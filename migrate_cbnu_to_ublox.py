#!/usr/bin/env python3
"""One-shot migration: convert cbnu lane1/lane2 lat/lon CSVs into ublox-format
extract dirs so the site uses pipeline.py's standard bag_extract loader.

Effect: removes the csv_pair branch from the data plane — all sites speak
the same format from this point on.

After this runs successfully:
  - data/sites/cbnu/source/lane1_extracted/{ublox_gps__fix,ublox_gps__navpvt}.csv
  - data/sites/cbnu/source/lane2_extracted/{ublox_gps__fix,ublox_gps__navpvt}.csv
  - The original lane1_data.csv / lane2_cross.csv can be deleted (kept here
    as a backup until you confirm the new pipeline runs end-to-end).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import to_utm

FS = 10.0          # sample rate (Hz) — team confirms walking measurement
HACC_M = 0.05      # RTK Fixed precision (m)
SACC_MPS = 0.05    # RTK Fixed precision (m/s)
CARRSOLN = 2       # 2 = RTK Fixed
T0_EPOCH = 0.0     # arbitrary base — pipeline.load_trip normalizes anyway

# Per-job interpolation: smooth the biggest position teleport by inserting
# N linearly-interpolated samples at the largest lat/lon gap.
# Why per-job: only lane2 (ego) has a splice (team-edited lane change);
# lane1 (emergency) is continuous.
INTERP_SECONDS = {
    "lane2_extracted": 0.7,   # 0.7s × 10Hz = 7 rows at splice
}

JOBS = [
    ("data/sites/cbnu/source/lane1_data.csv",
     "data/sites/cbnu/source/lane1_extracted"),
    ("data/sites/cbnu/source/lane2_cross.csv",
     "data/sites/cbnu/source/lane2_extracted"),
]


def smooth_lane_change(lat: np.ndarray, lon: np.ndarray, n_insert: int):
    """Detect the largest position gap and linearly interpolate n_insert
    rows across it. Returns (lat_new, lon_new, i_splice).

    If no gap clearly dominates (< ~2× median step), returns inputs
    unchanged so smoothing is safe even on already-smooth data.
    """
    if n_insert <= 0 or len(lat) < 3:
        return lat, lon, -1
    # Step distance in meters (approximate, lat≈111km/°, lon scaled by cos(lat))
    dlat = np.diff(lat) * 111000.0
    dlon = np.diff(lon) * 111000.0 * np.cos(np.radians(lat[:-1].mean()))
    step = np.hypot(dlat, dlon)
    i = int(np.argmax(step))
    if step[i] < 2.0 * np.median(step):
        return lat, lon, -1
    # exclusive endpoints — interp rows go BETWEEN i and i+1
    u = np.linspace(0.0, 1.0, n_insert + 2)[1:-1]
    lat_ins = lat[i] + u * (lat[i + 1] - lat[i])
    lon_ins = lon[i] + u * (lon[i + 1] - lon[i])
    lat_new = np.concatenate([lat[:i + 1], lat_ins, lat[i + 1:]])
    lon_new = np.concatenate([lon[:i + 1], lon_ins, lon[i + 1:]])
    return lat_new, lon_new, i


def synthesize(src_csv: Path, out_dir: Path) -> int:
    df = pd.read_csv(src_csv)
    cols = {c.lower(): c for c in df.columns}
    lat = df[cols["latitude"]].to_numpy(dtype=float)
    lon = df[cols["longitude"]].to_numpy(dtype=float)
    if len(lat) < 2:
        raise ValueError(f"{src_csv}: < 2 rows")

    interp_s = INTERP_SECONDS.get(out_dir.name, 0.0)
    n_insert = int(round(interp_s * FS))
    lat, lon, i_splice = smooth_lane_change(lat, lon, n_insert)
    if i_splice >= 0:
        print(f"  interp: +{n_insert} rows at splice index {i_splice}→{i_splice+1} "
              f"({interp_s:.1f}s smoothing)")
    n = len(lat)

    t = np.arange(n) / FS
    bag_t = t + T0_EPOCH
    secs = np.floor(bag_t).astype(np.int64)
    nsecs = np.round((bag_t - secs) * 1e9).astype(np.int64)

    x, y = to_utm(lat, lon)
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    velE = dx * FS
    velN = dy * FS
    speed = np.hypot(velE, velN)

    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "bag_t.secs": secs,
        "bag_t.nsecs": nsecs,
        "latitude": lat,
        "longitude": lon,
    }).to_csv(out_dir / "ublox_gps__fix.csv", index=False)

    flags = (CARRSOLN & 0x3) << 6   # carrSoln in bits 6-7 of NavPVT flags

    pd.DataFrame({
        "bag_t.secs": secs,
        "bag_t.nsecs": nsecs,
        "heading": np.round(heading_deg * 1e5).astype(np.int64),  # 1e-5 deg
        "gSpeed": np.round(speed * 1e3).astype(np.int64),         # mm/s
        "velE":   np.round(velE  * 1e3).astype(np.int64),
        "velN":   np.round(velN  * 1e3).astype(np.int64),
        "hAcc":   np.full(n, int(round(HACC_M  * 1e3)), dtype=np.int64),
        "sAcc":   np.full(n, int(round(SACC_MPS * 1e3)), dtype=np.int64),
        "flags":  np.full(n, flags, dtype=np.int64),
    }).to_csv(out_dir / "ublox_gps__navpvt.csv", index=False)
    return n


def main() -> int:
    for src, out in JOBS:
        src_p = Path(src)
        out_p = Path(out)
        if not src_p.is_file():
            print(f"[skip] missing: {src_p}")
            continue
        n = synthesize(src_p, out_p)
        print(f"[wrote] {out_p}/{{ublox_gps__fix,ublox_gps__navpvt}}.csv "
              f"({n} samples @ {FS:.0f} Hz, hAcc={HACC_M*100:.0f}cm, carrSoln={CARRSOLN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
