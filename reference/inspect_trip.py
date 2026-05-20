#!/usr/bin/env python3
"""Plot an extracted trip's trajectory + heading + speed + RTK quality.

Reads <extract_dir>/ublox_gps__fix.csv and ublox_gps__navpvt.csv,
converts lat/lon to UTM 52N, saves <extract_dir>_inspect.png.

Usage:
    python3 inspect_trip.py gps1_extracted gps2_extracted
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

LL_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)


def t_secs(df, secs="header.stamp.secs", nsecs="header.stamp.nsecs"):
    if secs not in df.columns:
        secs, nsecs = "bag_t.secs", "bag_t.nsecs"
    return df[secs].astype(np.int64) + df[nsecs].astype(np.int64) * 1e-9


def carrsoln(flags):
    arr = flags.to_numpy().astype(np.int64)
    return (arr >> 6) & 0x3  # bits 6-7 of NavPVT.flags


def plot_trip(extract_dir: Path):
    fix = pd.read_csv(extract_dir / "ublox_gps__fix.csv")
    pvt = pd.read_csv(extract_dir / "ublox_gps__navpvt.csv")

    t_fix = t_secs(fix).to_numpy()
    t_fix -= t_fix[0]
    x, y = LL_TO_UTM.transform(fix["longitude"].to_numpy(),
                               fix["latitude"].to_numpy())
    x0 = x.mean()
    y0 = y.mean()
    x_rel = x - x0
    y_rel = y - y0

    t_pvt = t_secs(pvt, "bag_t.secs", "bag_t.nsecs").to_numpy()
    t_pvt -= t_pvt[0]
    heading_deg = pvt["heading"].to_numpy() * 1e-5
    speed = pvt["gSpeed"].to_numpy() * 1e-3
    hAcc = pvt["hAcc"].to_numpy() * 1e-3
    cs = carrsoln(pvt["flags"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{extract_dir.name}  ({len(fix)} fix msgs, "
                 f"{t_fix[-1]:.1f}s, mean UTM=({x0:.1f}, {y0:.1f}))",
                 fontsize=12)

    ax = axes[0, 0]
    sc = ax.scatter(x_rel, y_rel, c=t_fix, cmap="viridis", s=8)
    ax.plot(x_rel[0], y_rel[0], "go", ms=14, label="start")
    ax.plot(x_rel[-1], y_rel[-1], "rx", ms=14, mew=3, label="end")
    for frac in (0.25, 0.5, 0.75):
        idx = int(len(x_rel) * frac)
        ax.annotate(f"{t_fix[idx]:.0f}s", (x_rel[idx], y_rel[idx]),
                    fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
    ax.set_xlabel("UTM East offset (m)")
    ax.set_ylabel("UTM North offset (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(sc, ax=ax, label="time (s)")
    ax.set_title("trajectory (color = time)")

    ax = axes[0, 1]
    ax.plot(t_pvt, heading_deg, "b-", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading (deg)")
    ax.grid(True, alpha=0.3)
    ax.set_title("heading vs time")

    ax = axes[1, 0]
    ax.plot(t_pvt, speed, "g-", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ground speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("speed vs time (low = U-turn candidate)")

    ax = axes[1, 1]
    ax.plot(t_pvt, hAcc, "r-", lw=1, label="hAcc (m)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hAcc (m)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.plot(t_pvt, cs, "k.", ms=2, alpha=0.5)
    ax2.set_ylabel("carrSoln (0=none, 1=Float, 2=Fixed)")
    ax2.set_ylim(-0.5, 2.5)
    ax.set_title("RTK quality (hAcc log scale + carrSoln dots)")

    plt.tight_layout()
    out = extract_dir.parent / f"{extract_dir.name}_inspect.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")

    fix_t0 = int(fix["header.stamp.secs"].iloc[0])
    print(f"  bag start (epoch s) = {fix_t0}")
    print(f"  duration = {t_fix[-1]:.1f}s, fix msgs = {len(fix)}")
    print(f"  hAcc  median = {np.median(hAcc)*100:.1f}cm, "
          f"min = {hAcc.min()*100:.1f}cm, max = {hAcc.max()*100:.1f}cm")
    print(f"  carrSoln distribution: "
          f"None={int((cs==0).sum())}, Float={int((cs==1).sum())}, "
          f"Fixed={int((cs==2).sum())}")
    print(f"  heading range: {heading_deg.min():.1f} .. {heading_deg.max():.1f} deg")


def main():
    if len(sys.argv) < 2:
        print("usage: inspect_trip.py <extract_dir> [...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        plot_trip(Path(d))


if __name__ == "__main__":
    main()
