#!/usr/bin/env python3
"""End-to-end pipeline: bag extracts -> lanes.json + frames.json.

Inputs:
    --bag1   directory of extracted CSVs for the EMERGENCY vehicle
    --bag2   directory of extracted CSVs for the EGO vehicle
    --out    output directory for lanes.json + frames.json (default: web/)

The web UI (nav.html) reads only the two JSONs. To swap in new data later,
re-extract bags and re-run this script — no front-end changes needed.

Algorithm:
  1. Load fix + navpvt, convert to UTM (EPSG:32652).
  2. Detect U-turn per trip (min smoothed speed in middle 50% + heading flip).
  3. Trim to outbound (0..t_turn).
  4. Lane geometry from bag1: lane R = outbound smoothed, lane L = return
     reversed + smoothed. Resampled at uniform 1m arc-length spacing.
  5. Resample both trips onto a common time grid (default 10 Hz).
  6. Per frame: ego's 10m forward polygon (lane-width-aligned rectangle),
     project emergency vehicle into ego frame, set same_lane_ahead boolean.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

LL_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
UTM_TO_LL = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Site registry — single source of truth for "where is the data, what kind".
# Adding a new site: append an entry here and run pipeline.py --site <id>.
# ---------------------------------------------------------------------------
SITES = {
    "ochang": {
        "label": "충북대학교 오창캠퍼스 C-TRACK 외곽순환도로",
        "label_short": "오창",
        "input": {
            "type": "bag_extract",
            "bag1": "data/sites/ochang/source/gps1_extracted",   # emergency
            "bag2": "data/sites/ochang/source/gps2_extracted",   # ego
        },
        "final": "data/sites/ochang/final.json",
        "out": "web/sites/ochang",
        "mirror_to_web_root": False,
        "sample_rate_hz": None,          # use bag timestamps
        "rtk_quality": "dgps",
        "has_uturn": True,
        "lane_width_m": 3.5,
        # speed_real_factor: stored → real-world. Ochang IS real vehicle so 1.0.
        # speed_vehicle_factor: stored → "demo vehicle" speed. Ochang inflates
        # 3.5× so its max ~14 km/h becomes ~50 km/h, matching CBNU's vehicle
        # mode range and giving the toggle a visible effect.
        "speed_real_factor": 1.0,
        "speed_vehicle_factor": 3.5,
        # Future-path overlay: 5s lookahead works for ochang's 32s demo.
        "future_path_s": 5.0,
        "future_path_width_m": 1.8,
        # Marker sizes (CSS px) for the map overlay.
        "marker_px": 14,
        "marker_label_px": 10,
    },
    "cbnu": {
        "label": "충북대학교 본캠퍼스",
        "label_short": "충북대",
        "input": {
            "type": "bag_extract",   # migrated via migrate_cbnu_to_ublox.py
            "bag1": "data/sites/cbnu/source/lane1_extracted",  # emergency
            "bag2": "data/sites/cbnu/source/lane2_extracted",  # ego (lane change)
        },
        "final": "data/sites/cbnu/final.json",
        "out": "web/sites/cbnu",
        "mirror_to_web_root": False,
        "rtk_quality": "rtk_fixed",      # informational — synthesized hAcc/carrSoln in ublox CSVs
        "has_uturn": False,              # one-way drive, no U-turn
        "lane_width_m": 3.5,
        # Original team data was 1 Hz walking; we replay it as ~10 Hz "vehicle"
        # for demo pace. The CV-KF tuned for ochang vehicles rejects every
        # measurement (NIS huge, mahalanobis gate fails) under this scale shift.
        # RTK Fixed input is already cm-level, so skip smoothing entirely.
        "skip_kf": True,
        # CBNU: 1Hz walking re-paced as 10Hz demo. Multiply stored speed by 0.1
        # to recover the actual walking speed for the "실제" UI mode.
        "speed_real_factor": 0.1,
        "speed_vehicle_factor": 1.0,    # stored is already vehicle-paced
        # CBNU's 4.2s demo is shorter than ochang — 1s future path keeps
        # the band inside the visible trip instead of running to data's end.
        "future_path_s": 1.0,
        "future_path_width_m": 0.9,
        # Smaller markers/labels — the campus area is tight and vehicles
        # often overlap.
        "marker_px": 10,
        "marker_label_px": 8,
    },
}


def to_utm(lat, lon):
    x, y = LL_TO_UTM.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def from_utm_pair(x, y):
    lon, lat = UTM_TO_LL.transform(np.asarray(x), np.asarray(y))
    return np.asarray(lat), np.asarray(lon)


def csv_t(df, secs="bag_t.secs", nsecs="bag_t.nsecs"):
    return df[secs].to_numpy().astype(np.int64) + df[nsecs].to_numpy().astype(np.int64) * 1e-9


def load_trip(extract_dir: Path, max_hacc_m=float("inf"), min_carrsoln=0):
    fix = pd.read_csv(extract_dir / "ublox_gps__fix.csv")
    pvt = pd.read_csv(extract_dir / "ublox_gps__navpvt.csv")

    t_fix = csv_t(fix)
    t0 = float(t_fix[0])
    t = t_fix - t0

    lat = fix["latitude"].to_numpy()
    lon = fix["longitude"].to_numpy()

    t_pvt = csv_t(pvt) - t0
    heading_nav = pvt["heading"].to_numpy() * 1e-5
    speed = pvt["gSpeed"].to_numpy() * 1e-3
    velE = pvt["velE"].to_numpy() * 1e-3
    velN = pvt["velN"].to_numpy() * 1e-3
    hAcc = pvt["hAcc"].to_numpy() * 1e-3
    sAcc = pvt["sAcc"].to_numpy() * 1e-3
    cs = (pvt["flags"].to_numpy().astype(np.int64) >> 6) & 0x3

    heading_at_fix = np.interp(t, t_pvt, heading_nav)
    speed_at_fix = np.interp(t, t_pvt, speed)
    velE_at_fix = np.interp(t, t_pvt, velE)
    velN_at_fix = np.interp(t, t_pvt, velN)
    hAcc_at_fix = np.interp(t, t_pvt, hAcc)
    sAcc_at_fix = np.interp(t, t_pvt, sAcc)
    cs_at_fix = np.round(np.interp(t, t_pvt, cs)).astype(int)

    keep = (hAcc_at_fix <= max_hacc_m) & (cs_at_fix >= min_carrsoln)
    if keep.sum() < len(keep):
        print(f"  filter dropped {(~keep).sum()}/{len(keep)} samples "
              f"(hAcc>{max_hacc_m}m or carrSoln<{min_carrsoln})")

    return {
        "t0_epoch": t0,
        "t": t[keep],
        "lat": lat[keep],
        "lon": lon[keep],
        "heading_nav": heading_at_fix[keep],
        "speed": speed_at_fix[keep],
        "velE": velE_at_fix[keep],
        "velN": velN_at_fix[keep],
        "hAcc": hAcc_at_fix[keep],
        "sAcc": sAcc_at_fix[keep],
        "carrSoln": cs_at_fix[keep],
    }


def add_utm_and_tangent_heading(trip):
    x, y = to_utm(trip["lat"], trip["lon"])
    trip["x"] = x
    trip["y"] = y

    # Heading from path tangent (more robust than nav-COG at low speed).
    dx = np.gradient(x)
    dy = np.gradient(y)
    bearing_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    # Light smoothing
    w = max(11, (len(bearing_deg) // 50) | 1)
    if len(bearing_deg) > w:
        # Smooth the unit-vector form to avoid 0/360 wrap issues
        c = savgol_filter(np.cos(np.radians(bearing_deg)), w, 2)
        s = savgol_filter(np.sin(np.radians(bearing_deg)), w, 2)
        bearing_deg = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
    trip["heading_path"] = bearing_deg
    return trip


def apply_bias(trip, dx_east, dy_north):
    """Shift all UTM positions by (dx, dy) and refresh lat/lon. Heading unchanged."""
    if not (dx_east or dy_north):
        return
    trip["x"] = trip["x"] + dx_east
    trip["y"] = trip["y"] + dy_north
    lat, lon = from_utm_pair(trip["x"], trip["y"])
    trip["lat"] = lat
    trip["lon"] = lon


def smooth_trip_with_kf(trip, sigma_a_mps2=1.0):
    """Run CV-KF + RTS smoother on the trip. Updates positions, speed, and
    heading_path in place. Returns a dict with before/after diagnostics."""
    from gps_kf import FilterConfig, GPSTrackFilter
    raw_x = trip["x"].copy()
    raw_y = trip["y"].copy()
    raw_heading = trip["heading_path"].copy() if "heading_path" in trip else None

    tracker = GPSTrackFilter(FilterConfig(sigma_a_mps2=sigma_a_mps2))
    ms = [tracker.adapter.build(trip["t"][i], trip["x"][i], trip["y"][i],
                                trip["velE"][i], trip["velN"][i],
                                trip["hAcc"][i], trip["sAcc"][i],
                                int(trip["carrSoln"][i]))
          for i in range(len(trip["t"]))]
    result = tracker.run_smoothed(ms)

    trip["x"] = result.x
    trip["y"] = result.y
    lat, lon = from_utm_pair(trip["x"], trip["y"])
    trip["lat"] = lat
    trip["lon"] = lon
    trip["speed"] = result.speed
    trip["heading_path"] = result.heading_deg

    raw_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([raw_x, raw_y]), axis=0) ** 2)))
    smt_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([result.x, result.y]), axis=0) ** 2)))
    return {
        "raw_step_jitter_m": raw_jit,
        "smoothed_step_jitter_m": smt_jit,
        "median_nis": float(np.median(result.nis)),
        "accepted_pct": float(100 * result.accepted.mean()),
        "n_samples": int(len(result.t)),
    }


def _lanes_for_centerline(pts_latlng, lane_w):
    """Returns dense lane R/L polylines plus the resampled centerline + per-point
    unit tangent (East,North) so callers can do heading-aware lane assignment."""
    pts = np.asarray(pts_latlng, dtype=float)
    half = lane_w / 2
    cx, cy = to_utm(pts[:, 0], pts[:, 1])
    P = np.column_stack([cx, cy])
    N = len(P)
    segR = np.zeros((N - 1, 2))
    for i in range(N - 1):
        d = P[i + 1] - P[i]
        L = np.hypot(*d) or 1e-9
        u = d / L
        segR[i] = (u[1], -u[0])
    vR = np.zeros_like(P)
    for i in range(N):
        if i == 0:
            vR[i] = segR[0]
        elif i == N - 1:
            vR[i] = segR[-1]
        else:
            a, b = segR[i - 1], segR[i]
            bsum = a + b
            bn = np.hypot(*bsum) or 1.0
            d = bsum / bn
            cos = a @ d
            scale = 5.0 if abs(cos) < 0.2 else 1.0 / cos
            vR[i] = d * scale
    Rsparse = P + half * vR
    Lsparse = P - half * vR

    def resample_uniform(poly, ds=1.0):
        seg = np.sqrt(((np.diff(poly, axis=0)) ** 2).sum(axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < ds:
            return poly[:, 0], poly[:, 1], s
        s_u = np.arange(0.0, s[-1], ds)
        xu = np.interp(s_u, s, poly[:, 0])
        yu = np.interp(s_u, s, poly[:, 1])
        return xu, yu, s_u

    Rx, Ry, sR = resample_uniform(Rsparse)
    Lx, Ly, sL = resample_uniform(Lsparse)
    # Resample the centerline itself the same way + derive per-point unit tangent
    Cx, Cy, sC = resample_uniform(P, ds=1.0)
    if len(Cx) >= 2:
        dCx = np.gradient(Cx)
        dCy = np.gradient(Cy)
        Lm = np.hypot(dCx, dCy)
        Lm[Lm < 1e-9] = 1e-9
        tan_x = dCx / Lm
        tan_y = dCy / Lm
    else:
        tan_x = np.array([1.0])
        tan_y = np.array([0.0])
    return {"R": {"x": Rx, "y": Ry, "s": sR},
            "L": {"x": Lx, "y": Ly, "s": sL},
            "center": {"x": Cx, "y": Cy, "tan_x": tan_x, "tan_y": tan_y}}


def snap_junctions(roads_pts, snap_radius_m=3.0):
    """Group all endpoints (first/last vertex of each road) within snap_radius_m
    of each other, replacing each group with a shared centroid in lat/lng.
    Ensures connected roads visually meet at exactly the same coordinate.
    Returns a (possibly modified) deep-ish copy of roads_pts."""
    if len(roads_pts) < 2:
        return roads_pts
    endpoints = []  # (road_idx, vertex_idx, x, y)
    for ri, pts in enumerate(roads_pts):
        if len(pts) < 2:
            continue
        x, y = to_utm(np.array([pts[0][0], pts[-1][0]]),
                      np.array([pts[0][1], pts[-1][1]]))
        endpoints.append((ri, 0, float(x[0]), float(y[0])))
        endpoints.append((ri, len(pts) - 1, float(x[1]), float(y[1])))
    n = len(endpoints)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    r2 = snap_radius_m ** 2
    for i in range(n):
        for j in range(i + 1, n):
            if endpoints[i][0] == endpoints[j][0]:
                continue
            dx = endpoints[i][2] - endpoints[j][2]
            dy = endpoints[i][3] - endpoints[j][3]
            if dx * dx + dy * dy < r2:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups = {}
    for i in range(n):
        g = find(i)
        groups.setdefault(g, []).append(i)
    out = [list(pts) for pts in roads_pts]
    n_snapped = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        cx = float(np.mean([endpoints[m][2] for m in members]))
        cy = float(np.mean([endpoints[m][3] for m in members]))
        lat, lon = from_utm_pair(np.array([cx]), np.array([cy]))
        snap_ll = [float(lat[0]), float(lon[0])]
        for m in members:
            ri, vi, _, _ = endpoints[m]
            out[ri][vi] = snap_ll
            n_snapped += 1
    if n_snapped:
        print(f"  snapped {n_snapped} endpoints into {sum(1 for v in groups.values() if len(v)>=2)} junctions "
              f"(radius {snap_radius_m:.1f}m)")
    return out


def lanes_from_road_setup(road_setup):
    """Build lanes from manual N-point centerline(s).

    Accepts new format:  {"roads": [{"centerline_latlng": [[lat,lng], ...]}, ...]}
    or legacy format:    {"centerline_latlng": [[lat,lng], ...]}
    The first road is "main" — its R/L lanes are used for lane assignment
    (KDTree). All roads' lanes are returned for rendering.
    """
    lane_w = float(road_setup["lane_width_m"])
    if "roads" in road_setup:
        roads_pts = [r["centerline_latlng"] for r in road_setup["roads"]]
    else:
        roads_pts = [road_setup["centerline_latlng"]]
    if not roads_pts:
        raise ValueError("road_setup has no roads")

    # Snap nearby road endpoints into shared junctions so centerlines visually connect.
    snap_r = float(road_setup.get("junction_snap_m", 3.0))
    roads_pts = snap_junctions(roads_pts, snap_radius_m=snap_r)

    per_road = [_lanes_for_centerline(p, lane_w) for p in roads_pts]
    # Attach the (snapped) raw click-points so renderers can draw exactly
    # the polyline the user defined (no midpoint approximation).
    for r, raw_pts in zip(per_road, roads_pts):
        r["center"]["raw_latlng"] = raw_pts
    # Per-road centerline KDTree for heading-aware assignment.
    road_assign = []
    for r in per_road:
        cx, cy = r["center"]["x"], r["center"]["y"]
        road_assign.append({
            "center_xy": np.column_stack([cx, cy]),
            "tan": np.column_stack([r["center"]["tan_x"], r["center"]["tan_y"]]),
            "tree": cKDTree(np.column_stack([cx, cy])),
        })
    # Combined R/L for backward-compat / debug (unused by new lane assignment).
    Rx = np.concatenate([r["R"]["x"] for r in per_road])
    Ry = np.concatenate([r["R"]["y"] for r in per_road])
    Lx = np.concatenate([r["L"]["x"] for r in per_road])
    Ly = np.concatenate([r["L"]["y"] for r in per_road])
    total_R = sum((r["R"]["s"][-1] if len(r["R"]["s"]) else 0.0) for r in per_road)
    total_L = sum((r["L"]["s"][-1] if len(r["L"]["s"]) else 0.0) for r in per_road)

    return {
        "R": {"x": Rx, "y": Ry, "s": np.array([0.0, total_R])},
        "L": {"x": Lx, "y": Ly, "s": np.array([0.0, total_L])},
        "tree_R": cKDTree(np.column_stack([Rx, Ry])),
        "tree_L": cKDTree(np.column_stack([Lx, Ly])),
        "sep_mean_m": lane_w, "sep_median_m": lane_w,
        "source": f"manual road_setup ({len(per_road)} road(s))",
        "per_road": per_road,
        "road_assign": road_assign,
        "lane_half_w": lane_w / 2,
    }


def find_turnaround(trip, min_frac=0.2, max_frac=0.8):
    t = trip["t"]
    speed = trip["speed"]
    heading = trip["heading_path"]

    dt = float(np.median(np.diff(t)))
    win = max(11, int(round(2.0 / dt)) | 1)
    sp = savgol_filter(speed, win, 2) if len(speed) > win else speed.copy()

    i_lo = int(len(t) * min_frac)
    i_hi = int(len(t) * max_frac)
    i = i_lo + int(np.argmin(sp[i_lo:i_hi]))

    pre = heading[max(0, i - 40):i]
    post = heading[i:min(len(heading), i + 40)]
    pre_mean = np.degrees(np.arctan2(np.sin(np.radians(pre)).mean(),
                                     np.cos(np.radians(pre)).mean()))
    post_mean = np.degrees(np.arctan2(np.sin(np.radians(post)).mean(),
                                      np.cos(np.radians(post)).mean()))
    diff = abs(((post_mean - pre_mean + 180.0) % 360.0) - 180.0)

    return {
        "i": int(i),
        "t": float(t[i]),
        "speed_at_turn": float(sp[i]),
        "heading_change_deg": float(diff),
    }


def resample_polyline(x, y, ds=1.0, smooth_window=11, smooth_order=3):
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < ds:
        return x.copy(), y.copy(), s
    s_u = np.arange(0.0, s[-1], ds)
    xu = np.interp(s_u, s, x)
    yu = np.interp(s_u, s, y)
    if len(s_u) > smooth_window:
        xu = savgol_filter(xu, smooth_window, smooth_order)
        yu = savgol_filter(yu, smooth_window, smooth_order)
    return xu, yu, s_u


def build_lanes(trip1, turn1, ds=1.0, edge_trim_samples=30):
    i_turn = turn1["i"]
    i_out_end = max(edge_trim_samples, i_turn - edge_trim_samples)
    i_ret_start = min(len(trip1["x"]) - edge_trim_samples, i_turn + edge_trim_samples)

    xR_raw = trip1["x"][:i_out_end]
    yR_raw = trip1["y"][:i_out_end]
    xR, yR, sR = resample_polyline(xR_raw, yR_raw, ds=ds)

    xL_raw = trip1["x"][i_ret_start:][::-1]
    yL_raw = trip1["y"][i_ret_start:][::-1]
    xL, yL, sL = resample_polyline(xL_raw, yL_raw, ds=ds)

    tree_R = cKDTree(np.column_stack([xR, yR]))
    d_RL, _ = tree_R.query(np.column_stack([xL, yL]))

    return {
        "R": {"x": xR, "y": yR, "s": sR},
        "L": {"x": xL, "y": yL, "s": sL},
        "tree_R": tree_R,
        "tree_L": cKDTree(np.column_stack([xL, yL])),
        "sep_mean_m": float(d_RL.mean()),
        "sep_median_m": float(np.median(d_RL)),
    }


def resample_trip_to_grid(trip, t_grid):
    """Interpolate trip onto t_grid. Clamps beyond endpoints."""
    out = {"t": t_grid}
    for key in ("lat", "lon", "speed", "hAcc"):
        out[key] = np.interp(t_grid, trip["t"], trip[key])
    # Heading: interpolate via unit-vector to avoid wrap.
    rad = np.radians(trip["heading_path"])
    c = np.interp(t_grid, trip["t"], np.cos(rad))
    s = np.interp(t_grid, trip["t"], np.sin(rad))
    out["heading"] = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
    out["x"] = np.interp(t_grid, trip["t"], trip["x"])
    out["y"] = np.interp(t_grid, trip["t"], trip["y"])
    out["carrSoln"] = np.round(np.interp(t_grid, trip["t"], trip["carrSoln"])).astype(int)
    return out


def apply_override(t, overrides, fallback):
    for ov in overrides:
        if ov["start_s"] <= t <= ov["end_s"]:
            return ov["lane"]
    return fallback


def assign_lane_heading_aware(x, y, heading_deg, road_assign, lane_half_w,
                              cos_min=0.5, deadband_m=0.3, prev_lane=None):
    """Assign 1/2-lane label using each road's tangent and the vehicle heading.

    For each candidate road:
      - Find nearest centerline point + its tangent.
      - Skip if |cos(heading, tangent)| < cos_min (i.e., >60° apart in either
        direction — handles intersections by rejecting wrong-axis roads).
      - Compute signed cross-track. cos>0: vehicle goes with road's clicked
        direction; cos<0: vehicle goes opposite (lane label flips).
    Pick the road with smallest |cross|. Korean convention:
      - 1차선 (left of motion)
      - 2차선 (right of motion)

    Hysteresis: if |signed-cross| < deadband_m AND prev_lane was the OTHER
    lane, stay sticky to prev_lane. Prevents label oscillation when the
    vehicle drives nearly on the centerline (sub-cm crossings would otherwise
    flip the label every few frames).
    """
    h = np.radians(heading_deg)
    veh = np.array([np.sin(h), np.cos(h)])      # (east, north) for bearing
    best = None
    for ri, r in enumerate(road_assign):
        if len(r["center_xy"]) == 0:
            continue
        d, i = r["tree"].query([x, y])
        tan = r["tan"][i]
        cos = float(veh @ tan)
        if abs(cos) < cos_min:
            continue
        right_of_road = np.array([tan[1], -tan[0]])
        cp = r["center_xy"][i]
        diff = np.array([x - cp[0], y - cp[1]])
        cross = float(diff @ right_of_road)
        veh_right_signed_d = cross if cos > 0 else -cross
        new_lane = "2차선" if veh_right_signed_d > 0 else "1차선"
        # Hysteresis: stay on previous lane if inside the deadband and prev is opposite.
        if prev_lane and abs(veh_right_signed_d) < deadband_m and prev_lane != new_lane:
            lane = prev_lane
        else:
            lane = new_lane
        d_perp = abs(cross)
        in_lane = d_perp <= lane_half_w
        score = (0 if in_lane else 1, d_perp)
        cand = {"lane": lane, "d_perp_m": d_perp, "in_lane": in_lane,
                "road_idx": ri, "score": score}
        if best is None or cand["score"] < best["score"]:
            best = cand
    return best


def future_path_band_latlng(xs, ys, hs_deg, i_start, n_ahead, width_m,
                            n_control_points=4, n_fine=40):
    """Future-path band with INDEPENDENT left/right edge fitting.

    Architecture (per user direction):
      1. Caller pre-smooths the ENTIRE trip once (time-axis consistency).
      2. Pick sparse control points along this future window.
      3. Compute left/right edge points at each control point via perpendicular
         offset of the local control-point tangent.
      4. **Fit PCHIP independently through left edge points AND right edge
         points** (arc-length parameterized) → each boundary becomes smooth
         on its own, without tangent-noise propagating through the perp offset.
      5. Polygon = smoothed_left + smoothed_right reversed.

    The independent fit decouples each edge from instantaneous tangent
    estimation, which is what made the previous version wobble.
    """
    from scipy.interpolate import PchipInterpolator
    n = len(xs)
    if i_start >= n - 1 or n_ahead <= 0:
        return []
    i_end = min(n, i_start + n_ahead + 1)
    px = np.asarray(xs[i_start:i_end])
    py = np.asarray(ys[i_start:i_end])
    if len(px) < 2:
        return []
    nc = max(2, min(int(n_control_points), len(px)))
    idx = np.linspace(0, len(px) - 1, nc).round().astype(int)
    cx, cy = px[idx], py[idx]

    # Per-control-point tangent from neighbouring control points (gradient).
    # We only need direction here; magnitude doesn't matter for the offset.
    dx = np.gradient(cx); dy = np.gradient(cy)
    L = np.hypot(dx, dy); L[L < 1e-9] = 1e-9
    tx, ty = dx / L, dy / L
    rx, ry = ty, -tx                  # right perpendicular at each control point
    half = width_m / 2
    left_ctrl  = np.column_stack([cx - half * rx, cy - half * ry])
    right_ctrl = np.column_stack([cx + half * rx, cy + half * ry])

    def smooth_edge(ctrl_xy):
        seg = np.hypot(np.diff(ctrl_xy[:, 0]), np.diff(ctrl_xy[:, 1]))
        seg = np.maximum(seg, 1e-3)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 1e-2 or len(ctrl_xy) < 3:
            return ctrl_xy
        psx = PchipInterpolator(s, ctrl_xy[:, 0])
        psy = PchipInterpolator(s, ctrl_xy[:, 1])
        u = np.linspace(0, s[-1], n_fine)
        return np.column_stack([psx(u), psy(u)])

    left = smooth_edge(left_ctrl)
    right = smooth_edge(right_ctrl)
    polygon = np.vstack([left, right[::-1]])
    la, lo = from_utm_pair(polygon[:, 0], polygon[:, 1])
    return list(zip(la.tolist(), lo.tolist()))


def lane_band_latlng(lanes, road_idx, x, y, fwd_unit, behind_m, ahead_m, half_width):
    """Return a closed polygon (list of [lat,lng]) covering ego's current LANE
    from behind_m behind to ahead_m ahead, with `half_width` perpendicular offset
    on each side (= full lane width when half_width = lane_width / 2).
    Lane chosen by signed cross-track using fwd_unit so it's stable across
    lane changes."""
    if road_idx < 0 or "per_road" not in lanes or road_idx >= len(lanes["per_road"]):
        return []
    rl = lanes["per_road"][road_idx]
    # Use centerline + tangent (precomputed in lanes_from_road_setup)
    cx = rl["center"]["x"]; cy = rl["center"]["y"]
    if len(cx) < 2:
        return []
    tx_arr = rl["center"]["tan_x"]; ty_arr = rl["center"]["tan_y"]
    # Project ego onto centerline
    dist = np.hypot(cx - x, cy - y)
    i_min = int(np.argmin(dist))
    tx, ty = tx_arr[i_min], ty_arr[i_min]
    cos_align = float(fwd_unit[0] * tx + fwd_unit[1] * ty)
    # Right-perpendicular of road tangent in (east, north): (ty, -tx)
    rx_arr = ty_arr; ry_arr = -tx_arr
    # Signed cross-track relative to road tangent
    cross = (x - cx[i_min]) * rx_arr[i_min] + (y - cy[i_min]) * ry_arr[i_min]
    # Lane center offset = +half_width if ego is on road-right (cross>0), else -half_width
    lane_center_sign = 1 if cross >= 0 else -1
    # Pick index range along the centerline
    bi = int(round(behind_m)); ai = int(round(ahead_m))
    if cos_align >= 0:
        i_start, i_end = max(0, i_min - bi), min(len(cx), i_min + ai + 1)
    else:
        i_start, i_end = max(0, i_min - ai), min(len(cx), i_min + bi + 1)
    if i_end - i_start < 2:
        return []
    cseg = np.column_stack([cx[i_start:i_end], cy[i_start:i_end]])
    rseg = np.column_stack([rx_arr[i_start:i_end], ry_arr[i_start:i_end]])
    lane_center = cseg + lane_center_sign * half_width * rseg
    # Polygon edges: lane_center ± half_width along the perpendicular
    left_edge  = lane_center + half_width * rseg
    right_edge = lane_center - half_width * rseg
    polygon = np.vstack([left_edge, right_edge[::-1]])
    la, lo = from_utm_pair(polygon[:, 0], polygon[:, 1])
    return list(zip(la.tolist(), lo.tolist()))


def per_frame(ego, emerg, lanes, forward_m, lane_half_w, heading_tol_deg,
              ego_overrides=None, emerg_overrides=None, rear_m=None,
              overlay_behind_m=25.0, overlay_ahead_m=5.0,
              future_path_s=1.0, future_path_width_m=2.2,
              tick_hz=10.0):
    if rear_m is None:
        rear_m = forward_m
    future_steps = int(round(future_path_s * tick_hz))
    ego_overrides = ego_overrides or []
    emerg_overrides = emerg_overrides or []
    n = len(ego["t"])
    frames = []
    prev_ego_lane = None
    prev_emerg_lane = None
    for i in range(n):
        xe, ye = ego["x"][i], ego["y"][i]
        he_deg = ego["heading"][i]
        he = np.radians(he_deg)
        cos_h, sin_h = np.cos(he), np.sin(he)
        # Heading unit vector in (East, North) is (sin, cos) for compass bearing.
        fwd = np.array([sin_h, cos_h])
        right = np.array([cos_h, -sin_h])

        corners_xy = []
        for s in (0.0, forward_m):
            for d in (-lane_half_w, +lane_half_w):
                if s == 0.0 and d > 0:
                    pass
                corners_xy.append((xe + s * fwd[0] + d * right[0],
                                   ye + s * fwd[1] + d * right[1]))
        # Order corners to make a non-self-intersecting polygon:
        # bottom-left, bottom-right, top-right, top-left
        corner_order = [
            (0.0, -lane_half_w),
            (0.0, +lane_half_w),
            (forward_m, +lane_half_w),
            (forward_m, -lane_half_w),
        ]
        corners_xy = [(xe + s * fwd[0] + d * right[0],
                       ye + s * fwd[1] + d * right[1]) for s, d in corner_order]
        latlng = []
        for cx, cy in corners_xy:
            la, lo = from_utm_pair(np.array([cx]), np.array([cy]))
            latlng.append([float(la[0]), float(lo[0])])

        xm, ym = emerg["x"][i], emerg["y"][i]
        hm_deg = emerg["heading"][i]
        dx = xm - xe
        dy = ym - ye
        s_em = dx * fwd[0] + dy * fwd[1]
        d_em = dx * right[0] + dy * right[1]
        dist = float(np.hypot(dx, dy))
        head_diff = abs(((hm_deg - he_deg + 180.0) % 360.0) - 180.0)

        # Heading-aware lane assignment per road (must run BEFORE same_lane check).
        if "road_assign" in lanes:
            ego_a = assign_lane_heading_aware(xe, ye, he_deg, lanes["road_assign"],
                                              lane_half_w, prev_lane=prev_ego_lane)
            emerg_a = assign_lane_heading_aware(xm, ym, hm_deg, lanes["road_assign"],
                                                lane_half_w, prev_lane=prev_emerg_lane)
            ego_lane_auto = ego_a["lane"] if ego_a else None
            emerg_lane_auto = emerg_a["lane"] if emerg_a else None
            prev_ego_lane = ego_lane_auto if ego_lane_auto else prev_ego_lane
            prev_emerg_lane = emerg_lane_auto if emerg_lane_auto else prev_emerg_lane
            d_eR = ego_a["d_perp_m"] if ego_a else float("nan")
            d_eL = float("nan"); d_mR = emerg_a["d_perp_m"] if emerg_a else float("nan"); d_mL = float("nan")
            ego_road_idx = ego_a["road_idx"] if ego_a else -1
            emerg_road_idx = emerg_a["road_idx"] if emerg_a else -1
        else:
            d_eR, _ = lanes["tree_R"].query([xe, ye])
            d_eL, _ = lanes["tree_L"].query([xe, ye])
            ego_lane_auto = "2차선" if d_eR < d_eL else "1차선"
            d_mR, _ = lanes["tree_R"].query([xm, ym])
            d_mL, _ = lanes["tree_L"].query([xm, ym])
            emerg_lane_auto = "2차선" if d_mR < d_mL else "1차선"
            ego_road_idx = 0; emerg_road_idx = 0
        ego_lane = apply_override(float(ego["t"][i]), ego_overrides, ego_lane_auto)
        emerg_lane = apply_override(float(emerg["t"][i]), emerg_overrides, emerg_lane_auto)

        # Topological same-lane check: same road AND same lane label.
        on_same_road = (ego_road_idx == emerg_road_idx and ego_road_idx >= 0
                        and ego_lane is not None and emerg_lane is not None)
        topo_same_lane = on_same_road and (ego_lane == emerg_lane)
        # Fallback for bag1-derived lanes: relative-frame lateral check.
        rel_same_lane = (abs(d_em) <= lane_half_w) and (head_diff <= heading_tol_deg)
        same_lane = topo_same_lane if "road_assign" in lanes else rel_same_lane
        same_lane_ahead  = bool(same_lane and  0.0 <= s_em <= forward_m)
        same_lane_behind = bool(same_lane and -rear_m   <= s_em <  0.0)
        # Alert fires only when emergency is BEHIND ego in same lane (rear blind-spot).
        same_lane_alert  = same_lane_behind

        frames.append({
            "t": float(ego["t"][i]),
            "ego": {
                "lat": float(ego["lat"][i]),
                "lon": float(ego["lon"][i]),
                "heading": float(he_deg),
                "speed": float(ego["speed"][i]),
                "hAcc": float(ego["hAcc"][i]),
                "carrSoln": int(ego["carrSoln"][i]),
                "lane": ego_lane,
                "road_idx": int(ego_road_idx),
                "d_perp_m": float(d_eR),
            },
            "emergency": {
                "lat": float(emerg["lat"][i]),
                "lon": float(emerg["lon"][i]),
                "heading": float(hm_deg),
                "speed": float(emerg["speed"][i]),
                "hAcc": float(emerg["hAcc"][i]),
                "carrSoln": int(emerg["carrSoln"][i]),
                "lane": emerg_lane,
                "road_idx": int(emerg_road_idx),
                "d_perp_m": float(d_mR),
            },
            "forward_polygon_latlng": latlng,
            "ego_lane_band_latlng": lane_band_latlng(
                lanes, ego_road_idx, xe, ye, fwd,
                overlay_behind_m, overlay_ahead_m, lane_half_w),
            "emerg_lane_band_latlng": lane_band_latlng(
                lanes, emerg_road_idx, xm, ym,
                np.array([np.sin(np.radians(hm_deg)), np.cos(np.radians(hm_deg))]),
                0.0, 30.0, lane_half_w),
            "ego_future_band_latlng": future_path_band_latlng(
                ego["x"], ego["y"], ego["heading"], i, future_steps,
                future_path_width_m),
            "emerg_future_band_latlng": future_path_band_latlng(
                emerg["x"], emerg["y"], emerg["heading"], i, future_steps,
                future_path_width_m),
            "rel": {
                "distance_m": dist,
                "ahead_m": float(s_em),
                "lateral_m": float(d_em),
                "heading_diff_deg": float(head_diff),
            },
            "same_lane_ahead": same_lane_ahead,
            "same_lane_behind": same_lane_behind,
            "same_lane_alert": same_lane_alert,
        })
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES.keys()),
                    help="Resolve --bag1/--bag2/--out/--final from SITES registry. "
                         "Mutually exclusive with the manual --bag1/--bag2 args.")
    ap.add_argument("--bag1", help="emergency vehicle extract dir (or CSV path)")
    ap.add_argument("--bag2", help="ego vehicle extract dir (or CSV path)")
    ap.add_argument("--out", default="web", help="output dir (default: web)")
    ap.add_argument("--final", help="path to final.json (overrides site default)")
    ap.add_argument("--lane-width-m", type=float, default=3.5)
    ap.add_argument("--forward-m", type=float, default=10.0)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--heading-tol-deg", type=float, default=30.0)
    ap.add_argument("--rear-m", type=float, default=30.0,
                    help="rear alert range (m). matches emergency forward band default 30m.")
    ap.add_argument("--overlay-behind-m", type=float, default=25.0,
                    help="how far behind ego the lane overlay extends (m).")
    ap.add_argument("--overlay-ahead-m", type=float, default=5.0,
                    help="how far ahead of ego the lane overlay extends (m).")
    ap.add_argument("--future-path-s", type=float, default=5.0,
                    help="lookahead seconds for future-path band visualization.")
    ap.add_argument("--future-path-width-m", type=float, default=2.2,
                    help="width of future-path band (m). default narrower than lane.")
    ap.add_argument("--no-kf", action="store_true",
                    help="skip CV-KF + RTS smoothing (use raw bias-corrected data).")
    ap.add_argument("--kf-sigma-a", type=float, default=1.0,
                    help="process-noise RMS acceleration for CV-KF (m/s^2).")
    ap.add_argument("--start-speed-mps", type=float, default=0.5,
                    help="trim leading samples until speed exceeds this (m/s) sustained for --start-sustain-s")
    ap.add_argument("--start-sustain-s", type=float, default=1.0,
                    help="how long speed must stay above threshold to count as 'moving'")
    ap.add_argument("--max-hacc-m", type=float, default=float("inf"),
                    help="filter samples with hAcc above this (m). default: no filter")
    ap.add_argument("--min-carrsoln", type=int, default=0,
                    help="filter samples with carrSoln below this. set 2 for RTK Fixed only.")
    ap.add_argument("--site-label", default=None,
                    help="display label written into JSON (default: site config or 'ochang_ctrack_outer')")
    args = ap.parse_args()

    # Resolve site config — manual args win over site config when both given.
    site_id = args.site or ("ochang" if (args.bag1 and "gps1" in args.bag1) else None)
    site_cfg = SITES.get(args.site, {}) if args.site else {}
    input_cfg = site_cfg.get("input", {})

    bag1 = args.bag1 or input_cfg.get("bag1")
    bag2 = args.bag2 or input_cfg.get("bag2")
    if not (bag1 and bag2):
        ap.error("must provide either --site or both --bag1 and --bag2")
    out_path = args.out if args.out != "web" else site_cfg.get("out", "web")
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = Path(args.final) if args.final else Path(site_cfg.get("final", "final.json"))
    has_uturn = site_cfg.get("has_uturn", True)
    skip_kf_site = site_cfg.get("skip_kf", False)
    if site_cfg.get("lane_width_m"):
        args.lane_width_m = site_cfg["lane_width_m"]
    if site_cfg.get("future_path_s") is not None:
        args.future_path_s = site_cfg["future_path_s"]
    if site_cfg.get("future_path_width_m") is not None:
        args.future_path_width_m = site_cfg["future_path_width_m"]
    site_label = args.site_label or site_cfg.get("label") or "ochang_ctrack_outer"
    print(f"[site] id={site_id or 'manual'} out={out_dir}  has_uturn={has_uturn}  skip_kf={skip_kf_site}")

    def load_one(path_str, label):
        p = Path(path_str)
        print(f"[load] {label} = {p}")
        t = load_trip(p, args.max_hacc_m, args.min_carrsoln)
        add_utm_and_tangent_heading(t)
        print(f"  {len(t['t'])} samples, duration {t['t'][-1]:.1f}s, "
              f"hAcc median {np.median(t['hAcc']):.3f}m, "
              f"carrSoln dist {dict(zip(*np.unique(t['carrSoln'], return_counts=True)))}")
        return t

    trip1 = load_one(bag1, "bag1 (emergency)")
    trip2 = load_one(bag2, "bag2 (ego)")

    print("[turnaround]")
    if has_uturn:
        turn1 = find_turnaround(trip1)
        turn2 = find_turnaround(trip2)
    else:
        # No U-turn (one-way drive). Use endpoint so trim covers full duration.
        def endpoint(trip):
            return {"i": len(trip["t"]) - 1, "t": float(trip["t"][-1]),
                    "speed_at_turn": float(trip["speed"][-1]),
                    "heading_change_deg": 0.0}
        turn1 = endpoint(trip1)
        turn2 = endpoint(trip2)
    print(f"  bag1: t={turn1['t']:.1f}s, speed_min={turn1['speed_at_turn']:.2f}m/s, "
          f"heading change={turn1['heading_change_deg']:.0f}°")
    print(f"  bag2: t={turn2['t']:.1f}s, speed_min={turn2['speed_at_turn']:.2f}m/s, "
          f"heading change={turn2['heading_change_deg']:.0f}°")

    # ----- emit raw_trips.json for the editor (UNCORRECTED — editor uses this to compute bias) -----
    def downsample(trip, target_hz=5):
        dt = float(np.median(np.diff(trip["t"])))
        step = max(1, int(round(1.0 / target_hz / dt)))
        keep = slice(None, None, step)
        return {
            "t": trip["t"][keep].tolist(),
            "lat": trip["lat"][keep].tolist(),
            "lon": trip["lon"][keep].tolist(),
            "speed": trip["speed"][keep].tolist(),
            "heading": trip["heading_path"][keep].tolist(),
            "hAcc": trip["hAcc"][keep].tolist(),
        }

    raw_trips_json = {
        "bag1": {
            "name": "emergency", "source": str(args.bag1),
            "t0_epoch": trip1["t0_epoch"],
            "duration_s": float(trip1["t"][-1]),
            "auto": {"turnaround_t": turn1["t"]},
            "data": downsample(trip1),
        },
        "bag2": {
            "name": "ego", "source": str(args.bag2),
            "t0_epoch": trip2["t0_epoch"],
            "duration_s": float(trip2["t"][-1]),
            "auto": {"turnaround_t": turn2["t"]},
            "data": downsample(trip2),
        },
    }
    (out_dir / "raw_trips.json").write_text(json.dumps(raw_trips_json, ensure_ascii=False))
    print(f"  wrote {out_dir/'raw_trips.json'} "
          f"({(out_dir/'raw_trips.json').stat().st_size/1024:.1f} KB)")

    # ----- read optional final.json / edit.json -----
    # When --site is given: ONLY look at that site's final.json (no root fallback —
    # otherwise cbnu would pick up ochang's road geometry by accident).
    # Legacy mode (no --site): walk root → web/ candidates.
    edits = {}
    if args.site or args.final:
        candidates = [final_path]
    else:
        candidates = [Path("final.json"), Path("edit.json"),
                      out_dir / "final.json", out_dir / "edit.json"]
    for candidate in candidates:
        if candidate.exists():
            edits = json.loads(candidate.read_text())
            print(f"  loaded {candidate}: keys {list(edits.keys())}")
            break
    if not edits:
        print(f"  (no final.json found — looked at {[str(c) for c in candidates]})")

    # ----- apply bias (if road_setup present) -----
    road_setup = edits.get("road_setup")
    if not road_setup and not has_uturn:
        # CSV/one-way trip with no manual road yet — synthesize a default
        # centerline from trip1's path so the editor has something to render.
        # The user refines this in editor.html and saves a real final.json.
        pts = list(zip(trip1["lat"].tolist(), trip1["lon"].tolist()))
        step = max(1, len(pts) // 20)
        pts = pts[::step]
        print(f"  no road_setup — using bag1 path as default centerline "
              f"({len(pts)} pts); refine via editor.html?site={site_id}")
        road_setup = {"roads": [{"centerline_latlng": pts}],
                      "lane_width_m": args.lane_width_m}
    if road_setup:
        bias = road_setup.get("bias_m", {})
        dx = float(bias.get("dx_east", 0))
        dy = float(bias.get("dy_north", 0))
        if dx or dy:
            apply_bias(trip1, dx, dy)
            apply_bias(trip2, dx, dy)
            print(f"  applied bias correction dx={dx:+.3f}m dy={dy:+.3f}m")

    # ----- KF + RTS smoothing on each trip (replaces raw positions/heading) -----
    if not args.no_kf and not skip_kf_site:
        diag1 = smooth_trip_with_kf(trip1, args.kf_sigma_a)
        diag2 = smooth_trip_with_kf(trip2, args.kf_sigma_a)
        print(f"[kf+rts] bag1: {diag1['n_samples']} samples, median NIS {diag1['median_nis']:.2f}, "
              f"jitter {diag1['raw_step_jitter_m']:.3f}→{diag1['smoothed_step_jitter_m']:.3f}m")
        print(f"[kf+rts] bag2: {diag2['n_samples']} samples, median NIS {diag2['median_nis']:.2f}, "
              f"jitter {diag2['raw_step_jitter_m']:.3f}→{diag2['smoothed_step_jitter_m']:.3f}m")

    # ----- build lanes (here, after bias applied so bag1-derived lanes are also bias-corrected) -----
    print("[lanes]")
    if road_setup:
        lane_w = float(road_setup.get("lane_width_m", args.lane_width_m))
        args.lane_width_m = lane_w
        lanes = lanes_from_road_setup(road_setup)
        lanes_source = "manual road_setup"
    else:
        lanes = build_lanes(trip1, turn1)
        lanes_source = "bag1 outbound (R) + bag1 return reversed (L)"
    print(f"  source: {lanes_source}")
    print(f"  lane R: {len(lanes['R']['x'])} pts ({lanes['R']['s'][-1]:.1f}m)")
    print(f"  lane L: {len(lanes['L']['x'])} pts ({lanes['L']['s'][-1]:.1f}m)")
    print(f"  R-L separation: mean {lanes['sep_mean_m']:.2f}m, "
          f"median {lanes['sep_median_m']:.2f}m  (expected ~{args.lane_width_m:.1f}m)")

    print("[frames]")

    def trim_window(trip, key):
        """Apply trim window. Manual edits override auto detection."""
        ed = edits.get(key, {}).get("trim", {})
        if "start_s" in ed and "end_s" in ed:
            s, e = float(ed["start_s"]), float(ed["end_s"])
            m = (trip["t"] >= s) & (trip["t"] <= e)
            src = "manual"
        else:
            # auto: leading-stationary + outbound (up to turnaround)
            turn = turn1 if key == "bag1" else turn2
            sp = trip["speed"]
            dt = float(np.median(np.diff(trip["t"])))
            win = max(1, int(round(args.start_sustain_s / dt)))
            moving = sp > args.start_speed_mps
            i_start = 0
            for i in range(len(moving) - win + 1):
                if moving[i:i + win].all():
                    i_start = i
                    break
            t_start = float(trip["t"][i_start])
            m = (trip["t"] >= t_start) & (trip["t"] <= turn["t"])
            src = "auto"
        out = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in trip.items()}
        out["t"] = out["t"] - out["t"][0]
        kept_start = float(trip["t"][m][0]) if m.any() else 0.0
        return out, src, kept_start

    trip1_out, src1, t1_start = trim_window(trip1, "bag1")
    trip2_out, src2, t2_start = trim_window(trip2, "bag2")

    # Per-trip speed factor: compress trip duration by dividing t by factor.
    def apply_speed_factor(trip, key):
        f = float(edits.get(key, {}).get("speed_factor", 1.0))
        if f != 1.0:
            trip["t"] = trip["t"] / f
        return f
    f1 = apply_speed_factor(trip1_out, "bag1")
    f2 = apply_speed_factor(trip2_out, "bag2")

    # Optional linear extrapolation past the trim end (last heading × last speed).
    def extrapolate(trip, key):
        extra_m = float(edits.get(key, {}).get("extrapolate_after_m", 0.0))
        if extra_m <= 0:
            return 0.0
        last_h_deg = float(trip["heading_path"][-1])
        override = edits.get(key, {}).get("extrapolate_speed_mps")
        if override:
            last_speed = float(override)
        else:
            moving = trip["speed"] > 1.0
            cruise = float(np.median(trip["speed"][moving])) if moving.any() else 3.0
            last_speed = max(cruise, 3.0)
        duration = extra_m / last_speed
        dt = float(np.median(np.diff(trip["t"]))) if len(trip["t"]) > 1 else 0.1
        n = max(2, int(round(duration / dt)))
        h_rad = np.radians(last_h_deg)
        vx, vy = np.sin(h_rad) * last_speed, np.cos(h_rad) * last_speed
        t_extra = trip["t"][-1] + np.arange(1, n + 1) * dt
        dt_step = t_extra - trip["t"][-1]
        x_extra = trip["x"][-1] + vx * dt_step
        y_extra = trip["y"][-1] + vy * dt_step
        lat_extra, lon_extra = from_utm_pair(x_extra, y_extra)
        trip["t"]  = np.concatenate([trip["t"],  t_extra])
        trip["x"]  = np.concatenate([trip["x"],  x_extra])
        trip["y"]  = np.concatenate([trip["y"],  y_extra])
        trip["lat"]= np.concatenate([trip["lat"], lat_extra])
        trip["lon"]= np.concatenate([trip["lon"], lon_extra])
        trip["heading_path"] = np.concatenate(
            [trip["heading_path"], np.full(n, last_h_deg)])
        trip["speed"] = np.concatenate([trip["speed"], np.full(n, last_speed)])
        trip["hAcc"]  = np.concatenate([trip["hAcc"],  np.full(n, trip["hAcc"][-1])])
        trip["carrSoln"] = np.concatenate(
            [trip["carrSoln"], np.full(n, trip["carrSoln"][-1])])
        return extra_m
    ex1 = extrapolate(trip1_out, "bag1")
    ex2 = extrapolate(trip2_out, "bag2")
    if ex1 or ex2:
        print(f"  extrapolation: bag1 +{ex1:.0f}m, bag2 +{ex2:.0f}m")
    sf1 = f" ×{f1:g} speed" if f1 != 1.0 else ""
    sf2 = f" ×{f2:g} speed" if f2 != 1.0 else ""
    print(f"  bag1 trim: {src1}, starts at {t1_start:.1f}s, kept {trip1_out['t'][-1]:.1f}s{sf1}")
    print(f"  bag2 trim: {src2}, starts at {t2_start:.1f}s, kept {trip2_out['t'][-1]:.1f}s{sf2}")

    # Capture lane overrides for application after frame build
    lane_overrides = {
        "ego": edits.get("bag2", {}).get("lane_overrides", []),
        "emergency": edits.get("bag1", {}).get("lane_overrides", []),
    }

    T_end = max(trip1_out["t"][-1], trip2_out["t"][-1])
    dt = 1.0 / args.tick_hz
    t_grid = np.arange(0.0, T_end + dt, dt)

    e1 = resample_trip_to_grid(trip1_out, t_grid)
    e2 = resample_trip_to_grid(trip2_out, t_grid)

    # Heavy time-axis smoothing of the resampled trip positions so the
    # future-path band is consistent across consecutive frames (each frame
    # slices a window from these pre-smoothed arrays — no in-frame savgol
    # whose window-shift would cause polygon flicker between frames).
    # Resync lat/lon back from the smoothed UTM so marker (uses lat/lon) and
    # polygon (uses x/y) share one position — without this they drift apart
    # at the trip boundary (savgol edge behaviour) and the polygon tail
    # appears behind the marker.
    for arr in (e1, e2):
        n = len(arr["x"])
        if n >= 31:
            arr["x"] = savgol_filter(arr["x"], 31, 2)
            arr["y"] = savgol_filter(arr["y"], 31, 2)
            arr["lat"], arr["lon"] = from_utm_pair(arr["x"], arr["y"])

    # Convert lane override times from original bag time to scenario time.
    def shift(ovs, kept_start):
        return [{"start_s": o["start_s"] - kept_start,
                 "end_s": o["end_s"] - kept_start,
                 "lane": o["lane"]} for o in ovs]

    frames = per_frame(e2, e1, lanes,
                       forward_m=args.forward_m,
                       lane_half_w=args.lane_width_m / 2,
                       heading_tol_deg=args.heading_tol_deg,
                       rear_m=args.rear_m,
                       overlay_behind_m=args.overlay_behind_m,
                       overlay_ahead_m=args.overlay_ahead_m,
                       future_path_s=args.future_path_s,
                       future_path_width_m=args.future_path_width_m,
                       tick_hz=args.tick_hz,
                       ego_overrides=shift(lane_overrides["ego"], t2_start),
                       emerg_overrides=shift(lane_overrides["emergency"], t1_start))

    n_ahead  = sum(1 for f in frames if f["same_lane_ahead"])
    n_behind = sum(1 for f in frames if f["same_lane_behind"])
    n_any    = sum(1 for f in frames if f["same_lane_alert"])
    print(f"  {len(frames)} frames @ {args.tick_hz:.0f}Hz · "
          f"alert {n_any} ({n_any/max(1,len(frames))*100:.0f}%) = "
          f"ahead {n_ahead} + behind {n_behind}")

    # ----- emit JSON -----
    def lane_latlng(lane):
        la, lo = from_utm_pair(lane["x"], lane["y"])
        return list(zip(la.tolist(), lo.tolist()))

    lanes_out = []
    roads_out = []
    if "per_road" in lanes:
        for ri, road_lanes in enumerate(lanes["per_road"]):
            lanes_out.append({"id": "R", "road_idx": ri,
                              "name": f"lane R (road {ri + 1})",
                              "centerline_latlng": lane_latlng(road_lanes["R"])})
            lanes_out.append({"id": "L", "road_idx": ri,
                              "name": f"lane L (road {ri + 1})",
                              "centerline_latlng": lane_latlng(road_lanes["L"])})
            # Use the SNAPPED user-clicked centerline directly so junctions
            # share exact coordinates and visual lines connect seamlessly.
            raw = road_lanes["center"].get("raw_latlng")
            if raw is None:
                rx, ry = road_lanes["R"]["x"], road_lanes["R"]["y"]
                lx, ly = road_lanes["L"]["x"], road_lanes["L"]["y"]
                n = min(len(rx), len(lx))
                mx, my = (rx[:n] + lx[:n]) / 2, (ry[:n] + ly[:n]) / 2
                la, lo = from_utm_pair(mx, my)
                raw = list(zip(la.tolist(), lo.tolist()))
            roads_out.append({"idx": ri, "name": f"road {ri + 1}",
                              "centerline_latlng": raw})
    else:
        lanes_out = [
            {"id": "R", "road_idx": 0, "name": "lane R",
             "centerline_latlng": lane_latlng(lanes["R"])},
            {"id": "L", "road_idx": 0, "name": "lane L",
             "centerline_latlng": lane_latlng(lanes["L"])},
        ]
        rx, ry = lanes["R"]["x"], lanes["R"]["y"]
        lx, ly = lanes["L"]["x"], lanes["L"]["y"]
        n = min(len(rx), len(lx))
        mx, my = (rx[:n] + lx[:n]) / 2, (ry[:n] + ly[:n]) / 2
        la, lo = from_utm_pair(mx, my)
        roads_out.append({"idx": 0, "name": "road 1",
                          "centerline_latlng": list(zip(la.tolist(), lo.tolist()))})
    lanes_json = {
        "site": site_label,
        "site_id": site_id,
        "lane_width_m": args.lane_width_m,
        "lane_half_width_m": args.lane_width_m / 2,
        "source": lanes_source,
        "sep_mean_m": lanes["sep_mean_m"],
        "sep_median_m": lanes["sep_median_m"],
        "roads": roads_out,
        "lanes": lanes_out,
    }

    frames_json = {
        "site": site_label,
        "site_id": site_id,
        "speed_real_factor": float(site_cfg.get("speed_real_factor", 1.0)),
        "speed_vehicle_factor": float(site_cfg.get("speed_vehicle_factor", 1.0)),
        "marker_px": int(site_cfg.get("marker_px", 14)),
        "marker_label_px": int(site_cfg.get("marker_label_px", 10)),
        "tick_hz": args.tick_hz,
        "duration_s": float(T_end),
        "forward_m": args.forward_m,
        "lane_width_m": args.lane_width_m,
        "heading_tol_deg": args.heading_tol_deg,
        "ego_source": str(args.bag2),
        "emergency_source": str(args.bag1),
        "turnaround": {"bag1": turn1, "bag2": turn2},
        "rtk_quality_note": (
            "carrSoln 0=None 1=Float 2=Fixed. Currently all DGPS (carrSoln=0); "
            "lane assignment is best-effort under ~1m GPS noise."
        ),
        "frames": frames,
    }

    (out_dir / "lanes.json").write_text(json.dumps(lanes_json, indent=2, ensure_ascii=False))
    (out_dir / "frames.json").write_text(json.dumps(frames_json, ensure_ascii=False))
    print(f"[write] {out_dir/'lanes.json'}  ({(out_dir/'lanes.json').stat().st_size/1024:.1f} KB)")
    print(f"[write] {out_dir/'frames.json'} ({(out_dir/'frames.json').stat().st_size/1024:.1f} KB)")

    # ----- emit final CSVs (trimmed + lane-labeled per vehicle) -----
    import csv as _csv
    for name, role in (("ego", "ego"), ("emergency", "emergency")):
        path = out_dir / f"final_{name}.csv"
        with open(path, "w", newline="") as fp:
            w = _csv.writer(fp)
            w.writerow(["scenario_t_s", "lat", "lon", "heading_deg", "speed_mps",
                        "hAcc_m", "carrSoln", "lane", "road_idx",
                        "rel_distance_m", "rel_ahead_m", "rel_lateral_m",
                        "same_lane_alert"])
            for f in frames:
                r = f[role]
                w.writerow([f"{f['t']:.3f}", r["lat"], r["lon"],
                            f"{r['heading']:.2f}", f"{r['speed']:.3f}",
                            f"{r['hAcc']:.3f}", r["carrSoln"],
                            r["lane"] or "", r["road_idx"],
                            f"{f['rel']['distance_m']:.2f}",
                            f"{f['rel']['ahead_m']:.2f}",
                            f"{f['rel']['lateral_m']:.2f}",
                            int(f["same_lane_alert"])])
        print(f"[write] {path}  ({path.stat().st_size/1024:.1f} KB, {len(frames)} rows)")
    # ----- mirror to legacy web/ root for the default site (ochang) so existing
    #       nav.html / editor.html (without ?site=) keep working unchanged.
    if site_cfg.get("mirror_to_web_root"):
        web_root = Path("web")
        web_root.mkdir(parents=True, exist_ok=True)
        for name in ("lanes.json", "frames.json", "raw_trips.json",
                     "final_ego.csv", "final_emergency.csv"):
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, web_root / name)
        print(f"  mirrored {out_dir}/* -> {web_root}/ (legacy default-site path)")

    # ----- write top-level sites.json index for the UI's site picker -----
    sites_index_path = Path("web/sites.json")
    sites_index_path.parent.mkdir(parents=True, exist_ok=True)
    sites_index = json.loads(sites_index_path.read_text()) if sites_index_path.exists() else {
        "default": "ochang", "sites": []
    }
    # Re-derive entry order from SITES registry so it's deterministic.
    entries = []
    for sid, cfg in SITES.items():
        out_rel = cfg.get("out", f"web/sites/{sid}")
        # Strip leading "web/" because the UI fetches relative to nav.html.
        out_rel = out_rel[4:] if out_rel.startswith("web/") else out_rel
        entries.append({
            "id": sid,
            "label": cfg.get("label_short", sid),
            "label_full": cfg.get("label", sid),
            "path": out_rel,
            "ready": (Path(cfg.get("out", f"web/sites/{sid}")) / "frames.json").exists(),
        })
    sites_index = {"default": "ochang", "sites": entries}
    sites_index_path.write_text(json.dumps(sites_index, ensure_ascii=False, indent=2))
    print(f"[write] {sites_index_path}")
    print("[done]")


if __name__ == "__main__":
    sys.exit(main())
