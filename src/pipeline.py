#!/usr/bin/env python3
"""End-to-end pipeline: bag extracts -> lanes.json + frames.json.

Inputs:
    --bag1   directory of extracted CSVs for the EMERGENCY vehicle
    --bag2   directory of extracted CSVs for the EGO vehicle
    --out    output directory for lanes.json + frames.json (default: web/)

The web UI (nav.html) reads only the two JSONs. To swap in new data later,
re-extract bags and re-run this script — no front-end changes needed.

Architecture (top-down):
    PipelineRunner          orchestrates one site end-to-end
        TripLoader              ublox bag CSV -> Trip
        Turnaround              U-turn / endpoint detection
        KFSmoother              CV-KF + RTS (delegates to gps_kf)
        RoadGeometry            road_setup -> lane polylines + KDTrees
        LaneAssigner            heading-aware lane label w/ hysteresis
        FrameBuilder            per-tick frame computation
        SiteEdits               wraps final.json (trim, overrides, bias)
        OutputWriter            JSON + CSV emission

The math is unchanged vs. the prior procedural version — only the layering
and ownership of state are formalised here. JSON output is byte-identical.
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Module-level CRS transformers (preserved name/signature for external scripts)
# ---------------------------------------------------------------------------

LL_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
UTM_TO_LL = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)


def to_utm(lat, lon):
    """Lat/lon (deg) -> EPSG:32652 East/North (m). Public for external scripts."""
    x, y = LL_TO_UTM.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def from_utm_pair(x, y):
    """EPSG:32652 East/North -> lat/lon."""
    lon, lat = UTM_TO_LL.transform(np.asarray(x), np.asarray(y))
    return np.asarray(lat), np.asarray(lon)


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
        "rtk_quality": "rtk_fixed",
        "has_uturn": False,
        "lane_width_m": 3.5,
        # Original team data was 1 Hz walking; we replay it as ~10 Hz "vehicle"
        # for demo pace. The CV-KF tuned for ochang vehicles rejects every
        # measurement under this scale shift; RTK Fixed is already cm-level so
        # smoothing is unnecessary.
        "skip_kf": True,
        "speed_real_factor": 0.1,
        "speed_vehicle_factor": 1.0,
        "future_path_s": 1.0,
        "future_path_width_m": 0.9,
        "marker_px": 10,
        "marker_label_px": 8,
    },
}


# ---------------------------------------------------------------------------
# Trip — one vehicle's trajectory, mutable in-place.
# Replaces the dict-with-arrays representation. Keeps the same field names
# so downstream code reads as before.
# ---------------------------------------------------------------------------

ARRAY_FIELDS = (
    "t", "lat", "lon", "x", "y",
    "heading_nav", "heading_path",
    "speed", "velE", "velN",
    "hAcc", "sAcc", "carrSoln",
)


class Trip:
    """Per-vehicle trajectory: timestamps + per-sample arrays.

    Mutable in place — bias correction, KF smoothing, trim, and extrapolation
    all rewrite the relevant arrays. `mask()` returns a new sliced Trip.
    """

    __slots__ = ("t0_epoch",) + ARRAY_FIELDS

    def __init__(self, t0_epoch: float = 0.0, **arrays):
        self.t0_epoch = float(t0_epoch)
        for f in ARRAY_FIELDS:
            setattr(self, f, arrays.get(f))

    def mask(self, m: np.ndarray) -> "Trip":
        out = Trip(t0_epoch=self.t0_epoch)
        for f in ARRAY_FIELDS:
            v = getattr(self, f)
            setattr(out, f, v[m] if v is not None else None)
        return out

    def shift_time(self, t0: float) -> None:
        self.t = self.t - t0

    # Compatibility helpers — earlier code addressed Trip as a dict.
    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)


# ---------------------------------------------------------------------------
# TripLoader — bag-extract CSV -> Trip
# ---------------------------------------------------------------------------

class TripLoader:
    """Loads u-blox NavPVT + Fix CSVs into a Trip.

    `max_hacc_m` / `min_carrsoln` are per-sample quality filters applied at
    load time. `add_utm_and_tangent_heading` runs UTM conversion and computes
    a path-tangent heading (more robust than NavPVT COG at low speed).
    """

    def __init__(self, max_hacc_m: float = float("inf"), min_carrsoln: int = 0):
        self._max_hacc = float(max_hacc_m)
        self._min_carrsoln = int(min_carrsoln)

    @staticmethod
    def _csv_time(df, secs="bag_t.secs", nsecs="bag_t.nsecs") -> np.ndarray:
        return (df[secs].to_numpy().astype(np.int64)
                + df[nsecs].to_numpy().astype(np.int64) * 1e-9)

    def load(self, extract_dir: Path) -> Trip:
        fix = pd.read_csv(extract_dir / "ublox_gps__fix.csv")
        pvt = pd.read_csv(extract_dir / "ublox_gps__navpvt.csv")

        t_fix = self._csv_time(fix)
        t0 = float(t_fix[0])
        t = t_fix - t0

        lat = fix["latitude"].to_numpy()
        lon = fix["longitude"].to_numpy()

        t_pvt = self._csv_time(pvt) - t0
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

        keep = ((hAcc_at_fix <= self._max_hacc)
                & (cs_at_fix >= self._min_carrsoln))
        if keep.sum() < len(keep):
            print(f"  filter dropped {(~keep).sum()}/{len(keep)} samples "
                  f"(hAcc>{self._max_hacc}m or carrSoln<{self._min_carrsoln})")

        return Trip(
            t0_epoch=t0,
            t=t[keep],
            lat=lat[keep], lon=lon[keep],
            heading_nav=heading_at_fix[keep],
            speed=speed_at_fix[keep],
            velE=velE_at_fix[keep], velN=velN_at_fix[keep],
            hAcc=hAcc_at_fix[keep], sAcc=sAcc_at_fix[keep],
            carrSoln=cs_at_fix[keep],
        )

    @staticmethod
    def add_utm_and_tangent_heading(trip: Trip) -> Trip:
        x, y = to_utm(trip.lat, trip.lon)
        trip.x = x
        trip.y = y

        dx = np.gradient(x)
        dy = np.gradient(y)
        bearing_deg = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
        w = max(11, (len(bearing_deg) // 50) | 1)
        if len(bearing_deg) > w:
            c = savgol_filter(np.cos(np.radians(bearing_deg)), w, 2)
            s = savgol_filter(np.sin(np.radians(bearing_deg)), w, 2)
            bearing_deg = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
        trip.heading_path = bearing_deg
        return trip


def apply_bias(trip: Trip, dx_east: float, dy_north: float) -> None:
    """Shift UTM positions by (dx, dy) and refresh lat/lon. Heading unchanged."""
    if not (dx_east or dy_north):
        return
    trip.x = trip.x + dx_east
    trip.y = trip.y + dy_north
    lat, lon = from_utm_pair(trip.x, trip.y)
    trip.lat = lat
    trip.lon = lon


# ---------------------------------------------------------------------------
# KFSmoother — CV-KF + RTS, delegates to gps_kf module
# ---------------------------------------------------------------------------

class KFSmoother:
    """Wraps `gps_kf.GPSTrackFilter` for offline batch smoothing of a Trip."""

    def __init__(self, sigma_a_mps2: float = 1.0):
        self._sigma_a = float(sigma_a_mps2)

    def smooth(self, trip: Trip) -> dict:
        from gps_kf import FilterConfig, GPSTrackFilter
        raw_x = trip.x.copy()
        raw_y = trip.y.copy()

        tracker = GPSTrackFilter(FilterConfig(sigma_a_mps2=self._sigma_a))
        ms = [tracker.adapter.build(trip.t[i], trip.x[i], trip.y[i],
                                    trip.velE[i], trip.velN[i],
                                    trip.hAcc[i], trip.sAcc[i],
                                    int(trip.carrSoln[i]))
              for i in range(len(trip.t))]
        result = tracker.run_smoothed(ms)

        trip.x = result.x
        trip.y = result.y
        lat, lon = from_utm_pair(trip.x, trip.y)
        trip.lat = lat
        trip.lon = lon
        trip.speed = result.speed
        trip.heading_path = result.heading_deg

        raw_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([raw_x, raw_y]), axis=0) ** 2)))
        smt_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([result.x, result.y]), axis=0) ** 2)))
        return {
            "raw_step_jitter_m": raw_jit,
            "smoothed_step_jitter_m": smt_jit,
            "median_nis": float(np.median(result.nis)),
            "accepted_pct": float(100 * result.accepted.mean()),
            "n_samples": int(len(result.t)),
        }


# ---------------------------------------------------------------------------
# Turnaround — U-turn / endpoint detection
# ---------------------------------------------------------------------------

class Turnaround:
    """Find the U-turn or use the trip endpoint as the outbound boundary."""

    @staticmethod
    def find(trip: Trip, min_frac: float = 0.2, max_frac: float = 0.8) -> dict:
        t = trip.t
        speed = trip.speed
        heading = trip.heading_path

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

    @staticmethod
    def endpoint(trip: Trip) -> dict:
        return {"i": len(trip.t) - 1, "t": float(trip.t[-1]),
                "speed_at_turn": float(trip.speed[-1]),
                "heading_change_deg": 0.0}


# ---------------------------------------------------------------------------
# Polyline / centerline utilities
# ---------------------------------------------------------------------------

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


def _resample_uniform(poly, ds=1.0):
    seg = np.sqrt(((np.diff(poly, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < ds:
        return poly[:, 0], poly[:, 1], s
    s_u = np.arange(0.0, s[-1], ds)
    xu = np.interp(s_u, s, poly[:, 0])
    yu = np.interp(s_u, s, poly[:, 1])
    return xu, yu, s_u


def _lanes_for_centerline(pts_latlng, lane_w):
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

    Rx, Ry, sR = _resample_uniform(Rsparse)
    Lx, Ly, sL = _resample_uniform(Lsparse)
    Cx, Cy, sC = _resample_uniform(P, ds=1.0)
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


def _snap_junctions(roads_pts, snap_radius_m=3.0):
    """Group endpoints of multiple roads within snap_radius_m and replace
    each cluster with its shared centroid so visual lines meet exactly."""
    if len(roads_pts) < 2:
        return roads_pts
    endpoints = []
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
        print(f"  snapped {n_snapped} endpoints into "
              f"{sum(1 for v in groups.values() if len(v) >= 2)} junctions "
              f"(radius {snap_radius_m:.1f}m)")
    return out


# ---------------------------------------------------------------------------
# RoadGeometry — manual road_setup -> lane polylines + KDTrees
# ---------------------------------------------------------------------------

class RoadGeometry:
    """Manual N-point centerlines turned into R/L lane polylines and KDTrees.

    Constructed from either the new `{"roads": [...]}` schema or the legacy
    single-`centerline_latlng` form. Owns:

      - per-road resampled R/L lane geometry (UTM + arc-length)
      - per-road centerline + unit tangent for heading-aware assignment
      - per-road KDTree(centerline) so a LaneAssigner can resolve quickly
    """

    def __init__(self, road_setup: dict, default_lane_width: float):
        if "roads" in road_setup:
            roads_pts = [r["centerline_latlng"] for r in road_setup["roads"]]
        elif "centerline_latlng" in road_setup:
            roads_pts = [road_setup["centerline_latlng"]]
        else:
            raise ValueError("road_setup has no roads")
        if not roads_pts:
            raise ValueError("road_setup has no roads")

        lane_w = float(road_setup.get("lane_width_m", default_lane_width))
        snap_r = float(road_setup.get("junction_snap_m", 3.0))
        roads_pts = _snap_junctions(roads_pts, snap_radius_m=snap_r)

        per_road = [_lanes_for_centerline(p, lane_w) for p in roads_pts]
        for r, raw_pts in zip(per_road, roads_pts):
            r["center"]["raw_latlng"] = raw_pts

        road_assign = []
        for r in per_road:
            cx, cy = r["center"]["x"], r["center"]["y"]
            road_assign.append({
                "center_xy": np.column_stack([cx, cy]),
                "tan": np.column_stack([r["center"]["tan_x"],
                                        r["center"]["tan_y"]]),
                "tree": cKDTree(np.column_stack([cx, cy])),
            })

        Rx = np.concatenate([r["R"]["x"] for r in per_road])
        Ry = np.concatenate([r["R"]["y"] for r in per_road])
        Lx = np.concatenate([r["L"]["x"] for r in per_road])
        Ly = np.concatenate([r["L"]["y"] for r in per_road])
        total_R = sum((r["R"]["s"][-1] if len(r["R"]["s"]) else 0.0) for r in per_road)
        total_L = sum((r["L"]["s"][-1] if len(r["L"]["s"]) else 0.0) for r in per_road)

        self.lane_width_m = lane_w
        self.lane_half_w = lane_w / 2
        self.per_road = per_road
        self.road_assign = road_assign
        self.lanes = {
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

    @classmethod
    def from_bag1(cls, trip1: Trip, turn1: dict,
                  ds: float = 1.0, edge_trim_samples: int = 30) -> dict:
        """Fallback lane-builder: derive R from outbound, L from return-reversed.
        Returns a `lanes` dict matching the road_setup branch (without per-road
        structure — no heading-aware assignment is possible)."""
        i_turn = turn1["i"]
        i_out_end = max(edge_trim_samples, i_turn - edge_trim_samples)
        i_ret_start = min(len(trip1.x) - edge_trim_samples, i_turn + edge_trim_samples)

        xR_raw = trip1.x[:i_out_end]
        yR_raw = trip1.y[:i_out_end]
        xR, yR, sR = resample_polyline(xR_raw, yR_raw, ds=ds)

        xL_raw = trip1.x[i_ret_start:][::-1]
        yL_raw = trip1.y[i_ret_start:][::-1]
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


# ---------------------------------------------------------------------------
# LaneAssigner — heading-aware lane label with hysteresis
# ---------------------------------------------------------------------------

class LaneAssigner:
    """Assign 1/2-lane label using each road's tangent and the vehicle heading.

    - For each candidate road: nearest centerline point + its tangent.
      Skip if |cos(heading, tangent)| < cos_min (i.e., >60° apart in either
      direction — handles intersections by rejecting wrong-axis roads).
    - Signed cross-track determines side; sign flips when the vehicle drives
      against the centerline's clicked direction.
    - Hysteresis: if |signed-cross| < deadband_m AND the previous frame's
      lane was the OTHER side, stay sticky to the previous lane. Avoids
      label oscillation when driving near the centerline.

    The instance is stateless across `assign()` calls — caller supplies
    `prev_lane`. (Per-vehicle hysteresis lives on FrameBuilder.)
    """

    def __init__(self, road_assign, lane_half_w: float,
                 cos_min: float = 0.5, deadband_m: float = 0.3):
        self._road_assign = road_assign
        self._lane_half_w = float(lane_half_w)
        self._cos_min = float(cos_min)
        self._deadband_m = float(deadband_m)

    def assign(self, x: float, y: float, heading_deg: float,
               prev_lane: str | None = None):
        h = np.radians(heading_deg)
        veh = np.array([np.sin(h), np.cos(h)])
        best = None
        for ri, r in enumerate(self._road_assign):
            if len(r["center_xy"]) == 0:
                continue
            d, i = r["tree"].query([x, y])
            tan = r["tan"][i]
            cos = float(veh @ tan)
            if abs(cos) < self._cos_min:
                continue
            right_of_road = np.array([tan[1], -tan[0]])
            cp = r["center_xy"][i]
            diff = np.array([x - cp[0], y - cp[1]])
            cross = float(diff @ right_of_road)
            veh_right_signed_d = cross if cos > 0 else -cross
            new_lane = "2차선" if veh_right_signed_d > 0 else "1차선"
            if (prev_lane and abs(veh_right_signed_d) < self._deadband_m
                    and prev_lane != new_lane):
                lane = prev_lane
            else:
                lane = new_lane
            d_perp = abs(cross)
            in_lane = d_perp <= self._lane_half_w
            score = (0 if in_lane else 1, d_perp)
            cand = {"lane": lane, "d_perp_m": d_perp, "in_lane": in_lane,
                    "road_idx": ri, "score": score}
            if best is None or cand["score"] < best["score"]:
                best = cand
        return best


# ---------------------------------------------------------------------------
# Resampling onto a uniform time grid
# ---------------------------------------------------------------------------

def resample_trip_to_grid(trip: Trip, t_grid: np.ndarray) -> dict:
    """Linear-interp Trip onto a common time grid. Returns a plain dict
    (consumed by FrameBuilder)."""
    out = {"t": t_grid}
    for key in ("lat", "lon", "speed", "hAcc"):
        out[key] = np.interp(t_grid, trip.t, getattr(trip, key))
    rad = np.radians(trip.heading_path)
    c = np.interp(t_grid, trip.t, np.cos(rad))
    s = np.interp(t_grid, trip.t, np.sin(rad))
    out["heading"] = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
    out["x"] = np.interp(t_grid, trip.t, trip.x)
    out["y"] = np.interp(t_grid, trip.t, trip.y)
    out["carrSoln"] = np.round(np.interp(t_grid, trip.t,
                                          trip.carrSoln)).astype(int)
    return out


def _apply_override(t: float, overrides, fallback):
    for ov in overrides:
        if ov["start_s"] <= t <= ov["end_s"]:
            return ov["lane"]
    return fallback


# ---------------------------------------------------------------------------
# FrameBuilder — per-tick frames
# ---------------------------------------------------------------------------

@dataclass
class FrameConfig:
    forward_m: float = 10.0
    rear_m: float = 30.0
    lane_half_w: float = 1.75
    heading_tol_deg: float = 30.0
    overlay_behind_m: float = 25.0
    overlay_ahead_m: float = 5.0
    future_path_s: float = 1.0
    future_path_width_m: float = 2.2
    tick_hz: float = 10.0


class FrameBuilder:
    """Builds the per-tick frame list from resampled ego + emergency arrays.

    Encapsulates:
      - heading-aware LaneAssigner + per-vehicle hysteresis state
      - forward polygon (rectangle aligned with ego's heading)
      - ego/emergency lane band overlays (centerline ± half-width)
      - future-path bands (PCHIP-smoothed independent left/right edges)
    """

    def __init__(self, lanes: dict, cfg: FrameConfig):
        self._lanes = lanes
        self._cfg = cfg
        self._has_road_assign = "road_assign" in lanes
        self._assigner = (
            LaneAssigner(lanes["road_assign"], cfg.lane_half_w)
            if self._has_road_assign else None
        )

    # -------- helpers --------

    def _future_path_band(self, xs, ys, hs_deg, i_start, n_ahead, width_m,
                          n_control_points=4, n_fine=40):
        """Future-path band with INDEPENDENT left/right edge fitting.

        Architecture:
          1. Caller pre-smooths the ENTIRE trip once (time-axis consistency).
          2. Pick sparse control points along this future window.
          3. Compute left/right edge points at each control point via
             perpendicular offset of the local control-point tangent.
          4. PCHIP-fit left+right edges INDEPENDENTLY (arc-length parameterised).
             Each boundary becomes smooth on its own without tangent-noise
             propagating through the perpendicular offset.
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

        dx = np.gradient(cx); dy = np.gradient(cy)
        L = np.hypot(dx, dy); L[L < 1e-9] = 1e-9
        tx, ty = dx / L, dy / L
        rx, ry = ty, -tx
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

    def _lane_band(self, road_idx, x, y, fwd_unit, behind_m, ahead_m, half_width):
        """Closed polygon over the vehicle's CURRENT lane from behind_m
        behind to ahead_m ahead, half_width perpendicular each side."""
        lanes = self._lanes
        if (road_idx < 0 or "per_road" not in lanes
                or road_idx >= len(lanes["per_road"])):
            return []
        rl = lanes["per_road"][road_idx]
        cx = rl["center"]["x"]; cy = rl["center"]["y"]
        if len(cx) < 2:
            return []
        tx_arr = rl["center"]["tan_x"]; ty_arr = rl["center"]["tan_y"]
        dist = np.hypot(cx - x, cy - y)
        i_min = int(np.argmin(dist))
        tx, ty = tx_arr[i_min], ty_arr[i_min]
        cos_align = float(fwd_unit[0] * tx + fwd_unit[1] * ty)
        rx_arr = ty_arr; ry_arr = -tx_arr
        cross = (x - cx[i_min]) * rx_arr[i_min] + (y - cy[i_min]) * ry_arr[i_min]
        lane_center_sign = 1 if cross >= 0 else -1
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
        left_edge  = lane_center + half_width * rseg
        right_edge = lane_center - half_width * rseg
        polygon = np.vstack([left_edge, right_edge[::-1]])
        la, lo = from_utm_pair(polygon[:, 0], polygon[:, 1])
        return list(zip(la.tolist(), lo.tolist()))

    # -------- main entry --------

    def build(self, ego: dict, emerg: dict,
              ego_overrides: list | None = None,
              emerg_overrides: list | None = None) -> list:
        cfg = self._cfg
        future_steps = int(round(cfg.future_path_s * cfg.tick_hz))
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
            fwd = np.array([sin_h, cos_h])
            right = np.array([cos_h, -sin_h])

            # Forward 4-corner polygon (bottom-L, bottom-R, top-R, top-L).
            corner_order = [
                (0.0, -cfg.lane_half_w),
                (0.0, +cfg.lane_half_w),
                (cfg.forward_m, +cfg.lane_half_w),
                (cfg.forward_m, -cfg.lane_half_w),
            ]
            corners_xy = [(xe + s * fwd[0] + d * right[0],
                           ye + s * fwd[1] + d * right[1])
                          for s, d in corner_order]
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

            # Lane assignment (heading-aware) — must run BEFORE same_lane check.
            if self._has_road_assign:
                ego_a = self._assigner.assign(xe, ye, he_deg, prev_ego_lane)
                emerg_a = self._assigner.assign(xm, ym, hm_deg, prev_emerg_lane)
                ego_lane_auto = ego_a["lane"] if ego_a else None
                emerg_lane_auto = emerg_a["lane"] if emerg_a else None
                prev_ego_lane = ego_lane_auto if ego_lane_auto else prev_ego_lane
                prev_emerg_lane = emerg_lane_auto if emerg_lane_auto else prev_emerg_lane
                d_eR = ego_a["d_perp_m"] if ego_a else float("nan")
                d_mR = emerg_a["d_perp_m"] if emerg_a else float("nan")
                ego_road_idx = ego_a["road_idx"] if ego_a else -1
                emerg_road_idx = emerg_a["road_idx"] if emerg_a else -1
            else:
                d_eR, _ = self._lanes["tree_R"].query([xe, ye])
                d_eL, _ = self._lanes["tree_L"].query([xe, ye])
                ego_lane_auto = "2차선" if d_eR < d_eL else "1차선"
                d_mR, _ = self._lanes["tree_R"].query([xm, ym])
                d_mL, _ = self._lanes["tree_L"].query([xm, ym])
                emerg_lane_auto = "2차선" if d_mR < d_mL else "1차선"
                ego_road_idx = 0
                emerg_road_idx = 0

            ego_lane = _apply_override(float(ego["t"][i]),
                                       ego_overrides, ego_lane_auto)
            emerg_lane = _apply_override(float(emerg["t"][i]),
                                         emerg_overrides, emerg_lane_auto)

            on_same_road = (ego_road_idx == emerg_road_idx
                            and ego_road_idx >= 0
                            and ego_lane is not None
                            and emerg_lane is not None)
            topo_same_lane = on_same_road and (ego_lane == emerg_lane)
            rel_same_lane = (abs(d_em) <= cfg.lane_half_w
                             and head_diff <= cfg.heading_tol_deg)
            same_lane = topo_same_lane if self._has_road_assign else rel_same_lane
            same_lane_ahead  = bool(same_lane and  0.0 <= s_em <= cfg.forward_m)
            same_lane_behind = bool(same_lane and -cfg.rear_m <= s_em <  0.0)
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
                "ego_lane_band_latlng": self._lane_band(
                    ego_road_idx, xe, ye, fwd,
                    cfg.overlay_behind_m, cfg.overlay_ahead_m, cfg.lane_half_w),
                "emerg_lane_band_latlng": self._lane_band(
                    emerg_road_idx, xm, ym,
                    np.array([np.sin(np.radians(hm_deg)),
                              np.cos(np.radians(hm_deg))]),
                    0.0, 30.0, cfg.lane_half_w),
                "ego_future_band_latlng": self._future_path_band(
                    ego["x"], ego["y"], ego["heading"], i, future_steps,
                    cfg.future_path_width_m),
                "emerg_future_band_latlng": self._future_path_band(
                    emerg["x"], emerg["y"], emerg["heading"], i, future_steps,
                    cfg.future_path_width_m),
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


# ---------------------------------------------------------------------------
# SiteEdits — final.json wrapper
# ---------------------------------------------------------------------------

class SiteEdits:
    """Wrapper around final.json content (road_setup + per-bag edits).

    Methods are intentionally narrow: they each pull one piece (bias,
    trim window, lane overrides, extrapolation, speed factor) so the
    runner reads as a sequence of explicit applications.
    """

    def __init__(self, raw: dict):
        self.raw = raw or {}

    @classmethod
    def load_first(cls, candidates: Iterable[Path]) -> "SiteEdits":
        for p in candidates:
            if p.exists():
                d = json.loads(p.read_text())
                print(f"  loaded {p}: keys {list(d.keys())}")
                return cls(d)
        print(f"  (no final.json found — looked at {[str(c) for c in candidates]})")
        return cls({})

    @property
    def road_setup(self):
        return self.raw.get("road_setup")

    def bias(self):
        rs = self.road_setup or {}
        b = rs.get("bias_m", {})
        return float(b.get("dx_east", 0)), float(b.get("dy_north", 0))

    def trim(self, key: str):
        return self.raw.get(key, {}).get("trim", {})

    def lane_overrides(self, key: str):
        return self.raw.get(key, {}).get("lane_overrides", [])

    def speed_factor(self, key: str) -> float:
        return float(self.raw.get(key, {}).get("speed_factor", 1.0))

    def extrapolate_after_m(self, key: str) -> float:
        return float(self.raw.get(key, {}).get("extrapolate_after_m", 0.0))

    def extrapolate_speed_mps(self, key: str):
        return self.raw.get(key, {}).get("extrapolate_speed_mps")


# ---------------------------------------------------------------------------
# Trip post-processing operations (trim, extrapolate, speed-factor)
# Free functions — they mutate Trip in place and are called once each.
# ---------------------------------------------------------------------------

def downsample_for_editor(trip: Trip, target_hz: int = 5) -> dict:
    dt = float(np.median(np.diff(trip.t)))
    step = max(1, int(round(1.0 / target_hz / dt)))
    keep = slice(None, None, step)
    return {
        "t": trip.t[keep].tolist(),
        "lat": trip.lat[keep].tolist(),
        "lon": trip.lon[keep].tolist(),
        "speed": trip.speed[keep].tolist(),
        "heading": trip.heading_path[keep].tolist(),
        "hAcc": trip.hAcc[keep].tolist(),
    }


def trim_trip(trip: Trip, edits: SiteEdits, key: str, turn: dict,
              start_speed_mps: float, start_sustain_s: float
              ) -> tuple[Trip, str, float]:
    """Apply trim window (manual override or auto leading-stationary + outbound)."""
    ed = edits.trim(key)
    if "start_s" in ed and "end_s" in ed:
        s, e = float(ed["start_s"]), float(ed["end_s"])
        m = (trip.t >= s) & (trip.t <= e)
        src = "manual"
    else:
        sp = trip.speed
        dt = float(np.median(np.diff(trip.t)))
        win = max(1, int(round(start_sustain_s / dt)))
        moving = sp > start_speed_mps
        i_start = 0
        for i in range(len(moving) - win + 1):
            if moving[i:i + win].all():
                i_start = i
                break
        t_start = float(trip.t[i_start])
        m = (trip.t >= t_start) & (trip.t <= turn["t"])
        src = "auto"
    kept_start = float(trip.t[m][0]) if m.any() else 0.0
    out = trip.mask(m)
    out.t = out.t - out.t[0]
    return out, src, kept_start


def apply_speed_factor(trip: Trip, edits: SiteEdits, key: str) -> float:
    f = edits.speed_factor(key)
    if f != 1.0:
        trip.t = trip.t / f
    return f


def extrapolate_trip(trip: Trip, edits: SiteEdits, key: str) -> float:
    """Linearly extrapolate past trim end with last heading × cruise speed."""
    extra_m = edits.extrapolate_after_m(key)
    if extra_m <= 0:
        return 0.0
    last_h_deg = float(trip.heading_path[-1])
    override = edits.extrapolate_speed_mps(key)
    if override:
        last_speed = float(override)
    else:
        moving = trip.speed > 1.0
        cruise = float(np.median(trip.speed[moving])) if moving.any() else 3.0
        last_speed = max(cruise, 3.0)
    duration = extra_m / last_speed
    dt = float(np.median(np.diff(trip.t))) if len(trip.t) > 1 else 0.1
    n = max(2, int(round(duration / dt)))
    h_rad = np.radians(last_h_deg)
    vx, vy = np.sin(h_rad) * last_speed, np.cos(h_rad) * last_speed
    t_extra = trip.t[-1] + np.arange(1, n + 1) * dt
    dt_step = t_extra - trip.t[-1]
    x_extra = trip.x[-1] + vx * dt_step
    y_extra = trip.y[-1] + vy * dt_step
    lat_extra, lon_extra = from_utm_pair(x_extra, y_extra)
    trip.t  = np.concatenate([trip.t,  t_extra])
    trip.x  = np.concatenate([trip.x,  x_extra])
    trip.y  = np.concatenate([trip.y,  y_extra])
    trip.lat = np.concatenate([trip.lat, lat_extra])
    trip.lon = np.concatenate([trip.lon, lon_extra])
    trip.heading_path = np.concatenate(
        [trip.heading_path, np.full(n, last_h_deg)])
    trip.speed = np.concatenate([trip.speed, np.full(n, last_speed)])
    trip.hAcc  = np.concatenate([trip.hAcc,  np.full(n, trip.hAcc[-1])])
    trip.carrSoln = np.concatenate(
        [trip.carrSoln, np.full(n, trip.carrSoln[-1])])
    return extra_m


# ---------------------------------------------------------------------------
# OutputWriter — emit the same JSON+CSV files the front-end expects
# ---------------------------------------------------------------------------

class OutputWriter:
    """Pure I/O. The shapes are pinned here so nav.html and editor.html keep
    working unchanged. Any front-end-facing schema change should pass through
    this class."""

    @staticmethod
    def write_raw_trips(out_dir: Path, trip1: Trip, trip2: Trip,
                        turn1: dict, turn2: dict,
                        bag1_src: str, bag2_src: str) -> None:
        raw_trips_json = {
            "bag1": {
                "name": "emergency", "source": bag1_src,
                "t0_epoch": trip1.t0_epoch,
                "duration_s": float(trip1.t[-1]),
                "auto": {"turnaround_t": turn1["t"]},
                "data": downsample_for_editor(trip1),
            },
            "bag2": {
                "name": "ego", "source": bag2_src,
                "t0_epoch": trip2.t0_epoch,
                "duration_s": float(trip2.t[-1]),
                "auto": {"turnaround_t": turn2["t"]},
                "data": downsample_for_editor(trip2),
            },
        }
        path = out_dir / "raw_trips.json"
        path.write_text(json.dumps(raw_trips_json, ensure_ascii=False))
        print(f"  wrote {path} ({path.stat().st_size/1024:.1f} KB)")

    @staticmethod
    def _lane_latlng(lane: dict) -> list:
        la, lo = from_utm_pair(lane["x"], lane["y"])
        return list(zip(la.tolist(), lo.tolist()))

    @classmethod
    def write_lanes_and_frames(cls, out_dir: Path,
                                lanes: dict, frames: list,
                                site_label: str, site_id: str | None,
                                site_cfg: dict, args, T_end: float,
                                turn1: dict, turn2: dict,
                                lanes_source: str) -> None:
        lanes_out = []
        roads_out = []
        if "per_road" in lanes:
            for ri, road_lanes in enumerate(lanes["per_road"]):
                lanes_out.append({"id": "R", "road_idx": ri,
                                  "name": f"lane R (road {ri + 1})",
                                  "centerline_latlng": cls._lane_latlng(road_lanes["R"])})
                lanes_out.append({"id": "L", "road_idx": ri,
                                  "name": f"lane L (road {ri + 1})",
                                  "centerline_latlng": cls._lane_latlng(road_lanes["L"])})
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
                 "centerline_latlng": cls._lane_latlng(lanes["R"])},
                {"id": "L", "road_idx": 0, "name": "lane L",
                 "centerline_latlng": cls._lane_latlng(lanes["L"])},
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
        (out_dir / "lanes.json").write_text(
            json.dumps(lanes_json, indent=2, ensure_ascii=False))
        (out_dir / "frames.json").write_text(
            json.dumps(frames_json, ensure_ascii=False))
        print(f"[write] {out_dir/'lanes.json'}  "
              f"({(out_dir/'lanes.json').stat().st_size/1024:.1f} KB)")
        print(f"[write] {out_dir/'frames.json'} "
              f"({(out_dir/'frames.json').stat().st_size/1024:.1f} KB)")

    @staticmethod
    def write_final_csvs(out_dir: Path, frames: list) -> None:
        for name, role in (("ego", "ego"), ("emergency", "emergency")):
            path = out_dir / f"final_{name}.csv"
            with open(path, "w", newline="") as fp:
                w = _csv.writer(fp)
                w.writerow(["scenario_t_s", "lat", "lon", "heading_deg",
                            "speed_mps", "hAcc_m", "carrSoln", "lane",
                            "road_idx", "rel_distance_m", "rel_ahead_m",
                            "rel_lateral_m", "same_lane_alert"])
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
            print(f"[write] {path}  ({path.stat().st_size/1024:.1f} KB, "
                  f"{len(frames)} rows)")

    @staticmethod
    def write_sites_index() -> None:
        sites_index_path = Path("web/sites.json")
        sites_index_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for sid, cfg in SITES.items():
            out_rel = cfg.get("out", f"web/sites/{sid}")
            out_rel = out_rel[4:] if out_rel.startswith("web/") else out_rel
            entries.append({
                "id": sid,
                "label": cfg.get("label_short", sid),
                "label_full": cfg.get("label", sid),
                "path": out_rel,
                "ready": (Path(cfg.get("out", f"web/sites/{sid}"))
                          / "frames.json").exists(),
            })
        sites_index = {"default": "ochang", "sites": entries}
        sites_index_path.write_text(json.dumps(sites_index,
                                              ensure_ascii=False, indent=2))
        print(f"[write] {sites_index_path}")


# ---------------------------------------------------------------------------
# PipelineRunner — orchestration
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Runs one site end-to-end. Each phase is a method so the high-level
    sequence stays readable, and individual steps are testable in isolation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.site_id = args.site or ("ochang"
                                     if (args.bag1 and "gps1" in args.bag1)
                                     else None)
        self.site_cfg = SITES.get(args.site, {}) if args.site else {}

    # -------- phase: resolve & report --------

    def _resolve_paths(self) -> tuple[str, str, Path, Path, bool, bool, str]:
        args = self.args
        input_cfg = self.site_cfg.get("input", {})
        bag1 = args.bag1 or input_cfg.get("bag1")
        bag2 = args.bag2 or input_cfg.get("bag2")
        if not (bag1 and bag2):
            raise SystemExit("must provide either --site or both --bag1 and --bag2")
        out_path = args.out if args.out != "web" else self.site_cfg.get("out", "web")
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        final_path = (Path(args.final) if args.final
                      else Path(self.site_cfg.get("final", "final.json")))
        has_uturn = self.site_cfg.get("has_uturn", True)
        skip_kf_site = self.site_cfg.get("skip_kf", False)
        if self.site_cfg.get("lane_width_m"):
            args.lane_width_m = self.site_cfg["lane_width_m"]
        if self.site_cfg.get("future_path_s") is not None:
            args.future_path_s = self.site_cfg["future_path_s"]
        if self.site_cfg.get("future_path_width_m") is not None:
            args.future_path_width_m = self.site_cfg["future_path_width_m"]
        site_label = (args.site_label or self.site_cfg.get("label")
                      or "ochang_ctrack_outer")
        print(f"[site] id={self.site_id or 'manual'} out={out_dir}  "
              f"has_uturn={has_uturn}  skip_kf={skip_kf_site}")
        # NOTE: args.bag1/bag2 intentionally left as user typed them (None for
        # --site mode). raw_trips.json and frames.json serialize str(args.bag1)
        # directly to preserve the prior wire format ("None" when --site used).
        return bag1, bag2, out_dir, final_path, has_uturn, skip_kf_site, site_label

    # -------- phase: load + UTM --------

    def _load_trip(self, path: str, label: str) -> Trip:
        loader = TripLoader(self.args.max_hacc_m, self.args.min_carrsoln)
        print(f"[load] {label} = {path}")
        trip = loader.load(Path(path))
        TripLoader.add_utm_and_tangent_heading(trip)
        carr_counts = dict(zip(*np.unique(trip.carrSoln, return_counts=True)))
        print(f"  {len(trip.t)} samples, duration {trip.t[-1]:.1f}s, "
              f"hAcc median {np.median(trip.hAcc):.3f}m, "
              f"carrSoln dist {carr_counts}")
        return trip

    # -------- phase: turnaround --------

    def _turnarounds(self, trip1: Trip, trip2: Trip, has_uturn: bool):
        print("[turnaround]")
        if has_uturn:
            turn1 = Turnaround.find(trip1)
            turn2 = Turnaround.find(trip2)
        else:
            turn1 = Turnaround.endpoint(trip1)
            turn2 = Turnaround.endpoint(trip2)
        print(f"  bag1: t={turn1['t']:.1f}s, "
              f"speed_min={turn1['speed_at_turn']:.2f}m/s, "
              f"heading change={turn1['heading_change_deg']:.0f}°")
        print(f"  bag2: t={turn2['t']:.1f}s, "
              f"speed_min={turn2['speed_at_turn']:.2f}m/s, "
              f"heading change={turn2['heading_change_deg']:.0f}°")
        return turn1, turn2

    # -------- phase: lanes --------

    def _build_lanes(self, edits: SiteEdits, trip1: Trip, turn1: dict,
                     has_uturn: bool) -> tuple[dict, str]:
        args = self.args
        road_setup = edits.road_setup
        if not road_setup and not has_uturn:
            # No manual road yet — synthesize a centerline from bag1 path so
            # the editor has something to render. User refines in editor.html.
            pts = list(zip(trip1.lat.tolist(), trip1.lon.tolist()))
            step = max(1, len(pts) // 20)
            pts = pts[::step]
            print(f"  no road_setup — using bag1 path as default centerline "
                  f"({len(pts)} pts); refine via editor.html?site={self.site_id}")
            road_setup = {"roads": [{"centerline_latlng": pts}],
                          "lane_width_m": args.lane_width_m}

        print("[lanes]")
        if road_setup:
            geom = RoadGeometry(road_setup, args.lane_width_m)
            args.lane_width_m = geom.lane_width_m
            lanes = geom.lanes
            lanes_source = "manual road_setup"
        else:
            lanes = RoadGeometry.from_bag1(trip1, turn1)
            lanes_source = "bag1 outbound (R) + bag1 return reversed (L)"
        print(f"  source: {lanes_source}")
        print(f"  lane R: {len(lanes['R']['x'])} pts ({lanes['R']['s'][-1]:.1f}m)")
        print(f"  lane L: {len(lanes['L']['x'])} pts ({lanes['L']['s'][-1]:.1f}m)")
        print(f"  R-L separation: mean {lanes['sep_mean_m']:.2f}m, "
              f"median {lanes['sep_median_m']:.2f}m  "
              f"(expected ~{args.lane_width_m:.1f}m)")
        return lanes, lanes_source

    # -------- phase: frame build --------

    def _build_frames(self, lanes: dict, trip1: Trip, trip2: Trip,
                      turn1: dict, turn2: dict,
                      edits: SiteEdits) -> tuple[list, float]:
        args = self.args
        print("[frames]")

        trip1_out, src1, t1_start = trim_trip(
            trip1, edits, "bag1", turn1,
            args.start_speed_mps, args.start_sustain_s)
        trip2_out, src2, t2_start = trim_trip(
            trip2, edits, "bag2", turn2,
            args.start_speed_mps, args.start_sustain_s)

        f1 = apply_speed_factor(trip1_out, edits, "bag1")
        f2 = apply_speed_factor(trip2_out, edits, "bag2")
        ex1 = extrapolate_trip(trip1_out, edits, "bag1")
        ex2 = extrapolate_trip(trip2_out, edits, "bag2")
        if ex1 or ex2:
            print(f"  extrapolation: bag1 +{ex1:.0f}m, bag2 +{ex2:.0f}m")
        sf1 = f" ×{f1:g} speed" if f1 != 1.0 else ""
        sf2 = f" ×{f2:g} speed" if f2 != 1.0 else ""
        print(f"  bag1 trim: {src1}, starts at {t1_start:.1f}s, "
              f"kept {trip1_out.t[-1]:.1f}s{sf1}")
        print(f"  bag2 trim: {src2}, starts at {t2_start:.1f}s, "
              f"kept {trip2_out.t[-1]:.1f}s{sf2}")

        T_end = max(trip1_out.t[-1], trip2_out.t[-1])
        dt = 1.0 / args.tick_hz
        t_grid = np.arange(0.0, T_end + dt, dt)

        e1 = resample_trip_to_grid(trip1_out, t_grid)
        e2 = resample_trip_to_grid(trip2_out, t_grid)

        # Heavy time-axis smoothing of resampled UTM positions so future-path
        # bands are consistent across consecutive frames (each frame slices a
        # window from these pre-smoothed arrays — no in-frame savgol).
        # Resync lat/lon from smoothed UTM so marker and polygon agree at
        # trip boundaries (savgol edge behaviour would otherwise drift them).
        for arr in (e1, e2):
            n = len(arr["x"])
            if n >= 31:
                arr["x"] = savgol_filter(arr["x"], 31, 2)
                arr["y"] = savgol_filter(arr["y"], 31, 2)
                arr["lat"], arr["lon"] = from_utm_pair(arr["x"], arr["y"])

        def shift(ovs, kept_start):
            return [{"start_s": o["start_s"] - kept_start,
                     "end_s": o["end_s"] - kept_start,
                     "lane": o["lane"]} for o in ovs]
        ego_ov   = shift(edits.lane_overrides("bag2"), t2_start)
        emerg_ov = shift(edits.lane_overrides("bag1"), t1_start)

        cfg = FrameConfig(
            forward_m=args.forward_m,
            rear_m=args.rear_m,
            lane_half_w=args.lane_width_m / 2,
            heading_tol_deg=args.heading_tol_deg,
            overlay_behind_m=args.overlay_behind_m,
            overlay_ahead_m=args.overlay_ahead_m,
            future_path_s=args.future_path_s,
            future_path_width_m=args.future_path_width_m,
            tick_hz=args.tick_hz,
        )
        builder = FrameBuilder(lanes, cfg)
        frames = builder.build(e2, e1, ego_overrides=ego_ov,
                                emerg_overrides=emerg_ov)

        n_ahead  = sum(1 for f in frames if f["same_lane_ahead"])
        n_behind = sum(1 for f in frames if f["same_lane_behind"])
        n_any    = sum(1 for f in frames if f["same_lane_alert"])
        print(f"  {len(frames)} frames @ {args.tick_hz:.0f}Hz · "
              f"alert {n_any} ({n_any/max(1,len(frames))*100:.0f}%) = "
              f"ahead {n_ahead} + behind {n_behind}")
        return frames, T_end

    # -------- top-level run() --------

    def run(self) -> int:
        args = self.args
        bag1, bag2, out_dir, final_path, has_uturn, skip_kf_site, site_label = \
            self._resolve_paths()

        trip1 = self._load_trip(bag1, "bag1 (emergency)")
        trip2 = self._load_trip(bag2, "bag2 (ego)")
        turn1, turn2 = self._turnarounds(trip1, trip2, has_uturn)

        OutputWriter.write_raw_trips(out_dir, trip1, trip2, turn1, turn2,
                                     str(args.bag1), str(args.bag2))

        if args.site or args.final:
            candidates = [final_path]
        else:
            candidates = [Path("final.json"), Path("edit.json"),
                          out_dir / "final.json", out_dir / "edit.json"]
        edits = SiteEdits.load_first(candidates)

        dx, dy = edits.bias()
        if dx or dy:
            apply_bias(trip1, dx, dy)
            apply_bias(trip2, dx, dy)
            print(f"  applied bias correction dx={dx:+.3f}m dy={dy:+.3f}m")

        if not args.no_kf and not skip_kf_site:
            smoother = KFSmoother(args.kf_sigma_a)
            diag1 = smoother.smooth(trip1)
            diag2 = smoother.smooth(trip2)
            print(f"[kf+rts] bag1: {diag1['n_samples']} samples, "
                  f"median NIS {diag1['median_nis']:.2f}, "
                  f"jitter {diag1['raw_step_jitter_m']:.3f}"
                  f"→{diag1['smoothed_step_jitter_m']:.3f}m")
            print(f"[kf+rts] bag2: {diag2['n_samples']} samples, "
                  f"median NIS {diag2['median_nis']:.2f}, "
                  f"jitter {diag2['raw_step_jitter_m']:.3f}"
                  f"→{diag2['smoothed_step_jitter_m']:.3f}m")

        lanes, lanes_source = self._build_lanes(edits, trip1, turn1, has_uturn)
        frames, T_end = self._build_frames(lanes, trip1, trip2, turn1, turn2,
                                            edits)

        OutputWriter.write_lanes_and_frames(
            out_dir, lanes, frames, site_label, self.site_id,
            self.site_cfg, args, T_end, turn1, turn2, lanes_source)
        OutputWriter.write_final_csvs(out_dir, frames)

        if self.site_cfg.get("mirror_to_web_root"):
            web_root = Path("web")
            web_root.mkdir(parents=True, exist_ok=True)
            for name in ("lanes.json", "frames.json", "raw_trips.json",
                         "final_ego.csv", "final_emergency.csv"):
                src = out_dir / name
                if src.exists():
                    shutil.copy2(src, web_root / name)
            print(f"  mirrored {out_dir}/* -> {web_root}/ (legacy default-site path)")

        OutputWriter.write_sites_index()
        print("[done]")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
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
    return ap.parse_args()


def main() -> int:
    return PipelineRunner(_parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
