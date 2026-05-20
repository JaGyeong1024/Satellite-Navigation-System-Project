#!/usr/bin/env python3
"""Flatten nav.html's data (web/frames.json + web/lanes.json) into CSVs.

Outputs (next to the source JSONs):
    nav_frames.csv   one row per tick — both vehicles, rel geometry, alerts,
                     forward polygon corners
    nav_lanes.csv    long-format: one row per (lane_id, vertex_idx)

Why two files: frames are time-indexed (363 rows here), lanes are static
geometry (different N per lane). Forcing them into one file means NaN
padding, which kills downstream tooling. Two tidy files > one ugly file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FRAME_COLS = [
    "t",
    "ego_lat", "ego_lon", "ego_heading_deg", "ego_speed_mps",
    "ego_hAcc_m", "ego_carrSoln", "ego_lane", "ego_road_idx", "ego_d_perp_m",
    "emerg_lat", "emerg_lon", "emerg_heading_deg", "emerg_speed_mps",
    "emerg_hAcc_m", "emerg_carrSoln", "emerg_lane", "emerg_road_idx", "emerg_d_perp_m",
    "rel_distance_m", "rel_ahead_m", "rel_lateral_m", "rel_heading_diff_deg",
    "same_lane_ahead", "same_lane_behind", "same_lane_alert",
    "fwd_p1_lat", "fwd_p1_lon",
    "fwd_p2_lat", "fwd_p2_lon",
    "fwd_p3_lat", "fwd_p3_lon",
    "fwd_p4_lat", "fwd_p4_lon",
]


def flatten_frame(fr: dict) -> list:
    e, m, r = fr["ego"], fr["emergency"], fr["rel"]
    fp = fr["forward_polygon_latlng"]
    if len(fp) != 4:
        raise ValueError(f"expected 4-corner forward polygon, got {len(fp)}")
    return [
        fr["t"],
        e["lat"], e["lon"], e["heading"], e["speed"],
        e["hAcc"], e["carrSoln"], e["lane"], e["road_idx"], e["d_perp_m"],
        m["lat"], m["lon"], m["heading"], m["speed"],
        m["hAcc"], m["carrSoln"], m["lane"], m["road_idx"], m["d_perp_m"],
        r["distance_m"], r["ahead_m"], r["lateral_m"], r["heading_diff_deg"],
        int(fr["same_lane_ahead"]),
        int(fr["same_lane_behind"]),
        int(fr["same_lane_alert"]),
        fp[0][0], fp[0][1],
        fp[1][0], fp[1][1],
        fp[2][0], fp[2][1],
        fp[3][0], fp[3][1],
    ]


def write_frames(frames_json_path: Path, out_path: Path) -> int:
    data = json.loads(frames_json_path.read_text())
    rows = [flatten_frame(fr) for fr in data["frames"]]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLS)
        w.writerows(rows)
    return len(rows)


def write_lanes(lanes_json_path: Path, out_path: Path) -> int:
    data = json.loads(lanes_json_path.read_text())
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lane_id", "lane_name", "vertex_idx", "lat", "lon"])
        total = 0
        for lane in data["lanes"]:
            for i, (lat, lon) in enumerate(lane["centerline_latlng"]):
                w.writerow([lane["id"], lane["name"], i, lat, lon])
                total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--web-dir", default="web",
                    help="directory containing frames.json + lanes.json")
    args = ap.parse_args()

    web = Path(args.web_dir)
    frames_csv = web / "nav_frames.csv"
    lanes_csv = web / "nav_lanes.csv"

    n_frames = write_frames(web / "frames.json", frames_csv)
    n_vertices = write_lanes(web / "lanes.json", lanes_csv)

    print(f"[write] {frames_csv}  ({n_frames} rows, {len(FRAME_COLS)} cols)")
    print(f"[write] {lanes_csv}   ({n_vertices} rows)")


if __name__ == "__main__":
    main()
