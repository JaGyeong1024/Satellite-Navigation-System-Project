#!/usr/bin/env python3
"""Build the ego-camera output for a site.

Single-source convention:
  - INPUT  (immutable):  data/sites/<site>/source/camera/<epoch>.jpg
                         data/sites/<site>/source/<bag2>/ublox_gps__fix.csv
                         data/sites/<site>/final.json (bag2.trim)
  - OUTPUT (rebuilt):    web/sites/<site>/ego_camera/<epoch>.jpg
                         web/sites/<site>/ego_camera_index.json

All transforms (trim windowing, crop, resampling) happen HERE — at the
processing layer between the immutable source and the UI. Adjusting crop
or any filter regenerates the web output; the source is never touched.

Usage:
    python3 build_ego_camera.py --site ochang [--crop 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import SITES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=list(SITES.keys()))
    ap.add_argument("--crop", type=int, default=10,
                    help="uniform pixels to crop from each edge of source JPG (default 10)")
    ap.add_argument("--crop-x", type=int, default=None,
                    help="override left+right crop (each side). Defaults to --crop value")
    ap.add_argument("--crop-y", type=int, default=None,
                    help="override top+bottom crop (each side). Defaults to --crop value")
    ap.add_argument("--quality", type=int, default=92,
                    help="JPEG quality for output (default 92)")
    args = ap.parse_args()

    cfg = SITES[args.site]
    if cfg["input"]["type"] != "bag_extract":
        sys.exit(f"site {args.site}: no bag-extract camera data")

    site_root = Path("data/sites") / args.site
    src_camera = site_root / "source" / "camera"
    if not src_camera.is_dir():
        sys.exit(f"no source camera dir at {src_camera}")

    ego_dir = Path(cfg["input"]["bag2"])
    fix_df = pd.read_csv(ego_dir / "ublox_gps__fix.csv")
    secs = fix_df["bag_t.secs"].to_numpy()
    nsecs = fix_df["bag_t.nsecs"].to_numpy()
    epoch = secs + nsecs * 1e-9
    t0_epoch = float(epoch[0])
    trip_t = epoch - t0_epoch

    final = json.loads(Path(cfg["final"]).read_text())
    trim = final.get("bag2", {}).get("trim", {})
    trim_start = float(trim.get("start_s", 0.0))
    trim_end = float(trim.get("end_s", 1e9))
    kept_mask = (trip_t >= trim_start) & (trip_t <= trim_end)
    if not kept_mask.any():
        sys.exit(f"no GPS samples in trim window [{trim_start}, {trim_end}]")
    epoch_lo = float(epoch[kept_mask][0])
    epoch_hi = float(epoch[kept_mask][-1])

    # Scan source JPGs, filter by epoch, transform, write to web output dir.
    out_dir = Path(cfg["out"]) / "ego_camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous output (idempotent rebuild — UI always reflects latest config)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    crop_x = args.crop_x if args.crop_x is not None else args.crop
    crop_y = args.crop_y if args.crop_y is not None else args.crop

    jpgs = sorted(src_camera.glob("*.jpg"))
    print(f"[ego_camera build] site={args.site}")
    print(f"  source: {src_camera}  ({len(jpgs)} JPGs)")
    print(f"  trim epoch [{epoch_lo:.3f} .. {epoch_hi:.3f}]")
    if crop_x > 0 or crop_y > 0:
        print(f"  transform: crop x={crop_x}px (L/R), y={crop_y}px (T/B) each side")

    kept = []
    for jpg in jpgs:
        try:
            sec_str, nsec_str = jpg.stem.split("_")
            ep = int(sec_str) + int(nsec_str) * 1e-9
        except ValueError:
            continue
        if not (epoch_lo <= ep <= epoch_hi):
            continue
        kept.append((ep, jpg))

    if not kept:
        sys.exit("no source JPGs match trim window")

    total_bytes = 0
    needs_transform = crop_x > 0 or crop_y > 0
    for i, (ep, jpg) in enumerate(kept):
        dst = out_dir / jpg.name
        if needs_transform:
            with Image.open(jpg) as im:
                w, h = im.size
                im.crop((crop_x, crop_y, w - crop_x, h - crop_y)).save(
                    dst, quality=args.quality, optimize=True)
        else:
            # No transform — straight byte copy (cheaper than PIL roundtrip)
            dst.write_bytes(jpg.read_bytes())
        total_bytes += dst.stat().st_size
        if i % 200 == 0 or i == len(kept) - 1:
            print(f"  {i + 1:>5}/{len(kept)}", end="\r", flush=True)
    print()

    index = [
        {"t": round(ep - epoch_lo, 4), "file": f"ego_camera/{jpg.name}"}
        for ep, jpg in kept
    ]
    # Probe one output frame for exact dimensions so the UI can size its
    # container to match (no object-fit cropping).
    with Image.open(out_dir / kept[0][1].name) as probe:
        out_w, out_h = probe.size

    idx_json = {
        "site_id": args.site,
        "source_dir": str(src_camera),
        "crop_x_px": crop_x,
        "crop_y_px": crop_y,
        "output_width": out_w,
        "output_height": out_h,
        "trim_start_s": trim_start,
        "trim_end_s": trim_end,
        "trim_t0_epoch": epoch_lo,
        "fps_estimate": round(len(kept) / (epoch_hi - epoch_lo), 2) if epoch_hi > epoch_lo else 0,
        "frames": index,
    }
    idx_path = Path(cfg["out"]) / "ego_camera_index.json"
    idx_path.write_text(json.dumps(idx_json, ensure_ascii=False, indent=2))
    print(f"[write] {idx_path}  ({len(kept)} frames, ~{idx_json['fps_estimate']:.1f} fps)")
    print(f"  total output: {total_bytes / 1024 / 1024:.1f} MB at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
