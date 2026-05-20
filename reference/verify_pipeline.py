#!/usr/bin/env python3
"""Sanity-plot the pipeline output. Writes web/verify.png."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

LL_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)


def latlng_to_xy(latlng):
    arr = np.asarray(latlng)
    x, y = LL_TO_UTM.transform(arr[:, 1], arr[:, 0])
    return np.asarray(x), np.asarray(y)


def main(web_dir="web"):
    web = Path(web_dir)
    lanes = json.loads((web / "lanes.json").read_text())
    frames = json.loads((web / "frames.json").read_text())["frames"]

    xR, yR = latlng_to_xy(lanes["lanes"][0]["centerline_latlng"])
    xL, yL = latlng_to_xy(lanes["lanes"][1]["centerline_latlng"])
    x0, y0 = float((xR.mean() + xL.mean()) / 2), float((yR.mean() + yL.mean()) / 2)

    ego_lat = np.array([f["ego"]["lat"] for f in frames])
    ego_lon = np.array([f["ego"]["lon"] for f in frames])
    em_lat  = np.array([f["emergency"]["lat"] for f in frames])
    em_lon  = np.array([f["emergency"]["lon"] for f in frames])
    xe, ye = LL_TO_UTM.transform(ego_lon, ego_lat)
    xm, ym = LL_TO_UTM.transform(em_lon, em_lat)

    t = np.array([f["t"] for f in frames])
    dist = np.array([f["rel"]["distance_m"] for f in frames])
    ahead = np.array([f["rel"]["ahead_m"] for f in frames])
    lateral = np.array([f["rel"]["lateral_m"] for f in frames])
    flag = np.array([f["same_lane_ahead"] for f in frames])
    ego_lane = np.array([1 if f["ego"]["lane"] == "L" else 0 for f in frames])
    em_lane  = np.array([1 if f["emergency"]["lane"] == "L" else 0 for f in frames])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"pipeline verify — {len(frames)} frames, "
                 f"lane sep median {lanes['sep_median_m']:.2f}m, "
                 f"same_lane_ahead {int(flag.sum())} frames",
                 fontsize=12)

    ax = axes[0, 0]
    ax.plot(xR - x0, yR - y0, "b-", lw=2, label="lane R (bag1 outbound)")
    ax.plot(xL - x0, yL - y0, "r-", lw=2, label="lane L (bag1 return rev)")
    ax.plot(xe - x0, ye - y0, "g.", ms=2, label="ego (bag2 outbound)")
    ax.plot(xm - x0, ym - y0, "m.", ms=2, label="emergency (bag1 outbound)")
    sf = flag
    ax.plot(xe[sf] - x0, ye[sf] - y0, "yo", ms=4, mec="black",
            label=f"same_lane_ahead ({int(sf.sum())})")
    ax.set_xlabel("UTM East offset (m)")
    ax.set_ylabel("UTM North offset (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("lanes + trips (mean-centered)")

    ax = axes[0, 1]
    ax.plot(t, dist, "k-", lw=1, label="ego↔emergency dist (m)")
    ax.fill_between(t, 0, dist.max(), where=flag, color="orange", alpha=0.3,
                    label="same_lane_ahead")
    ax.set_xlabel("scenario time (s)")
    ax.set_ylabel("distance (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("distance & same-lane flag")

    ax = axes[1, 0]
    ax.plot(t, ahead, "b-", lw=1, label="ahead (m, along ego heading)")
    ax.plot(t, lateral, "r-", lw=1, label="lateral (m, ego-right positive)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(10, color="b", ls="--", alpha=0.4, label="forward 10m")
    ax.axhline(+1.75, color="r", ls="--", alpha=0.4, label="lane half-width ±1.75m")
    ax.axhline(-1.75, color="r", ls="--", alpha=0.4)
    ax.set_xlabel("scenario time (s)")
    ax.set_ylabel("relative position (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("emergency relative to ego")

    ax = axes[1, 1]
    ax.plot(t, ego_lane + 0.02, "g-", lw=1.5, label="ego lane (0=R, 1=L)")
    ax.plot(t, em_lane - 0.02, "m-", lw=1.5, label="emergency lane")
    ax.set_xlabel("scenario time (s)")
    ax.set_ylabel("lane id")
    ax.set_yticks([0, 1], ["R", "L"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("lane assignment over time")

    plt.tight_layout()
    out = web / "verify.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web")
