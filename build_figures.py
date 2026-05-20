#!/usr/bin/env python3
"""Build presentation figures — Satellite Navigation Systems project (ochang).

Paper-style figures: English labels, no inline interpretation/annotation.
Captions and explanations live in `data/slide_content_draft.md`.

Output: data/figures/fig_*.png + data/figure_metrics.json
Run from project root:
    python3 build_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import savgol_filter
from scipy.stats import chi2

from gps_kf import FilterConfig, GPSTrackFilter

LL2U = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
OUT = Path("data/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 110,
    "font.family": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

def load_navpvt(extract_dir):
    pvt = pd.read_csv(f"{extract_dir}/ublox_gps__navpvt.csv")
    fix = pd.read_csv(f"{extract_dir}/ublox_gps__fix.csv")
    t = pvt["bag_t.secs"].to_numpy() + pvt["bag_t.nsecs"].to_numpy() * 1e-9
    t0 = t[0]
    t -= t0
    flags = pvt["flags"].to_numpy().astype(np.int64)
    return {
        "name": Path(extract_dir).name,
        "t": t,
        "hAcc": pvt["hAcc"].to_numpy() * 1e-3,
        "sAcc": pvt["sAcc"].to_numpy() * 1e-3,
        "gSpeed_kmh": pvt["gSpeed"].to_numpy() * 1e-3 * 3.6,
        "velE": pvt["velE"].to_numpy() * 1e-3,
        "velN": pvt["velN"].to_numpy() * 1e-3,
        "carrSoln": (flags >> 6) & 0x3,
        "diffSoln": (flags >> 1) & 0x1,
        "numSV": pvt["numSV"].to_numpy(),
        "pDOP": pvt["pDOP"].to_numpy() * 0.01,
        "fix_lat": fix["latitude"].to_numpy(),
        "fix_lon": fix["longitude"].to_numpy(),
        "fix_t": (fix["bag_t.secs"].to_numpy() + fix["bag_t.nsecs"].to_numpy() * 1e-9 - t0),
    }


bag1 = load_navpvt("data/sites/ochang/source/gps1_extracted")
bag2 = load_navpvt("data/sites/ochang/source/gps2_extracted")
final_cfg = json.load(open("data/sites/ochang/final.json"))
lanes_data = json.load(open("web/sites/ochang/lanes.json"))
frames_data = json.load(open("web/sites/ochang/frames.json"))


# ---------------------------------------------------------------------------
# (carrSoln/diffSoln are constants throughout the recording — carrSoln=0 100%,
# diffSoln=1 100%. Conveyed as slide text rather than as a flat-line plot.)
# Numbers are still emitted to figure_metrics.json for slide bullets.
# ---------------------------------------------------------------------------

def metric_correction_status():
    return {
        "carrSoln_zero_pct": float(100 * (bag2["carrSoln"] == 0).mean()),
        "diffSoln_on_pct": float(100 * (bag2["diffSoln"] == 1).mean()),
    }


# ---------------------------------------------------------------------------
# hAcc + sAcc + numSV
# ---------------------------------------------------------------------------

def fig_hacc_sacc_numsv():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    bins = np.linspace(0, 1.6, 32)
    axes[0].hist(bag1["hAcc"], bins=bins, alpha=0.55, color="C3", label="Emergency")
    axes[0].hist(bag2["hAcc"], bins=bins, alpha=0.55, color="C0", label="Ego")
    med_h = float(np.median(np.concatenate([bag1["hAcc"], bag2["hAcc"]])))
    axes[0].axvline(med_h, color="k", ls=":", lw=1.5, label=f"median = {med_h:.2f} m")
    axes[0].axvline(0.03, color="green", ls="--", lw=1.5, label="RTK Fixed (~3 cm)")
    axes[0].set_xlabel("hAcc (m)")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("(a) Horizontal 1-σ (hAcc)")
    axes[0].legend(fontsize=9)

    bins_s = np.linspace(0, 0.8, 32)
    axes[1].hist(bag1["sAcc"], bins=bins_s, alpha=0.55, color="C3", label="Emergency")
    axes[1].hist(bag2["sAcc"], bins=bins_s, alpha=0.55, color="C0", label="Ego")
    med_s = float(np.median(np.concatenate([bag1["sAcc"], bag2["sAcc"]])))
    axes[1].axvline(med_s, color="k", ls=":", lw=1.5, label=f"median = {med_s:.2f} m/s")
    axes[1].set_xlabel("sAcc (m/s)")
    axes[1].set_title("(b) Velocity 1-σ (sAcc)")
    axes[1].legend(fontsize=9)

    axes[2].plot(bag1["t"], bag1["numSV"], "C3-", lw=1.2, alpha=0.7, label="Emergency")
    axes[2].plot(bag2["t"], bag2["numSV"], "C0-", lw=1.2, alpha=0.7, label="Ego")
    med_n = float(np.median(np.concatenate([bag1["numSV"], bag2["numSV"]])))
    axes[2].axhline(med_n, color="k", ls=":", lw=1.5, label=f"median = {med_n:.0f}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("numSV")
    axes[2].set_title("(c) Satellites used")
    axes[2].set_ylim(0, max(bag1["numSV"].max(), bag2["numSV"].max()) + 3)
    axes[2].legend(fontsize=9)

    fig.suptitle("Receiver-Reported 1-σ Uncertainty and Satellite Visibility",
                 fontsize=12, y=1.02)
    fig.savefig(OUT / "fig_hacc_sacc_numsv.png")
    plt.close(fig)
    print("  fig_hacc_sacc_numsv.png")
    return {"hAcc_median_m": med_h, "sAcc_median_mps": med_s, "numSV_median": med_n}


# ---------------------------------------------------------------------------
# 3) pDOP
# ---------------------------------------------------------------------------

def fig_dop_geometry():
    fig, ax = plt.subplots()
    ax.plot(bag1["t"], bag1["pDOP"], "C3-", lw=1.2, alpha=0.75, label="Emergency")
    ax.plot(bag2["t"], bag2["pDOP"], "C0-", lw=1.2, alpha=0.75, label="Ego")
    med = float(np.median(np.concatenate([bag1["pDOP"], bag2["pDOP"]])))
    ax.axhline(med, color="k", ls=":", lw=1.5, label=f"median = {med:.2f}")
    ax.axhspan(0, 2, color="green", alpha=0.07)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("pDOP")
    ax.set_ylim(0, max(2.5, bag2["pDOP"].max() * 1.1))
    ax.set_title("Position Dilution of Precision (pDOP)")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "fig_dop_geometry.png")
    plt.close(fig)
    print("  fig_dop_geometry.png")
    return {"pDOP_median": med,
            "pDOP_max": float(np.max(np.concatenate([bag1["pDOP"], bag2["pDOP"]])))}


# ---------------------------------------------------------------------------
# 4) Error decomposition — high-freq noise vs slow drift
# ---------------------------------------------------------------------------

def fig_error_decomposition():
    x, y = LL2U.transform(bag1["fix_lon"], bag1["fix_lat"])
    t = bag1["fix_t"]
    mask = (t > 18) & (t < 58)
    x_o, y_o = x[mask], y[mask]
    ux, uy = x_o[-1] - x_o[0], y_o[-1] - y_o[0]
    L = np.hypot(ux, uy); ux /= L; uy /= L
    nx, ny = -uy, ux
    perp = (x_o - x_o[0]) * nx + (y_o - y_o[0]) * ny
    perp_smooth = savgol_filter(perp, 21, 3) if len(perp) > 21 else perp
    noise = perp - perp_smooth

    fig, ax = plt.subplots()
    t_o = t[mask]
    ax.plot(t_o, perp - perp.mean(), "C0-", lw=1, alpha=0.55, label="raw")
    ax.plot(t_o, perp_smooth - perp.mean(), "C3-", lw=2,
            label="low-frequency (Savitzky-Golay)")
    ax.fill_between(t_o,
                    (perp_smooth - perp.mean()) - noise.std(),
                    (perp_smooth - perp.mean()) + noise.std(),
                    color="C3", alpha=0.15,
                    label=f"±1σ high-freq = {noise.std() * 100:.1f} cm")
    bias_range = float(perp_smooth.max() - perp_smooth.min())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateral residual (m)")
    ax.set_title("Lateral Residual Decomposition along an Outbound Segment")
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(OUT / "fig_error_decomposition.png")
    plt.close(fig)
    print("  fig_error_decomposition.png")
    return {"noise_rms_cm": float(noise.std() * 100), "drift_range_m": bias_range}


# ---------------------------------------------------------------------------
# (fig_trajectory_raw_vs_kf dropped — NavPVT은 수신기 내부에서 이미 평활화된
# 솔루션이라 추가 KF의 positional 변화가 미미 (~1%). KF의 진짜 효과는 heading
# 안정성에 있고 그건 다음 figure가 정량적으로 보여줌.)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Heading: gradient vs KF
# ---------------------------------------------------------------------------

def fig_heading_kf_vs_gradient():
    x, y = LL2U.transform(bag1["fix_lon"], bag1["fix_lat"])
    t = bag1["fix_t"]
    dx = np.gradient(x); dy = np.gradient(y)
    hdg_grad = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

    velE = np.interp(t, bag1["t"], bag1["velE"])
    velN = np.interp(t, bag1["t"], bag1["velN"])
    hAcc = np.interp(t, bag1["t"], bag1["hAcc"])
    sAcc = np.interp(t, bag1["t"], bag1["sAcc"])
    cs = np.round(np.interp(t, bag1["t"], bag1["carrSoln"])).astype(int)
    tracker = GPSTrackFilter(FilterConfig())
    ms = [tracker.adapter.build(ti, xi, yi, vE, vN, ha, sa, ci)
          for ti, xi, yi, vE, vN, ha, sa, ci in zip(t, x, y, velE, velN, hAcc, sAcc, cs)]
    smoothed = tracker.run_smoothed(ms)
    hdg_kf = smoothed.heading_deg

    mask = (t > 20) & (t < 58)
    dh_grad = np.degrees(np.std(np.diff(np.unwrap(np.radians(hdg_grad[mask])))))
    dh_kf = np.degrees(np.std(np.diff(np.unwrap(np.radians(hdg_kf[mask])))))

    fig, ax = plt.subplots()
    ax.plot(t[mask], hdg_grad[mask], "C0-", lw=0.8, alpha=0.5,
            label=f"gradient (σΔ = {dh_grad:.2f}°)")
    ax.plot(t[mask], hdg_kf[mask], "C3-", lw=1.8,
            label=f"KF velocity (σΔ = {dh_kf:.2f}°)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (deg, compass)")
    ax.set_title("Heading Estimation: Path Gradient vs Kalman Velocity")
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(OUT / "fig_heading_kf_vs_gradient.png")
    plt.close(fig)
    print("  fig_heading_kf_vs_gradient.png")
    return {
        "heading_std_gradient_deg": float(dh_grad),
        "heading_std_kf_deg": float(dh_kf),
        "reduction_pct": float(100 * (1 - dh_kf / dh_grad)),
    }


# ---------------------------------------------------------------------------
# 7) NIS distribution
# ---------------------------------------------------------------------------

def fig_nis_distribution():
    nis_all = []
    for b, role in ((bag1, "Emergency"), (bag2, "Ego")):
        x, y = LL2U.transform(b["fix_lon"], b["fix_lat"])
        t = b["fix_t"]
        velE = np.interp(t, b["t"], b["velE"])
        velN = np.interp(t, b["t"], b["velN"])
        hAcc = np.interp(t, b["t"], b["hAcc"])
        sAcc = np.interp(t, b["t"], b["sAcc"])
        cs = np.round(np.interp(t, b["t"], b["carrSoln"])).astype(int)
        tracker = GPSTrackFilter(FilterConfig())
        ms = [tracker.adapter.build(ti, xi, yi, vE, vN, ha, sa, ci)
              for ti, xi, yi, vE, vN, ha, sa, ci in zip(t, x, y, velE, velN, hAcc, sAcc, cs)]
        result = tracker.run(ms)
        nis_all.append((role, result.nis))

    fig, ax = plt.subplots()
    bins = np.logspace(-3, 2, 40)
    for name, nis in nis_all:
        ax.hist(nis, bins=bins, alpha=0.5,
                label=f"{name} (median {np.median(nis):.2f})")
    expected = chi2.ppf(0.5, df=4)
    gate = chi2.ppf(0.997, df=4)
    ax.axvline(expected, color="green", ls="--", lw=1.5,
               label=f"χ²(4) median = {expected:.2f}")
    ax.axvline(gate, color="red", ls="--", lw=1.5,
               label=f"χ² gate (p = 0.997) = {gate:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("NIS")
    ax.set_ylabel("Samples")
    ax.set_title("Normalized Innovation Squared (NIS) Distribution")
    ax.legend(fontsize=10, loc="upper right")
    fig.savefig(OUT / "fig_nis_distribution.png")
    plt.close(fig)
    print("  fig_nis_distribution.png")
    return {
        "nis_median_emergency": float(np.median(nis_all[0][1])),
        "nis_median_ego": float(np.median(nis_all[1][1])),
        "chi2_4dof_median_expected": float(expected),
    }


# ---------------------------------------------------------------------------
# 8) Alert timeline
# ---------------------------------------------------------------------------

def fig_alert_timeline():
    f = frames_data["frames"]
    t = np.array([fr["t"] for fr in f])
    dist = np.array([fr["rel"]["distance_m"] for fr in f])
    alert = np.array([fr["same_lane_alert"] for fr in f])
    behind = np.array([fr["same_lane_behind"] for fr in f])
    ahead = np.array([fr["same_lane_ahead"] for fr in f])

    fig, ax = plt.subplots()
    ax.plot(t, dist, "k-", lw=1.6, label="Ego–Emergency distance")
    ax.fill_between(t, 0, dist.max() * 1.05, where=behind, color="C3", alpha=0.25,
                    label="Behind, same lane (alert)")
    ax.fill_between(t, 0, dist.max() * 1.05, where=ahead & ~behind, color="C1",
                    alpha=0.18, label="Ahead, same lane")
    ax.set_xlabel("Scenario time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Same-Lane Detection Timeline")
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(OUT / "fig_alert_timeline.png")
    plt.close(fig)
    print("  fig_alert_timeline.png")
    return {
        "alert_pct": float(100 * alert.mean()),
        "alert_n": int(alert.sum()),
        "behind_n": int(behind.sum()),
        "ahead_n": int(ahead.sum()),
        "total_frames": int(len(alert)),
    }


# ---------------------------------------------------------------------------

def main():
    print("[figures] writing →", OUT)
    metrics = {}
    metrics["correction_status"] = metric_correction_status()
    metrics["fig_hacc_sacc_numsv"] = fig_hacc_sacc_numsv()
    metrics["fig_dop_geometry"] = fig_dop_geometry()
    metrics["fig_error_decomposition"] = fig_error_decomposition()
    metrics["fig_heading_kf_vs_gradient"] = fig_heading_kf_vs_gradient()
    metrics["fig_nis_distribution"] = fig_nis_distribution()
    metrics["fig_alert_timeline"] = fig_alert_timeline()
    Path("data/figure_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print("[wrote] data/figure_metrics.json")
    print("[done]")


if __name__ == "__main__":
    main()
