#!/usr/bin/env python3
"""Constant-velocity Kalman filter for GPS trajectory smoothing.

State:        x = [x_m, y_m, vx_mps, vy_mps]   (UTM East, North)
Measurement:  z = [x_m, y_m, vE_mps, vN_mps]   (from u-blox NavPVT)
Motion:       constant velocity + white-noise acceleration.

R is per-sample: hAcc / sAcc from NavPVT, inflated by carrSoln quality.
H is linear → plain KF (no EKF). U-turn is out of scope for this project,
and lane changes are absorbed by process-noise tuning.

Run standalone:
    python gps_kf.py --extract gps1_extracted --out gps1_filtered.csv
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import chi2

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterConfig:
    """Tunable parameters. Defaults target u-blox at vehicle speeds."""

    sigma_a_mps2: float = 1.0
    """RMS acceleration for white-noise-acceleration Q. Raise to trust
    measurements more during lane changes; lower for smoother straights."""

    carrsoln_r_scale: tuple[float, float, float] = (1.5, 1.1, 1.0)
    """Multiplicative R-scale by carrSoln (None, Float, Fixed). u-blox hAcc
    is already an honest 1-sigma estimate, so this only adds a small margin
    for non-RTK fixes. Use NIS to verify calibration on your data."""

    mahalanobis_p: float = 0.997
    """Innovation-gating probability. A measurement whose normalized
    innovation-squared exceeds chi2.ppf(p, dof) is rejected as an outlier."""

    min_hacc_m: float = 0.05
    """Floor on reported hAcc — avoids overconfidence at RTK Fixed."""

    min_sacc_mps: float = 0.05
    """Floor on reported sAcc."""


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """Timestamped measurement with its already-scaled covariance."""
    t: float
    z: np.ndarray         # shape (m,)
    R: np.ndarray         # shape (m, m)
    source: str = "gps"


# ---------------------------------------------------------------------------
# Motion model — strategy interface (Open/Closed: add CA/CTRV without
# touching the filter core).
# ---------------------------------------------------------------------------

class MotionModel(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @abstractmethod
    def F(self, dt: float) -> np.ndarray: ...

    @abstractmethod
    def Q(self, dt: float) -> np.ndarray: ...

    def predict(self, x: np.ndarray, P: np.ndarray, dt: float
                ) -> tuple[np.ndarray, np.ndarray]:
        F = self.F(dt)
        return F @ x, F @ P @ F.T + self.Q(dt)


class ConstantVelocity(MotionModel):
    """4-state CV [x, y, vx, vy] with decoupled white-noise acceleration."""

    def __init__(self, sigma_a_mps2: float):
        self._sigma_a2 = float(sigma_a_mps2) ** 2

    @property
    def state_dim(self) -> int:
        return 4

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        q = self._sigma_a2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = dt4 / 4 * q
        Q[2, 2] = Q[3, 3] = dt2 * q
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = dt3 / 2 * q
        return Q


# ---------------------------------------------------------------------------
# Filter core
# ---------------------------------------------------------------------------

@dataclass
class FilterState:
    t: float
    x: np.ndarray
    P: np.ndarray


@dataclass
class UpdateRecord:
    """Diagnostics from a single update step.

    The `_predicted`/`_filtered` fields and `F_used` are required by the RTS
    smoother — without them the backward pass would have to re-run the forward
    pass to recover the predicted moments."""
    t: float
    accepted: bool
    nis: float                       # normalized innovation squared
    gate: float                      # chi-square threshold
    innovation: np.ndarray
    posterior_state: np.ndarray
    F_used: np.ndarray = None        # F(dt) used to predict TO this step
    x_predicted: np.ndarray = None   # state before this step's update
    P_predicted: np.ndarray = None
    x_filtered: np.ndarray = None    # state after this step's update
    P_filtered: np.ndarray = None


class KalmanFilter:
    """Linear KF with Joseph-form covariance update and chi-square gating.

    The engine is decoupled from any specific motion or measurement model.
    Callers pass a MotionModel at construction and an H matrix per update.
    """

    def __init__(self, model: MotionModel, config: FilterConfig):
        self._model = model
        self._cfg = config
        self._state: FilterState | None = None
        self.history: list[UpdateRecord] = []

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> FilterState:
        if self._state is None:
            raise RuntimeError("Filter not initialized")
        return self._state

    def initialize(self, t: float, x0: np.ndarray, P0: np.ndarray) -> None:
        if x0.shape != (self._model.state_dim,):
            raise ValueError(
                f"x0 shape {x0.shape}, expected ({self._model.state_dim},)")
        self._state = FilterState(t=float(t), x=x0.copy(), P=P0.copy())

    def predict(self, t: float) -> None:
        st = self.state
        dt = float(t) - st.t
        if dt < 0:
            raise ValueError(f"Backwards time step: dt={dt}")
        if dt == 0.0:
            return
        x, P = self._model.predict(st.x, st.P, dt)
        self._state = FilterState(t=float(t), x=x, P=P)

    def update(self, m: Measurement, H: np.ndarray) -> UpdateRecord:
        st_prev = self.state
        dt = float(m.t) - st_prev.t
        F_used = (self._model.F(dt) if dt > 0
                  else np.eye(self._model.state_dim))

        self.predict(m.t)
        st = self.state
        x_pred = st.x.copy()
        P_pred = st.P.copy()

        y = m.z - H @ st.x                            # innovation
        S = H @ st.P @ H.T + m.R                       # innovation covariance
        nis = float(y @ np.linalg.solve(S, y))
        gate = float(chi2.ppf(self._cfg.mahalanobis_p, df=len(m.z)))
        accepted = nis <= gate

        if accepted:
            # K = P H^T S^-1 via solve for numerical stability.
            K = np.linalg.solve(S, (st.P @ H.T).T).T
            I = np.eye(self._model.state_dim)
            IKH = I - K @ H
            x_new = st.x + K @ y
            P_new = IKH @ st.P @ IKH.T + K @ m.R @ K.T   # Joseph form
            self._state = FilterState(t=m.t, x=x_new, P=P_new)
            posterior = x_new
        else:
            log.debug("rejected measurement at t=%.3f: NIS=%.2f > %.2f",
                      m.t, nis, gate)
            posterior = st.x
            x_new = x_pred
            P_new = P_pred

        rec = UpdateRecord(t=m.t, accepted=accepted, nis=nis, gate=gate,
                           innovation=y, posterior_state=posterior,
                           F_used=F_used,
                           x_predicted=x_pred, P_predicted=P_pred,
                           x_filtered=x_new.copy(), P_filtered=P_new.copy())
        self.history.append(rec)
        return rec


# ---------------------------------------------------------------------------
# Rauch-Tung-Striebel smoother (offline backward pass)
# ---------------------------------------------------------------------------

class RTSSmoother:
    """Backward smoother for an offline-replayed track. Takes the forward
    `UpdateRecord` history and produces smoothed states. Strictly reduces
    posterior variance under the linear/Gaussian assumption."""

    def __init__(self, model: MotionModel):
        self._model = model

    def smooth(self, records: list[UpdateRecord]) -> list[FilterState]:
        if len(records) < 2:
            return [FilterState(r.t, r.x_filtered.copy(), r.P_filtered.copy())
                    for r in records]
        n = len(records)
        out: list[FilterState | None] = [None] * n
        out[-1] = FilterState(records[-1].t,
                              records[-1].x_filtered.copy(),
                              records[-1].P_filtered.copy())
        for k in range(n - 2, -1, -1):
            rec_k = records[k]
            rec_kp1 = records[k + 1]
            F_kp1 = rec_kp1.F_used
            P_pred_kp1 = rec_kp1.P_predicted
            x_pred_kp1 = rec_kp1.x_predicted
            # C_k = P_filt[k] @ F.T @ inv(P_pred[k+1])
            # Solve P_pred^T y = (P_filt F^T)^T  →  y = P_pred^-T (P_filt F^T)^T
            C_k = np.linalg.solve(P_pred_kp1.T,
                                  (rec_k.P_filtered @ F_kp1.T).T).T
            x_s = rec_k.x_filtered + C_k @ (out[k + 1].x - x_pred_kp1)
            P_s = (rec_k.P_filtered
                   + C_k @ (out[k + 1].P - P_pred_kp1) @ C_k.T)
            out[k] = FilterState(rec_k.t, x_s, P_s)
        return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# u-blox NavPVT → Measurement adapter
# ---------------------------------------------------------------------------

class UbloxNavPVTAdapter:
    """Builds CV-state measurements from u-blox-style position+velocity.

    Inputs are expected in SI units (caller does the *1e-3 / *1e-5 NavPVT
    unscaling) and UTM (caller does the lat/lon → UTM conversion). Keeping
    those concerns outside this class makes it reusable for any RTK GNSS
    that produces position + velocity + per-sample sigma.
    """

    def __init__(self, config: FilterConfig):
        self._cfg = config

    def H(self) -> np.ndarray:
        """Maps CV state [x, y, vx, vy] → measurement [x, y, vE, vN]."""
        return np.eye(4)

    def build(self, t: float, x_utm: float, y_utm: float,
              vE_mps: float, vN_mps: float,
              hAcc_m: float, sAcc_mps: float,
              carrSoln: int) -> Measurement:
        hAcc = max(float(hAcc_m), self._cfg.min_hacc_m)
        sAcc = max(float(sAcc_mps), self._cfg.min_sacc_mps)
        scale = self._cfg.carrsoln_r_scale[max(0, min(int(carrSoln), 2))]
        R = np.diag([
            (scale * hAcc) ** 2,
            (scale * hAcc) ** 2,
            (scale * sAcc) ** 2,
            (scale * sAcc) ** 2,
        ])
        z = np.array([x_utm, y_utm, vE_mps, vN_mps], dtype=float)
        return Measurement(t=float(t), z=z, R=R, source="ublox_navpvt")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class TrackResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray
    heading_deg: np.ndarray       # compass bearing (0=N, 90=E)
    P_diag: np.ndarray            # (n, state_dim) per-sample variance diags
    nis: np.ndarray
    accepted: np.ndarray          # bool


class GPSTrackFilter:
    """High-level driver: run a CV-KF over a stream of measurements."""

    def __init__(self, config: FilterConfig | None = None):
        self._cfg = config or FilterConfig()
        self._model = ConstantVelocity(self._cfg.sigma_a_mps2)
        self._adapter = UbloxNavPVTAdapter(self._cfg)
        self._kf = KalmanFilter(self._model, self._cfg)

    @property
    def adapter(self) -> UbloxNavPVTAdapter:
        return self._adapter

    def run(self, measurements: Iterable[Measurement]) -> TrackResult:
        ms = list(measurements)
        if len(ms) < 2:
            raise ValueError("Need at least 2 measurements to bootstrap velocity")

        self._initialize_from_first_two(ms[0], ms[1])
        H = self._adapter.H()

        ts, xs, vs, Ps, niss, accs = [], [], [], [], [], []
        for m in ms:
            rec = self._kf.update(m, H)
            st = self._kf.state
            ts.append(st.t)
            xs.append(st.x[:2].copy())
            vs.append(st.x[2:].copy())
            Ps.append(np.diag(st.P).copy())
            niss.append(rec.nis)
            accs.append(rec.accepted)

        xs = np.asarray(xs)
        vs = np.asarray(vs)
        speed = np.linalg.norm(vs, axis=1)
        # Compass bearing from east/north velocity: atan2(E, N).
        heading = (np.degrees(np.arctan2(vs[:, 0], vs[:, 1])) + 360.0) % 360.0

        return TrackResult(
            t=np.asarray(ts),
            x=xs[:, 0], y=xs[:, 1],
            vx=vs[:, 0], vy=vs[:, 1],
            speed=speed, heading_deg=heading,
            P_diag=np.asarray(Ps),
            nis=np.asarray(niss),
            accepted=np.asarray(accs, dtype=bool),
        )

    def run_smoothed(self, measurements: Iterable[Measurement]) -> TrackResult:
        """Run forward KF + RTS backward smoother. Suitable for offline bag
        replay where future measurements are available."""
        forward = self.run(measurements)
        smoother = RTSSmoother(self._model)
        smoothed = smoother.smooth(self._kf.history)
        ts = np.array([s.t for s in smoothed])
        xs = np.array([s.x for s in smoothed])
        Pdiag = np.array([np.diag(s.P) for s in smoothed])
        vs = xs[:, 2:]
        speed = np.linalg.norm(vs, axis=1)
        heading = (np.degrees(np.arctan2(vs[:, 0], vs[:, 1])) + 360.0) % 360.0
        return TrackResult(
            t=ts, x=xs[:, 0], y=xs[:, 1], vx=vs[:, 0], vy=vs[:, 1],
            speed=speed, heading_deg=heading,
            P_diag=Pdiag, nis=forward.nis, accepted=forward.accepted,
        )

    def _initialize_from_first_two(self, m0: Measurement, m1: Measurement) -> None:
        dt = m1.t - m0.t
        if dt <= 0:
            raise ValueError(f"Non-positive dt between first two samples: {dt}")
        x0 = np.array([
            m0.z[0], m0.z[1],
            (m1.z[0] - m0.z[0]) / dt,
            (m1.z[1] - m0.z[1]) / dt,
        ])
        # Initial velocity uncertainty: two position errors over dt.
        vx_var = (m0.R[0, 0] + m1.R[0, 0]) / (dt * dt)
        vy_var = (m0.R[1, 1] + m1.R[1, 1]) / (dt * dt)
        P0 = np.diag([m0.R[0, 0], m0.R[1, 1], vx_var, vy_var])
        self._kf.initialize(t=m0.t, x0=x0, P0=P0)


# ---------------------------------------------------------------------------
# Standalone demo / verification
# ---------------------------------------------------------------------------

def _demo() -> None:
    import argparse
    from pathlib import Path

    import pandas as pd
    from pyproj import Transformer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", required=True,
                    help="extracted bag dir (e.g. gps1_extracted)")
    ap.add_argument("--out", default=None, help="optional CSV of filtered track")
    ap.add_argument("--sigma-a", type=float, default=FilterConfig.sigma_a_mps2,
                    help="process-noise RMS acceleration (m/s^2)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ll_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
    extract = Path(args.extract)

    fix = pd.read_csv(extract / "ublox_gps__fix.csv")
    pvt = pd.read_csv(extract / "ublox_gps__navpvt.csv")

    def bag_t(df: pd.DataFrame) -> np.ndarray:
        return (df["bag_t.secs"].to_numpy().astype(np.int64)
                + df["bag_t.nsecs"].to_numpy().astype(np.int64) * 1e-9)

    t_fix = bag_t(fix)
    t0 = float(t_fix[0])
    t_fix -= t0
    x_utm, y_utm = ll_to_utm.transform(fix["longitude"].to_numpy(),
                                       fix["latitude"].to_numpy())

    t_pvt = bag_t(pvt) - t0
    velE = pvt["velE"].to_numpy() * 1e-3
    velN = pvt["velN"].to_numpy() * 1e-3
    hAcc = pvt["hAcc"].to_numpy() * 1e-3
    sAcc = pvt["sAcc"].to_numpy() * 1e-3
    carrSoln = (pvt["flags"].to_numpy().astype(np.int64) >> 6) & 0x3

    # Align NavPVT onto fix timestamps.
    velE_i = np.interp(t_fix, t_pvt, velE)
    velN_i = np.interp(t_fix, t_pvt, velN)
    hAcc_i = np.interp(t_fix, t_pvt, hAcc)
    sAcc_i = np.interp(t_fix, t_pvt, sAcc)
    cs_i = np.round(np.interp(t_fix, t_pvt, carrSoln)).astype(int)

    cfg = FilterConfig(sigma_a_mps2=args.sigma_a)
    tracker = GPSTrackFilter(cfg)
    measurements = [
        tracker.adapter.build(t, x, y, vE, vN, hA, sA, cs)
        for t, x, y, vE, vN, hA, sA, cs in zip(
            t_fix, x_utm, y_utm, velE_i, velN_i, hAcc_i, sAcc_i, cs_i)
    ]
    result = tracker.run(measurements)

    raw_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([x_utm, y_utm]),
                                            axis=0) ** 2)))
    filt_jit = float(np.sqrt(np.mean(np.diff(np.column_stack([result.x, result.y]),
                                             axis=0) ** 2)))

    log.info("[gps_kf] %d samples processed", len(measurements))
    log.info("  accepted   : %d/%d (%.1f%%)",
             result.accepted.sum(), len(result.accepted),
             100 * result.accepted.mean())
    log.info("  median NIS : %.2f  (expected ~3.36 for 4-DoF, gate %.2f)",
             float(np.median(result.nis)),
             float(chi2.ppf(cfg.mahalanobis_p, df=4)))
    log.info("  step jitter: raw=%.3f m  filtered=%.3f m  (%.0f%% reduction)",
             raw_jit, filt_jit, 100 * (1 - filt_jit / raw_jit))

    if args.out:
        pd.DataFrame({
            "t": result.t, "x_utm": result.x, "y_utm": result.y,
            "vx": result.vx, "vy": result.vy,
            "speed": result.speed, "heading_deg": result.heading_deg,
            "nis": result.nis, "accepted": result.accepted,
        }).to_csv(args.out, index=False)
        log.info("  wrote %s", args.out)


if __name__ == "__main__":
    _demo()
