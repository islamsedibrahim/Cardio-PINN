"""Stage 7: Cardio-PINN parametric surrogate on the user's own LV.

Port of Buoso, Joyce & Kozerke (MedIA 2021) from TensorFlow 1.10 to PyTorch:

* a reduced displacement basis ``Phi`` (here: POD of the FE heartbeat on
  this patient's mesh, instead of the shape-model functional bases, so it
  applies to any imported anatomy);
* a small network ``(p_endo, t, s_act) -> a`` with Cardio-PINN's Swish
  activation, output scaled by the amplitude range; biventricular twins take
  both cavity pressures, ``(p_LV, p_RV, t, s_act) -> a``;
* trained by minimising the total potential energy of ``u = Phi a``
  (Cardio-PINN ``CardioLoss``), optionally anchored to the FE snapshots;
* coupled to the same Windkessel with Cardio-PINN's Newton/secant pressure
  updates, so a heartbeat costs milliseconds.

When NVIDIA PhysicsNeMo is installed its ``FullyConnected`` model is used as
the network backbone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ..mechanics.cycle import Chamber, CirculationParams, CycleResult
from ..mechanics.model import MMHG, MechanicsModel


@dataclass
class SurrogateConfig:
    n_modes: int = 12
    hidden_layers: int = 4
    hidden_neurons: int = 48
    epochs: int = 1500
    batch: int = 16
    learning_rate: float = 2e-3
    p_max_mmhg: float = 180.0
    p_max_rv_mmhg: float = 80.0  # RV pressure range sampled for biventricular twins
    t_max_ms: float = 500.0
    s_range: tuple = (0.4, 1.6)
    # FE-equilibrium anchoring (0 = pure physics as in Cardio-PINN). The reduced energy is as
    # ill-conditioned as the FE problem, so Adam alone under-converges; anchors fix the accuracy
    # (EF error 33.6 vs 33.5 % at s=0.6, 63.5 vs 61.7 % at s=1) and physics interpolates between them.
    data_weight: float = 300.0
    anchor_scales: tuple = (0.6, 1.4)  # extra FE equilibria at these contractilities
    anchor_pressure_factors: tuple = (1.0,)  # ... at these fractions of the beat pressure (warm-started solves)
    anchor_times_ms: tuple = (100.0, 180.0, 260.0)
    use_physicsnemo: bool = True
    seed: int = 0


class _Swish(torch.nn.Module if torch else object):
    def forward(self, x):
        return x * torch.sigmoid(30.0 * x)  # Cardio-PINN mySwish


def _build_net(cfg: SurrogateConfig, n_out, n_in=3):
    if cfg.use_physicsnemo:
        try:
            from physicsnemo.models.mlp.fully_connected import FullyConnected

            return FullyConnected(in_features=n_in, out_features=n_out, num_layers=cfg.hidden_layers,
                                  layer_size=cfg.hidden_neurons, activation_fn="silu"), "physicsnemo.FullyConnected"
        except Exception:
            pass
    layers, d = [], n_in
    for _ in range(cfg.hidden_layers):
        lin = torch.nn.Linear(d, cfg.hidden_neurons)
        torch.nn.init.xavier_normal_(lin.weight)
        layers += [lin, _Swish()]
        d = cfg.hidden_neurons
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers).double(), "torch.MLP(swish30)"


class CardioPINNSurrogate:
    def __init__(self, model: MechanicsModel, fe: CycleResult, cfg: SurrogateConfig = None):
        self.model = model
        self.cfg = cfg or SurrogateConfig()
        torch.manual_seed(self.cfg.seed)
        U = fe.displacements.reshape(len(fe.times), -1).astype(np.float64)
        _, S, Vt = np.linalg.svd(U, full_matrices=False)
        energy = np.cumsum(S**2) / np.sum(S**2)
        r = int(min(self.cfg.n_modes, len(S)))
        self.pod_energy = float(energy[r - 1])
        self.Phi = torch.as_tensor(Vt[:r].T.copy(), dtype=model.dtype, device=model.device)  # (3N, r)
        A = U @ Vt[:r].T  # snapshot amplitudes
        self.a_scale = torch.as_tensor(np.maximum(np.abs(A).max(0), 1e-6) * 1.5, dtype=model.dtype, device=model.device)
        # FE displacements are relative to the imaged mesh; the energy needs them relative to the
        # stress-free reference (they differ by model.offset after passive unloading)
        self.offset = torch.as_tensor(model.offset, dtype=model.dtype, device=model.device)
        self.K = model.n_cavities
        # activation level tau(t): mean active tension at unit contractility / T_max. The network sees
        # s * tau(t) instead of s, so relaxed states (tau = 0) cannot depend on contractility
        self._tau_t = np.linspace(0.0, self.cfg.t_max_ms, 401)
        tm = max(model.cfg.t_max_kpa * 1e3, 1e-9)
        self._tau = np.array([model.active_tension(float(t), 1.0).mean() / tm for t in self._tau_t])
        self.p_max = [self.cfg.p_max_mmhg, self.cfg.p_max_rv_mmhg][: self.K]
        self.net, self.backbone = _build_net(self.cfg, r, n_in=self.K + 2)
        self.net = self.net.to(model.device).double()
        # FE anchors
        self.fe_pressures = np.stack([fe.pressure_mmhg] + [fe.chambers[n]["pressure_mmhg"]
                                                           for n in model.cavity_names[1:]], 1)  # (T,K)
        self.anchor_x = self._x(self.fe_pressures, np.clip(fe.times, 0, None), np.ones(len(fe.times)))
        self.anchor_a = torch.as_tensor(A, dtype=model.dtype, device=model.device)
        self.history = []
        self._fe = fe

    def _x(self, ps, t_ms, s):
        """Network input from pressures (B,K) mmHg, times (B,) ms and contractility scales (B,)."""
        ps = np.atleast_2d(np.asarray(ps, float))
        cols = [ps[:, k] / self.p_max[k] for k in range(self.K)]
        t = np.clip(np.asarray(t_ms, float).reshape(-1), 0.0, self.cfg.t_max_ms)
        act = np.asarray(s, float).reshape(-1) * np.interp(t, self._tau_t, self._tau)
        cols += [t / self.cfg.t_max_ms, act]
        return torch.as_tensor(np.stack(np.broadcast_arrays(*cols), 1), dtype=self.model.dtype,
                               device=self.model.device)

    def _pressures(self, p_mmhg):
        ps = np.atleast_1d(np.asarray(p_mmhg, float))
        return np.concatenate([ps, np.zeros(self.K - len(ps))])[: self.K]

    def add_fe_anchors(self, log=print):
        """Equilibria at other contractilities (few FE solves) so the s-dependence is anchored."""
        fe, cfg = self._fe, self.cfg
        X, Y = [self.anchor_x], [self.anchor_a]
        beat = fe.times >= 0
        for s in cfg.anchor_scales:
            for t in cfg.anchor_times_ms:
                for pf in cfg.anchor_pressure_factors:
                    k = int(np.argmin(np.abs(fe.times - t) + (~beat) * 1e9))
                    ps = self.fe_pressures[k] * pf
                    u, _, _, _ = self.model.solve(fe.displacements[k] - self.model.offset,
                                                  self.model.active_tension(t, s),
                                                  loads=[("pressure", p * MMHG, 0.0, 0.0) for p in ps])
                    a = (u + self.model.offset).reshape(-1) @ self.Phi.cpu().numpy()
                    X.append(self._x(ps, t, s))
                    Y.append(torch.as_tensor(a[None], dtype=self.model.dtype, device=self.model.device))
        self.anchor_x, self.anchor_a = torch.cat(X), torch.cat(Y)
        log(f"surrogate: {len(self.anchor_x)} FE anchors ({len(cfg.anchor_scales) * len(cfg.anchor_times_ms) * len(cfg.anchor_pressure_factors)} new)")

    # ------------------------------------------------------------------
    def amplitudes(self, x):
        return self.net(x) * self.a_scale

    def displacement(self, p_mmhg, t_ms, s=1.0):
        """``p_mmhg``: LV pressure, or [p_LV, p_RV] on biventricular twins."""
        x = self._x(self._pressures(p_mmhg), t_ms, s)
        with torch.no_grad():
            a = self.amplitudes(x)
            return (a @ self.Phi.T).reshape(-1, 3).cpu().numpy()

    def volumes_ml(self, p_mmhg, t_ms, s=1.0):
        """Every cavity's volume (mL) at pressures ``p_mmhg`` (one per cavity)."""
        x = self._x(self._pressures(p_mmhg), t_ms, s)
        with torch.no_grad():
            U = (self.amplitudes(x) @ self.Phi.T).reshape(1, -1, 3) - self.offset
            return np.array([float(self.model.cavity_volume_batch(U, k)[0]) / 1000.0 for k in range(self.K)])

    def volume_ml(self, p_mmhg, t_ms, s=1.0):
        return float(self.volumes_ml(p_mmhg, t_ms, s)[0])

    def _ta_batch(self, t_ms, s):
        return torch.stack([torch.as_tensor(self.model.active_tension(float(t), float(k)), dtype=self.model.dtype)
                            for t, k in zip(t_ms, s)]).to(self.model.device)

    # ------------------------------------------------------------------
    def train(self, log=print, progress=None):
        cfg = self.cfg
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
        rng = np.random.default_rng(cfg.seed)
        scale = 1.0 / (self.model.cfg.material.a_iso * self.model.total_volume)
        if cfg.data_weight > 0 and cfg.anchor_scales and len(self.anchor_x) == len(self._fe.times):
            self.add_fe_anchors(log)
        for ep in range(cfg.epochs):
            ps = np.stack([rng.uniform(0, pm, cfg.batch) for pm in self.p_max], 1)
            t = rng.uniform(0, cfg.t_max_ms, cfg.batch)
            s = rng.uniform(*cfg.s_range, cfg.batch)
            x = self._x(ps, t, s)
            Ta = self._ta_batch(t, s)
            a = self.amplitudes(x)
            U = (a @ self.Phi.T).reshape(cfg.batch, -1, 3) - self.offset
            p_t = torch.as_tensor(ps * MMHG, dtype=self.model.dtype, device=self.model.device)
            Pi, _ = self.model.potential_batch(U, Ta, p_t if self.K > 1 else p_t[:, 0])
            # per-sample normalisation: each sample's minimiser is unchanged, but high-pressure
            # samples no longer dominate the gradient
            loss = (Pi / (Pi.detach().abs() + 1e-3 / scale)).mean()
            if cfg.data_weight > 0:
                a_fe = self.amplitudes(self.anchor_x)
                loss = loss + cfg.data_weight * (((a_fe - self.anchor_a) / self.a_scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if ep % 100 == 0 or ep == cfg.epochs - 1:
                self.history.append((ep, float(loss.detach())))
                log(f"surrogate epoch {ep:5d}  loss {float(loss.detach()):.5e}")
            if progress:
                progress((ep + 1) / cfg.epochs)
        return self.history

    def state_dict(self):
        return {"net": self.net.state_dict(), "Phi": self.Phi.cpu(), "a_scale": self.a_scale.cpu(),
                "config": asdict(self.cfg), "backbone": self.backbone, "pod_energy": self.pod_energy,
                "cavities": list(self.model.cavity_names)}


def _secant(fun, x0, x1, tol=0.05, it=30):
    f0, f1 = fun(x0), fun(x1)
    for _ in range(it):
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = float(np.clip(x2, -10.0, 400.0))
        x0, f0, x1, f1 = x1, f1, x2, fun(x2)
        if abs(x1 - x0) < tol:
            break
    return x1


def simulate_cycle_surrogate(sur: CardioPINNSurrogate, circ: CirculationParams = None, s_act=1.0, dt_ms=5.0,
                             min_ejection_ms=40.0, sweeps=3):
    """Windkessel-coupled heartbeat using the surrogate's V_k(p_LV, p_RV, t, s) (milliseconds).

    The surrogate's V(p, t) carries small noise, so a semilunar valve closes only on sustained
    reverse flow after a minimum ejection period."""
    circ = circ or CirculationParams()
    K = sur.K
    prm = circ.chambers(K)
    ps = np.array([c["edp"] for c in prm], float)
    v_ed = sur.volumes_ml(ps, 0.0, s_act)
    ch = [Chamber(c, dt_ms, circ.cycle_ms, float(v), min_reverse_steps=2, min_ejection_ms=min_ejection_ms)
          for c, v in zip(prm, v_ed)]
    rows = [[(c["edp"], float(v), c["p_dia"], 0.0, "ivc")] for c, v in zip(prm, v_ed)]
    times = [0.0]

    def solve_step(t, ps):
        ps = ps.copy()
        for _ in range(sweeps if K > 1 else 1):  # Gauss-Seidel over the ventricles
            for k, c in enumerate(ch):
                def Vk(q, k=k):
                    pp = ps.copy()
                    pp[k] = q
                    return sur.volumes_ml(pp, t, s_act)[k]

                if c.phase in ("ivc", "ivr"):
                    ps[k] = _secant(lambda q: Vk(q) - c.v_ref, ps[k], ps[k] + 5.0)
                elif c.phase == "ejection":
                    a, v_star = c.ejection_line()
                    ps[k] = _secant(lambda q: q - a * (v_star - Vk(q)), ps[k], ps[k] + 5.0)
                else:
                    ps[k] = c.fill_pressure(t)
        return ps, sur.volumes_ml(ps, t, s_act)

    n = int(circ.cycle_ms / dt_ms)
    for step in range(1, n + 1):
        t = step * dt_ms
        for _ in range(2 * K + 1):
            ps_new, vs = solve_step(t, ps)
            if not any([c.transition(t, p, v) for c, p, v in zip(ch, ps_new, vs)]):
                break
        ps = ps_new
        for k, c in enumerate(ch):
            rows[k].append(c.commit(float(ps[k]), float(vs[k])))
        times.append(t)

    def trace(r):
        return {"pressure_mmhg": np.array([x[0] for x in r]), "volume_ml": np.array([x[1] for x in r]),
                "arterial_mmhg": np.array([x[2] for x in r]), "flow_ml_s": np.array([x[3] for x in r]),
                "phase": [x[4] for x in r]}

    tr = [trace(r) for r in rows]
    Vol, P = tr[0]["volume_ml"], tr[0]["pressure_mmhg"]
    edv_, esv = float(Vol[0]), float(Vol.min())
    out = {
        "times": np.array(times), "pressure_mmhg": P, "volume_ml": Vol, "aortic_mmhg": tr[0]["arterial_mmhg"],
        "phase": tr[0]["phase"],
        "metrics": {"edv_ml": edv_, "esv_ml": esv, "stroke_volume_ml": edv_ - esv,
                    "ejection_fraction_pct": 100 * (edv_ - esv) / edv_, "peak_lv_pressure_mmhg": float(np.max(P)),
                    "contractility_scale": s_act},
        "chambers": {},
    }
    for k in range(1, K):
        c, name = tr[k], prm[k]["name"].lower()
        out["chambers"][prm[k]["name"]] = c
        v = c["volume_ml"]
        out["metrics"].update({f"{name}_edv_ml": float(v[0]), f"{name}_esv_ml": float(v.min()),
                               f"{name}_stroke_volume_ml": float(v[0] - v.min()),
                               f"{name}_ejection_fraction_pct": float(100 * (v[0] - v.min()) / v[0]),
                               f"peak_{name}_pressure_mmhg": float(c["pressure_mmhg"].max()),
                               "pa_systolic_mmhg" if name == "rv" else f"{name}_peak_arterial_pressure_mmhg":
                                   float(c["arterial_mmhg"].max())})
    return out


def calibrate_contractility(sur: CardioPINNSurrogate, target_ef_pct: float, circ: CirculationParams = None,
                            lo=None, hi=None, iters=12):
    """Personalise active tension so the twin reproduces a measured EF (e.g. from echo)."""
    lo = lo if lo is not None else sur.cfg.s_range[0]
    hi = hi if hi is not None else sur.cfg.s_range[1]
    ef = lambda s: simulate_cycle_surrogate(sur, circ, s, dt_ms=5.0)["metrics"]["ejection_fraction_pct"]  # noqa: E731
    f_lo, f_hi = ef(lo) - target_ef_pct, ef(hi) - target_ef_pct
    if f_lo * f_hi > 0:
        s = lo if abs(f_lo) < abs(f_hi) else hi
        return s, ef(s), False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = ef(mid) - target_ef_pct
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    s = 0.5 * (lo + hi)
    return s, ef(s), True
