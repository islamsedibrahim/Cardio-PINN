"""Stage 7: Cardio-PINN parametric surrogate on the user's own LV.

Port of Buoso, Joyce & Kozerke (MedIA 2021) from TensorFlow 1.10 to PyTorch:

* a reduced displacement basis ``Phi`` (here: POD of the FE heartbeat on
  this patient's mesh, instead of the shape-model functional bases, so it
  applies to any imported anatomy);
* a small network ``(p_endo, t, s_act) -> a`` with Cardio-PINN's Swish
  activation, output scaled by the amplitude range;
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

from ..mechanics.cycle import CirculationParams, CycleResult
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


def _build_net(cfg: SurrogateConfig, n_out):
    if cfg.use_physicsnemo:
        try:
            from physicsnemo.models.mlp.fully_connected import FullyConnected

            return FullyConnected(in_features=3, out_features=n_out, num_layers=cfg.hidden_layers,
                                  layer_size=cfg.hidden_neurons, activation_fn="silu"), "physicsnemo.FullyConnected"
        except Exception:
            pass
    layers, d = [], 3
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
        self.net, self.backbone = _build_net(self.cfg, r)
        self.net = self.net.to(model.device).double()
        # FE anchors
        beat = fe.times > -1e9
        self.anchor_x = torch.as_tensor(np.stack([fe.pressure_mmhg[beat] / self.cfg.p_max_mmhg,
                                                  np.clip(fe.times[beat], 0, None) / self.cfg.t_max_ms,
                                                  np.ones(beat.sum())], 1), dtype=model.dtype, device=model.device)
        self.anchor_a = torch.as_tensor(A[beat], dtype=model.dtype, device=model.device)
        self.history = []
        self._fe = fe

    def add_fe_anchors(self, log=print):
        """Equilibria at other contractilities (few FE solves) so the s-dependence is anchored."""
        fe, cfg = self._fe, self.cfg
        X, Y = [self.anchor_x], [self.anchor_a]
        beat = fe.times >= 0
        for s in cfg.anchor_scales:
            for t in cfg.anchor_times_ms:
                for pf in cfg.anchor_pressure_factors:
                    k = int(np.argmin(np.abs(fe.times - t) + (~beat) * 1e9))
                    p = float(fe.pressure_mmhg[k]) * pf
                    u, _, _, _ = self.model.solve(fe.displacements[k], self.model.active_tension(t, s), "pressure",
                                                  p_pa=p * MMHG)
                    a = u.reshape(-1) @ self.Phi.cpu().numpy()
                    X.append(torch.tensor([[p / cfg.p_max_mmhg, t / cfg.t_max_ms, s]], dtype=self.model.dtype,
                                          device=self.model.device))
                    Y.append(torch.as_tensor(a[None], dtype=self.model.dtype, device=self.model.device))
        self.anchor_x, self.anchor_a = torch.cat(X), torch.cat(Y)
        log(f"surrogate: {len(self.anchor_x)} FE anchors ({len(cfg.anchor_scales) * len(cfg.anchor_times_ms) * len(cfg.anchor_pressure_factors)} new)")

    # ------------------------------------------------------------------
    def amplitudes(self, x):
        return self.net(x) * self.a_scale

    def displacement(self, p_mmhg, t_ms, s=1.0):
        x = torch.tensor([[p_mmhg / self.cfg.p_max_mmhg, t_ms / self.cfg.t_max_ms, s]], dtype=self.model.dtype,
                         device=self.model.device)
        with torch.no_grad():
            a = self.amplitudes(x)
            return (a @ self.Phi.T).reshape(-1, 3).cpu().numpy()

    def volume_ml(self, p_mmhg, t_ms, s=1.0):
        x = torch.tensor([[p_mmhg / self.cfg.p_max_mmhg, t_ms / self.cfg.t_max_ms, s]], dtype=self.model.dtype,
                         device=self.model.device)
        with torch.no_grad():
            U = (self.amplitudes(x) @ self.Phi.T).reshape(1, -1, 3)
            return float(self.model.cavity_volume_batch(U)[0]) / 1000.0

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
            p = rng.uniform(0, cfg.p_max_mmhg, cfg.batch)
            t = rng.uniform(0, cfg.t_max_ms, cfg.batch)
            s = rng.uniform(*cfg.s_range, cfg.batch)
            x = torch.as_tensor(np.stack([p / cfg.p_max_mmhg, t / cfg.t_max_ms, s], 1), dtype=self.model.dtype,
                                device=self.model.device)
            Ta = self._ta_batch(t, s)
            a = self.amplitudes(x)
            U = (a @ self.Phi.T).reshape(cfg.batch, -1, 3)
            Pi, _ = self.model.potential_batch(U, Ta, torch.as_tensor(p * MMHG, dtype=self.model.dtype,
                                                                     device=self.model.device))
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
                "config": asdict(self.cfg), "backbone": self.backbone, "pod_energy": self.pod_energy}


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
                             min_ejection_ms=40.0):
    """Windkessel-coupled heartbeat using the surrogate's V(p, t, s) (milliseconds)."""
    circ = circ or CirculationParams()
    V = lambda p, t: sur.volume_ml(p, t, s_act)  # noqa: E731
    edv = V(circ.edp, 0.0)
    times, P, Vol, PA, phases = [0.0], [circ.edp], [edv], [circ.p_aortic_diastolic], ["ivc"]
    p, p_c, phase, v_prev, v_ref = circ.edp, circ.p_aortic_diastolic, "ivc", edv, edv
    beta = 1.0 / (1.0 + dt_ms / 1000.0 / (circ.r_periph * circ.c_art))
    t_fill, p_fill = None, None
    t_eject, reverse_steps = None, 0
    n = int(circ.cycle_ms / dt_ms)
    for k in range(1, n + 1):
        t = k * dt_ms
        for _ in range(3):
            if phase in ("ivc", "ivr"):
                p_new = _secant(lambda q: V(q, t) - v_ref, p, p + 5.0)
                v = V(p_new, t)
                p_c_new = beta * p_c
                if phase == "ivc" and p_new >= p_c_new:
                    phase, t_eject, reverse_steps = "ejection", t, 0
                    continue
                if phase == "ivr" and p_new <= circ.p_atrial:
                    phase, t_fill, p_fill = "filling", t, max(p_new, 1.0)
                    continue
            elif phase == "ejection":
                a = beta / circ.c_art + circ.z_char / (dt_ms / 1000.0)  # mmHg / mL
                v_star = v_prev + beta * p_c / a
                p_new = _secant(lambda q: q - a * (v_star - V(q, t)), p, p + 5.0)
                v = V(p_new, t)
                q_flow = (v_prev - v) / (dt_ms / 1000.0)
                # the surrogate's V(p, t) carries small noise: close the aortic valve only on
                # sustained reverse flow after a minimum ejection period
                reverse_steps = reverse_steps + 1 if q_flow < 0 else 0
                if reverse_steps >= 2 and t - t_eject >= min_ejection_ms:
                    phase, v_ref = "ivr", v_prev
                    continue
                q_flow = max(q_flow, 0.0)
                p_c_new = beta * (p_c + q_flow * dt_ms / 1000.0 / circ.c_art)
            else:
                frac = min((t - t_fill) / max(circ.cycle_ms - t_fill, dt_ms), 1.0)
                p_new = p_fill + (circ.edp - p_fill) * frac
                v = V(p_new, t)
                p_c_new = beta * p_c
            break
        p, p_c, v_prev = p_new, p_c_new, v
        times.append(t)
        P.append(p)
        Vol.append(v)
        PA.append(p if phase == "ejection" else p_c)
        phases.append(phase)
    Vol = np.array(Vol)
    edv_, esv = float(Vol[0]), float(Vol.min())
    return {
        "times": np.array(times), "pressure_mmhg": np.array(P), "volume_ml": Vol, "aortic_mmhg": np.array(PA),
        "phase": phases,
        "metrics": {"edv_ml": edv_, "esv_ml": esv, "stroke_volume_ml": edv_ - esv,
                    "ejection_fraction_pct": 100 * (edv_ - esv) / edv_, "peak_lv_pressure_mmhg": float(np.max(P)),
                    "contractility_scale": s_act},
    }


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
