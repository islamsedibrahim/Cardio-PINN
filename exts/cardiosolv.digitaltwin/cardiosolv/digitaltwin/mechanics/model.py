"""Stage 6a: hyperelastic LV mechanics on the user's myocardium (PyTorch).

Total potential energy, as in Cardio-PINN (Buoso et al., MedIA 2021):

    Pi(u) = sum_e V_e [ s * Psi_HO(C) + Psi_act(C; T_a,e(t)) ] + Pi_cavity(V_cav(u)) + Pi_bc(u)

* ``Psi_HO``: Holzapfel-Ogden with Cardio-PINN's parameters (Sack et al. 2018),
  isochoric invariants and a volumetric penalty.
* ``Psi_act``: Cardio-PINN active term ``T_a/2 [(I4f-1) + eta((I4s-1)+(I4n-1))]`` (eta=0 default),
  with ``T_a`` per element driven by the EP activation time
  (electromechanical coupling).
* ``Pi_cavity``: ``-p V`` (prescribed pressure) or ``alpha/2 (V* - V)^2``
  (isovolumetric phases and 3-element Windkessel ejection, see ``cycle.py``).
* ``Pi_bc``: basal longitudinal anchoring + pericardial springs.

Equilibrium is found by L-BFGS on the nodal displacements. The same energy is
the physics loss of the Cardio-PINN surrogate (``surrogate/``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - reported in the UI
    torch = None

from ..core.coordinates import VentricularCoordinates
from ..core.geometry_layer import ENDO
from ..core.volume_mesh import TetMesh

MMHG = 133.322  # Pa


@dataclass
class MaterialParams:
    """Holzapfel-Ogden (Pa) - Sack et al. 2018, as used in Cardio-PINN."""

    a_iso: float = 1.05e3
    b_iso: float = 7.52
    a_f: float = 3.465e3
    b_f: float = 14.472
    a_s: float = 0.481e3
    b_s: float = 12.548
    a_fs: float = 0.283e3
    b_fs: float = 3.088
    bulk: float = 3.5e5  # volumetric penalty (Cardio-PINN: 1.05e6; lower limits P1 locking)
    stiff_scale: float = 0.75


@dataclass
class MechanicsConfig:
    material: MaterialParams = field(default_factory=MaterialParams)
    t_max_kpa: float = 85.0  # Cardio-PINN max_act = 0.85e5 Pa
    # Cardio-PINN uses 0.3; with isochoric invariants (I4s-1)+(I4n-1) acts as an isotropic
    # active stiffening that opposes wall thickening (EF 40% vs 62% on the same anatomy), so the
    # default here is pure fibre-directed active stress.
    eta_transverse: float = 0.0
    twitch_rise_ms: float = 70.0
    twitch_decay_ms: float = 80.0
    twitch_extra_ms: float = 30.0  # active tension outlasts the AP by this much
    k_base_axial: float = 2.0e3  # Pa/mm basal longitudinal spring (per node area-free)
    k_base_plane: float = 50.0
    k_pericardium: float = 20.0
    max_iter: int = 250
    tol_grad: float = 1e-7
    device: str = "cpu"


def twitch(tau, apd, cfg: MechanicsConfig):
    """Normalised active-tension transient vs time since activation (ms)."""
    dur = apd + cfg.twitch_extra_ms
    rise = np.tanh(np.clip(tau, 0, None) / cfg.twitch_rise_ms) ** 2
    decay = np.tanh(np.clip(dur - tau, 0, None) / cfg.twitch_decay_ms) ** 2
    return np.where((tau > 0) & (tau < dur), rise * decay, 0.0)


class MechanicsModel:
    def __init__(self, mesh: TetMesh, coords: VentricularCoordinates, long_axis, cfg: MechanicsConfig = None,
                 activation_time=None, apd=None):
        if torch is None:
            raise RuntimeError("PyTorch is required for the biomechanics stage (bundled with Isaac Sim).")
        self.cfg = cfg or MechanicsConfig()
        self.mesh = mesh
        self.dtype = torch.float64
        dev = torch.device(self.cfg.device)
        self.device = dev
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=dev)  # noqa: E731

        X = mesh.points
        self.X = T(X)
        self.tets = torch.as_tensor(mesh.tets, device=dev)
        p = X[mesh.tets]
        Dm = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
        self.Dm_inv = T(np.linalg.inv(Dm))
        self.vol = T(np.abs(np.linalg.det(Dm)) / 6.0)
        self.f0, self.s0, self.n0 = T(coords.fiber), T(coords.sheet), T(coords.normal)
        self.total_volume = float(self.vol.sum())

        # cavity surface (endocardium) and its rim
        endo = mesh.boundary_faces[mesh.face_labels == ENDO]
        if len(endo) == 0:
            raise RuntimeError("No endocardial faces on the computational mesh.")
        e = np.concatenate([endo[:, [0, 1]], endo[:, [1, 2]], endo[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        rim = np.unique(e[cnt[inv.ravel()] == 1])
        if len(rim) < 3:
            xl = long_axis.project(X[np.unique(endo)])
            rim = np.unique(endo)[xl >= np.percentile(xl, 95)]
        self.endo_faces = torch.as_tensor(endo, device=dev)
        self.rim = torch.as_tensor(rim, device=dev)
        self.cavity_sign = 1.0
        v0 = float(self.cavity_volume(torch.zeros_like(self.X)))
        self.cavity_sign = 1.0 if v0 > 0 else -1.0
        self.V0 = abs(v0)

        # boundary conditions
        sets = mesh.node_sets()
        base = sets["base"]
        if len(base) < 3:
            xl = long_axis.project(X)
            base = np.nonzero(xl >= 0.97)[0]
        self.base = torch.as_tensor(base, device=dev)
        self.axis = T(long_axis.direction)
        epi = sets["epi"]
        self.epi = torch.as_tensor(epi, device=dev)
        # outward epicardial node normals
        surf_n = np.zeros_like(X)
        bf = mesh.boundary_faces
        fn = np.cross(X[bf[:, 1]] - X[bf[:, 0]], X[bf[:, 2]] - X[bf[:, 0]]) * 0.5
        for k in range(3):
            np.add.at(surf_n, bf[:, k], fn)
        self.epi_area_n = T(surf_n[epi])  # area-weighted normals

        # electromechanical coupling
        self.act_el = None
        self.apd_el = None
        if activation_time is not None:
            self.act_el = np.asarray(activation_time)[mesh.tets].mean(1)
            self.apd_el = np.asarray(apd)[mesh.tets].mean(1) if apd is not None else np.full(mesh.n_tets, 280.0)

    # ------------------------------------------------------------------
    def active_tension(self, t_ms, scale=1.0):
        """Element active tension (Pa) at time t."""
        if self.act_el is None:
            return np.full(self.mesh.n_tets, scale * self.cfg.t_max_kpa * 1e3 * float(twitch(np.array([t_ms]), 280.0, self.cfg)[0]))
        return scale * self.cfg.t_max_kpa * 1e3 * twitch(t_ms - self.act_el, self.apd_el, self.cfg)

    def deformation_gradient(self, u):
        x = self.X + u
        xt = x[self.tets]
        Ds = torch.stack([xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 0], xt[:, 3] - xt[:, 0]], dim=2)
        return Ds @ self.Dm_inv

    def invariants(self, F):
        J = torch.linalg.det(F)
        C = F.transpose(1, 2) @ F
        Jc = torch.clamp(J, min=1e-6)
        Cb = C * Jc[:, None, None] ** (-2.0 / 3.0)
        I1 = Cb.diagonal(dim1=1, dim2=2).sum(-1)
        q = lambda a, b: torch.einsum("ei,eij,ej->e", a, Cb, b)  # noqa: E731
        return J, I1, q(self.f0, self.f0), q(self.s0, self.s0), q(self.n0, self.n0), q(self.f0, self.s0)

    def strain_energy(self, u, Ta):
        m = self.cfg.material
        F = self.deformation_gradient(u)
        J, I1, I4f, I4s, I4n, I8 = self.invariants(F)
        ex = lambda z: torch.exp(torch.clamp(z, max=40.0))  # noqa: E731
        relu = torch.nn.functional.relu
        psi = (m.a_iso / (2 * m.b_iso) * (ex(m.b_iso * (I1 - 3)) - 1)
               + m.a_f / (2 * m.b_f) * (ex(m.b_f * relu(I4f - 1) ** 2) - 1)
               + m.a_s / (2 * m.b_s) * (ex(m.b_s * relu(I4s - 1) ** 2) - 1)
               + m.a_fs / (2 * m.b_fs) * (ex(m.b_fs * I8**2) - 1)) * m.stiff_scale
        psi = psi + 0.5 * m.bulk * (J - 1) ** 2 + 1e3 * m.bulk * relu(0.3 - J) ** 3
        eta = self.cfg.eta_transverse
        psi_act = 0.5 * Ta * ((I4f - 1) + eta * ((I4s - 1) + (I4n - 1)))
        return ((psi + psi_act) * self.vol).sum()

    def cavity_volume(self, u):
        x = self.X + u
        c = x[self.rim].mean(0)
        f = self.endo_faces
        a, b, d = x[f[:, 0]] - c, x[f[:, 1]] - c, x[f[:, 2]] - c
        return -self.cavity_sign * (a * torch.cross(b, d, dim=1)).sum() / 6.0

    def bc_energy(self, u):
        cfg = self.cfg
        ub = u[self.base]
        ax = (ub @ self.axis)
        plane = ub - ax[:, None] * self.axis
        e = 0.5 * cfg.k_base_axial * (ax**2).sum() + 0.5 * cfg.k_base_plane * (plane**2).sum()
        un = (u[self.epi] * self.epi_area_n).sum(1)
        e = e + 0.5 * cfg.k_pericardium * (torch.relu(un) ** 2).sum() / max(self.mesh.spacing, 1e-9) ** 2
        return e

    def total_energy(self, u, Ta, mode, p_pa=0.0, alpha=0.0, v_star=0.0):
        W = self.strain_energy(u, Ta) + self.bc_energy(u)
        V = self.cavity_volume(u)
        if mode == "pressure":
            W = W - p_pa * V
        else:
            W = W + 0.5 * alpha * (v_star - V) ** 2
        return W, V

    # ------------------------------------------------------------------
    def solve(self, u0, Ta, mode="pressure", p_pa=0.0, alpha=0.0, v_star=0.0, max_iter=None):
        """Static equilibrium; returns (u, cavity volume mm^3, cavity pressure Pa, info)."""
        cfg = self.cfg
        u = torch.as_tensor(u0, dtype=self.dtype, device=self.device).clone().requires_grad_(True)
        Ta_t = torch.as_tensor(Ta, dtype=self.dtype, device=self.device)
        scale = 1.0 / (cfg.material.a_iso * self.total_volume)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=max_iter or cfg.max_iter, history_size=30,
                                line_search_fn="strong_wolfe", tolerance_grad=cfg.tol_grad,
                                tolerance_change=1e-14)
        state = {}

        def closure():
            opt.zero_grad()
            W, V = self.total_energy(u, Ta_t, mode, p_pa, alpha, v_star)
            loss = W * scale
            loss.backward()
            state["V"] = float(V.detach())
            state["W"] = float(W.detach())
            return loss

        opt.step(closure)
        with torch.no_grad():
            V = float(self.cavity_volume(u))
            p = p_pa if mode == "pressure" else alpha * (v_star - V)
            g = u.grad.abs().max().item() if u.grad is not None else 0.0
        n_iter = opt.state[opt._params[0]].get("n_iter", 0)
        return u.detach().cpu().numpy(), V, p, {"iterations": n_iter, "grad_inf": g, "energy": state.get("W", 0.0)}

    # ------------------------------------------------------------------
    def potential_batch(self, U, Ta, p_pa):
        """Pressure-loaded total potential for a batch: U (B,N,3), Ta (B,E), p (B,) -> (B,), V (B,)."""
        m = self.cfg.material
        x = self.X[None] + U
        xt = x[:, self.tets]  # (B,E,4,3)
        Ds = torch.stack([xt[:, :, 1] - xt[:, :, 0], xt[:, :, 2] - xt[:, :, 0], xt[:, :, 3] - xt[:, :, 0]], dim=3)
        F = Ds @ self.Dm_inv[None]
        J = torch.linalg.det(F)
        C = F.transpose(2, 3) @ F
        Cb = C * torch.clamp(J, min=1e-6)[..., None, None] ** (-2.0 / 3.0)
        I1 = Cb.diagonal(dim1=2, dim2=3).sum(-1)
        q = lambda a, b: torch.einsum("ei,beij,ej->be", a, Cb, b)  # noqa: E731
        I4f, I4s, I4n, I8 = q(self.f0, self.f0), q(self.s0, self.s0), q(self.n0, self.n0), q(self.f0, self.s0)
        ex = lambda z: torch.exp(torch.clamp(z, max=40.0))  # noqa: E731
        relu = torch.nn.functional.relu
        psi = (m.a_iso / (2 * m.b_iso) * (ex(m.b_iso * (I1 - 3)) - 1)
               + m.a_f / (2 * m.b_f) * (ex(m.b_f * relu(I4f - 1) ** 2) - 1)
               + m.a_s / (2 * m.b_s) * (ex(m.b_s * relu(I4s - 1) ** 2) - 1)
               + m.a_fs / (2 * m.b_fs) * (ex(m.b_fs * I8**2) - 1)) * m.stiff_scale
        psi = psi + 0.5 * m.bulk * (J - 1) ** 2 + 1e3 * m.bulk * relu(0.3 - J) ** 3
        psi = psi + 0.5 * Ta * ((I4f - 1) + self.cfg.eta_transverse * ((I4s - 1) + (I4n - 1)))
        W = (psi * self.vol[None]).sum(1)
        # boundary springs
        cfg = self.cfg
        ub = U[:, self.base]
        ax = ub @ self.axis
        plane = ub - ax[..., None] * self.axis
        W = W + 0.5 * cfg.k_base_axial * (ax**2).sum(1) + 0.5 * cfg.k_base_plane * (plane**2).sum((1, 2))
        un = (U[:, self.epi] * self.epi_area_n[None]).sum(2)
        W = W + 0.5 * cfg.k_pericardium * (relu(un) ** 2).sum(1) / max(self.mesh.spacing, 1e-9) ** 2
        V = self.cavity_volume_batch(U)
        return W - p_pa * V, V

    def cavity_volume_batch(self, U):
        x = self.X[None] + U
        c = x[:, self.rim].mean(1, keepdim=True)
        f = self.endo_faces
        a, b, d = x[:, f[:, 0]] - c, x[:, f[:, 1]] - c, x[:, f[:, 2]] - c
        return -self.cavity_sign * (a * torch.cross(b, d, dim=2)).sum((1, 2)) / 6.0

    def element_fields(self, u, Ta):
        """Fibre stretch, fibre Cauchy stress (kPa), J and Green-Lagrange fibre strain per element."""
        with torch.no_grad():
            u = torch.as_tensor(u, dtype=self.dtype, device=self.device)
            F = self.deformation_gradient(u)
            J = torch.linalg.det(F)
            Ff = torch.einsum("eij,ej->ei", F, self.f0)
            lam = torch.linalg.norm(Ff, dim=1)
            m = self.cfg.material
            I4 = lam**2 * torch.clamp(J, min=1e-6) ** (-2.0 / 3.0)
            # passive fibre stress (dominant HO fibre term) + active, pushed forward
            dpsi = m.stiff_scale * m.a_f * torch.relu(I4 - 1) * torch.exp(torch.clamp(m.b_f * torch.relu(I4 - 1) ** 2, max=40))
            Ta_t = torch.as_tensor(Ta, dtype=self.dtype, device=self.device)
            sigma_ff = (2 * dpsi + Ta_t) * lam**2 / torch.clamp(J, min=1e-6)
            return {
                "fiber_stretch": lam.cpu().numpy(),
                "fiber_strain": (0.5 * (lam**2 - 1)).cpu().numpy(),
                "fiber_stress_kpa": (sigma_ff / 1e3).cpu().numpy(),
                "jacobian": J.cpu().numpy(),
            }

    def config_dict(self):
        return asdict(self.cfg)
