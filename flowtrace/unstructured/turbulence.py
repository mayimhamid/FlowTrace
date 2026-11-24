r"""RANS turbulence models for the pressure-based unstructured solver.

Models
------
``k-omega-sst``
    Menter's SST model in its 2003 form (Menter, Kuntz & Langtry, *Turbulence,
    Heat and Mass Transfer 4*, 2003), the form used by Fluent's default
    *SST k-omega* model and NASA's Turbulence Modeling Resource (TMR):

    .. math::
        \nabla\cdot(\rho U k) = \nabla\cdot((\mu + \sigma_k \mu_t)\nabla k) + \tilde P_k - \beta^* \rho k \omega

        \nabla\cdot(\rho U \omega) = \nabla\cdot((\mu + \sigma_\omega \mu_t)\nabla \omega) + \gamma \rho S^2
            - \beta \rho \omega^2 + 2 (1 - F_1) \rho \sigma_{\omega 2} \frac{\nabla k\cdot\nabla\omega}{\omega}

    with :math:`\mu_t = \rho a_1 k / \max(a_1 \omega, S F_2)` and
    :math:`\tilde P_k = \min(\mu_t S^2, 10 \beta^* \rho k \omega)`.

``k-epsilon-realizable``
    Shih et al. (*Computers & Fluids* 24, 1995) realizable k-epsilon with the
    variable :math:`C_\mu` used by Fluent (A0 = 4.04).

Wall treatment
--------------
* SST: y+-insensitive ("automatic"). The near-wall cell value of omega blends
  the viscous-sublayer and log-layer solutions,
  :math:`\omega_P = \sqrt{\omega_{vis}^2 + \omega_{log}^2}`, and the wall
  shear stress comes from Spalding's single-formula law of the wall, which is
  exactly the laminar stress for y+ < 5 and the log law for y+ > 30. The model
  is therefore accurate on meshes that resolve the sublayer (y+ ~ 1) and
  consistent on coarser ones.
* Realizable k-epsilon: scalable standard wall functions (Launder & Spalding
  with y* limited to 11.225, as Fluent's *Scalable Wall Functions*): wall
  shear from the log law, wall-cell production from the wall shear, and the
  near-wall epsilon fixed to its equilibrium value.

Boundary conditions
-------------------
Inlets (and backflow at outlets) use the turbulence intensity ``I`` and the
viscosity ratio ``mu_t/mu`` (Fluent's defaults 5 % and 10), or explicit k
and omega/epsilon values; outlets and symmetry planes are zero-gradient.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

KAPPA = 0.41  # von Karman constant in SST and Spalding's wall law
# Fluent's standard / scalable wall functions pair E = 9.793 with kappa = 0.4187 (log-law intercept
# B = ln(E)/kappa = 5.45). Combined with 0.41, E = 9.793 gives B = 5.57 and under-predicts the wall
# shear (measured: flat plate Cf -2 %, channel U+ up to +1.7 % against DNS).
KAPPA_WALL = 0.4187
E_WALL = 9.793
E_SPALDING = 9.8
B_SPALDING = math.log(E_SPALDING) / KAPPA
CMU = 0.09
YSTAR_LIM = 11.225
PR_T = 0.85  # turbulent Prandtl number (energy)

MODELS = ("laminar", "k-omega-sst", "k-epsilon-realizable", "k-epsilon-standard")


def spalding_uplus(re_y: np.ndarray, iters: int = 30, *,
                   kappa: float = KAPPA, B: float = B_SPALDING, tol: float = 1e-10, xp=np) -> np.ndarray:
    """Solve u+ y+(u+) = Re_y for the Spalding wall law."""
    re_y = xp.maximum(xp.asarray(re_y, dtype=float), 0.0)
    eb = math.exp(-kappa * B)
    up = xp.minimum(xp.sqrt(re_y), xp.log1p(re_y) / kappa + B)
    up = xp.where(re_y > 0, up, 1.0)
    for _ in range(iters):
        ku = kappa * up
        exp_neg = xp.exp(-ku)
        tail = -xp.expm1(-ku) - exp_neg * (ku + ku**2 / 2)
        yp_scaled = up * exp_neg + eb * (tail - exp_neg * ku**3 / 6)
        dyp_scaled = exp_neg + kappa * eb * tail
        # Scale g and its analytic derivative by exp(-kappa u+)/u+.
        step = (yp_scaled - (re_y * exp_neg) / up) / (yp_scaled / up + dyp_scaled)
        up = xp.where(re_y > 0, xp.maximum(up - step, 0.5 * up), up)
        if xp is np and xp.all((re_y == 0) | (xp.abs(step) <= tol * up)):
            break
    return xp.where(re_y > 0, up, 0.0)


def spaldings_law_wall_shear(
    u_tan: np.ndarray,
    y_wall: np.ndarray,
    rho: np.ndarray,
    nu: np.ndarray,
    kappa: float = KAPPA,
    B: float = B_SPALDING,
    max_iter: int = 30,
    tol: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves Spalding's C-infinity unified law of the wall via vectorized Newton-Raphson.
    Eliminates buffer-layer switching discontinuities at y+ = 11.225.

    Returns
    -------
    tau_w : (n_wall_faces,) ndarray
        Wall shear stress magnitude.
    u_tau : (n_wall_faces,) ndarray
        Friction velocity.
    y_plus : (n_wall_faces,) ndarray
        Dimensionless wall distance.
    """
    u_mag = np.abs(np.asarray(u_tan, dtype=float))
    nu = np.maximum(nu, 1e-15)
    u_plus = spalding_uplus(u_mag * y_wall / nu, max_iter, kappa=kappa, B=B, tol=tol)
    u_tau = np.divide(u_mag, u_plus, out=np.zeros_like(u_plus), where=u_plus > 0)
    y_plus = y_wall * u_tau / nu
    tau_w = rho * (u_tau**2)
    return tau_w, u_tau, y_plus


def linearize_kepsilon_sources(
    rho: np.ndarray,
    nu: np.ndarray,
    k: np.ndarray,
    eps: np.ndarray,
    P_k: np.ndarray,
    S_mag: np.ndarray,
    model: str = "realizable",
    C2: float = 1.9,
    C1_eps: float = 1.44,
    C2_eps: float = 1.92,
    k_min: float = 1e-12,
    eps_min: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Patankar negative-slope source term linearization (S = S_C + S_P * phi)
    for Realizable and Standard k-epsilon turbulence models.
    Guarantees S_C >= 0 and S_P <= 0 everywhere for M-matrix positivity preservation.
    """
    k_safe = np.maximum(k, k_min)
    eps_safe = np.maximum(eps, eps_min)
    P_k_safe = np.maximum(P_k, 0.0)

    Sc_k = P_k_safe
    Sp_k = -(rho * eps_safe) / k_safe

    if model.lower() == "realizable":
        eta = (S_mag * k_safe) / eps_safe
        C1 = np.maximum(0.43, eta / (eta + 5.0))
        prod_eps = rho * C1 * S_mag * eps_safe

        sqrt_nu_eps = np.sqrt(np.maximum(nu * eps_safe, 0.0))
        denom = k_safe + sqrt_nu_eps
        denom_sq = denom**2

        Sp_eps = -rho * C2 * (eps_safe * (2.0 * k_safe + 1.5 * sqrt_nu_eps)) / denom_sq
        Sc_dest = rho * C2 * ((eps_safe**2) * (k_safe + 0.5 * sqrt_nu_eps)) / denom_sq
        Sc_eps = prod_eps + Sc_dest

    elif model.lower() == "standard":
        eps_over_k = eps_safe / k_safe
        prod_eps = C1_eps * eps_over_k * P_k_safe
        Sp_eps = -2.0 * rho * C2_eps * eps_over_k
        Sc_dest = rho * C2_eps * (eps_safe**2) / k_safe
        Sc_eps = prod_eps + Sc_dest
    else:
        raise ValueError(f"Unsupported k-epsilon model variant: {model}")

    return Sc_k, Sp_k, Sc_eps, Sp_eps


def assemble_and_relax_scalar_system(
    A_diag: np.ndarray,
    A_offdiag_sum: np.ndarray,
    rhs: np.ndarray,
    phi_old: np.ndarray,
    Sc: np.ndarray,
    Sp: np.ndarray,
    volume: np.ndarray,
    alpha: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Injects linearized source terms into the FVM matrix diagonal and RHS,
    then applies Patankar implicit under-relaxation.
    """
    A_P = A_diag - Sp * volume
    b_P = rhs + Sc * volume
    A_P = np.maximum(A_P, A_offdiag_sum + 1e-15)

    inv_alpha = 1.0 / alpha
    A_P_relaxed = A_P * inv_alpha
    b_P_relaxed = b_P + ((1.0 - alpha) * inv_alpha) * A_P * phi_old
    return A_P_relaxed, b_P_relaxed


def spalding_utau(xp, ut, y, nu) -> np.ndarray:
    """Friction velocity from Spalding's law for wall-parallel speed ``ut`` at distance ``y``."""
    ut_h = np.asarray(ut if xp is np else xp.asnumpy(ut), dtype=float)
    y_h = np.asarray(y if xp is np else xp.asnumpy(y), dtype=float)
    up = spalding_uplus(ut_h * y_h / nu)
    utau = np.where(up > 0, ut_h / np.maximum(up, 1e-300), 0.0)
    return utau if xp is np else xp.asarray(utau)


def strain_invariants(xp, gU, dim):
    """|S| = sqrt(2 S_ij S_ij), |Omega| = sqrt(2 W_ij W_ij), sum S_ij S_ij, S_ij S_jk S_ki."""
    S2 = 0.0
    W2 = 0.0
    Sij = [[0.5 * (gU[i][:, j] + gU[j][:, i]) for j in range(dim)] for i in range(dim)]
    for i in range(dim):
        for j in range(dim):
            S2 = S2 + Sij[i][j] ** 2
            W2 = W2 + (0.5 * (gU[i][:, j] - gU[j][:, i])) ** 2
    S3 = 0.0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                S3 = S3 + Sij[i][j] * Sij[j][k] * Sij[k][i]
    return xp.sqrt(2.0 * S2), xp.sqrt(2.0 * W2), S2, W2, S3


class TurbulenceModel:
    """Shared machinery: wall distance, BCs, scalar transport, wall shear."""

    name = "base"
    second = "omega"  # name of the second transported variable
    default_relax = 0.8

    def __init__(self, solver):
        from .simple import INLET, OUTLET, PRESSURE, WALL

        self.s = s = solver
        fv, xp = s.fv, s.xp
        self.xp = xp
        c = s.case
        self.rho, self.mu = c.density, c.viscosity
        self.nu = self.mu / self.rho
        kind = s.bkind_h
        self.wall_b = kind == WALL
        self.inlet_b = kind == INLET
        self.open_b = (kind == OUTLET) | (kind == PRESSURE)
        h = s.be.to_numpy
        Pb = h(fv.Pb)
        nb = h(fv.nb).astype(np.float64)
        db = h(fv.db).astype(np.float64)
        # wall-normal distance of the owner cell of every wall face
        self.yb = np.abs(np.einsum("fd,fd->f", db, nb))
        self.y = self._wall_distance()
        nc = s.nc
        # wall-adjacent cells and their (smallest) wall distance
        wf = np.nonzero(self.wall_b)[0]
        self.wall_faces = wf
        self.wall_cells = np.unique(Pb[wf])
        ywc = np.full(nc, np.inf)
        np.minimum.at(ywc, Pb[wf], self.yb[wf])
        self.y_wall_cell = ywc
        self.is_wall_cell = np.isfinite(ywc)
        if self.wall_cells.size:
            self.y[self.wall_cells] = ywc[self.wall_cells]
        self.y_d = xp.asarray(self.y, dtype=s.dtype)
        self.yb_d = xp.asarray(self.yb, dtype=s.dtype)
        self.wall_b_d = xp.asarray(self.wall_b)
        self.wall_cells_d = xp.asarray(self.wall_cells)
        self.wall_distance_d = xp.asarray(ywc[self.wall_cells], dtype=s.dtype)
        self.inlet_b_d, self.open_b_d = xp.asarray(self.inlet_b), xp.asarray(self.open_b)
        self.wall_area_d = xp.where(self.wall_b_d, fv.Sb_mag, 0.0)
        self.wall_area_cell_d = fv.Cb_d @ self.wall_area_d
        # inlet / backflow values
        spd = np.linalg.norm(h(s.ub_fixed).astype(np.float64), axis=1)
        u_ref = float(spd[self.inlet_b].max()) if self.inlet_b.any() else float(c.turb_reference_velocity or 1.0)
        self.u_ref = max(u_ref, 1e-6)
        kin = np.zeros(s.fv.n_bf)
        sin = np.zeros(s.fv.n_bf)
        for z in s.mesh.boundary_zones():
            spec = s.zone_specs[z.name]
            m = h(fv.bzone) == z.id
            I = spec.turb_intensity if spec.turb_intensity is not None else c.turb_intensity
            r = spec.turb_viscosity_ratio if spec.turb_viscosity_ratio is not None else c.turb_viscosity_ratio
            U = np.where(spd[m] > 0, spd[m], self.u_ref)
            k = 1.5 * (I * U) ** 2 if spec.k is None else np.full(m.sum(), spec.k)
            kin[m] = np.maximum(k, 1e-12 * self.u_ref ** 2)
            sin[m] = self.second_from(kin[m], r, getattr(spec, self.second))
        self.kb_in = xp.asarray(kin, dtype=s.dtype)
        self.sb_in = xp.asarray(sin, dtype=s.dtype)
        # initial field
        init = c.turb_init or {}
        k0 = init.get("k", 1.5 * (c.turb_intensity * self.u_ref) ** 2)
        s0 = init.get(self.second, float(self.second_from(np.array([k0]), c.turb_viscosity_ratio, None)[0]))
        self.k = xp.full(nc, k0, dtype=s.dtype)
        self.w = xp.full(nc, s0, dtype=s.dtype)  # omega or epsilon
        self.mut = xp.zeros(nc, dtype=s.dtype)
        self.k_min = 1e-14 * self.u_ref ** 2
        self.w_min = 1e-14
        self.relax_k = c.relax_k if c.relax_k is not None else self.default_relax
        self.relax_w = c.relax_w if c.relax_w is not None else self.default_relax
        self._mu_w = None  # effective wall viscosity per boundary face
        self.yplus = xp.zeros(s.fv.n_bf, dtype=s.dtype)
        self._update_mut(None)

    # ------------------------------------------------------------------
    def second_from(self, k, ratio, explicit):
        raise NotImplementedError

    def _wall_distance(self) -> np.ndarray:
        """Distance from every cell centre to the nearest wall face (exact for 2-D segments)."""
        s = self.s
        mesh, geo = s.mesh, s.fv.geometry
        xc = geo.cell_centre
        wall_zone_ids = [z.id for z in mesh.boundary_zones() if s.zone_specs[z.name].kind == "wall"]
        wf = np.nonzero(np.isin(mesh.face_zone, wall_zone_ids))[0]
        if wf.size == 0:
            return np.full(s.nc, 1e30)
        from scipy.spatial import cKDTree

        fc = geo.face_centre[wf]
        kq = min(8, wf.size)
        _, idx = cKDTree(fc).query(xc, k=kq)
        idx = np.atleast_2d(idx).reshape(xc.shape[0], kq)
        best = np.full(xc.shape[0], np.inf)
        if mesh.dim == 2:
            a = mesh.nodes[mesh.face_nodes[mesh.face_ptr[wf]]]
            b = mesh.nodes[mesh.face_nodes[mesh.face_ptr[wf] + 1]]
            for j in range(kq):
                f = idx[:, j]
                ab = b[f] - a[f]
                t = np.clip(np.einsum("cd,cd->c", xc - a[f], ab) / np.maximum(np.einsum("cd,cd->c", ab, ab), 1e-300), 0, 1)
                d = np.linalg.norm(xc - (a[f] + t[:, None] * ab), axis=1)
                best = np.minimum(best, d)
        else:
            n = geo.face_area_vec[wf]
            area = np.linalg.norm(n, axis=1)
            n = n / area[:, None]
            rad = np.sqrt(area / np.pi) * 1.2
            for j in range(kq):
                f = idx[:, j]
                v = xc - fc[f]
                dn = np.abs(np.einsum("cd,cd->c", v, n[f]))
                dt = np.linalg.norm(v - np.einsum("cd,cd->c", v, n[f])[:, None] * n[f], axis=1)
                d = np.where(dt <= rad[f], dn, np.sqrt(dn ** 2 + (dt - rad[f]) ** 2))
                best = np.minimum(best, d)
        return np.maximum(best, 1e-300)

    # ------------------------------------------------------------------
    def wall_shear_viscosity(self, U) -> np.ndarray:
        """Effective viscosity mu_w on every boundary face (walls only differ from mu)."""
        raise NotImplementedError

    def momentum_viscosity(self, U):
        """(interior-face, boundary-face) effective viscosity for the momentum equations."""
        s, xp = self.s, self.xp
        fv = s.fv
        mut_f = fv.face_interp(self.mut)
        mub = self.mu + self.mut[fv.Pb]
        mu_w = self.wall_shear_viscosity(U)
        mub = xp.where(self.wall_b_d, mu_w, mub)
        return self.mu + mut_f, mub

    def energy_diffusivity(self, k_over_cp: float):
        fv = self.s.fv
        mut_f = fv.face_interp(self.mut)
        pr_t = self.s.case.prandtl_turbulent
        gb = k_over_cp + self.xp.where(self.wall_b_d, 0.0, self.mut[fv.Pb] / pr_t)
        return k_over_cp + mut_f / pr_t, gb

    def transpose_stress(self, gU, viscosity):
        """Explicit transpose and trace terms in the symmetric deviatoric stress."""
        s, xp = self.s, self.xp
        fv, dim = s.fv, s.dim
        mu_f, mu_b = viscosity
        div_u = sum(gU[j][:, j] for j in range(dim))
        div_f = fv.face_interp(div_u)
        # Wall functions prescribe wall traction. Apply the correction on
        # interior faces and open boundaries.
        mu_b = mu_b * (s.m_inlet + s.m_pres)
        out = xp.zeros((s.nc, dim), dtype=s.dtype)
        for k in range(dim):
            gT = xp.stack([gU[j][:, k] for j in range(dim)], axis=1)
            interior = xp.sum(fv.face_interp_vec(gT) * fv.S, axis=1) - (2.0 / 3.0) * div_f * fv.S[:, k]
            boundary = xp.sum(gT[fv.Pb] * fv.Sb, axis=1) - (2.0 / 3.0) * div_u[fv.Pb] * fv.Sb[:, k]
            out[:, k] = fv.Cdiff_d @ (mu_f * interior) + fv.Cb_d @ (mu_b * boundary)
        return out

    # ------------------------------------------------------------------
    def _bvals(self, phi, inflow_vals, wall_zero_grad=True):
        """Boundary values: fixed at inlets and outlet backflow, zero-gradient elsewhere."""
        s, xp = self.s, self.xp
        fv = s.fv
        fixed = self.inlet_b_d | (self.open_b_d & (s.Fb < 0))
        vals = xp.where(fixed, inflow_vals, phi[fv.Pb])
        return vals, fixed.astype(s.dtype)

    def _transport(self, phi, inflow_vals, sigma_mut, Sc, Sp, relax, fixed_cells=None, fixed_vals=None,
                   floor=0.0, name=None):
        s, xp = self.s, self.xp
        fv = s.fv
        phib, fixed = self._bvals(phi, inflow_vals)
        grad = fv.gradient(phi, phib)
        gam_c = self.mu + sigma_mut * self.mut  # cell diffusivity
        gf = fv.face_interp(gam_c)
        gb = gam_c[fv.Pb]
        diag, aPN, aNP, Db = s._coefficients((gf, gb), fixed)
        rhs = s._rhs((gf, gb), phi, grad, phib, Db, Sc * fv.vol, scheme=s.case.turb_scheme)
        diag = diag + Sp * fv.vol
        diag, rhs = s._time_terms(name, phi, diag, rhs)  # transient runs: d(rho phi)/dt (no-op when steady)
        if fixed_cells is not None and fixed_cells.size:
            big = 1e10 * xp.max(diag)
            fc = xp.asarray(fixed_cells)
            diag[fc] = diag[fc] + big
            rhs[fc] = rhs[fc] + big * xp.asarray(fixed_vals, dtype=s.dtype)
        A = fv.matrix(diag, aPN, aNP)
        # residual over the cells the equation actually controls: not the fixed near-wall
        # cells, and not cells held at the lower bound (e.g. free-stream k decaying to the
        # floor upstream of a flat plate), whose clipped residual can never vanish
        mask = phi > floor * (1.0 + 1e-9)
        if fixed_cells is not None and fixed_cells.size:
            mask[xp.asarray(fixed_cells)] = False
        r = xp.where(mask, xp.abs(rhs - A @ phi), 0.0)
        den = xp.where(mask, xp.abs(diag * phi), 0.0)
        res = float(xp.sum(r)) / max(float(xp.sum(den)), 1e-300)
        dr = diag / relax
        Ar = fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - relax) / relax) * diag * phi
        new, _ = s._solve(Ar, b, phi, dr, tol=1e-3, maxiter=60)
        new = xp.maximum(new, floor)
        return new, res

    def _gradient(self, phi, inflow_vals):
        b, _ = self._bvals(phi, inflow_vals)
        return self.s.fv.gradient(phi, b)

    # ------------------------------------------------------------------
    def fields(self) -> Dict[str, np.ndarray]:
        h = self.s.be.to_numpy
        out = {"turb-kinetic-energy": h(self.k).astype(np.float64),
               self.field_name: h(self.w).astype(np.float64),
               "viscosity-turb": h(self.mut).astype(np.float64),
               "viscosity-ratio": h(self.mut).astype(np.float64) / self.mu,
               "wall-distance": self.y.copy()}
        yp = np.zeros(self.s.nc)
        Pb = h(self.s.fv.Pb)
        if self.wall_faces.size:
            np.maximum.at(yp, Pb[self.wall_faces], h(self.yplus)[self.wall_faces])
        out["y-plus"] = yp
        return out

    def state(self) -> dict:
        h = self.s.be.to_numpy
        return {"k": h(self.k), "w": h(self.w), "mut": h(self.mut)}

    def load(self, st: dict):
        xp, dt = self.xp, self.s.dtype
        if "k" in st:
            self.k = xp.asarray(st["k"], dtype=dt)
            self.w = xp.asarray(st["w"], dtype=dt)
            self.mut = xp.asarray(st["mut"], dtype=dt)

    def _wall_tangential_speed(self, U):
        fv, xp = self.s.fv, self.xp
        relative = U[fv.Pb] - self.s.ub_fixed
        tangential = relative - xp.sum(relative * fv.nb, axis=1)[:, None] * fv.nb
        return xp.linalg.norm(tangential, axis=1)

    def _wall_shear_device(self):
        xp = self.xp
        speed = self._wall_tangential_speed(self.s.U)
        mu_w = self.wall_shear_viscosity(self.s.U)
        distance = xp.where(self.wall_b_d, self.yb_d, 1.0)
        return xp.where(self.wall_b_d, mu_w * speed / distance, 0.0)

    def wall_shear(self) -> np.ndarray:
        """Report the wall shear magnitude on every boundary face."""
        return self.s.be.to_numpy(self._wall_shear_device())


class KOmegaSST(TurbulenceModel):
    name = "k-omega-sst"
    second = "omega"
    field_name = "specific-diss-rate"
    sk1, sw1, b1 = 0.85, 0.5, 0.075
    sk2, sw2, b2 = 1.0, 0.856, 0.0828
    bstar, a1 = 0.09, 0.31
    g1 = b1 / bstar - sw1 * KAPPA ** 2 / math.sqrt(bstar)
    g2 = b2 / bstar - sw2 * KAPPA ** 2 / math.sqrt(bstar)

    def second_from(self, k, ratio, explicit):
        if explicit is not None:
            return np.full(np.shape(k), float(explicit))
        return self.rho * np.asarray(k) / (self.mu * ratio)

    def _blend(self):
        xp = self.xp
        k, w, y = self.k, self.w, self.y_d
        gk = self._gradient(k, self.kb_in)
        gw = self._gradient(w, self.sb_in)
        kw = xp.sum(gk * gw, axis=1)
        cd = xp.maximum(2.0 * self.rho * self.sw2 * kw / w, 1e-10)
        sk = xp.sqrt(k)
        a = xp.maximum(sk / (self.bstar * w * y), 500.0 * self.nu / (y * y * w))
        arg1 = xp.minimum(a, 4.0 * self.rho * self.sw2 * k / (cd * y * y))
        F1 = xp.tanh(arg1 ** 4)
        arg2 = xp.maximum(2.0 * sk / (self.bstar * w * y), 500.0 * self.nu / (y * y * w))
        F2 = xp.tanh(arg2 ** 2)
        return F1, F2, kw

    def _update_mut(self, S):
        xp = self.xp
        if S is None:
            self.mut = self.rho * self.k / self.w
            return
        _, F2, _ = self._blend()
        self.mut = self.rho * self.a1 * self.k / xp.maximum(self.a1 * self.w, S * F2)

    def _omega_wall(self):
        """Near-wall cell omega: sqrt(omega_vis^2 + omega_log^2)."""
        xp = self.xp
        yc = self.wall_distance_d
        k = self.k[self.wall_cells_d]
        w_vis = 6.0 * self.nu / (self.b1 * yc**2)
        w_log = xp.sqrt(xp.maximum(k, 0.0)) / (CMU ** 0.25 * KAPPA * yc)
        return xp.hypot(w_vis, w_log)

    def wall_shear_viscosity(self, U):
        xp = self.xp
        re_y = self._wall_tangential_speed(U) * self.yb_d / self.nu
        up = spalding_uplus(re_y, xp=xp).astype(self.s.dtype, copy=False)
        ok = (re_y > 1e-12) & (up > 0)
        safe_up = xp.where(ok, up, 1.0)
        mu_w = xp.where(ok, self.mu * re_y / safe_up**2, self.mu)
        self.yplus = xp.where(self.wall_b_d & ok, re_y / safe_up, 0.0)
        return xp.where(self.wall_b_d, mu_w, self.mu)

    def solve(self, gU) -> Dict[str, float]:
        s, xp = self.s, self.xp
        S, _, _, _, _ = strain_invariants(xp, gU, s.dim)
        S2 = S * S
        F1, F2, kw = self._blend()
        rho, k, w = self.rho, self.k, self.w
        bl = lambda a, b: F1 * a + (1.0 - F1) * b  # noqa: E731
        limit = 10.0 * self.bstar * rho * k * w
        Pk = xp.minimum(self.mut * S2, limit)
        knew, rk = self._transport(k, self.kb_in, bl(self.sk1, self.sk2), Pk, self.bstar * rho * w,
                                   self.relax_k, floor=self.k_min, name="k")
        cdt = (1.0 - F1) * 2.0 * rho * self.sw2 * kw / w
        # Pk / nu_t with the small-eddy-viscosity limit taken algebraically.
        limited_strain = xp.divide(Pk, xp.where(self.mut > 0.0, self.mut, 1.0))
        limited_strain = xp.where(self.mut > 0.0, limited_strain, S2)
        Sc = bl(self.g1, self.g2) * rho * limited_strain + xp.maximum(cdt, 0.0)
        Sp = bl(self.b1, self.b2) * rho * w + xp.maximum(-cdt, 0.0) / w
        wnew, rw = self._transport(w, self.sb_in, bl(self.sw1, self.sw2), Sc, Sp, self.relax_w,
                                   fixed_cells=self.wall_cells_d, fixed_vals=self._omega_wall(), floor=self.w_min, name="w")
        self.k, self.w = knew, wnew
        self._update_mut(S)
        return {"k": rk, "omega": rw}


class KEpsilonRealizable(TurbulenceModel):
    name = "k-epsilon-realizable"
    second = "epsilon"
    field_name = "turb-diss-rate"
    default_relax = 0.5
    sk, se, C2, A0 = 1.0, 1.2, 1.9, 4.04

    def second_from(self, k, ratio, explicit):
        if explicit is not None:
            return np.full(np.shape(k), float(explicit))
        return self.rho * CMU * np.asarray(k) ** 2 / (self.mu * ratio)

    def _cmu(self, gU):
        xp = self.xp
        dim = self.s.dim
        S, _, S2, W2, S3 = strain_invariants(xp, gU, dim)
        frame = self.s.mrf_omega_cells
        if frame is not None:
            # The solver stores absolute velocity, so its antisymmetric
            # gradient is Omega_ij. Apply -2 epsilon_ijk Omega_frame,k
            # to its three independent components.
            pairs = ((1, 0, 2),) if dim == 2 else ((2, 1, 0), (0, 2, 1), (1, 0, 2))
            W2 = 2.0 * sum((0.5 * (gU[i][:, j] - gU[j][:, i]) + 2.0 * frame[:, axis])**2
                           for i, j, axis in pairs)
        St = xp.sqrt(S2)
        Wr = xp.where(St > 0, S3 / xp.maximum(St ** 3, 1e-300), 0.0)
        phi = xp.arccos(xp.clip(math.sqrt(6.0) * Wr, -1.0, 1.0)) / 3.0
        As = math.sqrt(6.0) * xp.cos(phi)
        Ustar = xp.sqrt(S2 + W2)
        return 1.0 / (self.A0 + As * self.k * Ustar / self.w), S

    def _eps_sources(self, S, k, eps, Gk):
        """Realizable epsilon equation: production rho C1 S eps, destruction with Patankar negative-slope linearization."""
        xp, rho = self.xp, self.rho
        eta = S * k / xp.maximum(eps, 1e-12)
        C1 = xp.maximum(0.43, eta / (eta + 5.0))
        prod_eps = rho * C1 * S * eps
        sqrt_nu_eps = xp.sqrt(xp.maximum(self.nu * eps, 0.0))
        denom = k + sqrt_nu_eps
        denom_sq = denom ** 2
        Sp_eps = rho * self.C2 * (eps * (2.0 * k + 1.5 * sqrt_nu_eps)) / denom_sq
        Sc_dest = rho * self.C2 * ((eps ** 2) * (k + 0.5 * sqrt_nu_eps)) / denom_sq
        return prod_eps + Sc_dest, Sp_eps

    def _update_mut(self, gU):
        if gU is None:
            self.mut = self.rho * CMU * self.k ** 2 / self.w
            return
        cmu, _ = self._cmu(gU)
        r = self.s.case.relax_mut
        self.mut = r * (self.rho * cmu * self.k ** 2 / self.w) + (1.0 - r) * self.mut

    def _ustar(self):
        return CMU ** 0.25 * self.xp.sqrt(self.xp.maximum(self.k[self.s.fv.Pb], 0.0))

    def wall_shear_viscosity(self, U):
        xp = self.xp
        us = self._ustar()
        ystar = xp.maximum(self.rho * us * self.yb_d / self.mu, YSTAR_LIM)
        mu_w = self.rho * KAPPA_WALL * us * self.yb_d / xp.log(E_WALL * ystar)
        self.yplus = xp.where(self.wall_b_d, self.rho * us * self.yb_d / self.mu, 0.0)
        return xp.where(self.wall_b_d, xp.maximum(mu_w, self.mu), self.mu)

    def solve(self, gU) -> Dict[str, float]:
        s, xp = self.s, self.xp
        rho, k, eps = self.rho, self.k, self.w
        _, S = self._cmu(gU)
        Gk = self.mut * S * S
        wc = self.wall_cells_d
        if wc.size:
            tau = self._wall_shear_device()
            us = self._ustar()
            floor = np.finfo(s.dtype).tiny
            denom = KAPPA_WALL * rho * us * self.yb_d
            prod = tau**2 / xp.maximum(denom, floor)
            total = s.fv.Cb_d @ (prod * self.wall_area_d)
            Gk[wc] = total[wc] / xp.maximum(self.wall_area_cell_d[wc], floor)
        eps_w = CMU ** 0.75 * xp.maximum(k[wc], 0.0) ** 1.5 / (KAPPA_WALL * self.wall_distance_d)
        knew, rk = self._transport(k, self.kb_in, 1.0 / self.sk, Gk, rho * eps / k, self.relax_k, floor=self.k_min, name="k")
        # epsilon sources with the freshly updated k: lagging k here lets the two
        # equations drift apart (eps/k blows up and k collapses to zero)
        Sc, Sp = self._eps_sources(S, knew, eps, Gk * (knew / k))
        enew, re = self._transport(eps, self.sb_in, 1.0 / self.se, Sc, Sp, self.relax_w,
                                   fixed_cells=self.wall_cells_d, fixed_vals=eps_w, floor=self.w_min, name="w")
        self.k, self.w = knew, enew
        self._update_mut(gU)
        return {"k": rk, "epsilon": re}


class KEpsilonStandard(KEpsilonRealizable):
    """Launder & Spalding (1974) standard k-epsilon (C_mu = 0.09, C1 = 1.44, C2 = 1.92)."""

    name = "k-epsilon-standard"
    sk, se, C1s, C2s = 1.0, 1.3, 1.44, 1.92

    def _cmu(self, gU):
        S, _, _, _, _ = strain_invariants(self.xp, gU, self.s.dim)
        return CMU, S

    def _eps_sources(self, S, k, eps, Gk):
        xp, rho = self.xp, self.rho
        prod_eps = self.C1s * (eps / xp.maximum(k, 1e-12)) * Gk
        Sp_eps = 2.0 * rho * self.C2s * (eps / xp.maximum(k, 1e-12))
        Sc_dest = rho * self.C2s * (eps ** 2) / xp.maximum(k, 1e-12)
        return prod_eps + Sc_dest, Sp_eps


def make_turbulence(solver) -> Optional[TurbulenceModel]:
    name = solver.case.turbulence
    if name in (None, "", "laminar"):
        return None
    if name == "k-omega-sst":
        return KOmegaSST(solver)
    if name == "k-epsilon-realizable":
        return KEpsilonRealizable(solver)
    if name == "k-epsilon-standard":
        return KEpsilonStandard(solver)
    raise ValueError(f"unknown turbulence model {name!r}; choose from {MODELS}")
