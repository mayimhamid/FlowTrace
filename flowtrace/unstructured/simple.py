r"""Pressure-based segregated solver (SIMPLE / SIMPLEC) on unstructured meshes.

This mirrors the algorithm behind Fluent's default pressure-based solver:

* collocated cell-centred finite volumes on arbitrary polyhedra;
* momentum: upwind, deferred-correction bounded second-order upwind, or
  skew-symmetric convection, central diffusion with explicit non-orthogonal
  correction (over-relaxed decomposition), least-squares gradients;
* Rhie-Chow face-flux interpolation to suppress pressure checkerboarding;
* pressure-correction equation solved with AMG-preconditioned CG;
* SIMPLE (alpha_p = 0.3) or SIMPLEC (consistent d-coefficients, alpha_p = 1);
* steady or transient (implicit Euler / BDF2 with inner iterations per time
  step and Choi's Rhie-Chow correction);
* energy with P-1 radiation, passive species transport;
* Fluent-style scaled residuals (momentum/energy: sum|r| / sum|a_P phi_P|,
  continuity: sum|mass imbalance| / max of the first five iterations).

All per-iteration work is vectorised over faces/cells and runs on NumPy or
CuPy; sparse assembly writes directly into a fixed CSR pattern.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict

import numpy as np

from ..hardware.backend import get_backend
from ..linalg.amg import make_amg
from ..linalg.krylov import bicgstab, pcg
from ..mesh.unstructured import UnstructuredMesh
from .case import FLUENT_TO_KIND, BoundarySpec, UnstructuredCase
from .fvmesh import FVMesh

WALL, INLET, OUTLET, PRESSURE, SYMMETRY = range(5)


def rotation_velocity(x: np.ndarray, rot: dict, dim: int) -> np.ndarray:
    """omega * axis x (x - origin) for points x (n, dim); rot = {"omega", "origin", "axis"}."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, dim))
    w = float(rot.get("omega", 0.0))
    origin = np.asarray((list(rot.get("origin", [0.0, 0.0, 0.0])) + [0.0, 0.0, 0.0])[:3], dtype=float)
    ax = np.asarray(rot.get("axis", [0.0, 0.0, 1.0]), dtype=float)
    ax = ax / np.linalg.norm(ax)
    r = np.zeros((x.shape[0], 3))
    r[:, :dim] = x - origin[:dim]
    return (w * np.cross(ax[None, :], r))[:, :dim]
def auto_relax_p(non_orthogonality_deg: float) -> float:
    """Initial SIMPLEC pressure under-relaxation (the stability guard lowers it if needed).

    SIMPLEC needs none on orthogonal meshes. On non-orthogonal ones the skewed part of the
    p' Laplacian is neglected and accumulates without it (measured: warped 96^2/128^2 meshes
    diverged at 1.0 and converge at 0.8, which costs 2-6 % iterations).

    The worst angle alone cannot set a safe value: warped triangles (43 deg) converge at 0.8,
    uniformly sheared quads (45 deg) diverge at 0.5 and converge at 0.33, because their
    skew does not alternate in sign from face to face.
    """
    return 1.0 if non_orthogonality_deg <= 5.0 else 0.8


class NonOrthogonalPressureCorrector:
    """
    Deferred non-orthogonal corrector loop for collocated SIMPLE/SIMPLEC.
    Reuses the assembled orthogonal pressure-Poisson matrix and AMG hierarchy
    across inner non-orthogonal sweeps.
    """

    def __init__(
        self,
        owner: np.ndarray,
        neighbor: np.ndarray,
        Sf: np.ndarray,
        d_vec: np.ndarray,
        weights: np.ndarray,
        n_cells: int,
        grad_operator,
        n_non_ortho_correctors: int = 2,
    ):
        self.owner = owner
        self.neighbor = neighbor
        self.n_cells = n_cells
        self.weights = weights
        self.grad_operator = grad_operator
        self.n_non_ortho = n_non_ortho_correctors

        # Over-relaxed face decomposition: Sf = alpha_f * d + T_f
        Sf_dot_d = np.einsum("ij,ij->i", Sf, d_vec)
        Sf_sq = np.einsum("ij,ij->i", Sf, Sf)
        self.alpha_f = Sf_sq / np.maximum(Sf_dot_d, 1e-30)
        self.T_f = Sf - self.alpha_f[:, None] * d_vec

    def solve_pressure_correction(
        self,
        F_star: np.ndarray,
        rho_d_f: np.ndarray,
        amg_solve,
        p_prime_init = None,
    ):
        """
        Solve the orthogonal system and apply the configured correction sweeps.
        """
        own = self.owner
        nei = self.neighbor
        w = self.weights

        g_orth = rho_d_f * self.alpha_f
        rhs_base = np.zeros(self.n_cells, dtype=F_star.dtype)
        np.add.at(rhs_base, own, -F_star)
        np.add.at(rhs_base, nei, F_star)

        p_prime = (
            np.zeros(self.n_cells, dtype=F_star.dtype)
            if p_prime_init is None
            else p_prime_init.copy()
        )
        non_ortho_flux = np.zeros_like(F_star)
        for sweep in range(self.n_non_ortho + 1):
            rhs = rhs_base.copy()
            if sweep:
                grad = self.grad_operator(p_prime)
                grad_f = w[:, None] * grad[own] + (1.0 - w[:, None]) * grad[nei]
                non_ortho_flux = rho_d_f * np.einsum("ij,ij->i", grad_f, self.T_f)
                np.add.at(rhs, own, non_ortho_flux)
                np.add.at(rhs, nei, -non_ortho_flux)
            p_prime = amg_solve(rhs, p_prime)

        grad_p_prime = self.grad_operator(p_prime)
        orth_flux = g_orth * (p_prime[nei] - p_prime[own])

        F_corr = F_star - orth_flux - non_ortho_flux
        return p_prime, grad_p_prime, F_corr


_KIND_CODE = {"wall": WALL, "inlet": INLET, "outlet": OUTLET, "pressure": PRESSURE, "symmetry": SYMMETRY}
T_ADIABATIC, T_FIXED, T_FLUX = range(3)


class SimpleSolver:
    def __init__(self, mesh: UnstructuredMesh, case: UnstructuredCase, backend=None):
        self.case = case
        self.mesh = mesh
        self.be = backend if backend is not None else get_backend(case.backend)
        self.xp = xp = self.be.xp
        self.dtype = np.dtype(np.float32 if case.precision == "fp32" else np.float64)
        t0 = time.perf_counter()
        if case.periodic and not mesh.periodic:  # zone pairs given on the case: pair them on (a copy of) the mesh
            from ..mesh.unstructured import _shallow, make_periodic

            mesh = _shallow(mesh)
            for za, zb in case.periodic:
                mesh = make_periodic(mesh, za, zb)
            self.mesh = mesh
        self.fv = fv = FVMesh(mesh, self.be, self.dtype)
        self.dim = dim = mesh.dim
        self.nc = mesh.n_cells
        self.energy = case.energy
        self.timers: Dict[str, float] = defaultdict(float)
        self.iteration = 0
        self.step_count = 0
        self.converged = False
        self._res_max5: Dict[str, float] = {}
        self.history = []
        self._bridge = None  # concurrent scalar lanes (see bridge.py); None = serial
        self._flow_threads = 0  # CPU threads of the flow's native kernels (0: library default)
        self._tuner = None
        self._lane_notes = []
        self._amg = None
        self._amg_age = 10 ** 9
        self.amg_rebuild = 10
        self.last_linear = {}
        self._gp = self._gp_eff = self._gU = self._gT = None  # lagged gradients for boundary reconstruction
        self.external_energy = False  # True when a conjugate (fluid+solid) solver owns the energy equation
        self.relax_p = float(case.relax_p) if case.relax_p is not None else auto_relax_p(fv.non_orthogonality_deg)
        self._build_boundaries()
        U0 = np.zeros((self.nc, dim))
        iv = list(case.init_velocity) + [0.0] * 3
        U0[:] = iv[:dim]
        self.U = xp.asarray(U0, dtype=self.dtype)
        self.p = xp.zeros(self.nc, dtype=self.dtype)
        self.T = xp.full(self.nc, case.init_T, dtype=self.dtype) if self.energy else None
        ub = self._velocity_bvals(self.U)
        self.F = case.density * xp.sum(fv.face_interp_vec(self.U) * fv.S, axis=1)
        self.Fb = self._boundary_flux_fixed(ub)
        self.mom_source = None  # (nc, dim), integrated over each cell
        self._momentum_forcing = None
        self._forcing_step = None
        self.energy_source = None  # (nc,) volumetric heat source per unit volume
        from .turbulence import make_turbulence

        self.turb = make_turbulence(self)
        self.Y = {name: xp.zeros(self.nc, dtype=self.dtype) for name in case.species}
        self._build_species_boundaries()
        self.rad = None
        if case.radiation == "p1":
            from .radiation import P1Model

            self.rad = P1Model(self)
        # transient bookkeeping
        self.time = 0.0
        self.time_step = 0
        self.inner = 0
        self._old: Dict[str, tuple] = {}
        self._F_old: list = []
        if case.unsteady:
            self._store_old()
        # a user-set relaxation is left alone; transient runs restart every time step from a larger
        # residual than the previous step's converged inner iterations, which a growth test would misread
        self.guard_enabled = case.relax_p is None and not case.unsteady
        self._capture = self._energy_capture = None  # systems of the latest iteration, see diagnostics()
        self._reset_guard()
        self._configure_lanes()
        self.setup_seconds = time.perf_counter() - t0

    def initialize(self, fields: dict) -> None:
        """Initialise from cell values with Fluent field names (e.g. an interpolated
        .ip file): x-/y-/z-velocity, pressure, temperature. Face mass fluxes are
        rebuilt from the velocity so the first iteration starts consistently."""
        xp, dim = self.xp, self.dim
        U = self.be.to_numpy(self.U).astype(np.float64)
        for k, name in enumerate(("x-velocity", "y-velocity", "z-velocity")[:dim]):
            if name in fields:
                U[:, k] = np.asarray(fields[name], dtype=np.float64)
        self.U = xp.asarray(U, dtype=self.dtype)
        if "pressure" in fields:
            p = np.asarray(fields["pressure"], dtype=np.float64)
            if self.singular_p:
                p = p - p.mean()
            self.p = xp.asarray(p, dtype=self.dtype)
        if self.energy and "temperature" in fields:
            self.T = xp.asarray(np.asarray(fields["temperature"], dtype=np.float64), dtype=self.dtype)
        ub = self._velocity_bvals(self.U)
        self.F = self.case.density * xp.sum(self.fv.face_interp_vec(self.U) * self.fv.S, axis=1)
        self.Fb = self._boundary_flux_fixed(ub)
        self._gp = self._gp_eff = self._gU = self._gT = None
        if self._bridge is not None:
            self._bridge.mark_dirty()
        self._capture = None  # the fields changed: the last iteration's systems no longer describe them
        self._reset_guard()

    def set_sources(self, momentum=None, energy=None) -> None:
        """Volumetric sources: momentum (nc, dim) [force/volume], energy (nc,) [heat/volume].

        Momentum may also be a callable ``momentum(centres, time)`` returning
        force per volume as a NumPy array of shape (nc, dim). For transient
        runs it is sampled at the end of each physical time step. For steady
        runs it is sampled at time zero. Closed and periodic domains remove
        the callback's volume-weighted mean. A uniform body force is ``case.body_force``.
        """
        xp = self.xp
        mom_source = energy_source = None
        if momentum is not None and not callable(momentum):
            m = np.asarray(momentum, dtype=np.float64).reshape(self.nc, self.dim)
            mom_source = xp.asarray(m * self.be.to_numpy(self.fv.vol)[:, None], dtype=self.dtype)
        if energy is not None:
            if not self.energy:
                raise ValueError("energy source given but the energy equation is off")
            e = np.asarray(energy, dtype=np.float64).reshape(self.nc)
            energy_source = xp.asarray(e * self.be.to_numpy(self.fv.vol) / self.case.cp, dtype=self.dtype)
        self.mom_source, self.energy_source = mom_source, energy_source
        self._momentum_forcing = momentum if callable(momentum) else None
        self._forcing_step = None

    def _sample_momentum_forcing(self) -> None:
        if self._momentum_forcing is None or self._forcing_step == self.time_step:
            return
        time = self.time + self.case.dt if self.case.unsteady else 0.0
        centres = self.cell_centres().view()
        centres.flags.writeable = False
        force = np.asarray(self._momentum_forcing(centres, time), dtype=np.float64)
        if force.shape != (self.nc, self.dim) or not np.isfinite(force).all():
            raise ValueError(f"momentum forcing must return a finite array of shape {(self.nc, self.dim)}")
        xp, vol = self.xp, self.fv.vol
        integrated = xp.asarray(force, dtype=self.dtype) * vol[:, None]
        if self.singular_p and not (self.bkind_h == INLET).any():
            mean = xp.sum(integrated, axis=0) / xp.sum(vol)
            integrated = integrated - vol[:, None] * mean
        self.mom_source = integrated
        self._forcing_step = self.time_step

    # ------------------------------------------------------------------
    def _build_boundaries(self):
        fv, case, xp, dim = self.fv, self.case, self.xp, self.dim
        n = fv.n_bf
        kind = np.zeros(n, dtype=np.int64)
        ub = np.zeros((n, dim))
        pb = np.zeros(n)
        tk = np.zeros(n, dtype=np.int64)
        tv = np.zeros(n)
        nb = self.be.to_numpy(fv.nb)
        xfb = self.be.to_numpy(fv.xfb).astype(np.float64)
        Pb_h = self.be.to_numpy(fv.Pb)
        self._build_mrf()
        self.zone_specs = {}
        for z in self.mesh.boundary_zones():
            spec = case.boundaries.get(z.name)
            if spec is None:
                spec = BoundarySpec(kind=FLUENT_TO_KIND.get(z.type, "wall"))
                if spec.kind == "inlet":
                    spec.normal_speed = 1.0
            self.zone_specs[z.name] = spec
            m = fv.bzone == z.id
            if spec.kind == "periodic" or z.type in ("periodic", "periodic-shadow"):
                if m.any():
                    raise ValueError(f"zone {z.name!r} is periodic but not paired (case.periodic / make_periodic)")
                continue
            kind[m] = _KIND_CODE[spec.kind]
            if spec.kind in ("wall", "inlet"):
                if spec.velocity is not None:
                    v = (list(spec.velocity) + [0.0, 0.0, 0.0])[:dim]
                    ub[m] = v
                elif spec.normal_speed is not None:
                    ub[m] = -spec.normal_speed * nb[m]  # area vectors point outwards
                elif spec.kind == "wall" and spec.rotation is not None:
                    ub[m] = rotation_velocity(xfb[m], spec.rotation, dim)
                elif spec.kind == "wall" and self.mrf_zones:
                    # as in Fluent, a wall bounding a moving reference frame moves with it
                    ub[m] = self._frame_at(xfb[m], Pb_h[m])
            pb[m] = spec.pressure
            if spec.kind == "wall":
                tk[m] = {"adiabatic": T_ADIABATIC, "temperature": T_FIXED, "flux": T_FLUX}[spec.thermal]
            elif spec.kind in ("inlet", "pressure"):
                tk[m] = T_FIXED
            tv[m] = spec.value
        self.bkind_h = kind
        d = lambda a: xp.asarray(a, dtype=self.dtype)  # noqa: E731
        self.bkind = xp.asarray(kind)
        self.ub_fixed = d(ub)
        self.pb_fixed = d(pb)
        self.tkind = xp.asarray(tk)
        self.tval = d(tv)
        self.m_wall = d(kind == WALL)
        self.m_inlet = d(kind == INLET)
        self.m_pres = d((kind == OUTLET) | (kind == PRESSURE))
        self.m_sym = d(kind == SYMMETRY)
        self.m_dirvel = d((kind == WALL) | (kind == INLET) | (kind == SYMMETRY))
        self.singular_p = not bool(((kind == OUTLET) | (kind == PRESSURE)).any())

    def _build_mrf(self):
        """Moving reference frames (absolute-velocity formulation, as Fluent's MRF).

        In an MRF cell zone rotating with Omega about ``origin``: convection uses the
        relative mass flux F - rho (Omega x r)_f . S_f, continuity the absolute flux,
        and the momentum equation gets the Coriolis-type source -rho Omega x U.
        """
        c, fv, xp, dim = self.case, self.fv, self.xp, self.dim
        self.mrf_zones = {}
        self._frame_zone = np.full(self.nc, -1, dtype=np.int64)
        self.mrf_flux = self.mrf_flux_b = None
        self.mrf_omega_cells = None
        if not c.mrf:
            return
        rots = []
        for i, (zname, rot) in enumerate(c.mrf.items()):
            z = self.mesh.zone_by_name(zname)
            if z is None or z.kind != "cell":
                raise ValueError(f"MRF zone {zname!r} is not a cell zone of this mesh")
            self._frame_zone[self.mesh.cell_zone == z.id] = i
            self.mrf_zones[zname] = rot
            rots.append(rot)
        self._rots = rots
        h = self.be.to_numpy
        P, N, Pb = h(fv.P), h(fv.N), h(fv.Pb)
        xf, xfb = h(fv.xf).astype(np.float64), h(fv.xfb).astype(np.float64)
        S, Sb = h(fv.S).astype(np.float64), h(fv.Sb).astype(np.float64)
        # frame of a face: the owner's, else the neighbour's (an MRF interface should be a
        # surface of revolution, where the frame velocity is tangential either way)
        zf = np.where(self._frame_zone[P] >= 0, self._frame_zone[P], self._frame_zone[N])
        Ff = np.zeros(P.size)
        for i, rot in enumerate(rots):
            m = zf == i
            Ff[m] = np.einsum("fd,fd->f", rotation_velocity(xf[m], rot, dim), S[m])
        zb = self._frame_zone[Pb]
        Fbb = np.zeros(Pb.size)
        for i, rot in enumerate(rots):
            m = zb == i
            Fbb[m] = np.einsum("fd,fd->f", rotation_velocity(xfb[m], rot, dim), Sb[m])
        rho = c.density
        self.mrf_flux = xp.asarray(rho * Ff, dtype=self.dtype)
        self.mrf_flux_b = xp.asarray(rho * Fbb, dtype=self.dtype)
        om = np.zeros((self.nc, 3))
        for i, rot in enumerate(rots):
            ax = np.asarray(rot.get("axis", [0.0, 0.0, 1.0]), dtype=float)
            om[self._frame_zone == i] = float(rot["omega"]) * ax / np.linalg.norm(ax)
        self.mrf_omega_cells = xp.asarray(om, dtype=self.dtype)

    def _frame_at(self, x, cells):
        """Frame velocity at points x whose adjacent cells are ``cells`` (0 outside MRF zones)."""
        out = np.zeros((x.shape[0], self.dim))
        z = self._frame_zone[cells]
        for i, rot in enumerate(getattr(self, "_rots", [])):
            m = z == i
            out[m] = rotation_velocity(x[m], rot, self.dim)
        return out

    def _conv(self):
        """Mass fluxes that convect momentum/scalars (relative to MRF frames)."""
        if self.mrf_flux is None:
            return self.F, self.Fb
        return self.F - self.mrf_flux, self.Fb - self.mrf_flux_b

    def _velocity_bvals(self, U, gU=None):
        """Boundary face velocities; zero-gradient values are reconstructed with the
        (lagged) cell gradient along the face, which keeps skewed boundary cells
        second-order accurate."""
        fv, xp = self.fv, self.xp
        UP = U[fv.Pb]
        if gU is not None:
            UP = UP + xp.stack([xp.sum(g[fv.Pb] * fv.db_t, axis=1) for g in gU], axis=1)
        un = xp.sum(UP * fv.nb, axis=1)
        sym = UP - un[:, None] * fv.nb
        fixed = (self.m_wall + self.m_inlet)[:, None]
        return fixed * self.ub_fixed + self.m_sym[:, None] * sym + self.m_pres[:, None] * UP

    def _pressure_bvals(self, p, gp=None, p_fixed=None):
        """Wall/inlet/symmetry face pressure: linear extrapolation from the owner
        cell with the (lagged) gradient; fixed value on pressure boundaries."""
        fv = self.fv
        pP = p[fv.Pb]
        if gp is not None:
            pP = pP + self.xp.sum(gp[fv.Pb] * fv.db, axis=1)
        if p_fixed is None:
            p_fixed = self.pb_fixed
        return self.m_pres * p_fixed + (1.0 - self.m_pres) * pP

    def _boundary_flux_fixed(self, ub):
        fv = self.fv
        flux = self.case.density * self.xp.sum(ub * fv.Sb, axis=1)
        return flux * (self.m_inlet + self.m_pres)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_gamma(gamma):
        """Diffusivity as (interior-face, boundary-face) values; scalars broadcast."""
        if isinstance(gamma, tuple):
            return gamma
        return gamma, gamma

    def _coefficients(self, gamma, dirichlet, conv=None, *, skew_symmetric=False, upwind_diffusion=0.0):
        """Implicit convection and diffusion: (diag, aPN, aNP, Db).

        ``gamma`` is a scalar or a tuple (interior-face values, boundary-face values).
        ``conv``: the convecting (F, Fb); default the solver's current fluxes.
        """
        fv, xp = self.fv, self.xp
        F, Fb = conv if conv is not None else self._conv()
        gf, gb = self._split_gamma(gamma)
        D = gf * fv.alpha
        Db = gb * fv.alpha_b * dirichlet
        if skew_symmetric:
            # Central face transport minus half the cell mass imbalance times
            # phi. The central diagonal cancels exactly on interior faces.
            # Physical boundaries retain their upwind flux with the same split.
            half_flux = 0.5 * F
            D = D + upwind_diffusion
            diag = fv.scatter(D, D, Db + 0.5 * xp.abs(Fb))
            return diag, half_flux - D, -half_flux - D, Db
        aPN = -(D + xp.maximum(-F, 0.0))  # coefficient of phi_N in P's row
        aNP = -(D + xp.maximum(F, 0.0))  # coefficient of phi_P in N's row
        diag = fv.scatter(D + xp.maximum(F, 0.0), D + xp.maximum(-F, 0.0), Db + xp.maximum(Fb, 0.0))
        return diag, aPN, aNP, Db

    def _skew_upwind_diffusion(self, U, gU, F):
        """Bounded upwind dissipation where reconstructed gradients miss the velocity jump."""
        fv, xp = self.fv, self.xp
        jump = U[fv.N] - U[fv.P]
        reconstructed = xp.stack(
            [xp.sum(fv.face_interp_vec(grad) * fv.d, axis=1) for grad in gU], axis=1
        )
        jump2 = xp.sum(jump * jump, axis=1)
        alignment = xp.sum(reconstructed * jump, axis=1) / xp.where(jump2 > 0.0, jump2, 1.0)
        fraction = xp.where(jump2 > 0.0, xp.clip(1.0 - 2.0 * alignment, 0.0, 1.0), 0.0)
        return 0.5 * xp.abs(F) * fraction

    def _rhs(self, gamma, phi, grad, phi_b, Db, extra=None, scheme=None, conv=None):
        """Boundary sources + explicit non-orthogonal and second-order corrections."""
        fv, xp = self.fv, self.xp
        F, Fb = conv if conv is not None else self._conv()
        gface, _ = self._split_gamma(gamma)
        rhs = fv.Cb_d @ ((Db + xp.maximum(-Fb, 0.0)) * phi_b)
        # non-orthogonal diffusion flux into P: gamma (grad phi)_f . T_f, outward from P
        gf = fv.face_interp_vec(grad)
        rhs = rhs + fv.Cdiff_d @ (gface * xp.sum(gf * fv.Tvec, axis=1))
        if (scheme or self.case.scheme) == "second-order":
            P, N = fv.P, fv.N
            phP, phN = phi[P], phi[N]
            hoP = phP + xp.sum(grad[P] * fv.dP_f, axis=1)
            hoN = phN + xp.sum(grad[N] * fv.dN_f, axis=1)
            up = F >= 0
            ho = xp.clip(xp.where(up, hoP, hoN), xp.minimum(phP, phN), xp.maximum(phP, phN))
            rhs = rhs - fv.Cdiff_d @ (F * (ho - xp.where(up, phP, phN)))
        if extra is not None:
            rhs = rhs + extra
        return rhs

    def _residual(self, A, diag, x, b):
        xp = self.xp
        r = float(xp.sum(xp.abs(b - A @ x)))
        s = float(xp.sum(xp.abs(diag * x)))
        return r / s if s > 0 else r

    def _relax_solve(self, diag, aPN, aNP, rhs, phi, alpha, tol=0.1, maxiter=40):
        dr = diag / alpha
        Ar = self.fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - alpha) / alpha) * diag * phi
        x, _ = self._solve(Ar, b, phi, dr, tol=tol, maxiter=maxiter)
        return x

    # ---------------- transient ----------------
    def _time_coeffs(self):
        """(c0, c1, c2) of d(rho phi)/dt ~ rho V/dt (c0 phi - c1 phi^n - c2 phi^(n-1)), or None if steady."""
        if not self.case.unsteady:
            return None
        if self.case.time_scheme == "bdf2" and self.time_step >= 1:
            return 1.5, 2.0, -0.5
        return 1.0, 1.0, 0.0

    def _time_snapshot(self, names):
        """(time coefficients, {name: (old, older)}): what _time_terms reads, captured for a lane."""
        tc = self._time_coeffs()
        return tc, ({n: self._old[n] for n in names if n in self._old} if tc is not None else {})

    def _time_terms(self, name, phi, diag, rhs, snap=None):
        tc, olds = snap if snap is not None else (self._time_coeffs(), self._old)
        if tc is None or name is None:
            return diag, rhs
        c0, c1, c2 = tc
        rv = self.case.density * self.fv.vol / self.case.dt
        old, older = olds.get(name, (phi, phi))
        return diag + c0 * rv, rhs + rv * (c1 * old + (c2 * older if c2 else 0.0))

    def _store_old(self):
        self._sync_lanes()
        cur = {f"U{k}": self.U[:, k].copy() for k in range(self.dim)}
        if self.energy and self.T is not None:
            cur["T"] = self.T.copy()
        if self.turb is not None:
            cur["k"] = self.turb.k.copy()
            cur["w"] = self.turb.w.copy()
        for name, Y in getattr(self, "Y", {}).items():
            cur[f"Y:{name}"] = Y.copy()
        for name, v in cur.items():
            prev = self._old.get(name, (v, v))[0]
            self._old[name] = (v, prev)
        Fprev = self._F_old[0] if self._F_old else self.F.copy()
        self._F_old = [self.F.copy(), Fprev]

    def _advance_time(self):
        self._sync_lanes()
        self.time += self.case.dt
        self.time_step += 1
        self.inner = 0
        self._store_old()

    @property
    def t(self):
        return self.time if self.case.unsteady else self.iteration

    # ---------------- concurrent scalar lanes (bridge.py) ----------------
    # T and Y are properties: any read from outside a lane first collects the lanes in flight,
    # and any write tells cross-device lanes to refresh their resident copy.
    @property
    def T(self):
        self._sync_lanes()
        return self.__dict__.get("_T")

    @T.setter
    def T(self, value):
        self._sync_lanes()
        self._T = value
        b = self.__dict__.get("_bridge")
        if b is not None:
            b.mark_dirty("energy")

    @property
    def Y(self) -> dict:
        self._sync_lanes()
        b = self.__dict__.get("_bridge")
        if b is not None:
            b.mark_dirty("species")  # the caller may write into the dict
        return self.__dict__.get("_Y", {})

    @Y.setter
    def Y(self, value):
        self._sync_lanes()
        self._Y = value
        b = self.__dict__.get("_bridge")
        if b is not None:
            b.mark_dirty("species")

    def _sync_lanes(self) -> None:
        b = self.__dict__.get("_bridge")
        if b is not None and b.busy and not b.in_lane_thread():
            b.await_all()

    def _set_bridge(self, bridge) -> None:
        old = self.__dict__.get("_bridge")
        if old is not None:
            old.close()  # collects its lanes: the solver state is complete again
        self._bridge = bridge

    def _configure_lanes(self) -> None:
        """``case.lanes``: 'off' (serial), 'auto' (measured placement, see LaneTuner), or a dict
        {lane: 'cpu' | 'gpu'} for a fixed assignment."""
        from .bridge import Bridge, LaneTuner, eligible_lanes

        mode = getattr(self.case, "lanes", "off")
        if mode in (None, "off", False):
            return
        if self.be.is_gpu and getattr(self.xp, "__fake__", False):
            return  # a test double of CuPy has no streams or events
        lanes, why = eligible_lanes(self)
        self._lane_notes = [f"{k}: stays serial ({v})" for k, v in why.items()]
        if not lanes:
            return
        if isinstance(mode, dict):
            assignment = {k: v for k, v in mode.items() if k in lanes}
            if assignment:
                threads = mode.get("threads") or {}
                self._set_bridge(Bridge(self, assignment, threads))
                self._flow_threads = int(threads.get("flow", 0))
            return
        if mode == "auto":
            if self.case.max_iter < LaneTuner.MIN_ITERATIONS:
                self._lane_notes.append(f"serial: max_iter < {LaneTuner.MIN_ITERATIONS}, too short to measure lanes")
                return
            self._tuner = LaneTuner(self, lanes)
            return
        raise ValueError(f"case.lanes must be 'off', 'auto' or a dict, not {mode!r}")

    def lanes_report(self) -> dict:
        """Where the scalar equations ran and why (for describe() and the dashboard)."""
        out = {"mode": "serial", "notes": list(self._lane_notes)}
        if self._tuner is not None:
            out["tuning"] = self._tuner.decision or {"phase": self._tuner.phase}
        if self._bridge is not None:
            out["mode"] = "concurrent"
            out.update(self._bridge.report())
        return out

    @property
    def dt(self):
        return self.case.dt if self.case.unsteady else None

    # ---------------- species ----------------
    def _build_species_boundaries(self):
        """Per-species boundary values and Dirichlet masks (inlets, openings, walls with a value)."""
        xp, fv = self.xp, self.fv
        self.Yb, self.Ydir = {}, {}
        for name in self.case.species:
            val = np.zeros(fv.n_bf)
            fixed = np.zeros(fv.n_bf)
            for z in self.mesh.boundary_zones():
                m = fv.bzone == z.id
                spec = self.zone_specs.get(z.name)
                if spec is None or not m.any():
                    continue
                v = (spec.species or {}).get(name)
                if spec.kind in ("inlet", "pressure"):
                    val[m] = v or 0.0
                    fixed[m] = 1.0 if spec.kind == "inlet" else 0.0
                elif spec.kind == "outlet":
                    val[m] = v or 0.0  # backflow value
                elif spec.kind == "wall" and v is not None:
                    val[m] = v
                    fixed[m] = 1.0
            self.Yb[name] = xp.asarray(val, dtype=self.dtype)
            self.Ydir[name] = xp.asarray(fixed, dtype=self.dtype)

    def _species_inputs(self) -> dict:
        """Everything the species equations read that changes between iterations (snapshots)."""
        c, fv = self.case, self.fv
        rho = c.density
        if self.turb is not None:
            mut = self.turb.mut
            gam = (rho * c.species_diffusivity + fv.face_interp(mut) / c.schmidt_turbulent,
                   rho * c.species_diffusivity + mut[fv.Pb] / c.schmidt_turbulent * (1.0 - self.m_wall))
        else:
            gam = rho * c.species_diffusivity
        F, Fb = self._conv()
        return {"F": F, "Fb": Fb, "Fb_abs": self.Fb, "gam": gam,
                "time": self._time_snapshot([f"Y:{n}" for n in self._Y])}

    def _species_kernel(self, Ys: dict, inp: dict):
        """Solve every species from ``Ys`` with the snapshot ``inp``; returns (new dict, residuals).
        Reads only static solver data besides its arguments (it may run in a lane thread)."""
        c, fv, xp = self.case, self.fv, self.xp
        res, out = {}, {}
        gam, conv = inp["gam"], (inp["F"], inp["Fb"])
        inflow = (self.m_inlet > 0) | ((self.m_pres > 0) & (inp["Fb_abs"] < 0))
        for name, Y in Ys.items():
            fixed = (self.Ydir[name] > 0) | inflow
            Yb = xp.where(fixed, self.Yb[name], Y[fv.Pb])
            gY = fv.gradient(Y, Yb)
            diag, aPN, aNP, Db = self._coefficients(gam, self.Ydir[name], conv)
            rhs = self._rhs(gam, Y, gY, Yb, Db, scheme=c.scalar_convection, conv=conv)
            diag, rhs = self._time_terms(f"Y:{name}", Y, diag, rhs, inp.get("time"))
            A = fv.matrix(diag, aPN, aNP)
            res[name] = self._residual(A, diag, Y, rhs)
            out[name] = self._relax_solve(diag, aPN, aNP, rhs, Y, 1.0, tol=1e-3, maxiter=60)
        return out, res

    def _species(self) -> dict:
        Y, res = self._species_kernel(self._Y, self._species_inputs())
        self._Y = Y
        return res

    def _wall_temperature(self, T=None):
        """Boundary-face temperatures: fixed values, else the adjacent cell (adiabatic/flux/outflow)."""
        TP = (self.T if T is None else T)[self.fv.Pb]
        return self.xp.where(self.tkind == T_FIXED, self.tval, TP)

    def _solve(self, A, b, x0, diag, tol=0.1, maxiter=30):
        inv = 1.0 / diag
        if self.fv.nat is not None:  # the whole Krylov loop in C++ (same algorithm and stopping rules)
            return self.fv.nat.bicgstab_jacobi(A, b, x0, inv, tol, maxiter, relative_to_initial=True, lib=self.fv._lib)
        if self.fv.cu is not None:
            return self.fv.cu.bicgstab_jacobi(A, b, x0, inv, tol, maxiter)
        x, info = bicgstab(lambda v: A @ v, b, self.xp, M=lambda v: inv * v, x0=x0, tol=tol, maxiter=maxiter,
                           relative_to_initial=True)
        return x, info

    # ------------------------------------------------------------------
    def step(self) -> float:
        if self.finished:
            return 0.0
        if self._flow_threads > 0 and not self.be.is_gpu:
            from .. import native

            native.set_threads(self._flow_threads)  # this thread's OpenMP team, restored after the step
            try:
                return self._step()
            finally:
                native.set_threads(native.max_threads())
        return self._step()

    def _step(self) -> float:
        # the previous iteration's systems are released first, so keeping them for
        # diagnostics() never adds to the peak memory of an iteration
        self._capture = self._energy_capture = None
        c, fv, xp = self.case, self.fv, self.xp
        rho, mu = c.density, c.viscosity
        dim = self.dim
        tm = self.timers
        t0 = time.perf_counter()
        U, p = self.U, self.p
        ub = self._velocity_bvals(U, self._gU)
        pb = self._pressure_bvals(p, self._gp)
        gp_ls = fv.gradient(p, pb)
        gU = [fv.gradient(U[:, k], ub[:, k]) for k in range(dim)]
        self._gU = gU
        use_gg = c.pressure_gradient == "green-gauss"
        self._gp = fv.gradient_gg(p, pb, gp_ls) if use_gg else gp_ls
        gp = self._gp
        p_open = self.pb_fixed
        if self.turb is not None:
            k = self.turb.k
            kb, _ = self.turb._bvals(k, self.turb.kb_in)
            scale = (2.0 / 3.0) * rho
            p = p + scale * k
            p_open = p_open + scale * kb
            pb = self._pressure_bvals(p, self._gp_eff, p_open)
            gp = fv.gradient(p, pb)
            if use_gg:
                gp = fv.gradient_gg(p, pb, gp)
            self._gp_eff = gp
        t1 = time.perf_counter()
        tm["gradients"] += t1 - t0

        # ---------------- momentum ----------------
        self._sample_momentum_forcing()
        src = -gp * fv.vol[:, None]
        if self.energy and any(c.buoyancy[:dim]):
            bv = xp.asarray(np.asarray(c.buoyancy[:dim]), dtype=self.dtype)
            src = src + (fv.vol * (self.T - c.t_ref))[:, None] * bv[None, :]
        drive = np.asarray(c.body_force[:dim], dtype=np.float64) - np.asarray(
            (list(c.periodic_pressure_gradient) + [0.0] * 3)[:dim], dtype=np.float64)  # mean -dp/dx drives periodic flow
        if np.any(drive):
            bf = xp.asarray(drive, dtype=self.dtype)
            src = src + fv.vol[:, None] * bf[None, :]
        if self.mom_source is not None:
            src = src + self.mom_source
        if self.mrf_omega_cells is not None:  # -rho Omega x U (absolute-velocity MRF)
            om = self.mrf_omega_cells
            if dim == 2:
                cr = xp.stack([-om[:, 2] * U[:, 1], om[:, 2] * U[:, 0]], axis=1)
            else:
                cr = xp.stack([om[:, 1] * U[:, 2] - om[:, 2] * U[:, 1], om[:, 2] * U[:, 0] - om[:, 0] * U[:, 2],
                               om[:, 0] * U[:, 1] - om[:, 1] * U[:, 0]], axis=1)
            src = src - rho * fv.vol[:, None] * cr
        gm = mu
        if self.turb is not None:
            gm = self.turb.momentum_viscosity(U)  # mu + mu_t on faces, wall-function mu_w on walls
            src = src + self.turb.transpose_stress(gU, gm)
        res = {}
        Ustar = xp.empty_like(U)
        au = self.relax_u
        conv = self._conv()
        skew = c.scheme == "skew-symmetric"
        upwind_diffusion = self._skew_upwind_diffusion(U, gU, conv[0]) if skew else 0.0
        diag, aPN, aNP, Db = self._coefficients(
            gm, self.m_dirvel, conv, skew_symmetric=skew, upwind_diffusion=upwind_diffusion
        )
        rhs_all = [self._rhs(gm, U[:, k], gU[k], ub[:, k], Db, src[:, k], conv=conv) for k in range(dim)]
        tcoef = self._time_coeffs()
        if tcoef is not None:
            rv = rho * fv.vol / c.dt
            diag = diag + tcoef[0] * rv
            for k in range(dim):
                old, older = self._old[f"U{k}"]
                rhs_all[k] = rhs_all[k] + rv * (tcoef[1] * old + tcoef[2] * older)
        A = fv.matrix(diag, aPN, aNP)
        # Fluent-style scaled momentum residuals; the denominators are floored so
        # that a velocity component that is identically ~0 (2-D flow in a 3-D
        # mesh) does not divide round-off by round-off.
        num = [float(xp.sum(xp.abs(rhs_all[k] - A @ U[:, k]))) for k in range(dim)]
        den = [float(xp.sum(xp.abs(diag * U[:, k]))) for k in range(dim)]
        floor = 1e-3 * max(max(den), 1e-300)
        for k, name in enumerate(("x-velocity", "y-velocity", "z-velocity")[:dim]):
            res[name] = num[k] / max(den[k], floor)
        diag_r = diag / au
        Ar = fv.matrix(diag_r, aPN, aNP)
        its, brs, infos = [], [], []
        for k in range(dim):
            b = rhs_all[k] + ((1.0 - au) / au) * diag * U[:, k]
            x, info = self._solve(Ar, b, U[:, k], diag_r)
            Ustar[:, k] = x
            its.append(info.iterations)
            brs.append(b)
            infos.append(info)
        t2 = time.perf_counter()
        tm["momentum"] += t2 - t1

        # ---------------- Rhie-Chow fluxes ----------------
        if self.algo == "SIMPLEC":
            sum_nb = fv.scatter(-aPN, -aNP)
            denom = diag_r - sum_nb
            denom = xp.where(denom > 1e-12 * diag_r, denom, diag_r)
        else:
            denom = diag_r
        dP = fv.vol / denom
        dF = fv.face_interp(dP)
        gpf = fv.face_interp_vec(gp)
        P, N = fv.P, fv.N
        # Rhie-Chow: compact minus interpolated pressure gradient, both projected on
        # the orthogonal part alpha*d of the area vector. (Projecting the
        # interpolated gradient on S instead leaves a spurious grad(p).T mass flux
        # on non-orthogonal faces, which MMS shows as a non-converging pressure.)
        Fst = rho * (xp.sum(fv.face_interp_vec(Ustar) * fv.S, axis=1)
                     - dF * fv.alpha * (p[N] - p[P] - xp.sum(gpf * fv.d, axis=1)))
        if tcoef is not None:
            # Choi's correction: keeps the converged face flux independent of the time step
            r = rho * dF / c.dt
            Un = xp.stack([self._old[f"U{k}"][0] for k in range(dim)], axis=1)
            corr = tcoef[1] * (self._F_old[0] - rho * xp.sum(fv.face_interp_vec(Un) * fv.S, axis=1))
            if tcoef[2]:
                Unm = xp.stack([self._old[f"U{k}"][1] for k in range(dim)], axis=1)
                corr = corr + tcoef[2] * (self._F_old[1] - rho * xp.sum(fv.face_interp_vec(Unm) * fv.S, axis=1))
            Fst = Fst + r * corr
        Pb = fv.Pb
        Fb_in = rho * xp.sum(self.ub_fixed * fv.Sb, axis=1) * self.m_inlet
        Fb_pr = rho * (xp.sum(Ustar[Pb] * fv.Sb, axis=1)
                       - dP[Pb] * fv.alpha_b * (p_open - p[Pb] - xp.sum(gp[Pb] * fv.db, axis=1)))
        Fbst = Fb_in + Fb_pr * self.m_pres

        # ---------------- pressure correction ----------------
        cf = rho * dF * fv.alpha
        cb = rho * dP[Pb] * fv.alpha_b * self.m_pres
        diag_p = fv.scatter(cf, cf, cb)
        Ap = fv.matrix(diag_p, -cf, -cf)
        mass = fv.net_outflow(Fst, Fbst)
        res["continuity"] = float(xp.sum(xp.abs(mass)))
        t3 = time.perf_counter()
        tm["assembly"] += t3 - t2
        if self._amg is None or self._amg_age >= self.amg_rebuild:
            Ah = Ap.get() if self.be.is_gpu else Ap
            self._amg = make_amg(Ah, self.be, singular=self.singular_p)
            self._amg_age = 0
        self._amg_age += 1
        if self.singular_p:
            def project(v):
                return v - v.mean()
        else:
            project = None
        pressure_rhs = -mass
        pp, pinfo = pcg(lambda v: Ap @ v, pressure_rhs, xp, M=self._amg, tol=c.pressure_tol, maxiter=200, project=project)

        # Deferred non-orthogonal corrector sweeps (reusing frozen Ap and AMG preconditioner)
        n_no = c.n_non_ortho if c.n_non_ortho is not None else (2 if fv.non_orthogonality_deg > 5.0 else 0)
        F_no = 0.0
        Fb_no = 0.0
        if n_no > 0:
            for _ in range(n_no):
                ppb = (1.0 - self.m_pres) * pp[Pb]
                gpp = fv.gradient(pp, ppb)
                gpp_f = fv.face_interp_vec(gpp)
                F_no = rho * dF * xp.sum(gpp_f * fv.Tvec, axis=1)
                Tvec_b = fv.Sb - fv.alpha_b[:, None] * fv.db
                Fb_no = rho * dP[Pb] * xp.sum(gpp[Pb] * Tvec_b, axis=1) * self.m_pres
                pressure_rhs = -mass + fv.net_outflow(F_no, Fb_no)
                pp, pinfo = pcg(lambda v: Ap @ v, pressure_rhs, xp, M=self._amg, x0=pp,
                                tol=c.pressure_tol, maxiter=200, project=project)
        # Reuse the lagged tangential fluxes from the final Poisson RHS.
        t4 = time.perf_counter()
        tm["pressure"] += t4 - t3

        # ---------------- corrections ----------------
        self.F = Fst + cf * (pp[P] - pp[N]) - F_no
        self.Fb = Fbst + cb * pp[Pb] - Fb_no
        p_prev = self.p
        self.p = p_prev + self.relax_p * pp
        if self.singular_p:
            self.p = self.p - self.p.mean()
        ppb = (1.0 - self.m_pres) * pp[Pb]
        gpp = fv.gradient(pp, ppb)
        self.U = Ustar - dP[:, None] * gpp
        # references (not copies) to the systems of this iteration, for diagnostics();
        # every array here is rebound, never modified in place, by later iterations
        cap = {"U_old": U, "p_old": p_prev, "A": A, "b": rhs_all, "Ar": Ar, "br": brs, "U_star": Ustar,
               "mom_info": infos, "alpha_u": au, "Fst": Fst, "Fbst": Fbst, "mass": mass, "Ap": Ap, "pp": pp,
               "pressure_rhs": pressure_rhs, "p_info": pinfo, "F": self.F, "Fb": self.Fb, "U_new": self.U, "p_new": self.p}

        # ---------------- energy ----------------
        bridge = self._bridge
        t4e = time.perf_counter()
        tm["correct"] += t4e - t4
        if self.energy and not self.external_energy:
            if bridge is not None and bridge.has("energy"):
                bridge.cycle("energy", self._energy_inputs())  # collect iteration n-1, launch n
            else:
                res.update(self._energy())
                cap["energy"] = self._energy_capture
        t5 = time.perf_counter()
        tm["energy"] += t5 - t4e

        # ---------------- turbulence ----------------
        if self.turb is not None:
            ubc = self._velocity_bvals(self.U, gU)
            gUc = [fv.gradient(self.U[:, k], ubc[:, k]) for k in range(dim)]
            k_old, w_old = self.turb.k, self.turb.w
            res.update(self.turb.solve(gUc))
            # Keep the pressure-correction variable fixed while the separately
            # transported k changes. Public pressure remains mechanical pressure.
            self.p = self.p + (2.0 / 3.0) * rho * (k_old - self.turb.k)
            if self.singular_p:
                self.p = self.p - self.p.mean()
            cap["p_new"] = self.p
            cap["turbulence"] = {"k": (k_old, self.turb.k), self.turb.second: (w_old, self.turb.w)}
            tm["turbulence"] += time.perf_counter() - t5
        if self._Y:
            t6 = time.perf_counter()
            if bridge is not None and bridge.has("species"):
                bridge.cycle("species", self._species_inputs())
            else:
                res.update(self._species())
            tm["species"] += time.perf_counter() - t6
        if bridge is not None:
            if any(l.launches == 1 and l.busy for l in bridge.lanes.values()):
                bridge.await_all()  # first iteration of the lanes: report a complete set of residuals
            res.update(bridge.residuals)  # lane residuals: one iteration behind the flow's
            if bridge.capture is not None:
                cap["energy"] = bridge.capture

        self.iteration += 1
        self.step_count = self.iteration
        self.last_linear = {"momentum_iters": its, "pressure_iters": pinfo.iterations,
                            "pressure_residual": pinfo.residual}
        prev = getattr(self, "_last_res", {}), getattr(self, "_last_raw", {})
        self._last_raw = dict(res)  # continuity: un-normalised sum |mass imbalance|
        if self.iteration <= 5:
            self._res_max5["continuity"] = max(self._res_max5.get("continuity", 0.0), res["continuity"], 1e-300)
        # Fluent normalises continuity by its largest value in the first five
        # iterations. Flows that start without any imbalance (periodic, warm
        # starts) would be normalised by round-off, so floor the denominator at a
        # small fraction of the total face mass flux.
        flux_scale = float(xp.sum(xp.abs(self.F))) + float(xp.sum(xp.abs(self.Fb)))
        res["continuity"] /= max(self._res_max5["continuity"], 1e-3 * flux_scale)
        self._last_res = res
        cap["iteration"] = self.iteration
        rolled_back = self._stability_guard(self._last_raw["continuity"], flux_scale)
        cap["rolled_back"] = rolled_back  # the guard discarded this iteration's fields
        self._capture = cap
        if rolled_back:
            self._last_res, self._last_raw = prev  # report the restored state, not the diverged one
            return 1.0
        if c.unsteady:
            self.inner += 1
            if self.inner >= c.inner_iterations or self._below(res, c.inner_tol):
                self._advance_time()
        elif self.iteration > 5 and self._below(res, c.tol):
            self.converged = True
        if self._tuner is not None:
            self._tuner.after_step(time.perf_counter() - t0)
        if self.finished and self._bridge is not None:
            self._sync_lanes()  # the final state is complete in every field
            self._last_res = {**self._last_res, **self._bridge.residuals}
        return 1.0

    def _below(self, res: dict, tol: float) -> bool:
        """All residuals below ``tol``. With lanes in flight the scalar residuals in ``res`` are one
        iteration old: collect the lanes and test their fresh residuals too, so that convergence is
        only declared on a state that is converged in every field."""
        if not max(res.values()) < tol:  # written so that a NaN residual is never "below"
            return False
        b = self._bridge
        if b is None or not b.busy:
            return True
        b.await_all()
        res.update(b.residuals)
        self._last_res = res
        return max(res.values()) < tol

    # divergence guard: roll back and lower the pressure under-relaxation
    GUARD_GROWTH = 20.0  # continuity above GUARD_GROWTH x its running minimum = diverging
    GUARD_ROUNDOFF = 1e-6  # ... where that minimum is floored at this fraction of the total face mass flux
    GUARD_EVERY = 20  # snapshot interval (iterations), taken only while the residual is healthy
    GUARD_FACTOR = 0.6
    GUARD_MIN_RELAX = 0.1
    GUARD_FALLBACK = True  # after relax_p reaches its floor: SIMPLEC -> SIMPLE (relax_u 0.3), then lower relax_u
    GUARD_MIN_RELAX_U = 0.2  # once relax_p is at its floor the guard lowers the momentum relaxation too

    def _reset_guard(self):
        self.guard_events = []
        self.algo = self.case.algorithm  # the guard may fall back from SIMPLEC to SIMPLE
        self.relax_u = self.case.relax_u  # lowered by the guard on convection-dominated, skewed cases
        self.guard_exhausted_at = None  # iteration at which divergence was seen with relax_p at its floor
        self._guard_min = float("inf")
        self._guard_snap = (self.state_dict(), float("inf")) if self.guard_enabled else None

    def _stability_guard(self, raw_continuity: float, flux_scale: float) -> bool:
        """Detect a diverging pressure-velocity coupling and recover from it.

        When the raw continuity residual becomes non-finite or grows ``GUARD_GROWTH``
        times above its minimum so far (floored at ``GUARD_ROUNDOFF`` x the face mass
        flux, so that round-off noise in periodic or converged flows is not mistaken
        for growth), restore the fields of the last healthy snapshot and multiply the
        pressure under-relaxation by ``GUARD_FACTOR``. The iteration counter is not
        rewound: discarded iterations count as work, and ``max_iter`` bounds the total.
        Returns True on a rollback. Only active when ``case.relax_p`` is None.
        """
        if not self.guard_enabled or self.external_energy:
            # a conjugate solver owns the solid temperatures, which a rollback here would not restore
            return False
        r = raw_continuity
        ref = max(self._guard_min, self.GUARD_ROUNDOFF * flux_scale)
        bad = not math.isfinite(r) or (self.iteration > 10 and r > self.GUARD_GROWTH * ref)
        if bad and self._guard_snap is not None and self.relax_p * self.GUARD_FACTOR >= self.GUARD_MIN_RELAX:
            snap, snap_r = self._guard_snap
            at = self.iteration
            self.load_state(snap)
            self.iteration = self.step_count = at
            self.relax_p *= self.GUARD_FACTOR
            self._amg_age = 10 ** 9
            self._guard_min = snap_r
            self.guard_events.append({"detected_at": at, "restored_to": int(snap["iteration"]),
                                      "relax_p": round(self.relax_p, 4), "continuity": r})
            return True
        stage2 = self.GUARD_FALLBACK and (self.algo == "SIMPLEC" or self.relax_u * 0.7 >= self.GUARD_MIN_RELAX_U - 1e-12)
        if bad and self._guard_snap is not None and stage2:
            # relax_p is at its floor. A high-Reynolds flow on a coarse or skewed mesh (elbow,
            # Re ~ 1e4) needs the classical robust setting: SIMPLE with momentum relaxation 0.3
            # (measured: converges in ~340 iterations where SIMPLEC at any relax_p diverged); a
            # case that is already SIMPLE gets its momentum relaxation lowered instead.
            snap, snap_r = self._guard_snap
            at = self.iteration
            self.load_state(snap)
            self.iteration = self.step_count = at
            if self.algo == "SIMPLEC":
                self.algo, self.relax_u, self.relax_p = "SIMPLE", min(self.relax_u, 0.3), 0.3
            else:
                self.relax_u *= 0.7
            self._amg_age = 10 ** 9
            self._guard_min = snap_r
            self.guard_events.append({"detected_at": at, "restored_to": int(snap["iteration"]), "algorithm": self.algo,
                                      "relax_p": round(self.relax_p, 4), "relax_u": round(self.relax_u, 4),
                                      "continuity": r})
            return True
        if bad and self.guard_exhausted_at is None:  # cannot lower relax_p further: record it, do not roll back
            self.guard_exhausted_at = self.iteration
        if math.isfinite(r) and self.iteration > 5:  # past the start-up transient
            self._guard_min = min(self._guard_min, r)
            if self.iteration % self.GUARD_EVERY == 0 and r <= 2.0 * self._guard_min:
                self._guard_snap = (self.state_dict(), r)
        return False

    def _energy_inputs(self) -> dict:
        """Everything the energy equation reads that changes between iterations (snapshots)."""
        c = self.case
        gam = c.conductivity / c.cp
        if self.turb is not None:
            gam = self.turb.energy_diffusivity(gam)  # k/cp + mu_t/Pr_t
        F, Fb = self._conv()
        return {"F": F, "Fb": Fb, "Fb_abs": self.Fb, "gam": gam, "src": self.energy_source,
                "time": self._time_snapshot(["T"])}

    def _energy(self) -> dict:
        T, res, cap = self._energy_kernel(self._T, self._energy_inputs())
        self._T = T
        self._energy_capture = cap
        return res

    def _energy_kernel(self, T, inp: dict):
        """One energy (+ P-1) iteration from ``T`` with the snapshot ``inp``.

        Returns (new T, residuals {"energy", "p1"?}, capture). Besides its arguments it reads only
        static solver data and the lagged state it owns (``_gT``, ``rad.G``), so it may run in a lane."""
        c, fv, xp = self.case, self.fv, self.xp
        gam = inp["gam"]
        conv = (inp["F"], inp["Fb"])
        TP = T[fv.Pb]
        fixed = self.tkind == T_FIXED
        flux = self.tkind == T_FLUX
        out = self.bkind == OUTLET
        if self._gT is not None:
            TP = TP + xp.sum(self._gT[fv.Pb] * fv.db_t, axis=1)
        Tb = xp.where(fixed, self.tval, TP)
        Tb = xp.where(out & (inp["Fb_abs"] < 0), self.tval, Tb)  # backflow enters at the outlet's T
        gT = fv.gradient(T, Tb)
        self._gT = gT
        q_in = fv.Cb_d @ (flux * self.tval * fv.Sb_mag / c.cp)
        if inp["src"] is not None:
            q_in = q_in + inp["src"]
        diag, aPN, aNP, Db = self._coefficients(gam, fixed.astype(self.dtype), conv)
        rhs = self._rhs(gam, T, gT, Tb, Db, q_in, scheme=c.scalar_convection, conv=conv)
        if self.rad is not None:
            dd, rr = self.rad.energy_terms(T)
            diag = diag + dd
            rhs = rhs + rr
        diag, rhs = self._time_terms("T", T, diag, rhs, inp.get("time"))
        A = fv.matrix(diag, aPN, aNP)
        r = self._residual(A, diag, T, rhs)
        aT = c.relax_T
        dr = diag / aT
        Ar = fv.matrix(dr, aPN, aNP)
        b = rhs + ((1.0 - aT) / aT) * diag * T
        T_new, info = self._solve(Ar, b, T, dr, tol=1e-3, maxiter=60)
        cap = {"A": A, "b": rhs, "T_old": T, "Ar": Ar, "br": b, "T_new": T_new, "alpha": aT, "info": info}
        res = {"energy": r}
        if self.rad is not None:
            res["p1"] = self.rad.solve(T_new, self._wall_temperature(T_new))
        return T_new, res, cap

    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:
        if self.case.unsteady:
            return self.time >= self.case.t_end - 1e-9 * self.case.dt or self.iteration >= self.case.max_iter
        return self.converged or self.iteration >= self.case.max_iter

    def residuals(self) -> dict:
        """scaled: Fluent-style (continuity normalised by the largest of the first
        five iterations); raw: continuity as the absolute sum |mass imbalance|.

        Momentum/energy entries are the pre-solve residuals of the assembled, unrelaxed
        system at the iteration-start field, sum|b - A phi| / sum|a_P phi_P|;
        continuity is sum|m*| of the predictor (Rhie-Chow) fluxes, before the pressure
        correction. :meth:`diagnostics` separates these from the solved and corrected states.
        """
        r = dict(getattr(self, "_last_res", {}))
        raw = dict(getattr(self, "_last_raw", r))
        return {"raw": raw, "scaled": r, "normalized": r}

    def diagnostics(self) -> dict:
        """Algebraic, conservation and health diagnostics of the most recent outer iteration:
        pre-solve and solved-system residuals (componentwise backward errors), predictor
        and corrected local continuity, global boundary imbalance, field change per
        iteration and non-finite counts. Computed on demand from the systems the
        iteration solved; definitions in :mod:`flowtrace.unstructured.diagnostics`."""
        from .diagnostics import evaluate

        self._sync_lanes()
        if self._bridge is not None and self._capture is not None and self._bridge.capture is not None:
            self._capture["energy"] = self._bridge.capture
        return evaluate(self)

    def describe(self) -> dict:
        amg = self._amg
        return {
            "solver": f"pressure-based {self.algo}",
            "turbulence": self.case.turbulence,
            "backend": self.be.name, "device": self.be.device, "precision": str(self.dtype),
            "cells": self.nc, "faces": self.mesh.n_faces, "dim": self.dim,
            "convection": self.case.scheme,
            "relax_p": self.relax_p,
            "stability_rollbacks": list(self.guard_events),
            "stability_guard_exhausted_at": self.guard_exhausted_at,
            "pressure_solver": "AMG-PCG (" + ("pyamg" if amg is not None and not hasattr(amg, "levels")
                                               else "FlowTrace SA-AMG") + ")",
            "max_non_orthogonality_deg": round(self.fv.non_orthogonality_deg, 2),
            "setup_seconds": round(self.setup_seconds, 3),
            "scalar_lanes": self.lanes_report(),
        }

    def fields(self, names=None) -> Dict[str, np.ndarray]:
        self._sync_lanes()
        h = self.be.to_numpy
        U = h(self.U).astype(np.float64)
        out = {"x-velocity": U[:, 0], "y-velocity": U[:, 1]}
        if self.dim == 3:
            out["z-velocity"] = U[:, 2]
        out["velocity-magnitude"] = np.linalg.norm(U, axis=1)
        out["pressure"] = h(self.p).astype(np.float64)
        if self.energy:
            out["temperature"] = h(self.T).astype(np.float64)
        if self.turb is not None:
            out.update(self.turb.fields())
        if self.rad is not None:
            out["incident-radiation"] = h(self.rad.G).astype(np.float64)
        for name, Y in self.Y.items():
            out[name] = h(Y).astype(np.float64)
        if names:
            out = {k: v for k, v in out.items() if k in names}
        return out

    def cell_centres(self) -> np.ndarray:
        return self.fv.geometry.cell_centre

    def mass_balance(self) -> dict:
        """Net mass flow through every boundary zone (should sum to ~0)."""
        Fb = self.be.to_numpy(self.Fb)
        out = {}
        for z in self.mesh.boundary_zones():
            m = self.fv.bzone == z.id
            if m.any():
                out[z.name] = float(Fb[m].sum())
        return out

    def periodic_mass_flow(self) -> dict:
        """Mass flow entering through zone_a (= leaving through zone_b) for each periodic pair."""
        F = self.be.to_numpy(self.F)
        return {f"{p['zones'][0]}->{p['zones'][1]}": -float(F[p["faces"]].sum()) for p in self.fv.periodic}

    def _wall_viscosity(self):
        """Wall viscosity per boundary face, as the momentum equation uses it."""
        if self.turb is not None:
            return self._split_gamma(self.turb.momentum_viscosity(self.U))[1] * self.xp.ones(self.fv.n_bf, self.dtype)
        return self.xp.full(self.fv.n_bf, self.case.viscosity, dtype=self.dtype)

    def wall_report(self) -> dict:
        """Per wall zone: area, forces (pressure + viscous), Cd/Cl, y+, skin friction, heat flows.

        Cd/Cl use ``case.ref_velocity`` and ``case.ref_area``. The forces equal
        :meth:`wall_forces`; heat flows are conduction (``heat_flow_conv``) and P-1
        radiation (``heat_flow_rad``) into the fluid.
        """
        self._sync_lanes()
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        rho, mu = c.density, c.viscosity
        Pb, nb = fv.Pb, fv.nb
        rel = self.U[Pb] - self.ub_fixed
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        muw = self._wall_viscosity()
        tau_vec = muw[:, None] * rel_t / fv.yb[:, None]
        tau = xp.sqrt(xp.sum(tau_vec * tau_vec, axis=1))
        yplus = fv.yb * xp.sqrt(tau / rho) * rho / mu
        forces = self.wall_forces()
        qconv = None
        if self.energy:
            kc = c.conductivity
            qconv = h(kc * fv.alpha_b * (self._wall_temperature() - self.T[Pb]) * (self.tkind == T_FIXED)
                      + xp.where(self.tkind == T_FLUX, self.tval * fv.Sb_mag, 0.0))
        qrad = h(self.rad.wall_flux(self._wall_temperature())) if self.rad is not None else None
        hp = {"tau": h(tau), "yp": h(yplus), "A": h(fv.Sb_mag)}
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        out = {}
        for z in self.mesh.boundary_zones():
            if z.name not in forces:
                continue
            m = fv.bzone == z.id
            A = hp["A"][m]
            f = forces[z.name]
            Ft = np.asarray(f["total"])
            rep = {"area": float(A.sum()), "force_pressure": f["pressure"], "force_viscous": f["viscous"],
                   "force_total": f["total"], "cd": float(Ft[0] / (qdyn * c.ref_area)),
                   "cl": float(Ft[1] / (qdyn * c.ref_area)),
                   "yplus_avg": float((hp["yp"][m] * A).sum() / A.sum()), "yplus_max": float(hp["yp"][m].max()),
                   "cf_avg": float((hp["tau"][m] * A).sum() / A.sum() / qdyn)}
            if qconv is not None:
                rep["heat_flow_conv"] = float(qconv[m].sum())
            if qrad is not None:
                rep["heat_flow_rad"] = float(qrad[m].sum())
            out[z.name] = rep
        return out

    def wall_profile(self, zone: str) -> dict:
        """Face-wise distributions on one wall zone (x, y, Cf, Cp, y+), sorted by x, for plotting."""
        xp, fv, c = self.xp, self.fv, self.case
        h = self.be.to_numpy
        z = self.mesh.zone_by_name(zone)
        m = fv.bzone == z.id
        rho, mu = c.density, c.viscosity
        Pb, nb = fv.Pb, fv.nb
        rel = self.U[Pb] - self.ub_fixed
        rel_t = rel - xp.sum(rel * nb, axis=1)[:, None] * nb
        tau = self._wall_viscosity() * xp.sqrt(xp.sum(rel_t * rel_t, axis=1)) / fv.yb
        qdyn = 0.5 * rho * c.ref_velocity ** 2
        xfb = h(fv.xfb)[m]
        order = np.argsort(xfb[:, 0])
        return {"x": xfb[order, 0], "y": xfb[order, 1], "cf": h(tau)[m][order] / qdyn,
                "cp": h(self._pressure_bvals(self.p, self._gp))[m][order] / qdyn,
                "yplus": h(fv.yb * xp.sqrt(tau / rho) * rho / mu)[m][order]}

    def wall_forces(self) -> dict:
        """Force exerted by the fluid on every wall zone (Fluent: Report > Forces), per unit depth in 2-D.

        Built from the discrete terms of the momentum equation, so it balances the
        solution exactly: the face pressure ``p_f S_f`` (the reconstructed boundary
        pressure) plus the wall shear ``mu_w alpha_b (u_P - u_w)_t``, where ``mu_w`` is
        the wall viscosity the momentum equation uses (laminar ``mu``, or the
        wall-function / SST value with turbulence). Returns
        ``{zone: {"pressure": [..], "viscous": [..], "total": [..]}}``.
        """
        h = self.be.to_numpy
        fv = self.fv
        pb = h(self._pressure_bvals(self.p, self._gp)).astype(np.float64)
        if self.turb is not None:
            mu_b = h(self._split_gamma(self.turb.momentum_viscosity(self.U))[1]).astype(np.float64)
            mu_b = np.broadcast_to(mu_b, pb.shape)
        else:
            mu_b = np.full(pb.shape, float(self.case.viscosity))
        Sb, nb = h(fv.Sb).astype(np.float64), h(fv.nb).astype(np.float64)
        alpha_b = h(fv.alpha_b).astype(np.float64)
        du = h(self.U).astype(np.float64)[h(fv.Pb)] - h(self.ub_fixed).astype(np.float64)
        du_t = du - np.einsum("fd,fd->f", du, nb)[:, None] * nb
        f_p = pb[:, None] * Sb  # S points out of the fluid, i.e. into the wall
        f_v = (mu_b * alpha_b)[:, None] * du_t
        out = {}
        for z in self.mesh.boundary_zones():
            m = (fv.bzone == z.id) & (self.bkind_h == WALL)
            if not m.any():
                continue
            fp, fvis = f_p[m].sum(axis=0), f_v[m].sum(axis=0)
            out[z.name] = {"pressure": fp.tolist(), "viscous": fvis.tolist(), "total": (fp + fvis).tolist()}
        return out

    def heat_flow(self) -> dict:
        """Heat flow into the domain through every boundary zone (conduction + enthalpy flux).

        At steady state (no volumetric sources) the values sum to ~0; Fluent's
        *Report > Fluxes > Total Heat Transfer Rate* reports the same quantity
        with the opposite sign convention for outflow.
        """
        self._sync_lanes()
        if not self.energy:
            return {}
        h = self.be.to_numpy
        c, fv = self.case, self.fv
        T = h(self.T).astype(np.float64)
        Pb = h(fv.Pb)
        TP = T[Pb]
        tk, tv = h(self.tkind), h(self.tval).astype(np.float64)
        kind = self.bkind_h
        Fb = h(self.Fb).astype(np.float64)
        alpha_b = h(fv.alpha_b).astype(np.float64)
        Sb = h(fv.Sb_mag).astype(np.float64)
        Tb = np.where(tk == T_FIXED, tv, TP)
        Tb = np.where((kind == OUTLET) & (Fb < 0), tv, Tb)
        cond = c.conductivity * alpha_b * (Tb - TP) * (tk == T_FIXED)
        cond = cond + np.where(tk == T_FLUX, tv * Sb, 0.0)
        conv = -c.cp * Fb * Tb  # Fb > 0 is outflow
        q = cond + conv
        out = {}
        for z in self.mesh.boundary_zones():
            out[z.name] = float(q[fv.bzone == z.id].sum())
        return out

    def state_dict(self) -> dict:
        self._sync_lanes()
        h = self.be.to_numpy
        st = {"U": h(self.U), "p": h(self.p), "F": h(self.F), "Fb": h(self.Fb), "iteration": self.iteration}
        if self.energy:
            st["T"] = h(self.T)
        if self.turb is not None:
            st.update(self.turb.state())
            if self._gp is not None:
                st["grad_p"] = h(self._gp)
            if self._gp_eff is not None:
                st["grad_p_eff"] = h(self._gp_eff)
            if self._gU is not None:
                st["grad_U"] = h(self.xp.stack(self._gU))
        st["time"], st["time_step"] = self.time, self.time_step
        st["res_continuity_norm"] = self._res_max5.get("continuity", 0.0)
        if self.rad is not None:
            st["G"] = h(self.rad.G)
        for name, Y in self.Y.items():
            st[f"Y:{name}"] = h(Y)
        return st

    def load_state(self, st: dict):
        self._sync_lanes()
        if self._bridge is not None:
            self._bridge.mark_dirty()  # cross-device lanes reload T, G and Y (and drop their lagged gradient)
        xp = self.xp
        self.U = xp.asarray(st["U"], dtype=self.dtype)
        self.p = xp.asarray(st["p"], dtype=self.dtype)
        self.F = xp.asarray(st["F"], dtype=self.dtype)
        self.Fb = xp.asarray(st["Fb"], dtype=self.dtype)
        if self.energy and "T" in st:
            self.T = xp.asarray(st["T"], dtype=self.dtype)
        if self.rad is not None and "G" in st:
            self.rad.G = xp.asarray(st["G"], dtype=self.dtype)
        for name in self.Y:
            if f"Y:{name}" in st:
                self.Y[name] = xp.asarray(st[f"Y:{name}"], dtype=self.dtype)
        self.iteration = int(st.get("iteration", 0))
        self._gp = xp.asarray(st["grad_p"], dtype=self.dtype) if "grad_p" in st else None
        self._gp_eff = xp.asarray(st["grad_p_eff"], dtype=self.dtype) if "grad_p_eff" in st else None
        self._gU = xp.asarray(st["grad_U"], dtype=self.dtype) if "grad_U" in st else None
        self._gT = None
        self._forcing_step = None
        self._capture = None  # diagnostics describe an iteration, not a restored state
        if self.turb is not None:
            self.turb.load(st)
        if float(st.get("res_continuity_norm", 0.0)) > 0:
            self._res_max5["continuity"] = float(st["res_continuity_norm"])
        self.time = float(st.get("time", 0.0))
        self.time_step = int(st.get("time_step", 0))
        if self.case.unsteady:
            self._old, self._F_old = {}, []
            self._store_old()
