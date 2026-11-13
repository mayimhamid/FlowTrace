"""
2-D steady, incompressible flow and heat transfer: collocated finite-volume SIMPLE solver.

Governing equations (constant properties, unit depth)

    continuity   div(u)                          = 0
    momentum     div(rho u u) = div(mu grad u) - dp/dx      (and the same for v)
    energy       div(rho u T) = div((k/cp) grad T)          (temperature is a passive scalar)

Numerics
    * uniform Cartesian mesh, every unknown (u, v, p, T) stored at the cell centres;
    * convection: first-order upwind, or second-order upwind by deferred correction
      (implicit upwind matrix + explicit high-order correction, so the matrix stays an M-matrix);
    * diffusion: central, two-point;
    * Rhie-Chow momentum interpolation (with the Majumdar under-relaxation correction) gives the
      face fluxes, which removes the pressure checkerboard of a collocated grid;
    * SIMPLE: momentum predictor -> pressure-correction Poisson equation -> correct p, u, v and
      the face fluxes -> energy equation;
    * implicit under-relaxation  aP/alpha * phi_P = sum(a_nb phi_nb) + b + (1-alpha)/alpha * aP phi_P_old.

Only NumPy and SciPy are needed. Run this file for a lid-driven cavity (Re = 100) that is
checked against Ghia, Ghia and Shin (1982), and for a heated channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SIDES = ("west", "east", "south", "north")


# --------------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------------
class Mesh:
    """Uniform nx-by-ny mesh of the rectangle [0, Lx] x [0, Ly]. Arrays are shaped (ny, nx).

    Cell (j, i) has the centre (xc[i], yc[j]). The x-faces are numbered 0..nx (array shape
    (ny, nx+1)), the y-faces 0..ny (shape (ny+1, nx)); face 0 is the west / south boundary.
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        if nx < 2 or ny < 2:
            raise ValueError("the mesh needs at least 2 x 2 cells")
        self.nx, self.ny, self.Lx, self.Ly = nx, ny, Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.vol = self.dx * self.dy          # cell volume (unit depth)
        self.xc = (np.arange(nx) + 0.5) * self.dx
        self.yc = (np.arange(ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(self.xc, self.yc)
        self.n = nx * ny
        self.idx = np.arange(self.n).reshape(ny, nx)   # unknown number of cell (j, i)


# --------------------------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------------------------
@dataclass
class Boundary:
    """One side of the rectangle.

    kind  "wall"    no-slip wall moving with (u, v); temperature T fixed, or None = adiabatic.
          "inlet"   prescribed velocity (u, v) and temperature T.
          "outlet"  fixed pressure p, zero-gradient velocity and temperature.
    """

    kind: str
    u: float = 0.0
    v: float = 0.0
    T: Optional[float] = None
    p: float = 0.0

    def __post_init__(self):
        if self.kind not in ("wall", "inlet", "outlet"):
            raise ValueError(f"unknown boundary kind {self.kind!r}")


@dataclass
class BoundaryConditions:
    west: Boundary
    east: Boundary
    south: Boundary
    north: Boundary
    _by_side: Dict[str, Boundary] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self._by_side = {s: getattr(self, s) for s in SIDES}

    def __getitem__(self, side: str) -> Boundary:
        return self._by_side[side]

    @property
    def has_outlet(self) -> bool:
        return any(b.kind == "outlet" for b in self._by_side.values())

    def velocity_dirichlet(self, side: str):
        """(is_dirichlet, u_b, v_b): walls and inlets fix the velocity, an outlet does not."""
        b = self[side]
        return b.kind != "outlet", b.u, b.v

    def temperature_dirichlet(self, side: str):
        """(is_dirichlet, T_b): inlets and isothermal walls fix T; adiabatic walls and outlets do not."""
        b = self[side]
        if b.kind == "inlet":
            return True, 0.0 if b.T is None else b.T
        if b.kind == "wall" and b.T is not None:
            return True, b.T
        return False, 0.0


# --------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------
class SimpleSolver:
    """Steady SIMPLE solver on a collocated grid with Rhie-Chow interpolation."""

    def __init__(self, mesh: Mesh, bc: BoundaryConditions, rho: float = 1.0, mu: float = 0.01,
                 k: float = 1.0, cp: float = 1.0, solve_energy: bool = False,
                 scheme: str = "upwind2", alpha_p: float = 0.3, alpha_u: float = 0.7,
                 alpha_T: float = 0.8):
        if scheme not in ("upwind1", "upwind2"):
            raise ValueError("scheme must be 'upwind1' or 'upwind2'")
        self.m, self.bc = mesh, bc
        self.rho, self.mu, self.k, self.cp = rho, mu, k, cp
        self.energy, self.scheme = solve_energy, scheme
        self.alpha_p, self.alpha_u, self.alpha_T = alpha_p, alpha_u, alpha_T
        ny, nx = mesh.ny, mesh.nx

        # cell-centred unknowns
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.T = np.zeros((ny, nx))
        # face-normal mass fluxes  F = rho (u.n_x) A  on x-faces and  rho v A  on y-faces (signed with +x / +y)
        self.Fx = np.zeros((ny, nx + 1))
        self.Fy = np.zeros((ny + 1, nx))
        self.history: List[dict] = []
        self._mass_norm: Optional[float] = None
        self._u_ref = max([1e-12] + [abs(b.u) + abs(b.v) for b in (bc[s] for s in SIDES)])
        self._T_ref = max([1e-12] + [abs(bc.temperature_dirichlet(s)[1]) for s in SIDES])
        self.initialize()

    # ---------------------------------------------------------------- initial state
    def initialize(self, u0: float = 0.0, v0: float = 0.0, T0: float = 0.0, p0: float = 0.0):
        """Uniform initial fields; face fluxes consistent with them; boundary fluxes from the BCs."""
        m = self.m
        self.u[:], self.v[:], self.T[:], self.p[:] = u0, v0, T0, p0
        self.Fx[:, 1:-1] = self.rho * 0.5 * (self.u[:, 1:] + self.u[:, :-1]) * m.dy
        self.Fy[1:-1, :] = self.rho * 0.5 * (self.v[1:, :] + self.v[:-1, :]) * m.dx
        self._set_boundary_fluxes()
        self.u_prev, self.v_prev = self.u.copy(), self.v.copy()

    # ---------------------------------------------------------------- boundary helpers
    def _set_boundary_fluxes(self):
        """Mass flux through the domain boundary: wall/inlet from the BC, outlet from the cell value."""
        m, bc, rho = self.m, self.bc, self.rho
        for side, face, cell, comp, area, sign_col in (
            ("west", (slice(None), 0), (slice(None), 0), "u", m.dy, "x"),
            ("east", (slice(None), -1), (slice(None), -1), "u", m.dy, "x"),
            ("south", (0, slice(None)), (0, slice(None)), "v", m.dx, "y"),
            ("north", (-1, slice(None)), (-1, slice(None)), "v", m.dx, "y"),
        ):
            dirichlet, ub, vb = bc.velocity_dirichlet(side)
            vel = (ub if comp == "u" else vb) if dirichlet else getattr(self, comp)[cell]
            F = self.Fx if sign_col == "x" else self.Fy
            F[face] = rho * vel * area

    def _boundary_values(self, phi: np.ndarray, dirichlet: Dict[str, tuple]):
        """Face values of phi on the four domain boundaries: the Dirichlet value, else the cell value."""
        out = {}
        cells = {"west": phi[:, 0], "east": phi[:, -1], "south": phi[0, :], "north": phi[-1, :]}
        for s in SIDES:
            is_d, val = dirichlet[s]
            out[s] = np.full_like(cells[s], val) if is_d else cells[s].copy()
        return out

    def _pressure_boundary_values(self, p: np.ndarray):
        """Pressure on boundary faces: fixed at an outlet, linear extrapolation from the two nearest cells elsewhere."""
        ex = {"west": 1.5 * p[:, 0] - 0.5 * p[:, 1], "east": 1.5 * p[:, -1] - 0.5 * p[:, -2],
              "south": 1.5 * p[0, :] - 0.5 * p[1, :], "north": 1.5 * p[-1, :] - 0.5 * p[-2, :]}
        for s in SIDES:
            if self.bc[s].kind == "outlet":
                ex[s] = np.full_like(ex[s], self.bc[s].p)
        return ex

    def _gradient(self, phi: np.ndarray, bvals: Dict[str, np.ndarray]):
        """Green-Gauss cell gradient: (1/V) sum_f phi_f n_f A_f, with phi_f the average of the two cells."""
        m = self.m
        fx = np.empty((m.ny, m.nx + 1))
        fx[:, 1:-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])
        fx[:, 0], fx[:, -1] = bvals["west"], bvals["east"]
        fy = np.empty((m.ny + 1, m.nx))
        fy[1:-1, :] = 0.5 * (phi[1:, :] + phi[:-1, :])
        fy[0, :], fy[-1, :] = bvals["south"], bvals["north"]
        return (fx[:, 1:] - fx[:, :-1]) / m.dx, (fy[1:, :] - fy[:-1, :]) / m.dy

    # ---------------------------------------------------------------- convection-diffusion assembly
    def _assemble(self, phi: np.ndarray, gamma: float, dirichlet: Dict[str, tuple], alpha: float,
                  source: Optional[np.ndarray] = None):
        """Assemble the relaxed convection-diffusion system for one transported scalar.

        Finite-volume balance over cell P (outward fluxes):
            sum_f F_f phi_f - sum_f Gamma A_f (dphi/dn)_f = S V
        with the upwind convective flux and the central diffusive flux
            a_E = D_e + max(-F_e, 0),  a_W = D_w + max(F_w, 0),  (D = Gamma A / d)
            a_P = sum a_nb                 (the net-flux term F_e - F_w + ... vanishes once continuity holds)
        A boundary face with a fixed value has the half-cell distance d = dx/2 and adds a_B to a_P and
        a_B * phi_B to b; a zero-gradient face contributes nothing.
        Returns (matrix A, right side b, relaxed diagonal aP, relaxed b has the (1-alpha) term included).
        """
        m = self.m
        ny, nx, dx, dy = m.ny, m.nx, m.dx, m.dy
        Fe, Fw, Fn, Fs = self.Fx[:, 1:], self.Fx[:, :-1], self.Fy[1:, :], self.Fy[:-1, :]
        aE = gamma * dy / dx + np.maximum(-Fe, 0.0)
        aW = gamma * dy / dx + np.maximum(Fw, 0.0)
        aN = gamma * dx / dy + np.maximum(-Fn, 0.0)
        aS = gamma * dx / dy + np.maximum(Fs, 0.0)
        b = np.zeros((ny, nx)) if source is None else source * m.vol

        # domain-boundary faces: half-cell diffusion distance, inflow-only convection
        bface = {"west": (aW, (slice(None), 0), Fw[:, 0], gamma * dy / (0.5 * dx)),
                 "east": (aE, (slice(None), -1), -Fe[:, -1], gamma * dy / (0.5 * dx)),
                 "south": (aS, (0, slice(None)), Fs[0, :], gamma * dx / (0.5 * dy)),
                 "north": (aN, (-1, slice(None)), -Fn[-1, :], gamma * dx / (0.5 * dy))}
        for side, (arr, sl, inflow, D) in bface.items():
            is_d, val = dirichlet[side]
            if is_d:
                ab = D + np.maximum(inflow, 0.0)
                arr[sl] = ab
                b[sl] += ab * np.broadcast_to(val, ab.shape)
            else:
                arr[sl] = 0.0
        aP = aE + aW + aN + aS

        # implicit under-relaxation
        aPr = aP / alpha
        b = b + (1.0 - alpha) * aPr * phi

        idx = m.idx
        rows = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(), idx.ravel()])
        cols = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(), idx.ravel()])
        data = np.concatenate([-aE[:, :-1].ravel(), -aW[:, 1:].ravel(), -aN[:-1, :].ravel(), -aS[1:, :].ravel(), aPr.ravel()])
        A = sp.csc_matrix((data, (rows, cols)), shape=(m.n, m.n))
        return A, b, aPr

    def _second_order_correction(self, phi: np.ndarray, bvals: Dict[str, np.ndarray]) -> np.ndarray:
        """Deferred-correction source for second-order upwind (interior faces only).

        phi_f(SOU) = phi_U + grad(phi)_U . (x_f - x_U), the value extrapolated from the upwind cell U.
        The matrix keeps phi_f = phi_U, so the difference F (phi_f(SOU) - phi_U) is moved to the right side:
        -corr for the cell the flux leaves, +corr for the cell it enters.
        """
        m = self.m
        gx, gy = self._gradient(phi, bvals)
        corr = np.zeros((m.ny, m.nx))
        F = self.Fx[:, 1:-1]
        up_w = F > 0.0
        d = np.where(up_w, 0.5 * m.dx * gx[:, :-1], -0.5 * m.dx * gx[:, 1:])
        c = F * d
        corr[:, :-1] -= c
        corr[:, 1:] += c
        F = self.Fy[1:-1, :]
        up_s = F > 0.0
        d = np.where(up_s, 0.5 * m.dy * gy[:-1, :], -0.5 * m.dy * gy[1:, :])
        c = F * d
        corr[:-1, :] -= c
        corr[1:, :] += c
        return corr

    # ---------------------------------------------------------------- one SIMPLE iteration
    def iterate(self) -> dict:
        m, bc, rho = self.m, self.bc, self.rho
        ny, nx = m.ny, m.nx

        # ---- step 1: momentum predictor with the current pressure p*
        vdir = {s: (bc.velocity_dirichlet(s)[0], None) for s in SIDES}
        dirichlet_u = {s: (vdir[s][0], bc.velocity_dirichlet(s)[1]) for s in SIDES}
        dirichlet_v = {s: (vdir[s][0], bc.velocity_dirichlet(s)[2]) for s in SIDES}
        pb = self._pressure_boundary_values(self.p)
        gpx, gpy = self._gradient(self.p, pb)

        u_old, v_old = self.u.copy(), self.v.copy()
        bu = self._boundary_values(self.u, dirichlet_u)
        bv = self._boundary_values(self.v, dirichlet_v)
        # the matrix is the same for u and v (same fluxes, same fixed/free pattern of the boundaries)
        A, b_u, aPr = self._assemble(self.u, self.mu, dirichlet_u, self.alpha_u)
        _, b_v, _ = self._assemble(self.v, self.mu, dirichlet_v, self.alpha_u)
        if self.scheme == "upwind2":
            b_u += self._second_order_correction(self.u, bu)
            b_v += self._second_order_correction(self.v, bv)
        b_u -= gpx * m.vol          # pressure force  -(dp/dx) V
        b_v -= gpy * m.vol

        # momentum residual: ||b - A phi||_1 / sum(aP |phi|)  (before the solve)
        scale = self._u_ref
        floor = 1e-3 * scale
        res_u = np.abs(b_u.ravel() - A @ self.u.ravel()).sum() / np.sum(aPr * np.maximum(np.abs(self.u), floor))
        res_v = np.abs(b_v.ravel() - A @ self.v.ravel()).sum() / np.sum(aPr * np.maximum(np.abs(self.v), floor))
        lu = spla.splu(A)
        self.u = lu.solve(b_u.ravel()).reshape(ny, nx)
        self.v = lu.solve(b_v.ravel()).reshape(ny, nx)

        # coefficient linking velocity to the pressure gradient:  u = u_hat - d grad(p),  d = V / aP
        d = m.vol / aPr.reshape(ny, nx)

        # ---- Rhie-Chow face fluxes from the predicted velocities
        gpx, gpy = self._gradient(self.p, pb)
        self._rhie_chow(d, gpx, gpy, u_old, v_old)
        self._set_boundary_fluxes()

        # ---- step 2: pressure-correction equation from continuity
        imbalance = self.Fx[:, 1:] - self.Fx[:, :-1] + self.Fy[1:, :] - self.Fy[:-1, :]   # net outflow of F*
        mass_res = np.abs(imbalance).sum()
        pc = self._solve_pressure_correction(d, imbalance)

        # ---- step 3: correct velocity, face fluxes and pressure
        pcb = {s: (np.zeros(m.ny) if s in ("west", "east") else np.zeros(m.nx)) for s in SIDES}
        cells = {"west": pc[:, 0], "east": pc[:, -1], "south": pc[0, :], "north": pc[-1, :]}
        for s in SIDES:   # zero-gradient p' except at an outlet, where p' = 0
            pcb[s] = np.zeros_like(cells[s]) if bc[s].kind == "outlet" else cells[s].copy()
        gcx, gcy = self._gradient(pc, pcb)
        self.u -= d * gcx
        self.v -= d * gcy
        self._correct_fluxes(d, pc)
        self.p += self.alpha_p * pc
        if not bc.has_outlet:                       # pressure only matters up to a constant
            self.p -= self.p.mean()
        self.u_prev, self.v_prev = u_old, v_old     # cell velocities that belong to the previous face fluxes

        # ---- energy equation (passive scalar)
        res_T = 0.0
        if self.energy:
            res_T = self._solve_energy()

        # residual normalisation: mass residual relative to the largest of the first iterations
        it = len(self.history) + 1
        if it <= 5:
            self._mass_norm = max(self._mass_norm or 0.0, mass_res, 1e-300)
        rec = {"it": it, "continuity": mass_res / self._mass_norm, "u": res_u, "v": res_v, "T": res_T}
        self.history.append(rec)
        return rec

    # ---------------------------------------------------------------- Rhie-Chow interpolation
    def _rhie_chow(self, d, gpx, gpy, u_old, v_old):
        """Momentum interpolation of the face velocities.

            u_f = <u>_f  -  <d>_f ( (p_E - p_P)/dx - <dp/dx>_f )  +  (1 - alpha_u)( u_f^old - <u^old>_f )

        The bracket replaces the average of the two cell pressure gradients by the compact face
        gradient, which couples neighbouring pressures and kills the checkerboard mode. The last
        term (Majumdar) makes the result independent of the under-relaxation factor.
        """
        m, rho, au = self.m, self.rho, self.alpha_u
        uf_old = self.Fx[:, 1:-1] / (rho * m.dy)
        vf_old = self.Fy[1:-1, :] / (rho * m.dx)
        u_bar = 0.5 * (self.u[:, 1:] + self.u[:, :-1])
        v_bar = 0.5 * (self.v[1:, :] + self.v[:-1, :])
        d_x = 0.5 * (d[:, 1:] + d[:, :-1])
        d_y = 0.5 * (d[1:, :] + d[:-1, :])
        dpdx = (self.p[:, 1:] - self.p[:, :-1]) / m.dx - 0.5 * (gpx[:, 1:] + gpx[:, :-1])
        dpdy = (self.p[1:, :] - self.p[:-1, :]) / m.dy - 0.5 * (gpy[1:, :] + gpy[:-1, :])
        u_old_bar = 0.5 * (self.u_prev[:, 1:] + self.u_prev[:, :-1])
        v_old_bar = 0.5 * (self.v_prev[1:, :] + self.v_prev[:-1, :])
        uf = u_bar - d_x * dpdx + (1.0 - au) * (uf_old - u_old_bar)
        vf = v_bar - d_y * dpdy + (1.0 - au) * (vf_old - v_old_bar)
        self.Fx[:, 1:-1] = rho * uf * m.dy
        self.Fy[1:-1, :] = rho * vf * m.dx

    # ---------------------------------------------------------------- pressure correction
    def _pc_coefficients(self, d):
        """Face coefficients c_f = rho <d>_f A_f / delta_f of  F'_f = -c_f (p'_nb - p'_P)  and outlet coefficients."""
        m, rho = self.m, self.rho
        cx = rho * 0.5 * (d[:, 1:] + d[:, :-1]) * m.dy / m.dx
        cy = rho * 0.5 * (d[1:, :] + d[:-1, :]) * m.dx / m.dy
        cb = {"west": rho * d[:, 0] * m.dy / (0.5 * m.dx), "east": rho * d[:, -1] * m.dy / (0.5 * m.dx),
              "south": rho * d[0, :] * m.dx / (0.5 * m.dy), "north": rho * d[-1, :] * m.dx / (0.5 * m.dy)}
        return cx, cy, cb

    def _solve_pressure_correction(self, d, imbalance):
        """Solve  aP p'_P - sum a_nb p'_nb = -(net outflow of F*)  (continuity of the corrected fluxes)."""
        m = self.m
        ny, nx = m.ny, m.nx
        cx, cy, cb = self._pc_coefficients(d)
        aE = np.zeros((ny, nx)); aW = np.zeros((ny, nx)); aN = np.zeros((ny, nx)); aS = np.zeros((ny, nx))
        aE[:, :-1] = cx; aW[:, 1:] = cx; aN[:-1, :] = cy; aS[1:, :] = cy
        aP = aE + aW + aN + aS
        sl = {"west": (slice(None), 0), "east": (slice(None), -1), "south": (0, slice(None)), "north": (-1, slice(None))}
        for s in SIDES:
            if self.bc[s].kind == "outlet":
                aP[sl[s]] += cb[s]                      # p' = 0 on an outlet face
        idx = m.idx
        rows = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(), idx.ravel()])
        cols = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(), idx.ravel()])
        data = np.concatenate([-aE[:, :-1].ravel(), -aW[:, 1:].ravel(), -aN[:-1, :].ravel(), -aS[1:, :].ravel(), aP.ravel()])
        A = sp.lil_matrix(sp.csr_matrix((data, (rows, cols)), shape=(m.n, m.n)))
        rhs = -imbalance.ravel().copy()
        if not self.bc.has_outlet:                       # singular (Neumann) problem: pin p' of cell 0
            A[0, :] = 0.0
            A[0, 0] = 1.0
            rhs[0] = 0.0
        return spla.spsolve(A.tocsc(), rhs).reshape(ny, nx)

    def _correct_fluxes(self, d, pc):
        """F_f <- F_f - c_f (p'_nb - p'_P); at an outlet face p' = 0 on the boundary side."""
        cx, cy, cb = self._pc_coefficients(d)
        self.Fx[:, 1:-1] -= cx * (pc[:, 1:] - pc[:, :-1])
        self.Fy[1:-1, :] -= cy * (pc[1:, :] - pc[:-1, :])
        if self.bc["east"].kind == "outlet":
            self.Fx[:, -1] += cb["east"] * pc[:, -1]
        if self.bc["west"].kind == "outlet":
            self.Fx[:, 0] -= cb["west"] * pc[:, 0]
        if self.bc["north"].kind == "outlet":
            self.Fy[-1, :] += cb["north"] * pc[-1, :]
        if self.bc["south"].kind == "outlet":
            self.Fy[0, :] -= cb["south"] * pc[0, :]

    # ---------------------------------------------------------------- energy
    def _solve_energy(self) -> float:
        """rho cp u.grad(T) = k lap(T), written as div(F T) = div((k/cp) grad T) with the corrected fluxes."""
        dirichlet = {s: self.bc.temperature_dirichlet(s) for s in SIDES}
        bT = self._boundary_values(self.T, dirichlet)
        A, b, aPr = self._assemble(self.T, self.k / self.cp, dirichlet, self.alpha_T)
        if self.scheme == "upwind2":
            b += self._second_order_correction(self.T, bT)
        floor = 1e-3 * self._T_ref
        res = np.abs(b.ravel() - A @ self.T.ravel()).sum() / np.sum(aPr * np.maximum(np.abs(self.T), floor))
        self.T = spla.spsolve(A, b.ravel()).reshape(self.m.ny, self.m.nx)
        return res

    # ---------------------------------------------------------------- driver
    def solve(self, max_iter: int = 2000, tol: float = 1e-6, verbose: bool = True, every: int = 50) -> bool:
        """Iterate until the scaled residuals of continuity, u, v (and T) are below tol."""
        for _ in range(max_iter):
            r = self.iterate()
            vals = [r["continuity"], r["u"], r["v"]] + ([r["T"]] if self.energy else [])
            if not np.all(np.isfinite(vals)):
                raise FloatingPointError("the iteration diverged; lower the under-relaxation factors or refine the mesh")
            if verbose and (r["it"] % every == 0 or r["it"] == 1):
                T = f"  T {r['T']:.2e}" if self.energy else ""
                print(f"it {r['it']:5d}  continuity {r['continuity']:.2e}  u {r['u']:.2e}  v {r['v']:.2e}{T}")
            if max(vals) < tol:
                if verbose:
                    print(f"converged in {r['it']} iterations")
                return True
        return False

    # ---------------------------------------------------------------- post-processing
    def centerline_u(self):
        """u(y) on the vertical centre line x = Lx/2 (average of the two columns around it)."""
        nx = self.m.nx
        col = 0.5 * (self.u[:, nx // 2 - 1] + self.u[:, nx // 2]) if nx % 2 == 0 else self.u[:, nx // 2]
        return self.m.yc, col

    def mass_imbalance(self) -> float:
        """Largest cell-wise net mass outflow of the converged face fluxes (should be ~0)."""
        return float(np.abs(self.Fx[:, 1:] - self.Fx[:, :-1] + self.Fy[1:, :] - self.Fy[:-1, :]).max())

    def plot(self, path: Optional[str] = None):
        """Contours of |u| with velocity vectors and of T (needs matplotlib; optional)."""
        import matplotlib.pyplot as plt

        m = self.m
        n = 2 if self.energy else 1
        fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
        sp_ = np.hypot(self.u, self.v)
        c = axs[0, 0].contourf(m.X, m.Y, sp_, 20, cmap="viridis")
        axs[0, 0].streamplot(m.X, m.Y, self.u, self.v, color="w", linewidth=0.6, density=1.2)
        fig.colorbar(c, ax=axs[0, 0], label="|u|")
        axs[0, 0].set_title("velocity magnitude")
        if self.energy:
            c = axs[0, 1].contourf(m.X, m.Y, self.T, 20, cmap="inferno")
            fig.colorbar(c, ax=axs[0, 1], label="T")
            axs[0, 1].set_title("temperature")
        for a in axs.ravel():
            a.set_aspect("equal")
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=130)
        else:
            plt.show()


# --------------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982), Re = 100, u on the vertical centre line (y, u)
GHIA_RE100 = [(0.0547, -0.04192), (0.1016, -0.06434), (0.1719, -0.10150), (0.2813, -0.15662),
              (0.4531, -0.21090), (0.5, -0.20581), (0.6172, -0.13641), (0.7344, 0.00332),
              (0.8516, 0.23151), (0.9531, 0.68717), (0.9766, 0.84123)]


def lid_driven_cavity(n: int = 32, re: float = 100.0, scheme: str = "upwind2"):
    """Unit square, lid moving with u = 1, rho = 1, mu = 1/Re. Compared with Ghia et al. (Re = 100)."""
    wall = Boundary("wall")
    bc = BoundaryConditions(west=wall, east=wall, south=wall, north=Boundary("wall", u=1.0))
    s = SimpleSolver(Mesh(n, n), bc, rho=1.0, mu=1.0 / re, scheme=scheme)
    s.solve(max_iter=3000, tol=1e-7)
    y, u = s.centerline_u()
    err = max(abs(np.interp(yg, y, u) - ug) for yg, ug in GHIA_RE100)
    print(f"cavity {n}x{n}, Re {re:g}, {scheme}: max |u - u_Ghia| on the centre line = {err:.4f}, "
          f"max cell mass imbalance {s.mass_imbalance():.1e}")
    return s


def heated_channel(nx: int = 60, ny: int = 20):
    """Developing laminar flow in a plane channel (height 1, length 10) with a hot lower wall (T = 1),
    cold inflow (T = 0) and an adiabatic upper wall. Re = rho U H / mu = 100, Pr = mu cp / k = 1."""
    bc = BoundaryConditions(west=Boundary("inlet", u=1.0, T=0.0), east=Boundary("outlet"),
                            south=Boundary("wall", T=1.0), north=Boundary("wall"))
    s = SimpleSolver(Mesh(nx, ny, Lx=10.0, Ly=1.0), bc, rho=1.0, mu=0.01, k=0.01, cp=1.0,
                     solve_energy=True, scheme="upwind1")
    s.initialize(u0=1.0)
    s.solve(max_iter=3000, tol=1e-6)
    u_out = s.u[:, -1]
    print(f"channel: outlet centre-line u = {u_out.max():.3f} (fully developed Poiseuille: 1.5), "
          f"mean outlet T = {s.T[:, -1].mean():.3f}, max cell mass imbalance {s.mass_imbalance():.1e}")
    return s


if __name__ == "__main__":
    cavity = lid_driven_cavity(32, 100.0)
    channel = heated_channel()
    try:
        cavity.plot("cavity.png")
        channel.plot("channel.png")
    except ImportError:
        print("matplotlib is not installed: plots skipped")
