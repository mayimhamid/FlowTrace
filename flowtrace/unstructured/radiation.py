r"""P-1 radiation model (Fluent: Models > Radiation > P1).

Incident radiation G obeys

    -div(Gamma grad G) + a G = 4 a sigma T^4,      Gamma = 1 / (3 (a + sigma_s))

and feeds the energy equation with -div(q_r) = a G - 4 a sigma T^4 (linearised
in T for stability). Walls use the Marshak condition

    -Gamma dG/dn = eps / (2 (2 - eps)) (4 sigma T_w^4 - G_w),

which, eliminated together with the near-wall gradient, is a conductance
in series with the cell: q = A B / (A + B) (G_P - 4 sigma T_w^4). Inlets,
outlets and openings radiate as black surfaces at their temperature.
Temperatures must be absolute (K).
"""

from __future__ import annotations

import numpy as np

SIGMA_SB = 5.670374419e-8


class P1Model:
    def __init__(self, solver):
        s = self.s = solver
        xp, c = s.xp, s.case
        self.xp = xp
        coefficients = np.asarray([c.absorption, c.scattering])
        extinction = float(coefficients.sum())
        if not np.isfinite(extinction) or np.any(coefficients < 0) or extinction == 0:
            raise ValueError("P1 requires finite non-negative coefficients and positive extinction")
        self.gamma = 1.0 / (3.0 * extinction)
        eps = np.ones(s.fv.n_bf)
        rad = np.zeros(s.fv.n_bf, dtype=bool)
        for z in s.mesh.boundary_zones():
            m = s.fv.bzone == z.id
            spec = s.zone_specs.get(z.name)
            if spec is None or spec.kind in ("symmetry", "periodic"):
                continue
            rad[m] = True
            eps[m] = spec.emissivity if spec.kind == "wall" else 1.0
        eps = np.clip(eps, 1e-6, 1.0)
        self.eps = xp.asarray(eps, dtype=s.dtype)
        self.rad = xp.asarray(rad.astype(float), dtype=s.dtype)
        T = s.T
        self.G = 4.0 * SIGMA_SB * T ** 4

    def _wall_conductance(self):
        fv = self.s.fv
        A = self.gamma * fv.alpha_b  # cell-to-face diffusion conductance (times area)
        B = self.eps / (2.0 * (2.0 - self.eps)) * fv.Sb_mag
        return self.rad * A * B / (A + B)

    def solve(self, T, Tb):
        """Update G from the current temperature field; returns the scaled residual."""
        s, fv, xp, c = self.s, self.s.fv, self.xp, self.s.case
        a = c.absorption
        V = fv.vol
        D = self.gamma * fv.alpha
        cb = self._wall_conductance()
        diag = fv.scatter(D, D, cb) + a * V
        emit = 4.0 * SIGMA_SB * T ** 4
        Gb = self.G[fv.Pb]
        gG = fv.gradient(self.G, Gb)
        gf = fv.face_interp_vec(gG)
        rhs = (fv.Cb_d @ (cb * 4.0 * SIGMA_SB * Tb ** 4) + a * emit * V
               + fv.Cdiff_d @ (self.gamma * xp.sum(gf * fv.Tvec, axis=1)))
        A = fv.matrix(diag, -D, -D)
        r = s._residual(A, diag, self.G, rhs)
        self.G, _ = s._solve(A, rhs, self.G, diag, tol=1e-6, maxiter=200)
        return r

    def energy_terms(self, T):
        """(extra diagonal, extra rhs) for the energy equation in temperature form (divided by cp)."""
        c, V = self.s.case, self.s.fv.vol
        a = c.absorption
        T = self.xp.maximum(T, 0.0)
        T3 = T ** 3
        diag = 16.0 * a * SIGMA_SB * T3 * V / c.cp
        rhs = (a * self.G + 12.0 * a * SIGMA_SB * T3 * T) * V / c.cp
        return diag, rhs

    def wall_flux(self, Tb):
        """Radiative heat flux into the domain on every boundary face (W)."""
        cb = self._wall_conductance()
        return cb * (4.0 * SIGMA_SB * Tb ** 4 - self.G[self.s.fv.Pb])


class RadiationDeviceBuffers:
    """Owned CUDA buffers for a caller's external radiation backend."""

    def __init__(self, solver):
        s = self.s = solver
        fv, xp = s.fv, s.xp
        nodes = s.mesh.nodes
        indices, faces = [], []
        for local_face, face in enumerate(fv.bf_idx):
            ids = s.mesh.face_nodes[s.mesh.face_ptr[face]:s.mesh.face_ptr[face + 1]]
            pts = nodes[ids]
            turns = np.cross(pts - np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0) - pts)
            normal = np.sum(turns, axis=0)
            if np.any(turns @ normal < -1e-12 * (normal @ normal)):
                raise ValueError("external radiation requires convex boundary polygons")
            for k in range(1, len(ids) - 1):
                indices.append((ids[0], ids[k], ids[k + 1]))
                faces.append(local_face)
        triangles = nodes[np.asarray(indices, dtype=np.int64).reshape(-1, 3)]
        self.triangles = xp.asarray(triangles, dtype=s.dtype)
        self.triangle_boundary_face = xp.asarray(faces, dtype=xp.int32)
        self.absorption = xp.full(s.nc, s.case.absorption, dtype=s.dtype)
        self.source = xp.zeros(s.nc, dtype=s.dtype)
        self.temperature = s.T.copy()

    def pointers(self):
        s = self.s
        arrays = {"cell_centroids": s.fv.xc, "temperatures": self.temperature,
                  "absorption": self.absorption, "boundary_triangles": self.triangles,
                  "triangle_boundary_face": self.triangle_boundary_face, "radiative_source": self.source}
        return {name: int(array.data.ptr) for name, array in arrays.items()} | {
            "cell_count": s.nc, "triangle_count": len(self.triangles), "dtype": s.dtype.name,
            "stream": int(s.xp.cuda.get_current_stream().ptr),
        }
