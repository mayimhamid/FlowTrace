"""Mesh certificate: how far the reported numbers are from the mesh-independent answer.

A converged residual says the discrete equations are solved; it says nothing about how far
the answer is from the one a fine enough mesh would give. The usual remedy, a mesh-independence
study, is manual work that is often skipped. ``certify`` does it as part of the run:

1. the converged solution is the coarse level;
2. the mesh is refined uniformly (:func:`flowtrace.mesh.refine.refine_2d`: same polygonal
   domain, same zones) and solved again, started from the coarse solution;
3. every reported quantity (pressure drop per inlet, wall forces, heat flows, kinetic energy)
   gets an error band from the change between the levels, following Roache's Grid Convergence
   Index: two levels use the formal order of the scheme with safety factor 3; three levels
   use the observed order (limited to [0.5, formal]) with 1.25 when the sequence converges
   monotonically, and three times the largest change when it oscillates or grows;
4. the difference between the coarse field and the restricted fine field is returned per cell:
   a map of where the mesh is too coarse.

The band is an estimate, not a bound. A quantity whose changes do not shrink is reported as
"not converging" (typical at singular corners: a uniform inlet profile meeting a no-slip wall)
instead of being given a small band.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from ..mesh.refine import refine_2d

FS_TWO, FS_THREE = 3.0, 1.25
P_MIN = 0.5


@dataclass
class GCIReport:
    observed_order: float
    extrapolated_value: float
    gci_fine_pct: float
    gci_medium_pct: float
    asymptotic_ratio: float
    convergence_regime: str


def compute_gci_certificate(
    f1: float,
    f2: float,
    f3: float,
    r21: float,
    r32: float,
    formal_order: float = 2.0,
    Fs: float = 1.25,
) -> GCIReport:
    """
    Computes Roache's Grid Convergence Index (GCI) and Richardson extrapolation
    for three systematically refined grids (h1 < h2 < h3) using fixed-point
    iteration for non-constant refinement ratios.
    """
    eps32 = f3 - f2
    eps21 = f2 - f1

    if abs(eps21) < 1e-15:
        return GCIReport(
            observed_order=formal_order,
            extrapolated_value=f1,
            gci_fine_pct=0.0,
            gci_medium_pct=0.0,
            asymptotic_ratio=1.0,
            convergence_regime="EXACT_CONVERGED",
        )

    ratio = eps32 / eps21
    s = 1.0 if ratio > 0.0 else -1.0

    # Fixed-point iteration for observed order p
    p = formal_order
    ln_r21 = np.log(r21)
    abs_ratio = abs(ratio)

    for _ in range(100):
        q_p = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs_ratio) + q_p) / ln_r21
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = 0.5 * (p + p_new)

    f_ext_21 = (r21**p * f1 - f2) / (r21**p - 1.0)
    ea_21 = abs((f1 - f2) / max(abs(f1), 1e-15))
    ea_32 = abs((f2 - f3) / max(abs(f2), 1e-15))

    gci_21 = (Fs * ea_21) / (r21**p - 1.0)
    gci_32 = (Fs * ea_32) / (r32**p - 1.0)

    # Asymptotic range check: |eps32| / (r21^p * |eps21|) should approach 1.0
    asymptotic_ratio = abs(eps32) / max((r21**p) * abs(eps21), 1e-15)

    if s < 0:
        regime = "OSCILLATORY"
    elif 0.85 <= asymptotic_ratio <= 1.15 and p >= 0.5 * formal_order:
        regime = "ASYMPTOTIC_MONOTONIC"
    else:
        regime = "PRE_ASYMPTOTIC"

    return GCIReport(
        observed_order=float(p),
        extrapolated_value=float(f_ext_21),
        gci_fine_pct=float(gci_21 * 100.0),
        gci_medium_pct=float(gci_32 * 100.0),
        asymptotic_ratio=float(asymptotic_ratio),
        convergence_regime=regime,
    )


def functionals(solver) -> Dict[str, float]:
    """The scalar results of a run that a user reads off: one number each, defined on any mesh."""
    h = solver.be.to_numpy
    fv, mesh = solver.fv, solver.mesh
    out: Dict[str, float] = {}
    pb = h(solver._pressure_bvals(solver.p, solver._gp)).astype(np.float64)
    A = h(fv.Sb_mag).astype(np.float64)
    zones = [z for z in mesh.boundary_zones() if (fv.bzone == z.id).any()]

    def mean_p(z):
        m = fv.bzone == z.id
        return float((pb[m] * A[m]).sum() / A[m].sum())

    outlets = [z for z in zones if solver.zone_specs[z.name].kind in ("outlet", "pressure")]
    p_out = float(np.mean([mean_p(z) for z in outlets])) if outlets else 0.0
    for z in zones:
        if solver.zone_specs[z.name].kind == "inlet":
            out[f"pressure drop {z.name} -> outlet"] = mean_p(z) - p_out
    for name, f in solver.wall_forces().items():
        for k, ax in enumerate("xyz"[:solver.dim]):
            out[f"force {ax} on {name}"] = float(f["total"][k])
    U = h(solver.U).astype(np.float64)
    vol = h(fv.vol).astype(np.float64)
    out["mean kinetic energy"] = float((0.5 * (U * U).sum(axis=1) * vol).sum() / vol.sum())
    if solver.energy:
        T = h(solver.T).astype(np.float64)
        out["mean temperature"] = float((T * vol).sum() / vol.sum())
        for name, q in solver.heat_flow().items():
            spec = solver.zone_specs.get(name)
            if spec is not None and spec.kind == "wall" and spec.thermal == "temperature":
                out[f"heat flow through {name}"] = float(q)
    return out


def error_band(values: List[float], r: float, p_formal: float) -> dict:
    """Error band of ``values[-1]`` (finest) from values on 2 or 3 levels refined by the ratio ``r``.

    Returns {"value", "band", "order", "status"}; status is one of "two-level", "asymptotic",
    "slow" (order below 0.5), "irregular" (order well above the formal one: the changes shrink, but
    not as a power of the mesh size), "oscillating", "not converging", "converged" (no change at round-off).
    """
    f1 = float(values[-1])
    e21 = float(values[-2]) - f1
    tiny = 1e-12 * max(abs(f1), abs(float(values[-2])), 1e-300)
    if len(values) == 2:
        band = FS_TWO * abs(e21) / (r ** p_formal - 1.0)
        return {"value": f1, "band": band, "order": None, "status": "converged" if abs(e21) <= tiny else "two-level"}
    e32 = float(values[-3]) - float(values[-2])
    if abs(e21) <= tiny and abs(e32) <= tiny:
        return {"value": f1, "band": 0.0, "order": None, "status": "converged"}
    ratio = e21 / e32 if abs(e32) > tiny else math.inf
    if 0.0 < ratio < 1.0:
        p = math.log(1.0 / ratio) / math.log(r)
        ok = P_MIN <= p <= p_formal + 0.5
        p_use = min(max(p, P_MIN), p_formal)
        band = (FS_THREE if ok else FS_TWO) * abs(e21) / (r ** p_use - 1.0)
        status = "asymptotic" if ok else "slow" if p < P_MIN else "irregular"
        return {"value": f1, "band": band, "order": p, "status": status}
    status = "oscillating" if ratio < 0.0 else "not converging"
    return {"value": f1, "band": FS_TWO * max(abs(e21), abs(e32)), "order": None, "status": status}


def _warm_fields(solver, parent: np.ndarray) -> dict:
    f = solver.fields()
    keep = ("x-velocity", "y-velocity", "z-velocity", "pressure", "temperature")
    return {k: np.asarray(v)[parent] for k, v in f.items() if k in keep}


def _restrict(values: np.ndarray, vol: np.ndarray, parent: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(parent, weights=values * vol, minlength=n) / np.bincount(parent, weights=vol, minlength=n)


def certify(solver, levels: int = 2, make_solver: Optional[Callable] = None, run: Optional[Callable] = None,
            log: Optional[Callable[[str], None]] = None, max_cells: int = 400_000) -> dict:
    """Refine and re-solve ``levels - 1`` times starting from the converged ``solver``.

    ``make_solver(mesh, case)`` builds a solver (default: same class and backend), ``run(solver,
    level)`` iterates it to convergence (default: a plain loop). Returns the certificate dict;
    ``["solver"]`` is the finest solver and ``["error_map"]`` the per-cell velocity difference
    between the two finest levels, on the finest mesh, relative to the largest speed.
    """
    if levels not in (2, 3):
        raise ValueError("levels must be 2 or 3")
    if solver.dim != 2:
        raise NotImplementedError("the mesh certificate needs uniform refinement, implemented for 2-D meshes")
    if solver.case.unsteady:
        raise NotImplementedError("the mesh certificate is for steady runs")
    log = log or (lambda msg: None)
    if make_solver is None:
        def make_solver(mesh, case):
            return type(solver)(mesh, case, backend=solver.be)
    if run is None:
        def run(s, level):
            while not s.finished:
                s.step()
    h = solver.be.to_numpy
    chain = [solver]
    cells = [solver.nc]
    vals = [functionals(solver)]
    converged = [bool(solver.converged)]
    parent = None
    for lvl in range(1, levels):
        cur = chain[-1]
        if cur.nc * 4 > max_cells and lvl > 0:
            log(f"certificate: level {lvl} would exceed {max_cells:,} cells, stopping at {len(chain)} levels")
            break
        mesh_f, parent = refine_2d(cur.mesh)
        case_f = copy.deepcopy(cur.case)
        case_f.periodic = []  # already applied to (and absent from) a refinable mesh
        case_f.max_iter = max(int(cur.case.max_iter), 2 * int(cur.iteration) + 500)
        fine = make_solver(mesh_f, case_f)
        fine.initialize(_warm_fields(cur, parent))
        log(f"certificate: level {lvl}: {mesh_f.n_cells:,} cells (refined from {cur.nc:,}), started from the coarser solution")
        run(fine, lvl)
        chain.append(fine)
        cells.append(fine.nc)
        vals.append(functionals(fine))
        converged.append(bool(fine.converged))
    if len(chain) < 2:
        raise RuntimeError("the mesh certificate needs at least one refined level")
    fine, coarse = chain[-1], chain[-2]
    r = math.sqrt(cells[-1] / cells[-2])
    p_formal = 2.0 if solver.case.scheme == "second-order" else 1.0
    Uf = h(fine.U).astype(np.float64)
    Uc = h(coarse.U).astype(np.float64)
    vol_f = h(fine.fv.vol).astype(np.float64)
    vol_c = h(coarse.fv.vol).astype(np.float64)
    u_ref = max(float(np.linalg.norm(Uf, axis=1).max()), 1e-300)
    R = np.stack([_restrict(Uf[:, k], vol_f, parent, coarse.nc) for k in range(solver.dim)], axis=1)
    diff = np.linalg.norm(Uc - R, axis=1) / u_ref  # per coarse cell
    scale_p = r ** p_formal - 1.0
    field = {"mean": float((diff * vol_c).sum() / vol_c.sum() / scale_p), "max": float(diff.max() / scale_p)}
    quantities = {}
    for name in vals[-1]:
        series = [v[name] for v in vals[-len(chain):]]
        b = error_band(series, r, p_formal)
        mag = max(abs(b["value"]), 1e-300)
        b["relative"] = b["band"] / mag
        b["levels"] = series
        quantities[name] = b
    # quantities that are zero by symmetry (a force component ~ round-off) carry no relative meaning
    big = max((abs(q["value"]) for n, q in quantities.items() if n.startswith("force")), default=0.0)
    for n, q in quantities.items():
        if n.startswith("force") and abs(q["value"]) < 1e-6 * big:
            q["relative"] = q["band"] / big
    worst = max(quantities.items(), key=lambda kv: kv[1]["relative"])
    notes = []
    if not all(converged):
        notes.append("a level stopped at its iteration limit before reaching the tolerance: bands include iteration error")
    turb = getattr(fine, "turb", None)
    if turb is not None and "epsilon" in str(solver.case.turbulence):
        yp = [w["yplus_avg"] for w in fine.wall_report().values()]
        if yp and min(yp) < 11.0:
            notes.append(f"wall functions with y+ ~ {min(yp):.0f} on the finest level (below the log layer): "
                         "the bands include the wall model's sensitivity to the first-cell height")
    bad = [n for n, q in quantities.items() if q["status"] in ("oscillating", "not converging")]
    if bad:
        notes.append("not converging with the mesh: " + ", ".join(bad) + " (a singular corner, an unsteady flow "
                     "or a mesh still too coarse)")
    return {"levels": len(chain), "cells": cells, "refinement_ratio": r, "formal_order": p_formal,
            "converged": converged, "quantities": quantities, "velocity_field_error": field,
            "worst": {"name": worst[0], "relative": worst[1]["relative"]}, "notes": notes,
            "error_map": diff[parent] / scale_p, "solver": fine, "mesh": fine.mesh}


def format_certificate(cert: dict) -> List[str]:
    """Plain-text lines of a certificate (run log, CLI)."""
    lines = [f"Mesh certificate: {cert['levels']} levels, {' -> '.join(f'{c:,}' for c in cert['cells'])} cells; "
             f"values on the finest level, +/- estimated discretisation error"]
    for name, q in cert["quantities"].items():
        order = f", observed order {q['order']:.2f}" if q.get("order") is not None else ""
        lines.append(f"  {name}: {q['value']:.5g} +/- {q['band']:.2g} ({100 * q['relative']:.2g} %; {q['status']}{order})")
    fe = cert["velocity_field_error"]
    lines.append(f"  velocity field on the coarser level: mean error {100 * fe['mean']:.2g} %, "
                 f"worst cell {100 * fe['max']:.2g} % of the largest speed")
    lines += [f"  note: {n}" for n in cert["notes"]]
    return lines
