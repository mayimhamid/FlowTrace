"""Automatic solver options from the mesh and the boundary conditions.

``auto_configure`` looks at what the case is (inlet speed and size, fluid properties, wall-cell
height, mesh skewness, temperatures and species named on the boundaries) and sets the options a
careful user would set by hand, returning every decision with its reason so the run log shows it:

* mesh length unit: ``units="auto"`` reads an extent of 100 or more as millimetres and scales the mesh (reported);
* flow regime from Re = rho U L / mu: laminar below 2300, otherwise a turbulence model, chosen by
  the estimated wall y+ (viscous sublayer resolved -> k-omega SST, wall functions -> realizable k-epsilon);
* energy equation when the boundaries carry different or fixed temperatures;
* species transport for the species named on inlets;
* bounded (first-order) scalar transport on skewed meshes, where the second-order scheme was
  measured to overshoot the boundary temperatures by 10 % of their range (elbow, 45 deg);
* first-order momentum on extremely skewed meshes, and an iteration budget that grows with the mesh.

Only the options listed in ``AUTO_FIELDS`` are touched; the fluid properties and the boundary values
always stay the user's.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..mesh.unstructured import UnstructuredMesh, compute_geometry, scale_mesh

AUTO_FIELDS = ("turbulence", "energy", "scheme", "scalar_scheme", "species", "max_iter", "init_T", "t_ref",
               "ref_velocity", "ref_area", "unsteady", "dt", "t_end", "time_scheme")
# flow past a bluff body sheds vortices above this Reynolds number (circular cylinder ~47, square ~45):
# a steady solution does not exist as a stable state, and steady iterations do not converge to one
RE_SHEDDING = 47.0
SHEDDING_DT = 0.025  # time step in units of L/U: about 250 steps per shedding period at St ~ 0.15
SHEDDING_T_END = 120.0  # run length in units of L/U: onset from a symmetric start plus ~10 periods
UNIT_FACTORS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "in": 0.0254}


def guess_units(mesh: UnstructuredMesh) -> Tuple[str, str]:
    """Length unit of a mesh from its extent: pipes, elbows and test sections are drawn in mm far more
    often than they are 100 m long, so an extent of 100 or more is read as millimetres."""
    span = float(np.ptp(mesh.nodes, axis=0).max())
    if span >= 100.0:
        return "mm", f"mesh extent {span:g}: looks like millimetres"
    return "m", f"mesh extent {span:g}: metres"


def _inlets(case):
    return {n: b for n, b in case.boundaries.items() if b.kind == "inlet"}


def _zone_faces(mesh, name):
    z = mesh.zone_by_name(name)
    return np.nonzero((mesh.neighbour < 0) & (mesh.face_zone == z.id))[0] if z is not None else np.zeros(0, int)


def _non_orthogonality(mesh, geo, limit: int = 2_000_000) -> float:
    inner = np.nonzero(mesh.neighbour >= 0)[0]
    if inner.size == 0:
        return 0.0
    if inner.size > limit:
        inner = inner[:: inner.size // limit + 1]
    d = geo.cell_centre[mesh.neighbour[inner]] - geo.cell_centre[mesh.owner[inner]]
    S = geo.face_area_vec[inner]
    cos = np.einsum("fd,fd->f", S, d) / np.maximum(np.linalg.norm(S, axis=1) * np.linalg.norm(d, axis=1), 1e-300)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max())


def _immersed_body(mesh, case, inlets):
    """(zone name, cross-flow size) of the largest wall zone that does not touch the domain's outer
    bounding box (a cylinder, an airfoil, a vehicle), or None for an internal flow."""
    lo, hi = mesh.nodes.min(0), mesh.nodes.max(0)
    tol = 1e-3 * float((hi - lo).max())
    # mean flow direction: into the domain through the inlets (outward area vectors reversed)
    from ..mesh.unstructured import _face_geometry

    fc, S = _face_geometry(mesh)
    flow = np.zeros(mesh.dim)
    for name in inlets:
        flow -= S[_zone_faces(mesh, name)].sum(0)
    if not np.linalg.norm(flow) > 0:
        return None
    flow /= np.linalg.norm(flow)
    best = None
    for name, b in case.boundaries.items():
        if b.kind != "wall":
            continue
        f = _zone_faces(mesh, name)
        if f.size == 0:
            continue
        nodes = np.unique(np.concatenate([mesh.face_nodes[mesh.face_ptr[i]:mesh.face_ptr[i + 1]] for i in f]))
        x = mesh.nodes[nodes]
        if np.any(x.min(0) <= lo + tol) or np.any(x.max(0) >= hi - tol):
            continue  # touches the outer boundary: a channel or tunnel wall
        perp = x - np.outer(x @ flow, flow)  # positions projected on the plane normal to the flow
        size = float(max(np.ptp(perp @ e) for e in np.eye(mesh.dim) if abs(e @ flow) < 0.999) if mesh.dim > 1 else 0.0)
        if mesh.dim == 2:
            n = np.array([-flow[1], flow[0]])
            size = float(np.ptp(x @ n))
        if size > 0 and (best is None or size > best[1]):
            best = (name, size)
    return best


def auto_configure(mesh: UnstructuredMesh, case, units: str = "auto") -> Tuple[UnstructuredMesh, List[str], Dict]:
    """Return ``(mesh, notes, applied)``: the mesh scaled to metres, one note per decision and the
    option values set on ``case`` (which is modified in place)."""
    notes: List[str] = []
    applied: Dict[str, object] = {}

    def set_(name, value, why):
        setattr(case, name, value)
        applied[name] = value
        notes.append(f"auto: {name} = {value!r}: {why}")

    # ---- length unit
    if units == "auto":
        # Fluent mesh files carry no unit. The guess is applied and reported; a non-dimensional mesh
        # more than 100 units wide must be given as units="m".
        units, why = guess_units(mesh)
        notes.append(f"auto: mesh unit = {units!r}: {why}" + ("" if units == "m" else
                     "; set the mesh unit to m if the file really is in metres"))
    if units not in UNIT_FACTORS:
        raise ValueError(f"unknown mesh unit {units!r}; use one of {', '.join(UNIT_FACTORS)}")
    if UNIT_FACTORS[units] != 1.0:
        mesh = scale_mesh(mesh, UNIT_FACTORS[units])
        notes.append(f"mesh scaled by {UNIT_FACTORS[units]:g} ({units} -> m)")
    geo = compute_geometry(mesh)
    skew = _non_orthogonality(mesh, geo)

    # ---- flow regime
    inlets = _inlets(case)
    speed, length = 0.0, 0.0
    for name, b in inlets.items():
        u = b.normal_speed if b.normal_speed is not None else (np.linalg.norm(b.velocity) if b.velocity else 0.0)
        f = _zone_faces(mesh, name)
        area = float(geo.face_area[f].sum())
        d = area if mesh.dim == 2 else math.sqrt(4.0 * area / math.pi)  # width, or equivalent diameter
        speed = max(speed, abs(float(u)))
        if area * abs(float(u)) >= length * speed:  # the inlet that carries the most
            length = d
    body = _immersed_body(mesh, case, inlets) if speed > 0 else None
    if body is not None:  # external flow: the body, not the inlet, sets the length scale (D of a cylinder)
        length = body[1]
    if speed > 0 and length > 0:
        re = case.density * speed * length / case.viscosity
        what = f"cross-flow size of '{body[0]}'" if body else "inlet width" if mesh.dim == 2 else "inlet diameter"
        notes.append(f"flow estimate: U = {speed:g} m/s, L = {length:g} m ({what}), Re = {re:.3g}")
        if body is not None and case.ref_velocity == 1.0 and case.ref_area == 1.0:
            # Fluent's Reference Values: drag and lift coefficients of the body per unit depth (2-D)
            area = length if mesh.dim == 2 else length ** 2
            set_("ref_velocity", float(speed), "inlet speed (reference for Cd, Cl)")
            set_("ref_area", float(area), f"cross-flow size of '{body[0]}'" + (" x unit depth" if mesh.dim == 2 else " squared"))
        if body is not None:
            applied["body_zone"] = body[0]
        if body is not None and re > RE_SHEDDING and not case.unsteady:
            # the physical state is periodic vortex shedding: solve it in time instead of iterating
            # towards a steady state that the flow does not have
            tau = length / speed
            set_("unsteady", True, f"Re {re:.3g} > ~{RE_SHEDDING:g} past the bluff body '{body[0]}': the wake sheds "
                 "vortices, a steady solution is not physical")
            set_("dt", SHEDDING_DT * tau, f"{SHEDDING_DT:g} L/U (about 250 steps per shedding period)")
            set_("t_end", SHEDDING_T_END * tau, f"{SHEDDING_T_END:g} L/U: onset of shedding and about ten periods")
            set_("time_scheme", "bdf2", "second order in time")
            applied["shedding_steps"] = int(round(SHEDDING_T_END / SHEDDING_DT))
        walls = [n for n, b in case.boundaries.items() if b.kind == "wall"]
        wf = np.concatenate([_zone_faces(mesh, n) for n in walls]) if walls else np.zeros(0, int)
        if re < 2300.0:
            set_("turbulence", "laminar", f"Re {re:.3g} < 2300")
        else:
            h = 1e30
            if wf.size:
                # wall-adjacent cell height ~ cell volume / wall face area
                own = mesh.owner[wf]
                h = float(np.median(geo.cell_volume[own] / np.maximum(geo.face_area[wf], 1e-300)) / 2.0)
            yplus = h * (0.05 * speed) * case.density / case.viscosity
            model = "k-omega-sst" if yplus <= 5.0 else "k-epsilon-realizable"
            set_("turbulence", model, f"Re {re:.3g} >= 2300; estimated first-cell y+ ~ {yplus:.3g} "
                 + ("(sublayer resolved)" if yplus <= 5.0 else "(wall functions)"))
    else:
        notes.append("no inlet speed given (pressure- or force-driven case): turbulence model left as set")

    # ---- thermal
    temps = {round(float(b.value), 9) for b in case.boundaries.values()
             if (b.kind in ("inlet", "pressure")) or (b.kind == "wall" and b.thermal != "adiabatic")}
    thermal_walls = any(b.kind == "wall" and b.thermal != "adiabatic" for b in case.boundaries.values())
    inlet_T = {round(float(b.value), 9) for b in inlets.values()}
    if (len(inlet_T - {0.0}) >= 1 and len(inlet_T) >= 2) or thermal_walls:
        set_("energy", True, "boundaries carry different or fixed temperatures")
        ref = [t for t in temps if t > 0.0]
        if ref and case.init_T == 0.0:
            set_("init_T", float(np.mean(ref)), "mean of the boundary temperatures")
        if ref and case.t_ref == 0.0:
            set_("t_ref", float(np.mean(ref)), "mean of the boundary temperatures (buoyancy reference)")
        if skew > 30.0:
            set_("scalar_scheme", "upwind", f"{skew:.0f} deg non-orthogonality: second-order energy overshoots "
                 "the boundary temperatures here, first-order stays bounded")

    # ---- species named on the inlets
    names = sorted({k for b in case.boundaries.values() for k in (b.species or {})})
    if names and not case.species:
        set_("species", names, "species given on the boundaries")
        if skew > 30.0 and "scalar_scheme" not in applied:
            set_("scalar_scheme", "upwind", f"{skew:.0f} deg non-orthogonality: bounded species transport")

    # ---- discretisation and iteration budget
    if skew > 60.0:
        set_("scheme", "upwind", f"{skew:.0f} deg non-orthogonality: second-order momentum is not robust")
    budget = 3000 if mesh.n_cells < 100_000 else 5000
    if case.unsteady:  # max_iter counts every inner iteration of every time step
        budget = max(budget, int(math.ceil(case.t_end / case.dt)) * int(case.inner_iterations) + 1)
    if case.max_iter < budget:
        set_("max_iter", budget, f"{mesh.n_cells:,} cells")
    notes.append(f"mesh: max non-orthogonality {skew:.1f} deg")
    return mesh, notes, applied
