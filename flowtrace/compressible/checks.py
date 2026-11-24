"""Physical checks of a finished density-based run, and comparison with a Fluent data file.

The checks are conservation statements a correct steady solution satisfies whatever the mesh:

* global mass balance: the mass flows through the boundaries sum to zero;
* total temperature: with adiabatic walls and no work, the total enthalpy of a steady flow is the
  inflow's. Mass-averaged over the outflow boundaries it must equal the inflow value; no cell may
  exceed the largest inflow total temperature by more than a few percent (overshoot is a sign of
  a boundary condition or a limiter creating energy);
* positivity of density and pressure.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# Fluent data-file names of the fields compared, and the scale each difference is divided by
COMPARED = ("pressure", "temperature", "density", "x-velocity", "y-velocity", "z-velocity")


def physical_checks(solver) -> List[dict]:
    h = solver.be.to_numpy
    g = solver.gamma
    f = solver.fields()
    mb = solver.mass_balance()
    inflow = sum(-v for v in mb.values() if v < 0)
    out: List[dict] = []
    net = abs(sum(mb.values()))
    rel = net / inflow if inflow > 0 else net
    out.append({"name": "Global mass balance", "value": rel, "unit": "relative", "pass": bool(rel < 1e-3)})
    T0 = f["temperature"] * (1.0 + 0.5 * (g - 1.0) * f["mach-number"] ** 2)
    specs = solver.zone_specs
    t_in = [s.T0 for s in specs.values() if s.kind == "pressure-inlet"]
    t_in += [s.T * (1.0 + 0.5 * (g - 1.0) * s.mach ** 2) for s in specs.values() if s.kind == "far-field"]
    iso = any(s.kind == "wall" and s.wall_T is not None for s in specs.values())
    if t_in and not iso:
        over = float(T0.max() / max(t_in) - 1.0)
        out.append({"name": "Total temperature overshoot above the inflow value", "value": over, "unit": "relative",
                    "pass": bool(over < 0.05)})
        mdot_b = h(solver.mdot_b).astype(np.float64)
        Pb = h(solver.fv.Pb)
        outm = mdot_b > 0
        if outm.any() and len(set(t_in)) == 1:
            t_out = float((mdot_b[outm] * T0[Pb][outm]).sum() / mdot_b[outm].sum())
            dev = t_out / t_in[0] - 1.0
            out.append({"name": "Mass-averaged total temperature at the outflow vs inflow", "value": dev,
                        "unit": "relative", "pass": bool(abs(dev) < 0.02)})
    ok = bool(np.isfinite(f["pressure"]).all() and f["pressure"].min() > 0 and f["density"].min() > 0)
    out.append({"name": "Positive density and pressure", "value": float(f["pressure"].min()), "unit": "Pa (minimum)",
                "pass": ok})
    return out


def compare_with_fluent(solver, data_path: str) -> Dict[str, dict]:
    """Cell-by-cell differences between this solution and a Fluent ``.dat.h5`` of the same mesh.

    Per field: the volume-weighted mean and the largest absolute difference, as fractions of the
    range of the Fluent field. The two solvers differ in scheme and wall treatment, so agreement
    of a few percent in the mean is what a correct solution gives; shocks and boundary layers
    carry the largest local differences."""
    from ..mesh.fluent_h5 import read_data_h5

    ref = read_data_h5(data_path, solver.nc)
    f = solver.fields()
    vol = solver.be.to_numpy(solver.V).astype(np.float64)
    out: Dict[str, dict] = {}
    for name in COMPARED:
        if name not in ref or name not in f:
            continue
        a, b = np.asarray(ref[name], dtype=np.float64), f[name]
        if a.shape != b.shape or not np.isfinite(a).all():
            continue
        scale = float(a.max() - a.min()) or 1.0
        d = np.abs(a - b)
        out[name] = {"fluent": [float(a.min()), float(a.max())], "flowtrace": [float(b.min()), float(b.max())],
                     "mean": float((d * vol).sum() / vol.sum() / scale), "max": float(d.max() / scale)}
    if not out:
        raise ValueError(f"{data_path}: no cell field of this mesh to compare ({solver.nc} cells expected)")
    return out


def format_comparison(cmp: Dict[str, dict]) -> List[str]:
    lines = ["difference from the Fluent data file, as a fraction of each Fluent field's range"]
    for k, v in cmp.items():
        lines.append(f"{k}: mean {100 * v['mean']:.2f} %, largest {100 * v['max']:.1f} %  "
                     f"(Fluent {v['fluent'][0]:.4g} to {v['fluent'][1]:.4g}, FlowTrace {v['flowtrace'][0]:.4g} to {v['flowtrace'][1]:.4g})")
    return lines
