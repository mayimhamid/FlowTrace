"""Device bridge: wall time to convergence with the scalar lanes off (serial) and on (measured placement).

Heated turbulent duct (k-omega SST, energy, one passive species), steady, FP64, run to convergence
(tol 1e-5) from the same start. For every size and backend: the serial solver (``lanes="off"``, the same
iterates as FlowTrace before the bridge) and ``lanes="auto"`` (placement and core budget measured on the
run itself, tuning iterations included in the wall time). Reported: iterations, wall time, speed-up and
the largest difference of the converged fields relative to their range.

Run:  python benchmarks/device_bridge.py [--sizes 32,64,128] [--backends numpy,cupy]
Writes benchmarks/results/device_bridge.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("x-velocity", "pressure", "temperature", "dye", "turb-kinetic-energy")


def duct(n: int, backend: str, lanes, max_iter: int):
    from flowtrace.mesh.unstructured import cartesian_mesh_2d
    from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
    from flowtrace.unstructured.simple import SimpleSolver

    m = cartesian_mesh_2d(4 * n, n, 4.0, 1.0)
    b = {"left": BoundarySpec("inlet", normal_speed=1.0, thermal="temperature", value=300.0, species={"dye": 1.0}),
         "right": BoundarySpec("outlet"), "bottom": BoundarySpec("wall", thermal="temperature", value=350.0),
         "top": BoundarySpec("wall")}
    c = UnstructuredCase(boundaries=b, density=1.0, viscosity=1e-4, energy=True, conductivity=1e-4, cp=1.0,
                         turbulence="k-omega-sst", species=["dye"], backend=backend, init_T=300.0,
                         init_velocity=[1, 0, 0], max_iter=max_iter, tol=1e-5, lanes=lanes)
    return SimpleSolver(m, c)


def solve(n, backend, lanes, max_iter):
    s = duct(n, backend, lanes, max_iter)
    t = time.perf_counter()
    while not s.finished:
        s.step()
    s.be.synchronize()
    wall = time.perf_counter() - t
    return s, wall, s.fields()


def run(sizes, backends, max_iter, out_dir):
    from flowtrace.hardware.backend import available_backends
    from flowtrace.hardware.probe import cached_profile

    rows = []
    for name in backends:
        if name not in available_backends():
            print(f"skip {name}: not available")
            continue
        for n in sizes:
            s0, w0, f0 = solve(n, name, "off", max_iter)
            s1, w1, f1 = solve(n, name, "auto", max_iter)
            # relative to the field's range, or to its magnitude when the range collapses (at steady
            # state the dye fills the duct uniformly: range ~1e-9, value 1)
            diff = max(float(np.abs(f0[k] - f1[k]).max() / max(np.ptp(f0[k]), 1e-3 * np.abs(f0[k]).max(), 1e-30))
                       for k in FIELDS)
            tun = s1.lanes_report().get("tuning") or {}
            row = {"backend": name, "cells": s0.nc, "serial_iterations": s0.iteration, "serial_s": round(w0, 2),
                   "serial_converged": s0.converged, "bridge_iterations": s1.iteration, "bridge_s": round(w1, 2),
                   "bridge_converged": s1.converged, "speedup": round(w0 / w1, 2), "max_rel_field_diff": diff,
                   "chosen": tun.get("chosen"), "steady_ms": tun.get("ms_per_iteration")}
            rows.append(row)
            print(f"{name:5} {s0.nc:>7} cells  serial {w0:7.1f} s ({s0.iteration} its)  bridge {w1:7.1f} s "
                  f"({s1.iteration} its)  x{w0 / w1:.2f}  diff {diff:.1e}  -> {row['chosen']}", flush=True)
    p = cached_profile()
    d = {"date": dt.datetime.now().isoformat(timespec="seconds"), "cpu": p.cpu_model,
         "gpu": p.gpus[0].name if p.gpus else None, "rows": rows,
         "reproduce": "python benchmarks/device_bridge.py --sizes " + ",".join(map(str, sizes))
                      + " --backends " + ",".join(backends)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "device_bridge.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    with open(os.path.join(out_dir, "device_bridge.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(d))
    return d


def render_markdown(d) -> str:
    dev = {"numpy": "CPU", "cupy": "GPU"}
    lines = [f"Heated turbulent duct (k-omega SST + energy + 1 species), steady, FP64, to tol 1e-5. "
             f"{d['cpu']}" + (f", {d['gpu']}" if d["gpu"] else "") + f". {d['date'][:10]}.", "",
             "| flow on | cells | serial: its / s | bridge: its / s | speed-up | max field diff | placement chosen |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in d["rows"]:
        lines.append(f"| {dev.get(r['backend'], r['backend'])} | {r['cells']:,} | {r['serial_iterations']} / "
                     f"{r['serial_s']} | {r['bridge_iterations']} / {r['bridge_s']} | {r['speedup']:.2f}x | "
                     f"{r['max_rel_field_diff']:.1e} | {r['chosen']} |")
    lines += ["", "Field difference: max |serial - bridge| / range of the serial field (or 1e-3 x its magnitude when "
              "the range collapses), over x-velocity, pressure, temperature, dye and k. "
              "Bridge wall time includes the tuning iterations.", "",
              f"Reproduce: `{d['reproduce']}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--backends", default="numpy,cupy")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    run([int(x) for x in a.sizes.split(",")], a.backends.split(","), a.max_iter, a.out)
