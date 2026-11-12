"""Concurrent scalar lanes (flowtrace.unstructured.bridge): exactness, consistency and placement."""

import numpy as np
import pytest

from flowtrace.mesh.unstructured import cartesian_mesh_2d
from flowtrace.unstructured import bridge as bridge_mod
from flowtrace.unstructured.bridge import eligible_lanes
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase
from flowtrace.unstructured.simple import SimpleSolver

FIELDS = ("x-velocity", "y-velocity", "pressure", "temperature", "dye")
# one thread per lane and for the flow: every reduction has the same order as the serial solver
CPU1 = {"energy": "cpu", "species": "cpu", "threads": {"flow": 1, "energy": 1, "species": 1}}


def _duct(lanes, n=12, turbulence="k-omega-sst", **kw):
    m = cartesian_mesh_2d(4 * n, n, 4.0, 1.0)
    b = {"left": BoundarySpec("inlet", normal_speed=1.0, thermal="temperature", value=300.0, species={"dye": 1.0}),
         "right": BoundarySpec("outlet"), "bottom": BoundarySpec("wall", thermal="temperature", value=350.0),
         "top": BoundarySpec("wall")}
    args = dict(boundaries=b, density=1.0, viscosity=1e-3, energy=True, conductivity=1e-3, cp=1.0,
                turbulence=turbulence, species=["dye"], backend="numpy", init_T=300.0, init_velocity=[1, 0, 0],
                max_iter=30, tol=1e-12, lanes=lanes)
    args.update(kw)
    return SimpleSolver(m, UnstructuredCase(**args))


def _run(s):
    while not s.finished:
        s.step()
    return s.fields()


def _serial_one_thread():
    s = _duct("off")
    s._flow_threads = 1
    return s


def test_cpu_lanes_are_bit_identical_to_the_serial_solver():
    ref = _run(_serial_one_thread())
    s = _duct(CPU1)
    assert s.lanes_report()["mode"] == "concurrent"
    out = _run(s)
    for k in FIELDS + ("turb-kinetic-energy",):
        assert np.array_equal(ref[k], out[k]), k
    assert s._bridge.lanes["energy"].launches == 30 and not s._bridge.busy


def test_unsteady_lanes_are_bit_identical():
    kw = dict(unsteady=True, dt=0.05, t_end=0.25, inner_iterations=4, turbulence="laminar", max_iter=10 ** 6)
    ref = _duct("off", **kw)
    ref._flow_threads = 1
    ref = _run(ref)
    out = _run(_duct(CPU1, **kw))
    for k in FIELDS:
        assert np.array_equal(ref[k], out[k]), k


def test_convergence_is_declared_on_a_state_converged_in_every_field():
    kw = dict(turbulence="laminar", max_iter=3000, tol=1e-6, n=8)
    a = _duct("off", **kw)
    a._flow_threads = 1
    fa = _run(a)
    b = _duct(CPU1, **kw)
    fb = _run(b)
    assert a.converged and b.converged
    # the lanes' residuals reach the flow one iteration late: at most one more iteration than serial
    assert a.iteration <= b.iteration <= a.iteration + 1
    res = b.residuals()["scaled"]
    assert res["energy"] < 1e-6 and res["dye"] < 1e-6 and not b._bridge.busy
    for k in FIELDS:
        scale = max(np.abs(fa[k]).max(), 1.0)
        assert np.abs(fa[k] - fb[k]).max() < 1e-4 * scale, k


def test_state_round_trip_with_lanes_in_flight():
    """state_dict / load_state (checkpoints, the stability guard) collect the lanes first."""
    ref = _serial_one_thread()
    for _ in range(12):
        ref.step()
    st_ref = ref.state_dict()
    s = _duct(CPU1)
    for _ in range(12):
        s.step()
    assert s._bridge.busy  # a lane is still running when the state is read
    st = s.state_dict()
    for k in ("U", "p", "T", "Y:dye", "k"):
        assert np.array_equal(st_ref[k], st[k]), k
    # restore an older state while lanes run, continue: same as the serial solver doing the same
    s.load_state(st)
    ref.load_state(st_ref)
    for _ in range(5):
        s.step()
        ref.step()
    assert np.array_equal(s.fields()["temperature"], ref.fields()["temperature"])


def test_buoyancy_keeps_energy_serial_and_species_concurrent():
    s = _duct(CPU1, buoyancy=[0.0, -0.1, 0.0])
    lanes, why = eligible_lanes(s)
    assert lanes == ["species"] and "buoyancy" in why["energy"]
    assert set(s._bridge.lanes) == {"species"}
    assert any("buoyancy" in n for n in s.lanes_report()["notes"])
    s.step()
    assert "energy" in s.residuals()["scaled"]


def test_short_runs_and_flow_only_cases_stay_serial():
    assert _duct("auto", max_iter=20)._tuner is None
    s = _duct("auto", energy=False, species=[], max_iter=500)
    assert s._tuner is None and s._bridge is None


def test_auto_measures_and_decides(monkeypatch):
    monkeypatch.setattr(bridge_mod, "gpu_available", lambda: False)
    s = _duct("auto", max_iter=200, turbulence="laminar")
    while s._tuner.phase != "done":
        s.step()
    d = s.lanes_report()["tuning"]
    assert "serial" in d["ms_per_iteration"] and "lanes on the CPU" in d["ms_per_iteration"]
    assert d["chosen"] and d["speedup"] >= 1.0
    # whatever was chosen, the run goes on and stays complete
    for _ in range(3):
        s.step()
    f = s.fields()
    assert np.all(np.isfinite(f["temperature"])) and np.all(np.isfinite(f["dye"]))


def _gpu():
    try:
        import cupy  # noqa: F401

        if getattr(cupy, "__fake__", False):
            return False
        return bridge_mod.gpu_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="no usable CUDA GPU / CuPy")
@pytest.mark.parametrize("backend,lanes", [("numpy", {"energy": "gpu", "species": "gpu"}),
                                           ("numpy", {"energy": "gpu", "species": "cpu"}),
                                           ("cupy", {"energy": "gpu", "species": "gpu"}),
                                           ("cupy", {"energy": "cpu", "species": "gpu"})])
def test_cross_device_lanes_converge_to_the_serial_solution(backend, lanes):
    kw = dict(turbulence="laminar", max_iter=3000, tol=1e-7, n=8, backend=backend)
    fa = _run(_duct("off", **kw))
    b = _duct(lanes, **kw)
    fb = _run(b)
    assert b.converged
    for k in FIELDS:
        scale = max(np.abs(fa[k]).max(), 1.0)
        assert np.abs(fa[k] - fb[k]).max() < 1e-4 * scale, k
