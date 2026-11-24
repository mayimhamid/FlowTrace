"""Automatic solver options, mesh units, bounded scalar transport and thermal checks."""

import os

import numpy as np
import pytest

from flowtrace.mesh.fluent_h5 import read_any_mesh
from flowtrace.mesh.unstructured import agglomerate_cells, cartesian_mesh_2d, compute_geometry, scale_mesh
from flowtrace.unstructured.autoconfig import auto_configure, guess_units
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.validation.checks import validate_unstructured

ELBOW = os.path.join(os.path.dirname(__file__), "..", "examples", "elbow-mix.msh.txt")


def _elbow_case(mesh):
    b = default_boundaries(mesh)
    b["inf1"] = BoundarySpec("inlet", normal_speed=0.4, thermal="temperature", value=293.0)
    b["inf2"] = BoundarySpec("inlet", normal_speed=1.2, thermal="temperature", value=313.0)
    return UnstructuredCase(boundaries=b, density=998.2, viscosity=1.003e-3, conductivity=0.6, cp=4182,
                            backend="numpy", tol=1e-5)


def test_scale_mesh_scales_lengths_and_leaves_the_original():
    m = cartesian_mesh_2d(4, 4, 2.0, 1.0)
    s = scale_mesh(m, 1e-3)
    assert np.allclose(compute_geometry(s).cell_volume.sum(), 2e-6) and m.nodes.max() == 2.0
    with pytest.raises(ValueError):
        scale_mesh(m, 0.0)


def test_guess_units_reads_a_400_unit_elbow_as_millimetres():
    mesh, _ = read_any_mesh(ELBOW)
    assert guess_units(mesh)[0] == "mm" and guess_units(cartesian_mesh_2d(4, 4))[0] == "m"


def test_auto_configure_picks_regime_energy_and_bounded_scalars():
    mesh, _ = read_any_mesh(ELBOW)
    case = _elbow_case(mesh)
    guessed, notes, _ = auto_configure(mesh, _elbow_case(mesh))  # units="auto": the guess is applied and reported
    assert guessed.nodes.max() == pytest.approx(0.2) and any("mesh unit = 'mm'" in n for n in notes)
    same, _, _ = auto_configure(mesh, _elbow_case(mesh), units="m")
    assert same.nodes.max() == 200.0
    scaled, notes, applied = auto_configure(mesh, case, units="mm")
    assert scaled.nodes.max() == pytest.approx(0.2) and mesh.nodes.max() == 200.0
    assert case.turbulence == "k-epsilon-realizable" and case.energy and case.scalar_scheme == "upwind"
    assert case.init_T == pytest.approx(303.0) and any("Re" in n for n in notes) and "energy" in applied
    # a slow, viscous flow is laminar and, without temperatures, has no energy equation
    slow = {"inf1": BoundarySpec("inlet", normal_speed=0.01), "inf2": BoundarySpec("inlet", normal_speed=0.01),
            "out": BoundarySpec("outlet")}
    lam = UnstructuredCase(boundaries=slow, density=1000.0, viscosity=0.1)
    auto_configure(mesh, lam, units="mm")
    assert lam.turbulence == "laminar" and not lam.energy
    with pytest.raises(ValueError):
        auto_configure(mesh, lam, units="furlong")


def test_auto_configured_elbow_converges_bounded_and_conserves_energy():
    mesh, _ = read_any_mesh(ELBOW)
    case = _elbow_case(mesh)
    mesh, _, _ = auto_configure(mesh, case, units="mm")
    s = SimpleSolver(mesh, case)
    while not s.finished:
        s.step()
    assert s.converged
    T = s.fields()["temperature"]
    assert 293.0 - 1e-2 <= T.min() and T.max() <= 313.0 + 1e-2
    checks = {c["name"].split(" (")[0]: c for c in validate_unstructured(s)["checks"]}
    assert all(c["pass"] for c in checks.values()) and "Temperature within the boundary values" in checks
    mb = s.mass_balance()  # the inlet mass flows of the Fluent mixing-elbow tutorial
    assert mb["inf1"] == pytest.approx(-39.928, abs=1e-3) and mb["out"] == pytest.approx(69.874, abs=1e-3)


def test_polygonal_2d_cells_run_like_quads():
    """2x2 blocks of quads merged into polygons (collinear 2-node faces): the lid-driven cavity converges."""
    m = cartesian_mesh_2d(24, 24)
    c = np.arange(m.n_cells)
    poly = agglomerate_cells(m, (c // 24 // 2) * 100 + (c % 24) // 2)
    case = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, viscosity=0.01,
                            max_iter=1500, tol=1e-8, backend="numpy")
    s = SimpleSolver(poly, case)
    while not s.finished:
        s.step()
    assert poly.n_cells == 144 and s.converged


SQUARE = os.path.join(os.path.dirname(__file__), "data", "SquareCylinder-Coarse.msh")


def test_bluff_body_above_the_shedding_reynolds_number_is_run_in_time():
    """Domain 80 units long, body side 4: not millimetres. At Re = 400 the wake sheds vortices, so the
    case is switched to a transient run; at Re = 20 it stays steady."""
    mesh, _ = read_any_mesh(SQUARE)
    assert guess_units(mesh)[0] == "m"
    case = UnstructuredCase(boundaries=default_boundaries(mesh), density=1.0, viscosity=0.01)
    same, notes, applied = auto_configure(mesh, case)
    assert same.nodes.max() == mesh.nodes.max() and applied["body_zone"] == "BluffWalls"
    assert case.unsteady and case.time_scheme == "bdf2" and case.ref_area == pytest.approx(4.0)
    assert case.dt == pytest.approx(0.025 * 4.0) and case.t_end == pytest.approx(120.0 * 4.0)
    assert case.max_iter >= round(case.t_end / case.dt) * case.inner_iterations
    slow = UnstructuredCase(boundaries=default_boundaries(mesh), density=1.0, viscosity=0.2)
    _, _, applied = auto_configure(mesh, slow)
    assert not slow.unsteady and applied["body_zone"] == "BluffWalls"


def test_body_force_report_finds_the_shedding_frequency():
    from flowtrace.runtime.engine import EventHub, Job

    job = Job({"kind": "unstructured"}, EventHub(), ".")
    case = UnstructuredCase(ref_velocity=2.0, ref_area=0.5)
    t = np.arange(0.0, 40.0, 0.01)
    f = 0.15 * 2.0 / 0.5  # Strouhal 0.15
    log = {"t": t, "cd": 1.8 + 0.02 * np.sin(4 * np.pi * f * t), "cl": 0.4 * np.sin(2 * np.pi * f * t), "zone": "body"}
    rep = job._body_force_report(log, case)
    assert rep["shedding"] and rep["strouhal"] == pytest.approx(0.15, rel=0.02)
    assert rep["cd_mean"] == pytest.approx(1.8, abs=0.005) and rep["cl_amplitude"] == pytest.approx(0.4, rel=0.01)
    steady = job._body_force_report({"t": t, "cd": 1.5 + 0 * t, "cl": 0 * t, "zone": "body"}, case)
    assert not steady["shedding"] and any("did not oscillate" in line for line in steady["lines"])
