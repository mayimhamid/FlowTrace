"""Choked converging-diverging nozzle: pressure inlet, axisymmetric implicit solve, Fluent case settings."""

import math

import numpy as np
import pytest

from flowtrace.compressible.case import CompBC, CompressibleCase
from flowtrace.compressible.checks import physical_checks
from flowtrace.compressible.solver import CompressibleSolver
from flowtrace.fluent.case_settings import choose_solver, compressible_spec, parse_settings
from flowtrace.mesh.unstructured import cartesian_mesh_2d

G, R, P0, T0 = 1.4, 287.0, 5e5, 3000.0
R_IN, R_T, R_EX = 0.025, 0.010, 0.016


def _nozzle_mesh(nx=72, ny=12):
    """A rectangle mapped onto a smooth nozzle: radius R(x) from inlet through a throat to the exit."""
    m = cartesian_mesh_2d(nx, ny, 1.0, 1.0, names=("inlet", "outlet", "axis", "wall"))
    x = m.nodes[:, 0]
    conv = R_T + (R_IN - R_T) * 0.5 * (1.0 + np.cos(np.pi * np.clip(x / 0.6, 0.0, 1.0)))
    div = R_T + (R_EX - R_T) * np.sin(0.5 * np.pi * np.clip((x - 0.6) / 0.4, 0.0, 1.0)) ** 2
    m.nodes[:, 1] *= np.where(x < 0.6, conv, div)
    m.nodes[:, 0] = 0.1 * x
    return m


def _mach_from_area(ratio, supersonic):
    f = lambda M: (1.0 / M) * ((2.0 / (G + 1.0)) * (1.0 + 0.5 * (G - 1.0) * M * M)) ** ((G + 1.0) / (2.0 * (G - 1.0))) - ratio  # noqa: E731
    lo, hi = (1.0, 10.0) if supersonic else (1e-6, 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (f(mid) > 0) == (f(lo) > 0) else (lo, mid)
    return 0.5 * (lo + hi)


def _run(axisymmetric, order=1, max_iter=2500):
    case = CompressibleCase(boundaries={"inlet": CompBC("pressure-inlet", p0=P0, T0=T0, p=0.95 * P0),
                                        "outlet": CompBC("pressure-outlet", p=5000.0), "axis": CompBC("symmetry"),
                                        "wall": CompBC("slip-wall")},
                            axisymmetric=axisymmetric, order=order, init_p=0.95 * P0, init_T=T0, max_iter=max_iter,
                            tol=1e-6, cfl_max=50.0, backend="numpy")
    s = CompressibleSolver(_nozzle_mesh(), case)
    while not s.finished:
        s.step()
    return s


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_choked_nozzle_mass_flow_and_exit_mach_match_the_isentropic_solution(axisymmetric):
    """Before the characteristic pressure inlet the inlet mass flow cycled between two values and never
    met the outlet's; before the axisymmetric terms of the implicit operator the axisymmetric run diverged."""
    s = _run(axisymmetric)
    mb = s.mass_balance()
    assert s.cfl == pytest.approx(50.0) and max(s.residuals()["scaled"].values()) < 1e-3
    assert abs(mb["inlet"] + mb["outlet"]) < 2e-3 * mb["outlet"]
    flux = P0 / math.sqrt(T0) * math.sqrt(G / R) * (2.0 / (G + 1.0)) ** ((G + 1.0) / (2.0 * (G - 1.0)))  # per throat area
    area = (lambda r: 0.5 * r * r) if axisymmetric else (lambda r: r)  # per radian / per unit depth
    assert mb["outlet"] == pytest.approx(flux * area(R_T), rel=0.03)
    f = s.fields()
    cc = s.cell_centres()
    exit_mach = f["mach-number"][cc[:, 0] > 0.098]
    assert exit_mach.mean() == pytest.approx(_mach_from_area(area(R_EX) / area(R_T), True), rel=0.06)
    inlet_mach = f["mach-number"][cc[:, 0] < 0.002]
    assert inlet_mach.mean() == pytest.approx(_mach_from_area(area(R_IN) / area(R_T), False), rel=0.06)
    checks = {c["name"]: c for c in physical_checks(s)}
    assert all(c["pass"] for c in checks.values()), checks
    assert f["temperature"].max() <= T0 * 1.01  # nothing hotter than the reservoir


THREADS = '''(39 (2 fluid nozzle 1)(
 (material . air)
))
(39 (6 pressure-inlet inlet 1)(
 (p0 (constant . 400000) (profile "" ""))
 (t0 (constant . 3000) (profile "" ""))
 (p (constant . 360000) (profile "" ""))
 (direction-spec . 1)
))
(39 (1 interior int_nozzle 1)(
))
(39 (7 pressure-outlet outlet 1)(
 (p (constant . -96325) (profile "" ""))
 (t0 (constant . 300) (profile "" ""))
))
(39 (8 wall wall 1)(
))
(39 (5 axis axis 1)(
))'''
RAMPANT = ('(rp-seg? #t)(rp-kw? #t)(kw-sst-on? #t)(rp-inviscid? #f)(operating-pressure 101325)'
           '(materials ((air fluid (chemical-formula . #f) (density (ideal-gas . #f) (constant . 1.225)) '
           '(specific-heat (constant . 1006.43) (polynomial 1 2)) (thermal-conductivity (constant . 0.0242)) '
           '(viscosity (constant . 1.7894e-05) (sutherland 1 2 3)) (molecular-weight (constant . 28.966)))'
           ' (aluminum solid (density (constant . 2719)))))')


def test_fluent_case_settings_choose_the_compressible_solver_with_the_case_values():
    st = parse_settings(RAMPANT, THREADS)
    assert st["material"]["density_law"] == "ideal-gas" and st["models"]["turbulence"] == "k-omega-sst"
    assert st["zones"]["inlet"]["p0"] == pytest.approx(501325.0)  # gauge + operating pressure
    assert st["zones"]["outlet"]["p"] == pytest.approx(5000.0)
    mesh = cartesian_mesh_2d(4, 4, names=("inlet", "outlet", "axis", "wall"),
                             types=("pressure-inlet", "pressure-outlet", "axis", "wall"))
    assert choose_solver(mesh, st)[0] == "compressible"
    assert choose_solver(mesh, None)[0] == "incompressible"  # no settings, no far field: constant density
    spec, notes = compressible_spec(mesh, st)
    assert spec["axisymmetric"] and spec["turbulence"] == "k-omega-sst"
    assert spec["gas"]["gamma"] == pytest.approx(1.399, abs=1e-3) and spec["gas"]["R"] == pytest.approx(287.04, abs=0.01)
    assert spec["boundaries"]["inlet"] == {"kind": "pressure-inlet", "p0": pytest.approx(501325.0), "T0": 3000.0,
                                           "p": pytest.approx(461325.0)}
    assert spec["boundaries"]["outlet"]["p"] == pytest.approx(5000.0) and spec["boundaries"]["axis"]["kind"] == "symmetry"
    constant = parse_settings(RAMPANT.replace("(ideal-gas . #f) (constant . 1.225)", "(constant . 1.225) (ideal-gas . #f)"), THREADS)
    assert choose_solver(mesh, constant)[0] == "incompressible"
