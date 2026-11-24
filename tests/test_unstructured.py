import numpy as np
import pytest
from scipy.interpolate import griddata

from flowtrace.hardware.backend import get_backend
from flowtrace.mesh.unstructured import cartesian_mesh_2d, extrude_to_3d
from flowtrace.unstructured.case import BoundarySpec, UnstructuredCase, default_boundaries
from flowtrace.unstructured.simple import SimpleSolver
from flowtrace.validation.reference import compare_ghia

be = get_backend("numpy")


def run(s):
    while not s.finished:
        s.step()
    return s


def test_skew_symmetric_momentum_does_no_internal_work():
    mesh = cartesian_mesh_2d(5, 4, triangulate=True, jitter=0.15)
    solver = SimpleSolver(mesh, UnstructuredCase(scheme="skew-symmetric"), be)
    fv = solver.fv
    rng = np.random.default_rng(42)
    flux = rng.normal(size=fv.n_if)
    boundary_flux = np.zeros(fv.n_bf)
    assert np.linalg.norm(fv.net_outflow(flux, boundary_flux)) > 0.1
    diag, a_pn, a_np, _ = solver._coefficients(
        0.0, solver.m_dirvel, (flux, boundary_flux), skew_symmetric=True
    )
    operator = fv.matrix(diag, a_pn, a_np)
    velocity = rng.normal(size=(solver.nc, solver.dim))
    np.testing.assert_allclose(operator.toarray(), -operator.toarray().T, atol=1e-14)
    assert abs(np.sum(velocity * (operator @ velocity))) < 1e-13

    smooth_velocity = solver.cell_centres()
    linear_gradients = [np.broadcast_to(row, smooth_velocity.shape) for row in np.eye(solver.dim)]
    np.testing.assert_allclose(solver._skew_upwind_diffusion(smooth_velocity, linear_gradients, flux), 0.0)

    unresolved_gradients = [np.zeros_like(velocity) for _ in range(solver.dim)]
    diffusion = solver._skew_upwind_diffusion(velocity, unresolved_gradients, flux)
    diag, a_pn, a_np, _ = solver._coefficients(
        0.0, solver.m_dirvel, (flux, boundary_flux), skew_symmetric=True, upwind_diffusion=diffusion
    )
    dissipative_operator = fv.matrix(diag, a_pn, a_np)
    actual = np.sum(velocity * (dissipative_operator @ velocity))
    expected = np.sum(diffusion * np.sum((velocity[fv.N] - velocity[fv.P])**2, axis=1))
    assert actual == pytest.approx(expected)
    assert actual > 0.0


@pytest.mark.parametrize("time_scheme", ["euler", "bdf2"])
def test_transient_momentum_forcing_from_rest_in_3d(time_scheme):
    ny, nz = 8, 4
    mesh = extrude_to_3d(cartesian_mesh_2d(3, ny), nz, 1.0)
    rho, nu, dt = 2.5, 0.04, 0.025
    case = UnstructuredCase(
        scheme="skew-symmetric", density=rho, viscosity=rho * nu,
        periodic=[["left", "right"], ["bottom", "top"]],
        unsteady=True, time_scheme=time_scheme, dt=dt, t_end=4 * dt,
        inner_iterations=35, inner_tol=1e-10, pressure_tol=1e-7, lanes="off",
    )
    solver = SimpleSolver(mesh, case, be)
    np.testing.assert_array_equal(solver.U, 0.0)
    sampled_times = []

    def force(centres, time):
        sampled_times.append(time)
        source = np.zeros_like(centres)
        mode = np.sin(2 * np.pi * centres[:, 1]) * np.cos(2 * np.pi * centres[:, 2])
        source[:, 0] = rho * time * (mode + 0.25)
        return source

    solver.set_sources(momentum=force)
    run(solver)
    times = dt * np.arange(1, 5)
    np.testing.assert_allclose(sampled_times, times)
    np.testing.assert_allclose(solver.mom_source.sum(axis=0), 0.0, atol=1e-14)

    # Periodic in y and zero wall-normal derivative at the z symmetry planes.
    # The cell-centred Fourier mode has this exact finite-volume decay rate.
    decay = 4 * nu * (ny**2 * np.sin(np.pi / ny)**2 + nz**2 * np.sin(np.pi / nz)**2)
    previous = older = 0.0
    for step, time in enumerate(times):
        if step and time_scheme == "bdf2":
            amplitude = (dt * time + 2 * previous - 0.5 * older) / (1.5 + dt * decay)
        else:
            amplitude = (dt * time + previous) / (1 + dt * decay)
        previous, older = amplitude, previous
    _, y, z = solver.cell_centres().T
    np.testing.assert_allclose(solver.U[:, 0], amplitude * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * z),
                               rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(solver.U[:, 1:], 0.0, atol=1e-11)
    np.testing.assert_allclose(solver.fv.net_outflow(solver.F, solver.Fb), 0.0, atol=1e-11)


def test_turbulent_pressure_is_absorbed_before_the_momentum_predictor():
    mesh = cartesian_mesh_2d(6, 4)
    case = UnstructuredCase(turbulence="k-omega-sst", density=2.0, max_iter=1)
    solver = SimpleSolver(mesh, case, be)
    solver.turb.k = 1.0 + solver.cell_centres()[:, 0]
    solver.p = -(2.0 / 3.0) * case.density * solver.turb.k
    solver.step()
    np.testing.assert_allclose(solver.U, 0.0, atol=1e-12)


def _ghia_err(s):
    xc, f = s.cell_centres(), s.fields()
    y = np.linspace(0.02, 0.98, 49)
    u = griddata(xc[:, :2], f["x-velocity"], np.column_stack([np.full_like(y, 0.5), y]))
    v = griddata(xc[:, :2], f["y-velocity"], np.column_stack([y, np.full_like(y, 0.5)]))
    return compare_ghia(100, y, u, y, v)


def test_simplec_cavity_skewed_triangles():
    m = cartesian_mesh_2d(32, 32, triangulate=True, jitter=0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         max_iter=3000, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    e = _ghia_err(s)
    assert e["u_max_err"] < 0.03 and e["v_max_err"] < 0.03, e


def test_simple_algorithm_converges_too():
    m = cartesian_mesh_2d(24, 24)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0])}, viscosity=0.01,
                         algorithm="SIMPLE", max_iter=4000, tol=1e-5)
    assert c.relax_p == 0.3
    s = run(SimpleSolver(m, c, be))
    assert s.converged


def test_channel_mass_balance_and_heat():
    m = cartesian_mesh_2d(60, 12, 5.0, 1.0, names=("inlet", "outlet", "bottom", "top"),
                          types=("velocity-inlet", "pressure-outlet", "wall", "wall"))
    bnd = default_boundaries(m)
    assert bnd["inlet"].kind == "inlet" and bnd["outlet"].kind == "outlet"
    bnd["inlet"] = BoundarySpec("inlet", normal_speed=1.0, value=0.0)
    bnd["bottom"] = BoundarySpec("wall", thermal="temperature", value=1.0)
    c = UnstructuredCase(boundaries=bnd, viscosity=0.02, energy=True, conductivity=0.02, max_iter=1500, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    assert s.converged
    mb = s.mass_balance()
    assert abs(mb["inlet"] + 1.0) < 1e-9 and abs(sum(mb.values())) < 1e-8
    T = s.fields()["temperature"]
    assert -1e-8 <= T.min() and T.max() <= 1.0 + 1e-6


def test_3d_extruded_cavity_is_two_dimensional():
    m = extrude_to_3d(cartesian_mesh_2d(16, 16), 2, 0.2)
    c = UnstructuredCase(boundaries={"top": BoundarySpec("wall", velocity=[1.0, 0.0, 0.0])}, viscosity=0.01,
                         max_iter=400, tol=1e-6)
    s = run(SimpleSolver(m, c, be))
    f = s.fields()
    assert np.abs(f["z-velocity"]).max() < 1e-8
    assert f["x-velocity"].min() < -0.15
    r = s.residuals()["scaled"]
    assert r["z-velocity"] < 1e-3
