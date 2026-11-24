# Pressure-based unstructured solver (SIMPLE / SIMPLEC)

`flowtrace.unstructured.simple.SimpleSolver` follows the algorithm behind Fluent's default
pressure-based segregated solver. It uses collocated, cell-centred finite volumes on arbitrary
polygons and polyhedra.

## Mesh data

Faces store owner \(P\) and neighbour \(N\) with the area vector \(\mathbf S_f\) pointing from
\(P\) to \(N\) (outwards on boundaries), as in Fluent. Left-handed faces are re-oriented on
import. For each interior face:

* \(\mathbf d = \mathbf x_N - \mathbf x_P\). For periodic pairs, \(\mathbf x_N\) is seen through
  the translation.
* Over-relaxed decomposition \(\mathbf S = \alpha\mathbf d + \mathbf T\) with
  \(\alpha = \mathbf S\cdot\mathbf S/\mathbf S\cdot\mathbf d\).
* Linear interpolation weight from the projection of the face centre on \(\mathbf d\).

## Discretisation

* **Gradients:** least squares (inverse-distance weighted) with boundary face values.
  Zero-gradient boundary values are reconstructed along the face with the lagged cell gradient,
  \(\phi_b = \phi_P + \nabla\phi_P\cdot\mathbf d_{b,t}\), which keeps skewed boundary cells
  consistent.
* **Diffusion:** implicit \(\Gamma_f\alpha(\phi_N-\phi_P)\) plus the explicit non-orthogonal
  correction \(\Gamma_f\,\overline{\nabla\phi}_f\cdot\mathbf T\).
* **Convection:** implicit first-order upwind with a deferred, bounded second-order upwind
  correction are available, as is skew-symmetric momentum convection.
* **Pressure force in momentum:** the standard Green–Gauss gradient on face pressures,
  \((\nabla p)_P = V_P^{-1}\sum_f p_f\mathbf S_f\), with linearly interpolated, skewness-corrected
  \(p_f\).
  With a least-squares cell gradient instead, the manufactured-solution error stalled at O(1) on
  triangles. That option remains available as `pressure_gradient="least-squares"` for
  comparison.

### Forced momentum and skew-symmetric convection

In the constant-density laminar limit, the momentum equation is

\[
\rho\left[\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u\right]
=-\nabla p+\mu\Delta\mathbf u+\mathbf f_{\mathrm{ext}},
\qquad \nabla\cdot\mathbf u=0.
\]

The force per unit volume is \(\rho\mathbf f\) in Equation (1.1) of
[OpenAI's forced finite-time blowup paper](https://openai.com/index/navier-stokes-solution/).
The solver samples prescribed forcing at the new physical time for implicit Euler and BDF2.
For a closed or periodic domain, the callback force is projected to zero volume-weighted mean.

Select `scheme="skew-symmetric"` for centered momentum transport. For each
interior face, the solver starts with \(F_f(\mathbf u_P+\mathbf u_N)/2\) and subtracts half
the cell mass imbalance times the local velocity. This leaves \(F_f\mathbf u_N/2\) in
the owner equation and \(-F_f\mathbf u_P/2\) in the neighbour equation. Their work
cancels pairwise, even when an intermediate flux has a residual mass imbalance.
Periodic faces use the same pairing. Physical boundary faces use upwind transport
with the same half-imbalance subtraction. A bounded upwind term adds dissipation
where the reconstructed gradient under-resolves a velocity jump.

For turbulent flow, the momentum predictor and Rhie–Chow interpolation use
\(p_{\rm eff}=p+2\rho k/3\). Exported pressure is \(p\). The transpose and deviatoric
trace corrections use the effective viscosity on interior faces and open boundaries.
Wall traction uses the selected wall treatment.

## Pressure–velocity coupling

Momentum is solved with under-relaxation \(\alpha_u\) (BiCGSTAB, Jacobi preconditioned) to give
\(\mathbf u^*\). Face mass fluxes use Rhie–Chow interpolation:

\[
F_f = \rho\Big[\overline{\mathbf u^*}_f\cdot\mathbf S_f
  - \overline{d}_f\,\alpha_f\big(p_N - p_P - \overline{\nabla p}_f\cdot\mathbf d\big)\Big],
\qquad d = \frac{V}{a_P - \sum a_{nb}}\ (\text{SIMPLEC}),\ \frac{V}{a_P}\ (\text{SIMPLE}).
\]

Both pressure terms are projected on the same orthogonal direction \(\alpha\mathbf d\). The
pressure correction \(\sum_f \rho\,\overline d_f\alpha_f(p'_P-p'_N) = -\dot m_P\) is solved by
AMG-preconditioned CG. pyamg is used on the CPU, and FlowTrace's smoothed-aggregation AMG,
applied with cuSPARSE, on the GPU. The inner tolerance defaults to 0.1, Fluent's AMG default;
this was measured to leave outer iterations unchanged and save 15–50 % of the wall time.

The transient Rhie-Chow history uses Choi's correction with coefficients
\((c_1,c_2)=(1,0)\) for implicit Euler and \((2,-1/2)\) for BDF2. BDF2 starts
with an Euler step to establish the second time level.

The momentum predictor solves the assembled sparse system once. Equivalently,
the diagonal times the predicted velocity equals the explicit source minus the
pressure force and minus the off-diagonal matrix applied to the predicted velocity.

Deferred non-orthogonal correctors lag the tangential pressure flux. The final
face correction uses the same tangential flux as the final Poisson right-hand
side. Its residual therefore gives the corrected mass imbalance up to roundoff.
The default is two sweeps above 5 degrees of non-orthogonality and zero otherwise.

**Pressure under-relaxation.** SIMPLE defaults to 0.3. SIMPLEC defaults to 1.0
through 5 degrees of non-orthogonality and 0.8 above it. An explicit `relax_p`
takes precedence.

## Stability measurements with orthogonal pressure correction

The pressure-correction equation keeps only the orthogonal part of its Laplacian
(\(\alpha = \mathbf S\cdot\mathbf S/\mathbf S\cdot\mathbf d\)); the non-orthogonal remainder is
neglected, as in Fluent's default and in OpenFOAM without non-orthogonal correctors. A
controlled sequence shows what that costs: lid-driven cavity, Re = 10, 24 × 24 quads sheared
uniformly to the angle θ (so every face has non-orthogonality θ), 4,000 iterations at most.

| θ | min. orthogonal quality | α_p = 1 | 0.8 | 0.5 | 0.33 | 0.2 | automatic + guard | SIMPLE (α_p 0.3) |
|---|---|---|---|---|---|---|---|---|
| 0°–20° | ≥ 0.94 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 30° | 0.87 | ✗ | ✗ (late) | ✓ | ✓ | ✓ | ✓ (1 rollback, 0.48) | ✗ |
| 40° | 0.77 | ✗ | ✗ | ✓ (slow) | ✓ | ✓ | ✓ (1, 0.48) | ✗ |
| 45° | 0.71 | ✗ | ✗ | ✗ (late) | ✓ | ✓ | ✓ (2, 0.29) | ✗ |
| 50° | 0.64 | ✗ | ✗ | ✗ | ✓ (slow) | ✓ | ✓ (2, 0.29) | ✗ |
| 55° | 0.57 | ✗ | ✗ | ✗ | stalls | ✓ | ✓ (3, 0.17) | ✗ |
| 60° | 0.50 | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (3, 0.17) | ✗ |

The largest stable α_p follows \(\cos^2\theta\) at every angle (0.75, 0.59, 0.50, 0.41, 0.33,
0.25): the neglected part grows like \(\tan^2\theta\) relative to the retained one. Plain SIMPLE
diverges as well, so the cause is the neglected non-orthogonal coupling, not SIMPLEC's
consistent coefficients. Meshes whose skew alternates in sign from face to face (warped quads,
triangles) tolerate far more: warped triangles at 43° converge at 0.8.

The stability guard (automatic `relax_p` only) recovered every case in the table. It stops at
`relax_p` = 0.1, which by the cos² law covers uniform skew up to about 70°. Beyond that, or with
an explicit `relax_p`, a diverging run now ends with an error naming the mesh non-orthogonality,
the guard's state and the cos² estimate. These measurements used zero non-orthogonal corrector sweeps.

## Residuals and convergence

Scaled residuals as in Fluent: momentum and energy use
\(\sum|r| / \sum|a_P\phi_P|\); continuity uses \(\sum|\dot m|\) divided by its largest value in
the first five iterations. For flows that start without any imbalance (periodic channels, warm
starts), that denominator is floored at \(10^{-3}\sum|F|\), so convergence is not judged against
round-off. `residuals()["raw"]` reports the absolute values.

Where they are evaluated: the momentum and energy residuals belong to the assembled,
*unrelaxed* system at the field the iteration started from (before its linear solve); the
continuity residual is the imbalance of the Rhie–Chow *predictor* fluxes, before the pressure
correction. When every velocity component is exactly zero in every cell (a cavity started from
rest, first iteration) \(\sum|a_P\phi_P| = 0\) and the momentum value is the numerator divided by
a \(10^{-300}\) floor, a very large number (\(\sim 10^{302}\)); convergence is only tested from the
sixth iteration on. The [solver diagnostics](diagnostics.md) separate the pre-solve, solved and
corrected states and are defined in that case.

## Energy

\(\nabla\cdot(F T) = \nabla\cdot\big((k/c_p + \mu_t/Pr_t)\nabla T\big) + q'''/c_p\), with fixed
temperature, heat flux or adiabatic walls, inlet temperatures and outlet backflow values.
`heat_flow()` reports conduction plus enthalpy flux per boundary zone; at steady state the values
sum to zero, which is checked by the benchmarks.

## Verification

Manufactured solution on smoothly warped quads and triangles (`mms_simplec_quads`,
`mms_simplec_tri`): observed order about 2 for velocity and temperature; pressure 2.0 on quads and
between first and second order on triangles: 1.2 for the finest grid pair of the full tier,
1.4 on the coarser grids of the quick tier.
Global mass and energy balances hold to ~1e-12.
