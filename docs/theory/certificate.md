# Mesh certificate

**Converged is not the same as accurate.** A residual of 10⁻⁵ says the discrete equations are
solved on *this* mesh. It says nothing about how far the pressure drop or the drag is from the
value a fine enough mesh would give. The remedy is a mesh-independence study: refine, solve
again, compare. It is manual work, and it is the step most often skipped.

The mesh certificate does that study as part of the run and puts the outcome next to every
number:

```text
pressure drop inf2 -> outlet: 136.6 +/- 3.4 (2.5 %; asymptotic, observed order 1.94)
pressure drop inf1 -> outlet: 73.767 +/- 33 (44 %; slow, observed order 0.15)
```

```bash
flowtrace solve examples/elbow-mix.msh.txt --auto --units mm --certify 3 \
    --bc inf1=inlet:0.4 --bc inf2=inlet:1.2 --rho 998.2 --mu 1.003e-3
```

In the dashboard: *Fluent mesh › Mesh certificate*. From Python:
`flowtrace.unstructured.certify.certify(solver, levels=3)`.

## What it does

1. The converged run is the coarse level.
2. The mesh is refined uniformly (`flowtrace.mesh.refine.refine_2d`): every edge is split at
   its midpoint and every cell with *k* corners becomes *k* quadrilaterals. The refined mesh
   covers exactly the same polygonal domain with the same zones, so two levels differ by
   discretisation error only.
3. The refined case starts from the coarse solution and is iterated to the same tolerance.
   With `--certify 3` this is done twice (4× and 16× the cells of a quad mesh).
4. Each reported quantity gets a band from the change between levels (Roache's Grid
   Convergence Index):

   | Levels | Sequence | Order used | Safety factor | Status |
   |---|---|---|---|---|
   | 2 | – | formal (2, or 1 for first-order upwind) | 3 | `two-level` |
   | 3 | monotone, observed order *p* in [0.5, formal + 0.5] | min(*p*, formal) | 1.25 | `asymptotic` |
   | 3 | monotone, *p* < 0.5 | 0.5 | 3 | `slow` |
   | 3 | monotone, *p* > formal + 0.5 | formal | 3 | `irregular` |
   | 3 | changes sign | – | 3 × largest change | `oscillating` |
   | 3 | changes grow | – | 3 × largest change | `not converging` |

   The band is `Fs |f_fine − f_coarse| / (rᵖ − 1)` with `r = sqrt(N_fine / N_coarse)`.
5. The difference between the coarse velocity field and the restricted fine one is kept per
   cell as the field `mesh-error-estimate` (fraction of the largest speed): a map of where the
   mesh is too coarse.

The quantities are the area-averaged pressure drop from each inlet to the outlets, the force
on each wall zone, the volume-averaged kinetic energy and, with the energy equation, the mean
temperature and the heat flow through each fixed-temperature wall. The run then continues with
the finest level: the reports, the exported Fluent package and the picture are the refined
solution.

## What it does not do

* **It is an estimate, not a bound.** `tests/test_certify.py` checks on the lid-driven cavity
  that the band of the kinetic energy from 12² → 24² → 48² contains the 96² value, and that
  the lid force (which grows without bound at the lid corners) is *not* given a small band.
* **Singular quantities stay singular.** A uniform inlet profile that meets a no-slip wall, or
  a moving lid that meets a fixed wall, makes the pressure and the shear unbounded at the
  corner. The certificate reports such a quantity as `slow`, `oscillating` or `not converging`
  with a wide band; that is the correct answer, and no mesh will change it.
* **Wall functions.** Refining lowers y⁺. When the finest level's y⁺ falls below the log layer
  the certificate says so: the band then includes the wall model's sensitivity to the
  first-cell height.
* 2-D steady cases without periodic or solid zones. Other runs are skipped with the reason in
  the log. Refinement keeps the polygonal boundary of the input mesh; it does not re-project
  new nodes onto a curved CAD surface.
