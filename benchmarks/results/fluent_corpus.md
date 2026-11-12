# Fluent mesh corpus

Read + quality + 5 SIMPLEC iterations with default boundary conditions (`benchmarks/fluent_mesh_corpus.py`).

| file | format | size_mb | read_s | dim | cells | faces | zones | min_oq | repaired | solve5_s | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cavity_quad.msh | fluent-msh | 0.0762 | 0.000967 | 2 | 1600 | 3280 | 4 | 1 | 0 | 0.407 | ok |
| elbow-mix.msh.txt | fluent-msh | 0.0538 | 0.00379 | 2 | 648 | 1374 | 4 | 0.673 | 0 | 0.0445 | ok |
| heated_duct_tri.msh | fluent-msh | 0.0965 | 0.00327 | 2 | 2400 | 3680 | 4 | 0.581 | 0 | 0.0483 | ok |
| SquareCylinder-Coarse.msh | fluent-msh | 0.479 | 0.0247 | 2 | 5824 | 11838 | 4 | 0.711 | 0 | 0.156 | ok |
| SquareCylinder-Fine.msh | fluent-msh | 1.4 | 0.0405 | 2 | 16464 | 33278 | 4 | 0.705 | 0 | 0.532 | ok |
| SquareCylinder-Finest.msh | fluent-msh | 1.97 | 0.0399 | 2 | 22984 | 46398 | 4 | 0.703 | 0 | 0.847 | ok |
| SquareCylinder-Medium.msh | fluent-msh | 0.919 | 0.0364 | 2 | 10904 | 22078 | 4 | 0.709 | 0 | 0.372 | ok |
| FFF-7.cas.h5 | cff-hdf5 | 0.89 | 0.0834 | 2 | 12508 | 25300 | 4 | 0.478 | 0 | 0.245 | ok |
| rocket_geom.msh.h5 | cff-hdf5 | 140 | 2.21 | 3 | 3189603 | 10880486 | 11 | 0.193 | 0 |  | ok |
| elbow-mix.msh.txt | fluent-msh | 0.0538 | 0.0056 | 2 | 648 | 1374 | 4 | 0.673 | 0 | 0.0284 | ok |
