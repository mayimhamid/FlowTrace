Heated turbulent duct (k-omega SST + energy + 1 species), steady, FP64, to tol 1e-5. AMD Ryzen 9 7940HS w/ Radeon 780M Graphics, NVIDIA GeForce RTX 4060 Laptop GPU. 2026-11-12.

| flow on | cells | serial: its / s | bridge: its / s | speed-up | max field diff | placement chosen |
|---|---:|---:|---:|---:|---:|---|
| CPU | 4,096 | 180 / 4.54 | 180 / 3.23 | 1.41x | 1.2e-11 | lanes on the CPU |
| CPU | 16,384 | 380 / 50.4 | 380 / 15.46 | 3.26x | 2.7e-06 | lanes on the CPU @ 4 CPU threads |
| CPU | 65,536 | 841 / 221.3 | 841 / 124.62 | 1.78x | 6.7e-07 | energy on the CPU, species on the GPU |
| GPU | 4,096 | 180 / 13.81 | 180 / 7.57 | 1.82x | 5.6e-08 | lanes on the CPU |
| GPU | 16,384 | 380 / 36.61 | 380 / 19.02 | 1.92x | 2.3e-06 | lanes on the CPU @ 8 CPU threads |
| GPU | 65,536 | 841 / 72.23 | 841 / 74.64 | 0.97x | 3.1e-05 | serial |

Field difference: max |serial - bridge| / range of the serial field (or 1e-3 x its magnitude when the range collapses), over x-velocity, pressure, temperature, dye and k. Bridge wall time includes the tuning iterations.

Reproduce: `python benchmarks/device_bridge.py --sizes 32,64,128 --backends numpy,cupy`
