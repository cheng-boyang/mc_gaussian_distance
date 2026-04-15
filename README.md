# Flies in the Room

Imagine two flies navigating an empty room. Each fly has an uncertain position —
modelled as a 3-D Gaussian distribution — and needs to continuously estimate its
distance to the other to avoid a collision. The room is discretized into a dense
voxel grid, where each voxel stores the Gaussian parameters of a fly's possible
location. At every moment, the flies must compute `E[||X - Y||_2]` in real time,
fast enough to react before a collision occurs.

This scenario motivates the core engineering challenge of the project: how cheaply
and how quickly can we estimate the expected distance between two uncertain 3-D
positions, at the scale of millions of voxel pairs per second?

The project answers this with a neural-network surrogate sampler deployed on an
analog-input Compute-in-Memory (CIM) array, benchmarked against deterministic
Gauss-Hermite quadrature as the ground truth.

---

## Problem

A 3-D grid of `grid_size³` voxels (default 100³ = 1 M voxels). Each voxel stores a
diagonal 3-D Gaussian `[mu_x, mu_y, mu_z, std_x, std_y, std_z]`.

Each benchmark run randomly selects two voxels and estimates the expected L2 distance
between their distributions:

```
E[||Z||_2]   where   Z = X - Y ~ N(mu1 - mu2, diag(std1² + std2²))
```

---

## Files

| File | Role |
|------|------|
| `flies_in_the_room.py` | Main script (NN training, benchmarking, CIM model, plots) |
| `gaussian_distance_cpu.py` | Dependency — provides Gauss-Hermite quadrature ground truth |
| `results_csv/` | Output directory for CSV results and histogram plots |
| `archive/` | Earlier Part 1–4 scripts (classical estimators, GPU/CIM breakdowns) |

---

## NN Surrogate

- **Architecture:** `Linear(15→h1) → ReLU → Linear(h1→h2) → ReLU → Linear(h2→3)`
- **Input (15D):** `eps(3) + mu1(3) + std1(3) + mu2(3) + std2(3)`
- **Output (3D):** sample from `Z = X - Y` (no final activation)
- **Loss:** MSE between predicted Z sample and exact reparameterization target `(mu1-mu2) + eps * sqrt(std1² + std2²)`
- **Distance estimate:** `mean(||sample||_2)` over N eps draws per run

---

## Ground Truth

Deterministic Gauss-Hermite quadrature (`order³` grid points, default 15³ = 3375).
Used as the reference for accuracy metrics (MAE, relative error).

---

## CIM Analytical Model

Estimates throughput for the NN inference flow on an **analog-input** CIM array with
an **analog RNG** (no DAC; ADC readout only).

Throughput bottleneck: `min(rng_throughput / 3, adc_rate / adc_ops_per_sample)`

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--grid-size` | 100 | Voxels per dimension (total: grid_size³) |
| `--mu-range` | 5.0 | Mu drawn from Uniform[-mu_range, mu_range] |
| `--std-min` | 0.1 | Minimum std per voxel |
| `--std-max` | 2.0 | Maximum std per voxel |
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--train-steps` | 3000 | NN training iterations |
| `--train-batch-size` | 4096 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--n-samples` | 1000 | MC eps draws per run |
| `--k-runs` | 100 | Number of benchmark runs |
| `--warmup-runs` | 2 | Warmup runs before timing |
| `--quadrature-order` | 15 | Gauss-Hermite quadrature order (points = order³) |
| `--seed` | 123 | Random seed |
| `--device` | `auto` | `cpu` / `cuda` / `auto` |
| `--array-rows` | 64 | CIM array row count |
| `--array-cols` | 64 | CIM array column count |
| `--adc-rate` | 1e9 | Aggregate ADC bandwidth (conversions/s) |
| `--rng-throughput` | 5.12e9 | Analog RNG throughput (samples/s) |
| `--rng-efficiency` | 0.36e-12 | Analog RNG energy efficiency (J/sample) |
| `--csv` | `results_csv/flies_in_the_room.csv` | Output CSV path |
| `--plot` | `results_csv/flies_in_the_room.png` | Output histogram plot path |
| `--hist-bins` | 30 | Number of histogram bins |

---

## Example Run

```bash
# Default run (auto-detects GPU)
python flies_in_the_room.py

# Custom grid, more runs, specific CIM parameters
python flies_in_the_room.py --grid-size 50 --k-runs 200 --n-samples 2000 \
    --array-rows 64 --array-cols 64 --adc-rate 1e9

# CPU only, smaller grid
python flies_in_the_room.py --device cpu --grid-size 20 --k-runs 50
```

---

## Outputs

| File | Content |
|------|---------|
| `results_csv/flies_in_the_room.csv` | Per-run NN estimates, ground truth, errors, and summary metrics |
| `results_csv/flies_in_the_room.png` | Two-panel histogram: distance distributions + NN vs ground truth scatter |

---

## Notes

- `gaussian_distance_cpu.py` must remain in the same directory as `flies_in_the_room.py` (local import).
- Quadrature cost scales as `order³`; the default order 15 (3375 points) adds ~10 ms per run on CPU.
- CIM throughput figures are analytical estimates, not hardware-counter measurements.
- GPU inference requires a CUDA-visible PyTorch environment.
