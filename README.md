# MC Gaussian Distance

This project studies the expected L2 (Euclidean) distance between two independent 3-D Gaussian distributions,

- `X ~ N(mu1, Sigma1)`
- `Y ~ N(mu2, Sigma2)`

using the reduction `Z = X - Y ~ N(mu1 - mu2, Sigma1 + Sigma2)`, then estimating `E[||Z||_2]`.

## Overview

The project is organized into **five parts**:

| Part | Topic |
|------|-------|
| 1 | Classical estimators (CPU quadrature, CPU/GPU Monte Carlo) |
| 2 | Neural-network surrogate sampler (diagonal covariance) |
| 3 | NN inference benchmarks (CPU vs GPU throughput, perturbed-parameter latency) |
| 4 | Analytical hardware breakdown models (GPU, CIM digital-input, CIM analog-input) |
| 5 | End-to-end scenario: *flies_in_the_room* (voxel map + NN + CIM) |

---

## Part 1 — Classical Estimators

### `gaussian_distance_cpu.py`

Core CPU library. Provides:
- Parameter validation and reduced-distribution construction
- **Deterministic** Gauss-Hermite quadrature (`deterministic_expected_norm_cpu`) — exact up to quadrature order; evaluates `m^d` grid points (does not scale to high dimensions)
- **Monte Carlo** estimation (`monte_carlo_expected_norm_cpu`)
- Analytical operation-count helpers (`OperationCounts`) used by the hardware scripts
- Supports full covariance matrices

No CLI. Imported by other scripts.

---

### `gaussian_distance_gpu.py`

GPU Monte Carlo implementation using PyTorch CUDA. Sampling and norm computation stay on-device until the final scalar reduction. Supports full covariance matrices.

No CLI. Imported by other scripts.

---

### `benchmark.py`

Runs and compares CPU deterministic, CPU Monte Carlo, and GPU Monte Carlo across fixed test cases. Prints a summary table.

No CLI arguments.

```bash
python benchmark.py
```

---

### `test_example.py`

Lightweight unit tests and runnable examples. Covers deterministic-vs-MC consistency checks and edge cases.

No CLI arguments.

```bash
python test_example.py
# Run a single test class
python -m unittest test_example.GaussianDistanceTests.test_deterministic_matches_point_mass_exactly
```

---

### `diagnose_cuda.py`

Checks CUDA visibility for the current Python/PyTorch environment and prints device info.

No CLI arguments.

```bash
python diagnose_cuda.py
```

---

## Part 2 — Neural-Network Surrogate

### `nn_gaussian_distance.py`

Implements `TwoLayerGaussianSampler`, a two-layer ReLU network that learns to draw samples from the reduced difference distribution `Z = X - Y` in the diagonal-covariance case.

- **Input (21D):** `eps(3) + mu1(3) + std1(3) + mu2(3) + std2(3)`
- **Output (3D):** sample from Z (no final activation); take `||sample||_2` for the distance contribution
- **Architecture:** `Linear(21→h1) → ReLU → Linear(h1→h2) → ReLU → Linear(h2→3)`
- Trained with a hand-rolled Adam loop against the exact reparameterization target `Z = (mu1-mu2) + eps * sqrt(std1^2 + std2^2)`
- Monte Carlo distance estimate: mean of `||sample||_2` over N eps draws

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--train-steps` | 3000 | Adam training iterations |
| `--batch-size` | 4096 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 42 | Random seed |
| `--device` | auto | `cpu` / `cuda` / `auto` |
| `--num-samples` | 100000 | MC samples for inference test |

```bash
python nn_gaussian_distance.py --device cpu
python nn_gaussian_distance.py --device cuda --hidden1 128 --hidden2 128
```

---

## Part 3 — NN Inference Benchmarks

### `benchmark_nn_inference.py`

Trains the NN surrogate once on CPU, then benchmarks Monte Carlo distance estimation throughput on both CPU and GPU across multiple sample sizes. Results written to CSV.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--train-steps` | 3000 | Training iterations |
| `--train-batch-size` | 4096 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 123 | Random seed |
| `--mu1` | `0,0,0` | Mean of distribution 1 |
| `--std1` | `1,1,1` | Std dev of distribution 1 |
| `--mu2` | `1,2,3` | Mean of distribution 2 |
| `--std2` | `1,1,1` | Std dev of distribution 2 |
| `--sample-sizes` | `10000,100000,1000000` | Comma-separated sample counts to benchmark |
| `--cpu-batch-size` | 50000 | Batch size for CPU inference |
| `--gpu-batch-size` | 250000 | Batch size for GPU inference |
| `--warmup-samples` | 50000 | GPU warmup samples before timing |
| `--csv` | `results_csv/nn_inference_benchmark.csv` | Output CSV path |

```bash
python benchmark_nn_inference.py
python benchmark_nn_inference.py --sample-sizes 100000,1000000 --gpu-batch-size 500000
```

---

### `benchmark_nn_perturbed_params.py`

Benchmarks the extended flow where NN input parameters are themselves sampled from Gaussian perturbation models before each inference batch. Measures per-batch GPU latency across K independent parameter batches.

Supports saving and loading trained model checkpoints to skip re-training.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--train-steps` | 3000 | Training iterations |
| `--train-batch-size` | 4096 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 123 | Random seed |
| `--device` | `cuda` | Inference device |
| `--k-batches` | 20 | Number of parameter batches to benchmark |
| `--param-batch-size` | 50000 | Samples per parameter batch |
| `--warmup-batches` | 2 | Warmup batches before timing |
| `--mu1` | `0,0,0` | Center mean of distribution 1 |
| `--std1` | `1,1,1` | Center std of distribution 1 |
| `--mu2` | `1,2,3` | Center mean of distribution 2 |
| `--std2` | `1,1,1` | Center std of distribution 2 |
| `--mu1-perturb-std` | `0.1,0.1,0.1` | Perturbation std for mu1 |
| `--std1-perturb-std` | `0.05,0.05,0.05` | Perturbation std for std1 |
| `--mu2-perturb-std` | `0.1,0.1,0.1` | Perturbation std for mu2 |
| `--std2-perturb-std` | `0.05,0.05,0.05` | Perturbation std for std2 |
| `--save-model` | _(none)_ | Path to save trained model checkpoint (.pt) |
| `--load-model` | _(none)_ | Path to load checkpoint, skipping training |
| `--csv` | `results_csv/nn_perturbed_params_latency.csv` | Output CSV path |

```bash
# Train, save checkpoint, and benchmark
python benchmark_nn_perturbed_params.py --device cuda --k-batches 20 --save-model model.pt

# Load checkpoint and benchmark (skips training)
python benchmark_nn_perturbed_params.py --device cuda --k-batches 100 --load-model model.pt
```

---

## Part 4 — Analytical Hardware Breakdown Models

These scripts compute theoretical operation counts and throughput estimates without running real hardware. All output CSVs to `results_csv/`.

### `profile_breakdown.py`

Generates stage-by-stage operation breakdown tables for the classical (non-NN) estimators using the `OperationCounts` helpers in `gaussian_distance_cpu.py`.

No CLI arguments.

```bash
python profile_breakdown.py
```

---

### `hardware_estimate.py`

Converts the analytical operation/byte model into rough throughput and energy estimates for CPU and GPU hardware targets.

No CLI arguments.

```bash
python hardware_estimate.py
```

---

### `nn_gpu_inference_breakdown.py`

Analytical breakdown of NN surrogate Monte Carlo inference on a GPU, counting multiplies, adds, ReLUs, reads, and writes per stage: RNG → NN surrogate → Euclidean distance.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--num-samples` | 1000000 | Total Monte Carlo samples |
| `--csv` | `results_csv/nn_gpu_inference_breakdown.csv` | Output CSV path |

```bash
python nn_gpu_inference_breakdown.py
python nn_gpu_inference_breakdown.py --num-samples 1000000 --hidden1 128 --hidden2 128
```

---

### `nn_cim_inference_breakdown.py`

Analytical breakdown of the same NN inference flow under a **digital-input** Compute-in-Memory (CIM) model. The NN layers are tiled onto CIM arrays; both DAC (input) and ADC (output) conversions are counted.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--num-samples` | 1000000 | Total Monte Carlo samples |
| `--array-rows` | 128 | CIM array row count |
| `--array-cols` | 128 | CIM array column count |
| `--csv` | `results_csv/nn_cim_inference_breakdown.csv` | Output CSV path |

```bash
python nn_cim_inference_breakdown.py
python nn_cim_inference_breakdown.py --array-rows 64 --array-cols 64 --num-samples 1000000
```

---

### `nn_cim_analog_breakdown.py`

Analytical breakdown under an **analog-input** CIM model with an **analog RNG**. Two key differences from the digital-input model:
- **No DAC**: inputs arrive in analog form; only ADC readout is counted
- **Analog RNG**: Gaussian samples are generated on-chip with finite throughput (Sa/s) and energy efficiency (J/Sa); zero digital RNG cost

Throughput is the bottleneck of RNG rate and ADC rate: `min(rng_throughput / 3, adc_rate / adc_ops_per_sample)`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden1` | 64 | First hidden layer width |
| `--hidden2` | 64 | Second hidden layer width |
| `--num-samples` | 1000000 | Total Monte Carlo samples |
| `--array-rows` | 128 | CIM array row count |
| `--array-cols` | 128 | CIM array column count |
| `--adc-rate` | 1e9 | Aggregate ADC bandwidth (conversions/s) |
| `--rng-throughput` | 5.12e9 | Analog RNG throughput (samples/s) |
| `--rng-efficiency` | 0.36e-12 | Analog RNG energy efficiency (J/sample) |
| `--csv` | `results_csv/nn_cim_analog_breakdown.csv` | Output CSV path |

```bash
python nn_cim_analog_breakdown.py
python nn_cim_analog_breakdown.py --array-rows 64 --array-cols 64 --adc-rate 1e9
python nn_cim_analog_breakdown.py --adc-rate 1e12 --rng-throughput 5.12e9 --rng-efficiency 0.36e-12
```

---

## Part 5 — End-to-End Scenario: Flies in the Room

### `flies_in_the_room.py`

A complete end-to-end scenario that ties together the NN surrogate, GPU benchmarking, and CIM analytical model in a single script.

**Scenario:** A 3-D grid of `grid_size^3` voxels (default 100³ = 1 M voxels), each storing a diagonal 3-D Gaussian `[mu_x, mu_y, mu_z, std_x, std_y, std_z]`. Each benchmark run randomly selects two voxels, then estimates `E[||Z||_2]` where `Z = X - Y`.

**NN surrogate:**
- Input (15D): `eps(3) + mu1(3) + std1(3) + mu2(3) + std2(3)`
- Output (3D): sample from `Z = X - Y` (no final activation)
- MC distance estimate: `mean(||sample||_2)` over N eps draws per run
- Training target: exact reparameterization `Z = (mu1-mu2) + eps * sqrt(std1^2 + std2^2)`

**Outputs per run:**
- NN MC estimate
- Deterministic Gauss-Hermite quadrature ground truth (order 15 = 15³ grid)
- Absolute and relative error vs ground truth

**Summary metrics:**
- CPU and GPU measured throughput (MC samples/s)
- CIM analytical throughput (ADC or RNG bottleneck)
- Analog RNG energy per sample (pJ)
- Histogram plot: overlapping distributions of NN estimates vs quadrature ground truth

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
| `--n-samples` | 1000 | MC samples (eps draws) per run |
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

```bash
# Default run (auto-detects GPU)
python flies_in_the_room.py

# Larger grid, more runs, custom CIM parameters
python flies_in_the_room.py --grid-size 50 --k-runs 200 --n-samples 2000 \
    --array-rows 64 --array-cols 64 --adc-rate 1e9
```

---

## Output Files

All benchmark and breakdown scripts write CSVs (and optionally plots) to `results_csv/`.

| File | Generated by |
|------|-------------|
| `nn_inference_benchmark.csv` | `benchmark_nn_inference.py` |
| `nn_perturbed_params_latency.csv` | `benchmark_nn_perturbed_params.py` |
| `nn_gpu_inference_breakdown.csv` | `nn_gpu_inference_breakdown.py` |
| `nn_cim_inference_breakdown.csv` | `nn_cim_inference_breakdown.py` |
| `nn_cim_analog_breakdown.csv` | `nn_cim_analog_breakdown.py` |
| `flies_in_the_room.csv` | `flies_in_the_room.py` |
| `flies_in_the_room.png` | `flies_in_the_room.py` |

---

## Notes

- Classical CPU/GPU estimators support **full covariance** matrices.
- The NN surrogate assumes **diagonal covariance**, using standard deviation vectors as input.
- Hardware breakdown scripts are **analytical models**, not hardware-counter measurements.
- GPU scripts require a CUDA-visible PyTorch environment; tests involving GPU are skipped when CUDA is unavailable.
