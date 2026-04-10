# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests
python test_example.py

# Run a single unittest class
python -m unittest test_example.GaussianDistanceTests.test_deterministic_matches_point_mass_exactly

# Run all classical benchmarks
python benchmark.py

# Check CUDA availability
python diagnose_cuda.py

# Train and evaluate the NN surrogate
python nn_gaussian_distance.py --device cpu
python nn_gaussian_distance.py --device cuda

# NN inference benchmark (CPU vs GPU)
python benchmark_nn_inference.py

# Perturbed-parameter benchmark
python benchmark_nn_perturbed_params.py --device cuda --k-batches 20

# Hardware breakdown reports (write to results_csv/)
python profile_breakdown.py
python hardware_estimate.py
python nn_gpu_inference_breakdown.py
python nn_cim_inference_breakdown.py --array-size 128
```

## Architecture

The project estimates `E[||X - Y||_2]` for independent 3D Gaussians via the reduction `Z = X - Y ~ N(mu1-mu2, Sigma1+Sigma2)`, then estimating `E[||Z||_2]`.

### Two independent estimator stacks

**Classical estimators** (`gaussian_distance_cpu.py`, `gaussian_distance_gpu.py`):
- Support full covariance matrices (not just diagonal)
- CPU path: Gauss-Hermite quadrature (`deterministic_expected_norm_cpu`) or Monte Carlo (`monte_carlo_expected_norm_cpu`). The quadrature grid is `m^d` points, so it does not scale to high dimensions.
- GPU path: PyTorch CUDA batched sampling + norm, kept on-device until the final reduction
- Both return a `DistanceResult` dataclass; the CPU path also has a `ProfiledDistanceResult` variant with per-stage timings
- `OperationCounts` helpers in the CPU module provide analytical flop/byte models for hardware estimation

**NN surrogate** (`nn_gaussian_distance.py`):
- Restricted to **diagonal covariance**; inputs are `(eps, mu1, std1, mu2, std2)` concatenated as a 21-dim feature vector
- `TwoLayerGaussianSampler`: Linear(21→h1) → ReLU → Linear(h1→h2) → ReLU → Linear(h2→3)
- Trained with a hand-rolled Adam loop (no `torch.optim`) to match the closed-form `exact_reduced_sample`
- `monte_carlo_distance_nn` runs batched inference; `monte_carlo_distance_exact` provides the ground-truth reference

### Hardware / analytical models
`hardware_estimate.py` and the `nn_*_inference_breakdown.py` scripts are pure analytical models (no hardware counters). The CIM array size in `nn_cim_inference_breakdown.py` is tunable via `--array-size`.

### Output
All benchmark and breakdown scripts write CSVs to `results_csv/`.

## Key constraints
- GPU scripts require a CUDA-visible PyTorch environment; tests involving GPU are `skipUnless(torch.cuda.is_available())`.
- The NN path only handles 3D Gaussians with diagonal covariance; the classical path handles arbitrary dimension and full covariance.
