# MC Gaussian Distance

This project studies the expected Euclidean distance between two independent 3D Gaussian distributions,

- `X ~ N(mu1, Sigma1)`
- `Y ~ N(mu2, Sigma2)`

by reducing the problem to the difference distribution

- `Z = X - Y ~ N(mu1 - mu2, Sigma1 + Sigma2)`

and estimating

- `E[||Z||_2]`

using several approaches:

- CPU deterministic quadrature
- CPU Monte Carlo
- GPU Monte Carlo
- a neural-network surrogate sampler for the diagonal-covariance case
- hardware-oriented operation and energy models for GPU and CIM-style execution

## Directory Structure

- [gaussian_distance_cpu.py](/home/bcheng4/Workspace/mc_gaussian_distance/gaussian_distance_cpu.py)
- [gaussian_distance_gpu.py](/home/bcheng4/Workspace/mc_gaussian_distance/gaussian_distance_gpu.py)
- [benchmark.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark.py)
- [test_example.py](/home/bcheng4/Workspace/mc_gaussian_distance/test_example.py)
- [diagnose_cuda.py](/home/bcheng4/Workspace/mc_gaussian_distance/diagnose_cuda.py)
- [profile_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/profile_breakdown.py)
- [hardware_estimate.py](/home/bcheng4/Workspace/mc_gaussian_distance/hardware_estimate.py)
- [nn_gaussian_distance.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_gaussian_distance.py)
- [benchmark_nn_inference.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark_nn_inference.py)
- [benchmark_nn_perturbed_params.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark_nn_perturbed_params.py)
- [nn_gpu_inference_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_gpu_inference_breakdown.py)
- [nn_cim_inference_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_cim_inference_breakdown.py)
- [results_csv](/home/bcheng4/Workspace/mc_gaussian_distance/results_csv)

## Script Overview

### Classical Gaussian-Distance Estimators

- [gaussian_distance_cpu.py](/home/bcheng4/Workspace/mc_gaussian_distance/gaussian_distance_cpu.py)
  CPU implementation of the Gaussian distance problem. It includes:
  - Gaussian parameter validation
  - reduced-distribution construction
  - deterministic Gauss-Hermite quadrature
  - CPU Monte Carlo estimation
  - operation-count helpers

- [gaussian_distance_gpu.py](/home/bcheng4/Workspace/mc_gaussian_distance/gaussian_distance_gpu.py)
  GPU Monte Carlo implementation using PyTorch CUDA. Sampling and norm evaluation remain on GPU as much as possible.

- [benchmark.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark.py)
  Compares CPU deterministic, CPU Monte Carlo, and GPU Monte Carlo across the project test cases.

- [test_example.py](/home/bcheng4/Workspace/mc_gaussian_distance/test_example.py)
  Lightweight tests and runnable examples.

### Environment and Hardware Diagnostics

- [diagnose_cuda.py](/home/bcheng4/Workspace/mc_gaussian_distance/diagnose_cuda.py)
  Checks CUDA visibility for the current Python process and environment.

- [profile_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/profile_breakdown.py)
  Generates operation-breakdown tables and CSVs for the classical estimators.

- [hardware_estimate.py](/home/bcheng4/Workspace/mc_gaussian_distance/hardware_estimate.py)
  Converts the operation/byte model into rough CPU/GPU throughput and energy estimates.

### Neural-Network Surrogate

- [nn_gaussian_distance.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_gaussian_distance.py)
  Implements a two-layer ReLU neural network that learns a sampler for the reduced difference distribution in the diagonal-covariance case. Inputs are:
  - `eps ~ N(0, I_3)`
  - `mu1, std1, mu2, std2`

  Output is a 3D sample intended to follow the target reduced distribution.

- [benchmark_nn_inference.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark_nn_inference.py)
  Benchmarks CPU vs GPU inference for the trained NN surrogate.

- [benchmark_nn_perturbed_params.py](/home/bcheng4/Workspace/mc_gaussian_distance/benchmark_nn_perturbed_params.py)
  Benchmarks the extended flow where NN input parameters themselves are sampled from Gaussian perturbation models before inference. Default is `K=20` parameter batches.

### NN Hardware-Breakdown Models

- [nn_gpu_inference_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_gpu_inference_breakdown.py)
  Analytical breakdown of GPU NN inference into:
  - RNG
  - NN surrogate
  - Euclidean distance

  Reports counts such as multiplies, adds, ReLUs, reads, and writes.

- [nn_cim_inference_breakdown.py](/home/bcheng4/Workspace/mc_gaussian_distance/nn_cim_inference_breakdown.py)
  Analytical breakdown of the same NN inference flow under a compute-in-memory model. Reports:
  - CIM MAC operations
  - CIM array uses
  - ADC operations
  - DAC operations
  - tile accumulation adds
  - reads and writes

  The CIM array size is tunable via command-line arguments.

## Output Files

Generated CSV files are written by default into:

- [results_csv](/home/bcheng4/Workspace/mc_gaussian_distance/results_csv)

This folder contains benchmark outputs, operation-breakdown tables, hardware-estimate tables, and NN/CIM analysis CSVs.

## Typical Workflow

1. Run the classical estimators:

```bash
python benchmark.py
python test_example.py
```

2. Diagnose CUDA visibility if needed:

```bash
python diagnose_cuda.py
```

3. Generate classical operation/hardware reports:

```bash
python profile_breakdown.py
python hardware_estimate.py
```

4. Train and test the NN surrogate:

```bash
python nn_gaussian_distance.py --device cpu
```

5. Benchmark NN inference:

```bash
python benchmark_nn_inference.py
```

6. Benchmark the perturbed-parameter flow:

```bash
python benchmark_nn_perturbed_params.py --device cuda --k-batches 20
```

7. Generate NN hardware-breakdown tables:

```bash
python nn_gpu_inference_breakdown.py
python nn_cim_inference_breakdown.py
```

## Notes

- The classical CPU/GPU estimators support full covariance matrices.
- The NN surrogate path currently assumes diagonal covariance, using standard deviation vectors as input.
- Many hardware and energy scripts are analytical models, not direct hardware-counter measurements.
- GPU scripts require a CUDA-visible PyTorch environment.
