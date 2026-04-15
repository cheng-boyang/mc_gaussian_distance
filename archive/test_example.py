"""Example usage and lightweight tests for Gaussian distance estimators."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None

from gaussian_distance_cpu import (
    deterministic_expected_distance_cpu,
    monte_carlo_expected_distance_cpu,
    reduced_gaussian_parameters,
)
from gaussian_distance_gpu import monte_carlo_expected_distance_gpu


CASE_1 = {
    "mu1": np.array([0.0, 0.0, 0.0]),
    "mu2": np.array([1.0, 2.0, 3.0]),
    "sigma1": np.eye(3),
    "sigma2": np.eye(3),
}

CASE_2 = {
    "mu1": np.array([0.0, 0.0, 0.0]),
    "mu2": np.array([1.0, 1.0, 1.0]),
    "sigma1": np.diag([1.0, 2.0, 3.0]),
    "sigma2": np.diag([2.0, 1.0, 0.5]),
}

CASE_3 = {
    "mu1": np.array([0.0, 0.0, 0.0]),
    "mu2": np.array([0.5, -1.5, 2.0]),
    "sigma1": 0.25 * np.eye(3),
    "sigma2": 4.0 * np.eye(3),
}

CASE_4 = {
    "mu1": np.array([1.0, -1.0, 0.5]),
    "mu2": np.array([-0.5, 0.5, 1.0]),
    "sigma1": np.array([[1.0, 0.9, 0.0], [0.9, 0.81, 0.0], [0.0, 0.0, 0.2]]),
    "sigma2": np.diag([0.1, 0.2, 0.3]),
}

CASE_5 = {
    "mu1": np.array([2.0, -1.0, 0.5]),
    "mu2": np.array([-1.0, 3.0, -2.5]),
    "sigma1": np.zeros((3, 3)),
    "sigma2": np.zeros((3, 3)),
}

ALL_CASES = [CASE_1, CASE_2, CASE_3, CASE_4, CASE_5]


class GaussianDistanceTests(unittest.TestCase):
    """Basic correctness tests and usage examples."""

    def test_reduction_shapes(self) -> None:
        mu_z, sigma_z = reduced_gaussian_parameters(**CASE_1)
        self.assertEqual(mu_z.shape, (3,))
        self.assertEqual(sigma_z.shape, (3, 3))
        np.testing.assert_allclose(mu_z, np.array([-1.0, -2.0, -3.0]))
        np.testing.assert_allclose(sigma_z, 2.0 * np.eye(3))

    def test_cpu_case_1_runs(self) -> None:
        result = monte_carlo_expected_distance_cpu(**CASE_1, num_samples=50_000, seed=7)
        self.assertGreater(result.estimate, 0.0)
        self.assertGreaterEqual(result.variance_estimate, 0.0)

    def test_cpu_case_2_runs(self) -> None:
        result = monte_carlo_expected_distance_cpu(**CASE_2, num_samples=50_000, seed=11)
        self.assertGreater(result.estimate, 0.0)
        self.assertGreaterEqual(result.variance_estimate, 0.0)

    def test_deterministic_matches_point_mass_exactly(self) -> None:
        result = deterministic_expected_distance_cpu(**CASE_5, quadrature_order=7)
        exact = math.sqrt(34.0)
        self.assertAlmostEqual(result.estimate, exact, places=12)
        self.assertEqual(result.variance_estimate, 0.0)

    def test_deterministic_and_monte_carlo_are_close(self) -> None:
        det_result = deterministic_expected_distance_cpu(**CASE_1, quadrature_order=15)
        mc_result = monte_carlo_expected_distance_cpu(**CASE_1, num_samples=200_000, seed=123)
        self.assertAlmostEqual(det_result.estimate, mc_result.estimate, delta=4e-2)

    def test_all_cases_run_on_cpu(self) -> None:
        for case in ALL_CASES:
            det_result = deterministic_expected_distance_cpu(**case, quadrature_order=9)
            mc_result = monte_carlo_expected_distance_cpu(**case, num_samples=20_000, seed=5)
            self.assertGreaterEqual(det_result.estimate, 0.0)
            self.assertGreaterEqual(mc_result.estimate, 0.0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_matches_cpu_reasonably(self) -> None:
        cpu_result = monte_carlo_expected_distance_cpu(**CASE_2, num_samples=100_000, seed=123)
        gpu_result = monte_carlo_expected_distance_gpu(
            **CASE_2,
            num_samples=100_000,
            seed=123,
            dtype=torch.float64,
            batch_size=50_000,
        )
        self.assertAlmostEqual(cpu_result.estimate, gpu_result.estimate, delta=5e-2)


if __name__ == "__main__":
    print("Example usage:")
    det_result = deterministic_expected_distance_cpu(**CASE_1, quadrature_order=15)
    mc_result = monte_carlo_expected_distance_cpu(**CASE_1, num_samples=100_000, seed=123)
    print(f"Case 1 CPU deterministic estimate: {det_result.estimate:.6f}")
    print(f"Case 1 CPU deterministic runtime:  {det_result.runtime_seconds:.4f} s")
    print(f"Case 1 CPU Monte Carlo estimate:   {mc_result.estimate:.6f}")
    print(f"Case 1 CPU Monte Carlo stderr:     {mc_result.variance_estimate ** 0.5:.6e}")
    print(f"Case 1 CPU Monte Carlo runtime:    {mc_result.runtime_seconds:.4f} s")

    if torch is not None and torch.cuda.is_available():
        result_2 = monte_carlo_expected_distance_gpu(
            **CASE_2,
            num_samples=100_000,
            seed=123,
            dtype=torch.float64,
            batch_size=50_000,
        )
        print(f"Case 2 GPU estimate: {result_2.estimate:.6f}")
        print(f"Case 2 GPU stderr:   {result_2.variance_estimate ** 0.5:.6e}")
        print(f"Case 2 GPU runtime:  {result_2.runtime_seconds:.4f} s")
    else:
        print("CUDA is not available; skipping GPU example.")

    print("\nRunning tests...")
    unittest.main(argv=["test_example.py"], exit=False)
