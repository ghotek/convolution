import time

import numpy as np

from convolution.broadening import calculate_broadening_parameters


class TestPerformanceBroadening:
    """Performance tests for calculate_broadening_parameters function."""

    def setup_method(self):
        """Setup for performance tests."""
        self.gamma_hole = 0.1
        self.gamma_max = 2.0
        self.E_cent = 1.5
        self.E_larg = 3.0
        self.E_Fermi = 0.5

        # Warm up the JIT compiler
        small_array = np.linspace(-5, 5, 10)
        _ = calculate_broadening_parameters(
            small_array, self.gamma_hole, self.gamma_max, self.E_cent, self.E_larg, self.E_Fermi
        )

    def test_performance_many_calls(self):
        """Test performance with many repeated calls."""
        energy = np.linspace(-5, 5, 1000)
        n_calls = 1000

        start_time = time.perf_counter()
        for _ in range(n_calls):
            result = calculate_broadening_parameters(
                energy, self.gamma_hole, self.gamma_max, self.E_cent, self.E_larg, self.E_Fermi
            )
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_calls
        calls_per_second = n_calls / total_time

        print(f"\nPerformance with {n_calls} calls:")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Average per call: {avg_time * 1e6:.2f} µs")
        print(f"  Calls per second: {calls_per_second:.0f}")


if __name__ == "__main__":
    print("\nRunning performance test...")

    perf_class = TestPerformanceBroadening()
    perf_class.setup_method()

    perf_methods = ["test_performance_many_calls"]

    for method in perf_methods:
        getattr(perf_class, method)()
        print(f"[+] {method}")

    print("\nAll tests passed!")
