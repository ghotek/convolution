import time

import numpy as np

from convolution.tail import extend_spectrum_tail
from convolution.broadening import calculate_broadening_parameters
from convolution.lorentz import (
    prepare_energy_bins, convolve_lorentzian_prepared,
    convolve_lorentzian
)


class TestPerformanceLorentz:
    """Performance tests for calculate_broadening_parameters function."""
    
    def setup_method(self):
        """Setup for performance tests."""
        self.energy = np.linspace(-10.0, 60.0, 1000)
        self.xanes  = np.random.uniform(0.0, 1.0, size=len(self.energy))

        self.gamma_hole = 1.0
        self.gamma_max = 15.0
        self.E_cent = 30.0
        self.E_larg = 30.0
        self.E_Fermi = 0.0
        
        self.extended_energy, self.extended_xanes = extend_spectrum_tail(
            energy=self.energy, xanes=self.xanes, n_tail=500
            )
        self.broadening = calculate_broadening_parameters(
            self.extended_energy, self.gamma_hole, self.gamma_max,
            self.E_cent, self.E_larg,
            self.E_Fermi
        )

        self.e1, self.e2 = prepare_energy_bins(energy=self.extended_energy)
        
        # Warm up the JIT compiler
        _ = convolve_lorentzian(
            energy=self.extended_energy,
            xanes=self.extended_xanes,
            gammas=self.broadening,
            E_Fermi=self.E_Fermi
        )
        _ = convolve_lorentzian_prepared(
            energy=self.extended_energy, e1=self.e1, e2=self.e2,
            xanes=self.extended_xanes,
            gammas=self.broadening,
            E_Fermi=self.E_Fermi
        )
    
    def test_performance_many_calls(self):
        """Test performance with many repeated calls."""
        n_calls = 1000
        
        start_time = time.perf_counter()
        for _ in range(n_calls):
            result = convolve_lorentzian(
                energy=self.extended_energy,
                xanes=self.extended_xanes,
                gammas=self.broadening,
                E_Fermi=self.E_Fermi
            )
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / n_calls
        calls_per_second = n_calls / total_time
        
        print(f"\n\n{convolve_lorentzian.__name__}()")
        print(f" Performance with {n_calls} calls:")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Average per call: {avg_time * 1e6:.2f} µs")
        print(f"  Calls per second: {calls_per_second:.0f}")

        start_time = time.perf_counter()
        for _ in range(n_calls):
            result = convolve_lorentzian_prepared(
                energy=self.extended_energy, e1=self.e1, e2=self.e2,
                xanes=self.extended_xanes,
                gammas=self.broadening,
                E_Fermi=self.E_Fermi
            )
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / n_calls
        calls_per_second = n_calls / total_time
        
        print(f"\n\n{convolve_lorentzian_prepared.__name__}()")
        print(f" Performance with {n_calls} calls:")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  Average per call: {avg_time * 1e6:.2f} µs")
        print(f"  Calls per second: {calls_per_second:.0f}")


if __name__ == "__main__":
    print("\nRunning performance test...")
    
    perf_class = TestPerformanceLorentz()
    perf_class.setup_method()
    
    perf_methods = [
        'test_performance_many_calls'
    ]
    
    for method in perf_methods:
        getattr(perf_class, method)()
        print(f"[+] {method}")
    
    print("\nAll tests passed!")
