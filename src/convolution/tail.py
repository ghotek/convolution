import numpy as np

from numba import njit


@njit(cache=True)
def extend_spectrum_tail(
        energy: np.ndarray, xanes: np.ndarray, n_tail: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extends spectrum with tail points.
    """
    if len(energy) < 2:
        return energy, xanes
    
    de = energy[-1] - energy[-2]

    n = len(energy)
    total_size = n + n_tail

    extended_energy = np.empty(total_size)
    extended_xanes = np.empty(total_size)
    
    for i in range(n):
        extended_energy[i] = energy[i]
        extended_xanes[i]  = xanes[i]

    for i in range(n, total_size):
        extended_energy[i] = energy[-1] + de * (i - n + 1)
        extended_xanes[i]  = xanes[-1]

    return extended_energy, extended_xanes
