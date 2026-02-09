import numpy as np


def extend_spectrum_tail(
        energy: np.ndarray, xanes: np.ndarray, n_tail: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extends spectrum with tail points.
    """
    if len(energy) < 2:
        return energy, xanes
    
    de = energy[-1] - energy[-2]
    
    energy_tail = energy[-1] + de * np.arange(1, n_tail + 1)
    xanes_tail = np.full(n_tail, xanes[-1])
    
    extended_energy = np.concatenate([energy, energy_tail])
    extended_xanes = np.concatenate([xanes, xanes_tail])
    
    return extended_energy, extended_xanes
