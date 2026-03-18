import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def calculate_broadening_parameters(
    energy: np.ndarray,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float,
) -> np.ndarray:
    """
    Calculates broadening parameters using arctangent model.
    """

    gammas = np.full_like(energy, gamma_hole)
    if gamma_max > 1e-10 and E_larg > 1e-10 and E_cent > 1e-10:
        inv_pi = 1.0 / math.pi
        inv_E_cent = 1.0 / E_cent
        p = (math.pi / 3.0) * gamma_max / E_larg

        for i in prange(len(energy)):
            E_rel = energy[i] - E_Fermi
            if E_rel > 0:
                arg = p * (E_rel * inv_E_cent - (E_cent / E_rel) ** 2)
                gamma_energy = gamma_max * (0.5 + math.atan(arg) * inv_pi)
                gammas[i] = gamma_hole + gamma_energy
            else:
                gammas[i] = gamma_hole

    return gammas
