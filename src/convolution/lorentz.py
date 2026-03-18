import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def prepare_energy_bins(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(energy)
    e1 = np.empty(n)
    e2 = np.empty(n)

    e1[0] = 1.5 * energy[0] - 0.5 * energy[1]
    for i in range(1, n):
        e1[i] = 0.5 * (energy[i] + energy[i - 1])

    e2[n - 1] = 1.5 * energy[n - 1] - 0.5 * energy[n - 2]
    for i in range(n - 1):
        e2[i] = 0.5 * (energy[i] + energy[i + 1])

    return e1, e2


@njit(parallel=True, fastmath=True, cache=True)
def convolve_lorentzian_prepared(
    energy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float,
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.

    Note: Convolution runs over ALL energy points, but integration only
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    i_fermi = np.searchsorted(energy, E_Fermi)

    if i_fermi >= n:
        return xanes.copy()

    inv_pi = 1.0 / np.pi

    # Convolution loop over ALL energy points
    for ie in prange(n):
        bb = 0.5 * gammas[ie]  # Fortran uses gamma/2

        if abs(bb) <= 1e-10:
            convoluted[ie] = xanes[ie]
            continue

        E_out = energy[ie]
        inv_bb = 1.0 / bb

        integral = 0.0

        e1j = E_Fermi
        de2 = (e2[i_fermi] - E_out) * inv_bb
        de1 = (e1j - E_out) * inv_bb
        integral += (math.atan(de1) - math.atan(de2)) * xanes[i_fermi]

        for j in range(i_fermi + 1, n):
            de2 = (e2[j] - E_out) * inv_bb
            de1 = (e1[j] - E_out) * inv_bb

            integral += (math.atan(de1) - math.atan(de2)) * xanes[j]

        convoluted[ie] = -integral * inv_pi

    return convoluted


@njit(cache=True)
def convolve_lorentzian(
    energy: np.ndarray, xanes: np.ndarray, gammas: np.ndarray, E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.

    Note: Convolution runs over ALL energy points, but integration only
    includes states above Fermi (unoccupied states).
    """
    e1, e2 = prepare_energy_bins(energy=energy)
    convolution = convolve_lorentzian_prepared(
        energy=energy, e1=e1, e2=e2, xanes=xanes, gammas=gammas, E_Fermi=E_Fermi
    )

    return convolution
