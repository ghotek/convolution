import numpy as np
from numba import njit

from convolution.broadening import calculate_broadening_parameters
from convolution.gauss import convolve_gaussian, convolve_gaussian_prepared, prepare_gauss
from convolution.lorentz import convolve_lorentzian, convolve_lorentzian_prepared
from convolution.tail import extend_spectrum_tail


@njit(cache=True)
def convolve_prepared(
    extended_energy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    extended_xanes: np.ndarray,
    *,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float,
    sigma_gauss: float = 0.0,
    vibration: float = 0.0,
) -> np.ndarray:
    gammas = calculate_broadening_parameters(
        energy=extended_energy,
        gamma_hole=gamma_hole,
        gamma_max=gamma_max,
        E_cent=E_cent,
        E_larg=E_larg,
        E_Fermi=E_Fermi,
    )

    convolution = convolve_lorentzian_prepared(
        energy=extended_energy, e1=e1, e2=e2, xanes=extended_xanes, gammas=gammas, E_Fermi=E_Fermi
    )

    if sigma_gauss > 1e-9 or vibration > 1e-9:
        Ef, de, nenerg, nj, Xa = prepare_gauss(energy=extended_energy, xanes=convolution)
        convolution = convolve_gaussian_prepared(
            Ef=Ef,
            de=de,
            nenerg=nenerg,
            nj=nj,
            Xa=Xa,
            E_cut=E_Fermi,
            sigma_gauss=sigma_gauss,
            vibration=vibration,
        )

    return convolution


@njit(cache=True)
def convolve(
    energy: np.ndarray,
    xanes: np.ndarray,
    *,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float,
    sigma_gauss: float = 0.0,
    vibration: float = 0.0,
) -> np.ndarray:
    extended_energy, extended_xanes = extend_spectrum_tail(energy=energy, xanes=xanes, n_tail=500)
    gammas = calculate_broadening_parameters(
        energy=extended_energy,
        gamma_hole=gamma_hole,
        gamma_max=gamma_max,
        E_cent=E_cent,
        E_larg=E_larg,
        E_Fermi=E_Fermi,
    )

    convolution = convolve_lorentzian(
        energy=extended_energy, xanes=extended_xanes, gammas=gammas, E_Fermi=E_Fermi
    )

    if sigma_gauss > 1e-9 or vibration > 1e-9:
        convolution = convolve_gaussian(
            energy=extended_energy,
            xanes=convolution,
            E_cut=E_Fermi,
            sigma_gauss=sigma_gauss,
            vibration=vibration,
        )

    return convolution
