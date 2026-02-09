import numpy as np

from numba import njit, prange
from numba import int64, float64
from numba import types


def calculate_broadening_parameters(
    energy: np.ndarray,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float
) -> np.ndarray:
    """
    Calculates broadening parameters using arctangent model.
    Corresponds to gammarc subroutine in Fortran.
    
    Key differences from original:
    - Uses correct arctangent formula from Fortran
    - Handles zero/negative energies properly
    """
    # Relative energy from Fermi level
    E = energy - E_Fermi
    
    # Initialize with gamma_hole
    gammas = np.full_like(energy, gamma_hole)
    
    # Only apply broadening above Fermi level
    mask = E > 0
    
    if gamma_max > 1e-10 and E_larg > 1e-10 and E_cent > 1e-10:
        # Fortran formula: Gamma_max * (0.5 + atan(p*(E/Ec - (Ec/E)**2)) / pi)
        # where p = (pi/3) * Gamma_max / El
        p = (np.pi / 3.0) * gamma_max / E_larg
        
        E_safe = np.where(mask, E, 1.0)  # Avoid division by zero
        arg = p * (E_safe / E_cent - (E_cent / E_safe)**2)
        
        gamma_energy = gamma_max * (0.5 + np.arctan(arg) / np.pi)
        gammas = np.where(mask, gamma_hole + gamma_energy, gamma_hole)
    
    return gammas


@njit(cache=True)
def calculate_broadening_parameters_fast(
    energy: np.ndarray,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float
) -> np.ndarray:
    """
    Calculates broadening parameters using arctangent model.
    Corresponds to gammarc subroutine in Fortran.
    
    Key differences from original:
    - Uses correct arctangent formula from Fortran
    - Handles zero/negative energies properly
    """
    # Relative energy from Fermi level
    E = energy - E_Fermi
    
    # Initialize with gamma_hole
    gammas = np.full_like(energy, gamma_hole)
    
    # Only apply broadening above Fermi level
    mask = E > 0
    
    if gamma_max > 1e-10 and E_larg > 1e-10 and E_cent > 1e-10:
        # Fortran formula: Gamma_max * (0.5 + atan(p*(E/Ec - (Ec/E)**2)) / pi)
        # where p = (pi/3) * Gamma_max / El
        p = (np.pi / 3.0) * gamma_max / E_larg
        
        E_safe = np.where(mask, E, 1.0)  # Avoid division by zero
        arg = p * (E_safe / E_cent - (E_cent / E_safe)**2)
        
        gamma_energy = gamma_max * (0.5 + np.arctan(arg) / np.pi)
        gammas = np.where(mask, gamma_hole + gamma_energy, gamma_hole)
    
    return gammas

# subroutine gammarc(Ecent,Elarg,Gamma_max,E_cut,nelor, Elor,betalor)

#   use declarations
#   implicit none

#   integer ie, nelor

#   real(kind=db):: E, Ec, Ecent, E_cut, El, Elarg, Gamma_max, p
#   real(kind=db), dimension(nelor):: Elor, betalor

#   Ec = max( Ecent, 1.E-10_db )
#   El = max( Elarg, 1.E-10_db )
#   p = ( pi / 3 ) * Gamma_max / El

#   do ie = 1,nelor
#     E = Elor(ie) - E_cut
#     if ( E <= 0._db ) then
#       betalor(ie) = 0._db
#     else
#       betalor(ie) = Gamma_max * ( 0.5 + atan( p*(E/Ec - (Ec/E)**2)) / pi)
#     endif
#   end do

#   return
# end

def mbrd(
    energy: np.ndarray,
    gamma_hole: float,
    gamma_max: float,
    E_cent: float,
    E_larg: float,
    E_Fermi: float
    ):
    Ec = max(E_cent, 1e-10)
    El = max(E_larg, 1e-10)
    p = (np.pi / 3) * gamma_max / El
    
    broadening = np.zeros(len(energy))
    for i in range(len(energy)):
        E = energy[i] - E_Fermi
        if (E <= 0.0):
            broadening[i] = 0.0
        else:
            broadening[i] = gamma_max * (0.5 + np.arctan( p*(E/Ec - (Ec/E) ** 2)) / np.pi)
    
    return gamma_hole + broadening



def extend_spectrum_tail(
    energy: np.ndarray,
    xanes: np.ndarray,
    n_tail: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extends spectrum with tail points.
    Fortran uses njp = 500 points beyond energy range.
    """
    if len(energy) < 2:
        return energy, xanes
    
    # Use last energy step for extension
    de = energy[-1] - energy[-2]
    
    # Create tail
    energy_tail = energy[-1] + de * np.arange(1, n_tail + 1)
    xanes_tail = np.full(n_tail, xanes[-1])
    
    extended_energy = np.concatenate([energy, energy_tail])
    extended_xanes = np.concatenate([xanes, xanes_tail])
    
    return extended_energy, extended_xanes


@njit(cache=True)
def extend_spectrum_tail_fast(
    energy: np.ndarray,
    xanes: np.ndarray,
    n_tail: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extends spectrum with tail points.
    Fortran uses njp = 500 points beyond energy range.
    """
    if len(energy) < 2:
        return energy, xanes
    
    # Use last energy step for extension
    de = energy[-1] - energy[-2]
    
    # Create tail
    energy_tail = energy[-1] + de * np.arange(1, n_tail + 1)
    xanes_tail = np.full(n_tail, xanes[-1])
    
    extended_energy = np.concatenate((energy, energy_tail))
    extended_xanes = np.concatenate((xanes, xanes_tail))
    
    return extended_energy, extended_xanes

def convolve_lorentzian_integral(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    
    # Find Fermi level index
    i_fermi = np.searchsorted(energy, E_Fermi)
    
    # Calculate energy bin edges
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            # First point: e1 is max of E_Fermi or extrapolated left edge
            e1[i] = max(E_Fermi, 1.5 * energy[0] - 0.5 * energy[1])
        else:
            e1[i] = 0.5 * (energy[i] + energy[i-1])
        
        if i == n - 1:
            e2[i] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
        else:
            e2[i] = 0.5 * (energy[i] + energy[i+1])
    
    # Convolution loop over ALL energy points
    for ie in range(n):
        E_out = energy[ie]
        bb = gammas[ie] / 2.0  # Fortran uses gamma/2
        
        # Check if we have finite broadening
        has_gamma = abs(bb) > 1e-10
        
        if not has_gamma:
            # Delta function case - no broadening
            convoluted[ie] = xanes[ie] if ie < len(xanes) else 0.0
            continue
        
        integral = 0.0
        
        # Integration from Fermi level upward (only unoccupied states contribute)
        for j in range(i_fermi, n):
            if j >= len(xanes):
                break
            
            # Calculate atan integral over bin [e1[j], e2[j]]
            de2 = (e2[j] - E_out) / bb
            de1 = (e1[j] - E_out) / bb
            
            lorentz_integral = (np.arctan(de1) - np.arctan(de2))
            
            integral += lorentz_integral * xanes[j]
        
        convoluted[ie] = -integral / np.pi
    
    return convoluted

@njit(cache=True)
def convolve_lorentzian_integral_vectorization(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    
    # Find Fermi level index
    i_fermi = np.searchsorted(energy, E_Fermi)
    
    # Calculate energy bin edges
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    
    # First point: e1 is max of E_Fermi or extrapolated left edge
    e1[0]  = max(E_Fermi, 1.5 * energy[0] - 0.5 * energy[1])
    e1[1:] = 0.5 * (energy[1:] + energy[:-1])

    e2[:-1] = 0.5 * (energy[:-1] + energy[1:])
    e2[-1] = 1.5 * energy[-1] - 0.5 * energy[-2]            
    
    # Convolution loop over ALL energy points
    for ie in range(n):
        E_out = energy[ie]
        bb = gammas[ie] / 2.0  # Fortran uses gamma/2
        
        # Check if we have finite broadening
        has_gamma = abs(bb) > 1e-10
        
        if not has_gamma:
            # Delta function case - no broadening
            convoluted[ie] = xanes[ie] if ie < len(xanes) else 0.0
            continue
        
        integral = 0.0

        de2 = (e2 - E_out) / bb
        de1 = (e1 - E_out) / bb

        lorentz_integral = (np.arctan(de1) - np.arctan(de2))

        integral = np.dot(lorentz_integral[i_fermi:], xanes[i_fermi:].copy())
        
        convoluted[ie] = -integral / np.pi
    
    return convoluted


@njit(cache=True)
def convolve_lorentzian_integral_numba_cycle_v1(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    
    # Find Fermi level index
    i_fermi = np.searchsorted(energy, E_Fermi)
    
    # Calculate energy bin edges
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            # First point: e1 is max of E_Fermi or extrapolated left edge
            e1[i] = max(E_Fermi, 1.5 * energy[0] - 0.5 * energy[1])
        else:
            e1[i] = 0.5 * (energy[i] + energy[i-1])
        
        if i == n - 1:
            e2[i] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
        else:
            e2[i] = 0.5 * (energy[i] + energy[i+1])
    
    # Convolution loop over ALL energy points
    for ie in range(n):
        E_out = energy[ie]
        bb = gammas[ie] / 2.0  # Fortran uses gamma/2
        
        # Check if we have finite broadening
        has_gamma = abs(bb) > 1e-10
        
        if not has_gamma:
            # Delta function case - no broadening
            convoluted[ie] = xanes[ie] if ie < len(xanes) else 0.0
            continue
        
        integral = 0.0
        
        # Integration from Fermi level upward (only unoccupied states contribute)
        for j in range(i_fermi, n):
            if j >= len(xanes):
                break
            
            # Calculate atan integral over bin [e1[j], e2[j]]
            de2 = (e2[j] - E_out) / bb
            de1 = (e1[j] - E_out) / bb
            
            lorentz_integral = (np.arctan(de1) - np.arctan(de2))
            
            integral += lorentz_integral * xanes[j]
        
        convoluted[ie] = -integral / np.pi
    
    return convoluted

@njit(cache=True)
def convolve_lorentzian_integral_numba_cycle_v2(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    
    # Find Fermi level index
    i_fermi = np.searchsorted(energy, E_Fermi)
    
    # Calculate energy bin edges
    e1 = np.zeros(n)
    e2 = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            # First point: e1 is max of E_Fermi or extrapolated left edge
            e1[i] = 1.5 * energy[0] - 0.5 * energy[1]
        else:
            e1[i] = 0.5 * (energy[i] + energy[i-1])
        
        if i == n - 1:
            e2[i] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
        else:
            e2[i] = 0.5 * (energy[i] + energy[i+1])

    if i_fermi < n:
        e1[i_fermi] = E_Fermi
    
    # Convolution loop over ALL energy points
    for ie in range(n):
        E_out = energy[ie]
        bb = gammas[ie] / 2.0  # Fortran uses gamma/2
        
        # Check if we have finite broadening
        has_gamma = abs(bb) > 1e-10
        
        if not has_gamma:
            # Delta function case - no broadening
            convoluted[ie] = xanes[ie] if ie < len(xanes) else 0.0
            continue
        
        integral = 0.0
        
        # Integration from Fermi level upward (only unoccupied states contribute)
        for j in range(i_fermi, n):
            if j >= len(xanes):
                break
            
            # Calculate atan integral over bin [e1[j], e2[j]]
            de2 = (e2[j] - E_out) / bb
            de1 = (e1[j] - E_out) / bb
            
            lorentz_integral = (np.arctan(de1) - np.arctan(de2))
            
            integral += lorentz_integral * xanes[j]
        
        convoluted[ie] = -integral / np.pi
    
    return convoluted

@njit(cache=True)
def convolve_lorentzian_integral_numba_cycle_v3(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    i_fermi = np.searchsorted(energy, E_Fermi)

    if i_fermi >= n:
        return convoluted
    
    # Precompute energy bin edges
    e1 = np.empty(n)
    e2 = np.empty(n)
    
    e1[0] = 1.5 * energy[0] - 0.5 * energy[1]
    for i in range(1, n):
        e1[i] = 0.5 * (energy[i] + energy[i-1])
    
    e2[n-1] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
    for i in range(n-1):
        e2[i] = 0.5 * (energy[i] + energy[i+1])

    e1[i_fermi] = E_Fermi

    inv_pi = 1.0 / np.pi
    
    # Convolution loop over ALL energy points
    for ie in range(n):
        bb = 0.5 * gammas[ie]  # Fortran uses gamma/2
        
        if abs(bb) <= 1e-10:
            convoluted[ie] = xanes[ie]
            continue
        
        E_out = energy[ie]
        inv_bb = 1.0 / bb

        integral = 0.0
        for j in range(i_fermi, n):    
            de2 = (e2[j] - E_out) * inv_bb
            de1 = (e1[j] - E_out) * inv_bb
            
            integral += (np.arctan(de1) - np.arctan(de2)) * xanes[j]
        
        convoluted[ie] = -integral * inv_pi
    
    return convoluted

@njit(parallel=True, cache=True)
def convolve_lorentzian_integral_numba_cycle_v4(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    i_fermi = np.searchsorted(energy, E_Fermi)

    if i_fermi >= n:
        return convoluted
    
    # Precompute energy bin edges
    e1 = np.empty(n)
    e2 = np.empty(n)
    
    e1[0] = 1.5 * energy[0] - 0.5 * energy[1]
    for i in prange(1, n):
        e1[i] = 0.5 * (energy[i] + energy[i-1])
    
    e2[n-1] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
    for i in prange(n-1):
        e2[i] = 0.5 * (energy[i] + energy[i+1])

    e1[i_fermi] = E_Fermi

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
        for j in prange(i_fermi, n):    
            de2 = (e2[j] - E_out) * inv_bb
            de1 = (e1[j] - E_out) * inv_bb
            
            integral += (np.arctan(de1) - np.arctan(de2)) * xanes[j]
        
        convoluted[ie] = -integral * inv_pi
    
    return convoluted

@njit(parallel=True, cache=True)
def convolve_lorentzian_integral_numba_cycle_v5(
    energy: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    i_fermi = np.searchsorted(energy, E_Fermi)

    if i_fermi >= n:
        return convoluted
    
    # Precompute energy bin edges
    e1 = np.empty(n)
    e2 = np.empty(n)
    
    e1[0] = 1.5 * energy[0] - 0.5 * energy[1]
    for i in range(1, n):
        e1[i] = 0.5 * (energy[i] + energy[i-1])
    
    e2[n-1] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
    for i in range(n-1):
        e2[i] = 0.5 * (energy[i] + energy[i+1])

    e1[i_fermi] = E_Fermi

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
        for j in range(i_fermi, n):    
            de2 = (e2[j] - E_out) * inv_bb
            de1 = (e1[j] - E_out) * inv_bb
            
            integral += (np.arctan(de1) - np.arctan(de2)) * xanes[j]
        
        convoluted[ie] = -integral * inv_pi
    
    return convoluted

@njit(cache=True)
def prepare_energy_bins(energy: np.ndarray):
    n = len(energy)
    e1 = np.empty(n)
    e2 = np.empty(n)
    
    e1[0] = 1.5 * energy[0] - 0.5 * energy[1]
    for i in range(1, n):
        e1[i] = 0.5 * (energy[i] + energy[i-1])
    
    e2[n-1] = 1.5 * energy[n-1] - 0.5 * energy[n-2]
    for i in range(n-1):
        e2[i] = 0.5 * (energy[i] + energy[i+1])

    return e1, e2

@njit(parallel=True, fastmath=True, cache=True)
def convolve_lorentzian_integral_numba_cycle_v6(
    energy: np.ndarray, e1: np.ndarray, e2: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
    Note: Convolution runs over ALL energy points, but integration only 
    includes states above Fermi (unoccupied states).
    """
    n = len(energy)
    convoluted = np.zeros(n)
    i_fermi = np.searchsorted(energy, E_Fermi)

    if i_fermi >= n:
        return convoluted

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
        integral += (np.arctan(de1) - np.arctan(de2)) * xanes[i_fermi]
        
        for j in range(i_fermi + 1, n):
            de2 = (e2[j] - E_out) * inv_bb
            de1 = (e1[j] - E_out) * inv_bb

            integral += (np.arctan(de1) - np.arctan(de2)) * xanes[j]
        
        convoluted[ie] = -integral * inv_pi

    return convoluted


import math

@njit(parallel=True, fastmath=True, cache=True)
def convolve_lorentzian_integral_numba_cycle_v6_math(
    energy: np.ndarray, e1: np.ndarray, e2: np.ndarray,
    xanes: np.ndarray,
    gammas: np.ndarray,
    E_Fermi: float
) -> np.ndarray:
    """
    Lorentzian convolution using integral formula from Fortran cflor subroutine.
    
    Key insight: Fortran integrates over energy bins, not point convolution.
    
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

def convolve_gaussian(
    energy: np.ndarray,
    xanes: np.ndarray,
    sigma: float,
    E_cut: float = 0.0,
    vibration: float = 0.0
) -> np.ndarray:
    """
    Gaussian convolution exactly matching Fortran gaussi subroutine.
    
    Args:
        energy: Energy array
        xanes: Absorption spectrum
        sigma: Gaussian width (deltar in Fortran)
        E_cut: Fermi energy for vibration broadening
        vibration: Additional energy-dependent broadening parameter
    """
    if abs(sigma) < 1e-10 and abs(vibration) < 1e-10:
        return xanes.copy()
    
    nenerg = len(energy)
    nj = 10
    
    # Fortran arrays use indices: ne1=-9, 1, 2, ..., nenerg, nenerg+1, ..., nenerg+10
    # Python: we'll use standard 0-based indexing
    # Fortran index i maps to Python index (i + nj - 1) for i >= 1
    # Fortran index -9 to 0 maps to Python index 0 to 9
    
    total_size = nenerg + 2 * nj
    Ef = np.zeros(total_size)
    Xa = np.zeros(total_size)
    de = np.zeros(total_size)
    
    # Copy original data: Fortran 1:nenerg -> Python nj:nj+nenerg
    Ef[nj:nj + nenerg] = energy
    Xa[nj:nj + nenerg] = xanes
    
    # Create boundary points before: Fortran 0:-9 -> Python 9:0:-1
    if nenerg > 1:
        def_step = Ef[nj + 1] - Ef[nj]  # Ef(2) - Ef(1)
    else:
        def_step = 1.0
    
    for i in range(nj):
        # Fortran: ie = 0, -1, -2, ..., -9
        # Python: i = 0, 1, 2, ..., 9
        py_idx = nj - 1 - i  # 9, 8, 7, ..., 0
        Ef[py_idx] = Ef[py_idx + 1] - def_step
        Xa[py_idx] = Xa[nj]  # Xa(1) in Fortran
    
    # Don't create points after nenerg in Fortran: ne2 = nenerg
    # So we only go up to nj + nenerg
    
    # Calculate de: Fortran de(ne1:ne2) -> Python de[0:nj+nenerg]
    # de(ne1) = Ef(ne1+1) - Ef(ne1) -> de[0] = Ef[1] - Ef[0]
    de[0] = Ef[1] - Ef[0]
    
    # de(ne1+1:ne2-1) -> de[1:nj+nenerg-1]
    for i in range(1, nj + nenerg - 1):
        de[i] = 0.5 * (Ef[i + 1] - Ef[i - 1])
    
    # de(ne2) = Ef(ne2) - Ef(ne2-1) -> de[nj+nenerg-1]
    if nj + nenerg > 1:
        de[nj + nenerg - 1] = Ef[nj + nenerg - 1] - Ef[nj + nenerg - 2]
    
    # Main loop: Fortran ie = 1, nenerg
    Y_out = xanes.copy()
    
    sigma_fwhm = sigma / np.sqrt(8 * np.log(2))
    for ie in range(nenerg):
        ie_py = nj + ie  # Position in extended array
        
        # Energy-dependent width
        vib = 2.0 * vibration * (Ef[ie_py] - E_cut + 0.5)
        vib = max(0.0, vib)
        b = sigma_fwhm + vib
        
        if abs(b) < 1e-10:
            Y_out[ie] = Xa[ie_py]
            continue
        
        gaus = np.zeros(total_size)
        Pdt = 0.0
        
        # Loop: Fortran je = ne1, ne2 -> Python 0 to nj+nenerg-1
        for je_py in range(nj + nenerg):
            
            # Check for valid bin width
            if abs(de[je_py]) < 1e-15:
                continue
            
            n_sub = max(int(10.0 * de[je_py] / b), 1)
            pas = de[je_py] / (n_sub + 1)
            
            # Starting energy
            if je_py == 0:  # je == ne1 in Fortran
                E = Ef[je_py] - 0.5 * (Ef[je_py + 1] - Ef[je_py])
            else:
                E = Ef[je_py] - 0.5 * (Ef[je_py] - Ef[je_py - 1])
            
            # Sub-sampling
            for i_sub in range(1, n_sub + 1):  # Fortran: do i = 1,n
                E = E + pas
                
                # Interpolation condition from Fortran:
                # if( ( E < Ef(je) .and. je /= ne1 ) .or. je == ne2 )
                if (E < Ef[je_py] and je_py != 0) or je_py == nj + nenerg - 1:
                    # Interpolate between je-1 and je
                    if je_py > 0:
                        dE = Ef[je_py] - Ef[je_py - 1]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py - 1]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py - 1] + p * Xa[je_py]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                else:
                    # Interpolate between je and je+1
                    if je_py < total_size - 1:
                        dE = Ef[je_py + 1] - Ef[je_py]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py] + p * Xa[je_py + 1]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                
                # Gaussian weight
                fac = -0.5 * ((E - Ef[ie_py]) / b)**2
                if fac > -600.0:
                    efac = np.exp(fac)
                    gaus[je_py] += efac * Yint
                    Pdt += efac * de[je_py] / n_sub
            
            # Fortran: gaus(je) = ( gaus(je) / n ) * de(je)
            gaus[je_py] = (gaus[je_py] / n_sub) * de[je_py]
        
        # Final sum: Fortran Y(ie) = sum( gaus(ne1:ne2) ) / Pdt
        if Pdt > 1e-30:
            Y_out[ie] = np.sum(gaus[0:nj + nenerg]) / Pdt
        else:
            Y_out[ie] = Xa[ie_py]
    
    return Y_out

@njit(cache=True)
def convolve_gaussian_numba_cycle(
    energy: np.ndarray,
    xanes: np.ndarray,
    sigma: float,
    E_cut: float = 0.0,
    vibration: float = 0.0
) -> np.ndarray:
    """
    Gaussian convolution exactly matching Fortran gaussi subroutine.
    
    Args:
        energy: Energy array
        xanes: Absorption spectrum
        sigma: Gaussian width (deltar in Fortran)
        E_cut: Fermi energy for vibration broadening
        vibration: Additional energy-dependent broadening parameter
    """
    if abs(sigma) < 1e-10 and abs(vibration) < 1e-10:
        return xanes.copy()
    
    nenerg = len(energy)
    nj = 10
    
    # Fortran arrays use indices: ne1=-9, 1, 2, ..., nenerg, nenerg+1, ..., nenerg+10
    # Python: we'll use standard 0-based indexing
    # Fortran index i maps to Python index (i + nj - 1) for i >= 1
    # Fortran index -9 to 0 maps to Python index 0 to 9
    
    total_size = nenerg + 2 * nj
    Ef = np.zeros(total_size)
    Xa = np.zeros(total_size)
    de = np.zeros(total_size)
    
    # Copy original data: Fortran 1:nenerg -> Python nj:nj+nenerg
    Ef[nj:nj + nenerg] = energy
    Xa[nj:nj + nenerg] = xanes
    
    # Create boundary points before: Fortran 0:-9 -> Python 9:0:-1
    if nenerg > 1:
        def_step = Ef[nj + 1] - Ef[nj]  # Ef(2) - Ef(1)
    else:
        def_step = 1.0
    
    for i in range(nj):
        # Fortran: ie = 0, -1, -2, ..., -9
        # Python: i = 0, 1, 2, ..., 9
        py_idx = nj - 1 - i  # 9, 8, 7, ..., 0
        Ef[py_idx] = Ef[py_idx + 1] - def_step
        Xa[py_idx] = Xa[nj]  # Xa(1) in Fortran
    
    # Don't create points after nenerg in Fortran: ne2 = nenerg
    # So we only go up to nj + nenerg
    
    # Calculate de: Fortran de(ne1:ne2) -> Python de[0:nj+nenerg]
    # de(ne1) = Ef(ne1+1) - Ef(ne1) -> de[0] = Ef[1] - Ef[0]
    de[0] = Ef[1] - Ef[0]
    
    # de(ne1+1:ne2-1) -> de[1:nj+nenerg-1]
    for i in range(1, nj + nenerg - 1):
        de[i] = 0.5 * (Ef[i + 1] - Ef[i - 1])
    
    # de(ne2) = Ef(ne2) - Ef(ne2-1) -> de[nj+nenerg-1]
    if nj + nenerg > 1:
        de[nj + nenerg - 1] = Ef[nj + nenerg - 1] - Ef[nj + nenerg - 2]
    
    # Main loop: Fortran ie = 1, nenerg
    Y_out = xanes.copy()
    
    sigma_fwhm = sigma / np.sqrt(8 * np.log(2))
    for ie in range(nenerg):
        ie_py = nj + ie  # Position in extended array
        
        # Energy-dependent width
        vib = 2.0 * vibration * (Ef[ie_py] - E_cut + 0.5)
        vib = max(0.0, vib)
        b = sigma_fwhm + vib
        
        if abs(b) < 1e-10:
            Y_out[ie] = Xa[ie_py]
            continue
        
        gaus = np.zeros(total_size)
        Pdt = 0.0
        
        # Loop: Fortran je = ne1, ne2 -> Python 0 to nj+nenerg-1
        for je_py in range(nj + nenerg):
            
            # Check for valid bin width
            if abs(de[je_py]) < 1e-15:
                continue
            
            n_sub = max(int(10.0 * de[je_py] / b), 1)
            pas = de[je_py] / (n_sub + 1)
            
            # Starting energy
            if je_py == 0:  # je == ne1 in Fortran
                E = Ef[je_py] - 0.5 * (Ef[je_py + 1] - Ef[je_py])
            else:
                E = Ef[je_py] - 0.5 * (Ef[je_py] - Ef[je_py - 1])
            
            # Sub-sampling
            for i_sub in range(1, n_sub + 1):  # Fortran: do i = 1,n
                E = E + pas
                
                # Interpolation condition from Fortran:
                # if( ( E < Ef(je) .and. je /= ne1 ) .or. je == ne2 )
                if (E < Ef[je_py] and je_py != 0) or je_py == nj + nenerg - 1:
                    # Interpolate between je-1 and je
                    if je_py > 0:
                        dE = Ef[je_py] - Ef[je_py - 1]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py - 1]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py - 1] + p * Xa[je_py]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                else:
                    # Interpolate between je and je+1
                    if je_py < total_size - 1:
                        dE = Ef[je_py + 1] - Ef[je_py]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py] + p * Xa[je_py + 1]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                
                # Gaussian weight
                fac = -0.5 * ((E - Ef[ie_py]) / b)**2
                if fac > -600.0:
                    efac = np.exp(fac)
                    gaus[je_py] += efac * Yint
                    Pdt += efac * de[je_py] / n_sub
            
            # Fortran: gaus(je) = ( gaus(je) / n ) * de(je)
            gaus[je_py] = (gaus[je_py] / n_sub) * de[je_py]
        
        # Final sum: Fortran Y(ie) = sum( gaus(ne1:ne2) ) / Pdt
        if Pdt > 1e-30:
            Y_out[ie] = np.sum(gaus[0:nj + nenerg]) / Pdt
        else:
            Y_out[ie] = Xa[ie_py]
    
    return Y_out


@njit(cache=True)
def prepare_gauss(
    energy: np.ndarray,
    xanes: np.ndarray
    ):
    nenerg = len(energy)
    nj = 10
    
    # Fortran arrays use indices: ne1=-9, 1, 2, ..., nenerg, nenerg+1, ..., nenerg+10
    # Python: we'll use standard 0-based indexing
    # Fortran index i maps to Python index (i + nj - 1) for i >= 1
    # Fortran index -9 to 0 maps to Python index 0 to 9
    
    total_size = nenerg + 2 * nj
    Ef = np.zeros(total_size)
    Xa = np.zeros(total_size)
    de = np.zeros(total_size)
    
    # Copy original data: Fortran 1:nenerg -> Python nj:nj+nenerg
    Ef[nj:nj + nenerg] = energy
    Xa[nj:nj + nenerg] = xanes
    
    # Create boundary points before: Fortran 0:-9 -> Python 9:0:-1
    if nenerg > 1:
        def_step = Ef[nj + 1] - Ef[nj]  # Ef(2) - Ef(1)
    else:
        def_step = 1.0
    
    for i in range(nj):
        # Fortran: ie = 0, -1, -2, ..., -9
        # Python: i = 0, 1, 2, ..., 9
        py_idx = nj - 1 - i  # 9, 8, 7, ..., 0
        Ef[py_idx] = Ef[py_idx + 1] - def_step
        Xa[py_idx] = Xa[nj]  # Xa(1) in Fortran
    
    # Don't create points after nenerg in Fortran: ne2 = nenerg
    # So we only go up to nj + nenerg
    
    # Calculate de: Fortran de(ne1:ne2) -> Python de[0:nj+nenerg]
    # de(ne1) = Ef(ne1+1) - Ef(ne1) -> de[0] = Ef[1] - Ef[0]
    de[0] = Ef[1] - Ef[0]
    
    # de(ne1+1:ne2-1) -> de[1:nj+nenerg-1]
    for i in range(1, nj + nenerg - 1):
        de[i] = 0.5 * (Ef[i + 1] - Ef[i - 1])
    
    # de(ne2) = Ef(ne2) - Ef(ne2-1) -> de[nj+nenerg-1]
    if nj + nenerg > 1:
        de[nj + nenerg - 1] = Ef[nj + nenerg - 1] - Ef[nj + nenerg - 2]

    return Ef, Xa, de, nenerg, nj

@njit(parallel=True, fastmath=True, cache=True)
def convolve_gaussian_numba_cycle_v1(
    Ef: np.ndarray, Xa: np.ndarray, de: np.ndarray,
    nenerg: int, nj: int,
    sigma: float,
    E_cut: float = 0.0,
    vibration: float = 0.0
) -> np.ndarray:
    
    total_size = nenerg + 2 * nj

    Y_out = np.empty(nenerg)

    fwhm = sigma / 2.3548200450309493
    for ie in prange(nenerg):
        ie_py = nj + ie  # Position in extended array
        
        # Energy-dependent width
        vib = 2.0 * vibration * (Ef[ie_py] - E_cut + 0.5)
        vib = max(0.0, vib)
        b = fwhm + vib
        
        if abs(b) < 1e-10:
            Y_out[ie] = Xa[ie_py]
            continue
        
        gaus = np.zeros(total_size)
        Pdt = 0.0
        
        # Loop: Fortran je = ne1, ne2 -> Python 0 to nj+nenerg-1
        for je_py in range(nj + nenerg):
            
            # Check for valid bin width
            if abs(de[je_py]) < 1e-15:
                continue
            
            n_sub = max(int(10.0 * de[je_py] / b), 1)
            pas = de[je_py] / (n_sub + 1)
            
            # Starting energy
            if je_py == 0:  # je == ne1 in Fortran
                E = Ef[je_py] - 0.5 * (Ef[je_py + 1] - Ef[je_py])
            else:
                E = Ef[je_py] - 0.5 * (Ef[je_py] - Ef[je_py - 1])
            
            # Sub-sampling
            for i_sub in range(1, n_sub + 1):  # Fortran: do i = 1,n
                E = E + pas
                
                # Interpolation condition from Fortran:
                # if( ( E < Ef(je) .and. je /= ne1 ) .or. je == ne2 )
                if (E < Ef[je_py] and je_py != 0) or je_py == nj + nenerg - 1:
                    # Interpolate between je-1 and je
                    if je_py > 0:
                        dE = Ef[je_py] - Ef[je_py - 1]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py - 1]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py - 1] + p * Xa[je_py]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                else:
                    # Interpolate between je and je+1
                    if je_py < total_size - 1:
                        dE = Ef[je_py + 1] - Ef[je_py]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py]) / dE
                            p = max(0.0, min(1.0, p))  # Clamp to [0,1]
                            Yint = (1.0 - p) * Xa[je_py] + p * Xa[je_py + 1]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                
                # Gaussian weight
                fac = -0.5 * ((E - Ef[ie_py]) / b)**2
                if fac > -600.0:
                    efac = np.exp(fac)
                    gaus[je_py] += efac * Yint
                    Pdt += efac * de[je_py] / n_sub
            
            # Fortran: gaus(je) = ( gaus(je) / n ) * de(je)
            gaus[je_py] = (gaus[je_py] / n_sub) * de[je_py]
        
        # Final sum: Fortran Y(ie) = sum( gaus(ne1:ne2) ) / Pdt
        if Pdt > 1e-30:
            Y_out[ie] = np.sum(gaus[0:nj + nenerg]) / Pdt
        else:
            Y_out[ie] = Xa[ie_py]
    
    return Y_out


def convolve_xanes_integral(
    energy: np.ndarray,
    xanes: np.ndarray,
    gamma_hole: float,
    E_cent: float,
    E_larg: float,
    gamma_max: float,
    E_Fermi: float,
    sigma_gauss: float = 0.0,
    vibration: float = 0.0
) -> np.ndarray:
    """
    Main convolution function matching FDMNES Fortran implementation.
    
    Args:
        energy: Energy array (eV)
        xanes: Absorption spectrum
        gamma_hole: Core-hole lifetime width (eV)
        E_cent: Center energy for arctangent model (eV)
        E_larg: Width parameter for arctangent model (eV)
        gamma_max: Maximum broadening width (eV)
        E_Fermi: Fermi energy (eV)
        sigma_gauss: Gaussian width (eV), default 0
        vibration: Energy-dependent Gaussian broadening parameter, default 0
        
    Returns:
        Convoluted spectrum
    """
    initial_length = len(energy)
    
    if initial_length == 0:
        return np.empty(0)
    
    # Extend spectrum with tail
    extended_energy, extended_xanes = extend_spectrum_tail(energy, xanes)
    
    # Calculate energy-dependent broadening
    gammas = calculate_broadening_parameters(
        extended_energy, gamma_hole, gamma_max, E_cent, E_larg, E_Fermi
    )
    
    # Lorentzian convolution
    convoluted = convolve_lorentzian_integral(
        extended_energy, extended_xanes, gammas, E_Fermi
    )
    
    # Gaussian convolution if needed
    if sigma_gauss > 1e-9 or vibration > 1e-9:
        sigma_gauss_fwhm = sigma_gauss / np.sqrt(8 * np.log(2))
        convoluted = convolve_gaussian(
            extended_energy, convoluted, sigma_gauss_fwhm, E_Fermi, vibration
        )
    
    # Return original length
    return convoluted[:initial_length]

@njit(cache=True)
def convolve_xanes_integral_numba_cycle(
    energy: np.ndarray,
    xanes: np.ndarray,
    gamma_hole: float,
    E_cent: float,
    E_larg: float,
    gamma_max: float,
    E_Fermi: float,
    sigma_gauss: float = 0.0,
    vibration: float = 0.0
) -> np.ndarray:
    """
    Main convolution function matching FDMNES Fortran implementation.
    
    Args:
        energy: Energy array (eV)
        xanes: Absorption spectrum
        gamma_hole: Core-hole lifetime width (eV)
        E_cent: Center energy for arctangent model (eV)
        E_larg: Width parameter for arctangent model (eV)
        gamma_max: Maximum broadening width (eV)
        E_Fermi: Fermi energy (eV)
        sigma_gauss: Gaussian width (eV), default 0
        vibration: Energy-dependent Gaussian broadening parameter, default 0
        
    Returns:
        Convoluted spectrum
    """
    initial_length = len(energy)
    
    if initial_length == 0:
        return np.empty(0)
    
    # Extend spectrum with tail
    extended_energy, extended_xanes = extend_spectrum_tail_fast(energy, xanes)
    
    # Calculate energy-dependent broadening
    gammas = calculate_broadening_parameters_fast(
        extended_energy, gamma_hole, gamma_max, E_cent, E_larg, E_Fermi
    )
    
    # Lorentzian convolution
    convoluted = convolve_lorentzian_integral_numba_cycle_v2(
        extended_energy, extended_xanes, gammas, E_Fermi
    )
    
    # Gaussian convolution if needed
    if sigma_gauss > 1e-9 or vibration > 1e-9:
        sigma_gauss_fwhm = sigma_gauss / np.sqrt(8 * np.log(2))
        convoluted = convolve_gaussian_numba_cycle(
            extended_energy, convoluted, sigma_gauss_fwhm, E_Fermi, vibration
        )
    
    # Return original length
    return convoluted[:initial_length]
