import math

import numpy as np

from numba import njit, prange


@njit(cache=True)
def prepare_gauss(
        energy: np.ndarray,
        xanes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    nenerg = len(energy)
    nj = 10
    
    total_size = nenerg + 2 * nj
    Ef = np.zeros(total_size)
    Xa = np.zeros(total_size)
    de = np.zeros(total_size)
    
    Ef[nj:nj + nenerg] = energy
    Xa[nj:nj + nenerg] = xanes
    
    if nenerg > 1:
        def_step = Ef[nj + 1] - Ef[nj]
    else:
        def_step = 1.0
    
    for i in range(nj):
        py_idx = nj - 1 - i
        Ef[py_idx] = Ef[py_idx + 1] - def_step
        Xa[py_idx] = Xa[nj]
    
    de[0] = Ef[1] - Ef[0]
    
    for i in range(1, nj + nenerg - 1):
        de[i] = 0.5 * (Ef[i + 1] - Ef[i - 1])
    
    if nj + nenerg > 1:
        de[nj + nenerg - 1] = Ef[nj + nenerg - 1] - Ef[nj + nenerg - 2]

    return Ef, de, nenerg, nj, Xa

@njit(parallel=True, fastmath=True, cache=True)
def convolve_gaussian_prepared(
        Ef: np.ndarray,
        de: np.ndarray,
        nenerg: int, nj: int,
        Xa: np.ndarray,
        E_cut: float,
        sigma_gauss: float,
        vibration: float = 0.0
    ) -> np.ndarray:
    
    total_size = nenerg + 2 * nj

    Y_out = np.empty(nenerg)

    fwhm = sigma_gauss / 2.3548200450309493
    for ie in prange(nenerg):
        ie_py = nj + ie

        vib = 2.0 * vibration * (Ef[ie_py] - E_cut + 0.5)
        vib = max(0.0, vib)

        b = fwhm + vib
        inv_b = 1.0 / b
        
        if abs(b) < 1e-10:
            Y_out[ie] = Xa[ie_py]
            continue
        
        gaus_acc = 0.0
        Pdt = 0.0
        
        for je_py in range(nj + nenerg):
            
            if abs(de[je_py]) < 1e-15:
                continue
            
            n_sub = max(int(10.0 * de[je_py] * inv_b), 1)
            inv_n_sub = 1.0 / n_sub
            pas = de[je_py] / (n_sub + 1)
            
            if je_py == 0:
                E = Ef[je_py] - 0.5 * (Ef[je_py + 1] - Ef[je_py])
            else:
                E = Ef[je_py] - 0.5 * (Ef[je_py] - Ef[je_py - 1])
            
            for i_sub in range(1, n_sub + 1):
                E = E + pas
                
                if (E < Ef[je_py] and je_py != 0) or je_py == nj + nenerg - 1:
                    if je_py > 0:
                        dE = Ef[je_py] - Ef[je_py - 1]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py - 1]) / dE
                            p = max(0.0, min(1.0, p))
                            Yint = (1.0 - p) * Xa[je_py - 1] + p * Xa[je_py]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                else:
                    if je_py < total_size - 1:
                        dE = Ef[je_py + 1] - Ef[je_py]
                        if abs(dE) > 1e-15:
                            p = (E - Ef[je_py]) / dE
                            p = max(0.0, min(1.0, p))
                            Yint = (1.0 - p) * Xa[je_py] + p * Xa[je_py + 1]
                        else:
                            Yint = Xa[je_py]
                    else:
                        Yint = Xa[je_py]
                
                fac = -0.5 * ((E - Ef[ie_py]) * inv_b)**2
                if fac > -600.0:
                    efac = math.exp(fac)
                    gaus_acc += efac * Yint * inv_n_sub * de[je_py]
                    Pdt += efac * de[je_py] * inv_n_sub
            
        if Pdt > 1e-30:
            Y_out[ie] = gaus_acc / Pdt
        else:
            Y_out[ie] = Xa[ie_py]
    
    return Y_out

@njit(cache=True)
def convolve_gaussian(
        energy: np.ndarray,
        xanes:  np.ndarray,
        E_cut: float,
        sigma_gauss: float,
        vibration: float = 0.0
    ) -> np.ndarray:
    Ef, de, nenerg, nj, Xa = prepare_gauss(
        energy=energy, xanes=xanes
        )
    convolution = convolve_gaussian_prepared(
        Ef=Ef, de=de,
        nenerg=nenerg, nj=nj,
        Xa=Xa,
        E_cut=E_cut,
        sigma_gauss=sigma_gauss,
        vibration=vibration
    )

    return convolution