import numpy as np
from pathlib import Path
from importlib.resources import files


def photoion_xsec_nl_Verner95(E, fit_pars):
        # Photoionization cross section for a single atomic (n,l) subshell using the
        # analytic fit of Verner et al. (1995, A&AS 109, 125).
        # Energies in eV, cross sections in cm^2.
    
        l,Eth,E0,sigma0,ya,p,yw = fit_pars

        sigma = np.zeros_like(E)
        E_gtr_thresh = E[E >= Eth]

        x = E_gtr_thresh/E0
        f = ((x-1)**2 + yw**2) * x**(0.5*p-l-5.5) * (1+np.sqrt(x/ya))**(-p)
        sigma[E>=Eth] = sigma0 * f * 1e-18
        return sigma


def photoion_xsec_H2(E):    
    # Photoionization cross section for molecular hydrogen from 10.1086/305420 in [cm^2]
    # Energy is in eV

    x = E/15.4
    EkeV = E/1000
    sigmaH2 = 45.57*(1-2.003/x**(0.5) - 4.806/x + 50.577/x**(1.5) - 171.044/x**2 + 231.608/x**(2.5)-81.885/x**3)/EkeV**(3.5)  #barn
    return sigmaH2 * 1e-24 #cm^2


# Fit parameters for photoion_xsec_nl_Verner95
Verner95_xsec_tab = np.loadtxt(files("beyond21").joinpath("data", "PhotoionXsec_Verner95.txt"))

def tau_MW(E_keV,fmol,NH):
    """
    Milky Way optical depth (Eq.8)

    Parameters
    ----------
    E_keV : Photon energy [keV].
    fmol : Molecular hydrogen fraction (sets H2 and HI abundances).
    NH : Hydrogen column density [cm^-2].

    Returns
    -------
    tau : ISM optical depth
    """

    E_eV = E_keV*1000

    # Specie abundance from 10.1086/317016
    species_Az = {
    0 : fmol/2, # H2  
    1:  1-fmol,  # H
    2:  10**(10.99-12),  # He
    6:  10**(8.38-12),   # C
    7:  10**(7.88-12),   # N
    8:  10**(8.69-12),   # O
    10:  10**(7.94-12),   # Ne
    11: 10**(6.16-12),   # Na
    12: 10**(7.40-12),   # Mg
    13: 10**(6.33-12),   # Al
    14: 10**(7.27-12),   # Si
    15: 10**(5.42-12),   # P
    16: 10**(7.09-12),   # S
    17: 10**(5.12-12),   # Cl
    18: 10**(6.41-12),   # Ar
    20: 10**(6.20-12),   # Ca
    22: 10**(4.81-12),   # Ti
    24: 10**(5.51-12),   # Cr
    25: 10**(5.34-12),   # Mn
    26: 10**(7.43-12),   # Fe
    27: 10**(4.92-12),   # Co
    28: 10**(6.05-12),   # Ni
}

    Z_loaded = Verner95_xsec_tab[:,0].astype(int) # Atomic numbers of loaded species in Verner95 table
    
    # Compute cross sections for all species and weight by abundance
    Xsec_arr = np.zeros((31))
    Xsec_arr[0] = photoion_xsec_H2(E_eV)
    for i,Z in enumerate(Z_loaded):
        fit_params = Verner95_xsec_tab[i,3:]
        Xsec_arr[Z] += photoion_xsec_nl_Verner95(E_eV, fit_params)
    
    sig_eff = [species_Az[key]*Xsec_arr[key] for key in species_Az.keys()]
    tot_xsec = np.sum(sig_eff,axis = 0)
    return tot_xsec*NH