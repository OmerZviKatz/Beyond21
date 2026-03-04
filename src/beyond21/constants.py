import numpy as np
from colossus.cosmology import cosmology
from numba import jit


################################### 
#          Basic constants        #
################################### 

#Basics
kB      = 8.6173324e-5          #[eV/K] Boltzman constant 
m_p     = 0.938272081e9         #[eV] Proton mass
m_e     = 510998.9461           #[eV] Electron mass
m_He    = 3.97107 * m_p         #[eV] Helium mass
hbar    = 6.58211951e-16        #[eV*sec]
Planck  = 2 * np.pi * hbar      #[eV*sec] Planck's constant
c       = 299792458e2           #[cm/s] speed of light
GN      = 6.708829880230113e-57 #[1/eV^2]
alphaEM = 1/137.035999139       #[] Fine structure constant

#Atomic
r_Bohr =  1 / (m_e*alphaEM)     #[1/eV] Bohr radius
E_Ryd = 13.60569253              #[eV] Rydeberg energy
Freq_Ryd = E_Ryd/Planck          #[Hz] Rydberg frequency
Lambda_Lya = 1.21567e-5        #[cm] Ly-alpha wavelength
Freq_Lya = c/Lambda_Lya        #[Hz] Ly-alpha frequency
E_Lya = Freq_Lya*Planck        #[eV] Lyman alpha energy
#E_Lya = 1.21567 * pow(10, -5) / (hbar * c)      # [eV] Lyman alpha energy
stefboltz    = np.pi**2 / (60 * (hbar**3) * (c**2)) #[eV^-3 cm^-2 s^-1 ] Stefan-Boltzmann constant
a_r = (4 * stefboltz / c) * (hbar * c) ** 3 #[] Radiation constant see Eq.2 in 1904.09296
Thomson_xsec = 6.652458734e-25 / (hbar * c) ** 2   #[1/eV^2] Thomson cross section

Freq_21 = 1420.405751768e6 #Hz
E_21 = Freq_21*hbar*2*np.pi
A10 = 2.85e-15 #s^-1
Tstar = 0.0628  # Lyman alpha temperature (h*nu0/kB) - Kelvin

################################### 
#         From units to eV        #
################################### 

#Time and frequency
Sec  = 1 / hbar
Day  = 86400 * Sec
Year = 365 * Day
Hz   = 1 / Sec


#Length
Centimeter = 5.067730716156396e4
Meter      = 100 * Centimeter
Mpc        = 3.086e24 * Centimeter
kpc        = 1e-3 * Mpc
pc         = 1e-3 * kpc
Angstrom   = 1e-10 * Meter


#Mass and energy
Kilogram = 5.6095883571872e35
joule    = Kilogram * Meter**2 / Sec**2
erg      = 1e-7 * joule
M_s      = 1.98847e30 * Kilogram


############################################
#           Common unit conversions        #
############################################
cm3_to_Mpc3 = (Mpc / Centimeter)**3

