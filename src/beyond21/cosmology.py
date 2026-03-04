import numpy as np
from colossus.cosmology import cosmology
import beyond21.constants as unit
from colossus.lss import mass_function

class Cosmology:
    """
    Cosmology wrapper.

    - Creates a fresh Colossus cosmology on init.
    - Sets it active immediately.
    - Provides HMF: d n / d ln M via Colossus.

    Conventions:
      - HMF input mass M is Msun/h (Colossus convention for q_in="M")
      - HMF output dndlnM is (Mpc/h)^-3
    """

    def __init__(self, Cosmo_params):
        if Cosmo_params is None:
            warnings.warn("No Cosmo_params given. Assuming Planck18 cosmology.",UserWarning)
            Cosmo_params = {}
            
        Om0 = Cosmo_params["Om0"]
        Ob0 = Cosmo_params["Ob0"]
        H0 = Cosmo_params["H0"]
        sigma8 = Cosmo_params["sigma8"]
        ns = Cosmo_params["ns"]
        Neff = Cosmo_params["Neff"]
        w0 = Cosmo_params["w0"]
        wa = Cosmo_params["wa"]
        self.Y_He = Cosmo_params["Y_He"]
        self.hmf_kwargs = Cosmo_params["hmf_kwargs"]
        self.Tcmb0 = 2.725 # CMB tempeature
        
        self.set_Colossus_cosmo(Om0, Ob0, H0, sigma8, ns, Neff, w0, wa)
        self.set_internal_properties()
        
        
    def set_Colossus_cosmo(self, Om0, Ob0, H0, sigma8, ns, Neff, w0, wa):
        w0 = float(w0)
        wa = float(wa)

        if w0 == -1.0 and wa == 0.0:
            de_model = "lambda"
        else:
            de_model = "w0wa"

        params = {
            "flat": True,
            "Om0": float(Om0),
            "Ob0": float(Ob0),
            "H0": float(H0),
            "sigma8": float(sigma8),
            "ns": float(ns),
            "Tcmb0": self.Tcmb0,
            "Neff": float(Neff),
            "relspecies": True,
            "de_model": de_model,
        }

        if de_model == "w0wa":
            params["w0"] = w0
            params["wa"] = wa

        # Set cosmology
        self.cosmo = cosmology.setCosmology("custom", **params)


    def set_internal_properties(self):

        # Hubble
        self.h = self.cosmo.h                           #Hubble
        self.H0 = self.cosmo.H0*unit.Meter*1000/unit.Mpc #[1/s]

        # Abundances
        self.Omega_lambda = self.cosmo.Ode0             #Dark energy
        self.Omega_m      = self.cosmo.Om0              #Matter
        self.Omega_r      = self.cosmo.Or0              #Radiation
        self.Omega_b      = self.cosmo.Ob0              #Baryons
        self.Omega_DM     = self.Omega_m - self.Omega_b      #DM

        # Mean energy densities today [eV/cm^3] 
        self.rho_crit = float(self.cosmo.rho_c(0)*unit.M_s/unit.kpc**3*unit.Centimeter**3*self.cosmo.h**2)
        self.rho_baryon = self.rho_crit * self.Omega_b
        self.rho_DM = self.rho_crit * self.Omega_DM

        # Mean number densities today [1/cm^3]
        self.nHe = self.rho_baryon * self.Y_He / unit.m_He 
        self.nH = self.rho_baryon * (1-self.Y_He) / unit.m_p 
        self.nB = self.nH + self.nHe

        self.mu_b = self.rho_baryon/self.nB # Mean baryon mass 
        return 

    def TCMB(self,rs):
        return self.Tcmb0 * unit.kB * rs
        
    def hubble(self,rs):
        #Hubble parameter at rs = 1+z in units of [1/s]
        return self.H0*np.sqrt(self.Omega_r*rs**4 + self.Omega_m*rs**3 + self.Omega_lambda)

    def dzdt(self,z):
        return -self.Cosmo.hubble(1+z)*(1+z)

    def rs_from_log_a(self,log_a): 
        return (1 / np.exp(log_a))

    def print_cosmo(self):
        for key, value in self.cosmo.__dict__.items():
            print(f"{key}: {value}")
        return 

    def dndlnm(self, Mh, z, PS = False):
        ''' 
        Parameters: 
            Mh - Halo mass in eV 
            z - redshift
        Return: 
            dndlnm in (Mpc/h)^-3 converted to cm^-3
        '''
        Mpc_to_cm = unit.Mpc/unit.Centimeter

        if PS:
            hmf_kwargs = {"mdef": "fof", "model": "press74"}
            return mass_function.massFunction(Mh / (unit.M_s / self.h), z, q_in="M", q_out="dndlnM", **hmf_kwargs) * (Mpc_to_cm / self.h) ** -3  # Notice that the input mass for hmf is changed to M_s/h as required in COLOSSUS.

        return mass_function.massFunction(Mh / (unit.M_s / self.h), z, q_in="M", q_out="dndlnM", **self.hmf_kwargs) * (Mpc_to_cm / self.h) ** -3  # Notice that the input mass for hmf is changed to M_s/h as required in COLOSSUS.


