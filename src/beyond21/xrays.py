import beyond21.constants as unit
import numpy as np
from scipy import integrate
from beyond21.utils.interp_reg_grid import reg_grid_interp
import beyond21.xrays_mw_abs as MW
from pathlib import Path
import beyond21.interpolations as pre
from scipy.integrate import cumulative_trapezoid
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     

    
class XrayHeatingReion:
    """
    This class computes the X-ray background produced by early galaxies
    (parameterized through an X-ray luminosity-SFR calibration and a spectral model),
    propagates it through the IGM, and evaluates the resulting volumetric heating and 
    hydrogen ionization rates in the bulk IGM outside of UV-ionized regions.

    Parameters
    ----------
    Xray_params : Dictionary of X-ray spectral and normalization parameters with keys: 
        Mandatory: 
            'LSFRII' : Luminosity to SFR ratio of Pop-II stars [erg/s/M_sun/yr]
            'alpha_s' : Soft pectral index of the X-ray SED 
            'E_min' : Minimum photon that escapes host galaxy [keV]
            'E_max': Maximum photon energy that contributes CXB
        Optional:
            'LSFRIII' : Luminosity to SFR ratio of Pop-III stars [erg/s/M_sun/yr]
            'alpha_h' : Hard pectral index of the X-ray SED 
            'E_break' : Break energy for the broken power law SED [keV]

    SFRD_interp : 
        SFRD interpolator(s) returning the comoving SFRD as a function of redshift.
        If two populations are used, provide (SFRDII(z), SFRDIII(z)).
    Q_ion_interp : Interpolator for the UV-ionized volume filling factor Q_ion(z). 
    populations : 
        Flag controlling whether the emissivity includes two populations. 
        Default is single population.
        populations == 'PopII+PopIII' triggers two population model.
    zstar : Maximum redshift for star formation.
    zmin : Minimum redshift used when constructing precomputed (z, x_e) grids.
    xe_max, xe_min : Bounds for the ionized-fraction grid used in interpolation utilities.
    zlen, xe_len : Grid sizes in redshift and ionized fraction for interpolation tables.
    include_HeII : 
        If True, include HeII photoionization/heating channels in the deposition
        calculation (off by default).

    """

    Eth = {'HI': 13.6, 'HeI': 24.59, 'HeII': 54.42}  # Threshold ionization energy [eV] 
    
    def __init__(self, CosmoObj, Xray_params, SFRD_interp, Q_ion_interp = None, populations = False, zstar = 50, zmin = 1, xe_max = 0.9999, xe_min = 1e-5, zlen = 100, xe_len = 25, include_HeII = False):
        self.Cosmo = CosmoObj
        self.include_HeII = include_HeII #Flag that decides whether to include HeII in heating and ionization or not

        #X-ray params
        self.Xray_params = Xray_params
        self.Q_ion_interp = Q_ion_interp
        self.SFRD_interp = SFRD_interp
        self.zstar = zstar
        self.zmin = zmin
        self.zlen = zlen
        self.xe_max = xe_max
        self.xe_min = xe_min
        self.xe_len = xe_len
        self.populations = populations # A flag that tells if there are two populations. We start with false then change according to input

        self.species_fit_params = {
            'HI': [4.298*1e-1, 5.475*1e4, 3.288*1e1, 2.963, 0, 0, 0, 13.60],
            'HeI': [1.361*1e1, 9.492*1e2, 1.469, 3.188, 2.039, 4.434*1e-1, 2.136, 24.59],
            'HeII': [1.720, 1.369*1e4, 3.288*1e1, 2.963, 0, 0, 0, 54.42]
        } #Fit parameters for photoionization xsec



        
    def photoions_xsec_Verner96(self, E, specie):
        # Photoionization cross section [cm^2] based on the analytic fit of 10.1086/177435.
        # Energies in eV, specie = 'HI', 'HeI', or 'HeII'

        E0, sigma0, ya, p, yw, y0, y1, Eth = self.species_fit_params[specie]

        x = E/E0-y0
        y = np.sqrt(x**2 + y1**2)
        f = ((x-1)**2 + yw**2) * y**(0.5*p-5.5) * (1+np.sqrt(y/ya))**(-p)
        sigma = sigma0 * f * 1e-18
        return sigma


    def SpecificXrayNumberEmissivity(self, z, E):
        # Compute the specific X-ray number emissivity per comoving volume [1 / eV / cm^3 / s]
        # at a given redshift,z, and photon energy [eV](Eq.25).
        
        erg_to_solar_mass=unit.erg/unit.M_s
        sec_to_year=unit.Sec/unit.Year
        
        # Normalization set by input L[<2keV]/SFR -> compute L/V = L[<2keV]/SFR*SFRD
        if self.populations == 'PopII+PopIII':
            SFRDII, SFRDIII = self.SFRD_interp[0](z), self.SFRD_interp[1](z) # [eV/cm^3/s]
            LumBySFRII = self.Xray_params['LSFRII']*erg_to_solar_mass/sec_to_year #dimless
            LumBySFRIII = self.Xray_params['LSFRIII']*erg_to_solar_mass/sec_to_year #dimless
            dLdV = LumBySFRII*SFRDII + LumBySFRIII*SFRDIII #[eV/cm^3/s]
        else:
            SFRD = self.SFRD_interp(z) # [eV/cm^3/s]
            LumBySFRII = self.Xray_params['LSFRII']*erg_to_solar_mass/sec_to_year #dimless
            dLdV = LumBySFRII*SFRD #[eV/cm^3/s]

        # Load spectral parameters
        a_s = self.Xray_params['alpha_s'] 
        Emin = self.Xray_params['E_min']*1000
        Emax = self.Xray_params['E_max']*1000
        a_h = self.Xray_params.get('alpha_h', None)
        Ematch = self.Xray_params.get('E_break', None)
        if Ematch:
            Ematch *= 1000 # Change to eV
        
        # Spectral shape (Eq. 32)
        fE = np.zeros_like(E) 
        if a_h != None:
            mask_low = (E >= Emin) & (E < Ematch)  # E in [Emin, Ematch)
            mask_high = (E >= Ematch) & (E <= Emax)  # E in [Ematch, Emax]            
            fE[mask_low] = E[mask_low] ** (-a_s - 1)
            fE[mask_high] = Ematch ** (a_h - a_s) * E[mask_high] ** (-a_h - 1)
        else:
            mask_high = (E >= Emin)
            fE[mask_high] = E**(-a_s - 1)

        # Normalize to L[<2keV]/SFR (Eq. 33)
        if a_s == 1:
            NormFact = 1/np.log(2000/Emin) 
        else: 
            NormFact = (1-a_s)/(2000**(1-a_s)-Emin**(1-a_s)) 
        
        return dLdV*fE*NormFact # [1/eV/s/cm^3]


    # # Exact computation
    # def optical_depth(self, z, zp_arr, E_arr):
    #     ''' 
    #     Compute the optical depth of a photon emitted at z' and observed at z with energy E, tau(E,z,z') (Eq.35):

    #     Parameters
    #     ----------
    #         z : float 
    #             Final propagation redshift 
    #         zp_arr : 1D array 
    #             Initial propagation redshift (zp>z)
    #         E_arr : 1D array
    #             Energy at z [eV] 
    #     Returns
    #     -------
    #         tau_2D : 2D array 
    #             Array of optical depths with shape (len(zp_arr)), len(E_arr))
    #     '''

    #     # tau is an integral over z'' between [z',z]
    #     # Create 2D matrix with rows running from zp to z for all zp's in zp_arr
    #     zp_col = zp_arr[:,None] # Column vector
    #     zpp_len = int(max(2, np.max(zp_arr - z)))
    #     zpp_matrix = np.linspace(zp_col, z, zpp_len, axis=1)[:,:,0] # (len(zp_arr), zpp_len)
    #     rs_pp_matrix = 1 + zpp_matrix 

    #     # Compute prefactors and ionized fraction outside integral to improve runtime
    #     preFact = -unit.c*rs_pp_matrix**2/self.Cosmo.hubble(rs_pp_matrix)

    #     Q_ion_pp = self.Q_ion_interp(zpp_matrix) 
        
    #     # Energy at z_pp 
    #     E_pp = E_arr * rs_pp_matrix[:,:,None] /(1+z) # (len(zp_arr), zpp_len, len(E_arr))
       
    #     # Broadcast energy dimension
    #     preFact = np.broadcast_to(preFact[:, :, None], E_pp.shape)
    #     Q_ion_pp = np.broadcast_to(Q_ion_pp[:, :, None], E_pp.shape)
    #     zpp = np.broadcast_to(zpp_matrix[:,:,None], E_pp.shape)

    #     # Solve integral
    #     integrand = np.zeros_like(E_pp,dtype = float)
    #     integrand += preFact * self.photoions_xsec_Verner96(E_pp, 'HI') * self.Cosmo.nH * (1-Q_ion_pp)
    #     integrand += preFact * self.photoions_xsec_Verner96(E_pp, 'HeI') * self.Cosmo.nHe * (1-Q_ion_pp)                           
        

    #     tau_2D = np.trapz(integrand, zpp,axis = 1)
    #     #print(tau_2D)
    #     return(tau_2D)


    # Interpolation - Fast
    def optical_depth(self, z, zp_arr, E_arr):
        '''
        Compute the optical depth of a photon emitted at z' and observed at z with energy E, tau(E,z,z') (Eq.35):

        Parameters
        ----------
            z : float 
                Final propagation redshift 
            zp_arr : 1D array 
                Initial propagation redshift (zp>z)
            E_arr : 1D array
                Energy at z [eV] 
        Returns
        -------
            tau_2D : 2D array 
                Array of optical depths with shape (len(zp_arr)), len(E_arr))
        '''

        """
        tau(E, z, z') for all z' in zp_arr and energies in E_arr.
        Output shape: (len(zp_arr), len(E_arr))
        """

        # tau is an integral over z'' between [z',z]
        # Create and array zpp running from z to max(zp)
        zmax = np.max(zp_arr)
        Nzpp = int(10 * zmax)
        zpp = np.linspace(z, zmax, Nzpp)              # (Nzpp,)
        rs  = 1.0 + zpp                               # (Nzpp,)

        # Compute prefactors and ionized fraction outside integral to improve runtime
        preFact = unit.c * rs**2 / self.Cosmo.hubble(rs)    # (Nzpp,)
        one_minus_Q = 1.0 - self.Q_ion_interp(zpp)           # (Nzpp,)

        # Energy at z_pp
        E_pp = (rs[:, None] / (1.0 + z)) * E_arr[None, :]    # (Nzpp, NE)

        # 4) cross sections (Nzpp, NE))
        sig_HI  = self.photoions_xsec_Verner96(E_pp, "HI")
        sig_HeI = self.photoions_xsec_Verner96(E_pp, "HeI")

        # Write integrand for tau(E_arr,z,max(zp)) and solve the cumulative integral
        absorption = self.Cosmo.nH * sig_HI + self.Cosmo.nHe * sig_HeI           # (Nzpp, NE)
        integrand  = (preFact * one_minus_Q)[:, None] * absorption               # (Nzpp, NE)
        cum_tau = cumulative_trapezoid(integrand, zpp, axis=0, initial=0.0)      # (Nzpp, NE)

        # The cumulative integral at each zpp index is simply tau(E_arr,z,zpp)
        # Interpolate over zpp to get get tau(E_arr,z,zp)
        tau_2D = np.empty((len(zp_arr), len(E_arr)), dtype=float)
        for j in range(len(E_arr)):
            tau_2D[:, j] = np.interp(zp_arr, zpp, cum_tau[:, j])

        return tau_2D




    
    def JX(self, z, E_arr):
        # Compute the X-ray specific number intensity J_X(E, z) [1/eV/cm^2/s/sr] (Eq.26) 
        # at redshift z and energies E_arr [eV].

        E_arr = np.atleast_1d(E_arr)
        
        # Create array of zprime (redshift integration variable)
        Nz = max(2, int(np.ceil((np.log(self.zstar) - np.log(z)) * 100)))
        zp = np.exp(np.linspace(np.log(z), np.log(self.zstar), Nz))  # (Nz,)

        # broadcasting grids
        Z = zp[:, None]                    # (Nz, 1)
        E = E_arr[None, :]                 # (1, Ne)
        Eprime = E * (1.0 + Z) / (1.0 + z) # (Nz, Ne)

        # Integrate over zp
        prefact = unit.c / (4.0 * np.pi) * (1.0 + z)**2
        emiss = self.SpecificXrayNumberEmissivity(Z, Eprime)   # (Nz, Ne)
        tau   = self.optical_depth(z,zp,E_arr)                 # (Nz, Ne)
        
        integrand = emiss / self.Cosmo.hubble(1+zp)[:, None] * np.exp(-tau)       # (Nz, Ne)
        integral = np.trapz(integrand, zp, axis=0)                   # (Ne,)

        return prefact * integral


    def Heat_and_Ion_Rates(self,z,xHII_IGM):
        # Compute X-ray heating and hydrogen ionization volumetric rates in regions outside of UV ionized bubbles (Eqs.46 and 43).
        # xHII_IGM is the mean ionized fraction in the bulk IGM outside of UV ionized bubbles
        
       
        # Start by computing the u, the volumetric energy deposited by secondary events (Eq.36) for each specie.
        # This is an integral from Eth to infinity, but in practice can be take between 100eV to 2keV
        E_arr = np.arange(100,2000,100) 
        JX = self.JX(z,E_arr) # X-ray specific number flux [1/eV/cm^2/s/sr]

        nHI = self.Cosmo.nH * (1 + z) ** 3 * (1-xHII_IGM)   # number density of HI outside of UV ionized regions [cm^{-3}]
        xsec_HI = self.photoions_xsec_Verner96(E_arr, 'HI')      # pohotionization cross section with HI [cm^2]
        integrand_HI = (E_arr - self.Eth['HI']) * xsec_HI * JX    
        utot = 4 * np.pi * nHI * np.trapz(integrand_HI, E_arr)      # [eV/cm^3/s]

        nHeI = self.Cosmo.nHe * (1 + z) ** 3 * (1-xHII_IGM) # number density of HeI outside of UV ionized regions [cm^{-3}]
        xsec_HeI = self.photoions_xsec_Verner96(E_arr, 'HeI')    # pohotionization cross section with HeI [cm^2]
        integrand_HeI = (E_arr - self.Eth['HeI']) * xsec_HeI * JX
        utot += 4 * np.pi * nHeI * np.trapz(integrand_HeI, E_arr)   # [eV/cm^3/s]

        if self.include_HeII:
            nHeII = self.Cosmo.nHe * (1 + z) ** 3 * xHII_IGM    # number density of HeI outside of UV ionized regions [cm^{-3}]
            xsec_HeII = self.photoions_xsec_Verner96(E_arr, 'HeII')    # pohotionization cross section with HeII [cm^2]
            integrand_HeII = (E_arr - self.Eth['HeII']) * xsec_HeII * JX
            utot += 4 * np.pi * nHeII * np.trapz(integrand_HeII, E_arr)   # [eV/cm^3/s]

        # Fraction of secondary energy going into each channel
        fheat = pre.fheat_interp(xHII_IGM)
        fion = pre.fion_interp(xHII_IGM)
        
        # Compute hydrogen ionization rate per unit volume due to primary and secondary interactions. 
        integrand_HI = xsec_HI * JX
        PrimaryIonRate = 4 * np.pi * nHI * np.trapz(integrand_HI, E_arr)   # [1/cm^3/s] primary ionizations only
        SecondaryIonRate = utot * fion / self.Eth['HI']  # [1/cm^3/s] Secondary ionizations
        IonRate = SecondaryIonRate + PrimaryIonRate # [1/cm^3/s] total ionization rate
        
        # Compute heating rate per unit volume
        HeatRate = utot*fheat
                
        return HeatRate, IonRate #[eV / cm^3 / s], [1 / cm^3 / s]


    def heat_and_ion_rate_grid_interpolation_funcions(self):
        # Construct 2D interpolation functions for X-ray heating ionization rates (Eqs.46 and 43) 
        # f(log10(x_e), z)

        xe_arr=np.logspace(np.log10(self.xe_min),np.log10(self.xe_max),self.xe_len) #xe's we want to scan
        z_arr = np.linspace(self.zmin,self.zstar-1,self.zlen)

        heatgrid = np.zeros((self.xe_len,self.zlen))
        reiongrid = np.zeros((self.xe_len,self.zlen))

        for i,z in enumerate(z_arr):
            heat_ion = self.Heat_and_Ion_Rates(z,xe_arr)
            heatgrid[:,i] = heat_ion[0]
            reiongrid[:,i] = heat_ion[1]

        # This assume the grid is really regular (equaly space in both log10(xe_arr) and z_arr). If this changes use sorted grid instead        
        self.HeatRate_interp = reg_grid_interp(heatgrid,np.log10(xe_arr) ,z_arr).interp2D_single
        self.ReionRate_interp =reg_grid_interp(reiongrid,np.log10(xe_arr),z_arr).interp2D_single

        return ([self.HeatRate_interp,self.ReionRate_interp]) # [eV / cm^3 / s], [1 / cm^3 / s]


    def CXB(self, zX, Emin, Emax, attenuate = False, NH = 1e20, fmol = 0.2):
        """
        Compute the CXB intensity in a given observed energy band sourced by z>zX sources.

        Parameters
        ----------
        zX : float
            Minimal source redshift. Only emission from
            z ≥ zX contributes to the unresolved CXB.
        Emin : float
            Lower bound of the observed energy band [keV].
        Emax : float
            Upper bound of the observed energy band [keV].
        attenuate : bool, optional
            If True, include attenuation by the IGM and by Milky Way
            absorption. 
        NH : float, optional
            Hydrogen column density of the Milky Way [cm^-2] used for
            ISM attenuation. 
        fmol : float, optional
            Molecular hydrogen fraction used in the Milky Way absorption
            model. 

        Returns
        -------
        IX: float
            CXB intensity in the observed band [keV / cm^2 / s / sr].
        """

        # Create energy array over observed band
        E_arr = 1000 * np.linspace(Emin, Emax, int(np.ceil(Emax-Emin))*50) # [eV]
        
        # Create array for all redshifts that contribute to unresolved X-rays
        z_arr = np.logspace(np.log10(zX), np.log10(self.zstar), 250)  #

        # Compute emissivity for each (E, z) pair
        integrand = np.zeros((len(E_arr), len(z_arr))) 
        if attenuate:
            for i, E in enumerate(E_arr):
                E_z = E * (1 + z_arr)  # [eV]
                tau_IGM = self.optical_depth(0, z_arr, E)
                tau_IGM = np.squeeze(tau_IGM)
                tau_ISM = MW.tau_MW(E/1000,fmol,NH)

                eps_x_arr = self.SpecificXrayNumberEmissivity(z_arr,E_z)*np.exp(-tau_IGM)*np.exp(-tau_ISM)
                integrand[i, :] = eps_x_arr / self.Cosmo.hubble(1+z_arr) # [1/eV/cm^3]
        else:
            for i, E in enumerate(E_arr):
                E_z = E * (1 + z_arr)  # [eV]
                eps_x_arr = self.SpecificXrayNumberEmissivity(z_arr,E_z)
                integrand[i, :] = eps_x_arr / self.Cosmo.hubble(1+z_arr) # [1/eV/cm^3]
        
        # Integrate over z first
        integral_over_z = np.trapz(integrand, z_arr, axis=1)

        # Integrate over E
        final_integral = np.trapz(E_arr * integral_over_z, E_arr) / (4 * np.pi) # [eV/cm^3/sr]

        return unit.c*final_integral/1000 # [keV/cm^2/s/sr]


    # #No optical depth
    # def CXB_notau(self, z_un, Emin, Emax):
    #     """
    #     z_un - Maximal redshift of resolved sources
    #     Emin, Emax - Range of observed band [keV]
    #     """

    #     # Create energy array over observed band
    #     E_arr = 1000 * np.linspace(Emin, Emax, int(np.ceil(Emax-Emin))*100) # [eV]
    #     # Create redshift for all redshifts that contribute to unresolved X-rays
    #     z_arr = np.logspace(np.log10(z_un), np.log10(self.zstar), 1000)  #

    #     # Compute epsilon_x(E(1+z), z) / H(z) for each (E, z) pair
    #     integrand = np.zeros((len(E_arr), len(z_arr))) 
    #     for i, E in enumerate(E_arr):
    #         E_z = E * (1 + z_arr)  # [eV]
    #         eps_x_arr = self.SpecificXrayNumberEmissivity(z_arr,E_z)
    #         integrand[i, :] = eps_x_arr / unit.hubble(1+z_arr) # [1/eV/cm^3]
    #     # Integrate over z first
    #     integral_over_z = np.trapz(integrand, z_arr, axis=1)

    #     # Integrate over E
    #     final_integral = np.trapz(E_arr * integral_over_z, E_arr) / (4 * np.pi) # [eV/cm^3/sr]

    #     return unit.c*final_integral/1000 # [keV/cm^2/s/sr]



    


   