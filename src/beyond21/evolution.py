import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import warnings

import beyond21.constants as consts
import beyond21.sfrd_ion_uv as fobj
import beyond21.xrays as Xobj
import beyond21.interpolations as pre
import beyond21.non_ion_uv as Uobj
import beyond21.igm_mdm.inter_galactic_medium as IGM
import beyond21.igm_mdm.mdm as IGM_mDM


class evolver():
    
    def __init__(self, cosmo, star_formation_params = None, xray_params = None, reion_params = None, DM_params = None, photoheat = True, Lya_Heat = True, CMB_Heat = True):
        self.cosmo = cosmo
        self.Pop = star_formation_params.get('model') #Stellar populations of interest ('PopII' / 'PopIII' / 'PopII+PopIII')
        self.CMB_Heat = CMB_Heat
        self.Lya_Heat = Lya_Heat

        self.generate_UVion_SFRD_interps(reion_params, star_formation_params, photoheat)
        self.generate_Xrays_heat_ion_interps(xray_params)
        self.generate_stellar_Lya_interps(reion_params)
        if DM_params != None:
            self.init_IDM(DM_params)
        self.Ts_prev = 0
    
    def generate_UVion_SFRD_interps(self,reion_params,star_formation_params, photoheat):
        sfr_ion = fobj.SFRD_UVion(self.cosmo, star_formation_params,reion_params, photoheat = photoheat)
        self.sfr_ion = sfr_ion
        self.SFRD_interp, self.Q_ion_interp = sfr_ion.SFRD_and_Qion_interp()

    def generate_Xrays_heat_ion_interps(self,xray_params):
        self.xrays = Xobj.XrayHeatingReion(
            self.cosmo,xray_params, self.SFRD_interp, Q_ion_interp = self.Q_ion_interp, 
            populations = self.Pop, zstar = 50, zmin = 1, xe_max = 0.99999, xe_min = 1e-5, 
            zlen = 50, xe_len = 25, include_HeII = False 
        )
        self.HeatRate_interp, self.ReionRate_interp = self.xrays.heat_and_ion_rate_grid_interpolation_funcions()
        return

    def generate_stellar_Lya_interps(self, reion_params): 
        # Only stellar Ly-alpha no secondary from X-rays
        self.lya = Uobj.NonIonUV(self.cosmo, reion_params, populations = self.Pop)

        z_arr_for_Jalpha = np.linspace(1,50,150)
        J_alpha_arr = self.lya.Jalpha_star(z_arr_for_Jalpha, self.SFRD_interp)
        self.Jalphastar_interps =[interp1d(z_arr_for_Jalpha,J_alpha_arr[0]),interp1d(z_arr_for_Jalpha,J_alpha_arr[1]),interp1d(z_arr_for_Jalpha,J_alpha_arr[2])]            
        return 

    def init_IDM(self,DM_params):
        self.DM_params = DM_params
        
        if DM_params == None:
                #No DM-SM interactions
                self.IDM = False
                self.CDM = False
                self.mm, self.sigma_e, self.f_m, self.mC, self.alphaI_alphaC = np.ones(7)

        elif DM_params.get('mC') == None:
            #IDM-SM interactions but no IDM-CDM interactions
            self.IDM = True
            self.CDM = False
            self.mm = DM_params['mm']
            Q = DM_params['Q']
            self.f_m = DM_params['f_m']
            red_mass_eI = consts.m_e * self.mm / (consts.m_e + self.mm)
            self.sigma_e = Q**2 * 2 * 16 * np.pi * consts.alphaEM ** 2 * red_mass_eI ** 2 / (consts.alphaEM ** 4 * consts.m_e ** 4)
            self.mC = 1e7 #mC and alphaI_alphaC won't be used because CDM = False
            self.alphaI_alphaC =1e-20
            self.m_phi = DM_params['m_phi'] if ('m_phi' in DM_params) else 1e-6

        else:
            #IDM-SM interactions and IDM-CDM interactions 
            self.IDM = True
            self.CDM = True
            self.mm = DM_params['mm']
            Q = DM_params['Q']
            self.f_m = DM_params['f_m']
            self.mC = DM_params['mC']
            alphaI_alphaC = DM_params['alphaI_alphaC']
            red_mass_eI = consts.m_e * self.mm / (consts.m_e + self.mm)
            self.sigma_e = Q**2 * 2 * 16 * np.pi * consts.alphaEM ** 2 * red_mass_eI ** 2 / (consts.alphaEM ** 4 * consts.m_e ** 4)
            self.m_phi = DM_params['m_phi'] if ('m_phi' in DM_params) else 1e-6
            
            def max_alphaI_alphac(m_I, m_C, f, sigmabar_e, CDM):
                # Calculates the maximum alpham_alphac value according to all bounds, given the input parameters.
                max_CMB= 7.4 * pow(10 ,-19) * (m_I / 1e9) ** 2 * (m_C / 1e8) ** 2 * (1e9 / (m_C + m_I)) * (1 + self.cosmo.Omega_b / (f * self.cosmo.Omega_DM)) 
                max_SI = np.sqrt(pow(10, -11) * (m_C / 1e9) ** 3)              
                max_tight_coupling = (0.08 * np.sqrt(Q) * np.sqrt(m_C / 1e8) * (1e9 / m_I) ** 0.25) ** 4 * consts.alphaEM ** 2  
                return np.min((max_SI*np.ones_like(max_CMB), max_CMB, max_tight_coupling))

            if DM_params['alphaI_alphaC'] == 'max':
                self.alphaI_alphaC = max_alphaI_alphac(self.mm, self.mC, self.f_m, self.sigma_e,self.CDM)
            else:
                self.alphaI_alphaC = alphaI_alphaC

    def EvolveIGM(self, z_min = 10, z_max = 1200, Nz = 250, ivp_kwargs = None):
        
        # Time array for ODE solver - we evolve in log(a), a is the scale factor.
        if z_max < 1200:
            raise ValueError("Evolution must start at z_max>=1200")

        log_a = np.linspace(np.log(1/(1+z_max)),np.log(1/(z_min+1)),Nz) 

        # Set initial conditions
        init_TCMB = self.cosmo.TCMB(1+z_max)          # CMB temperature (eV)
        init_Tb = 0.99999*init_TCMB    # Baryons kinetic temperature (eV)
        init_xHII = 0.999                          #Ionized hydrogen fraction
        init_cond_array = [init_TCMB-init_Tb,init_xHII]
                
        # solve_ivp kwargs
        defaults = {"rtol": 1e-4, "max_step": 0.01}
        opts = {**defaults, **(ivp_kwargs or {})}

        # Solve evolution
        soln = solve_ivp(IGM.ODEs_SM, [log_a[0], log_a[-1]],init_cond_array, method='BDF',t_eval=log_a,
                         args=(self.cosmo, self.Jalphastar_interps, self.lya.Jalpha_X,self.SFRD_interp, self.Q_ion_interp, self.HeatRate_interp, self.ReionRate_interp, 
                               self.Lya_Heat, self.CMB_Heat), **opts)
        
        #Organize output
        self.rs = self.cosmo.rs_from_log_a(soln['t'])            # Redshifts (1+z) for which we solved
        self.TCMB = self.cosmo.TCMB(self.rs)/consts.kB             #CMB temperature [K]
        soln_vec = np.transpose(soln['y'])
        Delta_CMB_b_arr = soln_vec[:, 0]/consts.kB           # CMB-baryon temperature [K]
        self.Tbaryon = self.TCMB - Delta_CMB_b_arr         # baryon temperature [K] 
        self.xHII_IGM = soln_vec[:, 1]                     # xHI in the IGM (outside of ionized bubbles)
        if self.Q_ion_interp == None:
            self.FillingFact = np.ones_like(self.rs)*1e-10 # Volume filling fraction of fully ionized bubbles - set to 0 in this case
        else:
            self.FillingFact = self.Q_ion_interp(self.rs-1)    
        self.xHII = self.FillingFact + (1-self.FillingFact) * self.xHII_IGM # Averaged xHII (outside + in ionized bubbles)
        self.xHI = 1-self.xHII                                              # Averaged xHI
        
        #T21 evolution
        self.Tspin, self.T21 = self.T21Evolution()

        #CMB optical depth to reionization
        self.tau_e()
        return

    def EvolveIGM_2cDM(self, z_min = 10, z_max = 1200, Nz = 250, ivp_kwargs = None):
        
        # Time array for ODE solver - we evolve in log(a), a is the scale factor.
        if z_max < 1200:
            raise ValueError("Evolution must start at z_max>=1200")

        log_a = np.linspace(np.log(1/(1+z_max)),np.log(1/(z_min+1)),Nz) 

        # Set initial conditions
        init_TCMB = self.cosmo.TCMB(1+z_max)          # CMB temperature (eV)
        init_Tb = 0.99999*init_TCMB                   # Baryons kinetic temperature (eV)
        init_T_IDM = 0.99999*init_Tb                  # IDM temperature (eV)
        init_T_CDM = 1e-6                             # CDM temperature (eV)
        init_V_baryon_IDM = 1e-5 / consts.c           # baryon-IDM relative velocity in c
        init_V_IDM_CDM = 29 * 1e5 / consts.c          # IDM-CDM relative velocity in c
        init_xHII = 0.999                             #Ionized hydrogen fraction
        init_cond_array = [init_TCMB-init_Tb, init_xHII, init_Tb - init_T_IDM, np.log(init_V_baryon_IDM), np.log(init_T_CDM), np.log(init_V_IDM_CDM)] 

        # solve_ivp kwargs
        defaults = {"rtol": 1e-4, "max_step": 0.01}
        opts = {**defaults, **(ivp_kwargs or {})}

        # Solve evolution
        soln = solve_ivp(IGM_mDM.ODEs_2cDM, [log_a[0], log_a[-1]],init_cond_array, method='BDF',t_eval=log_a,
                         args=(self.cosmo, self.Jalphastar_interps, self.lya.Jalpha_X, self.Q_ion_interp, self.HeatRate_interp, self.ReionRate_interp,
                         self.Lya_Heat,self.CMB_Heat, self.f_m, self.mm, self.mC, self.sigma_e, self.alphaI_alphaC, self.m_phi, self.IDM, self.CDM), **opts)

            
        #Organize output
        self.rs = self.cosmo.rs_from_log_a(soln['t'])            # Redshifts (1+z) for which we solved
        self.TCMB = self.cosmo.TCMB(self.rs)/consts.kB           # CMB temperature [K]
        soln_vec = np.transpose(soln['y'])
        Delta_CMB_b_arr = soln_vec[:, 0]/consts.kB           # CMB-baryon temperature [K]
        Delta_baryon_IDM = soln_vec[:, 2]/consts.kB          # Baryon-IDM temperature [K]
        self.Tbaryon = self.TCMB - Delta_CMB_b_arr         # Baryon temperature [K] 
        self.TIDM = self.Tbaryon - Delta_baryon_IDM        # IDM temperature [K]
        self.TCDM = np.exp(soln_vec[:,4])/consts.kB          # CDM temperature [K]
        self.xHII_IGM = soln_vec[:, 1]                     # xHI in the IGM (outside of ionized bubbles)
        self.V_mb_arr=np.exp(soln_vec[:, 3])*consts.c/1e5  # V_mb [km/s]
        self.V_mc_arr=np.exp(soln_vec[:, 5])*consts.c/1e5  # V_mc [km/s]

        if self.Q_ion_interp == None:
            self.FillingFact = np.ones_like(self.rs)*1e-10 # Volume filling fraction of fully ionized bubbles - set to 0 in this case
        else:
            self.FillingFact = self.Q_ion_interp(self.rs-1)    
        self.xHII = self.FillingFact + (1-self.FillingFact) * self.xHII_IGM # Averaged xHII (outside + in ionized bubbles)
        self.xHI = 1-self.xHII                                              # Averaged xHI
        
        #T21 evolution
        self.Tspin, self.T21 = self.T21Evolution()

        #CMB optical depth to reionization
        self.tau_e()
        return

    def T21Evolution(self):
        '''
        Returns T21,Ts evolutions according to the xe, TK evolutions of the object 
        '''
        # Densities evolution
        nH = self.cosmo.nH*self.rs**3 # Mean hydrogen density [cm^-3]
        nHI = nH * self.xHI 
        nHII = nH * self.xHII
        ne = np.where(self.rs <=51, nHII * (self.cosmo.nH + self.cosmo.nHe)/self.cosmo.nH, nHII) # Mean electron density [cm^-3]. Assume astrophysic ionization (below z<50) is similar for HI, HeI

        #Find the index in of the self.rs element that is closest value to z = 50 (above rs=50 we assume Lyalpha and X-rays are zero)
        max_star_rs_index = (np.abs(self.rs - 50)).argmin() + 1

        #Lyman alpha fluxes. Zero for z>50
        Jalphastar = np.zeros(len(self.rs))
        Jalphastar[max_star_rs_index:] = self.Jalphastar_interps[0](self.rs[max_star_rs_index:]-1)

        # Lyman alpha from X-ray photo-ionization secondaries
        eps_X_heat = np.zeros(len(self.rs))
        for valid_idx in range(max_star_rs_index, len(self.rs)):
            eps_X_heat[valid_idx] = self.HeatRate_interp([np.log10(self.xHII_IGM[valid_idx]), self.rs[valid_idx]-1]) 
        JalphaX = np.zeros(len(self.rs))
        JalphaX[max_star_rs_index:] = self.lya.Jalpha_X(eps_X_heat[max_star_rs_index:],self.xHII_IGM[max_star_rs_index:], self.rs[max_star_rs_index:]-1)
        
        self.Jalpha = JalphaX + Jalphastar

        #Calculate Tspin and T21
        Tspin = IGM.Ts_calc(self.cosmo.hubble(self.rs), self.Tbaryon, self.xHI, self.Jalpha, self.rs-1, nHI, nHII, ne, self.TCMB)
        T21 = IGM.T21calc(self.cosmo,self.rs-1,Tspin,self.xHI, nH, self.TCMB)
        return [Tspin,T21]


    def tau_e(self):
        #return CMB optical depth to reionization
        if not (hasattr(self, 'xHII') and self.xHII.any()):
            raise ValueError("Cannot determine CMB optical depth to reionization because xHII data is not available. Please ensure that 'EvolveSM' has been executed successfully before calling 'verify_tau'.")
        if self.rs[-1] > 7:
            warnings.warn("CMB optical depth may be unreliable for z_min > 6 in models with late reionization.", UserWarning)
        if self.rs[0] < 35:
            warnings.warn("CMB optical depth may be unreliable for z_max < 35.", UserWarning)
        z_arr = np.linspace(0,50,250)
        xHII_interp = interp1d(self.rs-1,self.xHII,fill_value=(1, 0), bounds_error=False)
        xe = xHII_interp(z_arr)*(1+self.cosmo.nHe/self.cosmo.nH)
        integrand = (1+z_arr)**2*xe/self.cosmo.hubble(1+z_arr)
        self.tau = self.cosmo.nH*consts.c*consts.Thomson_xsec/consts.Centimeter**2*np.trapz(integrand,z_arr)
        return
            
