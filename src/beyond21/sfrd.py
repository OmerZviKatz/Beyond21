import numpy as np
from numpy import trapz
from importlib.resources import files
import beyond21.constants as consts


class SFRD:
    
    def __init__(self, cosmo, SF_params):
        self.cosmo = cosmo
        self.SF_params = SF_params
        
        if 'A_LW' in SF_params: #Run with LW feedback
            self.LW_feedback = True 
            self.JLW_interp = False
            self.fmod_interp = np.load(files("beyond21").joinpath("data", "fmod/fmod_interp_Ahn.npy"),allow_pickle = True).item() 
        else:
            self.LW_feedback = False

        if 'A_vrel' in SF_params: #Run with vrel feedback
            self.vrel_feedback = True 
        else:
            self.vrel_feedback = False

        
    def Mcut_eV(self,z, pop, JLW = None):
        
        Matom = 3.3 * 1e7 * consts.M_s * ((1 + z) / 21) ** (-3 / 2)  
        if pop == 'II':
            Mcut0 = self.SF_params['M_cutII']
        if pop == 'III':
            Mcut0 = self.SF_params['M_cutIII']
        
        if Mcut0 == 'Matom':
            return Matom
        
        Mcut0_eV = Mcut0 * consts.M_s
        
        if Mcut0_eV> Matom:
            return Mcut0_eV
        
        Mmol = 3.3 * 1e7 * consts.M_s * (1 + z) ** (-3 / 2) 
        
        if self.LW_feedback:
            if self.JLW_interp:
                LW_supression = 1 + self.SF_params['A_LW'] * self.JLW_interp(z)**self.SF_params['B_LW']
            elif JLW:
                LW_supression = 1 + self.SF_params['A_LW'] * JLW**self.SF_params['B_LW']
            else:
                raise ValueError("When LW_feedback = True McutIII must be run with JLW as input.")
            Mmol = Mmol * LW_supression

        if self.vrel_feedback:
            vrel_supression = (1 + self.SF_params['A_vrel']*0.92)**self.SF_params['B_vrel'] 
            Mmol = Mmol * vrel_supression
        
        if Mcut0_eV >= Mmol:
            return Mcut0_eV

        return Mmol


    def MstarII(self,Mh, z):
        ''' 
        Return: Mstar [eV]
        '''        
        Mpivot = self.SF_params['Mpivot']
        alphaII = self.SF_params['alphaII']
        betaII = self.SF_params['betaII']
        Fstar = self.SF_params['F_starII']

        # Double PL 
        den = (Mh/consts.M_s/Mpivot)**(alphaII) + (Mh/consts.M_s/Mpivot)**(betaII)
        fstar = Fstar/den
        fstar[fstar>1] = 1
        return self.cosmo.Omega_b/self.cosmo.Omega_m * fstar * Mh

    
    
    def MstarIII(self,Mh, z):
        temp = self.SF_params['F_starIII'] * pow(Mh/1e7/consts.M_s, -self.SF_params['alphaIII'])
        temp[temp>1] = 1
        fstarIII = temp
        return self.cosmo.Omega_b/self.cosmo.Omega_m*fstarIII*Mh

    
    def SFRII(self,Mh, z):
        # Mh [eV]
        'Return: SFR(z) [eV/s]'
        timescale = self.SF_params['eps_t'] / self.cosmo.hubble(1+z)
        MstarII = self.MstarII(Mh, z)
        return MstarII/timescale

    def SFRIII(self,Mh, z):
        # Mh [eV]
        'Return: SFR(z) [eV/s]'
        timescale = self.SF_params['eps_t'] / self.cosmo.hubble(1+z)
        MstarIII = self.MstarIII(Mh, z)
        return MstarIII/timescale

    def fgalII(self, Mh, McutII):
        '''
            Mh [eV], McutII [eV]'''
        return np.exp(-McutII/Mh)
    
    def fgalIII(self, Mh, McutII, McutIII):
        '''
            Mh [eV], McutII [eV], McutIII [eV]'''
        return np.exp(-McutIII/Mh) * np.exp(-Mh/McutII)

    def SFRDII_calc(self, z, Mh_arr, dndlnm, McutII):
        'Return: SFRD(z) [eV/cm^3/s]'
        lnMh_arr = np.log(Mh_arr)
        SFRII_arr = self.SFRII(Mh_arr, z)
        fgalII_arr = self.fgalII(Mh_arr,McutII)
        dndlnm_StarForming = dndlnm*fgalII_arr
        SFRDII = trapz(SFRII_arr * dndlnm_StarForming , lnMh_arr) #eV/cm^3
        return SFRDII 

    def SFRDIII_calc(self, z, Mh_arr, dndlnm, McutII, McutIII):
        'Return: SFRD(z) [eV/cm^3/s]'
        lnMh_arr = np.log(Mh_arr)
        SFRIII_arr = self.SFRIII(Mh_arr, z)
        fgalIII_arr = self.fgalIII(Mh_arr,McutII,McutIII)
        dndlnm_StarForming = dndlnm*fgalIII_arr
        SFRDIII = trapz(SFRIII_arr * dndlnm_StarForming , lnMh_arr) #eV/cm^3
        return SFRDIII
    

    def Mean_MUV_from_Mh(self,Mh,z,kUV,pop):
        if pop == 'II':
            SFR = self.SFRII(Mh, z)/consts.M_s/consts.Sec*consts.Year # [Msolar/yr]
        elif pop == 'III':
            SFR = self.SFRIII(Mh, z)/consts.M_s/consts.Sec*consts.Year
        LUV = SFR/kUV #erg/s/Hz
        MUV = 51.63 - 2.5 * np.log10(LUV) #mag
        return MUV
    

    def UVLF_Stoch_continuous(self,z,Muv,sigma_MUV,pop,Mh = None, kUV=1.15e-28):
        """
        Continuous stochastic UVLF (no binning).

        Parameters
        ----------
        Muv : (NMUV,) array
            UV magnitudes where phi is evaluated.
        Mh : (Nm,) array
            Halo mass grid [Msun].
        sigma_MUV : float
            Scatter in magnitudes.
        
        Returns
        -------
        phi : (NMUV,) array
            UV luminosity function [Mpc^-3 mag^-1]
        """

        Muv = np.asarray(Muv, float)
        if Mh is not None:
            Mh  = np.asarray(Mh, float)
        else:
            Mhlen = max(1000,len(Muv)/sigma_MUV*5)
            Mh = np.logspace(5,15,int(Mhlen))

        # ----- duty cycle -----
        McutII = self.Mcut_eV(z,'II')

        if pop == 'III':
            McutIII = self.Mcut_eV(z,'III')
            fgal = self.fgalIII(Mh*consts.M_s, McutII, McutIII)
        else:
            fgal = self.fgalII(Mh*consts.M_s, McutII)

        # ----- HMF -----
        dndlMh = self.cosmo.dndlnm(Mh*consts.M_s, z)  # [cm^-3]

        # convert to Mpc^-3
        cm3_to_Mpc3 = (consts.Mpc / consts.Centimeter)**3
        fact = dndlMh * fgal * cm3_to_Mpc3   # (Nm,)

        # ----- mean MUV(Mh) -----
        mu = self.Mean_MUV_from_Mh(Mh*consts.M_s, z, kUV, pop)  # (Nm,)

        # reshape for broadcasting
        mu  = mu[:, None]     # (Nm,1)
        Muv = Muv[None, :]    # (1,NMUV)

        # ----- Gaussian PDF -----
        inv_norm = 1.0 / (np.sqrt(2*np.pi) * sigma_MUV)

        P = inv_norm * np.exp(-0.5 * ((Muv - mu)/sigma_MUV)**2)
        # shape (Nm, NMUV)

        # ----- integrate over halo mass -----
        logMh = np.log(Mh)
        phi = np.trapz(fact[:, None] * P, logMh, axis=0)

        if pop == 'III' and self.photoheat==True:
            return phi*(1-self.Q_interp(z))
        return phi


    
    # def UVLF_Stoch(self,z,Muv_edges,sigma_MUV, pop, Mh= None, kUV = 1.15*pow(10,-28), pts_per_bin=25, normalize_P=False):
    #     """
    #     Parameters
    #     ----------
    #     Muv_edges [mag]: (Nb+1,) array
    #         Magnitude bin edges.
    #     Mh [Msolar]: (Nm,) array
    #         Halo mass grid (increasing).
    #     pts_per_bin : int
    #         Number of integration points per magnitude bin.
    #     normalize_P : bool
    #         If True, normalizes P over the full Muv range spanned by Muv_edges, per Mh.

    #     Returns
    #     -------
    #     Muv_centers : (Nb,) array
    #     phi : (Nb,) array  [Mpc^-3 mag^-1]
    #     """

    #     Muv_edges = np.asarray(Muv_edges, float)
    #     if Mh:
    #         Mh  = np.asarray(Mh, float)
    #     else:
    #         Mhlen = max(1000,len(Muv_edges)*pts_per_bin*10,len(Muv_edges)/sigma_MUV*5)

    #         Mh = np.logspace(5,15,int(Mhlen))

    #     Nb = len(Muv_edges) - 1


    #     McutII = self.Mcut_eV(z,'II')
    #     if pop == 'III':
    #         McutIII = self.Mcut_eV(z,'III')
    #         fgal = self.fgalIII(Mh*consts.M_s, McutII, McutIII)
    #     elif pop == 'II':
    #         fgal = self.fgalII(Mh*consts.M_s, McutII)

    #     dndlMh = self.dndlnm(Mh*consts.M_s, z) * (consts.Mpc / consts.Centimeter)**3# (Nm,) [Mpc^-3]
    #     fact = dndlMh * fgal 
    #     fact = fact[:, None] # Broadcase to (Nm,1) 


    #     # Mean UV magnitude at each halo mass
    #     mu = self.Mean_MUV_from_Mh(Mh*consts.M_s,z,kUV,pop)          # (Nm,)
    #     mu = mu[:, None]                                # (Nm,1)

    #     # Bin edges reshaped for broadcasting
    #     Mlo = Muv_edges[:-1][None, :]                   # (1,Nb)
    #     Mhi = Muv_edges[1:][None, :]                    # (1,Nb)

    #     # Analytic probability mass in each bin for each Mh
    #     inv = 1.0 / (np.sqrt(2.0) * sigma_MUV)
    #     Prob = 0.5 * (erf((Mhi - mu) * inv) - erf((Mlo - mu) * inv))   # (Nm,Nb)

    #     # Integrate over Mh for each bin
    #     logMh = np.log(Mh)
    #     # dMh = (ln 10) Mh dlog10Mh
    #     n_bins = np.trapz(fact * Prob, logMh, axis=0)
        

    #     dMUV = np.diff(Muv_edges)
    #     phi = n_bins / dMUV
    #     Muv_centers = 0.5 * (Muv_edges[:-1] + Muv_edges[1:])
    #     return Muv_centers, phi # [mag], [Mpc^-3 mag^-1]