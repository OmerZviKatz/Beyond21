import numpy as np
import beyond21.constants as unit

class UVLFs:
    
    def __init__(self, sfrd_ion_uv):
        self.SF = sfrd_ion_uv

    def Mean_MUV_from_Mh(self,Mh,z,kUV,pop):
        if pop == 'II':
            SFR = self.SF.SFRII(Mh, z)/unit.M_s/unit.Sec*unit.Year # [Msolar/yr]
        elif pop == 'III':
            SFR = self.SF.SFRIII(Mh, z)/unit.M_s/unit.Sec*unit.Year
        LUV = SFR/kUV #erg/s/Hz
        MUV = 51.63 - 2.5 * np.log10(LUV) #mag
        return MUV

    def UVLF_gaus_cont(self,z,Muv,sigma_MUV,pop,Mh = None, kUV=1.15e-28):
            """
            UVLF assuming a Gaussian PDF for MUV at fixed Mh, and no binning in MUV.

            Parameters
            ----------
            z : redshift
            Muv : UV magnitude grid where phi is evaluated.
            Mh : Halo mass grid [Msun].
            sigma_MUV : Scatter in magnitudes.
            
            Returns
            -------
            phi(z,Muv) : UV luminosity function [Mpc^-3 mag^-1]
            """

            Muv = np.asarray(Muv, float)
            if Mh is not None:
                Mh  = np.asarray(Mh, float)
            else:
                Mhlen = max(1000,len(Muv)/sigma_MUV*5)
                Mh = np.logspace(5,15,int(Mhlen))

            #---- compute fgal(Mh,z)*dndlnm(Mh,z) -----
            McutII = self.SF.Mcut_eV(z,'II')
            if pop == 'II':
                fgal = self.SF.fgalII(Mh*unit.M_s, McutII)
            elif pop == 'III':
                McutIII = self.SF.Mcut_eV(z,'III')
                fgal = self.SF.fgalIII(Mh*unit.M_s, McutII, McutIII)
            
            dndlMh = self.SF.Cosmo.dndlnm(Mh*unit.M_s, z)  # [cm^-3]
            fact = dndlMh * fgal * unit.cm3_to_Mpc3   # (Nm,)

            # ----- mean MUV(Mh) -----
            mu = self.Mean_MUV_from_Mh(Mh*unit.M_s, z, kUV, pop)  # (Nm,)

            # ----- Gaussian PDF -----
            mu  = mu[:, None]     # (Nm,1) reshape for broadcasting
            Muv = Muv[None, :]    # (1,NMUV)

            inv_norm = 1.0 / (np.sqrt(2*np.pi) * sigma_MUV)
            P = inv_norm * np.exp(-0.5 * ((Muv - mu)/sigma_MUV)**2) # shape (Nm, NMUV)
            
            # ----- integrate over halo mass -----
            logMh = np.log(Mh)
            phi = np.trapz(fact[:, None] * P, logMh, axis=0)

            if pop == 'III' and self.photoheat==True: # apply photoheating suppression to Pop III galaxies
                return phi*(1-self.SF.Q_interp(z))
            return phi

