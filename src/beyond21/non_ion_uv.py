import numpy as np
import beyond21.constants as unit
import beyond21.interpolations as pre

class NonIonUV:

    def __init__(self, CosmoObj, reion_params, populations = False):
        self.Cosmo = CosmoObj
        self.populations = populations
        self.NionII = reion_params['N_ionII']
        if populations == 'PopII+PopIII':
            self.NionIII = reion_params['N_ionIII']

    def z_maxn(self,z, n):
        # The maximum emission redshift of a photon observed at redshift z in the Lyman-n transition
        z_max = (1 + z) * (1 - (n + 1)**(-2)) / (1 - 1 / n ** 2) - 1
        return (z_max)

    def Freq_redshift(self, z, ztag, n):
        # Frequency [Hz] of a photon at redshift z that was emitted at redshift ztag via the n -> 1 Lyman de-excitation
        Ryd = 13.6 / unit.Planck  # Rydberge constant in Hz
        nu_n = Ryd * (1 - 1 / n ** 2)  
        nu_n_tag = nu_n * (1 + ztag) / (1 + z)
        return (nu_n_tag)

    def eps_star(self, nu, z, SFRD_interp):
        """
        Compute the specific Lyman-band number emissivity eps(nu, z) in units of [1 / cm^3 / Hz / s], see Eq.25.

        Parameters
        ----------
        nu : Emission frequencies [Hz].
        z : Emission redshift.
        SFRD_interp : 
            Interpolation function(s) for the Star Formation Rate Density [eV/sec/cm^3].
            If populations == 'PopII+PopIII', should be a tuple (SFRD_PopII, SFRD_PopIII).
        populations : 'PopII', 'PopIII', or 'PopII+PopIII'.
            
        Returns
        -------
        emissivity : ndarray
            eps(nu, z) in units of [1 / cm^3 / Hz / s].
        """

        nu = np.atleast_1d(nu)
        z = np.atleast_1d(z)

        #Determine the closest Lyman-level n below nu. This is the Lyman-level of absorption.
        Ryd = 13.6 / unit.Planck  # Rydberge constant in Hz
        n_lyman = np.floor(np.sqrt(1 / (1 - nu / Ryd))).astype(int)

        # Keep only n>=2 and n<=23 (see Eq. 27)
        valid_mask = (n_lyman >= 2) & (n_lyman <= 23)
        idx_valid = n_lyman[valid_mask] - 2  # n-2. Index for spectral tables. Tables start from n=2.


        def dNdnu_spectrum(Nion_i, norm_col, alpha_col):
            #Compute the spectral photon number density dN/dnu for a given stellar population using the borken power-law in spectral_distribtuion.
            #Tabulation follows dN/dnu = Norm * Nion / nu_alpha * (nu / nu_alpha)^(alpha - 1) 

            nu_alpha = unit.Freq_Lya
            dNdnu = np.zeros_like(n_lyman, dtype=float)
            norm = pre.spectral_distribution[norm_col][idx_valid] * Nion_i / nu_alpha
            alpha = pre.spectral_distribution[alpha_col][idx_valid]
            dNdnu[valid_mask] = norm * np.power(nu[valid_mask] / nu_alpha, alpha - 1)
            return dNdnu

        if self.populations == 'PopII':
            dNdnuII = dNdnu_spectrum(self.NionII, norm_col=1, alpha_col=2)
            return dNdnuII * SFRD_interp(z) / self.Cosmo.mu_b

        elif self.populations == 'PopII+PopIII':
            SFRD_interpII, SFRD_interpIII = SFRD_interp
            dNdnuII = dNdnu_spectrum(self.NionII, norm_col=1, alpha_col=2)
            dNdnuIII = dNdnu_spectrum(self.NionIII, norm_col=3, alpha_col=4)
            return (dNdnuII * SFRD_interpII(z) + dNdnuIII * SFRD_interpIII(z)) / self.Cosmo.mu_b

        else:
            raise ValueError(f"Invalid population: {populations}. Must be 'PopII', 'PopIII', or 'PopII+PopIII'.")
            
    def Jalpha_star(self, z, SFRD_interp):
        """
        Compute the stellar Ly-alpha intensity.

        Parameters
        ----------
        z : Redshift at which J_alpha is evaluated.
        SFRD_interp : Interpolation function for SFRD(z).
        populations : 'PopII' or 'PopII+PopIII'
        Nion : Number of ionizing photons per baryon in stars (NionII for PopII, or a tuple (NionII, NionIII) for PopII+PopIII).

        Returns
        -------
        list of arrays
            [J_alpha_star, J_alpha_star_continuum, J_alpha_star_injected],
            i.e. total, continuum (n=2), and injected (n>2 cascade) Ly-alpha intensities in units cm^-2s^-1 Hz^-1
        """

        z = np.atleast_1d(z) 

        # Probabilly to produce a Ly-alpha photon from a cascade starting at Lyman level n in [2,23]
        frecycle = np.array([
            1, 0, 0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476,
            0.3496, 0.3512, 0.3524, 0.3535, 0.3543, 0.3550, 0.3556, 0.3561,
            0.3565, 0.3569, 0.3572, 0.3575,0.3578
        ])  
        
        n_vals = np.arange(2.0, 24.0) # Lyman levels
        n_col = n_vals[:, None] 
        i_row = np.arange(0, 22)  # n-2. Index for frecycle

        J_alpha_star = np.ones_like(z)
        J_alpha_star_injected = np.ones_like(z)
        J_alpha_star_continuum = np.ones_like(z)

        for j, zval in enumerate(z):
            # ztag: Dummy variable to integrate over redshift.
            # Prepare matrix of ztag values. In each row ztag goes from z to zmax of Lyman level n
            zmax_n_col = self.z_maxn(zval, n_col) #column vector of zmax(levels)
            Ztag_matrix = np.linspace(zval, zmax_n_col, 5, axis=1)[:,:,0] 
            
            integrand_arr = self.eps_star( self.Freq_redshift(zval, Ztag_matrix, n_col), Ztag_matrix, SFRD_interp) / self.Cosmo.hubble(1 + Ztag_matrix)

            n_contribution = frecycle[i_row] * np.trapz(integrand_arr, Ztag_matrix,axis = 1)
            sum_n_cont = n_contribution[0]
            sum_n_inj = np.sum(n_contribution[1:])
            sum_n = sum_n_inj+sum_n_cont
            J_alpha_star_injected[j] = unit.c * (1 + zval) ** 2 / 4 / np.pi * sum_n_inj  # cm^-2s^-1
            J_alpha_star_continuum[j] = unit.c * (1 + zval) ** 2 / 4 / np.pi * sum_n_cont  # cm^-2s^-1
            J_alpha_star[j] = unit.c * (1 + zval) ** 2 / 4 / np.pi * sum_n
        return ([J_alpha_star,J_alpha_star_continuum,J_alpha_star_injected])

    def Jalpha_X(self, eps_X_heat, xe, z):
        # Parameters: X-ray heat transfer rate (eVcm^-3s^-1), ionized fraction, redshift
        # Return: Jalpha_X (cm^-2) 
        nu_alpha = unit.Freq_Lya
        fheat = pre.fheat_interp(xe)
        flya = pre.fLya_interp(xe)
        eps_X_alpha = eps_X_heat * flya / fheat 
        Jalpha_x = unit.c / 4 / np.pi * eps_X_alpha / unit.Planck / nu_alpha / self.Cosmo.hubble(1+z) / nu_alpha
        return (Jalpha_x)

    def dN_dnu_Lyman(self, nu, Pop):
        # Parameters: frequency, Pop = II or III
        # Return Lyman band dN/dnu per baryon in stars #1/Hz
        Ryd = 13.6 / unit.Planck  # Rydberge constant in Hz
        i = int(np.floor(np.sqrt(1 / (1 - nu / Ryd))))  # Lyman level
        nu_alpha = unit.Freq_Lya

        if not (2 <= i <= 23):
            return 0

        if Pop == 'II':
            alpha = pre.spectral_distribution[2][i-2]
            Norm = pre.spectral_distribution[1][i-2] * self.NionII / nu_alpha
            eps_nuII = Norm*pow(nu/nu_alpha,alpha-1)
            return eps_nuII

        if Pop == 'III':
            alpha = pre.spectral_distribution[4][i-2]
            Norm = pre.spectral_distribution[3][i-2] * self.NionIII / nu_alpha
            eps_nuIII = Norm*pow(nu/nu_alpha,alpha-1)
            return eps_nuIII
    
    def avg_LW_spect(self, Pop):
        # dN/dnu per baryon in stars averaged over LW band
        nu_arr = np.linspace(11.2,13.5999,100)/unit.Planck
        dN_dnu = [self.dN_dnu_Lyman(nu,Pop) for nu in nu_arr] #1/Hz
        eps_b_avg= np.sum(dN_dnu * nu_arr * unit.Planck  /unit.erg) / len(nu_arr) #erg/Hz
        return eps_b_avg 

    def z_int_LW(self,z,SFRD_interp):
        zp_arr = np.linspace(z,1.053*(1+z)-1,10)
        integrand = SFRD_interp(zp_arr)/self.Cosmo.hubble(1+zp_arr)*self.fmod_interp((1+zp_arr)/(1+z)) #eV/cm^3
        return np.trapz(integrand,zp_arr) #eV/cm^3

    def JLW(self,z,SFRD_interp, eps_b_avg):
        prefact = eps_b_avg/self.Cosmo.mu_b*unit.c*(1+z)**2/4/np.pi #erg/Hz/s/cm^2
        return prefact*self.z_int_LW(z,SFRD_interp)
