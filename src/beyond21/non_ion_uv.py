import numpy as np
import beyond21.constants as consts
import beyond21.interpolations as pre
import beyond21.lyman_spec as lyman_spec

class NonIonUV:

    def __init__(self, cosmo, reion_params, populations = False):
        self.cosmo = cosmo
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
        Ryd = 13.6 / consts.Planck  # Rydberge constant in Hz
        nu_n = Ryd * (1 - 1 / n ** 2)  
        nu_n_tag = nu_n * (1 + ztag) / (1 + z)
        return (nu_n_tag)
    
    def eps_star(self, nu, z, SFRD_interp):
        nu = np.asarray(nu)
        z = np.asarray(z)

        if self.populations == "PopII":
            dNdnuII = lyman_spec.dNdnu_Lyman(nu, self.NionII, 'II')
            return dNdnuII * SFRD_interp(z) / self.cosmo.mu_b

        if self.populations == "PopII+PopIII":
            SFRD_II, SFRD_III = SFRD_interp
            dNdnuII = lyman_spec.dNdnu_Lyman(nu, self.NionII, 'II')
            dNdnuIII = lyman_spec.dNdnu_Lyman(nu, self.NionIII, 'III')
            return (dNdnuII * SFRD_II(z) + dNdnuIII * SFRD_III(z)) / self.cosmo.mu_b

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
            i.e. total, continuum (n=2), and injected (n>2 cascade) Ly-alpha intensities in constss cm^-2s^-1 Hz^-1
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
            
            integrand_arr = self.eps_star( self.Freq_redshift(zval, Ztag_matrix, n_col), Ztag_matrix, SFRD_interp) / self.cosmo.hubble(1 + Ztag_matrix)

            n_contribution = frecycle[i_row] * np.trapz(integrand_arr, Ztag_matrix,axis = 1)
            sum_n_cont = n_contribution[0]
            sum_n_inj = np.sum(n_contribution[1:])
            sum_n = sum_n_inj+sum_n_cont
            J_alpha_star_injected[j] = consts.c * (1 + zval) ** 2 / 4 / np.pi * sum_n_inj  # cm^-2s^-1
            J_alpha_star_continuum[j] = consts.c * (1 + zval) ** 2 / 4 / np.pi * sum_n_cont  # cm^-2s^-1
            J_alpha_star[j] = consts.c * (1 + zval) ** 2 / 4 / np.pi * sum_n
        return ([J_alpha_star,J_alpha_star_continuum,J_alpha_star_injected])

    def Jalpha_X(self, eps_X_heat, xe, z):
        # Parameters: X-ray heat transfer rate (eVcm^-3s^-1), ionized fraction, redshift
        # Return: Jalpha_X (cm^-2) 
        nu_alpha = consts.Freq_Lya
        fheat = pre.fheat_interp(xe)
        flya = pre.fLya_interp(xe)
        eps_X_alpha = eps_X_heat * flya / fheat 
        Jalpha_x = consts.c / 4 / np.pi * eps_X_alpha / consts.Planck / nu_alpha / self.cosmo.hubble(1+z) / nu_alpha
        return (Jalpha_x)
