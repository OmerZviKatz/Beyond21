import numpy as np
from importlib.resources import files
import beyond21.constants as consts

spec = np.transpose(np.loadtxt(files("beyond21").joinpath("data", "spectral_distribution"), skiprows=10)) 

def dNdnu_Lyman(nu, Nion_i, pop):       
        Ryd_Hz = 13.6 / consts.Planck  # Rydberge constant in Hz
 
        nu = np.asarray(nu)
        n_lyman = np.floor(np.sqrt(1 / (1 - nu / Ryd_Hz))).astype(int)
        idx = n_lyman - 2  # want 0..21
        valid = (idx >= 0) & (idx <= 21)
        idxc = np.clip(idx, 0, 21)

        if pop == 'II':
            norm_col = 1
            alpha_col = 2
        elif pop == 'III':
            norm_col = 3
            alpha_col = 4

        norm_tab = spec[norm_col]
        alpha_tab = spec[alpha_col]

        nu_alpha = consts.Freq_Lya
        norm = np.take(norm_tab, idxc) * Nion_i / nu_alpha
        alpha = np.take(alpha_tab, idxc)

        dNdnu = norm * (nu / nu_alpha) ** (alpha - 1)
        return np.where(valid, dNdnu, 0.0)


def avg_LW_spect(Nion_i, Pop):
        # dN/dnu per baryon in stars averaged over LW band
        nu_arr = np.linspace(11.2,13.5999,100)/consts.Planck
        dN_dnu = np.array([dNdnu_Lyman(nu,Nion_i, Pop) for nu in nu_arr]) #1/Hz
        eps_b_avg= np.sum(dN_dnu * nu_arr * consts.Planck  /consts.erg) / (len(nu_arr)-1) #erg/Hz
        return eps_b_avg 

