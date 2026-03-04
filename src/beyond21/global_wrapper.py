import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
from scipy.interpolate import interp1d
import beyond21.constants as unit
import beyond21.evolution as Eobj
from beyond21.cosmology import Cosmology as CObj

warnings.simplefilter("always", category=UserWarning)
Blues = cm.get_cmap('Blues')
Reds = cm.get_cmap('Reds')
Greys = cm.get_cmap('Greys')

class GlobalWrapper(Eobj.evolver):
    def __init__(self, star_formation_params, xray_params, reion_params, cosmo_params = None,
                 photoheat = True, Lya_Heat = True, CMB_Heat = True):
        
        self.verify_cosmo_params(cosmo_params)
        self.verify_star_formation_params(star_formation_params)
        self.verify_xray_params(xray_params, star_formation_params['model'])
        self.verify_reion_params(reion_params, star_formation_params['model'])
        
        CosmoObj = CObj(self.cosmo_params)
        super().__init__(CosmoObj,
                         star_formation_params = star_formation_params,
                         xray_params = xray_params,
                         reion_params = reion_params,
                         photoheat = photoheat,
                         Lya_Heat = Lya_Heat,
                         CMB_Heat = CMB_Heat)

    
    def verify_star_formation_params(self, star_formation_params):
        if star_formation_params['model'] not in ['PopII', 'PopII+PopIII']:
            raise ValueError("Unsupported SFR model. Rerun with model = PopII or PopII+PopIII")
        
        if star_formation_params['model'] in ['PopII', 'PopII+PopIII']:
            for key in ['F_starII', 'alphaII', 'betaII', 'Mpivot', 'M_cutII','eps_t']:
                if key not in star_formation_params:
                     raise ValueError("Missing star_formation_params key '" + key + "'")
        
        if star_formation_params['model'] in ['PopIII', 'PopII+PopIII']:
            for key in ['F_starIII', 'alphaIII', 'M_cutIII', 'eps_t']:
                if key not in star_formation_params:
                     raise ValueError("Missing star_formation_params key '" + key + "'")
               
        if ('A_LW' in star_formation_params) != ('B_LW' in star_formation_params):
            raise ValueError("Invalid LW feedback parameters in 'star_formation_params'.\n"
                "You have provided only 'A_LW' or only 'B_LW'.\n"
                "- To run without LW feedback, remove both 'A_LW' and 'BLW'.\n"
                "- To run with LW feedback, include both 'A_LW' and 'BLW'.")

        if ('A_vrel' in star_formation_params) != ('B_vrel' in star_formation_params):
            raise ValueError("Invalid v_rel feedback parameters in 'star_formation_params'.\n"
                "You have provided only 'A_vrel' or only 'B_vrel'.\n"
                "- To run without v_rel feedback, remove both 'A_vrel' and 'B_vrel'.\n"
                "- To run with v_rel feedback, include both 'A_vrel' and 'B_vrel'.")
        self.star_formation_params = star_formation_params

    def verify_xray_params(self, xray_params,model):
        Xray_keys = ['E_min', 'E_max', 'alpha_s', 'alpha_h','E_break']
        
        if model in ['PopII', 'PopII+PopIII']:
            Xray_keys.append('LSFRII')
        if model in ['PopIII', 'PopII+PopIII']:
            Xray_keys.append('LSFRIII')

        for key in Xray_keys:
            if key not in xray_params:
                 raise ValueError("Missing xray_params key '" + key + "'")
        self.xray_params = xray_params

    def verify_reion_params(self,reion_params,model):
        if model in ['PopII', 'PopII+PopIII']:
            for key in ['F_escII', 'alpha_escII', 'N_ionII']:
                if key not in reion_params:
                     raise ValueError("Missing reion_params key '" + key + "'")

        if model in ['PopIII', 'PopII+PopIII']:
            for key in ['F_escIII', 'alpha_escIII', 'N_ionIII']:
                if key not in reion_params:
                     raise ValueError("Missing reion_params key '" + key + "'")
        self.reion_params = reion_params

    def verify_cosmo_params(self, cosmo_params):
        default_cosmo_params = {
            "Om0": 0.3111,
            "Ob0": 0.0490,
            "H0": 67.66,
            "sigma8": 0.8102,
            "ns": 0.9665,
            "Neff": 3.046,
            "w0": -1.0,
            "wa": 0.0,
            "Y_He": 0.24,
            "hmf_kwargs": {"mdef": "fof", "model": "sheth99"},
        }

        if cosmo_params is None:
            warnings.warn(
                "No cosmo_params given. Assuming Planck18 cosmology and ST HMF.",
                UserWarning
            )
            cosmo_params = {}

        else:
            required_keys = ["Om0", "Ob0", "H0", "sigma8", "ns", "Neff"]
            for key in required_keys:
                if key not in cosmo_params:
                    raise ValueError(f"Missing cosmo_params key '{key}'. Please provide a full set of cosmological parameters or leave cosmo_params as None to use default Planck18 values.")
        
        cosmo_params = {**default_cosmo_params, **cosmo_params}
        self.cosmo_params = cosmo_params

    def SFRD(self, z):
        # Wrapper SFRD function for convenience
        
        #Unit conversion
        cm_to_Mpc = unit.Centimeter/unit.Mpc
        eV_to_Ms = 1/unit.M_s
        sec_to_year = unit.Sec/unit.Year
        if self.Pop != 'PopII+PopIII':
            return self.SFRD_interp(z)* eV_to_Ms / cm_to_Mpc**3 / sec_to_year
        SFRDII = self.SFRD_interp[0](z)* eV_to_Ms / cm_to_Mpc**3 / sec_to_year
        SFRDIII = self.SFRD_interp[1](z)* eV_to_Ms / cm_to_Mpc**3 / sec_to_year
        return [SFRDII, SFRDIII] # [Msolar/yr/Mpc^3]

    
    def UVLF(self, z, Muv, sigma_MUV, Mh = None, kUV=1.15e-28):
        if self.Pop == 'PopII':
            UVLF = self.UVobj.UVLF_Stoch_continuous(z, Muv, sigma_MUV, 'II', Mh = None, kUV=1.15e-28)
        elif self.Pop == 'PopIII':
            UVLF = self.UVobj.UVLF_Stoch_continuous(z, Muv, sigma_MUV, 'II', Mh = None, kUV=1.15e-28)
        elif self.Pop == 'PopII+PopIII':
            UVLFII = self.UVobj.UVLF_Stoch_continuous(z, Muv, sigma_MUV, 'II', Mh = None, kUV=1.15e-28)
            UVLFIII = self.UVobj.UVLF_Stoch_continuous(z, Muv, sigma_MUV, 'III', Mh = None, kUV=1.15e-28)
            UVLF = (UVLFII,UVLFIII)
        return UVLF

    def JLW(self, z):
        return self.SF.J_LW(z)

    def default_figure(self, axis, xlabel = None, ylabel = None, xscale = 'log', yscale = 'log', xlim = None):
        fig, axis = plt.subplots(figsize=(7.5, 4))
        if not xlim:
            xmin,xmax = self.rs[-1],self.rs[0]
        else:
            xmin,xmax = xlim
        axis.set_xlim(xmin, xmax)    
        axis.tick_params(axis='both', which='major', length=7, width=1.2)
        axis.tick_params(axis='both', which='minor', length=4, width=0.8)
        axis.tick_params(labelsize=12)
        axis.set_xscale(xscale)
        axis.set_yscale(yscale)
        axis.set_xlabel(xlabel, fontsize=15)
        axis.set_ylabel(ylabel, fontsize=15)
        return (fig,axis)

    def default_plot_kwargs(self, **plot_kwargs):
        plot_kwargs.setdefault('linewidth', 3)
        plot_kwargs.setdefault('linestyle', 'solid')
        plot_kwargs.setdefault('color', plt.get_cmap('Blues')(0.7))
        return (plot_kwargs)

    def plot_quantity(self, x, y, xlabel, ylabel, axis, xscale = 'log', yscale = 'log', **plot_kwargs):
        ''' Plot quantity evolution.
    
            Parameters:
                axis (matplotlib axis, optional): 
                    To plot on a specific axis, pass it using the ax argument. 
                    If not provided, a new figure and axis will be created automatically.
             **plot_kwargs: Additional keyword arguments passed to ax.plot(), e.g. linewidth, linestyle, color, etc.
    
             Returns:
                 tuple: (fig, ax) - matplotlib figure and axis objects. '''
        
        fig = None
        if not axis:
            fig, axis = self.default_figure(axis, xlabel=xlabel, ylabel=ylabel, xscale = xscale, yscale=yscale, xlim = (np.min(x), np.max(x)))
        plot_kwargs = self.default_plot_kwargs(**plot_kwargs)
        
        axis.plot(x, y, **plot_kwargs)
        return fig, axis

    def plot_Tbaryon(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.rs, self.Tbaryon, 'Redshift (1 + z)', 'Baryon Temperature [K]', axis, **plot_kwargs)
        
    def plot_Tspin(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.rs, self.Tspin, 'Redshift (1 + z)', 'Spin Temperature [K]', axis, **plot_kwargs)

    def plot_T21(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.rs, self.T21*1000, 'Redshift (1 + z)', '$T_{21}$ [mK]', axis, yscale = 'linear', **plot_kwargs)

    def plot_TCMB(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.rs, self.TCMB, 'Redshift (1 + z)',  'CMB Temperature [K]', axis, **plot_kwargs)

    def plot_xHI(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.rs, self.xHI,'Redshift (1 + z)',  r'Neutral Hydrogen Fraction $x_{\mathrm{HI}}$', axis, **plot_kwargs)

    def plot_SFRD(self, z_arr = np.linspace(5,30,150), axis = None, **plot_kwargs):
        """ Plots the SFRD [Msolar/yr/Mpc^3] as a function of redshift
            axis, ***plot_kwargs like in plot_Tbaryon """

        fig = None
        if not axis:
            fig, axis = self.default_figure(axis, 
                                            xlabel = 'Redshift (1 + z)', 
                                            ylabel = r'SFRD [$M_{\odot}~{\rm yr}^{-1}~{\rm Mpc}^{-3}$]',
                                            xscale = 'linear', xlim = (1,61))
        plot_kwargs = self.default_plot_kwargs(**plot_kwargs) # Set default plot style if not provided

        if self.Pop == 'PopII':
            axis.plot(z_arr + 1, self.SFRD(z_arr), **plot_kwargs)
        elif self.Pop == 'PopIII':
            axis.plot(z_arr + 1, self.SFRD(z_arr), **plot_kwargs)
        elif self.Pop == 'PopII+PopIII':
            if plot_kwargs:
                print('Line styling is currently unavailable for two populations. Sorry for the inconvenience. \nIf needed it is possible to change styling directly in plot_SFRD.')   
            SFRDII,SFRDIII = self.SFRD(z_arr)
            axis.plot(z_arr + 1, SFRDII, linewidth = 3, label = 'PopII', color = Blues(0.8))
            axis.plot(z_arr + 1, SFRDIII, linewidth = 3, label = 'PopIII', color = Reds(0.7))
            axis.plot(z_arr + 1, SFRDII+SFRDIII, linewidth = 3, linestyle = 'dashed', label = 'PopII+PopIII', color = Greys(0.7))
        return fig, axis

    def plot_UVLF(self,z, MagArr = np.linspace(-20,-5, 50), sigma_MUV = 0.01, kUV=1.15e-28, axis = None, **plot_kwargs):
        """ Calculates the UVLF [1/Mpc^3/Mag] at redshift z over magnitude array Mag_arr 
            axis, ***plot_kwargs like in plot_Tbaryon """

        fig = None
        if not axis:
            fig, axis = self.default_figure(axis, 
                                            xlabel = 'Magnitude', 
                                            ylabel = r'$\Phi \left[{\rm Mag}^{-1}~{\rm Mpc}^{-3}\right]$',
                                            xscale = 'linear', xlim = (MagArr[0],MagArr[-1]))
        plot_kwargs = self.default_plot_kwargs(**plot_kwargs) # Set default plot style if not provided

        if self.Pop == 'PopII':
            axis.plot(MagArr, self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28), **plot_kwargs)
        elif self.Pop == 'PopIII':
            axis.plot(MagArr, self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28), **plot_kwargs)
        elif self.Pop == 'PopII+PopIII':
            if plot_kwargs:
                print('Line styling is currently unavailable for two populations. Sorry for the inconvenience. \nIf needed it is possible to change styling directly in plot_SFRD.')
            UVLFII,UVLFIII = self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28)
            axis.plot(MagArr, UVLFII, linewidth = 3, label = 'PopII', color = Blues(0.8))
            axis.plot(MagArr, UVLFIII, linewidth = 3, label = 'PopIII', color = Reds(0.7))
            axis.plot(MagArr, UVLFII+UVLFIII, linewidth = 3, linestyle = 'dashed', label = 'PopII+PopIII', color = Greys(0.7))
        return fig, axis

    def plot_JLW(self, z_arr = np.linspace(5,30,150), axis = None, **plot_kwargs):
        if not axis:
            return self.plot_quantity(1+z_arr, self.UVobj.JLW_interp(z_arr),'Redshift (1 + z)',  r'$J_{\rm LW} \ [10^{-21} \ {\rm erg \ Hz^{-1} \ s^{-1} \ cm^{-2}}]$', axis, **plot_kwargs, xscale = 'linear')
        return self.plot_quantity(1+z_arr, self.UVobj.JLW_interp(z_arr),'Redshift (1 + z)',  r'$J_{\rm LW}$', axis, **plot_kwargs)
