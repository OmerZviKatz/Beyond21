import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
Blues = cm.get_cmap('Blues')
Reds = cm.get_cmap('Reds')
Greys = cm.get_cmap('Greys')

class plotter():
    def __init__(self, evolver_obj):
        self.ev = evolver_obj

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

    def plot_quantity(self, quantity, ylabel, axis, yscale = 'log', **plot_kwargs):
        ''' Plot quantity evolution.
    
            Parameters:
                axis (matplotlib axis, optional): 
                    To plot on a specific axis, pass it using the ax argument. 
                    If not provided, a new figure and axis will be created automatically.
             **plot_kwargs: Additional keyword arguments passed to ax.plot(), e.g. linewidth, linestyle, color, etc.
    
             Returns:
                 tuple: (fig, ax) - matplotlib figure and axis objects. '''
        
        if quantity is None or not np.any(quantity):
            raise ValueError(f"{label} data is not available. Please ensure that 'EvolveSM' has been executed successfully.")

        fig = None
        if not axis:
            fig, axis = self.default_figure(axis, xlabel='Redshift (1 + z)', ylabel=ylabel, yscale=yscale)
        plot_kwargs = self.default_plot_kwargs(**plot_kwargs)
        
        axis.plot(self.ev.rs, quantity, **plot_kwargs)
        return fig, axis

    def plot_Tbaryon(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.ev.Tbaryon, 'Baryon Temperature [K]', axis, **plot_kwargs)
        
    def plot_Tspin(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.ev.Tspin, 'Spin Temperature [K]', axis, **plot_kwargs)

    def plot_T21(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.ev.T21*1000, '$T_{21}$ [mK]', axis, yscale = 'linear', **plot_kwargs)

    def plot_TCMB(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.ev.TCMB, 'CMB Temperature [K]', axis, **plot_kwargs)

    def plot_xHI(self, axis=None, **plot_kwargs):
        return self.plot_quantity(self.ev.xHI, r'Neutral Hydrogen Fraction $x_{\mathrm{HI}}$', axis, **plot_kwargs)

    def plot_SFRD(self, axis = None, **plot_kwargs):
        """ Plots the SFRD [Msolar/yr/Mpc^3] as a function of redshift
            axis, ***plot_kwargs like in plot_Tbaryon """

        fig = None
        if not axis:
            fig, axis = self.default_figure(axis, 
                                            xlabel = 'Redshift (1 + z)', 
                                            ylabel = r'SFRD [$M_{\odot}~{\rm yr}^{-1}~{\rm Mpc}^{-3}$]',
                                            xscale = 'linear', xlim = (1,61))
        plot_kwargs = self.default_plot_kwargs(**plot_kwargs) # Set default plot style if not provided

        z_arr = np.linspace(0,60,300)
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

        if self.ev.Pop == 'PopII':
            axis.plot(MagArr, self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28), **plot_kwargs)
        elif self.ev.Pop == 'PopIII':
            axis.plot(MagArr, self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28), **plot_kwargs)
        elif self.ev.Pop == 'PopII+PopIII':
            if plot_kwargs:
                print('Line styling is currently unavailable for two populations. Sorry for the inconvenience. \nIf needed it is possible to change styling directly in plot_SFRD.')
            UVLFII,UVLFIII = self.UVLF(z, MagArr, sigma_MUV, kUV=1.15e-28)
            axis.plot(MagArr, UVLFII, linewidth = 3, label = 'PopII', color = Blues(0.8))
            axis.plot(MagArr, UVLFIII, linewidth = 3, label = 'PopIII', color = Reds(0.7))
            axis.plot(MagArr, UVLFII+UVLFIII, linewidth = 3, linestyle = 'dashed', label = 'PopII+PopIII', color = Greys(0.7))
        return fig, axis

    

