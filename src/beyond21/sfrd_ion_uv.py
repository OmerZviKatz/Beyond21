import beyond21.constants as consts
import numpy as np
from numpy import trapz
from scipy.interpolate import interp1d
import beyond21.sfrd as Sobj
import beyond21.utils.interp_sorted_grid as interp
import beyond21.utils.interp_reg_grid as interp_reg
import beyond21.lyman_spec as lyman_spec

class SFRD_UVion(Sobj.SFRD):
    '''
    Non default parameters:
        star_formation_params - A dictionary listing the SFR model and its parameters. Can take one of three forms:
            A. Only pop-II stars 
                
        
    Default parameters:
        zmin, zstar - Minimal, maximal redshifts of the SFRD interpolation function produced by the SFRD object
        znum - The number of z points logspaced between zmin to zstar usrd to create SFRD(z) grids for interpolation functions
    '''    

    alpha_B_1e4 = 2.59e-13 #case B recombination coefficient at 10^4K
    
    def __init__(self, cosmo, SF_params, reion_params, photoheat = True):
        
        super().__init__(cosmo, SF_params)
        self.reion_params = reion_params
        self.photoheat = photoheat

        # average emissivity per baryon in LW band[erg/Hz]
        self.eps_b_avgII = lyman_spec.avg_LW_spect(reion_params['N_ionII'], 'II')
        if self.SF_params['model'] == 'PopII+PopIII':
           self.eps_b_avgIII = lyman_spec.avg_LW_spect(reion_params['N_ionIII'], 'III')


    ###############################
    # LW intensity - for feedback #  
    ###############################

    def z_int_LW(self,z,SFRD_interp):
        zp_arr = np.linspace(z,1.053*(1+z)-1,10)
        integrand = SFRD_interp(zp_arr)/self.cosmo.hubble(1+zp_arr)*self.fmod_interp((1+zp_arr)/(1+z)) #eV/cm^3
        return np.trapz(integrand,zp_arr) #eV/cm^3

    def JLW_calc(self,z,SFRD_interp, eps_b_avg):
        prefact = eps_b_avg/self.cosmo.mu_b*consts.c*(1+z)**2/4/np.pi #erg/Hz/s/cm^2
        return prefact*self.z_int_LW(z,SFRD_interp)


    ######################################################################
    # Ionizing UV photons, and ionized volume filling fraction evolution #
    ######################################################################

    def fescII(self, Mh):
        temp = self.reion_params['F_escII'] * pow(Mh/1e10/consts.M_s, -self.reion_params['alpha_escII']) 
        temp[temp>1] = 1
        return temp 
        
    def fescIII(self, Mh):
        temp = self.reion_params['F_escIII'] * pow(Mh/1e7/consts.M_s, -self.reion_params['alpha_escIII']) 
        temp[temp>1] = 1
        return temp

    def dNion_dz(self, k, z_desc, Mh_arr, dndlnm_mat, McutII, McutIII=None):
        """
        Compute dNion/dz at z_desc[k] using the *precomputed* HMF grid dndlnm_mat
        and finite differences on the z grid (no additional Cosmo.dndlnm calls).

        Parameters
        ----------
        k : int
            Index into z_desc (descending grid).
        z_desc : array
            Redshift grid used to build dndlnm_mat (descending).
        Mh_arr : array
            Halo mass grid.
        dndlnm_mat : array
            Precomputed HMF matrix with shape (Nz, len(Mh_arr)).
        McutII, McutIII : float
            Cut masses at the evaluation redshift (you already computed these in the loop).
            We keep them fixed for the derivative stencil (same behavior as your old code,
            where Mcut was not recomputed at zp/zm).
        """
        lnMh_arr = np.log(Mh_arr)

        # Precompute the gating functions once (same as old behavior: fixed across zp/zm)

        def Nion_at_index(idx):
            z = z_desc[idx]
            dndlnm = dndlnm_mat[idx, :]

            out = 0.0
            if self.SF_params["model"] in ["PopII", "PopII+PopIII"]:
                NionII = (
                    trapz(self.MstarII(Mh_arr, z) * dndlnm * self.fgalII(Mh_arr, McutII) * self.fescII(Mh_arr), lnMh_arr)
                    / self.cosmo.rho_baryon
                    * self.reion_params["N_ionII"]
                )
                out += NionII

            if self.SF_params["model"] in ["PopIII", "PopII+PopIII"]:
                NionIII = (
                    trapz(self.MstarIII(Mh_arr, z) * dndlnm * self.fgalIII(Mh_arr, McutII, McutIII) * self.fescIII(Mh_arr), lnMh_arr)
                    / self.cosmo.rho_baryon
                    * self.reion_params["N_ionIII"]
                )
                out += NionIII

            return out

        Nz = len(z_desc)

        # Central difference on the grid when possible; otherwise one-sided.
        if 0 < k < Nz - 1:
            Np = Nion_at_index(k + 1)
            Nm = Nion_at_index(k - 1)
            dz = z_desc[k + 1] - z_desc[k - 1]  # negative on a descending grid
            return (Np - Nm) / dz

        if k == 0:
            N0 = Nion_at_index(0)
            N1 = Nion_at_index(1)
            dz = z_desc[1] - z_desc[0]
            return (N1 - N0) / dz

        # k == Nz-1
        Nn = Nion_at_index(Nz - 1)
        Nn1 = Nion_at_index(Nz - 2)
        dz = z_desc[Nz - 1] - z_desc[Nz - 2]
        return (Nn - Nn1) / dz


    def dQdz(self, z, Q, dNion_dz):
        if Q >= 0.99:
            return 0.0
        H = self.cosmo.hubble(1 + z)
        clumping = 3   
        dNrec_dz = self.alpha_B_1e4 * Q**2 * (1 + z)**3 / H * self.cosmo.nH / (1 + z) * clumping * (1+self.cosmo.nHe/self.cosmo.nH)
        return dNion_dz + dNrec_dz


    def SFRD_and_Qion_interp(self, zstar=50, z_end=1, Nz=250):
        """
        Compute SFRD(z) interpolation functions for PopII, PopIII, 
        along the UV ionized filling fraction Qion(z) interpolation function. 
        Account for LW feedback and photoheating suppression of PopIII if flagged.
        If LW feedback is on, pass method JLW_calc to compute the evolving LW flux.
        
        Returns
        -------
        list
            [SFRD_interp, Qion_interp] if model is 'PopII' or 'PopIII',
            [[SFRDII_interp, SFRDIII_interp], Qion_interp] if model is 'PopII+PopIII'.
        
        Note: We solve from high to low-z (backwards in time) to properly account for feedback effects, 
        but all interpolation functions must be in ascending z order for efficiency.
        """

        model = self.SF_params['model']

        # Redshift grid 
        z_desc = np.linspace(zstar, z_end, Nz) # High to low z (evolution order)
        z_asc = z_desc[::-1]  # Low z to hight (for interpolations)
        dz_desc = z_desc[1] - z_desc[0]   

        # Initialize LW, SFRD and filling fraction arrays
        JLW_arr = np.full_like(z_desc, 1e-50)  # Initial JLW ~ 0
        SFRDII_arr = np.zeros(Nz)
        SFRDIII_arr = np.zeros(Nz)
        Q_arr = np.full_like(z_desc, 1e-10)

        # Create interpolation objects with pointers to SFRD_arr (will be updated dynamically as SFRD_arr is filled)
        SFRDII_interp_obj = interp_reg.reg_grid_interp(SFRDII_arr, z_asc, zero_out_of_bounds = True)
        SFRDIII_interp_obj = interp_reg.reg_grid_interp(SFRDIII_arr, z_asc, zero_out_of_bounds = True)
    
        
        # Compute HMF grids in advance for efficiency
        Mh_min = self.Mcut_eV(zstar, 'II',JLW = 1e-30)/consts.M_s / 10
        if model in ['PopIII', 'PopII+PopIII']:
            Mh_minIII = self.Mcut_eV(zstar, 'III',JLW = 1e-30)/consts.M_s / 10
            Mh_min = np.minimum(Mh_min,Mh_minIII)           
        
        Mh_arr = 10**np.arange(np.log10(Mh_min), 15, 0.1)*consts.M_s
        lnMh_arr = np.log(Mh_arr)
        self.dndlnm_mat = np.array([self.cosmo.dndlnm(Mh_arr, z) for z in z_desc]) # HMF matrix (Nz,len(Mh_arr))
        
        # Time integration loop (from high to low z, i follows descending z)
        for i in range(1, Nz):
            z = z_desc[i] # z at current step
            z_prev = z_desc[i-1] # z at previous step
            dndlnm_arr = self.dndlnm_mat[i,:]  # HMF at current z
            j = Nz - 1 - i  # Reverse index for ascending interpolation
            
            Q_prev = Q_arr[i-1] 
            JLW_prev = JLW_arr[i - 1]

            # 1) compute cutoff mass and SFRD at current z 
            if model in ['PopII', 'PopII+PopIII']:
                McutII = self.Mcut_eV(z, 'II',JLW = JLW_prev)
                SFRDII_arr[j] = self.SFRDII_calc(z, Mh_arr, dndlnm_arr, McutII)

            if model in ['PopIII', 'PopII+PopIII']:
                McutIII = self.Mcut_eV(z, 'III',JLW = JLW_prev)
                SFRDIII = self.SFRDIII_calc(z, Mh_arr, dndlnm_arr, McutII, McutIII)
                if self.photoheat:
                    SFRDIII = SFRDIII*(1-Q_prev)
                SFRDIII_arr[j] = SFRDIII

            # 2) update LW flux at current z
            if self.LW_feedback:
                if model == 'PopII+PopIII':
                    JLW_arr[i] = (
                        self.JLW_calc(z, SFRDII_interp_obj.interp1D, self.eps_b_avgII) +
                        self.JLW_calc(z, SFRDIII_interp_obj.interp1D, self.eps_b_avgIII)
                        ) * 1e21
                elif model == 'PopII':
                    JLW_arr[i] = self.JLW_calc(z, SFRDII_interp_obj.interp1D, self.eps_b_avgII) * 1e21
                

            # 3) compute ionization rate
            k_prev = i - 1  # z_prev index in z_desc / dndlnm_mat
            if model == "PopII":
                dNion_dz = self.dNion_dz(
                    k_prev, z_desc, Mh_arr, self.dndlnm_mat, McutII, McutIII=None
                )
            else:
                dNion_dz = self.dNion_dz(
                    k_prev, z_desc, Mh_arr, self.dndlnm_mat, McutII, McutIII=McutIII
    )
            # 4) update Q (RK4)
            Q_cur = rk4_step(self.dQdz, z_prev, Q_prev, dz_desc, dNion_dz)
            Q_arr[i] = min(Q_cur, 0.999999999999999) # Clip when universe is fully ionized
            
        # Create interpolation functions for outputs 
        self.Q_interp = interp1d(z_desc,Q_arr,fill_value=(1, 0), bounds_error=False)
        self.JLW_interp = interp1d(z_desc, JLW_arr, bounds_error=False, fill_value='extrapolate')

        # Prepare outputs based on model
        if model == 'PopII':
            return [SFRDII_interp_obj.interp1D,self.Q_interp]
        elif model == 'PopIII':
            return [SFRDIII_interp_obj.interp1D, self.Q_interp]
        elif model == 'PopII+PopIII':
            return [[SFRDII_interp_obj.interp1D, SFRDIII_interp_obj.interp1D],self.Q_interp]

    

def rk4_step(f, x, y, h, *args):
    # Single RK4 step for dy/dx = f(x, y, *args).
    # Returns y(x + h).
    
    k1 = f(x, y, *args)
    k2 = f(x + 0.5*h, y + 0.5*h*k1, *args)
    k3 = f(x + 0.5*h, y + 0.5*h*k2, *args)
    k4 = f(x + h, y + h*k3, *args)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6