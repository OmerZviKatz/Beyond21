import numpy as np
from scipy.optimize import root
import beyond21.constants as consts
import beyond21.interpolations as pre

global Ts_prev
Ts_prev = [0]

##############################
# Spin and 21-cm Temperatues #
##############################

def C10_calc(TK, nHI, np, ne):
    # Parameters: Tk- gas temperature in Kelvin, number densities [cm^-3]
    # Return: C10- collisional transition rate in s^-1
    Kappa_eH = pre.interp_kappa_eH(TK)
    Kappa_pH = pre.interp_kappa_pH(TK)
    Kappa_HH = pre.interp_kappa_HH(TK)
    C10 = nHI * Kappa_HH + np * Kappa_pH + ne * Kappa_eH
    return (C10)

def tauGP_func(H, nHI):
    # Hubble at z - s^-1
    gamma = 50 * 1e6  # HWHM of Lyalpha resonance in Hz
    return 3 * nHI * consts.Lambda_Lya ** 3 * gamma / 2 / H  # dimensionless

def Ts_inv(Ts, TK, TCMB, xHI, xc, tauGP, J, z):
    # Parameters: Ts- spin temperature in K (symoblic parameter), TK- gas temperature in kelvin, xe- ionized fraction, J - Lymann alpha flux in cm^−2s^−1 Hz^−1, z-redshift
    # Return: Ts^-1 in kelvin
    s_alpha,Tc = pre.Salpha_Tc_Interp(tauGP,TK,Ts[0])
    xalpha = 1.66 * 1e11 / (1 + z) * J * s_alpha  # dimensionless https://arxiv.org/pdf/1003.3878.pdf eq 7
    Ts_inv = (1 / TCMB + xalpha / Tc + xc / TK) / (1 + xalpha + xc)

    return Ts_inv

def Ts_calc(H, TK, xHI, J, z, nHI, nHII, ne, TCMB, Ts_prev=None):
    # Solve for spin temperature
    C10 = C10_calc(TK, nHI, nHII, ne)
    xc = C10 * consts.Tstar / (consts.A10 * TCMB)  # Collision coupling coefficient - dimensionless
    tauGP = tauGP_func(H,nHI)
    
    # Scalar case
    if np.isscalar(TK):
        T_init = Ts_prev if Ts_prev else min(TK, TCMB)
        sol = root(lambda Ts: Ts_inv(Ts, TK, TCMB, xHI, xc, tauGP, J, z) - 1/Ts, T_init)
        return sol.x[0]

    # Array case
    Tsarr = np.empty_like(TK)
    for i in range(len(TK)):
        if i==0:
            T_init = min(TK[i], TCMB[i])
        else:
            T_init = Tsarr[i-1]
        sol = root(lambda Ts: Ts_inv(Ts, TK[i], TCMB[i], xHI[i], xc[i], tauGP[i], J[i], z[i]) - 1/Ts, T_init)
        Tsarr[i] = sol.x[0]
        
    return Tsarr

def T21calc(cosmo,z, Ts, xHI, nH, TCMB):
    # Parameters: redshift, spin temperature in K, ionized fraction
    # Return: T21 in K 
    TCMB = cosmo.TCMB(z + 1) / consts.kB  # CMB temperature - Kelvin

    # nu0=1420*1e6 #Frequency of 21cm photon in Hz
    # tau=3*consts.c**3*consts.A10*xHI*nH/32/np.pi/nu0**3/cosmo.hubble(z)*consts.Tstar/Ts #Correct
    # delta_Tb=(Ts-TCMB)/(1+z)*(1-np.exp(-tau))
    # return delta_Tb

    return (27 * xHI * cosmo.Omega_b * cosmo.cosmo.h ** 2 / 0.023 * np.sqrt(0.15 / cosmo.Omega_m / cosmo.cosmo.h ** 2 * (1 + z) / 10) * (1 - cosmo.TCMB(1 + z) / consts.kB / Ts) / 1000)  # /1000 for mK


#############################################
# Ionization and Recombination Coefficients #
#############################################

def alpha_A(Tbaryon):
    '''
    Case A recombination coefficient [cm^3/s]

    The result for T>3 [Kelvin] is taken from table 1 (rate 2) of 10.1016/s1384-1076(97)00010-9 which is a fit to the data from  10.1086/171063
    For T<3 [Kelvin] fir is inavlid, instead take alpha(3K)*sqrt(3K/T) as an approximation. See Eq.10 in the 2nd paper for explanation.
    '''
    logT = np.log(Tbaryon);
    T3 = 3*consts.kB
    Tbaryon_kelvin = Tbaryon/consts.kB
    
    if Tbaryon>=T3:
        alpha = pow(np.exp(1), -28.6130338 - 0.72411256*logT - 2.02604473e-2*pow(logT, 2)
                - 2.38086188e-3*pow(logT, 3) - 3.21260521e-4*pow(logT, 4)
                - 1.42150291e-5*pow(logT, 5) + 4.98910892e-6*pow(logT, 6)
                + 5.75561414e-7*pow(logT, 7) - 1.85676704e-8*pow(logT, 8)
                - 3.07113524e-9 * pow(logT, 9))
        return alpha

    logT3 = np.log(T3)
    alpha3 = pow(np.exp(1), -28.6130338 - 0.72411256*logT3 - 2.02604473e-2*pow(logT3, 2)
            - 2.38086188e-3*pow(logT3, 3) - 3.21260521e-4*pow(logT3, 4)
            - 1.42150291e-5*pow(logT3, 5) + 4.98910892e-6*pow(logT3, 6)
            + 5.75561414e-7*pow(logT3, 7) - 1.85676704e-8*pow(logT3, 8)
            - 3.07113524e-9 * pow(logT3, 9))
    alpha = alpha3*np.sqrt(T3/Tbaryon)
    return alpha

def alpha_B(Tbaryon):
    # Case-B recombination coefficient [cm^3/s], input baryon temperature in eV.
    # Fudged result of "Total and effective radiative recombination coefficients" by Pequignot et al. (see 1011.3758).

    fudge_fac = 1.126
    t = 1.0e-4*Tbaryon/consts.kB
    return ( fudge_fac * 1.0e-13 * 4.309 * t ** (-0.6166) / (1 + 0.6703 * t ** 0.5300) )

def beta_ion(T_rad):
    # Input: T_rad [eV] : Temperature of background radiation (CMB for SM)
    # Output: beta_ion [1/s] : Case-B photoionization coefficient in s^-1

    red_mass = consts.m_p*consts.m_e/(consts.m_p + consts.m_e)
    ge = (2*np.pi*red_mass*T_rad)**(3/2)/consts.Planck**3/consts.c**3 #Eq 3 times nH
    return ge / 4 * np.exp(-consts.E_Ryd/4/T_rad)*alpha_B(T_rad)


###############################################################
#                     x_HII_IGM Evolution Eqs                 #
###############################################################
#This is the H_II fraction outside the ionized bubbles


def dxHII_dloga_3level_caseB_recombination(rs, xHII_IGM, Tbaryon, n_H, Hubble, TCMB):
    ''' return: dxHII/dlog(a) - the change in the ionized hydrogen xHII = nHII/nH according to the TLA approximation 1011.3758 '''
    RLya = 8 * np.pi * Hubble / (3 * n_H * (1 - xHII_IGM) * consts.Lambda_Lya**3) # [1/s] Lyman alpha rate
    alpha_recomb = alpha_B(Tbaryon) # [cm^3/s] Case B recombination coefficient
    Lambda2s1s = 8.22 #[1/s] Rate of photon decay from 2s to 1s in Hydrogen
    C=(3/4*RLya + Lambda2s1s/4)/(beta_ion(TCMB) + 3/4*RLya + Lambda2s1s/4) # Peebles constant

    dxHII_dloga = -(C / Hubble) * (n_H * alpha_recomb * xHII_IGM ** 2 - 4 * (1 - xHII_IGM) * beta_ion(TCMB) * np.exp(-consts.E_Lya / TCMB))  # Eq C13
    if xHII_IGM >= 0.99 and dxHII_dloga>0: 
        #Before recombination starts dxHII_dloga>0 causing numerical issues. Physically xHII<1 so we just set to dxHII_dloga = 0 in this case.
        return 0
    return dxHII_dloga

def dxHII_dloga_caseA_recombination(rs, xHII_IGM, Tbaryon, n_H, Hubble):
    Clump = 3 # IGM clumping factor
    return -alpha_A(Tbaryon)*2 * xHII_IGM**2 * n_H / Hubble

def dxHII_dloga_Xray_ionization(rs, xHII_IGM, Lambda, Hubble, n_H):    
    return (1-xHII_IGM) * Lambda / Hubble / n_H

    
###############################################################
#                  Baryonic Temperature Rates                 #
###############################################################

def dTb_dloga_Compton(rs, abundances, Tbaryon, Hubble, TCMB):
    ''' 
    Inputs:
        rs: Redshift (1+z)
        ionization: Ionization dictionary 
        Tbaryon [eV]: Temperature of baryons
    Outputs:
        dTb_dloga [eV]: Rate of change in Tbaryon due to Compton scattering with CMB photons 
    '''
    H = Hubble / consts.Sec
    return (abundances['e'] / (1 + abundances['He'] + abundances['e'])) * 8 * consts.Thomson_xsec* consts.a_r*pow(TCMB,4) / (3 * consts.m_e) * (TCMB - Tbaryon) / H  #Fix: should replace HII with e in ionization. Also we are using ionization quantities only in neutral region. Should it not be the average including bubbles?

def dTb_dloga_Xrays(rs, abundances, Xheat, Hubble, n_H): 
    '''
    Inputs:
        rs: Redshift (1+z)
        ionization: Ionization dictionary
        Xheat [eV/cm^3/s]: An interpolation function of X-ray heat rate per volume, taking [log_10(xHII),z] as input
    Outputs:
        dTb_dloga_Xrays [eV]: Rate of change in Tbaryon due to X-ray heating
    '''
    n = n_H * (1 + abundances['He'] + abundances['e']) 
    return 2 / 3 / Hubble / n * Xheat  

def dT_dloga_NumberChange (rs,dxHII_dloga,abundances,Tbaryon):
        if rs > 50:
            dxe_dloga = dxHII_dloga
        else:
            dxe_dloga = dxHII_dloga*(1 + abundances['He'])
        return -dxe_dloga * Tbaryon / (1 + abundances['He'] + abundances['e']) 

def dTb_dloga_Lya(rs, abundances, Q_HII, Tbaryon, Jalphastar_interps, JalphaX_interp, n_H, Xheat, TCMB, H):     
    '''
    Inputs:
        rs: Redshift (1+z)
        ionization: Ionization dictionary
        Q_HII: Volume fraction of the universe covered by UV ionized bubbles
        Tbaryon [eV]: Temperature of baryons
        J_alpha_interps:
        Xrays: True to include X-ray heating
        Xheat [eV/cm^3/s]: An interpolation function of X-ray heat rate taking [log_10(xHII),z] as input
    Outputs:
        dTb_dloga_Lya [eV]: Rate of change in Tbaryon due to Lyman-alpha heating
    '''

    Jalphastar = Jalphastar_interps[0](rs-1)
    Jalphastar_continuum = Jalphastar_interps[1](rs-1)
    Jalphastar_injected = Jalphastar_interps[2](rs-1)
    Tbaryon_kelvin = Tbaryon/consts.kB
    
    # Global averaged abundances
    xHI_avg = 1 - (Q_HII + (1-Q_HII) * abundances['HII'])
    nHI = n_H*xHI_avg
    nHII = n_H*(1-xHI_avg)
    ne = nHII * (1+abundances['He'])
    
    
    Jalpha = Jalphastar + JalphaX_interp(Xheat,abundances['HII'], rs-1) #Add Lyalpha intensity from X-rays
    if Ts_prev[0]!=0 and rs<49:
        Ts = Ts_calc(H,Tbaryon_kelvin, xHI_avg, Jalpha, rs-1, nHI, nHII, ne, TCMB/consts.kB, Ts_prev = Ts_prev[0]) #Tspin kelvin
    else:
        Ts = Ts_calc(H,Tbaryon_kelvin, xHI_avg, Jalpha, rs-1, nHI, nHII, ne, TCMB/consts.kB) #Tspin kelvin
    Ts_prev[0] = Ts

    
    tauGP = tauGP_func(H,nHI)
    J0 = n_H * consts.c / 4 / np.pi / consts.Freq_Lya
    
    Continuum_Heat_tab, Injected_Heat_tab = pre.LyalphaHeat_Interps(Tbaryon_kelvin,Ts,tauGP)
    Continuum_Heat = Continuum_Heat_tab*Jalphastar_continuum/J0*Tbaryon
    Injected_Heat = Injected_Heat_tab*Jalphastar_injected/J0*Tbaryon
    return Continuum_Heat+Injected_Heat

def dTb_dloga_CMB(rs, abundances, Tbaryon, TCMB, Hubble):     
    P_CMB = 3/4*TCMB/consts.E_21*consts.A10 #s^-1
    return 2/3/Hubble*P_CMB*consts.E_21**2/consts.m_p*(1+2*Tbaryon/consts.E_21)*(1-abundances['HII'])/(1+abundances['HII'])/(1+abundances['He']) #Fix: Shouldn't be ionization['e']?

    

###########################################################################
#                   Coupled ODEs for xHII_IGM and Tbaryon                 #
###########################################################################


def ODEs_SM(log_a,y,cosmo, Jalphastar_interps, JalphaX_interp, interp_SFRD,xi_interp, heatrate_grid_xe_z,ion_grid_xe_z,Lya_Heat,CMB_Heat):
    '''
    Parameters:
        log_a is the natural log of the scale factor. This is the parameter according to which we differentiate.
        y is an array containing the set of values we wish to solve for.
    '''

                            # Update fluid and cosmological properties to current step
                            #---------------------------------------------------------
    rs = cosmo.rs_from_log_a(log_a)            # Redshift (1+z)
    TCMB = cosmo.TCMB(rs)                 # CMB temperature [eV]
    DeltaT_CMB_kinetic, xHII_IGM = y     # CMB - gas kinetic temperature [eV], fraction of ionized hydrogen in IGM (nHII/nH) - outside of fully ionized bubbles
    Tbaryon = TCMB - DeltaT_CMB_kinetic  # Kinetic temperature of gas [eV]
    xHII_IGM = max(xHII_IGM,1e-5)        # Cut xHII below 1e-5 for numerical stability
    
    Hubble = cosmo.hubble(rs)      # Hubble parameter at rs [1/s]
    n_H = cosmo.nH * rs **3        # Mean hydrogen density [1/cm^3]
    n_b = cosmo.nB * rs **3        # Mean baryon (H+He) density [1/cm^3]
    Q_HII = xi_interp(rs - 1) if xi_interp else 1e-10 # Volume filling factor of fully ionized bubbles
    
    abundances = {
        #Abundances of different species x = nx/nH. 
        #At z<50 IGM ionization (outside of ionized bubbles) is by X-rays, We assume HeI and HI ionize equally. 
        'e' : xHII_IGM * (1+ cosmo.nHe/cosmo.nH) if rs <= 50 else xHII_IGM ,  
        'HI' : (1 - xHII_IGM),
        'HII' : xHII_IGM,
        'He' : cosmo.nHe / cosmo.nH
    }


                                                # Ionization rate 
                                                #----------------
    # Ionized fraction outside of bubbles
    if rs>50:
        #Xion,Xheat = self.Xray_obj.(rs-1,xHII_IGM)
        dxe_dloga = dxHII_dloga_3level_caseB_recombination(rs, xHII_IGM, Tbaryon, n_H, Hubble, TCMB)
    else:
        dxe_dloga = dxHII_dloga_caseA_recombination(rs, xHII_IGM, Tbaryon, n_H, Hubble)
        
        #Ionization by X-ray photons
        Xion = ion_grid_xe_z([np.log10(xHII_IGM),rs-1]) # X-ray ionization rate per volume [1/cm^3/s]
        dxe_dloga += dxHII_dloga_Xray_ionization(rs, xHII_IGM, Xion, Hubble, n_H) # Ionization rate per hydrogen
            
                                                # Temperature change rate
                                                #------------------------
    # TCMB-Tb
    dTb_dloga = -2 * Tbaryon + dTb_dloga_Compton(rs, abundances, Tbaryon, Hubble, TCMB) + dT_dloga_NumberChange (rs,dxe_dloga,abundances,Tbaryon)
    if rs<=50:
        Xheat = heatrate_grid_xe_z([np.log10(abundances['HII']),rs-1]) # X-ray heat rate per volume [eV/cm^3/s]
        dTb_dloga += dTb_dloga_Xrays(rs, abundances, Xheat, Hubble, n_H)
        if Lya_Heat:
            dTb_dloga += dTb_dloga_Lya(rs, abundances, Q_HII, Tbaryon, Jalphastar_interps, JalphaX_interp,  n_H, Xheat, TCMB, Hubble)
        if CMB_Heat:
            dTb_dloga += dTb_dloga_CMB(rs, abundances, Tbaryon, TCMB, Hubble)
    
    dDeltaT_CMB_kinetic = -TCMB - dTb_dloga


    return np.array([dDeltaT_CMB_kinetic, dxe_dloga] , dtype=float)
