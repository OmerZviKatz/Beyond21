# This module loads different pre-processed data and makes interpolation tables for use in the code. 
# This includes: 
    # Color temperature and Salpha grids for computing Ts 
    # Ly-alpha heating rates
    # Scattering rates for e-H, p-H and H-H collisions for computing Ts
    # Secondary ionization and heating fractions for computing X-ray heating, ionization, and excitation rates


from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from importlib.resources import files
from beyond21.utils.interp_sorted_grid import reg_grid_interp


#######################################
# PopII and PopIII Lyman-band spectra #  
#######################################

spectral_distribution = np.transpose(np.loadtxt(files("beyond21").joinpath("data", "spectral_distribution"), skiprows=10)) 

########################################################
# Energy deposition fractions in X-ray photoionization #  
########################################################

# heat, Lya, and HI ionization fractions for E=3keV taken from 10.1111/j.1365-2966.2010.16401.x
dep_data = np.loadtxt(files("beyond21").joinpath("data", "furlanetto_stoever_secondary_ionization.dat"), skiprows=1)
fheat_interp = interp1d(dep_data[:,0], dep_data[:,2], bounds_error=False, fill_value="extrapolate")
fLya_interp = interp1d(dep_data[:,0], dep_data[:,4], bounds_error=False, fill_value="extrapolate") 
fion_interp = interp1d(dep_data[:,0], dep_data[:,5], bounds_error=False, fill_value="extrapolate") 


##################################################
# Interpolation tables collision spin flip rates #  
##################################################

# e-H collisions from 10.1111/j.1365-2966.2006.11169.x
Tgas, kappa_eH = np.loadtxt(open(files("beyond21").joinpath("data", "hydrogen_scattering_rates/eH scattering rate.csv"), "rb"), delimiter=',', usecols=(0, 1), unpack=True)
interp_kappa_eH =interp1d(Tgas ,kappa_eH ,fill_value="extrapolate")

# p-H collisions from 10.1111/j.1365-2966.2006.11169.x
Tgas, kappa_pH = np.loadtxt(open(files("beyond21").joinpath("data", "hydrogen_scattering_rates/pH scattering rate.csv"), "rb"), delimiter=',', usecols=(0, 1), unpack=True)
interp_kappa_pH =interp1d(Tgas ,kappa_pH ,fill_value="extrapolate")

# H-H collisions from 10.1086/427682
Tgas, kappa_HH = np.loadtxt(open(files("beyond21").joinpath("data", "hydrogen_scattering_rates/HH scattering rate.csv"), "rb"), delimiter=',', usecols=(0, 1), unpack=True)
interp_kappa_HH =interp1d(Tgas ,kappa_HH ,fill_value="extrapolate")


##############################################################
# Interpolation tables for color temperature and Stilde grid #  
##############################################################

# Load pre-computed grids in two regimes
# 1<Tk<500, 1<Ts<500, 1e4<tauGP<1e7 (should be sufficient for SM)
Ts_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/Ts"))
Tk_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/Tk"))
tauGP_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/tauGP"))
Stilde_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/S_tilde")) #Stilde(TK,Ts,taugGP)
TC_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/TC"))
TC_interp = reg_grid_interp(TC_arr,np.log10(Tk_arr) ,np.log10(Ts_arr),np.log10(tauGP_arr)).interp3D_sorted_single
Stilde_interp = reg_grid_interp(Stilde_arr,np.log10(Tk_arr) ,np.log10(Ts_arr),np.log10(tauGP_arr)).interp3D_sorted_single

# 0.01<Tk<300, 0.01<Ts<300, 1e4<tauGP<1e7 (required for BSM scenarios where TK can drop below 1K)
Ts_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/Ts_2"))
Tk_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/Tk_2"))
tauGP_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/tauGP_2"))
Stilde_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/S_tilde_2")) #Stilde(TK,Ts,taugGP)
TC_arr = np.load(files("beyond21").joinpath("data", "lya_coupling_grids/TC_2"))
TC_interp_2 = reg_grid_interp(TC_arr,np.log10(Tk_arr) ,np.log10(Ts_arr),np.log10(tauGP_arr)).interp3D_sorted_single
Stilde_interp_2 = reg_grid_interp(Stilde_arr,np.log10(Tk_arr) ,np.log10(Ts_arr),np.log10(tauGP_arr)).interp3D_sorted_single

def Salpha_Tc_Interp(tauGP,TK,Ts):
    #if tauGP>=1e5 and tauGP<=1e7 and TK>=2 and Ts>=2:
    if tauGP>1e7 or TK>500 or Ts>500 or tauGP<1e4:
        # Use fitting function from 10.1111/j.1365-2966.2005.09949.x
        Tc = 1 / (1 / TK + 0.405535 / TK * (1 / Ts - 1 / TK))  # effective color temperature in Kelvin (Hirata)
        xi = pow(1e-7 * tauGP, 1 / 3) * pow(TK, -2 / 3)  # dimensionless
        s_alpha = (1 - 0.0631789 / TK + 0.115995 / TK ** 2 - 0.401403 / Ts / TK + 0.336463 / Ts / TK ** 2) / (1 + 2.98394 * xi + 1.53583 * xi ** 2 + 3.85289 * xi ** 3)  # dimensionless
        
    elif TK>1 and TK<500 and Ts>1 and Ts<500 and tauGP>1e4 and tauGP<1e7:
        Tc = TC_interp_2([np.log10(TK),np.log10(Ts),np.log10(tauGP)])[0]
        s_alpha = Stilde_interp_2([np.log10(TK),np.log10(Ts),np.log10(tauGP)])[0]
        
    elif Ts>0.01 and Ts<300 and TK>0.001 and TK<10 and tauGP>1e4 and tauGP<1e7:
        Tc = TC_interp([np.log10(TK),np.log10(Ts),np.log10(tauGP)])[0]
        s_alpha = Stilde_interp([np.log10(TK),np.log10(Ts),np.log10(tauGP)])[0]

    else:
        raise ValueError(f"TK={TK}, Ts={Ts}, or tauGP={tauGP} are of bounds for Salpha,Tc grids")
        #raise ValueError("tauGP,TK or Ts out of bounds for Salpha,Tc grids")
    return [s_alpha,Tc]


#########################################
# Interpolation for Ly-alpha heat rates #  
#########################################

#Grids from 10.1103/physrevd.98.103513. Valid for Tbaryon,Ts in [0.1,100]K and tauGP in [1e4,1e7]
heffs = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/heffs.npy"))
Tbaryon_heffs = np.logspace(np.log10(0.1), np.log10(100.0), num=175) 
Ts_heffs = np.logspace(np.log10(0.1), np.log10(100.0), num=175)
tauGP_heffs = np.logspace(4.0, 7.0)

LyalphaHeat_Injected_heffs = reg_grid_interp(heffs[2,:,:,:],Tbaryon_heffs,Ts_heffs,tauGP_heffs).interp3D_sorted_single
LyalphaHeat_Continuum_heffs = reg_grid_interp(heffs[3,:,:,:],Tbaryon_heffs,Ts_heffs,tauGP_heffs).interp3D_sorted_single

#Pre-computed grid - valid for Tbaryon,Ts in [1e-3,10]K and tauGP in [1e4,1e7]
Ts_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/Ts_Heat"))
Tbaryon_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/Tk_Heat"))
tauGP_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/tauGP_Heat"))
LyalphaHeat_Injected_raw = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/HeatI")) #Stilde(Tbaryon,Ts,taugGP)
LyalphaHeat_Continuum_raw = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids/HeatC"))

LyalphaHeat_Injected_low = reg_grid_interp(LyalphaHeat_Injected_raw,Tbaryon_arr,Ts_arr,tauGP_arr).interp3D_sorted_single
LyalphaHeat_Continuum_low = reg_grid_interp(LyalphaHeat_Continuum_raw,Tbaryon_arr,Ts_arr,tauGP_arr).interp3D_sorted_single

#Pre-computed grid - valid for Ts in [100,300], Tk in [1,100], and tauGP in [1e4,1e7]
Ts_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids_highTk/Ts_Heat"))
Tbaryon_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids_highTk/Tk_Heat"))
tauGP_arr = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids_highTk/tauGP_Heat"))
LyalphaHeat_Injected_raw = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids_highTk/HeatI")) 
LyalphaHeat_Continuum_raw = np.load(files("beyond21").joinpath("data", "lya_heat_grids/LyalphaHeating_Grids_highTk/HeatC"))

LyalphaHeat_Injected_high = reg_grid_interp(LyalphaHeat_Injected_raw,Tbaryon_arr,Ts_arr,tauGP_arr).interp3D_sorted_single
LyalphaHeat_Continuum_high = reg_grid_interp(LyalphaHeat_Continuum_raw,Tbaryon_arr,Ts_arr,tauGP_arr).interp3D_sorted_single

def LyalphaHeat_Interps(Tk,Ts,tauGP):
    if Tk>100 or tauGP<1e4:
        #Irelevant at high temperatures or low tauGP (low xHI so Lya scatterings)
        Continuum_Heat = 0
        Injected_Heat = 0
    elif Tk>0.1 and Ts>0.1 and Ts<100 and tauGP>1e4 and tauGP<1e7:
        Continuum_Heat = LyalphaHeat_Continuum_heffs([Tk,Ts,tauGP])[0]
        Injected_Heat = LyalphaHeat_Injected_heffs([Tk,Ts,tauGP])[0]
    elif Tk>1 and Ts>100 and Ts<300 and tauGP>1e4 and tauGP<1e7:
        Continuum_Heat = LyalphaHeat_Continuum_high([Tk,Ts,tauGP])[0]
        Injected_Heat = LyalphaHeat_Injected_high([Tk,Ts,tauGP])[0]
    elif Tk>1e-3 and Ts>1e-3 and Ts<10 and tauGP>1e4 and tauGP<1e7:
        Continuum_Heat = LyalphaHeat_Continuum_low([Tk,Ts,tauGP])[0]
        Injected_Heat = LyalphaHeat_Injected_low([Tk,Ts,tauGP])[0]
    else:
        raise ValueError("tauGP,TK or Ts out of bounds for LyalphaHeat grids")

    return (Continuum_Heat,Injected_Heat)
