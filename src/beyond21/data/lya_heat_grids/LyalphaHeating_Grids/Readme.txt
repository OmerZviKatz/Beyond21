heff:
#####

Source - https://github.com/ntveem/lyaheating

Build - heff[Tk index][Ts index][tauGP index][Continuum= 0 or  Injected = 1] 
[net energy loss efficiency (epsilons) = 0, energy loss efficiency to spins = 1, \tilde{salpha} = 2, Tceff temperature (K) = 3]

Range - 
	Tk: np.logspace(np.log10(0.1), np.log10(100.0), num=175) [Kelvin]
	Ts: np.logspace(np.log10(0.1), np.log10(100.0), num=175) [Kelvin]
	tauGP: np.logspace(4.0, 7.0)


ClusterGrids:
#############

Build - 
	HeatI[Tk index][Ts index][tauGP index] gives the epsilon for injected
	HeatC[Tk index][Ts index][tauGP index] gives the epsilon for continuum
Range:
	Tk_Heat: np.logspace(-3, 0, 60) [Kelvin]
	Ts_Heat: np.logspace(-3, 1, 80) [Kelvin]
	tauGP_Heat: np.logspace(4.0, 7.0, 60)
	
