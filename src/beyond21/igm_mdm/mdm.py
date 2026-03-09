import numpy as np
import beyond21.igm_mdm.mdm_integrals as ITID
import beyond21.igm_mdm.inter_galactic_medium as IGM
import beyond21.constants as consts
import warnings


global Ts_prev
Ts_prev = [0]


#                          Preliminary Functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def reduced_mass(m1,m2):
    return m1*m2/(m1+m2)

def u2th(m1,m2,T1,T2):
    return T1/m1+T2/m2

def v_small(V,limit):
    if V>limit:
        small_v=1
    else:
        small_v=0
    return small_v

def eps_calc(m_phi, redmass_m, uth2):
    ''' Dimensionless parameter epsilon required for I_T,I_D calculation for ions '''
    return (m_phi / (2 * redmass_m * np.sqrt(uth2)))

def r_calc(V_rel, uth2):
    ''' Dimensionless parameter r required for I_T,I_D calculation for ions and atoms '''
    return (V_rel / np.sqrt(uth2))

def xi_calc(redmass, uth2, specie):
    ''' Dimensionless parameter xi required for I_T,I_D calculation for atoms '''
    if specie == 'HI':
        aeff = consts.r_Bohr
        Z = 1
    elif specie == 'He':
        aeff = consts.r_Bohr / 1.69
        Z = 2
    xi = 1 / (aeff * redmass * np.sqrt(uth2))                         
    return (xi)

#################################################################################################################################
#                                                          Coupled Fluids Equations                                             #
#################################################################################################################################

def ODEs_2cDM(log_a,y,cosmo, J_alpha_interps,JalphaX_interp,xi_interp, heatrate_grid_xe_z,ion_grid_xe_z,Lya_Heat,CMB_Heat,f_m,m_m,m_c,sigma_e,alphaI_alphaC,m_phi,IDM,CDM):
    '''
    Parameters:
        log_a is the natural log of the scale factor. This is the parameter according to which we differentiate.
        y is an array containing the set of values we wish to solve for.
    '''

    # Solution vector at current itteration
    #--------------------------------------
    DeltaT_CMB_baryon, xHII_IGM, DeltaT_baryon_IDM, logV_rel_IDM_baryon, log_TCDM, logV_rel_IDM_CDM = y
    rs = cosmo.rs_from_log_a(log_a)

    # Temperatures and velocity at current itteration
    #------------------------------------------------
    T_CMB = cosmo.TCMB(rs)
    Tbaryon = T_CMB - DeltaT_CMB_baryon
    TIDM = Tbaryon - DeltaT_baryon_IDM
    TCDM = np.exp(log_TCDM)
    V_IB = np.exp(logV_rel_IDM_baryon)
    V_IC = np.exp(logV_rel_IDM_CDM)

    H_Hz = cosmo.hubble(rs)    # Hubble constant at rs. Units: [1/s]
    H = H_Hz * consts.hbar     # Hubble constant at rs. Units: [eV]
    

    # Abundances at current itteration
    #---------------------------------
    Q_HII = xi_interp(rs - 1) if xi_interp else 1e-10 # Volume filling factor of fully ionized bubbles
    xHII_IGM = max(xHII_IGM,1e-5)        # Cut xHII below 1e-5 for numerical stability

    abundances = {
        #Abundances of different species x = nx/nH. 
        #At z<50 IGM ionization (outside of ionized bubbles) is by X-rays, We assume HeI and HI ionize equally. 
        'e' : xHII_IGM * (1+ cosmo.nHe/cosmo.nH) if rs <= 50 else xHII_IGM ,  
        'HI' : (1 - xHII_IGM),
        'HII' : xHII_IGM,
        'He' : cosmo.nHe / cosmo.nH
    }

    n_H_cm = cosmo.nH * (rs ** 3)
    n_H = cosmo.nH * (rs ** 3) * (consts.hbar * consts.c) ** 3
    Ndensity = {key: n_H * fraction for key, fraction in abundances.items()}

    rho_DM = cosmo.rho_DM * (rs ** 3) * (consts.hbar * consts.c) ** 3
    rho_baryon = cosmo.rho_baryon * (rs ** 3) * (consts.hbar * consts.c) ** 3
    rho_IDM = rho_DM * f_m
    rho_CDM = rho_DM * (1 - f_m)

    # Reduced masses and u_th current itteration
    #-------------------------------------------

    species = ['e','HII','HI','He','CDM','IDM']
    
    specie_mass = {
        'e': consts.m_e,
        'HI': consts.m_p,
        'HII': consts.m_p,
        'He': consts.m_He,
        'CDM': m_c,
        'IDM': m_m
    } #Mass in eV for all species

    Temperatures = {key: Tbaryon for key in species[:-2]} 
    Temperatures['CDM'] = TCDM
    Temperatures['IDM'] = TIDM #Temperature in eV for all species

    red_mass = {key: reduced_mass(specie_mass[key],m_m) for key in species[:-1]} #Reduced mass with IDM in [eV]

    IDM_term = TIDM / m_m
    u2th = {key: Temperatures[key]/specie_mass[key] + IDM_term for key in species[:-1]} #u2th with IDM [dimless]



    # Calculate IT, ID integrals (Eqs A18, A19) with IDM for all species
    # --------------------------------------------------------------------
    Integrals = {key: 0 for key in species[:-1]}

    if IDM:     
        # Classical cross sections - Eq B11
        xsec =  {key: sigma_e * pow(red_mass[key] / red_mass['e'], 2) for key in species[0:2]} #Ions
        xsec.update({key: sigma_e * pow(red_mass[key] / red_mass['e'], 2)/2 for key in species[2:4]}) #Atoms
        Q = np.sqrt(consts.alphaEM ** 4 * consts.m_e ** 4 * xsec['e'] / (2*16 * np.pi * consts.alphaEM ** 2*red_mass['e']**2)) #2*16 and not 16 because of classical. We get sigma_e as input so we have to accound for that when deriving Q

        # Debye mass - mediator is SM photon. Debye mass regulates the integrals
        m_gamma = np.sqrt(4 * np.pi*consts.alphaEM * Ndensity['HII'] / Tbaryon)  
        
        # Integrals
        for i,key in enumerate(species[0:4]):
            r = r_calc(V_IB, u2th[key])

            if i<2: #Ions
                eps = eps_calc(np.sqrt(4 * m_gamma * Q * consts.alphaEM * red_mass[key] / np.exp(1)), 
                             red_mass[key], u2th[key])
                Integrals[key] = ITID.ITID_calc(red_mass[key], red_mass['e'], xsec[key], 
                                                u2th[key],'Ion', eps = eps, r = r)
            elif i<4: #Atoms
                xi = xi_calc(red_mass[key], u2th[key], key)
                Integrals[key] = ITID.ITID_calc(red_mass[key], red_mass['e'], xsec[key], 
                                                u2th[key], key, r = r, xi = xi)        
        if CDM:
            # Cross section
            sigma_CDM_IDM = 2 * 16 * np.pi * alphaI_alphaC * pow(red_mass['e'], 2) / pow(consts.alphaEM * consts.m_e,4)  

            # DM-CDM interactions are n=-4 which we regulate with the mediating particle's mass. 
            # Only log sensitive to mass. Follow eq B11
            m_phi = np.sqrt(16*np.pi**2*(1-f_m)*rho_DM*np.sqrt(alphaI_alphaC)/(m_c*TCDM))
            eff_m_phi=np.sqrt(4 * m_phi * np.sqrt(alphaI_alphaC) * red_mass['CDM'] / np.exp(1)) #Eq B11
            
            q_typ = red_mass['CDM'] * np.maximum(V_IC, np.sqrt(u2th['CDM'])) 
            if eff_m_phi*10 > q_typ:
                warnings.warn(
                "The effective mediator mass is comparable or larger than the typical momentum transfer in mDM-CDM interactions. The results may be inaccurate.",
                UserWarning
            )

            #Integrals
            eps_c = eps_calc(eff_m_phi, red_mass['CDM'], u2th['CDM']) 
            r_c = r_calc(V_IC, u2th['CDM'])
        
            Integrals['CDM'] = ITID.ITID_calc(red_mass['CDM'], red_mass['e'], sigma_CDM_IDM, 
                                              u2th['CDM'], 'CDM', eps = eps_c, r = r_c)



    # Solve Evolution Equations
    #------------------------------
    # This is based on c8-c13 of 1908.06986
    # Notice that we solve for temperature differences or log(temp) and for log(V)

                                    # Ionized fraction outside of bubbles
    if rs>50:
        dxe_dloga = IGM.dxHII_dloga_3level_caseB_recombination(rs, xHII_IGM, Tbaryon, n_H_cm, H_Hz, T_CMB)
    else:
        dxe_dloga = IGM.dxHII_dloga_caseA_recombination(rs, xHII_IGM, Tbaryon, n_H_cm, H_Hz)

        # Xray contribution
        Xion = ion_grid_xe_z([np.log10(xHII_IGM),rs-1]) # X-ray ionization rate per volume [1/cm^3/s]
        dxe_dloga += IGM.dxHII_dloga_Xray_ionization(rs, xHII_IGM, Xion, H_Hz, n_H_cm)

    
            
                                                # TCMB-Tbaryon

    dTb_dloga = -2 * Tbaryon + IGM.dTb_dloga_Compton(rs, abundances, Tbaryon, H_Hz, T_CMB)  + IGM.dT_dloga_NumberChange (rs,dxe_dloga,abundances,Tbaryon)
    if IDM:
        fact = 2 * rho_IDM / (3 * H * (1 + abundances['e'] + abundances['He']))
        for key in species[0:4]:
           dTb_dloga += fact * (abundances[key] * red_mass[key]/(m_m + specie_mass[key])) * (Integrals[key][1]+Integrals[key][0]*(TIDM - Tbaryon)/(m_m*u2th[key])) 

    if rs<=50:
        # Xray contribution
        Xheat = heatrate_grid_xe_z([np.log10(abundances['HII']),rs-1]) # X-ray heat rate per volume [eV/cm^3/s]
        dTb_dloga += IGM.dTb_dloga_Xrays(rs, abundances, Xheat, H_Hz,n_H_cm)
        
        if Lya_Heat:
            dTb_dloga += IGM.dTb_dloga_Lya(rs, abundances, Q_HII, Tbaryon, J_alpha_interps, JalphaX_interp, n_H_cm, Xheat, T_CMB, H_Hz)
        if CMB_Heat:
            dTb_dloga += IGM.dTb_dloga_CMB(rs, abundances, Tbaryon, T_CMB,H_Hz)
    
    dDeltaT_CMB_baryon=-T_CMB-dTb_dloga
    
                                                # Tbaryon - TIDM
    #Even if IDM is off we redshift TIDM and just don't use it anywhere
    dTm_dloga = -2 * TIDM  

    if IDM:   
        fact = 2/(3*H)
        for key in species[0:4]:
            dTm_dloga += fact * (Ndensity[key] * specie_mass[key] * red_mass[key] / (m_m + specie_mass[key])) *(Integrals[key][1] + Integrals[key][0] * (Tbaryon - TIDM) / (specie_mass[key] * u2th[key]))
        if CDM:
            fact = 2*rho_CDM/(3*H)
            dTm_dloga += fact * (red_mass['CDM']/(m_m+m_c)) * (Integrals['CDM'][1]+Integrals['CDM'][0]*(TCDM-TIDM)/(m_c*u2th['CDM']))
    dDeltaT_baryon_IDM=dTb_dloga-dTm_dloga


        
                                                # log(T_CDM)
    #Even if IDM or CDM is off we redshift Tc and just don't use it anywhere
    dlogTc_dloga = -2
        
    if CDM:
        fact = 2 * rho_IDM / (3 * H)
        IDM_contribution = (red_mass['CDM'] / (m_m + m_c)) * (Integrals['CDM'][1] + Integrals['CDM'][0] * (TIDM - TCDM) / (m_m * u2th['CDM']))
        dlogTc_dloga += fact * IDM_contribution/TCDM


                                                # log(Vmb)
    d_rs = 5.0
    S = 0.5 * (1.0 - np.tanh((rs - 1011.0) / d_rs))  # Smoothed step function in rs implementing the high-z velocity prescription
                                                     # used in arXiv:1908.06986. Future versions may include a more detailed treatment.
    limit = pow(10, -30) #The minimal Vrel value below which we set Vrel=0 to avoid overflows

                                                
    #If IDM is off we keep Vmb constant and just don't use it
    dlogVrel_mb_dloga = 0 

    if IDM:
        #Adiabatic
        dVrel_mb_dloga = - V_IB 
        
        #Drag
        fact = (rho_IDM / rho_baryon) + 1
        DragTerm_SM_IDM = 0
        for key in species[0:4]:
            DragTerm_SM_IDM += Ndensity[key] * specie_mass[key] / (m_m + specie_mass[key]) * Integrals[key][1] / (H * V_IB)
        dVrel_mb_dloga += -fact * DragTerm_SM_IDM
            
        if CDM:
            dVrel_mb_dloga += rho_CDM / (m_m + m_c) * Integrals['CDM'][1] / (H * V_IC)
        dVrel_mb_dloga *= S
        dlogVrel_mb_dloga = dVrel_mb_dloga / V_IB
        
        if (V_IB < limit and dVrel_mb_dloga < 0):
            # If V_rel<=10^-15 it is effectively 0 so no need to make it smaller. Avoid numerical issues
            dlogVrel_mb_dloga = 0

                                                    # log(Vmc)
    #If CDM is off we keep Vmc constant and just don't use it
    dlogVrel_mc_dloga = 0
    
    if CDM:
        fact = (rho_IDM + rho_CDM) / (m_m + m_c)
        dVrel_mc_dloga = -V_IC #Adiabatic
        dVrel_mc_dloga += DragTerm_SM_IDM
        dVrel_mc_dloga += -fact * Integrals['CDM'][1] / (H * V_IC) #CDM-IDM term
        dVrel_mc_dloga *= S
        dlogVrel_mc_dloga = dVrel_mc_dloga / V_IC
        
        if (V_IC < limit and dlogVrel_mc_dloga < 0):
                dlogVrel_mc_dloga = 0

    return np.array([dDeltaT_CMB_baryon, dxe_dloga, dDeltaT_baryon_IDM, dlogVrel_mb_dloga, dlogTc_dloga,dlogVrel_mc_dloga] , dtype=np.float64)