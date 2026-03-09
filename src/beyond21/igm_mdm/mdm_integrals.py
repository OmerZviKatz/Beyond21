from scipy.special import expi
import numpy as np
from beyond21.utils.interp_sorted_grid import sort_grid_interp
import beyond21.constants as consts
from importlib.resources import files

##########################################################
#Load Pre_processed_data and make interpolation functions#     Load grids for dimensionless part of IT,ID
##########################################################
with open(files("beyond21").joinpath("data", "mdm_integrals", "Atoms_n-4"), "rb") as gatom:
    Igridplotatom_n_minus4 = np.load(gatom)

xiatom_n_minus4 = Igridplotatom_n_minus4[:, 1, 0].flatten()          #Load epsilon values from grid
ratom_n_minus4 = Igridplotatom_n_minus4[1, :, 1].flatten()          #Load r1 values from grid
ITatom_n_minus4 = Igridplotatom_n_minus4[:, :, 2]         #Load IT(ions) values from grid
IDatom_n_minus4 = Igridplotatom_n_minus4[:, :, 3]         #Load ID(ions) values from grid

ITatoms_n_minus4_interpolating_function = sort_grid_interp(ITatom_n_minus4,xiatom_n_minus4,ratom_n_minus4).interp2D_sorted_single
IDatoms_n_minus4_interpolating_function = sort_grid_interp(IDatom_n_minus4,xiatom_n_minus4,ratom_n_minus4).interp2D_sorted_single


with open(files("beyond21").joinpath("data", "mdm_integrals", "Ions_n-4"), "rb") as gion:
    IgridplotCDM = np.load(gion)
epsCDM = IgridplotCDM[:, 1, 0].flatten()          #Load epsilon values from grid
rCDM = IgridplotCDM[1, :, 1].flatten()          #Load r1 values from grid
ITCDM = IgridplotCDM[:, :, 2]         #Load IT(ions) values from grid
IDCDM = IgridplotCDM[:, :, 3]         #Load ID(ions) values from grid

ITCDM_interpolating_function = sort_grid_interp(abs(ITCDM),epsCDM,rCDM).interp2D_sorted_single
IDCDM_interpolating_function = sort_grid_interp(abs(IDCDM),epsCDM,rCDM).interp2D_sorted_single

def ITID_calc(redmass_A, redmass_e, sigmabar, uth2, Ion_Atom, eps = None, r = None, xi = None):
    #Return: a length 2 array of [IT,ID] for the IDM-SM interaction of interest.

    n = -4
    alphaEM = consts.alphaEM

    if Ion_Atom == 'CDM':
        cD = sigmabar * pow(alphaEM * consts.m_e, 4) / (8 * pow(redmass_A * redmass_e, 2) * np.sqrt(uth2))  # The coefficient outside of the ID integral
        cT = np.sqrt(2 / np.pi) * cD  # The coefficient outside of the IT integral
        eps_r = [eps, r]

        if r > pow(10, -10):
            IntIT = ITCDM_interpolating_function(eps_r)[0]  # Interpolate according to n=-4 Ions grid.
            IntID = IDCDM_interpolating_function(eps_r)[0]
        else:
            # At sufficiently small r values we can approximate the integrals analytically
            if eps < 30:
                exp_times_expi = np.exp(eps ** 2 / 2) * expi(-eps ** 2 / 2)
                IntIT = -(1. + 0.5 * exp_times_expi * (2 + eps ** 2))
                IntID = r ** 2 * (-20 + 2 * r ** 2 * (5 + eps ** 2) + exp_times_expi * (-10 * (2 + eps ** 2) + r ** 2 * (6 + 7 * eps ** 2 + eps ** 4))) / (30 * np.sqrt(2 * np.pi))

            else:
                # At large epsilons we get overflows so another approximation is required
                IntIT = -(-4 / eps ** 4 + 32 / eps ** 6 - 288 / eps ** 8 + 3072 / eps ** 10 - 38400 / eps ** 12 + 552960 / eps ** 14 - 9031680 / eps ** 16)
                IntID = 2 * np.sqrt(2) * r ** 2 * (2257920 * (10 + 13 * r ** 2) - 138240 * (10 + 11 * r ** 2) * eps ** 2 + 9600 * (10 + 9 * r ** 2) * eps ** 4 - 768 * (10 + 7 * r ** 2) * eps ** 6 + 360 * (2 + r ** 2) * eps ** 8 - 8 * (10 + 3 * r ** 2) * eps ** 10 + (10 + r ** 2) * eps ** 12) / (15 * np.sqrt(np.pi) * eps ** 16)

    elif Ion_Atom == 'Ion':
        # In this case we calculate the integrals for ions.

        qref = alphaEM * consts.m_e
        cD = sigmabar * pow(np.sqrt(uth2), n + 3) * pow(2, n + 1) * pow(redmass_A / qref,n)  # The coefficient outside of the ID integral
        cT = np.sqrt(2 / np.pi) * cD  # The coefficient outside of the IT integral

        eps_r = [eps, r]
        if r > pow(10, -10):
            IntIT = ITCDM_interpolating_function(eps_r)[0]  
            IntID = IDCDM_interpolating_function(eps_r)[0]   
        else:
            # At sufficiently small r values we can approximate the integrals
            if eps < 30:
                exp_times_expi = np.exp(eps ** 2 / 2) * expi(-eps ** 2 / 2)
                IntIT = -1 - 0.5 * (2 + eps ** 2) * exp_times_expi
                IntID = r ** 2 * (-20 + 2 * r ** 2 * (5 + eps ** 2) + exp_times_expi * (-10 * (2 + eps ** 2) + r ** 2 * (6 + 7 * eps ** 2 + eps ** 4))) / (30 * np.sqrt(2 * np.pi))

            else:
                # At large epsilons we get overflows so another approximation is required.
                IntIT = 4 / eps ** 4 - 32 / eps ** 6 + 288 / eps ** 8 - 3072 / eps ** 10 + 38400 / eps ** 12 - 552960 / eps ** 14 + 9031680 / eps ** 16
                IntID = 2 * np.sqrt(2) * r ** 2 * (2257920 * (10 + 13 * r ** 2) - 138240 * (10 + 11 * r ** 2) * eps ** 2 + 9600 * (10 + 9 * r ** 2) * eps ** 4 - 768 * (10 + 7 * r ** 2) * eps ** 6 + 360 * (2 + r ** 2) * eps ** 8 - 8 * (10 + 3 * r ** 2) * eps ** 10 + (10 + r ** 2) * eps ** 12) / (15 * np.sqrt(np.pi) * eps ** 16)
            
    elif Ion_Atom in ['HI','He']:
        Z = 1
        if Ion_Atom == 'He':
            Z = 2
        qref = alphaEM * consts.m_e
        cD = pow(Z, 2) * sigmabar * pow(np.sqrt(uth2), n + 3) * pow(2, n + 1) * pow(redmass_A / qref,n)  # The coefficient outside of the ID integral
        cT = np.sqrt(2 / np.pi) * cD  # The coefficient outside of the IT integral
        xi_r = [xi, r]

        if r > pow(10, -10):
            IntIT = ITatoms_n_minus4_interpolating_function(xi_r)[0]  # Interpolate according to grids
            IntID = IDatoms_n_minus4_interpolating_function(xi_r)[0]
        else:
            # At sufficiently small r values we can approximate the integrals
            if xi < 10:
                exp_times_expi = np.exp(xi ** 2 / 2) * expi(-xi ** 2 / 2)
                IntIT = (-12 * (-4 - 8 * xi ** 2 + xi ** 4) + 2 * r ** 2 * (36 - 64 * xi ** 2 + xi ** 4 + xi ** 6) + exp_times_expi * (-6 * (48 - 24 * xi ** 2 - 6 * xi ** 4 + xi ** 6) + r ** 2 * (144 - 72 * xi ** 2 - 66 * xi ** 4 + 3 * xi ** 6 + xi ** 8))) / 288
                IntID = r ** 2 * (-20 * (-4 - 8 * xi ** 2 + xi ** 4) + 2 * r ** 2 * (36 - 64 * xi ** 2 + xi ** 4 + xi ** 6) + exp_times_expi * (-10 * (48 - 24 * xi ** 2 - 6 * xi ** 4 + xi ** 6) + r ** 2 * (144 - 72 * xi ** 2 - 66 * xi ** 4 + 3 * xi ** 6 + xi ** 8))) / (720 * np.sqrt(2 * np.pi))
            else:
                # At large xi values we get overflows so another approximation is required.
                IntIT = 8 * (6 + r ** 2) / (3 * xi ** 4) - 96 * (2 + r ** 2) / xi ** 6 + 400 * (6 + 5 * r ** 2) / xi ** 8 - 5632 * (6 + 7 * r ** 2) / xi ** 10
                IntID = 8 * np.sqrt(2 / np.pi) * r ** 2 * (-2112 * (10 + 7 * r ** 2) + 750 * (2 + r ** 2) * xi ** 2 - 12 * (10 + 3 * r ** 2) * xi ** 4 + (10 + r ** 2) * xi ** 6) / (15 * xi ** 10)

    return [cT * IntIT, cD * IntID]  # Returns a 1D array. 1st entry is IT value, 2nd entry is ID value