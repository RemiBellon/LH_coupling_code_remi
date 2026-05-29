'''
Physical parameters and configuration for modeling
'''
import math 
# =============================================
# Physics Constants
# =============================================
CONST = {
    'c0': 299792458,           # speed of light in vacuum (m/s)
    'qe': 1.602176634e-19,     # elementary charge (C)
    'me': 9.10938356e-31,      # electron mass (kg)
    'mi': 2.014*1.660539e-27,  # Deuterium mass (kg)
    'eps0': 8.854187817e-12,  # vacuum permittivity (F/m)
    'mu0': math.pi*4e-7,       # vacuum permeability (H/m)
}



# =============================================
# Antenna & wave parameters
# =============================================
# NO ANTENNA PARAMETER YET (we consider a single plane antenna with infinite extension in the vertical direction

WAVE = {
    'freq_LH': 3.7e9,     # Klystron frequency (Hz)
    'n_para': 2.,        # Parallel refractive index (imposed by multi-junctions phasing)
    'E_inc': 10.0,        # Incident electric field amplitude (V/m)
}

WAVE['omega_LH'] = 2*math.pi*WAVE['freq_LH']     # LH Wave angular frequency (rad/s)
WAVE['lambda0'] = CONST['c0']/WAVE['freq_LH']   # Vacuum LH wave wavelength (m)
WAVE['k0'] = WAVE['omega_LH']/CONST['c0']        # Free space wavenumber (1/m)


# =============================================
# FEM (+ PMLs) & mesh parameters
# =============================================
# DOMAIN parameters define the size of model box & the mesh resolution (before considering an adaptative mesh later)
DOMAIN = {
    'Lx_plasma': .04,                   # Plasma domain in radial direction (m)
    'Lx_pml': 0.01,                     # PLM domain in radial direction (m)
                                        # Total domain size in radial direction (m)
    'Lz_plasma': 0.15,                   # Plasma domain in toroidal direction (m)
    'Lz_pml': 0.1,                     # PLM domain in toroidal direction (m)

# Mesh resolution:
    'n_resol_per_wlgth': 8.,    
}
DOMAIN['Lx_tot'] = DOMAIN['Lx_plasma'] + DOMAIN['Lx_pml']
DOMAIN['Lz_tot'] = DOMAIN['Lz_plasma'] + 2*DOMAIN['Lz_pml']       # Total domain size in toroidal direction (m)

PML = {
    'Sx_r' : 1.0,
    'Sx_im': 1.0,  
    'px'   : 2.0,

    'Sz_r' : 1.0,
    'Sz_im': 2.0,  
    'pz'   : 2.0,
}

# =============================================
# Plasma parameters
# =============================================
PLASMA = {
    'B0': 3.7, # Total magnetic field at R_0 (T)
}
# =============================================
# Density profile type:
PLASMA['profile_type'] = 'constant_density'
PLASMA['ne_constant'] = 5e18
