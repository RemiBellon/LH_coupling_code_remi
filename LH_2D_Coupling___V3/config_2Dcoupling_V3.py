'''
Physical parameters and configuration for modeling
'''
import math 
# =============================================
# 1. Physical constants
# =============================================
CONST = {
    'c0': 299792458, # speed of light in vacuum (m/s)
    'qe': 1.602176634e-19, # elementary charge (C)
    'me': 9.10938356e-31, # electron mass (kg)
    'mi': 2.014*1.660539e-27, # Deuterium mass (kg)
    'eps_0' : 8.854187817e-12, # vacuum permittivity (F/m)
}


# =============================================
# 2. Geometry parameters
# =============================================
# GEOM parameters are defined from center of the tokamak
GEOM ={
    'R0' : 2.5,  # Great radius of WEST (m)
    'R_ant': 3.0 # Antenna radial position (m)
}

# =============================================
# 3. Antenna & wave parameters
# =============================================
# NO ANTENNA PARAMETER YET (we consider a single plane antenna with infinite extension in the vertical direction

WAVE = {
    'freq_LH': 3.7e9,     # Klystron frequency (Hz)
    'n_para': 2.0,        # Parallel refractive index (imposed by multi-junctions phasing)
    'E_inc': 10.0,        # Incident electric field amplitude (V/m)
}

WAVE['omega_wave'] = 2*math.pi*WAVE['freq_LH']     # LH Wave angular frequency (rad/s)
WAVE['k0'] = WAVE['omega_wave']/CONST['c0']  # Free space wavenumber (1/m)


# =============================================
# 4. FEM (+ PMLs) & mesh parameters
# =============================================
# DOMAIN parameters define the size of model box & the mesh resolution (before considering an adaptative mesh later)
DOMAIN = {
    'Lx_plasma': .04,                    # Plasma domain in radial direction (m)
    'Lx_pml': 0.01,                      # PLM domain in radial direction (m)
                                        # Total domain size in radial direction (m)
    'Lz_plasma_approx': 0.08,            # Plasma domain in toroidal direction (m)
    'Lz_pml': 0.05,                     # PLM domain in toroidal direction (m)
    
    'n_resol_isotropic': 200.0,

# Mesh resolution:
    'n_resol_per_wlgth': 4.,    

# Temporary unused:
    'nx_plasma': 250,           # Number of mesh points in plasma domain in radial direction
    'nx_pml': 50,               # Number of mesh points in PLM domain in radial direction
    'nz_plasma': 100,           # Number of mesh points in plasma domain in toroidal direction
    'nz_pml': 0,                # Number of mesh points in PLM domain in toroidal direction


    'periodic_z': False,
    'interp_poly_order': 2,     # Polynomial order for interpolation functions

}
DOMAIN['Lx_tot'] = DOMAIN['Lx_plasma'] + DOMAIN['Lx_pml']
DOMAIN['Lz_tot'] = DOMAIN['Lz_plasma_approx'] + 2*DOMAIN['Lz_pml']       # Total domain size in toroidal direction (m)


# =============================================
# 5. Plasma parameters
# =============================================
PLASMA = {
 # Magnetic field
    'B0_center_plasma': 3.7, # Total magnetic field at R_0 (T)
    'theta_B_deg': 0.0,      # Angle between B and horizontal plane (degrees)
    'phi_B_deg': 0.0,        # Angle between B and vertical plane (degrees)

# Particles density 
    'n_edge': 1e16,  # Density at the edge (m^-3)
    'n_core': 5e19,  # Density at the core (m^-3)
    'L_grad': 0.05,  # Caracteristic gradient lenght (m)
}

PLASMA['theta_B_rad'] = math.radians(PLASMA['theta_B_deg'])
PLASMA['phi_B_rad'] = math.radians(PLASMA['phi_B_deg'])

# =============================================
# Density profile type:
 
PLASMA['profile_type'] = 'constant_density'
PLASMA['ne_constant'] = 5e18

# PLASMA['profile_type'] = 'piecewise_linear_density'
PLASMA['lin_prof_x'] = [0.0, DOMAIN['Lx_plasma']/6, DOMAIN['Lx_plasma']]
PLASMA['lin_prof_n'] = [1e16, 7e18, 1e19]

# PLASMA['profile_type'] = 'exponential_density'