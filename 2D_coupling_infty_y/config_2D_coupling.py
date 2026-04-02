'''
Physical parameters and configuration for modeling
'''
import math 
# =============================================
# 1. Physical constants
# =============================================
CONST = {
    'c0': 299792458, # speed of light in vacuum (m/s)
    'q_e': 1.602176634e-19, # elementary charge (C)
    'm_e': 9.10938356e-31, # electron mass (kg)
    'm_i': 2.014*1.660539e-27, # Deuterium mass (kg)
    'eps_0' : 8.854187817e-12, # vacuum permittivity (F/m)
}


# =============================================
# 2. Geometry parameters
# =============================================
# GEOM parameters are defined from center of the tokamak
GEOM ={
    'R0' : 2.5, # Great radius of WEST (m)
    'R_ant': 3.0 # Antenna radial position (m)
}

# =============================================
# 3. Plasma parameters
# =============================================
PLASMA = {
 # Magnetic field
    'B0_center_plasma': 3.7, # Total magnetic field at R_0 (T)
    'theta_B_deg': -5.0,     # Angle between B and horizontal plane (degrees)
    'phi_B_deg': 0.0,        # Angle between B and vertical plane (degrees)

# Particles density 
    'n_edge': 1e16,  # Density at the edge (m^-3)
    'n_core': 5e19,  # Density at the core (m^-3)
    'L_grad': 0.05,  # Caracteristic gradient lenght (m)
}

PLASMA['theta_B_rad'] = math.radians(PLASMA['theta_B_deg'])
PLASMA['phi_B_rad'] = math.radians(PLASMA['phi_B_deg'])

# =============================================
# 4. Antenna & wave parameters
# =============================================
# NO ANTENNA PARAMETER YET (we consider a single plane antenna with infinite extension in the vertical direction

WAVE = {
    'f': 3.7e9,    # Klystron frequency (Hz)
    'n_para': 2.0, # Parallel refractive index (imposed by multi-junctions phasing)
    'E_inc': 1.0,  # Incident electric field amplitude (V/m)
}

WAVE['omega_wave'] = 2*math.pi*WAVE['f']     # LH Wave angular frequency (rad/s)
WAVE['k0'] = WAVE['omega_wave']/CONST['c0'] # Free space wavenumber (1/m)

# =============================================
# 5. FEM (+ PMLs) & mesh parameters
# =============================================
# DOMAIN parameters define the size of model box & the mesh resolution (before considering an adaptative mesh later)
DOMAIN = {
    'Lx_plasma': 0.1,  #Plasma domain in radial direction (m)
    'Lx_pml': 0.05,    #PLM domain in radial direction (m)
    'Lz_plasma': 0.4,         # Plasma domain in toroidal direction (m)
    'Lz_pml': 0.05,    # PLM domain in toroidal direction (m)

# Mesh resolution:
    'nx_plasma': 50,  # Number of mesh points in plasma domain in radial direction
    'nx_pml': 15,      # Number of mesh points in PLM domain in radial direction
    'nz_plasma': 100,         # Number of mesh points in plasma domain in toroidal direction
    'nz_pml': 15,      # Number of mesh points in PLM domain in toroidal direction
    'order': 2,          # Polynomial order for interpolation functions

# PMLs: attenuation parameters 
    'sigma_max_factor': 1e2,   # Maximum conductivity in PML (S/m)
    'degree': 2.0,             # Grading order for conductivity profile
}
DOMAIN['Lx_tot'] = DOMAIN['Lx_plasma'] + DOMAIN['Lx_pml'] # Total domain size in radial direction (m)
DOMAIN['Lz_tot'] = DOMAIN['Lz_plasma'] + 2*DOMAIN['Lz_pml']       # Total domain size in toroidal direction (m)