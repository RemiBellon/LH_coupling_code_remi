from solver_Hcurl_3D_V3 import * 
import config_2Dcoupling_V3 as cfg              # config = physical & simulation parameters 

import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.patheffects as pe


from time import *
import os
import datetime
import json
from pathlib import Path


# figure_save_dir = Path("/home/remi/Perso/Stage/M2_IRFM/Codes/LH_2D_Coupling___V3/Figures")
figure_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Figures")
figure_save_dir.mkdir(parents=True, exist_ok=True)
solver = LHCouplingSolver_Hcurl3D(cfg.__dict__)

# ======================================================================================================
#   Run Files Management
# ======================================================================================================
def setup_output_directory(base_folder="Results"):
    """
    Make a unique directory based on date and time to save run data. 
    Example: Results/Run_20260408_103015/
    """
    # Recover date & time then make the dir:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_folder, f"Run_{now}")
    
    # Make parent dir if doesn't exist:
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[SYSTEM] Dossier de sauvegarde créé : {run_dir}")
    return run_dir

def save_configuration(cfg_dict, save_dir):
    """
    Save simulation config dict in JASON format. And ignore system python object: Keep only dict and numpy variables 
    """
    config_path = os.path.join(save_dir, "simulation_parameters.json")
    
    # System variables filter:
    clean_dict = {}
    for key, value in cfg_dict.items():
        # Ignore every python object starting with "__" (ex: __builtins__, __name__, __file__)
        if not key.startswith('__') and isinstance(value, dict):
            clean_dict[key] = value

    # Encode the dict data:
    class ScientificEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy table to standard python lists
            try:
                # Try classic encoding:
                return super().default(obj)
            except TypeError:
                # If weird object consvert it to string format to avoid crashing
                return str(obj)

    # Save
    with open(config_path, 'w') as f:
        json.dump(clean_dict, f, indent=4, cls=ScientificEncoder)
    
    print(f"[SYSTEM] Configuration save in: {config_path}")


# ======================================================================================================
#   General Function to Compute Cutoff Density Layer & Position  
# ======================================================================================================
def create_density_profile(x_val,z_val, solver):
# Detect if the input is a symbolic FEM variable
    # TYPE is_ngsolve: bool
    is_ngsolve = type(x_val).__name__ == 'CoefficientFunction'
    print(f'--- [Create_density_profile]: is_ngsolve = {is_ngsolve} ---')
    # TYPE prof_type: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'constant_density')

    if prof_type == 'constant_density':
        print(f'create_density_profile: prof_type = {prof_type}')
        # TYPE ne_constant: float
        ne_constant = solver.cfg['PLASMA']['ne_constant']
        if is_ngsolve:
            # NGSolve accepts native floats as constants
            # TYPE: float
            return ne_constant
        else:
            # Numpy needs an array of the same shape
            # TYPE: np.ndarray
            return np.full_like(x_val, ne_constant)

    elif prof_type == 'exponential_density':
        print('create_density_profile: prof_type = ', prof_type)
        # TYPE Lx_plasma: float
        Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
        # TYPE n_edge, n_core: float
        n_edge = solver.cfg['PLASMA']['lin_prof_n'][0]
        n_core = solver.cfg['PLASMA']['lin_prof_n'][-1]
        # TYPE lambda_val: float
        lambda_val = np.log(n_core) / Lx_plasma
        
        if is_ngsolve:
            # TYPE: ngsolve.CoefficientFunction
            return exp(lambda_val * x_val)              # ne_exp = n_core * exp (x_val/Lx_plasma)
        else:
            # TYPE: np.ndarray
            return np.exp(lambda_val * x_val)           # " " " " " "

    elif prof_type == 'piecewise_linear_density':
        # We set the (x,y) coord of points that the curve must pass through
        print('create_density_profile: prof_type = ', prof_type)
        # TYPE x_pts, n_pts: np.ndarray
        x_pts = np.array(solver.cfg['PLASMA']['lin_prof_x'])
        ne_pts = np.array(solver.cfg['PLASMA']['lin_prof_n'])
        # TYPE smooth_width: float
        smooth_width = solver.cfg['PLASMA'].get('smooth_width', 0.006)
        print('create_density_profile: smooth_width = ', smooth_width)
        # TYPE slope_0: float
        slope_0 = (ne_pts[1] - ne_pts[0]) / (x_pts[1] - x_pts[0])
        # TYPE profile: np.ndarray | ngsolve.CoefficientFunction
        profile = ne_pts[0] + slope_0 * (x_val - x_pts[0])

        for i in range(1, len(x_pts) - 1):
            # Float conversion forces clean type matching for NGSolve nodes
            # TYPE x_c, n_c: float
            x_c, n_c = float(x_pts[i]), float(ne_pts[i])
            # TYPE s_prev, s_next: float
            s_prev = float((ne_pts[i] - ne_pts[i-1]) / (x_pts[i] - x_pts[i-1]))
            s_next = float((ne_pts[i+1] - ne_pts[i]) / (x_pts[i+1] - x_pts[i]))
            
            # TYPE L_prev, L_next: np.ndarray | ngsolve.CoefficientFunction
            L_prev = n_c + s_prev * (x_val - x_c)
            L_next = n_c + s_next * (x_val - x_c)
            
            # TYPE dx: np.ndarray | ngsolve.CoefficientFunction
            dx = x_val - x_c
            
            # MATH ENGINE ROUTING
            if is_ngsolve:
                # TYPE H_smooth: ngsolve.CoefficientFunction
                H_smooth = 0.5 * (1.0 + dx / sqrt(dx**2 + smooth_width**2))
            else:
                # TYPE H_smooth: np.ndarray
                H_smooth = 0.5 * (1.0 + dx / np.sqrt(dx**2 + smooth_width**2))
            
            profile = profile + (L_next - L_prev) * H_smooth
            
        return profile
        
    else:
        raise ValueError(f"Unknown profile type '{prof_type}'.")


# ==========================================================================
# 2D solver Benchmark
# ==========================================================================

def plot_wave_E_field_2D_map(mesh, gfu, cfg, figure_save_dir, 
                             component='Ez', value_type='real', 
                             plot_e_vectors=False, resolution=(300, 300)):
    """
    Plots the 2D Electrical field wave maps. 
    
    Parameters:
    - component: 'Ex', 'Ey', 'Ez', or 'norm'.
    - value_type: 'real' (instantaneous wave: E.real) or 'abs' (envelope amplitude: abs(E)).
    - plot_e_vectors: Boolean. If True, overlays the (Ex, Ez) vector field quiver.
    - resolution: Tuple (nx, nz) for grid resol.
    """
    print(f"--- Generating 2D E-Field Map: | Component: {component} | Type: {value_type} ---")
    
    # Geometry and Configuration Extraction from config dict file
    Lx_tot = cfg.DOMAIN['Lx_tot']
    Lx_plasma = cfg.DOMAIN['Lx_plasma']
    Lz_exact = cfg.DOMAIN['Lz_exact']
    
    # Reconstruct the 3D Field mathematically 
    Ep = gfu.components[0] # (Ex, Ez)
    Et = gfu.components[1] # (Ey)
    E_3D_full = CF((Ep[0], Et, Ep[1]))
    
    # Create the Grid
    nx, nz = resolution
    eps = 1e-6 # Strict guard to prevent evaluating exactly on the geometry borders
    x_coords = np.linspace(eps, Lx_tot - eps, nx)
    z_coords = np.linspace(eps, Lz_exact - eps, nz)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    # Vectorization using C++: Array are flatten and send to NGSolve as single batch 
    X_flat, Z_flat = X.flatten(), Z.flatten()
    
    try:
        # Ask NGSolve to map the entire array of coordinates at once
        mips = mesh(X_flat, Z_flat)
        # E_vals is returned as a numpy array of shape (N, 3) containing complex numbers
        E_vals = E_3D_full(mips)
    except TypeError:
        # Fallback for older NGSolve versions that do not support array mapping natively
        E_vals = np.array([E_3D_full(mesh(x, z)) if mesh(x, z) else (0j, 0j, 0j) 
                           for x, z in zip(X_flat, Z_flat)])

    # Reshape back to 2D matrices
    Ex = E_vals[:, 0].reshape(nx, nz)
    Ey = E_vals[:, 1].reshape(nx, nz)
    Ez = E_vals[:, 2].reshape(nx, nz)

    # Extract target data based on given configuration
    def extract_val(data_array, v_type):
        return data_array.real if v_type == 'real' else np.abs(data_array)

    if component == 'Ex':
        plot_data = extract_val(Ex, value_type)
    elif component == 'Ey':
        plot_data = extract_val(Ey, value_type)
    elif component == 'Ez':
        plot_data = extract_val(Ez, value_type)
    elif component == 'norm':
        # Total norm computation: sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)
        plot_data = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    else:
        raise ValueError("Invalid component. Choose 'Ex', 'Ey', 'Ez', or 'norm'.")

    # Initialize the Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use pcolormesh with shading='gouraud' for faster rendering than contourf
    cmap = 'magma' if value_type == 'abs' or component == 'norm' else 'coolwarm'
    vmax = np.max(plot_data)
    vmin = 0.0 if value_type == 'abs' or component == 'norm' else -vmax
    
    c = ax.pcolormesh(Z, X, plot_data, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(f"Wave Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=14)

    # PML Boundary Indicator
    ax.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=4, alpha=0.8, 
               label='Radial PML border', path_effects=[pe.withStroke(linewidth=6, foreground="black")])

    # Electric Field Vector Quiver (Jacquot 2013 Fig 3)
    if plot_e_vectors:
        # We define a stride so the arrows don't turn into a solid black block.
        # ~30 arrows per axis is usually visually optimal.
        step_x = max(1, nx // 30)
        step_z = max(1, nz // 30)
        
        # We plot the real part of the field to show the polarization at t=0
        ax.quiver(Z[::step_x, ::step_z], X[::step_x, ::step_z], 
                  Ez.real[::step_x, ::step_z], Ex.real[::step_x, ::step_z], 
                  color='cyan', alpha=0.7, pivot='mid', scale_units='xy')

    # Background B-Field Direction Indicator
    theta_rad = cfg.PLASMA['theta_B_rad']
    phi_rad = cfg.PLASMA['phi_B_rad']
    
    # Mathematical projection of 3D B-field onto the 2D (x, z) plane
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    # Normalize the 2D projected vector for plotting
    norm_b = np.sqrt(bx**2 + bz**2)
    if norm_b > 1e-6:
        bx, bz = bx / norm_b, bz / norm_b
        
        # Place arrow in the top right corner of the plasma domain
        arrow_z = Lz_exact * 0.85
        arrow_x = Lx_plasma * 0.85
        len_scale = Lz_exact * 0.08
        
        ax.quiver(arrow_z, arrow_x, bz * len_scale, bx * len_scale, 
                  color='lime', scale=1, scale_units='xy', width=0.005, pivot='tail', zorder=5,  
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.text(arrow_z + bz * len_scale, arrow_x + bx * len_scale, r'$\mathbf{B}_0$', 
                color='lime', fontsize=16, fontweight='bold', ha='left', va='bottom', 
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Formatting 
    ax.set_title(f"Lower Hybrid Coupling: {component} component", fontsize=16)
    ax.set_xlabel(r'Toroidal position $z$ [m]', fontsize=16)
    ax.set_ylabel(r'Radial position $x$ [m]', fontsize=16)
    ax.set_xlim(0, Lz_exact)
    ax.set_ylim(0, Lx_tot)
    ax.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, right=True, left=True)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=16)
    
    plt.tight_layout()
    
    vector_suffix = "_E_vect_field" if plot_e_vectors else ""
    filename = f"Map_{component}_{value_type}{vector_suffix}.pdf"
    plt.savefig(os.path.join(figure_save_dir, filename), dpi=300)
    plt.show()


    