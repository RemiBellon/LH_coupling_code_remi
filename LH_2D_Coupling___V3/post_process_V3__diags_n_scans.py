from solver_Hcurl_3D_V3 import * 
import config_2Dcoupling_V3 as cfg              # config = physical & simulation parameters 

import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors
import matplotlib.patheffects as pe

from scipy.fft import fft, fftfreq

from time import *
import os
import datetime
import json
from pathlib import Path


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



def benchmark_1D_radial_profile(mesh, gfu, cfg, figure_save_dir, component='Ez'):
    """
    Extracts 1D radial slice of the E-field and performs automated 
    analytical physics benchmarking (FFT for propagating, Log-fit for evanescent).
    Designed for CONSTANT DENSITY validation.
    """
    print(f"--- Running 1D Analytical Benchmark | Component: {component} ---")
    
    # 1. Geometry & Coordinate Setup
    Lx_plasma = cfg['DOMAIN']['Lx_plasma']
    Lz_exact = cfg['DOMAIN']['Lz_exact']
    z_mid = Lz_exact / 2.0
    
    # High resolution 1D array purely inside the plasma (ignore PML for the analytical fit)
    nx = 2000
    x_coords = np.linspace(1e-5, Lx_plasma - 1e-5, nx)
    dx = x_coords[1] - x_coords[0]
    
    # 2. High-Performance Vectorized Evaluation
    Ep = gfu.components[0] 
    Et = gfu.components[1] 
    E_3D_full = CF((Ep[0], Et, Ep[1]))
    
    # Map the 1D line directly into NGSolve C++ memory
    mips = mesh(x_coords, np.full_like(x_coords, z_mid))
    E_vals = E_3D_full(mips) # Shape: (nx, 3)
    
    if component == 'Ex': E_slice = E_vals[:, 0]
    elif component == 'Ey': E_slice = E_vals[:, 1]
    elif component == 'Ez': E_slice = E_vals[:, 2]
    
    E_real = E_slice.real
    E_abs = np.abs(E_slice)
    
    # 3. ANALYTICAL STIX COMPUTATION (The Theoretical Truth)
    omega = cfg['WAVE']['omega_wave']
    k0 = cfg['CONST']['c0'] / omega
    k0_vac = 2 * np.pi * k0
    n_para = cfg['WAVE']['n_para']
    
    n_e = cfg['PLASMA']['ne_constant'] # Must use constant density config for this test
    B0 = cfg['PLASMA']['B0_center_plasma']
    qe, me, mi, eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12
    
    w_pe2 = (n_e * qe**2) / (me * eps0)
    w_pi2 = (n_e * qe**2) / (mi * eps0)
    Om_ce = qe * B0 / me
    Om_ci = qe * B0 / mi
    
    S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
    P = 1 - w_pe2/omega**2 - w_pi2/omega**2
    D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
    
    # Booker Quartic for Slow Wave
    A_stix, B_stix = S, (S + P)*n_para**2 - (S**2 - D**2) - P*S
    C_stix = P * (n_para**2 - (S + D)) * (n_para**2 - (S - D))
    n_perp_sq = (-B_stix + np.sqrt(max(0, B_stix**2 - 4*A_stix*C_stix))) / (2*A_stix)
    
    # 4. PHYSICS REGIME DETECTION & PLOTTING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"1D Radial Benchmark ($z = {z_mid:.2f}$ m) | $n_\parallel = {n_para}$", fontsize=16)
    
    if n_perp_sq > 0:
        # ---------------------------------------------------------
        # REGIME A: PROPAGATING WAVE (FFT BENCHMARK)
        # ---------------------------------------------------------
        print("Regime: Propagating Wave. Executing FFT...")
        k_perp_theory = (omega / 3e8) * np.sqrt(n_perp_sq)
        lambda_theory = (2 * np.pi) / k_perp_theory
        
        # Subplot 1: Real Wave
        ax1.plot(x_coords, E_real, color='royalblue', lw=2)
        ax1.set_ylabel(f'Re({component}) [V/m]')
        ax1.set_title('Instantaneous Wavefront')
        ax1.grid(True, alpha=0.5)
        
        # Subplot 2: Spatial FFT
        k_axis = fftfreq(nx, d=dx) * 2 * np.pi # Spatial frequencies to Wavenumber (rad/m)
        fft_spectrum = np.abs(fft(E_real))
        
        # Filter positive half of the spectrum (ignore DC offset)
        pos_mask = (k_axis > 0)
        k_pos, spec_pos = k_axis[pos_mask], fft_spectrum[pos_mask]
        
        # Extract numerical peak
        k_perp_sim = k_pos[np.argmax(spec_pos)]
        lambda_sim = (2 * np.pi) / k_perp_sim
        error_pct = abs(lambda_sim - lambda_theory) / lambda_theory * 100
        
        ax2.plot(k_pos, spec_pos, color='crimson', lw=2, label='Simulation FFT')
        ax2.axvline(k_perp_theory, color='black', linestyle='--', lw=2, label=f'Theory $k_\perp$: {k_perp_theory:.2f} rad/m')
        ax2.set_xlim(0, k_perp_theory * 2)
        ax2.set_xlabel('Perpendicular Wavenumber $k_x$ [rad/m]')
        ax2.set_ylabel('Spectral Amplitude')
        ax2.set_title(f'FFT Spectrum | Error: {error_pct:.2f}% | $\lambda_{{sim}}$={lambda_sim*100:.2f}cm')
        ax2.legend()
        
    else:
        # ---------------------------------------------------------
        # REGIME B: EVANESCENT WAVE (LOG-FIT BENCHMARK)
        # ---------------------------------------------------------
        print("Regime: Evanescent Wave. Executing Log-Linear Regression...")
        alpha_theory = (omega / 3e8) * np.sqrt(-n_perp_sq)
        
        # Subplot 1: Envelope
        ax1.plot(x_coords, E_abs, color='darkorange', lw=2)
        ax1.set_ylabel(f'|{component}| Envelope [V/m]')
        ax1.set_title('Evanescent Decay')
        ax1.grid(True, alpha=0.5)
        
        # Subplot 2: Semi-Log Fit
        log_E_abs = np.log(np.maximum(E_abs, 1e-12)) # Prevent log(0)
        
        # Fit only the first 20% of the domain to avoid numerical noise floor
        fit_idx = int(nx * 0.2)
        slope, intercept = np.polyfit(x_coords[:fit_idx], log_E_abs[:fit_idx], 1)
        alpha_sim = -slope
        error_pct = abs(alpha_sim - alpha_theory) / alpha_theory * 100
        
        ax2.plot(x_coords, log_E_abs, color='purple', lw=2, label='Simulation $\ln|E|$')
        ax2.plot(x_coords[:fit_idx], slope * x_coords[:fit_idx] + intercept, color='lime', linestyle='--', lw=3, label=f'Linear Fit (Sim $\\alpha$: {alpha_sim:.2f})')
        ax2.set_xlabel('Radial Position $x$ [m]')
        ax2.set_ylabel('$\ln(|E|)$')
        ax2.set_title(f'Semi-Log Decay | Theory $\\alpha$: {alpha_theory:.2f} | Error: {error_pct:.2f}%')
        ax2.legend()

    plt.tight_layout()
    filename = f"Benchmark_1D_{component}_npara_{n_para}.png"
    plt.savefig(os.path.join(figure_save_dir, filename), dpi=300)
    plt.show()


def benchmark_mesh_convergence(solver, cfg, figure_save_dir):
    """
    Sweeps the mesh resolution (Points Per Wavelength) to generate
    the Pareto Frontier of Cost vs. Precision for the 2D LH Solver.
    Designed for CONSTANT DENSITY propagating regimes.
    """
    print("--- Initiating Algorithmic Scaling & Convergence Benchmark ---")
    
    # 1. Define the Sweep (Logarithmic spacing is standard for convergence studies)
    # Sweeping from 2.0 PPW to 10.0 PPW
    ppw_array = np.geomspace(2.0, 8.0, num=8)
    
    # Storage arrays
    dofs_list = []
    time_list = []
    l2_error_list = []
    
    # 2. Analytical Truth Setup (For L2 Error Computation)
    omega = cfg['WAVE']['omega_wave']
    k0 = cfg['CONST']['c0'] / omega
    n_para = cfg['WAVE']['n_para']
    n_e = cfg['PLASMA']['ne_constant']
    B0 = cfg['PLASMA']['B0_center_plasma']
    qe, me, mi, eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12
    
    w_pe2 = (n_e * qe**2) / (me * eps0)
    w_pi2 = (n_e * qe**2) / (mi * eps0)
    Om_ce = qe * B0 / me
    Om_ci = qe * B0 / mi
    
    S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
    P = 1 - w_pe2/omega**2 - w_pi2/omega**2
    D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
    
    B_stix = (S + P)*n_para**2 - (S**2 - D**2) - P*S
    C_stix = P * (n_para**2 - (S + D)) * (n_para**2 - (S - D))
    n_perp_sq = (-B_stix + np.sqrt(B_stix**2 - 4*S*C_stix)) / (2*S)
    k_perp = (omega / 3e8) * np.sqrt(n_perp_sq)
    
    E_inc_amp = cfg['WAVE']['E_inc']
    k_para = (omega / 3e8) * n_para
    
    # 3. THE RESOLUTION SWEEP
    for ppw in ppw_array:
        print(f"\nEvaluating Resolution: PPW = {ppw:.2f}")
        
        # Inject the new resolution into the config
        solver.cfg['DOMAIN']['n_resol_per_wlgth'] = ppw
        
        # Rebuild Mesh
        mesh = solver.build_mesh_with_PMLs()
        
        # Isolate the exact timing of the Weak Form Assembly and Matrix Inversion
        t_start = time.time()
        gfu, ndofs = solver.solve_helmholtz_2_5D_pml(mesh)
        t_end = time.time()
        
        cpu_time = t_end - t_start
        
        # 4. L2 Error Computation (Native NGSolve Integration)
        # In NGSolve 2D: 'x' is Radial, 'y' is Toroidal
        # The exact forward-propagating analytical field for the Ez component
        Ez_exact = E_inc_amp * exp(1j * k_perp * x + 1j * k_para * y)
        
        # Extract the Ez component from the simulation (HCurl space component 0, vector index 1)
        Ez_sim = gfu.components[0][1]
        
        # Integrate the Absolute Squared Error purely over the physical plasma region (ignore PML)
        error_expr = (Ez_sim - Ez_exact) * Conj(Ez_sim - Ez_exact)
        L2_error = sqrt(Integrate(error_expr, mesh, definedon=mesh.Materials("plasma_region"))).real
        
        print(f"--> DoFs: {ndofs} | CPU Time: {cpu_time:.3f}s | L2 Error: {L2_error:.4e}")
        
        dofs_list.append(ndofs)
        time_list.append(cpu_time)
        l2_error_list.append(L2_error)

    # Convert to Numpy Arrays for math
    DoFs = np.array(dofs_list)
    Times = np.array(time_list)
    Errors = np.array(l2_error_list)

    # 5. ASYMPTOTIC CONVERGENCE EXTRACTION
    # We use log-log linear regression to extract the slope (the order of convergence)
    slope_err, int_err = np.polyfit(np.log10(DoFs), np.log10(Errors), 1)
    slope_time, int_time = np.polyfit(np.log10(DoFs), np.log10(Times), 1)

    # 6. PROFESSIONAL PLOTTING
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graph A: L2 Error vs DoFs (Verifies Mathematical Correctness)
    axs[0].loglog(DoFs, Errors, 'bo-', markersize=8, label='Simulation Error')
    axs[0].loglog(DoFs, 10**(int_err) * DoFs**slope_err, 'k--', label=f'Trend ($O(N^{{{slope_err:.2f}}})$)')
    axs[0].set_title('Mesh Convergence (Precision)')
    axs[0].set_xlabel('Degrees of Freedom (N)')
    axs[0].set_ylabel('$L_2$ Error Norm')
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].legend()

    # Graph B: CPU Time vs DoFs (Verifies Algorithmic Scaling)
    axs[1].loglog(DoFs, Times, 'ro-', markersize=8, label='CPU Time')
    axs[1].loglog(DoFs, 10**(int_time) * DoFs**slope_time, 'k--', label=f'Trend ($O(N^{{{slope_time:.2f}}})$)')
    axs[1].set_title('Algorithmic Scaling (Cost)')
    axs[1].set_xlabel('Degrees of Freedom (N)')
    axs[1].set_ylabel('CPU Time [Seconds]')
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].legend()

    # Graph C: Pareto Frontier (Cost vs Precision)
    axs[2].loglog(Times, Errors, 'go-', markersize=8)
    # Annotate the optimal operating point (the "knee" of the curve)
    knee_idx = len(Times) // 2 
    axs[2].annotate(f'Optimal Target\n(PPW = {ppw_array[knee_idx]:.1f})', 
                    xy=(Times[knee_idx], Errors[knee_idx]), xytext=(Times[knee_idx]*1.5, Errors[knee_idx]*2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    axs[2].set_title('The Pareto Frontier')
    axs[2].set_xlabel('Computational Cost [Seconds]')
    axs[2].set_ylabel('$L_2$ Error Norm')
    axs[2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    filename = "Mesh_Convergence_Pareto_Frontier.png"
    plt.savefig(os.path.join(figure_save_dir, filename), dpi=300)
    plt.show()