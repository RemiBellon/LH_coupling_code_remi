from solver_Hcurl_3D_V3 import * 
import config_2Dcoupling_V3 as cfg              # config = physical & simulation parameters 

import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as colors

from time import *
import os
import datetime
import json
from pathlib import Path

figure_save_dir = Path("/home/remi/Perso/Stage/M2_IRFM/Codes/LH_2D_Coupling___V3/Figures")
# figure_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Figures")
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




# =====================================================================
# 3.1 PML DIAGNOSTIC --- Poynting flux
# =====================================================================
def pml_diag_poynting_flux(mesh, gfu, freq_LH):
    '''
    Compute radial Poynting vector flux + verification: power injected = power transmitted
    '''
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * freq_LH
    Px = (1.0 / (2.0 * omega * mu_0)) * (Conj(gfu)) * grad(gfu)[0]
    
    total_power = Integrate(Px, mesh.Materials("plasma_region")) # definedon="pml_region")
    print(f'Total transmitted Power Flux: {total_power:.4e} W')
    return total_power

# =====================================================================
# 3.2 PML DIAGNOSTIC --- SWR & Reflexion coeffs
# =====================================================================
def pml_diag_SWR_eta(mesh, gfu, kx, Lx_plasma, L_pml_r, Lz):
    """
    Evaluate SWR and reflection coeff (analytic and simulation)
    """
    print('--- Running PML diag ---')
   # Sample at the middle of the box in toroidal direction: 
    z_lmid_line = Lz / 2.0
    x_pml_coords = np.linspace(0, Lx_plasma, 500)

    E_abs =[]
    for x in x_pml_coords:
        pt = mesh(x, z_lmid_line)
        if pt:
           # To find standing wave we take the abs value of E field 
            E_abs.append(abs(gfu(pt)))
        
    E_abs_array = np.array(E_abs)

   # Compute SWR
    min_E, max_E = np.min(E_abs), np.max(E_abs)
    SWR = max_E / min_E
    eta_sim = (SWR - 1.0) / (SWR + 1.0)

   # Analytical predcition eta_pred
   # Because SetPML = HalfSpace==> Constant stretching ==> p = 0  
    Sr_Im = 1
    pr = 0
    eta_pred_fwd_wave = np.exp(-2. * abs(kx) * L_pml_r * (Sr_Im) / (1 + pr))
    eat_pred_evan_wave = (np.exp(-2. * abs(kx) * L_pml_r * (1 + Sr_Im) / (pr + 1))) / (np.exp(2 * abs(kx) * L_pml_r))

    print(f'SWR = {SWR:.6f}')
    print(f'eta_sim = {eta_sim:.6f}')
    print(f'eta_pred_fwd_wave = {eta_pred_fwd_wave:.6f}')
    print(f'eta_pred_evan_wave = {eat_pred_evan_wave:.6f}')

    plt.figure(figsize=(8,4))
    plt.plot(x_pml_coords, E_abs, color = 'Royalblue', label = r'$\|E\| Envelope$')
    plt.axhline(y=max_E, color = 'crimson', linestyle='--', alpha=0.5, label = r'$Max \|E\|$')
    plt.axhline(y=min_E, color = 'green', linestyle='--', alpha=0.5, label = r'$Min \|E\|$')
    plt.tick_params(direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)
    plt.xlabel(r'$x\ [m]$',fontsize=14)
    plt.ylabel(r'$\|E\|\ [V/m]$',fontsize=14)
    # plt.legend(loc = 'best', fontsize=14)
    plt.tight_layout()
    plt.savefig(figure_save_dir + "\Plane_Wave_E_field_envelope_vs_radiale_direction.png", dpi=300)
    plt.show()

# =====================================================================
# 3.3 MESH DIAG --- Mesh convergence 
# =====================================================================
def mesh_diag_L2_error(mesh, gfu, k):
    '''
    Compute the L2 norm of error between analytical and simulated plane wave.
    '''
    u_exact = exp(1j * k * x)
    error_expr = Norm(gfu - u_exact)**2
    L2_error = sqrt(Integrate(error_expr, mesh.Materials("plasma_region")))
    
    return L2_error

def mesh_diag_convergence_study(Lx_plasma, Lx_pml, Lz_approx, k_wave_vacuum, kz_exact, lambda_0, theta_deg):
    '''
    Precision vs computation time ==> optimal mesh resolution based on simulation parameters
    '''
    print('--- Run convergence study ---')
    resolutions = np.linspace(1e-3, 0.025, 200) 
    dofs_list, errors_list, times_list = [], [], []
    
    for res in resolutions:
        t0 = time.time()
        mesh = solver.build_mesh_with_PMLs()
        gfu, ndof = solver.solve_helmholtz_Hcurl_3D_pml(mesh)
        t_solve = time.time() - t0

        error = mesh_diag_L2_error(mesh, gfu, k_wave_vacuum)
        dofs_list.append(ndof)
        errors_list.append(error)
        times_list.append(t_solve)
        print(f'Res: {res:.3f}, Dofs:{ndof:5d}, L2 error:{error:.2e}, Time:{t_solve:.3f}s')

    # Convert to numpy arrays for vectorized math
    dofs_arr = np.array(dofs_list)
    errors_arr = np.array(errors_list)
    times_arr = np.array(times_list)
    
    # =====================================================================
    # APPLYING THE 3 SPECIFIC MASKS FOR LINEAR REGRESSIONS
    # =====================================================================
    # 1. Fit L2 Error vs DoFs (Condition: DoFs < 3e9)
    # Note: 3e9 is massive, this will likely include all points unless changed.
    mask_l2_dofs = dofs_arr < 9e3
    slope_l2_dofs, int_l2_dofs = np.polyfit(np.log10(dofs_arr[mask_l2_dofs]), np.log10(errors_arr[mask_l2_dofs]), 1)
    
    # 2. Fit CPU Time vs DoFs (Condition: DoFs > 1e4)
    mask_time_dofs = dofs_arr > 1e4
    slope_time_dofs, int_time_dofs = np.polyfit(np.log10(dofs_arr[mask_time_dofs]), np.log10(times_arr[mask_time_dofs]), 1)
    
    # 3. Fit L2 Error vs CPU Time (Condition: Time < 3e-2 s)
    mask_l2_time = times_arr < 1
    slope_l2_time, int_l2_time = np.polyfit(np.log10(times_arr[mask_l2_time]), np.log10(errors_arr[mask_l2_time]), 1)

    print("\n--- Fit Results ---")
    print(f"L2 Error vs DoFs Slope (p): {slope_l2_dofs:.3f}")
    print(f"CPU Time vs DoFs Slope (q): {slope_time_dofs:.3f}")
    print(f"L2 Error vs Time Slope:     {slope_l2_time:.3f}")

    # =====================================================================
    # GENERATING THE TREND LINES (Bounded to their masked regions)
    # =====================================================================
    # Create smooth X arrays strictly bounded between the min and max of the filtered data
    dofs_line_l2 = np.geomspace(min(dofs_arr[mask_l2_dofs]), max(dofs_arr[mask_l2_dofs]), 50)
    dofs_line_time = np.geomspace(min(dofs_arr[mask_time_dofs]), max(dofs_arr[mask_time_dofs]), 50)
    times_line_l2 = np.geomspace(min(times_arr[mask_l2_time]), max(times_arr[mask_l2_time]), 50)

    # Compute the Y arrays using the correctly unpacked slopes and intercepts
    L2_fit_dofs_y = 10**(slope_l2_dofs * np.log10(dofs_line_l2) + int_l2_dofs)
    Time_fit_dofs_y = 10**(slope_time_dofs * np.log10(dofs_line_time) + int_time_dofs)
    L2_fit_time_y = 10**(slope_l2_time * np.log10(times_line_l2) + int_l2_time)

    # =====================================================================
    # PLOTTING
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Subplot 1: DoFs vs Error and Time ---
    ax1.set_xlabel('Degrees of Freedom (DoFs)', fontsize=14)
    ax1.set_ylabel('L2 Error', fontsize=14)
    
    # Data points
    ax1.loglog(dofs_arr, errors_arr, marker='o', color='royalblue', linestyle='None', alpha=0.5, label="L2 Error (Data)")
    # Fit line
    ax1.loglog(dofs_line_l2, L2_fit_dofs_y, color='darkorange', linewidth=3, label=f"Error Fit (Slope: {slope_l2_dofs:.2f})")

    ax1.tick_params(axis='y', labelcolor="royalblue")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('CPU Time [s]', fontsize=14)
    
    # Data points
    ax1_twin.loglog(dofs_arr, times_arr, marker='s', color="crimson", linestyle="None", alpha=0.5, label="CPU time (Data)")
    # Fit line
    ax1_twin.loglog(dofs_line_time, Time_fit_dofs_y, color="k", linewidth=3, linestyle='--', label=f"Time Fit (Slope: {slope_time_dofs:.2f})")
    
    ax1_twin.tick_params(axis='y', labelcolor="crimson", direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    ax1.set_title('Algorithmic Scaling', fontsize=14)

    # --- Subplot 2: Pareto Frontier (Time vs Error) ---
    ax2.set_xlabel('CPU Time (s)', fontsize=12)
    ax2.set_ylabel('L2 Error', fontsize=12)
    
    # Data points
    ax2.loglog(times_arr, errors_arr, marker='D', color='royalblue', linestyle='None', alpha=0.6, markersize=6, label='Data Points')
    # Fit line
    ax2.loglog(times_line_l2, L2_fit_time_y, color='crimson', linewidth=3, label=f'Pareto Fit (Slope: {slope_l2_time:.2f})')
    ax2.tick_params(direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend()
    ax2.set_title('Pareto Frontier (Cost vs. Precision)', fontsize=14)

    fig.tight_layout()
    # Replace the path below with a local relative path or pure filename if possible
    plt.savefig("L2_Error_and_CPU_ti.svg", dpi=300)
    plt.show()
# =====================================================================
# 4. VISUALIZATION 
# =====================================================================
def plot_wave_3D(mesh, gfu):
    """
    Plots the REAL part of the field to show the physical oscillating wave at t=0.
    """
    Lx_tot = cfg.DOMAIN['Lx_tot'] 
    Lx_plasma = cfg.DOMAIN['Lx_plasma'] 
    Lz_exact = cfg.DOMAIN['Lz_exact'] 
    Ly_slice = cfg.DOMAIN['Ly_slice'] 

    print("Generating 2D wave E map...")
    nx, nz  = 300, 150
    eps = 1e-6
    x_coords, z_coords = np.linspace(0+eps, Lx_tot-eps, nx), np.linspace(0+eps, Lz_exact-eps, nz)
    X,Z = np.meshgrid(x_coords, z_coords, indexing='ij') 
    y_mid = Ly_slice /2.0
    Ez_vals = np.zeros((nx, nz))

    for i in range(nx):
        for j in range(nz):
            try:
                pt = mesh(X[i,j], y_mid, Z[i, j])
                if pt:
                    Ez_vals[i, j] = gfu(pt)[2].real
            except Exception as e:
                Ez_vals[i, j] = np.nan


    plt.figure(figsize=(10, 4))
    plt.contourf(Z, X, Ez_vals, levels=50, cmap='inferno', alpha=1)
    plt.colorbar(label='Physical Wave Field $Re(E_z)$')
    
    # Draw a line to show where the PML starts
    plt.axhline(y=Lx_plasma, color='black', linestyle='--', label='PML Entrance')
    
    # plt.title('Plane Wave Propagating')
    plt.xlabel(r'$z\ [m]$', fontsize=14)
    plt.ylabel(r'$x\ [m]$', fontsize=14)
    plt.legend(loc='upper left')
    plt.tick_params(direction='out', length=6, bottom=True, top=True, right=True, left=True)
    plt.tight_layout()
    plt.savefig(figure_save_dir / "2D_E_field_Re_Ez.png", dpi=300)
    plt.show()



def compute_density_and_cutoff(solver, resolution_x, resolution_z):
    """
    Function to generate the density 2D map and compute the cutoff layer position
    Return: (X,Z)=Meshgrid resolution, Density_map=2Darray of density value, 
            (n_cutoff,x_cutoff)=cutoff density value and cutoff layer position
    """
    print("\n--- Computing Density Map & Cutoff ---")
    
    # TYPE: float
    Lx_tot = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_tot']
    Lz_tot = solver.cfg['DOMAIN']['Lz_tot']
    print('Lx_tot= ', Lx_tot, 'Lx_plasma= ', Lx_plasma, 'Lz_tot= ', Lz_tot)
    
    # TYPE x_vals, z_vals: np.ndarray, SHAPE: (resolution,)
    x_vals = np.linspace(1e-6, Lx_tot - 1e-6, resolution_x)
    z_vals = np.linspace(1e-6, Lz_tot - 1e-6, resolution_z)
    
    # TYPE X, Z: np.ndarray, SHAPE: (resolution, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    print('len(X): ', len(X))
    print('len(Z): ', len(Z))


    # TYPE prof_type: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'piecewise_linear_density')
    print(f"Selected density profile: {prof_type}")
    Density_map = create_density_profile(X, Z, solver)
    # Mask on PML region (no-physical for density)
    Density_map = np.where(X <= Lx_plasma, Density_map, np.nan)

    # Radial critical density (Cutoff) computation
    # TYPE physical CONST: float
    eps_0 = solver.cfg['CONST']['eps_0']
    m_e = solver.cfg['CONST']['m_e']
    q_e = solver.cfg['CONST']['q_e']
    omega = solver.cfg['WAVE']['omega_wave']
    
    # TYPE n_cutoff: float
    n_cutoff = (eps_0 * m_e * omega**2) / (q_e**2)
    print(f"Computed Cutoff Density (n_c): {n_cutoff:.4e} m^-3")
    
    z_mid_idx = resolution_z // 2
    # TYPE x_slice, n_slice: np.ndarray, SHAPE: (resolution,)
    x_slice = X[:, z_mid_idx]
    n_slice = Density_map[:, z_mid_idx]
    
    # Find spatial coordinate of cutoff dynamically
    # TYPE idx_cutoff: int
    idx_cutoff = np.argmax(n_slice >= n_cutoff)
    
    # TYPE x_cutoff: float
    x_cutoff = x_slice[idx_cutoff] if n_slice[idx_cutoff] >= n_cutoff else np.nan
    
    if not np.isnan(x_cutoff):
        print(f"Cutoff layer dynamically located at x = {x_cutoff:.4f} m")
    else:
        print("Cutoff layer is not reached within the current plasma domain.")
        
    return X, Z, prof_type, Density_map, n_cutoff, x_cutoff




# ======================================================================================================
#   General Function to Compute Cutoff Density Layer & Position  
# ======================================================================================================
def Plot_E_field_2D_Map(solver, save_dir, resolution_x, resolution_z, component):
    """
    2D map of E field components or total norm in (x,z) plane (Top View)
    Toroidal (z) is horizontal, Radial (x) is vertical.
    """
    if solver.E_field is None:
        print("Error: E field has not been calculated yet.")
        return

    print(f"--- 2D map generation of ({component}) ---")
    
    Lx_tot = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_tot']
    Lz_plasma = solver.cfg['DOMAIN']['Lz_tot']
    print('Lz_plasma = ', Lz_plasma, ' m')
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    # Initialize the plot mesh
    x_vals = np.linspace(1e-6, Lx_tot - 1e-6, resolution_x)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution_z)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    print('len(x_vals) = ', len(x_vals), 'len(z_vals) = ', len(z_vals))
    print('len(X) = ', len(X), 'len(Z) = ', len(Z))
    print('resolution_x = ', resolution_x, 'resolution_z = ', resolution_z)
    E_vec_cf = CoefficientFunction(solver.E_field.components[0])
    Ey_cf = CoefficientFunction(solver.E_field.components[1])
    # Matrix to store amplitude values
    Field_abs = np.zeros((resolution_x, resolution_z))
    # print('Starting loops')
    for i in range(resolution_x):
        for j in range(resolution_z):
            mip = solver.mesh(X[i,j], Z[i,j]) 
            if mip:
                # Évaluation unique du champ 3D complet
                val = solver.E_tot_cf(mip) 
                
                Ex, Ey, Ez = val[0], val[1], val[2]
                
                if component == 'Ex':
                    Field_abs[i, j] = np.abs(Ex)
                elif component == 'Ey':
                    Field_abs[i, j] = np.abs(Ey)
                elif component == 'Ez':
                    Field_abs[i, j] = np.abs(Ez)
                elif component == 'norm':
                    Field_abs[i, j] = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
            else:
                Field_abs[i, j] = 0.0
                raise ValueError("Error, wrong component: Should be in : ['Ex', 'Ey', 'Ez', 'norm']")
    
    max_amp = np.max(Field_abs)
    print(f"Max {component} amplitude detected in grid: {max_amp:.4e} V/m")

    plt.figure(figsize=(12, 6))
    
    # 1. Plot 
    norm_scale = colors.LogNorm(vmin=max_amp*5e-2, vmax=max_amp)
    cmap_plot = plt.pcolormesh(Z, X, Field_abs, shading='gouraud', cmap='magma', norm=norm_scale)
    plt.colorbar(cmap_plot, label=f'Amplitude |{component}| [V/m]')
    
    # Horizontal line: radial PML
    plt.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma + Lx*0.015, 'Radial PML', color='white', ha='center', va='bottom')
    
    print('x_cutoff computation')

    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution_x, resolution_z)

    plt.axhline(y = x_cutoff, linestyle= '--', linewidth='2', color = 'crimson', label=r'$Radial\ critical\density\ position$')

    print("Add B vect to the graph ")
    # 2. Add B direction to graph
    theta_rad = solver.cfg['PLASMA']['theta_B_rad']
    phi_rad = solver.cfg['PLASMA']['phi_B_rad']
    
    # B vector components projection on (x,z)
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    arrow_len = Lx * 0.15 
    dz = bz * arrow_len
    dx = bx * arrow_len
    
    # arrow position
    z0 = Lz * 0.05
    x0 = Lx * 0.85
    
    plt.arrow(z0, x0, dz, dx, color='cyan', width=Lx*0.003, head_width=Lx*0.015, zorder=5)
    plt.text(z0 + dz*1.5, x0 + dx*1.5, r'$\vec{B}_{0}$', color='cyan', fontsize=14, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Toroidal position z [m]', fontsize=14)
    plt.ylabel('Radial position x [m]', fontsize=14)
    plt.title(f'2D map E field - {component} (Top View)', fontsize=16)
    
    # plt.axis('scaled') 
    plt.xlim(0, Lz)
    plt.ylim(0, Lx)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"Map_2D_{component}.pdf")
    print('file_path = ', file_path)
    plt.savefig(file_path, dpi=300)
    plt.show()
    print('--- Plot E field Function End---')



def Plot_Radial_Components_at_z_Target(solver, z_target, save_dir):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) pour une hauteur z donnée.
    """
    if solver.E_field is None: 
        return

    print(f"--- Extract radial profile (Ex, Ey, Ez) at z = {z_target} m ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot'] 
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma'] 
    
    x_points = np.linspace(1e-6, Lx - 1e-6, 1000)
    Ex_abs = np.zeros_like(x_points)
    Ey_abs = np.zeros_like(x_points)
    Ez_abs = np.zeros_like(x_points)
    
    for i, x_val in enumerate(x_points):
        mip = solver.mesh(x_val, z_target) 
        if mip:
            val = solver.E_tot_cf(mip)
            # On assigne bien la valeur à l'index [i] du tableau !
            Ex_abs[i] = np.abs(val[0])
            Ey_abs[i] = np.abs(val[1])
            Ez_abs[i] = np.abs(val[2])
        else:
            Ex_abs[i], Ey_abs[i], Ez_abs[i] = 0.0, 0.0, 0.0

    # Création du Subplot avec x partagé
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    components = [Ex_abs, Ey_abs, Ez_abs]
    print('shape(Ex): ', np.shape(components[0]))
    print('Ex: ', components[0])
    
    labels = [r'$|E_x|$ (Radial)', r'$|E_y|$ (Poloïdal)', r'$|E_z|$ (Toroïdal)']
    colors = ['green', 'darkorange', 'blue']

    for i, ax in enumerate(axs):
        ax.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
        ax.plot(x_points, components[i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_ylabel('Amplitude [V/m]', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Radial position x [m]', fontsize=14)
    fig.suptitle(f'Profil Radial des 3 composantes du champ à z={z_target}m', fontsize=16)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"radial_components.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()
    print('--- Plot Radial Component at z Target Function End---')


def Plot_Radial_Components_Averaged(solver, save_dir, z_min=0.05, z_max=0.35, z_res=50):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) en moyennant sur l'axe toroïdal (z).
    z_min et z_max définissent la fenêtre de moyennage (pour éviter les PMLs toroïdales).
    """
    if solver.E_field is None: 
        return

    print(f"--- Extraction and mean on z of radiale profile components ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot'] 
    Lx_plasma = solver.cfg['DOMAIN']['Lx_tot'] 
    
    # Résolution de l'extraction
    x_points = np.linspace(1e-6, Lx - 1e-6, 500)
    z_points = np.linspace(z_min, z_max, z_res)
    
    Ex_avg = np.zeros(len(x_points))
    Ey_avg = np.zeros(len(x_points))
    Ez_avg = np.zeros(len(x_points))
    
    for i, x_val in enumerate(x_points):
        Ex_sum, Ey_sum, Ez_sum = 0.0, 0.0, 0.0
        valid_pts = 0
        
        for z_val in z_points:
            mip = solver.mesh(x_val, z_val) 
            if mip:
                val = solver.E_tot_cf(mip) 
                
                Ex_sum += np.abs(val[0])
                Ey_sum += np.abs(val[1])
                Ez_sum += np.abs(val[2])
                valid_pts += 1
                
        # On calcule la moyenne
        if valid_pts > 0:
            Ex_avg[i] = Ex_sum / valid_pts
            Ey_avg[i] = Ey_sum / valid_pts
            Ez_avg[i] = Ez_sum / valid_pts

    # Création du Subplot
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    components = [Ex_avg, Ey_avg, Ez_avg]
    print('--- Plot Radial Component Av.: components = ', components)
    print('sum Ey = ', np.sum(Ey_avg))
    labels = [r'Averaged $|E_x|$', r'Averaged $|E_y|$', r'Averaged $|E_z|$']
    colors = ['green', 'darkorange', 'blue']

    for i, ax in enumerate(axs):
        ax.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
        ax.plot(x_points, components[i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_ylabel('Amplitude [V/m]', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlim(0, max(x_points))
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Radial position x [m]', fontsize=14)
    fig.suptitle(f'Radial profiles averaged on z $\in$ [{z_min}, {z_max}] m', fontsize=16)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"radial_components_averaged.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()
    print('--- Plot Radial Component Averaged Function End---')

def Plot_Density_Profile_2D(solver, save_dir, resolution_x, resolution_z) -> None:
    """
    2D map of density and 1D slice with Slow-Wave Cutoff layer calculation.
    """
    print("\n--- [PLOT] 2D & 1D map generation of Density n_e ---")
    # TYPE DOMAIN size: float
    Lx_tot = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    print('Compute Density_map and cutoff layer position.')
    # X,Z = meshgrid resolution; Density_map = 2d map of density values; 
    # n_cutoff,x_cutoff = cutoff density value and layer position relative to given profile density 
    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution_x, resolution_z)

    # --- 1D Slice Extraction ---
    x_mid_idx = resolution_x // 2
    x_slice = X[:, x_mid_idx]
    n_slice = Density_map[:, x_mid_idx]
    # print('n_slice = ', n_slice)
    # Plotting
    # TYPE fig: matplotlib.figure.Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D Map
    cmap_plot = ax1.pcolormesh(Z, X, Density_map, shading='nearest', cmap='viridis')
    fig.colorbar(cmap_plot, ax=ax1, label=r'Density $n_e$ [m$^{-3}$]')
    ax1.axhline(y=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Display cutoff layer on 2D map if it exists
    if not np.isnan(x_cutoff):
        ax1.axhline(y=x_cutoff, color='red', linestyle='-', linewidth=1.5, label='Cutoff Layer')
        ax1.legend(loc='lower right')
        
    ax1.set_xlabel('Toroidal position z [m]', fontsize=12)
    ax1.set_ylabel('Radial position x [m]', fontsize=12)
    ax1.set_title('2D map Plasma Density', fontsize=14)
    ax1.set_xlim(0, Lz)
    ax1.set_ylim(0, Lx_plasma)
    
    # 1D Slice
    ax2.plot(x_slice, n_slice, color='indigo', linewidth=2.5, linestyle='', marker='+', ms=1)
    ax2.axvline(x=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.5, label='PML Boundary')
    
    # Display Cutoff on 1D graph
    if not np.isnan(x_cutoff):
        ax2.axhline(y=n_cutoff, color='red', linestyle=':', linewidth=2, label=f'n_c = {n_cutoff:.2e}')
        ax2.text(n_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'n_c = {x_cutoff*100:.2f} cm', color='red')
        ax2.axvline(x=x_cutoff, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.text(x_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'x_c = {x_cutoff*100:.2f} cm', color='red')
    print('prof_type = ', prof_type)
    if prof_type == 'constant_density' or prof_type == 'piecewise_linear_density':
        ax2.set_yscale('linear')
        ax2.set_title('1D Density Profile (linear Scale)', fontsize=14)
    else : 
        ax2.set_yscale('log')
        ax2.set_title('1D Density Profile (Log Scale)', fontsize=14)
    
    ax2.set_xlabel('Radial position x [m]', fontsize=12)
    ax2.set_ylabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    ax2.set_xlim(0, Lx_tot)
    
    plt.tight_layout()
    import os
    file_path = os.path.join(save_dir, "Map_Density_Combined.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()
    print('--- Plot Desnity Profile 2D Function End---')

def Plot_B_Field_2D(solver, save_dir, resolution=100):
    """
    Génère une carte 2D de l'intensité du champ magnétique B0
    superposée à un champ de vecteurs (quiver) indiquant sa direction.
    """
    print("--- 2D map generation of (Magnetic Field B_tot + Vectors) ---")
    
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    x_vals = np.linspace(1e-6, Lx - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    B0_center = solver.cfg['PLASMA']['B0_center_plasma']
    R0 = solver.cfg['GEOM']['R0']
    R_ant = solver.cfg['GEOM']['R_ant']
    
    # 1. Calcul de la Magnitude (Masquage de la PML)
    B_map = B0_center * (R0 / (R_ant - X))
    B_map = np.where(X <= Lx_plasma, B_map, np.nan)

    # 2. Calcul des composantes pour le champ de vecteurs
    theta_rad = solver.cfg['PLASMA']['theta_B_rad']
    phi_rad = solver.cfg['PLASMA']['phi_B_rad']
    
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    Bx = B_map * bx
    Bz = B_map * bz

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Colormap (Fond)
    cmap_plot = plt.pcolormesh(Z, X, B_map, shading='nearest', cmap='cividis')
    plt.colorbar(cmap_plot, label=r'Intensité Champ Magnétique $|B_{tot}|$ [Tesla]')
    
    # Champ de Vecteurs (Quiver)
    # On saute des points (step) pour ne pas surcharger visuellement la carte avec trop de flèches
    step = resolution // 15
    plt.quiver(Z[::step, ::step], X[::step, ::step], 
               Bz[::step, ::step], Bx[::step, ::step], 
               color='white', alpha=0.9, pivot='mid')
    
    plt.axhline(y=Lx_plasma, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma - Lx*0.02, 'Limite PML (Non-physique)', color='red', ha='center', va='top')
    
    plt.xlabel('Toroidal position z [m]', fontsize=12)
    plt.ylabel('Radial position x [m]', fontsize=12)
    plt.title('2D map Magnetic Field with Vector Directions', fontsize=14)
    
    plt.xlim(0, Lz)
    plt.ylim(0, Lx_plasma)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"Map_2D_B_Field_Vectors.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()  
    print('--- Plot B field Function End---')