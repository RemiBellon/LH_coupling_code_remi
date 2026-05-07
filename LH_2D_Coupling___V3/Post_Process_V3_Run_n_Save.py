import os
import time
import datetime
import json
import numpy as np
import h5py
from ngsolve import *
from pathlib import Path

# ====================================================================
# UTILITIES
# ====================================================================
def setup_output_directory(base_folder="Results", save_data=True):
    """Creates a unique timestamped directory to prevent data overwriting."""
    if not save_data:
        print("<<<< TEST MODE: Don't save data >>>> ")
        return None
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_folder, f"Run_{now}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[SYSTEM] Output directory created: {run_dir}")
    return run_dir

def create_density_profile(x_val, z_vals, solver):
    """Generates the density profile (Constant, Exp, or Linear)."""
    is_ngsolve = type(x_val).__name__ == 'CoefficientFunction'
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'constant_density')

    if prof_type == 'constant_density':
        ne_constant = solver.cfg['PLASMA']['ne_constant']
        return ne_constant if is_ngsolve else np.full_like(x_val, ne_constant)
        
    elif prof_type == 'exponential_density':
        Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
        n_edge, n_core = solver.cfg['PLASMA']['lin_prof_n'][0], solver.cfg['PLASMA']['lin_prof_n'][-1]
        lambda_val = np.log(n_core) / Lx_plasma
        return exp(lambda_val * x_val) if is_ngsolve else np.exp(lambda_val * x_val)
        
    elif prof_type == 'piecewise_linear_density':
        x_pts, ne_pts = np.array(solver.cfg['PLASMA']['lin_prof_x']), np.array(solver.cfg['PLASMA']['lin_prof_n'])
        smooth_width = solver.cfg['PLASMA'].get('smooth_width', 0.006)
        slope_0 = (ne_pts[1] - ne_pts[0]) / (x_pts[1] - x_pts[0])
        profile = ne_pts[0] + slope_0 * (x_val - x_pts[0])

        for i in range(1, len(x_pts) - 1):
            x_c, n_c = float(x_pts[i]), float(ne_pts[i])
            s_prev = float((ne_pts[i] - ne_pts[i-1]) / (x_pts[i] - x_pts[i-1]))
            s_next = float((ne_pts[i+1] - ne_pts[i]) / (x_pts[i+1] - x_pts[i]))
            
            L_prev, L_next = n_c + s_prev * (x_val - x_c), n_c + s_next * (x_val - x_c)
            dx = x_val - x_c
            
            if is_ngsolve:
                H_smooth = 0.5 * (1.0 + dx / sqrt(dx**2 + smooth_width**2))
            else:
                H_smooth = 0.5 * (1.0 + dx / np.sqrt(dx**2 + smooth_width**2))
            
            profile = profile + (L_next - L_prev) * H_smooth
        return profile
    else:
        raise ValueError(f"Unknown profile type '{prof_type}'.")

# ====================================================================
# EXTRACTION MODULES
# ====================================================================
def run_2D_wave_map(mesh, gfu, cfg, save_dir, resolution=(300, 300)):
    """Extracts the full 2D complex wave field and saves to HDF5."""
    print("--- Extracting 2D Wave Map ---")
    Lx_tot, Lx_plasma, Lz_exact = cfg.DOMAIN['Lx_tot'], cfg.DOMAIN['Lx_plasma'], cfg.DOMAIN['Lz_exact']
    nx, nz = resolution
    eps = 1e-6 
    
    x_coords = np.linspace(eps, Lx_tot - eps, nx)
    z_coords = np.linspace(eps, Lz_exact - eps, nz)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))
    mips = mesh(X.flatten(), Z.flatten())
    E_vals = E_3D_full(mips)
    
    Ex = E_vals[:, 0].reshape(nx, nz)
    Ey = E_vals[:, 1].reshape(nx, nz)
    Ez = E_vals[:, 2].reshape(nx, nz)
    
    if save_dir is not None:
        h5_path = os.path.join(save_dir, "Wave_Map_2D.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('X', data=X, compression="gzip")
            h5f.create_dataset('Z', data=Z, compression="gzip")
            h5f.create_dataset('Ex', data=Ex, compression="gzip")
            h5f.create_dataset('Ey', data=Ey, compression="gzip")
            h5f.create_dataset('Ez', data=Ez, compression="gzip")
            h5f.attrs['Lx_plasma'] = Lx_plasma
            h5f.attrs['Lz_exact'] = Lz_exact
            h5f.attrs['theta_B_rad'] = cfg.PLASMA['theta_B_rad']
            h5f.attrs['phi_B_rad'] = cfg.PLASMA['phi_B_rad']
        print(f"--- 2D Map saved to {h5_path} ---")
        return h5_path
    else:
        print("TEST MODE: No h5 file data saving")

def run_1D_radial_profile(mesh, gfu, cfg, save_dir):
    """Extracts a high-res 1D radial slice and saves to HDF5."""
    print("--- Extracting 1D Radial Profile ---")
    Lx_plasma = cfg.DOMAIN['Lx_plasma']
    z_mid = cfg.DOMAIN['Lz_exact'] / 2.0
    
    nx = 2000
    x_coords = np.linspace(1e-5, Lx_plasma - 1e-5, nx)
    
    E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))
    mips = mesh(x_coords, np.full_like(x_coords, z_mid))
    E_vals = E_3D_full(mips) 
    
    if save_dir is not None:
        h5_path = os.path.join(save_dir, "Radial_Profile_1D.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('x_coords', data=x_coords, compression="gzip")
            h5f.create_dataset('Ex', data=E_vals[:, 0], compression="gzip")
            h5f.create_dataset('Ey', data=E_vals[:, 1], compression="gzip")
            h5f.create_dataset('Ez', data=E_vals[:, 2], compression="gzip")
        
            # Save physics params required for Plotting Stix Benchmarks
            h5f.attrs['omega_wave'] = cfg.WAVE['omega_wave']
            h5f.attrs['n_para'] = cfg.WAVE['n_para']
            h5f.attrs['ne_constant'] = cfg.PLASMA['ne_constant']
            h5f.attrs['B0_center'] = cfg.PLASMA['B0_center_plasma']
            h5f.attrs['z_mid'] = z_mid
        print(f"--- 1D Profile saved to {h5_path} ---")
        return h5_path
    else:
        print('TEST MODE: no h5 file data saving')


def run_mesh_convergence(solver, cfg, save_dir):
    """Runs the PPW sweep and saves DoFs, Times, and Errors to HDF5."""
    print("--- Running Algorithmic Scaling Sweep ---")
    ppw_array = np.geomspace(1.0, 20.0, num=round((12-1)*5))
    
    # Pre-calculate analytical field for Error integration
    omega, n_para = cfg.WAVE['omega_wave'], cfg.WAVE['n_para']
    n_e, B0 = cfg.PLASMA['ne_constant'], cfg.PLASMA['B0_center_plasma']
    qe, me, mi, eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12
    w_pe2, w_pi2 = (n_e * qe**2)/(me * eps0), (n_e * qe**2)/(mi * eps0)
    Om_ce, Om_ci = (qe * B0)/me, (qe * B0)/mi
    
    S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
    P = 1 - w_pe2/omega**2 - w_pi2/omega**2
    D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
    
    B_stix = (S + P)*n_para**2 - (S**2 - D**2) - P*S
    C_stix = P * (n_para**2 - (S + D)) * (n_para**2 - (S - D))
    n_perp = np.sqrt((-B_stix + np.sqrt(B_stix**2 - 4*S*C_stix)) / (2*S))
    k_perp = (omega / 3e8) * n_perp
    k_para = (omega / 3e8) * n_para
    E_inc_amp = cfg.WAVE['E_inc']

    dofs_list, time_list, l2_error_list = [], [], []

    for ppw in ppw_array:
        print(f"Evaluating PPW = {ppw:.2f}")
        cfg.DOMAIN['n_resol_per_wlgth'] = ppw
        
        mesh_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Meshes")
        mesh = solver.build_mesh_with_PMLs(mesh_save_dir)
        
        t0 = time.time()
        gfu, ndofs = solver.solve_helmholtz_Hcurl_2D_pml(mesh, cfg)
        cpu_time = time.time() - t0
        
        Ez_exact = E_inc_amp * exp(1j * k_perp * x + 1j * k_para * y)
        Ez_sim = gfu.components[0][1]
        error_expr = (Ez_sim - Ez_exact) * Conj(Ez_sim - Ez_exact)
        exact_expr = Ez_exact*Conj(Ez_exact)

        L2_error_abs = Integrate(error_expr, mesh, definedon=mesh.Materials("plasma_region")).real
        L2_exact = Integrate(exact_expr, mesh, definedon=mesh.Materials("plasma_region")).real
        
        L2_rel_error = np.sqrt(L2_error_abs / max(L2_exact, 1e-12))
        
        dofs_list.append(ndofs)
        time_list.append(cpu_time)
        l2_error_list.append(L2_rel_error)

    if save_dir is not None:
        h5_path = os.path.join(save_dir, "Mesh_Convergence.h5")
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('ppw_array', data=ppw_array)
            h5f.create_dataset('DoFs', data=np.array(dofs_list))
            h5f.create_dataset('Times', data=np.array(time_list))
            h5f.create_dataset('Errors', data=np.array(l2_error_list))
        print(f"--- Convergence data saved to {h5_path} ---")
        return h5_path
    else:
        print('<<<< TEST MODE: No h5 file data saving >>>>')





def run_pml_optimization(solver, cfg, save_dir=None):
    """
    EXECUTION ONLY: Runs Bivariate PML Scan and extracts both E-Field and Poynting SWR.
    """
    if save_dir is None:
        print("\n[SYSTEM] TEST MODE: Data will NOT be saved to disk.")

    print("--- Initiating Rigorous PML Optimization Scan (Compute Phase) ---")
    
    cfg.PLASMA['profile_type'] = 'constant_density'
    cfg.PLASMA['ne_constant'] = 3e18
    cfg.WAVE['n_para'] = 2.0 
    cfg.PML['p_degree'] = 2.0 
    cfg.PML['S_real'] = 1.0
    
    omega = cfg.WAVE['omega_wave']
    k_para = cfg.WAVE['n_para'] * omega / 3e8
    n_e, B0 = cfg.PLASMA['ne_constant'], cfg.PLASMA['B0_center_plasma']
    qe, me, mi, eps0, mu0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12, 4*np.pi*1e-7
    
    w_pe2, w_pi2 = (n_e * qe**2)/(me * eps0), (n_e * qe**2)/(mi * eps0)
    Om_ce, Om_ci = (qe * B0)/me, (qe * B0)/mi
    S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
    P = 1 - w_pe2/omega**2 - w_pi2/omega**2
    D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
    
    B_stix = (S + P)*cfg.WAVE['n_para']**2 - (S**2 - D**2) - P*S
    C_stix = P * (cfg.WAVE['n_para']**2 - (S + D)) * (cfg.WAVE['n_para']**2 - (S - D))
    n_perp = np.sqrt((-B_stix + np.sqrt(max(0, B_stix**2 - 4*S*C_stix))) / (2*S))
    lambda_perp = (2 * np.pi * 3e8) / (omega * n_perp)
    
    l_pml_ratios = np.linspace(0.5, 20.0, 100) 
    s_imag_array = np.geomspace(0.5, 50.0, 100) 
    
    gamma_E_matrix = np.zeros((len(s_imag_array), len(l_pml_ratios)))
    gamma_S_matrix = np.zeros((len(s_imag_array), len(l_pml_ratios)))
    
    Lx_plasma = cfg.DOMAIN['Lx_plasma']
    Lz_exact = cfg.DOMAIN['Lz_exact']
    z_mid = Lz_exact / 2.0
    x_eval = np.linspace(Lx_plasma * 0.25, Lx_plasma * 0.75, 1000)

    total_runs = len(s_imag_array) * len(l_pml_ratios)
    run_count = 0
    
    for i, s_imag in enumerate(s_imag_array):
        for j, l_ratio in enumerate(l_pml_ratios):
            run_count += 1
            print(f"Run {run_count}/{total_runs} | S_imag={s_imag:.2f}, L_PML={l_ratio:.2f}λ")
            
            cfg.PML['S_imag'] = s_imag
            cfg.DOMAIN['Lx_pml'] = l_ratio * lambda_perp
            cfg.DOMAIN['Lx_tot'] = cfg.DOMAIN['Lx_plasma'] + cfg.DOMAIN['Lx_pml']
            
            mesh_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Meshes")
            mesh = solver.build_mesh_with_PMLs(mesh_save_dir)
            gfu, _ = solver.solve_helmholtz_Hcurl_2D_pml(mesh, cfg)
            

            # --- 1. E-Field Evaluation ---
            E_xz = gfu.components[0] # The HCurl vector (Ex, Ez)
            Ey = gfu.components[1]   # The H1 scalar (Ey)
            
            Ex, Ez = E_xz[0], E_xz[1]
            E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))
            
            # --- 2. Rigorous 2.5D Poynting Vector Evaluation ---
            
            # A. Safe Gradient of the H1 component
            # Grad(Ey) returns a 2D vector: (dEy/dx, dEy/dz)
            grad_Ey = Grad(Ey)
            dEy_dx = grad_Ey[0]
            dEy_dz = grad_Ey[1]

            # B. Safe Curl of the HCurl component
            # In 2D, curl(E_xz) rigorously evaluates (dEz/dx - dEx/dz) without illegal gradients
            curl_Exz = curl(E_xz) 

            # C. Faraday's Law (H = 1/(i*w*mu) * curl E)
            Hx = (1j * k_para * Ez - dEy_dz) / (1j * omega * mu0)
            Hy = (-curl_Exz) / (1j * omega * mu0)
            Hz = (dEy_dx - 1j * k_para * Ex) / (1j * omega * mu0)

            # D. Poynting Vector Norm: S = 0.5 * Re(E x H*)
            # For purely radial reflection, we look at the radial flux Sx
            Sx = 0.5 * (Ey * Conj(Hz) - Ez * Conj(Hy)).real
            
            # To be absolutely rigorous for total energy, we look at the total in-plane magnitude
            Sz = 0.5 * (Ex * Conj(Hy) - Ey * Conj(Hx)).real
            S_tot_norm = sqrt(Sx**2 + Sz**2)


            # --- 3. Compute Metrics on the 1D extraction line ---
            mips = mesh(x_eval, np.full_like(x_eval, z_mid))
            
            E_vals = E_tot_norm(mips)
            S_vals = S_tot_norm(mips)
            
            # SWR for E-Field
            SWR_E = max(np.max(E_vals) / max(np.min(E_vals), 1e-12), 1.0001)
            gamma_E_matrix[i, j] = (SWR_E - 1.0) / (SWR_E + 1.0)
            
            # SWR for Poynting Vector
            SWR_S = max(np.max(S_vals) / max(np.min(S_vals), 1e-12), 1.0001)
            gamma_S_matrix[i, j] = (SWR_S - 1.0) / (SWR_S + 1.0)

    if save_dir is not None:
        h5_filepath = os.path.join(save_dir, "PML_Optimization_Data.h5")
        with h5py.File(h5_filepath, 'w') as h5f:
            h5f.create_dataset('gamma_E_matrix', data=gamma_E_matrix, compression="gzip")
            h5f.create_dataset('gamma_S_matrix', data=gamma_S_matrix, compression="gzip")
            h5f.create_dataset('s_imag_array', data=s_imag_array)
            h5f.create_dataset('l_pml_ratios', data=l_pml_ratios)
            metadata = {'lambda_perp': lambda_perp, 'n_para': cfg.WAVE['n_para'], 'n_e': n_e}
            h5f.attrs['physics_metadata'] = json.dumps(metadata)
        print(f"\n[SUCCESS] Raw data saved to {h5_filepath}")
        return h5_filepath
    return None


from scipy.stats import qmc # SciPy's Quasi-Monte Carlo library

def run_7D_sobol_pml_scan(solver, cfg, save_dir, m_power=8):
    """
    Executes a 7-Dimensional Sobol Sequence Scan for Robust PML Optimization.
    Number of simulations N = 2^m_power. (e.g., m=8 means 256 simulations).
    """
    N_simulations = 2**m_power
    print(f"--- Initiating 7D Sobol DoE: {N_simulations} Simulations ---")
    
    # 1. Define the 7D Parameter Space Bounds
    # Format: 'parameter_name': [Lower_Bound, Upper_Bound, is_log_scale]
    bounds = {
        'S_imag':   [0.5, 50.0, True],      # Log: Damping is exponential
        'L_pml':    [0.5, 3.0, False],      # Linear: Ratio of lambda_perp
        'S_real':   [1.0, 5.0, False],      # Linear: Evanscent compression
        'p_degree': [1.0, 4.0, False],      # Linear: Polynomial smoothing
        'n_para':   [-10, 10, False],      # Linear: Parallel spectrum
        'theta_B':  [0.0, np.pi/2, False],  # Linear: Pitch angle (0 to 90 deg)
        'n_e':      [1e17, 1e19, True]      # Log: Density variation
    }
    
    # 2. Generate the Sobol Sequence in [0, 1)^7
    # Scramble=True randomizes the starting point while maintaining uniform discrepancy
    sampler = qmc.Sobol(d=len(bounds), scramble=True)
    sobol_raw = sampler.random_base2(m=m_power) 
    
    # 3. Map [0, 1) to Physical Physics Bounds
    def map_bounds(val, lb, ub, is_log):
        if is_log:
            return 10**(np.log10(lb) + val * (np.log10(ub) - np.log10(lb)))
        else:
            return lb + val * (ub - lb)

    # Initialize storage arrays
    data_store = {key: np.zeros(N_simulations) for key in bounds.keys()}
    data_store['Gamma'] = np.zeros(N_simulations)
    
    Lx_plasma = cfg.DOMAIN['Lx_plasma']
    Lz_plasma_approx = cfg.DOMAIN['Lz_plasma_approx']
    z_mid = Lz_plasma_approx / 2.0
    x_eval = np.linspace(Lx_plasma * 0.25, Lx_plasma * 0.75, 1000)

    # 4. EXECUTE THE HIGH-DIMENSIONAL SCAN
    omega = cfg.WAVE['omega_wave']
    qe, me, mi, eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12

    for i in range(N_simulations):
        # Extract mapped parameters for this specific run
        p_dict = {}
        for j, (key, (lb, ub, is_log)) in enumerate(bounds.items()):
            p_dict[key] = map_bounds(sobol_raw[i, j], lb, ub, is_log)
            data_store[key][i] = p_dict[key]
            
        print(f"\nRun {i+1}/{N_simulations} | S_img:{p_dict['S_imag']:.1f}, L:{p_dict['L_pml']:.2f}, "
              f"p:{p_dict['p_degree']:.1f}, n//:{p_dict['n_para']:.2f}, ne:{p_dict['n_e']:.1e}")

        # --- A. Inject Physics into Config ---
        cfg.PLASMA['profile_type'] = 'constant_density'
        cfg.PLASMA['ne_constant'] = p_dict['n_e']
        cfg.PLASMA['theta_B_rad'] = p_dict['theta_B']
        cfg.WAVE['n_para'] = p_dict['n_para']
        
        cfg.PML['S_imag'] = p_dict['S_imag']
        cfg.PML['S_real'] = p_dict['S_real']
        cfg.PML['p_degree'] = p_dict['p_degree']
        
        # --- B. Calculate the specific lambda_perp for this plasma ---
        B0 = cfg.PLASMA['B0_center_plasma']
        w_pe2, w_pi2 = (p_dict['n_e'] * qe**2)/(me * eps0), (p_dict['n_e'] * qe**2)/(mi * eps0)
        Om_ce, Om_ci = (qe * B0)/me, (qe * B0)/mi
        
        S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
        P = 1 - w_pe2/omega**2 - w_pi2/omega**2
        D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
        
        B_stix = (S + P)*p_dict['n_para']**2 - (S**2 - D**2) - P*S
        C_stix = P * (p_dict['n_para']**2 - (S + D)) * (p_dict['n_para']**2 - (S - D))
        
        # Guard against evanescent parameters crashing the geometry
        n_perp_sq = (-B_stix + np.sqrt(max(0, B_stix**2 - 4*S*C_stix))) / (2*S)
        n_perp = np.sqrt(max(1e-6, n_perp_sq)) 
        lambda_perp = (2 * np.pi * 3e8) / (omega * n_perp)
        
        # --- C. Grow Domain and Solve ---
        cfg.DOMAIN['Lx_pml'] = p_dict['L_pml'] * lambda_perp
        cfg.DOMAIN['Lx_tot'] = cfg.DOMAIN['Lx_plasma'] + cfg.DOMAIN['Lx_pml']
        
        try:
            mesh_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Meshes")
            mesh = solver.build_mesh_with_PMLs(mesh_save_dir)
            gfu, _ = solver.solve_helmholtz_Hcurl_2D_pml(mesh, cfg)
            
            # Extract Field & SWR
            E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))
            E_vals = E_3D_full(mesh(x_eval, np.full_like(x_eval, z_mid)))
            Ez_abs = np.abs(E_vals[:, 2])
            
            SWR = max(np.max(Ez_abs) / max(np.min(Ez_abs), 1e-12), 1.0001)
            data_store['Gamma'][i] = (SWR - 1.0) / (SWR + 1.0)
            
        except Exception as e:
            print(f"  [!] Matrix inversion failed for this setup: {e}")
            data_store['Gamma'][i] = 1.0 # Maximum reflection assigned to failed physics states

    # 5. Save to ML-Ready HDF5 Database
    h5_filepath = os.path.join(save_dir, "PML_7D_Sobol_Data.h5")
    with h5py.File(h5_filepath, 'w') as h5f:
        for key, array in data_store.items():
            h5f.create_dataset(key, data=array, compression="gzip")
        
        metadata = {'m_power': m_power, 'N_simulations': N_simulations}
        h5f.attrs['sobol_metadata'] = json.dumps(metadata)
        h5f.attrs['bounds'] = json.dumps(bounds)
        
    print(f"\n[SUCCESS] 7D Sobol Data saved to {h5_filepath}")
    return h5_filepath