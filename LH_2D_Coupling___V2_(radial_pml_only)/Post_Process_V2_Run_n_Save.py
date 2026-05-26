import os
import time
import datetime
import json
import numpy as np
import h5py
import gc # Garbage Collector
from ngsolve import *
from pathlib import Path
from scipy.stats import qmc

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

def create_density_profile(x_val, z_val, solver):
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
    Lx_tot, Lx_plasma, Lz_exact = cfg.DOMAIN['Lx_tot'], cfg.DOMAIN['Lx_plasma'], cfg.DOMAIN['Lz_plasma']
    nx, nz = resolution
    eps = 1e-6 
    print('params recover in run_2D_wave_map')
    
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



def run_pml_scan_dataset(solver, cfg, save_dir, m_power=12):
    """
    11D Sobol Sequence with Iterative Checkpointing.
    m_power = 10 generates 1024 simulations.
    Plasma with radial and toroidal pmls
    """
    if save_dir is None:
        print("[ERROR] Ultimate Scan requires a save_dir to stream data. Aborting.")
        return None

    N_simulations = 2**m_power
    print(f"--- Initiating ULTIMATE 7D Sobol Data Lake: {N_simulations} Simulations ---")
    
    # 1. 11D Parameter Space Bounds
    bounds = {
        'n_para':       [1.5, 10.0, False],   # Toroidal reflection is highly sensitive to n_para now!
        'n_e':          [1e17, 1e19, True],   
        
        # RADIAL PML (Restricted to known good ranges from Phase 1)
        'Lx_ratio':     [10.0, 20.0, False],  
        'Sx_im':         [3.0, 8.0, False],    
        'Sx_r':         [1.0, 4.0, False],    
        'px':           [1.5, 3.0, False],    
        
        # TOROIDAL PML (Wide open for exploration)
        'Lz_ratio':     [1.0, 10.0, False],   # Multiple of lambda_para 
        'Sz_im':         [0.5, 30.0, True],    
        'Sz_r':         [1.0, 10.0, False],    
        'pz':           [1.0, 4.0, False]     
    }
    
    # Generate Sobol Sequence
    sampler = qmc.Sobol(d=len(bounds), scramble=True)
    sobol_raw = sampler.random_base2(m=m_power) 
    
    def map_bounds(val, lb, ub, is_log):
        return 10**(np.log10(lb) + val * (np.log10(ub) - np.log10(lb))) if is_log else lb + val * (ub - lb)

    # 2. Define the Master List of Variables to Record
    recorded_vars = [
        'S_imag', 'L_pml_ratio', 'S_real', 'p_degree',
        'n_para', 'n_e',
        'omega', 'B0', 'S_stix', 'P_stix', 'D_stix', 
        'n_perp_plus', 'n_perp_minus', 'lambda_perp', 'lambda_para', 'Lx_pml',
        'DoFs', 'CPU_Time',

        'Gamma_E_Radial', 'Gamma_E_Toroidal'
    ]

    # 3. Pre-allocate the HDF5 Database 
    h5_filepath = os.path.join(save_dir, "PML_scan_all_param_dataset.h5")
    with h5py.File(h5_filepath, 'w') as h5f:
        # Create empty datasets of size N for every variable
        for var in recorded_vars:
            h5f.create_dataset(var, shape=(N_simulations,), dtype='float64', compression="gzip")
        
        # Save exact metadata
        h5f.attrs['N_simulations'] = N_simulations
        h5f.attrs['bounds'] = json.dumps(bounds)
        
    # 4. EXECUTE THE SIMULATIONS
    omega = cfg.WAVE['omega_wave']
    B0 = cfg.PLASMA['B0_center_plasma']
    mu0 =  4*np.pi*1e-7
    eps0 = cfg.CONST['eps_0']
    me = cfg.CONST['me']
    mi = cfg.CONST['mi']
    qe = cfg.CONST['qe']
    c0 = cfg.CONST['c0']

    for i in range(N_simulations):
        t_start = time.time()
        
        # A. Map parameters for this run
        p_dict = {}
        for j, (key, (lb, ub, is_log)) in enumerate(bounds.items()):
            p_dict[key] = map_bounds(sobol_raw[i, j], lb, ub, is_log)
        print('\n') 
        print(f"\nRun {i+1}/{N_simulations} | S_i:{p_dict['S_imag']:.1f}, L:{p_dict['L_pml_ratio']:.1f}λ, "
              f"p:{p_dict['p_degree']:.1f}, n//:{p_dict['n_para']:.2f}, ne:{p_dict['n_e']:.1e}")
        print('\n')
        # B. Inject Physics
        cfg.PLASMA['profile_type'] = 'constant_density'
        cfg.PLASMA['ne_constant'] = p_dict['n_e']
        cfg.WAVE['n_para'] = p_dict['n_para']
        cfg.PML['S_imag'] = p_dict['S_imag']
        cfg.PML['S_real'] = p_dict['S_real']
        cfg.PML['p_degree'] = p_dict['p_degree']
        
        # C. Calculate Analytical Physics
        w_pe2, w_pi2 = (p_dict['n_e'] * qe**2)/(me * eps0), (p_dict['n_e'] * qe**2)/(mi * eps0)
        Om_ce, Om_ci = (qe * B0)/me, (qe * B0)/mi
        
        S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
        P = 1 - w_pe2/omega**2 - w_pi2/omega**2
        D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
        
        B_stix = (S + P)*p_dict['n_para']**2 - (S**2 - D**2) - P*S
        C_stix = P * (p_dict['n_para']**2 - (S + D)) * (p_dict['n_para']**2 - (S - D))
        
          # --- THE PRE-MESH PHYSICS FILTER ---
        delta = B_stix**2 - 4*S*C_stix
        # print(f"--- delta = {delta} ---")
        # print(f"--- B_stix = {B_stix}, S = {S}, C_stix = {C_stix} ---")
        # print(f"--- n_perp_sq_p = {n_perp_sq_p} ---")
        n_perp_sq_p = (-B_stix + np.sqrt(max(0, delta))) / (2*S) if delta >= 0 else -1.0
        n_perp_sq_m = (-B_stix - np.sqrt(max(0, delta))) / (2*S) if delta >= 0 else -1.0

        if n_perp_sq_p <= 0.0:
            
            print("  [!] Wave is Evanescent (n_perp^2 <= 0). Perfect reflection. Skipping FEM Solver.")
            n_perp_p, n_perp_m, lambda_perp, lambda_para, Lx_pml = 0.0, 0.0, 0.0, 0.0, 0.0
            ndofs, Gamma_E, Gamma_S = 0, 1.0, 1.0 # 100% reflection assigned immediately
            
        else:
            n_perp_p = np.sqrt(n_perp_sq_p)
            n_perp_m = np.sqrt(max(1e-6, n_perp_sq_m))
            
            lambda_perp = (2 * np.pi * c0) / (omega * n_perp_p)
            lambda_para = (2 * np.pi * c0) / (omega * p_dict['n_para'])
            
            Lx_plasma_dynamic = 2.0 * lambda_perp
            Lx_pml = p_dict['L_pml_ratio'] * lambda_perp
            Lx_tot = Lx_plasma_dynamic + Lx_pml
            Lz_exact = lambda_para
            
            aspect_ratio = Lx_tot / Lz_exact

            # THE PRE-MESH GEOMETRY FILTER
            if aspect_ratio > 10000 or aspect_ratio < 1e-4:
                print(f"  [!] Extreme Aspect Ratio ({aspect_ratio:.1f}). Netgen will crash. Skipping FEM Solver.")
                ndofs, Gamma_E, Gamma_S = 0, 1.0, 1.0
                
            else:
                cfg.DOMAIN['Lx_plasma'] = Lx_plasma_dynamic
                cfg.DOMAIN['Lx_pml'] = Lx_pml
                cfg.DOMAIN['Lx_tot'] = Lx_tot
                cfg.DOMAIN['Lz_plasma'] = Lz_exact 

        # print(f"\n[Run PMLs Scan] :")
        # print(f"  --> n_perp_p : {n_perp_p:.5e}, n_perp_m = {n_perp_m:.5e}")
        # print(f"  --> lambda_perp : {lambda_perp:.5e} m, lambda_para = {lambda_para:.5e} m")
        # print(f"  --> Lx_plm = {Lx_pml:.5e} m, Lx_tot = {Lx_tot:.5e} m")
        # print(f"  --> Lz_exact: {Lz_exact:.5e} m")
        
        try:
            mesh_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3_(radial&toroidal_pmls)/Meshes")
            mesh = solver.build_mesh_with_PMLs(mesh_save_dir)
            gfu, ndofs = solver.solve_helmholtz_Hcurl_2D_pml(mesh, cfg)
            
            # --- Professional 1D Field Extraction ---
                
            E_xz, Ey = gfu.components[0], gfu.components[1]
            Ex, Ez = E_xz[0], E_xz[1]                
            # Complex magnitude of the Electric Field
            E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))
            
            # Create a strict 1D array of points perfectly centered along the Z-axis
            z_mid = Lz_exact / 2.0
            x_vals = np.linspace(Lx_plasma_dynamic * 0.25, Lx_plasma_dynamic * 0.75, 500)
            
            # Map to NGSolve integration points (mips)
            mips_x = mesh(x_vals, np.full_like(x_vals, z_mid))
            
            # Evaluate field on the points (extracting the real magnitude)
            E_vals_x = np.array(E_tot_norm(mips)).real
            
            # Calculate Standing Wave Ratio (SWR) 
            SWR_Radial = max(np.max(E_vals_x) / np.max([np.min(E_vals_x), 1e-12]), 1.0001)
            Gamma_E_Radial = (SWR_Radial - 1.0) / (SWR_Radial + 1.0)
            
            # Toroidal SWR computation:
            x_eval_z = 0.1 * Lx_plasma
            z_eval = np.linspace(Lz_exact * 0.25, Lz_exact * .75, 500)
            mips_z = mesh(np.full_like(z_eval, x_eval_z), z_eval)
            E_vals_z = np.array(E_tot_norm(mips_z)).real

            SWR_Toroidal = max(np.max(E_vals_z) / max(np.min(E_vals_z), 1e-12), 1.0001)
            Gamma_E_Toroidal = (SWR_Toroidal - 1.0) / (SWR_Toroidal + 1.0)
            
            print(f"  --> Success | Gamma_Radial: {Gamma_E_R:.2e} | Gamma_Toroidal: {Gamma_E_T:.2e}")
            
        except Exception as e:
            print(f"  [!] Failed Physics State: {e}")
            ndofs, Gamma_E_Radial, Gamma_E_Toroidal = 0, 1.0, 1.0

        cpu_time = time.time() - t_start

        # E. The Streaming Save (CHECKPOINTING)
        data_to_write = {
            'S_imag': p_dict['S_imag'], 'L_pml_ratio': p_dict['L_pml_ratio'], 'S_real': p_dict['S_real'],
            'p_degree': p_dict['p_degree'], 'n_para': p_dict['n_para'],  
            'n_e': p_dict['n_e'], 'omega': omega, 'B0': B0, 'S_stix': S, 'P_stix': P, 'D_stix': D, 
            'n_perp_plus': n_perp_p, 'n_perp_minus': n_perp_m, 'lambda_perp': lambda_perp, 
            'lambda_para': lambda_para, 'Lx_pml': Lx_pml, 'DoFs': ndofs, 
            'CPU_Time': cpu_time, 'Gamma_E': Gamma_E, 'Gamma_S': Gamma_S
        }
        
        # Open file, write index i, instantly flush to disk
        with h5py.File(h5_filepath, 'a') as h5f:
            for key in recorded_vars:
                h5f[key][i] = data_to_write[key]
            h5f.flush() # CRITICAL: Forces OS to write to hard drive immediately
        print('--- Run ok & data saved correctly ---')
        # F. RAM Garbage Collection (Obliterate C++ Memory Leaks)
        try:
            del mesh, gfu
        except:
            pass
        gc.collect()

    print(f"\n[SUCCESS] Ultimate Data Lake built perfectly: {h5_filepath}")
    return h5_filepath
