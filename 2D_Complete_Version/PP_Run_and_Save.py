import os
import datetime
import numpy as np
import h5py
from ngsolve import *

from solver_2DHcurl_1DH1 import *

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

# ====================================================================
# POST-TREATMENT
# ====================================================================
def run_2D_wave_map(mesh, gfu, cfg, save_dir, mode, diag_data, resolution=(300, 300)):
    """Extracts the full 2D complex wave field and saves to HDF5."""
    print("--- Extracting 2D Wave Map ---")
    Lx_plasma, Lx_pml, Lx_tot = cfg.DOMAIN['Lx_plasma'], cfg.DOMAIN['Lx_pml'], cfg.DOMAIN['Lx_tot']
    Lz_plasma, Lz_pml, Lz_tot = cfg.DOMAIN['Lz_plasma'], cfg.DOMAIN['Lz_pml'], cfg.DOMAIN['Lz_tot']
    print('[run 2D E map]')
    print(f'Lx_plasma: {Lx_plasma:.2e}m, Lx_pml: {Lx_pml:.2e}m, Lx_tot: {Lx_tot:.2e}m')
    print(f'Lz_plasma: {Lz_plasma:.2e}m, Lz_pml: {Lz_pml:.2e}m, Lz_tot: {Lz_tot:.2e}m')
    nx, nz = resolution
    eps = 1e-6 
    print('params recover in run_2D_wave_map')
    
    x_coords = np.linspace(eps, Lx_tot - eps, nx)
    if mode == "RADIAL_ONLY":
        z_coords = np.linspace(eps, Lz_plasma - eps, nz)
    else: 
        z_coords = np.linspace(-Lz_pml + eps, Lz_plasma + Lz_pml - eps, nz)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))

    # --- Poynting vector computation -----
    omega_LH, mu0 = cfg.WAVE['omega_LH'], cfg.CONST['mu0']
    E_plane, E_outplane = gfu.components[0], gfu.components[1]
    curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
    H_3D_full = 1.0 / (1j * omega_LH * mu0) * curl_E_3D
    
    # S = 1/2 Re(E x H*)
    S_3D_full = 0.5 * Cross(E_3D_full, Conj(H_3D_full)).real
    
    x_flat, z_flat = X.flatten(), Z.flatten()
    Ex_flat, Ey_flat, Ez_flat = np.zeros_like(x_flat, dtype=complex), np.zeros_like(x_flat, dtype=complex), np.zeros_like(x_flat, dtype=complex)
    E_norm_flat = np.zeros_like(x_flat, dtype=float)
    Sx_flat, Sy_flat, Sz_flat = np.zeros_like(x_flat, dtype=float), np.zeros_like(x_flat, dtype=float), np.zeros_like(x_flat, dtype=float)

    print("  --> Safely interpolating FEM fields onto structured grid...")
    for i in range(len(x_flat)):
        try:
            mip = mesh(x_flat[i], z_flat[i])
            val_E = E_3D_full(mip)
            val_S = S_3D_full(mip)
            
            Ex_flat[i], Ey_flat[i], Ez_flat[i] = val_E[0], val_E[1], val_E[2]
            E_norm_flat[i] = np.sqrt(abs(val_E[0])**2 + abs(val_E[1])**2 + abs(val_E[2])**2)
            Sx_flat[i], Sy_flat[i], Sz_flat[i] = val_S[0], val_S[1], val_S[2]
        except Exception:
            pass
            
    Ex, Ey, Ez = Ex_flat.reshape(nx, nz), Ey_flat.reshape(nx, nz), Ez_flat.reshape(nx, nz)
    E_norm = E_norm_flat.reshape(nx, nz)
    Sx, Sy, Sz = Sx_flat.reshape(nx, nz), Sy_flat.reshape(nx, nz), Sz_flat.reshape(nx, nz)
    # -------------------------------------

    solver = solver = LHCouplingSolver_2DHcurl_1DH1(cfg.__dict__, mode)
    n_para, n_perp_p, n_perp_m = solver.compute_physics_parameters()
    print(f'PP_run_n_save: n_para: {n_para:.1f}, n_perp_p: {n_perp_p:.2e}, n_perp_m: {n_perp_m:.2e}')

    print('Params recovered')
    if save_dir is not None:
        h5_path = os.path.join(save_dir, "Wave_Map_2D.h5")
        print(f"h5_path = {h5_path}")
        print(f"type(h5_path) = {type(h5_path)}")
        with h5py.File(h5_path, 'w') as h5f:
            print('oui')
            h5f.create_dataset('X', data=X, compression="gzip")
            h5f.create_dataset('Z', data=Z, compression="gzip")
            h5f.create_dataset('Ex', data=Ex, compression="gzip")
            h5f.create_dataset('Ey', data=Ey, compression="gzip")
            h5f.create_dataset('Ez', data=Ez, compression="gzip")
            h5f.create_dataset('Sx', data=Sx, compression="gzip")
            h5f.create_dataset('Sy', data=Sy, compression="gzip")
            h5f.create_dataset('Sz', data=Sz, compression="gzip")
            h5f.create_dataset('E_norm', data=E_norm, compression="gzip")
            
            h5f.attrs['Lx_plasma'], h5f.attrs['Lx_pml'], h5f.attrs['Lx_tot'] = Lx_plasma, Lx_pml, Lx_tot
            h5f.attrs['Lz_plasma'], h5f.attrs['Lz_pml'], h5f.attrs['Lz_tot'] = Lz_plasma, Lz_pml, Lz_tot
            
            h5f.attrs['k_para'] = - cfg.WAVE['k0'] * n_para
            h5f.attrs['k_perp_p'] = -1.0 * cfg.WAVE['k0'] * n_perp_p
            
            if diag_data:
                for key, val in diag_data.items():
                    if val is not None:
                        print(f'key: {key}, type(key): {type(key)}, val: {val}')
                        h5f.attrs[key] = val
            print('PRINT H5 CONTENT:')
            for key, val in h5f.attrs.items():
                print(f' {key}: {val}')

        print(f"--- 2D Map saved to {h5_path} ---")
        return h5_path
    else:
        print("TEST MODE: No h5 file data saving")
