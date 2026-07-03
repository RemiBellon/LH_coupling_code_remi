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
def run_2D_wave_map(mesh, gfu, cfg, save_dir, geom_mode, box_medium, antenna_grill, diag_data, resolution=(400, 400)):
    """Extracts the full 2D complex wave field and saves to HDF5."""
    print("--- Extracting 2D Wave Map ---")
    Lx_plasma, Lx_pml, Lx_tot = cfg.DOMAIN['Lx_plasma'], cfg.DOMAIN['Lx_pml'], cfg.DOMAIN['Lx_tot']
    
    # Récupération stricte des dimensions Z
    Lz_plasma_src = cfg.DOMAIN['Lz_plasma']
    Lz_pml = cfg.DOMAIN.get('Lz_pml', 0.0)
    Lz_wall = cfg.DOMAIN.get('Lz_wall', 0.0)
    Lz_tot = Lz_plasma_src + 2.0 * Lz_wall + 2.0 * Lz_pml
    print(f'Lz_plasma_src: {Lz_plasma_src:.3f}, Lz_wall: {Lz_wall:.3f}, Lz_pml: {Lz_pml:.3f}, Lz_tot: {Lz_tot:.3f}')
    nx, nz = resolution
    eps, eps_m = 1e-6, 1e-5 
    
    Lx_wg_extract = cfg.DOMAIN.get('Lx_wg', 0.0) if antenna_grill is not None else 0.0
    x_coords = np.linspace(-Lx_wg_extract + eps, Lx_tot - eps, nx)
    
    # --- FIX 1: BORNES GÉOMÉTRIQUES STRICTES ET INFAILLIBLES ---
    if geom_mode == "1D":
        z_min_domain = 0.0
        z_max_domain = Lz_plasma_src + 2.0 * Lz_wall
    else: 
        z_min_domain = -Lz_pml
        z_max_domain = Lz_plasma_src + 2.0 * Lz_wall + Lz_pml
    print(f'Bornes géométriques fixées')
    z_coords = np.linspace(z_min_domain + eps, z_max_domain - eps, nz)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
    
    E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))

    # --- Poynting vector computation -----
    omega_LH, mu0 = cfg.WAVE['omega_LH'], cfg.CONST['mu0']
    E_plane, E_outplane = gfu.components[0], gfu.components[1]
    curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0]))
    H_3D_full = 1.0 / (1j * omega_LH * mu0) * curl_E_3D
    
    # S = 1/2 Re(E x H*)
    S_3D_full = 0.5 * Cross(E_3D_full, Conj(H_3D_full)).real
    
    x_flat, z_flat = X.flatten(), Z.flatten()
    Ex_flat, Ey_flat, Ez_flat = np.zeros_like(x_flat, dtype=complex), np.zeros_like(x_flat, dtype=complex), np.zeros_like(x_flat, dtype=complex)
    E_norm_flat = np.zeros_like(x_flat, dtype=float)
    Sx_flat, Sy_flat, Sz_flat = np.zeros_like(x_flat, dtype=float), np.zeros_like(x_flat, dtype=float), np.zeros_like(x_flat, dtype=float)

# --- FIX 2: MULTITHREADED C++ VECTORIZED INTERPOLATION ---
    print("  --> Pure Multithreaded C++ vectorized interpolation...")
    
    # 1. Masque Topologique Strict (Marge de 1 micron pour exclure le métal et les bords)
    eps_m = 1e-5
    valid_mask = np.zeros_like(x_flat, dtype=bool)
    
    # Zone Plasma & PML
    z_min_domain = -Lz_pml if geom_mode != "RADIAL_ONLY" else 0.0
    if geom_mode != "RADIAL_ONLY":
        z_max_domain = Lz_plasma_src + 2.0 * Lz_wall + Lz_pml
    else:
        z_max_domain = Lz_plasma_src + 2.0 * Lz_wall
    
    in_main = (x_flat >= eps_m) & (x_flat <= Lx_tot - eps_m) & \
              (z_flat >= z_min_domain + eps_m) & (z_flat <= z_max_domain - eps_m)
    valid_mask = valid_mask | in_main
    
    # Zone Guides d'Ondes (strictement dans le vide maillé)
    if Lx_wg_extract > 0 and antenna_grill is not None:
        Lz_wall = cfg.DOMAIN.get('Lz_wall', 0.02)
        instructions = antenna_grill.generate_mesh_instructions(z_start_position=Lz_wall)
        for inst in instructions:
            if inst['type'] in ['wg_active', 'wg_passive']:
                in_wg = (x_flat >= -Lx_wg_extract + eps_m) & (x_flat < -eps_m) & \
                        (z_flat >= inst['z_start'] + eps_m) & (z_flat <= inst['z_end'] - eps_m)
                valid_mask = valid_mask | in_wg

    # 2. Sous-échantillonnage Numpy (Buffer C direct)
    x_valid = x_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    
    Ex_valid, Ey_valid, Ez_valid = np.zeros(len(x_valid), dtype=complex), np.zeros(len(x_valid), dtype=complex), np.zeros(len(x_valid), dtype=complex)
    Sx_valid, Sy_valid, Sz_valid = np.zeros(len(x_valid), dtype=float), np.zeros(len(x_valid), dtype=float), np.zeros(len(x_valid), dtype=float)

    # 3. Évaluation NGSolve Native et Multithreadée
    # Pas de try...except, pas de .tolist(). Vitesse RAM maximale.
    CHUNK = 250000 
    with TaskManager():
        for i in range(0, len(x_valid), CHUNK):
            xc = x_valid[i : i+CHUNK]
            zc = z_valid[i : i+CHUNK]
            
            mips = mesh(xc, zc)
            val_E = np.array(E_3D_full(mips))
            val_S = np.array(S_3D_full(mips))
            
            Ex_valid[i:i+CHUNK], Ey_valid[i:i+CHUNK], Ez_valid[i:i+CHUNK] = val_E[:, 0], val_E[:, 1], val_E[:, 2]
            Sx_valid[i:i+CHUNK], Sy_valid[i:i+CHUNK], Sz_valid[i:i+CHUNK] = val_S[:, 0], val_S[:, 1], val_S[:, 2]

    # 4. Recomposition de la grille complète
    Ex_flat, Ey_flat, Ez_flat = np.zeros(len(x_flat), dtype=complex), np.zeros(len(x_flat), dtype=complex), np.zeros(len(x_flat), dtype=complex)
    Sx_flat, Sy_flat, Sz_flat = np.zeros(len(x_flat), dtype=float), np.zeros(len(x_flat), dtype=float), np.zeros(len(x_flat), dtype=float)

    Ex_flat[valid_mask], Ey_flat[valid_mask], Ez_flat[valid_mask] = Ex_valid, Ey_valid, Ez_valid
    Sx_flat[valid_mask], Sy_flat[valid_mask], Sz_flat[valid_mask] = Sx_valid, Sy_valid, Sz_valid
    
    # Calcul de la norme via Numpy
    E_norm_flat = np.sqrt(np.abs(Ex_flat)**2 + np.abs(Ey_flat)**2 + np.abs(Ez_flat)**2)
            
    Ex, Ey, Ez = Ex_flat.reshape(nx, nz), Ey_flat.reshape(nx, nz), Ez_flat.reshape(nx, nz)
    E_norm = E_norm_flat.reshape(nx, nz)
    Sx, Sy, Sz = Sx_flat.reshape(nx, nz), Sy_flat.reshape(nx, nz), Sz_flat.reshape(nx, nz)
    # -------------------------------------

    solver = LHCouplingSolver_2DHcurl_1DH1(cfg.__dict__, geom_mode, box_medium, antenna_grill)
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
            h5f.attrs['Lz_plasma'], h5f.attrs['Lz_pml'], h5f.attrs['Lz_tot'] = Lz_plasma_src, Lz_pml, Lz_tot
            
            h5f.attrs['k_para'] = - cfg.WAVE['k0'] * n_para
            h5f.attrs['k_perp_p'] = -1.0 * cfg.WAVE['k0'] * n_perp_p
            
            if diag_data:
                for key, val in diag_data.items():
                    if val is not None:
                        # --- FIX 2: SÉPARATION DES DATASETS ET ATTRIBUTS ---
                        if isinstance(val, np.ndarray) and val.size > 1:
                            h5f.create_dataset(key, data=val, compression="gzip")
                        elif isinstance(val, (complex, np.complex128, np.complex64)):
                            h5f.attrs[f"{key}_real"] = val.real
                            h5f.attrs[f"{key}_imag"] = val.imag
                        else:
                            h5f.attrs[key] = val
                            
            print('\nPRINT H5 METADATA:')
            for key, val in h5f.attrs.items():
                if isinstance(val, (float, np.floating)):
                    print(f' {key}: {val:.2e}')
                else:
                    print(f' {key}: {val}')

        print(f"--- 2D Map saved to {h5_path} ---")
        return h5_path
    else:
        print("TEST MODE: No h5 file data saving")
