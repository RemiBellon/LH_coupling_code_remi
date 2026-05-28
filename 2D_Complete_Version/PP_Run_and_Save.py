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

# ====================================================================
# POST-TREATMENT
# ====================================================================
def run_2D_wave_map(mesh, gfu, cfg, save_dir, mode, resolution=(300, 300)):
    """Extracts the full 2D complex wave field and saves to HDF5."""
    print("--- Extracting 2D Wave Map ---")
    Lx_plasma, Lx_pml, Lx_tot = cfg.DOMAIN['Lx_plasma'], cfg.DOMAIN['Lx_pml'], cfg.DOMAIN['Lx_tot']
    Lz_plasma, Lz_pml, Lz_tot = cfg.DOMAIN['Lz_plasma'], cfg.DOMAIN['Lz_pml'], cfg.DOMAIN['Lz_tot']

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
    mips = mesh(X.flatten(), Z.flatten())
    E_vals = E_3D_full(mips)
    
    Ex = E_vals[:, 0].reshape(nx, nz)
    Ey = E_vals[:, 1].reshape(nx, nz)
    Ez = E_vals[:, 2].reshape(nx, nz)
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
            h5f.attrs['Lx_plasma'] = Lx_plasma
            h5f.attrs['Lx_pml'] = Lx_pml
            h5f.attrs['Lx_tot'] = Lx_tot
            h5f.attrs['Lz_plasma'] = Lz_plasma
            h5f.attrs['Lz_pml'] = Lz_pml
            h5f.attrs['Lz_tot'] =  Lz_tot
        print(f"--- 2D Map saved to {h5_path} ---")
        return h5_path
    else:
        print("TEST MODE: No h5 file data saving")
