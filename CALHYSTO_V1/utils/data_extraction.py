import os
import numpy as np
import h5py
import scipy.constants as const
from ngsolve import CF, grad, curl, Conj, Cross, TaskManager
from config.schema import SimulationConfig
from utils.antenna_desc_2D import AntennaGrill
from utils.antenna_desc_2D import build_grill_from_config

def extract_2d_wave_map(config: SimulationConfig, mesh, E_field_gf, save_dir: str, nx: int = 1000, nz: int = 1000):
    """
    Highly optimized multithreaded extraction of the full wave field onto a structured Numpy grid.
    Saves data to an HDF5 file for decoupled, high-speed post-processing.
    """
    print("\n--- Extracting 2D Wave Map (Optimized C++ Interpolation) ---")

    dom = config.geometry.domain
    ant = config.geometry.antenna

    # 1. Define Strict Geometric Bounds
    Lx_wg = ant.dimensions.Lx_wg_active if ant else 0.0
    x_min, x_max = -Lx_wg, dom.Lx_plasma + dom.Lx_pml
    z_min, z_max = -dom.Lz_pml, dom.Lz_plasma + dom.Lz_pml

    # Add microscopic epsilon to avoid querying exact boundary faces
    eps = 1e-6
    x_coords = np.linspace(x_min + eps, x_max - eps, nx)
    z_coords = np.linspace(z_min + eps, z_max - eps, nz)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')

    # 2. Construct Symbolic Physics Functions
    E_plane, E_outplane = E_field_gf.components[0], E_field_gf.components[1]
    E_3D_full = CF((E_plane[0], E_outplane, E_plane[1]))
    x_flat, z_flat = X.flatten(), Z.flatten()

    eps_mask = 1e-5
    valid_mask = np.zeros_like(x_flat, dtype=bool)
    valid_mask |= (x_flat >= eps_mask)

    if ant is not None:
        grill, instructions = build_grill_from_config(config.geometry.antenna, config.geometry.domain)

        for inst in instructions:
            if inst['type'] in ['wg_active', 'wg_passive']:
                z_s = inst['z_start']
                z_e = inst['z_end']

                depth = inst.get('depth', Lx_wg)

                # Verify if coordinates fall cleanly inside this specific waveguide channel
                in_wg = (x_flat <= -eps_mask) & (x_flat >= -depth + eps_mask) & \
                        (z_flat >= z_s + eps_mask) & (z_flat <= z_e - eps_mask)
                valid_mask |= in_wg

    x_valid = x_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    Ex_valid = np.zeros(len(x_valid), dtype=complex)
    Ey_valid = np.zeros(len(x_valid), dtype=complex)
    Ez_valid = np.zeros(len(x_valid), dtype=complex)

    # 4. Multithreaded Chunk Evaluation
    CHUNK = 250000
    with TaskManager():
        for i in range(0, len(x_valid), CHUNK):
            xc = x_valid[i : i+CHUNK]
            zc = z_valid[i : i+CHUNK]

            mips = mesh(xc, zc)
            val_E = np.array(E_3D_full(mips))

            # Catch any stray NaNs that survived the geometric masking
            safe_idx = ~np.isnan(val_E[:, 0])

            temp_Ex = np.zeros(len(xc), dtype=complex)
            temp_Ey = np.zeros(len(xc), dtype=complex)
            temp_Ez = np.zeros(len(xc), dtype=complex)

            temp_Ex[safe_idx] = val_E[:, 0][safe_idx]
            temp_Ey[safe_idx] = val_E[:, 1][safe_idx]
            temp_Ez[safe_idx] = val_E[:, 2][safe_idx]

            Ex_valid[i:i+CHUNK] = temp_Ex
            Ey_valid[i:i+CHUNK] = temp_Ey
            Ez_valid[i:i+CHUNK] = temp_Ez

    # 5. Reconstruct the Full Dense Grid
    Ex_flat = np.zeros_like(x_flat, dtype=complex)
    Ey_flat = np.zeros_like(x_flat, dtype=complex)
    Ez_flat = np.zeros_like(x_flat, dtype=complex)

    Ex_flat[valid_mask] = Ex_valid
    Ey_flat[valid_mask] = Ey_valid
    Ez_flat[valid_mask] = Ez_valid

    Ex, Ey, Ez = Ex_flat.reshape(nx, nz), Ey_flat.reshape(nx, nz), Ez_flat.reshape(nx, nz)
    E_norm = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)

    # compute Poynting vector
    curl_E_3D = CF((-grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0]))
    H_3D_full = 1.0 / (1j * config.physics.wave.omega_LH * const.mu_0) * curl_E_3D
    S_3D_full = 0.5 * Cross(E_3D_full, Conj(H_3D_full)).real

    # 6. Save to HDF5
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, "Wave_Map_2D.h5")

    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset('X', data=X, compression="gzip")
        h5f.create_dataset('Z', data=Z, compression="gzip")
        h5f.create_dataset('Ex', data=Ex, compression="gzip")
        h5f.create_dataset('Ey', data=Ey, compression="gzip")
        h5f.create_dataset('Ez', data=Ez, compression="gzip")
        h5f.create_dataset('E_norm', data=E_norm, compression="gzip")

        h5f.attrs['Lx_plasma'] = dom.Lx_plasma
        h5f.attrs['Lz_plasma'] = dom.Lz_plasma
        h5f.attrs['Lz_wall'] = dom.Lz_wall

    print(f"--- Data strictly extracted and saved to {h5_path} ---")
    return h5_path