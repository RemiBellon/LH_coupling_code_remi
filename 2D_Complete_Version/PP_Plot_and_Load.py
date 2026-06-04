import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from scipy.fft import fft, fftfreq, fftshift
from ngsolve import *

def plot_2D_wave_map(h5_filepath, figure_save_dir, mode, component='Ez', value_type='real', plot_poynting=True, show_windows=True):
    print(f"--- Plotting 2D Map from {h5_filepath} ---")
    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        if component == 'E_norm':
            E_comp = h5f['E_norm'][:]
            plot_data = E_comp
            cmap = 'magma'
            vmin, vmax = 0.0, np.max(plot_data)
        else: 
            E_comp = h5f[component][:] # Automatically grabs Ex, Ey, or Ez
            plot_data = E_comp.real if value_type == 'real' else np.abs(E_comp)
            cmap = 'magma' if value_type == 'abs' else 'coolwarm'
            vmax = np.max(plot_data)
            vmin = 0.0 if value_type == 'abs' else -vmax
            
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lz_plasma = h5f.attrs['Lz_plasma']
        Sx, Sz = h5f['Sx'][:], h5f['Sz'][:]
        k_para, k_perp_p = h5f.attrs['k_para'], h5f.attrs['k_perp_p']

    fig, ax = plt.subplots(figsize=(14, 8))
    
    c = ax.pcolormesh(Z, X, plot_data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(c, ax=ax)  #, label=f'{component} field ({value_type if component != 'E_norm' else 'Absolute'})')
    cbar.set_label(f"Wave Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=14)

    ax.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=4, alpha=0.8, 
               label='Radial PML border', path_effects=[pe.withStroke(linewidth=6, foreground="black")])
    if mode != "RADIAL_ONLY":
        ax.axvline(x=0, color='white', linestyle='--', linewidth=4, alpha=0.8, 
               label='Radial PML border', path_effects=[pe.withStroke(linewidth=6, foreground="black")])
        ax.axvline(x=Lz_plasma, color='white', linestyle='--', linewidth=4, alpha=0.8, 
               label='Radial PML border', path_effects=[pe.withStroke(linewidth=6, foreground="black")])

    if plot_poynting and component:
        strm = ax.streamplot(Z[0,:], X[:, 0], Sz, Sx, color='black', linewidth=1.5, density=0.8, arrowstyle='->', arrowsize=1.5)
        z_center, x_center = Lz_plasma * 0.5, Lx_plasma * 0.5
        k_norm = np.sqrt(k_para**2 + k_perp_p**2)
        k_scale = 0.15 * Lx_plasma
        kx_plot, kz_plot = (k_perp_p/k_norm) * k_scale, (k_para/k_norm) * k_scale
        print(f'kx_plot: {kx_plot}, kz_plot: {kz_plot}')
        ax.quiver(z_center, x_center, kz_plot, kx_plot, color='yellow', scale=1, scale_units='xy', width=0.008, pivot='tail', zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax.text(z_center + 1.5*kz_plot, x_center + kx_plot, r'$\mathbf{k}$', color='Yellow', fontsize=18, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    if show_windows and 'x_target_R' in h5f.attrs:
        x_target_R = h5f.attrs['x_target_R']
        peak_z_R = h5f.attrs['peak_z_R']
        window_size_radial = h5f.attrs['window_size_radial']
        ax.avline(x=x_target_R, color="white", linesyle='--', alpha=0.5, label='Radial Target Line')
    
    
    theta_rad, phi_rad = 0, 0
    # B-Field Vector
    bx, bz = np.sin(phi_rad), np.cos(phi_rad) * np.cos(theta_rad)
    norm_b = np.sqrt(bx**2 + bz**2)
    if norm_b > 1e-6:
        bx, bz = bx / norm_b, bz / norm_b
        arrow_z, arrow_x, len_scale = Lz_plasma * 0.85, Lx_plasma * 0.85, Lz_plasma * 0.08
        ax.quiver(arrow_z, arrow_x, bz * len_scale, bx * len_scale, color='lime', scale=1, scale_units='xy', width=0.005, pivot='tail', zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.text(arrow_z + bz * len_scale, arrow_x + bx * len_scale, r'$\mathbf{B}_0$', color='lime', fontsize=16, fontweight='bold', ha='left', va='bottom', path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_title(f"Lower Hybrid Coupling: {component} component", fontsize=16)
    ax.set_xlabel(r'Toroidal position $z$ [m]', fontsize=16)
    ax.set_ylabel(r'Radial position $x$ [m]', fontsize=16)
    ax.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, right=True, left=True)
    
    plt.tight_layout()
    if figure_save_dir is not None:
        suffix = "_Poynting" if plot_poynting else ""
        plt.savefig(os.path.join(figure_save_dir, f"Map_{component}_{value_type}{suffix}.png"), dpi=300)
    plt.show()



    # ========================================================================================

def plot_n_para_spectrum(mesh, gfu, cfg, mode, x_eval, num_points=3000, pad_factor=8):
    # Extract domain sizes
    Lz_plasma, Lz_pml = cfg.DOMAIN['Lz_plasma'], cfg.DOMAIN['Lz_pml']
    if mode == "RADIAL_ONLY":
        z_min, z_max = 0.0, Lz_plasma 
        z_coords, dz = np.linspace(z_min, z_max, num_points, endpoint=True, retstep=True)
    else:
        z_min, z_max = -Lz_pml, Lz_plasma + Lz_pml
        z_coords, dz = np.linspace(z_min, z_max, num_points, endpoint=True, retstep=True)

    # Extract Ez field
    Ez_vals = np.zeros(num_points, dtype=complex)
    Ez_field = gfu.components[0][1]
    for i, z in enumerate(z_coords):
        try: 
            mip = mesh(x_eval, z)
            Ez_vals[i] = Ez_field(mip)
        except Exception:
            Ez_vals[i] = 0.0 + 0.0j

    # Compute spatial FFT
    n_fft = num_points * pad_factor
    Ez_fft = fftshift(fft(Ez_vals, n=n_fft))
    E_fft_norm = Ez_fft / num_points
    
    # Map spatial frequency to n_para 
    fz = fftshift(fftfreq(n_fft, d=dz)) # fz = cycle per meter in the z direction
    n_para_array = (2.0 * np.pi *fz)/cfg.WAVE['k0']

    # compute power spectrum:
    power_spectrum = np.abs(Ez_fft)**2
    power_spectrum /=np.max(power_spectrum) # normalize to 1.0

    plt.figure(figsize=(10, 6))
    plt.plot(n_para_array, power_spectrum, color='crimson', lw=2)
    plt.xlim(-10, 10)
    plt.ylim(1e-4, 1.1)

    # plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
     
    injected_n_para = np.array([2, -3])
    for n_para_value in injected_n_para:
        plt.axvline(x=n_para_value, color='Royalblue', linestyle=':', lw=2, label=r'$n_{//} = $'+f'{n_para_value}')
    
    plt.xlabel(r'Parallel Refractive Index ($n_\parallel$)', fontsize=14)
    plt.ylabel('Normalized Spectral Power (a.u.)', fontsize=14)
    plt.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, right=True, left=True)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return n_para_array, power_spectrum