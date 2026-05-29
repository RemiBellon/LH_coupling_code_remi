import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from scipy.fft import fft, fftfreq

def plot_2D_wave_map(h5_filepath, figure_save_dir, mode, component='Ez', value_type='real', plot_poynting=True):
    print(f"--- Plotting 2D Map from {h5_filepath} ---")
    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        E_comp = h5f[component][:] # Automatically grabs Ex, Ey, or Ez
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lz_plasma = h5f.attrs['Lz_plasma']
        Sx, Sz = h5f['Sx'][:], h5f['Sz'][:]
        k_para, k_perp_p = h5f.attrs['k_para'], h5f.attrs['k_perp_p']

    plot_data = E_comp.real if value_type == 'real' else np.abs(E_comp)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = 'magma' if value_type == 'abs' else 'coolwarm'
    vmax = np.max(plot_data)
    vmin = 0.0 if value_type == 'abs' else -vmax
    
    c = ax.pcolormesh(Z, X, plot_data, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(c, ax=ax)
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