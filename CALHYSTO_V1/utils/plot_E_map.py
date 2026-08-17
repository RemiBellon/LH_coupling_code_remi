import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config.schema import SimulationConfig
from utils.antenna_desc_2D import AntennaGrill
from utils.antenna_desc_2D import build_grill_from_config

def plot_2d_wave_map(config: SimulationConfig, h5_filepath: str, component: str = 'Ez', value_type: str = 'real'):
    """
    Generates an expert-level, publication-ready 2D wave map.
    """
    print(f"--- Plotting 2D Map from {h5_filepath} ---")

    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        E_comp = h5f[component][:]
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lz_plasma = h5f.attrs['Lz_plasma']
        Lz_wall = h5f.attrs['Lz_wall']

    # Physical Formatting
    if value_type == 'real':
        plot_data = E_comp.real
        cmap = 'RdBu_r'  # Diverging colormap for oscillating real waves
        vmax = np.percentile(np.abs(plot_data), 99.5)
        vmin = -vmax
    else:
        plot_data = np.abs(E_comp)
        cmap = 'inferno' # Sequential colormap for magnitude
        vmax = np.percentile(plot_data, 99.5)
        vmin = 0.0

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [Z.min(), Z.max(), X.min(), X.max()]

    c = ax.imshow(plot_data, origin='lower', extent=extent, cmap=cmap,
                  vmin=vmin, vmax=vmax, aspect='equal', interpolation='bicubic')

    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Electric Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=14)

    # Overwrite Antenna Geometry using exact Pydantic schema
    ant_cfg = config.geometry.antenna
    if ant_cfg:
        grill, instructions = build_grill_from_config(config.geometry.antenna, config.geometry.domain)
        max_depth = ant_cfg.dimensions.Lx_wg_active

        for inst in instructions:
            z_s = inst['z_start']
            z_w = inst['width']
            depth = inst.get('depth', 0.0)

            if inst['type'] in ['metal', 'metal_gap']:
                rect = patches.Rectangle((z_s, -max_depth), z_w, max_depth,
                                         linewidth=1.5, edgecolor='black', facecolor='silver', hatch='////')
                ax.add_patch(rect)
            elif inst['type'] == 'wg_passive':
                metal_h = max_depth - depth
                rect = patches.Rectangle((z_s, -max_depth), z_w, metal_h,
                                         linewidth=1.5, edgecolor='black', facecolor='silver', hatch='////')
                ax.add_patch(rect)
                # Short circuit boundary
                ax.plot([z_s, z_s + z_w], [-depth, -depth], color='red', lw=3)

        # Draw Side Walls
        ax.add_patch(patches.Rectangle((0.0, -max_depth), Lz_wall, max_depth,
                     linewidth=1.5, edgecolor='black', facecolor='silver', hatch='////'))
        ax.add_patch(patches.Rectangle((Lz_plasma - Lz_wall, -max_depth), Lz_wall, max_depth,
                     linewidth=1.5, edgecolor='black', facecolor='silver', hatch='////'))

    # Draw Plasma and PML Boundaries
    ax.axhline(y=0.0, color='black', lw=2, label='Antenna Mouth')
    ax.axhline(y=Lx_plasma, color='yellow', linestyle='--', lw=2, label='Radial PML')
    ax.axvline(x=0.0, color='lime', linestyle=':', lw=2, label='Toroidal PML (Bottom)')
    ax.axvline(x=Lz_plasma, color='lime', linestyle=':', lw=2, label='Toroidal PML (Top)')

    ax.set_xlabel("Toroidal Position $z$ [m]", fontsize=14)
    ax.set_ylabel("Radial Position $x$ [m]", fontsize=14)
    # ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Strictly define plotting limits so empty metal space isn't visually dominant
    ax.set_ylim(-ant_cfg.dimensions.Lx_wg_active * 1.1, Lx_plasma + config.geometry.domain.Lx_pml)

    plt.tight_layout()
    plt.show()