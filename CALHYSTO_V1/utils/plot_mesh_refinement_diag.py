import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from ngsolve import Mesh, specialcf
from config.schema import SimulationConfig
from utils.antenna_desc_2D import AntennaGrill
from utils.antenna_desc_2D import build_grill_from_config

def plot_true_mesh_size(config: SimulationConfig, mesh: Mesh, title: str):
    """
    Extracts and visually maps the exact element sizes of the current NGSolve mesh with metal shading.
    """
    print(f"--- Generating Mesh Size Diagnostic: {title} ---")
    actual_h_cf = specialcf.mesh_size

    ant = config.geometry.antenna
    dom = config.geometry.domain
    Lx_wg = ant.dimensions.Lx_wg_active if ant else 0.0

    x_min, x_max = -Lx_wg, dom.Lx_plasma
    z_min, z_max = -dom.Lz_pml, dom.Lz_plasma + dom.Lz_pml

    nx, nz = 300, 300
    x_vals = np.linspace(x_min, x_max, nx)
    z_vals = np.linspace(z_min, z_max, nz)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    H_actual = np.zeros_like(X)

    for i in range(nx):
        for j in range(nz):
            try:
                mip = mesh(X[i, j], Z[i, j])
                H_actual[i, j] = actual_h_cf(mip)
            except Exception:
                H_actual[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [Z.min(), Z.max(), X.min(), X.max()]

    # EXPERT FIX: Rigorous LogNorm for structural clarity
    c = ax.imshow(H_actual, origin='lower', extent=extent, cmap='viridis',
                  norm=LogNorm(vmin=1e-4, vmax=0.05), aspect='equal', interpolation='nearest')

    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Actual Element Size $h$ (m) [Log Scale]', fontsize=14)

    # Geometry Overlay
    if ant:
        grill, instructions = build_grill_from_config(config.geometry.antenna, config.geometry.domain)
        is_pam = (ant.topology == "PAM")
        for i in range(ant.grill_arrangement.num_modules):
            grill.add_module(
                num_active=ant.grill_arrangement.active_waveguides_per_module_row[i],
                is_PAM=is_pam, delta_phi_deg=0.0, power_module_W=0.0)
        max_depth = Lx_wg
        for inst in instructions:
            z_s, z_w = inst['z_start'], inst['width']
            depth = inst.get('depth', 0.0)

            if inst['type'] in ['metal', 'metal_gap']:
                ax.add_patch(patches.Rectangle((z_s, -max_depth), z_w, max_depth, linewidth=1, edgecolor='black', facecolor='silver', hatch='////'))
            elif inst['type'] == 'wg_passive':
                metal_h = max_depth - depth
                ax.add_patch(patches.Rectangle((z_s, -max_depth), z_w, metal_h, linewidth=1, edgecolor='black', facecolor='silver', hatch='////'))
                ax.plot([z_s, z_s + z_w], [-depth, -depth], color='red', lw=3)

        ax.add_patch(patches.Rectangle((0.0, -max_depth), dom.Lz_wall, max_depth, linewidth=1, edgecolor='black', facecolor='silver', hatch='////'))
        ax.add_patch(patches.Rectangle((dom.Lz_plasma - dom.Lz_wall, -max_depth), dom.Lz_wall, max_depth, linewidth=1, edgecolor='black', facecolor='silver', hatch='////'))

    ax.axhline(0, color='cyan', linestyle='--', linewidth=1.5, label='Antenna Aperture')
    ax.set_ylim(-Lx_wg * 1.05, dom.Lx_plasma)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Radial Position $x$ [m]', fontsize=14)
    ax.set_xlabel('Toroidal Position $z$ [m]', fontsize=14)
    plt.tight_layout()
    plt.show()