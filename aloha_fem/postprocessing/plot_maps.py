import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from physics.waveguide import WaveguidePhysics

def plot_2d_electric_field(mesh, E_field_gf, config, filename="E_field_2D.pdf", resolution=1000):
    """
    Plots the normalized 2D electric field norm |E|, overlaying exact active/passive 
    waveguide geometry and hatched metal regions.
    """
    print("\n--- Generating 2D Electric Field Map ---")
    
    # 1. Extract dimensions
    Lx_plasma = config.geometry.domain.Lx_plasma
    Lz_wall = config.geometry.domain.Lz_wall
    Lz_source = config.geometry.domain.Lz_plasma
    z_max = Lz_wall + Lz_source + Lz_wall

    wg_physics = WaveguidePhysics(config)
    wg_sequence = wg_physics.wg_sequence
    max_wg_length = wg_physics.max_wg_length

    # 2. Setup grid evaluation
    x_coords = np.linspace(-max_wg_length, Lx_plasma, resolution)
    z_coords = np.linspace(0.0, z_max, resolution)
    X, Z = np.meshgrid(x_coords, z_coords)
    E_norm_vals = np.zeros_like(X, dtype=float)

    # 3. Safely evaluate the field over the mesh
    E_plane, E_outplane = E_field_gf.components[0], E_field_gf.components[1]
    Ex, Ez = E_plane[0], E_plane[1]
    E_norm_cf = np.sqrt(Ex*np.conj(Ex) + E_outplane*np.conj(E_outplane) + Ez*np.conj(Ez))
    
    for i in range(resolution):
        for j in range(resolution):
            try:
                mip = mesh(float(X[i, j]), float(Z[i, j]))
                E_norm_vals[i, j] = E_norm_cf(mip).real
            except Exception:
                E_norm_vals[i, j] = np.nan

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use pcolormesh for proper coordinate mapping. Note that X is the vertical axis in our arrays
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='white')
    c = ax.pcolormesh(X, Z, E_norm_vals, shading='auto', cmap=cmap, vmin=0)
    cbar = fig.colorbar(c, ax=ax, label=r'Electric Field Norm $|E|$ [V/m]')

    # 5. Overlay Antenna Geometry
    if wg_sequence:
        # A. Solid metal walls (Left and Right Flanks)
        left_flank = patches.Rectangle((-max_wg_length, 0), max_wg_length, Lz_wall, 
                                       linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='//', zorder=10)
        ax.add_patch(left_flank)
        
        right_flank = patches.Rectangle((-max_wg_length, Lz_wall + Lz_source), max_wg_length, Lz_wall, 
                                        linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='//', zorder=10)
        ax.add_patch(right_flank)

        # B. Waveguides & Septa
        for idx, wg in enumerate(wg_sequence):
            w_wg = wg["z_end"] - wg["z_start"]
            
            # Highlight active waveguides (Empty outlines)
            if wg["type"] == "active":
                active_rect = patches.Rectangle((-wg["length"], wg["z_start"]), wg["length"], w_wg, 
                                                linewidth=1.5, edgecolor='red', facecolor='none', zorder=11)
                ax.add_patch(active_rect)
            
            # Semi-transparent overlay for passive waveguides + blocked back cavity
            elif wg["type"] == "passive":
                # The blocked metal behind the short circuit
                metal_depth = max_wg_length - wg["length"]
                blocked_rect = patches.Rectangle((-max_wg_length, wg["z_start"]), metal_depth, w_wg, 
                                                 linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='//', zorder=10)
                ax.add_patch(blocked_rect)
                
                # The vacuum cavity itself (semi-transparent so the field is visible)
                passive_rect = patches.Rectangle((-wg["length"], wg["z_start"]), wg["length"], w_wg, 
                                                 linewidth=1.5, edgecolor='black', facecolor='white', alpha=0.3, zorder=11)
                ax.add_patch(passive_rect)

            # Solid metal septa separating waveguides
            if idx < len(wg_sequence) - 1:
                z_sep_start = wg["z_end"]
                z_sep_end = wg_sequence[idx+1]["z_start"]
                septum = patches.Rectangle((-max_wg_length, z_sep_start), max_wg_length, z_sep_end - z_sep_start,
                                           linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='//', zorder=10)
                ax.add_patch(septum)

    # 6. Formatting
    ax.axvline(x=0.0, color='black', linestyle=':', lw=2, zorder=12, label='Plasma Interface')
    ax.set_xlabel('Radial Position $x$ [m]', fontsize=14)
    ax.set_ylabel('Toroidal Position $z$ [m]', fontsize=14)
    ax.set_title('Normalized 2D Electric Field Map', fontsize=16)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  -> Plot saved to {filename}")