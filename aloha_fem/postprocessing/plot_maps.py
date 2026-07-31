import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from postprocessing.plot_style import apply_style
from postprocessing.data_reader import SimulationData

def plot_2D_wave_map(data: SimulationData, component='Ez', value_type='real', save_dir=None):
    """
    Renders the 2D spatial wave fields using Matplotlib and overlays the antenna geometry.
    """
    if data.map_2d is None:
        print("[!] No 2D map data found in HDF5 file.")
        return
        
    apply_style()
    print(f"--- Plotting 2D Map for {component} ---")
    
    # 1. Extract Grid and Field
    X = data.map_2d["X"]
    Z = data.map_2d["Z"]
    E_comp = data.map_2d[component]
    
    if value_type == 'real':
        plot_data = np.real(E_comp)
        cmap = 'coolwarm'
    else:
        plot_data = np.abs(E_comp)
        cmap = 'magma'
        
    # Prevent metal corner singularities from ruining the color scale
    vmax = np.percentile(plot_data, 99.5) 
    vmin = 0.0 if value_type == 'abs' else -vmax

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 2. Render Field
    extent = [Z.min(), Z.max(), X.min(), X.max()]
    c = ax.imshow(plot_data, origin='lower', extent=extent, cmap=cmap, 
                  vmin=vmin, vmax=vmax, aspect='auto', interpolation='bicubic')
                  
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(f"Wave Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=16)

    # 3. Overlay Antenna Grill Geometry
    max_depth = max(p.length for p in data.ports) if data.ports else 0.05
    if data.ports:
        current_z = 0.0
        for port in data.ports:
            # Draw metal septum if gap exists
            if port.z_start > current_z + 1e-6:
                septa_width = port.z_start - current_z
                rect = patches.Rectangle((current_z, -max_depth), septa_width, max_depth, 
                                         linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///', zorder=10)
                ax.add_patch(rect)
                
            width = port.z_end - port.z_start
            
            # Block off passive short-circuits
            if port.type == 'passive':
                depth = port.length
                metal_height = max_depth - depth
                rect = patches.Rectangle((port.z_start, -max_depth), width, metal_height, 
                                         linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///', zorder=10)
                ax.add_patch(rect)
                # Red short circuit line
                ax.plot([port.z_start, port.z_end], [-depth, -depth], color='red', lw=3, zorder=11)
                
            current_z = port.z_end

        # Base boundaries
        ax.axhline(y=0.0, color='black', lw=1.5, zorder=12)
        ax.set_ylim(-max_depth * 1.05, X.max())

    # 4. Formatting
    ax.set_xlabel("Toroidal Position $z$ [m]")
    ax.set_ylabel("Radial Depth $x$ [m]")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{component}_2D_map.pdf"), dpi=300)
    plt.show()