import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from postprocessing.plot_style import apply_style
from postprocessing.data_reader import SimulationData

def plot_2D_wave_map(data: SimulationData, component='Ez', value_type='real', save_dir=None):
    """
    Renders the 2D spatial wave fields using Matplotlib and overlays the antenna geometry.
    To see the absolute uniform power, set component='E_norm'.
    """
    if data.map_2d is None:
        print("[!] No 2D map data found in HDF5 file.")
        return
        
    apply_style()
    print(f"--- Plotting 2D Map for {component} ---")
    
    X = data.map_2d["X"]
    Z = data.map_2d["Z"]
    E_comp = data.map_2d[component]
    
    # Adaptive colormap based on physical constraints
    if component == 'E_norm':
        plot_data = E_comp
        cmap = 'magma'
        vmax = np.percentile(plot_data, 90.5)
        vmin = 0.0
        label_str = "Wave Field Norm $|E|$ [V/m]"
    else:
        if value_type == 'real':
            plot_data = np.real(E_comp)
            cmap = 'coolwarm'
            vmax = np.percentile(np.abs(plot_data), 90.5) 
            vmin = -vmax
            label_str = f"Wave Field Real({component}) [V/m]"
        else:
            plot_data = np.abs(E_comp)
            cmap = 'magma'
            vmax = np.percentile(plot_data, 90.5) 
            vmin = 0.0
            label_str = f"Wave Field $|{component}|$ [V/m]"

    fig, ax = plt.subplots(figsize=(14, 8))
    
    extent = [Z.min(), Z.max(), X.min(), X.max()]
    c = ax.imshow(plot_data, origin='lower', extent=extent, cmap=cmap, 
                  vmin=vmin, vmax=vmax, aspect='auto', interpolation='bicubic')
                  
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(label_str, fontsize=16)

    max_depth = max(p.length for p in data.ports) if data.ports else 0.05
    if data.ports:
        current_z = 0.0
        for port in data.ports:
            if port.z_start > current_z + 1e-6:
                septa_width = port.z_start - current_z
                rect = patches.Rectangle((current_z, -max_depth), septa_width, max_depth, 
                                         linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///', zorder=10)
                ax.add_patch(rect)
                
            width = port.z_end - port.z_start
            
            if port.type == 'passive':
                depth = port.length
                metal_height = max_depth - depth
                rect = patches.Rectangle((port.z_start, -max_depth), width, metal_height, 
                                         linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///', zorder=10)
                ax.add_patch(rect)
                
                # Make the passive cavity semi-transparent to see the field
                rect_void = patches.Rectangle((port.z_start, -depth), width, depth, 
                                         linewidth=1, edgecolor='black', facecolor='dimgrey', alpha=0.1, zorder=10)
                ax.add_patch(rect_void)
                
                ax.plot([port.z_start, port.z_end], [-depth, -depth], color='red', lw=3, zorder=11)
            
        Z_max = Z.max()
        if current_z < Z_max:
            right_wall_width = Z_max - current_z
            rect = patches.Rectangle((current_z, -max_depth), right_wall_width, max_depth, 
                                     linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///', zorder=10)
            ax.add_patch(rect)

        # ax.axhline(y=0.0, color='black', lw=1.5, zorder=12)
        ax.set_ylim(-max_depth * 1.05, X.max())

    ax.set_xlabel("Toroidal Position $z$ [m]")
    ax.set_ylabel("Radial Depth $x$ [m]")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{component}_2D_map.pdf"), dpi=300)
    plt.show()