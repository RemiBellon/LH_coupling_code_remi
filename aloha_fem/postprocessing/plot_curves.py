import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from postprocessing.plot_style import apply_style
from postprocessing.data_reader import WaveguidePort, SimulationData
from config.schema import SimulationConfig
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

def plot_blueprint_from_yaml(yaml_filepath: str):
    """
    Reads the input.yaml file using the strict Pydantic schema and generates
    a mock SimulationData object to visualize the antenna geometry before running.
    """
    from config.schema import SimulationConfig
    config = SimulationConfig.from_yaml(yaml_filepath)

    if config.geometry.antenna is None:
        print("[!] No explicit antenna geometry found in YAML (Likely 1D mode).")
        return

    antenna_cfg = config.geometry.antenna
    arrangement = antenna_cfg.grill_arrangement
    dimensions = antenna_cfg.dimensions
    topology = antenna_cfg.topology

    mock_data = SimulationData.__new__(SimulationData)
    mock_data.ports = []

    current_z = 0.0
    port_index = 1
    
    # Force a Strict Colormap Normalization [0, 360]
    phase_norm = mcolors.Normalize(vmin=0.0, vmax=360.0)
    
    # 1. ADD LEFT EDGE PASSIVE WAVEGUIDE
    mock_data.ports.append(WaveguidePort(
        index=port_index, type="passive", z_start=current_z, 
        z_end=current_z + dimensions.wg_width, length=dimensions.wg_length_passive,
        phase_deg=0.0, power_reflectivity=1.0
    ))
    current_z += dimensions.wg_width + dimensions.septa_width
    port_index += 1

    # 2. MAIN MODULE LOOP
    for i, num_active_wgs in enumerate(arrangement.active_waveguides_per_module):
        delta_phi = arrangement.phase_shift_per_module_deg[i]
        
        for j in range(num_active_wgs):
            accumulated_phase = (j * delta_phi) % 360.0
            
            # Active waveguide
            mock_data.ports.append(WaveguidePort(
                index=port_index, type="active", z_start=current_z, 
                z_end=current_z + dimensions.wg_width, length=dimensions.wg_length_active,
                phase_deg=accumulated_phase, power_reflectivity=0.0
            ))
            current_z += dimensions.wg_width + dimensions.septa_width
            port_index += 1
            
            # Intra-module PAM passive
            if topology == "PAM" and j < num_active_wgs - 1:
                mock_data.ports.append(WaveguidePort(
                    index=port_index, type="passive", z_start=current_z, 
                    z_end=current_z + dimensions.wg_width, length=dimensions.wg_length_passive,
                    phase_deg=0.0, power_reflectivity=1.0
                ))
                current_z += dimensions.wg_width + dimensions.septa_width
                port_index += 1

        # Inter-module passive separator
        if i < arrangement.num_modules - 1:
            mock_data.ports.append(WaveguidePort(
                index=port_index, type="passive", z_start=current_z, 
                z_end=current_z + dimensions.wg_width, length=dimensions.wg_length_passive,
                phase_deg=0.0, power_reflectivity=1.0
            ))
            current_z += dimensions.wg_width + dimensions.septa_width
            port_index += 1

    # 3. ADD RIGHT EDGE PASSIVE WAVEGUIDE
    mock_data.ports.append(WaveguidePort(
        index=port_index, type="passive", z_start=current_z, 
        z_end=current_z + dimensions.wg_width, length=dimensions.wg_length_passive,
        phase_deg=0.0, power_reflectivity=1.0
    ))

    print(f"--- Pre-Run Blueprint Verification ({topology}) ---")
    # Assuming plot_antenna_blueprint is imported and available
    plot_antenna_blueprint(mock_data, yaml_filepath)

def plot_antenna_blueprint(data: SimulationData, yaml_filepath, save_dir=None):
    """
    Reads the geometric instructions from the HDF5 object and draws the physical antenna face.
    Inspired by the ALOHA blueprint structure.
    """
    from config.schema import SimulationConfig
    config = SimulationConfig.from_yaml(yaml_filepath)

    if config.geometry.antenna is None:
        print("[!] No explicit antenna geometry found in YAML (Likely 1D mode).")
        return

    antenna_cfg = config.geometry.antenna
    dimensions = antenna_cfg.dimensions
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    if not data.ports:
        print("[!] No antenna port data found to plot blueprint.")
        return

    max_z = data.ports[-1].z_end
    max_depth = max(p.length for p in data.ports)
    if max_depth <= 0: max_depth = 0.05

    current_z = 0.0 # To track and draw metal septa between waveguides
    rounding = dimensions.corner_radius
    for port in data.ports:
        # Draw preceding metal septum if there is a gap
        if port.z_start > current_z + 1e-6:
            box_style = f"round,pad=0,rounding_size={rounding}"
            septa_width = port.z_start - current_z
            septa = FancyBboxPatch((current_z, -max_depth), septa_width, max_depth,
                                   boxstyle=box_style, linewidth=.5, edgecolor='black', 
                                   facecolor='dimgrey', hatch='///')
            ax.add_patch(septa)

        width = port.z_end - port.z_start
        depth = port.length

        if port.type == 'passive':
            color = 'lightgrey'
            label = 'P'
        else:
            # Map electrical phase to HSV colormap
            norm_phase = (port.phase_deg) / 360.0
            color = plt.cm.hsv(norm_phase)
            label = f'{port.phase_deg:.0f}°'

        # Draw Waveguide Void
        rect = patches.Rectangle((port.z_start, -depth), width, depth,
                                 linewidth=.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)

        # Draw Short-Circuit for Passives
        if port.type == 'passive':
            ax.plot([port.z_start, port.z_end], [-depth, -depth], color='red', lw=1, zorder=5)
            # Draw metal block behind the short-circuit
            metal_height = max_depth - depth
            rect_back = patches.Rectangle((port.z_start, -max_depth), width, metal_height,
                                     linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///')
            ax.add_patch(rect_back)

        # Add Label
        ax.text(port.z_start + width/2, -depth/2, label, color='black',
                ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

        current_z = port.z_end

    # Global Formatting
    # ax.axhline(0, color='black', lw=2)
    ax.set_xlim(-0.005, max_z + 0.005)
    ax.set_ylim(-max_depth * 1.1, max_depth * 0.1)

    ax.tick_params(axis='y', which='both', left=True, labelleft=True)
    ax.set_yticks(np.linspace(0, -dimensions.wg_length_active, 5))
    ax.grid(None)
    ax.set_ylabel("Radial Depth $x$ [m]")
    ax.set_xlabel("Toroidal Position $z$ [m]")

    # Phase Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.08, pad=0.25, aspect=50)
    # cbar.set_label('Electrical Phase [Degrees]')

    if save_dir:
        plt.savefig(f"{save_dir}/Antenna_Blueprint.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_s_parameters(data: SimulationData, save_dir=None):
    """
    Plots the Power Reflectivity (|Gamma|^2) for each waveguide port.
    Visually distinguishes between active (driven) and passive (short-circuited) ports.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if not data.ports:
        print("[!] No S-Parameter data found.")
        return

    z_centers = [(p.z_start + p.z_end) / 2.0 for p in data.ports]
    reflectivities = [p.power_reflectivity for p in data.ports]
    types = [p.type for p in data.ports]

    # Separate data for styling
    z_active = [z for z, t in zip(z_centers, types) if t == "active"]
    ref_active = [r for r, t in zip(reflectivities, types) if t == "active"]

    z_passive = [z for z, t in zip(z_centers, types) if t == "passive"]
    ref_passive = [r for r, t in zip(reflectivities, types) if t == "passive"]

    # Stem plot for clear discrete waveguide visualization
    if z_active:
        markerline, stemline, baseline = ax.stem(z_active, ref_active, label="Active Ports", basefmt="k-")
        plt.setp(markerline, marker='o', markersize=8, color="royalblue", markeredgecolor="black")
        plt.setp(stemline, color="royalblue", linewidth=2)

    if z_passive:
        markerline, stemline, baseline = ax.stem(z_passive, ref_passive, label="Passive Ports", basefmt="k-")
        plt.setp(markerline, marker='s', markersize=8, color="darkgrey", markeredgecolor="black")
        plt.setp(stemline, color="darkgrey", linewidth=2, linestyle="--")

    ax.set_ylim(0, 1.1)
    ax.set_xlim(min(z_centers) - 0.02, max(z_centers) + 0.02)

    ax.set_xlabel("Waveguide Center Toroidal Position $z$ [m]")
    ax.set_ylabel(r"Power Reflectivity $|\Gamma|^2$")
    ax.set_title("Antenna S-Parameters")
    ax.legend(loc="upper right")

    if save_dir:
        plt.savefig(f"{save_dir}/S_Parameters.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_power_spectrum(data: SimulationData, save_dir=None):
    """Plots the rigorously normalized power spectrum vs parallel refractive index."""
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Using the intuitive dot notation
    ax.plot(data.spectrum.n_para, data.spectrum.dP_dn_para, color="crimson", lw=2)

    # Dynamic limits based on the requested n_para
    n_target = data.meta.n_para_req
    ax.set_xlim(-abs(n_target) * 3, abs(n_target) * 3)
    ax.set_ylim(1e-4, np.max(data.spectrum.dP_dn_para) * 1.1)

    # Overlay the target injection phasing
    ax.axvline(x=n_target, color="royalblue", linestyle=":", lw=2, label=rf"Target $n_\parallel = {n_target}$")
    ax.axvline(x=-n_target, color="royalblue", linestyle=":", lw=2)

    ax.set_xlabel(r"Parallel Refractive Index $n_\parallel$")
    ax.set_ylabel("Normalized Spectral Power [W]")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "n_para_spectrum.pdf"), dpi=300)
    plt.show()

def plot_aperture_field_amplitude(data: SimulationData, component="Ez", save_dir=None):
    """Plots the electric field amplitude along the toroidal axis."""
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    z_coords = data.fields.z_coords

    if component == "Ez":
        field_vals = np.abs(data.fields.Ez)
        c = "black"
    elif component == "E_norm":
        field_vals = data.fields.E_norm
        c = "forestgreen"
    else:
        raise ValueError("Unsupported component.")

    ax.plot(z_coords, field_vals, color=c, lw=2.5, label=f"FEM — $|{component}|$")

    # Ignore metal edge singularities for the y-axis scaling
    y_max = np.percentile(field_vals, 98) * 1.2
    ax.set_ylim(-y_max * 0.05, y_max)

    ax.set_xlabel("Toroidal Position $z$ [m]")
    ax.set_ylabel(f"Electric Field {component} [V/m]")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"aperture_field_{component}.pdf"), dpi=300)
    plt.show()