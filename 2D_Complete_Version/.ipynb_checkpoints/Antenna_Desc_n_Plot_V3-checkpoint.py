import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# 1. Define a strict Data Structure for a single Waveguide
@dataclass
class Waveguide:
    width: float
    complex_E: complex
    is_active: bool

class AntennaGrill:
    def __init__(self, b_active: float, d_septa: float, d_gap: float):
        """
        Initializes the global rules for the antenna structure.
        b_active: Width of the active waveguides (meters)
        d_septa: Width of the thin metal septum INSIDE a module (meters)
        d_gap: Width of the thick metal wall BETWEEN modules (meters)
        """
        self.b_active = b_active
        self.d_septa = d_septa
        self.d_gap = d_gap
        
        # A list containing sub-lists (modules) of Waveguide objects
        self.modules: List[List[Waveguide]] = [] 

    def add_module(self, num_active: int, delta_phi_deg: float, amplitude: float = 1.0, 
                   initial_phase_deg: float = 0.0, interleave_passive: bool = False, 
                   b_passive: float = None):
        """
        Generates a module. If interleave_passive=True, it builds a PAM.
        If interleave_passive=False, it builds a FAM.
        """
        module_wgs: List[Waveguide] = []
        current_phase_rad = np.radians(initial_phase_deg)
        delta_phi_rad = np.radians(delta_phi_deg)
        
        # Default passive width to active width if not specified
        if b_passive is None:
            b_passive = self.b_active

        for i in range(num_active):
            # 1. Create and append the ACTIVE waveguide
            E_val = amplitude * np.exp(1j * current_phase_rad)
            module_wgs.append(Waveguide(width=self.b_active, complex_E=E_val, is_active=True))
            current_phase_rad += delta_phi_rad
            
            # 2. Create and append the PASSIVE waveguide (if PAM mode is ON)
            # Typically, we insert a passive wg after every active wg, except maybe the very last one.
            # Let's insert it after every active one to make a true A-P-A-P pattern.
            if interleave_passive and i < num_active - 1:
                module_wgs.append(Waveguide(width=b_passive, complex_E=0.0+0j, is_active=False))
                
        self.modules.append(module_wgs)

    def generate_mesh_instructions(self, z_start_position: float = 0.0) -> List[Dict]:
        """
        Translates the objects into literal geometric boundaries for NGSolve.
        """
        instructions = []
        current_z = z_start_position
        global_wg_count = 0
        global_septa_count = 0
        
        for mod_idx, module_wgs in enumerate(self.modules):
            num_wgs_in_module = len(module_wgs)
            
            for wg_idx, wg in enumerate(module_wgs):
                # 1. Draw the Waveguide (Active or Passive)
                tag = 'wg_active' if wg.is_active else 'wg_passive'
                instructions.append({
                    'name': f'{tag}_{global_wg_count}',
                    'width': wg.width,
                    'z_start': current_z,
                    'z_end': current_z + wg.width,
                    'is_metal': False,
                    'complex_E_field': wg.complex_E
                })
                current_z += wg.width
                global_wg_count += 1
                
                # 2. Draw the Thin Septum (If NOT the last waveguide in this module)
                if wg_idx < num_wgs_in_module - 1:
                    instructions.append({
                        'name': f'septa_{global_septa_count}',
                        'width': self.d_septa,
                        'z_start': current_z,
                        'z_end': current_z + self.d_septa,
                        'is_metal': True,
                        'complex_E_field': 0.0 + 0j
                    })
                    current_z += self.d_septa
                    global_septa_count += 1
            
            # 3. Draw the Thick Inter-Module Gap (If NOT the very last module)
            if mod_idx < len(self.modules) - 1:
                instructions.append({
                    'name': f'mod_gap_{mod_idx}',
                    'width': self.d_gap,
                    'z_start': current_z,
                    'z_end': current_z + self.d_gap,
                    'is_metal': True,
                    'complex_E_field': 0.0 + 0j
                })
                current_z += self.d_gap
                
        return instructions
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_antenna_blueprint(instructions):
    """
    Reads the NGSolve geometric instructions and draws the physical antenna face.
    Waveguides are colored by their injected phase.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    max_z = instructions[-1]['z_end']
    
    for inst in instructions:
        z_start = inst['z_start']
        width = inst['width']
        
        if inst['is_metal']:
            # Draw Metal Septa / Gaps (Grey)
            rect = patches.Rectangle((z_start, 0), width, 1, linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///')
            ax.add_patch(rect)
            ax.text(z_start + width/2, 0.5, 'Metal', color='white', ha='center', va='center', fontsize=9, rotation=90)
        else:
            # Draw Waveguides
            E_val = inst['complex_E_field']
            amplitude = np.abs(E_val)
            
            if amplitude < 1e-6:
                # Passive Waveguide
                color = 'lightgrey'
                label = 'Passive\nWG'
            else:
                # Active Waveguide: Calculate phase in degrees for coloring
                phase_deg = np.degrees(np.angle(E_val))
                # Normalize phase [-180, 180] to [0, 1] for a colormap
                norm_phase = (phase_deg + 180) / 360.0
                color = plt.cm.hsv(norm_phase) # HSV colormap is perfect for circular phase
                label = f'Active\n{phase_deg:.0f}°'
                
            rect = patches.Rectangle((z_start, 0), width, 1, linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(z_start + width/2, 0.5, label, color='black', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.005, max_z + 0.005)
    ax.set_ylim(0, 1)
    ax.set_yticks([]) # Hide Y axis
    ax.set_xlabel("Toroidal Position $z$ [meters]", fontsize=12)
    ax.set_title("Antenna Geometric & Phase Blueprint", fontsize=14, fontweight='bold')
    
    # Add a custom Phase Legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=-180, vmax=180))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.3, aspect=50)
    cbar.set_label('Electrical Phase [Degrees]')
    
    plt.tight_layout()
    plt.show()

# --- HOW TO TEST IT ---
# grill = AntennaGrill(b_active=0.009, d_septa=0.002, d_gap=0.010)
# grill.add_module(num_active=6, delta_phi_deg=90.0, interleave_passive=True)
# instructions = grill.generate_mesh_instructions()
# plot_antenna_blueprint(instructions)