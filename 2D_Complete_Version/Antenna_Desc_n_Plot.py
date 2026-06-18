import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Waveguide:
    width: float
    complex_E: complex
    is_active: bool
    is_metal: bool = False

class AntennaGrill:
    def __init__(self, b_active: float, d_septa: float, d_gap: float, b_passive: float = None):
        """
        b_active: Largeur des guides actifs (m)
        d_septa: Épaisseur des cloisons métalliques internes (m)
        d_gap: Épaisseur du mur métallique entre les modules (m)
        b_passive: Largeur des guides passifs (par défaut = b_active)
        """
        self.b_active = b_active
        self.d_septa = d_septa
        self.d_gap = d_gap
        self.b_passive = b_passive if b_passive is not None else b_active
        self.modules: List[List[Waveguide]] = [] 

    def add_module(self, num_active: int, delta_phi_deg: float, amplitude: float = 10.0, 
                   initial_phase_deg: float = 0.0, is_PAM: bool = False):
        """
        Génère un module FAM (is_PAM=False) ou PAM (is_PAM=True).
        """
        module_wgs: List[Waveguide] = []
        current_phase_rad = np.radians(initial_phase_deg)
        delta_phi_rad = np.radians(delta_phi_deg)
        
        # Un module PAM de WEST commence par un guide passif de bord
        if is_PAM:
            module_wgs.append(Waveguide(width=self.b_passive, complex_E=0j, is_active=False))
            module_wgs.append(Waveguide(width=self.d_septa, complex_E=0j, is_active=False, is_metal=True))

        for i in range(num_active):
            # Guide Actif
            E_val = amplitude * np.exp(1j * current_phase_rad)
            module_wgs.append(Waveguide(width=self.b_active, complex_E=E_val, is_active=True))
            current_phase_rad += delta_phi_rad
            
            # Dans un FAM, on met juste un septa entre les actifs.
            # Dans un PAM, on met un septa, un passif, puis un septa.
            if i < num_active - 1:
                module_wgs.append(Waveguide(width=self.d_septa, complex_E=0j, is_active=False, is_metal=True))
                if is_PAM:
                    module_wgs.append(Waveguide(width=self.b_passive, complex_E=0j, is_active=False))
                    module_wgs.append(Waveguide(width=self.d_septa, complex_E=0j, is_active=False, is_metal=True))

        # Un module PAM de WEST se termine par un guide passif
        if is_PAM:
            module_wgs.append(Waveguide(width=self.d_septa, complex_E=0j, is_active=False, is_metal=True))
            module_wgs.append(Waveguide(width=self.b_passive, complex_E=0j, is_active=False))
                
        self.modules.append(module_wgs)

    def generate_mesh_instructions(self, z_start_position: float = 0.0) -> List[Dict]:
        """
        Traduit l'architecture en coordonnées z absolues pour construire la fonction par morceaux dans NGSolve.
        """
        instructions = []
        current_z = z_start_position
        
        for mod_idx, module_wgs in enumerate(self.modules):
            for wg in module_wgs:
                tag = 'metal' if wg.is_metal else ('wg_active' if wg.is_active else 'wg_passive')
                instructions.append({
                    'type': tag,
                    'width': wg.width,
                    'z_start': current_z,
                    'z_end': current_z + wg.width,
                    'complex_E_field': wg.complex_E
                })
                current_z += wg.width
            
            # Espace inter-module (sauf après le dernier module)
            if mod_idx < len(self.modules) - 1:
                instructions.append({
                    'type': 'metal_gap',
                    'width': wg.width,
                    'z_start': current_z,
                    'z_end': current_z + self.d_gap,
                    'complex_E_field': 0j
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
        
        if inst['type'] == 'metal':
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

