import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Dict

class AntennaGrill:
    def __init__(self, b_active: float, d_septa: float, d_gap: float,
                Lx_wg_active: float, Lx_wg_passive: float = None, b_passive: float = None):
        """
        b_active: Width of active waveguides (m)
        d_septa: Thickness of internal metallic septa (m)
        d_gap: Maintained for signature compatibility, though physically replaced by shared passives
        Lx_wg_active: Depth of the active waveguides (m)
        Lx_wg_passive: Depth of the short-circuited passive waveguides (m)
        b_passive: Width of passive waveguides (defaults to b_active)
        """
        self.b_active = b_active
        self.d_septa = d_septa
        self.d_gap = d_gap 
        self.Lx_wg_active = Lx_wg_active
        self.Lx_wg_passive = Lx_wg_passive if Lx_wg_passive is not None else Lx_wg_active
        self.b_passive = b_passive if b_passive is not None else b_active
        
        # We now store module configurations, not the pre-built waveguides
        self.modules_config = [] 

    def add_module(self, num_active: int, is_PAM: bool, delta_phi_deg: float, amplitude: float, initial_phase_deg: float):
        """
        Registers a module's parameters. The actual layout is calculated during global assembly.
        """
        self.modules_config.append({
            'num_active': num_active,
            'is_PAM': is_PAM,
            'delta_phi_deg': delta_phi_deg,
            'amplitude': amplitude,
            'initial_phase_deg': initial_phase_deg
        })

    def generate_mesh_instructions(self, z_start_position: float = 0.0, add_global_edge_passives: bool = True) -> List[Dict]:
        """
        Translates the architecture into absolute z-coordinates for the NGSolve mesh builder.
        Dynamically handles inter-module shared passives.
        """
        instructions = []
        current_z = z_start_position
        
        # Helper function to keep generation clean
        def add_wg(wg_type: str, width: float, depth: float, E_field: complex):
            nonlocal current_z
            instructions.append({
                'type': wg_type,
                'width': width,
                'depth': depth,
                'z_start': current_z,
                'z_end': current_z + width,
                'complex_E_field': E_field
            })
            current_z += width

        # ==========================================
        # 1. GLOBAL LEFT EDGE
        # ==========================================
        if add_global_edge_passives:
            add_wg('wg_passive', self.b_passive, self.Lx_wg_passive, 0j)
            add_wg('metal', self.d_septa, 0.0, 0j)

        # ==========================================
        # 2. MODULE ASSEMBLY & INTER-MODULE GAPS
        # ==========================================
        for mod_idx, mod in enumerate(self.modules_config):
            current_phase_rad = np.radians(mod['initial_phase_deg'])
            delta_phi_rad = np.radians(mod['delta_phi_deg'])
            
            for i in range(mod['num_active']):
                # A. Place Active Waveguide
                E_val = mod['amplitude'] * np.exp(1j * current_phase_rad)
                add_wg('wg_active', self.b_active, self.Lx_wg_active, E_val)
                current_phase_rad += delta_phi_rad
                
                # B. Place internal septa and passives (if not the last active in THIS module)
                if i < mod['num_active'] - 1:
                    add_wg('metal', self.d_septa, 0.0, 0j)
                    if mod['is_PAM']:
                        add_wg('wg_passive', self.b_passive, self.Lx_wg_passive, 0j)
                        add_wg('metal', self.d_septa, 0.0, 0j)
            
            # C. Inter-Module Connection (Shared Passive)
            if mod_idx < len(self.modules_config) - 1:
                add_wg('metal', self.d_septa, 0.0, 0j)
                add_wg('wg_passive', self.b_passive, self.Lx_wg_passive, 0j)
                add_wg('metal', self.d_septa, 0.0, 0j)

        # ==========================================
        # 3. GLOBAL RIGHT EDGE
        # ==========================================
        if add_global_edge_passives:
            add_wg('metal', self.d_septa, 0.0, 0j)
            add_wg('wg_passive', self.b_passive, self.Lx_wg_passive, 0j)
                
        return instructions
    
def plot_antenna_blueprint(instructions):
    """
    Reads the NGSolve geometric instructions and draws the physical antenna face.
    """
    fig, ax = plt.subplots(figsize=(14, 4)) 
    
    max_z = instructions[-1]['z_end']
    max_depth = max(inst['depth'] for inst in instructions)
    if max_depth <= 0: max_depth = 0.05 
    
    for inst in instructions:
        z_start = inst['z_start']
        width = inst['width']
        depth = inst['depth']

        if inst['type'] in ['metal', 'metal_gap']:
            # FIX: Anchored at -max_depth, going UP by max_depth
            rect = patches.Rectangle((z_start, -max_depth), width, max_depth, 
                                     linewidth=1, edgecolor='black', facecolor='dimgrey', hatch='///')
            ax.add_patch(rect)
            
            if width >= 0.002: 
                ax.text(z_start + width/2, -max_depth/2, 'M', color='white', 
                        ha='center', va='center', fontsize=8, rotation=90)
        
        else:
            E_val = inst['complex_E_field']
            
            if inst['type'] == 'wg_passive':
                color = 'lightgrey'
                label = 'P'
            else:
                phase_deg = np.degrees(np.angle(E_val))
                norm_phase = (phase_deg + 180) / 360.0
                color = plt.cm.hsv(norm_phase)
                label = f'{phase_deg:.0f}°'
                
            rect = patches.Rectangle((z_start, -depth), width, depth, 
                                     linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
            if inst['type'] == 'wg_passive':
                ax.plot([z_start, z_start + width], [-depth, -depth], color='red', lw=3, zorder=5)              
            
            ax.text(z_start + width/2, -depth/2, label, color='black', 
                    ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

    ax.axhline(0, color='black', lw=2)

    ax.set_xlim(-0.005, max_z + 0.005)
    # FIX: Tighter Y-limits prevent the massive white vertical spaces
    ax.set_ylim(-max_depth * 1.1, max_depth * 0.1)
    
    ax.set_yticks([]) 
    ax.set_ylabel("Radial Depth $x$ [m]", fontsize=12)
    ax.set_xlabel("Toroidal Position $z$ [m]", fontsize=12)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=-180, vmax=180))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.08, pad=0.25, aspect=50)
    cbar.set_label('Electrical Phase [Degrees]')
    
    # FIX: bbox_inches='tight' crops the final figure perfectly
    plt.savefig("Antenna_Blueprint.pdf", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()