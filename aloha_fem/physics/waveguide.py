import math
import cmath
from dataclasses import dataclass
from typing import List, Dict
from ngsolve import CF, IfPos, cos
import math

# Standard physical constants
C_LIGHT = 299792458.0           # Speed of light in vacuum (m/s)
MU_0 = 4 * math.pi * 1e-7       # Vacuum permeability (H/m)
EPS_0 = 1 / (MU_0 * C_LIGHT**2) # Vacuum permittivity (F/m)
Z_0 = math.sqrt(MU_0 / EPS_0)   # Vacuum impedance (Ohms)

@dataclass
class WaveguidePort: # 
    """Physical excitation for a single active waveguide port."""
    module_id: int
    waveguide_index: int
    E_amplitude: float      # Peak electric field (V/m)
    phase_rad: float        # Phase shift (radians)
    E_complex: complex      # Complex field amplitude ready for FEM

class WaveguidePhysics:
    def __init__(self, config):
        """Initializes the physics engine directly from the Pydantic config."""
        self.config = config
        self.freq = config.physics.wave.freq_LH
        self.wg_width = config.geometry.antenna.dimensions.wg_width
        self.wg_height = config.geometry.antenna.dimensions.wg_height
        self.grill_arrangement = config.geometry.antenna.grill_arrangement
        
        # Calculate fundamental TE10 properties immediately
        self._calculate_te10_properties()
        self.wg_sequence = self._generate_waveguide_sequence()

    def _calculate_te10_properties(self):
        """Calculates exact analytical properties of the TE10 mode."""
        self.f_c = C_LIGHT / (2 * self.wg_width)
        
        # Robustness check: Ensure the wave can physically propagate
        if self.freq <= self.f_c:
            raise ValueError(
                f"Physical impossibility: Operating frequency ({self.freq / 1e9} GHz) "
                f"is below the TE10 cutoff frequency ({self.f_c / 1e9} GHz) for a "
                f"waveguide of width {self.wg_width} m. The wave is evanescent."
            )

        # Wavenumbers
        self.k_0 = (2 * math.pi * self.freq) / C_LIGHT
        self.k_c = math.pi / self.wg_width
        self.beta = math.sqrt(self.k_0**2 - self.k_c**2)
        
        # Waveguide Impedance
        self.Z_TE = (self.k_0 / self.beta) * Z_0

    def _generate_waveguide_sequence(self):
        """Translates PAM/FAM configurations into a strict spatial sequence."""
        sequence = []
        ant_cfg = self.config.geometry.antenna
        if not ant_cfg: 
            self.max_wg_length = 0.0
            return sequence

        wg_width = ant_cfg.dimensions.wg_width
        septa_width = ant_cfg.dimensions.septa_width

        # Store the max length here so the mesh builder can access it easily
        self.max_wg_length = max(
            ant_cfg.dimensions.wg_length_active,
            ant_cfg.dimensions.wg_length_passive
        )

        current_z = self.config.geometry.domain.Lz_wall
        for mod_idx in range(ant_cfg.grill_arrangement.num_modules):
            num_active = ant_cfg.grill_arrangement.active_waveguides_per_module[mod_idx]

            if ant_cfg.topology == "FAM":
                mod_sequence = ["active"] * num_active
            elif ant_cfg.topology == "PAM":
                mod_sequence = []
                for _ in range(num_active):
                    mod_sequence.extend(["passive", "active"])
                mod_sequence.append("passive")
            else:
                raise ValueError(f"Unknown topology: {ant_cfg.topology}")

            for wg_type in mod_sequence:
                length = ant_cfg.dimensions.wg_length_active if wg_type == "active" else ant_cfg.dimensions.wg_length_passive
                sequence.append({
                    "type": wg_type,
                    "length": length,
                    "z_start": current_z,
                    "z_end": current_z + wg_width
                })
                current_z += wg_width + septa_width

        return sequence

    def get_port_excitations(self) -> List[WaveguidePort]:
        """
        Calculates the exact complex Electric field for every active waveguide 
        based on the power distribution per module.
        """
        ports = []
        
        # Unpack the list arrangements from the config
        num_mods = self.grill_arrangement.num_modules
        wg_act_per_mod = self.grill_arrangement.active_waveguides_per_module
        power_per_mod = self.grill_arrangement.power_per_module_W
        phase_shift_deg = self.grill_arrangement.phase_shift_per_module_deg

        for mod_idx in range(num_mods):
            mod_id = mod_idx + 1
            num_wg = wg_act_per_mod[mod_idx]
            total_mod_power = power_per_mod[mod_idx]
            delta_phi_deg = phase_shift_deg[mod_idx]
            
            # Divide power evenly among active waveguides in the module
            power_per_wg = total_mod_power / num_wg
            
            # Exact power-to-field formula
            # E_0 = sqrt((4 * P * Z_TE) / (width * height))
            E_0 = math.sqrt((4 * power_per_wg * self.Z_TE) / (self.wg_width * self.wg_height))
            
            for wg_idx in range(num_wg):
                # Calculate relative phase for this specific waveguide in the module
                current_phase_deg = wg_idx * delta_phi_deg
                current_phase_rad = math.radians(current_phase_deg)
                
                # Complex field representation: E_0 * exp(j * phi)
                E_cplx = E_0 * cmath.exp(1j * current_phase_rad)
                
                port = WaveguidePort(
                    module_id=mod_id,
                    waveguide_index=wg_idx + 1,
                    E_amplitude=E_0,
                    phase_rad=current_phase_rad,
                    E_complex=E_cplx
                )
                ports.append(port)
                
        return ports

    def build_spatial_source_function(self, spatial_var):
        """
        Translates discrete waveguide port excitations into a continuous NGSolve 
        CoefficientFunction over the spatial variable using the exact TE10 mode profile.
        """
        
        ports = self.get_port_excitations()
        Ez_inc_cf = CF(0.0 + 0.0j)
        
        port_idx = 0
        for wg in self.wg_sequence:
            z_start = wg["z_start"]
            z_end = wg["z_end"]
            z_center = (z_start + z_end) / 2.0
            
            if wg["type"] == "active":
                port = ports[port_idx]
                E_val = port.E_complex
                port_idx += 1
            else:
                E_val = 0.0 + 0.0j

            # 1.0 inside the WG, 0.0 outside
            is_inside_wg = IfPos(spatial_var - z_start, IfPos(z_end - spatial_var, 1.0, 0.0), 0.0)
            
            # Exact TE10 Cosine Profile: E_0 * cos(pi * (z - z_center) / width)
            # The field is maximal at z_center and strictly 0.0 at z_start and z_end
            mode_profile = cos((math.pi * (spatial_var - z_center)) / self.wg_width)
            
            Ez_inc_cf += is_inside_wg * mode_profile * E_val
            
        return Ez_inc_cf