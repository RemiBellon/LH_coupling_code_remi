import math
import cmath
from dataclasses import dataclass
from typing import List, Dict

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
        self.freq = config.physics.wave.freq_LH
        self.wg_width = config.geometry.antenna.dimensions.wg_width
        self.wg_height = config.geometry.antenna.dimensions.wg_height
        self.grill_arrangement = config.geometry.antenna.arrangement
        
        # Calculate fundamental TE10 properties immediately
        self._calculate_te10_properties()

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

    def build_spatial_source_function(self, spatial_var, wg_sequence: list):
        """
        Translates discrete waveguide port excitations into a continuous NGSolve 
        CoefficientFunction over the spatial variable (e.g., the toroidal z-axis).
        """
        from ngsolve import CF, IfPos
        
        # Get the complex amplitudes (E_0 * exp(i * phi)) for all ports
        # This function respects the module power division
        ports = self.get_port_excitations()
        
        # Initialize an empty complex field
        Ez_inc_cf = CF(0.0 + 0.0j)
        
        port_idx = 0
        for wg in wg_sequence:
            z_start = wg["z_start"]
            z_end = wg["z_end"]
            
            if wg["type"] == "active":
                # Extract the corresponding pre-calculated port physics
                port = ports[port_idx]
                E_val = port.E_complex
                port_idx += 1
            else:
                # Passive waveguides inject zero incident power
                E_val = 0.0 + 0.0j

            # NGSolve spatial mapping: 1.0 IF (z > z_start AND z < z_end) ELSE 0.0
            is_inside_wg = IfPos(spatial_var - z_start, IfPos(z_end - spatial_var, 1.0, 0.0), 0.0)
            
            # Add this waveguide's field to the total function
            Ez_inc_cf += is_inside_wg * E_val
            
        return Ez_inc_cf