import math
import numpy as np
import scipy.constants as const
from ngsolve import x, exp, IfPos, CoefficientFunction

class StixPhysics:
    def __init__(self, config):
        """Initializes the Stix tensor builder using the validated physics config."""
        self.freq = config.physics.wave.freq_LH
        self.omega = 2 * math.pi * self.freq
        
        # Fundamental physical constants
        self.e = const.e
        self.eps_0 = const.epsilon_0
        self.m_e = const.m_e
        
        # Assuming Deuterium plasma for exactness (Z=1)
        self.m_i = 2.014 * const.m_u 
        
        # Build the continuous NGSolve spatial functions
        self.n_e = self._build_piecewise_cf(config.physics.plasma.radial_density_profile, is_density=True)
        self.B_0 = self._build_piecewise_cf(config.physics.plasma.b_field_profile, is_density=False)

    def _build_piecewise_cf(self, profile, is_density: bool) -> CoefficientFunction:
        """
        Translates a Pydantic piecewise profile into an NGSolve CoefficientFunction.
        Uses IfPos() to evaluate different functions based on the x-coordinate.
        """
        points = profile.points
        segments = profile.segments
        
        # Start from the right-most point (deepest in the core)
        current_cf = CoefficientFunction(points[-1][1])
        
        # Iterate backwards through the segments to build nested IfPos conditions
        for i in range(len(segments) - 1, -1, -1):
            x1, val1 = points[i]
            x2, val2 = points[i+1]
            seg_type = segments[i]
            
            if seg_type == "constant":
                local_cf = CoefficientFunction(val1)
                
            elif seg_type == "linear":
                slope = (val2 - val1) / (x2 - x1)
                local_cf = val1 + slope * (x - x1)
                
            elif seg_type == "exponential":
                # val(x) = val1 * exp((x - x1) / decay_len)
                # decay_len = (x2 - x1) / ln(val2 / val1)
                decay_len = (x2 - x1) / math.log(val2 / val1)
                local_cf = val1 * exp((x - x1) / decay_len)
                
            else:
                raise ValueError(f"Unknown segment type: {seg_type}")
            
            # Stitch the current segment with the expression built so far
            current_cf = IfPos(x - x2, current_cf, local_cf)
            
        # Handle the vacuum gap before the antenna mouth (x < 0)
        # Density must strictly be zero, B-field remains constant
        vacuum_val = 0.0 if is_density else points[0][1]
        current_cf = IfPos(x - points[0][0], current_cf, CoefficientFunction(vacuum_val))
        return current_cf

    def solve_booker_roots(self, n_para: float, ne_val: float) -> dict:
        """
        Solves the exact Booker equation for a given local density and dominant n_parallel.
        Used strictly as a heuristic to estimate extreme refractive indices for mesh refinement,
        NOT for the weak formulation physics.
        """
        if ne_val <= 0.0:
            n_perp_vac = math.sqrt(max(1.0 - n_para**2, 0.0)) # Handle evanescence safely
            return {"n_perp_p": n_perp_vac, "n_perp_m": n_perp_vac}

        # Local plasma frequencies squared
        w_pe2 = (ne_val * self.e**2) / (self.m_e * self.eps_0)
        w_pi2 = (ne_val * self.e**2) / (self.m_i * self.eps_0)
        
        # Local Stix parameters
        S = 1.0 - w_pe2/(self.omega**2 - self.omega_ce**2) - w_pi2/(self.omega**2 - self.omega_ci**2)
        P = 1.0 - w_pe2/self.omega**2 - w_pi2/self.omega**2
        D = -(self.omega_ce * w_pe2)/(self.omega*(self.omega**2 - self.omega_ce**2)) + \
             (self.omega_ci * w_pi2)/(self.omega*(self.omega**2 - self.omega_ci**2))

        # Booker polynomial coefficients
        n_para_sq = n_para**2
        B_stix = (S + P)*n_para_sq - (S**2 - D**2) - P*S 
        C_stix = P * (n_para_sq - (S + D)) * (n_para_sq - (S - D))
        
        delta = complex(B_stix**2 - 4*S*C_stix)
        
        # Slow Wave (n_perp_p) and Fast Wave (n_perp_m)
        n_perp_p = np.sqrt((-B_stix + np.sqrt(delta)) / (2*S))
        n_perp_m = np.sqrt((-B_stix - np.sqrt(delta)) / (2*S))
        
        return {"n_perp_p": n_perp_p, "n_perp_m": n_perp_m}

    def get_stix_parameters(self) -> dict:
        """
        Computes the S, D, and P Stix parameters strictly based on cold plasma theory.
        Returns a dictionary of NGSolve CoefficientFunctions.
        """
        # Electron and Ion Plasma Frequencies Squared (omega_p^2)
        omega_pe_sq = (self.n_e * self.e**2) / (self.eps_0 * self.m_e)
        omega_pi_sq = (self.n_e * self.e**2) / (self.eps_0 * self.m_i) # Assuming Z_eff = 1

        # Electron and Ion Cyclotron Frequencies (with electrical charge sign)
        omega_ce = (-self.e * self.B_0) / self.m_e
        omega_ci = (self.e * self.B_0) / self.m_i

        # Denominators for perpendicular dynamics
        denom_e = self.omega**2 - omega_ce**2
        denom_i = self.omega**2 - omega_ci**2

        # Stix components
        S = 1 - (omega_pe_sq / denom_e) - (omega_pi_sq / denom_i)
        
        D_e = (omega_ce / self.omega) * (omega_pe_sq / denom_e)
        D_i = (omega_ci / self.omega) * (omega_pi_sq / denom_i)
        D = D_e + D_i
        
        P = 1 - (omega_pe_sq / self.omega**2) - (omega_pi_sq / self.omega**2)

        return {"S": S, "D": D, "P": P}