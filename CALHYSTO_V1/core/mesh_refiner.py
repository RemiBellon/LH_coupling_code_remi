import numpy as np
import scipy.constants as const
from ngsolve import Mesh, CoefficientFunction, specialcf, IfPos, sqrt, Integrate, BitArray, VOL
from config.schema import SimulationConfig

class WKBRefiner:
    """
    Executes a-priori mesh refinement based on the uniform density
    plasma Stix slow-wave dispersion relation.
    """
    def __init__(self, config: SimulationConfig, mesh: Mesh):
        self.cfg = config
        self.mesh = mesh

        self.eps0 = const.epsilon_0
        self.qe = const.e
        self.me = const.m_e
        self.mi = 3.34e-27

        self.omega = self.cfg.physics.wave.omega_LH
        self.k0 = self.cfg.physics.wave.k0
        self.n_para = self.cfg.physics.wave.n_para_req
        self.ppw = self.cfg.geometry.mesh.ppw_medium    # Pull points per wavelength from config
        self.h_min = 1e-4

    def _build_target_h_cf(self, density_cf: CoefficientFunction) -> CoefficientFunction:
        """ Computes the local target element size h(x,z) using the biquadratic Stix equation. """
        B0 = self.cfg.physics.plasma.b_field_profile.points[0][1]
        Om_ce = (self.qe * B0) / self.me
        Om_ci = (self.qe * B0) / self.mi

        w_pe2 = (density_cf * self.qe**2) / (self.me * self.eps0)
        w_pi2 = (density_cf * self.qe**2) / (self.mi * self.eps0)

        S = 1.0 - w_pe2/(self.omega**2 - Om_ce**2) - w_pi2/(self.omega**2 - Om_ci**2)
        P = 1.0 - w_pe2/self.omega**2 - w_pi2/self.omega**2
        D = -(Om_ce * w_pe2)/(self.omega*(self.omega**2 - Om_ce**2)) + \
             (Om_ci * w_pi2)/(self.omega*(self.omega**2 - Om_ci**2))

        # Stix Biquadratic Coefficients
        A_coeff = S
        B_coeff = S**2 - D**2 + P*S - self.n_para**2 * (P + S)
        C_coeff = P * ((self.n_para**2 - S)**2 - D**2)

        # Slow Wave Root: n_perp^2
        discriminant = B_coeff**2 - 4 * A_coeff * C_coeff
        discriminant_safe = IfPos(discriminant, discriminant, 0.0)
        n_perp_plus_sq = (B_coeff + sqrt(discriminant_safe)) / (2 * A_coeff)

        abs_n_perp_sq = IfPos(n_perp_plus_sq, n_perp_plus_sq, -n_perp_plus_sq)

        # Map to h_target: In vacuum (ne < 1e14), default to free space wavelength / ppw
        n_perp_mag = IfPos(density_cf - 1e14, sqrt(abs_n_perp_sq), 1.0)
        lambda_perp_plus = (2 * np.pi) / (self.k0 * n_perp_mag)

        h_ideal = lambda_perp_plus / self.ppw
        h_target_cf = IfPos(h_ideal - self.h_min, h_ideal, self.h_min)

        return h_target_cf

    def apply_apriori_refinement(self, density_cf: CoefficientFunction, max_passes: int = 3):
        """ Iteratively refines the mesh until it obeys the local WKB wavelength limit. """
        print("\n--- Initiating A-Priori WKB Mesh Refinement ---")
        h_target_cf = self._build_target_h_cf(density_cf)
        h_current_cf = specialcf.mesh_size

        # Mark element if its current size exceeds the target size
        refine_condition = IfPos(h_current_cf - h_target_cf, 1.0, 0.0)

        for pass_idx in range(max_passes):
            # Execute C++ Integration
            el_flags = Integrate(refine_condition, self.mesh, VOL, element_wise=True)

            num_marked = 0

            # Map the C++ array back to the native NGSolve Element IDs
            for i, el in enumerate(self.mesh.Elements(VOL)):
                should_refine = bool(el_flags[i] > 0)
                self.mesh.SetRefinementFlag(el, should_refine)
                if should_refine:
                    num_marked += 1

            print(f"Pass {pass_idx+1}: Refined {num_marked} elements.")

            if num_marked == 0:
                print("A-Priori Refinement Converged.")
                break

            # Trigger refinement on marked elements
            self.mesh.Refine()

        return self.mesh