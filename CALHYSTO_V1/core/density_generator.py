from ngsolve import x, y, IfPos, exp, CoefficientFunction, Mesh
from config.schema import SimulationConfig
import numpy as np

class DensityProfileBuilder:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.domain = config.geometry.domain

        self.x = x
        self.z = y  # NGSolve maps the second 2D coordinate to 'y', but physically it's toroidal 'z'

        self.xc = IfPos(self.x - self.domain.Lx_plasma, self.domain.Lx_plasma, self.x)
        zc_upper = IfPos(self.z - self.domain.Lz_plasma, self.domain.Lz_plasma, self.z)
        self.zc = IfPos(self.z, zc_upper, 0.0)

    def build_from_config(self, mesh: Mesh) -> tuple[CoefficientFunction, CoefficientFunction]:
        """
        Dynamically reads the YAML piecewise profile and assembles the CoefficientFunction.
        """
        profile_cfg = self.cfg.physics.plasma.radial_density_profile
        points = profile_cfg.points
        segments = profile_cfg.segments

        # Ensure points are ordered correctly
        points = sorted(points, key=lambda pt: pt[0])

        # Start from the outermost point (deepest in plasma) and build backwards
        final_x, final_ne = points[-1]
        bg_cf = CoefficientFunction(final_ne)

        # Loop backwards through the segments
        for i in reversed(range(len(segments))):
            x_start, ne_start = points[i]
            x_end, ne_end = points[i+1]
            seg_type = segments[i]

            if seg_type == "constant":
                segment_cf = CoefficientFunction(ne_start)
            elif seg_type == "linear":
                slope = (ne_end - ne_start) / (x_end - x_start)
                segment_cf = ne_start + slope * (self.xc - x_start)
            elif seg_type == "quadratic":
                # Polynomial ramp: n(x) = n0 + (n1 - n0) * ((x - x0)/(x1 - x0))^2
                normalized_x = (self.xc - x_start) / (x_end - x_start)
                segment_cf = ne_start + (ne_end - ne_start) * (normalized_x * normalized_x)
            elif seg_type == "exponential":
                # Assuming exponential form: n(x) = n0 * exp(x / L_n)
                # Ensure ne_start > 0 to avoid math domain errors
                L_n = (x_end - x_start) / np.log(ne_end / ne_start) if ne_start > 0 else 1.0
                segment_cf = ne_start * exp((self.xc - x_start) / (L_n))
            else:
                segment_cf = CoefficientFunction(ne_start)

            # Nest the conditional logic
            bg_cf = IfPos(self.xc - x_end, bg_cf, segment_cf)

        total_cf = bg_cf
        if self.cfg.physics.plasma.use_perturbations:
            print(f"--- Injecting {len(self.cfg.physics.plasma.perturbations)} perturbations into the plasma ---")
            for blob in self.cfg.physics.plasma.perturbations:
                # Gaussian evaluation natively in NGSolve
                blob_cf = blob.amplitude * exp(
                    -((self.xc - blob.x_c) / blob.sigma_x)**2
                    -((self.zc - blob.z_c) / blob.sigma_z)**2
                )
                total_cf = total_cf + blob_cf
        else:
            print("--- Perturbations disabled. Using baseline background profile. ---")

        # Physical safety check: Density cannot drop below zero
        total_cf = IfPos(total_cf, total_cf, 0.0)

        # 3. Apply Material Mapping
        mat_dict = {
            "vacuum_active": 0.0,
            "vacuum_passive": 0.0,
            "plasma": None, # Will be set below
            "radial_pml": None,
            "toroidal_left_pml": None,
            "toroidal_right_pml": None
        }

        # Background output map
        mat_dict_bg = mat_dict.copy()
        for key in ["plasma", "radial_pml", "toroidal_left_pml", "toroidal_right_pml"]:
            mat_dict_bg[key] = bg_cf
        mat_ne_bg = mesh.MaterialCF(mat_dict_bg, default=0.0)

        # Total output map
        mat_dict_tot = mat_dict.copy()
        for key in ["plasma", "radial_pml", "toroidal_left_pml", "toroidal_right_pml"]:
            mat_dict_tot[key] = total_cf
        mat_ne_tot = mesh.MaterialCF(mat_dict_tot, default=0.0)

        return mat_ne_bg, mat_ne_tot