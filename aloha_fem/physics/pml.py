import math
from ngsolve import x, y, z, CF, IfPos, CoefficientFunction

class JacquotPML:
    def __init__(self, config):
        """
        Initializes the Jacquot 2013 PML stretching functions.
        Extracts validated geometries and PML properties directly from SimulationConfig.
        """
        self.dim = config.simulation.dimension
        self.box_medium = config.simulation.box_medium
        self.config = config
        # PML Parameters
        self.pml_cfg = config.geometry.pml
        self.Lx_plasma, self.Lx_pml = config.geometry.domain.Lx_plasma, config.geometry.domain.Lx_pml
        self.Ly_plasma, self.Ly_pml, self.Ly_wall = config.geometry.domain.Ly_plasma, config.geometry.domain.Ly_pml, config.geometry.domain.Ly_wall
        self.Lz_plasma, self.Lz_pm, self.Lz_wall = config.geometry.domain.Lz_plasma, config.geometry.domain.Lz_pml, config.geometry.domain.Lz_wall

        # Build spatial stretching coefficients
        self.s_x, self.s_y, self.s_z = self._build_stretching_factors()

    def _build_stretching_factors(self) -> tuple[CoefficientFunction, CoefficientFunction]:
        """
        Constructs the complex polynomial stretching functions s_x and s_z using IfPos.
        Ensures the PML is completely null (value = 1.0) inside the physical domain.
        """
        # --- Radial (X) Stretching ---
        if self.pml_cfg.use_radial and self.Lx_pml > 0.0:
            Sx_r, Sx_im, px = self.pml_cfg.Sx_r, self.pml_cfg.Sx_im, self.pml_cfg.px

            # The sign adapts based on outward wave propagation characteristics
            sign_x = 1.0 if self.box_medium == "VACUUM" else -1.0 # absorb forward wave in vacuum and backward wave in plasma

            # Active only for x > Lx_plasma
            s_x = 1.0 + (Sx_r - 1.0 + 1j * sign_x * Sx_im) * \
                  IfPos(x - self.Lx_plasma, ((x - self.Lx_plasma) / self.Lx_pml)**px, 0.0)
        else:
            s_x = CF(1.0)

        if self.dim == "2D":
            poloidal_var, toroidal_var = None, y
        elif self.dim == "3D":
            poloidal_var, toroidal_var = y, z
        else:
            poloidal_var, toroidal_var = None, None

        # --- Toroidal (Z) Stretching ---
        if self.dim in ["2D", "3D"] and self.config.simulation.boundary_toroidal == "pml":
            Sz_r, Sz_im, pz = self.pml_cfg.Sz_r, self.pml_cfg.Sz_im, self.pml_cfg.pz
            z_right_boundary = self.Lz_plasma + 2.0 * self.Lz_wall
            
            s_z = 1.0 + (Sz_r - 1.0 + 1j * Sz_im) * \
                  IfPos(-toroidal_var, (-toroidal_var / self.Lz_pml)**pz, \
                  IfPos(toroidal_var - z_right_boundary, ((toroidal_var - z_right_boundary) / self.Lz_pml)**pz, 0.0))
        else:
            s_z = CF(1.0)
            
        # --- Poloidal (Y) Stretching (Only active in 3D) ---
        if self.dim == "3D" and self.config.simulation.boundary_poloidal == "pml":
            Sy_r, Sy_im, py = self.pml_cfg.Sy_r, self.pml_cfg.Sy_im, self.pml_cfg.py
            y_right_boundary = self.Ly_plasma 
            
            s_y = 1.0 + (Sy_r - 1.0 + 1j * Sy_im) * \
                  IfPos(-poloidal_var, (-poloidal_var / self.Ly_pml)**pz, \
                  IfPos(poloidal_var - z_right_boundary, ((poloidal_var - y_right_boundary) / self.Ly_pml)**py, 0.0))
        else:
            s_y = CF(1.0)

        return s_x, s_y, s_z

    def get_curl_tensor(self) -> CoefficientFunction:
        """Computes the 3D diagonal Lambda metric tensor."""
        return CF((
            self.s_x / (self.s_y * self.s_z), 0.0, 0.0,
            0.0, self.s_y / (self.s_x * self.s_z), 0.0,
            0.0, 0.0, self.s_z / (self.s_x * self.s_y)
        ), dims=(3, 3))

    def get_effective_dielectric_tensor(self, K_tensor: CoefficientFunction) -> CoefficientFunction:
        """
        Applies the PML stretching metric to the physical dielectric tensor.
        Args:
            K_tensor (CoefficientFunction): The 3x3 dielectric tensor (e.g., from StixPhysics).
        """
        # Extract individual matrix components for precise scaling
        K_xx, K_xy, K_xz = K_tensor[0,0], K_tensor[0,1], K_tensor[0,2]
        K_yx, K_yy, K_yz = K_tensor[1,0], K_tensor[1,1], K_tensor[1,2]
        K_zx, K_zy, K_zz = K_tensor[2,0], K_tensor[2,1], K_tensor[2,2]

        # Multiply by the complex determinant Jacobian stretching factors
        return CF((
            K_xx * (self.s_z / self.s_x), K_xy * (self.s_z / self.s_x), K_xz * (self.s_z / self.s_x),
            K_yx * (self.s_x * self.s_z), K_yy * (self.s_x * self.s_z), K_yz * (self.s_x * self.s_z),
            K_zx * (self.s_x / self.s_z), K_zy * (self.s_x / self.s_z), K_zz * (self.s_x / self.s_z)
        ), dims=(3, 3))