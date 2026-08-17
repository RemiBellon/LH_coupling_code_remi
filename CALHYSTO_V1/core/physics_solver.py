import numpy as np
from ngsolve import *
from config.schema import SimulationConfig
from utils.antenna_desc_2D import build_grill_from_config

class LowerHybridSolver:
    def __init__(self, config: SimulationConfig, mesh: Mesh):
        self.cfg = config
        self.mesh = mesh

        # Extract physical constants
        self.eps0 = 8.854e-12
        self.mu0 = 4 * np.pi * 1e-7
        self.qe = 1.602e-19
        self.me = 9.109e-31
        self.mi = 3.34e-27 # Assuming Deuterium
        self.Z0 = np.sqrt(self.mu0/self.eps0)
        self.omega_LH = self.cfg.physics.wave.omega_LH
        self.k0 = self.cfg.physics.wave.k0
        self.n_para = self.cfg.physics.wave.n_para_req

    def build_stix_tensor(self, density_cf: CoefficientFunction) -> CoefficientFunction:
        """
        Builds the cold plasma Stix tensor.
        Because density_cf is strictly 0.0 in the 'vacuum_active' and 'vacuum_passive'
        domains, this naturally reduces to the vacuum identity tensor there.
        """
        # Assume constant B-field
        B0 = self.cfg.physics.plasma.b_field_profile.points[0][1]
        self.Om_ce = (self.qe * B0) / self.me
        self.Om_ci = (self.qe * B0) / self.mi

        w_pe2 = (density_cf * self.qe**2) / (self.me * self.eps0)
        w_pi2 = (density_cf * self.qe**2) / (self.mi * self.eps0)

        S_cf = 1.0 - w_pe2/(self.omega_LH**2 - self.Om_ce**2) - w_pi2/(self.omega_LH**2 - self.Om_ci**2)
        P_cf = 1.0 - w_pe2/self.omega_LH**2 - w_pi2/self.omega_LH**2
        D_cf = -(self.Om_ce * w_pe2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ce**2)) + \
                (self.Om_ci * w_pi2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ci**2))

        # Assuming B-field is purely toroidal (z-direction) for this 2D approximation: bx=0, by=0, bz=1
        self.K_xx = S_cf
        self.K_xy = 1j * D_cf
        self.K_xz = CF(0.0)

        self.K_yx = -1j * D_cf
        self.K_yy = S_cf
        self.K_yz = CF(0.0)

        self.K_zx = CF(0.0)
        self.K_zy = CF(0.0)
        self.K_zz = P_cf

        self.stix_tensor = CF((self.K_xx, self.K_xy, self.K_xz,
                                  self.K_yx, self.K_yy, self.K_yz,
                                  self.K_zx, self.K_zy, self.K_zz), dims=(3,3))
        return self.stix_tensor

    def setup_pml_tensor(self) -> CoefficientFunction:
        """ Constructs dynamic PMLs bounded by the Pydantic schema constraints. """
        pml_cfg = self.cfg.geometry.pml
        dom = self.cfg.geometry.domain

        # Spatial variables
        x_sym, z_sym = x, y
        sign_x = -1.0 if self.cfg.simulation.box_medium == "PLASMA" else 1.0
        sign_z = 1.0

        Stretch_x = 1.0 + (pml_cfg.Sx_r - 1.0 + 1j * sign_x * pml_cfg.Sx_im) * \
                    IfPos(x_sym - dom.Lx_plasma, ((x_sym - dom.Lx_plasma) / dom.Lx_pml)**pml_cfg.px, 0.0)

        # Toroidal PMLs: Starts for z < 0 or z > Lz_plasma
        Stretch_z = 1.0 + (pml_cfg.Sz_r - 1.0 + 1j * sign_z * pml_cfg.Sz_im) * \
                    IfPos(-z_sym, (-z_sym / dom.Lz_pml)**pml_cfg.pz, \
                    IfPos(z_sym - dom.Lz_plasma, ((z_sym - dom.Lz_plasma) / dom.Lz_pml)**pml_cfg.pz, 0.0))

        self.eff_eps_tensor = CF((
            (Stretch_z / Stretch_x) * self.K_xx, (Stretch_z / Stretch_x) * self.K_xy, CF(0.0),
            (Stretch_x * Stretch_z) * self.K_yx, (Stretch_x * Stretch_z) * self.K_yy, CF(0.0),
            CF(0.0), CF(0.0), (Stretch_x / Stretch_z) * self.K_zz), dims=(3,3))

        self.pml_mu_inv_tensor = CF((Stretch_x / Stretch_z, 0.0, 0.0,
                                     0.0, 1.0/(Stretch_x * Stretch_z), 0.0,
                                     0.0, 0.0, Stretch_z / Stretch_x), dims=(3,3))

    def get_vacuum_port_admittance(self):
        """
        Computes the vacuum TM mode admittance.
        Since waveguides are physically meshed, the inlet is in vacuum.
        """
        n_perp_vac = 1.0 + 0.0j

        Y_11_ref = 0.0 + 0.0j
        Y_12_ref = 1.0 / (n_perp_vac * self.Z0)
        Y_21_ref = -1.0 / (n_perp_vac * self.Z0)
        Y_22_ref = 0.0 + 0.0j

        return Y_11_ref, Y_12_ref, Y_21_ref, Y_22_ref

    def get_incident_fields(self, Ez_inc_val: CoefficientFunction):
        """ Maps the phased Ez_inc to the transverse H-fields for the Robin boundary condition. """
        n_perp_vac = 1.0 + 0.0j

        Ey_inc = CF(0.0 + 0.0j)
        Ez_inc = Ez_inc_val
        Hy_inc = -(1.0 / (n_perp_vac * self.Z0)) * Ez_inc_val
        Hz_inc = CF(0.0 + 0.0j)

        return Ey_inc, Ez_inc, Hy_inc, Hz_inc

    def build_antenna_source_cf(self) -> CoefficientFunction:
        """
        Reconstructs the phased power excitation purely from the YAML layout.
        Maps the complex E-field to the specific waveguide positions.
        """
        from ngsolve import y, IfPos, exp

        ant = self.cfg.geometry.antenna
        dom = self.cfg.geometry.domain
        if ant is None:
            print("[INFO] No antenna geometry defined. Injecting uniform plane wave.")
            # Inject a wave with the requested parallel phase shift
            # Note: The toroidal z-coordinate maps to 'y' in the 2D NGSolve mesh
            return CF(1.0 + 0.0j) * exp(1j * self.k0 * self.n_para * y)


        grill, instructions = build_grill_from_config(ant, dom)

        Ez_cf = CF(0.0 + 0.0j)
        for inst in instructions:
            if inst['type'] == 'wg_active':
                z_start = inst['z_start']
                z_end = inst['z_end']
                E_val = inst['complex_E_field']

                is_inside = IfPos(y - z_start, IfPos(z_end - y, 1.0, 0.0), 0.0)
                Ez_cf = Ez_cf + is_inside * CF(E_val)

        return Ez_cf

    def solve(self, density_cf: CoefficientFunction):
        """ Main solver execution loop. """
        self.build_stix_tensor(density_cf)
        self.setup_pml_tensor()

        dirichlet_bnds = "metal"
        p_order = self.cfg.solver.fem_order

        fes_plane = HCurl(self.mesh, order=p_order, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(self.mesh, order=p_order, complex=True, dirichlet=dirichlet_bnds)
        self.fes = fes_plane * fes_outplane

        print(f"--- Solving full-wave system with {self.fes.ndof} DoFs ---")
        self.E_field = GridFunction(self.fes)
        E_plane, E_outplane = self.fes.TrialFunction()
        v_plane, v_outplane = self.fes.TestFunction()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1]))
        v_3D = CF((v_plane[0], v_outplane, v_plane[1]))

        # Mathematical mapping of the 3D curl operator in a 2D (x, z) domain
        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

        Y_11, Y_12, Y_21, Y_22 = self.get_vacuum_port_admittance()

        a = BilinearForm(self.fes)

        # Volume Integral
        a += ((self.pml_mu_inv_tensor * curl_E_3D) * curl_v_3D - \
              self.k0**2 * (self.eff_eps_tensor * E_3D) * v_3D) * dx

        # Boundary Integral (Admittance condition at the waveguide inlets)
        Ez_trace, Ey_trace = E_plane.Trace()[1], E_outplane.Trace()
        vz_trace, vy_trace = v_plane.Trace()[1], v_outplane.Trace()

        # print(f"E_plane dim: {E_plane.dim}")
        # print(f"E_plane Trace dim: {E_plane.Trace().dim}")
        # print(f"v_plane Trace dim: {v_plane.Trace().dim}")
        # print(f"E_plane is from: {E_plane.space.type}")

        # mip = self.mesh(0.0, 0.0)
        # print(self.stix_tensor(mip))

        a += 2j * self.omega_LH * self.mu0 * ((Y_21 * Ey_trace + Y_22 * Ez_trace) * vy_trace - \
             (Y_11 * Ey_trace + Y_12 * Ez_trace) * vz_trace) * ds("wg_inlet")

        # Assembling the Linear Form (Source Excitation)
        f = LinearForm(self.fes)
        Ez_inc_spatial = self.build_antenna_source_cf()
        Ey_inc, Ez_inc, Hy_inc, Hz_inc = self.get_incident_fields(Ez_inc_spatial)

        Ay = (Y_11 * Ey_inc + Y_12 * Ez_inc) - Hy_inc
        Az = (Y_21 * Ey_inc + Y_22 * Ez_inc) - Hz_inc

        f += 1j * self.omega_LH * self.mu0 * (Az * vy_trace - Ay * vz_trace) * ds("wg_inlet")

        # Matrix Inversion
        with TaskManager():
            a.Assemble()
            f.Assemble()

            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * self.E_field.vec

            # Use the solver defined in the YAML config
            inv = a.mat.Inverse(freedofs=self.fes.FreeDofs()) #, inverse=self.cfg.solver.linear_backend)
            self.E_field.vec.data += inv * res

        print("--- Linear System Solved Successfully ---")
        return self.E_field

    def apply_aposteriori_amr(self, density_cf: CoefficientFunction, max_steps: int = 3, tolerance: float = 1e-4):
        """
        Executes the AMR loop by measuring H-field discontinuities at element boundaries.
        """
        from ngsolve import Norm, Integrate, VOL

        for step in range(max_steps):
            print(f"\n--- A-Posteriori AMR Step {step+1}/{max_steps} ---")

            self.solve(density_cf)

            E_plane = self.E_field.components[0]
            curl_E = curl(E_plane)

            p_order = self.cfg.solver.fem_order
            fes_smooth = H1(self.mesh, order=p_order, complex=True)
            curl_E_smooth = GridFunction(fes_smooth)

            u, v = fes_smooth.TnT()
            m = BilinearForm(u * v * dx).Assemble()
            f = LinearForm(curl_E * v * dx).Assemble()

            curl_E_smooth.vec.data = m.mat.Inverse() * f.vec

            err_cf = Norm(curl_E - curl_E_smooth)**2
            el_errors = Integrate(err_cf, self.mesh, VOL, element_wise=True)

            max_err = max(el_errors)
            print(f"Maximum Element Error: {max_err:.2e}")

            if max_err < tolerance:
                print("AMR Converged based on tolerance.")
                break

            total_err = sum(el_errors)
            sorted_indices = sorted(range(len(el_errors)), key=lambda i: el_errors[i], reverse=True)

            target_indices = set()
            running_sum = 0

            for idx in sorted_indices:
                target_indices.add(idx)
                running_sum += el_errors[idx]
                if running_sum > 0.3 * total_err:
                    break

            marked_count = 0
            for i, el in enumerate(self.mesh.Elements(VOL)):
                should_refine = (i in target_indices)
                self.mesh.SetRefinementFlag(el, should_refine)
                if should_refine:
                    marked_count += 1

            print(f"Marking {marked_count} elements for refinement around singularities/gradients.")

            # Execute refinement natively using the internal flags
            self.mesh.Refine()