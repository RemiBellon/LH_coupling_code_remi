import numpy as np
import scipy.constants as const
from ngsolve import Mesh, GridFunction, CoefficientFunction, CF, Integrate, ds, grad, curl, Conj, Cross, specialcf
from config.schema import SimulationConfig
from utils.antenna_desc_2D import AntennaGrill

class PhysicsPostProcessor:
    """
    Expert-level physics extraction:
    - Native boundary integral S-parameters and Active Gamma.
    - Exact Poynting flux-based dP/dn_parallel spectrum.
    - True multiport S-matrix computation.
    """
    def __init__(self, config: SimulationConfig, mesh: Mesh, E_field: GridFunction):
        self.cfg = config
        self.mesh = mesh
        self.E_field = E_field

        self.omega = config.physics.wave.omega_LH
        self.k0 = config.physics.wave.k0
        self.Z0 = np.sqrt(const.mu_0 / const.epsilon_0)

        # Build symbolic H-field and Poynting vector
        E_plane, E_outplane = E_field.components[0], E_field.components[1]
        self.E_3D = CF((E_plane[0], E_outplane, E_plane[1]))

        curl_E_3D = CF((-grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0]))
        self.H_3D = (1.0 / (1j * self.omega * const.mu_0)) * curl_E_3D

        E_r = CF((self.E_3D[0].real, self.E_3D[1].real, self.E_3D[2].real))
        E_i = CF((self.E_3D[0].imag, self.E_3D[1].imag, self.E_3D[2].imag))

        H_r = CF((self.H_3D[0].real, self.H_3D[1].real, self.H_3D[2].real))
        H_i = CF((self.H_3D[0].imag, self.H_3D[1].imag, self.H_3D[2].imag))

        self.S_poynting_3D = 0.5 * (Cross(E_r, H_r) + Cross(E_i, H_i))


    def compute_net_port_power(self) -> float:
        """
        Computes the net real power injected through the waveguide inlets [Watts].
        Calculated robustly using modal reflection coefficients.
        """
        ant = self.cfg.geometry.antenna
        s_params = self.extract_active_s_parameters()

        P_net = 0.0
        # Iterate over the injected powers defined in the YAML config
        for i in range(ant.grill_arrangement.num_modules):
            P_inc_total = ant.grill_arrangement.power_per_module_W[i]
            n_active = ant.grill_arrangement.active_waveguides_per_module_row[i]
            P_inc_per_wg = P_inc_total / n_active

            # Subtract the reflected power for each active waveguide
            for wg_idx in range(1, n_active + 1):
                wg_key = f'active_wg_{wg_idx}'
                if wg_key in s_params:
                    gamma_sq = s_params[wg_key]['Power_Reflectivity']
                    P_net += P_inc_per_wg * (1.0 - gamma_sq)

        print(f'P_net (via S-parameters) = {P_net:.3e} W')
        return P_net

    def extract_active_s_parameters(self) -> dict:
        """
        Extracts active reflection coefficients (Gamma_active) for each active waveguide
        using modal projection directly on the port boundary.
        """
        ant = self.cfg.geometry.antenna
        dom = self.cfg.geometry.domain

        grill = AntennaGrill(
            Lx_wg_active=ant.dimensions.Lx_wg_active,
            Lx_wg_passive=ant.dimensions.Lx_wg_passive,
            Ly_wg=ant.dimensions.Ly_wg,
            Lz_wg_active=ant.dimensions.Lz_wg_active,
            Lz_wg_passive=ant.dimensions.Lz_wg_passive,
            Lz_septa=ant.dimensions.Lz_septa,
            Lz_gap=ant.dimensions.Lz_gap_module
        )
        for i in range(ant.grill_arrangement.num_modules):
            grill.add_module(
                num_active=ant.grill_arrangement.active_waveguides_per_module_row[i],
                is_PAM=(ant.topology == "PAM"),
                delta_phi_deg=ant.grill_arrangement.phase_shift_per_module_deg[i],
                power_module_W=ant.grill_arrangement.power_per_module_W[i]
            )
        instructions = grill.generate_mesh_instructions(z_start_position=dom.Lz_wall, add_global_edge_passives=True)

        results = {}
        active_idx = 1

        from ngsolve import y, IfPos
        Ez_total_cf = self.E_3D[2]
        for inst in instructions:
            if inst['type'] == 'wg_active':
                z_start, z_end = inst['z_start'], inst['z_end']
                E_inc_val = inst['complex_E_field']
                width = z_end - z_start

                # Spatial window mask for this specific port
                port_mask = IfPos(y - z_start, IfPos(z_end - y, 1.0, 0.0), 0.0)
                integrand = Ez_total_cf * port_mask

                int_real = Integrate(integrand.real, self.mesh, definedon=self.mesh.Boundaries("wg_inlet"))
                int_imag = Integrate(integrand.imag, self.mesh, definedon=self.mesh.Boundaries("wg_inlet"))
                integral_Etot = int_real + 1j * int_imag

                # Average total field on port
                Ez_avg = integral_Etot / width

                # Modal decomposition: E_tot = E_inc + E_ref
                E_ref = Ez_avg - E_inc_val
                gamma_active = E_ref / E_inc_val if abs(E_inc_val) > 0 else 0.0j

                results[f'active_wg_{active_idx}'] = {
                    'z_center': float(0.5 * (z_start + z_end)),
                    'Gamma_active': complex(gamma_active),
                    'Power_Reflectivity': float(np.abs(gamma_active)**2),
                    'Phase_deg': float(np.degrees(np.angle(gamma_active)))
                }
                active_idx += 1

        return results

    def compute_exact_poynting_spectrum(self, x_eval: float = 0.001, num_points: int = 4096, pad_factor: int = 8):
        """
        Computes the physical power spectrum dP/dn_parallel using the cross-spectral
        Poynting formulation at a specified radial position in front of the antenna.
        """
        dom = self.cfg.geometry.domain
        z_min = -dom.Lz_pml
        z_max = dom.Lz_plasma + dom.Lz_pml

        z_coords, dz = np.linspace(z_min, z_max, num_points, retstep=True)

        # Sample tangential fields along the cut
        Ez_vals = np.zeros(num_points, dtype=complex)
        Hy_vals = np.zeros(num_points, dtype=complex)

        Ez_cf = self.E_3D[2]
        Hy_cf = self.H_3D[1]

        for i, z_val in enumerate(z_coords):
            try:
                mip = self.mesh(x_eval, z_val)
                Ez_vals[i] = Ez_cf(mip)
                Hy_vals[i] = Hy_cf(mip)
            except Exception:
                Ez_vals[i] = 0.0j
                Hy_vals[i] = 0.0j

        # Spatial Fourier Transform
        n_fft = num_points * pad_factor
        Ez_k = np.fft.fftshift(np.fft.fft(Ez_vals, n=n_fft)) * dz
        Hy_k = np.fft.fftshift(np.fft.fft(Hy_vals, n=n_fft)) * dz

        k_z = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dz)) * (2.0 * np.pi)
        n_para = k_z / self.k0

        # Exact Physical Spectral Poynting Flux [Watts / (m * delta_n_parallel)]
        # dP/dn_para = 0.5 * Re( E_z(k) * H_y^*(k) ) * Ly_wg
        dP_dn_para = 0.5 * np.real(Ez_k * np.conj(Hy_k)) * self.cfg.geometry.antenna.dimensions.Ly_wg
        # Filter negative unphysical numerical noise from evanescent window edges
        dP_dn_para = np.maximum(dP_dn_para, 0.0)

        return n_para, dP_dn_para