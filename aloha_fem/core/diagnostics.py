import h5py
import numpy as np
from ngsolve import VTKOutput, Integrate, dx, Conj, BND, IfPos
import os
from datetime import datetime

class DiagnosticManager:
    def __init__(self, solver, output_dir="results"):
        """
        Manages exact physical extractions and hierarchical data saving.
        Args:
            solver: The solved FEMSolver2D instance.
            output_dir: Directory to save the .vtu and .h5 files.
        """
        self.solver = solver
        self.config = solver.config
        self.mesh = solver.mesh
        self.E_field = solver.E_field
        self.output_dir = output_dir

    def extract_s_parameters(self: list) -> dict:
        """
        Computes the exact complex reflection coefficient (Gamma) for every active waveguide
        using the rigorous modal overlap integral.
        """
        print("\n--- Extracting Rigorous S-Parameters ---")
        
        # Extract the Z-component (Toroidal) of the in-plane HCurl field[cite: 12]
        E_z_tot = self.E_field.components[0][1] 
        
        # Rebuild the exact incident TE10 field to use as the orthogonal projection basis[cite: 16]
        E_inc_cf = self.solver.wg.build_spatial_source_function(self.solver.toroidal_var)
        
        gamma_dict = {}
        port_idx = 1
        wg_sequence = self.solver.wg.wg_sequence 
        for wg in wg_sequence:
            if wg["type"] == "active":
                z_start, z_end = wg["z_start"], wg["z_end"]
                mask = IfPos(self.solver.toroidal_var - z_start, IfPos(z_end - self.solver.toroidal_var, 1.0, 0.0), 0.0)
                
                # Integrals evaluated exactly over the mesh boundaries using NGSolve BND flag
                numerator_expr = mask * (E_z_tot - E_inc_cf) * Conj(E_inc_cf)
                denominator_expr = mask * E_inc_cf * Conj(E_inc_cf)
                
                num = Integrate(numerator_expr, self.mesh, BND)
                den = Integrate(denominator_expr, self.mesh, BND)
                
                gamma = num / den if abs(den) > 1e-12 else 0.0 + 0.0j
                
                gamma_dict[f"port_{port_idx}"] = {
                    "Gamma_real": gamma.real,
                    "Gamma_imag": gamma.imag,
                    "Power_Reflectivity": np.abs(gamma)**2,
                    "Phase_deg": np.degrees(np.angle(gamma)),
                    # Add the missing geometric metadata expected by SimulationData
                    "type": wg["type"],
                    "z_start": wg["z_start"],
                    "z_end": wg["z_end"],
                    "length": wg["length"]
                }
                print(f"  -> WG_{port_idx} | |Gamma|^2 = {np.abs(gamma)**2:.4f} | Phase = {np.degrees(np.angle(gamma)):.1f}°")
                port_idx += 1
                
        return gamma_dict

    def extract_toroidal_field_profile(self, x_target=0.0, num_points=4000) -> dict:
        """
        Extracts Ex, Ey, Ez along the toroidal axis (z) at a specific radial depth (x).
        Uses exact mesh interpolation.
        """
        print(f"\n--- Extracting Field Profile at x={x_target:.4f} m ---")
        
        # Determine total physical Z domain (excluding Toroidal PMLs for clean spectra)[cite: 14]
        z_max_plasma = self.config.geometry.domain.Lz_plasma + 2.0 * self.config.geometry.domain.Lz_wall
        z_sweep = np.linspace(0.0, z_max_plasma, num_points)
        
        Ex, Ey, Ez = [], [], []
        
        for z in z_sweep:
            try:
                # Mesh Evaluation Point (mip)
                mip = self.mesh(x_target, z)
                Ex.append(self.E_field.components[0][0](mip)) # HCurl Radial[cite: 12]
                Ey.append(self.E_field.components[1](mip))    # H1 Poloidal (Out of plane)[cite: 12]
                Ez.append(self.E_field.components[0][1](mip)) # HCurl Toroidal[cite: 12]
            except Exception:
                Ex.append(0.0 + 0.0j)
                Ey.append(0.0 + 0.0j)
                Ez.append(0.0 + 0.0j)
                
        Ex = np.array(Ex, dtype=complex)
        Ey = np.array(Ey, dtype=complex)
        Ez = np.array(Ez, dtype=complex)
        
        return {
            "z_coords": z_sweep,
            "Ex": Ex, "Ey": Ey, "Ez": Ez
        }

    def compute_power_spectrum(self, field_data: dict, gamma_dict: dict) -> tuple:
        """
        Computes the normalized absolute power density spectrum P(n_para).
        Rigorously scales the FFT shape so its integral exactly matches the net coupled active power.
        """
        print("--- Computing Normalized Power Spectrum (ALOHA Benchmark) ---")
        
        # 1. Calculate Net Coupled Power P_net
        P_net = 0.0
        active_ports = self.solver.wg.get_port_excitations() # Retrieves E_amplitude for each port[cite: 16]
        
        wg_width = self.config.geometry.antenna.dimensions.wg_width
        wg_height = self.config.geometry.antenna.dimensions.wg_height
        Z_TE = self.solver.wg.Z_TE
        
        port_idx = 1
        for port in active_ports:
            # Revert the E_amplitude back to forward power: P = (E0^2 * w * h) / (4 * Z_TE)[cite: 16]
            p_forward = (port.E_amplitude**2 * wg_width * wg_height) / (4.0 * Z_TE)
            
            gamma_sq = gamma_dict[f"port_{port_idx}"]["Power_Reflectivity"]
            P_net += p_forward * (1.0 - gamma_sq)
            port_idx += 1
            
        print(f"  -> Total Net Coupled Power: {P_net:.4e} W")
        
        # 2. Compute Spatial FFT of Ez[cite: 10]
        z_sweep = field_data["z_coords"]
        Ez = field_data["Ez"]
        dz = z_sweep[1] - z_sweep[0]
        num_pts = len(z_sweep)
        
        # High-resolution padding (factor of 8) for smooth spectral curves
        Ez_fft = np.fft.fftshift(np.fft.fft(Ez, 8 * num_pts)) * dz
        k_z = np.fft.fftshift(np.fft.fftfreq(8 * num_pts, d=dz)) * 2.0 * np.pi
        
        n_para = k_z / self.solver.k0
        power_spectrum = np.abs(Ez_fft)**2
        
        # 3. Apply Rigorous Normalization
        dn_para = n_para[1] - n_para[0]
        current_integral = np.trapezoid(power_spectrum, dx=dn_para)
        
        if current_integral > 1e-15:
            normalized_spectrum = power_spectrum * (P_net / current_integral)
        else:
            normalized_spectrum = power_spectrum
            print("[!] Spectrum integral is zero. Normalization failed.")
            
        return n_para, normalized_spectrum



    def export_hdf5_database(self, custom_prefix="fem_results", save_data=True, gamma_dict=None, field_data=None, spectrum_data=None):
        """
        Creates a highly structured HDF5 database containing physics metadata and observables.
        Includes a timestamp to prevent file overwriting and a toggle to bypass saving.
        """
        if not save_data:
            print("\n--- Data saving disabled. Skipping HDF5 export. ---")
            return

        # Generate exact timestamp (Format: YYYYMMDD_HHMMSS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{custom_prefix}_{timestamp}.h5"
        
        print(f"\n--- Exporting HDF5 Database to {self.output_dir}/{filename} ---")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with h5py.File(f"{self.output_dir}/{filename}", "w") as f:
            # Metadata
            meta = f.create_group("Metadata")
            meta.attrs["freq_LH"] = self.config.physics.wave.freq_LH
            meta.attrs["n_para_req"] = self.config.physics.wave.n_para_req
            
            # S-Parameters
            if gamma_dict:
                s_param = f.create_group("S_Parameters")
                for port, data in gamma_dict.items():
                    port_grp = s_param.create_group(port)
                    for key, val in data.items():
                        port_grp.attrs[key] = val
                        
            # Electrical Fields
            if field_data:
                flds = f.create_group("Tangential_Fields")
                flds.create_dataset("z_coords", data=field_data["z_coords"])
                flds.create_dataset("Ex_real", data=field_data["Ex"].real)
                flds.create_dataset("Ex_imag", data=field_data["Ex"].imag)
                flds.create_dataset("Ey_real", data=field_data["Ey"].real)
                flds.create_dataset("Ey_imag", data=field_data["Ey"].imag)
                flds.create_dataset("Ez_real", data=field_data["Ez"].real)
                flds.create_dataset("Ez_imag", data=field_data["Ez"].imag)
                
            # Power Spectrum
            if spectrum_data:
                spec = f.create_group("Power_Spectrum")
                spec.create_dataset("n_para", data=spectrum_data[0])
                spec.create_dataset("dP_dn_para", data=spectrum_data[1])

    def export_paraview_vtk(self, filename="FEM_fields"):
        """Exports the full mesh and complex vector fields for 2D/3D visualization."""
        print(f"\n--- Exporting VTK Data to {self.output_dir}/{filename}.vtu ---")
        
        # 1. Extract the complex spatial components
        E_rad = self.E_field.components[0][0]
        E_pol = self.E_field.components[1]
        E_tor = self.E_field.components[0][1]

        # 2. Explicitly separate into Real and Imaginary parts for VTK compatibility
        vtk = VTKOutput(
            ma=self.mesh,
            coefs=[
                E_rad.real, E_rad.imag,
                E_pol.real, E_pol.imag,
                E_tor.real, E_tor.imag
            ],
            names=[
                "E_radial_real", "E_radial_imag",
                "E_poloidal_real", "E_poloidal_imag",
                "E_toroidal_real", "E_toroidal_imag"
            ],
            filename=f"{self.output_dir}/{filename}",
            subdivision=3
        )
        vtk.Do()
