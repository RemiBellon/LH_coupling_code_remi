
import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time 
from pathlib import Path

# =====================================================================
# 1. MESH GENERATION (Plasma Domain + PML Domain)
# =====================================================================
class LHCouplingSolver_2DHcurl_1DH1:
    def __init__(self, config_dict):
        self.cfg = config_dict          # type = dict ==> dict with physics and geometry values
        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                     # In the context y is vertical direction. In 2D only the plane (xOz) is describe.   


    def compute_n_perp_plus_minus(self) -> None:
        self.freq_LH = self.cfg['WAVE']['freq_LH']
        self.omega = 2 * np.pi * self.freq_LH
        self.n_para = self.cfg['WAVE']['n_para']
        self.c0 = self.cfg['CONST']['c0']
        self.k0_vacuum = self.omega / self.c0
        self.ne_constant = self.cfg['PLASMA']['ne_constant']
        self.B0 = self.cfg['PLASMA']['B0_center_plasma']
        self.qe, self.me, self.mi, self.eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12

        # --- 2. Rigorous Stix Physics Evaluation ---
        self.w_pe2 = (self.ne_constant * self.qe**2) / (self.me * self.eps0)
        self.w_pi2 = (self.ne_constant * self.qe**2) / (self.mi * self.eps0)
        self.Om_ce = (self.qe * self.B0) / self.me
        self.Om_ci = (self.qe * self.B0) / self.mi
        
        self.S = 1 - self.w_pe2/(self.omega**2 - self.Om_ce**2) - self.w_pi2/(self.omega**2 - self.Om_ci**2)
        self.P = 1 - self.w_pe2/self.omega**2 - self.w_pi2/self.omega**2
        self.D = -(self.Om_ce * self.w_pe2)/(self.omega*(self.omega**2 - self.Om_ce**2)) + (self.Om_ci * self.w_pi2)/(self.omega*(self.omega**2 - self.Om_ci**2))
        
        self.B_stix = (self.S + self.P)*self.n_para**2 - (self.S**2 - self.D**2) - self.P*self.S
        self.C_stix = self.P * (self.n_para**2 - (self.S + self.D)) * (self.n_para**2 - (self.S - self.D))
        
        # Calculate Slow Wave (n_perp_plus)
        delta = max(0.0, self.B_stix**2 - 4*self.S*self.C_stix)
        n_perp_sq_p = (-self.B_stix + np.sqrt(delta)) / (2*self.S)
        n_perp_p = np.sqrt(max(1e-6, n_perp_sq_p))
        n_perp_sq_m = (-self.B_stix - np.sqrt(delta)) / (2*self.S)
        n_perp_m = np.sqrt(max(1e-6, n_perp_sq_m))
        
        return n_perp_p, n_perp_m
    
    def build_mesh_with_PMLs(self, mesh_save_dir) -> None:
        '''
        Function to set the meshgrid size and shape adding PMLs.
            - Set the mesh size Lx_tot, and compute the exact Lz size to fit as a multiple of lambda_z in z direction.
        '''
        # --- Mesh size in radial (x axis) direction: --- 
 # --- 1. Extract Core Parameters ---
        self.Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        self.Lx_pml = self.cfg['DOMAIN']['Lx_pml']
        self.Lx_tot = self.cfg['DOMAIN']['Lx_tot']

        self.Lz_plasma = self.cfg['DOMAIN']['Lz_plasma']
        self.Lz_pml = self.cfg['DOMAIN']['Lz_pml']
        self.Lz_tot = self.cfg['DOMAIN']['Lz_tot']
        
        rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
        rect_pml_radial = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
        rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
        rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
        rect_pml_corner_radial_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
        rect_pml_corner_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma).Rectangle(self.Lx_pml, self.Lz_pml).Face()

        rect_plasma.edges.Min(occ.X).name = "bottom_source"
        rect_pml_toroidal_left.edges.Min(occ.Y).name = "left_wall_pec"
        rect_pml_corner_radial_left.edges.Min(occ.Y).name = "left_wall_pec"

        rect_pml_toroidal_right.edges.Max(occ.Y).name = "right_wall_pec"
        rect_pml_corner_radial_right.edges.Max(occ.Y).name = "right_wall_pec"

        rect_pml_corner_radial_left.edges.Max(occ.X).name = "top_wall_pec"
        rect_pml_radial.edges.Max(occ.X).name = "top_wall_pec"
        rect_pml_corner_radial_right.edges.Max(occ.X).name = "top_wall_pec"


        rect_pml_toroidal_left.edges.Min(occ.X).name = "bottom_wall_pec"
        rect_pml_toroidal_right.edges.Min(occ.X).name = "bottom_wall_pec"


        domain = occ.Glue([rect_plasma, rect_pml_radial, rect_pml_toroidal_left, rect_pml_corner_radial_left, rect_pml_corner_radial_right, rect_pml_toroidal_right])
        geo = occ.OCCGeometry(domain, dim=2)

                
        # --- 3. Wavelengths and Scaling ---
        n_perp_p, n_perp_m = LHCouplingSolver_2DHcurl_1DH1.compute_n_perp_plus_minus(self)        
        lambda0_vacuum = self.c0 / self.freq_LH
        lambda_perp_SW = lambda0_vacuum / n_perp_p
        lambda_para = lambda0_vacuum / self.cfg['WAVE']['n_para']
        # The mesh MUST resolve the smallest wavelength in the system (Slow Wave)
        self.h_max = lambda_perp_SW / self.cfg['DOMAIN']['n_resol_per_wlgth']
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.h_max)) 
        
        # --- 5. Consistency Check Output ---
        print(f"\n[MESH BUILDER] :")
        print(f"  --> SW n_perp_p : {n_perp_p:.5e}, n_perp_m = {n_perp_m:.5e}")
        print(f"  --> SW lambda_perp : {lambda_perp_SW:.5e} m, lambda_para = {lambda_para:.5e} m")
        print(f"  --> Lx_plasma = {self.Lx_plasma:.2e} m , Lx_pml = {self.Lx_pml:.2e} m, Lx_tot = {self.Lx_tot:.2e} m")
        print(f"  --> Lz_plasma = {self.Lz_plasma:.2e} m , Lz_pml = {self.Lz_pml:.2e} m, Lz_tot = {self.Lz_tot:.2e} m")
        print(f"  --> h_max resolution : {self.h_max:.5e} m")
        return self.mesh
    
# =====================================================================
# 2. PHYSICS IMPLEMENTATION - STIX TENSOR, B FIELD, etc...
# =====================================================================
    def build_physics_Stix_B_field(self, density_func) -> None:
        '''
        Function to gather every needed physical parameters to build the Stix tensor 
        that contain all the plasma/wave physics and geometry:
            - Compute arbitrary B field components relative to cartesian coordinates 
            - Compute every general Stix tensor elements 
            and return it as a native NGSolve CoefficientFunction 
        '''

       # --- Basic problem physical and geometry parameters --- 
        omega_wave = self.cfg['WAVE']['omega_wave']
        B0 = self.cfg['PLASMA']['B0_center_plasma']
        R0 = self.cfg['GEOM']['R0']
        R_ant = self.cfg['GEOM']['R_ant']
        eps_0 = self.cfg['CONST']['eps_0']
        me = self.cfg['CONST']['me']
        mi = self.cfg['CONST']['mi']
        qe = self.cfg['CONST']['qe']
        
       # --- B field direction & intensity (radial dependance) ---
        theta_B = self.cfg['PLASMA']['theta_B_rad']
        phi_B = self.cfg['PLASMA']['phi_B_rad']
        bx = np.sin(phi_B)
        by = np.cos(phi_B) * np.sin(theta_B)
        bz = np.cos(phi_B) * np.cos(theta_B)

        # --- Constant B field (T) ---
        B_tot = B0 # * (R0/(R_ant - x_in_plasma))  # type = ngsolve.Coefficient.Function
    
       # --- Cyclotron frequency (rad/s) ---
        self.Om_ce = qe * B_tot / me
        self.Om_ci = qe * B_tot / mi
    
       # --- Plasma density profile (m-3) & ions and electrons plasma pulsations ---
        n_e = density_func(self.x, self.z)  # type = float (constant density)
        print(f'ne = {n_e}, type(ne) = {type(n_e)}')
        self.w_pe2 = (n_e * qe**2) / (me * eps_0)
        self.w_pi2 = (n_e * qe**2) / (mi * eps_0)

        # --- General Stix tensor elements: type = float ---
        self.S = 1 - self.w_pe2/(omega_wave**2 - self.Om_ce**2) - self.w_pi2/(omega_wave**2 - self.Om_ci**2)
        self.P = 1 - self.w_pe2/omega_wave**2 - self.w_pi2/omega_wave**2
        self.D = - self.Om_ce * self.w_pe2/(omega_wave*(omega_wave**2 - self.Om_ce**2)) + self.Om_ci * self.w_pi2/(omega_wave*(omega_wave**2 - self.Om_ci**2))
        Q_stix = self.P - self.S

        self.K_xx = self.S*(1 - bx**2) + self.P*bx**2
        self.K_xy = 1j*self.D*bz + Q_stix*bx*by
        self.K_xz = -1j*self.D*by + Q_stix*bx*bz
        
        self.K_yx = -1j*self.D*bz + Q_stix*by*bx
        self.K_yy = self.S*(1 - by**2) + self.P*by**2
        self.K_yz = 1j*self.D*bx + Q_stix*by*bz
        
        self.K_zx = 1j*self.D*by + Q_stix*bz*bx
        self.K_zy = -1j*self.D*bx + Q_stix*bz*by
        self.K_zz = self.S*(1 - bz**2) + self.P*bz**2
        
        # --- Matrix format for Stix Tensor for NGSolve ---
        self.K_tensor = CoefficientFunction(
            (self.K_xx, self.K_xy, self.K_xz,
             self.K_yx, self.K_yy, self.K_yz,
             self.K_zx, self.K_zy, self.K_zz), dims=(3,3)
        )


# =====================================================================
# 3. 3D VECTOR WEAK FORM SOLVER (Standard Ex, Ey, Ez Basis)
# =====================================================================
    def solve_helmholtz_Hcurl_2D_pml(self, mesh, cfg):
        '''
        Solves the Weak Form using standard (Ex, Ey, Ez) coordinate mapping.
        E_3D[0] = Radial (x)
        E_3D[1] = Poloidal (y)
        E_3D[2] = Toroidal (z)
        '''
        # --- 1. Function Space Definition --- 
        dirichlet_bnds = "bottom_source|left_wall_pec|right_wall_pec|top_wall_pec|bottom_wall_pec"
        fes_plane = HCurl(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        self.fes = fes_plane * fes_outplane
        print(f'#DoFs = {self.fes.ndof} (= number of mesh points).')

        # --- 2. PML Stretching Functions ---
        Lx_plasma, Lz_plasma = self.cfg['DOMAIN']['Lx_plasma'], self.cfg['DOMAIN']['Lz_plasma']
        Lx_pml, Lz_pml = self.cfg['DOMAIN']['Lx_pml'], self.cfg['DOMAIN']['Lz_pml']
        Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_r, Sz_im, pz = self.cfg['PML']['Sz_r'], self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']
        
        # Radial Stretching (+1j for Backward Wave dampening)
        Stretch_x = 1.0 + (Sx_r - 1.0 + 1j * Sx_im) * \
                    IfPos(self.x - Lx_plasma, ((self.x - Lx_plasma) / Lx_pml)**px, 0.0)
            
        # Toroidal Stretching (-1j for Forward Wave dampening)
        Stretch_z = 1.0 + (Sz_r - 1.0 - 1j * Sz_im) * \
                    IfPos(-self.z, (-self.z / Lz_pml)**pz, \
                    IfPos(self.z - Lz_plasma, ((self.z - Lz_plasma) / Lz_pml)**pz, 0.0))

        # --- 3. Mapped Tensors in (Ex, Ey, Ez) Basis ---
        self.pml_tensor = CF((
            Stretch_z / Stretch_x, 0.0,                   0.0, 
            0.0,                   Stretch_x * Stretch_z, 0.0, 
            0.0,                   0.0,                   Stretch_x / Stretch_z
        ), dims=(3,3))

        # eps_eff = det(Lambda) * Lambda^-1 * K_tensor * Lambda^-T
        self.eff_eps_tensor = CF((
            self.K_xx * (Stretch_z / Stretch_x), self.K_xy * Stretch_z,           self.K_xz, 
            self.K_yx * Stretch_z,               self.K_yy * (Stretch_x * Stretch_z), self.K_yz * Stretch_x, 
            self.K_zx,                           self.K_zy * Stretch_x,           self.K_zz * (Stretch_x / Stretch_z)
        ), dims=(3,3))

        # --- 4. Smoothed Source Windowing ---
        z_start, L_aperture = 0.0, self.cfg['DOMAIN']['Lz_plasma']
        z_end = z_start + L_aperture
        
        window_func = IfPos(self.z - z_start,
                            IfPos(z_end - self.z, sin(pi * (self.z - z_start) / L_aperture)**2, 0.0), 0.0)

        k_z = self.k0_vacuum * self.cfg['WAVE']['n_para']
        wave_phase = exp(-1j * k_z * self.z)
        E0 = self.cfg['WAVE']['E_inc']
        
        # FIXED: Exciting Toroidal Ez-field (Index 2) to launch the Slow Wave
        self.E_inc_cf = CF((0.0, 0.0, E0 * window_func * wave_phase))

        # --- 5. Field Initialization & Dirichlet Mapping ---
        E_field = GridFunction(self.fes)
        
        # E_plane contains (Ex, Ez), so we assign indices 0 and 2
        E_field.components[0].Set(CF((self.E_inc_cf[0], self.E_inc_cf[2])), 
                                  BND, definedon=self.mesh.Boundaries("bottom_source"))
        
        # E_outplane contains Ey, so we assign index 1
        E_field.components[1].Set(self.E_inc_cf[1], 
                                  BND, definedon=self.mesh.Boundaries("bottom_source"))

        # --- 6. Weak Form Assembly (User's Exact Math) ---
        E_plane, E_outplane = self.fes.TrialFunction()
        v_plane, v_outplane = self.fes.TestFunction()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1]))

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

        a = BilinearForm(self.fes)
        a += (self.pml_tensor * curl_E_3D * curl_v_3D - \
              self.k0_vacuum**2 * self.eff_eps_tensor * E_3D * v_3D) * dx
        
        with TaskManager():
            a.Assemble()
            f = LinearForm(self.fes)
            f.Assemble()
    
            print("--- Solving the 3D vector linear system ---")
            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * E_field.vec
    
            inv = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            E_field.vec.data += inv * res
            
        self.E_field = E_field
        
        # --- 7. Gamma Reflection Coefficient Computation ---
        # Note: Unpacking adjusted for the new basis
        E_xz, Ey = self.E_field.components[0], self.E_field.components[1]
        Ex, Ez = E_xz[0], E_xz[1]                
        
        E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))

        z_mid = self.Lz_plasma / 2.0
        x_vals = np.linspace(self.Lx_plasma * 0.25, self.Lx_plasma * 0.75, 500)
        mips_x = mesh(x_vals, np.full_like(x_vals, z_mid))
        E_vals_x = np.array(E_tot_norm(mips_x)).real
            
        SWR_Radial = max(np.max(E_vals_x) / np.max([np.min(E_vals_x), 1e-12]), 1.000001)
        Gamma_E_Radial = (SWR_Radial - 1.0) / (SWR_Radial + 1.0)
            
        x_eval_z = 0.1 * self.Lx_plasma
        z_eval = np.linspace(self.Lz_plasma * 0.25, self.Lz_plasma * .75, 500)
        mips_z = mesh(np.full_like(z_eval, x_eval_z), z_eval)
        E_vals_z = np.array(E_tot_norm(mips_z)).real

        SWR_Toroidal = max(np.max(E_vals_z) / max(np.min(E_vals_z), 1e-12), 1.0001)
        Gamma_E_Toroidal = (SWR_Toroidal - 1.0) / (SWR_Toroidal + 1.0)
            
        print(f"  --> Success | Gamma_Radial: {Gamma_E_Radial:.2e} | Gamma_Toroidal: {Gamma_E_Toroidal:.2e}")
        print('--- System solved ---')
        
        return self.E_field, self.fes.ndof
