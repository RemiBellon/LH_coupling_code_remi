
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

        # --- 3. Wavelengths and Scaling ---
        lambda0_vacuum = self.c0 / self.freq_LH
        lambda_perp_SW = lambda0_vacuum / n_perp_p
        
        if self.n_para != 0: 
            lambda_para = lambda0_vacuum / abs(self.n_para)
            # BRUTAL FIX: Force Lz to be exactly 1 parallel wavelength for periodicity
            self.Lz_plasma = 1.0 * lambda_para
            self.cfg['DOMAIN']['Lz_plasma'] = self.Lz_plasma
        else:
            self.Lz_plasma= self.cfg['DOMAIN']['Lz_plasma']

        # The mesh MUST resolve the smallest wavelength in the system (Slow Wave)
        lambda_min = lambda0_vacuum / max(n_perp_p, self.cfg['WAVE']['n_para'], 1.0)
        self.h_max = lambda_min / self.cfg['DOMAIN']['n_resol_per_wlgth']

        # --- 4. OCC Geometry Construction ---
        # Domain 1 : Plasma
        rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
        rect_plasma.edges.Min(occ.X).name = "left_source"
        rect_plasma.edges.Max(occ.X).name = "plasma_pml_interface"
        
        # Explicit Topological Identification for Periodicity (Plasma)
        edge_bot_plasma = rect_plasma.edges.Min(occ.Y)
        edge_top_plasma = rect_plasma.edges.Max(occ.Y)
        edge_bot_plasma.name = "bottom_periodic"
        edge_top_plasma.name = "top_periodic"
        # CORRECTED NOTATION: Just pass the edge and a unique string identifier
        edge_top_plasma.Identify(edge_bot_plasma, "periodic_plasma")
        
        rect_plasma.mat("plasma_region")

        # Domain 2 : PML
        rect_pml = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
        rect_pml.edges.Max(occ.X).name = "right_pml_wall"
        
        # Explicit Topological Identification for Periodicity (PML)
        edge_bot_pml = rect_pml.edges.Min(occ.Y)
        edge_top_pml = rect_pml.edges.Max(occ.Y)
        edge_bot_pml.name = "bottom_periodic_pml"
        edge_top_pml.name = "top_periodic_pml"
        # CORRECTED NOTATION
        edge_top_pml.Identify(edge_bot_pml, "periodic_pml")
        
        rect_pml.mat("pml_region")

        # Glue and build
        domain = occ.Glue([rect_plasma, rect_pml])
        geo = occ.OCCGeometry(domain, dim=2)
        ngmesh = geo.GenerateMesh(maxh=self.h_max)
        ngmesh = geo.GenerateMesh(maxh=self.h_max)
        
        self.mesh = Mesh(ngmesh)
        
        # --- 5. Consistency Check Output ---
        print(f"\n[MESH BUILDER] :")
        print(f"  --> SW n_perp_p : {n_perp_p:.5e}, n_perp_m = {n_perp_m:.5e}")
        print(f"  --> SW lambda_perp : {lambda_perp_SW:.5e} m, lambda_para = {lambda_para:.5e} m")
        print(f"  --> Lx_plm = {self.Lx_pml:.5e} m, Lx_tot = {self.Lx_tot:.5e} m")
        print(f"  --> Lz_plasma : {self.Lz_plasma:.5e} m")
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
# 3. 3D VECTOR WEAK FORM SOLVER (Jacquot 2013 Artificial Medium)
# =====================================================================
    def solve_helmholtz_Hcurl_2D_pml(self, mesh, cfg):
        '''
        Function to compute and solve the Weak Form:
            - Set the function math space: HCurl (native NGSolve) to compute E_field solution functions on mesh triangles (or rectangles) edges. 
            HCurl forces tangential components continuity but allow normal components jumps.
            - Set the PML expression based on Jacquot2013 method. Sr_Re to attenuate evanescent waves 
            and Sr_Im to attenuate incident waves.  
            - 
        '''
        # --- Function math space to solve wave equation: --- 
        fes_plane = HCurl(mesh, order=2, complex=True, dirichlet="left_source|right_pml_wall")
        fes_outplane = H1(mesh, order=2, complex=True, dirichlet="left_source|right_pml_wall")
        fes = Periodic(fes_plane) * Periodic(fes_outplane)
        print(f'#DoFs = {fes.ndof} (= number of mesh points).')
        
        # Define the E vector components:
        (E_plane, E_outplane), (v_plane, v_outplane) = fes.TnT()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1])) 

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))


        # --- Jacquot 2013 Artificial PML Tensors ---
        Sx_r = self.cfg['PML'].get('Sx_r', 1.0)
        Sx_im = self.cfg['PML'].get('Sx_im', 1.0)
        px = self.cfg['PML'].get('px', 2.0)
    
        norm_dist = (self.x - self.Lx_plasma) / self.Lx_pml # Normalise the position depth in the pml region in [0, 1]
        # IfPos condition is verified if and only if x - Lx_plasma > 0 ==> the radial coords is in pml region 
        # in the other the stretch function is 0 
        stretch_func = IfPos(self.x - self.Lx_plasma, (Sx_r + 1j * Sx_im) * (norm_dist**px), 0.0)
        # Stretch function along x-axis only
        Sx = 1.0 + stretch_func

        # 3x3 Dielectric Tensor (epsilon) from Jacquot2013
        eps_pml_tensor = CF((1/Sx, 0.0, 0.0,
                     0.0,  Sx,  0.0,
                     0.0,  0.0, Sx), dims=(3,3))

        # 3x3 Inverse Permeability Tensor (mu^-1) for Jacquot2013
        mu_inv_tensor = CF((Sx,   0.0,  0.0,
                        0.0, 1/Sx,  0.0,
                        0.0,  0.0, 1/Sx), dims=(3,3))
    
        eff_eps_tensor = eps_pml_tensor * self.K_tensor
        # --- Weak Form expression: --- 

        a = BilinearForm(fes)
        a += (mu_inv_tensor * curl_E_3D * curl_v_3D - self.k0_vacuum**2 * eff_eps_tensor * E_3D * v_3D) * dx
    
        with TaskManager():
            a.Assemble()
    
        # --- No source term within the solved domain: ---
        f = LinearForm(fes)
        f.Assemble()

        # E_y is 0.0. The phase varies along the z-axis.
        E_inc_amp = self.cfg['WAVE']['E_inc']
        kz_exact = self.k0_vacuum * self.n_para

        E_z_inc = E_inc_amp * exp(1j * kz_exact * self.z)
        E_vector = CF((0.0, E_z_inc))
        # --- Create the grid function (= solution on the mesh) and inverse the matrix ,                   system --- 
        gfu = GridFunction(fes)
        gfu.components[0].Set(E_vector, definedon=mesh.Boundaries("left_source"))
        gfu.components[1].Set(0.0 , definedon=mesh.Boundaries("left_source"))


    
        print("--- Solving the 3D vector linear system ---")
        res = f.vec.CreateVector()
        res.data = f.vec - a.mat * gfu.vec
    
        inv = a.mat.Inverse(freedofs=fes.FreeDofs())
        gfu.vec.data += inv * res

        # --- Gamma_refl_coeff computation ---
        #        # Area of Gamma_r computatio: 
        z_mid = self.Lz_plasma / 2.0
        x_eval = np.linspace(self.Lx_plasma * 0.25, self.Lx_plasma * 0.75, 1000)

        E_3D_full = CF((gfu.components[0][0], gfu.components[1], gfu.components[0][1]))
        E_vals = E_3D_full(mesh(x_eval, np.full_like(x_eval, z_mid)))
        
        Ex_abs = np.abs(E_vals[:, 0])
        Ey_abs = np.abs(E_vals[:, 1])
        Ez_abs = np.abs(E_vals[:, 2])
        E_tot_norm = np.sqrt(Ex_abs**2 + Ey_abs**2 + Ez_abs**2)

        E_max = np.max(E_tot_norm)
        E_min = np.min(E_tot_norm)
        SWR = max(E_max/max(E_min, 1e-12), 1.0001)

        # print(f'--- SWR = {SWR:.4f} ---')
        # print(f'coeff gamma =  {(SWR - 1.0) / (SWR + 1.0)}') 
        print('--- System solved ---')
        return gfu, fes.ndof
