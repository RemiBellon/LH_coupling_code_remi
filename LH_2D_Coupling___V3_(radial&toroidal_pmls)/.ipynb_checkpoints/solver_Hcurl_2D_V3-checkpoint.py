
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


    def compute_n_perp_plus_minus(self):
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
        
        rect_plasma = occ.MoveTo(0, 0).Rectangle(Lx_plasma, Lz_plasma).Face()
        rect_pml_radial = occ.MoveTo(Lx_plasma, 0).Rectangle(Lx_pml, Lz_plasma).Face()
        rect_pml_toroidal_left = occ.MoveTo(0, -Lz_pml).Rectangle(Lx_plasma, Lz_pml).Face()
        rect_pml_toroidal_right = occ.MoveTo(0, Lz_plasma).Rectangle(Lx_plasma, Lz_pml).Face()
        rect_pml_corner_radial_left = occ.MoveTo(Lx_plasma, -Lz_pml).Rectangle(Lx_pml, Lz_pml).Face()
        rect_pml_corner_radial_right = occ.MoveTo(Lx_plasma, Lz_plasma).Rectangle(Lx_pml, Lz_pml).Face()

        rect_plasma.edges.Min(occ.X).name = "bottom_source"
        rect_pml_toroidal_left.edges.Min(occ.Y).name = "left_wall"
        rect_pml_corner_radial_left.edges.Min(occ.Y).name = "left_wall"

        rect_pml_toroidal_right.edges.Max(occ.Y).name = "right_wall"
        rect_pml_corner_radial_right.edges.Max(occ.Y).name = "right_wall"

        rect_pml_corner_radial_left.edges.Max(occ.X).name = "top_wall_pec"
        rect_pml_radial.edges.Max(occ.X).name = "top_wall_pec"
        rect_pml_corner_radial_right.edges.Max(occ.X).name = "top_wall_pec"


        rect_pml_toroidal_left.edges.Min(occ.X).name = "bottom_wall"
        rect_pml_toroidal_right.edges.Min(occ.X).name = "bottom_wall"


        domain = occ.Glue([rect_plasma, rect_pml_radial, rect_pml_toroidal_left, rect_pml_corner_radial_left, rect_pml_corner_radial_right, rect_pml_toroidal_right])
        geo = occ.OCCGeometry(domain, dim=2)

                
        # --- 3. Wavelengths and Scaling ---
        n_perp_p, n_perp= compute_n_perp_plus_minus(self)        
        lambda0_vacuum = self.c0 / self.freq_LH
        lambda_perp_SW = lambda0_vacuum / n_perp_p
        
        # The mesh MUST resolve the smallest wavelength in the system (Slow Wave)
        self.h_max = lambda_perp_SW / self.cfg['DOMAIN']['n_resol_per_wlgth']
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.h_max)) 
        
        # --- 5. Consistency Check Output ---
        print(f"\n[MESH BUILDER] :")
        print(f"  --> SW n_perp_p : {n_perp_p:.5e}, n_perp_m = {n_perp_m:.5e}")
        print(f"  --> SW lambda_perp : {lambda_perp_SW:.5e} m, lambda_para = {lambda_para:.5e} m")
        print(f"  --> Lx_plm = {self.Lx_pml:.5e} m, Lx_tot = {self.Lx_tot:.5e} m")
        print(f"  --> Lz_exact : {self.Lz_exact:.5e} m")
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
        dirichlet_bnds = "bottom_source|left_wall|right_wall|top_wall_pec"
        fes_plane = HCurl(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        fes = Periodic(fes_plane) * Periodic(fes_outplane)
        print(f'#DoFs = {fes.ndof} (= number of mesh points).')
        
        # Define the E vector components:
        (E_plane, E_outplane), (v_plane, v_outplane) = fes.TnT()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1])) 

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

        # --- PML: ---
            # Radial Stretching
        dist_x = IfPos(self.x - self.Lx_plasma, self.x -self.Lx_plasma, 0.0)
        stretch_x = (self.cfg['PML'].get(['Sx_r'], 1.0) + 1j*self.cfg['PML'].get['Sx_im']) * (dist_x/Lx_pml)**self.cfg['PML']['px_exponent']
        Sx = 1.0 + stretch_x

            # Toroidal Stretching
        dist_z_left = IfPos(self_z - self.Lz_plasma, self.z - self.Lz_plasma, 0.0)
        dist_z_right = IfPos(-self_z, -self_z, 0.0)
        dist_z = dist_z_left + dist_z_right
        stretch_z = ((self.cfg['PML']['Sz_r'] + 1j*self.cfg['PML']['Sz_im'])) * (dist_z/Lz_pml)**self.cfg['PML']['pz_exponent']
        Sz = 1.0 + stretch_z

        pml_tensor = CF((Sz/Sx, 0.0, 0.0, 
                         0.0, Sx*Sz, 0.0, 
                         0.0, 0.0, Sx/Sz), dims=(3,3))
        inv_pml_tensor = CF((Sx/Sz, 0.0, 0.0, 
                         0.0, 1/(Sx*Sz), 0.0, 
                         0.0, 0.0, Sz/Sx), dims=(3,3))

        eff_permittivity_tensor = pml_tensor * self.K_tensor
        
        # --- Weak Form expression: --- 

        a = BilinearForm(fes)
        a += (inv_pml_tensor * curl_E_3D * curl_v_3D - self.k0_vacuum**2 * eff_permittivity_tensor * E_3D * v_3D) * dx
    
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
        z_mid = self.Lz_exact / 2.0
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

        print(f'--- SWR = {SWR:.4f} ---')
        print(f'coeff gamma =  {(SWR - 1.0) / (SWR + 1.0)}') 
        print('--- System solved ---')
        return gfu, fes.ndof
