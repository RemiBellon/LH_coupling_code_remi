
import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time 
from pathlib import Path

# directory path to save mesh geometry data
mesh_save_dir = Path("/home/remi/Perso/Stage/M2_IRFM/Codes/LH_2D_Coupling___V3/Meshes")
mesh_save_dir.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 1. MESH GENERATION (Plasma Domain + PML Domain)
# =====================================================================
class LHCouplingSolver_Hcurl3D:
    def __init__(self, config_dict):
        self.cfg = config_dict          # type = dict ==> dict with physics and geometry values
        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                      # In the context y is vertical direction. In 2D only the plane (xOz) is describe. 
                                        # ==> So we change the notation from y to z (because z is not known by NGSolve as space coords)  

    def build_mesh_with_PMLs(self) -> None:
        '''
        Function to set the meshgrid size and shape adding PMLs.
            - Set the mesh size Lx_tot, and compute the exact Lz size to fit as a multiple of lambda_z in z direction.
        '''
        # --- Mesh size in radial (x axis) direction: --- 
        self.Lx_plasma = self.cfg['DOMAIN']['Lx_plasma'] 
        self.Lx_pml = self.cfg['DOMAIN']['Lx_pml']
        self.Lx_tot = self.cfg['DOMAIN']['Lx_tot']
        
        # --- Mesh size in toroidal (z axis) direction: ---
        # To fit correctly with toroidal periodic conditions: 
        # the z size of the mesh must be multiple of the z-oriented wavelength  
        self.Lz_plasma_approx = self.cfg['DOMAIN']['Lz_plasma_approx']
        self.n_para = self.cfg['WAVE']['n_para']
        self.freq_LH = self.cfg['WAVE']['freq_LH']
        self.c0 = self.cfg['CONST']['c0']
        
        lambda0_vacuum = self.c0 / self.freq_LH                         # lambda = c/f
        self.k0_vacuum = (2 * np.pi) / (lambda0_vacuum)                 # k = 2pi/lambda
        print(f'k0_vacuum = {self.k0_vacuum:.3f} m-1')
        if self.n_para != 0: 
            print(f'--- n_para = {self.n_para} is != 0 ---')
            lambda_z = (2 * np.pi) / (self.k0_vacuum * self.n_para )    # kz = k0 * n_para 
            print(f'lambda_z = {lambda_z:.3f} m')
            lambda_z_multiple = round(self.Lz_plasma_approx / lambda_z)
            print(f'lambda_z_multiple = {lambda_z_multiple:.3f}')
            self.Lz_exact = lambda_z_multiple * lambda_z
        elif self.n_para == 0:
            print(f'--- n_para = {self.n_para} is == 0 ---')
            self.Lz_exact = self.Lz_plasma_approx
        print(f'Then Lz_exact = {self.Lz_exact:.3f} m')

        # --- Mesh size in vertical (y axis) direction ---
        # In 2D mesh, to compute (ngsolve) curl function correctly in solver function,
        # we must have at least explicitly "2" elements in y direction with a distance = h_max (= mesh resolution)

        A = self.S
        B = (self.S + self.P)*self.n_para**2 - (self.S**2 - self.D**2) - self.P * self.S
        C = self.P * (self.n_para**2 - (self.S - self.D)) * (self.n_para**2 - (self.S - self.D))
        
        n_perp_plus = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
        n_perp_minus = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
        print(f'n_perp_plus = {n_perp_plus:.4f} ; n_perp_minus =  {n_perp_minus:.4f}')
        n_perp_max = max(n_perp_plus, n_perp_minus)
        print(f'n_perp_max = {n_perp_max:.4f} m-1')
        lambda_min = (2 * np.pi) / abs(self.k0_vacuum * n_perp_max)
        print(f'lambda_min = {lambda_min:.3f} m')
        
        self.n_resol_per_wlgth = self.cfg['DOMAIN']['n_resol_per_wlgth'] # Approx 10 pts/lambda
        self.h_max = lambda_min / self.n_resol_per_wlgth
        print(f'h_max = {self.h_max:.4f} m')
        
        self.Ly_2D = self.h_max # in m
        print(f'Ly_2D = {self.Ly_2D:.4f} m')
        print(f"--- Generating 2.5D Mesh ---")
        print('Lx_tot type = ', type(self.Lx_tot), 'Ly type = ', type(self.Ly_2D), 'Lz type = ', type(self.Lz_exact))
        print(f"Box: X={self.Lx_tot:.3f}m, Y={self.Ly_2D:.3f}m, Z={self.Lz_exact:.3f}m")


        # --- Create the box geometry using NGSolve native function ---
        # The plasma and PML boxes are create separately to play on both depth independantly
        plasma_box = occ.Box((0, 0, 0), (self.Lx_plasma, self.Ly_2D, self.Lz_exact))
        pml_box = occ.Box((self.Lx_plasma, 0, 0), (self.Lx_tot, self.Ly_2D, self.Lz_exact))
        
        # Name both boxes regions:
        plasma_box.mat("plasma")
        pml_box.mat("pml")
        plasma_box.faces.name = "plasma_region"
        pml_box.faces.name = "pml_region"

        # Glue the two boxes to make them share the same mesh nods where they're stuck together
        glued_shape = occ.Glue([plasma_box, pml_box])
        
        # Name the Faces for later boundary conditions: +/- 1e-6 to avoid boundary problem at exact surface position
        for f in glued_shape.faces:
            cx, cy, cz = f.center.x, f.center.y, f.center.z
            if cx < 1e-6: f.name = "left_source"
            elif cx > self.Lx_tot - 1e-6: f.name = "right_perf_el_cond"
            elif cz > self.Lz_exact - 1e-6: f.name = "top"
            elif cz < 1e-6: f.name = "bottom"
            elif cy > self.Ly_2D - 1e-6: f.name = "front"
            elif cy < 1e-6: f.name = "back"

        # Apply toroidal periodicity (z axis)
        for f_top in glued_shape.faces:
            if f_top.name == "top":
                for f_bot in glued_shape.faces:
                    if f_bot.name == "bottom" and abs(f_top.center.x - f_bot.center.x) < 1e-6 and abs(f_top.center.y - f_bot.center.y) < 1e-6:
                        f_top.Identify(f_bot, "periodic_z", occ.IdentificationType.PERIODIC)

        # Apply vertical periodicity (y axis) -> to force 2D Physics
        for f_front in glued_shape.faces:
            if f_front.name == "front":
                for f_back in glued_shape.faces:
                    if f_back.name == "back" and abs(f_front.center.x - f_back.center.x) < 1e-6 and abs(f_front.center.z - f_back.center.z) < 1e-6:
                        f_front.Identify(f_back, "periodic_y", occ.IdentificationType.PERIODIC)


        # --- Generate the Mesh --- 
        geo = occ.OCCGeometry(glued_shape)
        ngmesh = geo.GenerateMesh(maxh=self.h_max)
        mesh = Mesh(ngmesh)
        print('--- Mesh was generated ---')
        
        # Save and Export the mesh data as .vol file readable by netgen
        mesh_file_path = mesh_save_dir / "my_lh_mesh.vol"
        print(f"--- Saving mesh to: {mesh_file_path} ---")
        ngmesh.Save(str(mesh_file_path))
        
        # mesh.ngmesh.Save("my_mesh.vol")
        self.cfg['DOMAIN']['Lz_exact'] = self.Lz_exact 
        self.cfg['DOMAIN']['Ly_slice'] = self.Ly_2D
        return mesh
    
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
        Om_ce = qe * B_tot / me
        Om_ci = qe * B_tot / mi
    
       # --- Plasma density profile (m-3) & ions and electrons plasma pulsations ---
        n_e = density_func(self.x, self.z)  # type = float (constant density)
        print('type(ne) = ', type(n_e))
        w_pe2 = (n_e * qe**2) / (me * eps_0)
        w_pi2 = (n_e * qe**2) / (mi * eps_0)

       # --- General Stix tensor elements: type = float ---
        self.S = 1 - w_pe2/(omega_wave**2 - Om_ce**2) - w_pi2/(omega_wave**2 - Om_ci**2)
        self.P = 1 - w_pe2/omega_wave**2 - w_pi2/omega_wave**2
        self.D = Om_ce * w_pe2/(omega_wave*(omega_wave**2 - Om_ce**2)) + Om_ci * w_pi2/(omega_wave*(omega_wave**2 - Om_ci**2))
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
    def solve_helmholtz_Hcurl_3D_pml(self, mesh):
        '''
        Function to compute and solve the Weak Form:
            - Set the function math space: HCurl (native NGSolve) to compute E_field solution functions on mesh triangles (or rectangles) edges. 
            HCurl forces tangential components continuity but allow normal components jumps.
            - Set the PML expression based on Jacquot2013 method. Sr_Re to attenuate evanescent waves 
            and Sr_Im to attenuate incident waves.  
            - 
        '''
        # --- Function math space to solve wave equation: --- 
        base_fes = HCurl(mesh, order=3, complex=True, dirichlet="left_source|right_perf_el_cond")
        fes = Periodic(base_fes)
        print(f'#DoFs = {fes.ndof} (= number of mesh points).')
    
        # --- Jacquot 2013 Artificial PML Tensors ---
        Sr_Real = 1.0   # to attenuate evanescent waves
        Sr_Imag = -1.0  # Positive imaginary stretch for e^{ikx} convention
        pr = 2.0        # power degree of pml attenuation 
    
        norm_dist = (self.x - self.Lx_plasma) / self.Lx_pml # Normalise the position depth in the pml region in [0, 1]
        # IfPos condition is verified if and only if x - Lx_plasma > 0 ==> the radial coords is in pml region 
        # in the other the stretch function is 0 
        stretch_func = IfPos(self.x - self.Lx_plasma, (Sr_Real + 1j * Sr_Imag) * (norm_dist**pr), 0.0)
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
        self.E_field, v_test = fes.TnT() # Initialize the Trial & associated Test function for FEM
    
        a = BilinearForm(fes, symmetric=True)
        a += (mu_inv_tensor * curl(self.E_field) * curl(v_test) - self.k0_vacuum**2 * eff_eps_tensor * self.E_field * v_test) * dx
    
        with TaskManager():
            a.Assemble()
    
        # --- No source term within the solved domain: ---
        f = LinearForm(fes)
        f.Assemble()

        # E_y is 0.0. The phase varies along the z-axis.
        E_inc_amp = self.cfg['WAVE']['E_inc']
        kz_exact = self.k0_vacuum * self.n_para

        E_vector = CF((0.0, 0.0, E_inc_amp)) * exp(1j * kz_exact * self.z)
   
        # --- Create the grid function (= solution on the mesh) and inverse the matrix ,                   system --- 
        gfu = GridFunction(fes)
        gfu.Set(E_vector, definedon=mesh.Boundaries("left_source"))
    
        print("--- Solving the 3D vector linear system ---")
        res = f.vec.CreateVector()
        res.data = f.vec - a.mat * gfu.vec
    
        inv = a.mat.Inverse(freedofs=fes.FreeDofs())
        gfu.vec.data += inv * res

        print('--- System solved ---')
        return gfu, fes.ndof

# import config_2Dcoupling_V3 as cfg              # config = physical & simulation parameters 
# from solver_Hcurl_3D_V3 import *                # solver = FEM method + pmls
# import diagnostic_post_process_V3 as my_pp      # post_process = plotting functions

# solver = LHCouplingSolver_Hcurl3D(cfg.__dict__)
# solver.build_physics_Stix_B_field(lambda x_sym, z_sym: my_pp.create_density_profile(x_sym, z_sym, solver))
# mesh = solver.build_mesh_with_PMLs()
