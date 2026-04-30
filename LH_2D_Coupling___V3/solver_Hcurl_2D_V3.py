
import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time 
from pathlib import Path

# directory path to save mesh geometry data
mesh_save_dir = Path("/home/remi/Perso/Stage/M2_IRFM/Codes/LH_2D_Coupling___V3/Meshes")
print(f'mesh_save_dir = {mesh_save_dir}')
# mesh_save_dir = Path("/Home/RB286887/LH_coupling_code_remi/LH_2D_Coupling___V3/Meshes")
mesh_save_dir.mkdir(parents=True, exist_ok=True)

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

        Golant_acc_crit = 1 + self.w_pe2 / (self.Om_ce**2)
        print(f'--- n_para = {self.n_para} and Golant_acc_crit = {Golant_acc_crit} ---')

        self.freq_LH = self.cfg['WAVE']['freq_LH']
        self.c0 = self.cfg['CONST']['c0']
        
        lambda0_vacuum = self.c0 / self.freq_LH                             # lambda = c/f
        self.k0_vacuum = (2 * np.pi) / (lambda0_vacuum)                     # k = 2pi/lambda
        print(f'k0_vacuum = {self.k0_vacuum:.3f} m-1')
        if self.n_para != 0: 
            print(f'--- n_para = {self.n_para} is != 0 ---')
            lambda_z = (2 * np.pi) / abs(self.k0_vacuum * self.n_para )     # kz = k0 * n_para 
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
        C = self.P * (self.n_para**2 - (self.S + self.D)) * (self.n_para**2 - (self.S - self.D))
        
        n_perp_plus = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
        n_perp_minus = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
        print(f'n_perp_plus = {n_perp_plus:.4f} ; n_perp_minus =  {n_perp_minus:.4f}')
        n_perp_max = max(n_perp_plus, n_perp_minus)
        print(f'n_perp_max = {n_perp_max:.4f} m-1')
        lambda_min = (2 * np.pi) / abs(self.k0_vacuum * n_perp_max)
        print(f'lambda_min = {lambda_min:.3f} m')
        
        self.n_resol_per_wlgth = self.cfg['DOMAIN']['n_resol_per_wlgth'] # Approx 4 pts/lambda
        self.h_max = lambda_min / self.n_resol_per_wlgth
        print(f'h_max = {self.h_max:.4f} m')

        print(f"--- Generating 2.5D Mesh ---")
        print('Lx_tot type = ', type(self.Lx_tot), 'Lz type = ', type(self.Lz_exact))
        print(f"2D flat plane: X={self.Lx_tot:.3f}m, Z={self.Lz_exact:.3f}m")


        # --- Create the box geometry using NGSolve native function ---
        # The plasma and PML boxes are create separately to play on both depth independantly
        wp = occ.WorkPlane()
        plasma_face = wp.MoveTo(0,0).Rectangle(self.Lx_plasma, self.Lz_exact).Face()
        plasma_face.mat("plasma")
        plasma_face.faces.name = "plasma_region"

        
        pml_face = wp.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_exact).Face()
        pml_face.mat("pml")
        pml_face.faces.name = "pml_region"
        print('--- The facees are set ---')


        # Glue the two faces to make them share the same mesh nods where they're stuck together
        glued_shape = occ.Glue([plasma_face, pml_face])
        
        # Name the Faces for later boundary conditions: +/- 1e-6 to avoid boundary problem at exact surface position
        for e in glued_shape.edges:
            cx, cy = e.center.x, e.center.y
            if cx < 1e-6: e.name = "left_source"
            elif cx > self.Lx_tot - 1e-6: e.name = "right_perf_el_cond"
            elif cy > self.Lz_exact - 1e-6: e.name = "top"
            elif cy < 1e-6: e.name = "bottom"

        print('--- The faces are glued ---')
        # Apply toroidal periodicity (z axis)
        
        for ez_top in glued_shape.edges:
            if ez_top.name == "top":
                for ez_bot in glued_shape.edges:
                    if ez_bot.name == "bottom" and abs(ez_top.center.x - ez_bot.center.x) < 1e-6: #  and abs(f_top.center.y - f_bot.center.y) < 1e-6:
                        ez_top.Identify(ez_bot, "periodic_z", occ.IdentificationType.PERIODIC)
        print('--- z periodicity set ok ---')


        # --- Generate the Mesh --- 
        geo = occ.OCCGeometry(glued_shape, dim=2)
        ngmesh = geo.GenerateMesh(maxh=self.h_max)
        mesh = Mesh(ngmesh)
        print('--- Mesh was generated ---')
        
        # Save and Export the mesh data as .vol file readable by netgen
        mesh_file_path = mesh_save_dir / "my_lh_mesh.vol"
        print(f"--- Saving mesh to: {mesh_file_path} ---")
        ngmesh.Save(str(mesh_file_path))
        
        # mesh.ngmesh.Save("my_mesh.vol")
        self.cfg['DOMAIN']['Lz_exact'] = self.Lz_exact 
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
        fes_plane = HCurl(mesh, order=2, complex=True, dirichlet="left_source|right_perf_el_cond")
        fes_outplane = H1(mesh, order=2, complex=True, dirichlet="left_source|right_perf_el_cond")
        fes = Periodic(fes_plane) * Periodic(fes_outplane)
        print(f'#DoFs = {fes.ndof} (= number of mesh points).')
        
        # Define the E vector components:
        (E_plane, E_outplane), (v_plane, v_outplane) = fes.TnT()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1])) 

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_outplane), grad(E_plane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_outplane), grad(v_plane)[0] ))


        # --- Jacquot 2013 Artificial PML Tensors ---
        Sr_Real = 1.0   # to attenuate evanescent waves
        Sr_Imag = 1.0  # Positive imaginary stretch for e^{ikx} convention
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

        a = BilinearForm(fes, symmetric=True)
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

        print('--- System solved ---')
        return gfu, fes.ndof

# import config_2Dcoupling_V3 as cfg              # config = physical & simulation parameters 
# from solver_Hcurl_3D_V3 import *                # solver = FEM method + pmls
# import diagnostic_post_process_V3 as my_pp      # post_process = plotting functions

# solver = LHCouplingSolver_Hcurl3D(cfg.__dict__)
# solver.build_physics_Stix_B_field(lambda x_sym, z_sym: my_pp.create_density_profile(x_sym, z_sym, solver))
# mesh = solver.build_mesh_with_PMLs()
