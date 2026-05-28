'''
Solver_2DHcurl_1DH1 is a class that gather the functions to:
    - Build the mesh (cf. build_mesh_with_PMLs for more infos) based on 2D box sizes (plasma & pmls) in config_dict.py (cfg)
    - Build the physics (cf. build_physics) based on B field, ne, omega_LH (specified in config_dict.py) and Stix cold plasma approx in generalized cartesian coordinates.
    - Initialize and solve the Helmholtz wave equation on cartesian mesh (cf. solve_helmholtz_2DHcurl_1DH1_with_pml)
'''

import netgen.occ as occ
from ngsolve import *
import numpy as np

class LHCouplingSolver_2DHcurl_1DH1:
    def __init__(self, config_dict, mode="FULL_2D"):
        self.cfg = config_dict          # type = dict ==> dictionnary of dictionnaries with physics (wave, plasma) and geometry (domain, pmls) values
        self.mode = mode                # VACUUM, RADIAL_ONLY, FULL_2D
        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                      # In the context y is out of 2D plane direction. In 2D only the plane (xOz) is describe.   

        self.compute_physics_parameters()
    
    def compute_physics_parameters(self) -> None:
        self.omega_LH, self.k0, self.B0, self.n_para = self.cfg['WAVE']['omega_LH'], self.cfg['WAVE']['k0'], self.cfg['PLASMA']['B0'], self.cfg['WAVE']['n_para']
        self.qe, self.me, self.mi, self.eps0 = self.cfg['CONST']['qe'], self.cfg['CONST']['me'], self.cfg['CONST']['mi'], self.cfg['CONST']['eps0']
        
        if self.mode == "VACUUM":
            self.ne_constant = 0.0
        else: 
            self.ne_constant = self.cfg['PLASMA']['ne_constant']

        self.w_pe2 = (self.ne_constant * self.qe**2) / (self.me * self.eps0)
        self.w_pi2 = (self.ne_constant * self.qe**2) / (self.mi * self.eps0)
        self.Om_ce = (self.qe * self.B0) / self.me
        self.Om_ci = (self.qe * self.B0) / self.mi
        
        if self.mode == "VACUUM":
            self.D = 0.0
        else:
            self.D = -(self.Om_ce * self.w_pe2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ce**2)) + \
                      (self.Om_ci * self.w_pi2)/(self.omega_LH*(self.omega_LH**2 - self.Om_ci**2))

        self.S = 1 - self.w_pe2/(self.omega_LH**2 - self.Om_ce**2) - self.w_pi2/(self.omega_LH**2 - self.Om_ci**2)
        self.P = 1 - self.w_pe2/self.omega_LH**2 - self.w_pi2/self.omega_LH**2
    
        # 4. Dispersion Relation
        self.B_stix = (self.S + self.P)*self.n_para**2 - (self.S**2 - self.D**2) - self.P*self.S
        self.C_stix = self.P * (self.n_para**2 - (self.S + self.D)) * (self.n_para**2 - (self.S - self.D))
        
        delta = max(0.0, self.B_stix**2 - 4*self.S*self.C_stix)
        n_perp_sq_p = (-self.B_stix + np.sqrt(delta)) / (2*self.S)
        n_perp_sq_m = (-self.B_stix - np.sqrt(delta)) / (2*self.S)
        
        # Store as complex to natively handle evanescent (vacuum) states
        self.n_perp_p = np.sqrt(complex(n_perp_sq_p))
        self.n_perp_m = np.sqrt(complex(n_perp_sq_m))
    
    # =====================================================================
    # MESH GENERATION (Plasma + PML Domains)
    # =====================================================================
    def build_mesh_with_PMLs(self) -> None:
        '''
            Create the mesh object for C++ NGSolve solver: 
                - Recover Domain sizes from cfg_dict.py
                - Discretize the 2D Domain in Plasma & PMLs areas (manage radial PML only or Radial + Toroidal PMLs)
                - Define the external edges to set up boundary conditions (PEC, perdiodic, wave launcher) in solve_Helmholtz_2DHcurl_1DH1 
                - Glue all areas and set the mesh resolution based on the smallest wavelength (lambda_perp_SW, lambda_perp_FW, lambda_para)
                - GenerateMesh and print phiscics values to checkout  
        '''
        # Extract Domains sizes from cfg dict:
        self.Lx_plasma, self.Lz_plasma = self.cfg['DOMAIN']['Lx_plasma'], self.cfg['DOMAIN']['Lz_plasma']
        self.Lx_pml, self.Lz_pml = self.cfg['DOMAIN']['Lx_pml'], self.cfg['DOMAIN']['Lz_pml']
        self.Lx_tot, self.Lz_tot = self.cfg['DOMAIN']['Lx_tot'], self.cfg['DOMAIN']['Lz_tot']
        # Define every plasma and pmls areas = define rectangles from the bottom left corner and (x,z) sizes:
        # --- 1. Dynamic Geometry Assembly ---
        rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
        rect_plasma.edges.Min(occ.X).name = "bottom_source"

        if self.mode == "RADIAL_ONLY":
            rect_pml_radial = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
            rect_pml_radial.edges.Max(occ.X).name = "top_wall_pec"
            
            # Identify Periodic Boundaries
            edge_plasma_left = rect_plasma.edges.Min(occ.Y)
            edge_plasma_left.name = "plasma_left_periodic"
            edge_plasma_right = rect_plasma.edges.Max(occ.Y)
            edge_plasma_right.name = "plasma_right_periodic"
            edge_pml_left = rect_pml_radial.edges.Min(occ.Y)
            edge_pml_left.name = "pml_left_periodic"
            edge_pml_right = rect_pml_radial.edges.Max(occ.Y)
            edge_pml_right.name = "pml_right_periodic"
            
            edge_plasma_left.Identify(edge_plasma_right, "plasma_periodic")
            edge_pml_left.Identify(edge_pml_right, "pml_periodic")
            domain = occ.Glue([rect_plasma, rect_pml_radial])

        else: # FULL_2D or VACUUM (with toroidal PMLs)
            rect_pml_radial = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
            rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_corner_rad_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            rect_pml_corner_rad_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma).Rectangle(self.Lx_pml, self.Lz_pml).Face()

            # Assign PEC boundaries
            rect_pml_toroidal_left.edges.Min(occ.Y).name = "left_wall_pec"
            rect_pml_corner_rad_left.edges.Min(occ.Y).name = "left_wall_pec"
            rect_pml_toroidal_right.edges.Max(occ.Y).name = "right_wall_pec"
            rect_pml_corner_rad_right.edges.Max(occ.Y).name = "right_wall_pec"
            
            rect_pml_corner_rad_left.edges.Max(occ.X).name = "top_wall_pec"
            rect_pml_radial.edges.Max(occ.X).name = "top_wall_pec"
            rect_pml_corner_rad_right.edges.Max(occ.X).name = "top_wall_pec"

            rect_pml_toroidal_left.edges.Min(occ.X).name = "bottom_wall_pec"
            rect_pml_toroidal_right.edges.Min(occ.X).name = "bottom_wall_pec"

            domain = occ.Glue([rect_plasma, rect_pml_radial, rect_pml_toroidal_left, 
                               rect_pml_corner_rad_left, rect_pml_corner_rad_right, rect_pml_toroidal_right])

        geo = occ.OCCGeometry(domain, dim=2)
                
        # --- Intelligent Scaling (Resolving Infinite Wavelengths) ---      
        lambda_para = self.cfg['WAVE']['lambda0'] / self.n_para
        
        # Prevent meshing collapse in evanescent vacuum states by bounding the meshing index
        n_meshing = max(np.abs(self.n_perp_p.real), self.n_para, 1.0)
        lambda_meshing = self.cfg['WAVE']['lambda0'] / n_meshing
        
        self.h_max = lambda_meshing / self.cfg['DOMAIN']['n_resol_per_wlgth']
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.h_max))

        print(f"\n[MESH BUILDER - {self.mode}] :")
        print(f"  --> SW n_perp_p : {self.n_perp_p:.5e}, n_perp_m = {self.n_perp_m:.5e}")
        print(f"  --> Effective meshing index used : {n_meshing:.5e}")
        print(f"  --> h_max resolution : {self.h_max:.5e} m")
        
        return self.mesh
    
# =====================================================================
# PHYSICS IMPLEMENTATION - STIX TENSOR, B FIELD, etc...
# =====================================================================
    def build_physics_Stix_B_field(self) -> None:
        '''
        Function to gather every needed physical parameters to build the Stix tensor in cold plasma approximation
        that contain all the plasma/wave physics and geometry:
            - Compute arbitrary B field components relative to cartesian coordinates 
            - Compute every general Stix tensor elements 
            and return it as a native NGSolve CoefficientFunction 
        '''

        theta_B, phi_B = 0.0, 0.0
        bx = np.sin(phi_B)
        by = np.cos(phi_B) * np.sin(theta_B)
        bz = np.cos(phi_B) * np.cos(theta_B)
    
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
        
        self.K_tensor = CF((self.K_xx, self.K_xy, self.K_xz,
                            self.K_yx, self.K_yy, self.K_yz,
                            self.K_zx, self.K_zy, self.K_zz), dims=(3,3))

# =====================================================================
# SOLVE HELMHOLTZ 3D IN 2D BOX DOMAIN
# =====================================================================
    def solve_helmholtz_2DHcurl_1DH1_with_pml(self, mesh):
        '''
        Solves the Weak Form using standard (Ex, Ey, Ez) coordinate mapping: (Ex = E_3D[0] = Radial, Ey = E_3D[1] = Poloidal, Ez = E_3D[2] = Toroidal)
            - Set Hcurl finite element space (fes) for in plane E field components (Ex, Ez): Hcurl compute E field and needles between meshpoints
            H1 (fes) compute out of 2D plane E field component (Ey) and force the field vector continuity in every space directions
            - 
        '''

        if self.mode == "RADIAL_ONLY":
            dirichlet_bnds = "top_wall_pec"

        else: dirichlet_bnds = "left_wall_pec|right_wall_pec|top_wall_pec|bottom_wall_pec"
        fes_plane = HCurl(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=2, complex=True, dirichlet=dirichlet_bnds)
        
        if self.mode == "RADIAL_ONLY":
            fes_plane, fes_outplane = Periodic(fes_plane), Periodic(fes_outplane)
            
        self.fes = fes_plane * fes_outplane
        print(f'#DoFs = {self.fes.ndof} (= number of mesh points).')

        # --- Dynamic PML Stretching ---
        Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_r, Sz_im, pz = self.cfg['PML']['Sz_r'], self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']
        
        Stretch_x = 1.0 + (Sx_r - 1.0 + 1j * Sx_im) * \
                    IfPos(self.x - self.Lx_plasma, ((self.x - self.Lx_plasma) / self.Lx_pml)**px, 0.0)
            
        if self.mode == "RADIAL_ONLY":
            Stretch_z = CF(1.0)
        else:
            Stretch_z = 1.0 + (Sz_r - 1.0 - 1j * Sz_im) * \
                        IfPos(-self.z, (-self.z / self.Lz_pml)**pz, \
                        IfPos(self.z - self.Lz_plasma, ((self.z - self.Lz_plasma) / self.Lz_pml)**pz, 0.0))

        self.pml_tensor = CF((Stretch_z / Stretch_x, 0.0, 0.0, 
                              0.0, Stretch_x * Stretch_z, 0.0, 
                              0.0, 0.0, Stretch_x / Stretch_z), dims=(3,3))

        self.eff_eps_tensor = CF((
            self.K_xx * (Stretch_z / Stretch_x), self.K_xy * Stretch_z,           self.K_xz, 
            self.K_yx * Stretch_z,               self.K_yy * (Stretch_x * Stretch_z), self.K_yz * Stretch_x, 
            self.K_zx,                           self.K_zy * Stretch_x,           self.K_zz * (Stretch_x / Stretch_z)
        ), dims=(3,3))

        # --- Vector Assembly ---
        self.E_field = GridFunction(self.fes)
        
        E_plane, E_outplane = self.fes.TrialFunction()
        v_plane, v_outplane = self.fes.TestFunction()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1]))

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

        # --- Weak Form Assembly with ROBIN PORT BOUNDARY ---
        # Magnetic Surface Flux injection.
        a = BilinearForm(self.fes)
        a += (self.pml_tensor * curl_E_3D * curl_v_3D - \
              self.k0**2 * self.eff_eps_tensor * E_3D * v_3D) * dx
        
        # Transparent Absorbing Port (Removes back-reflections)
        k_x = self.k0 * self.n_perp_p
        # E_Plane_dot_v_Plane = InnerProduct(E_plane.Trace(), v_plane.Trace())
        E_Plane_dot_v_Plane = E_plane.Trace()[0] * v_plane.Trace()[0] + E_plane.Trace()[1] * v_plane.Trace()[1]
        a += 1j * k_x * (E_Plane_dot_v_Plane) * ds("bottom_source")
        
        with TaskManager():
            a.Assemble()
            
            # Incident Wave Injection Port
            f = LinearForm(self.fes)
            E0 = self.cfg['WAVE']['E_inc']
            k_z = self.k0 * self.n_para
            E_inc_z = E0 * exp(-1j * k_z * self.z) # Pure plane wave!
            
            E_inc_vec = CF((0.0, E_inc_z))
            # print(f'E_inc_vec type: {type(E_inc_vec)}, E_inc_tan = {E_inc_vec}')
            # print(f'v_plane.Trace() type: {type(v_plane.Trace())}, v_plane.Trace() = {v_plane.Trace()}')

            power_flux = E_inc_vec[0] * v_plane.Trace()[0] + E_inc_vec[1] * v_plane.Trace()[1]           
     
            # print('linear form assemble')
            f += 2j * k_x * power_flux * ds("bottom_source")
            f.Assemble()
    
            print("--- Solving the 3D vector linear system ---")
            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * self.E_field.vec
            inv = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            self.E_field.vec.data += inv * res
            
        # --- Gamma Reflection Computation (Remains Unchanged) ---
        E_xz, Ey = self.E_field.components[0], self.E_field.components[1]
        Ex, Ez = E_xz[0], E_xz[1]                
        E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))

        # Radial SWR
        z_mid = self.Lz_plasma / 2.0
        x_vals = np.linspace(self.Lx_plasma * 0.25, self.Lx_plasma * 0.75, 500)
        mips_x = mesh(x_vals, np.full_like(x_vals, z_mid))
        E_vals_x = np.array(E_tot_norm(mips_x)).real
        SWR_Radial = max(np.max(E_vals_x) / np.max([np.min(E_vals_x), 1e-12]), 1.000001)
        Gamma_E_Radial = (SWR_Radial - 1.0) / (SWR_Radial + 1.0)
            
        # Toroidal SWR
        x_eval_z = 0.1 * self.Lx_plasma
        z_eval = np.linspace(self.Lz_plasma * 0.25, self.Lz_plasma * .75, 500)
        mips_z = mesh(np.full_like(z_eval, x_eval_z), z_eval)
        E_vals_z = np.array(E_tot_norm(mips_z)).real
        SWR_Toroidal = max(np.max(E_vals_z) / max(np.min(E_vals_z), 1e-12), 1.0001)
        Gamma_E_Toroidal = (SWR_Toroidal - 1.0) / (SWR_Toroidal + 1.0)
            
        print(f"  --> Success | Gamma_Radial: {Gamma_E_Radial:.2e} | Gamma_Toroidal: {Gamma_E_Toroidal:.2e}")
        
        return self.E_field, self.fes.ndof