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
    def __init__(self, config_dict, mode):
        self.cfg = config_dict          # type = dict ==> dictionnary of dictionnaries with physics (wave, plasma) and geometry (domain, pmls) values
        self.mode = mode                # VACUUM, RADIAL_ONLY, FULL_2D
        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                      # In the context y is out of 2D plane direction. In 2D only the plane (xOz) is describe.   

        self.n_para, self.n_perp_p, self.n_perp_m = self.compute_physics_parameters()
        print(f'In LHCoupling class: n_para: {self.n_para:.1f}, n_perp_p: {self.n_perp_p:.2e}, n_perp_m: {self.n_perp_m:.2e}')
    
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
        
        return self.n_para, self.n_perp_p, self.n_perp_m
    
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
        self.Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        
        # self.Lx_pml = 1.0 * (self.cfg['WAVE']['lambda0']/self.n_perp_p.real)
        self.cfg['DOMAIN']['Lx_pml'] = self.Lx_pml = 1.0 * (self.cfg['WAVE']['lambda0']/self.n_perp_p.real)
        # self.Lx_pml = self.cfg['DOMAIN']['Lx_pml']
        print(f'Lx_pml = {self.Lx_pml:.3e}m')
        self.cfg['DOMAIN']['Lx_tot'] = self.Lx_plasma + self.Lx_pml
        
        self.lambda0, self.n_para = self.cfg['WAVE']['lambda0'], self.cfg['WAVE']['n_para']
        self.lambda_para = self.lambda0/abs(self.n_para.real)

        if self.mode == "RADIAL_ONLY" or "FULL_2D" and self.n_para != 0:
            self.Lz_plasma = 3.0 * self.lambda_para
            self.cfg['DOMAIN']['Lz_plasma'] = self.Lz_plasma
        else: 
            self.Lz_plasma = self.cfg['DOMAIN']['Lz_plasma']
        
        # self.Lz_pml = self.cfg['DOMAIN']['Lz_pml']
        self.cfg['DOMAIN']['Lz_pml'] = self.Lz_pml = 1.0 * self.lambda_para 
        self.cfg['DOMAIN']['Lz_tot'] = self.Lz_tot = self.Lz_plasma + 2*self.Lz_pml
        print(f'Lz_plasma = {self.Lz_plasma:.2e}m, Lz_pml = {self.Lz_pml:.2e}m')
        print(f'lambda_para = {self.lambda_para:.2e}m')

        # Define every plasma and pmls areas = define rectangles from the bottom left corner and (x,z) sizes:
        # --- 1. Dynamic Geometry Assembly ---
        rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
        rect_plasma.edges.Min(occ.X).name = "bottom_source"

        # Plasma meshing resolution
        n_meshing = max(np.abs(self.n_perp_p.real), self.n_para, 1.0)
        lambda_meshing = self.cfg['WAVE']['lambda0'] / n_meshing
        self.h_max_plasma = lambda_meshing / self.cfg['DOMAIN']['n_resol_per_wlgth']
        print(f'h_max_plasma: {self.h_max_plasma:.2e}')

        if self.mode == "RADIAL_ONLY":
            # PLASMA DOMAIN:
            rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
            rect_plasma.edges.Min(occ.X).name = "bottom_source"
            rect_plasma.maxh = self.h_max_plasma

            edge_plasma_left = rect_plasma.edges.Min(occ.Y)
            edge_plasma_left.name = "plasma_left_periodic"
            edge_plasma_right = rect_plasma.edges.Max(occ.Y)
            edge_plasma_right.name = "plasma_right_periodic"
            edge_plasma_left.Identify(edge_plasma_right, "plasma_periodic", occ.IdentificationType.PERIODIC)

            rect_pml = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
            Sx_mag_max = np.sqrt((Sx_r)**2 + (Sx_im)**2)
            h_min_pml = lambda_meshing / (PPW_pml * Sx_mag_max)
            print(f'h_min_pml: {h_min_pml}')
            rect_pml.edges.Max(occ.X).maxh = h_min_pml
            rect_pml.edges.Max(occ.X).name = "top_wall_pec"
            
            edge_pml_left = rect_pml.edges.Min(occ.Y)
            edge_pml_right = rect_pml.edges.Max(occ.Y)
            edge_pml_left.name = "periodic_pml_left"
            edge_pml_right.name = "periodic_pml_right"
            edge_pml_left.Identify(edge_pml_right, 'periodic_pml', occ.IdentificationType.PERIODIC)

            # Glue Plasma and all PML layers together
            domain = occ.Glue([rect_plasma, rect_pml])

        else: # FULL_2D or VACUUM (with toroidal PMLs)
            self.Lz_metal = 0.15 * self.Lz_plasma
            self.Lz_plasma_src = self.Lz_plasma - (2 * self.Lz_metal)

            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            h_min_pml_radial = lambda_meshing / (PPW_pml * Sx_norm_max) 
            Sz_norm_max = np.sqrt(self.cfg['PML']['Sz_r']**2 + self.cfg['PML']['Sz_im']**2)
            h_min_pml_toroidal = lambda_meshing / (PPW_pml * Sz_norm_max)

            h_min_pml_corner = lambda_meshing / (PPW_pml * np.sqrt(Sx_norm_max**2 + Sz_norm_max**2))
            print('[Build Mesh:]')
            print(f'PPW_pml: {PPW_pml:.1f}, Sx_norm_max:{Sx_norm_max:.2e}, h_min_pml_radial: {h_min_pml_radial:.2e}')
            print(f'Sz_norm_max: {Sz_norm_max:.2e}, h_min_pml_toroidal: {h_min_pml_toroidal:.2e}, h_min_pml_corner: {h_min_pml_corner:.2e}')

            rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_metal).Face()
            rect_plasma_left.maxh = self.h_max_plasma
            rect_plasma_left.edges.Min(occ.X).name = "bottom_wall_pec"
            rect_plasma_src = occ.MoveTo(0, self.Lz_metal).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
            rect_plasma_src.maxh = self.h_max_plasma
            rect_plasma_src.edges.Min(occ.X).name = "bottom_source"
            rect_plasma_right = occ.MoveTo(0, self.Lz_metal + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_metal).Face()
            rect_plasma_right.maxh = self.h_max_plasma
            rect_plasma_right.edges.Min(occ.X).name = "bottom_wall_pec"

            rect_pml_radial = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
            rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_corner_rad_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            rect_pml_corner_rad_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            
            # Assign PEC boundaries
            rect_pml_toroidal_left.edges.Min(occ.Y).name = "left_wall_pec"
            rect_pml_toroidal_left.edges.Min(occ.Y).maxh = h_min_pml_toroidal
            rect_pml_corner_rad_left.edges.Min(occ.Y).name = "left_wall_pec"
            rect_pml_corner_rad_left.edges.Min(occ.Y).maxh = h_min_pml_corner
            rect_pml_toroidal_right.edges.Max(occ.Y).name = "right_wall_pec"
            rect_pml_toroidal_right.edges.Max(occ.Y).maxh = h_min_pml_toroidal
            rect_pml_corner_rad_right.edges.Max(occ.Y).name = "right_wall_pec"
            rect_pml_corner_rad_right.edges.Max(occ.Y).maxh = h_min_pml_corner

            
            rect_pml_corner_rad_left.edges.Max(occ.X).name = "top_wall_pec"
            rect_pml_corner_rad_left.edges.Max(occ.X).maxh = h_min_pml_corner
            rect_pml_radial.edges.Max(occ.X).name = "top_wall_pec"
            rect_pml_radial.edges.Max(occ.X).maxh = h_min_pml_radial
            rect_pml_corner_rad_right.edges.Max(occ.X).name = "top_wall_pec"
            rect_pml_corner_rad_left.edges.Max(occ.X).maxh = h_min_pml_corner


            rect_pml_toroidal_left.edges.Min(occ.X).name = "bottom_wall_pec"
            rect_pml_toroidal_left.edges.Min(occ.X).maxh = h_min_pml_toroidal
            rect_pml_toroidal_right.edges.Min(occ.X).name = "bottom_wall_pec"
            rect_pml_toroidal_left.edges.Min(occ.X).maxh = h_min_pml_toroidal

            domain = occ.Glue([rect_plasma_left, rect_plasma_src, rect_plasma_right, rect_pml_radial, rect_pml_toroidal_left, 
                               rect_pml_corner_rad_left, rect_pml_corner_rad_right, rect_pml_toroidal_right])

        geo = occ.OCCGeometry(domain, dim=2)
                

        self.mesh = Mesh(geo.GenerateMesh(maxh=self.h_max_plasma))

        print(f"\n[MESH BUILDER - {self.mode}] :")
        print(f"  --> SW n_perp_p : {self.n_perp_p:.5e}, n_perp_m = {self.n_perp_m:.5e}")
        print(f"  --> Effective meshing index used : {n_meshing:.5e}")
        print(f"  --> h_max_plasma resolution : {self.h_max_plasma:.5e} m")
        
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
    def solve_helmholtz_2DHcurl_1DH1_with_pml(self, mesh, mode):
        '''
        Solves the Weak Form using standard (Ex, Ey, Ez) coordinate mapping: (Ex = E_3D[0] = Radial, Ey = E_3D[1] = Poloidal, Ez = E_3D[2] = Toroidal)
            - Set Hcurl finite element space (fes) for in plane E field components (Ex, Ez): Hcurl compute E field and needles between meshpoints
            H1 (fes) compute out of 2D plane E field component (Ey) and force the field vector continuity in every space directions
            - 
        '''

        if self.mode == "RADIAL_ONLY":
            dirichlet_bnds = "top_wall_pec"

        else: dirichlet_bnds = "left_wall_pec|right_wall_pec|top_wall_pec|bottom_wall_pec"
        fes_plane = HCurl(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)
        
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
        print(f'Stretch_z (RADIAL_ONLY) : {Stretch_z}')
        self.pml_tensor = CF((Stretch_x / Stretch_z, 0.0, 0.0, 
                              0.0, 1.0/(Stretch_x * Stretch_z), 0.0, 
                              0.0, 0.0, Stretch_z / Stretch_x), dims=(3,3))

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
            kz = self.k0 * self.n_para
            print(f'kz:{kz:.2e}')

            E_inc_z = E0 * exp(1j * kz * self.z) # + exp(-1.5j *kz *self.z)) # Pure plane wave!
            
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
            
        # --- Gamma Reflection Computation ---
        Ex, Ey, Ez = self.E_field.components[0][0], self.E_field.components[1], self.E_field.components[0][1]
        E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))
        window_size_radial = 1. * self.cfg['WAVE']['lambda0']/self.n_perp_p.real # = 1.5 * lambda_perp_+
        window_size_toroidal = 1. * self.lambda_para
        
        # --- Radial reflection coefficient ---
        x_fixed_target = self.Lx_plasma * 0.95
        z_sweeping_vals = np.linspace(0.01 * self.Lz_plasma, 0.99 * self.Lz_plasma, 1000)
        mips_target_R = self.mesh(np.full_like(z_sweeping_vals, x_fixed_target), z_sweeping_vals)
        E_target_R, valid_z_sweep_vals = [], []
        for z, mip in zip(z_sweeping_vals, mips_target_R):
            try: 
                E_target_R.append(E_tot_norm(mip).real)
                valid_z_sweep_vals.append(z)
            except:
                pass
        E_target_R_array, valid_z_sweep_vals_array = np.array(E_target_R), np.array(z_sweeping_vals)
        peak_E_z_idx = np.argmax(E_target_R_array)
        peak_z = valid_z_sweep_vals_array[peak_E_z_idx]

        x_vals = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
        mips_measure_R = self.mesh(x_vals, np.full_like(x_vals, peak_z))
        E_measure_R, valid_x_vals = [], []

        for x, mip in zip(x_vals, mips_measure_R):
            E_measure_R.append(E_tot_norm(mip).real)
            valid_x_vals.append(x)
        E_measure_R_array, valid_x_vals_array = np.array([E_measure_R]), np.array([valid_x_vals])
        mask_R = (valid_x_vals_array >= x_fixed_target - window_size_radial) & (valid_x_vals_array <= x_fixed_target + window_size_toroidal)
        E_window_R = E_measure_R_array[mask_R]

        if len(E_window_R) > 5:
            SWR_R = max(np.max(E_window_R) / np.max([np.min(E_window_R), 1e-12]), 1.00001)
            Gamma_R = (SWR_R - 1) / (SWR_R + 1)
            print(f'Radial Gamma_R: {Gamma_R:.2e}, computed at z={peak_z:.3e}m')
        else:
            print(f'E_window_R: {E_window_R}, len(E_window_R) < 5')


        # --- Toroidal reflection coefficient ---
        if self.n_para >= 0:
            z_fixed_target = 0.05 * self.Lz_plasma
        else:
            z_fixed_target = 0.95 * self.Lz_plasma
        print(f'z_fixed_target: {z_fixed_target:.2e}m')

        x_sweeping_vals = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
        mips_target_T = self.mesh(x_sweeping_vals, np.full_like(x_sweeping_vals, z_fixed_target))
        E_target_T, valid_x_sweep_vals = [], []
        for x, mip in zip(x_sweeping_vals, mips_target_T):
            try:
                E_target_T.append(E_tot_norm(mip).real)
                valid_x_sweep_vals.append(x)
            except: pass
        E_target_T_array, valid_x_sweep_vals_array = np.array(E_target_T), np.array(valid_x_sweep_vals)
        peak_E_x_idx = np.argmax(E_target_T_array)
        peak_x = valid_x_sweep_vals_array[peak_E_x_idx]

        z_vals = np.linspace(0.01 * self.Lz_plasma, 0.99 * self.Lz_plasma, 1000)
        mips_measure_T = self.mesh(np.full_like(z_vals, peak_x), z_vals)
        E_measure_T, valid_z_vals = [], []
        for z, mip in zip(z_vals, mips_measure_T):
            try:
                E_measure_T.append(E_tot_norm(mip).real)
                valid_z_vals.append(z)
            except: pass
        
        E_measure_T_array, valid_z_vals_array = np.array(E_measure_T), np.array(valid_z_vals)
        mask_T = (valid_z_vals_array >= z_fixed_target - window_size_toroidal) & (valid_z_vals_array <= z_fixed_target + window_size_toroidal)
        E_window_T = E_measure_T_array[mask_T]

        if len(E_window_T) > 5:
            SWR_T = max(np.max(E_window_T) / np.max([np.min(E_window_T), 1e-12]), 1.00001)
            Gamma_T = (SWR_T - 1) / (SWR_T + 1)
            print(f'Radial Gamma_T: {Gamma_T:.2e}, computed at x={peak_x:.3e}m')
        else:
            print(f'E_window_T: {E_window_T}, len(E_window_T) < 5')

        return self.E_field, self.fes.ndof
    

    
