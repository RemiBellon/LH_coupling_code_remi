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
        self.B_stix = (self.S + self.P)*np.abs(self.n_para)**2 - (self.S**2 - self.D**2) - self.P*self.S
        self.C_stix = self.P * (np.abs(self.n_para)**2 - (self.S + self.D)) * (np.abs(self.n_para)**2 - (self.S - self.D))
        
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
        print(f'mode: {self.mode}')
        self.Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        self.Lx_pml = self.cfg['DOMAIN']['Lx_pml'] # 1.0 * (self.cfg['WAVE']['lambda0']/np.abs(self.n_perp_p.real))
        self.cfg['DOMAIN']['Lx_tot'] = self.Lx_tot = self.Lx_plasma + self.Lx_pml
        print(f'==== \n'
            f'Lx_plasma: {self.Lx_plasma:.2e}m,   Lx_pml: {self.Lx_pml:.2e}m,    Lx_tot: {self.Lx_tot:.2e}m')
        
        self.lambda0, self.n_para = self.cfg['WAVE']['lambda0'], self.cfg['WAVE']['n_para']
        self.lambda_para = self.lambda0/abs(self.n_para)
        self.lambda_perp_p = self.lambda0/np.abs(self.n_perp_p)
        self.lambda_perp_m = self.lambda0/np.abs(self.n_perp_m)

        if self.mode in ["RADIAL_ONLY", "FULL_2D"] and self.n_para != 0:
            self.Lz_plasma = 3.0 * self.lambda_para
            self.cfg['DOMAIN']['Lz_plasma'] = self.Lz_plasma
        else: 
            self.Lz_plasma = self.cfg['DOMAIN']['Lz_plasma']
        
        # self.Lz_pml = self.cfg['DOMAIN']['Lz_pml']
        self.cfg['DOMAIN']['Lz_pml'] = self.Lz_pml = 1.0 * self.lambda_para 
        self.cfg['DOMAIN']['Lz_tot'] = self.Lz_tot = self.Lz_plasma + 2*self.Lz_pml
        print(f'Lz_plasma: {self.Lz_plasma:.2e}m,   Lz_pml: {self.Lz_pml:.2e}m,    Lz_tot:{self.Lz_tot:.2e}m')

        # Define every plasma and pmls areas = define rectangles from the bottom left corner and (x,z) sizes:
        rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
        rect_plasma.edges.Min(occ.X).name = "bottom_source"
        print(f'==== \n'
            f'n_∥: {self.n_para:.1f},   λ_∥: {self.lambda_para:.2e}m \n'
            f'n_⟂⁺:{self.n_perp_p:.2f}, λ_⟂⁺: {self.lambda_perp_p:.2e}m \n'
            f'n_⟂-:{self.n_perp_m:.2f}, λ_⟂⁻$: {self.lambda_perp_m:.2e}m')
        
        
        # Plasma meshing resolution
        n_index_meshing = max(np.abs(self.n_perp_p), np.abs(self.n_perp_m), np.abs(self.n_para), 1.0)
        lambda_meshing = self.cfg['WAVE']['lambda0'] / n_index_meshing
        self.maxh_plasma = lambda_meshing / self.cfg['DOMAIN']['PPW_plasma']
        print(f'==== \n'
            f'n_index_meshing: {n_index_meshing:.2f} \n'
            f'λ_meshing: {lambda_meshing:.2e}m \n'            
            f'maxh_plasma: {self.maxh_plasma:.2e}m')

        if self.mode == "RADIAL_ONLY":
            # PLASMA DOMAIN:
            rect_plasma = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_plasma).Face()
            rect_plasma.maxh = self.maxh_plasma
            
            rect_pml_radial = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_plasma).Face()
            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
            Sx_norm = np.sqrt((Sx_r)**2 + (Sx_im)**2)
            maxh_pml_radial = lambda_meshing / (PPW_pml * Sx_norm)
            
            print(f'==== \n'
                f'PPW_pml_radial: {PPW_pml:.1f} \n'
                f'Sx_r: {Sx_r:.2f},    Sx_im: {Sx_im:.2f},    Sx_norm: {Sx_norm:.2f} \n'
                f'maxh_pml_radial: {maxh_pml_radial:.3e}m \n')
            domain = occ.Glue([rect_plasma, rect_pml_radial])

            for e in domain.edges:
                # Left Wall (Source)
                if abs(e.center[0]) < 1e-6:
                    e.name = "bottom_source"
                # Right Wall (PML back wall)
                elif abs(e.center[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = max(maxh_pml_radial, 1e-4)
                    
            # Periodic Identification (Edge by Edge)
            for bot_edge in domain.edges:
                if abs(bot_edge.center[1]) < 1e-6: # If it's a bottom edge
                    bot_edge.name = "periodic_bot"
                    
                    # Find the exact corresponding top edge with the same X-center
                    for top_edge in domain.edges:
                        if abs(top_edge.center[1] - self.Lz_plasma) < 1e-6 and abs(top_edge.center[0] - bot_edge.center[0]) < 1e-6:
                            top_edge.name = "periodic_top"
                            bot_edge.Identify(top_edge, "periodic", occ.IdentificationType.PERIODIC)
            print('[DOMAIN 1D DEFINED]')

        else: # FULL_2D or VACUUM (with toroidal PMLs)
            self.Lz_metal = 0.15 * self.Lz_plasma
            self.Lz_plasma_src = self.Lz_plasma - (2 * self.Lz_metal)
            print(f'==== \n'
                f'Lz_metal: {self.Lz_metal:.2e}m \n'
                f'Lz_source: {self.Lz_plasma_src:.2e}m')
            
            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            maxh_pml_radial = lambda_meshing / (PPW_pml * Sx_norm_max) 
            Sz_norm_max = np.sqrt(self.cfg['PML']['Sz_r']**2 + self.cfg['PML']['Sz_im']**2)
            maxh_pml_toroidal = lambda_meshing / (PPW_pml * Sz_norm_max)
            maxh_pml_corner = lambda_meshing / (PPW_pml * np.sqrt(Sx_norm_max**2 + Sz_norm_max**2))

            print(f'==== \n'
                f'PPW_pml: {PPW_pml:.1f} \n'
                f'Sx_norm:{Sx_norm_max:.2e}, maxh_pml_radial: {maxh_pml_radial:.2e} \n'
                f'Sz_norm_max: {Sz_norm_max:.2e}, maxh_pml_toroidal: {maxh_pml_toroidal:.2e} \n'
                f'maxh_pml_corner: {maxh_pml_corner:.2e}')

            # 1. Build BARE geometry (NO NAMES, NO MAXH YET)
            rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_metal).Face()
            rect_plasma_src = occ.MoveTo(0, self.Lz_metal).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
            rect_plasma_right = occ.MoveTo(0, self.Lz_metal + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_metal).Face()

            rect_pml_radial_left = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_metal).Face()
            rect_pml_radial_middle = occ.MoveTo(self.Lx_plasma, self.Lz_metal).Rectangle(self.Lx_pml, self.Lz_plasma_src).Face()
            rect_pml_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_metal + self.Lz_plasma_src).Rectangle(self.Lx_pml, self.Lz_metal).Face()

            rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma).Rectangle(self.Lx_plasma, self.Lz_pml).Face()

            rect_pml_corner_rad_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            rect_pml_corner_rad_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma).Rectangle(self.Lx_pml, self.Lz_pml).Face()

            # 2. GLUE FIRST (Dissolves internal boundaries into a unified topology)
            domain = occ.Glue([rect_plasma_left, rect_plasma_src, rect_plasma_right, 
                               rect_pml_radial_left, rect_pml_radial_middle, rect_pml_radial_right, 
                               rect_pml_toroidal_left, rect_pml_toroidal_right, 
                               rect_pml_corner_rad_left, rect_pml_corner_rad_right])
            
            print('[DOMAIN 2D DEFINED AND GLUED]')

            # 3. TAG OUTER BOUNDARIES AFTER GLUING
            # We locate edges using their geometric coordinates on the glued domain
            
            # X-Axis boundaries (Left / Right)
            for e in domain.edges:
                # Leftmost edge (Antenna side, x=0)
                if abs(e.center[0]) < 1e-6:
                    # Is it the source or the metal wall?
                    if self.Lz_metal < e.center[1] < (self.Lz_metal + self.Lz_plasma_src):
                        e.name = "bottom_source"
                    else:
                        e.name = "bottom_wall_pec"
                
                # Rightmost outer edge (End of radial PML)
                elif abs(e.center[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = maxh_pml_radial  # Force extreme fine mesh only at the very end
                    
            # Y-Axis boundaries (Top / Bottom in physical space, which is Z-axis)
            for e in domain.edges:
                # Bottom-most edge (End of toroidal PML)
                if abs(e.center[1] - (-self.Lz_pml)) < 1e-6:
                    e.name = "left_wall_pec"
                    e.maxh = maxh_pml_toroidal
                
                # Top-most edge (End of toroidal PML)
                elif abs(e.center[1] - (self.Lz_plasma + self.Lz_pml)) < 1e-6: # Note: depending on coord system, this might be Lz_plasma + Lz_pml
                    e.name = "right_wall_pec"
                    e.maxh = maxh_pml_toroidal

        geo = occ.OCCGeometry(domain, dim=2)
        # Apply the global maximum size for the bulk plasma
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.maxh_plasma))
        print('[MESH GENERATED !]')
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
        print(f'==== \n'
            f'#DoFs = {self.fes.ndof}')

        # --- Dynamic PML Stretching ---
        Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_r, Sz_im, pz = self.cfg['PML']['Sz_r'], self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']
        
        Stretch_x = 1.0 + (Sx_r - 1.0 - 1j * Sx_im) * \
                    IfPos(self.x - self.Lx_plasma, ((self.x - self.Lx_plasma) / self.Lx_pml)**px, 0.0)
            
        if self.mode == "RADIAL_ONLY":
            Stretch_z = CF(1.0)
        else:
            Stretch_z = 1.0 + (Sz_r - 1.0 + 1j * Sz_im) * \
                        IfPos(-self.z, (-self.z / self.Lz_pml)**pz, \
                        IfPos(self.z - self.Lz_plasma, ((self.z - self.Lz_plasma) / self.Lz_pml)**pz, 0.0))
        # print(f'Stretch_z (RADIAL_ONLY) : {Stretch_z}')
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
        n_perp_p = self.n_perp_p
        if np.real(n_perp_p) > 0:
            n_perp_p = np.real(n_perp_p) + 1j * np.imag(n_perp_p)
        Y_z_term = 1j * self.k0 * (self.P / n_perp_p)
        Y_y_term = 1j * self.k0 * n_perp_p
        Ez_trace = E_plane.Trace()[1]
        Ey_trace = E_outplane.Trace()[0]
        vz_trace = v_plane.Trace()[1]
        vy_trace = v_outplane.Trace()[0]

        # 2. Bilinear Form: The exact absorbing operator for the reflected wave
        # This includes the critical -i*kz*Ex*vz term dictated by the curl operator!
        # Magnetic Surface Flux injection.
        a = BilinearForm(self.fes)
        a += (self.pml_tensor * curl_E_3D * curl_v_3D - \
              self.k0**2 * self.eff_eps_tensor * E_3D * v_3D) * dx
        a += (Y_z_term * Ez_trace *vz_trace + Y_y_term * Ey_trace * vy_trace) * ds("bottom_source")
        
        with TaskManager():
            a.Assemble()
            
            # Linear Form: The Incident Wave Source
            f = LinearForm(self.fes)
            E0 = self.cfg['WAVE']['E_inc']
            kz = self.k0 * np.abs(self.n_para)
            E_inc_z = E0 * exp(1j * kz * self.z) # + exp(-1.5j *kz *self.z)) # Pure plane wave!
            n_perp_inc = -np.abs(np.real(self.n_perp_p)) + 1j * np.imag(self.n_perp_p)

            Py = -1j * self.D * (self.P - n_perp_inc**2) / (n_perp_inc * self.n_para * (self.S - n_perp_inc**2 - self.n_para**2))
            E_inc_y = Py * E_inc_z
            f += (2.0 * Y_z_term * E_inc_z * vz_trace + 2.0 * Y_y_term * E_inc_y * vy_trace) * ds("bottom_source")
            
            f.Assemble()


            print("--- Solving the 3D vector linear system ---")
            res = f.vec.CreateVector()
            res.data = f.vec - a.mat * self.E_field.vec
            inv = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            self.E_field.vec.data += inv * res
            
       # =====================================================================
        # RIGOROUS GAMMA REFLECTION COMPUTATION 
        # =====================================================================
        E_xz, Ey = self.E_field.components[0], self.E_field.components[1]
        Ex, Ez = E_xz[0], E_xz[1]                
        E_tot_norm = sqrt(Ex*Conj(Ex) + Ey*Conj(Ey) + Ez*Conj(Ez))

        window_size_radial = 1.5 * self.cfg['WAVE']['lambda0'] / np.abs(self.n_perp_p.real)
        window_size_toroidal = 1.5 * self.lambda_para
        print(f'==== \n'
            f'w_size_R: {window_size_radial:.4e}m,    w_size_T: {window_size_toroidal:.4e}m')
        
        # ---------------------------------------------------------------------
        # 1. RADIAL SWR (Right Wall: Reflection along X-axis)
        # ---------------------------------------------------------------------
        # A) Target Line: Vertical line at x = 0.95 * Lx_plasma
        x_target_R = self.Lx_plasma * 0.95
        z_sweep = np.linspace(0.01 * self.Lz_plasma, 0.99 * self.Lz_plasma, 1000)
        mips_target_R = self.mesh(np.full_like(z_sweep, x_target_R), z_sweep)
        
        E_target_R, valid_z_R = [], []
        for z, mip in zip(z_sweep, mips_target_R):
            try: 
                E_target_R.append(E_tot_norm(mip).real)
                valid_z_R.append(z)
            except: pass
            
        peak_z_R = valid_z_R[np.argmax(E_target_R)]
        
        # B) Measure Line: Horizontal line at z = peak_z_R
        x_sweep = np.linspace(0.01 * self.Lx_plasma, self.Lx_plasma * 0.99, 1000)
        mips_measure_R = self.mesh(x_sweep, np.full_like(x_sweep, peak_z_R))
        
        E_measure_R, valid_x_R = [], []
        for x, mip in zip(x_sweep, mips_measure_R):
            try:
                E_measure_R.append(E_tot_norm(mip).real)
                valid_x_R.append(x)
            except: pass
            
        valid_x_R = np.array(valid_x_R)
        E_measure_R = np.array(E_measure_R)
        
        # Apply radial window strictly along the X-axis
        mask_R = (valid_x_R >= x_target_R - window_size_radial) & (valid_x_R <= x_target_R)
        E_window_R = E_measure_R[mask_R]
        
        if len(E_window_R) > 5:
            SWR_R = max(np.max(E_window_R) / np.max([np.min(E_window_R), 1e-12]), 1.000001)
            Gamma_R = (SWR_R - 1.0) / (SWR_R + 1.0)
            print(f"Gamma_R: {Gamma_R:.2e}, computed at z={peak_z_R:.3e}m")
        else:
            Gamma_R = 1.0
            print("Gamma_R FAILED: Window missed or too small.")

        # ---------------------------------------------------------------------
        # 2. TOROIDAL SWR (Top/Bottom Wall: Reflection along Z-axis)
        # ---------------------------------------------------------------------
        # A) Target Line: Horizontal line at z = 0.95 * Lz_plasma (if n_para > 0)
        if self.n_para.real >= 0:
            z_target_T = self.Lz_plasma * 0.95
        else:
            z_target_T = self.Lz_plasma * 0.05
            
        x_sweep_T = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
        mips_target_T = self.mesh(x_sweep_T, np.full_like(x_sweep_T, z_target_T))
        
        E_target_T, valid_x_T = [], []
        for x, mip in zip(x_sweep_T, mips_target_T):
            if E_tot_norm(mip).real !=0:
                try:
                    E_target_T.append(E_tot_norm(mip).real)
                    valid_x_T.append(x)
                except: pass
            
        peak_x_T = valid_x_T[np.argmax(E_target_T)]
        
        # B) Measure Line: Vertical line at x = peak_x_T
        z_sweep_T = np.linspace(0.01 * self.Lz_plasma, 0.99 * self.Lz_plasma, 1000)
        mips_measure_T = self.mesh(np.full_like(z_sweep_T, peak_x_T), z_sweep_T)
        
        E_measure_T, valid_z_T = [], []
        for z, mip in zip(z_sweep_T, mips_measure_T):
            try:
                E_measure_T.append(E_tot_norm(mip).real)
                valid_z_T.append(z)
            except: pass
            
        valid_z_T = np.array(valid_z_T)
        E_measure_T = np.array(E_measure_T)
        
        # Apply toroidal window strictly along the Z-axis
        if self.n_para.real >= 0:
            mask_T = (valid_z_T <= z_target_T) & (valid_z_T >= z_target_T - window_size_toroidal)
        else:
            mask_T = (valid_z_T >= z_target_T) & (valid_z_T <= z_target_T + window_size_toroidal)
            
        E_window_T = E_measure_T[mask_T]

        if len(E_window_T) > 5:
            SWR_T = max(np.max(E_window_T) / np.max([np.min(E_window_T), 1e-12]), 1.000001)
            Gamma_T = (SWR_T - 1.0) / (SWR_T + 1.0)
            print(f"Gamma_T: {Gamma_T:.2e}, computed at x={peak_x_T:.3e}m")
        else:
            Gamma_T = 1.0
            print("Gamma_T FAILED: Window missed or too small.")


        # ---------------------------------------------------------------------
        # Poynting vector computation
        # ---------------------------------------------------------------------
        self.mu0 = self.cfg['CONST']['mu0']
        E_sol_3D = CF((self.E_field.components[0][0], self.E_field.components[1], self.E_field.components[0][1]))
        curl_E_sol_3D = CF(( -grad(self.E_field.components[1])[1], 
                             -curl(self.E_field.components[0]), 
                              grad(self.E_field.components[1])[0] ))
        H_sol_3D = curl_E_sol_3D / (1j * self.omega_LH * self.mu0)
        S_sol_3D = 0.5 * Cross(E_sol_3D, Conj(H_sol_3D)).real
        
        S_x_cf, S_z_cf = S_sol_3D[0], S_sol_3D[2]
        

        def integrate_flux(cf, x_vals, z_vals, axis):
            vals, coords = [], []
            for xi, zi in zip(x_vals, z_vals):
                try:
                    mip = self.mesh(xi, zi)
                    vals.append(cf(mip))
                    coords.append(zi if axis=='z' else xi)
                except: pass
            return np.trapezoid(vals, x=coords) if len(coords) > 1 else 0.0

        x_limits = np.linspace(0.0, self.Lx_plasma, 1000)
        z_limits = np.linspace(0.0, self.Lz_plasma, 1000)
        # Compute Integrals
        P_in_net = integrate_flux(S_x_cf, np.full_like(z_limits,0), z_limits, 'z')
        P_out_R = integrate_flux(S_x_cf, np.full_like(z_limits, self.Lx_plasma), z_limits, 'z')
        P_out_T_right = integrate_flux(S_z_cf, x_limits,  np.full_like(x_limits, self.Lz_plasma), 'x')
        P_out_T_left = integrate_flux(S_z_cf, x_limits,  np.full_like(x_limits, self.Lz_plasma), 'x') # Negative because it points -z
        
        print(f'==== \n'
            f"Net Power Flow (W/m) => P_in: {P_in_net:.4e} | P_out_top: {P_out_R:.4e} \n"
             f"P_out_right: {P_out_T_right:.4e} | P_out_T_left: {P_out_T_left:.4e}")
        
        

        # ---------------------------------------------------------------------
        # Theoretical incident power
        # ---------------------------------------------------------------------
        Z0, E0 = np.sqrt(self.cfg['CONST']['mu0'] / self.cfg['CONST']['eps0']), self.cfg['WAVE']['E_inc']
        Ey_over_Ez_ratio = 1j * (self.D*(self.P - np.abs(self.n_perp_p)**2)) / (self.n_perp_p * self.n_para * (self.S - (self.n_para**2 + self.n_perp_p**2)))
        Sx = (self.P * E0**2) / (2 * Z0) * (Conj(self.n_perp_p)*np.abs(Ey_over_Ez_ratio)**2 + self.P / Conj(self.n_perp_p))
        Sx_val = Sx(self.mesh(1e-5, self.Lz_plasma / 2.0)).real
        if self.mode == "RADIAL_ONLY":
            Lz_source = self.cfg['DOMAIN']['Lz_plasma']
        else:
            Lz_metal = 0.15 * self.cfg['DOMAIN']['Lz_plasma']
            Lz_source = self.cfg['DOMAIN']['Lz_plasma'] - (2.0 * Lz_metal)
        P_ideal = np.abs(Sx_val) * Lz_source
        print(f'type(P_ideal): {type(P_ideal)}')
        print(f'Sx_val: {Sx_val:.4e} W/m^2\n'
            f'(Lz_source): {Lz_source:.3f} \n'
            f'P_ideal: {P_ideal:.4e} W/m \n'
            f'Gamma = 1-(P_out)/P_ideal: {(np.sqrt(1 - (P_out_R + P_out_T_right - P_out_T_left)/P_ideal)):.4e} \n')
        print(f'P_stix: {self.P:.3e}')
        
        # ---------------------------------------------------------------------
        # EXPORT DIAGNOSTIC DATA FOR PLOTTING
        # ---------------------------------------------------------------------
        diag_data = {
            'x_target_R': x_target_R,
            'peak_z_R': peak_z_R,
            'window_size_radial': window_size_radial,
            'z_target_T': z_target_T, #  if self.mode != "RADIAL_ONLY" else None,
            'peak_x_T': peak_x_T,     # if self.mode != "RADIAL_ONLY" else None,
            'window_size_toroidal': window_size_toroidal,
            'n_para': self.n_para, 
            'P_in_net': P_in_net, 
            'P_ideal': P_ideal,
            'P_out_R': P_out_R, 
            'P_out_T_right': P_out_T_right, 
            'P_out_T_left': P_out_T_left,
        }

        return self.E_field, self.fes.ndof, Gamma_R, Gamma_T, diag_data