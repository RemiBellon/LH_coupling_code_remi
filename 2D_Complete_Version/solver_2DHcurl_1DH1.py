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
    def __init__(self, config_dict, geom_mode, box_medium, antenna_grill):
        self.cfg = config_dict          # type = dict ==> dictionnary of dictionnaries with physics (wave, plasma) and geometry (domain, pmls) values
        self.geom_mode = geom_mode          # 1D or 2D
        self.box_medium = box_medium        # "VACUUM", "PLASMA"
        self.antenna_grill = antenna_grill 
        
        self.mesh = None                # type = ngsolve.comp.Mesh ==> Mesh to solve wave equation
        self.fes = None                 # type = ngsolve.comp.FESpace ==> Hcurl space function to solve wave equation
        self.E_field = None             # type = ngsolve.comp.GridFunction ==> Solution of the wave equation on the mesh/grid

        self.wg_medium = self.cfg['DOMAIN'].get('wg_medium', 'VACUUM').upper()
        self.x = x                      # type = ngsolve.fem.CoefficientFunction ==> Space coords to compute wave equation variables
        self.z = y                      # In the context y is out of 2D plane direction. In 2D only the plane (xOz) is describe.   

        self.n_para, self.n_perp_p, self.n_perp_m = self.compute_physics_parameters()
        print(f'In LHCoupling class: n_para: {self.n_para:.1f}, n_perp_p: {self.n_perp_p:.2e}, n_perp_m: {self.n_perp_m:.2e}')
        if geom_mode != "1D" and antenna_grill is not None:
            print(f'Waveguide Medium set to: {self.wg_medium}')
    
    def compute_physics_parameters(self) -> None:
        self.omega_LH, self.k0, self.B0 = self.cfg['WAVE']['omega_LH'], self.cfg['WAVE']['k0'], self.cfg['PLASMA']['B0']
        self.qe, self.me, self.mi, self.eps0, self.mu0 = self.cfg['CONST']['qe'], self.cfg['CONST']['me'], self.cfg['CONST']['mi'], self.cfg['CONST']['eps0'], self.cfg['CONST']['mu0']
        
        # Compute or recover n_para if antenna or not 
        if self.antenna_grill is None:
            if self.box_medium == 'VACUUM':
                self.n_para = 0.0
            else:
                self.n_para = self.cfg['WAVE']['n_para']
        else:
            # We assume uniform modules for the dominant n_para calculation
            module = self.antenna_grill.modules[0]
            # Find the first phase shift (delta_phi)
            active_wgs = [wg for wg in module if wg.is_active]
            if len(active_wgs) > 1:
                phase_diff_rad = np.angle(active_wgs[1].complex_E) - np.angle(active_wgs[0].complex_E)
                periodicity = self.antenna_grill.b_active + self.antenna_grill.d_septa
                # If PAM, add the passive width and extra septa
                if not active_wgs[1].is_active: # Basic check, refine based on your PAM logic
                   pass 
                
                # Simplified robust calculation for typical grills
                self.n_para = (self.cfg['CONST']['c0'] / self.omega_LH) * (abs(phase_diff_rad) / periodicity)
            else:
                 self.n_para = self.cfg['WAVE']['n_para'] # Fallback
                                                                                           
        if self.box_medium == "VACUUM":
            self.ne_constant = 0.0
        else: 
            self.ne_constant = self.cfg['PLASMA']['ne_constant']
        print(f'n_e = {self.ne_constant:.2e} m^-3')
        self.w_pe2 = (self.ne_constant * self.qe**2) / (self.me * self.eps0)
        self.w_pi2 = (self.ne_constant * self.qe**2) / (self.mi * self.eps0)
        self.Om_ce = (self.qe * self.B0) / self.me
        self.Om_ci = (self.qe * self.B0) / self.mi
        
        if self.box_medium == "VACUUM":
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
        print(f'geom_mode: {self.geom_mode}')
        print(f'box_medium: {self.box_medium}')
        
        self.Lx_plasma, self.Lx_pml = self.cfg['DOMAIN']['Lx_plasma'], self.cfg['DOMAIN']['Lx_pml'] 
        self.cfg['DOMAIN']['Lx_tot'] = self.Lx_tot = self.Lx_plasma + self.Lx_pml

        # Set the waveguide length if antenna or not 
        if self.antenna_grill is None:
            self.Lx_wg = 0.0            # No waveguide(s)
        else:
            self.Lx_wg = self.cfg['DOMAIN'].get('Lx_wg', 0.03)

        self.lambda0 = self.cfg['WAVE']['lambda0']
        self.lambda_para = self.lambda0/max(abs(self.n_para), 1e-6)
        self.lambda_perp_p = self.lambda0/np.abs(self.n_perp_p)
        self.lambda_perp_m = self.lambda0/np.abs(self.n_perp_m)

        if self.box_medium == "VACUUM":
            if self.geom_mode == "1D": # Apart from DoFs no constraint on Lz size and no toroidal PMLs in 1D Vacuum
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma']
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = 0.
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.
            elif self.geom_mode == "2D": # In vacuum, cannot base z box size and PMLs on lambda_para 
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma']
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = self.Lz_plasma_src/2
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.

        elif self.box_medium == "PLASMA":
            if self.geom_mode == "1D":# DoFs and periodic constraints on Lz size and no toroidal PMLs in 1D Plasma
                self.Lz_plasma_src = self.cfg['DOMAIN']['Lz_plasma'] = 3.*self.lambda_para
                self.Lz_pml = self.cfg['DOMAIN']['Lz_pml'] = 0.
                self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.

            elif self.geom_mode == "2D":
                self.cfg['DOMAIN']['Lz_pml'] = self.Lz_pml = 1.* self.lambda_para
            
                if self.antenna_grill is not None:
                    base_instruction = self.antenna_grill.generate_mesh_instructions(z_start_position=0.0)
                    self.Lz_antenna = self.Lz_plasma_src = base_instruction[-1]['z_end']
                    self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = self.cfg['DOMAIN'].get('Lz_wall', 0.02)
                    self.instructions = self.antenna_grill.generate_mesh_instructions(z_start_position=self.Lz_wall)
                else:
                    self.Lz_plasma_src = 3.* self.lambda_para
                    # No metal walls if no antenna
                    self.Lz_wall = self.cfg['DOMAIN']['Lz_wall'] = 0.
                    self.instructions = []

        self.cfg['DOMAIN']['Lz_tot'] = self.Lz_tot = self.Lz_plasma_src + 2.0 * self.Lz_wall + 2.0 * self.Lz_pml
        
        print(f'==== \n'
            f'Lx_plasma: {self.Lx_plasma:.2e}m,   Lx_pml: {self.Lx_pml:.2e}m,    Lx_tot: {self.Lx_tot:.2e}m\n'
            f'Lx_wg = {self.Lx_wg:.2e}m')
        print(f'Lz_antenna: {getattr(self, "Lz_antenna", 0):.2e}m, Lz_wall: {self.Lz_wall:.2e}m')
        print(f'Lz_plasma: {self.Lz_plasma_src:.2e}m, Lz_pml: {self.Lz_pml:.2e}m, Lz_tot: {self.Lz_tot:.2e}m')
        print(f'==== \n'
            f'n_∥: {self.n_para:.1f},   λ_∥: {self.lambda_para:.2e}m \n'
            f'n_⟂⁺:{self.n_perp_p:.2f}, λ_⟂⁺: {self.lambda_perp_p:.2e}m \n'
            f'n_⟂-:{self.n_perp_m:.2f}, λ_⟂⁻: {self.lambda_perp_m:.2e}m')

        # meshing resolution based on the shortest wavelength (max refractive index n) 
        n_index_meshing = max(np.abs(self.n_perp_p), np.abs(self.n_perp_m), np.abs(self.n_para), 1.0)
        shortest_lambda = self.cfg['WAVE']['lambda0'] / n_index_meshing
        self.maxh_plasma = shortest_lambda / self.cfg['DOMAIN']['PPW_plasma']

        print(f'==== \n'
            f'n_index_meshing: {n_index_meshing:.2f} \n'
            f'λ_meshing: {shortest_lambda:.2e}m \n'            
            f'maxh_plasma: {self.maxh_plasma:.2e}m')

        # --- Create the geometry based on the previously recovered dimensions
        # Set simulation box (rect_plasma is equivalent to rect vacuum in dimensions size) 
        rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_wall).Face()
        rect_plasma_src = occ.MoveTo(0, self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
        rect_plasma_right = occ.MoveTo(0, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_wall).Face()

        # Generate waveguide rectangles iff antenna is defined (not None)
        wg_faces = []
        if self.Lx_wg > 0:
            for inst in self.instructions:
                if inst['type'] in ['wg_active', 'wg_passive']:
                    width_wg = inst['z_end'] - inst['z_start']
                    face = occ.MoveTo(-self.Lx_wg, inst['z_start']).Rectangle(self.Lx_wg, width_wg).Face()
                    wg_faces.append(face)
        
        # Generate Geometry rectangles
        if self.geom_mode == "1D":
            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            maxh_pml_radial = shortest_lambda / (PPW_pml * Sx_norm_max)
        
            rect_pml_radial_left = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            rect_pml_radial_middle = occ.MoveTo(self.Lx_plasma, self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_plasma_src).Face()
            rect_pml_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            faces = [rect_plasma_src, rect_pml_radial_middle]
            if self.Lz_wall > 0:
                faces += [rect_plasma_left, rect_plasma_right, rect_pml_radial_left, rect_pml_radial_right]
            domain = occ.Glue(faces + wg_faces)

            for e in domain.edges:
                if abs(e.center[0]) < 1e-6:
                    if self.Lz_wall < e.center[1] < (self.Lz_wall + self.Lz_plasma_src):
                        e.name = "bottom_source"
                    else:
                        e.name = "bottom_wall_pec"
                elif abs(e.center[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = max(maxh_pml_radial, 1e-4)
                    
            for bot_edge in domain.edges:
                if abs(bot_edge.center[1]) < 1e-6:
                    bot_edge.name = "periodic_bot"
                    for top_edge in domain.edges:
                        if abs(top_edge.center[1] - self.Lz_plasma_src) < 1e-6 and abs(top_edge.center[0] - bot_edge.center[0]) < 1e-6:
                            top_edge.name = "periodic_top"
                            bot_edge.Identify(top_edge, "periodic", occ.IdentificationType.PERIODIC)
            print('[DOMAIN 1D DEFINED]')

        else: # FULL_2D or VACUUM (with toroidal PMLs)
            if self.antenna_grill is None:
                # self.Lz_wall = 0.15 * self.Lz_plasma_src
                # self.Lz_plasma_src = self.Lz_plasma_src - (2.0 * self.Lz_wall)
                print(f'==== \n'
                f'Lz_wall: {self.Lz_wall:.2e}m \n'
                f'Lz_source: {self.Lz_plasma_src:.2e}m')
            
            PPW_pml = self.cfg['PML'].get('ppw_pml', 50)
            Sx_norm_max = np.sqrt(self.cfg['PML']['Sx_r']**2 + self.cfg['PML']['Sx_im']**2)
            maxh_pml_radial = shortest_lambda / (PPW_pml * Sx_norm_max) 
            Sz_norm_max = np.sqrt(self.cfg['PML']['Sz_r']**2 + self.cfg['PML']['Sz_im']**2)
            maxh_pml_toroidal = shortest_lambda / (PPW_pml * Sz_norm_max)
            maxh_pml_corner = shortest_lambda / (PPW_pml * np.sqrt(Sx_norm_max**2 + Sz_norm_max**2))

            print(f'==== \n'
                f'PPW_pml: {PPW_pml:.1f} \n'
                f'Sx_norm:{Sx_norm_max:.2e}, maxh_pml_radial: {maxh_pml_radial:.2e} \n'
                f'Sz_norm_max: {Sz_norm_max:.2e}, maxh_pml_toroidal: {maxh_pml_toroidal:.2e} \n'
                f'maxh_pml_corner: {maxh_pml_corner:.2e}')

            # 1. Build BARE geometry (NO NAMES, NO MAXH YET)
            rect_plasma_left = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_wall).Face()
            rect_plasma_src = occ.MoveTo(0, self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_plasma_src).Face()
            rect_plasma_right = occ.MoveTo(0, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_plasma, self.Lz_wall).Face()

            rect_pml_radial_left = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_wall).Face()
            rect_pml_radial_middle = occ.MoveTo(self.Lx_plasma, self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_plasma_src).Face()
            rect_pml_radial_right = occ.MoveTo(self.Lx_plasma, self.Lz_wall + self.Lz_plasma_src).Rectangle(self.Lx_pml, self.Lz_wall).Face()

            rect_pml_toroidal_left = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            rect_pml_toroidal_right = occ.MoveTo(0, self.Lz_plasma_src + 2.0 * self.Lz_wall).Rectangle(self.Lx_plasma, self.Lz_pml).Face()

            rect_pml_corner_rad_left = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
            rect_pml_corner_rad_right = occ.MoveTo(self.Lx_plasma, self.Lz_plasma_src + 2.0 * self.Lz_wall).Rectangle(self.Lx_pml, self.Lz_pml).Face()

            # 2. GLUE FIRST (Dissolves internal boundaries into a unified topology)
            domain = occ.Glue([rect_plasma_left, rect_plasma_src, rect_plasma_right, 
                               rect_pml_radial_left, rect_pml_radial_middle, rect_pml_radial_right, 
                               rect_pml_toroidal_left, rect_pml_toroidal_right, 
                               rect_pml_corner_rad_left, rect_pml_corner_rad_right] + wg_faces)
            
            print('[DOMAIN 2D DEFINED AND GLUED]')

            for e in domain.edges:
                c = e.center
                if self.Lx_wg > 0 and abs(c[0] - (-self.Lx_wg)) < 1e-6:
                    e.name = "bottom_source"
                elif self.Lx_wg > 0 and c[0] < -1e-6 and abs(c[0] - (-self.Lx_wg)) > 1e-6:
                    e.name = "bottom_wall_pec"
                elif abs(c[0]) < 1e-6:
                    if self.Lx_wg > 0:
                        is_opening = False
                        for inst in self.instructions:
                            if inst['type'] in ['wg_active', 'wg_passive']:
                                if inst['z_start'] - 1e-5 < c[1] < inst['z_end'] + 1e-5:
                                    is_opening = True
                                    break
                        if not is_opening:
                            e.name = "bottom_wall_pec"
                            e.maxh = self.maxh_plasma / 5.0 # EDGE SINGULARITY REFINEMENT!
                        else:
                            e.maxh = self.maxh_plasma / 5.0
                    else:
                        if self.Lz_wall < c[1] < (self.Lz_wall + self.Lz_plasma_src):
                            e.name = "bottom_source"
                        else:
                            e.name = "bottom_wall_pec"
                elif abs(c[0] - self.Lx_tot) < 1e-6:
                    e.name = "top_wall_pec"
                    e.maxh = maxh_pml_radial
                elif abs(c[1] - (-self.Lz_pml)) < 1e-6:
                    e.name = "left_wall_pec"
                    e.maxh = maxh_pml_toroidal
                elif abs(c[1] - (self.Lz_plasma_src + self.Lz_pml)) < 1e-6:
                    e.name = "right_wall_pec"
                    e.maxh = maxh_pml_toroidal

        geo = occ.OCCGeometry(domain, dim=2)
        # Apply the global maximum size for the bulk plasma
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.maxh_plasma))
        print('[MESH GENERATED !]')
        return self.mesh


    def build_antenna_source_function(self):
        # Translates AntennaGrill instructions into a continuous NGSolve CoefficientFunction.

        if self.antenna_grill is None:
            # Fallback to the old single plane wave
            if self.box_medium == 'VACUUM':
                return self.cfg['WAVE']['E_inc'] 
            else:
                return self.cfg['WAVE']['E_inc'] * exp(1j * self.k0 * self.n_para * self.z)

        # Initialize an empty complex field
        Ez_inc_cf = CF(0.0 + 0.0j)
        
        # Build the piecewise function
        print(f'len(instruction): {len(self.instructions)}')
        for inst in self.instructions:
            # print(f"inst['type']: {inst['type']}")
            if inst['type'] == 'wg_active':
                z_start = inst['z_start']
                z_end = inst['z_end']
                E_val = inst['complex_E_field']
            
                print(f'z_start: {z_start}, z_end: {z_end}, E_val: {E_val}')
            # We want: 1.0 IF (z > z_start AND z < z_end) ELSE 0.0
            # Condition 1: z - z_start > 0
            # Condition 2: z_end - z > 0
            # IfPos: If (z - z_start > 0) -> Evaluate (If z_end - z > 0) -> Return E_val
            
                is_inside_wg = IfPos(self.z - z_start, IfPos(z_end - self.z, 1.0, 0.0), 0.0)
            
                # Add this segment's field to the total function
                Ez_inc_cf += is_inside_wg * E_val
        return Ez_inc_cf
        
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
        if self.box_medium == "PLASMA":
            box_is_plasma = 1.0
        else:
            box_is_plasma = 0.0

        if self.Lx_wg > 0 and self.wg_medium == "PLASMA":
            wg_is_plasma = 1.0
        else:
            wg_is_plasma = 0.0

        # Mappage spatial rigoureux : Guide d'onde (x < 0) vs Boîte principale (x >= 0)
        is_plasma = IfPos(self.x, box_is_plasma, wg_is_plasma)
        is_vacuum = 1.0 - is_plasma

        theta_B, phi_B = 0.0, 0.0
        bx = np.sin(phi_B)
        by = np.cos(phi_B) * np.sin(theta_B)
        bz = np.cos(phi_B) * np.cos(theta_B)
    
        Q_stix = self.P - self.S

        self.K_xx = is_plasma * (self.S*(1 - bx**2) + self.P*bx**2) + 1.0 * is_vacuum
        self.K_xy = is_plasma * (1j*self.D*bz + Q_stix*bx*by)
        self.K_xz = is_plasma * (-1j*self.D*by + Q_stix*bx*bz)
        
        self.K_yx = is_plasma * (-1j*self.D*bz + Q_stix*by*bx)
        self.K_yy = is_plasma * (self.S*(1 - by**2) + self.P*by**2) + 1.0 * is_vacuum
        self.K_yz = is_plasma * (1j*self.D*bx + Q_stix*by*bz)
        
        self.K_zx = is_plasma * (1j*self.D*by + Q_stix*bz*bx)
        self.K_zy = is_plasma * (-1j*self.D*bx + Q_stix*bz*by)
        self.K_zz = is_plasma * (self.S*(1 - bz**2) + self.P*bz**2) + 1.0 * is_vacuum
        
        self.K_tensor = CF((self.K_xx, self.K_xy, self.K_xz,
                            self.K_yx, self.K_yy, self.K_yz,
                            self.K_zx, self.K_zy, self.K_zz), dims=(3,3))
    
    def get_port_admittance(self):
        """ Computes the robust 2x2 Reflected Admittance Tensor depending on the medium """
        Z0 = np.sqrt(self.mu0 / self.eps0)
        port_medium = self.wg_medium if self.Lx_wg > 0 else self.box_medium
        
        if port_medium == "PLASMA":
            Py_S = (1j * self.D * (self.n_perp_p**2 - self.P)) / (self.n_perp_p * self.n_para * (self.n_perp_p**2 - self.S))
            Py_F = (1j * self.D * (self.n_perp_m**2 - self.P)) / (self.n_perp_m * self.n_para * (self.n_perp_m**2 - self.S))
            
            denom = Z0 * (Py_S - Py_F)
            Y_11_inc = (-self.P/self.n_perp_p + self.P/self.n_perp_m) / denom
            Y_12_inc = (self.P*Py_F/self.n_perp_p - self.P*Py_S/self.n_perp_m) / denom
            Y_21_inc = (self.n_perp_p*Py_S - self.n_perp_m*Py_F) / denom
            Y_22_inc = (Py_S*Py_F*(self.n_perp_m - self.n_perp_p)) / denom
            
            Y_11_ref = Y_11_inc
            Y_12_ref = -Y_12_inc
            Y_21_ref = -Y_21_inc
            Y_22_ref = Y_22_inc
            
        else: # VACUUM (Onde TM pure et rigoureuse)
            n_perp_vac_port = 1.0 + 0.0j if self.Lx_wg > 0 else np.sqrt(complex(1.0 - self.n_para**2))
            Y_11_ref, Y_22_ref = 0.0 + 0.0j, 0.0 + 0.0j
            Y_12_ref = n_perp_vac_port / Z0
            Y_21_ref = -n_perp_vac_port / Z0
            
        return Y_11_ref, Y_12_ref, Y_21_ref, Y_22_ref

    def get_incident_fields(self, Ez_inc_val):
        """ Computes the incident E and H fields strictly adapted to the medium """
        Z0 = np.sqrt(self.mu0 / self.eps0)
        port_medium = self.wg_medium if self.Lx_wg > 0 else self.box_medium
        
        if port_medium == "PLASMA":
            Py_S = (1j * self.D * (self.n_perp_p**2 - self.P)) / (self.n_perp_p * self.n_para * (self.n_perp_p**2 - self.S))
            Ey_inc = Py_S * Ez_inc_val
            Ez_inc = Ez_inc_val
            Hy_inc = (-self.P / (self.n_perp_p * Z0)) * Ez_inc_val
            Hz_inc = ((self.n_perp_p * Py_S) / Z0) * Ez_inc_val
            
        else: # VACUUM
            n_perp_vac_port = 1.0 + 0.0j if self.Lx_wg > 0 else np.sqrt(complex(1.0 - self.n_para**2))
            Ey_inc = CF(0.0 + 0.0j)
            Ez_inc = Ez_inc_val
            Hy_inc = -(n_perp_vac_port / Z0) * Ez_inc_val
            Hz_inc = CF(0.0 + 0.0j)
            
        return Ey_inc, Ez_inc, Hy_inc, Hz_inc

# =====================================================================
# SOLVE HELMHOLTZ 3D IN 2D BOX DOMAIN
# =====================================================================
    def solve_helmholtz_2DHcurl_1DH1_with_pml(self, mesh, geom_mode, box_medium):
        '''
        Solves the Weak Form using standard (Ex, Ey, Ez) coordinate mapping: (Ex = E_3D[0] = Radial, Ey = E_3D[1] = Poloidal, Ez = E_3D[2] = Toroidal)
            - Set Hcurl finite element space (fes) for in plane E field components (Ex, Ez): Hcurl compute E field and needles between meshpoints
            H1 (fes) compute out of 2D plane E field component (Ey) and force the field vector continuity in every space directions
            - 
        '''
        if geom_mode == "1D":
            dirichlet_bnds = "top_wall_pec"

        else: dirichlet_bnds = "left_wall_pec|right_wall_pec|top_wall_pec|bottom_wall_pec"
        
        fes_plane = HCurl(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)
        fes_outplane = H1(mesh, order=3, complex=True, dirichlet=dirichlet_bnds)
        
        if geom_mode == "1D":
            fes_plane, fes_outplane = Periodic(fes_plane), Periodic(fes_outplane)
            
        self.fes = fes_plane * fes_outplane
        print(f'==== \n'
            f'#DoFs = {self.fes.ndof}')

        # --- Dynamic PML Stretching ---
        Sx_r, Sx_im, px = self.cfg['PML']['Sx_r'], self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_r, Sz_im, pz = self.cfg['PML']['Sz_r'], self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']
        
        sign_x = 1.0 if self.box_medium == "VACUUM" else -1.0
        sign_z = 1.0 if self.box_medium == "VACUUM" else -1.0
        
        Stretch_x = 1.0 + (Sx_r - 1.0 + 1j * sign_x * Sx_im) * \
                    IfPos(self.x - self.Lx_plasma, ((self.x - self.Lx_plasma) / self.Lx_pml)**px, 0.0)
            
        if geom_mode == "1D":
            Stretch_z = CF(1.0)
        else:
            Stretch_z = 1.0 + (Sz_r - 1.0 + 1j * sign_z * Sz_im) * \
                        IfPos(-self.z, (-self.z / self.Lz_pml)**pz, \
                        IfPos(self.z - (self.Lz_plasma_src+ 2*self.Lz_wall), ((self.z - (self.Lz_plasma_src+ 2*self.Lz_wall)) / self.Lz_pml)**pz, 0.0))
        # print(f'Stretch_z (RADIAL_ONLY) : {Stretch_z}')
        self.pml_tensor = CF((Stretch_x / Stretch_z, 0.0, 0.0, 
                              0.0, 1.0/(Stretch_x * Stretch_z), 0.0, 
                              0.0, 0.0, Stretch_z / Stretch_x), dims=(3,3))

        self.eff_eps_tensor = CF((
            self.K_xx * (Stretch_z / Stretch_x), self.K_xy * (Stretch_z / Stretch_x), self.K_xz * (Stretch_z / Stretch_x), 
            self.K_yx * (Stretch_x * Stretch_z), self.K_yy * (Stretch_x * Stretch_z), self.K_yz * (Stretch_x * Stretch_z), 
            self.K_zx * (Stretch_x / Stretch_z), self.K_zy * (Stretch_x / Stretch_z), self.K_zz * (Stretch_x / Stretch_z)
        ), dims=(3,3))

        # --- Vector Assembly ---
        self.E_field = GridFunction(self.fes)
        E_plane, E_outplane = self.fes.TrialFunction()
        v_plane, v_outplane = self.fes.TestFunction()

        E_3D = CF((E_plane[0], E_outplane, E_plane[1])) 
        v_3D = CF((v_plane[0], v_outplane, v_plane[1]))

        curl_E_3D = CF(( -grad(E_outplane)[1], -curl(E_plane), grad(E_outplane)[0] ))
        curl_v_3D = CF(( -grad(v_outplane)[1], -curl(v_plane), grad(v_outplane)[0] ))

        Y_11_ref, Y_12_ref, Y_21_ref, Y_22_ref = self.get_port_admittance()

        # bi-linear form:
        a = BilinearForm(self.fes)
        # volume intergral (unknown) terms:
        a += (self.pml_tensor * curl_E_3D * curl_v_3D - \
              self.k0**2 * self.eff_eps_tensor * E_3D * v_3D)*dx
        # boundary intergral (unknown) terms:
        Ez_trace, Ey_trace = E_plane.Trace()[1], E_outplane.Trace()
        vz_trace, vy_trace = v_plane.Trace()[1], v_outplane.Trace()
        a += 1j * self.omega_LH * self.mu0 * ((Y_21_ref * Ey_trace + Y_22_ref * Ez_trace) * vy_trace - \
            (Y_11_ref * Ey_trace + Y_12_ref * Ez_trace) * vz_trace) * ds("bottom_source")

        with TaskManager():
            a.Assemble()
            f=LinearForm(self.fes)

            Ez_inc_spatial = self.build_antenna_source_function()
            Ey_inc, Ez_inc, Hy_inc, Hz_inc = self.get_incident_fields(Ez_inc_spatial)

            Ay = (Y_11_ref * Ey_inc + Y_12_ref * Ez_inc) - Hy_inc
            Az = (Y_21_ref * Ey_inc + Y_22_ref * Ez_inc) - Hz_inc
            f+= 1j * self.omega_LH * self.mu0 * (Az * vy_trace - Ay * vz_trace) * ds("bottom_source")
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
        window_size_toroidal = min(1.5 * self.lambda_para, 0.5*self.Lz_plasma_src)
        print(f'==== \n'
            f'w_size_R: {window_size_radial:.4e}m,    w_size_T: {window_size_toroidal:.4e}m')
        
        # ---------------------------------------------------------------------
        # 1. RADIAL SWR (Right Wall: Reflection along X-axis)
        # ---------------------------------------------------------------------
        x_target_R = self.Lx_plasma * 0.95
        z_sweep = np.linspace(0.1 * self.Lz_plasma_src, 0.9 * self.Lz_plasma_src, 500)
        
        # Récupération des composantes complexes Ez et Hy
        E_sol_3D = CF((self.E_field.components[0][0], self.E_field.components[1], self.E_field.components[0][1]))
        Ez_cf = E_sol_3D[2]
        H_sol_3D = curl_E_3D / (1j * self.omega_LH * self.mu0)
        Hy_cf = H_sol_3D[1]

        # Impédance d'onde radiale locale
        Z0_vac = np.sqrt(self.cfg['CONST']['mu0'] / self.cfg['CONST']['eps0'])
        Zx = Z0_vac / self.n_perp_p
        
        # Décomposition mathématique stricte
        E_plus_cf = 0.5 * (Ez_cf - Zx * Hy_cf)
        E_minus_cf = 0.5 * (Ez_cf + Zx * Hy_cf)
        
        # Coefficient de réflexion local
        Gamma_local_cf = Norm(E_minus_cf) / (Norm(E_plus_cf) + 1e-15)
        Gamma_R_vals = []
        for z in z_sweep:
            print(f'Computing Gamma_R at z = {z:.4e}m')
            try:
                mip = self.mesh(x_target_R, z)
                val = Gamma_local_cf(mip)
                print(f'val: {val:.2e}')
                if not np.isnan(val):
                    Gamma_R_vals.append(val)
            except: pass

        if len(Gamma_R_vals) > 0:
            # La vraie réflexion est la moyenne de la réflexion sur le front d'onde
            Gamma_R = np.mean(Gamma_R_vals)
            peak_z_R = self.Lz_plasma_src / 2.0 # Gardé pour l'affichage visuel
            print(f'Gamma_R (True Directional) = {Gamma_R:.3e}')
        else:
            Gamma_R = 1.0
            peak_z_R = self.Lz_plasma_src / 2.0

        # ---------------------------------------------------------------------
        # 2. TOROIDAL SWR (Top/Bottom Wall: Reflection along Z-axis)
        # ---------------------------------------------------------------------
        if self.n_para.real >= 0:
            z_target_T = (self.Lz_plasma_src+ 2*self.Lz_wall) * 0.95
        else:
            z_target_T = (self.Lz_plasma_src+ 2*self.Lz_wall) * 0.05
            
        x_sweep_T = np.linspace(0.01 * self.Lx_plasma, 0.99 * self.Lx_plasma, 1000)
        mips_target_T = self.mesh(x_sweep_T, np.full_like(x_sweep_T, z_target_T))
        
        E_target_T, valid_x_T = [], []
        for x, mip in zip(x_sweep_T, mips_target_T):
            try:
                val = E_tot_norm(mip).real
                if not np.isnan(val): # --- FIX 4: SAFE FILTER ---
                    E_target_T.append(val)
                    valid_x_T.append(x)
            except: pass
            
        if len(E_target_T) > 0:
            peak_x_T = valid_x_T[np.argmax(E_target_T)]
        else:
            peak_x_T = self.Lx_plasma / 2.0
            print("[WARNING] Toroidal Argmax failed. Using center.")
        
        # B) Measure Line: Vertical line at x = peak_x_T
        z_sweep_T = np.linspace(0.01 * self.Lz_plasma_src, 0.99 * self.Lz_plasma_src, 1000)
        mips_measure_T = self.mesh(np.full_like(z_sweep_T, peak_x_T), z_sweep_T)
        
        E_measure_T, valid_z_T = [], []
        for z, mip in zip(z_sweep_T, mips_measure_T):
            try:
                val = E_tot_norm(mip).real
                if not np.isnan(val):
                    E_measure_T.append(val)
                    valid_z_T.append(z)
            except: pass
            
        valid_z_T = np.array(valid_z_T)
        E_measure_T = np.array(E_measure_T)
        
        if self.n_para.real >= 0:
            mask_T = (valid_z_T <= z_target_T) & (valid_z_T >= z_target_T - window_size_toroidal)
        else:
            mask_T = (valid_z_T >= z_target_T) & (valid_z_T <= z_target_T + window_size_toroidal)
            
        E_window_T = E_measure_T[mask_T]

        if len(E_window_T) > 5:
            SWR_T = max(np.max(E_window_T) / np.max([np.min(E_window_T), 1e-12]), 1.000001)
            Gamma_T = (SWR_T - 1.0) / (SWR_T + 1.0)
            print(f'Gamma_T (SWR) = {Gamma_T:.3e}')
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
        z_limits = np.linspace(0.0, self.Lz_plasma_src, 1000)
        # Compute Integrals
        P_in_net = integrate_flux(S_x_cf, np.full_like(z_limits,0), z_limits, 'z')
        P_out_R = integrate_flux(S_x_cf, np.full_like(z_limits, self.Lx_plasma), z_limits, 'z')
        P_out_T_right = integrate_flux(S_z_cf, x_limits,  np.full_like(x_limits, self.Lz_plasma_src), 'x')
        P_out_T_left = integrate_flux(S_z_cf, x_limits,  np.full_like(x_limits, self.Lz_plasma_src), 'x') # Negative because it points -z
        
        print(f'==== \n'
            f"Net Power Flow (W/m) => P_in: {P_in_net:.4e} | P_out_top: {P_out_R:.4e} \n"
             f"P_out_right: {P_out_T_right:.4e} | P_out_T_left: {P_out_T_left:.4e}")
        
        

        # ---------------------------------------------------------------------
        # Theoretical incident power
        # ---------------------------------------------------------------------
        Z0, E0 = np.sqrt(self.cfg['CONST']['mu0'] / self.cfg['CONST']['eps0']), self.cfg['WAVE']['E_inc']
        
        if self.box_medium == "PLASMA":
            denom = self.n_perp_p * self.n_para * (self.S - (self.n_para**2 + self.n_perp_p**2))
            if np.abs(denom) > 1e-12: # Filtre de sécurité mathématique
                Ey_over_Ez_ratio = 1j * (self.D*(self.P - np.abs(self.n_perp_p)**2)) / denom
            else:
                Ey_over_Ez_ratio = 0.0 + 0.0j
            Sx = (self.P * E0**2) / (2 * Z0) * (Conj(self.n_perp_p)*np.abs(Ey_over_Ez_ratio)**2 + self.P / Conj(self.n_perp_p))
            Sx_val = Sx(self.mesh(1e-5, self.Lz_plasma_src / 2.0)).real
        else:
            # Pure TM Mode in Vacuum
            n_perp_vac_ideal = np.sqrt(complex(1.0 - self.n_para**2))
            Sx_val = ((E0**2) / (2 * Z0) * np.conj(n_perp_vac_ideal)).real

        if self.geom_mode == "RADIAL_ONLY":
            Lz_source = self.cfg['DOMAIN']['Lz_plasma']
        else:
            Lz_wall = 0.15 * self.cfg['DOMAIN']['Lz_plasma']
            Lz_source = self.cfg['DOMAIN']['Lz_plasma'] - (2.0 * Lz_wall)
            
        P_ideal = np.abs(Sx_val) * Lz_source
        print(f'type(P_ideal): {type(P_ideal)}')
        print(f'Sx_val: {Sx_val:.4e} W/m^2\n'
            f'(Lz_source): {Lz_source:.3f} \n'
            f'P_ideal: {P_ideal:.4e} W/m \n'
            f'Gamma = 1-(P_out)/P_ideal: {(np.sqrt(1 - (P_out_R + P_out_T_right - P_out_T_left)/P_ideal)):.4e} \n')
        print(f'P_stix: {self.P:.3e}')


        # =====================================================================
        # JACQUOT 2013 :
        # =====================================================================
        # 1. Theoretical Amplitude Reflection (eta_pred)
        k_perp_real = np.abs(self.n_perp_p.real) * self.k0
        k_para_real = np.abs(self.n_para.real) * self.k0
        
        Sx_im, px = self.cfg['PML']['Sx_im'], self.cfg['PML']['px']
        Sz_im, pz = self.cfg['PML']['Sz_im'], self.cfg['PML']['pz']
        
        # Factor 2.0 because eta is the AMPLITUDE reflection coefficient (|Gamma|)
        eta_pred_R = np.exp(-2.0 * k_perp_real * Sx_im * self.Lx_pml / (px + 1.0))
        eta_pred_T = np.exp(-2.0 * k_para_real * Sz_im * self.Lz_pml / (pz + 1.0))
        
        eta_sim_R = Gamma_R
        eta_sim_T = Gamma_T
        
        # 2. Power flux profile inside the Radial PML
        x_pml_sweep = np.linspace(self.Lx_plasma, self.Lx_tot, 150)
        Px_pml_profile = []
        z_measure = self.Lz_plasma_src / 2.0 
        
        for xi in x_pml_sweep:
            try:
                mip = self.mesh(xi, z_measure)
                Px_pml_profile.append(S_x_cf(mip).real)
            except:
                Px_pml_profile.append(0.0)
                
        Px_pml_profile = np.array(Px_pml_profile)
        # Normalize the profile by the power at the PML entrance (to start at 1.0)
        if np.abs(Px_pml_profile[0]) > 1e-15:
            Px_pml_profile_norm = Px_pml_profile / Px_pml_profile[0]
        else:
            Px_pml_profile_norm = Px_pml_profile

        # ---------------------------------------------------------------------
        # EXPORT DIAGNOSTIC DATA FOR PLOTTING
        # ---------------------------------------------------------------------
        diag_data = {
            'x_target_R': x_target_R, 'peak_z_R': peak_z_R, 'window_size_radial': window_size_radial,
            'z_target_T': z_target_T, 'peak_x_T': peak_x_T, 'window_size_toroidal': window_size_toroidal,
            'n_para': self.n_para, 'P_in_net': P_in_net, 'P_ideal': P_ideal,
            'P_out_R': P_out_R, 'P_out_T_right': P_out_T_right, 'P_out_T_left': P_out_T_left,
            'eta_sim_R': eta_sim_R, 'eta_pred_R': eta_pred_R,
            'eta_sim_T': eta_sim_T, 'eta_pred_T': eta_pred_T,
            'x_pml_sweep': x_pml_sweep,
            'Px_pml_profile_norm': Px_pml_profile_norm,
            'lambda_perp_real': (2.0 * np.pi) / k_perp_real
        }

        return self.E_field, Gamma_R, Gamma_T, diag_data