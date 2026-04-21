''' Modules, class & function for the 2D coupling solver '''
import numpy as np 
from ngsolve import * 
from ngsolve.meshes import MakeStructured2DMesh
import cmath 

class LHCouplingSolver:
    def __init__(self, config_dict):
        self.cfg = config_dict  # type = dict
        self.mesh = None        # type = ngsolve.comp.Mesh
        self.fes = None         # type = ngsolve.comp.FESpace
        self.E_field = None     # type = ngsolve.comp.GridFunction
# spatial variable definition:    type = ngsolve.fem.CoefficientFunction
        self.x_sym = x
        self.z_sym = y


# ======================================================================================================================


    def nx_initialization(self):
       # x axe is not periodic:
      # /!\ DANS LE CAS D'UN PROFIL DE DENSITÉ CONSTANT ==> IL FAUDRA LE RENDRE + ROBUSTE POUR DES PROFILES ARBITRAIRES 
        k0_vacuum = self.cfg['WAVE']['omega_wave'] / self.cfg['CONST']['c0']
        n_para = self.cfg['WAVE']['n_para']
        
       # Mesh size and resolution definition: 
        Lx_plasma_target = self.cfg['DOMAIN']['Lx_plasma_target']
        Lz_plasma_target = self.cfg['DOMAIN']['Lz_plasma_target']
        pts_x = self.cfg['DOMAIN']['pts_per_lambda_x']
        n_max = self.cfg['PLASMA']['ne_constant']
        omega_wave = self.cfg['WAVE']['omega_wave']
        B0 = self.cfg['PLASMA']['B0_center_plasma']
        R0 = self.cfg['GEOM']['R0']
        R_ant = self.cfg['GEOM']['R_ant']
        eps_0 = self.cfg['CONST']['eps_0']
        me = self.cfg['CONST']['m_e']
        mi = self.cfg['CONST']['m_i']
        qe = self.cfg['CONST']['q_e']

        Om_ce = qe * B0 / me
        Om_ci = qe * B0 / mi

        w_pe2_max = (n_max * qe**2) / (me * eps_0)
        w_pi2_max = (n_max * qe**2) / (mi * eps_0)
        
        S_max = 1 - w_pe2_max / (omega_wave**2 - Om_ce**2) - w_pi2_max / (omega_wave**2 - Om_ci**2)
        P_max = 1 - w_pe2_max / omega_wave**2 - w_pi2_max / omega_wave**2
        print('S_max = ', S_max , 'P_max = ', P_max, 'k0_vacuum = ', k0_vacuum)
        
        kx_max_sw = k0_vacuum**2 * (P_max / S_max) * (S_max - n_para**2)
        print('kx_max = ', kx_max_sw)
        if abs(kx_max_sw) < 1e-12:
            kx_norm = k0_vacuum
        else: 
            kx_norm = abs(cmath.sqrt(kx_max_sw + 0j))

       # Let's define the thinnest spatial scale in the plasma:
        L_scale_min = (2 * np.pi) / max(k0_vacuum, kx_norm) 
        print('L_scale_min = ', L_scale_min)
        Nx_lambda_plasma = Lx_plasma_target / L_scale_min

       # At least 30 points per lambda 
        nx = int(max(30, Nx_lambda_plasma * pts_x))
        print(f'Lx = {Lx_plasma_target} m. ({Nx_lambda_plasma} lambda in x direction). Resolution nx = {nx}.')
        
        return Lx_plasma_target, Nx_lambda_plasma, nx
    

    def build_mesh(self) -> None:
        '''
        - If meshgrid involves toroidal periodic boundary conditions: Lz must be a multiple of // wavelength.
        - In slab model approx Ey is a scalar while Ex and Ez are vectors. So Ey is solve in H1 and Ex,Ez are solved in Hcurl.
        (H1 and Hcurl are functions mathematical spaces to solve wave equation.)
        '''

       # Physical parameters to compute // wavelength:
        k0_vacuum = self.cfg['WAVE']['omega_wave'] / self.cfg['CONST']['c0']
        n_para = self.cfg['WAVE']['n_para']
        
       # Mesh size and resolution definition: 
        Lx_plasma_target = self.cfg['DOMAIN']['Lx_plasma_target']
        Lz_plasma_target = self.cfg['DOMAIN']['Lz_plasma_target']
        pts_x = self.cfg['DOMAIN']['pts_per_lambda_x']
        pts_z = self.cfg['DOMAIN']['pts_per_lambda_z']
        
       # --- X axis --- (No periodic conditions)
        Lx_plasma_target, Nx_lambda_plasma, nx = self.nx_initialization()

       #  --- Z axis --- (Exact Lz size for periodic bnds conditions)  
        if n_para == 0.0:
            nz = nx
            Lz_exact = self.cfg['DOMAIN']['Lz_plasma_target']
        else: 
            lambda_para = (2*np.pi)/(k0_vacuum * abs(n_para))
            N_lambda_z = max(1, round(Lz_plasma_target/lambda_para))
            Lz_exact = N_lambda_z * lambda_para
            nz = int(N_lambda_z * pts_z)
            print(f' Lz = {Lz_exact:.4f} m ({N_lambda_z} lambda in z direction). Resolution nz = {nz}.')


       # Meshgrid initialization:
        self.mesh = MakeStructured2DMesh(quads=False, nx=nx, ny=nz, mapping=lambda x, y: (x * Lx_plasma_target, y * Lz_exact))
        self.mesh.ngmesh.SetBCName(0, "bottom")
        self.mesh.ngmesh.SetBCName(1, "right")
        self.mesh.ngmesh.SetBCName(2, "top")
        self.mesh.ngmesh.SetBCName(3, "left")

        interp_poly_order = self.cfg['DOMAIN']['interp_poly_order'] 
       # Functions Math Space to solve wave equation.
        V_hcurl = HCurl(self.mesh, order=interp_poly_order, complex=True, dirichlet="top|bottom")
        V_h1 = H1(self.mesh, order=interp_poly_order, complex=True, dirichlet="top|bottom")
        self.fes = FESpace([V_hcurl, V_h1])
        print(f"Degrees of freedom: {self.fes.ndof}")

        self.cfg['DOMAIN']['Lx_tot'] = Lx_plasma_target
        self.cfg['DOMAIN']['Lz_tot'] = Lz_exact
        self.cfg['DOMAIN']['nx_plasma'] = nx
        self.cfg['DOMAIN']['nz_plasma'] = nz


# ======================================================================================================================


    def build_physics(self, density_func):
        '''
        General Stix tensor + density profile function initialization.
        '''
        Lx_plasma = self.cfg['DOMAIN']['Lx_tot'] # POUR LE MOMENT PAS DE PML !!!

       # Constants and parameters
       # TYPE GEOM, B field & freq parameters = float
        omega_wave = self.cfg['WAVE']['omega_wave']
        B0 = self.cfg['PLASMA']['B0_center_plasma']
        R0 = self.cfg['GEOM']['R0']
        R_ant = self.cfg['GEOM']['R_ant']
        eps_0 = self.cfg['CONST']['eps_0']
        me = self.cfg['CONST']['m_e']
        mi = self.cfg['CONST']['m_i']
        qe = self.cfg['CONST']['q_e']

        # Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
       # x_phys = x_sym if x_sym < Lx_plasma, otherwise x_phys = Lx_plasma
        x_in_plasma = IfPos(self.x_sym - Lx_plasma, Lx_plasma, self.x_sym)      # to avoid complex B field and density within pml domain
       # B field direction & intensity (radial dependance)
        theta_B = self.cfg['PLASMA']['theta_B_rad']
        phi_B = self.cfg['PLASMA']['phi_B_rad']
        bx = sin(phi_B)
        by = cos(phi_B) * sin(theta_B)
        bz = cos(phi_B) * cos(theta_B)

       # TYPE = ngsolve.Coefficient.Function
        B_tot = B0 # * (R0/(R_ant - x_in_plasma))          
    
       # Cyclotron frequency (rad/s)
        Om_ce = qe * B_tot / me
        Om_ci = qe * B_tot / mi
    
       # Plasma density profile
        n_e = density_func(x_in_plasma, self.z_sym)
        w_pe2 = (n_e * qe**2) / (me * eps_0)
        w_pi2 = (n_e * qe**2) / (mi * eps_0)

       # General Stix tensor elements: type = float
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
        
       # Matrix format for Stix Tensor for NGSolve
        self.K_tensor = CoefficientFunction(
            (self.K_xx, self.K_xy, self.K_xz,
             self.K_yx, self.K_yy, self.K_yz,
             self.K_zx, self.K_zy, self.K_zz), dims=(3,3)
        )


# ======================================================================================================================


    def solve_system(self):
        '''
        Set FEM weak form and solve it:
        - Use the NGSolve formalisme: TrialFunction, CoefficientFunction
        - Define the bilinear form
        - Define the Admittance Matrix (for source term = linear form = antenna) (before was just a scalar Robin condition in x=0 cf. V2)
        - solve system by inverting the matrix
        '''

       # E field vector and test function vector initialisation in NGSolve:
        (E_xz, E_y), (v_xz, v_y) = self.fes.TrialFunction(), self.fes.TestFunction()
        
       # Initialize the vectors:  
        E_tot = CoefficientFunction((E_xz[0], E_y, E_xz[1]))
        v_tot = CoefficientFunction((v_xz[0], v_y, v_xz[1]))

        # curl(E_vec) = dE_z/dx - dE_x/dz 
        def curl_3d(vec, y_comp):
            return CoefficientFunction((-grad(y_comp)[1], -curl(vec), grad(y_comp)[0]))

       # Bilinear Form: 
       # E and v curl computation
        curE = curl_3d(E_xz, E_y)
        curV = curl_3d(v_xz, v_y)
       
       # kx_plasma computation  
        k0 = self.cfg['WAVE']['k0']
        n_para = self.cfg['WAVE']['n_para']
        kx_plasma = k0 * sqrt((self.P / self.S) * (self.S - n_para**2) + 0j)

        a = BilinearForm(self.fes)
        a += (curE * curV) * dx
        a += - (k0**2) * (self.K_tensor * E_tot) * v_tot * dx
        print('Bilinear form is set.')
        
        # Build Surface Admittance Tensor:
        kz = k0 * n_para
        # perpendicular k (kx) for LH slow wave:  
        kx_sw = k0 * cmath.sqrt((self.P / self.S) * (self.S - n_para**2) + 0j)
        
        # --- Right boundary ---: Admittance 
        kx_fwd = kx_sw 

        # Reflected wave to absorb the wave reflectd to the antenna
        Ay_fwd = (kx_fwd * self.K_xy + kz * self.K_zy) / (kx_fwd * self.K_xx + kz * self.K_zx)
        Az_fwd = (kx_fwd * self.K_xz + kz * self.K_zz) / (kx_fwd * self.K_xx + kz * self.K_zx)
        
        # Admittance Tensor (antenna boundary condition)
        Y_yy_R = 1j * kx_fwd
        Y_zy_R = 1j * kz * Ay_fwd
        Y_zz_R = 1j * (kx_fwd + kz * Az_fwd)

        t_vec = CoefficientFunction((0,1))
        a += Y_yy_R * E_y * v_y * ds(definedon="right")
        a += Y_zy_R * E_y * (t_vec * v_xz.Trace()) * ds(definedon="right")
        a += Y_zz_R * (E_xz.Trace() * v_xz.Trace()) * ds(definedon="right")
        print('Right bilinear form ok')

        # --- Left boundary ---: Admittance
        kx_ref = -kx_sw
        Ay_ref = (kx_ref * self.K_xy + kz * self.K_zy)/ (kx_ref * self.K_xx + kz * self.K_zx)
        Az_ref = (kx_ref * self.K_xz + kz * self.K_zz) / (kx_ref * self.K_xx + kz * self.K_zx)

        Y_yy_L = 1j * kx_ref
        Y_zy_L = 1j * kz * Ay_ref
        Y_zz_L = 1j * (kx_ref + kz * Az_ref)
        
        
        a += Y_yy_L * E_y * v_y * ds(definedon="left")
        a += Y_zy_L * E_y * (t_vec * v_xz.Trace()) * ds(definedon="left")
        a += Y_zz_L * (E_xz.Trace() * v_xz.Trace()) * ds(definedon="left")
        print('Right bilinear form ok')


        # --- Analytic Plane Wave initialization ---  
        phase_x = exp(1j * kx_sw * self.x_sym)
        phase_z = exp(1j * kz * self.z_sym)
        E_amp = self.cfg['WAVE']['E_inc']

        Ez_exact = E_amp * phase_x * phase_z 
        Ey_exact = -Ay_fwd * Ez_exact
        Ex_exact = -Az_fwd * Ez_exact 
        print('Analytic plane Wave ok')

       # --- Source Terme --- On left Boundary = plane wave injection
        f = LinearForm(self.fes)
        Ez_inc_xz = CoefficientFunction((0.0, Ez_exact))
        Ey_inc_scalar = Ey_exact

        # Source S_inc = (Y_fwd - Y_ref) * E_inc
        f += (Y_zz_R - Y_zz_L) * (Ez_inc_xz * v_xz.Trace()) * ds(definedon="left") 
        print('f_zz ok')
        f += (Y_yy_R - Y_yy_L) * Ey_inc_scalar * ds(definedon="left") 
        print('f_yy ok')
        f += (Y_zy_R - Y_zy_L) * Ey_inc_scalar * (t_vec * v_xz.Trace()) * ds(definedon="left") 
        print('Linear Form is set.')

        with TaskManager():
            a.Assemble()
            f.Assemble()
            self.E_field = GridFunction(self.fes)

            # Defining Dirichlet condition 
            vec_exact = CoefficientFunction((Ex_exact, Ez_exact))
            self.E_field.components[0].Set(vec_exact, definedon=self.mesh.Boundaries("top|bottom")) 
            self.E_field.components[1].Set(Ey_exact, definedon=self.mesh.Boundaries("top|bottom")) 

            # Solving ith dirichlet conditions:
            f.vec.data -= a.mat * self.E_field.vec
            inv_mat = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            self.E_field.vec.data += inv_mat * f.vec 

           # for E_field.components: 1st index = (V_Hcurl, V_H1) et 2nd index = (Ex, Ez) or nothing because in V_H1 => only Ey component
            self.E_tot_cf = CoefficientFunction((self.E_field.components[0][0],
                                                self.E_field.components[1], self.E_field.components[0][1]))
            
        print("--- Système solved ---")
        print('--- In solver : self.E_field', self.E_field)
        return self.E_field 