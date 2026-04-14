''' Modules, class & function for the 2D coupling solver '''
import numpy as np 
from ngsolve import * 
from ngsolve.meshes import MakeStructured2DMesh
from ngsolve import IfPos

class LHCouplingSolver:
    def __init__(self, config_dict):
        self.cfg = config_dict  # type = dict
        self.mesh = None        # type = ngsolve.comp.Mesh
        self.fes = None         # type = ngsolve.comp.FESpace
        self.E_field = None     # type = ngsolve.comp.GridFunction
# spatial variable definition:    type = ngsolve.fem.CoefficientFunction
        self.x_sym = x
        self.z_sym = y
    

    def build_mesh(self) -> None:
# TYPE mesh parameters from cfg dict: float
        # Add plasma & PML points to get total points in each direction 
        nx_tot = self.cfg['DOMAIN']['nx_plasma'] + self.cfg['DOMAIN']['nx_pml']
        # Careful! There are 2 PMLs in z direction (left and right)
        nz_tot = self.cfg['DOMAIN']['nz_plasma'] + 2 * self.cfg['DOMAIN']['nz_pml']
        
        Lx_tot = self.cfg['DOMAIN']['Lx_tot']
        Lz_tot = self.cfg['DOMAIN']['Lz_tot']

        # Mesh aligned within plasma/PML borders
        self.mesh = MakeStructured2DMesh(quads=False, nx=nx_tot, ny=nz_tot, mapping=lambda x,y: (x*Lx_tot, y*Lz_tot))
        self.fes = FESpace([H1(self.mesh, order=self.cfg['DOMAIN']['order'], complex=True)]*3)
        print(f"Degrees of freedom: {self.fes.ndof}")

    def build_physics(self, density_func):
    # General Stix tensor + density profile function
    # Constants and parameters
# TYPE GEOM, B field & freq parameters = float
        omega_wave = self.cfg['WAVE']['omega_wave']
        B0 = self.cfg['PLASMA']['B0_center_plasma']
        R0 = self.cfg['GEOM']['R0']
        R_ant = self.cfg['GEOM']['R_ant']

    # B field direction & intensity (radial dependance)
        theta_B = self.cfg['PLASMA']['theta_B_rad']
        phi_B = self.cfg['PLASMA']['phi_B_rad']
        bx = sin(phi_B)
        by = cos(phi_B) * sin(theta_B)
        bz = cos(phi_B) * cos(theta_B)

        B_tot = B0 * (R0/(R_ant - self.x_sym))          # type = ngsolve.Coefficient.Function
    
    # Cyclotron frequency (rad/s)
        Om_ce = self.cfg['CONST']['q_e'] * B_tot / self.cfg['CONST']['m_e']
        Om_ci = self.cfg['CONST']['q_e'] * B_tot / self.cfg['CONST']['m_i']
    
    # Plasma profiles
        n_e = density_func(self.x_sym, self.z_sym)
        w_pe2 = (n_e * self.cfg['CONST']['q_e']**2) / (self.cfg['CONST']['m_e']*self.cfg['CONST']['eps_0']) 
        w_pi2 = (n_e * self.cfg['CONST']['q_e']**2)/ (self.cfg['CONST']['m_i']*self.cfg['CONST']['eps_0']) 
    
    # Basic Stix tensor elements: type = float
        S = 1 - w_pe2/(omega_wave**2 - Om_ce**2) - w_pi2/(omega_wave**2 - Om_ci**2)
        D = Om_ce * w_pe2/(omega_wave*(omega_wave**2 - Om_ce**2)) + Om_ci * w_pi2/(omega_wave*(omega_wave**2 - Om_ci**2))
        P = 1 - w_pe2/omega_wave**2 - w_pi2/omega_wave**2
        Q_stix = P - S

        K_xx = S*(1 - bx**2) + P*bx**2
        K_xy = 1j*D*bz + Q_stix*bx*by
        K_xz = -1j*D*by + Q_stix*bx*bz
        
        K_yx = -1j*D*bz + Q_stix*by*bx
        K_yy = S*(1 - by**2) + P*by**2
        K_yz = 1j*D*bx + Q_stix*by*bz
        
        K_zx = 1j*D*by + Q_stix*bz*bx
        K_zy = -1j*D*bx + Q_stix*bz*by
        K_zz = S*(1 - bz**2) + P*bz**2
        
        # Matrix format for Stix Tensor for NGSolve
        self.K_tensor = CoefficientFunction(
            (K_xx, K_xy, K_xz,
             K_yx, K_yy, K_yz,
             K_zx, K_zy, K_zz), dims=(3,3)
        )
# Build PMLs, TYPE = float
        Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        Lx_pml = self.cfg['DOMAIN']['Lx_pml']
        Lz_tot = self.cfg['DOMAIN']['Lz_tot']
        Lz_pml = self.cfg['DOMAIN']['Lz_pml']

        sigma_max = self.cfg['DOMAIN']['sigma_max_factor'] * omega_wave
        deg = self.cfg['DOMAIN']['degree']

# Radial PML (x > Lx_plasma)
        sig_x = IfPos(self.x_sym - Lx_plasma, sigma_max * ((self.x_sym - Lx_plasma)/Lx_pml)**deg, 0.0)
# Toroidal PML (z < Lz_pml & z > Lz - Lz_pml)
        sig_z_bot = IfPos(Lz_pml - self.z_sym, sigma_max * ((Lz_pml - self.z_sym)/Lz_pml)**deg, 0.0)
        sig_z_top = IfPos(self.z_sym - (Lz_tot - Lz_pml), sigma_max * ((self.z_sym - (Lz_tot - Lz_pml))/Lz_pml)**deg, 0.0)
        sig_z = sig_z_bot + sig_z_top

# Stretching complex factor 
        self.s_x = 1.0 + 1j * (sig_x / omega_wave)
        self.s_z = 1 + 1j * (sig_z /omega_wave)

    
    def _curl_2D_with_pml(self, E):
        Ex, Ey, Ez = E
# d/dx -> (1/s_x) * d/dx and same for d/dz 
        dx_Ex, dx_Ey , dx_Ez = Grad(Ex)[0]/self.s_x, Grad(Ey)[0]/self.s_x, Grad(Ez)[0]/self.s_x
        dz_Ex, dz_Ey , dz_Ez = Grad(Ex)[1]/self.s_z, Grad(Ey)[1]/self.s_z, Grad(Ez)[1]/self.s_z

        return CoefficientFunction( (dz_Ey, dz_Ex - dx_Ez, -dx_Ey) ) 


    def solve_system(self):
        '''
        Solve the weak formulation
        '''
        E = self.fes.TrialFunction()         # type = ngsolve.comp.ProxyFunction
        v = self.fes.TestFunction()          # "" "" 
        k0 = self.cfg['WAVE']['k0']          # type = float 

# differential operator in weak formulation
        curl_E = self._curl_2D_with_pml(E)
        curl_v = self._curl_2D_with_pml(v)

# Tensor product term: K * E:                  type = ngsolve.CoefficientFunction     
        K_E_x = self.K_tensor[0,0]*E[0] + self.K_tensor[0,1]*E[1] + self.K_tensor[0,2]*E[2]
        K_E_y = self.K_tensor[1,0]*E[0] + self.K_tensor[1,1]*E[1] + self.K_tensor[1,2]*E[2]
        K_E_z = self.K_tensor[2,0]*E[0] + self.K_tensor[2,1]*E[1] + self.K_tensor[2,2]*E[2]
        det_J = self.s_x * self.s_z
        
        # TYPE: ngsolve.comp.BilinearForm, LinearForm
        a = BilinearForm(self.fes)
        f = LinearForm(self.fes)

        # Formulation volumique : On utilise K_E_x, K_E_y, K_E_z
        a += (curl_E[0]*curl_v[0] + curl_E[1]*curl_v[1] + curl_E[2]*curl_v[2]) * det_J * dx
        a += - (k0**2) * (K_E_x*v[0] + K_E_y*v[1] + K_E_z*v[2]) * det_J * dx
# Robin boundary conditions
        n_para = self.cfg['WAVE']['n_para']
        # type = # TYPE: complex
        kx0 = k0 * np.sqrt(1.0 - n_para**2 + 0j)
        Ez_inc = self.cfg['WAVE']['E_inc']

# Absorption & reflexion term (TE mode)
        a += 1j * kx0 * E[1] * v[1] * ds(definedon = "left") # On Ey
        a += 1j * kx0 * E[2] * v[2] * ds(definedon = "left") # On Ez

# Incident/source term, on Ez only
        f = LinearForm(self.fes)
        Lz_tot = self.cfg['DOMAIN']['Lz_tot']
        Lz_pml = self.cfg['DOMAIN']['Lz_pml']
        source_mask = IfPos(self.z_sym - Lz_pml, 1.0, 0.0) * IfPos((Lz_tot - Lz_pml) - self.z_sym, 1.0, 0.0)
        f += 2j * kx0 * Ez_inc * v[2] * ds(definedon="left")

        with TaskManager():
            a.Assemble()
            f.Assemble()

            self.E_field = GridFunction(self.fes)
        # type = ngsolve.la.BaseMatrix
            inv_mat = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            self.E_field.vec.data = inv_mat * f.vec

            print("--- Système solved ---")
            return self.E_field
        