''' Modules, class & function for the 2D coupling solver '''
import numpy as np 
from ngsolve import * 
from ngsolve.meshes import MakeStructured2DMesh
from ngsolve import IfPos
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
    


    def build_mesh(self) -> None:
# TYPE mesh parameters from cfg dict: float
        # Add plasma & PML points to get total points in each direction 
        nx_tot = self.cfg['DOMAIN']['nx_plasma'] + self.cfg['DOMAIN']['nx_pml']
        # Careful! There are 2 PMLs in z direction (left and right)
        nz_tot = self.cfg['DOMAIN']['nz_plasma'] + 2 * self.cfg['DOMAIN']['nz_pml']
        
        Lx_tot = self.cfg['DOMAIN']['Lx_tot']
        Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        Lz_tot = self.cfg['DOMAIN']['Lz_tot']

 
        is_periodic = self.cfg['DOMAIN'].get('periodic_z', False)
        if is_periodic:
                print('Periodic bounds are activated on z (top and bottom)')


        # Mesh aligned within plasma/PML borders
        self.mesh = MakeStructured2DMesh(quads=False, nx=nx_tot, ny=nz_tot, periodic_y=is_periodic,
                                         mapping=lambda x,y: (x*Lx_tot, y*Lz_tot))
        
        # Definition PMLs with NGSolve Native functions
        sigma_max = self.cfg['DOMAIN']['sigma_max_factor']
        self.mesh.SetPML(pml.Cartesian((-1e9, -1e9), (Lx_plasma, 1e9), sigma_max * 1j), ".*")
        # Hcurl for Ex, Ez and H1 for Ey
        order = self.cfg['DOMAIN']['order']
        V_hcurl = HCurl(self.mesh, order=order, complex=True)
                       # dirichlet='right|top|bottom')
        V_h1 = H1(self.mesh, order=order, complex=True)
                  # dirichlet="right|top|bottom")
        self.fes = FESpace([V_hcurl, V_h1])
        print(f"Degrees of freedom: {self.fes.ndof}")



    def build_physics(self, density_func):
    # General Stix tensor + density profile function
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

        Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        # x_phys = x_sym if x_sym < Lx_plasma, otherwise x_phys = Lx_plasma
        x_in_plasma = IfPos(self.x_sym - Lx_plasma, Lx_plasma, self.x_sym)      # to avoid complex B field and density within pml domain
    # B field direction & intensity (radial dependance)
        theta_B = self.cfg['PLASMA']['theta_B_rad']
        phi_B = self.cfg['PLASMA']['phi_B_rad']
        bx = sin(phi_B)
        by = cos(phi_B) * sin(theta_B)
        bz = cos(phi_B) * cos(theta_B)

        B_tot = B0 * (R0/(R_ant - x_in_plasma))          # TYPE = ngsolve.Coefficient.Function
    
    # Cyclotron frequency (rad/s)
        Om_ce = qe * B_tot / me
        Om_ci = qe * B_tot / mi
    
    # Plasma profiles
        n_e = density_func(x_in_plasma, self.z_sym)
        w_pe2 = (n_e * qe**2) / (me * eps_0)
        w_pi2 = (n_e * qe**2) / (mi * eps_0)

    # Basic Stix tensor elements: type = float
        self.S = 1 - w_pe2/(omega_wave**2 - Om_ce**2) - w_pi2/(omega_wave**2 - Om_ci**2)
        self.P = 1 - w_pe2/omega_wave**2 - w_pi2/omega_wave**2
        self.D = Om_ce * w_pe2/(omega_wave*(omega_wave**2 - Om_ce**2)) + Om_ci * w_pi2/(omega_wave*(omega_wave**2 - Om_ci**2))
        Q_stix = self.P - self.S

        K_xx = self.S*(1 - bx**2) + self.P*bx**2
        K_xy = 1j*self.D*bz + Q_stix*bx*by
        K_xz = -1j*self.D*by + Q_stix*bx*bz
        
        K_yx = -1j*self.D*bz + Q_stix*by*bx
        K_yy = self.S*(1 - by**2) + self.P*by**2
        K_yz = 1j*self.D*bx + Q_stix*by*bz
        
        K_zx = 1j*self.D*by + Q_stix*bz*bx
        K_zy = -1j*self.D*bx + Q_stix*bz*by
        K_zz = self.S*(1 - bz**2) + self.P*bz**2
        
        # Matrix format for Stix Tensor for NGSolve
        self.K_tensor = CoefficientFunction(
            (K_xx, K_xy, K_xz,
             K_yx, K_yy, K_yz,
             K_zx, K_zy, K_zz), dims=(3,3)
        )


    def solve_system(self):
        # Récupération des fonctions de test et de forme (scindées)
        (E_vec, E_y), (v_vec, v_y) = self.fes.TrialFunction(), self.fes.TestFunction()
        
        # Reconstruction des vecteurs 3D pour les produits tensoriels
        E_tot = CoefficientFunction((E_vec[0], E_y, E_vec[1]))
        v_tot = CoefficientFunction((v_vec[0], v_y, v_vec[1]))

        # curl(E_vec) = dE_z/dx - dE_x/dz 
        def curl_3d(vec, y_comp):
            return CoefficientFunction((-grad(y_comp)[1], -curl(vec), grad(y_comp)[0]))

        curE = curl_3d(E_vec, E_y)
        curV = curl_3d(v_vec, v_y)

        k0 = self.cfg['WAVE']['k0']
        n_para = self.cfg['WAVE']['n_para']
        kx_plasma = k0 * sqrt((self.P / self.S) * (self.S - n_para**2) + 0j)


        # Forme bilinéaire
        a = BilinearForm(self.fes)
        a += (curE * curV) * dx
        a += - (k0**2) * (self.K_tensor * E_tot) * v_tot * dx

        # Robin condition on H1 compo = Ey
        a += 1j * kx_plasma * E_y * v_y * ds(definedon="left")
        # Robin condition on tangential HCurl
        a += 1j * kx_plasma * (E_vec.Trace() * v_vec.Trace()) * ds(definedon="left")

        # Term Source 
        f = LinearForm(self.fes)
        
        Ez_inc_scalar = self.cfg['WAVE']['E_inc'] * exp(-1j * k0 * n_para * self.z_sym)
        Ez_inc_vec = CoefficientFunction((Ez_inc_scalar, Ez_inc_scalar))
        f += 2j * kx_plasma * (Ez_inc_vec * v_vec.Trace()) * ds(definedon="left")

        with TaskManager():
            a.Assemble()
            f.Assemble()
            self.E_field = GridFunction(self.fes)
            self.E_field.vec.data = a.mat.Inverse(self.fes.FreeDofs()) * f.vec
            
            
            norm_f = Norm(f.vec)
            print(f"[DIAGNOSTIC] Norme du vecteur source (F) : {norm_f:.4e}")
            if norm_f < 1e-12:
                print("!!! ATTENTION : Le terme source est nul. NGSolve ne voit pas l'antenne.")

            self.E_field = GridFunction(self.fes)
            self.E_field.vec.data = a.mat.Inverse(self.fes.FreeDofs()) * f.vec
            norm_E = Norm(self.E_field.vec)
            print(f"[DIAGNOSTIC] Norme de la solution (E_field) : {norm_E:.4e}")
            if norm_E < 1e-12:
                print("!!! ATTENTION : La solution trouvée est nulle.")
            L2_E_vec = sqrt(Integrate(InnerProduct(self.E_field.components[0], self.E_field.components[0]), self.mesh).real)
        
            # Norme L2 du champ scalaire hors-plan (Ey) issu de H1
            L2_E_y = sqrt(Integrate(self.E_field.components[1] * Conj(self.E_field.components[1]), self.mesh).real)
        
            print(f"Énergie L2 du plan (Ex, Ez) [HCurl] : {L2_E_vec:.4e}")
            print(f"Énergie L2 hors-plan (Ey)   [H1]    : {L2_E_y:.4e}")
        
            if L2_E_vec < 1e-10:
                print("!!! ALERTE : Le champ (Ex, Ez) est physiquement nul dans tout le domaine.")
            else:
                print("-> OK : Le champ (Ex, Ez) s'est propagé avec succès.")
        

        

        E_vec_cf = self.E_field.components[0] # Composantes dans le plan (Ex, Ez)
        Ey_cf = self.E_field.components[1]    # Composante hors-plan (Ey)
        
        # On mappe proprement les index spatiaux : (Ex, Ey, Ez)
        self.E_tot_cf = CoefficientFunction((E_vec_cf[0], Ey_cf, E_vec_cf[1]))

        print("--- Système solved ---")
        print('--- In solver : self.E_field', self.E_field)
        return self.E_field 