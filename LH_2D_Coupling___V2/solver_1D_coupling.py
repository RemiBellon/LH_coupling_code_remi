import numpy as np 
import cmath
from ngsolve import * 
from ngsolve.meshes import Make1DMesh

class LH1DSolver:
    def __init__(self, config_dict):
        self.cfg = config_dict
        self.mesh = None
        self.fes = None
        self.Ez_field = None
        self.x_sym = x  # Variable spatiale symbolique 1D de NGSolve

    def build_mesh(self) -> None:
        # On utilise exactement les mêmes paramètres de maillage radial que le 2D
        nx_tot = self.cfg['DOMAIN']['nx_plasma'] + self.cfg['DOMAIN']['nx_pml']
        Lx_tot = self.cfg['DOMAIN']['Lx_tot']

        self.mesh = Make1DMesh(nx_tot, mapping=lambda x: x * Lx_tot)
        
        # Le 1D actuel ne résout qu'une composante complexe (Ez)
        self.fes = H1(self.mesh, order=self.cfg['DOMAIN']['order'], complex=True)
        print(f"[Solver 1D] Degrees of freedom: {self.fes.ndof}")

    def build_physics(self, density_func):
        # Récupération des constantes physiques communes
        omega = self.cfg['WAVE']['omega_wave']
        n_par = self.cfg['WAVE']['n_para']
        B0 = self.cfg['PLASMA']['B0_center_plasma']
        q_e = self.cfg['CONST']['q_e']
        m_e = self.cfg['CONST']['m_e']
        m_i = self.cfg['CONST']['m_i']
        eps_0 = self.cfg['CONST']['eps_0']
        c = self.cfg['CONST']['c0']
        
        # Profil de densité (évalué uniquement en x pour le 1D)
        n_e = density_func(self.x_sym)
        
        w_pe2 = (n_e * q_e**2) / (m_e * eps_0)
        w_pi2 = (n_e * q_e**2) / (m_i * eps_0)

        Omega_ce = -q_e * B0 / m_e  
        Omega_ci = q_e * B0 / m_i   

        # Tenseur de Stix (Approximation scalaire 1D)
        P = 1.0 - (w_pe2 / omega**2) - (w_pi2 / omega**2)
        S = 1.0 - (w_pe2 / (omega**2 - Omega_ce**2)) - (w_pi2 / (omega**2 - Omega_ci**2))

        k0 = omega / c
        self.Q = (k0**2) * (P / S) * (S - n_par**2)

        # Construction de la PML Radiale (exactement comme le 2D)
        Lx_plasma = self.cfg['DOMAIN']['Lx_plasma']
        Lx_pml = self.cfg['DOMAIN']['Lx_pml']
        sigma_max = self.cfg['DOMAIN']['sigma_max_factor'] * omega 
        deg = self.cfg['DOMAIN']['degree']
        
        sigma = IfPos(self.x_sym - Lx_plasma, sigma_max * ((self.x_sym - Lx_plasma)/Lx_pml)**deg, 0.0)
        self.s_x = 1.0 + 1j * (sigma / omega)

    def solve_system(self):
        u = self.fes.TrialFunction()
        v = self.fes.TestFunction()
        k0 = self.cfg['WAVE']['k0']
        n_par = self.cfg['WAVE']['n_para']

        a = BilinearForm(self.fes)
        f = LinearForm(self.fes)

        # Formulation faible 1D (Scalaire)
        a += ( (1.0 / self.s_x) * Grad(u)[0] * Grad(v)[0] - self.s_x * self.Q * u * v ) * dx

        E_inc = self.cfg['WAVE']['E_inc']
        k_x0 = k0 * cmath.sqrt(1.0 - n_par**2 + 0j) 
        
        a += 1j * k_x0 * u * v * ds(definedon="left")
        f += 2j * k_x0 * E_inc * v * ds(definedon="left")

        with TaskManager():
            a.Assemble()
            f.Assemble()

            self.Ez_field = GridFunction(self.fes)
            inv_mat = a.mat.Inverse(freedofs=self.fes.FreeDofs())
            self.Ez_field.vec.data = inv_mat * f.vec

        print("--- Système 1D solved ---")
        return self.Ez_field