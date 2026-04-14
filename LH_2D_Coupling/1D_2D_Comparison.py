import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

# Importation de tes classes 2D
import config_2D_coupling_V2 as cfg
from solver_2D_coupling_V2 import LHCouplingSolver

# =====================================================================
# 1. SOLVER 1D (Méthode LH_1D_Coupling)
# =====================================================================
def solve_1d_model(cfg_dict, x_eval):
    """
    Résout le système 1D : d^2E/dx^2 + k0^2 * eps_eff * E = 0
    En utilisant les paramètres exacts du dictionnaire 2D.
    """
    k0 = cfg_dict['WAVE']['k0']
    n_para = cfg_dict['WAVE']['n_para']
    omega = cfg_dict['WAVE']['omega_wave']
    eps0 = cfg_dict['CONST']['eps_0']
    me = cfg_dict['CONST']['m_e']
    qe = cfg_dict['CONST']['q_e']
    
    # Calcul de n_e(x) pour le 1D
    # On utilise ici la même logique PCHIP que le 2D pour la cohérence
    x_pts = np.array(cfg_dict['PLASMA']['lin_prof_x'])
    n_pts = np.array(cfg_dict['PLASMA']['lin_prof_n'])
    spline = PchipInterpolator(x_pts, n_pts)
    
    def get_stix_1d(x):
        ne = spline(x)
        wpe2 = (ne * qe**2) / (me * eps0)
        # On simplifie ici pour l'onde lente (Slow Wave) : P = 1 - wpe2/w2
        P = 1 - wpe2 / omega**2
        return P

    # Equation d'onde 1D (Slow Wave) : d2Ez/dx2 + k0^2 * P * (1 - n_para^2/S) * Ez = 0
    # Pour le benchmark de base, on utilise la relation simplifiée :
    def wave_system(x, y):
        Ez, dEz = y
        P = get_stix_1d(x)
        # kx^2 approx pour LH
        d2Ez = - k0**2 * P * (1 - n_para**2) * Ez 
        return [dEz, d2Ez]

    # Condition initiale à l'antenne (x=0)
    y0 = [cfg_dict['WAVE']['E_inc'], -1j * k0 * np.sqrt(1 - n_para**2 + 0j) * cfg_dict['WAVE']['E_inc']]
    
    sol = solve_ivp(wave_system, [0, x_eval[-1]], y0, t_eval=x_eval, method='RK45')
    return sol.y[0]

# =====================================================================
# 2. SCRIPT DE COMPARAISON
# =====================================================================
def run_benchmark():
    # TYPE: dict
    cfg_shared = {}
    for key, val in cfg.__dict__.items(): # scan cfg file  
        if not key.startswith('__') and isinstance(val, dict): # avoid python system hidden files starting with '__' that are not iterable ==> otherwise the code crash 
            cfg_shared[key] = copy.deepcopy(val)
    # 1. Résolution 2D
    solver2d = LHCouplingSolver(cfg_shared)
    solver2d.build_mesh()
    # On utilise une fonction de densité simple pour le benchmark
    solver2d.build_physics(lambda x, z: cfg_shared['PLASMA']['n_edge']) 
    solver2d.solve_system()
    
    # Extraction 2D (Moyenne sur Z)
    x_eval = np.linspace(0, cfg_shared['DOMAIN']['Lx_plasma'], 200)
    z_mid = cfg_shared['DOMAIN']['Lz_tot'] / 2
    
    Ez_2d = []
    for xv in x_eval:
        mip = solver2d.mesh(xv, z_mid)
        Ez_2d.append(solver2d.E_field.components[2](mip) if mip else 0)
    Ez_2d = np.array(Ez_2d)

    # 2. Résolution 1D
    Ez_1d = solve_1d_model(cfg_shared, x_eval)

    # 3. Post-Processing
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, np.abs(Ez_1d), 'k--', label='1D Analytique (Plan)')
    plt.plot(x_eval, np.abs(Ez_2d), 'r', label='2D NGSolve (Tranche Z)')
    plt.xlabel('x [m]')
    plt.ylabel('|Ez| [V/m]')
    plt.title('Vérification 1D vs 2D (Mêmes paramètres)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_benchmark()