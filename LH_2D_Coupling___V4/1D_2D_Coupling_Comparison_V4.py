import numpy as np
import matplotlib.pyplot as plt
from ngsolve import CoefficientFunction

# Import des modules locaux
import LH_2D_Coupling___V3.config_2D_coupling_V3 as cfg
from solver_2D_coupling_V2 import LHCouplingSolver
from solver_1D_coupling import LH1DSolver

def run_comparison():
    print("="*50)
    print(" DÉMARRAGE DU BENCHMARK 1D vs 2D (Densité Constante)")
    print("="*50)
    
    # 1. On force le profil constant dans le dictionnaire pour ce test
    cfg.PLASMA['profile_type'] = 'constant_density'
    ne_const = cfg.PLASMA['ne_constant']
    
    # Fonctions lambdas simples pour l'injection du profil constant
    density_1d = lambda x_sym: CoefficientFunction(ne_const)
    density_2d = lambda x_sym, z_sym: CoefficientFunction(ne_const)

    # ==========================================
    # RÉSOLUTION 1D
    # ==========================================
    solver_1d = LH1DSolver(cfg.__dict__)
    solver_1d.build_mesh()
    solver_1d.build_physics(density_1d)
    Ez_1d_gf = solver_1d.solve_system()

    # ==========================================
    # RÉSOLUTION 2D
    # ==========================================
    solver_2d = LHCouplingSolver(cfg.__dict__)
    solver_2d.build_mesh()
    solver_2d.build_physics(density_2d)
    E_2d_gf = solver_2d.solve_system()

    # ==========================================
    # EXTRACTION DES DONNÉES SUR L'AXE RADIAL (x)
    # ==========================================
    Lx_tot = cfg.DOMAIN['Lx_tot']
    Lx_plasma = cfg.DOMAIN['Lx_plasma']
    
    x_points = np.linspace(1e-6, Lx_tot - 1e-6, 1000)
    
    # Tableaux pour stocker les amplitudes absolues
    Ez_1d_abs = np.zeros_like(x_points)
    Ex_2d_abs = np.zeros_like(x_points)
    Ey_2d_abs = np.zeros_like(x_points)
    Ez_2d_abs = np.zeros_like(x_points)
    
    # Pour le 2D, on se place au milieu du domaine toroïdal (z = Lz / 2)
    z_mid = cfg.DOMAIN['Lz_tot'] / 2.0

    for i, x_val in enumerate(x_points):
        # --- Extraction 1D ---
        mip_1d = solver_1d.mesh(x_val)
        if mip_1d:
            # En 1D, le champ est un simple scalaire H1, l'extraction directe fonctionne
            Ez_1d_abs[i] = np.abs(Ez_1d_gf(mip_1d))
            
        # --- Extraction 2D ---
        mip_2d = solver_2d.mesh(x_val, z_mid)
        if mip_2d:
            # On utilise le CoefficientFunction global défini dans le solveur 2D
            val_2d = solver_2d.E_tot_cf(mip_2d)
            Ex_2d_abs[i] = np.abs(val_2d[0])
            Ey_2d_abs[i] = np.abs(val_2d[1])
            Ez_2d_abs[i] = np.abs(val_2d[2])
        else:
            Ez_1d_abs[i], Ex_2d_abs[i], Ey_2d_abs[i], Ez_2d_abs[i] = 0.0, 0.0, 0.0, 0.0

    # ==========================================
    # AFFICHAGE COMPARATIF
    # ==========================================
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- SUBPLOT 1: Comparaison directe de Ez (1D vs 2D) ---
    axs[0].axvspan(Lx_plasma, Lx_tot, color='red', alpha=0.1, label='PML Radiale')
    axs[0].plot(x_points, Ez_1d_abs, 'k-', linewidth=3, label=r'$|E_z|$ (Code 1D Scalaire)')
    axs[0].plot(x_points, Ez_2d_abs, 'r--', linewidth=2, label=r'$|E_z|$ (Code 2D Vectoriel)')
    
    axs[0].set_title(r"Comparaison de la composante Toroïdale $|E_z|$", fontsize=14)
    axs[0].set_ylabel('Amplitude [V/m]', fontsize=12)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].legend(loc='upper right')

    # --- SUBPLOT 2: Les composantes "perdues" par le 1D (Ex et Ey capturées par le 2D) ---
    axs[1].axvspan(Lx_plasma, Lx_tot, color='red', alpha=0.1, label='PML Radiale')
    axs[1].plot(x_points, Ex_2d_abs, color='green', linewidth=2, label=r'$|E_x|$ (Radial - 2D uniquement)')
    axs[1].plot(x_points, Ey_2d_abs, color='darkorange', linewidth=2, label=r'$|E_y|$ (Poloïdal - 2D uniquement)')
    
    axs[1].set_title(r"Composantes Radiale et Poloïdale (Exclusives au tenseur 2D)", fontsize=14)
    axs[1].set_xlabel('Position Radiale x [m]', fontsize=14)
    axs[1].set_ylabel('Amplitude [V/m]', fontsize=12)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()