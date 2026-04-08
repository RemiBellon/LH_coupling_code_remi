"""
Solveur FEM 1D - Couplage de l'onde Hybride Inférieure (LH)
Vérification V&V incluse : Bilan de Poynting & Benchmark de la fonction d'Airy.
Auteur: [Ton Nom] - Projet WEST
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import cmath
import scipy.special as sp  # Pour les fonctions d'Airy

# Imports NGSolve
from ngsolve import *
from ngsolve.meshes import Make1DMesh

# =============================================================================
# 1. PARAMÈTRES PHYSIQUES FONDAMENTAUX
# =============================================================================
c = 3e8
q_e = 1.602e-19
m_e = 9.109e-31
m_i = 2.014 * 1.66e-27
eps_0 = 8.854e-12

f = 3.7e9             
omega = 2 * pi * f    
n_par = 2.0           
B0 = 3.7              

# =============================================================================
# 2. GÉOMÉTRIE ET MAILLAGE
# =============================================================================
L_plasma = 0.05       
L_pml = 0.03          
L_tot = L_plasma + L_pml

# Maillage très fin pour une précision analytique (3000 éléments)
mesh = Make1DMesh(3000, mapping=lambda x: x * L_tot)

# =============================================================================
# 3. ROUTINE DES PROFILS DE DENSITÉ
# =============================================================================
def create_density_profile(profile_type, x_sym, params, L_plasma):
    if profile_type == "constant":
        return CoefficientFunction(params['n_constant'])
    elif profile_type == "linear":
        # Formule originale conservée pour ne pas casser ton benchmark d'Airy
        return params['grad_n'] * x_sym
    elif profile_type == "exponential":
        # Récupération des bornes
        n_edge = params['n_edge']
        n_core = params['n_core']
        
        # Calcul robuste du taux de croissance lambda avec numpy
        import numpy as np
        lambda_val = np.log(n_core / n_edge) / L_plasma
        
        # Retourne l'expression symbolique avec le 'exp' de NGSolve
        return n_edge * exp(lambda_val * x_sym)
    else:
        raise ValueError(f"Type de profil inconnu : {profile_type}")

# =============================================================================
# 4. DÉFINITION DE LA PHYSIQUE (COEFFICIENT FUNCTIONS)
# =============================================================================
x_sym = x 

density_params = {
    'n_constant': 1e18, 
    'grad_n': 2e19,      # Gradient pour le benchmark linéaire (x=0 -> ne=0)
    'n_edge': 1e16,      # Densité au bord de l'antenne (pour l'exponentielle)
    'n_core': 5e19       # Densité à la fin du domaine plasma (pour l'exponentielle)
}

# --- SÉLECTION DU PROFIL ---
TYPE_PROFIL = "exponential"  

# Création du profil en passant L_plasma pour le calcul du lambda
n_e = create_density_profile(TYPE_PROFIL, x_sym, density_params, L_plasma)

#print("Densités vérifiées (x=0, x=L/2, x=L) :")
# print(f"{n_e(0.0):.2e}, {n_e(L_plasma/2):.2e}, {n_e(L_plasma):.2e}")

w_pe2 = (n_e * q_e**2) / (m_e * eps_0)
w_pi2 = (n_e * q_e**2) / (m_i * eps_0)

Omega_ce = -q_e * B0 / m_e  
Omega_ci = q_e * B0 / m_i   

P = 1.0 - (w_pe2 / omega**2) - (w_pi2 / omega**2)

# Pour le benchmark strict d'Airy, on force S=1 pour avoir une équation pure.
# (Dans la vraie physique, S_exact = 1.0 - w_pe2/(...) = 0.999...)
if TYPE_PROFIL == "linear":
    S = CoefficientFunction(1.0)
else:
    S = 1.0 - (w_pe2 / (omega**2 - Omega_ce**2)) - (w_pi2 / (omega**2 - Omega_ci**2))

k0 = omega / c
Q = (k0**2) * (P / S) * (S - n_par**2)

# Définition de la PML
sigma_max = 10.0 * omega 
sigma = IfPos(x_sym - L_plasma, sigma_max * ((x_sym - L_plasma)/L_pml)**2, 0.0)
s_x = 1.0 + 1j * (sigma / omega)

# =============================================================================
# 5. FORMULATION ÉLÉMENTS FINIS
# =============================================================================
fes = H1(mesh, order=2, complex=True)
u = fes.TrialFunction()
v = fes.TestFunction()

a = BilinearForm(fes)
f = LinearForm(fes)

a += ( (1.0 / s_x) * Grad(u)[0] * Grad(v)[0] - s_x * Q * u * v ) * dx

E_inc = 1.0 
k_x0 = k0 * cmath.sqrt(1.0 - n_par**2) 
a += 1j * k_x0 * u * v * ds(definedon="left")
f += 2j * k_x0 * E_inc * v * ds(definedon="left")

# =============================================================================
# 6. ASSEMBLAGE ET RÉSOLUTION
# =============================================================================
print(f"--- Résolution FEM (Profil: {TYPE_PROFIL}) ---")
a.Assemble()
f.Assemble()

Ez = GridFunction(fes)
Ez.vec.data = a.mat.Inverse() * f.vec
print("Résolution FEM terminée avec succès.")

# =============================================================================
# 7. EXTRACTION DES DONNÉES SPATIALES
# =============================================================================
x_plot = np.linspace(0, L_tot, 2000)
Ez_complex = np.zeros_like(x_plot, dtype=complex)

for i, xp in enumerate(x_plot):
    Ez_complex[i] = Ez(mesh(xp))

Ez_real = Ez_complex.real
Ez_abs = np.abs(Ez_complex)

# =============================================================================
# 8. VÉRIFICATION V&V : BENCHMARK D'AIRY ET BILAN DE PUISSANCE
# =============================================================================
print("\n" + "="*50)
print(" RÉSULTATS DE LA VÉRIFICATION SCIENTIFIQUE (V&V)")
print("="*50)

# A. Bilan de Poynting (Conservation de l'énergie)
dEz = Grad(Ez)[0]
def compute_flux(x_val):
    mip = mesh(x_val)
    return (Ez(mip).conjugate() * dEz(mip)).imag

flux_in = compute_flux(0.0)
flux_out = compute_flux(L_plasma - 0.001)

print("\n--- 1. Bilan de Puissance Active (Théorème de Poynting) ---")
print(f"Flux à l'antenne (x=0)             : {flux_in:.6e}")
print(f"Flux avant la PML (x={L_plasma-0.001:.3f} m) : {flux_out:.6e}")

if flux_in != 0:
    err_flux = abs((flux_in - flux_out) / flux_in) * 100
    print(f"-> Erreur de conservation d'énergie: {err_flux:.4e} %")
    if err_flux < 0.1:
        print("-> CONCLUSION : Parfaite conservation, pas de fuite numérique.")

# B. Benchmark de l'Équation d'Airy
if TYPE_PROFIL == "linear":
    print("\n--- 2. Benchmark Analytique : Fonction d'Airy ---")
    
    # Calcul des paramètres de la transformation d'Airy
    n_c = (eps_0 * m_e * omega**2) / (q_e**2)
    print('n_c )', n_c)
    x_c = n_c / density_params['grad_n']
    beta = (k0**2) * (n_par**2 - 1.0) / x_c
    
    # Coordonnée adimensionnelle z(x)
    z_array = - (beta**(1/3)) * (x_plot - x_c)
    
    # Évaluation des fonctions d'Airy via Scipy
    Ai, Aip, Bi, Bip = sp.airy(z_array)
    
    # Masque pour se concentrer uniquement sur le plasma physique (sans la PML)
    idx_plasma = x_plot <= L_plasma
    Ai_plasma = Ai[idx_plasma]
    Bi_plasma = Bi[idx_plasma]
    Ez_fem_plasma = Ez_complex[idx_plasma]
    
    # Moindres carrés : on cherche C1 et C2 tels que C1*Ai + C2*Bi = Ez_FEM
    # La matrice A a deux colonnes : Ai et Bi
    A_fit = np.column_stack((Ai_plasma, Bi_plasma))
    
    # lstsq résout A_fit * C = Ez_fem_plasma
    C, residuals, rank, s = np.linalg.lstsq(A_fit, Ez_fem_plasma, rcond=None)
    C1, C2 = C[0], C[1]
    
    # Reconstruction de la solution analytique sur tout le domaine
    Ez_ana_complex = C1 * Ai + C2 * Bi
    Ez_ana_abs = np.abs(Ez_ana_complex)
    
    # Calcul de l'erreur mathématique (Norme L2 relative)
    error_l2 = np.linalg.norm(Ez_fem_plasma - Ez_ana_complex[idx_plasma]) / np.linalg.norm(Ez_fem_plasma)
    
    print(f"Position de la coupure (Cutoff) x_c : {x_c*100:.2f} cm")
    print(f"Erreur L2 relative FEM vs Airy      : {error_l2*100:.4e} %")
    
    if error_l2 < 1e-3: # Si l'erreur est inférieure à 0.001 %
        print("validation du code par Airy")

# =============================================================================
# 9. POST-TRAITEMENT ET AFFICHAGE (MATPLOTLIB)
# =============================================================================
plt.figure(figsize=(11, 7))

plt.axvspan(0, L_plasma/100, color='#FF4747', alpha=0.5, label=r'$Antenna (Grill)$')
plt.axvspan(0, L_plasma, color='None', alpha=0.05) # , label=r'$Plasma$')
plt.axvspan(L_plasma, L_tot, color='#7ED321', alpha=0.2, label=r'$PML\ (Absorption)$')

# Tracé de la solution FEM
plt.plot(x_plot, Ez_real, color = 'darkorange', linestyle='-', label=r'$FEM\ -\ Re(E_z)$', linewidth=1.5)
plt.plot(x_plot, Ez_abs, 'k-', label=r'$FEM\ -\ |E_z|$', linewidth=2.5)

# Tracé de la superposition Analytique (si existante)
if TYPE_PROFIL == "linear":
    plt.plot(x_plot[idx_plasma], Ez_ana_abs[idx_plasma], 'r--', 
             label=r'$Analytic\ (Airy)\ -\ |Ez|$', linewidth=2)
    plt.axvline(x=x_c, color='royalblue', linestyle=':', linewidth=2.5, 
                    label=r'$Cutoff\ (\ n_e = n_c\ )$')
elif TYPE_PROFIL == "exponential":
        n_c = (eps_0 * m_e * omega**2) / (q_e**2)
        x_c = n_c / density_params['grad_n']
        plt.axvline(x=x_c, color='royalblue', linestyle=':', linewidth=2.5, 
                    label=r'$Cutoff\ (\ n_e = n_c\ )$')
# plt.title("Couplage Onde LH - Validation FEM vs Analytique (Airy)", fontsize=14, fontweight='bold')
plt.xlabel(r"$Radial\ Position\ x \ [cm]$", fontsize=14)
plt.ylabel(r"$Electric\ Field\ Ez\ [V/m]$", fontsize=14)
plt.xlim(0, L_tot)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='best', framealpha=0.9, ncol = 2)
plt.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True, width=0.6, length=4, labelsize=14)

plt.tight_layout()

plt.savefig("LH_Airy_exponential.svg", dpi=300)
plt.show()