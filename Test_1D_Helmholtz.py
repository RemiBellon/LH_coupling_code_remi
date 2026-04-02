import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def run_fem_simulation():
    # ==========================================
    # 1. PARAMÈTRES PHYSIQUES & GÉOMÉTRIQUES
    # ==========================================
    f = 50e6              # Fréquence (50 MHz, typique ICRH)
    omega = 2 * np.pi * f
    c = 3e8               # Vitesse lumière
    k0 = omega / c        # Nombre d'onde dans le vide
    
    L_domain = 2.0        # Longueur du domaine (mètres)
    N_elements = 1000     # Nombre d'éléments (finesse du maillage)
    dx = L_domain / N_elements
    
    # Coordonnées des noeuds
    x = np.linspace(0, L_domain, N_elements + 1)
    
    # ==========================================
    # 2. DÉFINITION DU PROFIL DE PLASMA (epsilon)
    # ==========================================
    # On crée un profil de densité fictif :
    # 0 à 0.5m : VIDE (Mur vers Antenne)
    # 0.5m à 0.8m : ZONE DE COUPURE (Densité faible)
    # > 0.8m : COEUR DU PLASMA (Densité forte)
    
    # Epsilon du vide = 1
    # Epsilon du plasma (Modèle Cold Plasma simplifié) : e = 1 - (wp/w)^2
    # Pour voir de l'absorption, on ajoute une petite partie imaginaire (collisions)
    
    epsilon = np.ones(len(x), dtype=complex)
    
    # Création d'une rampe de densité
    x_plasma_start = 0.6
    density_profile = np.zeros_like(x)
    mask = x > x_plasma_start
    density_profile[mask] = (x[mask] - x_plasma_start) / (L_domain - x_plasma_start)
    
    # Fréquence plasma (fictive pour la démo)
    # On s'arrange pour que epsilon passe de positif (vide) à négatif (cutoff) 
    # ou change de régime. Ici on va simuler un indice de réfraction n^2.
    # Pour ICRH : n^2 ~ densité. 
    
    # PROFIL D'INDICE DE RÉFRACTION CARRÉ (n^2)
    # Vide : n^2 = 1
    # Plasma dense : n^2 élevé (ex: 500) pour onde Alfven/ICRH
    n_squared = np.ones(len(x), dtype=complex)
    
    # Zone de "Vide" et "Scrape-off" (n ~ 1) avant 0.6m
    # Zone de Plasma Dense (n >> 1) après 0.6m
    # Ajoutons une zone évanescente artificielle pour l'exemple :
    # Si n^2 < 0, l'onde décroit.
    
    # Modèle pédagogique :
    # - Zone 1 (0.0 - 0.5) : Vide (Propagation normale)
    # - Zone 2 (0.5 - 0.7) : Barrière (Evanescente, n^2 < 0)
    # - Zone 3 (0.7 - 2.0) : Plasma Dense (Propagation lente, n^2 > 0)
    
    n_squared[:] = 1.0 # Vide par défaut
    
    # Barrière (Cutoff)
    mask_barrier = (x > 0.5) & (x < 0.7)
    n_squared[mask_barrier] = -50.0 + 0.5j # Négatif = Évanescent !
    
    # Coeur Plasma
    mask_core = (x >= 0.7)
    n_squared[mask_core] = 200.0 + 10j # Positif = Propagatif + Amortissement
    
    epsilon = n_squared # Dans ce modèle scalaire simple

    # ==========================================
    # 3. ASSEMBLAGE DES MATRICES (FEM 1D)
    # ==========================================
    # Matrice de Rigidité (Stiffness) K : int(dN/dx * dN/dx)
    # Matrice de Masse M : int(epsilon * N * N)
    
    # Pour des éléments linéaires 1D, les matrices locales sont connues.
    # K_loc = 1/dx * [[1, -1], [-1, 1]]
    # M_loc = dx/6 * [[2, 1], [1, 2]] * epsilon_local
    
    # Construction vectorisée (Sparse)
    
    # Diagonale principale et décalées pour la matrice Laplacienne (Dérivée seconde)
    # D2 est l'opérateur d2/dx2 discret
    diagonals_K = [
    np.ones(N_elements, dtype=complex)*(-1), 
    np.ones(N_elements+1, dtype=complex)*2, 
    np.ones(N_elements, dtype=complex)*(-1)]
    K = diags(diagonals_K, [-1, 0, 1], shape=(N_elements+1, N_elements+1)) / dx
    
    # Matrice de Masse (Le terme k0^2 * epsilon * E)
    # Approximation "Mass Lumping" (diagonale) pour simplifier le code
    # C'est moins précis mais plus simple à lire pour un débutant
    M = diags([epsilon], [0], shape=(N_elements+1, N_elements+1)) * dx
    
    # L'opérateur final A = K - k0^2 * M
    # Attention au signe selon la convention ( -d2u/dx2 - k2 u = f )
    A = K - (k0**2) * M

    # ==========================================
    # 4. SOURCE ET CONDITIONS AUX LIMITES
    # ==========================================
    b = np.zeros(N_elements + 1, dtype=complex)
    
    # SOURCE : Antenne placée à x = 0.2m
    idx_antenne = int(0.2 / dx)
    b[idx_antenne] = 1.0 # Courant d'excitation J
    
    # CONDITIONS AUX LIMITES (Dirichlet : E=0 aux bouts)
    # Métal parfait aux deux extrémités
    # On force la solution à 0 aux indices 0 et -1
    # Méthode "Penalty" ou modification directe de la matrice
    
    # Modification directe (Ligne = 0, Diag = 1, RHS = 0)
    # Bord gauche (Mur)
    A = A.tolil() # Conversion pour modification efficace
    A[0, :] = 0
    A[0, 0] = 1
    b[0] = 0
    
    # Bord droit (Centre plasma/Mur opposé)
    A[-1, :] = 0
    A[-1, -1] = 1
    b[-1] = 0
    
    A = A.tocsr() # Conversion pour résolution rapide

    # ==========================================
    # 5. RÉSOLUTION
    # ==========================================
    E = spsolve(A, b)

    # ==========================================
    # 6. VISUALISATION
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Plot de l'amplitude du champ E
    plt.subplot(2, 1, 1)
    plt.plot(x, np.real(E), label='Re(E) - Oscillation instantanée')
    plt.plot(x, np.abs(E), 'k--', linewidth=2, label='|E| - Enveloppe')
    plt.axvline(0.2, color='g', linestyle=':', label='Antenne')
    plt.axvspan(0.5, 0.7, color='red', alpha=0.1, label='Zone Evanescente (Barrière)')
    plt.axvspan(0.7, 2.0, color='blue', alpha=0.1, label='Coeur Plasma (Propagatif)')
    plt.title(f"Simulation FEM 1D Full-Wave (f={f/1e6} MHz)")
    plt.ylabel("Champ Électrique E (V/m)")
    plt.legend()
    plt.grid(True)
    
    # Plot du profil diélectrique
    plt.subplot(2, 1, 2)
    plt.plot(x, np.real(epsilon), 'r', label="Re($\epsilon$) / Indice n²")
    plt.ylabel("Permittivité relative")
    plt.xlabel("Position x (m)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fem_simulation()