"""
Post-Processing functions 
"""
import os
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def setup_output_directory(base_folder="Results"):
    """
    Crée un dossier de sauvegarde unique basé sur la date et l'heure.
    Exemple: Results/Run_20260408_103015/
    """
    # Récupération de la date et heure actuelles
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_folder, f"Run_{now}")
    
    # Création du dossier (et du dossier parent s'il n'existe pas)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[SYSTEM] Dossier de sauvegarde créé : {run_dir}")
    return run_dir

def save_configuration(cfg_dict, save_dir):
    """
    Sauvegarde le dictionnaire de configuration au format JSON de manière robuste.
    Ignore les objets systèmes de Python et gère les tableaux Numpy.
    """
    config_path = os.path.join(save_dir, "simulation_parameters.json")
    
    # 1. Filtrage des variables système
    clean_dict = {}
    for key, value in cfg_dict.items():
        # On ignore tout ce qui commence par "__" (ex: __builtins__, __name__, __file__)
        if not key.startswith('__') and isinstance(value, dict):
            clean_dict[key] = value

    # 2. Encodeur sur-mesure pour la robustesse scientifique
    class ScientificEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convertit les tableaux Numpy en listes Python standard
            try:
                # Tente l'encodage classique
                return super().default(obj)
            except TypeError:
                # Si l'objet est vraiment bizarre (une fonction lambda, une classe C++ de NGSolve), 
                # on le transforme en chaîne de caractères (string) pour ne pas crasher.
                return str(obj)

    # 3. Sauvegarde
    with open(config_path, 'w') as f:
        json.dump(clean_dict, f, indent=4, cls=ScientificEncoder)
    
    print(f"[SYSTEM] Configuration sauvegardée dans : {config_path}")

# ======================================================================================================
# ======================================================================================================

def plot_density_profile_2d(solver, save_dir, resolution: int = 150) -> None:
    """
    2D map of density and 1D slice with robust Slow-Wave Cutoff layer calculation.
    """
    print("--- 2D & 1D map generation of (Electron Density n_e) ---")
    
    # TYPE: float
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    # TYPE: np.ndarray, SHAPE: (resolution,)
    x_vals = np.linspace(1e-6, Lx_plasma - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution)
    
    # TYPE: np.ndarray, SHAPE: (resolution, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    # TYPE: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'constant_profile')
    
    # Generate Density Map robustly
    if prof_type == 'exponential':
        print('map density exponential')
        n_edge = solver.cfg['PLASMA']['n_edge']
        n_core = solver.cfg['PLASMA']['n_core']
        lambda_val = np.log(n_core / n_edge) / Lx_plasma
        # TYPE: np.ndarray, SHAPE: (resolution, resolution)
        Density_map = n_edge * np.exp(lambda_val * X)
        
    elif prof_type == 'piecewise_linear':
        print('map density piecewise_linear')
        x_pts = np.array(solver.cfg['PLASMA']['lin_prof_x'])
        n_pts = np.array(solver.cfg['PLASMA']['lin_prof_n'])
        # Use the exact same PCHIP logic as the FEM solver to match perfectly
        spline = PchipInterpolator(x_pts, n_pts)
        Density_map = spline(X)
    else:
        Density_map = np.zeros_like(X)

    # Mask PML region
    Density_map = np.where(X <= Lx_plasma, Density_map, np.nan)

    # --- Robust Cutoff Calculation (P = 0 approximation for Slow Wave) ---
    # TYPE: float
    eps_0 = solver.cfg['CONST']['eps_0']
    m_e = solver.cfg['CONST']['m_e']
    q_e = solver.cfg['CONST']['q_e']
    omega = solver.cfg['WAVE']['omega_wave']
    
    # n_c = eps_0 * m_e * omega^2 / e^2
    # TYPE: float
    n_cutoff = (eps_0 * m_e * omega**2) / (q_e**2)

    # --- 1D Slice Extraction ---
    # TYPE: int
    z_mid_idx = resolution // 2
    # TYPE: np.ndarray, SHAPE: (resolution,)
    x_slice = X[:, z_mid_idx]
    n_slice = Density_map[:, z_mid_idx]
    
    # Find spatial coordinate of cutoff dynamically
    # TYPE: int
    idx_cutoff = np.argmax(n_slice >= n_cutoff)
    # TYPE: float
    x_cutoff = x_slice[idx_cutoff] if n_slice[idx_cutoff] >= n_cutoff else np.nan

    # Plotting
    # TYPE: matplotlib.figure.Figure, tuple
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D Map
    cmap_plot = ax1.pcolormesh(Z, X, Density_map, shading='nearest', cmap='viridis')
    fig.colorbar(cmap_plot, ax=ax1, label=r'Electron Density $n_e$ [m$^{-3}$]')
    ax1.axhline(y=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Display cutoff layer on 2D map if it exists in the domain
    if not np.isnan(x_cutoff):
        ax1.axhline(y=x_cutoff, color='red', linestyle='-', linewidth=1.5, label='Cutoff Layer')
        ax1.legend(loc='lower right')
        
    ax1.set_xlabel('Toroidal position z [m]', fontsize=12)
    ax1.set_ylabel('Radial position x [m]', fontsize=12)
    ax1.set_title('2D map Plasma Density', fontsize=14)
    ax1.set_xlim(0, Lz)
    ax1.set_ylim(0, Lx_plasma)
    
    # 1D Slice
    ax2.plot(x_slice, n_slice, color='indigo', linewidth=2.5)
    ax2.axvline(x=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.5, label='PML Boundary')
    
    # Display Cutoff on 1D graph
    if not np.isnan(x_cutoff):
        ax2.axhline(y=n_cutoff, color='red', linestyle=':', linewidth=2, label=f'n_c = {n_cutoff:.2e}')
        ax2.axvline(x=x_cutoff, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.text(x_cutoff + Lx*0.02, n_cutoff*1.5, f'x_c = {x_cutoff*100:.2f} cm', color='red')

    ax2.set_yscale('log') # Log scale is essential to see edge + core density correctly
    ax2.set_xlabel('Radial position x [m]', fontsize=12)
    ax2.set_ylabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.set_title('1D Density Profile (Log Scale)', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    ax2.set_xlim(0, Lx)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"Map_2D_Density_Combined.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show() 


def plot_radial_components(solver, z_target, save_dir):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) pour une hauteur z donnée.
    """
    if solver.E_field is None: 
        return

    print(f"--- Extraction des profils radiaux (Ex, Ey, Ez) à z = {z_target} m ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot'] 
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma'] 
    
    x_points = np.linspace(1e-6, Lx - 1e-6, 1000)
    Ex_abs = np.zeros_like(x_points)
    Ey_abs = np.zeros_like(x_points)
    Ez_abs = np.zeros_like(x_points)
    
    for i, x_val in enumerate(x_points):
        mip = solver.mesh(x_val, z_target) 
        if mip:
            Ex_abs[i] = np.abs(solver.E_field.components[0](mip))
            Ey_abs[i] = np.abs(solver.E_field.components[1](mip))
            Ez_abs[i] = np.abs(solver.E_field.components[2](mip))
        else:
            Ex_abs[i], Ey_abs[i], Ez_abs[i] = 0.0, 0.0, 0.0

    # Création du Subplot avec x partagé
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    components = [Ex_abs, Ey_abs, Ez_abs]
    print('shape(Ex): ', np.shape(components[0]))
    print('Ex: ', components[0])
    
    labels = [r'$|E_x|$ (Radial)', r'$|E_y|$ (Poloïdal)', r'$|E_z|$ (Toroïdal)']
    colors = ['green', 'darkorange', 'blue']

    for i, ax in enumerate(axs):
        ax.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
        ax.plot(x_points, components[i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_ylabel('Amplitude [V/m]', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Radial position x [m]', fontsize=14)
    fig.suptitle(f'Profil Radial des 3 composantes du champ à z={z_target}m', fontsize=16)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"radial_components.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()



def plot_2d_map(solver, save_dir, component='Ez', resolution=200):
    """
    2D map of E field components or total norm in (x,z) plane (Top View)
    Toroidal (z) is horizontal, Radial (x) is vertical.
    """
    if solver.E_field is None:
        print("Error: E field has not been calculated yet.")
        return

    print(f"--- 2D map generation of ({component}) ---")
    
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    # Initialize the plot mesh
    x_vals = np.linspace(1e-6, Lx - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution)
    X, Z = np.meshgrid(x_vals, z_vals)
    
    # Matrix to store amplitude values
    Field_abs = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            mip = solver.mesh(X[i, j], Z[i, j])
            
            if mip:
                Ex = solver.E_field.components[0](mip)
                Ey = solver.E_field.components[1](mip)
                Ez = solver.E_field.components[2](mip)
                
                if component == 'Ez':
                    Field_abs[i, j] = np.abs(Ez)
                elif component == 'Ey':
                    Field_abs[i, j] = np.abs(Ey)
                elif component == 'Ex':
                    Field_abs[i, j] = np.abs(Ex)
                elif component == 'norm':
                    Field_abs[i, j] = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
                else:
                    raise ValueError("Error, wrong component: Should be in : ['Ex', 'Ey', 'Ez', 'norm']")
            else:
                Field_abs[i, j] = 0.0

    max_amp = np.max(Field_abs)
    print(f"Max {component} amplitude detected in grid: {max_amp:.4e} V/m")

    plt.figure(figsize=(12, 6))
    
    # 1. Plot 
    cmap_plot = plt.pcolormesh(Z, X, Field_abs, shading='gouraud', cmap='magma')
    plt.colorbar(cmap_plot, label=f'Amplitude |{component}| [V/m]')
    
    # Horizontal line: radial PML
    plt.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma + Lx*0.015, 'Radial PML', color='white', ha='center', va='bottom')
    
    # Radial critical density position 
    plt.axhline(y = solver.cfg['PLASMA']['x_crit'], linestyle= '--', linewidth='2', color = 'crimson', label=r'$Radial\ critical\density\ position$')
    # print("solver.cfg[PLASMA]['x_crit']: ", solver.cfg['PLASMA']['x_crit'], ' m')
    # print("solver.cfg[PLASMA]['n_crit']: ", solver.cfg['PLASMA']['n_crit'], ' m-3')


    # 2. Add B direction to graph
    theta_rad = solver.cfg['PLASMA']['theta_B_rad']
    phi_rad = solver.cfg['PLASMA']['phi_B_rad']
    
    # B vector components projection on (x,z)
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    arrow_len = Lx * 0.15 
    dz = bz * arrow_len
    dx = bx * arrow_len
    
    # arrow position
    z0 = Lz * 0.05
    x0 = Lx * 0.85
    
    plt.arrow(z0, x0, dz, dx, color='cyan', width=Lx*0.003, head_width=Lx*0.015, zorder=5)
    plt.text(z0 + dz*1.5, x0 + dx*1.5, r'$\vec{B}_{0}$', color='cyan', fontsize=14, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Toroidal position z [m]', fontsize=14)
    plt.ylabel('Radial position x [m]', fontsize=14)
    plt.title(f'2D map E field - {component} (Top View)', fontsize=16)
    
    # plt.axis('scaled') 
    plt.xlim(0, Lz)
    plt.ylim(0, Lx)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"Map_2D_{component}.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()


def plot_radial_components_averaged(solver, save_dir, z_min=0.05, z_max=0.35, z_res=50):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) en moyennant sur l'axe toroïdal (z).
    z_min et z_max définissent la fenêtre de moyennage (pour éviter les PMLs toroïdales).
    """
    if solver.E_field is None: 
        return

    print(f"--- Extraction et moyennage toroïdal des profils radiaux ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot'] 
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma'] 
    
    # Résolution de l'extraction
    x_points = np.linspace(1e-6, Lx - 1e-6, 500)
    z_points = np.linspace(z_min, z_max, z_res)
    
    Ex_avg = np.zeros(len(x_points))
    Ey_avg = np.zeros(len(x_points))
    Ez_avg = np.zeros(len(x_points))
    
    for i, x_val in enumerate(x_points):
        Ex_sum, Ey_sum, Ez_sum = 0.0, 0.0, 0.0
        valid_pts = 0
        
        # On somme sur tous les points z pour un x donné
        for z_val in z_points:
            mip = solver.mesh(x_val, z_val) 
            if mip:
                Ex_sum += np.abs(solver.E_field.components[0](mip))
                Ey_sum += np.abs(solver.E_field.components[1](mip))
                Ez_sum += np.abs(solver.E_field.components[2](mip))
                valid_pts += 1
                
        # On calcule la moyenne
        if valid_pts > 0:
            Ex_avg[i] = Ex_sum / valid_pts
            Ey_avg[i] = Ey_sum / valid_pts
            Ez_avg[i] = Ez_sum / valid_pts

    # Création du Subplot
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    components = [Ex_avg, Ey_avg, Ez_avg]
    labels = [r'Moyenne $|E_x|$', r'Moyenne $|E_y|$', r'Moyenne $|E_z|$']
    colors = ['green', 'darkorange', 'blue']

    for i, ax in enumerate(axs):
        ax.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
        ax.plot(x_points, components[i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_ylabel('Amplitude [V/m]', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Radial position x [m]', fontsize=14)
    fig.suptitle(f'Profils Radiaux (Moyenne sur z $\in$ [{z_min}, {z_max}] m)', fontsize=16)
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"radial_components_averaged.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()


def plot_b_field_2d(solver, save_dir, resolution=100):
    """
    Génère une carte 2D de l'intensité du champ magnétique B0
    superposée à un champ de vecteurs (quiver) indiquant sa direction.
    """
    print("--- 2D map generation of (Magnetic Field B_tot + Vectors) ---")
    
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    x_vals = np.linspace(1e-6, Lx - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    B0_center = solver.cfg['PLASMA']['B0_center_plasma']
    R0 = solver.cfg['GEOM']['R0']
    R_ant = solver.cfg['GEOM']['R_ant']
    
    # 1. Calcul de la Magnitude (Masquage de la PML)
    B_map = B0_center * (R0 / (R_ant - X))
    B_map = np.where(X <= Lx_plasma, B_map, np.nan)

    # 2. Calcul des composantes pour le champ de vecteurs
    theta_rad = solver.cfg['PLASMA']['theta_B_rad']
    phi_rad = solver.cfg['PLASMA']['phi_B_rad']
    
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    Bx = B_map * bx
    Bz = B_map * bz

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Colormap (Fond)
    cmap_plot = plt.pcolormesh(Z, X, B_map, shading='nearest', cmap='cividis')
    plt.colorbar(cmap_plot, label=r'Intensité Champ Magnétique $|B_{tot}|$ [Tesla]')
    
    # Champ de Vecteurs (Quiver)
    # On saute des points (step) pour ne pas surcharger visuellement la carte avec trop de flèches
    step = resolution // 15
    plt.quiver(Z[::step, ::step], X[::step, ::step], 
               Bz[::step, ::step], Bx[::step, ::step], 
               color='white', alpha=0.9, pivot='mid')
    
    plt.axhline(y=Lx_plasma, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma - Lx*0.02, 'Limite PML (Non-physique)', color='red', ha='center', va='top')
    
    plt.xlabel('Toroidal position z [m]', fontsize=12)
    plt.ylabel('Radial position x [m]', fontsize=12)
    plt.title('2D map Magnetic Field with Vector Directions', fontsize=14)
    
    plt.xlim(0, Lz)
    plt.ylim(0, Lx_plasma)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"Map_2D_B_Field_Vectors.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()