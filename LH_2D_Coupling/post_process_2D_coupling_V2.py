"""
Post-Processing functions 
"""
import os
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ======================================================================================================
#   Run Files Management
# ======================================================================================================
def setup_output_directory(base_folder="Results"):
    """
    Make a unique directory based on date and time to save run data. 
    Example: Results/Run_20260408_103015/
    """
    # Recover date & time then make the dir:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_folder, f"Run_{now}")
    
    # Make parent dir if doesn't exist:
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[SYSTEM] Dossier de sauvegarde créé : {run_dir}")
    return run_dir

def save_configuration(cfg_dict, save_dir):
    """
    Save simulation config dict in JASON format. And ignore system python object: Keep only dict and numpy variables 
    """
    config_path = os.path.join(save_dir, "simulation_parameters.json")
    
    # System variables filter:
    clean_dict = {}
    for key, value in cfg_dict.items():
        # On ignore tout ce qui commence par "__" (ex: __builtins__, __name__, __file__)
        if not key.startswith('__') and isinstance(value, dict):
            clean_dict[key] = value

    # Encode the dict data:
    class ScientificEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy table to standard python lists
            try:
                # Try classic encoding:
                return super().default(obj)
            except TypeError:
                # If weird object consvert it to string format to avoid crashing
                return str(obj)

    # Save
    with open(config_path, 'w') as f:
        json.dump(clean_dict, f, indent=4, cls=ScientificEncoder)
    
    print(f"[SYSTEM] Configuration sauvegardée dans : {config_path}")


# ======================================================================================================
#   General Function to Compute Cutoff Density Layer & Position  
# ======================================================================================================
def compute_density_and_cutoff(solver, resolution: int = 150):
    """
    Function to generate the density 2D map and compute the cutoff layer position
    Return: (X,Z)=Meshgrid resolution, Density_map=2Darray of density value, 
            (n_cutoff,x_cutoff)=cutoff density value and cutoff layer position
    """
    print("\n--- Computing Density Map & Cutoff ---")
    
    # TYPE: float
    Lx_tot = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz_tot = solver.cfg['DOMAIN']['Lz_tot']

    # TYPE x_vals, z_vals: np.ndarray, SHAPE: (resolution,)
    x_vals = np.linspace(1e-6, Lx_tot - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz_tot - 1e-6, resolution)
    
    # TYPE X, Z: np.ndarray, SHAPE: (resolution, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    # TYPE prof_type: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'constant_density')
    print(f"Selected density profile: {prof_type}")
    
    # Generate Density Map
    if prof_type == 'constant_density':
        ne_constant = solver.cfg['PLASMA']['ne_constant'] 
        # TYPE Density_map: np.ndarray, SHAPE: (resolution, resolution)
        Density_map = np.full_like(X, ne_constant)
        
    elif prof_type == 'exponential_density':
        n_edge = solver.cfg['PLASMA']['n_edge']
        n_core = solver.cfg['PLASMA']['n_core']
        lambda_val = np.log(n_core / n_edge) / Lx_plasma
        # TYPE Density_map: np.ndarray, SHAPE: (resolution, resolution)
        Density_map = n_edge * np.exp(lambda_val * X)
        
    elif prof_type == 'piecewise_linear_density':
        x_pts = np.array(solver.cfg['PLASMA']['lin_prof_x'])
        n_pts = np.array(solver.cfg['PLASMA']['lin_prof_n'])
        spline = PchipInterpolator(x_pts, n_pts)
        # TYPE Density_map: np.ndarray, SHAPE: (resolution, resolution)
        Density_map = spline(X)
    else:
        print("Unknown profile type. Defaulting to zero array.")
        Density_map = np.zeros_like(X)

    # Mask on PML region (no-physical for density)
    Density_map = np.where(X <= Lx_plasma, Density_map, np.nan)

    # Radial critical density (Cutoff) computation
    # TYPE physical CONST: float
    eps_0 = solver.cfg['CONST']['eps_0']
    m_e = solver.cfg['CONST']['m_e']
    q_e = solver.cfg['CONST']['q_e']
    omega = solver.cfg['WAVE']['omega_wave']
    
    # TYPE n_cutoff: float
    n_cutoff = (eps_0 * m_e * omega**2) / (q_e**2)
    print(f"[INFO] Computed Cutoff Density (n_c): {n_cutoff:.4e} m^-3")
    
    z_mid_idx = resolution // 2
    # TYPE x_slice, n_slice: np.ndarray, SHAPE: (resolution,)
    x_slice = X[:, z_mid_idx]
    n_slice = Density_map[:, z_mid_idx]
    
    # Find spatial coordinate of cutoff dynamically
    # TYPE idx_cutoff: int
    idx_cutoff = np.argmax(n_slice >= n_cutoff)
    
    # TYPE x_cutoff: float
    x_cutoff = x_slice[idx_cutoff] if n_slice[idx_cutoff] >= n_cutoff else np.nan
    
    if not np.isnan(x_cutoff):
        print(f"Cutoff layer dynamically located at x = {x_cutoff:.4f} m")
    else:
        print("Cutoff layer is not reached within the current plasma domain.")
        
    return X, Z, prof_type, Density_map, n_cutoff, x_cutoff


# ======================================================================================================
#   General Function to Compute Cutoff Density Layer & Position  
# ======================================================================================================
def Plot_Density_Profile_2D(solver, save_dir, resolution: int = 150) -> None:
    """
    2D map of density and 1D slice with Slow-Wave Cutoff layer calculation.
    """
    print("\n--- [PLOT] 2D & 1D map generation of Density n_e ---")
    # TYPE DOMAIN size: float
    Lx_tot = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_tot']

    print('Compute Density_map and cutoff layer position.')
    # X,Z = meshgrid resolution; Density_map = 2d map of density values; 
    # n_cutoff,x_cutoff = cutoff density value and layer position relative to given profile density 
    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution)

    # --- 1D Slice Extraction ---
    z_mid_idx = resolution // 2
    x_slice = X[:, z_mid_idx]
    n_slice = Density_map[:, z_mid_idx]
    
    # Plotting
    # TYPE fig: matplotlib.figure.Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D Map
    cmap_plot = ax1.pcolormesh(Z, X, Density_map, shading='nearest', cmap='viridis')
    fig.colorbar(cmap_plot, ax=ax1, label=r'Electron Density $n_e$ [m$^{-3}$]')
    ax1.axhline(y=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Display cutoff layer on 2D map if it exists
    if not np.isnan(x_cutoff):
        ax1.axhline(y=x_cutoff, color='red', linestyle='-', linewidth=1.5, label='Cutoff Layer')
        ax1.legend(loc='lower right')
        
    ax1.set_xlabel('Toroidal position z [m]', fontsize=12)
    ax1.set_ylabel('Radial position x [m]', fontsize=12)
    ax1.set_title('2D map Plasma Density', fontsize=14)
    ax1.set_xlim(0, Lz)
    ax1.set_ylim(0, Lx_tot)
    
    # 1D Slice
    ax2.plot(x_slice, n_slice, color='indigo', linewidth=2.5)
    ax2.axvline(x=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.5, label='PML Boundary')
    
    # Display Cutoff on 1D graph
    if not np.isnan(x_cutoff):
        ax2.axhline(y=n_cutoff, color='red', linestyle=':', linewidth=2, label=f'n_c = {n_cutoff:.2e}')
        ax2.text(n_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'x_c = {x_cutoff*100:.2f} cm', color='red')
        ax2.axvline(x=x_cutoff, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.text(x_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'x_c = {x_cutoff*100:.2f} cm', color='red')

    ax2.set_yscale('log')
    ax2.set_xlabel('Radial position x [m]', fontsize=12)
    ax2.set_ylabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.set_title('1D Density Profile (Log Scale)', fontsize=14)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    ax2.set_xlim(0, Lx_tot)
    
    plt.tight_layout()
    import os
    file_path = os.path.join(save_dir, "Map_Density_Combined.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()


def Plot_E_field_2D_Map(solver, save_dir, component='Ez', resolution=200):
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
    
    print('x_cutoff computation')

    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution)

    plt.axhline(y = x_cutoff, linestyle= '--', linewidth='2', color = 'crimson', label=r'$Radial\ critical\density\ position$')

    print("Add B vect to the graph ")
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





def Plot_Radial_Components_at_z_Target(solver, z_target, save_dir):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) pour une hauteur z donnée.
    """
    if solver.E_field is None: 
        return

    print(f"--- Extract radial profile (Ex, Ey, Ez) at z = {z_target} m ---")
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






def Plot_Radial_Components_Averaged(solver, save_dir, z_min=0.05, z_max=0.35, z_res=50):
    """
    Trace un subplot (3x1) de l'atténuation de |Ex|, |Ey|, et |Ez| 
    le long de l'axe radial (x) en moyennant sur l'axe toroïdal (z).
    z_min et z_max définissent la fenêtre de moyennage (pour éviter les PMLs toroïdales).
    """
    if solver.E_field is None: 
        return

    print(f"--- Extraction and mean on z of radiale profile components ---")
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
    labels = [r'Averaged $|E_x|$', r'Averaged $|E_y|$', r'Averaged $|E_z|$']
    colors = ['green', 'darkorange', 'blue']

    for i, ax in enumerate(axs):
        ax.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
        ax.plot(x_points, components[i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_ylabel('Amplitude [V/m]', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlim(0, max(x_points))
        ax.legend(loc='upper right')

    axs[-1].set_xlabel('Radial position x [m]', fontsize=14)
    fig.suptitle(f'Radial profiles averaged on z $\in$ [{z_min}, {z_max}] m', fontsize=16)
    
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