"""
Post-Processing functions 
"""
import os
import datetime
import json
import numpy as np
from ngsolve import *
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from ngsolve import exp, IfPos, sqrt
import matplotlib.colors as colors

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
def create_density_profile(x_val,z_val, solver):
# Detect if the input is a symbolic FEM variable
    # TYPE is_ngsolve: bool
    is_ngsolve = type(x_val).__name__ == 'CoefficientFunction'
    
    # TYPE prof_type: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'constant_density')

    if prof_type == 'constant_density':
        print('create_density_profile: prof_type = ', prof_type)
        # TYPE ne_constant: float
        ne_constant = solver.cfg['PLASMA']['ne_constant']
        if is_ngsolve:
            # NGSolve accepts native floats as constants
            # TYPE: float
            return ne_constant
        else:
            # Numpy needs an array of the same shape
            # TYPE: np.ndarray
            return np.full_like(x_val, ne_constant)

    elif prof_type == 'exponential_density':
        print('create_density_profile: prof_type = ', prof_type)
        # TYPE Lx_plasma: float
        Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
        # TYPE n_edge, n_core: float
        n_edge = solver.cfg['PLASMA']['lin_prof_n'][0]
        n_core = solver.cfg['PLASMA']['lin_prof_n'][-1]
        # TYPE lambda_val: float
        lambda_val = np.log(n_core) / Lx_plasma
        
        if is_ngsolve:
            # TYPE: ngsolve.CoefficientFunction
            return exp(lambda_val * x_val)
        else:
            # TYPE: np.ndarray
            return np.exp(lambda_val * x_val)

    elif prof_type == 'piecewise_linear_density':
        print('create_density_profile: prof_type = ', prof_type)
        # TYPE x_pts, n_pts: np.ndarray
        x_pts = np.array(solver.cfg['PLASMA']['lin_prof_x'])
        n_pts = np.array(solver.cfg['PLASMA']['lin_prof_n'])
        # TYPE smooth_width: float
        smooth_width = solver.cfg['PLASMA'].get('smooth_width', 0.006)
        print('create_density_profile: smooth_width = ', smooth_width)
        # TYPE slope_0: float
        slope_0 = (n_pts[1] - n_pts[0]) / (x_pts[1] - x_pts[0])
        # TYPE profile: np.ndarray | ngsolve.CoefficientFunction
        profile = n_pts[0] + slope_0 * (x_val - x_pts[0])

        for i in range(1, len(x_pts) - 1):
            # Float conversion forces clean type matching for NGSolve nodes
            # TYPE x_c, n_c: float
            x_c, n_c = float(x_pts[i]), float(n_pts[i])
            # TYPE s_prev, s_next: float
            s_prev = float((n_pts[i] - n_pts[i-1]) / (x_pts[i] - x_pts[i-1]))
            s_next = float((n_pts[i+1] - n_pts[i]) / (x_pts[i+1] - x_pts[i]))
            
            # TYPE L_prev, L_next: np.ndarray | ngsolve.CoefficientFunction
            L_prev = n_c + s_prev * (x_val - x_c)
            L_next = n_c + s_next * (x_val - x_c)
            
            # TYPE dx: np.ndarray | ngsolve.CoefficientFunction
            dx = x_val - x_c
            
            # MATH ENGINE ROUTING
            if is_ngsolve:
                # TYPE H_smooth: ngsolve.CoefficientFunction
                H_smooth = 0.5 * (1.0 + dx / sqrt(dx**2 + smooth_width**2))
            else:
                # TYPE H_smooth: np.ndarray
                H_smooth = 0.5 * (1.0 + dx / np.sqrt(dx**2 + smooth_width**2))
            
            profile = profile + (L_next - L_prev) * H_smooth
            
        return profile
        
    else:
        raise ValueError(f"Unknown profile type '{prof_type}'.")



def compute_density_and_cutoff(solver, resolution_x, resolution_z):
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
    print('Lx_tot= ', Lx_tot, 'Lx_plasma= ', Lx_plasma, 'Lz_tot= ', Lz_tot)
    
    # TYPE x_vals, z_vals: np.ndarray, SHAPE: (resolution,)
    x_vals = np.linspace(1e-6, Lx_tot - 1e-6, resolution_x)
    z_vals = np.linspace(1e-6, Lz_tot - 1e-6, resolution_z)
    
    # TYPE X, Z: np.ndarray, SHAPE: (resolution, resolution)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    print('len(X): ', len(X))
    print('len(Z): ', len(Z))


    # TYPE prof_type: str
    prof_type = solver.cfg['PLASMA'].get('profile_type', 'piecewise_linear_density')
    print(f"Selected density profile: {prof_type}")
    Density_map = create_density_profile(X, Z, solver)
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
    print(f"Computed Cutoff Density (n_c): {n_cutoff:.4e} m^-3")
    
    z_mid_idx = resolution_z // 2
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
def Plot_E_field_2D_Map(solver, save_dir, resolution_x, resolution_z, component):
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
    x_vals = np.linspace(1e-6, Lx - 1e-6, resolution_x)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution_z)
    X, Z = np.meshgrid(x_vals, z_vals, indexing='ij')
    
    print('len(x_vals) = ', len(x_vals), 'len(z_vals) = ', len(z_vals))
    print('len(X) = ', len(X), 'len(Z) = ', len(Z))
    print('resolution_x = ', resolution_x, 'resolution_z = ', resolution_z)
    E_vec_cf = CoefficientFunction(solver.E_field.components[0])
    Ey_cf = CoefficientFunction(solver.E_field.components[1])
    # Matrix to store amplitude values
    Field_abs = np.zeros((resolution_x, resolution_z))
    # print('Starting loops')
    for i in range(resolution_x):
        for j in range(resolution_z):
            mip = solver.mesh(X[i,j], Z[i,j]) 
            if mip:
                # Évaluation unique du champ 3D complet
                val = solver.E_tot_cf(mip) 
                
                Ex, Ey, Ez = val[0], val[1], val[2]
                
                if component == 'Ex':
                    Field_abs[i, j] = np.abs(Ex)
                elif component == 'Ey':
                    Field_abs[i, j] = np.abs(Ey)
                elif component == 'Ez':
                    Field_abs[i, j] = np.abs(Ez)
                elif component == 'norm':
                    Field_abs[i, j] = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
            else:
                Field_abs[i, j] = 0.0
                raise ValueError("Error, wrong component: Should be in : ['Ex', 'Ey', 'Ez', 'norm']")
    
    max_amp = np.max(Field_abs)
    print(f"Max {component} amplitude detected in grid: {max_amp:.4e} V/m")

    plt.figure(figsize=(12, 6))
    
    # 1. Plot 
    norm_scale = colors.LogNorm(vmin=max_amp*5e-2, vmax=max_amp)
    cmap_plot = plt.pcolormesh(Z, X, Field_abs, shading='gouraud', cmap='magma', norm=norm_scale)
    plt.colorbar(cmap_plot, label=f'Amplitude |{component}| [V/m]')
    
    # Horizontal line: radial PML
    plt.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma + Lx*0.015, 'Radial PML', color='white', ha='center', va='bottom')
    
    print('x_cutoff computation')

    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution_x, resolution_z)

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
    print('--- Plot E field Function End---')



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
            val = solver.E_tot_cf(mip)
            # On assigne bien la valeur à l'index [i] du tableau !
            Ex_abs[i] = np.abs(val[0])
            Ey_abs[i] = np.abs(val[1])
            Ez_abs[i] = np.abs(val[2])
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
    print('--- Plot Radial Component at z Target Function End---')


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
        
        for z_val in z_points:
            mip = solver.mesh(x_val, z_val) 
            if mip:
                val = solver.E_tot_cf(mip) 
                
                Ex_sum += np.abs(val[0])
                Ey_sum += np.abs(val[1])
                Ez_sum += np.abs(val[2])
                valid_pts += 1
                
        # On calcule la moyenne
        if valid_pts > 0:
            Ex_avg[i] = Ex_sum / valid_pts
            Ey_avg[i] = Ey_sum / valid_pts
            Ez_avg[i] = Ez_sum / valid_pts

    # Création du Subplot
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    components = [Ex_avg, Ey_avg, Ez_avg]
    print('--- Plot Radial Component Av.: components = ', components)
    print('sum Ey = ', np.sum(Ey_avg))
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
    print('--- Plot Radial Component Averaged Function End---')

def Plot_Density_Profile_2D(solver, save_dir, resolution_x, resolution_z) -> None:
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
    X, Z, prof_type, Density_map, n_cutoff, x_cutoff = compute_density_and_cutoff(solver, resolution_x, resolution_z)

    # --- 1D Slice Extraction ---
    x_mid_idx = resolution_x // 2
    x_slice = X[:, x_mid_idx]
    n_slice = Density_map[:, x_mid_idx]
    # print('n_slice = ', n_slice)
    # Plotting
    # TYPE fig: matplotlib.figure.Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 2D Map
    cmap_plot = ax1.pcolormesh(Z, X, Density_map, shading='nearest', cmap='viridis')
    fig.colorbar(cmap_plot, ax=ax1, label=r'Density $n_e$ [m$^{-3}$]')
    ax1.axhline(y=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Display cutoff layer on 2D map if it exists
    if not np.isnan(x_cutoff):
        ax1.axhline(y=x_cutoff, color='red', linestyle='-', linewidth=1.5, label='Cutoff Layer')
        ax1.legend(loc='lower right')
        
    ax1.set_xlabel('Toroidal position z [m]', fontsize=12)
    ax1.set_ylabel('Radial position x [m]', fontsize=12)
    ax1.set_title('2D map Plasma Density', fontsize=14)
    ax1.set_xlim(0, Lz)
    ax1.set_ylim(0, Lx_plasma)
    
    # 1D Slice
    ax2.plot(x_slice, n_slice, color='indigo', linewidth=2.5, linestyle='', marker='+', ms=1)
    ax2.axvline(x=Lx_plasma, color='black', linestyle='--', linewidth=2, alpha=0.5, label='PML Boundary')
    
    # Display Cutoff on 1D graph
    if not np.isnan(x_cutoff):
        ax2.axhline(y=n_cutoff, color='red', linestyle=':', linewidth=2, label=f'n_c = {n_cutoff:.2e}')
        ax2.text(n_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'n_c = {x_cutoff*100:.2f} cm', color='red')
        ax2.axvline(x=x_cutoff, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ax2.text(x_cutoff + Lx_tot*0.02, n_cutoff*1.5, f'x_c = {x_cutoff*100:.2f} cm', color='red')
    print('prof_type = ', prof_type)
    if prof_type == 'constant_density' or prof_type == 'piecewise_linear_density':
        ax2.set_yscale('linear')
        ax2.set_title('1D Density Profile (linear Scale)', fontsize=14)
    else : 
        ax2.set_yscale('log')
        ax2.set_title('1D Density Profile (Log Scale)', fontsize=14)
    
    ax2.set_xlabel('Radial position x [m]', fontsize=12)
    ax2.set_ylabel(r'Electron Density $n_e$ [m$^{-3}$]', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    ax2.set_xlim(0, Lx_tot)
    
    plt.tight_layout()
    import os
    file_path = os.path.join(save_dir, "Map_Density_Combined.pdf")
    plt.savefig(file_path, dpi=300)
    plt.show()
    print('--- Plot Desnity Profile 2D Function End---')

def Plot_B_Field_2D(solver, save_dir, resolution=100):
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
    print('--- Plot B field Function End---')