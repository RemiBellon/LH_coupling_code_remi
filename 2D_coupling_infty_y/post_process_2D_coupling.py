"""
Post-Processing functions 
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_map(solver, component='Ez', resolution=150):
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

    # --- CHECKPOINT ---
    max_amp = np.max(Field_abs)
    print(f"[CHECKPOINT] Max {component} amplitude detected in grid: {max_amp:.4e} V/m")

    plt.figure(figsize=(12, 6))
    
    # 1. Plot avec rotation 90° (Z en abscisse, X en ordonnée)
    cmap_plot = plt.pcolormesh(Z, X, Field_abs, shading='gouraud', cmap='magma')
    plt.colorbar(cmap_plot, label=f'Amplitude |{component}| [V/m]')
    
    # Ligne horizontale démarquant la PML radiale
    plt.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lz/2, Lx_plasma + Lx*0.015, 'Radial PML', color='white', ha='center', va='bottom')
    
    # 2. Ajout de la projection du vecteur champ magnétique B0
    theta_rad = solver.cfg['PLASMA']['theta_B_rad']
    phi_rad = solver.cfg['PLASMA']['phi_B_rad']
    
    # Composantes unitaires dans le plan (x,z)
    bx = np.sin(phi_rad)
    bz = np.cos(phi_rad) * np.cos(theta_rad)
    
    # Mise à l'échelle de la flèche (environ 15% de la hauteur radiale du plot)
    arrow_len = Lx * 0.15 
    dz = bz * arrow_len
    dx = bx * arrow_len
    
    # Position de départ de la flèche (en haut à gauche de la zone plasma)
    z0 = Lz * 0.05
    x0 = Lx * 0.85
    
    # Tracé de la flèche et du texte
    plt.arrow(z0, x0, dz, dx, color='cyan', width=Lx*0.003, head_width=Lx*0.015, zorder=5)
    plt.text(z0 + dz*1.5, x0 + dx*1.5, r'$\vec{B}_{0}$', color='cyan', fontsize=14, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Toroidal position z [m]', fontsize=14)
    plt.ylabel('Radial position x [m]', fontsize=14)
    plt.title(f'2D map E field - {component} (Top View)', fontsize=16)
    
    # On force l'échelle 1:1 pour respecter les proportions physiques
    plt.axis('scaled') 
    plt.xlim(0, Lz)
    plt.ylim(0, Lx)
    plt.tight_layout()
    plt.savefig(rf"/home/remi/Perso/Stage/M2_IRFM/Codes/2D_coupling_infty_y/Figures/Map_2D_{component}.pdf", dpi=300)
    plt.show()

def plot_radial_profile(solver, z_target=0.10):
    if solver.E_field is None: 
        return

    print(f"--- Extraction du profil radial à z = {z_target} m ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot'] 
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma'] 
    
    x_points = np.linspace(1e-6, Lx - 1e-6, 1000)
    Ez_abs = np.zeros_like(x_points)
    
    for i, x_val in enumerate(x_points):
        mip = solver.mesh(x_val, z_target) 
        if mip:
            Ez_abs[i] = np.abs(solver.E_field.components[2](mip))
        else:
            Ez_abs[i] = 0.0

    plt.figure(figsize=(10, 5))
    plt.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
    plt.plot(x_points, Ez_abs, color='blue', linewidth=2, label=r'$|E_z|$')
    plt.xlabel('Radial position x [m]')
    plt.ylabel('Amplitude [V/m]')
    plt.title(f'Profil Radial de l\'onde LH à z={z_target}m')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"/home/remi/Perso/Stage/M2_IRFM/Codes/2D_coupling_infty_y/Figures/radial_profile.pdf", dpi=300)
    plt.show()