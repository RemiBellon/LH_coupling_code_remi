"""
Post-Processing functions 
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_2d_map(solver, component='Ez', resolution=150):
    """
    2D map of E field components or total norm in (x,z) plane
    """
    if solver.E_field is None:
        print("Error: E field has not been calculated yet.")
        return

    print(f"--- 2D map generation of ({component}) ---")
    
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    Lz = solver.cfg['DOMAIN']['Lz_plasma']

# Initialize the plot mesh
    x_vals = np.linspace(1e-6, Lx - 1e-6, resolution)
    z_vals = np.linspace(1e-6, Lz - 1e-6, resolution)
    X, Z = np.meshgrid(x_vals, z_vals)
    
# Matrix to store amplitude values
    Field_abs = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            mip = solver.mesh(X[i, j], Z[i, j])
            
            Ex, Ey, Ez = solver.E_field(mip)
            
            if component == 'Ez':
                Field_abs[i, j] = np.abs(Ez)
            elif component == 'Ey':
                Field_abs[i, j] = np.abs(Ey)
            elif component == 'Ex':
                Field_abs[i, j] = np.abs(Ex)
            elif component == 'norm':
                Field_abs[i, j] = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
            else:
                raise ValueError("Error, wrong component: Should be in : [Ex, Ey, Ez, norm]") 

# Plot initialization
    plt.figure(figsize=(12, 6))

    cmap_plot = plt.pcolormesh(X, Z, Field_abs, shading='gouraud', cmap='magma')
    plt.colorbar(cmap_plot, label=f'Amplitude |{component}| [V/m]')
    
    # Lignes démarquant la PML radiale
    plt.axvline(x=Lx_plasma, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.text(Lx_plasma + 0.002, Lz/2, 'Radial PML', color='white', rotation=90, va='center')
    
    plt.xlabel(r'$Radial\ position\ x\ [m]', fontsize=16)
    plt.ylabel(r'$Toroidal\ position\ z\ [m]', fontsize=16)
    plt.title(f'2D map E field - {component}', fontsize=16)
    
    plt.axis('scaled') 
    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    plt.tight_layout()
    
    plt.savefig(f"Map_2D_{component}.pdf", dpi=300)
    plt.show()




def plot_radial_profile(solver, z_target=0.10):
    """
    Trace l'atténuation de l'amplitude de Ez le long de l'axe x, pour un z donné.
    """
    if solver.E_field is None: return

    print(f"--- Extraction du profil radial à z = {z_target} m ---")
    Lx = solver.cfg['DOMAIN']['Lx_tot']
    Lx_plasma = solver.cfg['DOMAIN']['Lx_plasma']
    
    x_points = np.linspace(1e-6, Lx - 1e-6, 1000)
    Ez_abs = np.zeros_like(x_points)
    
    for i, x_val in enumerate(x_points):
        mip = solver.mesh(x_val, z_target) 
        Ez_abs[i] = np.abs(solver.E_field(mip)[2])

    plt.figure(figsize=(10, 5))
    plt.axvspan(Lx_plasma, Lx, color='red', alpha=0.1, label='PML')
    plt.plot(x_points, Ez_abs, color='blue', linewidth=2, label=r'$|E_z|$')
    plt.xlabel('Profondeur radiale x (m)')
    plt.ylabel('Amplitude (V/m)')
    plt.title(f'Profil Radial de l\'onde LH à z={z_target}m')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("radial_profile.pdf", dpi=300)
    plt.show()