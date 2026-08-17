from ngsolve import Mesh, CoefficientFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


# ==================================
# Plot density profile 1D
# ==================================
def plot_1d_radial_cut(config, mesh: Mesh,
                       density_cf: CoefficientFunction,
                       z_eval: float,
                       num_points: int = 1000):
    """
    Generates a 1D radial slice of the density profile.
    """
    print(f"--- Generating 1D Radial Cut at z = {z_eval:.4f} m ---")
    x_min = -1 * config.geometry.antenna.dimensions.Lx_wg_active
    x_max = config.geometry.domain.Lx_tot
    print(f'x_min={x_min:.3f} m, x_max={x_max:.3f} m')
    x_coords = np.linspace(x_min, x_max, num_points)
    ne_values = np.zeros_like(x_coords)

    for i, xi in enumerate(x_coords):
        try:
            mip = mesh(xi, z_eval)
            ne_values[i] = density_cf(mip)
        except Exception:
            ne_values[i] = 0.0  # Vacuum/Metal outside meshed domain

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_coords, ne_values, lw=2.5, color='darkblue',
            label=f'Density Profile at $z = {z_eval:.3f}$ m')

    # Physical Boundary Markers
    ax.axvline(x=0.0, color='black', linestyle=':', lw=2)
    ax.text(0.002, np.max(ne_values)*0.5, 'Antenna/Plasma\nInterface',
            rotation=90, va='bottom', fontsize=12, fontweight='bold')
    Lx_plasma = config.geometry.domain.Lx_plasma
    ax.axvline(x=Lx_plasma, color='crimson', linestyle='--', lw=2)
    ax.text(Lx_plasma - 0.002, np.max(ne_values)*0.05, 'Plasma/PML\nInterface',
            rotation=90, va='bottom', ha='right', color='crimson',
            fontsize=12, fontweight='bold')

    # Formatting
    ax.set_ylim(0, np.max(ne_values) * 1.1)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Radial Position $x$ [m]", fontsize=14)
    ax.set_ylabel("Electron Density $n_e$ [m$^{-3}$]", fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Use scientific notation for Y-axis
    plt.tick_params(direction="in", length=6, right=True, top=True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.offsetText.set_fontsize(12)

    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig("1d_radial_density_profile.pdf", dpi=300, bbox_inches='tight')
    plt.show()


# ==================================
# Plot density profile 2D
# ==================================
def plot_2d_density_perturbation(config, mesh: Mesh,
                                 bg_cf: CoefficientFunction,
                                 total_cf: CoefficientFunction,
                                 nx: int = 400, nz: int = 400):
    """
    Generates a publication-ready 2D map of the density perturbation (total - background).
    Uses a diverging colormap to clearly emphasize positive and negative blobs.
    """
    print("--- Generating 2D Density Perturbation Map ---")

    # Calculate the perturbation natively in NGSolve
    delta_cf = total_cf - bg_cf
    x_min = -1 * config.geometry.antenna.dimensions.Lx_wg_active
    x_max = config.geometry.domain.Lx_tot
    print(f'x_min={x_min:.3f} m, x_max={x_max:.3f} m')
    x_vals = np.linspace(x_min, x_max, nx)

    z_min = -1 * config.geometry.domain.Lz_pml
    z_max = config.geometry.domain.Lz_plasma + config.geometry.domain.Lz_pml
    print(f'z_min={z_min:.3f} m, z_max={z_max:.3f} m')
    z_vals = np.linspace(z_min, z_max, nz)

    X, Z_grid = np.meshgrid(x_vals, z_vals)
    Delta_n = np.zeros_like(X)

    for i in range(nx):
        for j in range(nz):
            try:
                mip = mesh(X[j, i], Z_grid[j, i])
                Delta_n[j, i] = delta_cf(mip)
            except Exception:
                Delta_n[j, i] = 0.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a diverging colormap centered at 0 (e.g., RdBu_r or coolwarm)
    vmax = np.max(np.abs(Delta_n))
    if vmax == 0: vmax = 1.0 # Fallback if no perturbations exist

    c = ax.pcolormesh(Z_grid, X, Delta_n, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, shading='gouraud')

    # Overlay contour lines for topographical clarity
    ax.contour(Z_grid, X, Delta_n, levels=15, colors='black', alpha=0.3, linewidths=0.5)

    ax.set_aspect('equal') # Physical space must not be visually stretched
    ax.set_title(r"Density Perturbation $\Delta n_e$ [m$^{-3}$]", fontsize=16, pad=15)
    ax.set_xlabel("Radial Position $x$ [m]", fontsize=14)
    ax.set_ylabel("Toroidal Position $z$ [m]", fontsize=14)

    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig("2d_density_perturbation.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# ==================================
# Plot density profile 3D
# ==================================
def plot_density_3d(config, mesh: Mesh, density_cf: CoefficientFunction,
                                nx: int = 300, nz: int = 300):

    print("--- Generating Publication-Quality 3D Density Plot ---")
    x_min = -1 * config.geometry.antenna.dimensions.Lx_wg_active
    x_max = config.geometry.domain.Lx_tot
    print(f'x_min={x_min:.3f} m, x_max={x_max:.3f} m')
    x_vals = np.linspace(x_min, x_max, nx)

    z_min = -1 * config.geometry.domain.Lz_pml
    z_max = config.geometry.domain.Lz_plasma + config.geometry.domain.Lz_pml
    print(f'z_min={z_min:.3f} m, z_max={z_max:.3f} m')
    z_vals = np.linspace(z_min, z_max, nz)
    X, Z_grid = np.meshgrid(x_vals, z_vals)
    Density = np.zeros_like(X)

    # Evaluate CoefficientFunction
    for i in range(nx):
        for j in range(nz):
            try:
                mip = mesh(X[j, i], Z_grid[j, i])
                Density[j, i] = density_cf(mip)
            except Exception:
                Density[j, i] = 0.0

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a light source for topographical shadowing
    ls = LightSource(azdeg=315, altdeg=45)

    # Shade the density data, blending the colormap with the simulated shadows
    rgb = ls.shade(Density, cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')

    # Plot the surface using the shaded facecolors, disabling default matplotlib shading
    surf = ax.plot_surface(X, Z_grid, Density, facecolors=rgb,
                           linewidth=0, antialiased=True, shade=False)

    # Project contour lines onto the "floor" to highlight the blob footprints
    cset = ax.contour(X, Z_grid, Density, zdir='z', offset=0, levels=25,
                      cmap='viridis', alpha=0.6, linewidths=1.5)

    # Lock in a strict, optimal viewing angle
    ax.view_init(elev=35, azim=235)

    # Formatting
    ax.set_xlabel("Radial Position x [m]", fontsize=14, labelpad=10)
    ax.set_ylabel("Toroidal Position z [m]", fontsize=14, labelpad=10)
    ax.set_zlabel("Density [m^-3]", fontsize=14, labelpad=15)

    # Prevent axes from cutting off large tick numbers
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_zlim(0, np.max(Density) * 1.05)

    # Fake a scalar mappable for the colorbar since we used facecolors directly
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m.set_array(Density)
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label("Density [m^-3]", fontsize=14)

    plt.tight_layout()
    plt.show()