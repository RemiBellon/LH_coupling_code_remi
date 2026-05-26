import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from scipy.fft import fft, fftfreq

def plot_2D_wave_map(h5_filepath, figure_save_dir, component='Ez', value_type='real', plot_e_vectors=False):
    print(f"--- Plotting 2D Map from {h5_filepath} ---")
    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        E_comp = h5f[component][:] # Automatically grabs Ex, Ey, or Ez
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lz_exact = h5f.attrs['Lz_exact']

    plot_data = E_comp.real if value_type == 'real' else np.abs(E_comp)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = 'magma' if value_type == 'abs' else 'coolwarm'
    vmax = np.max(plot_data)
    vmin = 0.0 if value_type == 'abs' else -vmax
    
    c = ax.pcolormesh(Z, X, plot_data, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(f"Wave Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=14)

    ax.axhline(y=Lx_plasma, color='white', linestyle='--', linewidth=4, alpha=0.8, 
               label='Radial PML border', path_effects=[pe.withStroke(linewidth=6, foreground="black")])

    if plot_e_vectors and component != 'Ey':
        with h5py.File(h5_filepath, 'r') as h5f:
            Ex_real = h5f['Ex'][:].real
            Ez_real = h5f['Ez'][:].real
        step_x, step_z = max(1, X.shape[0] // 30), max(1, Z.shape[1] // 30)
        ax.quiver(Z[::step_x, ::step_z], X[::step_x, ::step_z], 
                  Ez_real[::step_x, ::step_z], Ex_real[::step_x, ::step_z], 
                  color='cyan', alpha=0.7, pivot='mid', scale_units='xy')
    
    theta_rad, phi_rad = 0, 0
    # B-Field Vector
    bx, bz = np.sin(phi_rad), np.cos(phi_rad) * np.cos(theta_rad)
    norm_b = np.sqrt(bx**2 + bz**2)
    if norm_b > 1e-6:
        bx, bz = bx / norm_b, bz / norm_b
        arrow_z, arrow_x, len_scale = Lz_exact * 0.85, Lx_plasma * 0.85, Lz_exact * 0.08
        ax.quiver(arrow_z, arrow_x, bz * len_scale, bx * len_scale, color='lime', scale=1, scale_units='xy', width=0.005, pivot='tail', zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.text(arrow_z + bz * len_scale, arrow_x + bx * len_scale, r'$\mathbf{B}_0$', color='lime', fontsize=16, fontweight='bold', ha='left', va='bottom', path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_title(f"Lower Hybrid Coupling: {component} component", fontsize=16)
    ax.set_xlabel(r'Toroidal position $z$ [m]', fontsize=16)
    ax.set_ylabel(r'Radial position $x$ [m]', fontsize=16)
    ax.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, right=True, left=True)
    
    plt.tight_layout()
    if figure_save_dir is not None:
        suffix = "_E_vect_field" if plot_e_vectors else ""
        plt.savefig(os.path.join(figure_save_dir, f"Map_{component}_{value_type}{suffix}.png"), dpi=300)
    plt.show()

def plot_1D_radial_benchmark(h5_filepath, figure_save_dir, component='Ez'):
    print(f"--- Plotting 1D Benchmark from {h5_filepath} ---")
    with h5py.File(h5_filepath, 'r') as h5f:
        x_coords = h5f['x_coords'][:]
        E_comp = h5f[component][:]
        omega, n_para = h5f.attrs['omega_wave'], h5f.attrs['n_para']
        n_e, B0 = h5f.attrs['ne_constant'], h5f.attrs['B0_center']
        z_mid = h5f.attrs['z_mid']
        
    E_real, E_abs = E_comp.real, np.abs(E_comp)
    dx = x_coords[1] - x_coords[0]
    nx = len(x_coords)
    
    # Analytical Math (Done safely in Python, away from C++ constraints)
    qe, me, mi, eps0 = 1.6e-19, 9.1e-31, 3.34e-27, 8.854e-12
    w_pe2, w_pi2 = (n_e * qe**2)/(me * eps0), (n_e * qe**2)/(mi * eps0)
    Om_ce, Om_ci = (qe * B0)/me, (qe * B0)/mi
    S = 1 - w_pe2/(omega**2 - Om_ce**2) - w_pi2/(omega**2 - Om_ci**2)
    P = 1 - w_pe2/omega**2 - w_pi2/omega**2
    D = -(Om_ce * w_pe2)/(omega*(omega**2 - Om_ce**2)) + (Om_ci * w_pi2)/(omega*(omega**2 - Om_ci**2))
    
    B_stix = (S + P)*n_para**2 - (S**2 - D**2) - P*S
    C_stix = P * (n_para**2 - (S + D)) * (n_para**2 - (S - D))
    n_perp_sq = (-B_stix + np.sqrt(max(0, B_stix**2 - 4*S*C_stix))) / (2*S)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"1D Radial Benchmark ($z = {z_mid:.2f}$ m) | $n_\\parallel = {n_para}$", fontsize=16)
    
    if n_perp_sq > 0:
        k_perp_theory = (omega / 3e8) * np.sqrt(n_perp_sq)
        lambda_theory = (2 * np.pi) / k_perp_theory
        
        ax1.plot(x_coords, E_real, color='royalblue', lw=2)
        ax1.set_title('Instantaneous Wavefront')
        
        # Zero-Padding FFT for higher resolution peak finding
        pad_factor = 5
        k_axis = fftfreq(len(E_real), d=dx) * 2 * np.pi
        fft_spectrum = np.abs(fft(E_real))
        
        pos_mask = (k_axis > 0)
        k_pos, spec_pos = k_axis[pos_mask], fft_spectrum[pos_mask]
        k_perp_sim = k_pos[np.argmax(spec_pos)]
        lambda_sim = (2 * np.pi) / k_perp_sim
        error_pct = abs(lambda_sim - lambda_theory) / lambda_theory * 100
        
        ax2.plot(k_pos, spec_pos, color='crimson', lw=2, label='Simulation FFT (Padded)')
        ax2.axvline(k_perp_theory, color='black', linestyle='--', lw=2, label=f'Theory $k_\\perp$: {k_perp_theory:.2f} rad/m')
        ax2.set_xlim(0, k_perp_theory * 2)
        ax2.set_title(f'FFT Spectrum | Error: {error_pct:.2f}% | $\\lambda_{{sim}}$={lambda_sim*100:.2f}cm')
        ax2.legend()
    else:
        alpha_theory = (omega / 3e8) * np.sqrt(-n_perp_sq)
        ax1.plot(x_coords, E_abs, color='darkorange', lw=2)
        ax1.set_title('Evanescent Decay')
        
        log_E_abs = np.log(np.maximum(E_abs, 1e-12))
        fit_idx = int(nx * 0.2)
        slope, intercept = np.polyfit(x_coords[:fit_idx], log_E_abs[:fit_idx], 1)
        alpha_sim = -slope
        error_pct = abs(alpha_sim - alpha_theory) / alpha_theory * 100
        
        ax2.plot(x_coords, log_E_abs, color='purple', lw=2)
        ax2.plot(x_coords[:fit_idx], slope * x_coords[:fit_idx] + intercept, color='lime', linestyle='--', lw=3, label=f'Fit ($\\alpha$: {alpha_sim:.2f})')
        ax2.set_title(f'Semi-Log Decay | Theory $\\alpha$: {alpha_theory:.2f} | Error: {error_pct:.2f}%')
        ax2.legend()

    plt.tight_layout()
    if figure_save_dir is not None:
        plt.savefig(os.path.join(figure_save_dir, f"Benchmark_1D_{component}.png"), dpi=300)
    plt.show()

def plot_mesh_convergence(h5_filepath, figure_save_dir):
    print(f"--- Plotting Convergence from {h5_filepath} ---")
    with h5py.File(h5_filepath, 'r') as h5f:
        ppw_array = h5f['ppw_array'][:]
        DoFs = h5f['DoFs'][:]
        Times = h5f['Times'][:]
        Errors = h5f['Errors'][:]
        
    half_idx = len(DoFs) // 2
    slope_err, int_err = np.polyfit(np.log10(DoFs[half_idx:]), np.log10(Errors[half_idx:]), 1)
    slope_time, int_time = np.polyfit(np.log10(DoFs[half_idx:]), np.log10(Times[half_idx:]), 1)

    C_dof = DoFs[-1] / (ppw_array[-1]**2)
    C_time = Times[-1] / (ppw_array[-1]**(2 * slope_time))

    def dof_to_ppw(n): return np.sqrt(np.abs(n) / C_dof)
    def ppw_to_dof(p): return C_dof * (np.abs(p)**2)
    def time_to_ppw(t): return (np.abs(t) / C_time)**(1.0 / (2 * slope_time))
    def ppw_to_time(p): return C_time * (np.abs(p)**(2 * slope_time))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    def format_primary_axis(ax):
        ax.tick_params(direction='in', length=6, width=1.5, bottom=True, top=False, right=True, left=True, which='major')
        ax.tick_params(direction='in', length=4, width=1.0, bottom=True, top=False, right=True, left=True, which='minor')

    # Graph A
    axs[0].loglog(DoFs, Errors, ls='-', color='royalblue', marker='o', mfc='lightsteelblue', mec='royalblue', mew=2, ms=8, zorder=5)
    axs[0].loglog(DoFs[half_idx:], 10**(int_err) * DoFs[half_idx:]**slope_err, 'k--', lw=2)
    axs[0].set_title('Mesh Convergence (Precision)', pad=30)
    axs[0].set_ylabel('$L_2$ Error Norm')
    axs[0].grid(True, which="major", ls="-", alpha=0.3)
    format_primary_axis(axs[0])
    
    ax_ppw_A = axs[0].secondary_xaxis('top', functions=(dof_to_ppw, ppw_to_dof))
    ax_ppw_A.set_xlabel('Points Per Wavelength (PPW)')
    ax_ppw_A.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Graph B
    axs[1].loglog(DoFs, Times, ls='-', color='crimson', marker='o', mfc='lightpink', mec='crimson', mew=2, ms=8, zorder=5)
    axs[1].loglog(DoFs[half_idx:], 10**(int_time) * DoFs[half_idx:]**slope_time, 'k--', lw=2, label=f'O(N^{{{slope_time:.2f}}})')
    axs[1].set_title('Algorithmic Scaling (Cost)', pad=30)
    axs[1].set_ylabel('CPU Time [Seconds]')
    axs[1].grid(True, which="major", ls="-", alpha=0.3)
    axs[1].legend(loc='lower right')
    format_primary_axis(axs[1])
    
    ax_ppw_B = axs[1].secondary_xaxis('top', functions=(dof_to_ppw, ppw_to_dof))
    ax_ppw_B.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Graph C
    axs[2].loglog(Times, Errors, ls='-', color='forestgreen', marker='o', mfc='palegreen', mec='forestgreen', mew=2, ms=8, zorder=5)
    axs[2].set_title('The Pareto Frontier', pad=30)
    axs[2].set_xlabel('Computational Cost [Seconds]')
    axs[2].grid(True, which="major", ls="-", alpha=0.3)
    format_primary_axis(axs[2])
    
    ax_ppw_C = axs[2].secondary_xaxis('top', functions=(time_to_ppw, ppw_to_time))
    ax_ppw_C.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    fig.set_layout_engine('constrained')
    if figure_save_dir is not None:
        plt.savefig(os.path.join(figure_save_dir, "Mesh_Convergence_Linked_Axes.png"), dpi=300)
    plt.show()



def plot_pml_optimization(h5_filepath, figure_save_dir=None):
    """
    VISUALIZATION ONLY: Plots side-by-side comparison of E-Field vs Poynting Reflection.
    """
    print(f"--- Loading Data from {h5_filepath} ---")
    
    with h5py.File(h5_filepath, 'r') as h5f:
        gamma_E = h5f['gamma_E_matrix'][:]
        gamma_S = h5f['gamma_S_matrix'][:]
        s_imag_array = h5f['s_imag_array'][:]
        l_pml_ratios = h5f['l_pml_ratios'][:]
        metadata = json.loads(h5f.attrs['physics_metadata'])

    L_GRID, S_GRID = np.meshgrid(l_pml_ratios, s_imag_array)
    
    # Create Side-by-Side Subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    norm = colors.LogNorm(vmin=1e-5, vmax=1.0) # Lowered vmin to see the deep absorption of S
    
    # Plot 1: Electric Field
    c1 = axs[0].contourf(L_GRID, S_GRID, gamma_E, levels=np.logspace(-5, 0, 25), cmap='magma_r', norm=norm)
    axs[0].set_title(r"E-Field Reflection ($\Gamma_E$)", fontsize=16)
    axs[0].set_ylabel(r'Imaginary Stretching Factor ($S_{imag}$)', fontsize=14)
    
    # Plot 2: Poynting Vector
    c2 = axs[1].contourf(L_GRID, S_GRID, gamma_S, levels=np.logspace(-5, 0, 25), cmap='magma_r', norm=norm)
    axs[1].set_title(r"Poynting Power Reflection ($\Gamma_S$)", fontsize=16)

    # Formatting both axes
    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel(r'Physical PML Length ($L_{PML}$ / $\lambda_\perp$)', fontsize=14)
        ax.tick_params(direction='in', length=6, width=1.5, which='major', bottom=True, top=True, left=True, right=True)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Global Colorbar
    cbar = fig.colorbar(c2, ax=axs, extend='min', pad=0.02)
    cbar.set_label(r'Amplitude Reflection Coefficient ($\Gamma$)', fontsize=14)

    plt.suptitle(f"PML Efficiency Comparison | $n_\\parallel = {metadata['n_para']}$ | PPW = 4.0", fontsize=18, y=1.02)
    plt.tight_layout()
    
    if figure_save_dir is not None:
        plot_path = os.path.join(figure_save_dir, "PML_Optimization_Comparison.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Plot saved to {plot_path}")
