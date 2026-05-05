import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_pml_optimization(h5_filepath, figure_save_dir):
    """
    VISUALIZATION ONLY: Reads HDF5 matrix and generates publication plots.
    """
    print(f"--- Loading Data from {h5_filepath} ---")
    
    # 1. Read the HDF5 Database
    with h5py.File(h5_filepath, 'r') as h5f:
        gamma_matrix = h5f['gamma_matrix'][:]
        s_imag_array = h5f['s_imag_array'][:]
        l_pml_ratios = h5f['l_pml_ratios'][:]
        metadata = json.loads(h5f.attrs['physics_metadata'])
        
    # 2. Mathematical Processing (Finding Optimum)
    target_threshold = 1e-3
    valid_indices = np.argwhere(gamma_matrix < target_threshold)
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmin(valid_indices[:, 1])] # Shortest L_PML
    else:
        best_idx = np.unravel_index(np.argmin(gamma_matrix, axis=None), gamma_matrix.shape)
        
    opt_s = s_imag_array[best_idx[0]]
    opt_l = l_pml_ratios[best_idx[1]]
    opt_g = gamma_matrix[best_idx[0], best_idx[1]]

    # 3. Professional Plotting
    L_GRID, S_GRID = np.meshgrid(l_pml_ratios, s_imag_array)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    norm = colors.LogNorm(vmin=1e-4, vmax=1.0)
    c = ax.contourf(L_GRID, S_GRID, gamma_matrix, levels=np.logspace(-4, 0, 20), cmap='magma_r', norm=norm)
    cbar = fig.colorbar(c, ax=ax, extend='min')
    cbar.set_label(r'Amplitude Reflection Coefficient ($\Gamma$)', fontsize=14)
    
    ax.plot(opt_l, opt_s, '*', color='cyan', markersize=15, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.annotate(f'Optimum:\n$\Gamma$ = {opt_g:.2e}', 
                xy=(opt_l, opt_s), xytext=(20, 10), textcoords='offset points',
                color='black', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='cyan', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black', lw=2))

    ax.set_yscale('log')
    ax.set_title(f"PML Efficiency Scan | $n_\\parallel = {metadata['n_para']}$", fontsize=16)
    ax.set_xlabel(r'Physical PML Length ($L_{PML}$ / $\lambda_\perp$)', fontsize=14)
    ax.set_ylabel(r'Imaginary Stretching Factor ($S_{imag}$)', fontsize=14)
    
    # Clean ticks
    ax.tick_params(direction='in', length=6, width=1.5, which='major', bottom=True, top=True, left=True, right=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(figure_save_dir, "PML_Optimization_Contour_Final.pdf")
    plt.savefig(plot_path, dpi=300)
    print(f"[SUCCESS] Plot saved to {plot_path}")
    plt.show()