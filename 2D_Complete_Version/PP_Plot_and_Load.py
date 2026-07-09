import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from scipy.fft import fft, fftfreq, fftshift
from ngsolve import *

def plot_2D_wave_map(h5_filepath, figure_save_dir, mode, component='Ez', value_type='real', antenna_grill=None, 
                     Lx_wg=0.0, Lz_wall=0.02, plot_poynting=True, show_windows_R=True, show_windows_T=True, Poynting_box=True):
    print(f"--- Plotting 2D Map from {h5_filepath} ---")
    diag_data_loaded = {}
    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        if component == 'E_norm':
            E_comp = h5f['E_norm'][:]
            plot_data = E_comp
            cmap = 'magma'
            vmin, vmax = 0.0, np.max(plot_data)/10
        else: 
            E_comp = h5f[component][:] # Automatically grabs Ex, Ey, or Ez
            plot_data = E_comp.real if value_type == 'real' else np.abs(E_comp)
            cmap = 'magma' if value_type == 'abs' else 'coolwarm'
            
            # --- FIX 2: PREVENT SINGULARITY FLATTENING ---
            vmax = np.percentile(plot_data, 99.5) # Ignore extreme singularities at metal corners
            vmin = 0.0 if value_type == 'abs' else -vmax
        
        if show_windows_R:
            print(f'--- Show radial windows is True ---')
            if 'x_target_R' in h5f.attrs:
                diag_data_loaded['x_target_R'] = h5f.attrs['x_target_R']
                diag_data_loaded['peak_z_R'] = h5f.attrs['peak_z_R']
                diag_data_loaded['window_size_radial'] = h5f.attrs['window_size_radial']
                print(f'Loaded diagnostic data: x_target_R = {diag_data_loaded["x_target_R"]}, peak_z_R = {diag_data_loaded["peak_z_R"]}, window_size_radial = {diag_data_loaded["window_size_radial"]}')
        if show_windows_T:
            print(f'--- Show toroidal windows is True ---')
            if 'z_target_T' in h5f.attrs:
                diag_data_loaded['z_target_T'] = h5f.attrs['z_target_T']
                diag_data_loaded['peak_x_T'] = h5f.attrs['peak_x_T']
                diag_data_loaded['window_size_toroidal'] = h5f.attrs['window_size_toroidal']
                diag_data_loaded['n_para'] = h5f.attrs.get('n_para', 2.0)

            
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lx_tot = h5f.attrs['Lx_tot']
        Lz_plasma = h5f.attrs['Lz_plasma']
        Sx, Sz = h5f['Sx'][:], h5f['Sz'][:]
        k_para, k_perp_p = h5f.attrs['k_para'], h5f.attrs['k_perp_p']

    fig, ax = plt.subplots(figsize=(14, 8))
    
    extent = [Z.min(), Z.max(), X.min(), X.max()]
    c = ax.imshow(plot_data, origin='lower', extent=extent, cmap=cmap, 
                  vmin=vmin, vmax=vmax, aspect='auto', interpolation='bicubic')
    cbar = fig.colorbar(c, ax=ax) #, label=f'{component} field ({value_type if component != '|E|' else 'Absolute'})')
    cbar.set_label(f"Wave Field ${value_type.capitalize()}({component})$ [V/m]", fontsize=14)

    if plot_poynting and component:
        strm = ax.streamplot(Z[0,:], X[:, 0], Sz, Sx, color='black', linewidth=1.5, density=0.8, arrowstyle='->', arrowsize=1.5)
        z_center, x_center = Lz_plasma * 0.5, Lx_plasma * 0.5
        k_norm = np.sqrt(k_para**2 + k_perp_p**2)
        k_scale = 0.15 * Lx_plasma
        kx_plot, kz_plot = (k_perp_p/k_norm) * k_scale, (k_para/k_norm) * k_scale
        print(f'kx_plot: {kx_plot:.2e}, kz_plot: {kz_plot:.2e}')
        ax.quiver(z_center, x_center, kz_plot, kx_plot, color='yellow', scale=1, scale_units='xy', width=0.008, pivot='tail', zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax.text(z_center + 1.5*kz_plot, x_center + kx_plot, r'$\mathbf{k}$', color='Yellow', fontsize=18, fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    if Poynting_box:
        print(f'Poynting_box = True')
        ax.axhline(y=1e-5, color="lime", linestyle='-', lw=4, label='Source Power ($P_{in}$)')
        ax.axhline(y=0.95 * Lx_plasma, color='darkorange', linestyle=':', label='Radial Power ($P_{out,\ R}$)')
        if mode in ['RADIAL_ONLY', 'FULL_2D']:
            ax.axvline(x=0.95 * Lz_plasma, color='gold', linestyle=':', lw=4, label='Toroidal Power ($P_{out,\ right}$)')
            ax.axvline(x=0.05 * Lz_plasma, color='gold', linestyle=':', lw=4, label='Toroidal Power ($P_{out,\ left}$)')

    # B-Field Vector
    phi_rad = theta_rad = 0
    bx, bz = np.sin(phi_rad), np.cos(phi_rad) * np.cos(theta_rad)
    norm_b = np.sqrt(bx**2 + bz**2)
    if norm_b > 1e-6:
        bx, bz = bx / norm_b, bz / norm_b
        arrow_z, arrow_x, len_scale = Lz_plasma * 0.15, Lx_plasma * 0.85, Lz_plasma * 0.15
        ax.quiver(arrow_z, arrow_x, bz * len_scale, bx * len_scale, color='lime', scale=1, scale_units='xy', width=0.005, pivot='tail', zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.text(arrow_z + bz * len_scale, arrow_x + bx * len_scale, r'$\mathbf{B}_0$', color='lime', fontsize=16, fontweight='bold', ha='left', va='bottom', path_effects=[pe.withStroke(linewidth=2, foreground="black")])


    # =====================================================================
    # OVERLAY PHYSICAL ANTENNA GEOMETRY (METALLIC SEPTA)
    # =====================================================================
    if antenna_grill is None:
            print(f'--- No antenna grill ---')
            Lx_wg = 0.0
        
    if antenna_grill is not None and Lx_wg > 0:
        instructions = antenna_grill.generate_mesh_instructions(z_start_position=Lz_wall)
        print(f"--- Generating Mesh Instructions while antenna grill ---")
        # 1. Draw the internal metal septa and inter-module gaps
        for inst in instructions:
            if inst['type'] in ['metal', 'metal_gap']:
                z_s = inst['z_start']
                z_width = inst['z_end'] - inst['z_start']
                
                # Rectangle((Horizontal_Start, Vertical_Start), Horizontal_Width, Vertical_Width)
                rect = patches.Rectangle((z_s, -Lx_wg), z_width, Lx_wg, 
                                         linewidth=1, edgecolor='black', 
                                         facecolor='dimgrey', hatch='///', zorder=10)
                ax.add_patch(rect)
                
        # 2. Draw the top and bottom macroscopic metal walls (Lz_wall regions)
        Lz_antenna = instructions[-1]['z_end'] - Lz_wall
        
        if Lz_wall > 0:
            rect_bot_wall = patches.Rectangle((0.0, -Lx_wg), Lz_wall, Lx_wg, 
                                              linewidth=1, edgecolor='black', 
                                              facecolor='dimgrey', hatch='///', zorder=10)
            rect_top_wall = patches.Rectangle((Lz_wall + Lz_antenna, -Lx_wg), Lz_wall, Lx_wg, 
                                              linewidth=1, edgecolor='black', 
                                              facecolor='dimgrey', hatch='///', zorder=10)
            ax.add_patch(rect_bot_wall)
            ax.add_patch(rect_top_wall)

        # 3. Dynamically adjust the Vertical (Y) axis to reveal the recessed waveguide stubs (x < 0)
        ax.set_ylim(-Lx_wg * 1.05, Lx_tot)
            
    # =========================================================
    # DIAGNOSTIC OVERLAY (CROSSHAIRS & WINDOWS)
    # =========================================================
    if show_windows_R and 'x_target_R' in diag_data_loaded:
        print(f"--- Overlaying Diagnostic Windows ---")
        # Dynamic visual thickness for the boxes (2% of the domain size)
        dz_rad = 0.02 * Lz_plasma # Width along the Toroidal (horizontal) axis
        

        # -----------------------------------------------------
        # 1. RADIAL WINDOW (Cyan) - Measuring along Vertical X-axis
        # -----------------------------------------------------
        x_target_R = diag_data_loaded['x_target_R']
        peak_z_R = diag_data_loaded['peak_z_R']
        window_size_radial = diag_data_loaded['window_size_radial']
        
        # Target Line: Now a HORIZONTAL line near the top radial boundary
        ax.axhline(y=x_target_R, color='yellow', linestyle='--', alpha=0.6, label='Radial Target Line')
        # Measure Line: Now a VERTICAL line dropping from the target
        ax.axvline(x=peak_z_R, color='green', linestyle='-', lw=1.5, label='Radial Measure Line')
        
        # Window Box: Rectangle( (z_coord, x_coord), width_z, height_x )
        rect_R = patches.Rectangle(
            (peak_z_R - dz_rad/2, x_target_R - window_size_radial), 
            dz_rad, 
            window_size_radial, 
            linewidth=2, edgecolor='green', facecolor='none', hatch='//'
        )
        ax.add_patch(rect_R)
        
        # -----------------------------------------------------
        # 2. TOROIDAL WINDOW (Magenta) - Measuring along Horizontal Z-axis
        # -----------------------------------------------------
    if show_windows_T and 'z_target_T' in diag_data_loaded: 
        dx_tor = 0.02 * Lx_plasma # Height along the Radial (vertical) axis
    if 'z_target_T' in diag_data_loaded and diag_data_loaded['z_target_T'] is not None:
        z_target_T = diag_data_loaded['z_target_T']
        peak_x_T = diag_data_loaded['peak_x_T']
        window_size_toroidal = diag_data_loaded['window_size_toroidal']
        n_para = diag_data_loaded['n_para']
        
        # Target Line: Now a VERTICAL line near the right/left toroidal boundary
        ax.axvline(x=z_target_T, color='white', linestyle='-.', alpha=0.6, label='Toroidal Target Line')
        # Measure Line: Now a HORIZONTAL line shooting backward
        ax.axhline(y=peak_x_T, color='magenta', linestyle='-', lw=1.5, label='Toroidal Measure Line')
        
        if n_para >= 0:
            z_start = z_target_T - window_size_toroidal
        else:
            z_start = z_target_T 
            
        # Window Box: Rectangle( (z_coord, x_coord), width_z, height_x )
        rect_T = patches.Rectangle(
            (z_start, peak_x_T - dx_tor/2), 
            window_size_toroidal, 
            dx_tor, 
            linewidth=2, edgecolor='magenta', facecolor='none', hatch='\\\\'
        )
        ax.add_patch(rect_T)
    else:
        print('no z_target_T')
            
    # ax.legend(loc='upper left', fontsize=12, ncol=2, framealpha=0.9)

    # ---------------------------------------------------------
    # BOUNDARIES & PLOT FORMATTING
    # ---------------------------------------------------------
    # Radial boundary is now a horizontal ceiling
    print('Radial PML boundary domain showed')
    ax.axhline(y=Lx_plasma, color='k', linestyle='--', lw=2, label='Radial PML Boundary')

    if mode == "2D":
        # Toroidal boundaries are now vertical walls
        ax.axvline(x=Lz_plasma + 2.0 * Lz_wall, color='k', linestyle='--', lw=2, label='Top Toroidal PML Boundary')
        ax.axvline(x=0.0, color='k', linestyle='--', lw=2, label='Bottom Toroidal PML Boundary')
    else:
        print('no Full 2D')
        
    # ax.set_title(f"2D Map of {component} ({mode})")
    
    # CRITICAL CHANGE: Labels match the new inverted axes
    ax.set_xlabel("Toroidal z [m]", fontsize=16)
    ax.set_ylabel("Radial x [m]", fontsize=16)
    plt.tight_layout()
    
    if figure_save_dir is not None:
        fig_path = os.path.join(figure_save_dir, f"{component}_map.pdf")
        plt.savefig(fig_path, dpi=300)
        print(f"--- Plot saved to {fig_path} ---")
        
    plt.show()

 
    # ========================================================================================

def plot_n_para_spectrum(mesh, gfu, cfg, mode, figure_save_dir, x_eval, num_points=3000, pad_factor=8):
    # Extract domain sizes
    Lz_plasma, Lz_pml = cfg.DOMAIN['Lz_plasma'], cfg.DOMAIN['Lz_pml']
    if mode == "RADIAL_ONLY":
        z_min, z_max = 0.0, Lz_plasma 
        z_coords, dz = np.linspace(z_min, z_max, num_points, endpoint=True, retstep=True)
    else:
        z_min, z_max = -Lz_pml, Lz_plasma + Lz_pml
        z_coords, dz = np.linspace(z_min, z_max, num_points, endpoint=True, retstep=True)

    # Extract Ez field
    Ez_vals = np.zeros(num_points, dtype=complex)
    Ez_field = gfu.components[0][1]
    for i, z in enumerate(z_coords):
        try: 
            mip = mesh(x_eval, z)
            Ez_vals[i] = Ez_field(mip)
        except Exception:
            Ez_vals[i] = 0.0 + 0.0j

    # Compute spatial FFT
    n_fft = num_points * pad_factor
    Ez_fft = fftshift(fft(Ez_vals, n=n_fft))
    E_fft_norm = Ez_fft / num_points
    
    # Map spatial frequency to n_para 
    fz = fftshift(fftfreq(n_fft, d=dz)) # fz = cycle per meter in the z direction
    n_para_array = (2.0 * np.pi *fz)/cfg.WAVE['k0']

    # compute power spectrum:
    power_spectrum = np.abs(Ez_fft)**2
    power_spectrum /=np.max(power_spectrum) # normalize to 1.0

    plt.figure(figsize=(10, 6))
    plt.plot(n_para_array, power_spectrum, color='crimson', lw=2)
    plt.xlim(-50, 50)
    plt.ylim(1e-4, 1.1)

    # plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
     
    # injected_n_para = np.array([cfg.WAVE['n_para'].real])  # Injected n_para value(s)
    # for n_para_value in injected_n_para:
    #     plt.axvline(x=n_para_value, color='Royalblue', linestyle=':', lw=2, label=r'$n_{//} = $'+f'{n_para_value}')
    
    plt.xlabel(r'Parallel Refractive Index [$n_\parallel$}', fontsize=16)
    plt.ylabel('Normalized Spectral Power [a.u.]', fontsize=16)
    plt.tick_params(direction='in', length=6, width=1.5, bottom=True, top=True, right=True, left=True)
    fig_path = os.path.join(figure_save_dir, f"n_para_spectrum_{mode}.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return n_para_array, power_spectrum



from scipy.special import airy

def plot_1D_radial_slice_with_theory(h5_filepath, cfg, component='Ez', z_eval=None):
    """
    Extrait une coupe 1D radiale et génère un graphique de qualité publication 
    comparant la FEM à l'équation d'Airy.
    """
    print(f"--- Extraction de la coupe 1D publication pour {component} ---")
    
    # 1. Chargement des données HDF5
    with h5py.File(h5_filepath, 'r') as h5f:
        X, Z = h5f['X'][:], h5f['Z'][:]
        E_comp = h5f[component][:]
        Lz_plasma = h5f.attrs['Lz_plasma']
        Lx_plasma = h5f.attrs['Lx_plasma']
        Lx_tot = h5f.attrs['Lx_tot']
        n_para = h5f.attrs.get('n_para', cfg['WAVE']['n_para'].real)
        
    # 2. Détermination de la coordonnée Z
    if z_eval is None:
        z_eval = Lz_plasma / 2.0
    
    idx_z = np.argmin(np.abs(Z[0, :] - z_eval))
    x_slice = X[:, idx_z]
    E_fem = E_comp[:, idx_z]
    
    # On filtre les valeurs négatives (intérieur des guides d'ondes)
    valid_idx = x_slice >= 0.0
    x_slice = x_slice[valid_idx]
    E_fem = E_fem[valid_idx]
    
    # 3. Physique : Calcul de la Coupure et du Gradient
    k0 = cfg['WAVE']['k0']
    omega_LH = cfg['WAVE']['omega_LH']
    eps0, me, qe = cfg['CONST']['eps0'], cfg['CONST']['me'], cfg['CONST']['qe']
    
    n_cutoff = (eps0 * me * omega_LH**2) / (qe**2)
    points = cfg['PLASMA']['ne_points']
    
    x_c = None
    grad_n = 0.0
    
    for i in range(len(points)-1):
        x1, n1 = points[i][0], points[i][1]
        x2, n2 = points[i+1][0], points[i+1][1]
        if min(n1, n2) <= n_cutoff <= max(n1, n2):
            grad_n = (n2 - n1) / (x2 - x1)
            x_c = x1 + (n_cutoff - n1) / grad_n
            break

    # 4. Théorie Analytique (Fonction d'Airy)
    E_theo = np.zeros_like(x_slice, dtype=complex)
    if x_c is not None and grad_n > 0:
        beta = (k0**2) * (grad_n / n_cutoff) * (n_para**2 - 1.0)
        xi = - (beta**(1/3)) * (x_slice - x_c)
        Ai, _, _, _ = airy(xi)
        
        # Alignement de la phase complexe au niveau de la coupure (x = xc)
        idx_c = np.argmin(np.abs(x_slice - x_c))
        C_match = E_fem[idx_c] / Ai[idx_c] 
        E_theo = Ai * C_match
    
    # 5. Graphique de Qualité Publication
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- Fonds Colorés (Backgrounds) ---
    ax.axvspan(0, Lx_plasma, facecolor='#f0f8ff', alpha=0.6, label='Plasma')
    ax.axvspan(Lx_plasma, Lx_tot, facecolor='#e6e6e6', alpha=0.8, label='PML (Absorption)')
    
    # --- Courbes (Normes et Partie Réelle) ---
    ax.plot(x_slice, np.abs(E_fem), color='black', lw=2.5, label='FEM — $|E_z|$')
    
    # On ne trace la théorie que dans le plasma (elle n'existe pas dans la PML)
    mask_plasma = x_slice <= Lx_plasma
    # ax.plot(x_slice[mask_plasma], np.abs(E_theo)[mask_plasma], color='red', linestyle='--', lw=2.5, label='Analytic (Airy) — $|E_z|$')
    
    ax.plot(x_slice, np.real(E_fem), color='blue', lw=1.5, label='FEM — Re($E_z$)')
    
    # --- Ligne de Coupure ---
    # if x_c is not None:
    #     ax.axvline(x=x_c, color='red', linestyle=':', lw=1.5, label='Cutoff ($n_e = n_c$)')

    # --- Esthétique ---
    ax.set_xlim(0.0, Lx_tot)
    
    
    # Limites Y dynamiques (pour ignorer les singularités métalliques de bord)
    y_max = np.percentile(np.abs(E_fem), 98) * 1.2
    ax.set_ylim(-y_max, y_max)
    
    ax.set_xlabel("Radial Position $x$ [m]", fontsize=14, fontstyle='italic')
    ax.set_ylabel("Electric Field $E_z$ [V/m]", fontsize=14, fontstyle='italic')
    
    # Ticks vers l'intérieur (Standard IEEE/APS)
    ax.tick_params(direction='in', length=6, width=1.0, bottom=True, top=True, left=True, right=True)
    
    # Légende en 2 colonnes comme sur l'image
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(os.path.dirname(h5_filepath), f"1D_radial_slice_{component}.pdf"), dpi=300)
    plt.tight_layout()
    plt.show()



    