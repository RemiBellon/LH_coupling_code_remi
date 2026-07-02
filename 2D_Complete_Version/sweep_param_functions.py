import os
import copy
import h5py
import netgen.occ as occ
from ngsolve import *
import numpy as np
import datetime
import matplotlib.colors as mcolors

import config_dict as cfg    
from solver_2DHcurl_1DH1 import LHCouplingSolver_2DHcurl_1DH1

def generate_1D_radial_pml_database(base_cfg, geom_mode, box_medium, save_dir="Results"):
    """
    Exécute des balayages indépendants sur les paramètres de la PML Radiale
    pour plusieurs valeurs de n_parallel et sauvegarde dans un HDF5.
    """
    print("=========================================================")
    print("  GÉNÉRATION DATABASE : PML RADIALE 1D (JACQUOT 2013) ")
    print("=========================================================")
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.join(save_dir, f"PML_Radial_1D_Database_{timestamp}.h5")
    
    # 1. Les n_parallel à tester (Universalisme de la PML)
    if box_medium == "VACUUM":
        n_para_list = [0.]
    else: 
        # n_para_list = [1.3, 1.5, 1.7, 2.0]
        n_para_arr_negative = np.linspace(-2, -1.25, 50)
        n_para_arr_positive = np.linspace(1.25, 2., 50)
        n_para_arr_tot = np.concatenate((n_para_arr_negative, n_para_arr_positive))
        print(f"n_para_list: {n_para_arr_tot}")
        n_para_list = [2.]
    
    # 2. Définition des balayages indépendants
    # Format -> 'Nom_du_Groupe': (Valeurs_à_tester, 'clé_PML')
    scans = {
        'Sweep_Sx_real': (np.linspace(0.5, 5.0, 0), 'Sx_r'),
        'Sweep_Sx_imag': (np.linspace(0.5, 12.0, 0), 'Sx_im'),
        'Sweep_px': (np.linspace(1.0, 5.0, 0), 'px'),
        'Sweep_Resolution_PPW': (np.linspace(5.0, 60.0, 0), 'ppw_pml'), # Contrôle maxh_pml
        'Sweep_Resolution_PPW_Medium': (np.linspace(1.0, 60.0, 0), 'ppw_medium'), # Contrôle maxh_medium
        'Sweep_Lpml_Ratio': (np.linspace(.5, 4.5, 0), 'Lx_pml_ratio'),   # L_pml / lambda_perp
        'Sweep_n_para': (n_para_arr_tot, 'n_para'),
    }
    
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.attrs['Description'] = "Validation PML 1D Radiale (Onde Lente Backward)"
        
        for group_name, (param_values, param_key) in scans.items():
            print(f"\n---> Lancement du balayage : {group_name} ({param_key})")
            grp = h5f.create_group(group_name)
            grp.attrs['param_name'] = param_key
            grp.create_dataset('param_values', data=param_values)
            
            for n_para_val in n_para_list:
                print(f"  >> Test pour n_para = {n_para_val}")
                subgrp = grp.create_group(f"n_para_{n_para_val}")
                
                eta_sim_R_list, eta_pred_R_list = [], []
                eta_sim_T_list, eta_pred_T_list = [], []
                Px_profiles, x_pml_coords = [], []
                conservation_err_list = []
                fraction_radial_list = []
                fraction_toroidal_list = []
                Lx_pml_list = []
                dofs_list = []
                for val in param_values:
                    # --- FIX: SAFE DEEPCOPY ---
                    # On ne copie que les dictionnaires purs (WAVE, DOMAIN, PML, etc.)
                    # et on ignore les modules importés (math, ngsolve) ou les variables système (__name__)
                    scan_cfg = {}
                    for key, value in base_cfg.items():
                        if isinstance(value, dict) and not key.startswith('__'):
                            scan_cfg[key] = copy.deepcopy(value)
                            
                    scan_cfg['WAVE']['n_para'] = n_para_val
                    
                    # Injection du paramètre
                    if param_key in ['Sx_r', 'Sx_im', 'px', 'ppw_pml']:
                        scan_cfg['PML'][param_key] = val
                    elif param_key == "ppw_medium":
                        scan_cfg['DOMAIN']['ppw_medium'] = val
                    elif param_key == 'Lx_pml_ratio':
                        # Conversion du ratio en vraie distance physique L_pml
                        temp_solver = LHCouplingSolver_2DHcurl_1DH1(scan_cfg, geom_mode, box_medium, antenna_grill=None)
                        _, n_perp_p, _ = temp_solver.compute_physics_parameters()
                        lambda_perp = scan_cfg['WAVE']['lambda0'] / np.abs(n_perp_p.real)
                        scan_cfg['DOMAIN']['Lx_pml'] = val * lambda_perp
                        scan_cfg['DOMAIN']['Lx_tot'] = scan_cfg['DOMAIN']['Lx_plasma'] + scan_cfg['DOMAIN']['Lx_pml']
                        del temp_solver
                    elif param_key == 'n_para':
                        print(f"     [!] Changement de n_para à {val:.3f}")
                        scan_cfg['WAVE']['n_para'] = val
                    try:
                        solver = LHCouplingSolver_2DHcurl_1DH1(scan_cfg, geom_mode, box_medium, antenna_grill=None)
                        mesh = solver.build_mesh_with_PMLs()
                        solver.build_physics_Stix_B_field()
                        _, _, _, diag_data = solver.solve_helmholtz_2DHcurl_1DH1_with_pml(mesh, geom_mode, box_medium)
                        
                        dofs_list.append(solver.fes.ndof)
                        eta_sim_R_list.append(diag_data['eta_sim_R'])
                        eta_pred_R_list.append(diag_data['eta_pred_R'])
                        
                        Px_profiles.append(diag_data['Px_pml_profile_norm'])
                        depth_norm = (diag_data['x_pml_sweep'] - scan_cfg['DOMAIN']['Lx_plasma']) / diag_data['lambda_perp_real']
                        x_pml_coords.append(depth_norm)
                        
                        # --- POYNTING FLUX EXTRACTION ---
                        P_in = diag_data['P_in_net']
                        P_out_R = diag_data['P_out_Radial']
                        P_out_T = diag_data['P_out_Toroidal']
                        
                        # Protect against division by zero numerically
                        safe_P_in = max(abs(P_in), 1e-15)
                        
                        conservation_err_list.append(diag_data['power_error_plasma'])
                        fraction_radial_list.append(P_out_R / safe_P_in)
                        fraction_toroidal_list.append(P_out_T / safe_P_in)
                        Lx_pml_list.append(scan_cfg['DOMAIN']['Lx_pml'])
                        print(f'===== Lx_pml = {scan_cfg["DOMAIN"]["Lx_pml"]:.4f} m =====')
                    except Exception as e:
                        print(f"     [!] Echec pour {param_key}={val:.3f} : {e}")
                        dofs_list.append(np.nan)
                        eta_sim_R_list.append(np.nan)
                        eta_pred_R_list.append(np.nan)
                        eta_sim_T_list.append(np.nan)
                        eta_pred_T_list.append(np.nan)
                        conservation_err_list.append(np.nan)
                        fraction_radial_list.append(np.nan)
                        fraction_toroidal_list.append(np.nan)
                
                # Sauvegarde dans le sous-groupe n_para
                subgrp.create_dataset('dofs', data=dofs_list)
                subgrp.create_dataset('eta_sim_R', data=eta_sim_R_list)
                subgrp.create_dataset('eta_pred_R', data=eta_pred_R_list)
                subgrp.create_dataset('eta_sim_T', data=eta_sim_T_list)
                subgrp.create_dataset('eta_pred_T', data=eta_pred_T_list)
                
                # Sauvegarde des nouvelles métriques de Poynting
                subgrp.create_dataset('conservation_error', data=conservation_err_list)
                subgrp.create_dataset('fraction_radial', data=fraction_radial_list)
                subgrp.create_dataset('fraction_toroidal', data=fraction_toroidal_list)
                subgrp.create_dataset('Lx_pml', data=Lx_pml_list)
                if len(Px_profiles) > 0:
                    subgrp.create_dataset('Px_profiles', data=np.vstack(Px_profiles))
                    subgrp.create_dataset('x_normalized_depths', data=np.vstack(x_pml_coords))
                    
    print(f"\n[SUCCÈS] Base de données HDF5 générée : {h5_filename}")
    return h5_filename

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.figsize': (9, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# Dictionnaire pour avoir de beaux labels sur les axes X
FORMAT_LABELS = {
    'Sx_r': "$S_x^r$",
    'Sx_im': "$S_x^{im}$",
    'px': "$p_x$",
    'ppw_pml': "PPW in pml",
    'ppw_medium': "PPW in plasma",
    'Lx_pml_ratio': "$L_{pml} / \lambda_\perp$",
    'n_para': "Parallel refactive index $n_\parallel$"
}

def plot_sweeps_all_npara(h5_filepath):
    """
    Parcourt tous les balayages d'un fichier HDF5 et trace eta_sim vs paramètre.
    Superpose les courbes pour tous les n_parallel sur le même graphique.
    """
    print(f"--- Création des graphiques multi-spectres depuis : {h5_filepath} ---")
    
    with h5py.File(h5_filepath, 'r') as h5f:
        
        # Trouver tous les groupes de balayage (Sweep_...)
        sweep_groups = [k for k in h5f.keys() if k.startswith('Sweep_')]
        
        for sweep_name in sweep_groups:
            grp = h5f[sweep_name]
            param_values = grp['param_values'][:]
            print(f'param_values: {param_values}')
            param_key = grp.attrs['param_name'] # Lx_pml_ratio, Sx^r, Sx_im, px
            print(f' ==== key: {param_key} ====')
            # Identifier et trier les sous-groupes n_para
            n_para_keys = [k for k in grp.keys() if k.startswith('n_para_')]
            n_para_keys.sort(key=lambda x: float(x.split('_')[-1]))
            
            # Si le balayage a échoué ou est vide, on passe
            if not n_para_keys:
                continue
                
            fig, ax = plt.subplots()
            
            # Palette de couleurs pour différencier les n_para
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_para_keys)))
            
            for idx, n_key in enumerate(n_para_keys):
                n_para_val = float(n_key.split('_')[-1])
                subgrp = grp[n_key]
                try:
                    dofs = subgrp['dofs'][:]
                    print(f'dofs:', dofs)
                except: 
                    pass
                
                eta_sim = subgrp['eta_sim'][:]
                if len(eta_sim) > 0:
                    ax.set_yscale('log')
                eta_pred = subgrp['eta_pred'][:]
                Lx_pml = subgrp['Lx_pml'][:] if 'Lx_pml' in subgrp else None
                print(f'Gamma_R:', eta_sim)
                print(f'Lx_pml:', Lx_pml)
                # Tracé de la simulation (Ligne pleine + marqueurs)
                ax.plot(param_values[:], eta_sim[:], marker='', linestyle='-', 
                        color=colors[idx], linewidth=2, markersize=5, alpha=1)
                ax.plot(param_values[:], eta_sim[:], marker='o', linestyle='', 
                        color=colors[idx], markersize=5, alpha=1)
                parameter_is_ppw, parameter_is_Lx_pml_ratio = None, None
                try:
                    if param_key == "Lx_pml_ratio" and Lx_pml is not None:
                        color_eta_pred = 'green'
                        # ax2.set_ylabel(r"Predicted Reflection Coefficient $\Gamma_R^{pred}$", color=color_eta_pred, fontsize=14)
                        # ax2.yaxis.set_label_position('right')
                        # ax2.spines['right'].set_color(color_eta_pred)
                        # ax2.set_yscale('log')
                        # ax2.yaxis.set_tick_params(which='both', direction='in', length=6, width=1.5, right=True, 
                        #                         left=False, labelleft=False, labelright=True, color=color_eta_pred, labelcolor = color_eta_pred)
                        parameter_is_Lx_pml_ratio = 'Lx_pml_ratio'
                        ax.plot(param_values[:],eta_pred[:], marker='', linestyle='--', color=color_eta_pred, alpha=0.5)
                        ax.plot(param_values[:],eta_pred[:], marker='s', linestyle='', color=color_eta_pred, alpha=1)
                    if param_key == 'Sx_r':
                        color_Sx_r, color_Sx_im = "crimson", "mediumblue"
                        ax.plot(param_values[:],eta_pred[:], marker='', linestyle='--', color=color_eta_pred, alpha=0.5)
                        ax.plot(param_values[:],eta_pred[:], marker='s', linestyle='', color=color_eta_pred, alpha=1)

                    if param_key == 'ppw_medium' and dofs is not None:
                        ax2, color_ppw = ax.twinx(), 'mediumblue'
                        ax2.set_ylabel(r"Degrees of Freedom DoFs", color=color_ppw, fontsize=14)
                        ax2.yaxis.set_label_position('right')
                        ax2.spines['right'].set_color(color_ppw)
                        # if len(param_values) > 0 and len(dofs) > 0:
                        #     ax2.set_yscale('log')
                        ax2.yaxis.set_tick_params(which='both', direction='in', length=6, width=1.5, right=True, 
                                                left=False, labelleft=False, labelright=True, color=color_ppw, labelcolor = color_ppw)
                        parameter_is_ppw = "ppw_medium"
                        ax2.set_yscale('log')
                        ax2.plot(param_values, dofs, marker='s', linestyle='--', 
                            color=color_ppw, lw=2, markersize=6, alpha=0.4)
                        ax2.plot(param_values, dofs, marker='s', linestyle='', 
                            color=color_ppw, lw=2, markersize=6, alpha=0.4)
                    
                    elif param_key == 'ppw_pml' and dofs is not None:
                        ax2, color_ppw = ax.twinx(), 'firebrick'
                        ax2.set_ylabel(r"Degrees of Freedom DoFs", color=color_ppw, fontsize=14)
                        ax2.yaxis.set_label_position('right')
                        ax2.spines['right'].set_color(color_ppw)
                        if len(param_values) > 0 and len(dofs) > 0:
                            ax2.set_yscale('log')
                        ax2.yaxis.set_tick_params(which="both", direction='in', length=6, width=1.5, right=True,
                                                   left=False, labelleft=False, labelright=True, color=color_ppw, labelcolor = color_ppw)
                        parameter_is_ppw = "ppw_pml"
                        ax2.plot(param_values, dofs, marker='s', linestyle='--', 
                            color=color_ppw, lw=2, markersize=6, alpha=0.4)
                        ax2.plot(param_values, dofs, marker='s', linestyle='', 
                            color=color_ppw, lw=2, markersize=6, alpha=0.4)
                except:
                    pass   
                # if Lx_pml is not None:
                #     ax.plot(param_values, Lx_pml, marker='s', linestyle=':', 
                #             color=colors[idx], lw=2, markersize=6)
                # # Tracé de la théorie (Ligne pointillée, légèrement transparente)
                # ax.plot(param_values, eta_pred, linestyle='--', 
                #         color=colors[idx], lw=2, alpha=0.7)
            ax.spines['left'].set_color(colors[idx])
            # Mise en forme des axes
            ax.yaxis.set_tick_params(which="both", direction='in', length=6, width=1., right=False, 
                                     left=True, labelleft=True, labelright=False)
            x_label = FORMAT_LABELS.get(param_key, param_key)
            ax.set_xlabel(x_label, fontsize=14)
            # ax.set_ylim(2e-09, 2e-01)



            ax.set_ylabel(r"Reflection coefficient $\Gamma_R$", fontsize=14)
            # ax.set_title(f"Impact de {param_key} sur l'absorption PML")
            
            # ---------------------------------------------------------
            # CONSTRUCTION D'UNE LÉGENDE PERSONNALISÉE ET PROPRE
            # ---------------------------------------------------------
            legend_elements = []
            
            # 1. Éléments pour le type de donnée (Sim vs Théorie)
            # legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o', label=r'FEM ($\eta_{sim}$)'))
            # legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'Théorie ($\eta_{pred}$)'))e
            # legend_elements.append(Line2D([0], [0], color='none', label=' '))
            
            # 2. Éléments pour les couleurs des n_para
            for idx, n_key in enumerate(n_para_keys):
                n_val = float(n_key.split('_')[-1])
                legend_elements.append(Line2D([0], [0], color=colors[idx], lw=4, label=rf'$\Gamma_R^{{sim}}$'))# (n_\parallel = {n_val})$'))
                print(f'parameter_is_ppw: {parameter_is_ppw}')
                try:
                    if parameter_is_Lx_pml_ratio in ['Lx_pml_ratio']:
                        legend_elements.append(Line2D([0], [0], color='green', lw=4, label=r'$\Gamma_R^{{pred}}$'))
                    if parameter_is_ppw in ['ppw_medium', 'ppw_pml']:
                        if parameter_is_ppw == 'ppw_medium':
                            color_line="mediumblue"
                        else: 
                            color_line="firebrick"
                        legend_elements.append(Line2D([0], [0], color=color_line, lw=4, label=r'$DoFs$'))
                except:
                    pass
            # Placement de la légende à l'extérieur du graphe pour ne pas cacher les courbes
            ax.legend(handles=legend_elements, loc='upper center', framealpha=0.95, ncol=2)
            

            
            plt.tight_layout()
            
            # Sauvegarde de l'image
            save_name = f"Analyse_Spectrale_{sweep_name}.svg"
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"  -> Graphe sauvegardé : {save_name}")
            
            plt.show()


import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# Formatting Dictionary
# ---------------------------------------------------------

def plot_aggregated_thermodynamics(h5_filepaths, target_params=None):
    """
    Universally extracts and concatenates Poynting vector data from multiple HDF5 files.
    Generates two strictly rigorous diagnostic plots per parameter:
      1. Fractional Power Flow (Proves 1D Assumption and Energy Delivery)
      2. Conservation Error (Proves FEM Mesh Quality)
    """
    print(f"--- Poynting Diagnostics: Processing {len(h5_filepaths)} Datasets ---")
    
    aggregate_data = {}
    param_keys_map = {}
    
    # =========================================================================
    # 1. Universal Data Extraction
    # =========================================================================
    for filepath in h5_filepaths:
        try:
            with h5py.File(filepath, 'r') as h5f:
                sweep_groups = [k for k in h5f.keys() if k.startswith('Sweep_')]
                
                for sweep_name in sweep_groups:
                    grp = h5f[sweep_name]
                    param_key = grp.attrs.get('param_name', '')
                    
                    if target_params is not None and param_key not in target_params:
                        continue
                        
                    if sweep_name not in aggregate_data:
                        aggregate_data[sweep_name] = {}
                        param_keys_map[sweep_name] = param_key
                        
                    n_para_keys = [k for k in grp.keys() if k.startswith('n_para_')]
                    
                    for n_key in n_para_keys:
                        subgrp = grp[n_key]
                        
                        # We only process if the Poynting metrics actually exist in this dataset
                        if 'conservation_error' not in subgrp:
                            continue
                            
                        if n_key not in aggregate_data[sweep_name]:
                            aggregate_data[sweep_name][n_key] = {
                                'param_values': [], 'err': [], 'frac_rad': [], 'frac_tor': []
                            }
                            
                        aggregate_data[sweep_name][n_key]['param_values'].extend(grp['param_values'][:])
                        aggregate_data[sweep_name][n_key]['err'].extend(subgrp['conservation_error'][:])
                        aggregate_data[sweep_name][n_key]['frac_rad'].extend(subgrp['fraction_radial'][:])
                        aggregate_data[sweep_name][n_key]['frac_tor'].extend(subgrp['fraction_toroidal'][:])
                        
        except Exception as e:
            print(f"[!] Error reading {filepath}: {e}")
            continue

    # =========================================================================
    # 2. Rigorous Deduplication, Sorting, and Plotting
    # =========================================================================
    for sweep_name, n_para_dict in aggregate_data.items():
        if not n_para_dict: # Skip if no Poynting data was found for this sweep
            continue
            
        param_key = param_keys_map[sweep_name]
        sorted_n_paras = sorted(n_para_dict.keys(), key=lambda x: float(x.split('_')[-1]))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sorted_n_paras)))
        
        # Initialize the two figures
        fig_frac, ax_frac = plt.subplots(figsize=(9, 6))
        fig_err, ax_err = plt.subplots(figsize=(9, 6))
        
        for idx, n_key in enumerate(sorted_n_paras):
            data = n_para_dict[n_key]
            n_para_val = float(n_key.split('_')[-1])
            
            # --- Dictionary Deduplication (The "Freshest Run Wins" Rule) ---
            unique_data = {}
            for p_val, err_v, rad_v, tor_v in zip(data['param_values'], data['err'], data['frac_rad'], data['frac_tor']):
                unique_data[p_val] = (err_v, rad_v, tor_v)
            
            p_vals_sorted = np.sort(list(unique_data.keys()))
            err_sorted = np.array([unique_data[x][0] for x in p_vals_sorted]) * 100.0  # Convert to %
            rad_sorted = np.array([unique_data[x][1] for x in p_vals_sorted])
            tor_sorted = np.array([unique_data[x][2] for x in p_vals_sorted])
            
            # Filter out NaNs (Failed Simulations)
            valid_mask = ~np.isnan(err_sorted)
            p_vals_clean = p_vals_sorted[valid_mask]
            err_clean = err_sorted[valid_mask]
            rad_clean = rad_sorted[valid_mask]
            tor_clean = tor_sorted[valid_mask]
            
            label_n = rf'($n_\parallel = {n_para_val}$)' if param_key != 'n_para' else ''
            
            # --- Plot A: Power Fractions ---
            ax_frac.plot(p_vals_clean, rad_clean, marker='o', linestyle='-', color=colors[idx], lw=2, label=rf'$P_{{rad}}/P_{{in}}$ {label_n}')
            ax_frac.plot(p_vals_clean, tor_clean, marker='x', linestyle=':', color=colors[idx], lw=2, alpha=0.7)

            # --- Plot B: Conservation Error ---
            if n_para_val == 0.0:
                ax_err.plot(p_vals_clean, err_clean, marker='D', linestyle='-', color=colors[idx], lw=2, label=rf'Vacuum: {label_n}')
            else:
                ax_err.plot(p_vals_clean, err_clean, marker='D', linestyle='-', color=colors[idx], lw=2, label=rf'{label_n}')

        # -----------------------------------------------------
        # Formatting Figure A: Fractions (Linear)
        # -----------------------------------------------------
        x_label = FORMAT_LABELS.get(param_key, param_key)
        
        ax_frac.set_xlabel(x_label, fontsize=14)
        ax_frac.set_ylabel("Fraction de la Puissance Injectée", fontsize=14)
        ax_frac.set_title(f"Preuve Dimensionnelle : Flux de Puissance ({param_key})")
        ax_frac.set_yscale('linear')
        ax_frac.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Idéal (100% Radial)')
        ax_frac.axhline(0.0, color='gray', linestyle='--', alpha=0.5, label='Idéal (0% Toroïdal)')
        ax_frac.grid(True, linestyle='--', alpha=0.6)
        
        # Legend construction for Fractions
        lines_f, labels_f = ax_frac.get_legend_handles_labels()
        lines_f.insert(0, Line2D([0], [0], color='gray', lw=2, linestyle=':', marker='x'))
        labels_f.insert(0, r'$P_{tor}/P_{in}$ (Fuite)')
        ax_frac.legend(lines_f, labels_f, loc='best', framealpha=0.95)
        
        fig_frac.tight_layout()
        save_name_frac = f"Thermodynamic_Fractions_{param_key}.png"
        fig_frac.savefig(save_name_frac, dpi=300, bbox_inches='tight')
        print(f"  -> Graphe (Fractions) sauvegardé : {save_name_frac}")

        # -----------------------------------------------------
        # Formatting Figure B: Error (Logarithmic)
        # -----------------------------------------------------
        ax_err.set_yscale('log')
        ax_err.set_xlabel(x_label, fontsize=14)
        ax_err.set_ylabel(r"$(P_{in} - P_{out})/(P_{in})$ (%)", fontsize=14)
        # ax_err.set_title(f"Santé Thermodynamique FEM ({param_key})")
        
        # Absolute Critical Threshold for FEM Physics (1%)
        # ax_err.axhline(1.0, color='red', linestyle='-', lw=2, alpha=0.5, label='Limite Critique (1%)')
        ax_err.grid(True, which="both", linestyle='--', alpha=0.6)
        
        ax_err.legend(loc='best', framealpha=0.95, ncol=2, fontsize=12)
        
        fig_err.tight_layout()
        save_name_err = f"1D_power_conservation_{param_key}.svg"
        fig_err.savefig(save_name_err, dpi=300, bbox_inches='tight')
        print(f"  -> Graphe (Erreur) sauvegardé : {save_name_err}")
        
        plt.close(fig_frac)
        plt.close(fig_err)


import h5py
def plot_concatenated_general_sweeps(h5_filepaths, target_params=None):
    """
    Ingests multiple HDF5 databases and generically concatenates data for ANY specified parameter.
    Applies rigorous dictionary deduplication and monotonic sorting.
    
    Parameters:
    h5_filepaths (list of str): Paths to the HDF5 files.
    target_params (list of str): List of 'param_name' to target (e.g., ['n_para', 'ppw_pml']). 
                                 If None, it plots ALL sweeps found in the files.
    """
    print(f"--- Universal Aggregation: Processing {len(h5_filepaths)} Datasets ---")
    
    aggregate_data = {}
    param_keys_map = {}
    
    # 1. Universal Data Extraction
    for filepath in h5_filepaths:
        try:
            with h5py.File(filepath, 'r') as h5f:
                # Find all sweep groups
                sweep_groups = [k for k in h5f.keys() if k.startswith('Sweep_')]
                
                for sweep_name in sweep_groups:
                    grp = h5f[sweep_name]
                    param_key = grp.attrs.get('param_name', '')
                    
                    # Filter based on user request
                    if target_params is not None and param_key not in target_params:
                        continue
                        
                    if sweep_name not in aggregate_data:
                        aggregate_data[sweep_name] = {}
                        param_keys_map[sweep_name] = param_key
                        
                    n_para_keys = [k for k in grp.keys() if k.startswith('n_para_')]
                    
                    for n_key in n_para_keys:
                        subgrp = grp[n_key]
                        
                        if n_key not in aggregate_data[sweep_name]:
                            aggregate_data[sweep_name][n_key] = {
                                'param_values': [], 'eta_sim': [], 'dofs': []
                            }
                            
                        # Safely extract data (Handles old datasets without DoFs)
                        p_vals = grp['param_values'][:]
                        eta = subgrp['eta_sim'][:]
                        dofs = subgrp['dofs'][:] if 'dofs' in subgrp else np.full_like(eta, np.nan)
                        
                        aggregate_data[sweep_name][n_key]['param_values'].extend(p_vals)
                        aggregate_data[sweep_name][n_key]['eta_sim'].extend(eta)
                        aggregate_data[sweep_name][n_key]['dofs'].extend(dofs)
                        
        except Exception as e:
            print(f"[!] Error reading {filepath}: {e}")
            continue

    # 2. Rigorous Deduplication, Sorting, and Plotting
    for sweep_name, n_para_dict in aggregate_data.items():
        param_key = param_keys_map[sweep_name]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Only create secondary axis if DoFs data actually exists
        has_dofs = any(not np.all(np.isnan(d['dofs'])) for d in n_para_dict.values())
        print(f'has_dofs for {sweep_name}: {has_dofs}')
        if has_dofs:
            ax2 = ax1.twinx() 
            
        sorted_n_paras = sorted(n_para_dict.keys(), key=lambda x: float(x.split('_')[-1]))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_n_paras)))
        
        for idx, n_key in enumerate(sorted_n_paras):
            data = n_para_dict[n_key]
            n_para_val = float(n_key.split('_')[-1])
            
            # --- Dictionary Deduplication ---
            unique_data = {}
            for x_val, y_val, d_val in zip(data['param_values'], data['eta_sim'], data['dofs']):
                unique_data[x_val] = (y_val, d_val)
            
            # Sort monotonically by X-axis
            p_vals_sorted = np.sort(list(unique_data.keys()))
            eta_sorted = np.array([unique_data[x][0] for x in p_vals_sorted])
            dofs_sorted = np.array([unique_data[x][1] for x in p_vals_sorted])
            
            # Filter NaNs
            valid_mask = ~np.isnan(eta_sorted)
            p_vals_clean = p_vals_sorted[valid_mask]
            eta_clean = eta_sorted[valid_mask]
            dofs_clean = dofs_sorted[valid_mask]
            
            # Plotting
            label_n = rf'($n_\parallel = {n_para_val}$)' if param_key != 'n_para' else ''
            
            ax1.plot(p_vals_clean, eta_clean, marker='o', linestyle='-', 
                     color=colors[idx], lw=2, markersize=5, 
                     label=rf'$\Gamma_R^{{sim}}$ {label_n}')
            
            ax1.spines['left'].set_color(colors[idx])
            ax1.yaxis.set_tick_params(which='both', direction='in', length=6, width=1.5, right=False, 
                                                left=True, labelleft=True, labelright=False, color=colors[idx], labelcolor = colors[idx])
            if has_dofs and param_key == 'ppw_pml':
                # ax1.set_ylim(9e-04, 7e-03)
                ax2.spines['right'].set_color("firebrick")
                ax2.yaxis.set_tick_params(which='both', direction='in', length=6, width=1.5, right=True, 
                                                left=False, labelleft=False, labelright=True, color="firebrick", labelcolor = "firebrick")
                ax2.plot(p_vals_clean, dofs_clean, marker='s', linestyle='--', 
                         color="firebrick", lw=2, alpha=0.4)
                
            if has_dofs and param_key == 'ppw_medium':
                    ax2.spines['right'].set_color("mediumblue")
                    ax2.yaxis.set_tick_params(which='both', direction='in', length=6, width=1.5, right=True, 
                                                left=False, labelleft=False, labelright=True, color="mediumblue", labelcolor = "mediumblue")
                    ax2.plot(p_vals_clean, dofs_clean, marker='s', linestyle='--', 
                         color="mediumblue", lw=2, alpha=0.4)
                

        # 3. Formatting
        
        ax1.set_yscale('log')
        x_label = FORMAT_LABELS.get(param_key, param_key)
        ax1.set_xlabel(x_label, fontsize=14)
        
        ax1.set_ylabel(r"Reflection coefficient $\Gamma_R^{sim}$", color='black', fontsize=14)
        ax1.grid(True, ls="--", alpha=0.5)

        if has_dofs:
            ax2.set_ylabel("Computational Cost DoFs", color="mediumblue")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax2.set_yscale('log')
        # ax1.set_title(f"Agrégation Universelle : {param_key}")
        
        # Legend Management
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        if has_dofs:
            lines_1.append(Line2D([0], [0], color="mediumblue", lw=2, linestyle='--', marker='s'))
            labels_1.append("DoFs")
            
        ax1.legend(lines_1, labels_1, loc='upper center', framealpha=0.95,
                   fancybox=True, shadow=False, ncol=3)
        
        plt.tight_layout()
        save_name = f"Aggregated_{param_key}.svg"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"  -> Graphe agrégé sauvegardé : {save_name}")
        plt.close()



        import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.figsize': (9, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def extract_sweep_data(h5f, sweep_name):
    """
    Helper function to safely extract param_values, eta_sim, and eta_pred 
    from a specific sweep group in the HDF5 file.
    """
    if sweep_name not in h5f:
        print(f"[!] Warning: {sweep_name} not found in the dataset.")
        return None, None, None
        
    grp = h5f[sweep_name]
    param_values = grp['param_values'][:]
    
    # Dynamically find the n_para subgroup (in vacuum, this is usually n_para_0.0)
    n_para_keys = [k for k in grp.keys() if k.startswith('n_para_')]
    if not n_para_keys:
        return None, None, None
        
    subgrp = grp[n_para_keys[0]]
    eta_sim = subgrp['eta_sim'][:]
    eta_pred = subgrp['eta_pred'][:]
    
    return param_values, eta_sim, eta_pred

def plot_vacuum_pml_validation(h5_filepath):
    """
    Reads a 1D Vacuum HDF5 dataset and generates two specific comparison graphs:
    1. Gamma_R (Sim vs Pred) for Sx_r and Sx_im on the same axes.
    2. Gamma_R (Sim vs Pred) for px.
    """
    print(f"--- Generating 1D Vacuum Validation Plots from: {h5_filepath} ---")
    
    try:
        with h5py.File(h5_filepath, 'r') as h5f:
            
            # Extract data for Graph 1 (Sx_r and Sx_im)
            sx_r_vals, sim_r, pred_r = extract_sweep_data(h5f, 'Sweep_Sx_real')
            sx_i_vals, sim_i, pred_i = extract_sweep_data(h5f, 'Sweep_Sx_imag')
            
            # Extract data for Graph 2 (px)
            px_vals, sim_px, pred_px = extract_sweep_data(h5f, 'Sweep_px')
            
    except Exception as e:
        print(f"[!] Failed to read the HDF5 file: {e}")
        return

    # =========================================================================
    # GRAPH 1: S_x^r and S_x^{im} Comparison
    # =========================================================================
    if sx_r_vals is not None and sx_i_vals is not None:
        fig1, ax1 = plt.subplots()
        
        # Plot Sx_r (Real part) in Blue
        ax1.plot(sx_r_vals[2:-1], sim_r[2:-1], marker='o', linestyle='', color='royalblue', lw=2, label=r'Simulation: $S_x^r$')
        ax1.plot(sx_r_vals[2:-1], sim_r[2:-1], marker='', linestyle='--', color='royalblue', lw=2, alpha=0.5)
        # Plot Sx_im (Imaginary part) in Red
        ax1.plot(sx_i_vals[2:20], sim_i[2:20], marker='s', linestyle='', color='crimson', lw=2, label=r'Simulation: $S_x^{im}$')
        ax1.plot(sx_i_vals[2:20], sim_i[2:20], marker='', linestyle='-', color='crimson', lw=2, alpha=0.5)

        ax1.set_xlim(0.5, 5)
        ax1.set_ylim(1.8e-04, 1.95e-04)
        ax1.set_yscale('log')
        ax1.set_xlabel("Stretching Parameter $S_x$", fontsize=14)
        ax1.set_ylabel(r"Reflection Coefficient $\Gamma_R^{{sim}}$", fontsize=14)
        #ax1.set_title("Validation Théorique PML : Partie Réelle vs Imaginaire")
        
        # Move legend outside to prevent overlapping with the data curves
        ax1.legend(loc='upper left', framealpha=0.95)
        plt.tick_params(which="both", direction='in', length=6, width=1.5, right=True,
                                                   left=True, bottom=True, top=True)              
        plt.tight_layout()
        save_name1 = "1D_VACUUM_Gamma_R_sim_vs_Sx_r_n_Sx_im.svg"
        plt.savefig(save_name1, dpi=300, bbox_inches='tight')
        print(f"  -> Graphe 1 sauvegardé : {save_name1}")
        plt.show()

    # =========================================================================
    # GRAPH 2: Polynomial Degree (p_x)
    # =========================================================================
    if px_vals is not None:
        fig2, ax2 = plt.subplots()
        
        # Plot px in Green
        ax2.plot(px_vals, sim_px, marker='D', linestyle='', color='crimson', lw=2, label=r'Simulation: $p_x$')
        ax2.plot(px_vals, sim_px, marker='', linestyle='-', color='crimson', lw=2, alpha=0.5)

        ax2.plot(px_vals, pred_px, marker='', linestyle='--', color='forestgreen', lw=2, alpha=0.7, label=r'Theoretical: $p_x$')
        
        ax2.set_yscale('log')
        ax2.set_xlabel(r"Stretching Degree $p_x$", fontsize=14)
        ax2.set_ylabel(r"Reflection Coefficient $\Gamma_R^{{sim}}$",fontsize=14)
        #ax2.set_title("Validation Théorique PML : Influence du profil polynomial")
        
        ax2.legend(loc='best', framealpha=0.95)
        plt.tick_params(which="both", direction='in', length=6, width=1.5, right=True,
                                                   left=True, bottom=True, top=True)
        plt.tight_layout()
        save_name2 = "1D_VACUUM_Gamma_R_sim_vs_px.svg"
        plt.savefig(save_name2, dpi=300, bbox_inches='tight')
        print(f"  -> Graphe 2 sauvegardé : {save_name2}")
        plt.show()