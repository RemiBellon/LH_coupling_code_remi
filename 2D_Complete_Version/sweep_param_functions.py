import os
import copy
import h5py
import netgen.occ as occ
from ngsolve import *
import numpy as np
import datetime

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
        n_para_list = [1.5, 2., 3., 5.]
    
    # 2. Définition des balayages indépendants
    # Format -> 'Nom_du_Groupe': (Valeurs_à_tester, 'clé_PML')
    scans = {
        'Sweep_Sx_real': (np.linspace(0.5, 5.0, 3), 'Sx_r'),
        'Sweep_Sx_imag': (np.linspace(0.5, 12.0, 3), 'Sx_im'),
        'Sweep_px': (np.linspace(1.0, 5.0, 3), 'px'),
        'Sweep_Resolution_PPW': (np.linspace(10.0, 60.0, 3), 'ppw_pml'), # Contrôle maxh_pml
        'Sweep_Resolution_PPW_Medium': (np.linspace(10.0, 60.0, 3), 'ppw_medium'), # Contrôle maxh_medium
        'Sweep_Lpml_Ratio': (np.linspace(0.2, 4.5, 3), 'Lx_pml_ratio')   # L_pml / lambda_perp
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
                
                eta_sim_list, eta_pred_list = [], []
                Px_profiles, x_pml_coords = [], []
                conservation_err_list = []
                fraction_radial_list = []
                fraction_toroidal_list = []

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
                    if param_key in ['Sx_r', 'Sx_im', 'px', 'ppw_pml', 'ppw_medium']:
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

                    try:
                        solver = LHCouplingSolver_2DHcurl_1DH1(scan_cfg, geom_mode, box_medium, antenna_grill=None)
                        mesh = solver.build_mesh_with_PMLs()
                        solver.build_physics_Stix_B_field()
                        _, _, _, diag_data = solver.solve_helmholtz_2DHcurl_1DH1_with_pml(mesh, geom_mode, box_medium)
                        
                        eta_sim_list.append(diag_data['eta_sim_R'])
                        eta_pred_list.append(diag_data['eta_pred_R'])
                        
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
                        
                    except Exception as e:
                        print(f"     [!] Echec pour {param_key}={val:.3f} : {e}")
                        eta_sim_list.append(np.nan)
                        eta_pred_list.append(np.nan)
                        conservation_err_list.append(np.nan)
                        fraction_radial_list.append(np.nan)
                        fraction_toroidal_list.append(np.nan)
                
                # Sauvegarde dans le sous-groupe n_para
                subgrp.create_dataset('eta_sim', data=eta_sim_list)
                subgrp.create_dataset('eta_pred', data=eta_pred_list)
                
                # Sauvegarde des nouvelles métriques de Poynting
                subgrp.create_dataset('conservation_error', data=conservation_err_list)
                subgrp.create_dataset('fraction_radial', data=fraction_radial_list)
                subgrp.create_dataset('fraction_toroidal', data=fraction_toroidal_list)
                
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
    'Sx_r': "Partie réelle de l'étirement ($S_x^r$)",
    'Sx_im': "Facteur d'amortissement imaginaire ($S_x^{im}$)",
    'px': "Degré du polynôme d'étirement ($p_x$)",
    'ppw_pml': "Résolution de la PML (Points par longueur d'onde)",
    'ppw_medium': "Résolution dans le milieu (Points par longueur d'onde)",
    'Lx_pml_ratio': "Profondeur normalisée ($L_{pml} / \lambda_\perp$)"
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
            param_key = grp.attrs['param_name']
            
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
                
                eta_sim = subgrp['eta_sim'][:]
                eta_pred = subgrp['eta_pred'][:]
                
                # Tracé de la simulation (Ligne pleine + marqueurs)
                ax.plot(param_values, eta_sim, marker='o', linestyle='-', 
                        color=colors[idx], lw=2, markersize=6)
                
                # Tracé de la théorie (Ligne pointillée, légèrement transparente)
                # ax.plot(param_values, eta_pred, linestyle='--', 
                        # color=colors[idx], lw=2, alpha=0.7)

            # Mise en forme des axes
            ax.set_yscale('log')
            x_label = FORMAT_LABELS.get(param_key, param_key)
            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel(r"Réflectivité en amplitude ($\eta$)", fontweight='bold')
            ax.set_title(f"Impact de {param_key} sur l'absorption PML")
            
            # ---------------------------------------------------------
            # CONSTRUCTION D'UNE LÉGENDE PERSONNALISÉE ET PROPRE
            # ---------------------------------------------------------
            legend_elements = []
            
            # 1. Éléments pour le type de donnée (Sim vs Théorie)
            legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o', label=r'FEM ($\eta_{sim}$)'))
            legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'Théorie ($\eta_{pred}$)'))
            
            # Espace vide invisible pour séparer les sections de la légende
            legend_elements.append(Line2D([0], [0], color='none', label=' '))
            
            # 2. Éléments pour les couleurs des n_para
            for idx, n_key in enumerate(n_para_keys):
                n_val = float(n_key.split('_')[-1])
                legend_elements.append(Line2D([0], [0], color=colors[idx], lw=4, label=rf'$n_\parallel = {n_val}$'))
                
            # Placement de la légende à l'extérieur du graphe pour ne pas cacher les courbes
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
            
            plt.tight_layout()
            
            # Sauvegarde de l'image
            save_name = f"Analyse_Spectrale_{sweep_name}.png"
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"  -> Graphe sauvegardé : {save_name}")
            
            plt.show()



def plot_poynting_diagnostics(h5_filepath):
    """
    Reads the HDF5 database and plots the FEM Conservation Error and 
    Power Fractions alongside the Reflection Coefficient to diagnose PML stability.
    """
    print(f"--- Création des graphiques Poynting (Stabilité FEM) depuis : {h5_filepath} ---")
    
    with h5py.File(h5_filepath, 'r') as h5f:
        sweep_groups = [k for k in h5f.keys() if k.startswith('Sweep_')]
        
        for sweep_name in sweep_groups:
            grp = h5f[sweep_name]
            param_values = grp['param_values'][:]
            param_key = grp.attrs['param_name']
            
            n_para_keys = [k for k in grp.keys() if k.startswith('n_para_')]
            n_para_keys.sort(key=lambda x: float(x.split('_')[-1]))
            if not n_para_keys: continue
                
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx() # Create a secondary y-axis for the Conservation Error
            
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(n_para_keys)))
            
            for idx, n_key in enumerate(n_para_keys):
                n_para_val = float(n_key.split('_')[-1])
                subgrp = grp[n_key]
                
                eta_sim = subgrp['eta_sim'][:]
                cons_error = subgrp['conservation_error'][:] * 100.0 # Convert to percentage
                frac_radial = subgrp['fraction_radial'][:] * 100.0
                
                # Plot 1: Amplitude Reflection (Solid Line, Left Axis)
                ax1.plot(param_values, eta_sim, marker='o', linestyle='-', 
                         color=colors[idx], lw=2, markersize=6, 
                         label=rf'$\eta_{{sim}}$ ($n_\parallel = {n_para_val}$)')
                
                # Plot 2: Conservation Error (Dashed Line, Right Axis)
                ax2.plot(param_values, cons_error, marker='x', linestyle=':', 
                         color=colors[idx], lw=2, alpha=0.7)

            # Formatting Left Axis (Wave Physics)
            ax1.set_yscale('log')
            x_label = FORMAT_LABELS.get(param_key, param_key)
            ax1.set_xlabel(x_label, fontweight='bold')
            ax1.set_ylabel(r"Réflectivité en amplitude ($\eta$)", fontweight='bold', color='black')
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            
            # Formatting Right Axis (Numerical Stability)
            ax2.set_yscale('log')
            ax2.set_ylabel(r"Erreur de Conservation d'Énergie FEM (%)", fontweight='bold', color='gray')
            # Optional: Fix the right axis to show anything above 1% as a critical failure
            ax2.axhline(y=1.0, color='red', linestyle='-', alpha=0.3, label='Limite Critique (1%)')

            ax1.set_title(f"Diagnostic Thermodynamique vs {param_key}")
            
            # Combine legends from both axes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            
            # Append a dummy line for the right-axis legend context
            lines_1.append(Line2D([0], [0], color='gray', lw=2, linestyle=':', marker='x'))
            labels_1.append("Erreur Conservation")
            if lines_2:
                lines_1.append(lines_2[0])
                labels_1.append(labels_2[0])

            ax1.legend(lines_1, labels_1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                       fancybox=True, shadow=True, ncol=3)
            
            plt.tight_layout()
            save_name = f"Poynting_Diag_{sweep_name}.png"
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"  -> Graphe sauvegardé : {save_name}")
            plt.close()

