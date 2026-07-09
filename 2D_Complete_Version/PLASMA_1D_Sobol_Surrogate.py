import os
import gc
import copy
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd
from SALib.sample import sobol
# =============================================================================
#            PHASE 1: Set the dataset and prepare dataset generation
# =============================================================================
def generate_sobol_doe(N_base, save_csv=True):
    """
    Generates a Sobol-sampled Design of Experiments (DoE) for the PML surrogate model.
    
    Parameters:
    N_base (int): The base number of samples. Must be a power of 2 for optimal Sobol 
                  properties (e.g., 128, 256, 512). The total number of simulations 
                  will be N_base * (2D + 2), where D is the number of parameters.
    """
    print("--- Generating Sobol DoE for PML Surrogate Model ---")
    
    # 1. Define the Parameter Space (D = 6)
    # For logarithmic parameters, we define the bounds in Log10 space.
    log_ne_min, log_ne_max = np.log10(3e17), np.log10(3e18)
    log_npara_min, log_npara_max = np.log10(2.0), np.log10(3.0)
    
    problem = {
        'num_vars': 6,
        'names': ['Sx_r', 'Sx_im', 'px', 'Lx_pml_ratio', 'log10_ne_edge', 'log10_n_para'],
        'bounds': [
            [0.5, 5.0],                  # Sx_r
            [0.5, 5.0],                  # Sx_im
            [0.5, 5.0],                  # px
            [0.2, 5.0],                  # Lx_pml_ratio
            [log_ne_min, log_ne_max],    # log10(ne_edge)
            [log_npara_min, log_npara_max] # log10(n_para)
        ]
    }
    
    # 2. Generate the Sobol Sequence
    # calc_second_order=True is required to compute Total-Order indices later.
    # Total samples = N_base * (2 * D + 2) = 256 * (12 + 2) = 3584 simulations
    sobol_samples = sobol.sample(problem, N_base, calc_second_order=True)
    print(f"[+] Sobol sequence generated. Total unique simulations required: {sobol_samples.shape[0]}")
    
    # 3. Transform to a Pandas DataFrame for easy manipulation
    df_doe = pd.DataFrame(sobol_samples, columns=problem['names'])
    
    # 4. Map the logarithmic variables back to physical linear space for NGSolve
    df_doe['ne_edge_linear'] = 10 ** df_doe['log10_ne_edge']
    df_doe['n_para_linear'] = 10 ** df_doe['log10_n_para']
    
    # Reorder columns to group the physical inputs together at the end
    final_columns = ['Sx_r', 'Sx_im', 'px', 'Lx_pml_ratio', 'log10_ne_edge', 'ne_edge_linear', 'log10_n_para', 'n_para_linear']
    df_doe_physical = df_doe[final_columns].copy()
    
    # Add empty columns for the NGSolve outputs we will collect in Phase 2
    df_doe_physical['Gamma_R'] = np.nan
    df_doe_physical['DoFs'] = np.nan
    df_doe_physical['Conservation_Error'] = np.nan
    
    if save_csv:
        filename = "DoE_Sobol_PML_6D.csv"
        df_doe_physical.to_csv(filename, index_label="Sim_ID", sep=';', decimal=',')
        print(f"[+] DoE saved successfully to '{filename}'")
        
    return df_doe_physical, problem


import os
import gc
import copy
import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# HPC / VM SAFETY: Set environment variables at the absolute top level
# =============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NGS_NUM_THREADS'] = '1'

def worker_simulation(task):
    """
    Isolated worker function. 
    LAZY LOADING: We import NGSolve INSIDE the worker to completely isolate 
    its asyncio event loops from the main orchestrator process.
    """
    idx = task['Sim_ID']
    row = task['row_data']
    
    result = {
        'Sim_ID': idx,
        'Gamma_R': np.nan,
        'DoFs': np.nan,
        'Conservation_Error': np.nan,
        'Success': False,
        'Error_Msg': ""
    }
    
    # try:
    # --- LOCAL IMPORTS ---
    from ngsolve import TaskManager
    import config_dict as cfg
    from solver_2DHcurl_1DH1 import LHCouplingSolver_2DHcurl_1DH1
    import asyncio
    
    # Patch local asyncio loop for this specific worker to prevent Netgen crashes
    asyncio.set_event_loop(asyncio.new_event_loop())

    # 1. Clean Deepcopy of the configuration dictionary
    scan_cfg = {}
    for key, value in cfg.__dict__.items():
        if isinstance(value, dict) and not key.startswith('__'):
            scan_cfg[key] = copy.deepcopy(value)
    # 2. Map Sobol inputs to the configuration dictionary
    scan_cfg['PML']['Sx_r'] = row['Sx_r']
    scan_cfg['PML']['Sx_im'] = row['Sx_im']
    scan_cfg['PML']['px'] = row['px']
    scan_cfg['PLASMA']['ne_constant'] = row['ne_edge_linear']
    scan_cfg['WAVE']['n_para'] = row['n_para_linear']
    
    # 3. Dynamic L_pml computation
    temp_solver = LHCouplingSolver_2DHcurl_1DH1(scan_cfg, geom_mode="1D", box_medium="PLASMA", antenna_grill=None)
    _, n_perp_p, _ = temp_solver.compute_physics_parameters()
    
    lambda_perp = scan_cfg['WAVE']['lambda0'] / np.abs(n_perp_p.real)
    scan_cfg['DOMAIN']['Lx_pml'] = row['Lx_pml_ratio'] * lambda_perp
    scan_cfg['DOMAIN']['Lx_tot'] = scan_cfg['DOMAIN']['Lx_plasma'] + scan_cfg['DOMAIN']['Lx_pml']
    
    del temp_solver # Free memory immediately
    
    # 4. Execute the Ground-Truth FEM Solver
    solver = LHCouplingSolver_2DHcurl_1DH1(scan_cfg, geom_mode="1D", box_medium="PLASMA", antenna_grill=None)
    mesh = solver.build_mesh_with_PMLs()
    solver.build_physics_Stix_B_field()
    
    _, Gamma_R, _, diag_data = solver.solve_helmholtz_2DHcurl_1DH1_with_pml(mesh, "1D", "PLASMA")
    
    # 5. Populate successful results
    result['Gamma_R'] = Gamma_R
    result['DoFs'] = solver.fes.ndof
    result['Conservation_Error'] = diag_data['power_error_plasma']
    result['Success'] = True
        
    # except Exception as e:
    #     result['Error_Msg'] = str(e)
        
    # finally:
    # Force garbage collection
    if 'solver' in locals(): del solver
    if 'mesh' in locals(): del mesh
    gc.collect()
        
    return result

# =============================================================================
#              PHASE 2: Generate the datasat using Sobol Smapling
# =============================================================================
def run_phase2_batch_multiprocess(csv_filepath, max_workers=2):
    print(f"--- Starting Phase 2: Multiprocessed FEM Batch ---")
    print(f"[*] Allocating strictly {max_workers} CPU cores to protect VM resources.")
    
    df = pd.read_csv(csv_filepath, sep=";", index_col="Sim_ID", decimal=',')
    
    tasks = []
    for idx, row in df.iterrows():
        if pd.isna(row['Gamma_R']):
            tasks.append({'Sim_ID': idx, 'row_data': row})
            
    total_tasks = len(tasks)
    print(f"[*] Found {total_tasks} remaining simulations to execute.")
    
    if total_tasks == 0:
        return

    completed_count = 0
    save_interval = 10  
    
    # Use the mp_context argument to force 'spawn' safely through the executor
    mp_context = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        future_to_sim = {executor.submit(worker_simulation, task): task for task in tasks}
        
        for future in as_completed(future_to_sim):
            res = future.result()
            idx = res['Sim_ID']
            completed_count += 1
            
            if res['Success']:
                print(f"[{completed_count}/{total_tasks}] Sim {idx} SUCCESS -> Gamma_R: {res['Gamma_R']:.3e}")
                df.at[idx, 'Gamma_R'] = res['Gamma_R']
                df.at[idx, 'DoFs'] = res['DoFs']
                df.at[idx, 'Conservation_Error'] = res['Conservation_Error']
            else:
                print(f"[{completed_count}/{total_tasks}] Sim {idx} FAILED -> Error: {res['Error_Msg']}")
                
            if completed_count % save_interval == 0:
                df.to_csv(csv_filepath, sep=";", index_label="Sim_ID", decimal=',')
                print(f"    [Disk Checkpoint] Progress saved.")

    df.to_csv(csv_filepath, sep=";", index_label="Sim_ID", decimal=',')
    print(f"\n--- Phase 2 Complete. Final data saved ---")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
#              PHASE 3: Training the model (and test it) 
# =============================================================================
from sklearn.model_selection import learning_curve, KFold

def plot_rigorous_learning_curve(estimator, X_scaled, y, title="Learning Curve"):
    """
    Computes and plots a rigorous Learning Curve using K-Fold Cross-Validation.
    This guarantees the model is tested on multiple different 80/20 splits to rule out bias.
    """
    print(f"\n[*] Generating Learning Curve for: {title}")
    
    # 1. Define K-Fold Cross Validation (5 splits = testing 5 different 80/20 configurations)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 2. Define the training subset sizes (e.g., test using 20%, 40%, 60%, 80%, and 100% of available data)
    train_sizes_fractions = np.linspace(0.2, 1.0, 5)
    
    # 3. Compute the learning curve (Scikit-Learn handles the K-Fold looping automatically)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_scaled, y, cv=cv, n_jobs=-1, 
        train_sizes=train_sizes_fractions, scoring='r2'
    )
    
    # 4. Calculate Mean and Standard Deviation across the 5 different Folds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # 5. Plotting the Diagnostics
    plt.figure(figsize=(9, 6))
    plt.title(f"Rigorous Learning Curve: {title}", fontweight='bold')
    plt.xlabel("Number of Training Simulations", fontweight='bold')
    plt.ylabel(r"Cross-Validated $R^2$ Score", fontweight='bold')
    
    # The shaded regions represent the variance across the 5 different splits. 
    # If the band is narrow, there is zero bias.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.15, color="red")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.15, color="green")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training Score (Memorization)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Test Score (True Prediction)")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    save_name = f"Learning_Curve_{title.replace(' ', '_')}.png"
    plt.savefig(save_name, dpi=300)
    print(f"  -> Saved diagnostic plot: {save_name}")
    plt.show()

def train_surrogate_models(csv_filepath):
    print("--- Phase 3: Training Gaussian Process Surrogate Models ---")
    
    # 1. Load Data and Clean Failed Runs
    # Ensure we use the correct European formatting based on Phase 2
    df = pd.read_csv(csv_filepath, sep=";", decimal=",", index_col="Sim_ID")
    
    initial_count = len(df)
    df = df.dropna(subset=['Gamma_R', 'DoFs'])
    clean_count = len(df)
    print(f"[*] Loaded {initial_count} simulations. Dropped {initial_count - clean_count} failed runs.")
    
    if clean_count < 10:
        print("[!] ERROR: Not enough successful simulations to train a model.")
        return

    # 2. Define Inputs (Features) and Outputs (Targets)
    feature_cols = ['Sx_r', 'Sx_im', 'px', 'Lx_pml_ratio', 'log10_ne_edge', 'log10_n_para']
    X = df[feature_cols].values
    
    # Target 1: Efficiency (Log10 space)
    # Clip Gamma_R to avoid log10(0) if any simulation was "perfectly" absorbed
    Gamma_R_safe = np.clip(df['Gamma_R'].values, a_min=1e-12, a_max=None)
    y_gamma = np.log10(Gamma_R_safe)
    
    # Target 2: Computational Cost (Linear space)
    y_dofs = df['DoFs'].values

    # 3. Train-Test Split (80% Training, 20% Validation)
    X_train, X_test, yg_train, yg_test, yd_train, yd_test = train_test_split(
        X, y_gamma, y_dofs, test_size=0.2, random_state=42
    )
    
    # 4. Feature Scaling (Crucial for Dimensional Collapse)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define the Advanced Kernel (ARD Matern + White Noise)
    # length_scale=np.ones(6) enables Automatic Relevance Determination (ARD)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(6), nu=2.5) \
             + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-2))
    
    # 6. Initialize Gaussian Process Regressors
    # n_restarts_optimizer ensures it doesn't get stuck in local mathematical minima
    gpr_gamma = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=42)
    gpr_dofs = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    
    X_full_scaled = scaler.transform(X)
    plot_rigorous_learning_curve(gpr_gamma, X_full_scaled, y_gamma, title="Gamma_R Surrogate")
    plot_rigorous_learning_curve(gpr_dofs, X_full_scaled, y_dofs, title="DoFs Surrogate")

    # 7. Train the Models
    print("\n[*] Training Gamma_R Surrogate (This may take a minute on large datasets)...")
    gpr_gamma.fit(X_train_scaled, yg_train)
    
    print("[*] Training DoFs Surrogate...")
    gpr_dofs.fit(X_train_scaled, yd_train)
    
    # 8. Evaluate and Score the Models
    yg_pred, yg_std = gpr_gamma.predict(X_test_scaled, return_std=True)
    yd_pred, yd_std = gpr_dofs.predict(X_test_scaled, return_std=True)
    
    r2_gamma = r2_score(yg_test, yg_pred)
    r2_dofs = r2_score(yd_test, yd_pred)
    
    print(f"\n--- Model Validation Scores (R^2) ---")
    print(f"Gamma_R Model R^2 : {r2_gamma:.4f} (1.0 is perfect)")
    print(f"DoFs Model R^2    : {r2_dofs:.4f} (1.0 is perfect)")
    print(f"Optimized Gamma Kernel: {gpr_gamma.kernel_}")
    
    # 9. Save the Models and Scaler for Phase 4 & 5
    joblib.dump(gpr_gamma, 'surrogate_gamma.pkl')
    joblib.dump(gpr_dofs, 'surrogate_dofs.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("\n[+] Models and Scaler successfully saved to disk.")
    
    # 10. Generate Validation Plots
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Gamma_R
    plt.subplot(1, 2, 1)
    # plt.scatter(yg_test, yg_pred, alpha=0.7, color='blue', edgecolors='k')
    plt.errorbar(yg_test, yg_pred, yerr=yg_std, fmt='o', alpha=0.7, color='blue', ecolor='magenta', elinewidth=2, capsize=3)
    plt.plot([min(yg_test), max(yg_test)], [min(yg_test), max(yg_test)], 'r--', lw=2)
    plt.xlabel(r"True Value: $\log_{10}(\Gamma_R)$", fontweight='bold')
    plt.ylabel(r"Predicted Value: $\log_{10}(\Gamma_R)$", fontweight='bold')
    plt.title(f"Gamma_R Surrogate (R² = {r2_gamma:.3f})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: DoFs
    plt.subplot(1, 2, 2)
    plt.scatter(yd_test, yd_pred, alpha=0.7, color='green', edgecolors='k')
    plt.plot([min(yd_test), max(yd_test)], [min(yd_test), max(yd_test)], 'r--', lw=2)
    plt.xlabel("True Value: DoFs", fontweight='bold')
    plt.ylabel("Predicted Value: DoFs", fontweight='bold')
    plt.title(f"DoFs Surrogate (R² = {r2_dofs:.3f})")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("Surrogate_Validation.png", dpi=300)
    print("[+] Validation plot saved as 'Surrogate_Validation.png'")
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import joblib
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

def compute_sobol_indices():
    print("--- Phase 4: Global Sensitivity Analysis via Surrogate ---")
    
    # 1. Load the Trained GPR Model and Feature Scaler from Phase 3
    try:
        gpr_gamma = joblib.load('surrogate_gamma.pkl')
        scaler = joblib.load('feature_scaler.pkl')
    except FileNotFoundError:
        print("[!] ERROR: Surrogate models not found. Run Phase 3 first.")
        return

    # 2. Define the EXACT same parameter space as Phase 1
    log_ne_min, log_ne_max = np.log10(3e17), np.log10(3e19)
    log_npara_min, log_npara_max = np.log10(2.0), np.log10(100.0)
    
    problem = {
        'num_vars': 6,
        'names': ['Sx_r', 'Sx_im', 'px', 'Lx_pml_ratio', 'log10_ne_edge', 'log10_n_para'],
        'bounds': [
            [0.5, 5.0],
            [0.5, 5.0],
            [0.5, 5.0],
            [0.2, 5.0],
            [log_ne_min, log_ne_max],
            [log_npara_min, log_npara_max]
        ]
    }
    
    # 3. Generate a MASSIVE Sobol Sequence for converged statistics
    # N=10000 -> 10000 * (2*6 + 2) = 140,000 surrogate evaluations
    N_samples = 10000 
    print(f"[*] Generating {N_samples} base samples for Monte Carlo integration...")
    X_massive = sobol_sample.sample(problem, N_samples, calc_second_order=True)
    print(f"[*] Total surrogate evaluations required: {X_massive.shape[0]}")
    
    # 4. Predict using the Surrogate Model
    # We MUST scale the inputs using the exact same scaler fitted in Phase 3
    print("[*] Evaluating surrogate model...")
    X_massive_scaled = scaler.transform(X_massive)
    
    # We predict the log10(Gamma_R) array
    Y_pred_gamma = gpr_gamma.predict(X_massive_scaled)
    
    # 5. Compute Sobol Indices
    print("[*] Computing Sobol Variance Indices...")
    Si_gamma = sobol_analyze.analyze(problem, Y_pred_gamma, calc_second_order=True, print_to_console=False)
    
    # Extract S1 (First-order) and ST (Total-order)
    # Clip negative numerical noise artifacts to 0
    S1 = np.clip(Si_gamma['S1'], a_min=0.0, a_max=None)
    ST = np.clip(Si_gamma['ST'], a_min=0.0, a_max=None)
    
    # Sort parameters by Total-Order influence for cleaner plotting
    sorted_idx = np.argsort(ST)
    names_sorted = [problem['names'][i] for i in sorted_idx]
    S1_sorted = S1[sorted_idx]
    ST_sorted = ST[sorted_idx]
    
    # 6. Generate the Professional Sensitivity Bar Chart
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(names_sorted))
    
    # Plot ST (Total Effect) as light bars
    plt.barh(y_pos, ST_sorted, height=0.6, color='lightblue', edgecolor='black', label='Total-Order ($S_T$)')
    
    # Plot S1 (Main Effect) as overlapping solid bars
    plt.barh(y_pos, S1_sorted, height=0.6, color='navy', edgecolor='black', label='First-Order ($S_1$)')
    
    plt.yticks(y_pos, names_sorted, fontweight='bold')
    plt.xlabel('Fraction of Variance Explained', fontweight='bold')
    plt.title('Sobol Global Sensitivity Analysis (PML Reflection Coefficient)')
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add numerical labels to the bars
    for i in range(len(names_sorted)):
        plt.text(ST_sorted[i] + 0.01, i, f"{ST_sorted[i]:.3f}", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('Sobol_Sensitivity_Indices.png', dpi=300)
    print("[+] Sensitivity bar chart saved to 'Sobol_Sensitivity_Indices.png'")
    plt.show()
    
    # Print numerical conclusions
    print("\n--- Physical Conclusions ---")
    for i, name in reversed(list(enumerate(names_sorted))):
        interaction = ST_sorted[i] - S1_sorted[i]
        print(f"{name}: Total Impact = {ST_sorted[i]*100:.1f}% | Interaction = {interaction*100:.1f}%")





import numpy as np
import joblib
import pandas as pd
from scipy.optimize import differential_evolution

def robust_pml_optimization(max_allowed_dofs=15000):
    print("--- Phase 5: Robust Weighted Optimization ---")
    
    # 1. Load the Phase 3 Surrogate Models
    try:
        gpr_gamma = joblib.load('surrogate_gamma.pkl')
        gpr_dofs = joblib.load('surrogate_dofs.pkl')
        scaler = joblib.load('feature_scaler.pkl')
    except Exception as e:
        print(f"[!] Error loading models: {e}. Run Phase 3 first.")
        return

    # 2. Define the Environmental Grid (The Physical Scenarios)
    # We create a grid of 200 different plasma states to test every PML against
    log_ne_vals = np.linspace(np.log10(3e17), np.log10(3e19), 10)
    log_npara_vals = np.linspace(np.log10(2.0), np.log10(100.0), 20)
    
    env_grid = np.array(np.meshgrid(log_ne_vals, log_npara_vals)).T.reshape(-1, 2)
    n_scenarios = env_grid.shape[0]
    
    # 3. Define the Weighting Function for n_parallel
    # Heavily weights low n_para (e.g., n_para=2 gets weight ~1.0, n_para=100 gets weight ~0.001)
    linear_n_para = 10 ** env_grid[:, 1]
    weights = np.exp(-0.1 * (linear_n_para - 2.0))
    weights /= np.sum(weights) # Normalize

    # 4. Define the Objective Function for the Optimizer
    def objective_function(x):
        # x is the 4D array of decision variables: [Sx_r, Sx_im, px, Lx_ratio]
        
        # Tile the decision variables to match the 200 environmental scenarios
        X_decision = np.tile(x, (n_scenarios, 1))
        
        # Assemble the full 6D input array for the surrogate
        X_full = np.hstack((X_decision, env_grid))
        
        # Scale and Predict
        X_full_scaled = scaler.transform(X_full)
        log_gamma_pred = gpr_gamma.predict(X_full_scaled)
        dofs_pred = gpr_dofs.predict(X_full_scaled)
        
        # Convert log10(Gamma) back to linear Gamma for physical weighting
        gamma_linear = 10 ** log_gamma_pred
        
        # Compute Weighted Average Reflection
        weighted_gamma = np.sum(gamma_linear * weights)
        
        # Compute Max Cost
        max_dofs = np.max(dofs_pred)
        
        # Penalty Function for violating the DoF constraint
        if max_dofs > max_allowed_dofs:
            penalty = (max_dofs - max_allowed_dofs) * 1e3
            return weighted_gamma + penalty
            
        return weighted_gamma

    # 5. Define Bounds for the Decision Variables
    bounds = [
        (0.5, 5.0),  # Sx_r
        (0.5, 5.0),  # Sx_im
        (0.5, 5.0),  # px
        (0.2, 5.0)   # Lx_pml_ratio
    ]
    
    print(f"[*] Launching Differential Evolution Optimizer...")
    print(f"[*] Constraint: Max DoFs = {max_allowed_dofs}")
    print(f"[*] Weighting: Exponential decay favoring low n_parallel")
    
    # 6. Run the Genetic Algorithm
    result = differential_evolution(
        objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=100, 
        popsize=15, 
        disp=True
    )
    
    if result.success:
        opt_x = result.x
        print("\n==================================================")
        print(" [SUCCESS] ROBUST OPTIMAL PML CONFIGURATION FOUND")
        print("==================================================")
        print(f"  Sx_r         : {opt_x[0]:.3f}")
        print(f"  Sx_im        : {opt_x[1]:.3f}")
        print(f"  px           : {opt_x[2]:.3f}")
        print(f"  Lx_pml_ratio : {opt_x[3]:.3f}")
        print(f"  Weighted Expected Gamma_R : {result.fun:.3e}")
        print("==================================================")
        
        # 7. Uncertainty and Interval Estimation
        # We perturb the optimal solution by +/- 10% to check the stability valley
        print("\n[*] Estimating Stability Intervals (+/- 10% Perturbation)...")
        perturbations = [0.9, 1.1]
        for i, name in enumerate(['Sx_r', 'Sx_im', 'px', 'Lx_pml_ratio']):
            for p in perturbations:
                x_pert = opt_x.copy()
                x_pert[i] *= p
                # Clip to bounds
                x_pert[i] = np.clip(x_pert[i], bounds[i][0], bounds[i][1])
                new_gamma = objective_function(x_pert)
                degradation = (new_gamma - result.fun) / result.fun * 100
                direction = "-10%" if p == 0.9 else "+10%"
                print(f"  Shift {name} by {direction:4s} -> Performance degradation: {degradation:.1f}%")
                
    else:
        print("[!] Optimizer failed to converge.")
