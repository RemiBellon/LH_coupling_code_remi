# import netgen.gui
import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time 

saving_file_path= "\\canicula5-cifs.intra.cea.fr\RB286887\LH_coupling_code_remi"
# =====================================================================
# 1. MESH GENERATION (Plasma Domain + PML Domain)
# =====================================================================

def create_mesh_with_pml_3D(Lx_plasma, Lx_pml, Lz_approx, resolution, lambda_0, theta_deg):
    Lx_tot = Lx_plasma + Lx_pml
    
    # 1. Compute Exact Lz
    if abs(theta_deg) < 1e-6:
        kx_exact = 2 * np.pi / lambda_0
        kz_exact = 0.0
        Lz_exact = Lz_approx
    else:
        k_wave_vacuum = 2 * np.pi / lambda_0
        lambda_z = lambda_0 / np.abs(np.sin(np.radians(theta_deg)))
        lambda_z_multiple = max(1, round(Lz_approx / lambda_z))
        Lz_exact = lambda_z_multiple * lambda_z
        kz_exact = lambda_z_multiple * (2 * np.pi / Lz_exact) * np.sin(np.radians(theta_deg))
        kx_exact = np.sqrt(k_wave_vacuum**2 - kz_exact**2)

    # 2.5D Trick: Make the Y-dimension exactly 1 element thick
    Ly_slice = resolution 
    print(f"--- Generating 2.5D Mesh ---")
    print(f"Box: X={Lx_tot:.3f}, Y={Ly_slice:.3f}, Z={Lz_exact:.3f}")

    # Create 3D Boxes
    plasma_box = occ.Box((0, 0, 0), (Lx_plasma, Ly_slice, Lz_exact))
    pml_box = occ.Box((Lx_plasma, 0, 0), (Lx_tot, Ly_slice, Lz_exact))
    
    plasma_box.faces.name = "plasma_region"
    pml_box.faces.name = "pml_region"

    glued_shape = occ.Glue([plasma_box, pml_box])
    
    # Name the Faces
    for f in glued_shape.faces:
        cx, cy, cz = f.center.x, f.center.y, f.center.z
        if cx < 1e-6: f.name = "left_source"
        elif cx > Lx_tot - 1e-6: f.name = "right_perf_el_cond"
        elif cz > Lz_exact - 1e-6: f.name = "top"
        elif cz < 1e-6: f.name = "bottom"
        elif cy > Ly_slice - 1e-6: f.name = "front"
        elif cy < 1e-6: f.name = "back"

    # Apply Top/Bottom Periodicity (Z-axis)
    for f_top in glued_shape.faces:
        if f_top.name == "top":
            for f_bot in glued_shape.faces:
                if f_bot.name == "bottom" and abs(f_top.center.x - f_bot.center.x) < 1e-6 and abs(f_top.center.y - f_bot.center.y) < 1e-6:
                    f_top.Identify(f_bot, "periodic_z", occ.IdentificationType.PERIODIC)

    # Apply Front/Back Periodicity (Y-axis) -> Forces 2D Physics
    for f_front in glued_shape.faces:
        if f_front.name == "front":
            for f_back in glued_shape.faces:
                if f_back.name == "back" and abs(f_front.center.x - f_back.center.x) < 1e-6 and abs(f_front.center.z - f_back.center.z) < 1e-6:
                    f_front.Identify(f_back, "periodic_y", occ.IdentificationType.PERIODIC)

    geo = occ.OCCGeometry(glued_shape)
    ngmesh = geo.GenerateMesh(maxh=resolution)
    
    return Mesh(ngmesh), Lz_exact, kx_exact, kz_exact, Ly_slice

# =====================================================================
# 2. 3D VECTOR WEAK FORM SOLVER (Jacquot 2013 Artificial Medium)
# =====================================================================
def solve_helmholtz_Hcurl_3D_pml(mesh, k_wave_vacuum, kz_exact, Lx_plasma, Lx_pml):
    # Note: Using order=3 to save RAM in 3D. If your PC has >16GB RAM, you can try order=4.
    base_fes = HCurl(mesh, order=3, complex=True, dirichlet="left_source|right_perf_el_cond")
    fes = Periodic(base_fes)
    print(f'#DoFs = {fes.ndof}')
    
    # --- Jacquot 2013 Artificial PML Tensors ---
    Sr_Real = 1.0
    Sr_Imag = 1.0  # Positive imaginary stretch for e^{ikx} convention
    pr = 2.0
    
    norm_dist = (x - Lx_plasma) / Lx_pml
    stretch_func = IfPos(x - Lx_plasma, (Sr_Real + 1j * Sr_Imag) * (norm_dist**pr), 0.0)
    Sx = 1.0 + stretch_func

    # 3x3 Dielectric Tensor (epsilon)
    eps_tensor = CF((1/Sx, 0.0, 0.0,
                     0.0,  Sx,  0.0,
                     0.0,  0.0, Sx), dims=(3,3))

    # 3x3 Inverse Permeability Tensor (mu^-1)
    mu_inv_tensor = CF((Sx,   0.0,  0.0,
                        0.0, 1/Sx,  0.0,
                        0.0,  0.0, 1/Sx), dims=(3,3))
    
    u, v = fes.TnT()
    a = BilinearForm(fes, symmetric=True)
    a += (mu_inv_tensor * curl(u) * curl(v) - k_wave_vacuum**2 * eps_tensor * u * v) * dx
    
    with TaskManager():
        a.Assemble()
    
    f = LinearForm(fes)
    f.Assemble()

    kx_exact = np.sqrt(k_wave_vacuum**2 - kz_exact**2)
    Ex_comp = kz_exact / k_wave_vacuum
    Ez_comp = -kx_exact / k_wave_vacuum
    
    # E_y is 0.0. The phase varies along the z-axis.
    E_vector = CF((Ex_comp, 0.0, Ez_comp)) * exp(1j * kz_exact * z)
   
    gfu = GridFunction(fes)
    gfu.Set(E_vector, definedon=mesh.Boundaries("left_source"))
    
    print("Solving the 3D vector linear system...")
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    
    inv = a.mat.Inverse(freedofs=fes.FreeDofs())
    gfu.vec.data += inv * res
    
    return gfu, fes.ndof

# =====================================================================
# 3.1 PML DIAGNOSTIC --- Poynting flux
# =====================================================================
def pml_diag_poynting_flux(mesh, gfu, freq_LH):
    '''
    Compute radial Poynting vector flux + verification: power injected = power transmitted
    '''
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * freq_LH
    Px = (1.0 / (2.0 * omega * mu_0)) * (Conj(gfu)) * grad(gfu)[0]

    total_power = Integrate(Px, mesh.Materials("plasma_region")) # definedon="pml_region")
    print(f'Total transmitted Power Flux: {total_power:.4e} W')
    return total_power

# =====================================================================
# 3.2 PML DIAGNOSTIC --- SWR & Reflexion coeffs
# =====================================================================
def pml_diag_SWR_eta(mesh, gfu, kx, Lx_plasma, L_pml_r, Lz):
    """
    Evaluate SWR and reflection coeff (analytic and simulation)
    """
    print('--- Running PML diag ---')
   # Sample at the middle of the box in toroidal direction: 
    z_lmid_line = Lz / 2.0
    x_pml_coords = np.linspace(0, Lx_plasma, 500)

    E_abs =[]
    for x in x_pml_coords:
        pt = mesh(x, z_lmid_line)
        if pt:
           # To find standing wave we take the abs value of E field 
            E_abs.append(abs(gfu(pt)))
        
    E_abs_array = np.array(E_abs)

   # Compute SWR
    min_E, max_E = np.min(E_abs), np.max(E_abs)
    SWR = max_E / min_E
    eta_sim = (SWR - 1.0) / (SWR + 1.0)

   # Analytical predcition eta_pred
   # Because SetPML = HalfSpace==> Constant stretching ==> p = 0  
    Sr_Im = 1
    pr = 0
    eta_pred_fwd_wave = np.exp(-2. * abs(kx) * L_pml_r * (Sr_Im) / (1 + pr))
    eat_pred_evan_wave = (np.exp(-2. * abs(kx) * L_pml_r * (1 + Sr_Im) / (pr + 1))) / (np.exp(2 * abs(kx) * L_pml_r))

    print(f'SWR = {SWR:.6f}')
    print(f'eta_sim = {eta_sim:.6f}')
    print(f'eta_pred_fwd_wave = {eta_pred_fwd_wave:.6f}')
    print(f'eta_pred_evan_wave = {eat_pred_evan_wave:.6f}')

    plt.figure(figsize=(8,4))
    plt.plot(x_pml_coords, E_abs, color = 'Royalblue', label = r'$\|E\| Envelope$')
    plt.axhline(y=max_E, color = 'crimson', linestyle='--', alpha=0.5, label = r'$Max \|E\|$')
    plt.axhline(y=min_E, color = 'green', linestyle='--', alpha=0.5, label = r'$Min \|E\|$')
    plt.tick_params(direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)
    plt.xlabel(r'$x\ [m]$',fontsize=14)
    plt.ylabel(r'$\|E\|\ [V/m]$',fontsize=14)
    # plt.legend(loc = 'best', fontsize=14)
    plt.tight_layout()
    plt.savefig(saving_file_path + "\Plane_Wave_E_field_envelope_vs_radiale_direction.png", dpi=300)
    plt.show()

# =====================================================================
# 3.3 MESH DIAG --- Mesh convergence 
# =====================================================================
def mesh_diag_L2_error(mesh, gfu, k):
    '''
    Compute the L2 norm of error between analytical and simulated plane wave.
    '''
    u_exact = exp(1j * k * x)
    error_expr = Norm(gfu - u_exact)**2
    L2_error = sqrt(Integrate(error_expr, mesh.Materials("plasma_region")))
    
    return L2_error

def mesh_diag_convergence_study(Lx_plasma, Lx_pml, Lz_approx, k_wave_vacuum, kz_exact, lambda_0, theta_deg):
    '''
    Precision vs computation time ==> optimal mesh resolution based on simulation parameters
    '''
    print('--- Run convergence study ---')
    resolutions = np.linspace(1e-3, 0.025, 200) 
    dofs_list, errors_list, times_list = [], [], []
    
    for res in resolutions:
        t0 = time.time()
        mesh = create_mesh_with_pml_3D(Lx_plasma, Lx_pml, Lz_approx, res, lambda_0, theta_deg)
        gfu, ndof = solve_helmholtz_Hcurl_3D_pml(mesh, k_wave_vacuum, kz_exact, Lx_plasma, Lx_pml)
        t_solve = time.time() - t0

        error = mesh_diag_L2_error(mesh, gfu, k_wave_vacuum)
        dofs_list.append(ndof)
        errors_list.append(error)
        times_list.append(t_solve)
        print(f'Res: {res:.3f}, Dofs:{ndof:5d}, L2 error:{error:.2e}, Time:{t_solve:.3f}s')

    # Convert to numpy arrays for vectorized math
    dofs_arr = np.array(dofs_list)
    errors_arr = np.array(errors_list)
    times_arr = np.array(times_list)
    
    # =====================================================================
    # APPLYING THE 3 SPECIFIC MASKS FOR LINEAR REGRESSIONS
    # =====================================================================
    # 1. Fit L2 Error vs DoFs (Condition: DoFs < 3e9)
    # Note: 3e9 is massive, this will likely include all points unless changed.
    mask_l2_dofs = dofs_arr < 9e3
    slope_l2_dofs, int_l2_dofs = np.polyfit(np.log10(dofs_arr[mask_l2_dofs]), np.log10(errors_arr[mask_l2_dofs]), 1)
    
    # 2. Fit CPU Time vs DoFs (Condition: DoFs > 1e4)
    mask_time_dofs = dofs_arr > 1e4
    slope_time_dofs, int_time_dofs = np.polyfit(np.log10(dofs_arr[mask_time_dofs]), np.log10(times_arr[mask_time_dofs]), 1)
    
    # 3. Fit L2 Error vs CPU Time (Condition: Time < 3e-2 s)
    mask_l2_time = times_arr < 1
    slope_l2_time, int_l2_time = np.polyfit(np.log10(times_arr[mask_l2_time]), np.log10(errors_arr[mask_l2_time]), 1)

    print("\n--- Fit Results ---")
    print(f"L2 Error vs DoFs Slope (p): {slope_l2_dofs:.3f}")
    print(f"CPU Time vs DoFs Slope (q): {slope_time_dofs:.3f}")
    print(f"L2 Error vs Time Slope:     {slope_l2_time:.3f}")

    # =====================================================================
    # GENERATING THE TREND LINES (Bounded to their masked regions)
    # =====================================================================
    # Create smooth X arrays strictly bounded between the min and max of the filtered data
    dofs_line_l2 = np.geomspace(min(dofs_arr[mask_l2_dofs]), max(dofs_arr[mask_l2_dofs]), 50)
    dofs_line_time = np.geomspace(min(dofs_arr[mask_time_dofs]), max(dofs_arr[mask_time_dofs]), 50)
    times_line_l2 = np.geomspace(min(times_arr[mask_l2_time]), max(times_arr[mask_l2_time]), 50)

    # Compute the Y arrays using the correctly unpacked slopes and intercepts
    L2_fit_dofs_y = 10**(slope_l2_dofs * np.log10(dofs_line_l2) + int_l2_dofs)
    Time_fit_dofs_y = 10**(slope_time_dofs * np.log10(dofs_line_time) + int_time_dofs)
    L2_fit_time_y = 10**(slope_l2_time * np.log10(times_line_l2) + int_l2_time)

    # =====================================================================
    # PLOTTING
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Subplot 1: DoFs vs Error and Time ---
    ax1.set_xlabel('Degrees of Freedom (DoFs)', fontsize=14)
    ax1.set_ylabel('L2 Error', fontsize=14)
    
    # Data points
    ax1.loglog(dofs_arr, errors_arr, marker='o', color='royalblue', linestyle='None', alpha=0.5, label="L2 Error (Data)")
    # Fit line
    ax1.loglog(dofs_line_l2, L2_fit_dofs_y, color='darkorange', linewidth=3, label=f"Error Fit (Slope: {slope_l2_dofs:.2f})")

    ax1.tick_params(axis='y', labelcolor="royalblue")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('CPU Time [s]', fontsize=14)
    
    # Data points
    ax1_twin.loglog(dofs_arr, times_arr, marker='s', color="crimson", linestyle="None", alpha=0.5, label="CPU time (Data)")
    # Fit line
    ax1_twin.loglog(dofs_line_time, Time_fit_dofs_y, color="k", linewidth=3, linestyle='--', label=f"Time Fit (Slope: {slope_time_dofs:.2f})")
    
    ax1_twin.tick_params(axis='y', labelcolor="crimson", direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    ax1.set_title('Algorithmic Scaling', fontsize=14)

    # --- Subplot 2: Pareto Frontier (Time vs Error) ---
    ax2.set_xlabel('CPU Time (s)', fontsize=12)
    ax2.set_ylabel('L2 Error', fontsize=12)
    
    # Data points
    ax2.loglog(times_arr, errors_arr, marker='D', color='royalblue', linestyle='None', alpha=0.6, markersize=6, label='Data Points')
    # Fit line
    ax2.loglog(times_line_l2, L2_fit_time_y, color='crimson', linewidth=3, label=f'Pareto Fit (Slope: {slope_l2_time:.2f})')
    ax2.tick_params(direction='in', length="6", width="4", bottom=True, top=True, right=True, left=True)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend()
    ax2.set_title('Pareto Frontier (Cost vs. Precision)', fontsize=14)

    fig.tight_layout()
    # Replace the path below with a local relative path or pure filename if possible
    plt.savefig("L2_Error_and_CPU_ti.svg", dpi=300)
    plt.show()
# =====================================================================
# 4. VISUALIZATION 
# =====================================================================
def plot_wave_snapshot(mesh, gfu, Lx_plasma):
    """
    Plots the REAL part of the field to show the physical oscillating wave at t=0.
    """
    print("Generating 2D wave E map...")
    pts = np.array([v.point for v in mesh.vertices])
    X = pts[:, 0]
    Z = pts[:, 1]
    elements = []
    for el in mesh.Elements(VOL):
        if len(el.vertices) == 3:
            elements.append([v.nr for v in el.vertices])
    elements = np.array(elements)

    z_vals = np.array([gfu(mesh(*p))[1] for p in pts])

    triang = mtri.Triangulation(Z, X, elements)

    plt.figure(figsize=(10, 4))
    plt.tricontourf(triang, z_vals, levels=50, cmap='plasma', alpha=1)
    plt.colorbar(label='Physical Wave Field $Re(E_z)$')
    
    # Draw a line to show where the PML starts
    plt.axhline(y=Lx_plasma, color='black', linestyle='--', label='PML Entrance')
    
    # plt.title('Plane Wave Propagating')
    plt.xlabel(r'$z\ [m]$', fontsize=14)
    plt.ylabel(r'$x\ [m]$', fontsize=14)
    plt.legend(loc='upper left')
    plt.tick_params(direction='out', length=6, bottom=True, top=True, right=True, left=True)
    plt.tight_layout()
    plt.savefig(saving_file_path + "\2D_E_field_Re_Ez.png", dpi=300)
    plt.show()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # --- Physics Parameters ---
    freq_LH = 3.7e9  
    c0 = 3.0e8   
    lambda_vacuum = c0 / freq_LH
    k_wave_vacuum = 2 * np.pi / lambda_vacuum
    
    # --- Geometry Parameters ---
    Lx_plasma = 0.5
    Lz_approx = 0.4
    
    # Jacquot 2013 states PML depth around 0.5 to 1 wavelength is sufficient.
    Lx_pml = lambda_vacuum * 4 
    Lx_total = Lx_plasma + Lx_pml

    max_h = lambda_vacuum / 12.0 
    print('resol = Lx_plasma/max_h = ', Lx_plasma/max_h)
    theta_deg_target = 85 # in degree

    mesh, kx_exact, kz_exact, Lz_exact, Ly_slice = create_mesh_with_pml_3D(Lx_plasma, Lx_pml, Lz_approx, max_h, lambda_vacuum, theta_deg_target) 
    u_sol, _ = solve_helmholtz_Hcurl_3D_pml(mesh, k_wave_vacuum, kz_exact, Lx_plasma, Lx_pml)
    # pml_diag_poynting_flux(mesh, u_sol, freq_LH)
    # pml_diag_SWR_eta(mesh, u_sol, k_wave, Lx_plasma, L_pml, Lz)
    plot_wave_snapshot(mesh, u_sol, Lx_plasma)
    # mesh_diag_convergence_study(Lx_plasma, L_pml, Lz, k_wave)
