# import netgen.gui
import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time 

# =====================================================================
# 1. MESH GENERATION (Plasma Domain + PML Domain)
# =====================================================================
def create_mesh_with_pml(Lx_plasma, L_pml, Lz, resolution):
    """
    Creates a 2D rectangular mesh with a distinct PML region on the right.
    """
    Lx_total = Lx_plasma + L_pml
    print(fr"Generating mesh: Plasma: {Lx_plasma:.2f}(m)x{Lz:.2f}(m) + PML: {L_pml:.2f}x{Lz:.2f}(m)")
    
    # Create the full rectangle
    total_rect = occ.WorkPlane().Rectangle(Lx_total, Lz).Face()
    plasma_rect = occ.WorkPlane().Rectangle(Lx_plasma, Lz).Face()
    pml_rect = total_rect - plasma_rect

    plasma_rect.faces.name = "plasma_region"
    pml_rect.faces.name = "pml_region"

    glued_shape = occ.Glue([plasma_rect, pml_rect]) # both regions share nodes at the edges 
    for e in glued_shape.edges:
       # print('e = ', e)
        cx, cy = e.center.x, e.center.y
        if cx < 1e-6:
            e.name = "left_source"
        elif cx > Lx_total-1e-6:
            e.name = "right_perf_el_cond"
        elif cy > Lz - 1e-6:
            e.name = "top"
        elif cy < 1e-6:
            e.name = "bottom"

    for e_top in glued_shape.edges:
        if e_top.name == "top":
            for e_bot in glued_shape.edges:
                if e_bot.name == "bottom" and abs(e_top.center.x - e_bot.center.x) < 1e-6:
                    e_top.Identify(e_bot, "periodic", occ.IdentificationType.PERIODIC)

    geo = occ.OCCGeometry(glued_shape, dim=2)
    ngmesh = geo.GenerateMesh(maxh=resolution)
    return Mesh(ngmesh)

# =====================================================================
# 2. GENERAL WEAK FORM SOLVER
# =====================================================================
def solve_helmholtz(mesh, k, Lx_plasma):
    """
    Solves the Helmholtz equation with a purely progressive wave in vacuum.
    Uses NGSolve's native coordinate stretching for the PML.
    """
    # Define the Space. 
    # 'left_source' is for the injected wave.
    # 'right_pec' is the perfect conductor terminating the PML (u=0).
    base_fes = H1(mesh, order=4, complex=True, dirichlet="left_source|right_perf_el_cond")
    fes = Periodic(base_fes)
    print('#DoFs = ', fes.ndof)
    
    # --- Activate the PML Coordinate Stretching ---
    # We apply a PML starting at x = Lx_plasma, absorbing in the +x direction (1,0).
    # The 'alpha' parameter maps to the imaginary stretch factor S'' in the paper.
    mesh.SetPML(pml.HalfSpace(point=(Lx_plasma, 0), normal=(1, 0), alpha=1j), "pml_region")
    
    u, v = fes.TnT()
    
    # --- The Bilinear Form ---
    # Because mesh.SetPML automatically handles the complex coordinate mapping,
    # we just write the standard vacuum physics. No Robin condition
    a = BilinearForm(fes, symmetric=True)
    a += (grad(u)*grad(v) - k**2 * u * v) * dx
    
    with TaskManager():
        a.Assemble()
    
    # --- The Source (Linear Form) ---
    f = LinearForm(fes)
    f.Assemble() 
    
    # --- The Dirichlet Boundary Conditions ---
    gfu = GridFunction(fes)
    
    # 1. Inject the plane wave on the left (amplitude = 1, uniform phase)
    # This represents E_z = 1 at x=0.
    gfu.Set(1.0, definedon=mesh.Boundaries("left_source"))
    
    # 2. The right boundary ("right_pec") naturally defaults to 0.0, cf. Jacquot 2013
    
    # --- Solve the Linear System ---
    print("Solving the linear system...")
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    
    # freedofs ignores both the source nodes and the PEC nodes
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
    p_r = 0
    eta_pred_fwd_wave = np.exp(-2. * abs(kx) * L_pml_r * (Sr_Im) / (1 + p_r))
    eat_pred_evan_wave = (np.exp(-2. * abs(kx) * L_pml_r * (1 + Sr_Im) / (p_r + 1))) / (np.exp(2 * abs(kx) * L_pml_r))

    print(f'SWR = {SWR:.6f}')
    print(f'eta_sim = {eta_sim:.6f}')
    print(f'eta_pred_fwd_wave = {eta_pred_fwd_wave:.6f}')
    print(f'eta_pred_evan_wave = {eat_pred_evan_wave:.6f}')

    plt.figure(figsize=(8,4))
    plt.plot(x_pml_coords, E_abs, color = 'Royalblue', label = r'$\|E\| Envelope$')
    plt.axhline(y=max_E, color = 'crimson', linestyle='--', alpha=0.5, label = r'$Max \|E\|$')
    plt.axhline(y=min_E, color = 'green', linestyle='--', alpha=0.5, label = r'$Min \|E\|$')

    plt.xlabel(r'$x\ [m]$',fontsize=14)
    plt.ylabel(r'$\|E\|\ [V/m]$',fontsize=14)
    # plt.legend(loc = 'best', fontsize=14)
    plt.tight_layout()

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

def mesh_diag_convergence_study(Lx_plasma, L_pml, Lz, k_wave):
    '''
    Precision vs computation time ==> optimal mesh resolution based on simulation parameters
    '''
    print('--- Run convergence study ---')
    resolutions = np.logspace(1e-3, 0.05, 200) 
    dofs_list, errors_list, times_list = [], [], []
    
    for res in resolutions:
        t0 = time.time()
        mesh = create_mesh_with_pml(Lx_plasma, L_pml, Lz, res)
        gfu, ndof = solve_helmholtz(mesh, k_wave, Lx_plasma)
        t_solve = time.time() - t0

        error = mesh_diag_L2_error(mesh, gfu, k_wave)
        dofs_list.append(ndof)
        errors_list.append(error)
        times_list.append(t_solve)
        print(f'Res: {res:.3f}, Dofs:{ndof:5d}, L2 error:{error:.2e}, Time:{t_solve:.3f}s')

        dofs_arr, errors_arr, times_arr = np.array(dofs_list), np.array(errors_list), np.array(times_list)
        fit_mask = dofs_arr <3e8

        dofs_arr_fit, errors_arr_fit = dofs_arr[fit_mask], errors_arr[fit_mask]

        error_slope, _ = np.polyfit(np.log10(dofs_arr_fit), np.log10(errors_arr_fit), 1)
        time_slope, _ = np.polyfit(np.log10(dofs_arr), np.log10(times_arr), 1)
        print(f"L2 Error Slope (p): {error_slope:.3f}  --> Error scales as O(DoFs^{error_slope:.2f})")
        print(f"CPU Time Slope (q): {time_slope:.3f}   --> Time scales as O(DoFs^{time_slope:.2f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_xlabel('Degrees of Freedom (DoFs)', fontsize=14)
    ax1.set_ylabel('L2 Error', fontsize=14)
    ax1.loglog(dofs_list, errors_list, marker='o', color='royalblue', linestyle=':', linewidth=2, label=f"Error Slope: {error_slope:.2f} error/Dofs")
    ax1.tick_params(axis='y', labelcolor="Royalblue")
    ax1.grid(True, which="both", ls="--")

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('CPU Time [s]', fontsize=14)
    ax1_twin.loglog(dofs_list, times_list, marker='s', color="crimson", linestyle=":", label=f"Time Slope: {time_slope:.2f} s/Dofs")
    ax1_twin.tick_params(axis='y', labelcolor="crimson")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    ax1.set_title('Algorithmic Scaling', fontsize=14)

    # 2. Pareto Frontier Plot: Time vs Error
    ax2.loglog(times_list, errors_list, marker='D', color='indigo', linewidth=2, markersize=8)
    ax2.set_xlabel('CPU Time (s)', fontsize=12)
    ax2.set_ylabel('L2 Error', fontsize=12)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Annotate the points on the Pareto curve so you know which mesh generated them
    for i, dof in enumerate(dofs_list):
        ax2.annotate(f"{dof} DoFs", (times_list[i], errors_list[i]), 
                     textcoords="offset points", xytext=(10,-5), ha='left', fontsize=10)
    fig.tight_layout()
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

    z_vals = np.array([gfu(mesh(*p)).real for p in pts])

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
    plt.tight_layout()
    plt.show()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # --- Physics Parameters ---
    freq_LH = 3.7e9  
    c0 = 3.0e8   
    lambda_0 = c0 / freq_LH
    k_wave = 2 * np.pi / lambda_0
    
    # --- Geometry Parameters ---
    Lx_plasma = 0.4
    Lz = 0.4
    
    # Jacquot 2013 states PML depth around 0.5 to 1 wavelength is sufficient.
    L_pml = lambda_0 * 0.75 
    Lx_total = Lx_plasma + L_pml

    max_h = lambda_0 / 12.0 
    print('resol = Lx_plasma/max_h = ', Lx_plasma/max_h)


    mesh = create_mesh_with_pml(Lx_plasma, L_pml, Lz, max_h)
    u_sol, _ = solve_helmholtz(mesh, k_wave, Lx_plasma)
    pml_diag_poynting_flux(mesh, u_sol, freq_LH)
    pml_diag_SWR_eta(mesh, u_sol, k_wave, Lx_plasma, L_pml, Lz)
    plot_wave_snapshot(mesh, u_sol, Lx_plasma)
    mesh_diag_convergence_study(Lx_plasma, L_pml, Lz, k_wave)
