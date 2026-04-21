import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. MESH GENERATION (Plasma Domain + PML Domain)
# =====================================================================
def create_mesh_with_pml(Lx_plasma, L_pml, Ly, resolution):
    """
    Creates a 2D rectangular mesh with a distinct PML region on the right.
    """
    Lx_total = Lx_plasma + L_pml
    print(f"Generating mesh: Plasma({Lx_plasma}x{Ly}) + PML({L_pml}x{Ly})")
    
    # Create the full rectangle
    rect = occ.WorkPlane().Rectangle(Lx_total, Ly).Face()
    
    # Name the boundaries
    left = rect.edges.Min(occ.X)
    right = rect.edges.Max(occ.X)
    bottom = rect.edges.Min(occ.Y)
    top = rect.edges.Max(occ.Y)

    left.name = "left_source"
    right.name = "right_pec" # Jacquot: PML is ended by a Perfect Electric Conductor
    bottom.name = "bottom"
    top.name = "top"

    # Periodic top and bottom to simulate infinite Y space
    top.Identify(bottom, "periodic", occ.IdentificationType.PERIODIC)
    
    geo = occ.OCCGeometry(rect, dim=2)
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
    base_fes = H1(mesh, order=4, complex=True, dirichlet="left_source|right_pec")
    fes = Periodic(base_fes)
    print('#DoFs = ', fes.ndof)
    
    # --- Activate the PML Coordinate Stretching ---
    # We apply a PML starting at x = Lx_plasma, absorbing in the +x direction (1,0).
    # The 'alpha' parameter maps to the imaginary stretch factor S'' in the paper.
    mesh.SetPML(pml.HalfSpace(point=(Lx_plasma, 0), normal=(1, 0), alpha=1j))
    
    u, v = fes.TnT()
    
    # --- The Bilinear Form ---
    # Because mesh.SetPML automatically handles the complex coordinate mapping,
    # we just write the standard vacuum physics. We NO LONGER need the Robin condition.
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
    
    # 2. The right boundary ("right_pec") naturally defaults to 0.0, 
    # fulfilling the Jacquot requirement for a perfect conductor.
    
    # --- Solve the Linear System ---
    print("Solving the linear system...")
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    
    # freedofs ignores both the source nodes and the PEC nodes
    inv = a.mat.Inverse(freedofs=fes.FreeDofs())
    gfu.vec.data += inv * res
    
    # Turn off the PML mapping after solving so we can plot the physical grid properly
    mesh.UnSetPML()
    
    return gfu

# =====================================================================
# 3. VISUALIZATION 
# =====================================================================
def plot_wave_snapshot(mesh, gfu, Lx_total, Ly, Lx_plasma, nx=400, ny=100):
    """
    Plots the REAL part of the field to show the physical oscillating wave at t=0.
    """
    print("Generating 2D wave snapshot...")
    x_coords = np.linspace(0, Lx_total, nx)
    y_coords = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z_real = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            pt = mesh(X[i,j], Y[i,j])
            if pt: 
                # Take the REAL part to see the wave peaks and troughs
                Z_real[i,j] = gfu(pt).real 
                
    plt.figure(figsize=(10, 4))
    plt.contourf(X, Y, Z_real, levels=50, cmap='RdBu')
    plt.colorbar(label='Physical Wave Field $Re(E_z)$')
    
    # Draw a line to show where the PML starts
    plt.axvline(x=Lx_plasma, color='black', linestyle='--', label='PML Entrance')
    
    plt.title('Snapshot of Plane Wave Propagating into PML ($t=0$)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
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
    Ly = 0.4
    
    # Jacquot 2013 states PML depth around 0.5 to 1 wavelength is sufficient.
    L_pml = lambda_0 * 1.5 
    Lx_total = Lx_plasma + L_pml
    
    max_h = lambda_0 / 10.0 
    
    mesh = create_mesh_with_pml(Lx_plasma, L_pml, Ly, max_h)
    u_sol = solve_helmholtz(mesh, k_wave, Lx_plasma)
    
    plot_wave_snapshot(mesh, u_sol, Lx_total, Ly, Lx_plasma)