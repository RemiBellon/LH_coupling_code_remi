import netgen.occ as occ
from ngsolve import *
import numpy as np
import matplotlib.pyplot as plt
from ngsolve.webgui import Draw

# =====================================================================
# 1. MESH GENERATION
# =====================================================================
def create_rectangular_mesh(Lx, Ly, resolution):
    """
    Creates a 2D rectangular mesh using Netgen's OpenCASCADE (OCC) kernel.
    """
    print(f"Generating mesh: {Lx} x {Ly} with max element size {resolution}")
    
    # Create a rectangle from (0,0) to (Lx, Ly)
    rect = occ.WorkPlane().Rectangle(Lx, Ly).Face()
    
    # Name the boundaries so we can apply specific physics to them later
    left = rect.edges.Min(occ.X)
    right = rect.edges.Max(occ.X)
    bottom = rect.edges.Min(occ.Y)
    top = rect.edges.Max(occ.Y)

    left.name = "left"
    right.name = "right"
    bottom.name = "bottom"
    top.name = "top"

    top.Identify(bottom, "periodic", occ.IdentificationType.PERIODIC)
    # Generate the mesh
    geo = occ.OCCGeometry(rect, dim=2)
    ngmesh = geo.GenerateMesh(maxh=resolution)
    return Mesh(ngmesh)

# =====================================================================
# 2. GENERAL WEAK FORM SOLVER
# =====================================================================
def solve_helmholtz(mesh, k):
    """
    Solves the stationary wave (Helmholtz) equation.
    Injects a plane wave on the left boundary at angle 'theta'.
    Applies Robin (Sommerfeld) absorbing boundaries on the other edges.
    """
    # Define the high-order, complex-valued Finite Element Space
    # We set Dirichlet conditions on the 'left' boundary
    base_fes = H1(mesh, order=4, complex=True, dirichlet="left|right")
    fes = Periodic(base_fes)
    print('#DoFs = ', fes.ndof)
    
    # Trial (u) and Test (v) functions
    u, v = fes.TnT()
    
    # --- The Bilinear Form (The physics of the domain and Robin boundaries) ---
    a = BilinearForm(fes, symmetric=True)
    
    # 1. Domain Integral: \int (\nabla u \cdot \nabla v - k^2 u v) dx
    a += (grad(u)*grad(v) - k**2 * u * v) * dx
    
    # 2. Robin Boundary Integral: -ik \int u v ds (on absorbing walls)
    # We apply this to right, top, and bottom to let waves exit.
    a += -1j * k * u * v * ds("right")
    
    # Assemble the matrix
    with TaskManager(): # Uses multi-threading if available
        a.Assemble()
    
    # --- The Linear Form (The source terms) ---
    f = LinearForm(fes)
    f.Assemble() # It's 0 everywhere inside the domain
    
    # --- The Dirichlet Boundary Condition (The injected plane wave) ---
    theta_rad = 0.0
    ky = k * np.sin(theta_rad)
    print('ky = ', ky)
    # Define the exact incoming wave function: e^{i(kx*x + ky*y)}
    # NGSolve uses symbolic coordinate variables x, y, z
    u_in = exp(1j * (ky * y))
    
    # Create the grid function to hold our solution
    gfu = GridFunction(fes)
    # Interpolate the incoming wave strictly onto the 'left' boundary
    gfu.Set(u_in, definedon=mesh.Boundaries("left"))
    
    # --- Solve the Linear System ---
    print("Solving the linear system...")
    # We solve a.mat * u = f.vec. 
    # Because we have Dirichlet conditions, we must modify the right-hand side.
    # f_mod = f - A * u_dirichlet
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    
    # Invert the matrix using a sparse direct solver (UMFPACK)
    # freedofs ignores the nodes locked by the Dirichlet condition
    inv = a.mat.Inverse(freedofs=fes.FreeDofs())# , inverse="umfpack")
    
    # Add the solved free degrees of freedom back to the GridFunction
    gfu.vec.data += inv * res
    Draw(gfu);
    return gfu

# =====================================================================
# 3. VISUALIZATION (Amplitude 2D Map)
# =====================================================================
def plot_amplitude_map(mesh, gfu, Lx, Ly, nx=300, ny=300):
    """
    Extracts the data from the FEM mesh onto a regular numpy grid
    and plots the 2D amplitude map.
    """
    print("Generating 2D amplitude map...")
    
    # Create a regular grid of points
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Evaluate the GridFunction at each point
    # We take the absolute value (Norm) of the complex field
    Z_amp = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            pt = mesh(X[i,j], Y[i,j])
            # Check if point is inside the mesh to avoid errors
            if pt: 
                complex_val = gfu(pt)
                Z_amp[i,j] = complex_val.real
                
    # Plotting using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z_amp, levels=50, cmap='inferno')
    plt.colorbar(label='Wave Amplitude $|u|$')
    plt.title('2D Map of Wave Amplitude')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.axis('equal')
    plt.tight_layout()
    plt.show()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    # --- Physics Parameters ---
    freq_LH = 3.7e9  # LH freq in Hz
    omega_wave = 2 * np.pi * freq_LH  
    c0 = 3.0e8   # speed of light in vacuum in m/s 
    lambda_0 = c0/freq_LH
    print("lambda_0 = ", lambda_0)

    k_wave = 2 * np.pi / lambda_0

    
    # --- Geometry Parameters ---
    Lx = 0.4
    Ly = 0.4
    
    # Rule of thumb for standard FEM: resolution <= wavelength / 10
    # Because we use hp-FEM (order=4), we can be slightly looser, but 
    # keeping it tight ensures a beautiful visualization.
    max_h = lambda_0 / 10.0 
    print('max_h = ', max_h)
    # 1. Create Mesh
    mesh = create_rectangular_mesh(Lx, Ly, max_h)
    # 2. Solve Weak Form
    u_sol = solve_helmholtz(mesh, k_wave)
    
    # 3. Plot Result
    plot_amplitude_map(mesh, u_sol, Lx, Ly)
