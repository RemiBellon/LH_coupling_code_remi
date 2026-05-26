import sys
from pathlib import Path
from ngsolve import *
import netgen.gui       # Initializes the window


if len(sys.argv) < 2:           # sys.argv[0] is the script name itself. sys.argv[1] is the first argument.
    print("Error: No mesh file provided.")
    print("Usage: netgen mesh_visualizer.py <path_to_mesh.vol>")
    sys.exit(1)

mesh_path = Path(sys.argv[2])   # Extract the file path passed in the terminal
print(f'mesh path = {mesh_path}')

if not mesh_path.exists():      # Extract the file path passed in the terminal
    print(f"Error: The file '{mesh_path}' does not exist.")
    sys.exit(1)

print(f"Loading mesh from: {mesh_path}")
mesh = Mesh(str(mesh_path)) # Convert path to string for NGSolve

print(f"Materials (3D Volumes) found: {mesh.GetMaterials()}")
print(f"Boundaries (2D Faces) found: {mesh.GetBoundaries()}")
print(f"Number of 3D volume elements (tetrahedra): {mesh.ne}")

# CREATE THE COLOR MAP: Each region is associated to a number
domain_values = {"plasma": 1, "plasma_region":1, "pml": 2, "pml_region":2}
raw_color_map = mesh.MaterialCF(domain_values, default=-1)

fes_colors = L2(mesh, order=0)
gf_colors = GridFunction(fes_colors)
gf_colors.Set(raw_color_map)
Draw(gf_colors, mesh, "Domain_Regions")

