import pyvista as pv
import numpy as np

def plot_2D_wave_map(vtu_filepath: str, component="E_toroidal", value_type="real"):
    """
    Renders the exact unstructured FEM mesh data for 2D spatial fields.
    Bypasses array interpolation to maintain strict mathematical fidelity.
    """
    # Load the exact finite element mesh and solutions
    mesh = pv.read(vtu_filepath)
    
    # The fields are exported as complex numbers in the VTU
    field_data = mesh.point_data[component]
    
    if value_type == "real":
        plot_data = np.real(field_data)
        cmap = "coolwarm"
    elif value_type == "abs":
        plot_data = np.abs(field_data)
        cmap = "magma"
    else:
        raise ValueError("value_type must be 'real' or 'abs'")
        
    # Prevent extreme metal corner singularities from flattening the color scale[cite: 18]
    vmax = np.percentile(plot_data, 99.5)
    vmin = 0.0 if value_type == "abs" else -vmax
    
    # Assign the processed scalar back to the mesh for plotting
    mesh.point_data["Active_Plot"] = plot_data
    
    # Setup the PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Active_Plot",
        cmap=cmap,
        clim=[vmin, vmax],
        show_edges=False,
        scalar_bar_args={"title": f"{value_type.capitalize()}({component}) [V/m]"}
    )
    
    # Adjust camera to look perfectly at the 2D plane (X-Y in PyVista mapping)
    plotter.view_xy()
    plotter.show_axes()
    plotter.show()