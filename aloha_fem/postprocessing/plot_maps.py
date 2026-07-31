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
    real_part = mesh.point_data[f"{component}_real"]
    imag_part = mesh.point_data[f"{component}_imag"]

    if value_type == "real":
        plot_data = real_part
        cmap = "coolwarm"
    elif value_type == "abs":
        # Recombine the complex magnitude rigorously
        plot_data = np.sqrt(real_part**2 + imag_part**2)
        cmap = "magma"
    else:
        raise ValueError("value_type must be 'real' or 'abs'")
        
    vmax = np.percentile(plot_data, 99.5)
    vmin = 0.0 if value_type == "abs" else -vmax
    
    mesh.point_data["Active_Plot"] = plot_data
    
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="Active_Plot",
        cmap=cmap,
        clim=[vmin, vmax],
        show_edges=False,
        scalar_bar_args={"title": f"{value_type.capitalize()}({component}) [V/m]"}
    )
    
    plotter.view_xy()
    plotter.show_axes()
    plotter.show()