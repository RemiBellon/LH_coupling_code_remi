import os
from ngsolve import Mesh, TaskManager, VTKOutput
import netgen.occ as occ
from config.schema import SimulationConfig
from utils.antenna_desc_2D import AntennaGrill
from utils.antenna_desc_2D import build_grill_from_config

class MeshGenerator:
    '''

    '''
    def __init__(self, config):
        self.cfg = config
        self.domain = config.geometry.domain

        if config.simulation.dimension == "2D" and config.geometry.antenna is not None:
            self.antenna_dims_cfg = config.geometry.antenna.dimensions
            self.grill_arr_cfg = config.geometry.antenna.grill_arrangement
            self.has_antenna = True
        else:
            self.has_antenna = False

    def build_occ_geometry(self) -> occ.OCCGeometry:
        '''
        Dynamic Mesh Generation of a 2D antenna.
        - Recover dimension sizes from inputs.yaml file
        - PAM or FAM design
        - Rounded septa
        '''

        # ==== Mesh geometry construction ====
        # == Plasma domain ==
        plasma_rect = occ.MoveTo(0, 0).Rectangle(self.domain.Lx_plasma, self.domain.Lz_plasma).Face()
        print(f'--- Lz_plasma: {self.domain.Lz_plasma:.3e}m ---')
        plasma_rect.name = "plasma"
        plasma_rect.edges.name = "internal"

        # == PMLs domains ==
        radial_pml = occ.MoveTo(self.domain.Lx_plasma, 0).Rectangle(self.domain.Lx_pml, self.domain.Lz_plasma).Face()
        radial_pml.name = "radial_pml"

        toroidal_left_pml = occ.MoveTo(0, -self.domain.Lz_pml).Rectangle(self.domain.Lx_tot, self.domain.Lz_pml).Face()
        toroidal_left_pml.name = "toroidal_left_pml"

        toroidal_right_pml = occ.MoveTo(0, self.domain.Lz_plasma).Rectangle(self.domain.Lx_tot, self.domain.Lz_pml).Face()
        toroidal_right_pml.name = "toroidal_right_pml"

        for p in [radial_pml, toroidal_left_pml, toroidal_right_pml]:
            p.edges.name = "pml_outer"

        domain_faces = [plasma_rect, radial_pml, toroidal_left_pml, toroidal_right_pml]

        # == Antenna structure ==
        if self.has_antenna:
            grill, instructions = build_grill_from_config(self.cfg.geometry.antenna, self.geometry.domain)
            r_septa = self.antenna_dims_cfg.corner_radius

            for inst in instructions:
                # We only mesh the vacuum regions. Metal is the void outside the mesh.
                if inst['type'] in ['wg_active', 'wg_passive']:
                    wg_width = inst['width']
                    wg_depth = inst['depth']
                    # Shift toroidal placement by the left wall thickness to center it
                    z_start = inst['z_start']
                    wg_base = occ.MoveTo(-wg_depth, z_start).Rectangle(wg_depth, wg_width).Face()

                    # Left rounded corner
                    corner_left = occ.MoveTo(-r_septa, z_start - r_septa).Rectangle(r_septa, r_septa).Face()
                    disk_left = occ.MoveTo(-r_septa, z_start - r_septa).Circle(r_septa).Face()
                    fillet_left = corner_left - disk_left

                    # Right rounded corner
                    corner_right = occ.MoveTo(-r_septa, z_start + wg_width).Rectangle(r_septa, r_septa).Face()
                    disk_right = occ.MoveTo(-r_septa, z_start + wg_width + r_septa).Circle(r_septa).Face()
                    fillet_right = corner_right - disk_right

                    wg = wg_base + fillet_left + fillet_right


                    if inst['type'] == 'wg_active':
                        wg.name = "vacuum_active"
                        wg.edges.name = "metal"
                        wg.edges.Min(occ.X).name = "wg_inlet"
                    else:
                        wg.name = "vacuum_passive"
                        wg.edges.name = "metal"

                    domain_faces.append(wg)
        # ==== Geometry Generation ====
        domain_tot = occ.Glue(domain_faces)

        if not self.has_antenna:
            for e in domain_tot.edges:
                if abs(e.center[0]) < 1e-6:
                    # Target the physical source aperture, leaving the rest as PEC metal walls
                    if self.domain.Lz_wall - 1e-5 <= e.center[1] <= (self.domain.Lz_plasma + self.domain.Lz_wall) + 1e-5:
                        e.name = "wg_inlet"
                    else:
                        e.name = "metal"

        return occ.OCCGeometry(domain_tot, dim=2)

    def generate_base_mesh(self, save_mesh_file: bool) -> Mesh:
        geom = self.build_occ_geometry()
        with TaskManager():
            mesh = Mesh(geom.GenerateMesh())

        if save_mesh_file == True:
            output_folder = 'mesh_plot_results'
            os.makedirs(output_folder, exist_ok=True)

            region_colors = mesh.MaterialCF({
                "plasma": 1,
                "vacuum_active": 2,
                "vacuum_passive": 3,
                "radial_pml": 4,
                "toroidal_left_pml": 5,
                "toroidal_right_pml": 5}, default=0)

            output_path = os.path.join(output_folder, "antenna_mesh.vtu")

            vtk = VTKOutput(
                ma=mesh,
                coefs=[region_colors],
                names=["Region_ID"],
                filename=output_path,
                subdivision=2
            )
            vtk.Do()
            print(f'The mesh .vtu file is save in {output_path}.')
        return mesh
