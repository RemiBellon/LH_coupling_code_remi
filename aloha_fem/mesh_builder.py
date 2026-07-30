import math
import netgen.occ as occ
from ngsolve import Mesh, TaskManager

class MeshBuilder2D:
    def __init__(self, config, shortest_lambda: float):
        self.config = config.geometry
        self.dim = config.simulation.dimension

        self.Lx_plasma = self.config.domain.Lx_plasma
        self.Lx_pml = self.config.domain.Lx_pml
        self.Lz_source = self.config.domain.Lz_plasma
        self.Lz_wall = self.config.domain.Lz_wall
        self.Lz_pml = self.config.domain.Lz_pml

        self.base_maxh = shortest_lambda / self.config.mesh.ppw_medium
        self.pml_maxh = shortest_lambda / self.config.mesh.ppw_pml
        self.grading = self.config.mesh.grading

        # New: Tunable fillet radius ratio (e.g., 0.25 = w_sep/4)
        # We can expose this to the yaml config later if desired.
        self.fillet_ratio = 0.25

    def _build_filleted_septum(self, max_depth: float, z_start: float, z_end: float) -> occ.TopoDS_Shape:
        """
        Constructs the exact filleted metal septum in the negative x domain.
        r can vary up to (z_end - z_start)/2 (which recovers the semi-circle).
        """
        w_sep = z_end - z_start
        r = w_sep * self.fillet_ratio

        # Safety clamp to prevent unphysical geometries
        r = min(max(r, 1e-6), w_sep / 2.0)

        # Define corner points
        p_bot_left = occ.Pnt(-max_depth, z_start)
        p_bot_right = occ.Pnt(-r, z_start)
        p_front_bot = occ.Pnt(0.0, z_start + r)
        p_front_top = occ.Pnt(0.0, z_end - r)
        p_top_right = occ.Pnt(-r, z_end)
        p_top_left = occ.Pnt(-max_depth, z_end)

        # Calculate intermediate points exactly on the 45-degree mark of the arcs
        # This is required by OCC's ArcOfCircle(Start_Point, Mid_Point, End_Point)
        offset = r * (1.0 - 1.0 / math.sqrt(2.0))
        arc1_mid = occ.Pnt(-offset, z_start + offset)
        arc2_mid = occ.Pnt(-offset, z_end - offset)

        # Construct the closed wire
        segments = [occ.Segment(p_bot_left, p_bot_right)]

        # Bottom Fillet
        segments.append(occ.ArcOfCircle(p_bot_right, arc1_mid, p_front_bot))

        # Flat front face (only added if r < w_sep/2)
        if r < (w_sep / 2.0) - 1e-6:
            segments.append(occ.Segment(p_front_bot, p_front_top))

        # Top Fillet
        segments.append(occ.ArcOfCircle(p_front_top, arc2_mid, p_top_right))

        # Complete the loop
        segments.append(occ.Segment(p_top_right, p_top_left))
        segments.append(occ.Segment(p_top_left, p_bot_left))

        wire = occ.Wire(segments)
        return occ.Face(wire)

    def _generate_waveguide_sequence(self):
        """Translates PAM/FAM configurations into a strict spatial sequence."""
        sequence = []
        ant_cfg = self.config.antenna
        if not ant_cfg: return sequence

        wg_width = ant_cfg.dimensions.wg_width
        septa_width = ant_cfg.dimensions.septa_width

        self.max_wg_length = max(
            ant_cfg.dimensions.wg_length_active,
            ant_cfg.dimensions.wg_length_passive
        )

        current_z = self.Lz_wall
        for mod_idx in range(ant_cfg.arrangement.num_modules):
            num_active = ant_cfg.arrangement.active_waveguides_per_module[mod_idx]

            if ant_cfg.topology == "FAM":
                mod_sequence = ["active"] * num_active
            elif ant_cfg.topology == "PAM":
                mod_sequence = []
                for _ in range(num_active):
                    mod_sequence.extend(["passive", "active"])
                mod_sequence.append("passive")
            else:
                raise ValueError(f"Unknown topology: {ant_cfg.topology}")

            for wg_type in mod_sequence:
                length = ant_cfg.dimensions.wg_length_active if wg_type == "active" else ant_cfg.dimensions.wg_length_passive
                sequence.append({
                    "type": wg_type,
                    "length": length,
                    "z_start": current_z,
                    "z_end": current_z + wg_width
                })
                current_z += wg_width + septa_width

        return sequence

    def build_mesh(self) -> Mesh:
        wg_sequence = self._generate_waveguide_sequence()
        faces_to_glue = []

        # 1. Unified Bounding Box (Vacuum Core)
        if wg_sequence:
            wg_zone = occ.MoveTo(-self.max_wg_length, self.Lz_wall).Rectangle(self.max_wg_length, self.Lz_source).Face()
            plasma_zone = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_source + 2.0 * self.Lz_wall).Face()
            fluid_core = wg_zone + plasma_zone
        else:
            fluid_core = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_source + 2.0 * self.Lz_wall).Face()

        # 2. Boolean Subtractions (Metal Voids and Septa)
        if wg_sequence:
            for i, wg in enumerate(wg_sequence):
                # Subtract metal block behind shallower waveguides
                if wg["length"] < self.max_wg_length:
                    metal_block = occ.MoveTo(-self.max_wg_length, wg["z_start"]).Rectangle(
                        self.max_wg_length - wg["length"],
                        wg["z_end"] - wg["z_start"]
                    ).Face()
                    fluid_core = fluid_core - metal_block

                # Subtract the filleted septum separating this WG from the next
                if i < len(wg_sequence) - 1:
                    z_septa_start = wg["z_end"]
                    z_septa_end = wg_sequence[i+1]["z_start"]
                    septum = self._build_filleted_septum(self.max_wg_length, z_septa_start, z_septa_end)
                    fluid_core = fluid_core - septum

        faces_to_glue.append(fluid_core)

        # ---------------------------------------------------------
        # Build PML Domains (Standard Rectangles)
        # ---------------------------------------------------------
        if self.config.pml.use_radial and self.Lx_pml > 0:
            pml_rad = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, self.Lz_source + 2.0 * self.Lz_wall).Face()
            faces_to_glue.append(pml_rad)

        if self.config.pml.use_toroidal and self.Lz_pml > 0:
            z_top_interface = self.Lz_source + 2.0 * self.Lz_wall
            pml_tor_bot = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            pml_tor_top = occ.MoveTo(0, z_top_interface).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            faces_to_glue.extend([pml_tor_bot, pml_tor_top])

            if self.config.pml.use_radial and self.Lx_pml > 0:
                pml_corner_bot = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
                pml_corner_top = occ.MoveTo(self.Lx_plasma, z_top_interface).Rectangle(self.Lx_pml, self.Lz_pml).Face()
                faces_to_glue.extend([pml_corner_bot, pml_corner_top])

        domain = occ.Glue(faces_to_glue)

        # 4. Edges Tagging
        for edge in domain.edges:
            c = edge.center

            if wg_sequence:
                for wg in wg_sequence:
                    if abs(c[0] - (-wg["length"])) < 1e-6 and (wg["z_start"] - 1e-5 < c[1] < wg["z_end"] + 1e-5):
                        edge.name = "bottom_source_active" if wg["type"] == "active" else "bottom_source_passive"

            # Tag the complex metal aperture
            if c[0] < 1e-6 and c[0] > -self.max_wg_length - 1e-6:
                if "bottom_source" not in edge.name:
                    edge.name = "bottom_wall_pec"

                # Apply aggressive H-refinement only on the filleted arc segments
                if self.config.antenna and c[0] > - (self.config.antenna.dimensions.septa_width * self.fillet_ratio) - 1e-6:
                    edge.maxh = self.base_maxh / 8.0

        geo = occ.OCCGeometry(domain, dim=2)
        with TaskManager():
            mesh = Mesh(geo.GenerateMesh(maxh=self.base_maxh, grading=self.grading))

        return mesh