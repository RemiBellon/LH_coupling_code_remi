import math
import netgen.occ as occ
from ngsolve import Mesh, TaskManager
from physics.waveguide import WaveguidePhysics  

class MeshFactory:
    @staticmethod
    def build(config, shortest_lambda: float) -> Mesh:
        dim = config.simulation.dimension
        if dim == "1D":
            return MeshBuilder1D(config, shortest_lambda).build_mesh()
        elif dim == "2D":
            return MeshBuilder2D(config, shortest_lambda).build_mesh()
        elif dim == "3D":
            return MeshBuilder3D(config, shortest_lambda).build_mesh()
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

class MeshBuilder1D:
    def __init__(self, config, shortest_lambda: float):
        """
        Initializes 1D geometry builder.
        Relies on radial dimensions; toroidal/poloidal and 
        multijunction antenna topologies are abstracted away.
        """
        self.config = config.geometry
        self.dim = config.simulation.dimension
        
        # In 1D, we extract radial spatial dimensions only
        self.Lx_plasma = self.config.domain.Lx_plasma
        self.Lx_pml = self.config.domain.Lx_pml
        
        # Global mesh refinement variables
        self.base_maxh = shortest_lambda / self.config.mesh.ppw_medium
        self.pml_maxh = shortest_lambda / self.config.mesh.ppw_pml
        self.grading = self.config.mesh.grading

    def build_mesh(self) -> Mesh:
        """
        Constructs a 1-dimensional OCC geometry using colinear segments.
        Tags vertices for Dirichlet/Robin boundary conditions.
        """
        edges_to_glue = []
        
        # ---------------------------------------------------------
        # 1. Build Physical Plasma Domain (0 < x < Lx_plasma)
        # ---------------------------------------------------------
        p0 = occ.Pnt(0, 0, 0)
        p1 = occ.Pnt(self.Lx_plasma, 0, 0)
        
        plasma_segment = occ.Segment(p0, p1)
        plasma_segment.name = "plasma_bulk"
        plasma_segment.maxh = self.base_maxh
        edges_to_glue.append(plasma_segment)
        
        # ---------------------------------------------------------
        # 2. Build Radial PML (Lx_plasma < x < Lx_plasma + Lx_pml)
        # ---------------------------------------------------------
        # A 1D simulation respects use_radial, ignoring toroidal/poloidal.
        if self.config.pml.use_radial and self.Lx_pml > 0:
            p2 = occ.Pnt(self.Lx_plasma + self.Lx_pml, 0, 0)
            pml_segment = occ.Segment(p1, p2)
            pml_segment.name = "pml_rad"
            pml_segment.maxh = self.pml_maxh
            edges_to_glue.append(pml_segment)
            
        # ---------------------------------------------------------
        # 3. Topology Glue
        # ---------------------------------------------------------
        # Fusing colinear segments ensures the node at Lx_plasma is shared
        domain = occ.Glue(edges_to_glue)
        
        # ---------------------------------------------------------
        # 4. Strict Tagging for 1D Boundaries (Vertices)
        # ---------------------------------------------------------
        # Calculate absolute domain length
        Lx_tot = self.Lx_plasma + (self.Lx_pml if self.config.pml.use_radial else 0.0)
        
        # In a 1D FEM space, boundaries are applied to points, not edges.
        for vertex in domain.vertices:
            # x = 0 (The abstracted antenna aperture where the n_parallel spectrum is injected)
            if abs(vertex.point[0] - 0.0) < 1e-6:
                vertex.name = "left_source"
                
            # x = L_total (The terminating PEC or PML boundary)
            elif abs(vertex.point[0] - Lx_tot) < 1e-6:
                vertex.name = "right_wall_pec"
                
        # ---------------------------------------------------------
        # 5. 1D Mesh Generation
        # ---------------------------------------------------------
        # Declare dim=1 to prevent OCC from attempting to build 2D surfaces
        geo = occ.OCCGeometry(domain, dim=1)
        
        with TaskManager():
            # Apply standard grading logic (growth limits) along the line
            ng_mesh = geo.GenerateMesh(grading=self.grading)
            mesh = Mesh(ng_mesh)
            
        return mesh

    
class MeshBuilder2D:
    def __init__(self, config, shortest_lambda: float):
        self.config = config
        self.sim_config = config.simulation
        self.geom_config = config.geometry
        self.dim = config.simulation.dimension

        self.Lx_plasma = self.geom_config.domain.Lx_plasma
        self.Lx_pml = self.geom_config.domain.Lx_pml
        self.Lz_source = self.geom_config.domain.Lz_plasma
        self.Lz_wall = self.geom_config.domain.Lz_wall
        self.Lz_pml = self.geom_config.domain.Lz_pml

        self.base_maxh = shortest_lambda / self.geom_config.mesh.ppw_medium
        self.pml_maxh = shortest_lambda / self.geom_config.mesh.ppw_pml
        self.grading = self.geom_config.mesh.grading
        self.fillet_ratio = 0.25

    def _build_filleted_septum(self, max_depth: float, z_start: float, z_end: float) -> occ.TopoDS_Shape:
        w_sep = z_end - z_start
        r = w_sep * self.fillet_ratio
        r = min(max(r, 1e-6), w_sep / 2.0)

        p_bot_left = occ.Pnt(-max_depth, z_start, 0.0)
        p_bot_right = occ.Pnt(-r, z_start, 0.0)
        p_front_bot = occ.Pnt(0.0, z_start + r, 0.0)
        p_front_top = occ.Pnt(0.0, z_end - r, 0.0)
        p_top_right = occ.Pnt(-r, z_end, 0.0)
        p_top_left = occ.Pnt(-max_depth, z_end, 0.0)

        offset = r * (1.0 - 1.0 / math.sqrt(2.0))
        arc1_mid = occ.Pnt(-offset, z_start + offset, 0.0)
        arc2_mid = occ.Pnt(-offset, z_end - offset, 0.0)

        segments = [occ.Segment(p_bot_left, p_bot_right)]
        segments.append(occ.ArcOfCircle(p_bot_right, arc1_mid, p_front_bot))

        if r < (w_sep / 2.0) - 1e-6:
            segments.append(occ.Segment(p_front_bot, p_front_top))

        segments.append(occ.ArcOfCircle(p_front_top, arc2_mid, p_top_right))
        segments.append(occ.Segment(p_top_right, p_top_left))
        segments.append(occ.Segment(p_top_left, p_bot_left))

        wire = occ.Wire(segments)
        return occ.Face(wire)

    def build_mesh(self) -> Mesh:
        """
        Constructs the complete 2D geometry using OpenCASCADE (OCC) boolean operations.
        Returns a fully tagged and refined NGSolve Mesh object.
        """
        wg_physics = WaveguidePhysics(self.config)
        wg_sequence = wg_physics.wg_sequence
        self.max_wg_length = wg_physics.max_wg_length
        faces_to_glue = []

        # =================================================================
        # STEP 1: Construct the Base Fluid Domains (Vacuum + Plasma)
        # =================================================================
        # We start by creating solid blocks of "fluid" (areas where waves propagate).
        if wg_sequence:
            # wg_zone: A solid block from the back of the deepest waveguide up to x=0.
            # Starts at z = Lz_wall and spans Lz_source.
            wg_zone = occ.MoveTo(-self.max_wg_length, self.Lz_wall).Rectangle(self.max_wg_length, self.Lz_source).Face()
            # plasma_zone: A solid block from x=0 to Lx_plasma.
            plasma_zone = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_source + 2.0 * self.Lz_wall).Face()
            fluid_core = wg_zone + plasma_zone
        else:
            # If no antenna, the domain is strictly the plasma bulk.
            fluid_core = occ.MoveTo(0, 0).Rectangle(self.Lx_plasma, self.Lz_source + 2.0 * self.Lz_wall).Face()

        # =================================================================
        # STEP 2: Carve out the Metal Structures (Boolean Subtractions)
        # =================================================================
        if wg_sequence:
            for i, wg in enumerate(wg_sequence):
                # A. Short-circuited Passives: Subtract the metal block behind shallow waveguides
                if wg["length"] < self.max_wg_length:
                    metal_block = occ.MoveTo(-self.max_wg_length, wg["z_start"]).Rectangle(
                        self.max_wg_length - wg["length"], wg["z_end"] - wg["z_start"]).Face()
                    fluid_core = fluid_core - metal_block

                # B. Metal Septa: Subtract the filleted walls between adjacent waveguides
                if i < len(wg_sequence) - 1:
                    z_septa_start = wg["z_end"]
                    z_septa_end = wg_sequence[i+1]["z_start"]
                    septum = self._build_filleted_septum(self.max_wg_length, z_septa_start, z_septa_end)
                    fluid_core = fluid_core - septum
            
            # C. Right Metal Flank: If the antenna array is shorter than the defined Lz_source,
            #    the remaining space on the right must be carved out as solid metal.
            last_z = wg_sequence[-1]["z_end"]
            z_wg_zone_end = self.Lz_wall + self.Lz_source
            if z_wg_zone_end > last_z + 1e-6:
                right_flank = occ.MoveTo(-self.max_wg_length, last_z).Rectangle(
                    self.max_wg_length, z_wg_zone_end - last_z).Face()
                fluid_core = fluid_core - right_flank

        faces_to_glue.append(fluid_core)

        # =================================================================
        # STEP 3: Construct the Perfectly Matched Layers (PMLs)
        # =================================================================
        z_top_interface = self.Lz_source + 2.0 * self.Lz_wall
        
        # Radial PML (Right side)
        if self.geom_config.pml.use_radial and self.Lx_pml > 0:
            pml_rad = occ.MoveTo(self.Lx_plasma, 0).Rectangle(self.Lx_pml, z_top_interface).Face()
            faces_to_glue.append(pml_rad)

        # Toroidal PMLs (Top and Bottom)
        if self.geom_config.pml.use_toroidal and self.Lz_pml > 0:
            pml_tor_bot = occ.MoveTo(0, -self.Lz_pml).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            pml_tor_top = occ.MoveTo(0, z_top_interface).Rectangle(self.Lx_plasma, self.Lz_pml).Face()
            faces_to_glue.extend([pml_tor_bot, pml_tor_top])

            # Corner PMLs (Required for numerical stability where Toroidal and Radial PMLs meet)
            if self.geom_config.pml.use_radial and self.Lx_pml > 0:
                pml_corner_bot = occ.MoveTo(self.Lx_plasma, -self.Lz_pml).Rectangle(self.Lx_pml, self.Lz_pml).Face()
                pml_corner_top = occ.MoveTo(self.Lx_plasma, z_top_interface).Rectangle(self.Lx_pml, self.Lz_pml).Face()
                faces_to_glue.extend([pml_corner_bot, pml_corner_top])

        # =================================================================
        # STEP 4: Topological Glue
        # =================================================================
        # Glue fuses overlapping faces and resolves internal boundaries (e.g., waveguide apertures)
        domain = occ.Glue(faces_to_glue)
        
        # Determine absolute domain limits for outer boundary tagging
        z_min_domain = -self.Lz_pml if self.geom_config.pml.use_toroidal else 0.0
        z_max_domain = z_top_interface + (self.Lz_pml if self.geom_config.pml.use_toroidal else 0.0)

        # =================================================================
        # STEP 5: Periodic Boundary Mappings
        # =================================================================
        # RIGOROUS FIX: Periodic maps MUST strictly apply to the absolute outer edges of the mesh.
        if self.sim_config.boundary_toroidal == "periodic":
            bottom_edges = [e for e in domain.edges if abs(e.center[1] - z_min_domain) < 1e-6]
            top_edges = [e for e in domain.edges if abs(e.center[1] - z_max_domain) < 1e-6]
            for bot, top in zip(bottom_edges, top_edges):
                bot.Identify(top, "periodic_toroidal", occ.IdentificationType.PERIODIC)

        # =================================================================
        # STEP 6: Strict Edge Tagging for Boundary Conditions
        # =================================================================
        for edge in domain.edges:
            c = edge.center

            # A. Antenna Waveguide Back-Walls (Robin Boundary Condition Injection)
            if wg_sequence:
                for wg in wg_sequence:
                    if abs(c[0] - (-wg["length"])) < 1e-6 and (wg["z_start"] - 1e-5 < c[1] < wg["z_end"] + 1e-5):
                        edge.name = "bottom_source_active" if wg["type"] == "active" else "bottom_source_passive"

            # B. Antenna Metal Walls (Perfect Electric Conductor - PEC)
            # This captures all waveguide side-walls and the front face of the septa at x=0
            if c[0] < 1e-6 and c[0] > -self.max_wg_length - 1e-6:
                if edge.name is None or "bottom_source" not in edge.name:
                    edge.name = "bottom_wall_pec"
                
                # Apply aggressive mesh refinement only on the sharp filleted arc segments
                if self.geom_config.antenna and c[0] > - (self.geom_config.antenna.dimensions.septa_width * self.fillet_ratio) - 1e-6:
                    edge.maxh = self.base_maxh / 8.0
                    
            # C. Outer Boundaries (PEC encapsulation behind the PMLs)
            # Radial outer wall
            if abs(c[0] - (self.Lx_plasma + self.Lx_pml)) < 1e-6:
                edge.name = "top_wall_pec"
                edge.maxh = self.pml_maxh
                
            # Toroidal Left (Bottom) Wall
            if abs(c[1] - z_min_domain) < 1e-6:
                edge.name = "left_wall_pec"
                edge.maxh = self.pml_maxh
                
            # Toroidal Right (Top) Wall
            if abs(c[1] - z_max_domain) < 1e-6:
                edge.name = "right_wall_pec"
                edge.maxh = self.pml_maxh

        # =================================================================
        # STEP 7: Mesh Generation
        # =================================================================
        geo = occ.OCCGeometry(domain, dim=2)
        with TaskManager():
            mesh = Mesh(geo.GenerateMesh(maxh=self.base_maxh, grading=self.grading))
            
        return mesh

class MeshBuilder3D:
    def __init__(self, config, shortest_lambda: float):
        """
        Initializes the rigorous 3D geometry builder.
        Maps the full FAM/PAM antenna topology across multiple rows and columns.
        """
        self.config = config
        self.sim_config = config.simulation
        self.geom_config = config.geometry
        self.dim = config.simulation.dimension
        
        # Domain Mapping
        self.Lx_plasma = self.geom_config.domain.Lx_plasma
        self.Lx_pml = self.geom_config.domain.Lx_pml
        
        self.Ly_plasma = self.geom_config.domain.Ly_plasma
        self.Ly_wall = self.geom_config.domain.Ly_wall
        self.Ly_pml = self.geom_config.domain.Ly_pml
        
        self.Lz_source = self.geom_config.domain.Lz_plasma
        self.Lz_wall = self.geom_config.domain.Lz_wall
        self.Lz_pml = self.geom_config.domain.Lz_pml

        self.base_maxh = shortest_lambda / self.geom_config.mesh.ppw_medium
        self.pml_maxh = shortest_lambda / self.geom_config.mesh.ppw_pml
        self.grading = self.geom_config.mesh.grading
        
        self.fillet_ratio = 0.25
        self.sim_config = config.simulation

    def _build_filleted_profile(self, max_depth: float, v_start: float, v_end: float, plane: str, offset: float) -> occ.TopoDS_Shape:
        """
        Constructs a 2D exact filleted profile on a specific plane.
        - If plane="XZ": Profile separates waveguides toroidally (v maps to z). Extruded along Y.
        - If plane="XY": Profile separates rows poloidally (v maps to y). Extruded along Z.
        """
        w_sep = v_end - v_start
        r = w_sep * self.fillet_ratio
        r = min(max(r, 1e-6), w_sep / 2.0)

        def make_pnt(x, v):
            if plane == "XZ": return occ.Pnt(x, offset, v)
            elif plane == "XY": return occ.Pnt(x, v, offset)
            else: raise ValueError("Invalid plane")

        p_bot_left = make_pnt(-max_depth, v_start)
        p_bot_right = make_pnt(-r, v_start)
        p_front_bot = make_pnt(0.0, v_start + r)
        p_front_top = make_pnt(0.0, v_end - r)
        p_top_right = make_pnt(-r, v_end)
        p_top_left = make_pnt(-max_depth, v_end)

        offset_arc = r * (1.0 - 1.0 / math.sqrt(2.0))
        arc1_mid = make_pnt(-offset_arc, v_start + offset_arc)
        arc2_mid = make_pnt(-offset_arc, v_end - offset_arc)

        segments = [occ.Segment(p_bot_left, p_bot_right)]
        segments.append(occ.ArcOfCircle(p_bot_right, arc1_mid, p_front_bot))
        
        if r < (w_sep / 2.0) - 1e-6:
            segments.append(occ.Segment(p_front_bot, p_front_top))
            
        segments.append(occ.ArcOfCircle(p_front_top, arc2_mid, p_top_right))
        segments.append(occ.Segment(p_top_right, p_top_left))
        segments.append(occ.Segment(p_top_left, p_bot_left))

        wire = occ.Wire(segments)
        return occ.Face(wire)

    def build_mesh(self) -> Mesh:
        wg_physics = WaveguidePhysics(self.config)
        wg_sequence = wg_physics.wg_sequence
        self.max_wg_length = wg_physics.max_wg_length
        ant_cfg = self.geom_config.antenna
        
        # Base bounds
        Ly_tot = self.Ly_plasma + 2.0 * self.Ly_wall
        Lz_tot = self.Lz_source + 2.0 * self.Lz_wall

        # 1. Unified 3D Bounding Box (Vacuum Core)
        if wg_sequence:
            wg_zone = occ.Box(occ.Pnt(-self.max_wg_length, 0, 0), occ.Pnt(0, Ly_tot, Lz_tot))
            plasma_zone = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(self.Lx_plasma, Ly_tot, Lz_tot))
            fluid_core = wg_zone + plasma_zone
        else:
            fluid_core = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(self.Lx_plasma, Ly_tot, Lz_tot))

        # 2. Subtractions for 3D Multijunction Antenna
        if wg_sequence:
            wg_height = ant_cfg.dimensions.wg_height
            row_spacing = ant_cfg.grill.row_spacing
            num_rows = ant_cfg.grill.num_rows
            
            for r in range(num_rows):
                y_start = self.Ly_wall + r * (wg_height + row_spacing)
                y_end = y_start + wg_height
                
                # Poloidal Septa (Horizontal plates between rows)
                if r < num_rows - 1:
                    profile_xy = self._build_filleted_profile(
                        self.max_wg_length, y_end, y_end + row_spacing, plane="XY", offset=self.Lz_wall
                    )
                    # Extrude along Z across the entire width of the module
                    horiz_septum = occ.Extrude(profile_xy, occ.Vec(0, 0, self.array_z_width))
                    fluid_core = fluid_core - horiz_septum

                # Toroidal Septa (Vertical plates between waveguides in a row)
                for i, wg in enumerate(wg_sequence):
                    # Block the back of passive waveguides
                    if wg["length"] < self.max_wg_length:
                        metal_block = occ.Box(
                            occ.Pnt(-self.max_wg_length, y_start, wg["z_start"]),
                            occ.Pnt(-wg["length"], y_end, wg["z_end"])
                        )
                        fluid_core = fluid_core - metal_block

                    # Septum separating this WG from the next
                    if i < len(wg_sequence) - 1:
                        z_sep_start = wg["z_end"]
                        z_sep_end = wg_sequence[i+1]["z_start"]
                        
                        profile_xz = self._build_filleted_profile(
                            self.max_wg_length, z_sep_start, z_sep_end, plane="XZ", offset=y_start
                        )
                        # Extrude along Y for the height of the waveguide
                        vert_septum = occ.Extrude(profile_xz, occ.Vec(0, wg_height, 0))
                        fluid_core = fluid_core - vert_septum
                        
            # Subtract massive metal blocks flanking the entire array 
            # to carve out the exact array shape from the unified wg_zone
            left_flank = occ.Box(occ.Pnt(-self.max_wg_length, 0, 0), occ.Pnt(0, Ly_tot, self.Lz_wall))
            right_flank = occ.Box(occ.Pnt(-self.max_wg_length, 0, self.Lz_wall + self.array_z_width), occ.Pnt(0, Ly_tot, Lz_tot))
            bot_flank = occ.Box(occ.Pnt(-self.max_wg_length, 0, self.Lz_wall), occ.Pnt(0, self.Ly_wall, self.Lz_wall + self.array_z_width))
            
            top_y_array = self.Ly_wall + num_rows * wg_height + (num_rows - 1) * row_spacing
            top_flank = occ.Box(occ.Pnt(-self.max_wg_length, top_y_array, self.Lz_wall), occ.Pnt(0, Ly_tot, self.Lz_wall + self.array_z_width))
            
            fluid_core = fluid_core - left_flank - right_flank - bot_flank - top_flank

        solids_to_glue = [fluid_core]

        # 3. 3D PML Domains (Strictly Orthogonal Boxes)
        if self.geom_config.pml.use_radial and self.Lx_pml > 0:
            solids_to_glue.append(occ.Box(occ.Pnt(self.Lx_plasma, 0, 0), occ.Pnt(self.Lx_plasma + self.Lx_pml, Ly_tot, Lz_tot)))
            
        if self.geom_config.pml.use_toroidal and self.Lz_pml > 0:
            solids_to_glue.append(occ.Box(occ.Pnt(0, 0, -self.Lz_pml), occ.Pnt(self.Lx_plasma, Ly_tot, 0)))
            solids_to_glue.append(occ.Box(occ.Pnt(0, 0, Lz_tot), occ.Pnt(self.Lx_plasma, Ly_tot, Lz_tot + self.Lz_pml)))
            
        if self.geom_config.pml.use_poloidal and self.Ly_pml > 0:
            solids_to_glue.append(occ.Box(occ.Pnt(0, -self.Ly_pml, 0), occ.Pnt(self.Lx_plasma, 0, Lz_tot)))
            solids_to_glue.append(occ.Box(occ.Pnt(0, Ly_tot, 0), occ.Pnt(self.Lx_plasma, Ly_tot + self.Ly_pml, Lz_tot)))

        domain = occ.Glue(solids_to_glue)
        
        # 4. Periodic Boundary Mappings (Before Meshing)
        if self.sim_config.boundary_toroidal == "periodic":
            bot_z_faces = [f for f in domain.faces if abs(f.center[2] - 0.0) < 1e-6]
            top_z_faces = [f for f in domain.faces if abs(f.center[2] - Lz_tot) < 1e-6]
            for bot, top in zip(bot_z_faces, top_z_faces): 
                bot.Identify(top, "periodic_toroidal", occ.IdentificationType.PERIODIC)
                
        if self.sim_config.boundary_poloidal == "periodic":
            bot_y_faces = [f for f in domain.faces if abs(f.center[1] - 0.0) < 1e-6]
            top_y_faces = [f for f in domain.faces if abs(f.center[1] - Ly_tot) < 1e-6]
            for bot, top in zip(bot_y_faces, top_y_faces): 
                bot.Identify(top, "periodic_poloidal", occ.IdentificationType.PERIODIC)
        # 5. 3D Face Tagging & Strict Refinement
        for face in domain.faces:
            c = face.center
            
            # Tag Active vs Passive Sources at the back walls
            if wg_sequence and c[0] < -1e-6:
                for r in range(ant_cfg.grill.num_rows):
                    y_s = self.Ly_wall + r * (ant_cfg.dimensions.wg_height + ant_cfg.grill.row_spacing)
                    y_e = y_s + ant_cfg.dimensions.wg_height
                    
                    if y_s - 1e-5 < c[1] < y_e + 1e-5:
                        for wg in wg_sequence:
                            if abs(c[0] - (-wg["length"])) < 1e-6 and (wg["z_start"] - 1e-5 < c[2] < wg["z_end"] + 1e-5):
                                face.name = "back_source_active" if wg["type"] == "active" else "back_source_passive"

            # Tag all remaining interior metal boundaries
            if c[0] < 1e-6 and "source" not in face.name:
                face.name = "metal_wall_pec"
                
                # Apply H-refinement to the complex filleted surfaces at the aperture
                if ant_cfg and c[0] > - (ant_cfg.dimensions.septa_width * self.fillet_ratio) - 1e-6:
                    face.maxh = self.base_maxh / 8.0

        geo = occ.OCCGeometry(domain, dim=3)
        with TaskManager():
            mesh = Mesh(geo.GenerateMesh(maxh=self.base_maxh, grading=self.grading))
            
        return mesh