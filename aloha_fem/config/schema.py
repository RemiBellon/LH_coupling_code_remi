import math
from typing import Literal, List, Tuple, Union, Optional
from pydantic import BaseModel, Field, model_validator, computed_field

# ==========================================
# 1. PIECEWISE PROFILE BUILDER
# ==========================================
class PiecewiseProfile(BaseModel):
    type: Literal["piecewise"] = "piecewise"
    points: List[Tuple[Union[float, Literal["Lx_plasma"]], float]] = Field(..., description="List of (x, value) pairs")
    segments: List[Literal["linear", "exponential", "constant"]]

    @model_validator(mode='after')
    def validate_piecewise_geometry(self):
        num_points = len(self.points)
        num_segments = len(self.segments)
        # Ensure the number of segment match the number of points -1: radial density profile correclty designed
        if num_segments != num_points - 1:
            raise ValueError(
                f"Invalid piecewise profile: {num_points} points requires exactly "
                f"{num_points - 1} segments, but {num_segments} were provided."
            )

        # Ensure density points position arrangement
        for i in range(num_points - 1):
            x1 = self.points[i][0]
            x2 = self.points[i+1][0]
            if isinstance(x1, float) and isinstance(x2, float):
                if x1 >= x2:
                    raise ValueError(
                        f"Profile points must be strictly increasing in x. "
                        f"Error at index {i}: x={x1} >= x={x2}."
                    )
        # Ensure constant segment between 2 consecutive points with the same density
        for i, segment_type in enumerate(self.segments):
            if segment_type == "constant":
                val1, val2 = self.points[i][1], self.points[i+1][1]
                if val1 != val2:
                    raise ValueError(
                        f"Segment {i} is 'constant', but values {val1} and {val2} do not match."
                    )
        return self

# ==========================================
# 2. PHYSICS CONFIGURATION
# ==========================================
class WaveConfig(BaseModel):
    freq_LH: float = Field(gt=0.0)
    n_para_req: float

    @computed_field
    def omega_LH(self) -> float:
        return 2 * math.pi * self.freq_LH

    @computed_field
    def k0(self) -> float:
        return self.omega_LH / 299792458.0

# BaseModel = Name data structure for intuitive data recovery
class PlasmaConfig(BaseModel):  # Plasma properties sub-structure
    radial_density_profile: PiecewiseProfile
    b_field_profile: PiecewiseProfile
    pitch_angle_profile: PiecewiseProfile

class PhysicsConfig(BaseModel): # Global physics properties main structure
    wave: WaveConfig            # ex: wave.freq_LH gives 3.7GHz
    plasma: PlasmaConfig        # ex: plasma.radial_density_profile gives (list of (position density) pairs)

# ==========================================
# 3. GEOMETRY CONFIGURATION
# ==========================================
# constraint: gt = (strictly) greater than... , ge = greater or equal to...
class AntennaDimensions(BaseModel): # Antenna geometry dimensions sub-structure:
    wg_width: float = Field(gt=0.0)
    septa_width: float = Field(gt=0.0)
    wg_height: float = Field(gt=0.0)
    wg_length_active: float = Field(gt=0.0)
    wg_length_passive: float = Field(ge=0.0)

class AntennaArrangement(BaseModel):        # Antenna grill sub-structure
    num_rows: int = Field(ge=1)
    row_spacing: float = Field(ge=0.0)
    num_modules: int = Field(ge=1)
    active_waveguides_per_module: List[int]
    power_per_module_W: List[float]
    phase_shift_per_module_deg: List[float]

    @model_validator(mode='after')           # Conflict values checkouts
    def validate_lists_match_modules(self):
        n_mod = self.num_modules
        if len(self.active_waveguides_per_module) != n_mod:
            raise ValueError(f"'active_waveguides_per_module' list length must equal num_modules ({n_mod}).")
        if len(self.power_per_module_W) != n_mod:
            raise ValueError(f"'power_per_module_W' list length must equal num_modules ({n_mod}).")
        if len(self.phase_shift_per_module_deg) != n_mod:
            raise ValueError(f"'phase_shift_per_module_deg' list length must equal num_modules ({n_mod}).")
        return self

class AntennaConfig(BaseModel): # AntennaConfig main structure
    topology: Literal["FAM", "PAM"]
    dimensions: AntennaDimensions
    grill_arrangement: AntennaArrangement

    @computed_field
    def total_width(self) -> float: # Total toroidal antenna width
        # Width calculation (can be expanded based on exact topology logic)
        total_wgs = sum(self.grill_arrangement.active_waveguides_per_module)
        total_septa = total_wgs - 1
        return (total_wgs * self.dimensions.wg_width) + (total_septa * self.dimensions.septa_width)

class DomainConfig(BaseModel):
    # Radial (x)
    Lx_plasma: float = Field(gt=0.0)    # radial dimension of plasma box
    Lx_pml: float = Field(ge=0.0)       # radial PML depth
    # Poloidal (y) - Required for 3D
    Ly_plasma: float = Field(ge=0.0)    # poloidal dimension of plasma box
    Ly_pml: float = Field(ge=0.0)       # poloidal pml depth
    # Toroidal (z)
    Lz_plasma: float = Field(gt=0.0)    # toroidal dimension of plasma box
    Lz_pml: float = Field(ge=0.0)       # toroidal pml depth
    Lz_wall: float = Field(ge=0.0)      # toroidal metal wall width (edge of antenna)

    @computed_field
    def Lx_tot(self) -> float:
        """Automatically calculates total X dimension (Plasma + PML)"""
        return self.Lx_plasma + self.Lx_pml

    @computed_field
    def Lz_tot(self) -> float:
        """Automatically calculates total Z dimension"""
        return self.Lz_plasma + (2 * self.Lz_pml) + (2 * self.Lz_wall)

class MeshConfig(BaseModel):            # mesh refinement param = sub-structure of GeometryConfig
    ppw_medium: float = Field(gt=0.0)   # point per wavelength (compute smallest wavelength) in bulk domain
    ppw_pml: float = Field(gt=0.0)      # ""    ""    ""  in pml domain
    grading: float = Field(gt=0.0, le=1.0)  # smoothing mesh refinement

class PMLConfig(BaseModel):     # sub-structure of GeometryConfig
    use_radial: bool            # true = add radial pml
    use_toroidal: bool          # true = toroidal pml
    use_poloidal: bool          # true = poloidal pml (no use yet)
    # radial PML stretching function parameters
    Sx_r: float = 2.0
    Sx_im: float = 2.0
    px: float = 2.5
    # toroidal PML stretching function parameters
    Sz_r: float = 1.88
    Sz_im: float = 4.0
    pz: float = 3.0

class GeometryConfig(BaseModel):
    antenna: Optional[AntennaConfig] = None
    domain: DomainConfig    # bulk and pml box dimensions
    mesh: MeshConfig        # mesh refinement parameters
    pml: PMLConfig          # PML stretching function parameters set

# ==========================================
# 4. SOLVER & SIMULATION CONFIGURATION
# ==========================================
class SolverConfig(BaseModel): # sub-structure of SimulationConfig
    fem_order: int = Field(ge=1, le=5) # included in [1, 5]
    linear_backend: Literal["mumps", "umfpack", "pardiso"] = "mumps" # type of c++ solver
    max_threads: int = Field(gt=0)      # number of threads to invert matrix problem

class SimModeConfig(BaseModel): # sub-structure of SimulationConfig
    dimension: Literal["1D", "2D", "3D"]                # 1D=toroidal periodic boundary conditions, 2D=radial and toroidal pmls, 3D not use yet
    mode: Literal["DirectAperture", "ExplicitGeometry"] # DirectAperture= module power injection + forced phase shift (no multijunction geometry description)
    box_medium: Literal["VACUUM", "PLASMA"]             # bulk medium
    boundary_toroidal: Literal["periodic", "pml", "pec"] = "pml"

class SimulationConfig(BaseModel): # main structure
    simulation: SimModeConfig   # model dimension, antenna description and bulk medium
    physics: PhysicsConfig      # injected wave properties and plasma profiles (density and B field)
    geometry: GeometryConfig    # model domain dimension, mesh refinement and pml description (choice and stretching function param)
    solver: SolverConfig        # degree of pol, C++ matrix solver and num of allocated thread

    @model_validator(mode='after')
    def enforce_strict_physics_and_geometry(self):
        dim = self.simulation.dimension
        antenna = self.geometry.antenna
        pml = self.geometry.pml
        domain = self.geometry.domain

        # 1. Dimension vs Geometry rules --> Checkout for model/variables conflicts
        if dim == "1D":
            if antenna is not None:
                raise ValueError("1D simulations do not support explicit antenna geometries. Remove the 'antenna' block.")
            if pml.use_toroidal or pml.use_poloidal:
                raise ValueError("In 1D, only radial PML (use_radial) is physically meaningful. Set others to false.")

        if dim == "2D" and pml.use_poloidal:
            raise ValueError("2D simulations are mapped to the x-z plane. 'use_poloidal' must be false.")

        if dim in ["2D", "3D"] and antenna is not None:
            if domain.Lz_plasma < antenna.total_width:
                raise ValueError(
                    f"Domain toroidal length (Lz_plasma={domain.Lz_plasma} m) is too small "
                    f"to fit the antenna ({antenna.total_width} m)."
                )

        # 2. Resolve "Lx_plasma" strings in profiles and rigorously check geometry bounds
        profiles_to_check = [
            self.physics.plasma.radial_density_profile,
            self.physics.plasma.b_field_profile
        ]

        for profile in profiles_to_check:
            resolved_points = []
            for (x, val) in profile.points:
                # Replace string variable with actual float
                resolved_x = domain.Lx_plasma if x == "Lx_plasma" else float(x)
                resolved_points.append((resolved_x, val))

            # Check for strict increasing order after resolving variables
            for i in range(len(resolved_points) - 1):
                if resolved_points[i][0] >= resolved_points[i+1][0]:
                    raise ValueError(
                        f"Profile points must be strictly increasing in x. "
                        f"Error: x={resolved_points[i][0]} >= x={resolved_points[i+1][0]}."
                    )

            # Ensure the final resolved point perfectly matches the domain boundary
            final_x = resolved_points[-1][0]
            if final_x != domain.Lx_plasma:
                raise ValueError(
                    f"The final coordinate of the profile (x={final_x}) does not match "
                    f"Lx_plasma (x={domain.Lx_plasma}). Use 'Lx_plasma' in YAML or ensure exact float match."
                )

            # Override the Pydantic field with purely resolved floats for the rest of the code
            profile.points = resolved_points

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulationConfig":
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)