=== Code architecture ===
aloha_fem/
├── config/
│   ├── schemas.py          # Pydantic data validation models
│   └── loader.py           # YAML/TOML configuration parser
├── physics/
│   ├── stix.py             # Cold plasma dielectric tensor physics
│   ├── waveguide.py        # Waveguide mode propagation & impedances
│   └── nonlinear.py        # Ponderomotive force & density modification (Future)
├── geometry/
│   ├── antenna.py          # Antenna & waveguide array geometry definition
│   ├── domain.py           # Computation domain & tokamak curvature mapping
│   ├── pml.py              # PML complex coordinate stretching formulations
│   └── mesh.py             # Mesh generator wrapper (singularity grading)
├── solver/
│   ├── bc.py               # Boundary Condition assembly (Robin, PEC, Interface)
│   ├── assembler.py        # Galerkin FEM matrix assembly (Maxwell equation)
│   ├── linear_solver.py    # Direct/Iterative matrix solvers (MUMPS, SciPy)
│   └── engine.py           # Non-linear iteration & solver orchestration
├── postprocess/
│   ├── sparameters.py      # Orthogonal mode projection & S-matrix extraction
│   ├── spectrum.py         # Brambilla N_parallel power spectrum & directivity
│   ├── fields.py           # E/H field evaluation & Poynting flux calculations
│   └── viz.py              # Plotting routines (Curved mesh fields, S-param vs ne0)
├── io/
│   ├── matlab.py           # Legacy MATLAB MAT / ALOHA file importer
│   └── hdf5.py             # Native HDF5 exporter/importer
├── simulation.py           # Top-level User API (The "Facade")
└── main.py                 # Command Line Interface (CLI) entry point


* Pydantic model: type checking and validation  
config/===> Garantee inputs are consistent before FEM assembling
	schema.py----------------: Contains Pydantic models for type checking and validation of simulation params before running
		PlasmaConfig.py----: Store plasma properties: density profile (ne), decay length (lambda_n), RF frequency (omega), magnetic field profile (B)
					=====> Can accept "analytical type" such as "linear", "exponential": described by (position, density) pairs and a function to connect points)
					=====> Or "experimental type" in dataset (depends on the file format: H5, IMAS, ect...) 
		AntennaConfig.py---: 2 modes available: 
						@ DirectApertureMode: Fast plasma coupling scan	
							- User directly defines klystron power and phase step and wether active or passive waveguides (Benchmarking vs ALOHA 1D/2D)
						@ ExplicitGeometryMode: Engineering & Antenna design 
							- internal klystron feeding port, power splitters, physical channel lengths. 
							- Phase shift at antenna mouth is computed naturally 
					=====> Module Level: Klystron power input, main waveguide dimensions (height, width, depth), fundamental mode excitation
					=====> Multijunction Topology: number of waveguides (precise active or passive) per module, internal waveguide length and septa thickness							=====> Aperture Mouth: 
							- Final grill layout : number rows/columns of waveguides facing the plasma 
							- N-parallel spectrum: Computed Power spectrum vs n_// (launched power into the plasma)
		DomainConfig.py----: Box dimensions, PML choices (radial, toroidal, poloidal) and PLASMA/PMLs mesh refinement and boundary conditions
		SolverConfig.py----: FEM polynomial degree, mesh refinement (on high E gradient), 
		SimulationConfig.py: Main config gathering all config class and process. /!\ Set up simu in 1 line: config = SimulationConfig.from_yaml("input.yaml")
	loader.py----------------: Load_config(yaml_path): Parses YAML files into validated SimulationConfig objects

physics/===> Plasma & Wave physics independent of FEM/mesh solver
	stix.py-------------------------: 
		Class StixTensor----------: Compute cold plasma tensor elements: S, D, P given n_e(x,y,z) and B(x,y,z)
		get_dielectric_matrix-----: Return complex cold plasma Stix tensor
	waveguide.py--------------------: 
		Class WaveguideMode-------:
	nonlinear.py--------------------:
		Class PonderomotiveForce--:

geometry/===> 
	antenna.py----------------------------:
		Class AntennaGeometry-----------:
	domain.py-----------------------------:
		Class SimulationDomain----------:
		Class CurvatureMapper-----------:
	pml.py--------------------------------:
		Class PMLFactor-----------------:
	mesh.py-------------------------------:
		Class MeshGenerator-------------:
		apply_singularity_refinement()--:

solver/===>
	bc.py--:
		Class WaveguidePortBC--:
		Class PECBC------------:
	assembler.py-----------------:
		Class MaxwellAssembler--:
	linear_solver.py--------------:
		Class LinearSolver------:
	engine.py---------------------:
		Class FEMEngine---------:
		solve_non_linear_loop---:

postprocess/===> Gather all the function 
	sparameter.py-------:
		Class SparameterExtract--:
		compute_refl_coeff()-----:
	spectrum.py--------------------:
		Power_vs_n_para----------: Compute power spectrum with Fourier Transform on Ez, and compute directivity
	fields.py----------------------:
		Class FieldEval----------/


=============== Example SimulationConfig.py ========================================
from pydantic import BaseModel
from .physics_schema import PhysicsConfig
from .geometry_schema import GeometryConfig
from .domain_schema import DomainConfig
from .solver_schema import SolverConfig

class SimulationConfig(BaseModel):
    physics: PhysicsConfig
    geometry: GeometryConfig
    domain: DomainConfig
    solver: SolverConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulationConfig":
        """Loads and validates the entire YAML file in one line."""
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
====================================================================================
#
#
=============== Example Verification function in SimulationConfig.py ========================================
@model_validator(mode="after")
    def validate_cross_module_physics(self):
        # Prevent selecting poloidal PML in a 2D planar simulation
        if self.geometry.dimension == "2D" and self.domain.pml.use_poloidal:
            raise ValueError("Poloidal PML cannot be enabled for a 2D simulation.")
            
        # Ensure phase vector length matches the waveguide count
        n_wg = self.geometry.antenna.num_waveguides
        n_phases = len(self.geometry.antenna.phases)
        if n_phases != n_wg:
            raise ValueError(
                f"Antenna has {n_wg} waveguides, but {n_phases} phase values were provided!"
            )
        return self
=============================================================================================================
#
#
=============== Data-Structure Waveguides Properties ========================================
class WaveguideChannel(BaseModel):
    channel_id: int
    is_active: bool = True  # True = driven/active, False = passive/terminated
    width: float            # Toroidal width
    height: float           # Poloidal height (for 2D/3D)
    length: float           # Physical length of the channel
    phase_offset: float = 0.0  # Extra fixed phase shift (if active)

class ModuleConfig(BaseModel):
    module_id: int
    input_power: float     # Klystron input power (Watts)
    channels: List[WaveguideChannel]
=============================================================================================



=== Intern file communication ===
Pydantic model = validate the feasibility of a simulation set of params before running it
models.py---------------------: Pydantic model  
config/setting.py-------------: Set up environment (file paths, ect..)
core/logic.py-----------------: Functions (inputs=Pydantic model) do the process (solver) and output a Pydantic model
pipeline/orchestrator.py-----: Import model and logic then read yaml file pass it to Pydantic, then hand the validated data to logic (to solve the matrix problem) 


=== Workflow ===
* User write and run .yaml simulation param file into main (-> it gives .yaml file to orchestrator.py: validate (or not => Value Error) the parameters compatibility)
* orchestrator.py feeds logic.py (solver computation) with correct simulation params data 
	* (if any specific custom error occurs in logic.py => exception.py then stop the process)
Once system solved: 
* The results data are Packed into a Pydantic Output Model (JSON or dictionnary format)