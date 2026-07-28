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
		get_dielectric_matrix-----: Return complex cold plasma Stix tensor using rotate B field tensor 
	waveguide.py--------------------: Compute analytical waveguide mode and port boundary conditions
		Class WaveguideMode-------: Compute analytical TE10 mode impedance to define absorbing port condition at back end of waveguides (klyston isolator), project E_tot on TE10 to compute power reflection coefficient S_11 (amplitude and phase). 
(***)	nonlinear.py--------------------: (...)
(***)		Class PonderomotiveForce--: (...)

geometry/===> 
	antenna.py----------------------------: Creates launcher aperture & rounded corners
		Class AntennaGeometry-----------: Use coordinates of metallic walls, port entrance locations, and aperture boundaries facing the plasma to generate antenna boundaries (no 90° angle corners-> replace by circular arcs)
	domain.py-----------------------------: Combines antenna + vacuum gap + plasma
		Class SimulationDomain----------: Build complete simulation domain: antenna geometry to Plasma medium to PMLs: 
	(***)	Class CurvatureMapper-----------: (...)
	pml.py--------------------------------: Coordinate stretching function definition 
		Class PMLFactor-----------------: Get stretching functions (Sx, Sy, Sz) parameters ([Jacquot 2013] formalism) from .yaml files and compute the combined PML/Stix tensor. Then passes the tensor to solver/assembler.py
	mesh.py-------------------------------: Generate the domain mesh 
		Class MeshGenerator-------------: Convert continuous geometric description (from Class SimulationDomain) into nodes/(edges elements)/cells discretization using Netgen 
		apply_singularity_refinement()--: Mesh necessite refinement at Antenna/Plasma interface (because vacuum to plasma & waveguides corners = fixed domain area) + high resolution in LH wave resonance cones and coarse mesh in vacuum or into plasma where no E field. (require pml ppw study to evaluate mesh in pml). Mesh refinement is smoothed (by a NGSolve intern function 10 or 15% element growth in size) to avoid spurious reflections. 

solver/===>
	bc.py-------------------------: Set the boundary conditions used by the solver 
		Class WaveguidePortBC---: Robin boundary condition to inject power and to absorb reflected power waves (1D: bc=scalar, 2D: bc=segment line)
		Class PECBC-------------: Perfect Electrical Conductor conditions applied at every PMLs back end = edge limits of simulations box & to antenna metallic part (careful of rounded corners), n \cross E = 0 -> tangential field is 0
	assembler.py------------------: Build the matrix system Ax=b
		Class MaxwellAssembler--: Gathers PML/Stix tensor + boundary conditions + weak form expression
	linear_solver.py--------------: Solve E field to weak form problem 
		Class LinearSolver------: choice of solver (default=...)
	engine.py---------------------: C++ solver
		Class FEMEngine---------: It takes the parsed SimulationConfig and the generated mesh, then it calls MaxwellAssembler to build the base matrices, it maps the boundary conditions in bc.py, finally hands the matrix to LinearSolver -> Raw E field solution
	(***)	solve_non_linear_loop---: for AMR on E field lines: based on E field gradient or density profile gradient (not a priority)

postprocess/===> Gather all the plot functions
	sparameter.py-------:
		Class SparameterExtract--:
		compute_refl_coeff()-----:
	spectrum.py--------------------:
		Power_vs_n_para----------: Compute power spectrum with Fourier Transform on Ez, and compute directivity
	fields.py----------------------:
		Class FieldEval----------/


* .yaml data --> SimulationConfig --> AntennaGeometry (build the antenna pattern) --> SimulationDomain gather the antenna geometry to plasma to PMLs regions (data from .yaml file) --> give the geometry to MeshGenerator (build the averaged mesh) then run apply_singularity_refinement() (not necessary) / * PMLFactor build the stretching functions for PML sub-domains 
* THEN: solver accept the mesh and the Stix/PML tensor to solve the problem. 

YAML Config (Parameters)
            │
            ▼
 ┌──────────────────────┐
 │   AntennaGeometry    │ (antenna.py: Creates launcher aperture & rounded fillets)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │   SimulationDomain   │ (domain.py: Combines antenna + vacuum gap + plasma)
 └──────────┬───────────┘
            │
            ├──────────────────────────┐
            ▼                          ▼
 ┌──────────────────────┐   ┌──────────────────────┐
 │    MeshGenerator     │   │      PMLFactor       │ (pml.py: Coordinate stretching)
 │      (mesh.py)       │   └──────────┬───────────┘
 └──────────┬───────────┘              │
            │ Mesh Grid Nodes          │ Complex Tensor
            ▼                          ▼
 ┌─────────────────────────────────────────────────┐
 │               Solver / Assembler                │
 └─────────────────────────────────────────────────┘




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