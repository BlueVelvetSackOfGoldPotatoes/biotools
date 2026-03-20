# OpenMM — High-Performance Molecular Dynamics Library

OpenMM is a high-performance **toolkit for molecular simulation** that provides a flexible, extensible API for defining custom forces, integrators, and simulation workflows. It emphasizes GPU acceleration and serves as the dynamics backend for many other tools in the biomolecular simulation ecosystem.

## What It Does

OpenMM is both a standalone molecular dynamics engine and an embeddable library. It enables:

- **GPU-accelerated molecular dynamics**: Runs on NVIDIA (CUDA), AMD (OpenCL), and Intel GPUs with near-optimal performance
- **Custom force definitions**: Define arbitrary mathematical force expressions that are compiled to GPU kernels at runtime — no GPU programming required
- **Custom integrators**: Define new integration algorithms using a simple scripting language
- **Standard simulations**: Protein folding, ligand binding, membrane dynamics, nucleic acid simulations
- **Enhanced sampling**: Metadynamics, replica exchange, umbrella sampling, alchemical free energy calculations
- **Polarizable force fields**: AMOEBA, Drude oscillator models
- **Quantum nuclear effects**: Ring Polymer MD (RPMD)
- **Implicit solvent**: GBSA models for faster simulations without explicit water

## Biological Scale

**Molecular / Atomistic** — simulates biomolecular systems at atomic resolution (Angstroms), typically 10^3 to 10^6 atoms.

## Key Features

### Force Field Support
- AMBER, CHARMM, OPLS force fields
- AMOEBA polarizable force field
- Drude polarizable force field
- Custom force expressions (compiled to GPU kernels at runtime)
- Implicit solvent (GBSA)

### Integrators
- Langevin (stochastic dynamics with heat bath)
- Brownian dynamics
- Verlet (NVE)
- Variable time-step integrators
- Custom integrators (user-defined)

### GPU Acceleration
- CUDA platform (NVIDIA GPUs)
- OpenCL platform (AMD, Intel, NVIDIA)
- CPU platform (reference/optimized)
- Mixed precision for optimal performance

### Advanced Methods
- Alchemical free energy calculations (FEP, TI)
- Replica exchange MD
- Metadynamics (via OpenMM-PLUMED plugin)
- Constant pH simulations
- Markov State Model construction

## Directory Structure

```
openmm/
├── openmmapi/                  # Core C++ API
├── platforms/
│   ├── cuda/                   # CUDA GPU platform
│   ├── opencl/                 # OpenCL GPU platform
│   ├── cpu/                    # CPU platform
│   └── reference/              # Reference (correctness testing)
├── plugins/
│   ├── amoeba/                 # AMOEBA force field
│   ├── drude/                  # Drude polarizable model
│   └── rpmd/                   # Ring Polymer MD
├── wrappers/python/            # Python bindings
├── examples/                   # Example simulations
├── build/
│   ├── libOpenMM.so            # Core shared library
│   ├── libOpenMMCPU.so         # CPU platform
│   ├── libOpenMMPME.so         # PME electrostatics
│   ├── libOpenMMAmoeba.so      # AMOEBA plugin
│   ├── libOpenMMDrude.so       # Drude plugin
│   └── libOpenMMRPMD.so        # RPMD plugin
└── ...
```

## Usage

### Python API
```python
from openmm import app, unit
from openmm import LangevinMiddleIntegrator

# Load a PDB structure
pdb = app.PDBFile('protein.pdb')
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# Create the system
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=app.HBonds
)

# Set up integrator and simulation
integrator = LangevinMiddleIntegrator(
    300*unit.kelvin,          # Temperature
    1.0/unit.picoseconds,     # Friction
    0.004*unit.picoseconds    # Time step
)

simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Minimize and run
simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1000))
simulation.step(100000)
```

### Custom Forces
```python
from openmm import CustomExternalForce

# Apply a custom harmonic restraint
force = CustomExternalForce('k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
force.addGlobalParameter('k', 100.0)
force.addPerParticleParameter('x0')
force.addPerParticleParameter('y0')
force.addPerParticleParameter('z0')
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | PDB/PDBx | Atomic coordinates |
| Input | XML | Force field parameter files |
| Input | Python/C++ | API-driven simulation setup |
| Output | DCD | Trajectory files |
| Output | PDB | Structure snapshots |
| Output | CSV/custom | Energy, temperature, and state data via reporters |

## Build

```bash
cd ~/Documents/biotools/openmm
mkdir -p build && cd build
cmake .. -DOPENMM_BUILD_CUDA_LIB=ON
make -j$(nproc)
make install
make PythonInstall
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **Boltz / ESM** | Predicted structures refined through OpenMM molecular dynamics |
| **GROMACS** | Alternative MD engine; complementary strengths (GROMACS for throughput, OpenMM for flexibility) |
| **Vivarium** | OpenMM simulations can be wrapped as Vivarium processes |
| **DeepXDE / Modulus** | Train neural surrogates on OpenMM simulation data |
| **ReaDDy** | Mesoscale potentials informed by OpenMM atomistic simulations |

## License

MIT License (LGPL for some plugins)

## References

- Eastman, P. et al. (2017). "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics." *PLOS Computational Biology*, 13(7):e1005659.
- Eastman, P. et al. (2013). "OpenMM 4: A Reusable, Extensible, Hardware Independent Library for High Performance Molecular Simulation." *Journal of Chemical Theory and Computation*, 9(1):461-469.
