# ReaDDy — Reaction Diffusion Dynamics

ReaDDy (**Rea**ction **D**iffusion **Dy**namics) is an open-source, particle-based **reaction-diffusion simulator** that combines Brownian dynamics with reactive potentials. Unlike purely stochastic simulators, ReaDDy supports soft interaction potentials between particles, enabling accurate simulation of crowded molecular environments with excluded volume effects.

## What It Does

ReaDDy simulates the mesoscale dynamics of biological systems where both spatial diffusion and chemical reactions are important:

- **Particle-based diffusion**: Molecules diffuse via overdamped Langevin (Brownian) dynamics in 3D
- **Interaction potentials**: Particles interact through soft potentials (harmonic, Lennard-Jones-like) that model excluded volume, crowding, and attraction
- **Reactive potentials**: Particles can react (bind, unbind, convert, fuse, fission) when they come within a reaction radius
- **Topology-based reactions**: Bond formation and breaking, polymer dynamics, complex assembly/disassembly
- **Crowded environments**: Accurate modeling of molecular crowding effects that are absent in point-particle simulators

### Applications
- Protein complex assembly in crowded cytoplasm
- Membrane-associated reaction networks
- Polymer dynamics (actin, microtubules)
- Receptor clustering and signaling
- Chromatin organization
- Phase separation and condensate dynamics
- Bridging atomistic and cellular-scale models

## Biological Scale

**Subcellular / Mesoscale** — bridges molecular (nm) and cellular (μm) scales. Typically simulates 10 nm – 10 μm systems with particles representing individual proteins, protein complexes, or coarse-grained molecular assemblies.

## Key Algorithms

### Dynamics
- **Overdamped Langevin dynamics** (Brownian dynamics): Euler-Maruyama integration of the overdamped Langevin equation
- **Customizable potentials**: Harmonic repulsion, Lennard-Jones, screened electrostatics, user-defined potentials
- **Cell-linked list** spatial decomposition for efficient O(N) neighbor search

### Reactions
- **Fusion**: Two particles combine into one (A + B → C)
- **Fission**: One particle splits into two (A → B + C)
- **Conversion**: One particle type changes to another (A → B)
- **Enzymatic**: Catalytic reactions (A + B → B + C)
- **Topology reactions**: Bond formation, bond breaking, topology type changes
- **Reaction rates**: Microscopic rates converted from macroscopic (Smoluchowski theory)

### Topologies
- Particles connected by bonds forming molecular complexes
- Bond potentials: harmonic, FENE
- Angle potentials: harmonic, cosine
- Dihedral potentials
- Topology-change reactions for dynamic assembly/disassembly

### Output
- **HDF5-based trajectory** output for efficient storage and analysis
- Particle positions, types, and topology information at each time step
- Observable tracking (molecule counts, distances, RDFs)

## Directory Structure

```
readdy/
├── readdy/
│   ├── main/                   # Core C++ simulation engine
│   │   ├── model/              # Particle, topology, reaction models
│   │   ├── kernel/             # CPU kernel implementation
│   │   └── io/                 # HDF5 trajectory I/O
│   └── ...
├── wrappers/python/            # Python bindings (pybind11)
├── contrib/
│   ├── Catch2/                 # Testing framework
│   ├── fmt/                    # String formatting
│   ├── json/                   # JSON parsing
│   ├── pybind11/               # Python bindings
│   └── spdlog/                 # Logging
├── local_deps/
│   ├── c-blosc/                # Data compression
│   └── hdf5/                   # HDF5 library
├── build/
│   └── libreaddy.so            # Shared library
└── ...
```

## Usage

```python
import readdy

# Create a simulation system
system = readdy.ReactionDiffusionSystem(
    box_size=[10, 10, 10],
    periodic_boundary_conditions=[True, True, True]
)

# Define particle types with diffusion coefficients
system.add_species("A", diffusion_constant=1.0)
system.add_species("B", diffusion_constant=1.0)
system.add_species("C", diffusion_constant=0.5)

# Add interaction potentials (excluded volume)
system.potentials.add_harmonic_repulsion("A", "A", force_constant=10.0, interaction_distance=1.0)

# Add reactions
system.reactions.add("bind: A +(0.5) B -> C", rate=1.0)
system.reactions.add("unbind: C -> A +(0.5) B", rate=0.1)

# Run simulation
simulation = system.simulation(kernel="CPU")
simulation.output_file = "trajectory.h5"
simulation.observe.particle_positions(stride=100)
simulation.run(n_steps=10000, timestep=0.01)
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | Python API | System setup, particle types, potentials, reactions |
| Output | HDF5 (`.h5`) | Trajectory data (positions, types, topologies) |
| Output | Observables | Particle counts, radial distribution functions, custom observables |

## Build

```bash
cd ~/Documents/biotools/readdy
mkdir -p build && cd build
cmake .. -DREADDY_BUILD_SHARED_COMBINED=ON
make -j$(nproc)
```

### Dependencies
- CMake 3.x
- C++17 compiler
- HDF5 (built from `local_deps/`)
- c-blosc (built from `local_deps/`)
- pybind11 (bundled)
- Python 3.x + NumPy

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **GROMACS / OpenMM** | Atomistic MD parameters inform ReaDDy's coarse-grained potentials |
| **Smoldyn** | Alternative spatial stochastic simulator (point particles vs. soft potentials) |
| **MCell** | Alternative 3D reaction-diffusion simulator (mesh-based geometry) |
| **COPASI** | Well-mixed kinetics from COPASI inform ReaDDy reaction rates |
| **Vivarium** | ReaDDy simulations can be composed as Vivarium processes |

## License

BSD 3-Clause License

## References

- Schoeneberg, J. et al. (2019). "ReaDDy 2: Fast and flexible software framework for interacting-particle reaction dynamics." *PLOS Computational Biology*, 15(2):e1006830.
- Hoffmann, M. et al. (2019). "ReaDDy - A Software for Particle-Based Reaction-Diffusion Dynamics in Crowded Cellular Environments." *PLOS ONE*, 14(4).
