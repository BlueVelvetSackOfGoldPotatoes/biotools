# Tissue Forge — Interactive Particle-Based Tissue Simulator

Tissue Forge is an interactive, **particle-based physics engine** for biological and biophysics simulations. It models cells, subcellular structures, and tissue-like assemblies as particles interacting through customizable potentials, bonds, and surfaces, featuring real-time 3D visualization.

## What It Does

Tissue Forge provides an interactive simulation environment where researchers can:

- **Model cells and tissues** as particles with customizable interactions
- **Visualize in real-time**: Built-in OpenGL 3D rendering that updates as the simulation runs
- **Define custom forces**: Arbitrary pair potentials (Lennard-Jones, Morse, harmonic, Coulomb, DPD, custom expressions)
- **Create bonds and structures**: Bond, angle, and dihedral potentials for connected structures
- **Simulate transport**: Secretion, absorption, and diffusion of chemical species
- **Handle events**: Stochastic transitions, cell division, death, and type changes
- **Interact during simulation**: Pause, modify parameters, add/remove particles, and probe the system while it runs

### Applications
- Cell sorting and tissue self-organization
- Epithelial sheet dynamics
- Cell migration and collective motion
- Morphogenesis simulations
- Phase separation and droplet dynamics
- Developmental biology models
- Educational demonstrations of biophysics

## Biological Scale

**Cellular / Tissue** — real-time interactive simulations, typically 100s to 10,000s of particles representing cells or subcellular objects.

## Key Features

### Physics Engine
- **Dissipative Particle Dynamics (DPD)**: Momentum-conserving thermostat for mesoscale dynamics
- **Pair potentials**: Lennard-Jones, Morse, harmonic, Coulomb, DPD, and custom expressions
- **Bonded interactions**: Harmonic bonds, angle potentials, dihedral potentials
- **Boundary conditions**: Periodic, reflective, and absorptive walls
- **Chemical species**: Secretion, uptake, and diffusion of substrates

### Interactivity
- Real-time OpenGL rendering
- Interactive parameter modification during simulation
- Mouse/keyboard control for adding particles, applying forces
- Python and C API for programmatic control

### Cell Biology Features
- **Cell division**: Configurable division rules (size-based, time-based, stochastic)
- **Cell death**: Apoptosis with configurable kinetics
- **Type switching**: Differentiation and phenotype changes
- **Flux and secretion**: Chemical signaling between particles

## Directory Structure

```
tissue-forge/
├── source/
│   ├── mdcore/                 # Core molecular dynamics engine
│   ├── rendering/              # OpenGL rendering
│   ├── models/                 # Biological model implementations
│   └── ...
├── extern/
│   ├── corrade/                # Magnum dependency
│   ├── libsbml/                # SBML support
│   └── ...
├── wraps/
│   ├── python/                 # Python bindings
│   └── C/                      # C API
├── build/Release/lib/
│   ├── libtissue-forge.so      # Core shared library
│   └── libtissue-forge-c.so    # C API shared library
└── ...
```

## Usage

### Python
```python
import tissue_forge as tf

# Initialize with window
tf.init(dt=0.01, dim=[20, 20, 20])

# Define cell types
class CellA(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    style = {"color": "red"}

class CellB(tf.ParticleTypeSpec):
    radius = 0.5
    dynamics = tf.Overdamped
    style = {"color": "blue"}

A = CellA.get()
B = CellB.get()

# Define interactions
pot = tf.Potential.morse(d=1.0, a=5.0, r0=1.0, min=0.5, max=3.0)
tf.bind.types(pot, A, B)

# Add particles
[A(tf.FVector3(tf.random_point())) for _ in range(100)]
[B(tf.FVector3(tf.random_point())) for _ in range(100)]

# Run with real-time visualization
tf.run()
```

### C API
```c
#include "tissue-forge-c.h"

tfSimulator_Config config;
tfSimulator_Config_init(&config);
config.dt = 0.01;
tfSimulator_init(config);

// Define types, potentials, particles...
tfSimulator_run(1000);
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | Python/C API | Programmatic simulation setup |
| Input | JSON | Configuration files |
| Output | Screenshots | Real-time visualization captures |
| Output | Trajectories | Particle positions and states over time |
| Output | Real-time | Interactive 3D visualization |

## Build

```bash
cd ~/Documents/biotools/tissue-forge
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Dependencies
- CMake 3.x
- C++17 compiler
- OpenGL (for rendering)
- Magnum/Corrade (graphics framework, bundled)
- libSBML (bundled in `extern/`)
- Python 3.x + pybind11

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **PhysiCell** | Alternative multicellular simulator (XML-configured batch runs vs. interactive real-time) |
| **CompuCell3D** | Alternative multicellular simulator (CPM lattice-based vs. particle-based) |
| **BETSE** | Bioelectric modeling (complementary physics for tissue electrics) |
| **Vivarium** | Can be composed as a process in multiscale simulations |
| **local_deps** | Uses Eigen and Assimp from shared dependency builds |
| **COPASI** | SBML models for intracellular dynamics (via libSBML) |

## License

MIT License

## References

- Sego, T.J. et al. (2023). "Tissue Forge: Interactive biological and biophysics simulation environment." *PLOS Computational Biology*, 19(10):e1010768.
