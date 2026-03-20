# Smoldyn — Spatial Stochastic Biochemical Simulator

Smoldyn (Smoluchowski Dynamics) is a **spatial stochastic simulator** for biochemical reaction networks. Molecules are represented as point particles that diffuse in continuous 2D or 3D space and react upon collision using Smoluchowski reaction dynamics. Surfaces (membranes, organelles) are modeled as triangulated meshes that molecules can bind to, reflect from, or pass through.

## What It Does

Smoldyn enables researchers to study how the spatial organization of cells affects biochemical reactions:

- **Individual molecule tracking**: Each molecule is a point particle with a defined diffusion coefficient and species identity
- **Brownian dynamics**: Molecules undergo random walk diffusion calibrated by exact first-passage time theory
- **Smoluchowski reactions**: Bimolecular reactions occur probabilistically when molecules approach within a binding radius
- **Surface interactions**: Molecules can be reflected, absorbed, transmitted, or adsorbed at membrane surfaces
- **Compartment logic**: Define compartments bounded by surfaces with distinct reaction rules
- **Rule-based reactions**: Integration with BioNetGen for complex reaction networks
- **Lattice hybrid**: Combine particle-based and lattice-based simulation for multi-scale problems

### Applications
- Signaling at the cell membrane (receptor activation, clustering)
- Intracellular signal transduction in realistic geometry
- Calcium dynamics near channels and pumps
- Bacterial chemotaxis
- Gene regulation in the nucleus
- Synaptic transmission
- Min protein oscillation in bacteria

## Biological Scale

**Subcellular** — individual molecules in 2D or 3D space (100s to millions of particles), spanning nanometers to micrometers.

## Key Algorithms

### Diffusion
- **Exact Brownian dynamics**: Random-walk steps calibrated by analytical first-passage time solutions for accuracy
- **Surface diffusion**: 2D diffusion on triangulated mesh surfaces
- **Drift**: External forces and drift terms

### Reactions
- **Smoluchowski bimolecular reactions**: Binding radius and unbinding radius computed from macroscopic rate constants
- **Unimolecular reactions**: First-order decay with exact exponential statistics
- **Surface reactions**: Adsorption, desorption, and surface-catalyzed reactions
- **Rule-based**: BioNetGen (BNGL) rule parsing and execution

### Surfaces
- Triangulated mesh surfaces as boundaries
- Reflection, absorption, transmission, and periodic boundary interactions
- Surface molecules (2D diffusion on membranes)
- Jump between surfaces (e.g., endocytosis/exocytosis)

### Hybrid Methods
- Lattice-particle hybrid for multi-scale simulation
- NSM (Next Subvolume Method) lattice regions coupled with Smoldyn particle regions

## Directory Structure

```
Smoldyn/
├── source/
│   ├── Smoldyn/                # Core simulation engine (C)
│   │   ├── smoldyn.h
│   │   ├── smolmol.c           # Molecule management
│   │   ├── smolreact.c         # Reaction processing
│   │   ├── smolsurf.c          # Surface interactions
│   │   ├── smolwall.c          # Wall/boundary conditions
│   │   └── ...
│   ├── libSteve/               # Utility library
│   ├── BioNetGen/              # BioNetGen integration
│   └── pybind11/               # Python bindings
├── examples/                   # Example configuration files
├── docs/                       # Documentation
├── build/
│   └── smoldyn                 # Compiled binary
└── ...
```

## Usage

### Command Line
```bash
# Run a Smoldyn configuration file
~/Documents/biotools/Smoldyn/build/smoldyn config.txt
```

### Python
```python
import smoldyn

# Create simulation
s = smoldyn.Simulation(low=[-10, -10, -10], high=[10, 10, 10])

# Define species
A = s.addSpecies("A", difc=1.0, color="red")
B = s.addSpecies("B", difc=1.0, color="blue")
C = s.addSpecies("C", difc=0.5, color="green")

# Add molecules
s.addMolecules(A, 1000)
s.addMolecules(B, 1000)

# Define reactions
s.addReaction("bind", subs=[A, B], prds=[C], rate=1e6)
s.addReaction("unbind", subs=[C], prds=[A, B], rate=0.1)

# Run
s.run(stop=100, dt=0.01)
```

### Configuration File (`.txt`)
```
# Smoldyn configuration
dim 3
boundaries 0 -10 10
boundaries 1 -10 10
boundaries 2 -10 10

species A B C
difc A 1
difc B 1
difc C 0.5

reaction bind A + B -> C 1e6
reaction unbind C -> A + B 0.1

mol 1000 A u u u
mol 1000 B u u u

time_start 0
time_stop 100
time_step 0.01

output_files output.txt
cmd i 0 100 1 molcount output.txt
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.txt` | Smoldyn configuration files |
| Input | `.bngl` | BioNetGen rule files (for rule-based models) |
| Input | Python | Python API for programmatic setup |
| Output | Text files | Molecule counts, positions, reaction data |
| Output | Visualization | Real-time OpenGL rendering (optional) |

## Build

```bash
cd ~/Documents/biotools/Smoldyn
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **BioNetGen** | Smoldyn can read `.bngl` rules for complex reaction networks |
| **MCell** | Alternative 3D reaction-diffusion simulator (mesh-based Monte Carlo) |
| **ReaDDy** | Alternative particle-based simulator (with soft potentials vs. point particles) |
| **COPASI** | Well-mixed kinetics inform Smoldyn reaction rates |
| **Vivarium** | Smoldyn can be composed as a Vivarium process |
| **libsmoldyn** | Embeddable library version for integration into other simulators |

## License

LGPL v3

## References

- Andrews, S.S. (2017). "Smoldyn: particle-based simulation with rule-based modeling, improved molecular interaction and a library interface." *Bioinformatics*, 33(5):710-717.
- Andrews, S.S. & Bray, D. (2004). "Stochastic simulation of chemical reactions with spatial resolution and single molecule detail." *Physical Biology*, 1(3):137-151.
