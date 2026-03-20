# MCell — Monte Carlo Cell Simulator

MCell (Monte Carlo Cell) is a program for simulating **3D reaction-diffusion of individual molecules** within and between cells. It tracks individual molecules as they diffuse via random walks and interact within realistic three-dimensional geometries derived from electron microscopy reconstructions or idealized shapes.

## What It Does

MCell enables researchers to study how the spatial organization of cells affects biochemical signaling. Key capabilities include:

- **Particle-based diffusion**: Individual molecules undergo Brownian motion (random walks) in continuous 3D space bounded by triangulated mesh geometries
- **Surface and volume molecules**: Molecules can diffuse freely in 3D volumes or on 2D membrane surfaces
- **Reaction kinetics**: Bimolecular reactions occur probabilistically when molecules collide, with rates governed by user-specified rate constants
- **Complex geometries**: Simulate within realistic 3D cellular ultrastructure from EM reconstructions (synaptic clefts, dendritic spines, organelle membranes, nuclear pores)
- **Rule-based reactions**: Integrate BioNetGen/NFSim for intracellular signaling with combinatorially complex reaction networks
- **Checkpointing**: Save and restart simulations for long-running computations

### Applications
- Synaptic neurotransmitter release and receptor activation
- Calcium signaling in dendritic spines
- Intracellular signaling cascades in realistic 3D geometry
- Membrane receptor clustering and signal transduction
- Drug-receptor binding in cellular microenvironments

## Biological Scale

**Subcellular** — individual molecules in realistic 3D cellular geometries (nanometers to micrometers). Typically models hundreds to millions of individual molecules.

## Key Algorithms

### Diffusion
- **Monte Carlo random walk**: Molecules take discrete random displacement steps calibrated to their diffusion coefficient
- **Surface diffusion**: 2D random walks constrained to triangulated mesh surfaces
- **Geometry interaction**: Reflection, absorption, and transparency at mesh surfaces

### Reactions
- **Bimolecular collision detection**: Probabilistic reaction upon molecular proximity
- **Unimolecular reactions**: Stochastic first-order decay/transformation
- **Surface reactions**: Molecules can bind to, unbind from, or be absorbed/reflected by surfaces
- **Network-free reactions**: Via NFSim integration for complex reaction networks

### Geometry
- **Triangulated mesh boundaries**: Arbitrary 3D geometries from any mesh source
- **Surface regions**: Named regions on meshes with distinct properties
- **Periodic boundaries**: For bulk simulations

## Directory Structure

```
mcell/
├── src/
│   ├── mcell3/                 # MCell3 core engine (C)
│   ├── mcell4/                 # MCell4 engine (C++ with Python API)
│   ├── bng/                    # BioNetGen integration
│   └── ...
├── libs/
│   ├── pybind11/               # Python binding support
│   ├── jsoncpp/                # JSON parsing
│   └── ...
├── libmcell/                   # Shared library interface
├── build/                      # Build artifacts
└── ...
```

## Usage

### MCell3 (MDL-based)
```bash
# Run an MDL model
~/Documents/biotools/mcell/build/mcell model.mdl
```

### MCell4 (Python API)
```python
import mcell as m

# Create simulation
model = m.Model()
model.config.time_step = 1e-6  # seconds

# Define species
vol_mol = m.Species('A', diffusion_constant_3d=1e-6)

# Define geometry, release sites, reactions...
# Run simulation
model.run_iterations(1000)
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.mdl` | Model Description Language files (MCell3) |
| Input | Python | MCell4 Python API scripts |
| Input | Mesh files | 3D geometry (from CellBlender, mesh_tools, etc.) |
| Output | `.dat` | Reaction data counts (molecule numbers over time) |
| Output | VIZ data | Molecule position visualization data |
| Output | DREAM.3D | Spatial data outputs |

## Build

```bash
cd ~/Documents/biotools/mcell
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **CellBlender** | Primary GUI for setting up MCell simulations in Blender |
| **mesh_tools** | Mesh conversion and repair for simulation geometries |
| **neuropil_tools** | EM reconstruction processing for neuronal ultrastructure |
| **GAMer** | High-quality mesh generation for simulation domains |
| **NFSim / BioNetGen** | Rule-based intracellular reactions (via nfsimCInterface/libbng) |
| **nfsimCInterface** | C API for embedding NFSim within MCell |
| **libbng** | Lightweight BioNetGen library for network generation |
| **mcell_tests** | Test suite for validating MCell builds |
| **mcell_tools** | Build and utility scripts for the MCell ecosystem |

## License

MIT License (MCell4), GPL v2 (MCell3)

## References

- Stiles, J.R. & Bartol, T.M. (2001). "Monte Carlo Methods for Simulating Realistic Synaptic Microphysiology Using MCell." *Computational Neuroscience: Realistic Modeling for Experimentalists*.
- Kerr, R.A. et al. (2008). "Fast Monte Carlo Simulation Methods for Biological Reaction-Diffusion Systems in Solution and on Surfaces." *SIAM J. Sci. Comput.*, 30(6):3126-3149.
