# CompuCell3D — Multiscale Multicellular Modeling Environment

CompuCell3D (CC3D) is a **multiscale modeling environment** for simulating multicellular systems based on the **Cellular Potts Model** (CPM, also known as the Glazier-Graner-Hogeweg model). Cells are represented as connected domains of lattice sites that evolve through a modified Metropolis algorithm, enabling realistic simulation of cell sorting, migration, growth, division, death, and morphogenesis.

## What It Does

CompuCell3D simulates the emergent behaviors of multicellular systems by integrating:

- **Cell-level mechanics**: Adhesion, volume constraints, surface area constraints, chemotaxis — all encoded in a Hamiltonian energy function
- **Subcellular signaling**: SBML-based ODE networks running independently within each cell
- **Tissue-level fields**: PDE diffusion solvers for chemical signals (morphogens, nutrients, waste)
- **Cell behaviors**: Growth, division, death, differentiation, migration, contact-inhibited locomotion

### Typical Applications
- Cell sorting and tissue boundary formation
- Tumor growth and invasion
- Angiogenesis (blood vessel formation)
- Embryonic morphogenesis
- Wound healing
- Delta-Notch lateral inhibition
- Reaction-diffusion pattern formation

## Biological Scale

**Cellular / Tissue** — models 10s to 10,000s of cells on a lattice (1 μm – 1 mm). Each lattice site represents a fraction of a cell, and cells typically span 10–50 lattice sites.

## Key Algorithms

### Cellular Potts Model (CPM)
- Stochastic **pixel-copy dynamics** minimizing a Hamiltonian energy function
- Energy terms: volume constraint, surface area constraint, cell-cell adhesion (type-specific contact energies), chemotaxis, haptotaxis
- **Modified Metropolis algorithm**: Boltzmann acceptance probability for lattice site reassignment
- Configurable temperature parameter controlling stochasticity

### Subcellular Models
- **SBML integration**: Each cell can run its own ODE-based intracellular signaling model
- Feedback between intracellular state and cell-level behaviors (phenotype switching, growth rate, chemotactic response)

### PDE Solvers
- Finite difference solvers for diffusing chemical fields
- SteadyStateDiffusion, FlexibleDiffusion, ReactionDiffusion solvers
- Secretion, absorption, and field manipulation by individual cells

### Cell Behaviors
- Configurable cell cycle models (growth → division)
- Programmed cell death (apoptosis)
- Directed migration via chemotaxis and haptotaxis
- Cell type transitions and differentiation

## Language & Interface

CompuCell3D is written in **C++** with extensive **Python bindings**. Models are defined using:
- **XML configuration** for basic simulation parameters and plugins
- **Python steppables** for custom cell behaviors, analysis, and visualization
- **CC3D project files** (`.cc3d`) bundling XML, Python scripts, and resources

## Directory Structure

```
CompuCell3D/
├── CompuCell3D/
│   ├── core/                       # Core C++ simulation engine
│   │   ├── CompuCell3D/            # Main library source
│   │   ├── Demos/                  # Example simulations
│   │   └── ...
│   ├── DeveloperZone/              # Developer documentation
│   └── ...
├── build/
│   └── core/CompuCell3D/
│       └── libCC3DCompuCellLib.so  # Core shared library
├── docker/                         # Docker configurations
└── ...
```

## Usage

```bash
# Run a CC3D simulation from the command line
cd ~/Documents/biotools/CompuCell3D
python -m cc3d.run_script simulation_folder/

# Or use the graphical Player application
cc3d_player simulation.cc3d
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.cc3d` | Project files bundling simulation configuration |
| Input | `.xml` | Simulation parameters, plugins, steppables |
| Input | `.py` | Python steppable scripts for custom behaviors |
| Output | VTK | Lattice snapshots for 3D visualization |
| Output | CSV | Numerical data (cell counts, field values) |
| Output | PNG | Screenshot images from simulation |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **COPASI** | SBML models from COPASI can serve as intracellular submodels in each CC3D cell |
| **BioNetGen** | Rule-based reaction models can be embedded via SBML export |
| **mesh_tools** | Output meshes can be processed for further analysis |
| **PhysiCell** | Alternative multicellular simulator using off-lattice agent-based approach |
| **Morpheus** | Alternative CPM simulator with different strengths (declarative XML, FEM solvers) |
| **Vivarium** | Can be composed as a process in Vivarium's multiscale framework |

## Build

```bash
cd ~/Documents/biotools/CompuCell3D
mkdir -p build && cd build
cmake ../CompuCell3D
make -j$(nproc)
```

## License

MIT License

## References

- Swat, M.H. et al. (2012). "Multi-Scale Modeling of Tissues Using CompuCell3D." *Methods in Cell Biology*, 110:325-366.
- Glazier, J.A. & Graner, F. (1993). "Simulation of the differential adhesion driven rearrangement of biological cells." *Physical Review E*, 47(3):2128-2154.
