# Morpheus — Multicellular Modeling & Simulation Environment

Morpheus is an integrated **modeling and simulation environment** for the study of multiscale and multicellular systems. It provides a declarative XML-based model description language (MorpheusML), a graphical user interface, and a simulation engine that supports Cellular Potts Models, continuous field PDEs, ODEs, and event-based rules in a unified framework.

## What It Does

Morpheus allows researchers to build and simulate multicellular models by combining:

- **Cellular Potts Model (CPM)**: Lattice-based cell mechanics with configurable energy terms (adhesion, volume, surface area, chemotaxis)
- **PDE solvers**: Finite-element methods on unstructured grids for diffusible signals (morphogens, nutrients)
- **ODE integration**: Per-cell intracellular dynamics (Runge-Kutta, CVODE)
- **Event-based rules**: Conditional cell behaviors (division, death, differentiation, type switching)
- **Population-level models**: Lattice-gas cellular automata
- **Declarative model specification**: XML-based MorpheusML language for reproducible, shareable models

### Applications
- Embryonic pattern formation and morphogenesis
- Cell sorting and tissue boundary formation
- Chemotaxis and collective cell migration
- Turing pattern formation
- Tumor growth modeling
- Wound healing simulations
- Delta-Notch lateral inhibition

## Biological Scale

**Cellular / Tissue** — simulates multicellular systems from tens to thousands of cells with coupled intracellular signaling and tissue-level fields.

## Key Features

### Cellular Potts Model
- Extended Metropolis algorithm for cell dynamics
- Configurable Hamiltonian: volume constraint, surface constraint, adhesion (type-specific contact energies), chemotaxis, haptotaxis
- Cell division, death, and differentiation
- Contact-inhibited locomotion

### Field Solvers
- Finite-element PDE solvers on unstructured grids
- Reaction-diffusion systems
- Secretion and absorption by individual cells
- Custom field equations

### Intracellular Dynamics
- ODE systems per cell (Runge-Kutta, CVODE)
- SBML model import
- Cell-autonomous gene regulatory networks
- Coupling between intracellular state and cell phenotype

### Model Specification
- **MorpheusML**: Declarative XML format
- GUI-based model editor
- Built-in model library with published examples
- Model validation and consistency checks

## Directory Structure

```
morpheus/
├── src/
│   ├── core/                   # Core simulation engine
│   ├── plugins/                # CPM, PDE, ODE plugins
│   ├── gui/                    # Qt-based graphical interface
│   └── ...
├── 3rdparty/
│   ├── eigen/                  # Linear algebra library
│   └── tiny-process/           # Process management
├── build/
│   └── morpheus               # Compiled binary
├── examples/                   # Example MorpheusML models
└── ...
```

## Usage

### GUI
```bash
# Launch the Morpheus GUI
~/Documents/biotools/morpheus/build/morpheus
```

### Command Line
```bash
# Run a MorpheusML model from the command line
~/Documents/biotools/morpheus/build/morpheus -f model.xml
```

### Example MorpheusML Model
```xml
<MorpheusModel version="4">
  <Space>
    <Lattice class="square">
      <Size value="200 200 0"/>
    </Lattice>
  </Space>
  <CellTypes>
    <CellType name="cells" class="biological">
      <VolumeConstraint target="200" strength="1"/>
      <SurfaceConstraint target="60" strength="1"/>
    </CellType>
  </CellTypes>
  <CPM>
    <Interaction>
      <Contact type1="cells" type2="cells" value="5"/>
    </Interaction>
    <MonteCarloSampler stepper="edgelist">
      <MCSDuration value="1"/>
      <MetropolisKinetics temperature="5"/>
    </MonteCarloSampler>
  </CPM>
</MorpheusModel>
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | MorpheusML (`.xml`) | Declarative model specification |
| Input | SBML | Intracellular ODE models |
| Output | VTK | Lattice and field snapshots for 3D visualization |
| Output | TIFF, PNG | 2D visualization images |
| Output | CSV | Numerical data (cell properties, field values) |

## Build

```bash
cd ~/Documents/biotools/morpheus
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Dependencies
- CMake 3.x
- Qt5 (for GUI)
- Eigen (bundled or from local_deps)
- CVODE (for ODE integration)
- xtensor (fetched during build)

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **CompuCell3D** | Alternative CPM simulator (CC3D uses Python scripting; Morpheus uses declarative XML) |
| **PhysiCell** | Alternative multicellular simulator (off-lattice agent-based vs. CPM) |
| **BETSE** | Bioelectric tissue modeling (complementary physics) |
| **COPASI** | SBML models from COPASI can be embedded as intracellular dynamics |
| **Vivarium** | Can be composed as a process in multiscale simulations |

## License

BSD 3-Clause License

## References

- Staruschenko, S. et al. (2014). "Morpheus: a user-friendly modeling environment for multiscale and multicellular systems biology." *Bioinformatics*, 30(9):1331-1332.
- Deutsch, A. & Dormann, S. (2005). "Cellular Automaton Modeling of Biological Pattern Formation." Birkhauser.
