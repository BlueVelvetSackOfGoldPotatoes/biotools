# BETSE — BioElectric Tissue Simulation Engine

BETSE is an open-source, cross-platform Python application for simulating **bioelectric phenomena** in two-dimensional cell clusters. It models the interplay between ion channel activity, gap junction communication, electroosmotic flows, mechanical deformation, and gene regulatory networks across interconnected cells.

## What It Does

BETSE simulates the collective bioelectric behavior of 2D cell networks — the voltages, ion concentrations, and current flows that emerge when populations of cells communicate through gap junctions and respond to their ionic environment. This is critical for understanding:

- **Embryonic patterning**: How bioelectric gradients guide morphogenesis and tissue organization
- **Wound healing**: How injury currents propagate and drive regenerative responses
- **Cancer**: How bioelectric disruptions may contribute to tumorigenesis
- **Ion channel diseases (channelopathies)**: How mutations in channels affect tissue-scale electrical behavior

## Biological Scale

Cellular / Tissue — operates at the scale of 2D epithelial cell sheets, from tens to thousands of cells (1 μm – 10 mm).

## Key Scientific Features

### Electrophysiology
- **Goldman-Hodgkin-Katz (GHK)** electrodiffusion flux equation for membrane transport
- **Nernst-Planck** transport through membranes and gap junctions
- **Hodgkin-Huxley** gating kinetics for 50+ voltage-gated ion channel types (Nav, Kv, Cav, HCN, Morris-Lecar, TRP families)
- Full complement of active transporters: Na/K-ATPase, Ca-ATPase, V-ATPase, H/K-ATPase, NCX exchanger, NKCC/KCC co-transporters

### Cell & Tissue Modeling
- **Voronoi cell cluster generation** with Lloyd relaxation for realistic tissue geometry
- **Gap junction** communication between neighboring cells (configurable connectivity)
- **Extracellular matrix (ECM)** PDE transport for full spatiotemporal ion dynamics
- Configurable ion profiles (mammalian, amphibian, custom)

### Subcellular Organelles
- **Endoplasmic reticulum** Ca²⁺ dynamics (IP3 receptors, ryanodine receptors, SERCA pumps, CICR)
- **Mitochondrial** membrane potential modeling (electron transport chain, MCU, NCLX)
- **Microtubule** electrophoretic alignment (Broersma drag, rotational diffusion)
- **Nuclear** transport through nuclear pore complexes

### Physics
- **Electroosmotic flow** (Helmholtz-Smoluchowski equation)
- **Tissue deformation** (steady-state and time-dependent) with galvanotropism
- **Osmotic pressure** modeling
- **Helmholtz-Hodge decomposition** for divergence-free flow fields

### Gene Regulatory Networks (GRNs)
- Arbitrary molecule species, reactions, transporters, and modulators
- Network-driven ion channel and pump expression
- Integration of GRN dynamics with bioelectric state

### Solvers
- **Equivalent circuit (FAST)** mode for rapid simulations
- **Full Nernst-Planck (FULL)** mode with complete ECM transport
- Poisson solvers: Jacobi, SOR, Conjugate Gradient, graph-based

## Installation

BETSE is a Python package. Within this biotools collection it is installed in the shared virtual environment:

```bash
# Activate the environment
source ~/Documents/biotools/.venv/bin/activate

# Run BETSE
betse --help
```

### From Source

```bash
cd ~/Documents/biotools/betse
pip install -e .
```

### Dependencies
- Python 3.x
- NumPy, SciPy, Matplotlib
- Pillow, dill, ruamel.yaml
- Optional: NetworkX (for network analysis), Graphviz (for visualization)

## Usage

BETSE uses a YAML configuration file to define all simulation parameters:

```bash
# Generate a sample configuration file
betse config my_sim.yaml

# Initialize the seed (cell cluster geometry)
betse seed my_sim.yaml

# Initialize ion concentrations and membrane voltages
betse init my_sim.yaml

# Run the full simulation
betse sim my_sim.yaml

# Plot results
betse plot init my_sim.yaml
betse plot sim my_sim.yaml
```

## Input/Output

- **Input**: YAML configuration files defining cell geometry, ion channels, tissue profiles, simulation parameters
- **Output**: CSV data files, PNG/SVG plots, animations of membrane voltage, ion concentrations, current flows, and other bioelectric quantities

## Connections to Other Biotools

- **BETSEE** provides a graphical frontend for configuring and visualizing BETSE simulations
- **cpp_ports/betse/** contains a complete C++17 header-only reimplementation (14 headers, ~9,300 lines)
- Results can inform higher-scale tissue models (PhysiCell, CompuCell3D) or be compared against physics-informed neural network surrogates (DeepXDE, Modulus)

## License

BSD 2-Clause License

## References

- Pietak, A. & Levin, M. (2016). "Exploring Instructive Physiological Signaling with the Bioelectric Tissue Simulation Engine." *Frontiers in Bioengineering and Biotechnology*, 4:55.
- Pietak, A. & Levin, M. (2017). "Bioelectric gene and reaction networks: computational modelling of genetic, biochemical, and bioelectrical dynamics in pattern regulation." *Journal of The Royal Society Interface*, 14(134).
