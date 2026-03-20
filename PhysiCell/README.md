# PhysiCell — Open Source Physics-Based Cell Simulator

PhysiCell is an open-source, **agent-based framework** for simulating multicellular systems in 3D tissue environments. Each cell is an independent software agent with its own phenotype, signaling state, and mechanical interactions. The microenvironment is modeled on a regular Cartesian mesh with diffusing substrates (oxygen, glucose, chemokines, drugs, etc.).

## What It Does

PhysiCell simulates the life cycle and collective behavior of large populations of cells in 3D tissue:

- **Cell mechanics**: Adhesion and repulsion via potential functions; cells push, pull, and rearrange
- **Cell cycle**: Configurable cell cycle models (Ki67, flow cytometry-based, live/dead) with growth, division, and death
- **Cell death**: Apoptosis and necrosis with distinct morphological and kinetic characteristics
- **Microenvironment**: 3D diffusion of substrates (oxygen, glucose, signaling molecules, drugs) on a Cartesian mesh
- **Chemotaxis**: Cells migrate along chemical gradients
- **Intracellular signaling**: Hooks for Boolean signaling networks, SBML ODE models, and custom rule sets
- **Custom rules**: User-defined cell behavior rules for phenotype switching, migration, and interaction

### Applications
- Tumor growth and invasion
- Immune cell interactions (T-cell killing, macrophage recruitment)
- Drug delivery and pharmacokinetics
- Tissue engineering and wound healing
- Microbial communities (biofilms)
- COVID-19 tissue infection dynamics

## Biological Scale

**Cellular / Tissue** — simulates 10s to millions of cells in 3D (1 μm – 10 mm), with substrate diffusion fields on a regular Cartesian mesh.

## Key Algorithms

### Biomechanics
- Cell-cell adhesion and repulsion via potential functions
- Velocity Verlet integration for cell positions
- Basement membrane interactions

### Diffusion
- Finite-volume method on a regular 3D Cartesian mesh
- Dirichlet and Neumann boundary conditions
- Multiple diffusing substrates with configurable decay rates

### Cell Phenotype
- Customizable cell cycle models (Ki67 basic/advanced, flow cytometry, live/dead)
- Death models (apoptosis with deterministic timing, necrosis with swelling/lysis)
- Secretion, uptake, and net export of substrates
- Motility with persistence time and chemotactic bias
- Cell-cell adhesion and repulsion strengths (type-specific)

### Intracellular
- Boolean signaling network integration (MaBoSS)
- SBML ODE model integration
- Custom rules (CSV-based or programmatic)

### Parallel
- OpenMP-based multithreading for large-scale simulations

## Directory Structure

```
PhysiCell/
├── core/                       # Core simulation engine
│   ├── PhysiCell_cell.cpp/h    # Cell agent implementation
│   ├── PhysiCell_phenotype.cpp/h # Cell phenotype models
│   ├── PhysiCell_utilities.cpp/h # Diffusion and microenvironment
│   └── ...
├── modules/                    # Optional modules
│   ├── PhysiCell_pathology.cpp # Virtual pathology
│   ├── PhysiCell_SVG.cpp       # SVG output
│   └── ...
├── addons/
│   └── PhysiMeSS/              # Microenvironment structures (fibers)
├── config/
│   └── PhysiCell_settings.xml  # Default configuration
├── sample_projects/            # Example projects
│   ├── biorobots/
│   ├── cancer_biorobots/
│   ├── heterogeneity/
│   └── ...
├── sample_projects_intracellular/
│   └── boolean/                # Boolean network examples
├── tests/                      # Test suite
└── Makefile
```

## Usage

```bash
cd ~/Documents/biotools/PhysiCell

# Build the default project
make

# Run with configuration file
./project config/PhysiCell_settings.xml

# Build a sample project
make reset
make cancer_biorobots-sample
make
./cancer_biorobots
```

### Configuration (PhysiCell_settings.xml)
```xml
<PhysiCell_settings version="devel-version">
  <domain>
    <x_min>-500</x_min><x_max>500</x_max>
    <y_min>-500</y_min><y_max>500</y_max>
    <z_min>-10</z_min><z_max>10</z_max>
    <dx>20</dx><dy>20</dy><dz>20</dz>
  </domain>
  <cell_definitions>
    <cell_definition name="default" ID="0">
      <phenotype>
        <cycle code="5" name="live">
          <phase_durations units="min">
            <duration index="0" fixed_duration="false">300</duration>
          </phase_durations>
        </cycle>
      </phenotype>
    </cell_definition>
  </cell_definitions>
</PhysiCell_settings>
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | XML | Simulation settings (domain, cell types, substrates) |
| Input | C++ module | Custom cell behaviors and project-specific code |
| Input | CSV | Custom rules for cell behavior |
| Output | `.mat` | MultiCellDS snapshots (cell data) |
| Output | SVG | 2D cross-section visualization |
| Output | XML | Microenvironment data |

## Build

```bash
cd ~/Documents/biotools/PhysiCell
make    # Uses Makefile (g++ with OpenMP)
```

### Dependencies
- C++11 compiler (g++ or clang++)
- OpenMP (for parallel execution)
- Python 3.x (for PhysiCell Studio, optional)

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **BioNetGen / COPASI** | SBML intracellular models imported into PhysiCell cells |
| **CompuCell3D** | Alternative multicellular simulator (lattice-based CPM vs. off-lattice agents) |
| **Morpheus** | Alternative multicellular simulator with CPM and declarative XML |
| **BETSE** | Bioelectric tissue modeling (complementary physics) |
| **Tissue Forge** | Alternative particle-based tissue simulator (real-time interactive) |
| **GAMer / mesh_tools** | Mesh generation for simulation domain boundaries |
| **Vivarium** | Can be composed as a process in multiscale simulations |

## License

BSD 3-Clause License

## References

- Ghaffarizadeh, A. et al. (2018). "PhysiCell: An Open Source Physics-Based Cell Simulator for 3-D Multicellular Systems." *PLOS Computational Biology*, 14(2):e1005991.
- Ozik, J. et al. (2018). "High-throughput cancer hypothesis testing with an integrated PhysiCell-EMEWS workflow." *BMC Bioinformatics*, 19(Suppl 18):483.
