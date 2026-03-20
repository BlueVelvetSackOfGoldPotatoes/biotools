# CellBlender — Blender Addon for MCell Simulations

CellBlender is a **Blender addon** that provides a complete graphical environment for creating, running, and analyzing **MCell** (Monte Carlo Cell) reaction-diffusion simulations. It leverages Blender's powerful 3D modeling tools to bridge biological geometry and particle-based simulation.

## What It Does

CellBlender enables researchers to:

- **Model 3D cellular geometries** using Blender's mesh tools — import from electron microscopy reconstructions, create idealized shapes, or sculpt custom geometries
- **Define molecular species** with diffusion coefficients, colors, and volume/surface designations
- **Specify reactions** between molecular species with kinetic rate constants
- **Place release sites** for initial molecule distributions (point, surface, volume, or region-based)
- **Configure surface classes** defining membrane properties (absorptive, reflective, transparent, concentration clamp)
- **Generate MDL files** (MCell Model Description Language) automatically from the visual setup
- **Run MCell simulations** directly from within Blender with progress monitoring
- **Visualize results** including animated molecule trajectories and reaction count time series
- **Perform parameter sweeps** over rate constants, concentrations, or geometry parameters

## Biological Scale

**GUI / Configuration** — configures MCell simulations at the subcellular scale (nanometer to micrometer geometries: synaptic clefts, dendritic spines, organelle membranes, entire cell compartments).

## Key Components

### Geometry Management
- Import mesh geometries from OBJ, STL, OFF, and other 3D formats
- Define **surface regions** on meshes (e.g., postsynaptic density, synaptic active zone, spine head)
- Assign surface classes to regions with specific molecular interaction rules
- Boolean mesh operations for creating complex compartmentalized geometries

### Molecule & Reaction Editor
- Define volume molecules (diffuse freely in 3D) and surface molecules (diffuse on 2D membrane surfaces)
- Specify unimolecular and bimolecular reactions with forward/reverse rate constants
- Configure release patterns (instantaneous burst, constant rate, periodic, custom schedules)
- Define complex BioNetGen-style rule-based reactions

### MDL Generation
- Automatic generation of complete MCell MDL files from the visual data model
- Includes geometry, molecules, reactions, release sites, surface classes, observation outputs
- Live MDL preview panel for real-time inspection and debugging

### Simulation Control & Visualization
- Launch and monitor MCell runs from within Blender
- 3D molecule position visualization with species-specific colors
- Time series plots of reaction data (molecule counts over time)
- Parameter sweep configuration for batch exploration

## Directory Structure

```
cellblender/
├── __init__.py                 # Blender addon registration
├── cellblender_main.py         # Main addon panel
├── cellblender_molecules.py    # Molecule species editor
├── cellblender_reactions.py    # Reaction editor
├── cellblender_release.py      # Release site configuration
├── cellblender_surfaces.py     # Surface class editor
├── cellblender_objects.py      # Geometry/object management
├── data_model.py               # Internal data model
├── io_mesh_mcell_mdl/          # MDL import/export
├── parameter_system/           # Parameter management
├── bng/                        # BioNetGen integration
└── ...
```

## Installation

CellBlender is installed as a Blender addon:

1. In Blender: **Edit > Preferences > Add-ons > Install**
2. Navigate to the `cellblender/` directory and select the addon
3. Enable the "CellBlender" addon in the addon list

The source is located at:
```
~/Documents/biotools/cellblender/
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | The simulation engine that CellBlender configures and drives |
| **mesh_tools** | Mesh conversion, repair, and analysis utilities for preparing geometries |
| **neuropil_tools** | Processes EM reconstruction data into CellBlender-compatible meshes |
| **GAMer** | High-quality tetrahedral mesh generation from biological geometries |
| **NFSim / BioNetGen** | Rule-based intracellular reactions executed within MCell simulations |
| **VTK** | Visualization and mesh I/O support |
| **cpp_ports/cellblender/** | C++17 header-only port of the data model + MDL generator, plus a React web frontend |

## License

GPL (as a Blender addon, following Blender's licensing)

## References

- Czech, D.R. et al. (2019). "CellBlender: A tool for creating MCell simulations." *Frontiers in Neuroinformatics*.
- Kerr, R.A. et al. (2008). "Fast Monte Carlo Simulation Methods for Biological Reaction-Diffusion Systems in Solution and on Surfaces." *SIAM J. Sci. Comput.*, 30(6):3126-3149.
