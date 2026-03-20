# Biotools — Multi-Scale Computational Biology Pipeline

A comprehensive collection of **computational biology simulation tools** spanning the full hierarchy of biological organization — from individual atoms and molecules up through subcellular reaction networks, whole cells, and multicellular tissues. This pipeline integrates 16 built C++ repositories, 10 C++ header-only ports, 2 React frontend applications, and 7 Python packages in a unified workspace.

## Overview

The biotools pipeline enables researchers to:

- **Simulate molecular dynamics** at atomistic resolution (GROMACS, OpenMM)
- **Predict protein structures** from sequence (Boltz, ESM)
- **Model subcellular reaction-diffusion** kinetics (Smoldyn, MCell, ReaDDy, COPASI, BioNetGen, NFSim)
- **Simulate multicellular tissue** bioelectrics, mechanics, and growth (BETSE, PhysiCell, CompuCell3D, Morpheus, Tissue Forge)
- **Bridge scales** with a composable simulation framework (Vivarium)
- **Build physics-informed neural network surrogates** (DeepXDE, Modulus/PhysicsNeMo)
- **Operate on graph-structured biological data** (PyTorch Geometric)
- **Process, repair, and analyze biological meshes** (GAMer, mesh_tools, neuropil_tools)
- **Configure and visualize simulations** through GUIs (BETSEE, CellBlender)

## Scale Hierarchy

```
ATOMS/MOLECULES           SUBCELLULAR               CELLULAR/TISSUE          SURROGATE/ML
(Angstroms – nm)          (nm – um)                 (um – mm)                (any scale)

  GROMACS ──┐                                                                DeepXDE (PINNs)
  OpenMM  ──┤─→ Atomistic    Smoldyn ────────┐                              Modulus (FNO,
  Boltz   ──┤   structures   MCell ──────────┤─→ Reaction-                    DeepONet,
  ESM     ──┘   & dynamics   ReaDDy ─────────┤   diffusion                   MeshGraphNet)
                              COPASI ─────────┤   dynamics                  PyG (GNN)
                              BioNetGen ──────┤
                              NFSim ──────────┘
                                      │
                                      ▼
                              PhysiCell ──────┐
                              CompuCell3D ────┤─→ Multicellular
                              Morpheus ───────┤   tissue dynamics
                              Tissue Forge ───┤
                              BETSE ──────────┘
                                      ▲
                                      │
                              Vivarium (bridging)
```

## Directory Structure

### Molecular / Atomistic Simulators
| Directory | Description | Language |
|-----------|-------------|----------|
| [gromacs/](gromacs/) | All-atom molecular dynamics engine (SIMD, GPU) | C++ |
| [openmm/](openmm/) | GPU-accelerated MD library with custom forces | C++/Python |
| [boltz/](boltz/) | Diffusion-based protein structure prediction | Python |
| [esm/](esm/) | Meta's protein language models (ESM-2, ESMFold) | Python |

### Subcellular Reaction-Diffusion
| Directory | Description | Language |
|-----------|-------------|----------|
| [Smoldyn/](Smoldyn/) | Spatial stochastic simulator (Smoluchowski dynamics) | C |
| [mcell/](mcell/) | 3D Monte Carlo reaction-diffusion in EM geometries | C/C++ |
| [readdy/](readdy/) | Particle-based reaction-diffusion with soft potentials | C++ |
| [COPASI/](COPASI/) | Biochemical network simulator (ODE/SSA/hybrid) | C++ |
| [bionetgen/](bionetgen/) | Rule-based modeling language & network generator | Perl/C++ |
| [nfsim/](nfsim/) | Network-free stochastic simulator | C++ |
| [nfsimCInterface/](nfsimCInterface/) | C API wrapper for NFSim | C/C++ |
| [libbng/](libbng/) | Lightweight embeddable BioNetGen library | C++ |

### Cellular / Tissue Simulators
| Directory | Description | Language |
|-----------|-------------|----------|
| [PhysiCell/](PhysiCell/) | Agent-based 3D multicellular simulator | C++ |
| [CompuCell3D/](CompuCell3D/) | Cellular Potts Model framework | C++/Python |
| [morpheus/](morpheus/) | Multicellular modeling (CPM + PDE + ODE + GUI) | C++ |
| [tissue-forge/](tissue-forge/) | Interactive particle-based tissue simulator | C++ |
| [betse/](betse/) | Bioelectric tissue simulation engine | Python |

### Bridging & Composition
| Directory | Description | Language |
|-----------|-------------|----------|
| [vivarium-core/](vivarium-core/) | Composable multi-scale simulation framework | Python |

### Machine Learning / Surrogates
| Directory | Description | Language |
|-----------|-------------|----------|
| [deepxde/](deepxde/) | Physics-informed neural networks for PDEs | Python |
| [modulus/](modulus/) | NVIDIA PhysicsNeMo (FNO, DeepONet, MeshGraphNet) | Python |
| [pytorch_geometric/](pytorch_geometric/) | Graph neural network library | Python |

### Mesh & Geometry
| Directory | Description | Language |
|-----------|-------------|----------|
| [gamer/](gamer/) | Geometry-preserving adaptive mesh generation | C++ |
| [mesh_tools/](mesh_tools/) | 60+ mesh processing command-line tools | C |
| [neuropil_tools/](neuropil_tools/) | Neuropil ultrastructure analysis from EM | Python |

### GUI / Visualization
| Directory | Description | Language |
|-----------|-------------|----------|
| [betsee/](betsee/) | Graphical frontend for BETSE | Python/Qt |
| [cellblender/](cellblender/) | Blender addon for MCell simulations | Python |

### C++ Header-Only Ports
| Directory | Description | Headers | Lines |
|-----------|-------------|---------|-------|
| [cpp_ports/](cpp_ports/) | C++17 ports of 10 Python packages | 52 | ~41,800 |

Includes ports of: BETSE, Modulus, PyG, DeepXDE, Boltz, ESM, Vivarium, BETSEE (+ React), CellBlender (+ React), neuropil_tools. All are header-only with zero external dependencies.

### Build Dependencies
| Directory | Description |
|-----------|-------------|
| [copasi-dependencies/](copasi-dependencies/) | Build dependencies for COPASI (libSBML, clapack, etc.) |
| [local_deps/](local_deps/) | Shared dependency builds (Eigen, Assimp, nlohmann/json, libxml2) |
| [VTK/](VTK/) | Visualization Toolkit (mesh I/O, rendering) |
| [mcell_tests/](mcell_tests/) | MCell & CellBlender test suite |
| [mcell_tools/](mcell_tools/) | MCell build & utility scripts |

## Quick Start

### Environment Setup
```bash
# Activate the shared Python environment
source ~/Documents/biotools/.venv/bin/activate
```

### Run Key Tools

```bash
# Molecular dynamics (GROMACS)
~/Documents/biotools/gromacs/build/bin/gmx mdrun -deffnm simulation

# Protein structure prediction (Boltz)
boltz predict sequence.fasta --output_dir results/

# Biochemical network simulation (COPASI)
~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE model.cps

# Spatial stochastic simulation (Smoldyn)
~/Documents/biotools/Smoldyn/build/smoldyn config.txt

# Multicellular simulation (PhysiCell)
cd ~/Documents/biotools/PhysiCell && make && ./project config/PhysiCell_settings.xml

# Rule-based modeling (BioNetGen)
perl ~/Documents/biotools/bionetgen/bng2/BNG2.pl model.bngl

# Network-free simulation (NFSim)
~/Documents/biotools/nfsim/build/NFsim -xml model.xml -sim 100 -oSteps 1000

# Multicellular CPM (Morpheus)
~/Documents/biotools/morpheus/build/morpheus -f model.xml

# Bioelectric tissue simulation (BETSE)
betse sim my_simulation.yaml

# Mesh processing
~/Documents/biotools/mesh_tools/mesh2mcell/mesh2mcell input.obj output.mdl
```

### Use C++ Ports (No Build Required)
```cpp
#include "betse/betse.h"          // Bioelectric simulation
#include "boltz/boltz.h"          // Structure prediction
#include "deepxde/deepxde.h"      // PINNs
#include "modulus/modulus.h"       // Physics-ML
#include "pyg/pyg.h"              // Graph neural networks
#include "esm/esm.h"              // Protein language models
#include "vivarium/vivarium.h"    // Multi-scale composition
```

```bash
g++ -std=c++17 -O2 -I ~/Documents/biotools/cpp_ports my_program.cpp -o my_program
```

## How Tools Connect Across Scales

### Molecular → Subcellular
- GROMACS/OpenMM produce equilibrium structures and force parameters
- Boltz/ESM predict protein structures from sequence, providing coordinates for MD or binding site geometries for MCell
- Coarse-grained MD parameters inform ReaDDy particle potentials

### Subcellular → Cellular
- COPASI/BioNetGen ODE/SSA models become intracellular submodels inside PhysiCell cells
- NFSim (via nfsimCInterface/libbng) provides network-free intracellular reactions inside MCell geometries
- Smoldyn/MCell reaction-diffusion outputs inform effective rates for tissue-scale models

### Cellular → Tissue
- PhysiCell, CompuCell3D, and Morpheus simulate multicellular dynamics at tissue scale
- BETSE models bioelectric pattern formation across cell sheets
- Tissue Forge provides interactive real-time tissue mechanics

### Bridging
- Vivarium composes processes at different scales into a single simulation with a shared state tree

### Surrogate/ML
- DeepXDE learns PDE solutions (approximating fields from GROMACS, COPASI, or PhysiCell)
- Modulus provides FNO/DeepONet surrogates for expensive PDE solvers
- PyG operates on graph-structured biological data (molecular graphs, cell networks, meshes)

### Mesh/Geometry
- GAMer generates high-quality meshes from EM reconstructions or molecular surfaces
- mesh_tools converts, refines, repairs, and analyzes meshes for MCell
- neuropil_tools processes neuronal ultrastructure from EM reconstructions

## Built Artifacts

### Executables
| Tool | Binary |
|------|--------|
| GROMACS | `gromacs/build/bin/gmx` |
| Smoldyn | `Smoldyn/build/smoldyn` |
| COPASI (CLI) | `COPASI/build/copasi/CopasiSE/CopasiSE` |
| Morpheus | `morpheus/build/morpheus` |
| NFSim | `nfsim/build/NFsim` |
| BioNetGen | `bionetgen/bng2/BNG2.pl` |
| run_network | `bionetgen/bng2/Network3/bin/run_network` |
| BETSE | `.venv/bin/betse` |
| mesh_tools | `mesh_tools/` (60 executables in subdirectories) |

### Shared Libraries
| Library | Path |
|---------|------|
| GROMACS | `gromacs/build/lib/libgromacs.so` |
| OpenMM | `openmm/build/libOpenMM.so` |
| ReaDDy | `readdy/build/libreaddy.so` |
| Tissue Forge | `tissue-forge/build/Release/lib/libtissue-forge.so` |
| GAMer | `gamer/build/lib/libgamer.so` |
| libbng | `libbng/build/bng/liblibbng.a` |
| nfsimCInterface | `nfsimCInterface/build/libnfsim_c.so` |
| CompuCell3D | `CompuCell3D/build/core/CompuCell3D/libCC3DCompuCellLib.so` |

### Python Packages (in .venv/)
| Package | Description |
|---------|-------------|
| betse | Bioelectric tissue simulation |
| boltz | Protein structure prediction |
| deepxde | Physics-informed neural networks |
| esm | Protein language models |
| torch_geometric | Graph neural networks |
| physicsnemo | NVIDIA PhysicsNeMo/Modulus |
| vivarium | Multi-scale simulation framework |

## Total Scale Coverage

| Biological Scale | Tools | Range |
|-----------------|-------|-------|
| Atomistic/Molecular | GROMACS, OpenMM, Boltz, ESM | 1 A – 100 nm |
| Subcellular | Smoldyn, MCell, ReaDDy, COPASI, BioNetGen, NFSim | 1 nm – 10 um |
| Cellular/Tissue | BETSE, PhysiCell, CompuCell3D, Morpheus, Tissue Forge | 1 um – 10 mm |
| Bridging | Vivarium | All scales |
| Surrogate/ML | DeepXDE, Modulus, PyG | Any scale |
| Mesh/Geometry | GAMer, mesh_tools, neuropil_tools | Any scale |
| GUI | BETSEE, CellBlender | Config/Viz |

## Further Reading

See [BIOTOOLS_REFERENCE.md](BIOTOOLS_REFERENCE.md) for the comprehensive technical reference with detailed API documentation, header contents, and algorithm descriptions for every tool in the pipeline.
