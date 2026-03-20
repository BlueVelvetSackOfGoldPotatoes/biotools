# cpp_ports — C++17 Header-Only Ports of Python Packages

This directory contains **complete C++17 header-only reimplementations** of 10 Python packages from the biotools pipeline. Every port compiles with any C++17-compliant compiler and has **zero external dependencies** beyond the standard library.

## What It Does

These ports translate the core algorithms and data structures from Python/PyTorch packages into standalone C++ headers. This enables:

- **Native C++ integration**: Use BETSE bioelectric simulations, Boltz structure prediction, ESM protein language models, DeepXDE PINNs, Modulus neural operators, PyG graph neural networks, and Vivarium composition directly from C++ code
- **No Python runtime**: Run these algorithms without a Python interpreter or any Python dependencies
- **Single-header inclusion**: Each port is one or a few `#include`-able headers — no build system, no linking, no configuration
- **Standard library only**: No Eigen, no Boost, no BLAS — every port implements its own linear algebra, FFT, tensor operations, and neural network primitives from scratch

## Ports Overview

| Port | Headers | Lines | Description |
|------|---------|-------|-------------|
| **betse/** | 14 | ~9,300 | Bioelectric tissue simulation (GHK flux, 50+ ion channels, gap junctions, ECM, GRN, organelles, deformation) |
| **modulus/** | 9 | ~6,600 | Physics-informed ML (FNO, DeepONet, AFNO, MeshGraphNet, GraphCast, SIREN, UNet, 25+ activations) |
| **pyg/** | 8 | ~5,100 | Graph neural networks (30+ convolution layers, pooling, normalization, GAE/VGAE, SchNet, DimeNet) |
| **deepxde/** | 8 | ~4,400 | PINNs for PDEs (tape-based AD, feedforward/ResNet/SIREN, geometry, boundary conditions, losses) |
| **boltz/** | 5 | ~4,500 | Protein structure prediction (MSA module, Pairformer, diffusion, IPA, confidence heads, PDB I/O) |
| **esm/** | 2 | ~3,800 | Protein language models (ESM-1/2, MSA Transformer, ESMFold, GVP inverse folding) |
| **vivarium/** | 3 | ~3,000 | Multi-scale composition (Store tree, Process, Engine, Emitter, Composer, division) |
| **betsee/** | 1 + React | ~2,100 | BETSE GUI (Qt5 desktop app + React web frontend) |
| **cellblender/** | 1 + React | ~1,600 | MCell GUI (data model, MDL generator + React web frontend) |
| **neuropil_tools/** | 1 | ~1,400 | Neuropil analysis (mesh I/O, region analysis, spine morphometry, contact detection) |

**Total: 52 headers, ~41,800 lines of C++17**

## Usage

Simply include the headers in your C++ project:

```cpp
// Bioelectric simulation
#include "betse/betse.h"

// Protein structure prediction
#include "boltz/boltz.h"

// Physics-informed neural networks
#include "deepxde/deepxde.h"

// Protein language models
#include "esm/esm.h"

// Physics-informed ML (FNO, DeepONet, etc.)
#include "modulus/modulus.h"

// Graph neural networks
#include "pyg/pyg.h"

// Multi-scale simulation composition
#include "vivarium/vivarium.h"

// Neuropil ultrastructure analysis
#include "neuropil_tools/neuropil_tools.h"
```

Compile with C++17:
```bash
g++ -std=c++17 -O2 -I ~/Documents/biotools/cpp_ports my_program.cpp -o my_program
```

No CMake, no Makefile, no linking — just include and compile.

## React Frontend Applications

Two ports include modern React web frontends as alternatives to the original Python desktop GUIs:

### BETSEE React App (`betsee/react-app/`)
```bash
cd ~/Documents/biotools/cpp_ports/betsee/react-app
npm install && npm run dev
```
Components: GeneralSettings, SpaceSettings, IonSettings, ChannelSettings, PhysicsSettings, TissueSettings, NetworkSettings, ExportSettings, SimulationControl, ResultsViewer, LogViewer, FileManager

### CellBlender React App (`cellblender/react-app/`)
```bash
cd ~/Documents/biotools/cpp_ports/cellblender/react-app
npm install && npm run dev
```
Components: MoleculeEditor, ReactionEditor, ReleaseSiteEditor, SurfaceClassEditor, GeometryViewer, MDLPreview, SimulationPanel, ResultsViewer, ParameterSweep, ProjectManager

## Design Principles

1. **Header-only**: Every port is fully contained in `.h` files — no `.cpp` files, no compiled libraries
2. **C++17 standard**: Uses `std::any`, `std::optional`, `std::variant`, structured bindings, fold expressions, and other C++17 features
3. **Zero dependencies**: All linear algebra (matrix multiply, decompositions, eigenvalues), FFT (Cooley-Tukey), random number generation, and file I/O are implemented from scratch
4. **Faithful to originals**: Algorithms, class names, and APIs follow the Python originals as closely as C++ idioms allow
5. **Self-contained tensors**: Each port that needs tensor operations includes its own dense tensor class with autograd where needed

## Directory Structure

```
cpp_ports/
├── betse/                      # 14 headers
│   ├── betse.h                 # Master include
│   ├── betse_types.h           # Core types, Vec2, matrices, ions, channels
│   ├── betse_channels.h        # 50+ ion channel kinetics
│   ├── betse_math.h            # FD operators, Poisson solvers, HH decomposition
│   ├── betse_physics.h         # Electroosmosis, deformation, osmotic pressure
│   ├── betse_networks.h        # Gene regulatory networks
│   ├── betse_organelles.h      # ER, mitochondria, microtubules, nucleus
│   ├── betse_tissue.h          # Tissue profiles, events, ion presets
│   ├── betse_phase.h           # Simulation phase runner, callbacks
│   ├── betse_solver.h          # FAST and FULL solvers, CG, SOR
│   ├── betse_config.h          # Configuration structures
│   ├── betse_enums.h           # Enumerations
│   ├── betse_export.h          # CSV/visualization data export
│   └── betse_util.h            # Geometry, interpolation, signal, Voronoi
├── modulus/                    # 9 headers
├── pyg/                        # 8 headers
├── deepxde/                    # 8 headers
├── boltz/                      # 5 headers
├── esm/                        # 2 headers
├── vivarium/                   # 3 headers
├── betsee/                     # 1 header + react-app/
├── cellblender/                # 1 header + react-app/
└── neuropil_tools/             # 1 header
```

## License

Each port follows the license of its original Python package.
