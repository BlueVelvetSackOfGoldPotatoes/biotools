# Biotools

Multi-scale computational biology pipeline, C++ ML benchmark suite, and Turing morphogenesis platform.

This repository contains three major projects and a collection of reference documentation for 30+ computational biology tools:

1. **cpp_ports/** — 10 header-only C++17 reimplementations of Python computational biology packages (~42K lines), plus 2 React web frontends
2. **substrate/** — C++17 ML benchmark suite with 26 benchmarks across 17 model families, including a biological cell simulation engine
3. **turing_cell_ring/** — Turing morphogenesis paper-example platform (C++ backend + React frontend + LaTeX report)
4. **Tool documentation** — In-depth README.md files for 30+ external computational biology tools used in the pipeline

---

## Repository Structure

```
biotools/
│
├── README.md                       # This file
├── BIOTOOLS_REFERENCE.md           # Comprehensive technical reference for the full pipeline
├── .gitignore
│
├── cpp_ports/                      # C++17 header-only ports of 10 Python packages
│   ├── README.md
│   ├── betse/                      # Bioelectric tissue simulation (14 headers, ~9,300 lines)
│   │   ├── betse.h                 # Master include — Simulator class, SimConfig, SimState
│   │   ├── betse_channels.h        # 50+ voltage-gated ion channel models (Nav, Kv, Cav, HCN, etc.)
│   │   ├── betse_math.h            # FD operators, Poisson solvers, Helmholtz-Hodge decomposition
│   │   ├── betse_physics.h         # Electroosmotic flow, deformation, osmotic pressure
│   │   ├── betse_networks.h        # Gene regulatory networks, molecule transport, modulators
│   │   ├── betse_organelles.h      # ER (IP3R/RyR/SERCA), mitochondria, microtubules
│   │   ├── betse_tissue.h          # Tissue profiles, wound events, GJ blocking
│   │   ├── betse_solver.h          # Fast (RC) and Full (Nernst-Planck) solvers
│   │   ├── betse_config.h          # Configuration structures (all YAML parameters)
│   │   ├── betse_phase.h           # Simulation phase runner, callbacks, CSV export
│   │   ├── betse_enums.h           # Enumerations (lattice type, picker type, etc.)
│   │   ├── betse_export.h          # Data snapshot and export pipeline
│   │   ├── betse_types.h           # Vec2, matrices, Ion enum, GHK flux, physical constants
│   │   ├── betse_util.h            # Voronoi generation, geometry, interpolation, signal functions
│   │   ├── CMakeLists.txt
│   │   └── main.cpp               # Test suite
│   │
│   ├── betsee/                     # BETSE GUI (C++ Qt5 desktop + React web frontend)
│   │   ├── betsee.h                # Qt5 GUI: main window, config tree, parameter editors, undo/redo
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   └── react-app/             # React web frontend
│   │       ├── src/
│   │       │   ├── App.tsx                              # Main layout with sidebar navigation
│   │       │   ├── components/
│   │       │   │   ├── config/
│   │       │   │   │   ├── general/GeneralSettings.tsx  # Time, temperature, solver parameters
│   │       │   │   │   ├── space/SpaceSettings.tsx      # Cell geometry, lattice, cluster preview
│   │       │   │   │   ├── ions/IonSettings.tsx         # Ion profiles, Nernst calculator
│   │       │   │   │   ├── channels/ChannelSettings.tsx # Channel types, gating parameters, curves
│   │       │   │   │   ├── tissue/TissueSettings.tsx    # Tissue profiles, Dm values
│   │       │   │   │   ├── network/NetworkSettings.tsx  # GRN molecules, reactions, transporters
│   │       │   │   │   ├── physics/PhysicsSettings.tsx  # Electroosmosis, deformation, pressure
│   │       │   │   │   ├── exports/ExportSettings.tsx   # CSV, plots, animations
│   │       │   │   │   ├── interventions/InterventionSettings.tsx  # Voltage clamps, cuts, GJ blocks
│   │       │   │   │   └── shared/ConfigWidgets.tsx     # Reusable form components with validation
│   │       │   │   ├── simulation/SimulationControl.tsx # Run/pause/stop, progress, phase pipeline
│   │       │   │   ├── results/ResultsViewer.tsx        # Vmem heatmap, ion time series, currents
│   │       │   │   ├── log/LogViewer.tsx                # Color-coded simulation log
│   │       │   │   ├── file/FileManager.tsx             # New/open/save, presets
│   │       │   │   └── sidebar/Sidebar.tsx              # Navigation sidebar
│   │       │   ├── hooks/useSimulation.ts               # State management with localStorage persistence
│   │       │   ├── types/simulation.ts                  # TypeScript type definitions
│   │       │   ├── types/presets.ts                     # 7 preset configurations
│   │       │   └── utils/
│   │       │       ├── simulation-engine.ts             # Goldman-equation mock results engine
│   │       │       ├── config-persistence.ts            # YAML serialization, localStorage CRUD
│   │       │       ├── validation.ts                    # Input validation for all config fields
│   │       │       └── format.ts                        # Number formatting utilities
│   │       └── package.json
│   │
│   ├── modulus/                    # NVIDIA PhysicsNeMo (9 headers, ~6,600 lines)
│   │   ├── modulus.h               # Tensor with FFT, Linear, MLP, FNO1D, DeepONet, Adam, Trainer
│   │   ├── modulus_activations.h   # 25+ activation functions (GELU, SiLU, Mish, Stan, etc.)
│   │   ├── modulus_models.h        # FNO2D/3D, AFNO, MeshGraphNet, GraphCast, UNet, Pix2Pix, SIREN, RNNs
│   │   ├── modulus_loss.h          # MSE, L1, Huber, PDE residual, boundary, integral losses
│   │   ├── modulus_geometry.h      # Box, Sphere, Cylinder, CSG, marching cubes, STL I/O
│   │   ├── modulus_graph.h         # Mesh graph construction, kNN, Delaunay, icosahedral mesh
│   │   ├── modulus_solver.h        # LR schedulers, gradient clipping, checkpointing, training loop
│   │   ├── modulus_datapipes.h     # Dataset, DataLoader, normalizers, augmentation
│   │   └── modulus_deploy.h        # Binary/NumPy export, ONNX builder, profiling
│   │
│   ├── pyg/                        # PyTorch Geometric (8 headers, ~5,100 lines)
│   │   ├── pyg.h                   # Tensor with autograd, Data/Batch, Linear, GRU, Adam/SGD, losses
│   │   ├── pyg_conv.h              # 45+ GNN convolutions (GCN, GAT, SAGE, GIN, Transformer, etc.)
│   │   ├── pyg_pool.h              # Global/TopK/SAG/Edge/ASA pooling, graclus, voxel grid
│   │   ├── pyg_norm.h              # GraphNorm, InstanceNorm, LayerNorm, PairNorm, etc.
│   │   ├── pyg_models.h            # GAE, VGAE, Node2Vec, GraphUNet, SchNet, DimeNet, GNNExplainer
│   │   ├── pyg_transforms.h        # Feature normalization, self-loops, positional encodings
│   │   ├── pyg_loader.h            # NeighborLoader, ClusterLoader, LinkNeighborLoader
│   │   └── pyg_dense.h             # Dense GCN/GAT/SAGE/GIN, DiffPool, MinCutPool
│   │
│   ├── deepxde/                    # Physics-Informed Neural Networks (8 headers, ~4,400 lines)
│   │   ├── deepxde.h               # Var (tape-based AD), FeedForward, PINNSolver, Adam, geometries
│   │   ├── deepxde_geometry.h      # Disk, Triangle, Polygon, Cuboid, Sphere, Cylinder, CSG, PointCloud
│   │   ├── deepxde_data.h          # PDEData, TimePDE, InversePDE, DataDriven, collocation management
│   │   ├── deepxde_nn.h            # ResNet, PFNN, SIREN, ModifiedMLP, DeepONet, FNO1D, MIONet
│   │   ├── deepxde_icbc.h          # Dirichlet, Neumann, Robin, Periodic, PointSet, Operator BCs
│   │   ├── deepxde_losses.h        # MSE, MAE, Huber, Sobolev, gradient-enhanced + SGD/RMSProp/L-BFGS
│   │   ├── deepxde_callbacks.h     # EarlyStopping, LRScheduler, Checkpoint, Resampler, MovieDumper
│   │   └── deepxde_all.h           # Convenience include-all
│   │
│   ├── boltz/                      # Protein structure prediction (5 headers, ~4,500 lines)
│   │   ├── boltz.h                 # MSA module, Pairformer, AtomTransformer, confidence heads, pipeline
│   │   ├── boltz_data.h            # PDB/mmCIF/A3M parsers, tokenization, featurization, cropping
│   │   ├── boltz_diffusion.h       # VP/VE/EDM noise schedules, Heun/Euler/SDE samplers, Kabsch SVD
│   │   ├── boltz_loss.h            # FAPE, smooth lDDT, distogram, confidence losses
│   │   └── boltz_training.h        # AlphaFold LR scheduler, EMA, AdamW, training loop
│   │
│   ├── esm/                        # Protein language models (2 headers, ~3,800 lines)
│   │   ├── esm.h                   # Alphabet, BatchConverter, MultiheadAttention, ESM-2, MSA Transformer, GVP
│   │   └── esm_extended.h          # Quaternion/Rigid, Triangle ops, IPA, ESMFold, structure module
│   │
│   ├── vivarium/                   # Multi-scale composition framework (3 headers, ~3,000 lines)
│   │   ├── vivarium_types.h        # Value, State, Schema, Topology, updaters, dividers, registry
│   │   ├── vivarium.h              # Store, Process, Step, Composer, Emitter (RAM/File), Engine
│   │   └── vivarium_processes.h    # GrowthRate, Timeline, Clock, Division, Injector
│   │
│   ├── cellblender/                # MCell GUI (C++ data model + React web frontend)
│   │   ├── cellblender.h           # DataModel, MDLWriter, MDLParser, ResultsParser, ParameterSweep
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp
│   │   └── react-app/
│   │       └── src/
│   │           ├── App.tsx                      # Tabbed interface (11 tabs)
│   │           ├── components/
│   │           │   ├── MoleculeEditor.tsx        # Define molecular species
│   │           │   ├── ReactionEditor.tsx        # Define reactions with rates
│   │           │   ├── ReleaseSiteEditor.tsx     # Configure molecule release
│   │           │   ├── SurfaceClassEditor.tsx    # Surface properties
│   │           │   ├── GeometryViewer.tsx        # 3D wireframe viewer with rotation/zoom
│   │           │   ├── MDLPreview.tsx            # Live MDL code preview
│   │           │   ├── SimulationPanel.tsx       # Run MCell, monitor progress
│   │           │   ├── ResultsViewer.tsx         # Time series plots with stddev bands
│   │           │   ├── ParameterSweep.tsx        # Parameter sweep configuration
│   │           │   ├── ModSurfRegions.tsx        # Surface region assignments
│   │           │   └── ProjectManager.tsx        # Save/load projects
│   │           ├── hooks/useDataModel.ts         # State management (40+ actions)
│   │           └── utils/
│   │               ├── mdlGenerator.ts           # Client-side MDL generation
│   │               ├── geometry.ts               # Mesh analysis (area, volume, topology)
│   │               ├── projectIO.ts              # JSON project save/load
│   │               └── defaults.ts               # Default object creation
│   │
│   └── neuropil_tools/             # Neuropil ultrastructure analysis (1 header, ~1,400 lines)
│       └── neuropil_tools.h        # Mesh I/O (OBJ/OFF/STL/MDL), area/volume, smoothing, contacts, spines
│
├── substrate/                      # C++17 ML benchmark suite
│   ├── README.md
│   ├── SPECS.md                    # Full repository specification
│   ├── Makefile                    # Build system (make benchmarks)
│   ├── build_all.sh
│   ├── core/                       # Shared runtime
│   │   ├── tensor/                 # Tensor type, CPU/CUDA matmul routing
│   │   ├── nn/                     # Layers: linear, activations, normalization, conv, embedding
│   │   ├── optim/                  # SGD, Adam, RMSProp + LR schedulers
│   │   ├── losses/                 # CE, MSE, BCE, NLL, KLDiv, Huber
│   │   ├── metrics/               # Classification, regression, clustering metrics
│   │   ├── data/                   # MNIST IDX loader, batching, sequence batching
│   │   ├── io/                     # RunLogger, run artifact writer
│   │   ├── online/                 # TrainableModel contract, continuous runtime
│   │   ├── benchmark/              # Discrete env framework, game engines (chess, go, connect4, etc.)
│   │   └── bio/                    # Biology runtime, parameter dynamics
│   ├── models/                     # 17 model families
│   │   ├── actor_critic/           # Actor-critic RL
│   │   ├── cells/                  # Biological cell simulation subsystem
│   │   │   ├── src/                # Cell models, cell circuit, cell tasks, tissue 2D, developmental
│   │   │   ├── cellengine/         # Full physics-based evolutionary cell engine (CUDA)
│   │   │   └── standalone_ogre/    # Optional OGRE 3D visualization
│   │   ├── cnn/                    # Convolutional neural networks
│   │   ├── rnn/                    # Recurrent neural networks
│   │   ├── lstm/                   # Long short-term memory
│   │   ├── transformer/           # Transformer architecture
│   │   ├── vit/                    # Vision Transformer
│   │   ├── gnn/                    # Graph neural networks
│   │   ├── diffusion/             # Diffusion models
│   │   ├── forests/               # Random forests
│   │   ├── trees/                  # Decision trees
│   │   ├── clustering/            # Clustering algorithms
│   │   ├── markov/                # Markov chains
│   │   ├── hebbian/               # Hebbian learning
│   │   ├── forward_forward/       # Forward-forward algorithm
│   │   ├── reinforcement/         # Reinforcement learning
│   │   └── hybrid/                # Model registry + TrainableModel adapter
│   ├── benchmarks/                 # 26 executable benchmark entrypoints
│   │   ├── benchmark_mlp.cpp
│   │   ├── benchmark_cnn.cpp
│   │   ├── benchmark_transformer.cpp
│   │   ├── benchmark_cellengine.cpp
│   │   ├── benchmark_games.cpp
│   │   └── ... (26 total)
│   ├── tests/                      # Test suite
│   ├── analytics/                  # Python plotting + Streamlit dashboard
│   ├── scripts/                    # Bio sweep, random search, smoke tests
│   ├── ui/                         # React + Node dashboard
│   │   ├── src/                    # React frontend source
│   │   └── api/                    # Node.js API server
│   ├── docs/                       # Architecture, code reviews, plugin guides
│   └── legacy/                     # Older standalone MLP app (reference only)
│
├── turing_cell_ring/               # Turing morphogenesis paper-example platform
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── backend/
│   │   ├── include/turing/         # C++ headers (model, simulator, analysis, types)
│   │   ├── src/                    # model.cpp, simulator.cpp, analysis.cpp, validate.cpp, json.cpp
│   │   └── tests/                  # Regression tests
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx             # Family/preset/engine selection, profile charts, heatmaps
│   │   │   └── components/         # React visualization components
│   │   └── package.json
│   ├── report/
│   │   ├── turing_ring_validation_report.tex   # LaTeX report
│   │   ├── turing_ring_validation_report.pdf   # Compiled report
│   │   ├── figures/                             # Hypothesis and expansion figures
│   │   └── generate_hypothesis_figures.py
│   └── results/                    # Saved JSON validation artifacts
│
├── BIOTOOLS_REFERENCE.md           # Comprehensive technical reference (965 lines)
│                                   # Covers all 16 C++ repos, 10 C++ ports, 2 React apps, 7 Python packages
│
└── <tool>/README.md                # Per-tool documentation (30 folders, README only)
    ├── betse/README.md             # Bioelectric tissue simulation
    ├── betsee/README.md            # BETSE GUI
    ├── bionetgen/README.md         # Rule-based modeling
    ├── boltz/README.md             # Protein structure prediction
    ├── cellblender/README.md       # MCell GUI (Blender addon)
    ├── CompuCell3D/README.md       # Cellular Potts Model
    ├── COPASI/README.md            # Biochemical network simulator
    ├── copasi-dependencies/README.md
    ├── deepxde/README.md           # Physics-informed neural networks
    ├── esm/README.md               # Protein language models
    ├── gamer/README.md             # Geometry-preserving adaptive mesher
    ├── gromacs/README.md           # Molecular dynamics
    ├── libbng/README.md            # Lightweight BioNetGen library
    ├── local_deps/README.md        # Shared dependency builds
    ├── mcell/README.md             # Monte Carlo cell simulator
    ├── mcell_tests/README.md       # MCell test suite
    ├── mcell_tools/README.md       # MCell build utilities
    ├── mesh_tools/README.md        # 60+ mesh processing tools
    ├── modulus/README.md           # NVIDIA PhysicsNeMo
    ├── morpheus/README.md          # Multicellular modeling
    ├── neuropil_tools/README.md    # Neuropil ultrastructure analysis
    ├── nfsim/README.md             # Network-free stochastic simulator
    ├── nfsimCInterface/README.md   # C API wrapper for NFSim
    ├── openmm/README.md            # GPU-accelerated MD library
    ├── PhysiCell/README.md         # Agent-based multicellular simulator
    ├── pytorch_geometric/README.md # Graph neural networks
    ├── readdy/README.md            # Reaction diffusion dynamics
    ├── Smoldyn/README.md           # Spatial stochastic simulator
    ├── tissue-forge/README.md      # Interactive tissue simulator
    ├── vivarium-core/README.md     # Multi-scale composition framework
    └── VTK/README.md               # Visualization Toolkit
```

---

## cpp_ports — C++17 Header-Only Ports

10 complete C++17 reimplementations of Python computational biology and ML packages. Every port is **header-only with zero external dependencies** — just `#include` and compile with `-std=c++17`.

| Port | Headers | Lines | What It Does |
|------|---------|-------|-------------|
| **betse** | 14 | ~9,300 | Bioelectric tissue simulation: 50+ ion channels, GHK flux, gap junctions, electroosmosis, tissue deformation, ER/mitochondria, gene regulatory networks |
| **modulus** | 9 | ~6,600 | Physics-ML: FNO1D/2D/3D, DeepONet, AFNO, MeshGraphNet, GraphCast, UNet, Pix2Pix, SIREN, 25+ activations, Adam, geometry+CSG |
| **pyg** | 8 | ~5,100 | Graph neural networks: 45+ convolution layers (GCN, GAT, SAGE, GIN, Transformer), pooling, autograd, GAE/VGAE, SchNet |
| **deepxde** | 8 | ~4,400 | PINNs: tape-based reverse-mode AD, FNN/ResNet/SIREN/DeepONet, 12 geometry types, all BC types, Adam/SGD/L-BFGS |
| **boltz** | 5 | ~4,500 | Protein structure prediction: MSA module, Pairformer, diffusion (EDM/Heun/SDE), IPA, confidence heads (pLDDT/pTM) |
| **esm** | 2 | ~3,800 | Protein language models: ESM-2, MSA Transformer, ESMFold, GVP inverse folding, Alphabet/BatchConverter |
| **vivarium** | 3 | ~3,000 | Multi-scale composition: Store (hierarchical state tree), Process, Engine, Emitter (RAM/File), Composer |
| **betsee** | 1 + React | ~2,100 | BETSE GUI: Qt5 desktop app + React web frontend with 10 config panels, results viewer, simulation control |
| **cellblender** | 1 + React | ~1,600 | MCell GUI: DataModel, MDL generator/parser, results parser + React frontend with 3D viewer, MDL preview |
| **neuropil_tools** | 1 | ~1,400 | Neuropil analysis: mesh I/O (OBJ/OFF/STL/MDL), surface area/volume, Laplacian smoothing, spine morphometry |

**Total: 52 headers, ~41,800 lines of C++17. Two React web frontends.**

### Quick start

```bash
# Include any port header and compile — no build system needed
g++ -std=c++17 -O2 -I cpp_ports my_program.cpp -o my_program
```

---

## substrate — C++17 ML Benchmark Suite

Pure C++17 neural-network, ML, and biological cell simulation benchmark suite with 26 benchmarks across 17 model families.

**Model families:** MLP, CNN, RNN, LSTM, Transformer, ViT, GNN, diffusion, decision trees, random forests, clustering, Markov chains, Hebbian learning, forward-forward, reinforcement learning, actor-critic, and biological cells (including a full physics-based evolutionary cell engine with optional CUDA acceleration and OGRE 3D visualization).

```bash
cd substrate
make -j$(nproc) benchmarks    # Build all 26 benchmarks
./bin/benchmark_transformer    # Run transformer benchmark
./bin/benchmark_cellengine     # Run evolutionary cell engine
```

See [substrate/README.md](substrate/README.md) and [substrate/SPECS.md](substrate/SPECS.md) for full documentation.

---

## turing_cell_ring — Turing Morphogenesis Platform

Paper-example toolkit for A. M. Turing's *The Chemical Basis of Morphogenesis* (1952). C++ backend implementing reaction-diffusion ring models with eigenspectral analysis, React frontend for visualization, and LaTeX validation report.

**Implemented families:** Section 10 twenty-cell ring (reduced + full chemistry engines), Table 2 six-cell stable ring, oscillatory cases (e) and (f) with travelling-wave diagnostics.

```bash
cd turing_cell_ring
mkdir build && cd build && cmake .. && make
./turing_ring --family section10_example1 --preset paper-slow
```

See [turing_cell_ring/README.md](turing_cell_ring/README.md) for full documentation.

---

## Tool Documentation

Each tool folder contains an in-depth README.md documenting what the tool does, its biological scale, key algorithms, usage examples, input/output formats, build instructions, and connections to other tools in the pipeline.

The tools span the full biological scale hierarchy:

| Scale | Tools |
|-------|-------|
| Atomistic / Molecular | GROMACS, OpenMM, Boltz, ESM |
| Subcellular | Smoldyn, MCell, ReaDDy, COPASI, BioNetGen, NFSim |
| Cellular / Tissue | BETSE, PhysiCell, CompuCell3D, Morpheus, Tissue Forge |
| Bridging | Vivarium |
| Surrogate / ML | DeepXDE, Modulus, PyTorch Geometric |
| Mesh / Geometry | GAMer, mesh_tools, neuropil_tools |
| GUI | BETSEE, CellBlender |

See [BIOTOOLS_REFERENCE.md](BIOTOOLS_REFERENCE.md) for the comprehensive technical reference covering all tools, APIs, build locations, and integration patterns.

---

## Build & Run

### cpp_ports (no build system needed)
```bash
g++ -std=c++17 -O2 -I cpp_ports my_code.cpp -o my_code
```

### substrate
```bash
cd substrate && make -j$(nproc) benchmarks
```

### turing_cell_ring
```bash
cd turing_cell_ring && mkdir build && cd build && cmake .. && make
```

### React frontends
```bash
cd cpp_ports/betsee/react-app && npm install && npm run dev
cd cpp_ports/cellblender/react-app && npm install && npm run dev
cd substrate/ui && npm install && npm run dev
cd turing_cell_ring/frontend && npm install && npm run dev
```

---

## License

Individual components follow the licenses of their upstream projects. Original work (cpp_ports, substrate, turing_cell_ring) is under their respective licenses as specified in each subdirectory.
