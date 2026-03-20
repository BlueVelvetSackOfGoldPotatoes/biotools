# Biotools Reference Manual

Comprehensive technical reference for the biotools multi-scale computational biology pipeline.

---

## 1. Overview

The biotools collection is a **multi-scale computational biology pipeline** that spans the full hierarchy of biological organization --- from individual atoms and molecules up through subcellular reaction networks, whole cells, and multicellular tissues. It integrates 16 built C++ repositories, 10 C++ header-only ports of Python packages, 2 React frontend applications, and 7 Python packages in a shared conda environment.

The pipeline enables researchers to:

- Simulate molecular dynamics at atomistic resolution (GROMACS, OpenMM)
- Predict protein structures from sequence (Boltz, ESM)
- Model subcellular reaction-diffusion kinetics (Smoldyn, MCell, ReaDDy, COPASI, BioNetGen, NFSim)
- Simulate multicellular tissue bioelectrics, mechanics, and growth (BETSE, PhysiCell, CompuCell3D, Morpheus, Tissue Forge)
- Bridge scales with a composable simulation framework (Vivarium)
- Build physics-informed neural network surrogates (DeepXDE, Modulus/PhysicsNeMo)
- Operate on graph-structured biological data (PyTorch Geometric / PyG)
- Process, repair, and analyze biological meshes (GAMer, mesh_tools, neuropil_tools)
- Configure and visualize simulations through GUIs (BETSEE, CellBlender)

All C++ ports are header-only, C++17, and depend on no external libraries beyond the standard library. The built repositories use CMake and produce shared libraries and/or executables.

---

## 2. Directory Structure

```
~/Documents/biotools/
|
|-- PhysiCell/              # Agent-based multicellular simulator
|-- Smoldyn/                # Spatial stochastic biochemical simulator
|-- gromacs/                # All-atom molecular dynamics
|-- openmm/                 # GPU-accelerated molecular dynamics library
|-- readdy/                 # Reaction-diffusion dynamics
|-- COPASI/                 # Biochemical network simulator (ODE/SSA)
|-- CompuCell3D/            # Cellular Potts model framework
|-- morpheus/               # Multicellular modeling environment
|-- tissue-forge/           # Interactive tissue mechanics
|-- mcell/                  # Monte Carlo cell simulator (3D reaction-diffusion)
|-- bionetgen/              # Rule-based modeling language
|-- nfsim/                  # Network-free stochastic simulator
|-- nfsimCInterface/        # C API wrapper for NFSim
|-- libbng/                 # Lightweight BioNetGen library
|-- gamer/                  # Geometry-preserving Adaptive Mesher
|-- mesh_tools/             # 51+ mesh processing command-line tools
|-- VTK/                    # Visualization Toolkit (dependency)
|-- copasi-dependencies/    # Build dependencies for COPASI
|-- local_deps/             # Shared local dependency builds
|-- mcell_tests/            # MCell test suites
|-- mcell_tools/            # MCell helper utilities
|
|-- betse/                  # Python BETSE source (pip-installed)
|-- betsee/                 # Python BETSEE source (pip-installed)
|-- boltz/                  # Python Boltz source
|-- deepxde/                # Python DeepXDE source
|-- esm/                    # Python ESM source
|-- modulus/                # Python PhysicsNeMo/Modulus source
|-- pytorch_geometric/      # Python PyTorch Geometric source
|-- vivarium-core/          # Python Vivarium source
|-- cellblender/            # Python CellBlender source
|-- neuropil_tools/         # Python neuropil tools source
|
|-- cpp_ports/              # C++ header-only ports
|   |-- betse/              # 14 headers, 9,311 lines
|   |-- modulus/            # 9 headers, 6,597 lines
|   |-- pyg/                # 8 headers, 5,065 lines
|   |-- deepxde/            # 8 headers, 4,397 lines
|   |-- boltz/              # 5 headers, 4,474 lines
|   |-- esm/                # 2 headers, 3,789 lines
|   |-- vivarium/           # 3 headers, 3,033 lines
|   |-- betsee/             # 1 header + React app, 2,091 lines
|   |-- cellblender/        # 1 header + React app, 1,631 lines
|   |-- neuropil_tools/     # 1 header, 1,441 lines
|
|-- .venv/                  # Conda/pip virtual environment
```

---

## 3. Built C++ Repositories (16)

---

### 3.1 PhysiCell

**Path:** `~/Documents/biotools/PhysiCell/`

**What it does:** PhysiCell is an open-source, agent-based framework for simulating multicellular systems in 3-D tissue environments. Each cell is an off-lattice software agent with independent phenotype, signaling state, and mechanical interactions. The microenvironment is modeled on a regular Cartesian mesh with diffusing substrates (oxygen, glucose, chemokines, etc.).

**Biological scale:** Cellular / Tissue (10s to millions of cells in 3-D)

**Key algorithms:**
- Biomechanics: Cell-cell adhesion and repulsion via potential functions; velocity Verlet integration
- Diffusion: Finite-volume method on a regular 3-D Cartesian mesh with Dirichlet/Neumann boundaries
- Phenotype: Customizable cell cycle models (Ki67, flow cytometry, live/dead), death models (apoptosis, necrosis)
- Intracellular: Hooks for Boolean signaling networks, SBML ODE models, and custom rules
- Parallel: OpenMP-based multithreading

**Input/Output:**
- Input: XML settings file (`PhysiCell_settings.xml`), custom C++ module
- Output: MultiCellDS `.mat` snapshots, SVG cross-section images, PhysiCell Studio-compatible outputs

**How to run:**
```bash
cd ~/Documents/biotools/PhysiCell
make                        # compiles default project
./project config/PhysiCell_settings.xml
```

**Connections:** Can import BioNetGen/SBML intracellular models; meshes can be prepared with GAMer or mesh_tools.

**Version:** 1.14.x

---

### 3.2 Smoldyn

**Path:** `~/Documents/biotools/Smoldyn/`

**What it does:** Smoldyn (Smoluchowski Dynamics) is a spatial stochastic simulator for biochemical reaction networks. Molecules are represented as point particles that diffuse in continuous 2-D or 3-D space and react upon collision using Smoluchowski reaction dynamics. Surfaces (membranes, organelles) are modeled as triangulated meshes.

**Biological scale:** Subcellular (individual molecules, 100s to millions of particles)

**Key algorithms:**
- Brownian dynamics (random-walk diffusion with exact first-passage time corrections)
- Smoluchowski bimolecular reactions (binding radius, unbinding radius)
- Surface interactions (reflection, absorption, transmission, periodic boundaries)
- Compartment logic and rule-based reactions

**Input/Output:**
- Input: Smoldyn configuration text files (`.txt`)
- Output: Molecule counts, molecule positions, visualization snapshots

**Binary:** `~/Documents/biotools/Smoldyn/build/smoldyn`

**How to run:**
```bash
~/Documents/biotools/Smoldyn/build/smoldyn config.txt
```

**Connections:** Can read BioNetGen (`.bngl`) rules; integrates with libsmoldyn for embedding in other simulators; used by VCell.

---

### 3.3 GROMACS

**Path:** `~/Documents/biotools/gromacs/`

**What it does:** GROMACS (GROningen MAchine for Chemical Simulations) is a high-performance molecular dynamics engine for simulating biomolecular systems --- proteins, lipids, nucleic acids, and their interactions with solvents and ions. It is one of the fastest MD codes available.

**Biological scale:** Molecular / Atomistic (individual atoms, ~10^3 to ~10^7 atoms)

**Key algorithms:**
- Velocity Verlet and leap-frog integrators
- PME (Particle Mesh Ewald) for long-range electrostatics
- LINCS and SETTLE constraint solvers
- Free energy perturbation, replica exchange, metadynamics
- Coarse-grained (Martini) support
- SIMD-optimized nonbonded kernels, GPU offloading (CUDA/OpenCL/SYCL)

**Input/Output:**
- Input: `.gro` (coordinates), `.top` (topology), `.mdp` (run parameters)
- Output: `.trr` / `.xtc` (trajectories), `.edr` (energies), `.xvg` (analysis)

**Binaries:** `~/Documents/biotools/gromacs/build/bin/gmx`
**Libraries:** `~/Documents/biotools/gromacs/build/lib/libgromacs.so.12.0.0`, `libgmxapi.so.0.4.0`

**How to run:**
```bash
export PATH=~/Documents/biotools/gromacs/build/bin:$PATH
gmx grompp -f params.mdp -c structure.gro -p topology.top -o run.tpr
gmx mdrun -deffnm run
```

**Connections:** Structures from Boltz/ESM can be solvated and simulated; trajectories feed into coarse-grained or continuum models.

---

### 3.4 OpenMM

**Path:** `~/Documents/biotools/openmm/`

**What it does:** OpenMM is a high-performance toolkit for molecular simulation that provides a flexible API for defining custom forces and integrators. It emphasizes GPU acceleration and is the backend for many other tools (e.g., OpenMM-based folding in Boltz).

**Biological scale:** Molecular / Atomistic

**Key algorithms:**
- Custom force expressions compiled to GPU kernels at runtime
- Langevin, Brownian, Verlet, and custom integrators
- PME electrostatics, GBSA implicit solvent
- Drude polarizable force field, AMOEBA
- Alchemical free energy calculations
- Ring Polymer MD (RPMD) for quantum nuclear effects

**Libraries:**
- `~/Documents/biotools/openmm/build/libOpenMM.so`
- `libOpenMMCPU.so`, `libOpenMMPME.so`, `libOpenMMAmoeba.so`, `libOpenMMDrude.so`, `libOpenMMRPMD.so`

**Input/Output:**
- Input: PDB/PDBx structures, XML force field files, Python/C++ API calls
- Output: Trajectories (PDB, DCD), state data, energies via reporters

**Connections:** Used as the dynamics engine when refining structures from ESM/Boltz; can be called from Vivarium composites.

---

### 3.5 ReaDDy

**Path:** `~/Documents/biotools/readdy/`

**What it does:** ReaDDy (Reaction Diffusion Dynamics) is a particle-based reaction-diffusion simulator that combines Brownian dynamics with reactive potentials. Unlike Smoldyn, ReaDDy supports soft interaction potentials between particles, enabling simulation of crowded molecular environments with excluded volume.

**Biological scale:** Subcellular (mesoscale: 10 nm -- 10 um)

**Key algorithms:**
- Overdamped Langevin dynamics (Brownian dynamics)
- Reactive potentials: particles react when they collide within a reaction radius
- Topology-based reactions: bond formation/breaking, polymer dynamics
- Cell-linked list spatial decomposition for neighbor search
- HDF5-based trajectory output

**Library:** `~/Documents/biotools/readdy/build/libreaddy.so`

**Connections:** Bridges molecular simulations (GROMACS/OpenMM) and cellular models by modeling the mesoscale.

---

### 3.6 COPASI

**Path:** `~/Documents/biotools/COPASI/`

**What it does:** COPASI (COmplex PAthway SImulator) is a comprehensive tool for simulation and analysis of biochemical networks. It supports deterministic (ODE) and stochastic (SSA, tau-leaping) simulation, steady-state analysis, metabolic control analysis, parameter estimation, sensitivity analysis, and optimization.

**Biological scale:** Subcellular (well-mixed biochemical networks)

**Key algorithms:**
- ODE solvers: LSODA (automatic stiff/non-stiff), Radau5, ODEPACK
- Stochastic: Gillespie SSA (direct, next-reaction), tau-leaping, adaptive SSA, hybrid ODE/SSA
- Steady-state analysis: Newton's method, damped Newton
- Metabolic control analysis (MCA), Lyapunov exponents
- Parameter estimation: Levenberg-Marquardt, genetic algorithms, particle swarm, simulated annealing
- SBML import/export

**Binaries:**
- `~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE` (command-line)
- `~/Documents/biotools/COPASI/copasi/CopasiUI` (Qt GUI)

**Input/Output:**
- Input: COPASI `.cps` files, SBML `.xml`, SED-ML
- Output: Time course data, steady-state reports, parameter estimation results (CSV/TSV)

**How to run:**
```bash
~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE model.cps
```

**Connections:** Can import/export SBML models shared with BioNetGen, PhysiCell intracellular models.

---

### 3.7 CompuCell3D

**Path:** `~/Documents/biotools/CompuCell3D/`

**What it does:** CompuCell3D (CC3D) is a multiscale modeling environment based on the Cellular Potts Model (CPM, also known as Glazier-Graner-Hogeweg model). Cells are represented as connected domains of lattice sites that evolve via a modified Metropolis algorithm. CC3D supports subcellular ODE networks, PDE diffusion fields, and cell-level behaviors.

**Biological scale:** Cellular / Tissue (10s to 10,000s of cells on a lattice)

**Key algorithms:**
- Cellular Potts Model: stochastic pixel-copy dynamics minimizing a Hamiltonian (volume, surface, adhesion, chemotaxis terms)
- PDE solvers for diffusible signals (finite difference, SteadyStateDiffusion, FlexibleDiffusion)
- SBML-based intracellular models per cell
- Cell growth, division, death, differentiation
- Chemotaxis, haptotaxis, contact-inhibited locomotion

**Libraries:** `~/Documents/biotools/CompuCell3D/build/core/CompuCell3D/libCC3DCompuCellLib.so` and supporting libs

**Input/Output:**
- Input: CC3D project files (`.cc3d`), Python steppables, XML configuration
- Output: VTK lattice snapshots, CSV data, screenshots

**Connections:** Can embed SBML models from COPASI; output meshes can be processed with mesh_tools.

---

### 3.8 Morpheus

**Path:** `~/Documents/biotools/morpheus/`

**What it does:** Morpheus is an integrated modeling and simulation environment for multicellular systems. It provides a declarative XML-based model description language and a GUI. Morpheus supports Cellular Potts Models, continuous field PDEs, ODEs, and event-based rules in a unified framework.

**Biological scale:** Cellular / Tissue

**Key algorithms:**
- Cellular Potts Model (extended Metropolis)
- Finite-element PDE solvers on unstructured grids
- ODE integration (Runge-Kutta, CVODE)
- Population-level models, lattice-gas cellular automata
- SBML support

**Binary:** `~/Documents/biotools/morpheus/build/morpheus`

**Input/Output:**
- Input: MorpheusML XML model files
- Output: VTK, TIFF, PNG, CSV

---

### 3.9 Tissue Forge

**Path:** `~/Documents/biotools/tissue-forge/`

**What it does:** Tissue Forge is an interactive, particle-based physics engine for biological and biophysics simulations. It models cells, subcellular objects, and tissue-like structures as particles interacting through customizable potentials, bonds, and surfaces. It features real-time 3-D visualization.

**Biological scale:** Cellular / Tissue (real-time interactive)

**Key algorithms:**
- Dissipative Particle Dynamics (DPD)
- Customizable pair potentials (Lennard-Jones, Morse, harmonic, Coulomb)
- Bond, angle, and dihedral potentials
- Boundary conditions, flux, and secretion
- Event handling and stochastic transitions
- Real-time OpenGL rendering

**Libraries:** `~/Documents/biotools/tissue-forge/build/Release/lib/libtissue-forge.so`, `libtissue-forge-c.so`

**Input/Output:**
- Input: Python/C API calls, JSON configuration
- Output: Screenshots, particle trajectories, real-time visualization

---

### 3.10 MCell

**Path:** `~/Documents/biotools/mcell/`

**What it does:** MCell (Monte Carlo Cell) is a program for simulating 3-D reaction-diffusion of molecules within and between cells. It tracks individual molecules as they diffuse and interact in realistic 3-D geometries derived from electron microscopy reconstructions.

**Biological scale:** Subcellular (synaptic cleft, dendritic spines, organelle membranes)

**Key algorithms:**
- Monte Carlo random walk diffusion in 3-D triangulated mesh geometries
- Surface and volume molecule tracking
- Bimolecular reactions via collision detection on surfaces
- Complex geometry handling (reflection, absorption, transparency)
- Checkpointing and restarting

**Input/Output:**
- Input: MDL (Model Description Language) files, mesh geometry files
- Output: Reaction data counts, molecule visualization data (VIZ), DREAM.3D outputs

**Connections:** Geometries prepared by CellBlender (Blender addon) or mesh_tools; NFSim/BioNetGen for rule-based reactions; neuropil_tools for EM reconstruction processing.

---

### 3.11 BioNetGen

**Path:** `~/Documents/biotools/bionetgen/`

**What it does:** BioNetGen is a rule-based modeling framework for biochemical systems. Instead of enumerating every possible molecular species and reaction, modelers write rules that describe classes of reactions. BioNetGen can automatically generate the full reaction network, or simulate directly using NFSim.

**Biological scale:** Subcellular (biochemical reaction networks)

**Key algorithms:**
- Rule-based model specification (BNGL language)
- Network generation via graph rewriting
- ODE generation and simulation (via `run_network`)
- SSA (Gillespie) simulation
- Network-free simulation (via NFSim)
- Parameter scanning and bifurcation analysis

**Binary:** `~/Documents/biotools/bionetgen/bng2/Network3/bin/run_network`
**Script:** `~/Documents/biotools/bionetgen/bng2/BNG2.pl`

**Input/Output:**
- Input: `.bngl` rule files
- Output: `.net` network files, `.gdat` / `.cdat` time series data, `.xml` for NFSim

---

### 3.12 NFSim

**Path:** `~/Documents/biotools/nfsim/`

**What it does:** NFSim (Network-Free Simulator) stochastically simulates rule-based models without generating the full reaction network. This enables simulation of models with combinatorially large state spaces (e.g., multi-site phosphorylation, scaffold complexes) that are intractable for network-based approaches.

**Biological scale:** Subcellular (complex biochemical networks)

**Key algorithms:**
- Network-free stochastic simulation (extended Gillespie-type)
- Graph-based molecule representations with pattern matching
- On-the-fly reaction firing
- Complex tracking and observables

**Binary:** `~/Documents/biotools/nfsim/build/NFsim`

**Input/Output:**
- Input: BioNetGen XML (`.xml`) generated from `.bngl` files
- Output: Time series of observable values (`.gdat`)

---

### 3.13 nfsimCInterface

**Path:** `~/Documents/biotools/nfsimCInterface/`

**What it does:** A C API wrapper around NFSim that allows other programs (particularly MCell) to embed network-free simulation as a library call rather than a separate process.

**Biological scale:** Subcellular (API bridge)

**Libraries:**
- `~/Documents/biotools/nfsimCInterface/build/libnfsim_c.so`
- `~/Documents/biotools/nfsimCInterface/build/libnfsim_c_static.a`

**Connections:** Called by MCell for intracellular rule-based reaction simulation.

---

### 3.14 libbng

**Path:** `~/Documents/biotools/libbng/`

**What it does:** A lightweight, embeddable C++ library providing core BioNetGen capabilities. Designed for integration into other simulation tools that need rule-based reaction network generation without the full BioNetGen/Perl stack.

**Library:** `~/Documents/biotools/libbng/build/bng/liblibbng.a`

**Connections:** Provides BioNetGen functionality to MCell and other C++ tools.

---

### 3.15 GAMer

**Path:** `~/Documents/biotools/gamer/`

**What it does:** GAMer (Geometry-preserving Adaptive MeshER) generates high-quality tetrahedral and surface meshes from molecular and cellular structures. It is designed for generating finite-element meshes from biological geometries reconstructed from electron microscopy or other imaging modalities.

**Biological scale:** Mesh / Geometry (all scales)

**Key algorithms:**
- Surface mesh smoothing (angle-weighted Laplacian)
- Surface mesh refinement and coarsening
- Tetrahedral mesh generation (Delaunay-based)
- Molecular surface meshing from PDB/PQR coordinates
- Geometry-preserving operations (normal smoothing, feature preservation)

**Libraries:**
- `~/Documents/biotools/gamer/build/lib/libgamer.so`
- `~/Documents/biotools/gamer/build/lib/libgamer.a`

**Input/Output:**
- Input: PDB/PQR molecular structures, OFF/OBJ/STL surface meshes
- Output: OFF, OBJ, STL surface meshes; tetrahedral meshes for FEM

**Connections:** Generates meshes used by MCell, COPASI spatial solvers, and finite-element solvers.

---

### 3.16 mesh_tools

**Path:** `~/Documents/biotools/mesh_tools/`

**What it does:** A collection of 51+ command-line mesh processing utilities for converting, analyzing, repairing, and transforming triangulated surface meshes. These tools form the mesh processing backbone for the MCell/CellBlender workflow.

**60 executables** in `~/Documents/biotools/mesh_tools/` (built from C source in subdirectories).

**Key tools include:**
| Tool | Function |
|------|----------|
| `mesh2mcell` | Convert mesh to MCell MDL format |
| `mesh2obj` | Convert to Wavefront OBJ |
| `mesh2stl` | Convert to STL |
| `mesh2vtk` | Convert to VTK format |
| `mesh2off` | Convert to OFF format |
| `mesh2gts` | Convert to GTS format |
| `obj2mesh` | Convert OBJ to internal mesh format |
| `meshrefine` | Refine mesh by subdivision |
| `meshsimplify` | Reduce mesh polygon count |
| `meshheal` | Repair mesh defects (holes, degeneracies) |
| `meshmorph` | Morph between two meshes |
| `meshfuse` | Fuse overlapping meshes |
| `meshmerge` | Merge separate meshes |
| `meshclip` | Clip mesh by a plane |
| `meshflip` | Flip face normals |
| `meshscale` | Scale mesh dimensions |
| `meshtranslate` | Translate mesh position |
| `meshalyzer` | Compute mesh statistics (area, volume, quality) |
| `meshfilter` | Filter mesh by criteria |
| `meshoffset` | Offset mesh surface |
| `meshstitch` | Stitch mesh boundaries |
| `meshorphan` | Remove orphan vertices |
| `mesh_separate` | Separate disconnected components |
| `mesh_tag_region` | Tag mesh regions for MCell surface classes |
| `contour_tiler` | Tile between serial-section contours |
| `contour_plotting` | Plot contour traces |
| `dx2mesh` | Convert OpenDX to mesh |
| `filtermesh` | Apply filters to mesh data |

**Input/Output:**
- Input: Mesh files in various formats (internal, OBJ, OFF, STL, DX, IRIT, GTS, SMF, RIB)
- Output: Converted/processed mesh files

**Connections:** Output feeds directly into MCell geometry, GAMer, CellBlender, and VTK visualization.

---

## 4. C++ Ports of Python Repositories (10)

All ports are **header-only C++17** with **no external dependencies**. They reside in `~/Documents/biotools/cpp_ports/`.

---

### 4.1 betse/ -- Bioelectric Tissue Simulation Engine

**Path:** `~/Documents/biotools/cpp_ports/betse/`
**Size:** 14 headers, 9,311 lines
**Original:** [github.com/betsee/betse](https://github.com/betsee/betse)

**What it does:** Complete C++ reimplementation of BETSE, which simulates bioelectric patterns in 2-D cell clusters. Models the interplay between ion channel activity, gap junction communication, electroosmotic flows, mechanical deformation, and gene regulatory networks across interconnected cells.

**Biological scale:** Cellular / Tissue (2-D epithelial sheets, 10s to 1000s of cells)

**Key algorithms:**
- Goldman-Hodgkin-Katz (GHK) electrodiffusion flux equation
- Nernst-Planck transport through membranes and gap junctions
- Hodgkin-Huxley gating kinetics for 50+ voltage-gated ion channel types (Nav, Kv, Cav, HCN, Morris-Lecar, TRP)
- Na/K-ATPase, Ca-ATPase, V-ATPase, H/K-ATPase, NCX exchanger, NKCC/KCC co-transporters
- Voronoi cell cluster generation with Lloyd relaxation
- Helmholtz-Hodge decomposition for divergence-free flow fields
- Electroosmotic flow (Helmholtz-Smoluchowski)
- Tissue deformation (steady-state and time-dependent) with galvanotropism
- Endoplasmic reticulum Ca2+ dynamics (IP3R, RyR, SERCA, CICR)
- Mitochondrial membrane potential (ETC, MCU, NCLX)
- Microtubule electrophoretic alignment (Broersma drag, rotational diffusion)
- Gene regulatory networks (molecules, reactions, transporters, modulators)
- Lateral redistribution of pumps/channels on membrane surfaces
- Full ECM (extracellular matrix) PDE transport
- Finite-difference operators (gradient, divergence, Laplacian, curl, Gaussian filter)
- Poisson solvers (Jacobi, SOR, CG, graph-based)
- Equivalent circuit (FAST) and full Nernst-Planck (FULL) solver modes
- CSV data export pipeline, visualization data extraction

**Headers and key contents:**

| Header | Key Classes/Structs/Functions |
|--------|------------------------------|
| `betse_types.h` | `Vec2`, `DenseMatrix`, `SparseMatrix`, `Ion` enum, `ChannelType` enum (50+ types), `SimPhaseKind`, physical constants (`F_FARADAY`, `R_GAS`, `K_BOLTZMANN`, etc.), `electroflux()` (Goldman flux), `vecutil` namespace |
| `betse_channels.h` | `ChannelState` (m/h gates, open probability, reversal voltage), `update_mh()` (semi-implicit Euler), `update_ml()` (Morris-Lecar), `compute_open_prob()`, kinetics for Nav1.2/1.3/1.6, Kv1.1-3.4, Kir2.1, Cav1.2/1.3/2.1-3.3, HCN1/2/4, ClLeak, CatLeak, TRP wound channel, all Morris-Lecar variants |
| `betse_math.h` | `Grid2D`, `fd::gradient()`, `fd::divergence()`, `fd::laplacian()`, `fd::gaussian_filter()`, `fd::curl_2d()`, `fd::integrator()`, `solve_poisson_jacobi()`, `solve_graph_poisson()`, `HHDecomp`, `helmholtz_hodge_2d()`, `helmholtz_hodge_cells()`, `single_cell_div_free()`, `WaveSolver`, `modulate::gradient_x/y/r()`, `modulate::periodic()`, `modulate::f_sweep()`, `ghk_voltage()`, `bicarbonate_buffer()` |
| `betse_physics.h` | `CurrentState`, `compute_current()`, `FlowState`, `compute_electroosmotic_flow()`, `compute_ecm_flow()`, `OsmoticState`, `compute_osmotic_pressure()`, `DeformState`, `compute_deformation_steady()`, `compute_deformation_time()`, `MoveChannelState`, `update_move_pump()`, `update_move_channel()`, `compute_env_voltage_simple()`, `compute_screening_constant()` |
| `betse_networks.h` | `MoleculeExt`, `ReactionExt`, `Transporter`, `NetworkChannel`, `MembraneModulator`, `MasterOfNetworks` (GRN engine with `move_molecule()`, `run_pump()`, `step_reactions()`, `step_growth_decay()`, `apply_modulators()`, `compute_extra_charge()`), `pump_VATP()`, `pump_HKATP()`, `pump_NCX()`, `pump_NKCC()`, `pump_KCC()` |
| `betse_organelles.h` | `IP3Receptor`, `EndoReticulumFull` (IP3R + RyR + SERCA + leak), `MitochondriaFull` (ETC + MCU + NCLX), `NuclearTransport` (NPC-mediated diffusion), `MicrotubulesFull` (electric-field-driven alignment, Broersma drag) |
| `betse_tissue.h` | `TissuePicker`, `TissueProfileFull`, `TissueEvent` (cutting, voltage pulse, Dm change, wound, GJ block), `TissueManager`, `TissueGrowth`, `IonProfilePreset` (mammal, amphibian, basic), `get_ion_profile()`, `ChannelRef`, `get_channelpedia_database()` (27 reference channels) |
| `betse_phase.h` | `PhaseCallbacks`, `ExtendedSnapshot`, `TimeSeriesRecorder`, `ScheduledEvent`, `EventType` enum, `TissueHandler`, `CSVExporter`, `SolverType` enum, `solve_ecm_transport()`, `update_concentrations_from_flux()`, `nernst_potential()`, `SimPhaseRunner` |
| `betse_solver.h` | `FastSolver` (equivalent circuit RC model), `FullSolver` (Nernst-Planck with ECM), `solve_cg()` (conjugate gradient), `solve_poisson_sor()`, `apply_boundary_voltage()`, `solve_ecm_voltage()`, `compute_magnetic_field()` |
| `betse_config.h` | `CSVExportItem`, `PlotExportItem`, `AnimExportItem`, `ExportConfig`, `GRNMoleculeConfig`, `GRNReactionConfig`, `GRNTransporterConfig`, `GRNModulatorConfig`, `GRNConfig`, `TissueProfileConfig`, `CutEventConfig`, `FullSimConfig` (all YAML configuration parameters) |
| `betse_enums.h` | `CellLatticeType`, `CellsPickerType`, `GrnUnpicklePhaseType`, `IonProfileType`, `SimExportType`, `BoundarySide`, `ColormapType`, `DeformSolverType`, `GJConnectivityType`, `ElectroDiffusionType`, `OsmoticModelType`, `VisualLayerType`, `SimDataChannel` (30+ data channels) |
| `betse_export.h` | `DataSnapshot`, `TimeSeriesDB`, `ExportPipelineItem`, `CSVWriter` (write_vmem, write_ions, write_molecules, write_current, write_pressure, write_full, write_spatial_snapshot), `ExportPipeline`, `VisDataExtractor` |
| `betse_util.h` | `units` namespace (mV/V, um/m, mM/M, pH conversions), `geometry` namespace (point_in_polygon, polygon_area, centroid, convex_hull, point_to_segment_dist, circles_intersect), `interp` namespace (lerp, bilinear, nearest_cell, IDW), `signal` namespace (pulse, step, ramp, periodic_sin2, chirp, gaussian_pulse), `VoronoiGenerator` (cell generation + Lloyd relaxation), `fileio` namespace (read_csv, write_array_csv), `stats` namespace, `random` namespace |
| `betse.h` | Master include, `SimConfig` struct |

---

### 4.2 modulus/ -- NVIDIA Modulus / PhysicsNeMo

**Path:** `~/Documents/biotools/cpp_ports/modulus/`
**Size:** 9 headers, 6,597 lines

**What it does:** C++ port of NVIDIA's Modulus (PhysicsNeMo) physics-informed machine learning framework. Implements neural network architectures for solving PDEs and learning physics from data, including Fourier Neural Operators, DeepONet, and MeshGraphNet.

**Biological scale:** Surrogate / ML (learns to approximate physics at any scale)

**Key algorithms:**
- Dense Tensor with FFT (Cooley-Tukey) for spectral operations
- MLP with backpropagation and Adam optimizer
- Fourier Neural Operator (FNO1D, FNO2D, FNO3D) with spectral convolution
- Deep Operator Network (DeepONet)
- Adaptive Fourier Neural Operator (AFNO)
- MeshGraphNet, GraphCastNet
- SRResNet, Pix2Pix, UNet, SIREN
- One2ManyRNN, SwinRNN, DLWP
- Physics constraint (PDE residual loss)
- 25+ activation functions (GELU, SiLU, Mish, Stan, capped variants, etc.)
- Loss functions (MSE, L1, Huber, relative L2, log-cosh, integral, boundary, PDE residual)
- Learning rate schedulers (constant, step, exponential, cosine annealing, warmup-decay, one-cycle)
- Data pipelines (Dataset, DataLoader, Normalizer, MinMaxNormalizer, augmentation)
- Model serialization (binary export/import, NumPy `.npy` export)
- ONNX graph description builder
- Profiling and model summary utilities
- Geometry primitives (Box, Sphere, Cylinder, Cone, Torus, Plane, CSG operations, STL import, tessellation)
- Mesh graph construction (edge features, kNN, radius, Delaunay, multi-mesh, graph partitioning)

**Headers:**
| Header | Key Contents |
|--------|-------------|
| `modulus.h` | `Tensor` (dense multi-dim array with FFT), `Parameter`, `activation` namespace, `MLP`, `FNO1D`, `DeepONet`, `PhysicsConstraint`, `Adam`, `TrainingLoop` |
| `modulus_activations.h` | 25+ activation functions with gradients (GELU, SiLU, Mish, Stan, SELU, CELU, etc.), `get_activation_extended()` dispatcher |
| `modulus_models.h` | `FNO2D`, `FNO3D`, `AFNO`, `MeshGraphNet`, `GraphCastNet`, `SRResNet`, `Pix2Pix`, `UNet`, `SIREN`, `One2ManyRNN`, weight initialization (Xavier, Kaiming, truncated normal) |
| `modulus_geometry.h` | `Vec3`, `Vec2`, `Box`, `Sphere`, `Cylinder`, `Cone`, `Torus`, `Plane`, `Line`, `Circle`, `CSGUnion/Intersection/Difference`, `STLReader`, `tessellate_geometry()`, `ParameterizedGeometry` |
| `modulus_graph.h` | `compute_edge_features_relative()`, `compute_edge_features_with_angles()`, `knn_graph()`, `radius_graph()`, `delaunay_graph_2d()`, `multi_mesh_graph()`, `GraphPartitioner` |
| `modulus_loss.h` | `mse_loss`, `l1_loss`, `huber_loss`, `relative_l2_loss`, `log_cosh_loss`, `WeightedLoss`, `IntegralLoss`, `BoundaryLoss`, `PointwiseLoss`, `PDEResidualLoss` |
| `modulus_solver.h` | LR schedulers (Constant, Step, Exponential, CosineAnnealing, WarmupDecay, OneCycle), `Solver` (full training loop with validation, checkpointing, early stopping, gradient clipping) |
| `modulus_datapipes.h` | `Dataset`, `DataLoader`, `Normalizer`, `MinMaxNormalizer`, `augmentation` namespace, CSV data loader |
| `modulus_deploy.h` | `ModelInfo`, `export_parameters_binary()`, `import_parameters_binary()`, `export_npy()`, `ONNXGraphBuilder`, `Profiler`, `print_model_summary()` |

---

### 4.3 pyg/ -- PyTorch Geometric

**Path:** `~/Documents/biotools/cpp_ports/pyg/`
**Size:** 8 headers, 5,065 lines

**What it does:** C++ port of PyTorch Geometric for graph neural networks. Implements graph data structures, 30+ convolution layers, pooling operators, normalization layers, and high-level model architectures.

**Biological scale:** Surrogate / ML (graph-structured biological data)

**Headers and key contents:**
| Header | Key Contents |
|--------|-------------|
| `pyg.h` | `Tensor` (2-D with autograd), `GradNode`, `Data` (graph), `HeteroData`, `Batch`, scatter/degree/softmax utilities, `Linear`, `GRU`, `BatchNorm1d`, `Embedding`, `Dropout`, Adam/SGD optimizers, cross-entropy/MSE/BCE losses, autograd (relu, sigmoid, log_softmax, etc.), negative_sampling, subgraph extraction |
| `pyg_conv.h` | 30+ convolution layers: `GCNConv`, `GATConv`, `GATv2Conv`, `SAGEConv`, `GraphConv`, `GINConv`, `GINEConv`, `EdgeConv`, `ChebConv`, `SGConv`, `TAGConv`, `ARMAConv`, `APPNPConv`, `GCN2Conv`, `TransformerConv`, `SuperGATConv`, `PNAConv`, `GeneralConv`, `NNConv`, `CGConv`, `MEGNetConv`, `SplineConv` (approx), `ClusterGCNConv`, `FeaStConv`, `LEConv`, `GMMConv`, `EGConv`, `PDNConv`, `SSGConv`, `RGCNConv`, `FiLMConv`, `WLConv` |
| `pyg_pool.h` | `global_mean_pool`, `global_max_pool`, `global_add_pool`, `SAGPooling`, `TopKPooling`, `ASAPooling`, `MemPooling`, `PANPooling`, `EdgePooling`, voxel grid, `graclus_coarsen` |
| `pyg_norm.h` | `GraphNorm`, `InstanceNorm`, `LayerNorm`, `PairNorm`, `MeanSubtractionNorm`, `DiffGroupNorm`, `GraphSizeNorm`, `MessageNorm` |
| `pyg_models.h` | `GAE`, `VGAE`, `Node2Vec`, `JumpingKnowledge`, `GraphUNet`, `SchNet`, `DimeNet` (simplified), `SignedGCN`, `RECT_L`, `MetaPath2Vec`, `GNNExplainer`, `RENet`, `BasicGNN`, `DeepGCNLayer` |
| `pyg_transforms.h` | `NormalizeFeatures`, `AddSelfLoops`, `RemoveSelfLoops`, `ToUndirected`, `RandomNodeSplit`, `RandomLinkSplit`, `AddRandomWalkPE`, `AddLaplacianPE`, `VirtualNode`, `GDC`, `SIGN` |
| `pyg_loader.h` | `NeighborLoader`, `ClusterLoader`, `RandomNodeLoader`, `LinkNeighborLoader` |
| `pyg_dense.h` | `DenseGCNConv`, `DenseGATConv`, `DenseSAGEConv`, `DenseGraphConv`, `DenseGINConv`, `dense_diff_pool`, `dense_mincut_pool` |

---

### 4.4 deepxde/ -- DeepXDE

**Path:** `~/Documents/biotools/cpp_ports/deepxde/`
**Size:** 8 headers, 4,397 lines

**What it does:** C++ port of DeepXDE, a library for solving differential equations using physics-informed neural networks (PINNs). Implements tape-based automatic differentiation, feedforward networks, PDE residual computation, geometry classes, boundary/initial conditions, and a PINN solver.

**Biological scale:** Surrogate / ML (PDE solving at any scale)

**Headers:**
| Header | Key Contents |
|--------|-------------|
| `deepxde.h` | `Var` (AD variable with tape-based reverse-mode), AD operators (+, -, *, /, sin, cos, exp, log, pow, tanh, sigmoid), `FeedForward` (MLP with backprop), `PINNSolver`, `Adam` optimizer, `Interval`, `Rectangle`, `GeometryXTime` |
| `deepxde_geometry.h` | `Disk`, `Triangle`, `Polygon`, `Cuboid`, `Sphere`, `Cylinder`, `CSGUnion/Intersection/Difference`, `TimeDomain`, `GeometryXTime`, `PointCloud`, sampling methods |
| `deepxde_data.h` | `PDEData`, `DatasetBC`, `TimePDE`, `InversePDE`, `DataDriven`, `MFData`, `FuncData`, collocation point management |
| `deepxde_nn.h` | Extended architectures: `ResNet`, `PFNN` (parallel), `SIREN`, `ModifiedMLP`, `DeepONet`, `FNO1D`, `PIDeepONet`, ensembles, multi-fidelity |
| `deepxde_icbc.h` | `DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`, `IC`, `PointSetBC`, `OperatorBC`, `InterfaceBC`, hard/soft constraint methods |
| `deepxde_losses.h` | MSE, MAE, Huber, relative L2, SSIM-like, custom weighted, gradient-enhanced, log-cosh, Sobolev, adversarial losses |
| `deepxde_callbacks.h` | `EarlyStopping`, `LRScheduler` (step, exponential, cosine, warmup), `ModelCheckpoint`, `VariableValue`, `PDEResidualResampler`, `TesterCallback`, `Timer` |
| `deepxde_all.h` | Convenience include-all header |

---

### 4.5 boltz/ -- Boltz Protein Structure Prediction

**Path:** `~/Documents/biotools/cpp_ports/boltz/`
**Size:** 5 headers, 4,474 lines

**What it does:** C++ port of the Boltz protein structure prediction architecture. Covers the full pipeline from sequence/MSA input through structure prediction with diffusion-based generation.

**Biological scale:** Molecular (protein structure prediction from sequence)

**Key algorithms:**
- Data parsing (FASTA, A3M/MSA, PDB, mmCIF)
- Tokenization and featurization
- MSA module (Outer Product Mean, Triangular Multiplicative Updates, Triangular Attention, Pair-Weighted Averaging)
- Pairformer (attention with pair bias)
- Diffusion module (VP/VE/EDM noise schedules, Heun/Euler samplers, SDE/ODE)
- Guidance potentials (VDW, connections, stereo bonds, chirality, planarity)
- Confidence heads (pLDDT, pTM, iPTM, PAE, PDE)
- Structure module (IPA, backbone updates)
- Loss functions (FAPE, smooth lDDT, distogram, B-factor)
- Training infrastructure (AlphaFold LR scheduler, EMA, Adam/AdamW, gradient clipping, checkpointing)

**Headers:**
| Header | Key Contents |
|--------|-------------|
| `boltz.h` | Constants, `Rigid` (SE(3) transform), `BackboneAtoms`, `MSARow`, `TokenData`, `PairData`, attention modules, MSA module, Pairformer, AtomTransformer, confidence heads, `BoltzPredictor` (full pipeline), PDB/mmCIF writer |
| `boltz_data.h` | Full PDB/mmCIF parsers, A3M parser, `BoltzTokenizer`, `BoltzCropper`, `AffinityCropper`, featurization pipeline, static/dynamic filters, sampling strategies, data module |
| `boltz_diffusion.h` | `VPSchedule`, `VESchedule`, `EDMSchedule`, `EulerSampler`, `HeunSampler`, guidance potentials, `AtomDiffusion` module, weighted rigid alignment |
| `boltz_loss.h` | `fape_loss_extended()`, `smooth_lddt_loss()`, `distogram_loss()`, `InteractionWeights`, confidence losses, validation metrics (RMSD, MAE) |
| `boltz_training.h` | `AlphaFoldLRScheduler`, `EMA`, `AdamW` optimizer, training loop with recycling, validation, checkpointing |

---

### 4.6 esm/ -- Evolutionary Scale Modeling

**Path:** `~/Documents/biotools/cpp_ports/esm/`
**Size:** 2 headers, 3,789 lines

**What it does:** C++ port of Meta's ESM protein language models. Covers ESM-1, ESM-1b, ESM-2, MSA Transformer, ESMFold, and GVP-based inverse folding.

**Biological scale:** Molecular (protein language modeling, structure prediction)

**Key contents:**
| Header | Key Contents |
|--------|-------------|
| `esm.h` | `Matrix`, `Tensor3D`, `Tensor4D`, `Tensor5D`, math utilities (GELU, softmax, layer norm, matmul, rotary embeddings), `Alphabet` (ESM-1/ESM-2/MSA), `FastaBatchedDataset`, `BatchConverter`, `MSABatchConverter`, `LearnedPositionalEmbedding`, `SinusoidalPositionalEmbedding`, `RotaryEmbedding`, `MultiheadAttention` (with rotary, bias_kv, causal), `RowSelfAttention`, `ColumnSelfAttention`, `TransformerLayer`, `AxialTransformerLayer`, `ProteinBertModel` (ESM-1/1b), `ESM2Model`, `MSATransformerModel`, `ContactPredictionHead`, `RobertaLMHead`, `GVPModule`, `GVPConv`, `GVPConvLayer`, `GVPEncoder`, `GVPTransformerEncoder`, `TransformerDecoder`, pretrained model registry |
| `esm_extended.h` | `Quaternion`, `Rigid`, `TriangleMultiplicativeUpdate`, `TriangleAttention`, `InvariantPointAttention`, `StructureModule`, `ESMFoldModel`, `DihedralFeatures`, `NormalizedResidualBlock`, `CoordBatchConverter`, PDB I/O, MSA position embedding, incremental decoding, extended pretrained registry |

---

### 4.7 vivarium/ -- Vivarium-core

**Path:** `~/Documents/biotools/cpp_ports/vivarium/`
**Size:** 3 headers, 3,033 lines

**What it does:** C++ port of Vivarium-core, a composable multi-scale simulation framework. Vivarium provides the "glue" that connects different simulators at different scales into a single hierarchical simulation.

**Biological scale:** Bridging (framework for composing simulations across all scales)

**Key concepts:**
- **Store:** Hierarchical state tree (like a nested dictionary with schema metadata)
- **Process:** A continuous-time simulation step that reads from and writes to the Store
- **Step:** A discrete-time simulation step
- **Composer:** Builds composite simulations from processes and wiring topology
- **Emitter:** Records simulation data (console, RAM, file)
- **Engine:** Orchestrates time advancement, process execution, and state updates

**Headers:**
| Header | Key Contents |
|--------|-------------|
| `vivarium_types.h` | `HierarchyPath`, `Value` (std::any), `State`, `Update`, `Schema`, `Topology`, `PortSchema`, `Flow`, value casting utilities, `Updater` (accumulate, set, merge, replace, nonnegative_accumulate, bounds), `Divider` (split, binomial, set, zero, no_divide), `Registry` (process/updater/divider/emitter registration), `deep_merge()`, `assoc_path()`, `inverse_topology()`, JSON serialization |
| `vivarium.h` | `Store` (hierarchical state tree with schema, topology, updaters, dividers), `Process` (ports_schema, next_update, time step), `Step`, `Deriver`, `Composer` (generate, initial_state, topology), `Emitter` (RAMEmitter, ConsoleEmitter, FileEmitter), `Engine` (run simulation, update, apply_updates, divide, complete timeline) |
| `vivarium_processes.h` | `GrowthRate` (exponential biomass growth), `TimelineProcess` (scheduled interventions), `Clock` (time tracking), `Division` / `DivideCondition` / `MetaDivision` (cell division logic), `Injector` (parameter injection), `Remove` (cell removal), toy composers for testing |

---

### 4.8 betsee/ -- BETSEE GUI (C++ + React)

**Path:** `~/Documents/biotools/cpp_ports/betsee/`
**Size:** 1 C++ header (2,091 lines) + React frontend

**What it does:** C++ port of BETSEE (BioElectric Tissue Simulation Engine Environment), the graphical frontend for BETSE. The C++ header implements a Qt5-based desktop GUI; the React frontend provides a modern web-based alternative.

**Biological scale:** GUI (configuration and visualization for BETSE simulations)

**`betsee.h` key contents:** Qt5-based `BetseeMainWindow` with simulation tree, parameter editors, log viewer, results display, config serialization, process management for running BETSE backend.

**React app** (`betsee/react-app/`):
| Component | Function |
|-----------|----------|
| `App.tsx` | Main layout with sidebar navigation |
| `GeneralSettings.tsx` | Time, temperature, solver parameters |
| `SpaceSettings.tsx` | Cell geometry, lattice, tissue radius |
| `IonSettings.tsx` | Ion concentrations and membrane permeabilities |
| `ChannelSettings.tsx` | Ion channel type selection and parameters |
| `PhysicsSettings.tsx` | Deformation, flow, osmotic pressure toggles |
| `TissueSettings.tsx` | Tissue profile definitions |
| `NetworkSettings.tsx` | GRN molecule/reaction/transporter configuration |
| `ExportSettings.tsx` | CSV/plot/animation export settings |
| `SimulationControl.tsx` | Run/pause/stop simulation controls |
| `ResultsViewer.tsx` | Time series charts and spatial plots |
| `LogViewer.tsx` | Simulation log output |
| `FileManager.tsx` | Load/save configuration files |
| `useSimulation.ts` | React hook for simulation state management |

---

### 4.9 cellblender/ -- CellBlender (C++ + React)

**Path:** `~/Documents/biotools/cpp_ports/cellblender/`
**Size:** 1 C++ header (1,631 lines) + React frontend

**What it does:** C++ port of CellBlender, the Blender addon for setting up and analyzing MCell simulations. The C++ backend handles MDL generation, data model management, geometry processing, and simulation control. The React frontend provides a web-based interface.

**`cellblender.h` key contents:** `Vec3`, `Face`, `GeometryObject`, `SurfaceRegion`, `MoleculeSpecies`, `Reaction`, `ReleaseSite`, `SurfaceClass`, `ReleasePattern`, `ReactionOutput`, `Partition`, `DataModel` (full simulation configuration), `MDLGenerator` (MDL file writer), `ResultsParser`, `ParameterSweep`.

**React app** (`cellblender/react-app/`):
| Component | Function |
|-----------|----------|
| `App.tsx` | Main tabbed interface |
| `MoleculeEditor.tsx` | Define molecular species (volume/surface, diffusion, color) |
| `ReactionEditor.tsx` | Define reactions with rates |
| `ReleaseSiteEditor.tsx` | Configure molecule release sites |
| `SurfaceClassEditor.tsx` | Define surface properties |
| `GeometryViewer.tsx` | 3-D wireframe/solid geometry preview |
| `MDLPreview.tsx` | Live MDL code preview |
| `SimulationPanel.tsx` | Run MCell, monitor progress |
| `ResultsViewer.tsx` | Plot reaction data output |
| `ParameterSweep.tsx` | Configure parameter sweep runs |
| `ProjectManager.tsx` | Save/load project files |
| `ModSurfRegions.tsx` | Modify surface region assignments |
| `useDataModel.ts` | React hook for data model state |
| `mdlGenerator.ts` | Client-side MDL generation |

---

### 4.10 neuropil_tools/ -- Neuropil Analysis Toolkit

**Path:** `~/Documents/biotools/cpp_ports/neuropil_tools/`
**Size:** 1 header, 1,441 lines

**What it does:** C++ port of the neuropil_tools Python package for analyzing neuropil ultrastructure from electron microscopy reconstructions. Provides mesh processing, region analysis, contact pattern detection, and spine/PSD measurement for MCell/Reconstruct/Blender workflows.

**Biological scale:** Mesh / Geometry (nanometer-scale neuronal ultrastructure)

**Key contents:** `Vec3`, `Triangle`, `Mesh` (vertices + faces + region tags), `NamedRegion`, `NeuropilObject`, `ContourPoint`, `ContourTrace`, `ContactPattern`, `SpinePSDData`, `ConnectivityRecord`, mesh I/O (OBJ, OFF, STL, MDL), surface area/volume calculation, mesh smoothing, region extraction, contact surface analysis, spine morphometry, Reconstruct series file parsing, Blender mesh export.

---

## 5. Python Packages (7)

Installed in the conda/pip environment at `~/Documents/biotools/.venv/`:

| Package | Description | Scale |
|---------|-------------|-------|
| **betse** | Bioelectric Tissue Simulation Engine (Python original) | Cellular/Tissue |
| **boltz** | Protein structure prediction (diffusion-based, AlphaFold-class) | Molecular |
| **deepxde** | Physics-informed neural networks for PDEs | Surrogate/ML |
| **esm** | Evolutionary Scale Modeling protein language models (Meta) | Molecular |
| **torch_geometric** | Graph neural network library (PyTorch Geometric) | Surrogate/ML |
| **physicsnemo** | NVIDIA PhysicsNeMo/Modulus for physics-ML | Surrogate/ML |
| **vivarium** | Composable multi-scale simulation framework | Bridging |

---

## 6. Multi-Scale Integration Map

### Scale Hierarchy

```
ATOMS/MOLECULES          SUBCELLULAR              CELLULAR/TISSUE         SURROGATE/ML
(Angstroms-nm)          (nm-um)                  (um-mm)                 (any scale)

GROMACS ----+
OpenMM  ----|---> Atomistic     Smoldyn --------+                       DeepXDE (PINNs)
Boltz   ----|    structures     MCell ----------|---> Reaction-          Modulus (FNO,
ESM     ----+    & dynamics     ReaDDy ---------|    diffusion           DeepONet,
                                COPASI ---------|    dynamics            MeshGraphNet)
                                BioNetGen ------|                        PyG (GNN)
                                NFSim ----------+

                                        |
                                        v

                                PhysiCell ------+
                                CompuCell3D ----|---> Multicellular
                                Morpheus -------|    tissue dynamics
                                Tissue Forge ---|
                                BETSE ----------+

                                        ^
                                        |
                              Vivarium (bridging)
```

### How Tools Connect Across Scales

**Molecular --> Subcellular:**
- GROMACS/OpenMM produce equilibrium structures and force parameters
- Boltz/ESM predict protein structures from sequence, providing initial coordinates for MD or binding site geometries for MCell
- Coarse-grained MD parameters inform ReaDDy particle potentials

**Subcellular --> Cellular:**
- COPASI/BioNetGen ODE/SSA models become intracellular submodels inside PhysiCell cells
- NFSim (via nfsimCInterface/libbng) provides network-free intracellular reactions inside MCell geometries
- Smoldyn/MCell reaction-diffusion outputs inform effective rates for tissue-scale models

**Cellular --> Tissue:**
- PhysiCell, CompuCell3D, and Morpheus simulate multicellular dynamics
- BETSE models bioelectric pattern formation across cell sheets
- Tissue Forge provides interactive real-time tissue mechanics

**Bridging:**
- Vivarium composes processes at different scales into a single simulation hierarchy with a shared state tree

**Surrogate/ML:**
- DeepXDE learns PDE solutions (e.g., approximating diffusion fields computed by GROMACS/COPASI)
- Modulus provides FNO/DeepONet surrogates for expensive PDE solvers
- PyG operates on graph-structured biological data (molecular graphs, cell interaction networks, mesh graphs)

**Mesh/Geometry:**
- GAMer generates high-quality meshes from EM reconstructions or molecular surfaces
- mesh_tools converts, refines, repairs, and analyzes meshes for MCell
- neuropil_tools processes neuronal ultrastructure reconstructions

**GUI:**
- BETSEE configures and visualizes BETSE simulations
- CellBlender sets up MCell simulations with geometry, molecules, reactions, and release sites

---

## 7. Build Locations

### Executables

| Tool | Binary Path |
|------|------------|
| GROMACS | `~/Documents/biotools/gromacs/build/bin/gmx` |
| Smoldyn | `~/Documents/biotools/Smoldyn/build/smoldyn` |
| COPASI (CLI) | `~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE` |
| COPASI (GUI) | `~/Documents/biotools/COPASI/copasi/CopasiUI` |
| Morpheus | `~/Documents/biotools/morpheus/build/morpheus` |
| NFSim | `~/Documents/biotools/nfsim/build/NFsim` |
| BioNetGen | `~/Documents/biotools/bionetgen/bng2/BNG2.pl` |
| run_network | `~/Documents/biotools/bionetgen/bng2/Network3/bin/run_network` |
| BETSE (Python) | `~/Documents/biotools/.venv/bin/betse` |
| mesh_tools | `~/Documents/biotools/mesh_tools/` (60 executables in subdirectories) |

### Shared Libraries

| Library | Path |
|---------|------|
| GROMACS | `~/Documents/biotools/gromacs/build/lib/libgromacs.so.12.0.0` |
| GROMACS API | `~/Documents/biotools/gromacs/build/lib/libgmxapi.so.0.4.0` |
| OpenMM | `~/Documents/biotools/openmm/build/libOpenMM.so` |
| OpenMM CPU | `~/Documents/biotools/openmm/build/libOpenMMCPU.so` |
| OpenMM PME | `~/Documents/biotools/openmm/build/libOpenMMPME.so` |
| OpenMM Amoeba | `~/Documents/biotools/openmm/build/libOpenMMAmoeba.so` |
| OpenMM Drude | `~/Documents/biotools/openmm/build/libOpenMMDrude.so` |
| OpenMM RPMD | `~/Documents/biotools/openmm/build/libOpenMMRPMD.so` |
| ReaDDy | `~/Documents/biotools/readdy/build/libreaddy.so` |
| Tissue Forge | `~/Documents/biotools/tissue-forge/build/Release/lib/libtissue-forge.so` |
| Tissue Forge C | `~/Documents/biotools/tissue-forge/build/Release/lib/libtissue-forge-c.so` |
| GAMer | `~/Documents/biotools/gamer/build/lib/libgamer.so` |
| GAMer (static) | `~/Documents/biotools/gamer/build/lib/libgamer.a` |
| libbng | `~/Documents/biotools/libbng/build/bng/liblibbng.a` |
| nfsimCInterface | `~/Documents/biotools/nfsimCInterface/build/libnfsim_c.so` |
| nfsimCInterface (static) | `~/Documents/biotools/nfsimCInterface/build/libnfsim_c_static.a` |
| CompuCell3D core | `~/Documents/biotools/CompuCell3D/build/core/CompuCell3D/libCC3DCompuCellLib.so` |
| BioNetGen core | `~/Documents/biotools/bionetgen/bng-graph/BNGcore/BNGcore.so` |

### C++ Header-Only Ports (no build required)

All headers are ready to `#include` directly:

| Port | Include Path |
|------|-------------|
| BETSE | `~/Documents/biotools/cpp_ports/betse/betse.h` |
| Modulus | `~/Documents/biotools/cpp_ports/modulus/modulus.h` |
| PyG | `~/Documents/biotools/cpp_ports/pyg/pyg.h` |
| DeepXDE | `~/Documents/biotools/cpp_ports/deepxde/deepxde.h` |
| Boltz | `~/Documents/biotools/cpp_ports/boltz/boltz.h` |
| ESM | `~/Documents/biotools/cpp_ports/esm/esm.h` |
| Vivarium | `~/Documents/biotools/cpp_ports/vivarium/vivarium.h` |
| BETSEE | `~/Documents/biotools/cpp_ports/betsee/betsee.h` |
| CellBlender | `~/Documents/biotools/cpp_ports/cellblender/cellblender.h` |
| Neuropil Tools | `~/Documents/biotools/cpp_ports/neuropil_tools/neuropil_tools.h` |

### React Frontends

| App | Path | Dev Server |
|-----|------|-----------|
| BETSEE | `~/Documents/biotools/cpp_ports/betsee/react-app/` | `npm run dev` |
| CellBlender | `~/Documents/biotools/cpp_ports/cellblender/react-app/` | `npm run dev` |

---

## 8. Total Scale Coverage Summary

| Biological Scale | Tools | Scale Range |
|-----------------|-------|-------------|
| Atomistic/Molecular | GROMACS, OpenMM, Boltz, ESM | 1 A -- 100 nm |
| Subcellular | Smoldyn, MCell, ReaDDy, COPASI, BioNetGen, NFSim | 1 nm -- 10 um |
| Cellular/Tissue | BETSE, PhysiCell, CompuCell3D, Morpheus, Tissue Forge | 1 um -- 10 mm |
| Bridging | Vivarium | All scales |
| Surrogate/ML | DeepXDE, Modulus, PyG | Any scale |
| Mesh/Geometry | GAMer, mesh_tools, neuropil_tools | Any scale |
| GUI | BETSEE, CellBlender | Config/Viz |

---

*This document covers all 16 built C++ repositories, 10 C++ header-only ports (with 59 total headers and 41,828 lines of C++), 2 React frontend applications, and 7 Python packages in the biotools pipeline.*
