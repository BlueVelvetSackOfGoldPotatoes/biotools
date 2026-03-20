# GROMACS — GROningen MAchine for Chemical Simulations

GROMACS is a high-performance **molecular dynamics (MD) engine** for simulating the behavior of biomolecular systems — proteins, lipids, nucleic acids, and their interactions with solvents and ions at atomistic or coarse-grained resolution. It is one of the fastest and most widely used MD codes in the world.

## What It Does

GROMACS computes the forces between all atoms in a molecular system and integrates Newton's equations of motion to produce a trajectory — a time series of atomic positions and velocities. This enables researchers to:

- **Simulate protein dynamics**: Folding, conformational changes, ligand binding, allosteric transitions
- **Study membrane systems**: Lipid bilayer assembly, membrane protein insertion, transport
- **Analyze nucleic acids**: DNA/RNA dynamics, protein-DNA interactions
- **Compute thermodynamic quantities**: Free energies of binding, solvation, or mutation via alchemical methods
- **Explore enhanced sampling**: Replica exchange MD, metadynamics, umbrella sampling
- **Run coarse-grained simulations**: Martini force field for longer timescales and larger systems

## Biological Scale

**Molecular / Atomistic** — simulates individual atoms, typically 10^3 to 10^7 atoms (Angstroms to tens of nanometers), over nanosecond to microsecond timescales.

## Key Algorithms

### Integration
- **Velocity Verlet** and **leap-frog** integrators
- Multiple time-stepping for efficiency
- Constraints: LINCS (bonds to hydrogen), SETTLE (rigid water)

### Long-Range Interactions
- **Particle Mesh Ewald (PME)** for long-range electrostatics
- Reaction-field electrostatics
- Verlet neighbor lists with buffered pair interactions

### Enhanced Sampling
- Replica Exchange Molecular Dynamics (REMD)
- Metadynamics (with PLUMED plugin)
- Umbrella sampling
- Free energy perturbation (FEP) and thermodynamic integration (TI)
- Accelerated weight histogram (AWH) method

### Performance
- **SIMD-optimized** nonbonded kernels (SSE, AVX, AVX-512)
- **GPU acceleration** via CUDA, OpenCL, and SYCL backends
- Multi-level parallelism: thread-MPI, OpenMP, MPI
- Domain decomposition for distributed memory systems
- Offloading of PME, bonded, and nonbonded calculations to GPU

### Force Fields
- AMBER, CHARMM, OPLS-AA, GROMOS all-atom force fields
- Martini coarse-grained force field
- TIP3P, TIP4P, SPC/E water models

## Directory Structure

```
gromacs/
├── src/
│   ├── gromacs/                # Core library source
│   │   ├── mdrun/              # MD simulation engine
│   │   ├── gmxpreprocess/      # System preparation tools
│   │   ├── trajectoryanalysis/ # Analysis framework
│   │   └── ...
│   ├── programs/               # Command-line tools (gmx)
│   └── external/               # Bundled dependencies (FFTW, etc.)
├── share/
│   └── top/                    # Force field topology files
├── build/
│   ├── bin/gmx                 # Main executable
│   └── lib/
│       ├── libgromacs.so.12.0.0  # Core shared library
│       └── libgmxapi.so.0.4.0   # C++ API library
├── python_packaging/           # Python bindings (gmxapi)
└── ...
```

## Usage

### Typical MD Workflow

```bash
export PATH=~/Documents/biotools/gromacs/build/bin:$PATH

# 1. Prepare the system (generate topology, add solvent, ions)
gmx pdb2gmx -f protein.pdb -o processed.gro -water tip3p
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt cubic
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
gmx genion -s ions.tpr -o system.gro -p topol.top -pname NA -nname CL -neutral

# 2. Energy minimization
gmx grompp -f minim.mdp -c system.gro -p topol.top -o em.tpr
gmx mdrun -deffnm em

# 3. Equilibration (NVT then NPT)
gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
gmx mdrun -deffnm nvt
gmx grompp -f npt.mdp -c nvt.gro -p topol.top -o npt.tpr
gmx mdrun -deffnm npt

# 4. Production MD
gmx grompp -f md.mdp -c npt.gro -p topol.top -o md.tpr
gmx mdrun -deffnm md

# 5. Analysis
gmx rms -s md.tpr -f md.xtc -o rmsd.xvg
gmx energy -f md.edr -o energy.xvg
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.gro` | Coordinate files (GROMACS format) |
| Input | `.pdb` | Coordinate files (Protein Data Bank) |
| Input | `.top` | Molecular topology (atoms, bonds, force field) |
| Input | `.mdp` | Run parameter files (simulation settings) |
| Input | `.tpr` | Portable binary run input (preprocessed) |
| Output | `.trr` / `.xtc` | Trajectory files (full precision / compressed) |
| Output | `.edr` | Energy data |
| Output | `.xvg` | Analysis output (RMSD, RMSF, energy, etc.) |

## Build

```bash
cd ~/Documents/biotools/gromacs
mkdir -p build && cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DGMX_GPU=CUDA  # or OpenCL/SYCL
make -j$(nproc)
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **Boltz / ESM** | Predicted structures serve as starting configurations for MD |
| **OpenMM** | Alternative MD engine with complementary strengths (custom forces, Python-native) |
| **ReaDDy** | Coarse-grained parameters from GROMACS inform mesoscale particle potentials |
| **DeepXDE / Modulus** | Train neural surrogates on GROMACS simulation data |
| **Vivarium** | MD simulations can be wrapped as Vivarium processes |

## License

LGPL v2.1 (with some components under other open-source licenses)

## References

- Abraham, M.J. et al. (2015). "GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers." *SoftwareX*, 1-2:19-25.
- Lindahl, E. et al. (2001). "GROMACS 3.0: a package for molecular simulation and trajectory analysis." *Journal of Molecular Modeling*, 7:306-317.
