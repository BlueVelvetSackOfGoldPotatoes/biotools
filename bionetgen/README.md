# BioNetGen — Rule-Based Modeling Framework

BioNetGen is a **rule-based modeling framework** for specifying and simulating biochemical reaction networks. Instead of manually enumerating every molecular species and reaction, modelers write compact rules describing classes of biochemical transformations. BioNetGen can then automatically generate the full reaction network or simulate it directly using network-free methods via NFSim.

## What It Does

Many biological signaling pathways involve molecules with multiple binding sites and modification states, leading to **combinatorial explosion** in the number of possible molecular species. BioNetGen addresses this fundamental challenge by allowing modelers to:

- Define molecular species as **structured objects** with typed binding sites, internal states, and compartment locations
- Write **reaction rules** that describe how patterns of molecules interact (e.g., "any kinase phosphorylates any substrate with a free SH2 domain")
- **Automatically generate** the complete reaction network via graph rewriting from a compact set of rules
- Simulate without network generation using **network-free** methods (via NFSim) for models with astronomical state spaces
- Perform **deterministic (ODE)** and **stochastic (SSA/Gillespie)** simulations
- Run **parameter scans** and **bifurcation analysis**

## Biological Scale

**Subcellular** — biochemical reaction networks (well-mixed or spatially resolved when coupled with other tools). Models range from simple enzyme kinetics to full receptor signaling cascades with hundreds of rules and millions of possible species.

## The BNGL Language

BioNetGen uses its own domain-specific language (BNGL) for model specification:

```bngl
begin model
  begin molecule types
    L(r)           # Ligand with receptor-binding site
    R(l, Y~U~P)   # Receptor with ligand site and phosphosite (U=unphosphorylated, P=phosphorylated)
  end molecule types

  begin seed species
    L(r)       1000
    R(l, Y~U)  200
  end seed species

  begin reaction rules
    # Ligand-receptor binding (reversible)
    L(r) + R(l) <-> L(r!1).R(l!1)  kon, koff

    # Receptor phosphorylation (only when ligand-bound)
    L(r!1).R(l!1, Y~U) -> L(r!1).R(l!1, Y~P)  kp
  end reaction rules

  begin observables
    Molecules  Bound    L(r!1).R(l!1)
    Molecules  PhosphoR R(Y~P)
  end observables
end model

generate_network({overwrite=>1})
simulate({method=>"ode", t_end=>100, n_steps=>1000})
```

## Key Features

### Simulation Methods
- **Network generation** via graph rewriting (`generate_network` command)
- **ODE simulation** via the `run_network` binary (CVODE solver)
- **SSA simulation** (Gillespie direct method) via `run_network`
- **Network-free simulation** via NFSim for combinatorially complex models
- **Hybrid** deterministic/stochastic methods

### Analysis Capabilities
- Parameter scanning over arbitrary model parameters
- Sensitivity analysis
- Steady-state computation
- Bifurcation analysis

### Model Features
- Molecules with sites, states, and bonds
- Rate laws: mass-action, Michaelis-Menten, Hill functions, custom expressions
- Compartments for spatial organization
- Energy-based models for thermodynamic consistency
- Functions and expressions for complex rate formulations

## Directory Structure

```
bionetgen/
├── bng2/
│   ├── BNG2.pl              # Main BioNetGen Perl script
│   ├── Network3/
│   │   └── bin/
│   │       └── run_network   # Compiled ODE/SSA simulation binary
│   ├── Models2/              # Example BNGL models
│   └── Perl2/                # Core Perl modules for BNGL processing
├── bng-graph/
│   └── BNGcore/
│       └── BNGcore.so        # Core graph processing shared library
├── PhiBPlot/                 # Visualization tools
└── ...
```

## Usage

```bash
# Run a BNGL model
perl ~/Documents/biotools/bionetgen/bng2/BNG2.pl model.bngl

# Direct ODE/SSA simulation of a generated network
~/Documents/biotools/bionetgen/bng2/Network3/bin/run_network model.net

# Network generation followed by ODE simulation (in BNGL file)
# Add to end of .bngl file:
#   generate_network({overwrite=>1})
#   simulate({method=>"ode", t_end=>100, n_steps=>1000})
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.bngl` | Rule-based model specification files |
| Output | `.net` | Generated reaction network (species + reactions) |
| Output | `.gdat` | Time series data for groups/observables |
| Output | `.cdat` | Time series data for individual species concentrations |
| Output | `.xml` | XML model representation for NFSim consumption |

## Build

The `run_network` simulation engine is compiled from C++ source:

```bash
cd ~/Documents/biotools/bionetgen/bng2/Network3
mkdir -p build && cd build
cmake ..
make
```

The main BNG2.pl script requires Perl (typically available on all Unix systems).

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **NFSim** | Network-free stochastic simulation of BioNetGen models |
| **nfsimCInterface** / **libbng** | C/C++ APIs for embedding BioNetGen capabilities |
| **MCell** | Uses BioNetGen rules for intracellular reaction networks in 3D geometries |
| **Smoldyn** | Can read `.bngl` rule files for spatial simulation |
| **PhysiCell** | Imports SBML models that BioNetGen can export |
| **COPASI** | Model exchange via SBML format |

## License

MIT License (BioNetGen core), with individual components under their own licenses.

## References

- Faeder, J.R., Blinov, M.L. & Hlavacek, W.S. (2009). "Rule-Based Modeling of Biochemical Systems with BioNetGen." *Methods in Molecular Biology*, 500:113-167.
- Harris, L.A. et al. (2016). "BioNetGen 2.2: advances in rule-based modeling." *Bioinformatics*, 32(21):3366-3368.
