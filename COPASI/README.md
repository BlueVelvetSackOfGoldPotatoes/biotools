# COPASI — COmplex PAthway SImulator

COPASI is a comprehensive software application for **simulation and analysis of biochemical networks** and their dynamics. It supports deterministic (ODE) and stochastic (Gillespie SSA, tau-leaping) simulation, steady-state analysis, metabolic control analysis, parameter estimation, sensitivity analysis, optimization, and more.

## What It Does

COPASI is designed for researchers studying biochemical reaction networks — from simple enzyme kinetics to genome-scale metabolic models. It provides:

- **Time-course simulation**: Deterministic (ODE) and stochastic (SSA, hybrid) methods for computing how species concentrations change over time
- **Steady-state analysis**: Find and analyze equilibrium states using Newton's method
- **Metabolic Control Analysis (MCA)**: Compute elasticities, control coefficients, and response coefficients
- **Parameter estimation**: Fit model parameters to experimental data using optimization algorithms (Levenberg-Marquardt, genetic algorithms, particle swarm, simulated annealing)
- **Sensitivity analysis**: Identify which parameters most influence model behavior
- **Optimization**: Find parameter values that minimize/maximize objective functions
- **Lyapunov exponents**: Analyze dynamical stability

## Biological Scale

**Subcellular** — well-mixed biochemical networks. Models range from a few reactions (simple enzyme kinetics) to hundreds of reactions (genome-scale metabolic models, signaling cascades).

## Key Algorithms

### Simulation Methods
| Method | Algorithm | Use Case |
|--------|-----------|----------|
| Deterministic | LSODA (auto stiff/non-stiff) | Standard ODE simulation |
| Deterministic | Radau5 | Stiff systems |
| Stochastic | Gillespie SSA (direct) | Exact stochastic kinetics |
| Stochastic | Next-reaction method | Efficient SSA variant |
| Stochastic | Tau-leaping | Approximate stochastic (faster) |
| Stochastic | Adaptive SSA | Auto-tuning stochastic |
| Hybrid | ODE/SSA | Mixed deterministic-stochastic |

### Analysis Methods
- **Steady-state**: Newton's method with damping, integration-based fallback
- **MCA**: Full elasticity and control coefficient computation
- **Parameter estimation**: Levenberg-Marquardt, genetic algorithms, differential evolution, particle swarm optimization, simulated annealing, scatter search, SRES
- **Sensitivity analysis**: Local and global methods
- **Linear stability analysis**: Eigenvalue computation at steady state

### Model Standards
- Full **SBML** import/export (Systems Biology Markup Language)
- **SED-ML** support for reproducible simulation experiments
- Native COPASI `.cps` format with full model and task specification

## Directory Structure

```
COPASI/
├── copasi/
│   ├── CopasiSE/              # Command-line solver
│   ├── CopasiUI/              # Qt-based graphical interface
│   ├── model/                 # Model representation
│   ├── trajectory/            # Time-course simulation
│   ├── steadystate/           # Steady-state analysis
│   ├── optimization/          # Optimization algorithms
│   ├── parameterFitting/      # Parameter estimation
│   ├── sensitivities/         # Sensitivity analysis
│   ├── mca/                   # Metabolic Control Analysis
│   ├── sbml/                  # SBML import/export
│   ├── bindings/              # Python and R bindings
│   └── ...
├── build/
│   └── copasi/CopasiSE/
│       └── CopasiSE           # Compiled command-line binary
├── copasi-dependencies/       # -> ../copasi-dependencies/
├── sbml-testsuite/            # SBML compliance tests
└── ...
```

## Usage

### Command-Line (CopasiSE)
```bash
# Run a COPASI model
~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE model.cps

# Import and simulate an SBML model
~/Documents/biotools/COPASI/build/copasi/CopasiSE/CopasiSE --importSBML model.xml
```

### Python Bindings
```python
import COPASI
dataModel = COPASI.CRootContainer.addDatamodel()
dataModel.loadModel("model.cps")
task = dataModel.getTask("Time-Course")
task.process(True)
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.cps` | COPASI native model + task files |
| Input | `.xml` | SBML models |
| Input | SED-ML | Simulation experiment descriptions |
| Output | CSV/TSV | Time course data, steady-state values |
| Output | `.cps` | Model with results |
| Output | SBML | Exported models |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **BioNetGen** | Model exchange via SBML |
| **PhysiCell** | COPASI ODE models as intracellular submodels |
| **CompuCell3D** | SBML models embedded per cell |
| **Vivarium** | Can wrap COPASI as a Vivarium Process for multiscale composition |
| **copasi-dependencies** | Required build dependencies (libSBML, etc.) |

## Build

```bash
cd ~/Documents/biotools/COPASI
mkdir -p build && cd build
cmake -DCOPASI_DEPENDENCY_DIR=../../copasi-dependencies/build ..
make -j$(nproc)
```

## License

Artistic License 2.0

## References

- Hoops, S. et al. (2006). "COPASI — a COmplex PAthway SImulator." *Bioinformatics*, 22(24):3067-3074.
- Bergmann, F.T. et al. (2017). "COPASI and its applications in biotechnology." *Journal of Biotechnology*, 261:215-220.
