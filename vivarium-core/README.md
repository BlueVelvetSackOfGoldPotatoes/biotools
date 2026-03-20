# Vivarium Core — Composable Multi-Scale Simulation Framework

Vivarium is a **composable simulation framework** for building multi-scale models of biological systems. It provides the "glue" that connects different simulation tools operating at different scales into a single, coherent hierarchical simulation with a shared state tree.

## What It Does

Biological systems span many scales simultaneously — from molecular interactions to intracellular networks to multicellular tissues. No single simulator can cover all scales. Vivarium solves this by:

- **Composing simulators**: Connect any simulation tool as a "Process" that reads from and writes to a shared hierarchical state tree
- **Managing time**: Coordinate time advancement across processes with different time scales (e.g., fast molecular dynamics + slow cell cycle)
- **Hierarchical state**: Organize simulation state in a nested tree structure where each branch can have its own schema, updaters, and dividers
- **Cell division**: Built-in support for cell division with configurable state partitioning
- **Modular design**: Add, remove, or swap simulation components without rewriting code
- **Data recording**: Configurable emitters for saving simulation data (RAM, file, console)

### Example Multi-Scale Composition
```
Whole Cell Simulation
├── Metabolism (COPASI ODE model, dt=1s)
│   ├── ATP concentration
│   ├── Amino acid pools
│   └── ...
├── Gene Expression (stochastic, dt=10s)
│   ├── mRNA counts
│   ├── Protein counts
│   └── ...
├── Division (event-based)
│   └── Triggers when volume > threshold
└── Growth (continuous, dt=1s)
    └── Volume, mass
```

## Biological Scale

**Bridging** — framework for composing simulations across all scales, from molecular to cellular to tissue.

## Key Concepts

### Store (Hierarchical State Tree)
The Store is a hierarchical data structure (like a nested dictionary) that holds the complete simulation state. Each node in the tree can have:
- **Schema**: Type information and metadata
- **Updater**: How updates are applied (accumulate, set, merge, replace, nonnegative_accumulate, bounds)
- **Divider**: How state is partitioned during cell division (split, binomial, set, zero)
- **Topology**: Wiring that connects Process ports to Store paths

### Process
A Process is a continuous-time simulation step. It:
- Declares its **ports_schema**: what state variables it reads and writes
- Implements **next_update(timestep)**: computes state changes for a given time interval
- Has a characteristic **time_step**: how often it should be called

### Step
Like a Process, but fires at discrete events rather than continuous time.

### Composer
A Composer wires together multiple Processes and their topologies to create a composite simulation. It generates:
- Process instances
- Wiring topology (which Process ports connect to which Store paths)
- Initial state

### Engine
The Engine orchestrates the simulation:
- Advances time
- Calls each Process at appropriate intervals
- Applies updates to the Store
- Handles cell division events
- Records data via Emitters

## Directory Structure

```
vivarium-core/
├── vivarium/
│   ├── core/
│   │   ├── store.py            # Hierarchical state tree
│   │   ├── process.py          # Process base class
│   │   ├── composer.py         # Composition framework
│   │   ├── engine.py           # Simulation engine
│   │   ├── emitter.py          # Data recording
│   │   ├── registry.py         # Process/updater registration
│   │   └── ...
│   ├── processes/              # Built-in processes
│   │   ├── timeline.py         # Scheduled interventions
│   │   ├── clock.py            # Time tracking
│   │   ├── growth_rate.py      # Exponential growth
│   │   ├── divide_condition.py # Division triggers
│   │   └── ...
│   └── library/                # Utilities and examples
├── doc/                        # Documentation
├── tests/                      # Test suite
└── ...
```

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/vivarium-core
pip install -e .
```

### Dependencies
- Python 3.x
- NumPy
- Parsimonious (for topology parsing)
- Pymongo (optional, for MongoDB emitter)

## Usage

```python
from vivarium.core.engine import Engine
from vivarium.core.process import Process

# Define a custom process
class GrowthProcess(Process):
    defaults = {'growth_rate': 0.01}

    def ports_schema(self):
        return {
            'global': {
                'volume': {
                    '_default': 1.0,
                    '_updater': 'accumulate',
                    '_divider': 'split',
                }
            }
        }

    def next_update(self, timestep, states):
        volume = states['global']['volume']
        return {'global': {'volume': volume * self.parameters['growth_rate'] * timestep}}

# Compose and run
composite = {
    'growth': {
        '_type': 'process',
        'address': 'local:GrowthProcess',
        'config': {'growth_rate': 0.02},
        'inputs': {'global': ['global']},
        'outputs': {'global': ['global']},
    }
}

engine = Engine(composite=composite)
engine.update(1000)  # Run for 1000 time units
data = engine.emitter.get_data()
```

## Built-in Processes

| Process | Description |
|---------|-------------|
| `GrowthRate` | Exponential biomass/volume growth |
| `TimelineProcess` | Scheduled interventions at specified times |
| `Clock` | Tracks simulation time |
| `DivideCondition` | Triggers cell division based on conditions |
| `MetaDivision` | Manages division of hierarchical state |
| `Injector` | Injects parameter changes |
| `Remove` | Removes cells/agents from simulation |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **COPASI** | Wrap COPASI ODE models as Vivarium Processes for biochemical kinetics |
| **PhysiCell** | Wrap PhysiCell as a Vivarium Process for multicellular dynamics |
| **GROMACS / OpenMM** | Wrap MD simulations as Vivarium Processes for molecular-scale dynamics |
| **Smoldyn / ReaDDy** | Wrap spatial stochastic simulators as Vivarium Processes |
| **BETSE** | Wrap bioelectric simulations as Vivarium Processes |
| **cpp_ports/vivarium/** | C++17 header-only port (3 headers, ~3,000 lines) of the core framework |

## License

Apache License 2.0

## References

- Agmon, E. et al. (2022). "Vivarium: an interface and engine for integrative multiscale modeling in computational biology." *Bioinformatics*, 38(7):1972-1979.
