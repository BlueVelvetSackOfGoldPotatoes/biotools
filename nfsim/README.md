# NFSim — Network-Free Stochastic Simulator

NFSim (Network-Free Simulator) is a stochastic simulation engine that simulates **rule-based biochemical models without generating the full reaction network**. This enables simulation of models with combinatorially large or infinite state spaces — such as multi-site phosphorylation cascades, scaffold-mediated signaling, and polymerization — that are intractable for traditional network-based approaches.

## What It Does

In many biological signaling systems, a small number of molecular components can generate astronomically large numbers of distinct molecular species through combinations of binding states, modifications, and complex formation. Traditional simulation approaches require enumerating all species and reactions, which becomes impossible for complex models.

NFSim solves this by:

- **Tracking individual molecules** as graph objects with sites, states, and bonds
- **Firing reactions on-the-fly** by pattern matching rules against the current molecular population
- **Never generating the full network**: Only species that actually form during simulation are ever created
- **Supporting complex observables**: Track molecular patterns across the population in real time

### Example Use Cases
- Multi-site phosphorylation (e.g., EGFR with 20+ phosphorylation sites → 2^20+ species)
- Scaffold-mediated signaling complexes
- Receptor oligomerization and clustering
- Actin polymerization and filament dynamics
- Flagellar motor assembly
- Chemotaxis adaptation networks

## Biological Scale

**Subcellular** — complex biochemical reaction networks with combinatorially large state spaces.

## Key Algorithms

- **Network-free stochastic simulation**: Extended Gillespie-type algorithm that operates on molecular graphs rather than species populations
- **Graph-based molecule representation**: Each molecule is a labeled graph with sites, internal states, and bonds to other molecules
- **Pattern matching**: Reaction rules are matched against the current population using subgraph isomorphism
- **On-the-fly reaction firing**: Reactions are selected and executed without pre-computing the network
- **Complex tracking**: Molecular complexes are tracked as connected components of the molecular graph
- **Observable evaluation**: Efficient counting of molecular patterns for output

## Directory Structure

```
nfsim/
├── src/
│   ├── NFcore/                 # Core simulation engine
│   │   ├── molecule.h/cpp      # Molecule representation
│   │   ├── reactionClass.h/cpp # Reaction rule classes
│   │   ├── system.h/cpp        # System state management
│   │   └── ...
│   ├── NFinput/                # Input parsing (BioNetGen XML)
│   ├── NFfunction/             # Function evaluation
│   ├── NFoutput/               # Output generation
│   └── ...
├── models/                     # Example models
│   ├── actin/                  # Actin polymerization
│   ├── chemotaxisAdaptation/   # Chemotaxis signaling
│   ├── fceRI_compendium/       # FcεRI receptor signaling
│   ├── flagellar_motor/        # Bacterial flagellar motor
│   ├── multisite_phos/         # Multi-site phosphorylation
│   ├── tlbr/                   # Trivalent ligand bivalent receptor
│   └── ...
├── build/
│   └── NFsim                   # Compiled binary
└── tools/                      # Utility scripts
```

## Usage

NFSim reads BioNetGen XML files generated from `.bngl` models:

```bash
# Step 1: Generate XML from a BNGL model using BioNetGen
perl ~/Documents/biotools/bionetgen/bng2/BNG2.pl model.bngl

# Step 2: Run NFSim on the generated XML
~/Documents/biotools/nfsim/build/NFsim -xml model.xml -sim 100 -oSteps 1000

# Common options:
#   -xml <file>     Input BioNetGen XML file
#   -sim <time>     Simulation time
#   -oSteps <n>     Number of output steps
#   -seed <n>       Random number seed
#   -utl <n>        Universal traversal limit (truncation)
#   -gml <n>        Global molecule limit
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | BioNetGen XML (`.xml`) | Generated from `.bngl` files by BioNetGen |
| Output | `.gdat` | Time series of observable values |
| Output | Console | Simulation progress and statistics |

## Example Models

| Model | Description |
|-------|-------------|
| `multisite_phos` | Multi-site phosphorylation demonstrating combinatorial complexity |
| `fceRI_compendium` | FcεRI receptor signaling with extensive complex formation |
| `chemotaxisAdaptation` | Bacterial chemotaxis receptor methylation/demethylation |
| `flagellar_motor` | Bacterial flagellar motor assembly |
| `actin` | Actin filament polymerization dynamics |
| `tlbr` | Trivalent ligand, bivalent receptor aggregation |

## Build

```bash
cd ~/Documents/biotools/nfsim
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **BioNetGen** | Generates the XML input files that NFSim reads |
| **nfsimCInterface** | C API wrapper enabling MCell to call NFSim as a library |
| **libbng** | Alternative approach: network generation instead of network-free simulation |
| **MCell** | Embeds NFSim for intracellular rule-based reactions in 3D geometry |
| **Smoldyn** | Can couple with rule-based models |

## License

MIT License

## References

- Sneddon, M.W., Faeder, J.R. & Emonet, T. (2011). "Efficient modeling, simulation and coarse-graining of biological complexity with NFsim." *Nature Methods*, 8(2):177-183.
