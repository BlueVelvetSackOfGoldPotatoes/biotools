# libbng — Lightweight BioNetGen Library

libbng is a **lightweight, embeddable C++ library** that provides core BioNetGen capabilities for rule-based biochemical reaction network generation. It is designed for integration into other simulation tools that need rule-based reaction processing without the full BioNetGen/Perl infrastructure.

## What It Does

libbng extracts the essential graph-rewriting and network-generation algorithms from BioNetGen into a standalone C++ library. This enables other simulation engines (particularly MCell) to:

- Parse and process rule-based model specifications
- Perform graph-based pattern matching on molecular complexes
- Generate reaction networks from rules on-the-fly
- Evaluate observables over complex molecular populations
- Integrate rule-based chemistry directly into C++ simulation codes

## Biological Scale

**Subcellular** — provides an API bridge for rule-based biochemical reaction network processing.

## Key Components

- **Graph rewriting engine**: Pattern matching and reaction rule application on molecular graphs
- **Species generation**: Enumerate molecular species from rules
- **Observable evaluation**: Track molecular patterns across populations
- **Canonical form computation**: Using nauty library for graph isomorphism detection
- **Efficient hashing**: sparsehash for fast species and pattern lookup

## Directory Structure

```
libbng/
├── bng/                        # Core library source
│   ├── ...                     # BNG graph rewriting engine
│   └── CMakeLists.txt
├── libs/
│   ├── nauty/                  # Graph automorphism library (for canonical forms)
│   └── sparsehash/             # Google sparse hash maps
├── build/
│   └── bng/
│       └── liblibbng.a         # Static library (built artifact)
└── CMakeLists.txt
```

## Build

```bash
cd ~/Documents/biotools/libbng
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Produces: `build/bng/liblibbng.a` (static library)

## Usage

Link against `liblibbng.a` in your C++ project:

```cmake
target_link_libraries(my_simulator PRIVATE libbng)
target_include_directories(my_simulator PRIVATE ${LIBBNG_INCLUDE_DIR})
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | Primary consumer — uses libbng for intracellular rule-based reactions |
| **BioNetGen** | Full-featured parent project; libbng extracts the core C++ engine |
| **NFSim** | Alternative network-free approach; libbng provides network-generation alternative |
| **nfsimCInterface** | Companion C API wrapper for NFSim (different approach to same problem) |

## Dependencies

- CMake 3.x
- C++11 compiler
- nauty (included in `libs/`)
- sparsehash (included in `libs/`)

## License

MIT License
