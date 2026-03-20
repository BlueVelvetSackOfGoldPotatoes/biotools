# nfsimCInterface — C API Wrapper for NFSim

nfsimCInterface provides a **C language interface** to the NFSim (Network-Free Simulator) API, enabling other programs — particularly MCell — to embed network-free stochastic simulation as a library call rather than spawning a separate process.

## What It Does

NFSim is normally used as a standalone command-line tool. nfsimCInterface wraps NFSim's C++ internals behind a clean C API, allowing:

- **Library embedding**: Other simulation engines can call NFSim functions directly without IPC overhead
- **MCell integration**: MCell uses this interface to run rule-based intracellular reactions within its 3D spatial simulation
- **Programmatic control**: Initialize, step, query, and reset NFSim simulations from C/C++ code
- **Shared memory**: Molecular populations are accessible directly, avoiding serialization/deserialization

## Biological Scale

**Subcellular** — API bridge enabling network-free rule-based reaction simulation within other spatial simulation tools.

## API Overview

The C interface exposes functions for:

```c
// Initialize NFSim with a BioNetGen XML model
int nfsim_init(const char* xml_file);

// Advance the simulation by one step
int nfsim_step(double dt);

// Query molecule counts and observable values
int nfsim_get_observable(const char* name, double* value);

// Add/remove molecules
int nfsim_add_molecules(const char* species, int count);

// Reset the simulation
int nfsim_reset();

// Cleanup
int nfsim_cleanup();
```

## Directory Structure

```
nfsimCInterface/
├── src/
│   ├── nfsim_c.h               # C API header
│   ├── nfsim_c.cpp             # C wrapper implementation
│   └── ...
├── build/
│   ├── libnfsim_c.so           # Shared library
│   └── libnfsim_c_static.a     # Static library
└── CMakeLists.txt
```

## Build

```bash
cd ~/Documents/biotools/nfsimCInterface
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Produces:
- `libnfsim_c.so` — shared library for dynamic linking
- `libnfsim_c_static.a` — static library for static linking

## Usage

### Linking
```cmake
target_link_libraries(my_simulator PRIVATE nfsim_c)
target_include_directories(my_simulator PRIVATE ${NFSIM_C_INCLUDE_DIR})
```

### In Code
```c
#include "nfsim_c.h"

// Initialize with a BioNetGen XML model
nfsim_init("model.xml");

// Run simulation steps
for (int i = 0; i < 1000; i++) {
    nfsim_step(0.001);  // step by 1 ms
}

// Query results
double count;
nfsim_get_observable("PhosphoProtein", &count);

// Cleanup
nfsim_cleanup();
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | Primary consumer — embeds NFSim for intracellular rule-based reactions |
| **NFSim** | The simulation engine being wrapped |
| **BioNetGen** | Generates the XML model files consumed by this interface |
| **libbng** | Alternative approach — provides network generation rather than network-free simulation |

## Dependencies

- NFSim source/headers
- CMake 3.x
- C++11 compiler

## License

MIT License
