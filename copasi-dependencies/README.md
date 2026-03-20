# copasi-dependencies — Build Dependencies for COPASI

This repository contains the **source code and build infrastructure** for all external libraries required to compile COPASI. It provides a self-contained way to build COPASI's dependencies from source with consistent, tested configurations.

## What It Contains

copasi-dependencies builds and installs the following libraries:

| Library | Description | Purpose in COPASI |
|---------|-------------|-------------------|
| **libSBML** | Systems Biology Markup Language library | SBML model import/export |
| **libCombine** | COMBINE archive library | Reading/writing COMBINE archives |
| **clapack** | C translation of LAPACK | Linear algebra for numerical methods |
| **cppunit** | C++ unit testing framework | Testing infrastructure |
| **crossguid** | Cross-platform GUID generation | Unique identifiers |
| **cpu_features** | CPU feature detection | Runtime optimization |
| **NativeJIT** | JIT compiler for mathematical expressions | Fast expression evaluation |

## Directory Structure

```
copasi-dependencies/
├── src/
│   ├── libSBML/                # SBML library source
│   ├── libCombine/             # COMBINE archive library source
│   ├── clapack/                # LAPACK C translation
│   ├── cppunit/                # Unit testing framework
│   ├── crossguid/              # GUID generation
│   ├── cpu_features/           # CPU feature detection
│   └── NativeJIT/              # Expression JIT compiler
├── build/
│   ├── share/libsbml/          # Installed libSBML
│   └── ...                     # Other installed libraries
└── CMakeLists.txt              # Top-level build script
```

## Build

```bash
cd ~/Documents/biotools/copasi-dependencies
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

The built libraries are installed into the `build/` directory and can be referenced when building COPASI:

```bash
cd ~/Documents/biotools/COPASI/build
cmake -DCOPASI_DEPENDENCY_DIR=../../copasi-dependencies/build ..
```

## Connections to Other Biotools

| Tool | Relationship |
|------|-------------|
| **COPASI** | These dependencies are required to build COPASI |
| **libSBML** | Also used by Tissue Forge and other SBML-consuming tools |

## License

Individual components are under their own licenses (LGPL for libSBML, BSD for clapack, etc.).
