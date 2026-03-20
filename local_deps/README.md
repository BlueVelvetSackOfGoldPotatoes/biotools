# local_deps — Shared Local Dependency Builds

This directory contains **shared library dependencies** that are built locally and used by multiple tools in the biotools pipeline. Rather than duplicating these dependencies across individual projects, they are compiled once here and linked from a common prefix.

## What It Contains

local_deps builds and installs several foundational libraries that other biotools projects depend on:

### Built Dependencies

| Library | Description | Used By |
|---------|-------------|---------|
| **Eigen** | C++ template library for linear algebra (matrices, vectors, solvers) | GAMer, Morpheus, Tissue Forge, and others |
| **Assimp** | Open Asset Import Library for 3D model format I/O (OBJ, STL, FBX, etc.) | Tissue Forge, mesh tools |
| **nlohmann/json** | Modern JSON library for C++ | Various tools needing JSON configuration |
| **libxml2** | XML C parser and toolkit | SBML processing, configuration files |

## Directory Structure

```
local_deps/
├── build/
│   ├── assimp/                 # Assimp source and build
│   ├── eigen/                  # Eigen source
│   ├── json/                   # nlohmann/json source
│   └── libxml2/                # libxml2 source
├── prefix/
│   ├── include/                # Installed headers
│   │   ├── eigen3/             # Eigen headers
│   │   ├── assimp/             # Assimp headers
│   │   ├── nlohmann/           # JSON headers
│   │   └── libxml2/            # libxml2 headers
│   └── lib/                    # Installed libraries
│       ├── libassimp.so        # Assimp shared library
│       ├── libxml2.so          # libxml2 shared library
│       └── cmake/              # CMake find-package configs
└── CMakeLists.txt
```

## Usage

When building other biotools projects, point CMake to the local_deps prefix:

```bash
cmake -DCMAKE_PREFIX_PATH=~/Documents/biotools/local_deps/prefix ..
```

Or set individual dependency paths:

```bash
cmake -DEigen3_DIR=~/Documents/biotools/local_deps/prefix/lib/cmake/eigen3 \
      -Dassimp_DIR=~/Documents/biotools/local_deps/prefix/lib/cmake/assimp5 \
      ..
```

## Build

```bash
cd ~/Documents/biotools/local_deps
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This fetches, builds, and installs all dependencies into the `prefix/` directory.

## Connections to Other Biotools

| Dependency | Consumers |
|------------|-----------|
| **Eigen** | GAMer, Morpheus, CompuCell3D, Tissue Forge |
| **Assimp** | Tissue Forge (3D model import) |
| **nlohmann/json** | ReaDDy, Tissue Forge, various config parsers |
| **libxml2** | COPASI (SBML processing), Tissue Forge (libSBML) |
