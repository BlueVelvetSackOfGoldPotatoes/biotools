# GAMer — Geometry-preserving Adaptive MeshER

GAMer is a mesh generation and processing tool that creates **high-quality triangulated surface and tetrahedral volume meshes** from biological structures. It is specifically designed for generating finite-element meshes from geometries reconstructed from electron microscopy, molecular surfaces, and other biological imaging modalities.

## What It Does

GAMer addresses a critical need in computational biology: converting raw biological geometry data (from EM reconstructions, PDB structures, or imaging) into high-quality meshes suitable for numerical simulation. It provides:

- **Surface mesh generation** from molecular structures (PDB/PQR coordinates)
- **Surface mesh improvement**: Smoothing, refinement, coarsening, and quality optimization while preserving geometric features
- **Tetrahedral volume mesh generation** for finite-element analysis
- **Geometry-preserving operations** that maintain biologically important features (curvature, topology) during mesh processing
- **Python bindings** (PyGAMer) for scripting and integration with other tools

### Applications
- Generating meshes for MCell reaction-diffusion simulations
- Creating finite-element meshes for electrostatics calculations
- Processing electron microscopy reconstruction surfaces
- Molecular surface mesh generation for visualization and analysis
- Mesh quality improvement for numerical stability

## Biological Scale

**Mesh / Geometry** — operates at all scales, from molecular surfaces (Angstroms) to cellular organelle reconstructions (micrometers).

## Key Algorithms

### Surface Mesh Operations
- **Angle-weighted Laplacian smoothing**: Smooth surface meshes while preserving shape
- **Surface refinement**: Subdivide triangles to increase mesh resolution
- **Surface coarsening**: Reduce triangle count while maintaining shape fidelity
- **Normal smoothing**: Smooth vertex normals for improved surface representation
- **Feature preservation**: Detect and protect sharp features and ridges during processing

### Volume Mesh Generation
- **Delaunay-based tetrahedral meshing**: Generate quality tetrahedral meshes from surface meshes
- **Constrained Delaunay triangulation**: Maintain surface mesh faces in the volumetric mesh
- **Quality metrics**: Radius-edge ratio, dihedral angles, volume

### Molecular Surface Meshing
- Generate solvent-excluded surfaces from PDB/PQR atomic coordinates
- Gaussian surface representation
- Configurable probe radius and mesh density

## Directory Structure

```
gamer/
├── src/                        # Core C++ source
│   ├── gamer.h                 # Main header
│   ├── SurfaceMesh.cpp         # Surface mesh operations
│   ├── TetMesh.cpp             # Tetrahedral mesh operations
│   └── ...
├── pygamer/                    # Python bindings (PyGAMer)
├── libraries/
│   └── tetgen/                 # TetGen tetrahedral mesher (dependency)
├── tools/                      # Utility scripts
├── tests/                      # Test suite
├── build/
│   └── lib/
│       ├── libgamer.so         # Shared library
│       └── libgamer.a          # Static library
└── ...
```

## Usage

### Python (PyGAMer)
```python
import pygamer

# Read a surface mesh
mesh = pygamer.readOFF("surface.off")

# Smooth the surface
pygamer.smoothMesh(mesh, max_iter=10, preserve_ridges=True)

# Refine the mesh
pygamer.refineMesh(mesh, max_area=0.1)

# Generate tetrahedral mesh
tetmesh = pygamer.makeTetMesh(mesh, quality=1.4)

# Write output
pygamer.writeOFF("smoothed.off", mesh)
```

### C++ Library
```cpp
#include <gamer/gamer.h>

auto mesh = gamer::readOFF("surface.off");
gamer::smoothMesh(mesh, 10, true);
gamer::writeOFF("output.off", mesh);
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | PDB/PQR | Molecular structures (atomic coordinates) |
| Input | OFF | Object File Format surface meshes |
| Input | OBJ | Wavefront OBJ meshes |
| Input | STL | STL surface meshes |
| Output | OFF, OBJ, STL | Processed surface meshes |
| Output | Tetrahedral | Volume meshes for FEM solvers |

## Build

```bash
cd ~/Documents/biotools/gamer
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Dependencies
- CMake 3.10+
- Eigen (fetched automatically)
- CASC (fetched automatically)
- TetGen (included in `libraries/`)
- Python 3.x + pybind11 (for PyGAMer)

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | GAMer-generated meshes serve as simulation geometries |
| **CellBlender** | Meshes imported into CellBlender for MCell setup |
| **mesh_tools** | Complementary mesh processing utilities |
| **COPASI** | Meshes for spatial solver domains |
| **Tissue Forge** | Surface meshes for particle-based tissue simulations |
| **VTK** | Visualization of generated meshes |

## License

LGPL v2.1

## References

- Lee, C.T. et al. (2020). "An Open-Source Mesh Generation Platform for Biophysical Modeling Using Realistic Cellular Geometries." *Biophysical Journal*, 118(5):1003-1008.
- Yu, Z. et al. (2008). "Feature-preserving adaptive mesh generation for molecular shape modeling and simulation." *Journal of Molecular Graphics and Modelling*, 26(8):1370-1380.
