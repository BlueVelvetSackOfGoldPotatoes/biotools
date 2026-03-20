# VTK — The Visualization Toolkit

VTK (Visualization Toolkit) is an open-source software system for **3D computer graphics, image processing, and scientific visualization**. In this biotools pipeline, VTK serves as a foundational dependency providing mesh I/O, rendering, and visualization capabilities used by several other tools.

## What It Does

VTK provides a comprehensive set of capabilities for scientific visualization:

- **3D rendering**: OpenGL-based rendering of surfaces, volumes, and scientific data
- **Mesh I/O**: Read and write numerous mesh and data formats (VTK, STL, OBJ, PLY, DICOM, and many more)
- **Data processing**: Filtering, transformation, and analysis of 3D datasets
- **Volume rendering**: Direct volume rendering of scalar fields
- **Streamlines and glyphs**: Flow visualization
- **Isosurfaces**: Marching cubes and contour generation
- **Image processing**: 2D and 3D image filters
- **Parallel rendering**: Distributed visualization for large datasets

### Role in Biotools
VTK is primarily a **build dependency** for other tools in the pipeline rather than a tool used directly. It provides:
- Mesh format support for mesh_tools, GAMer, and CellBlender
- VTK file format output for CompuCell3D and Morpheus
- Rendering capabilities for visualization tools
- Scientific data processing infrastructure

## Key Components

### I/O Modules
- VTK legacy and XML format readers/writers
- STL, OBJ, PLY mesh I/O
- DICOM medical imaging format
- CGNS computational fluid dynamics format
- HDF5-based formats
- Many more (100+ format readers)

### Filters
- Surface extraction (marching cubes, flying edges)
- Mesh smoothing, decimation, subdivision
- Clip, cut, and threshold operations
- Gradient, divergence, vorticity computation
- Delaunay triangulation
- Boolean operations on meshes

### Rendering
- OpenGL-based 3D rendering
- Ray casting volume rendering
- WebGPU support (experimental)
- VR/AR support (OpenVR, OpenXR)
- Web-based rendering (Emscripten/WebAssembly)

## Directory Structure

```
VTK/
├── Common/                     # Core data structures
├── Filters/                    # Data processing filters
├── IO/                         # File format readers/writers
├── Rendering/                  # Rendering backends
│   ├── Core/                   # Core rendering
│   ├── OpenGL2/                # OpenGL rendering
│   ├── WebGPU/                 # WebGPU rendering
│   └── VR/                     # Virtual reality
├── Interaction/                # User interaction
├── Imaging/                    # Image processing
├── ThirdParty/                 # Bundled dependencies
│   ├── eigen/                  # Linear algebra
│   ├── hdf5/                   # HDF5 library
│   ├── freetype/               # Font rendering
│   ├── libxml2/                # XML parsing
│   └── ...
├── Examples/                   # Example programs
├── Documentation/              # Documentation
└── ...
```

## Build

VTK is a large project. A minimal build for biotools:

```bash
cd ~/Documents/biotools/VTK
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DVTK_BUILD_TESTING=OFF \
    -DVTK_GROUP_ENABLE_Rendering=WANT \
    -DVTK_GROUP_ENABLE_StandAlone=WANT
make -j$(nproc)
```

## Connections to Other Biotools

| Tool | Relationship |
|------|-------------|
| **mesh_tools** | Uses VTK for mesh2vtk format conversion |
| **GAMer** | VTK mesh I/O support |
| **CompuCell3D** | Outputs VTK lattice snapshots |
| **Morpheus** | Outputs VTK field and cell data |
| **CellBlender** | VTK-based visualization support |
| **Tissue Forge** | Rendering infrastructure |

## License

BSD 3-Clause License

## References

- Schroeder, W., Martin, K. & Lorensen, B. (2006). "The Visualization Toolkit: An Object-Oriented Approach to 3-D Graphics." 4th Edition, Kitware.
