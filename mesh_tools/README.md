# mesh_tools — Mesh Processing Toolkit

mesh_tools is a collection of **60+ command-line mesh processing utilities** for converting, analyzing, repairing, refining, and transforming triangulated surface meshes. These tools form the mesh processing backbone for the MCell/CellBlender workflow and are broadly useful for any biological mesh processing pipeline.

## What It Does

mesh_tools provides a comprehensive set of standalone executables, each performing a specific mesh operation. Together they enable:

- **Format conversion**: Convert between mesh formats (OBJ, STL, OFF, VTK, GTS, MCell MDL, DX, SMF, RIB, IRIT)
- **Mesh repair**: Heal defects (holes, non-manifold edges, degenerate triangles, orphan vertices)
- **Mesh refinement**: Subdivide meshes for higher resolution
- **Mesh simplification**: Reduce polygon count while preserving shape
- **Mesh analysis**: Compute statistics (surface area, volume, quality metrics)
- **Geometric operations**: Scale, translate, rotate, clip, offset, flip normals
- **Mesh manipulation**: Merge, fuse, separate, stitch, morph between meshes
- **Contour processing**: Tile between serial-section contours (from EM reconstructions)
- **Region tagging**: Mark mesh regions for MCell surface class assignments

## Biological Scale

**Mesh / Geometry** — operates at all scales, from molecular surface meshes to cellular reconstructions.

## Tools Reference

### Format Converters

| Tool | Description |
|------|-------------|
| `mesh2mcell` | Convert to MCell MDL format |
| `mesh2obj` | Convert to Wavefront OBJ |
| `mesh2stl` | Convert to STL (stereolithography) |
| `mesh2vtk` | Convert to VTK format |
| `mesh2off` | Convert to OFF (Object File Format) |
| `mesh2gts` | Convert to GNU Triangulated Surface |
| `obj2mesh` | Convert OBJ to internal mesh format |
| `dx2mesh` | Convert OpenDX to mesh |

### Mesh Processing

| Tool | Description |
|------|-------------|
| `meshrefine` | Refine mesh by subdivision |
| `meshsimplify` | Reduce polygon count (decimation) |
| `meshheal` | Repair mesh defects (holes, degeneracies) |
| `meshmorph` | Morph between two meshes |
| `meshfuse` | Fuse overlapping meshes |
| `meshmerge` | Merge separate meshes into one |
| `meshclip` | Clip mesh by a plane |
| `meshflip` | Flip face normals |
| `meshstitch` | Stitch mesh boundaries together |
| `meshorphan` | Remove orphan (unconnected) vertices |
| `meshfilter` / `filtermesh` | Filter mesh by criteria |
| `meshoffset` | Offset mesh surface by a distance |

### Geometric Transforms

| Tool | Description |
|------|-------------|
| `meshscale` | Scale mesh dimensions |
| `meshtranslate` | Translate mesh position |

### Analysis

| Tool | Description |
|------|-------------|
| `meshalyzer` | Compute mesh statistics (area, volume, quality metrics) |
| `mesh_separate` | Separate disconnected mesh components |
| `mesh_tag_region` | Tag mesh regions for MCell surface classes |

### Contour Processing

| Tool | Description |
|------|-------------|
| `contour_tiler` | Tile between serial-section contours |
| `contour_plotting` | Plot contour traces |
| `reconstruct_interpolate` | Interpolate between reconstruction contours |

## Directory Structure

```
mesh_tools/
├── mesh2mcell/                 # Each tool has its own subdirectory
├── mesh2obj/
├── mesh2stl/
├── mesh2vtk/
├── mesh2off/
├── meshrefine/
├── meshsimplify/
├── meshheal/
├── meshmorph/
├── meshalyzer/
├── contour_tiler/
├── filtermesh/
├── ...                         # 60+ tool subdirectories
└── README.md
```

Each subdirectory contains C source files and a Makefile or build script.

## Usage

```bash
# Convert OBJ to MCell MDL format
~/Documents/biotools/mesh_tools/mesh2mcell/mesh2mcell input.obj output.mdl

# Repair a mesh
~/Documents/biotools/mesh_tools/meshheal/meshheal input.obj output.obj

# Compute mesh statistics
~/Documents/biotools/mesh_tools/meshalyzer/meshalyzer input.obj

# Refine a mesh
~/Documents/biotools/mesh_tools/meshrefine/meshrefine input.obj output.obj

# Convert between formats
~/Documents/biotools/mesh_tools/mesh2stl/mesh2stl input.obj output.stl
```

## Input/Output

| Format | Extension | Read | Write |
|--------|-----------|------|-------|
| Internal mesh | `.mesh` | Yes | Yes |
| Wavefront OBJ | `.obj` | Yes | Yes |
| Object File Format | `.off` | Yes | Yes |
| STL | `.stl` | Yes | Yes |
| VTK | `.vtk` | Yes | Yes |
| MCell MDL | `.mdl` | No | Yes |
| GNU Triangulated Surface | `.gts` | Yes | Yes |
| OpenDX | `.dx` | Yes | No |
| SMF | `.smf` | Yes | Yes |
| IRIT | `.irit` | Yes | No |
| RenderMan RIB | `.rib` | Yes | No |

## Build

Most tools are built individually from their subdirectories:

```bash
cd ~/Documents/biotools/mesh_tools/mesh2mcell
make
```

Or build all tools:
```bash
cd ~/Documents/biotools/mesh_tools
for dir in */; do (cd "$dir" && make 2>/dev/null); done
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | mesh_tools prepares geometries for MCell simulations (mesh2mcell, mesh_tag_region) |
| **CellBlender** | Meshes are converted/repaired before import into CellBlender |
| **GAMer** | Complementary mesh generation and improvement tool |
| **neuropil_tools** | Processes EM reconstructions that mesh_tools then converts |
| **VTK** | mesh2vtk enables VTK-based visualization |
| **contour_tiler** | Creates meshes from serial-section EM contour traces |

## License

Various open-source licenses (check individual tool directories)
