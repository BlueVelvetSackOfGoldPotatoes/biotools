# neuropil_tools — Neuropil Ultrastructure Analysis Toolkit

neuropil_tools is a Python package for **analyzing neuropil ultrastructure** from electron microscopy (EM) reconstructions. It provides mesh processing, region analysis, contact pattern detection, and spine/PSD (postsynaptic density) measurement tools for MCell, Reconstruct, and Blender workflows.

## What It Does

neuropil_tools processes 3D neuronal reconstructions from serial-section or volume EM to:

- **Parse reconstruction data**: Read Reconstruct series files and extract contour traces, object definitions, and section metadata
- **Mesh processing**: Load, process, and export meshes in multiple formats (OBJ, OFF, STL, MDL)
- **Region analysis**: Extract and analyze named regions on mesh surfaces (e.g., PSD, active zone, spine head, spine neck)
- **Contact pattern detection**: Identify and characterize contact surfaces between neuronal structures (synapses, glial contacts)
- **Spine morphometry**: Measure dendritic spine geometry (head volume, neck length, PSD area)
- **Connectivity analysis**: Map synaptic connectivity patterns between reconstructed neurons
- **Export for simulation**: Prepare meshes and region annotations for MCell simulations via CellBlender

### Applications
- Quantitative analysis of synaptic ultrastructure
- Spine morphology classification and measurement
- Synaptic connectivity mapping
- Preparing EM reconstruction data for MCell reaction-diffusion simulation
- Neuropil volume fraction analysis

## Biological Scale

**Mesh / Geometry** — nanometer-scale neuronal ultrastructure (synaptic clefts: ~20 nm; dendritic spines: ~0.1–2 μm; neuropil volumes: ~1–100 μm^3).

## Key Features

### Mesh I/O
- Read/write OBJ, OFF, STL, MDL mesh formats
- Surface area and volume computation
- Mesh smoothing and cleanup
- Region extraction from tagged meshes

### Reconstruction Processing
- Parse Reconstruct series (`.ser`) files
- Extract contour traces and object metadata
- Convert between reconstruction and mesh representations

### Analysis
- Contact surface area measurement between adjacent structures
- Spine morphometry: head volume, neck length/diameter, PSD area
- Connectivity records mapping pre/post-synaptic partners
- Surface region statistics

### Blender Integration
- Export meshes in formats compatible with Blender/CellBlender
- Transfer region annotations for MCell surface class assignment

## Directory Structure

```
neuropil_tools/
├── neuropil_tools/
│   ├── __init__.py             # Package initialization
│   ├── mesh.py                 # Mesh processing
│   ├── regions.py              # Region analysis
│   ├── contacts.py             # Contact pattern detection
│   ├── spines.py               # Spine morphometry
│   ├── reconstruct.py          # Reconstruct series parsing
│   └── ...
└── ...
```

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/neuropil_tools
pip install -e .
```

## Usage

```python
from neuropil_tools import Mesh, NeuropilObject

# Load a mesh from EM reconstruction
mesh = Mesh.from_obj("dendrite_segment.obj")

# Compute surface area and volume
print(f"Surface area: {mesh.surface_area()} um^2")
print(f"Volume: {mesh.volume()} um^3")

# Extract a named region (e.g., PSD)
psd_region = mesh.extract_region("PSD_01")
print(f"PSD area: {psd_region.surface_area()} um^2")

# Analyze contacts between structures
contacts = analyze_contacts(dendrite_mesh, axon_mesh, threshold=0.02)
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **MCell** | Prepares EM reconstructions for MCell reaction-diffusion simulations |
| **CellBlender** | Meshes with region annotations imported into CellBlender |
| **mesh_tools** | Complementary mesh conversion and repair utilities |
| **GAMer** | Mesh quality improvement for simulation-ready geometries |
| **cpp_ports/neuropil_tools/** | C++17 header-only port (1 header, ~1,400 lines) |

## License

MIT License
