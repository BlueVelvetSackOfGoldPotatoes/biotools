# PyTorch Geometric (PyG) — Graph Neural Network Library

PyTorch Geometric (PyG) is a comprehensive **graph neural network (GNN) library** built on PyTorch. It provides a wide range of graph convolution operators, pooling methods, data loaders, and model architectures for learning on graph-structured data.

## What It Does

PyG enables deep learning on graph-structured data, which is ubiquitous in biology:

- **Molecular graphs**: Atoms as nodes, bonds as edges — for property prediction, drug discovery, molecular generation
- **Protein structure graphs**: Residues as nodes, spatial contacts as edges — for function prediction, binding site detection
- **Cell interaction networks**: Cells as nodes, physical/signaling contacts as edges — for tissue modeling
- **Gene regulatory networks**: Genes as nodes, regulatory relationships as edges
- **Metabolic networks**: Metabolites/reactions as graph structures
- **Mesh graphs**: Vertices as nodes, mesh edges as edges — for physics simulation on unstructured grids

### Key Capabilities
- **Node classification**: Predict properties of individual nodes (e.g., protein residue function)
- **Graph classification**: Predict properties of entire graphs (e.g., molecular activity)
- **Link prediction**: Predict missing edges (e.g., protein-protein interactions)
- **Graph generation**: Generate new graphs (e.g., novel molecular structures)
- **Graph regression**: Predict continuous values for graphs (e.g., binding affinity)

## Biological Scale

**Surrogate / ML** — operates on graph-structured biological data at any scale, from molecular graphs to tissue-level cell interaction networks.

## Key Features

### Convolution Operators (30+)
| Layer | Description |
|-------|-------------|
| `GCNConv` | Graph Convolutional Network |
| `GATConv` / `GATv2Conv` | Graph Attention Network |
| `SAGEConv` | GraphSAGE (inductive learning) |
| `GINConv` / `GINEConv` | Graph Isomorphism Network |
| `TransformerConv` | Graph Transformer |
| `ChebConv` | Chebyshev spectral convolution |
| `SchNet` | Continuous-filter convolution for molecules |
| `DimeNet` | Directional message passing for molecules |
| `PNAConv` | Principal Neighbourhood Aggregation |
| `RGCNConv` | Relational GCN for heterogeneous graphs |
| And many more... | |

### Pooling Operators
- Global: mean, max, add pooling
- Hierarchical: SAGPooling, TopKPooling, ASAPooling, EdgePooling
- Dense: DiffPool, MinCutPool

### Data Handling
- `Data` / `HeteroData` graph data structures
- `Batch` for mini-batch processing
- `NeighborLoader`, `ClusterLoader` for large-scale graphs
- `RandomNodeSplit`, `RandomLinkSplit` for train/val/test splitting

### Model Architectures
- GAE / VGAE (Graph Autoencoders)
- Node2Vec (graph embeddings)
- GraphUNet
- SchNet, DimeNet (molecular property prediction)
- JumpingKnowledge (multi-scale aggregation)
- GNNExplainer (interpretability)

### Transforms
- Feature normalization, self-loop addition
- Positional encodings (random walk, Laplacian)
- Graph diffusion convolution (GDC)
- Virtual node augmentation

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/pytorch_geometric
pip install -e .
```

### Dependencies
- Python 3.x
- PyTorch
- torch-scatter, torch-sparse, torch-cluster, torch-spline-conv (optional C++ extensions)
- NumPy, SciPy

## Usage

```python
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes, 16 features each
data = Data(x=x, edge_index=edge_index)

# Define a GNN
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN()
out = model(data)  # Node embeddings
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **Modulus** | MeshGraphNet and GraphCastNet build on GNN foundations |
| **Boltz / ESM** | Protein structure graphs for property prediction |
| **GROMACS / OpenMM** | Molecular graph representations from simulation data |
| **PhysiCell / CompuCell3D** | Cell interaction graphs from multicellular simulations |
| **DeepXDE** | Complementary approach (PINNs vs. GNNs for physics) |
| **cpp_ports/pyg/** | C++17 header-only port (8 headers, ~5,100 lines) with 30+ convolution layers |

## License

MIT License

## References

- Fey, M. & Lenssen, J.E. (2019). "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
