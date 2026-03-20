# NVIDIA PhysicsNeMo (Modulus) — Physics-Informed Machine Learning

NVIDIA PhysicsNeMo (formerly Modulus) is a **physics-informed machine learning framework** for building, training, and deploying neural network models that solve physics problems. It provides state-of-the-art architectures for learning PDE solutions, operator mappings, and physics surrogates.

## What It Does

PhysicsNeMo enables researchers to use deep learning for physics simulation by providing:

- **Neural network architectures** designed for physics: Fourier Neural Operators (FNO), DeepONet, MeshGraphNet, GraphCast, AFNO, and more
- **Physics-informed training**: Embed PDE residuals, boundary conditions, and conservation laws directly into the loss function
- **Operator learning**: Train networks that learn the mapping from input parameters to solutions (not just a single solution)
- **Large-scale training**: Distributed multi-GPU training for big physics problems
- **Weather & climate**: State-of-the-art architectures for weather forecasting (GraphCast, DLWP, FourCastNet)
- **CFD surrogates**: Fast approximations for fluid dynamics, heat transfer, and aerodynamics
- **Healthcare**: Blood flow modeling, brain anomaly detection

### Why for Biology?
In this biotools pipeline, PhysicsNeMo serves as the framework for building **neural surrogates** that approximate expensive biological simulations:
- Learn diffusion field solutions from COPASI or PhysiCell data
- Approximate molecular force fields from GROMACS trajectories
- Build fast tissue mechanics surrogates from Tissue Forge simulations
- Train graph neural networks on biological network data

## Biological Scale

**Surrogate / ML** — learns to approximate physics at any scale, from molecular dynamics to tissue-level simulations.

## Key Architectures

### Operator Learning
| Architecture | Description |
|-------------|-------------|
| **FNO** (1D/2D/3D) | Fourier Neural Operator — learns operators in spectral space |
| **AFNO** | Adaptive Fourier Neural Operator — attention-based spectral method |
| **DeepONet** | Deep Operator Network — learns nonlinear operators from data |
| **PI-DeepONet** | Physics-Informed DeepONet |

### Graph Neural Networks
| Architecture | Description |
|-------------|-------------|
| **MeshGraphNet** | GNN for mesh-based physics simulation |
| **GraphCastNet** | GNN for weather/climate prediction |

### Image-Based
| Architecture | Description |
|-------------|-------------|
| **SRResNet** | Super-resolution for physics fields |
| **Pix2Pix** | Image-to-image translation for physics |
| **UNet** | Encoder-decoder for dense prediction |

### Sequence/Temporal
| Architecture | Description |
|-------------|-------------|
| **One2ManyRNN** | Recurrent network for temporal physics |
| **SwinRNN** | Swin Transformer-based RNN |
| **DLWP** | Deep Learning Weather Prediction |

### Specialized
| Architecture | Description |
|-------------|-------------|
| **SIREN** | Sinusoidal representations for implicit neural representations |
| **FourCastNet** | AFNO-based global weather forecasting |

## Directory Structure

```
modulus/
├── physicsnemo/                # Core Python package
│   ├── models/                 # Neural network architectures
│   ├── datapipes/              # Data loading and processing
│   ├── loss/                   # Physics-informed loss functions
│   ├── mesh/                   # Mesh utilities and geometry
│   ├── distributed/            # Multi-GPU training support
│   └── ...
├── examples/
│   ├── cfd/                    # Computational fluid dynamics examples
│   ├── weather/                # Weather forecasting examples
│   ├── healthcare/             # Healthcare applications
│   ├── molecular_dynamics/     # Molecular dynamics surrogates
│   └── ...
├── benchmarks/                 # Performance benchmarks
└── ...
```

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/modulus
pip install -e .
```

### Dependencies
- Python 3.x
- PyTorch (with CUDA for GPU training)
- NumPy, SciPy
- Hydra (configuration management)
- wandb / TensorBoard (experiment tracking)

## Usage

```python
import physicsnemo
from physicsnemo.models.fno import FNO

# Create a Fourier Neural Operator
model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
)

# Train on simulation data
# ... standard PyTorch training loop with physics-informed loss
```

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **DeepXDE** | Complementary PINN framework (DeepXDE for small problems, PhysicsNeMo for large-scale) |
| **PyG** | Graph neural network foundations used by MeshGraphNet |
| **GROMACS / OpenMM** | Training data from MD simulations |
| **COPASI / PhysiCell** | Training data from biochemical/cellular simulations |
| **cpp_ports/modulus/** | C++17 header-only port (9 headers, ~6,600 lines) |

## License

Apache License 2.0

## References

- NVIDIA. "PhysicsNeMo: A Framework for Physics-ML."
- Li, Z. et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
- Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet." *Nature Machine Intelligence*, 3:218-229.
