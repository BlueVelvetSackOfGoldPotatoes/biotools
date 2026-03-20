# DeepXDE — Physics-Informed Neural Networks for Differential Equations

DeepXDE is a Python library for solving **differential equations** (PDEs, ODEs, integro-differential equations) using **physics-informed neural networks (PINNs)** and related deep learning methods. It provides a high-level API for defining problems, specifying boundary/initial conditions, and training neural network solutions.

## What It Does

DeepXDE bridges the gap between traditional numerical PDE solvers and modern deep learning by:

- **Solving forward problems**: Given a PDE and boundary/initial conditions, train a neural network to approximate the solution field
- **Solving inverse problems**: Given partial observations, jointly learn the PDE solution and unknown parameters (diffusion coefficients, reaction rates, etc.)
- **Data-driven discovery**: Learn governing equations from data
- **Operator learning**: Train networks (DeepONet, FNO) to learn solution operators mapping inputs to solutions
- **Multi-fidelity learning**: Combine low- and high-fidelity data sources

### Why PINNs for Biology?
PINNs are particularly valuable in computational biology because:
- They can handle irregular geometries without meshing
- They naturally incorporate physical constraints (conservation laws, diffusion equations)
- They can solve inverse problems (parameter estimation from experimental data)
- They can serve as fast surrogates for expensive numerical simulations

## Biological Scale

**Surrogate / ML** — PINNs can approximate physics at any scale, from molecular diffusion to tissue-level transport equations. In this biotools pipeline, DeepXDE learns to approximate PDE solutions computed by tools like GROMACS, COPASI, or PhysiCell.

## Key Features

### Problem Types
- **PDEs**: Elliptic, parabolic, hyperbolic equations in arbitrary dimensions
- **Time-dependent PDEs**: Diffusion, reaction-diffusion, wave equations
- **ODEs**: Systems of ordinary differential equations
- **Integro-differential equations**: Equations with integral terms
- **Fractional PDEs**: Equations with fractional derivatives
- **Inverse problems**: Unknown coefficients, source terms, or boundary conditions

### Neural Network Architectures
- **Feedforward** (MLP) with various activation functions
- **ResNet**: Residual connections for deeper networks
- **PFNN**: Parallel feedforward for multi-output problems
- **SIREN**: Sinusoidal representation networks
- **Modified MLP**: With Fourier feature embeddings
- **DeepONet**: Deep Operator Network for operator learning
- **FNO**: Fourier Neural Operator
- **PI-DeepONet**: Physics-informed DeepONet

### Boundary & Initial Conditions
- Dirichlet, Neumann, Robin, Periodic boundary conditions
- Initial conditions for time-dependent problems
- Point set constraints from data
- Operator boundary conditions
- Hard and soft constraint enforcement

### Training Infrastructure
- Multiple backend support: **TensorFlow**, **PyTorch**, **JAX**, **PaddlePaddle**
- Adam, L-BFGS optimizers
- Learning rate scheduling (step, exponential, cosine annealing, warmup)
- Adaptive residual resampling (RAR)
- Early stopping and model checkpointing
- Gradient-enhanced training

### Geometry
- Primitives: Interval, Rectangle, Disk, Triangle, Polygon, Cuboid, Sphere, Cylinder
- CSG operations: Union, Intersection, Difference
- Time domains and space-time geometries
- Point clouds for complex geometries

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/deepxde
pip install -e .
```

### Dependencies
- Python 3.x
- One of: TensorFlow 2.x, PyTorch, JAX, PaddlePaddle
- NumPy, SciPy, Matplotlib
- scikit-learn (optional)

## Usage

```python
import deepxde as dde

# Define geometry and time domain
geom = dde.geometry.Rectangle([0, 0], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define the PDE: diffusion equation du/dt = D * laplacian(u)
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=2)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    D = 0.01
    return du_t - D * (du_xx + du_yy)

# Boundary and initial conditions
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2]), lambda _, on_initial: on_initial)

# Create problem and train
data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=10000, num_boundary=500, num_initial=500)
net = dde.nn.FNN([3] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=20000)
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | Python code | PDE definition, geometry, boundary conditions |
| Input | NumPy arrays | Observation data for inverse problems |
| Output | Trained model | Neural network approximating the PDE solution |
| Output | NumPy arrays | Solution evaluated at arbitrary points |
| Output | Plots | Training loss curves, solution visualizations |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **GROMACS / OpenMM** | DeepXDE can learn surrogates for expensive MD force fields |
| **COPASI** | Learn approximate ODE solutions or inverse-estimate kinetic parameters |
| **PhysiCell** | Approximate diffusion field solutions used in multicellular simulations |
| **Modulus** | Complementary physics-ML framework (FNO, DeepONet at larger scale) |
| **PyG** | Graph neural networks for irregular domain discretizations |
| **cpp_ports/deepxde/** | C++17 header-only port (8 headers, ~4,400 lines) with tape-based AD |

## License

Apache License 2.0

## References

- Lu, L. et al. (2021). "DeepXDE: A Deep Learning Library for Solving Differential Equations." *SIAM Review*, 63(1):208-228.
- Raissi, M. et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378:686-707.
