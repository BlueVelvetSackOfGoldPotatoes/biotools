# Boltz — Protein Structure Prediction

Boltz is an open-source **protein structure prediction** system that predicts 3D atomic structures of proteins and biomolecular complexes from amino acid sequences. It uses a diffusion-based generative architecture achieving AlphaFold-class accuracy, with support for proteins, nucleic acids, small molecules, and covalent modifications.

## What It Does

Given a protein sequence (and optionally a multiple sequence alignment), Boltz predicts:

- **3D atomic coordinates** for all atoms in the structure
- **Confidence scores**: pLDDT (per-residue), pTM (predicted TM-score), iPTM (interface TM-score for complexes), PAE (predicted aligned error), PDE (predicted distance error)
- **Multi-chain complexes** with accurate interface prediction
- **B-factor estimates** for structural flexibility
- **Ligand binding poses** and covalent modification geometry

### Applications
- Predicting structures of proteins without experimental data
- Modeling protein-protein, protein-nucleic acid, and protein-ligand interactions
- Generating starting conformations for molecular dynamics simulations
- Drug target characterization and virtual screening

## Biological Scale

**Molecular** — predicts biomolecular structures at atomic resolution (Angstroms to nanometers).

## Architecture & Key Algorithms

### Input Processing
- **Data parsing**: FASTA sequences, A3M/MSA files, PDB/mmCIF templates
- **Tokenization & featurization**: Converts sequence and structural features into model-ready tensors
- **MSA processing**: Extracts evolutionary covariation signal from multiple sequence alignments

### Neural Network
- **MSA Module**: Outer Product Mean, Triangular Multiplicative Updates, Triangular Attention, Pair-Weighted Averaging — captures evolutionary and structural relationships
- **Pairformer**: Attention with pair bias for learning residue-residue spatial relationships
- **Atom Transformer**: Processes atomic-level detail
- **Structure Module**: Invariant Point Attention (IPA) with SE(3) rigid body backbone updates

### Diffusion-Based Generation
- **Noise schedules**: VP (Variance Preserving), VE (Variance Exploding), EDM
- **Samplers**: Heun (second-order) and Euler (first-order) integrators
- **Guidance potentials**: Van der Waals, connection constraints, stereo bond constraints, chirality, planarity

### Confidence Estimation
- pLDDT, pTM/iPTM, PAE, PDE confidence heads
- FAPE (Frame Aligned Point Error), smooth lDDT, distogram, and B-factor losses

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/boltz
pip install -e .
```

### Dependencies
- Python 3.x
- PyTorch (with CUDA for GPU inference)
- NumPy, SciPy
- Biopython

## Usage

```bash
# Command-line prediction
boltz predict input.fasta --output_dir results/

# Or via Python API
from boltz import BoltzPredictor
predictor = BoltzPredictor(weights_path="model_weights.pt")
result = predictor.predict("sequence.fasta")
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | `.fasta` | Protein/nucleic acid sequences |
| Input | `.a3m` | Multiple sequence alignments |
| Input | `.pdb` / `.cif` | Template structures |
| Output | `.pdb` / `.cif` | Predicted 3D structures with coordinates |
| Output | JSON | Confidence metrics (pLDDT, pTM, PAE) |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **ESM** | Protein language model embeddings complement Boltz predictions |
| **GROMACS / OpenMM** | Refine Boltz-predicted structures via molecular dynamics |
| **DeepXDE / Modulus** | Build surrogate models for rapid structure-property prediction |
| **cpp_ports/boltz/** | C++17 header-only reimplementation (5 headers, ~4,500 lines) |

## License

MIT License

## References

- Wohlwend, J. et al. (2024). "Boltz-1: Democratizing Biomolecular Interaction Modeling." *bioRxiv*.
