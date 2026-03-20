# ESM — Evolutionary Scale Modeling

ESM is Meta AI's suite of **protein language models** that learn the patterns of protein sequences across evolution. Trained on millions of protein sequences, ESM models capture the fundamental principles of protein structure and function, enabling structure prediction, variant effect prediction, protein design, and inverse folding.

## What It Does

ESM provides several model families, each with distinct capabilities:

### ESM-2 (Protein Language Model)
- Learns contextualized representations of protein sequences via masked language modeling
- Extracts rich per-residue and per-sequence embeddings encoding structural and functional information
- Enables zero-shot variant effect prediction, contact map prediction, and downstream fine-tuning

### ESMFold (Structure Prediction)
- End-to-end single-sequence protein structure prediction — no MSA required
- Achieves near AlphaFold2-level accuracy at dramatically faster speed
- Uses ESM-2 embeddings as input to a structure module with Invariant Point Attention

### MSA Transformer
- Processes multiple sequence alignments as structured inputs
- Captures coevolutionary relationships between residue positions
- State-of-the-art unsupervised contact prediction

### ESM-IF1 (Inverse Folding)
- Given a protein backbone structure, predicts amino acid sequences that would fold into that structure
- Uses GVP (Geometric Vector Perceptrons) for structure encoding
- Enables fixed-backbone protein design

### Applications
- **Variant effect prediction**: Predict the functional impact of mutations without experimental data
- **Contact prediction**: Infer residue-residue contacts from sequence alone
- **Structure prediction**: Predict 3D structures from single sequences (ESMFold)
- **Protein design**: Generate sequences for desired structures (inverse folding)
- **Representation learning**: Extract embeddings for downstream ML tasks
- **Evolutionary analysis**: Study sequence-structure-function relationships

## Biological Scale

**Molecular** — protein language modeling and structure prediction at atomic resolution.

## Model Family

| Model | Parameters | Training Data | Key Capability |
|-------|-----------|---------------|----------------|
| ESM-2 (8M–15B) | 8M to 15B | UniRef50/90 | Sequence embeddings, variant prediction |
| ESM-1b | 650M | UniRef50 | Sequence embeddings (predecessor) |
| MSA Transformer | 100M | UniRef50 MSAs | Coevolution, contact prediction |
| ESMFold | ~700M | PDB + ESM-2 | Single-sequence structure prediction |
| ESM-IF1 | ~142M | PDB + CATH | Inverse folding / protein design |

## Installation

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/esm
pip install -e .
```

### Dependencies
- Python 3.x
- PyTorch
- fair-esm
- Biopython (optional, for PDB I/O)

## Usage

```python
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Prepare data
data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQ")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract per-residue representations
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
representations = results["representations"][33]
contact_map = results["contacts"]
```

### ESMFold Structure Prediction
```python
import esm
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQ"
with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("prediction.pdb", "w") as f:
    f.write(output)
```

## Input/Output

| Type | Format | Description |
|------|--------|-------------|
| Input | FASTA | Protein sequences |
| Input | A3M | Multiple sequence alignments (MSA Transformer) |
| Input | PDB | Backbone structures (inverse folding) |
| Output | Tensors | Per-residue and per-sequence embeddings |
| Output | Contact maps | Predicted residue-residue contacts |
| Output | PDB | Predicted 3D structures (ESMFold) |

## Connections to Other Biotools

| Tool | Integration |
|------|-------------|
| **Boltz** | Complementary structure prediction approach (diffusion-based vs. language model) |
| **GROMACS / OpenMM** | Refine ESMFold structures via molecular dynamics |
| **PyG** | Graph neural networks operating on protein structure graphs |
| **DeepXDE / Modulus** | Physics-informed surrogates for molecular properties |
| **cpp_ports/esm/** | C++17 header-only reimplementation (2 headers, ~3,800 lines) covering ESM-1/2, MSA Transformer, ESMFold, and GVP inverse folding |

## License

MIT License

## References

- Lin, Z. et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637):1123-1130.
- Rives, A. et al. (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences." *PNAS*, 118(15).
- Meier, J. et al. (2021). "Language models enable zero-shot prediction of the effects of mutations on protein function." *NeurIPS 2021*.
