# mcell_tests — MCell & CellBlender Test Suite

This repository contains the **comprehensive test suite** for MCell and CellBlender. It validates the correctness of MCell simulations, CellBlender model generation, and the integration between the two tools.

## What It Does

mcell_tests provides automated testing for the MCell ecosystem:

- **MCell3 MDL tests**: Validate MDL-based simulation correctness (diffusion, reactions, surface interactions, viz output)
- **MCell4 Python tests**: Test the MCell4 Python API
- **CellBlender integration tests**: Verify that CellBlender correctly generates MDL files and handles simulation workflows
- **Regression tests**: Ensure that changes to MCell or CellBlender don't break existing functionality
- **Validation tests**: Compare simulation results against analytical solutions or known benchmarks

## Directory Structure

```
mcell_tests/
├── tests/
│   ├── mdl/                    # MCell3 MDL-based tests
│   │   ├── 0001_*/             # Individual test cases (numbered)
│   │   ├── 0002_*/
│   │   └── ...
│   ├── pymcell4/               # MCell4 Python API tests
│   ├── cellblender/            # CellBlender integration tests
│   └── ...
├── scripts/                    # Test runner scripts
├── utils/                      # Test utility functions
└── ...
```

## Usage

Tests are typically run through the mcell_tools infrastructure:

```bash
# Run all tests
cd ~/Documents/biotools/mcell_tests
python run_tests.py

# Run specific test categories
python run_tests.py --test-suite mdl
python run_tests.py --test-suite pymcell4
```

## Dependencies

- **mcell**: The simulation engine being tested
- **mcell_tools**: Build and test orchestration scripts
- **CellBlender**: For integration tests
- Python 3.x

## Connections to Other Biotools

| Tool | Relationship |
|------|-------------|
| **MCell** | The simulation engine being tested |
| **CellBlender** | GUI tool tested for correct MDL generation |
| **mcell_tools** | Orchestrates test execution and reporting |
