# mcell_tools — MCell Build & Utility Scripts

mcell_tools is a collection of **build scripts, utility tools, and automation** for the MCell simulation ecosystem. It provides a unified way to clone, build, and test all MCell-related repositories.

## What It Does

mcell_tools streamlines the development workflow for the MCell ecosystem:

- **Repository management**: Scripts to clone all required MCell repositories (mcell, cellblender, mesh_tools, bionetgen, nfsim, nfsimCInterface, libbng, gamer, mcell_tests)
- **Build automation**: Coordinated build of MCell and all its dependencies in the correct order
- **Test orchestration**: Run the mcell_tests suite against a built MCell
- **Packaging**: Create distributable MCell packages
- **Utility scripts**: Various helper tools for development and debugging

## Directory Structure

```
mcell_tools/
├── scripts/
│   ├── clone_all.py            # Clone all required repositories
│   ├── build_all.py            # Build MCell and dependencies
│   ├── run_tests.py            # Execute test suite
│   └── ...
├── work/
│   └── build_mcell/            # Build workspace
│       └── bng2/               # BioNetGen build artifacts
└── ...
```

## Usage

```bash
cd ~/Documents/biotools/mcell_tools

# Clone all MCell ecosystem repositories
python scripts/clone_all.py

# Build everything
python scripts/build_all.py

# Run tests
python scripts/run_tests.py
```

### Required System Packages (Debian/Ubuntu)

```bash
sudo apt-get install cmake g++ python3 python3-dev git \
    libgl1-mesa-dev libglu1-mesa-dev
```

## Connections to Other Biotools

| Tool | Relationship |
|------|-------------|
| **MCell** | Primary build target |
| **CellBlender** | Built and packaged as part of MCell ecosystem |
| **mesh_tools** | Built as a dependency |
| **BioNetGen** | Cloned and built for rule-based reaction support |
| **NFSim** | Built as a dependency for network-free simulation |
| **nfsimCInterface** | Built for MCell-NFSim integration |
| **libbng** | Built for lightweight BioNetGen integration |
| **GAMer** | Built for mesh generation support |
| **mcell_tests** | Test suite executed by mcell_tools |

## License

MIT License
