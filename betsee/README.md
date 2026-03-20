# BETSEE — BioElectric Tissue Simulation Engine Environment

BETSEE is the **graphical user interface (GUI)** for the BETSE bioelectric tissue simulator. It provides a visual environment for configuring, running, monitoring, and analyzing BETSE simulations without needing to edit YAML configuration files by hand.

## What It Does

BETSEE wraps the full BETSE simulation engine in an interactive desktop application. Users can:

- **Configure** all simulation parameters through visual editors (ion channels, tissue profiles, GRN components, physics settings, export options)
- **Run** simulations with real-time progress monitoring and log output
- **Visualize** results including membrane voltage maps, ion concentration plots, current flow fields, and time series data
- **Manage** simulation projects with save/load functionality

## Biological Scale

GUI / Configuration — provides the interface for BETSE's cellular/tissue-scale (1 μm – 10 mm) bioelectric simulations.

## Architecture

BETSEE is built on **PyQt5** (Qt5 Python bindings) and follows a model-view-controller pattern:

- **Simulation tree**: Hierarchical view of all simulation components (cells, ions, channels, tissues, GRN, physics, exports)
- **Parameter editors**: Type-aware editors for every configurable parameter in BETSE
- **Process management**: Spawns BETSE as a subprocess, captures stdout/stderr in real-time
- **Results display**: Embedded matplotlib viewers for plots and animations
- **Configuration serialization**: Reads/writes BETSE YAML configuration files

## Installation

Within this biotools collection:

```bash
source ~/Documents/biotools/.venv/bin/activate
cd ~/Documents/biotools/betsee
pip install -e .
```

### Dependencies
- Python 3.x
- BETSE (must be installed first)
- PyQt5 or PySide2
- Matplotlib (for embedded plotting)

## Usage

```bash
# Launch the GUI
betsee

# Or launch with a specific configuration
betsee my_simulation.yaml
```

## Connections to Other Biotools

- **BETSE** is the simulation engine that BETSEE configures and drives
- **cpp_ports/betsee/** contains a C++17 header-only port of the GUI backend plus a modern React web frontend alternative
- The React frontend (in `cpp_ports/betsee/react-app/`) provides a web-based alternative interface with components for general settings, ion configuration, channel selection, tissue profiles, GRN editing, simulation control, and results viewing

## License

BSD 2-Clause License

## References

- Pietak, A. & Levin, M. (2016). "Exploring Instructive Physiological Signaling with the Bioelectric Tissue Simulation Engine." *Frontiers in Bioengineering and Biotechnology*, 4:55.
