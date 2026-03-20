# Cells Standalone OGRE Viewer

Optional sidecar project for visualizing the existing `models/cells/src/` simulator with OGRE.

## Scope
- Reuses the maintained `CellSim` implementation from the main repo.
- Builds outside the top-level `Makefile` path.
- Produces three CMake targets:
  - `cell_sim`
  - `cell_render_ogre`
  - `cell_app`

## Build

```bash
cd models/cells/standalone_ogre
cmake -S . -B build
cmake --build build -j
```

If you use vcpkg, point CMake at the toolchain in the usual way.

## Run

```bash
./build/cell_app
```

## Controls
- `Space`: pause/resume fixed-step simulation
- `D`: inject membrane/particle damage
- `R`: rebuild the simulation state
