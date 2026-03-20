# Cell-First AI: A Complete Guide to Building a Real-Time 3D Cell Simulator

## From Falsifying AI's Deepest Assumption to a Working C++ + OGRE Codebase

---

## Part I — The Narrative: Why We Are Doing This

### 1.1 The Core Insight: Science Advances by Falsification

Throughout the history of science, the biggest breakthroughs have come not from incremental improvement, but from *negating* something the entire field assumed was true. Relativity negated absolute space. Plate tectonics negated fixed continents. Germ theory negated spontaneous generation.

In every case, the pattern is the same: there is a long-held assumption that everyone treats as obviously correct. Someone identifies it, states its negation as a testable hypothesis, and builds an experiment to check. If the negation holds, the field reorganizes around a new foundation.

This project applies that same logic to artificial intelligence.

### 1.2 The Assumption We Are Attacking

Modern AI — specifically deep learning — rests on an implicit assumption that almost nobody states explicitly:

> **The point-neuron abstraction is sufficient for sample-efficient, robust, continual, and adaptive intelligence.**

What does this mean in practice? It means that when you build a neural network, each "neuron" is modeled as a single number (an activation) that takes a weighted sum of its inputs, applies a nonlinearity, and passes the result forward. Everything about the biological neuron — its internal biochemistry, its membrane dynamics, its energy metabolism, its shape, its local signaling — is thrown away. The assumption is that none of that matters for intelligence.

This assumption has been *spectacularly successful* for a certain class of problems: image classification, language modeling, game playing on fixed benchmarks. Deep learning scaled because the point-neuron abstraction is cheap to compute, easy to differentiate, and amenable to massive parallelism on GPUs.

But there are regimes where deep learning consistently struggles:

- **Continual learning**: networks catastrophically forget old tasks when trained on new ones.
- **Robustness under distribution shift**: performance degrades sharply when the test environment differs from training.
- **Recovery after damage**: remove part of a trained network and it breaks; there is no self-repair.
- **Sample efficiency in nonstationary environments**: adapting to a changing world requires enormous retraining.
- **Long-horizon stability**: maintaining coherent behavior over extended time without drift.
- **Energy efficiency**: biological brains use roughly 20 watts; comparable AI systems use kilowatts to megawatts.

These are not minor gaps. These are the capabilities that define robust, adaptive, embodied intelligence — the kind that every animal on Earth exhibits.

### 1.3 The Deeper Inversion: Not Neurons, but Cells

The initial instinct might be: "make the neuron model more realistic." But that stays inside the same paradigm. You still have a network of neuron-like units optimized by a task loss. You've changed a detail, not the foundation.

The stronger inversion goes further:

> **Intelligence-relevant computation does not require neuron-like architectures at all.**

Instead, the fundamental computational unit may be the **cell** — not the neuron specifically, but the general biological cell, of which the neuron is just one specialized type.

Why is this a bigger idea? Because cells — all cells, not just neurons — are computationally active systems. Every cell in your body:

- Maintains **multi-timescale internal state** (fast electrical signals, medium biochemical cascades, slow gene expression changes, very slow structural remodeling).
- Performs **local decision-making** based on signals from its immediate neighborhood.
- Communicates through **multiple channels** simultaneously (electrical, chemical, mechanical).
- Stores information in **state, morphology, and topology** — not just in "weights."
- Operates under **homeostatic constraints**: it must keep itself alive, maintain its internal environment, and repair damage, *before* it can do anything else.
- Participates in **collective behavior**: tissues, organs, and organisms emerge from cells coordinating without any central controller.

Neurons are one cell type that happens to be optimized for fast electrical signaling over long distances. But the underlying computational machinery — state maintenance, local signaling, self-repair, collective coordination — exists in *every* cell.

This suggests a radical reframing: what if intelligence is not a property of neural networks, but a general property of cellular collectives coordinating across scales? What if brains are one instance of a much broader phenomenon?

### 1.4 Why Prior "Bio-Inspired AI" Has Failed

This idea is not entirely new. Many researchers have explored biologically-inspired AI. Most of those efforts have failed to displace deep learning. Understanding *why* they failed is critical to not repeating the same mistakes.

**Spiking neural networks** attacked the assumption that rate-coded neurons are enough. They achieved interesting energy arguments and event-driven computation, but training remained harder than backpropagation, and benchmark gains were inconsistent. The lesson: changing the neuron's signaling model alone is not enough.

**Dendritic / compartmental neuron models** showed that dendrites perform local nonlinear computation and gating, offering richer credit assignment and context-sensitive processing. But implementations at scale tended to become "deep learning with biological decoration" — the training framework remained unchanged.

**Neural Cellular Automata (NCA)** are the closest prior work to this project's direction. They demonstrated that local cells with internal state and neighbor communication can grow, maintain, and regenerate patterns. But they mostly stopped at pattern formation. Controllability was hard, stability was fragile, and there was no clear path from "pretty self-organization" to general cognition.

**Morphogenetic / collective intelligence work** (particularly in the tradition of Levin and collaborators) provided a strong conceptual foundation: intelligence-like behavior exists across biological scales, not just in brains. But this remained primarily a theoretical framework, not a standardized model family with benchmarks beating deep learning baselines.

**Connectomics** showed that wiring matters, motifs matter, and cell types matter. But connectivity alone is not function — you also need dynamics, neuromodulation, development, and metabolic constraints.

**The common pattern across all these "failures"**: they changed one biological detail while keeping the rest of the deep learning frame intact. They kept fixed topology, task-loss-first optimization, centralized training, learning mostly in weights, and no intrinsic self-maintenance objective. In other words, they imported biological details without importing the *computational reasons* those details exist.

### 1.5 What Would Actually Count as a Revolution

A few-percent accuracy improvement on ImageNet will not disrupt deep learning. What would disrupt it is evidence of a **regime change** — qualitatively different capabilities on problems that matter:

- Same task performance at **10–100× less data**.
- Same performance at **10× less energy**.
- **Continual learning without replay** or catastrophic forgetting.
- **Fast adaptation from a few examples** in nonstationary environments.
- **Robust performance under distribution shift** without retraining.
- **Self-repair after damage** (remove components and the system recovers).

If cell-like computational primitives consistently enable these capabilities where parameter-matched neural baselines fail, that would not be an incremental gain. It would change what the field considers the basic unit of intelligence.

### 1.6 The Role of This Simulator

To make any of this scientific (rather than philosophical), you need a **controlled experimental substrate** where "cell-like" properties are explicit, measurable, and manipulable.

That is what this simulator provides.

It is a real-time 3D simulation of a single biological cell — with a deformable membrane, internal viscous fluid, and physically grounded constraints — written in C++ and visualized with the OGRE rendering engine. It is designed from the ground up to be:

- **Explicit**: every physical property is represented as a named function with a clear signature.
- **Deterministic**: the same initial state and parameters always produce the same trajectory.
- **Measurable**: constraint residuals, volume drift, energy, and other diagnostics are logged every frame.
- **Optimizable**: all parameters live in a single struct, and the step function can be wrapped in a loss and optimized.

The simulator is not the AI system itself. It is the *substrate* on which cell-first AI will be built and tested. Once the physical cell is stable, you introduce active policies (tension fields, protrusions, adhesion rules) as explicit functions, define tasks that require repair and adaptation, and compare cell-based agents against neural baselines.

### 1.7 The Concrete Falsifiable Question

To keep this scientific, state the experimental question sharply:

> **Hypothesis**: A cell-like system built from **spatially migrating, self-wiring cells** (multi-timescale internal state, local signaling, homeostatic constraints, and structural plasticity through contact) will outperform a parameter-matched point-neuron baseline on tasks that require persistent self-maintenance, damage recovery, and continual adaptation in nonstationary environments.

**What you measure**: (1) Time-to-recovery after membrane or fluid damage (removal of vertices/particles). (2) Cumulative reward on a locomotion/chemotaxis task where the environment distribution shifts every N steps. (3) Volume/area stability over 100k steps with intermittent perturbation.

**What the baseline is**: A standard recurrent neural network (GRU/LSTM) or small transformer controlling the same physical body, with the same total parameter count as the cell agent's policy + internal state variables.

**What constitutes a "win"**: The cell agent recovers from damage in ≤ 50% of the steps the baseline requires, *or* achieves ≥ 2× cumulative reward under distribution shift, *or* maintains volume within 2% of target while the baseline drifts beyond 5%. Any one of these, consistently across seeds, is evidence against the sufficiency of point-neuron abstractions.

**What constitutes a "loss"**: If the baseline matches or exceeds the cell agent on all three metrics, the cell-like structure adds complexity without benefit, and the hypothesis is falsified in this regime.

### 1.8 Where the Simulator Ends and the AI Layer Begins

The boundary is concrete:

- **Simulator scope**: Everything in `sim/` — the Step function, constraint projections, SPH fluid, coupling, and diagnostics. The simulator is a deterministic physics engine. It takes a state S, parameters Θ, and optional external inputs U, and produces the next state. It has no learning, no policy, no objective function.

- **AI layer scope** (not built in this guide): Policy functions that map local observations to active control signals (tension, protrusion, adhesion). An optimization loop that calls Step repeatedly, accumulates a loss, and updates policy parameters. Baseline agent implementations for comparison. Benchmark task definitions (damage schedules, environment shift protocols, reward functions).

The simulator exposes the AI layer's interface through two mechanisms: (1) the `SimView` struct, which provides observations, and (2) the external input `U_t` in the Step function, which accepts control signals.

In the **spatial migration** setting, the AI layer is not only “a policy that outputs forces.” It is a **development + learning loop**: cells move, form/prune synapses based on contact and compatibility, and self-organize into functional areas. Top-down teaching enters as a localized clamp/gating field, not as an output error broadcast. The AI layer never reaches into SimState directly — it reads SimView and writes U_t (including fields and teaching clamps).

---

## Part II — The Complete Technical Stack

### 2.1 What We Are Building

A single-cell simulator with four major components:

1. **Membrane**: A triangulated surface mesh (deformable shell) that represents the cell's outer boundary. It is a closed surface made of vertices connected by edges and organized into triangles. It can stretch, bend, and deform, but it resists changes to its total volume and surface area — just like a real cell membrane backed by a cortical cytoskeleton.

2. **Cytoplasm**: The internal fluid of the cell, represented as a collection of particles using Smoothed Particle Hydrodynamics (SPH). These particles have mass, velocity, density, and pressure. They interact with each other through kernel functions that approximate fluid behavior: pressure pushes particles apart when they're too close, viscosity makes them drag on each other, and density computation tells each particle how crowded its neighborhood is.

3. **Coupling**: The membrane and fluid must interact. The fluid exerts pressure on the membrane from inside (supporting its shape), and the membrane prevents fluid from leaking out. This is implemented as a boundary constraint: particles that would cross the membrane surface are pushed back inside, and the reaction forces are applied to the nearest membrane vertices.

4. **Visualization**: An OGRE-based renderer that displays the membrane as a lit, shaded surface and the fluid particles as small billboards or point sprites. The renderer accepts only plain arrays of positions, normals, and indices — it never touches physics data or implements physics rules.

### 2.2 The Simulation–Rendering Separation (Non-Negotiable)

The single most important architectural decision in this project is the **strict separation between simulation and rendering**.

The simulation owns the physics. It maintains positions, velocities, forces, constraints, and parameters. It advances the state forward in time. It knows nothing about OGRE, OpenGL, GPUs, or windows.

The renderer owns the visualization. It takes arrays of numbers (positions, normals, indices) and turns them into pixels on screen. It knows nothing about SPH kernels, constraint projections, or fluid dynamics.

The interface between them is a **view object** — a lightweight struct that exposes read-only spans (array references) into the simulation's current state:

```
SimView {
    membrane_positions: span of Vec3
    membrane_normals:   span of Vec3
    membrane_indices:   span of uint32
    particle_positions: span of Vec3
}
```

This separation exists for three reasons:

1. **Testability**: You can run the simulation headless (no window, no GPU) for automated testing, parameter sweeps, and optimization.
2. **Swappability**: You can replace OGRE with raw OpenGL, Vulkan, or a terminal-based visualizer without touching any simulation code.
3. **Optimizability**: The simulation step function remains a clean, deterministic function that can be called inside an optimization loop without rendering overhead.

### 2.3 The Physics Model in Detail

#### 2.3.1 The Membrane

The membrane is represented as a **triangulated surface mesh**. Think of it as a net made of triangles draped over a balloon.

**Data:**
- A list of **vertices**, each with a 3D position and a 3D velocity.
- A list of **triangles**, each defined by three vertex indices. These define the surface for rendering and for geometric computations (area, volume, normals).
- A list of **edges**, each defined by two vertex indices and a **rest length** (the natural, unstretched distance between those two vertices). These define the elastic structure.
- **Adjacency information**: for each edge, which two triangles share it (needed for bending constraints).

**How to generate it:**
The initial mesh is a procedural **icosphere** — a sphere built by starting with an icosahedron (20 triangles) and recursively subdividing each triangle into four smaller triangles, then projecting the new vertices onto the sphere surface. This gives a nearly-uniform triangulation with controllable resolution. Two to four levels of subdivision (producing roughly 320 to 5,120 triangles) is a good starting range.

**Constraints on the membrane:**

Each constraint is a function that takes the current (predicted) vertex positions, computes how far they are from satisfying a desired property, and nudges the positions to reduce the violation. This is the XPBD (Extended Position-Based Dynamics) approach: constraints operate on positions, not forces, which makes the system much more stable at large timesteps.

- **Stretch constraint**: For each edge, the distance between its two vertices should stay close to the rest length. If the edge is too long, pull the vertices together; if too short, push them apart. The compliance parameter controls how stiff this is. This prevents the membrane from stretching unrealistically.

- **Volume constraint**: The total volume enclosed by the membrane should stay close to a target value V₀. Volume is computed from the signed volumes of tetrahedra formed by each triangle and the origin (a standard formula). If volume is too low, push all vertices outward; if too high, push them inward. This is the most important constraint for making the cell look "alive" — it's what makes it bounce back when compressed.

  **Computing volume and its gradient (precise recipe):** For a closed triangle mesh, the enclosed volume is `V = (1/6) Σ_t (x_a · (x_b × x_c))`, where the sum is over all triangles t with vertices a, b, c in consistent winding order. The gradient of V with respect to vertex x_i is: `∂V/∂x_i = (1/6) Σ_{t ∈ triangles(i)} (x_b × x_c)` — i.e., for each triangle containing vertex i, you take the cross product of the *other* two vertices of that triangle (in the role they play when i is in position a; cycle the roles for b and c positions). This gives a per-vertex 3D gradient vector. The XPBD correction for the volume constraint is then: `Δx_i = −(C / (Σ_j w_j |∇_j C|² + α/Δt²)) · w_i · ∇_i C`, where C = V − V₀, w_i is the inverse mass of vertex i, and α is the compliance. **Do not skip the per-vertex gradient computation** — a naive "push all vertices along their normal" does not conserve momentum and produces drift.

- **Area constraint**: The total surface area should stay close to a target value A₀. This prevents the membrane from collapsing into a point (which would satisfy the volume constraint by becoming a tiny dense sphere) or ballooning out (which would satisfy stretch constraints by making all edges longer). Area is computed as the sum of triangle areas.

  **Computing area and its gradient:** Total area `A = Σ_t (1/2)|e₁ × e₂|` where e₁ = x_b − x_a and e₂ = x_c − x_a. The gradient of A with respect to vertex x_a of triangle t is: `∂A_t/∂x_a = −(1/(2|e₁ × e₂|)) · ((e₁ × e₂) × e₁ + (e₁ × e₂) × e₂)` — more practically, use the formula `∂A_t/∂x_i = (1/(2A_t)) · n_t × e_opposite`, where n_t is the triangle normal (unnormalized cross product) and e_opposite is the edge of the triangle opposite vertex i. Accumulate gradients for each vertex across all its adjacent triangles. The XPBD correction follows the same formula as for volume.

  **Warning — area vs. stretch constraint interaction:** The global area constraint and per-edge stretch constraints partially overlap: both resist changes in surface size. If both have low compliance (high stiffness), the solver will ping-pong between them — stretch pushes vertices one way, area pushes them back, and convergence stalls. To avoid this: (a) make the area constraint softer (higher compliance) than stretch, so it acts as a gentle global correction rather than a competing stiff constraint; (b) solve stretch *before* area in the constraint ordering, so area acts as a post-correction; (c) use warm-starting (carry over the Lagrange multiplier λ from the previous frame as the initial guess for the next frame) to help the solver converge faster. Start with area compliance at 10–100× the stretch compliance and tune from there.

- **Bending constraint**: For each pair of triangles sharing an edge, the dihedral angle (the angle between their normals) should stay close to a rest value. This gives the membrane resistance to folding and creasing. It is the most expensive constraint and can be added after the others are working.

**Membrane vertex masses:** Every XPBD correction requires inverse masses (w_i = 1/m_i) for each vertex. Compute vertex masses by distributing triangle mass uniformly: for each triangle, add one-third of the triangle's mass to each of its three vertices. Triangle mass = `ρ_membrane × A_t × thickness`, where ρ_membrane is the membrane surface density and thickness is a nominal membrane thickness (e.g., 0.01 × cell radius). If you set all vertices to equal mass (m = total_mass / n_vertices), that works as a starting point but produces slightly worse behavior at mesh irregularities.

Each constraint function returns a **residual** — a number measuring how much the constraint is still violated after projection. These residuals are logged for diagnostics. **The residual is measured after projection, not before** — it tells you how much error remains, which is the signal you use to decide if iteration counts and compliance values are adequate.

#### 2.3.2 The Cytoplasm (SPH Fluid)

The cytoplasm is represented as a set of **particles** — discrete chunks of fluid that move through space and interact with their neighbors.

**Data per particle:**
- Position (3D vector)
- Velocity (3D vector)
- Mass (scalar, usually uniform)
- Density (scalar, computed each step from neighbors)
- Pressure (scalar, derived from density)

**How SPH works (for someone who has never seen it):**

Imagine you have a jar of marbles, and each marble represents a small parcel of fluid. To figure out what each marble "feels" (pressure, viscosity), you look at the marbles near it and compute a weighted average, where closer marbles contribute more.

The "weighted average" uses **kernel functions** — smooth, bell-shaped functions that are large at distance zero and fall to zero at a cutoff distance *h* (the "smoothing length"). Different kernel shapes are used for different quantities:

- **Poly6 kernel**: Used for computing density. It is a smooth, non-negative function. You sum up the mass of all neighbors weighted by this kernel to get each particle's density.
- **Spiky gradient kernel**: Used for computing pressure forces. Its gradient points away from nearby particles, which is what creates the repulsive force when particles are too close (high density → high pressure → push apart).
- **Viscosity Laplacian kernel**: Used for computing viscous drag. It smooths out velocity differences between neighbors, making the fluid resist shearing motion.

**The fluid step (each frame):**
1. For each particle, find all neighbors within distance *h*. This requires an efficient spatial data structure (see below).
2. Compute each particle's density by summing neighbor contributions through the Poly6 kernel.
3. Compute pressure from density using an equation of state (e.g., P = k(ρ − ρ₀), where ρ₀ is the rest density and k is a stiffness constant).
4. Compute pressure forces (Spiky gradient) and viscosity forces (Viscosity Laplacian) for each particle.
5. Apply Position-Based Fluids (PBF) corrections: iteratively adjust particle positions to reduce density errors, making the fluid more incompressible.

**Position-Based Fluids (PBF)** is a technique that treats incompressibility as a constraint rather than a force. Instead of computing pressure forces and hoping they keep the density right, PBF directly moves particles to positions where the density constraint is better satisfied. This is much more stable at large timesteps, which is critical for real-time simulation.

#### 2.3.3 Neighbor Search

Every SPH computation requires knowing which particles are near each other. Naively checking all pairs is O(n²), which is far too slow for thousands of particles.

The solution is a **uniform grid spatial hash**:

1. Divide space into a grid of cubic cells, each with side length equal to the smoothing length *h*.
2. For each particle, compute which grid cell it belongs to (by dividing its position by *h* and rounding down).
3. Hash the grid cell coordinates to a bucket in a hash table.
4. To find neighbors of particle *i*, look at the 27 grid cells (3×3×3 neighborhood) around particle *i*'s cell and check distances to particles in those buckets.

This gives O(n) neighbor search in practice (each particle checks only a bounded number of candidates).

The neighbor grid must be **rebuilt every frame** because particles move. It must be **deterministic** — the same particle positions must always produce the same neighbor lists, regardless of build order. And it should be independently testable (you should be able to unit-test it without any physics code).

#### 2.3.4 Fluid–Membrane Coupling

The membrane and fluid must know about each other. The coupling described here is an **MVP collision-and-projection hack**, not a physically grounded immersed boundary method. It will produce artifacts (mild jitter near the boundary, occasional sticking, small energy injection). These are acceptable for the first working version. The goal is to get a stable, visible coupling running, then iterate toward cleaner methods (signed distance fields, proper immersed boundary forces) once the sim is otherwise stable.

**No-leak constraint**: After the fluid step, some particles may have moved outside the membrane surface. For each such particle:
1. Find the nearest triangle on the membrane.
2. Project the particle back to the surface and push it slightly inward.
3. Compute the reaction impulse (equal and opposite to the correction) and distribute it to the three vertices of the nearest triangle, weighted by barycentric coordinates.

**Inside/outside determination — the practical strategy**: Ray casting (count triangle crossings) is conceptually simple but too slow per-particle per-frame and fragile around degenerate triangles. Instead, use a **signed distance field (SDF)** approach:

- At the start of each step (after membrane positions are finalized), voxelize the membrane into a low-resolution 3D grid (cell size ≈ 2–4× smoothing length h). For each voxel, store the signed distance to the nearest triangle. The sign is determined by the dot product of the displacement with the triangle normal (positive = outside, negative = inside). This is an O(n_triangles × n_voxels) precomputation, but the grid is coarse (32³ to 64³ for a single cell) and rebuilt once per step, not per particle.
- For each particle, look up the signed distance from the nearest voxel (with trilinear interpolation for smoothness). If the signed distance is positive (outside), the particle has leaked and needs projection.
- The closest surface point for projection is approximated by stepping the particle inward along the SDF gradient by the signed distance value, then clamping to a small offset inside the surface (e.g., 0.1 × h).

This SDF approach is deterministic, handles degenerate triangles gracefully (the distance field smooths over them), and is fast enough for real-time use. Build the SDF once per step, query it per particle.

**Anti-tunneling for large Δt**: If Δt is large relative to particle speed, a particle can jump entirely through the membrane in one step without ever registering as "outside." To catch this, perform **continuous collision detection (CCD)** on fast-moving particles: for any particle whose predicted displacement exceeds 0.5 × h, test whether the line segment from old position to predicted position intersects any membrane triangle. If it does, clamp the particle to the intersection point and push inward. This is only needed for the fastest-moving particles (typically a small fraction), so the cost is bounded.

**Reaction force stability**: Distributing reaction impulses to membrane vertices can destabilize the membrane solver if the impulses are too large or too sudden. To prevent this: (a) clamp the reaction impulse magnitude to a maximum value (e.g., 0.5 × particle mass × h / Δt); (b) apply the reaction as a velocity correction *after* membrane constraint solving, not as a force before prediction (this prevents the membrane solver from fighting the coupling); (c) apply damping to the reaction (multiply by a coupling damping factor, e.g., 0.8, to dissipate energy injected by the projection).

**Why barycentric weighting?** If a particle hits the membrane near one vertex of a triangle, most of the reaction should go to that vertex. If it hits near the center, the reaction is split roughly equally among all three vertices. Barycentric coordinates (the natural coordinate system of a triangle) give exactly this weighting.

**Cytoplasm realism caveat**: Real cytoplasm is closer to a viscoelastic or active gel than a Newtonian fluid. The SPH model here treats it as a viscous Newtonian fluid, which is a deliberate simplification for the MVP. This means the simulation will not capture elastic recoil of the cytoskeleton, stress-stiffening under large deformation, or active contractile behavior from molecular motors. These effects can be added later by introducing elastic spring networks between particles (for viscoelasticity) or active force terms in the fluid step (for contractility). For the initial simulator, the Newtonian approximation is adequate — the internal fluid will flow, slosh, and exert pressure on the membrane, which is enough to validate the coupling and produce visually convincing cell motion.

**Diagnostics**: The coupling module should report the number of penetrations detected each frame and the maximum penetration depth. If these numbers are growing over time, the coupling is failing and parameters need adjustment.

### 2.4 The Key Abstraction: Everything is a Function

The entire simulator is designed so that every component of interest is an explicit, named function with a clear signature.

#### 2.4.1 The Step Function

The single most important function in the system:

```
S_{t+1} = Step(S_t, Θ, U_t, Δt)
```

- **S_t** is the full state: all membrane vertex positions and velocities, all particle positions and velocities, and any cached data (neighbor grid, normals).
- **Θ** is the parameter struct: every tunable number in the simulation (stiffness values, compliance parameters, viscosity, smoothing length, iteration counts, etc.).
- **U_t** is external input: any forces or boundary conditions applied from outside.
- **Δt** is the timestep.

This function is **deterministic**: given the same inputs, it always produces the same output. It is **pure** in the sense that it has no hidden state — everything it depends on is in its arguments. This is what makes it optimizable.

#### 2.4.2 Constraint Projection Functions

Each physical property of the cell is a constraint projection:

```
ProjectVolume(mesh, V₀, α_v) → corrected positions, residual
ProjectArea(mesh, A₀, α_a) → corrected positions, residual
ProjectStretch(mesh, edges, α_s) → corrected positions, residual
ProjectBend(mesh, α_b) → corrected positions, residual
ProjectFluidIncompressibility(particles, α_p) → corrected positions, residual
ProjectNoLeak(particles, mesh, α_n) → corrected positions, penetration count
```

Each takes predicted positions in, returns corrected positions out, and reports how much violation remains.

#### 2.4.3 Energy Functions (for Optimization)

Even though the runtime uses constraints (not forces), it is useful to define energy functions for debugging and parameter fitting:

```
E_total(S, Θ) = E_membrane + E_bend + E_volume + E_area + E_fluid + E_contact + E_active
```

If you can compute E_total, you can define a loss function and use optimization algorithms (CMA-ES, Bayesian optimization, or gradient-based methods if you add autodiff) to fit parameters or learn policies.

#### 2.4.4 Cell Policy Functions (Where Intelligence Will Live)

Once the physical substrate is stable, active behavior becomes a set of explicit policy functions:

```
τ(x, t) = TensionPolicy(local_state)    — cortex tension at point x
p(x, t) = ProtrusionPolicy(local_state)  — protrusion force at point x
a(x, t) = AdhesionPolicy(local_state)    — adhesion strength at point x
```

These are the functions you will eventually optimize or learn. They are the "knobs" that turn a passive physical cell into an active agent.

### 2.5 The Optimization-from-Day-1 Design

There are two kinds of optimization in this project, and both must be supported from the start.

#### 2.5.1 Engineering Optimization (Performance and Stability)

The simulation must run in real time. This means:

- **Fixed timestep**: The simulation always advances by the same Δt, regardless of rendering frame rate. This makes behavior reproducible and prevents instability from variable timesteps.
- **Fixed iteration budgets**: The constraint solver runs a fixed number of iterations per step (e.g., 10 iterations for membrane constraints, 5 for fluid). This makes runtime predictable.
- **Deterministic step order**: Constraints are always solved in the same order. The neighbor grid is always built the same way. There is no randomness unless explicitly seeded.
- **Predictable memory layout**: Particle data is stored in arrays (ideally Structure-of-Arrays for cache efficiency). No pointer-chasing, no dynamic allocation during the step.
- **Neighbor search as a first-class module**: The spatial hash is the performance bottleneck. It must have its own tests and benchmarks, separate from the rest of the simulation.

#### 2.5.2 Scientific Optimization (Parameter Fitting and Policy Learning)

The simulation must be usable as an optimization target. This means:

- **All parameters in one struct (Θ)**: No magic numbers buried in code. Every tunable value has a name and lives in the parameter struct.
- **Deterministic, reproducible Step()**: Given the same S, Θ, U, Δt, you always get the same S'. This is required for any optimization algorithm to work reliably.
- **Observable diagnostics**: Every frame, the simulation logs volume drift, area drift, constraint residuals, energy, penetration counts, and performance timings. These are the "sensors" that tell you whether the simulation is healthy.
- **Explicit loss functions**: You can define scalar objectives over trajectories:
  - L_volume_drift = (V(t) − V₀)² — penalizes volume instability
  - L_area_drift = (A(t) − A₀)² — penalizes area instability
  - L_energy = Σ‖v‖² — penalizes jitter
  - L_task = distance(centroid, goal) — penalizes failure to reach a target

These losses can be combined and minimized over Θ using black-box optimization (CMA-ES, Bayesian optimization) or, if you later add automatic differentiation, gradient-based methods.

### 2.6 Concrete Defaults (Starting Values That Work)

These are tested starting values. They are not optimal — they are stable. Tune from here.

**Geometry and scale:**
- Cell radius: 1.0 (all other quantities are in units relative to this).
- Icosphere subdivisions: 3 (giving 642 vertices, 1280 triangles, 1920 edges).
- Fluid particle count: ~5,000 (Poisson disk sampled inside the icosphere, spacing ≈ 0.08).
- Smoothing length h: 0.12 (≈ 1.5× particle spacing; this is the single most sensitive SPH parameter — too small and particles clump, too large and detail is lost).

**Timestep and iterations:**
- Δt: 0.005 seconds (200 Hz physics). This is small enough for XPBD stability with the compliance values below, and still allows 3–4 substeps per 60 Hz render frame.
- Membrane constraint iterations: 10 per step.
- Fluid PBF iterations: 4 per step.
- The step runs once per physics tick, not once per render frame. Use a fixed-step accumulator (see section 3.5).

**Compliance values (XPBD α, units: 1/(stiffness × Δt²)):**
- Stretch compliance: 1e-6 (very stiff — edges barely stretch).
- Volume compliance: 1e-7 (extremely stiff — volume should not drift).
- Area compliance: 1e-4 (soft — 100× softer than stretch, to avoid ping-ponging).
- Bending compliance: 1e-3 (soft — allow gentle folding; stiffen later if creasing is a problem).
- Fluid PBF relaxation (ε in PBF papers): 100.0 (the "tensile instability" correction; standard PBF default).

**Damping:**
- Global velocity damping: 0.98 per step (multiply all velocities by this factor after each step). This dissipates energy and prevents slow drift. A value of 1.0 means no damping (conservative); 0.95 is heavy damping.
- Coupling reaction damping: 0.8 (reaction impulses from no-leak projection are multiplied by this before application).

**Fluid parameters:**
- Rest density ρ₀: 1000.0 (water-like, in simulation units).
- Viscosity (XSPH coefficient): 0.01 (low — cytoplasm should flow, not freeze; increase to 0.1 for viscoelastic feel).
- Particle mass: ρ₀ × (4/3 π r³) / n_particles (so total fluid mass equals the volume at rest density).

**What to do when these don't work (and they won't, first try):**
If the membrane explodes: reduce Δt by 2× or increase stretch compliance by 10×.
If the membrane collapses: reduce volume compliance (make it stiffer).
If particles leak: reduce Δt or increase no-leak projection iterations.
If the fluid is too stiff / clumps: increase h slightly (e.g., 0.14) or reduce rest density.
If the sim is too slow: reduce particle count first (it dominates cost), then reduce iterations.

### 2.7 Integration Scheme (Precise Specification)

Both membrane vertices and fluid particles use the same integration scheme: **Symplectic Euler prediction + XPBD/PBF constraint projection + velocity update from position delta**. This is the standard PBD/XPBD integration and is different from force-based Verlet or explicit Euler.

The exact sequence for each substep:

```
1. For each vertex/particle i:
     v_i = v_i + Δt * (f_external_i / m_i)     // apply gravity, damping, external forces
     v_i = v_i * global_damping                  // velocity damping
     p_i = x_i + Δt * v_i                        // predicted position (stored separately)

2. Constraint projection (on predicted positions p_i):
     [fluid PBF iterations on particle p_i]
     [membrane stretch on vertex p_i]
     [membrane area on vertex p_i]
     [membrane volume on vertex p_i]
     [membrane bending on vertex p_i]
     [no-leak coupling: project leaked particles, apply reaction to vertex p_i]

3. Velocity update:
     v_i = (p_i - x_i) / Δt                     // new velocity from position correction
     x_i = p_i                                    // commit corrected positions
```

**Predicted positions** are stored in a separate array (`pred_x`, `pred_y`, `pred_z` for particles; a separate `pred_positions` vector for membrane vertices). They are allocated once and reused every step. They are *not* part of the persistent state — they are temporaries.

**Constraint ordering rationale:** Fluid PBF runs first because it only moves particles (not membrane vertices), and its position corrections affect the no-leak test that runs last. Membrane stretch runs before area/volume because stretch is the stiffest local constraint and converges fastest; running it first gives area/volume a better starting point. No-leak runs last because it needs both fluid and membrane positions to be nearly finalized. If you change this order, the system may still converge, but you may need more iterations.

### 2.8 Constraint Solve Ordering and What to Do When It Fails

The constraint order above (fluid → stretch → area → volume → bending → no-leak) is a starting point, not a guarantee. Signs that the ordering is wrong:

- **Residuals oscillate** instead of decreasing across iterations: constraints are fighting each other. Fix: soften the weaker constraint (increase its compliance) or add warm-starting.
- **Volume drifts monotonically** even though volume residual per step is small: energy injection from coupling or integration error is accumulating. Fix: add explicit volume correction as a post-step clamp (set V to V₀ by uniform scaling) or reduce Δt.
- **Stretch residuals blow up after area/volume projection**: area/volume corrections are too aggressive relative to stretch. Fix: increase area/volume compliance.

**Warm-starting**: For each constraint, carry over the Lagrange multiplier λ from the previous step as the initial value for the current step. This dramatically improves convergence for stiff constraints. In XPBD, this means storing one float per constraint (per edge for stretch, one global for volume, one global for area) and initializing `λ = λ_prev` instead of `λ = 0` at the start of each step.

---

## Part III — The Visualization Stack: OGRE Classic + OgreBites

### 3.1 Why OGRE

OGRE is a mature, open-source 3D rendering engine that provides everything you need to visualize the simulation without writing low-level OpenGL/Vulkan code:

- **Windowing and input** (via SDL2, handled by OgreBites)
- **Scene graph** (scene managers, scene nodes, cameras, lights)
- **Materials and shaders** (built-in lit/unlit materials, custom shaders later)
- **Resource management** (textures, meshes, materials loaded from config files)
- **Debug UI** (OgreBites TrayManager for FPS counters, sliders, toggles)
- **Application skeleton** (OgreBites::ApplicationContext handles initialization and teardown)

We use **OGRE classic (1.x)**, not OGRE-Next (2.x). Classic OGRE is stable, widely packaged (available through vcpkg), and has simpler setup. OGRE-Next offers a modern rendering path but requires more configuration (HLMS registration, specific data folders) that is unnecessary for this project's visualization needs.

### 3.2 What We Render

**The membrane** is rendered as a triangle mesh using an OGRE `ManualObject`. Each frame, the ManualObject is cleared and rebuilt from the simulation's current vertex positions, normals, and triangle indices. This is the simplest approach and is performant up to roughly **5,000–8,000 triangles** on most drivers. Beyond that (higher subdivision levels, or multiple cells), ManualObject rebuild becomes a bottleneck because it forces a CPU-side buffer upload every frame. The planned upgrade path is a **dynamic `HardwareVertexBuffer`**: create the mesh once (with `HBU_DYNAMIC_WRITE_ONLY_DISCARDABLE` usage), then lock/update the vertex buffer each frame without rebuilding the index buffer. The `MembraneRenderable::update()` API stays the same — the implementation changes underneath. Plan to switch to dynamic VBO once the simulation is stable and you are profiling for 60 fps.

**The fluid particles** are rendered as an OGRE `BillboardSet`. Each billboard is a small, camera-facing quad at a particle's position. Each frame, the billboard positions are updated from the simulation's particle position array. BillboardSet works reliably up to roughly **10,000–15,000 billboards**; beyond that, driver overhead from per-billboard position updates can drop below 60 fps. The fallback for larger particle counts is a **custom `SimpleRenderable`** (or `ManualObject` with `OT_POINT_LIST`) backed by a single dynamic vertex buffer, rendered with `GL_POINTS` and a point-sprite material. This gives the GPU a single draw call with one buffer upload, which scales to 100k+ particles. Again, the `ParticlesRenderable::update()` API does not change — only the implementation.

### 3.3 The OGRE Application Structure

The application is a single class (`CellApp`) that inherits from `OgreBites::ApplicationContext` and `OgreBites::InputListener`. It handles:

- **Setup**: Creating the scene manager, camera, viewport, and lights. Creating the `MembraneRenderable` and `ParticlesRenderable` objects. Constructing the simulation with initial conditions.
- **Per-frame update**: Advancing the simulation by one step, syncing the new state to the renderables, and updating the debug UI.
- **Input handling**: Keyboard and mouse events for camera control, pause/step, and parameter adjustment.
- **Debug UI**: OgreBites TrayManager provides sliders for simulation parameters (timestep, solver iterations, stiffness, particle display size) and toggles (wireframe, show normals, show particles, pause).

### 3.4 The Rendering Boundary

The rendering layer must accept **only plain arrays**:

```
MembraneRenderable::update(positions: span<Vec3>, normals: span<Vec3>, indices: span<uint32>)
ParticlesRenderable::update(positions: span<Vec3>)
```

No OGRE types appear in the simulation. No simulation types appear in the rendering. The bridge between them is the `SimView` struct, which holds spans pointing into the simulation's state arrays.

### 3.5 OGRE Timing and the Fixed-Step Accumulator

OGRE's frame event gives a **variable Δt** (the wall-clock time since the last frame). The simulation requires a **fixed Δt** for stability and determinism. These must be reconciled with a fixed-step accumulator:

```
float accumulator = 0.0f;
const float physics_dt = 0.005f; // fixed physics timestep

// In the OGRE frame callback:
accumulator += frame_event.timeSinceLastFrame;
accumulator = std::min(accumulator, 0.05f); // clamp to prevent spiral of death

while (accumulator >= physics_dt) {
    sim.step(physics_dt);
    accumulator -= physics_dt;
}

// Render with the latest sim state (no interpolation needed for MVP;
// add lerp between states later if visual stuttering is noticeable).
```

The clamp at 0.05 seconds (10 physics steps maximum per render frame) prevents the "spiral of death" where slow frames cause many substeps, which cause even slower frames. If the simulation cannot keep up at the physics rate, it slows down gracefully rather than exploding.

### 3.6 Resources and Materials

OGRE requires a `resources.cfg` file that lists directories containing materials, textures, and shaders. For this project:

- A `media/` directory holds material scripts for the membrane surface and the particle billboards.
- Optionally, OGRE's sample media path is included for default materials during early development.

At startup, `ResourceGroupManager::initialiseAllResourceGroups()` loads everything listed in the config.

**Minimum viable material for the membrane** (`media/membrane.material`):

```
material CellMembrane
{
    technique
    {
        pass
        {
            ambient 0.2 0.3 0.2
            diffuse 0.4 0.7 0.4
            specular 0.6 0.6 0.6 40
            scene_blend alpha_blend
            depth_write on
            cull_hardware none        // render both sides (membrane can be seen from inside)
        }
    }
}
```

**Minimum viable material for particle billboards** (`media/particle.material`):

```
material CellParticle
{
    technique
    {
        pass
        {
            lighting off
            scene_blend alpha_blend
            depth_write off           // critical: without this, alpha-blended billboards
                                      // will z-fight and produce visual artifacts
            depth_check on

            texture_unit
            {
                texture particle.png
            }
        }
    }
}
```

The `depth_write off` on the particle material is the single most common first-time OGRE mistake with billboards. Without it, alpha-blended particles write to the depth buffer and occlude each other incorrectly, producing flickering and popping. `depth_check on` ensures particles are still hidden behind the membrane when they should be.

For `particle.png`: create a 32×32 white circle with a radial alpha gradient (opaque center, transparent edge) in any image editor. OGRE needs this texture to exist at startup or billboard rendering will fail silently.

---

## Part IV — Pipelines: What Runs, In What Order

### 4.1 Build Pipeline

1. **Install OGRE** using vcpkg (or your distribution's package manager). vcpkg handles OGRE's dependencies (SDL2, FreeImage, etc.) automatically.
2. **Configure with CMake**, pointing to the vcpkg toolchain file. CMake finds OGRE via `find_package(OGRE CONFIG REQUIRED)`.
3. **Build three targets**:
   - `cell_sim`: The physics library. No OGRE includes. Can be tested headless.
   - `cell_render_ogre`: OGRE adapter code (MembraneRenderable, ParticlesRenderable, OgreBridge).
   - `cell_app`: The executable. Links cell_sim and cell_render_ogre. Contains the OgreBites application.
4. **Run** `cell_app`. It opens an OGRE window and starts simulating.

### 4.2 Runtime Pipeline (Per Frame)

Each frame proceeds in this order:

1. **Determine how many physics steps to run**: Use the fixed-step accumulator described in section 3.5. Accumulate the OGRE frame time, then run `sim.step(physics_dt)` as many times as the accumulator allows (clamped to a maximum of 10 steps to prevent spiral of death). The physics timestep is fixed at 0.005 seconds (see section 2.6 for defaults).
2. **Simulation step**: Call `sim.step(Δt)`. This advances all physics state (membrane positions/velocities, particle positions/velocities, neighbor grid, diagnostics).
3. **Render sync**: Call `membraneRenderable.update(sim.state().membranePositions, sim.state().membraneNormals, sim.state().membraneIndices)` and `particlesRenderable.update(sim.state().particlePositions)`. This uploads the new state to OGRE objects.
4. **OGRE renders** the frame (handled by OGRE's internal loop).
5. **UI update**: Refresh debug sliders, FPS counter, and diagnostic readouts.

### 4.3 Simulation Step Pipeline (Inside `sim.step()`)

Each call to `step(Δt)` runs the following sub-steps in order:

1. **Apply external forces** (gravity, user-applied forces) and **damping** (velocity damping for stability).
2. **Predict positions**: For each vertex and particle, compute a predicted position using semi-implicit Euler: `x_predicted = x + v * Δt + a * Δt²`. These predicted positions are the starting point for constraint solving.
3. **Build SPH neighbor grid**: Rebuild the spatial hash from predicted particle positions.
4. **Solve constraints iteratively** (fixed number of iterations):
   - **Fluid incompressibility (PBF)**: Adjust particle positions to reduce density errors.
   - **Membrane stretch**: Adjust vertex positions to bring edge lengths closer to rest lengths.
   - **Membrane area and volume**: Adjust vertex positions to bring total area and volume closer to targets.
   - **Membrane bending** (if enabled): Adjust vertex positions to preserve dihedral angles.
   - **No-leak coupling**: Project escaped particles back inside the membrane; apply reaction to membrane vertices.
5. **Update velocities**: For each vertex and particle, compute the new velocity from the position change: `v = (x_corrected − x_old) / Δt`.
6. **Compute and log diagnostics**: Volume drift, area drift, constraint residuals, penetration count, max penetration depth, performance timings.

---

## Part V — Project Layout: Every File, Explained

### 5.1 Top-Level Files

#### `README.md`
The project's documentation. Points to this guide and explains how to build and run.

#### `CMakeLists.txt`
The CMake build script. Defines three targets:
- **`cell_sim`**: Static library. Sources from `core/` and `sim/`. No OGRE dependencies. Compile with optimization flags (`-O2` or `-O3`) even in debug builds for performance-sensitive code (SPH kernels, constraint projections).
- **`cell_render_ogre`**: Static library. Sources from `render/`. Links against OGRE and OgreBites. Depends on `cell_sim` for the `SimView` type.
- **`cell_app`**: Executable. Sources from `app/`. Links against `cell_render_ogre` and `cell_sim`.

#### `resources.cfg`
OGRE resource configuration file. Contains paths to:
- `media/` (project-specific materials and textures)
- Optionally, OGRE's sample media directory (for default fonts and debug materials)

#### `media/`
Directory containing:
- `membrane.material`: An OGRE material script for the membrane surface. Start with a basic lit material (ambient + diffuse + specular). Later, add translucency or subsurface scattering for realism.
- `particle.material`: An OGRE material script for particle billboards. A simple unlit, alpha-blended material with a soft circle texture.
- `particle.png` (or similar): A small texture for particle billboards (a soft white circle with alpha falloff).

---

### 5.2 `core/` — Math, Utilities, and Infrastructure

These files provide the foundation that everything else builds on. They have no dependencies on simulation or rendering code.

#### `core/Vec3.h`

A minimal 3D vector type. This is the basic data type used throughout the simulation for positions, velocities, forces, and normals.

Must support: addition, subtraction, scalar multiplication, dot product, cross product, length, normalization, and component-wise access.

**Critical**: This type must be independent of OGRE's vector type. The rendering layer will convert `Vec3` to `Ogre::Vector3` at the boundary; the simulation never sees OGRE types.

Why not just use `glm::vec3` or `Ogre::Vector3`? Because the simulation must remain renderer-independent. A custom `Vec3` (or `glm::vec3` if you prefer, since glm has no rendering dependencies) keeps the dependency graph clean.

#### `core/Span.h`

If your compiler does not support C++20's `std::span`, provide a minimal span wrapper: a non-owning view into a contiguous array, defined by a pointer and a size. This is used at the simulation–rendering boundary to pass arrays without copying.

If you have C++20, just `#include <span>` and use `std::span` directly.

#### `core/Timer.h`

A high-resolution timer for two purposes:
1. **Fixed timestep control**: Measure elapsed wall-clock time per frame and decide how many simulation substeps to run.
2. **Profiling**: Measure how long each phase of the simulation step takes (neighbor search, constraint solving, etc.) for performance optimization.

Use `std::chrono::high_resolution_clock` internally.

#### `core/Log.h`

A structured logging system for diagnostics. Each frame, the simulation writes:
- Constraint residuals (stretch, area, volume, bending, fluid density error)
- Drift metrics (current volume vs. target, current area vs. target)
- Performance counters (neighbor search time, constraint solve time, total step time)
- Penetration metrics (count, max depth)

This can be as simple as writing comma-separated values to a file, or as sophisticated as a structured binary log. The key requirement is that it is always on during development, so you can diagnose instabilities after the fact.

---

### 5.3 `sim/` — Simulation Orchestration

#### `sim/SimParams.h`

Defines **Θ**, the canonical parameter struct. This is the single source of truth for every tunable number in the simulation.

**Must contain:**

*Membrane parameters:*
- `float stretch_compliance` — How much the membrane resists stretching (lower = stiffer).
- `float area_compliance` — How much the total surface area is allowed to deviate from target.
- `float volume_compliance` — How much the total volume is allowed to deviate from target.
- `float bend_compliance` — How much the membrane resists bending (folding/creasing).
- `float target_volume` — The desired enclosed volume (V₀).
- `float target_area` — The desired surface area (A₀).

*Fluid parameters:*
- `float rest_density` — The desired fluid density (ρ₀). Particles try to maintain this density.
- `float viscosity` — How much the fluid resists shearing (higher = more syrupy, like real cytoplasm).
- `float smoothing_length` — The SPH kernel radius (h). Determines how far each particle "sees."
- `float fluid_particle_mass` — Mass of each SPH particle.

*Solver parameters:*
- `int membrane_iterations` — How many times membrane constraints are projected per step.
- `int fluid_iterations` — How many PBF correction iterations per step.

*Coupling parameters:*
- `float noleak_compliance` — How aggressively escaped particles are pushed back inside.
- `float coupling_damping` — Damping applied to membrane–fluid interaction forces.

*Integration parameters:*
- `float dt` — The fixed timestep.
- `float global_damping` — Velocity damping applied to all elements for stability.

**Design rule**: If a number appears in any equation in the simulation, it must live in this struct. No magic constants in code.

#### `sim/SimState.h`

Defines **S**, the canonical state struct. This contains everything that changes over time.

**Must contain:**

*Membrane state:*
- `std::vector<Vec3> membrane_positions` — Current vertex positions.
- `std::vector<Vec3> membrane_velocities` — Current vertex velocities.
- `std::vector<Vec3> membrane_normals` — Current vertex normals (recomputed each step).
- `std::vector<Triangle> membrane_triangles` — Triangle indices (fixed after mesh generation).
- `std::vector<Edge> membrane_edges` — Edge list with rest lengths (fixed after mesh generation).

*Particle state:*
- `std::vector<Vec3> particle_positions` — Current particle positions.
- `std::vector<Vec3> particle_velocities` — Current particle velocities.
- `std::vector<float> particle_densities` — Current computed densities.
- `std::vector<float> particle_masses` — Particle masses (usually uniform).

*Caches and temporaries (owned by SimState, allocated once, reused every step):*
- `NeighborGrid neighbor_grid` — rebuilt from particle positions at the start of each step.
- `std::vector<Vec3> membrane_pred_positions` — predicted membrane vertex positions. Size = n_vertices, allocated at initialization. Used as the working buffer for all membrane constraint projections. After all constraints are solved, committed to `membrane_positions`.
- `std::vector<float> particle_pred_x, particle_pred_y, particle_pred_z` — predicted particle positions (SoA). Size = n_particles each, allocated at initialization. Used as the working buffer for PBF iterations and no-leak coupling. After all constraints are solved, committed to particle position arrays.

**How predicted positions are shared between solvers:** PBF and membrane constraints operate on different predicted-position arrays (particles and vertices, respectively). They do not directly share arrays. However, the no-leak coupling reads *both* predicted arrays (it needs to know where the particles are *and* where the membrane is) and writes corrections to both. This is why no-leak runs last in the constraint order — it acts as the final reconciliation between the two subsystems. Within a single constraint iteration, all solvers see the latest predicted positions from the *current* iteration (Gauss-Seidel style), not the positions from the start of the iteration (Jacobi style). This is important: Gauss-Seidel converges faster for the same iteration count.

**SimView**: A lightweight view struct that exposes read-only spans into the state arrays. This is what the renderer consumes.

```
struct SimView {
    std::span<const Vec3> membrane_positions;
    std::span<const Vec3> membrane_normals;
    std::span<const uint32_t> membrane_indices;  // flattened triangle indices
    std::span<const Vec3> particle_positions;
};
```

**SoA vs. SimView reconciliation**: There is a tension between the recommended SoA layout for particles (separate `pos_x[]`, `pos_y[]`, `pos_z[]` arrays for cache-efficient SPH computation) and the SimView's `span<const Vec3>` for particle positions (which expects an AoS layout: interleaved x,y,z per particle).

The canonical resolution: **SimState owns a render buffer** — a `std::vector<Vec3> particle_positions_packed` — that is populated at the end of each step by packing the SoA data:

```cpp
// At end of step(), after all constraints are solved:
for (size_t i = 0; i < particles.count; ++i) {
    particle_positions_packed[i] = Vec3{
        particles.pos_x[i], particles.pos_y[i], particles.pos_z[i]
    };
}
```

This costs O(n) per step with a trivial loop, which is negligible compared to SPH kernel evaluations. SimView then points into this packed buffer. The membrane is already AoS (vertex positions as `vector<Vec3>`), so no packing is needed for it.

If you start with AoS particles (which is fine for the first milestone), SimView can point directly into the particle position array with no packing step. The render buffer packing only becomes necessary if/when you migrate to SoA for performance.

**Flattened indices ownership**: `membrane_indices` in SimView is a span into a `std::vector<uint32_t>` that SimState owns. This vector is populated once during mesh construction by flattening the triangle list: for each `Triangle{a, b, c}`, push `a`, `b`, `c` into the flat array. It is rebuilt only when mesh topology changes (never during normal simulation).

#### `sim/CellSim.h`

The public API of the simulation. This is the only header that application code needs to include.

**Public interface:**
- **Constructor**: Takes initial conditions (mesh, particle positions) and a `SimParams` struct.
- **`void step(float dt)`**: Advances the simulation by one timestep. This is the Step function.
- **`const SimState& state() const`**: Returns a const reference to the current state (for creating SimViews).
- **`SimView view() const`**: Returns a SimView pointing into the current state.
- **`Diagnostics diagnostics() const`**: Returns the most recent diagnostic measurements (residuals, drift, penetration counts, timings).

**No OGRE includes.** This file must compile without any rendering dependencies.

#### `sim/CellSim.cpp`

Implements the Step pipeline described in Part IV, Section 4.3.

**Critical implementation rules:**
- Step must be **deterministic**: same inputs → same outputs. No uninitialized memory, no random numbers (unless explicitly seeded and documented), no dependency on thread scheduling.
- Constraints are always solved in a **fixed order**: fluid → stretch → area/volume → bending → no-leak. Changing this order changes behavior, so it must be explicit and documented.
- **Fixed iteration counts**: The number of constraint iterations is a parameter in Θ, not a convergence criterion. This makes runtime predictable.

---

### 5.4 `sim/membrane/` — Membrane Representation and Constraints

#### `sim/membrane/MembraneMesh.h`

The mesh data structure and geometry utilities.

**Data (owned):**
- Vertex positions and velocities. **Ownership decision (canonical)**: `SimState` owns the position and velocity arrays. `MembraneMesh` is a **non-owning view** that holds pointers/spans into SimState's arrays, plus it owns the topology data (triangles, edges, adjacency) which does not change during simulation. This means: SimState has `std::vector<Vec3> membrane_positions` and `std::vector<Vec3> membrane_velocities`; MembraneMesh has `std::span<Vec3> positions` (pointing into SimState) plus `std::vector<Triangle> triangles`, `std::vector<Edge> edges`, and adjacency tables. MembraneMesh is constructed once from SimState's arrays after mesh generation, and its spans remain valid as long as SimState's vectors are not reallocated (which they should not be during simulation — reserve capacity at initialization).
- Triangle list (indices).
- Edge list with rest lengths.
- Adjacency: for each edge, the two triangles that share it (needed for bending).

**Methods:**
- **`static MembraneMesh generateIcosphere(int subdivisions, float radius)`**: Creates an icosphere mesh. This is the initial cell shape. The function generates vertices, computes triangles, extracts edges with rest lengths, and builds adjacency tables.
- **`void computeNormals()`**: Recomputes vertex normals from triangle normals (area-weighted average of adjacent triangle normals). Called every frame after constraint solving.
- **`float computeVolume() const`**: Returns the total enclosed volume using the divergence theorem (sum of signed tetrahedra volumes).
- **`float computeArea() const`**: Returns the total surface area (sum of triangle areas).
- **`void rebuildAdjacency()`**: Rebuilds edge and adjacency tables. Called once after mesh creation or topology changes.

#### `sim/membrane/Constraints.h`

Declares the constraint projection functions.

**Function signatures (XPBD style):**

```cpp
// Stretch: for each edge, project endpoints toward rest length.
// Returns total residual (sum of squared length errors).
float projectStretch(
    std::span<Vec3> positions,      // in/out: predicted positions
    std::span<const Edge> edges,    // edge definitions with rest lengths
    float compliance,               // α_s: higher = softer
    float dt                        // timestep (for XPBD compliance scaling)
);

// Volume: project all vertices to correct total volume.
// Returns absolute volume error after projection.
float projectVolume(
    std::span<Vec3> positions,
    std::span<const Triangle> triangles,
    float target_volume,
    float compliance,
    float dt
);

// Area: project all vertices to correct total surface area.
// Returns absolute area error after projection.
float projectArea(
    std::span<Vec3> positions,
    std::span<const Triangle> triangles,
    float target_area,
    float compliance,
    float dt
);

// Bending: for each shared edge, project vertices to preserve dihedral angle.
// Returns total residual.
float projectBend(
    std::span<Vec3> positions,
    std::span<const Edge> edges,
    /* adjacency info for dihedral angle computation */
    float compliance,
    float dt
);
```

#### `sim/membrane/Constraints.cpp`

Implements the constraint projections.

**How stretch constraint projection works (in detail for someone new to this):**

For each edge connecting vertices i and j:
1. Compute the current distance: `d = |positions[i] − positions[j]|`.
2. Compute the error: `C = d − rest_length`.
3. Compute the correction direction: `n = (positions[i] − positions[j]) / d` (unit vector along the edge).
4. Compute the correction magnitude using XPBD compliance: `Δλ = −C / (w_i + w_j + α/Δt²)`, where `w_i` and `w_j` are inverse masses (usually 1/mass) and `α` is the compliance.
5. Move vertex i by `+Δλ * w_i * n` and vertex j by `−Δλ * w_j * n`.

This nudges the vertices toward the rest length without overshooting (the compliance and inverse masses prevent overcorrection).

**How volume constraint projection works (precise recipe):**

1. Compute current volume V using the signed tetrahedra formula: `V = (1/6) Σ_t dot(x_a, cross(x_b, x_c))` where the sum is over all triangles t with vertices a, b, c.
2. Compute the constraint error: `C = V − V₀`.
3. Compute the gradient `∇_i C` for each vertex i. As derived in section 2.3.1, for each triangle t containing vertex i, the contribution is `(1/6) * cross(x_b, x_c)` (when i is in the "a" position; permute for b and c positions). Sum over all triangles adjacent to vertex i.
4. Compute the denominator: `denom = Σ_i w_i * |∇_i C|² + α / Δt²`, where `w_i = 1/m_i` is the inverse mass and α is the volume compliance.
5. Compute the scaling factor: `s = −C / denom`.
6. Apply the correction: for each vertex i, `Δx_i = s * w_i * ∇_i C`.

The correction conserves center of mass (because the gradients sum to zero for a closed mesh) and the compliance α controls how stiff the constraint is. If `denom` is very small (degenerate mesh), clamp it to a minimum value (e.g., 1e-10) to avoid division by zero.

**Residual reporting**: Every projection function returns the residual *after* projection — compute V again after applying corrections and return `|V_corrected − V₀|`. This is used for diagnostics: if residuals are not decreasing across iterations, something is wrong with the parameters or the constraint formulation. Residuals are also the signal for the acceptance tests defined in Part V-B.

---

### 5.5 `sim/sph/` — SPH / Position-Based Fluids Core

#### `sim/sph/Particles.h`

The particle data container.

**Data layout**: Ideally Structure-of-Arrays (SoA) for cache efficiency during neighbor traversal:

```cpp
struct Particles {
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> vel_x, vel_y, vel_z;
    std::vector<float> mass;
    std::vector<float> density;
    std::vector<float> pressure;
    // Predicted positions (used during constraint solving):
    std::vector<float> pred_x, pred_y, pred_z;
    size_t count;
};
```

If SoA feels premature, start with Array-of-Structs (AoS) — a `std::vector<Particle>` — but keep an explicit plan to migrate to SoA once the simulation is stable and you are profiling.

**Initialization**: Particles are placed inside the initial icosphere mesh. A simple approach: generate particles on a regular 3D grid, then discard any that fall outside the membrane surface (using a signed distance test or ray casting). Alternatively, use Poisson disk sampling for a more uniform distribution.

#### `sim/sph/NeighborGrid.h`

The uniform grid spatial hash.

**Public API:**
- **`void build(const Particles& particles, float h)`**: Clears the grid and inserts all particles. After this call, the grid is ready for queries.
- **`template<typename F> void forNeighbors(size_t i, F&& callback)`**: Calls `callback(j)` for every particle `j` within distance `h` of particle `i` (including i itself, or excluding — be consistent and document which).

**Internal implementation:**
1. Compute grid cell for each particle: `cell = (floor(x/h), floor(y/h), floor(z/h))`.
2. Hash the cell coordinates to a bucket index: e.g., `hash = (cell.x * 73856093) ^ (cell.y * 19349663) ^ (cell.z * 83492791)`, modulo table size.
3. Store particle indices in buckets (a flat array with offsets, or a vector of vectors).
4. For queries, iterate over the 27 neighboring cells (3×3×3 block around the query particle's cell) and check actual distances.

**Determinism strategy (canonical approach — use this, not alternatives):** Use a **counting sort / radix approach** rather than a hash table with collision chains:

1. For each particle, compute its cell key: `key = hash(floor(x/h), floor(y/h), floor(z/h))`.
2. Allocate a flat array `sorted_indices` of size n_particles and a `cell_start` / `cell_end` array of size n_buckets.
3. First pass: count particles per bucket.
4. Prefix sum to get `cell_start[b]` for each bucket.
5. Second pass: place each particle index into `sorted_indices` at the appropriate offset *in particle-index order* (i.e., for two particles with the same key, the one with the smaller index goes first).
6. For queries, iterate `sorted_indices[cell_start[b] .. cell_end[b]]` for each of the 27 neighbor cells.

This is fully deterministic because ties are broken by particle index, not by insertion timing. It uses flat arrays with no pointer-chasing, making it cache-friendly. The counting sort is O(n + buckets) and typically faster than hash-table approaches for the particle counts in this project (5k–50k).

Table size: use the next prime above 2 × n_particles. Rehash only when particle count changes significantly.

#### `sim/sph/Kernels.h`

SPH kernel functions. These are the "weighting functions" that determine how much influence one particle has on another based on distance.

**Poly6 kernel** (for density computation):
```
W_poly6(r, h) = (315 / (64π h⁹)) * (h² − r²)³   for 0 ≤ r ≤ h
              = 0                                    for r > h
```
This is smooth and non-negative. Used to sum neighbor masses into a density estimate.

**Spiky gradient kernel** (for pressure forces):
```
∇W_spiky(r, h) = −(45 / (π h⁶)) * (h − r)² * (r_vec / r)   for 0 < r ≤ h
```
This has a sharp peak at r=0, which prevents particle clumping (the gradient is strongest when particles are very close).

**Viscosity Laplacian kernel** (for viscous forces):
```
∇²W_viscosity(r, h) = (45 / (π h⁶)) * (h − r)   for 0 ≤ r ≤ h
```
This is always positive, ensuring viscosity always dampens relative motion (never amplifies it).

All kernels must be implemented as **inline functions** for performance (they are called millions of times per frame).

#### `sim/sph/FluidStep.h`

The fluid simulation step, implementing Position-Based Fluids (PBF).

**The PBF algorithm:**

1. **Predict positions**: For each particle, apply external forces and compute predicted positions.
2. **Neighbor search**: Build the neighbor grid from predicted positions.
3. **Iterate** (fixed number of iterations):
   a. Compute density at each particle: `ρ_i = Σ_j m_j * W_poly6(|x_i − x_j|, h)`.
   b. Compute constraint value: `C_i = ρ_i / ρ₀ − 1` (zero when density equals rest density).
   c. Compute correction: `Δx_i = (1/ρ₀) * Σ_j (λ_i + λ_j) * ∇W_spiky(x_i − x_j, h)`, where λ values are computed from the constraint gradient.
   d. Apply corrections to predicted positions.
4. **Update velocities**: `v_i = (x_predicted − x_old) / Δt`.
5. **Apply viscosity**: `v_i += ε * Σ_j (v_j − v_i) * W_poly6(|x_i − x_j|, h)` (XSPH viscosity for smooth motion).

**Expose residual metrics:**
- Average density error: `(1/n) * Σ |ρ_i − ρ₀|`
- Max density error: `max_i |ρ_i − ρ₀|`

---

### 5.6 `sim/coupling/` — Membrane–Fluid Coupling

#### `sim/coupling/NoLeak.h`

The boundary coupling between fluid and membrane.

**Public API:**
```cpp
struct NoLeakResult {
    int penetration_count;
    float max_penetration_depth;
};

NoLeakResult applyNoLeak(
    Particles& particles,
    MembraneMesh& membrane,
    const CouplingParams& params
);
```

**Implementation (step by step):**

1. **Inside/outside test**: Use the signed distance field (SDF) described in section 2.3.4. Each particle looks up its signed distance from the precomputed voxel grid. Positive = outside (leaked). This is the default approach; do not use per-particle ray casting.

2. For each particle found to be outside:

   a. **Find the closest point on the membrane surface.** This requires a robust closest-point-on-triangle algorithm. For each candidate triangle (found via the SDF grid or a BVH), compute the closest point using the Voronoi region method:

   Given triangle with vertices A, B, C and query point P:
   - Compute edge vectors: `ab = B − A`, `ac = C − A`, `ap = P − A`.
   - Compute dot products: `d1 = dot(ab, ap)`, `d2 = dot(ac, ap)`.
   - If `d1 ≤ 0` and `d2 ≤ 0`, closest point is A (vertex region).
   - Continue with similar tests for vertex B, vertex C, and the three edge regions (AB, BC, CA).
   - If none of the vertex/edge regions match, P projects onto the interior of the triangle: `closest = A + u*ab + v*ac` where u, v are the barycentric coordinates from `dot(ab,ab)*v + dot(ab,ac)*u = d1` and `dot(ab,ac)*v + dot(ac,ac)*u = d2`.

   This method is numerically stable and handles degenerate (very thin) triangles gracefully — the vertex/edge tests catch them before the face projection divides by a near-zero determinant. Use the implementation from Ericson, *Real-Time Collision Detection*, section 5.1.5, which is the standard reference.

   b. Compute the penetration depth: `depth = |particle_position − closest_surface_point|`.

   c. Project the particle back: `particle_position = closest_surface_point − 0.1 * h * triangle_normal` (push inside by a fraction of the smoothing length, not just an epsilon — too small an offset and the particle will leak again next step).

   d. Compute reaction impulse: `impulse = particle_mass * (old_position − new_position) / Δt`. Distribute to the three vertices of the closest triangle using barycentric weights of the closest point. Apply the coupling damping factor (0.8) before adding to vertex velocities.

3. Return the penetration count and maximum penetration depth for diagnostics.

**Performance note**: For the MVP single cell (1280 triangles, 5000 particles), the SDF lookup is the fast path and brute-force triangle testing as a fallback for escaped particles is acceptable. For larger meshes or multi-cell scenarios, build a BVH (bounding volume hierarchy) over the triangles for the closest-point queries. AABB trees are simple to implement and give O(log n) per query.

---

### 5.7 `render/` — OGRE Adapters (Visualization Only)

#### `render/OgreBridge.h`

A small set of helper functions for converting between simulation types and OGRE types.

```cpp
inline Ogre::Vector3 toOgre(const Vec3& v) {
    return Ogre::Vector3(v.x, v.y, v.z);
}
```

This file is the *only* place where both `Vec3` and `Ogre::Vector3` are visible in the same scope. It prevents OGRE includes from leaking into simulation code.

#### `render/MembraneRenderable.h` and `render/MembraneRenderable.cpp`

Wraps an OGRE `ManualObject` and its `SceneNode`.

**Public API:**
```cpp
class MembraneRenderable {
public:
    void create(Ogre::SceneManager* mgr, const std::string& materialName);
    void update(std::span<const Vec3> positions,
                std::span<const Vec3> normals,
                std::span<const uint32_t> indices);
};
```

**Implementation of `update()`:**
1. `manualObject->clear();`
2. `manualObject->begin(materialName, Ogre::RenderOperation::OT_TRIANGLE_LIST);`
3. For each vertex: `manualObject->position(toOgre(positions[i])); manualObject->normal(toOgre(normals[i]));`
4. For each triangle index: `manualObject->index(indices[k]);`
5. `manualObject->end();`

This rebuilds the entire mesh every frame. It is simple and correct. For a single cell (~5000 triangles), the cost is negligible compared to the simulation step.

#### `render/ParticlesRenderable.h` and `render/ParticlesRenderable.cpp`

Wraps an OGRE `BillboardSet` and its `SceneNode`.

**Public API:**
```cpp
class ParticlesRenderable {
public:
    void create(Ogre::SceneManager* mgr, const std::string& materialName, float billboardSize);
    void update(std::span<const Vec3> positions);
};
```

**Implementation of `update()`:**
1. Ensure the billboard count matches the position count: create or remove billboards as needed.
2. For each billboard, set its position: `billboardSet->getBillboard(i)->setPosition(toOgre(positions[i]));`

---

### 5.8 `app/` — OGRE Application Entry Point

#### `app/CellApp.h` and `app/CellApp.cpp`

The main application class.

**Inheritance:**
```cpp
class CellApp : public OgreBites::ApplicationContext,
                public OgreBites::InputListener {
    // ...
};
```

**`setup()` implementation:**
1. Call base class setup (initializes OGRE root, loads resources).
2. Get the root scene manager: `auto* smgr = getRoot()->createSceneManager();`
3. Create camera, set position and look-at for a good initial view of the cell.
4. Create a viewport from the camera.
5. Create ambient and directional lights.
6. Create the simulation: construct a `CellSim` with an icosphere mesh and initial particle positions.
7. Create renderables: `MembraneRenderable` and `ParticlesRenderable`, attached to scene nodes.
8. Set up debug UI: TrayManager sliders for timestep, iterations, stiffness, particle size; toggles for wireframe, normals, pause, step.

**Per-frame update (in a frame listener callback):**
1. If not paused: `sim.step(params.dt);`
2. Get the sim view: `auto view = sim.view();`
3. Update renderables: `membraneRenderable.update(view.membrane_positions, view.membrane_normals, view.membrane_indices);`
4. `particlesRenderable.update(view.particle_positions);`
5. **Update diagnostic display** using the **pull model**: after each step, CellApp calls `sim.diagnostics()` which returns a `Diagnostics` struct (computed during the step and stored in CellSim). CellApp then pushes the values into OgreBites TrayManager labels:

```cpp
auto diag = sim.diagnostics();
volumeLabel->setCaption("Vol: " + std::to_string(diag.current_volume)
    + " / " + std::to_string(params.target_volume)
    + " (" + std::to_string(diag.volume_drift_pct) + "%)");
areaLabel->setCaption("Area: " + std::to_string(diag.current_area));
residualLabel->setCaption("Stretch res: " + std::to_string(diag.stretch_residual)
    + " | Vol res: " + std::to_string(diag.volume_residual));
leakLabel->setCaption("Leaks: " + std::to_string(diag.penetration_count)
    + " | Max depth: " + std::to_string(diag.max_penetration_depth));
fpsLabel->setCaption("Step: " + std::to_string(diag.step_time_ms) + " ms");
```

The `Diagnostics` struct contains at minimum: `current_volume`, `current_area`, `volume_drift_pct`, `area_drift_pct`, `stretch_residual`, `volume_residual`, `area_residual`, `bend_residual`, `fluid_density_error_avg`, `fluid_density_error_max`, `penetration_count`, `max_penetration_depth`, `step_time_ms`, `neighbor_build_time_ms`, `constraint_solve_time_ms`. The simulation computes all of these during the step (they are byproducts of the constraint projections, not extra work) and stores them in a member field. CellApp pulls them each frame. The simulation never pushes to the UI — it does not know the UI exists.

**Input handling:**
- Space: toggle pause.
- Right arrow: advance one step (when paused).
- W: toggle wireframe.
- N: toggle normal visualization.
- P: toggle particle visibility.
- Mouse: orbit camera (handled by OgreBites camera helper).

#### `app/main.cpp`

Minimal entry point:
```cpp
int main(int argc, char* argv[]) {
    CellApp app;
    app.initApp();
    app.getRoot()->startRendering();
    app.closeApp();
    return 0;
}
```

---

## Part V-B — Testing Story

### Unit Tests (run on every build)

The following tests must exist and pass before any milestone is considered complete:

**Geometry tests (`test/test_geometry.cpp`):**
- `computeVolume` on a known mesh (e.g., a regular tetrahedron with analytically known volume) returns the correct value within 1e-6 relative error.
- `computeArea` on the same mesh returns the correct value within 1e-6 relative error.
- `computeNormals` on a sphere mesh produces normals that all point outward (dot product with position vector > 0 for every vertex).
- `generateIcosphere` at subdivision level 0 produces exactly 12 vertices, 20 faces, 30 edges. At level 3: 642 vertices, 1280 faces, 1920 edges.

**Neighbor grid tests (`test/test_neighbor_grid.cpp`):**
- Place 100 particles at known positions. For each particle, verify that `forNeighbors` returns exactly the set of particles within distance h (brute-force comparison).
- Run `build` twice with the same particle positions in different insertion order. Verify identical neighbor lists (determinism test).
- Place a particle at the boundary of a grid cell. Verify it finds neighbors in adjacent cells (boundary handling test).

**Constraint tests (`test/test_constraints.cpp`):**
- Apply `projectStretch` to two vertices with a stretched edge. Verify the edge length moves toward rest length and the residual decreases.
- Apply `projectVolume` to a mesh with inflated volume. Verify volume moves toward target and the correction conserves center of mass (sum of vertex displacements weighted by mass equals zero).
- Apply `projectVolume` 100 times in a loop to a mesh 10% above target volume. Verify volume converges to within 0.1% of target.

**SPH kernel tests (`test/test_kernels.cpp`):**
- Verify Poly6 integrates to 1.0 (numerical integration over a sphere of radius h with uniform particle spacing).
- Verify Spiky gradient is zero at r = h and maximal near r = 0.
- Verify Viscosity Laplacian is always positive for 0 ≤ r ≤ h.

### Regression Tests (run nightly or before merges)

**Determinism regression (`test/test_determinism.cpp`):**
- Run 100 steps of the full simulation from a fixed initial state with fixed parameters. Record the final state (all positions, all velocities). Run again. Verify bitwise identical output. This catches: uninitialized memory, non-deterministic hash iteration, thread scheduling dependencies, and accidental use of random numbers.

**Stability regression (`test/test_stability.cpp`):**
- Run 10,000 steps from the default initial state. At each step, record volume, area, penetration count, and total kinetic energy.
- Assert: volume stays within 2% of V₀ for all 10,000 steps.
- Assert: area stays within 3% of A₀.
- Assert: penetration count is zero for at least 99% of steps.
- Assert: kinetic energy does not grow monotonically (the system is not injecting energy).
- If any assertion fails, the test prints the step number and the violating metric, making it possible to diagnose the regression.

### Acceptance Thresholds (encoded in the test suite)

These are the quantitative definitions of "working" for each milestone:

| Metric | Threshold | Where checked |
|---|---|---|
| Volume drift after 10k steps | < 2% of V₀ | `test_stability` |
| Area drift after 10k steps | < 3% of A₀ | `test_stability` |
| Penetration count per step (avg) | < 0.5% of particles | `test_stability` |
| Kinetic energy trend (linear fit slope) | ≤ 0 (not growing) | `test_stability` |
| Determinism (bitwise) | Exact match over 100 steps | `test_determinism` |
| Neighbor grid correctness | 100% match with brute force | `test_neighbor_grid` |

---

## Part VI — Implementation Milestones

### Milestone 1: Membrane-Only "Living Blob"

**Goal**: A deformable blob that preserves its volume and jiggles like a cell when poked.

**What to implement:**
- Icosphere mesh generation (2–3 subdivisions).
- Edge extraction with rest lengths.
- Stretch constraints (edge length preservation).
- Volume constraint (global volume preservation).
- OGRE membrane rendering via ManualObject.
- Basic camera, lights, and debug UI.

**What you should see**: A spherical mesh that resists compression and expansion. If you apply a force to one side, it deforms, then slowly returns toward a sphere. It should feel "squishy" — like poking a water balloon.

**What to test**: Volume should remain within 1–2% of the target after any deformation. Stretch residuals should decrease over solver iterations. The mesh should not invert (triangles should not flip inside out).

### Milestone 2: Cytoplasm Particles (Visualization Only)

**Goal**: Particles rendered inside the cell, without any fluid physics yet.

**What to implement:**
- Particle initialization (grid sampling inside the icosphere, discarding particles outside).
- OGRE BillboardSet rendering for particles.
- Billboard size slider in the debug UI.

**What you should see**: A cloud of small dots or spheres filling the inside of the membrane mesh. They don't move yet — this milestone validates the rendering pipeline.

**What to test**: All particles should visually appear inside the membrane. The billboard count should match the particle count. Billboard size should respond to the UI slider.

### Milestone 3: SPH + PBF Internal Flow

**Goal**: The cytoplasm moves as a viscous fluid inside the cell.

**What to implement:**
- Uniform grid neighbor search.
- SPH density and pressure computation.
- PBF incompressibility correction loop.
- Viscosity (XSPH).
- No-leak coupling (detect and project escaped particles, apply reaction to membrane).

**What you should see**: When the membrane deforms, the internal fluid flows in response. You should see vortex-like patterns and viscous sloshing. The fluid should not leak through the membrane. The membrane should be pushed outward by internal pressure.

**What to test**: Average density should stay within 5% of rest density. Penetration count should be near zero. No particles should be visible outside the membrane.

### Milestone 4: Stability, Diagnostics, and Optimization Readiness

**Goal**: The simulation is stable, instrumented, and ready for controlled experiments.

**What to implement:**
- Full XPBD compliance parameters for all constraints.
- Area constraint and bending constraint.
- Complete diagnostic logging (all residuals, drift metrics, timings).
- Performance profiling (per-phase timings displayed in UI).
- Parameter presets (soft cell, stiff cell, viscous cytoplasm, watery cytoplasm).

**What you should see**: A cell that maintains its shape, has smooth internal flow, responds to perturbations, and recovers. The diagnostic readouts should show stable, bounded values.

**What to test**: Run for 10,000 steps and verify that volume drift stays under 1%, area drift stays under 2%, no constraint residuals are growing, and no particles are leaking. This is your "physics is working" acceptance test.

---

## Part VII — What Drives Cell Motion: From Passive Physics to Active Behavior

Up to this point the simulator describes a single physical cell. For the **brain-like digit learner** you specified a stronger requirement: **literal spatial migration** (cells move in 2D/3D, connectivity depends on distance and contact). This turns the project into *developmental morphogenesis + learning*.

This part therefore has two layers:

1. How a **single cell** moves (useful for validating actuation, energy use, and homeostasis).
2. How a **population of migrating cells** self-organizes into tissue-like structures and wiring.

### 7.1 Single-Cell Motion (still useful as a primitive)

The single-cell motion models remain valid and serve as “unit tests” for actuation:

- Active matter polarity (persistent random walk, chemotaxis gradients).
- Protrusion/contraction asymmetry (cortex mechanics).
- Adhesion/friction to a substrate (crawling).
- Homeostasis-first control (motion as a byproduct of maintaining viable internal ranges).
- Explicit RL (only after viability penalties exist; otherwise policy exploits simulator artifacts).

These are the right tools when you want to validate that your membrane + cytoplasm system can be actuated in a controlled way and that energy/homeostasis can regulate behavior.

### 7.2 Migrating-Cell Tissue Motion (the core of the “brain-like” plan)

A brain-like system is not a static graph. It develops. Cells migrate, segregate into regions, wire locally, and only then behave like a circuit. In your abstraction, each cell is a real-time stateful agent with:

- a fast bioelectric state (excitability)
- a chemical footprint (multi-channel modulators and identity)
- a slow structural program (growth, adhesion, synaptogenesis propensity)
- an energy budget and homeostatic constraints
- the ability to move, form contacts, and grow/prune connections

#### 7.2.1 Space and regions

Start with a 2D sheet (cortical sheet analogue). Define spatial “proto-areas” as fields rather than hard-coded layers:

- a region field `G_area(x)` that biases cell differentiation (retina/V1/association/decision-like zones)
- guidance gradients `G_grad(x)` that bias migration direction and wiring directionality
- optional barriers/boundaries to encourage compartmentalization

These are not labels. They are developmental constraints.

#### 7.2.2 Migration dynamics

Each cell i has position `x_i`, polarity `p_i`, and velocity `v_i`.

At each step, compute forces:

- short-range repulsion from overcrowding (maintain tissue density)
- adhesion to compatible neighbors (cell-type dependent)
- guidance from `G_grad` and `G_area`
- optional electrical guidance (follow bioelectric gradients)

Update:

- `p_i ← normalize((1−β)p_i + β*(∇G_grad(x_i) + κ∇M_att(x_i) + ξ_i))`
- `v_i ← v_max(s_i) * p_i + F_repulsion + F_adhesion + F_guidance`
- `x_i ← x_i + dt * v_i`

Noise `ξ_i` is the exploration source and should be modulated by energy: low energy increases variability.

#### 7.2.3 Synaptogenesis as contact-dependent structural plasticity

Connectivity is not fixed. Synapses form when cells remain in proximity and are compatible.

Maintain a contact timer `τ_ij` for neighbor pairs within radius `r_contact`.

Form an edge (j→i) when `τ_ij` exceeds a threshold and a compatibility score is high:

- `P_form = σ(u0 + u1*compat(s_i, s_j) + u2*a_j + u3*a_i − u4*ρ_i)`
- if sampled true for K consecutive steps, create synapse with small initial weight

Prune when:
- distance exceeds `r_prune` for T steps, or
- synapse stays weak/unutilized (low eligibility / low co-activity)

This is the wiring mechanism that replaces “designing a network.”

#### 7.2.4 Homeostasis, energy, and stability

Migration and wiring are constrained by viability:

- each cell maintains `E_i` energy and a homeostasis penalty `H_i`
- movement and plasticity consume energy
- high `H_i` suppresses further plasticity and can bias migration away from stressful regions

This makes development self-stabilizing rather than runaway.

### 7.3 How “teacher guidance” enters in a migrating-tissue system

External teaching is modeled as **localized fields**:

- `M_att(x,t)` attention / plasticity gate (where learning is allowed)
- `M_teach(x,t)` teaching clamp (which goal attractor is active)
- `M_novel(x,t)` novelty/arousal (when to learn)

Early in development these fields are driven externally (teacher). Later they are produced internally by learned predictive circuits (internal teacher).

This avoids broadcasting output errors to all cells.

## Part VIII — From Cells to Circuits: Developmental Morphogenesis Instead of Fixed Graphs

### 8.1 The scaling shift caused by literal migration

If cells **literally migrate** and **wire by contact**, you are no longer building “a neural network made of modules.” You are building a **developing tissue** whose graph is an emergent property of:

- migration
- adhesion and segregation into regions
- synaptogenesis and pruning
- homeostasis and energy constraints

This is closer to “developmental neuroscience in silico” than to architecture design.

The single-cell membrane+SPH simulator remains valuable as a physical primitive, but the digit-learner uses a **reduced migrating-cell tissue simulator** (2D sheet first) where each cell is a stateful agent and the graph changes over time.

### 8.2 The migrating-cell abstraction (what a “cell” is in the circuit)

Each cell i exposes a real-time footprint:

- Position `x_i ∈ ℝ²` (or ℝ³ later), polarity `p_i`, velocity `v_i`
- Fast bioelectric state `V_i` and activity `a_i = σ((V_i − θ_i)/T)`
- Chemical footprint vector `c_i ∈ ℝ^K` (plasticity gate sensitivity, neuromodulator receptivity, identity channels)
- Structural program `s_i ∈ ℝ^M` (slow; controls adhesion, growth, excitability drift, wiring propensity)
- Energy `E_i` and homeostasis penalty `H_i`
- Synapses `w_ij` with eligibility traces `e_ij` and maturity `g_ij`

This is the computational unit. No internal ion channels or gene networks are simulated; those are folded into (V, c, s, E, H).

### 8.3 Spatial fields (how top-down and bottom-up are represented)

Instead of hard layers, represent developmental and teaching influences as fields sampled at `x_i`:

- Sensory field `I(x,t)` (MNIST mapped onto a “retina zone” and optionally revealed over time)
- Region identity field `G_area(x)` and guidance gradients `G_grad(x)` (development scaffold)
- Modulator fields:
  - `M_att(x,t)` attention/plasticity gate
  - `M_teach(x,t)` goal clamp (external teacher early; internal teacher later)
  - `M_novel(x,t)` novelty/arousal

Cells read these fields locally. No global error broadcast is needed.

---

## Part IX — The MNIST Pipeline as a Developing Tissue (Top-Down + Bottom-Up, No Backprop)

MNIST is a convenient benchmark, but learning letters/digits in brains is *active* (saccades, stroke dynamics, self-correction). To preserve that developmental logic, treat MNIST as an **active perception** task:

- map the 28×28 image into a sensory field in a “retina zone”
- reveal the image through a moving window (“saccade”) or progressive exposure so time matters

### 9.1 Tissue layout: proto-areas as spatial zones

On the 2D sheet, define four contiguous zones (soft boundaries via `G_area(x)`):

- Retina zone: mostly stable cells, strong sensory drive
- V1 zone: dense local wiring, feature formation
- Association zone: longer-range integration, concept assemblies
- Decision zone: 10 attractor assemblies (one per digit) with competition

Cells can migrate, but guidance and adhesion make them preferentially remain in or near their zone (developmental compartmentalization).

### 9.2 Bottom-up learning (representation formation)

Bottom-up learning builds stroke/edge features without labels:

- cells in retina and V1 learn via local competitive Hebbian rules + homeostasis
- synapses form and prune based on contact persistence and co-activity
- intrinsic excitability θ_i adapts to keep mean activity in range (homeostatic plasticity)
- energy limits activity and plasticity, preventing runaway

A simple, stable baseline learning rule for existing synapses j→i:

- `Δw_ij ∝ η * a_i * a_j − η * a_i^2 * w_ij`  (Oja-like)
- periodic synaptic scaling so Σ_j |w_ij| stays bounded

This makes early zones self-organize feature detectors under active perception.

### 9.3 Top-down teaching without output-error broadcast (external teacher → internal teacher)

This section replaces the earlier “global δ broadcast” plan.

#### 9.3.1 External teacher (early development)

The teacher provides only two interventions, both local:

1. A goal clamp in the decision zone: bias the correct digit attractor y via `M_teach(x,t)` localized to that zone.
2. An attention/plasticity gate `M_att(x,t)` that turns learning on where attention should be focused.

The teacher does **not** provide a numeric error signal to all synapses.

#### 9.3.2 Two-phase settling (free vs taught) as the supervision mechanism

For each training example (image x, label y):

- Free phase: present sensory field; let tissue dynamics settle for T steps → activities `a^-`
- Taught phase: same sensory field; apply localized goal clamp to correct attractor y for T steps → activities `a^+`

Then update synapses locally using only pre/post activity correlations:

- `Δw_ij ∝ η * (a_i^+ a_j^+ − a_i^- a_j^-) * gate_i`
- `gate_i = σ(k1*M_att(x_i) + k2*M_teach(x_i) − k3*H_i)`

This is local, biologically plausible, and avoids broadcasting “expected output differences.”

#### 9.3.3 Internal teacher (later development): learned top-down prediction

To internalize teaching, you add feedback/prediction pathways:

- cells in higher zones form feedback synapses into lower zones
- those feedback synapses are trained to predict lower activity

Define local prediction for a lower cell i:

- `â_i = Σ_k w_fb_{ik} a_k(high)`
- local mismatch: `ε_i = a_i − â_i`

Update feedback weights by local prediction error:

- `Δw_fb_{ik} ∝ η_fb * ε_i * a_k(high) * gate_i`

As feedback improves, higher-level digit assemblies can generate expectations about lower-level features. This is the internal teacher: “imagine the goal; compare to what is seen.”

#### 9.3.4 Self-correction loop (no external teacher)

Later training/fine-tuning proceeds with reduced external clamping:

- set a goal attractor internally (from instruction/context)
- use feedback predictions to generate expected lower-level activity patterns
- local mismatches ε_i drive attention (increase M_att where mismatch is high) and plasticity (when viable)

This implements “expected outcome vs actual outcome” without a global output error: mismatch is local everywhere.

### 9.4 Wiring and migration during learning (development never fully stops)

Because cells migrate and rewire, you need slow/fast timescales:

- fast: activity dynamics and synaptic weight updates
- medium: synapse formation/pruning based on contact persistence
- slow: migration and region segregation (guided by `G_area` / `G_grad`)

During training, migration is typically slower than activity by 10–100× so the tissue doesn’t churn too fast. Migration can be alternated in epochs (development step every N examples).

### 9.5 What “wins” on MNIST in this regime

Accuracy is not the only objective. The intended regime advantages are:

- damage recovery: remove a region of cells or sever connections and observe reorganization
- continual learning: train on subsets of digits sequentially and test forgetting
- robustness under distribution shifts: introduce changes in writing style distribution
- energy-aware computation: measure total activity/energy required to achieve competence

The key falsification angle remains: do these capabilities improve relative to parameter-matched point-neuron baselines?

---

## Part X — Connecting Everything: From 3D Cell Physics to Developmental Circuit Intelligence

### 10.1 The full stack (updated for migration)

This project now has **two** physically grounded levels:

1. **3D physical cell simulator** (Parts I–VI): membrane + cytoplasm physics and actuation. This validates how cell-level constraints, energy, and control can generate realistic behavior.
2. **Migrating-cell tissue simulator** (Parts VII–IX): many cells as stateful agents in space; wiring emerges by contact and compatibility; top-down teaching is a field/clamp; internal teacher emerges via learned feedback predictions.

The second level is the one that implements “brain-like digit learning.” The first level is the calibration/validation substrate for the cell abstraction.

### 10.2 Experimental program (updated)

Phase 1: build and stabilize the 3D physical cell simulator (as written).  
Phase 2: implement a migrating-cell tissue sheet (2D), with region fields, migration, synaptogenesis, and stable activity dynamics.  
Phase 3: run active-perception MNIST (saccade/reveal) with two-phase settling supervision (free vs taught clamp), and learn feedback predictors for internal teaching.  
Phase 4: compare against baselines on accuracy *and* regime metrics (repair, continual learning, distribution shift, energy).

---

## Summary (updated)

This guide covers:

- why the assumption to falsify is “point-neuron sufficiency”
- how a physically grounded 3D cell simulator is built and validated
- how literal spatial migration changes the AI layer into developmental morphogenesis + learning
- how digits/letters can be learned with top-down and bottom-up signals without backprop and without global output-error broadcasts
- how external teaching (localized clamps and plasticity gates) can be internalized as learned top-down prediction (internal teacher)

## Part X — Connecting Everything: From 3D Physics to Circuit Intelligence

### 10.1 The Full Stack

The complete system has three levels:

1. **3D Cell Simulator** (this guide, Parts I–VI): The physics ground truth. A single cell with membrane, fluid, constraints, and active forces. Used to validate cell behavior and calibrate reduced models.

2. **CellModule Abstraction** (Part VIII): The fast surrogate. A compact dynamical system that preserves multi-timescale state, homeostasis, and energy dynamics without physics overhead. Used as the building block for circuits.

3. **Cell-Inhabited Circuit** (Part IX): The intelligence layer. Populations of CellModules connected by locally-learned weights, organized into a processing pipeline, solving tasks (MNIST as the first benchmark).

Each level informs the next: the 3D simulator grounds the CellModule; the CellModule enables the circuit; the circuit tests the hypothesis.

### 10.2 The Experimental Program

The falsification program has four phases:

**Phase 1 — Physical cell** (Milestones 1–4 in Part VI): Build and stabilize the 3D simulator. Validate that constraints, fluid, and coupling work. This is the engineering foundation.

**Phase 2 — Active cell** (Part VII): Add polarity, active forces, adhesion, and motion. Validate that the cell can crawl, explore, and respond to signals. Calibrate CellModule parameters against the full simulation.

**Phase 3 — Cell circuit on MNIST** (Part IX): Build the four-layer circuit. Implement three-factor learning. Train on MNIST. Measure accuracy, damage recovery, continual learning, and energy efficiency.

**Phase 4 — Controlled comparison** (Section 1.7): Run parameter-matched neural baselines (MLP, RNN, small transformer) on the same tasks with the same evaluation protocol. Report results as a falsification test: do cell-like modules provide a regime advantage, or don't they?

If Phase 4 shows a consistent advantage for cell-like modules on damage recovery and continual adaptation, the project has produced evidence against the sufficiency of point-neuron abstractions. If not, the hypothesis is falsified in this regime, and the failure mode tells you what to try next.

---

## Summary

This guide has covered:

- **Why**: The assumption that point-neuron abstractions are sufficient for robust, adaptive intelligence may be the most important assumption to falsify in AI. The stronger inversion treats cells (not neurons) as the fundamental computational unit.

- **What**: A real-time 3D cell simulator with a deformable membrane (triangulated shell with XPBD constraints), internal viscous fluid (SPH particles with PBF), fluid–membrane coupling (no-leak boundary projection), and OGRE-based visualization.

- **How it's organized**: Strict separation between simulation (C++, no rendering dependencies) and visualization (OGRE classic + OgreBites). Everything important is an explicit function with a clear signature. All parameters live in one struct. The step function is deterministic. Diagnostics are always on.

- **How to build it**: Incrementally, in four milestones — membrane blob, particle visualization, SPH fluid, and stability/diagnostics — each building on the last and each independently testable.

- **What drives cell motion**: Five families of approaches (active matter, cortex mechanics, CPM-style energy minimization, homeostasis-first control, explicit RL), with concrete implementation recipes for the simulator and a recommended build order.

- **How to scale to circuits**: The CellModule abstraction — a reduced dynamical model calibrated against the full 3D simulator — enables cell-like computation at circuit scale without physics overhead.

- **How to test the hypothesis**: An MNIST pipeline using cell-inhabited circuits with three-factor local learning, eligibility traces, homeostatic constraints, and energy-aware computation. Compared against parameter-matched neural baselines on accuracy, damage recovery, continual learning, and energy efficiency.

- **The falsification target**: If cell-like modules consistently outperform point-neuron baselines on robustness and adaptation tasks, the field's fundamental abstraction is insufficient. If they don't, the hypothesis is cleanly falsified in this regime, and the failure points to what to try next. Either outcome advances the science.