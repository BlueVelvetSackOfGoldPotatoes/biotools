# CellEngine Deep Dive

Date: 2026-03-10

## 1. Executive summary

This benchmark has **three different kinds of learning/adaptation** happening at once, and it is easy to confuse them:

1. **Evolutionary learning across genomes**
   The cell controller itself is not trained with gradient descent. A genetic algorithm searches over 96-float genomes. Each genome defines how a tissue of cells develops and how those cells behave.

2. **Within-lifetime cellular adaptation**
   Once a genome has instantiated a tissue, the cells still change state during and across trials. Gap-junction couplings adapt locally, sensitivity adapts homeostatically, energy is consumed and replenished, and gene-expression values drift slowly.

3. **Gradient-based RL for the comparator baselines**
   The DQN and A2C baselines are separate controllers trained directly on the active task state. They are not cell-based. They exist to provide a control baseline and a damage/recovery baseline.

The most important practical consequence is this:

- `full` mode searches for a good **genome**.
- `replay` mode does **not** evolve anything. It loads one saved genome and runs a single traced episode.
- Replay no longer retrains RL on the fly in the UI path; it reuses cached comparator artifacts when available.

## 1.1 Implemented vs planned matrix

| Area | Current state |
| --- | --- |
| Shared 96-float genome and developmental phenotype map | implemented |
| `fixed2d`, `grown2d`, `grown3d` morphologies | implemented |
| Task family: cartpole / mass-spring / worm / pong | implemented |
| Replay export for body / cells / edges / RL traces | implemented |
| ODD telemetry: current / stress / mechanics / gap decomposition | implemented |
| ODD interventions and counterfactual ranking | partial |
| Long-range neurite / axon-like wiring | planned |
| Rich GRN, migration, division during runtime | planned |
| Fully GPU-resident batched rollout | planned |

## 2. What is actually implemented vs. what the spec describes

The file `models/cellspec.md` describes a broad research program. The current implementation in `models/cells/cellengine/cellengine.cpp` is a **working simplified version**, not a literal implementation of the full long-form spec.

What is implemented now:

- One shared 96-float genome per organism.
- Development as a mapping from geometric features to 8 continuous gene-expression channels.
- Explicit developmental growth from seed voxels into a connected body.
- Optional legacy `fixed2d`, `grown2d`, and `grown3d` body modes.
- 3D voxel adjacency and replay export with `x/y/z` coordinates.
- Variable-size tissues up to the configured `CELLENGINE_MAX_CELLS` budget, including hundred-cell regimes.
- A fast electrophysiology/mechanics update with local electrical coupling.
- A CUDA fast-cell-tick execution path with runtime NVRTC compilation and CPU fallback.
- A true lattice chemical field with spatial diffusion and decay.
- A slower gene-expression drift update.
- Teacher-assisted evaluation trials.
- Genetic search over genomes.
- A task-general environment interface with `cartpole_balance`, `mass_spring_balance`, `worm_drag_race`, and `pong_return`.
- RL comparators: Double-DQN and a lightweight actor-critic called `A2C` in the code.
- Replay export for body state, per-cell state, per-edge coupling state, and RL traces.

What is still not implemented from the ambitious/full spec:

- No full cell migration or runtime remeshing during the task.
- No detailed gene regulatory network beyond the 8-channel slow phenotype vector.

So the correct framing is:

- this is a serious **operational prototype** of the cell-first benchmark,
- not the final full biological simulator described in the long spec.

## 3. Files that matter

Core implementation:

- `models/cells/cellengine/cellengine.cpp`
- `models/cells/cellengine/cellengine.h`

Spec and intent:

- `models/cellspec.md`

Shared RL baseline implementation:

- `models/reinforcement/src/rl_models.h`
- `models/reinforcement/src/rl_models.cpp`

Current solved benchmark artifacts:

- `reports/cellengine_latest/summary.json`
- `reports/cellengine_latest/report.md`
- `reports/cellengine_latest/champion_genome.csv`

UI replay viewer:

- `ui/src/components/CellEngineReplayPanel.jsx`
- `ui/api/server.mjs`

## 4. The benchmark in one sentence

The system evolves a shared genome that grows a connected tissue, differentiates it into a continuous phenotype field, and uses that tissue to control a physical task by converting stress into cell activity and contractility back into force.

## 5. The body that the genome controls

There are now three body modes:

- `fixed2d`: the original hard-coded 29-cell `T` template
- `grown2d`: genome-driven development on a 2D lattice
- `grown3d`: genome-driven development on a 3D voxel lattice

In the growth modes, development starts from anchored seed voxels and expands through explicit cell-addition steps before evaluation begins. Candidate frontier voxels are scored from:

- geometric position
- local occupancy density
- support from lower voxels
- bridge/branch structure
- provisional gene-expression values induced at that location

This gives real evolvable morphology while keeping the genome size fixed at 96.

Three structural tags are derived from geometry:

- `ground`: all cells with `y == 0`
- `hinge`: cells with `|x| <= 1` and `y <= 2`
- `motor`: ground cells with `|x| >= 2`

Five geometric features are computed per cell:

- normalized `x`
- normalized `y`
- proximity to the hinge region
- surface proximity
- neighbor density

These features are the developmental input used to assign each cell its gene-expression program.

The body also precomputes:

- `force_direction_x`: how a cell's contractile output projects into left/right actuation
- `mech_advantage`: how effectively that cell's force reaches the ground/support

This is how the tissue becomes physically asymmetric without hand-coded cell types.

## 6. The genome

### 6.1 Size and role

The genome is a flat vector of **96 floats**.

Every cell shares the same genome.

Cells differ because each cell sits at a different body position, sees different geometry features, receives different stress, and therefore ends up with a different continuous gene-expression vector `G`.

### 6.2 Developmental part of the genome

The first 48 genome values are the development map:

- `8 x 5` weights `W[g][f]`
- `8` biases

Each cell gets an 8-dimensional gene-expression vector:

```text
G_i[g] = sigmoid(sum_f W[g][f] * feature_i[f] + bias[g])
```

So the genome does not say:

- "cell 7 is a sensor"
- "cell 12 is a motor"

Instead it says:

- "cells with these geometric features should express these continuous tendencies"

That is much closer to developmental programming than to standard neural-network parameterization.

### 6.3 Runtime scalar parameters decoded from the genome

The rest of the genome decodes to global scalar hyperparameters for the tissue dynamics.

These include:

- FHN-like voltage dynamics parameters
- calcium dynamics parameters
- coupling and Hebbian plasticity rates
- mechanosensitivity and chemical coefficients
- energy replenishment and energy costs
- slow gene-drift rate
- teacher fraction and teacher gains/scales
- stiffness, contractility, activity decay
- force smoothing and stress smoothing

### 6.4 Honest note on gene semantics

The intended semantics are roughly:

- `G[0]`: excitability
- `G[1]`: contractility
- `G[2]`: mechanosensitivity
- `G[3]`: secretion magnitude
- `G[4]`: secretion sign/flavor
- `G[5]`: adhesion class
- `G[6]`: coupling-related program
- `G[7]`: adaptation-rate program

But the actual implementation is not perfectly symmetric about all eight channels.

Important implementation details:

- `G[5]` directly affects adhesion-style initialization of coupling.
- `G[7]` directly affects Hebbian adaptation speed.
- `G[6]` is **not** directly read in the runtime equations. It still affects phenotype indirectly because the initial coupling baseline depends on average phenotype difference across all `G` dimensions, but there is no explicit `g6` multiplier in the runtime update.

This matters because the UI labels `gene_expr_6` as coupling-related. That is directionally reasonable, but not literally a one-line runtime control knob in the current code.

## 7. How a genome becomes a tissue

For a given genome:

1. Decode the 96 floats into development weights plus dynamic constants.
2. Build the selected body template (`fixed2d`, `grown2d`, or `grown3d`).
3. Compute each cell's 8-dimensional `G` from its geometry.
4. Initialize fast/medium/slow state arrays:
   - membrane potential `V`
   - recovery variable `Wrec`
   - calcium `Ca`
   - sensitivity `sigma`
   - energy
   - mechanical activation `a_mech`
   - chemical output `chem_out`
   - stress
   - activity average
   - fatigue
   - gap-junction strengths
5. Initialize gap couplings between neighbors.

Initial gap coupling is based on:

- a genome-level `coupling_base`
- average phenotype difference across all 8 gene-expression components
- explicit adhesion-like similarity using `G[5]`

So initial connectivity already reflects the developmental phenotype map.

## 8. The cell state variables

Each cell carries the following main dynamic state:

- `V`: fast electrical state, membrane-potential analogue
- `Wrec`: FHN-style recovery variable
- `Ca`: calcium-like activation state
- `sigma`: sensitivity/homeostatic gain
- `energy`: metabolic budget in `[0, 1]`
- `a_mech`: contractile output
- `chem_out`: local chemical output
- `stress`: externally written mechanical load
- `activity_avg`: smoothed activity level
- `fatigue`: slow low-energy trace
- `gap`: adaptive neighbor coupling strengths
- `G`: slow phenotype/gene-expression state

This is the core reason the model is not just a neural network with a biology skin. There are multiple interacting physical-ish and physiological state variables, not only an activation vector.

## 9. The physics loop

The benchmark family now supports multiple environments. The cartpole description below is the most mature worked example, not the only task.

The environment state is:

- cart position `x`
- cart velocity `x_dot`
- pole angle `theta`
- pole angular velocity `theta_dot`

The system integrates the standard cartpole equations with RK4 at `dt = 0.02` seconds.

Failure happens when:

- `|theta| > 15 degrees`

Important:

- **cart position is not terminal** in this benchmark implementation
- the task is basically "keep the pole up", not "keep the cart centered"

That is why you can see large `x` drift in successful runs.

## 10. How the environment talks to the cells

The flow is:

1. The active task state produces mechanical loads.
2. Those loads are projected into a tissue stress field.
3. Each cell reads its local stress as mechanosensory input.
4. The cells update their internal state.
5. The tissue produces a summed horizontal force.
6. That force is sent back into the cartpole ODE.

### 10.1 Stress projection

The code computes three main physical contributors:

- hinge moment
- lateral pole/cart force
- gravity load

Those are written into an external stress term and relaxed over the body graph for a few iterations.

The stress solve is not a full FEM solver. It is a compact graph-based relaxation that propagates load over neighbors, with effective stiffness determined partly by phenotype.

### 10.2 Force readout

The tissue's output force is:

```text
sum_i a_mech[i] * force_direction_x[i] * mech_advantage[i]
```

That summed force is then scaled and clipped by the environment force limit.

Only some cells are geometrically good at generating left/right action. The body shape matters.

## 11. What one cell update actually does

At each physics step, the code performs several cell substeps.

Default configuration:

- `physics_dt = 0.02`
- `cell_ticks_per_physics = 4`
- so each cell substep is `0.005`

For each active cell on each substep:

1. Compute electrical neighbor input `I_elec` from gap junctions.
2. Compute chemical neighbor input `I_chem` from neighbors' `chem_out`.
3. Compute mechanical input `I_mech` from local stress and mechanosensitivity.
4. Combine them into a total input current.
5. Update `V` and `Wrec` with an FHN-like system.
6. Update `Ca` from thresholded voltage.
7. Update `activity_avg` as a smoothed magnitude of voltage.
8. Update `sigma` toward homeostatic targets.
9. Compute `a_mech` from phenotype, calcium, stress, energy, and gating.
10. Compute `chem_out`.
11. Update energy by replenishment minus activity costs.
12. Update outgoing gap couplings by a Hebbian-like rule.

The effective current looks like:

```text
I_total = sigma * (I_elec + chem_coeff * I_chem + I_mech + chemical_balance * chem_out_self) * energy
```

The voltage dynamics are a FitzHugh-Nagumo style surrogate:

```text
dV = (V - V^3 / 3 - W + I_total) / tau_v
dW = eps * (V + a - bW)
```

Calcium rises when voltage exceeds a threshold.

Mechanical output depends on:

- contractility program `G[1]`
- mechanosensitive contribution
- calcium
- energy
- a gating term from voltage
- geometry bias near the surface/base

So the cell is not just firing or not firing. It is balancing electrical, chemical, energetic, and mechanical variables.

## 12. How adaptation happens inside the tissue

This is the most important conceptual part.

The cells are not static after development.

### 12.1 Fast adaptation

At the fast timescale, cells change:

- `V`
- `Wrec`
- `Ca`
- `sigma`
- `energy`
- `a_mech`
- `chem_out`
- `activity_avg`
- `gap`

### 12.2 Local structural adaptation

Gap couplings change with a local Hebbian-like rule.

In code, each directed gap value moves according to:

```text
gap += dt * (hebb_scale * corr_ij - decay_rate * (gap - gap_base))
```

where `corr_ij` is approximated from the product of the two cells' voltages.

Important properties:

- no global error signal
- no task loss inside the tissue
- no backprop through time
- only local co-activity and decay

This is why the model can be described as "learning-like" within lifetime without being standard ML training.

### 12.3 Homeostatic adaptation

`sigma` changes toward a target and also reacts to recent activity.

This gives the tissue a local sensitivity-control mechanism:

- overactive cells can damp down
- underactive cells can regain sensitivity

### 12.4 Slow phenotype drift

After each trial, `G` drifts slowly toward a target constructed from:

- current `G`
- the cell's own calcium
- average neighbor calcium

This is a crude but meaningful model of phenotype plasticity.

### 12.5 State persistence across trials

During evaluation, the organism is not fully reset between trials.

The task state resets.

But much of the tissue state carries over:

- gap structure
- energy trajectory
- fatigue
- slowly drifting phenotype

That is how teaching trials can shape later autonomous behavior.

## 13. Teaching

The benchmark includes a built-in teacher phase for the cell controller.

For some fraction of early evaluation trials, a PD teacher contributes an extra force:

```text
teacher_force = -(Kp * theta + Kd * theta_dot) * teacher_scale
```

This means early trials are easier.

But the tissue does not receive a supervised target. Instead it simply experiences the state trajectories and stress patterns of a more successful balancing episode.

During teacher-active trials, Hebbian plasticity is also boosted by `teacher_hebb_boost`.

Conceptually:

- the teacher does not tell the cells what action to take
- the teacher changes the environment trajectory
- the cells adapt locally while living through that trajectory

That is closer to guided developmental experience than to imitation learning.

## 14. Damage

### 14.1 Cell-controller damage

For the cell controller, damage means:

- randomly deactivate a fraction of cells
- zero their voltage/calcium/energy/output
- collapse their outgoing gap connections
- zero any connections that point into dead neighbors

This is a structural lesion to the tissue.

### 14.2 RL damage

For RL baselines, damage means something completely different:

- randomly zero a fraction of neural-network parameters

So the damage regimes are **not identical**.

They are analogous "capacity lesions", not the same physical perturbation.

That matters when interpreting the damage comparison.

## 15. Evaluation and scoring

### 15.1 Trial budgets

The current default budgets are:

- search: smaller/cheaper
- final clean: `100 trials x 500 ticks`
- damaged: `40 trials x 500 ticks`

The main solve threshold is:

```text
survival_ratio >= 0.80
```

on the full clean evaluation.

### 15.2 What the score means during evolution

The genetic search does not optimize only survival.

It optimizes a composite score:

```text
score = survival_ratio
      + 0.18 * sensor_responsiveness
      + 0.18 * signal_connectivity
      + 0.25 * force_correlation
      + 0.12 * normalized_adaptation_gain
      + 0.05 * mean_energy
      - 0.03 * normalized_force_cost
```

So evolution is encouraged to find genomes that:

- survive
- respond to perturbation
- propagate signals
- produce action in the correct direction
- adapt over the course of evaluation
- use energy reasonably
- avoid unnecessary force magnitude

The benchmark therefore searches for something broader than raw survival.

## 16. How evolutionary learning happens

The cell controller is learned with a genetic algorithm.

### 16.1 Population initialization

The population contains:

- random genomes
- a number of hand-seeded hint genomes

The hint genomes bias the search toward useful developmental programs, such as stronger mechanosensitivity near the hinge and stronger contractility near the outer ground cells.

### 16.2 Per-generation loop

For each generation:

1. Evaluate every genome on the search budget.
2. Rank genomes by the composite score.
3. Keep the top elites unchanged.
4. Fill the rest of the next population with tournament-selected crossover + mutation children.

Operators:

- tournament selection
- gene-wise crossover
- Gaussian mutation clipped to `[-1, 1]`

### 16.3 Local refinement

After the GA finishes, the best genome is locally refined by repeated mutation-only hill climbing for 36 steps.

That final local refinement is important. The final champion is not just the raw GA elite.

### 16.4 Auto-attempts

`full` mode can perform multiple increasingly aggressive search attempts.

Across attempts it increases things like:

- population size
- generation count
- search trials
- search ticks

This is why the system can keep iterating toward a solve automatically.

## 17. How RL learning happens

The RL baselines do not use cell state at all.

They observe a 4D state tensor:

- normalized cart position
- squashed cart velocity
- normalized pole angle
- squashed pole angular velocity

Actions are discrete:

- push left
- push right

Reward is extremely simple:

- `+1` each non-terminal step
- `-1` on the terminal step

So the RL baselines are solving a standard sparse-ish cartpole problem.

## 18. DQN in this codebase

The DQN comparator is a fairly standard feedforward Double-DQN.

Architecture:

- input 4
- hidden 64
- hidden 64
- output 2 Q-values

Training details:

- replay buffer capacity `20000`
- batch size `64`
- target network copied every `120` train steps
- `double_q = true`
- Adam optimizer
- epsilon-greedy exploration
- epsilon start `1.0`
- epsilon end `0.03`
- epsilon multiplicative decay `0.995`
- gamma `0.995`

Loss:

- mean squared TD error on the chosen action

Double-DQN behavior:

- action selection for the bootstrap target uses the online net
- action evaluation uses the target net

In short:

- this is a real Double-DQN baseline, not a toy tabular baseline
- but it is still a small MLP on the active task state

## 19. A2C in this codebase

The code calls the second RL baseline `A2C`, but the honest description is:

- a lightweight actor-critic with separate policy and value networks
- trained with full-episode Monte Carlo returns
- using a value baseline to form the advantage

Architecture:

- policy network: `4 -> 64 -> 64 -> 2 logits`
- value network: `4 -> 64 -> 64 -> 1`

Training details:

- policy sampled stochastically during training via softmax
- returns computed by backward discounted accumulation
- advantage = `return - value`
- advantages are standardized during the main training phase
- Adam optimizers for both policy and value networks
- gamma `0.99`

What it is **not** in the strict canonical sense:

- not vectorized multi-environment A2C
- no explicit GAE
- no entropy regularization term
- no clipped PPO-style update

So if you say "A2C" here, the best precise interpretation is:

- a compact on-policy actor-critic baseline implemented in the spirit of A2C
- not a full industrial-strength canonical A2C implementation

## 20. Damage and recovery for RL

After clean evaluation:

- DQN damage zeros a fraction of online-network parameters
- A2C damage zeros a fraction of both policy and value parameters

Then both baselines are re-evaluated in damaged form.

Then both are fine-tuned for a configured number of damage episodes and evaluated again as `recovered`.

So RL recovery means:

- lesion the controller
- continue gradient-based training for a short period
- measure how much performance returns

That is very different from the cell system, where recovery emerges from local state adaptation and tissue dynamics.

## 21. Replay mode and why the UI can feel slow

When you press `Run replay` in the React UI:

1. The UI sends a POST request to `/api/cellengine/replay`.
2. The Node API launches `bin/benchmark_cellengine` in replay mode.
3. The binary loads the selected genome.
4. It runs a **single traced episode** for the cell controller.
5. It writes:
   - `replay_trace.csv`
   - `replay_body.csv`
   - `replay_cells.csv`
   - `replay_edges.csv`
6. If RL is enabled, it also trains a fresh RL baseline and writes:
   - `rl_replay_trace.csv`

So the replay button is not only loading a static file. It can trigger actual backend computation.

That is why the startup delay is real.

The fastest way to reduce replay startup cost is:

- choose `RL comparator = disabled`

because then the backend skips fresh RL training for that replay request.

## 22. What the React UI is actually showing

The UI is built from the replay artifacts.

### 22.1 `replay_body.csv`

Static per-cell structure:

- cell coordinates
- hinge/ground/motor flags
- mechanical leverage
- force direction
- geometry features
- final gene-expression values `gene_expr_0..7`

### 22.2 `replay_cells.csv`

Per-tick, per-cell dynamic state:

- active flag
- voltage
- recovery
- calcium
- sigma
- energy
- mechanical output
- chemical output
- stress
- activity

### 22.3 `replay_edges.csv`

Per-tick pairwise coupling state:

- source and destination cell ids
- forward gap
- reverse gap
- average gap

### 22.4 `replay_trace.csv`

Per-tick organism/environment state:

- cart position
- pole angle
- velocities and accelerations
- organism force
- teacher force
- energy/stress/activity aggregates
- damage marker

The UI is therefore a visualization of recorded telemetry, not a live training environment.

## 23. Current solved result

The current saved solved result in `reports/cellengine_latest/summary.json` is:

### Cell controller

- clean survival ratio: `0.928960`
- clean mean ticks: `464.48 / 500`
- clean success rate: `91.0%`
- damaged survival ratio: `0.911150`
- damaged success rate: `85.0%`

### RL baselines

DQN:

- clean survival ratio: `0.924920`
- damaged survival ratio: `0.020700`
- recovered survival ratio: `0.438150`

A2C:

- clean survival ratio: `1.000000`
- damaged survival ratio: `1.000000`
- recovered survival ratio: `1.000000`

So the accurate interpretation is:

- the cell controller **does solve the pole-balancing task**
- it is also robust under the implemented cell-damage protocol
- the strongest RL baseline in this repo, the actor-critic comparator, scores higher on raw clean control
- the DQN comparator is weaker than the cell controller under damage

## 24. What "solved" means here

There are two reasonable meanings of "solved":

1. **Benchmark solved**
   The implementation's configured solve threshold is exceeded.

2. **The pole actually balances**
   The controller keeps the pole upright for the episode budget in most trials.

For the current cell controller, both are true.

It is not perfect, but it genuinely balances the pole autonomously.

## 25. Important limitations and mismatches

This section matters because without it the benchmark can be misunderstood.

### 25.1 It is a simplified biological controller, not a literal cell simulator

The cells are abstract dynamical units with biologically inspired state variables. They are not detailed biophysical cells.

### 25.2 Morphology is fixed

The genome does not currently evolve the body shape.

### 25.3 Some decoded knobs are currently unused or underused

In the current `cellengine.cpp` implementation:

- `mech_filter` is decoded but not used in the runtime equations
- `damage_resilience` is decoded but not used in the runtime equations
- `G[6]` has no direct dedicated runtime multiplier, although it affects phenotype distance and therefore indirectly affects coupling initialization

So the genome-to-dynamics mapping is not yet fully saturated.

### 25.4 The PD reference is not a trustworthy gold standard here

The saved results show the PD reference performing very poorly on this exact environment setup.

That means `pd_gold` should not be interpreted as an actual upper bound in the usual control-theory sense for this implementation. It is just a reference controller under the current force scaling and task conventions.

### 25.5 RL damage is not the same as cell damage

Cell damage is structural lesioning of the tissue.

RL damage is random parameter deletion.

Those are analogous but not identical.

## 26. The cleanest mental model to keep in mind

If you want one concise, accurate mental model, use this:

- A genome defines how a fixed tissue differentiates.
- That tissue has local electrical, chemical, energetic, and mechanical state.
- The cartpole writes stress into the tissue.
- The tissue turns that into coordinated contractile force.
- Evolution shapes the genome.
- Local Hebbian/homeostatic processes shape the tissue's state during life.
- RL comparators solve the same cartpole directly from the 4D global state using standard neural RL.

## 27. Practical guidance for interpreting the viewer

When using the React viewer:

- `Run replay` means "generate a fresh traced episode", not "just play an old animation".
- The left cell panel is a visualization of the selected genome's tissue state over the traced episode.
- The right RL panel is a separate controller trained on the same task state space.
- Large cart drift does not necessarily mean failure in this benchmark.
- The important failure variable is the pole angle corridor.

## 28. Bottom line

The implemented benchmark demonstrates that:

- a non-backpropagated cell-like tissue controller can be evolved to balance the pole,
- that controller can retain strong performance after explicit cell lesioning,
- local adaptive mechanisms inside the tissue matter to its behavior,
- and the resulting system is qualitatively different from the standard RL baselines even when the raw task is the same.

At the same time, the current system is still a simplified operational prototype, not the final full biological architecture envisioned by the long spec.
