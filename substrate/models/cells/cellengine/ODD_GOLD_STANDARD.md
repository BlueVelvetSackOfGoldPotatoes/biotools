# CellEngine ODD Gold Standard

Date: 2026-03-11

This document upgrades the existing [DEEP_DIVE.md](/home/pessegueirolunar/Documents/MNIST/models/cells/cellengine/DEEP_DIVE.md) into an ODD-style causal specification for the current CellEngine implementation.

ODD here means:

- Overview
- Design concepts
- Details

This version is written to answer the stricter question:

- which variables contribute to which outcomes,
- by what exact coefficient or range when the implementation exposes one,
- and where interpretation is confounded by mediation, carryover, or shared causes.

It is therefore both:

- an ODD for the model as implemented, and
- a requirements document for making attribution claims gold-standard rather than impressionistic.

## 1. Overview

### 1.1 Purpose

The implemented system evolves a shared 96-float genome that differentiates a selected body template (`fixed2d`, `grown2d`, or `grown3d`) into a controller for the active task. The most mature and best-analyzed case is cartpole balance, where environment-induced stress is converted into local cellular dynamics and then summed back into horizontal force.

### 1.1.1 Implemented vs planned matrix

| Area | Current state |
| --- | --- |
| Shared genome + developmental phenotype map | implemented |
| `fixed2d`, `grown2d`, `grown3d` body modes | implemented |
| Task family: cartpole / mass-spring / worm / pong | implemented |
| ODD telemetry: current / stress / mechanics / gap decomposition | implemented |
| Counterfactual intervention suite | partial |
| Long-range neurite / axon layer | planned |
| Rich GRN, migration, division during runtime | planned |

### 1.2 State variables and scales

Environment state depends on task. For the cartpole case analyzed most heavily in this document:

- `x`
- `x_dot`
- `theta`
- `theta_dot`

Fast per-cell state:

- `V`
- `Wrec`
- `Ca`
- `sigma`
- `energy`
- `a_mech`
- `chem_out`
- `stress`
- `activity_avg`

Slow per-cell state:

- `fatigue`
- `gap`
- `gap_base`
- `G[0..7]`

Structural body variables:

- body cell coordinates `(x, y, z)`
- `hinge`
- `ground`
- `motor`
- `force_direction_x`
- `mech_advantage`
- geometric features `feature_x_norm`, `feature_y_norm`, `feature_hinge_prox`, `feature_surface_prox`, `feature_neighbor_density`

Scales:

- physics step `dt = 0.02`
- cell substeps per physics step `= 4`
- cell substep `dt = 0.005`
- failure threshold `|theta| > 15 deg`

### 1.3 Process overview

At each physics step:

1. Cartpole state is integrated.
2. Mechanical load is written into per-cell `stress`.
3. Cells are advanced for 4 substeps.
4. Tissue force is read out from `a_mech`.
5. A smoothed force command is applied to the cartpole.
6. Aggregate telemetry is recorded.

At the end of each trial:

1. A slow phenotype drift update is applied.
2. Fatigue is updated.
3. Most tissue state persists across trials.

Across generations:

1. Genomes are evaluated.
2. A composite score is computed.
3. GA selection, crossover, mutation, and local refinement are applied.

## 2. Design Concepts

### 2.1 Emergence

No cell is assigned a hard-coded semantic type beyond structural geometry flags. Functional specialization emerges from:

- geometry-dependent developmental mapping into `G`,
- local state dynamics,
- local coupling adaptation,
- and asymmetric mechanical readout through `force_direction_x` and `mech_advantage`.

### 2.2 Sensing, communication, actuation

Sensing:

- cells sense mechanical stress through `I_mech`
- cells sense neighbor voltage through `I_elec`
- cells sense neighbor chemical output through `I_chem`

Communication:

- electrical coupling through adaptive gap junctions
- local chemical influence through neighbor `chem_out`

Actuation:

- each cell contributes `a_mech`
- organism force is the weighted sum of `a_mech * force_direction_x * mech_advantage`

### 2.3 Adaptation

There are three distinct adaptive layers:

- evolutionary search over genomes
- within-trial fast adaptation of `V`, `Ca`, `sigma`, `energy`, `a_mech`, `chem_out`, `gap`
- across-trial slow adaptation of `G` and `fatigue`

### 2.4 Stochasticity

Stochasticity enters through:

- genome initialization and mutation
- trial seeds
- damage target selection
- RL comparator training and evaluation seeds

### 2.5 Observation

Current replay artifacts expose:

- aggregate organism telemetry
- per-cell dynamic state
- per-edge coupling state
- static body and final gene-expression values

They do not yet expose the full causal decomposition needed for gold-standard attribution.

## 3. Details

### 3.1 Body construction

Body geometry is fixed:

- 9 ground cells at `y = 0`, `x = -4..4`
- 9 cells at `y = 1`, `x = -4..4`
- 11 cells in a vertical column at `x = 0`, `y = 2..12`

Derived geometry:

- `x_norm = x / 4.0`
- `y_norm = y / 12.0`
- `hinge_prox = exp(-dist((x, y), (0, 1.5)) / 1.8)`
- `surface_prox = 1 - neighbors / kMaxNeighborDirs`
- `neighbor_density = neighbors / kMaxNeighborDirs`

Structural readout terms:

- `force_direction_x = clamp(-x_norm, -1, 1)` for ground cells, else `0`
- `mech_advantage = 0.7 + 0.3 * abs(x_norm)` for ground cells
- `mech_advantage = 0.35 + 0.2 * abs(x_norm)` for non-ground cells with `y <= 1`
- `mech_advantage = 0.10 + 0.18 * hinge_prox` otherwise

Implication:

- geometry is a first-order confound because it influences both phenotype assignment and force readout.

### 3.2 Developmental map

Each cell receives an 8D phenotype vector:

```text
G_i[g] = sigmoid(sum_f W[g][f] * feature_i[f] + bias[g])
```

Parameter ranges:

- `W[g][f] in [-3.0, 3.0]`
- `bias[g] in [-2.2, 2.2]`

This means each geometric feature can strongly increase or suppress each program channel before sigmoid compression.

### 3.3 Decoded global parameters

The runtime genome decodes into scalar parameters with exact ranges. High-impact ones are:

- `coupling_base in [0.05, 0.55]`
- `hebb_rate in [0.02, 0.18]`
- `decay_rate in [0.01, 0.10]`
- `adhesion_coeff in [0.08, 0.40]`
- `chem_coeff in [0.00, 0.50]`
- `mech_base in [0.5, 2.5]`
- `sigma_target in [0.15, 0.65]`
- `sigma_alpha in [0.01, 0.08]`
- `sigma_beta in [0.02, 0.18]`
- `energy_replenish in [0.025, 0.12]`
- `cost_vm in [0.004, 0.03]`
- `cost_ca in [0.002, 0.03]`
- `cost_mech in [0.004, 0.035]`
- `g_drift_rate in [0.0005, 0.010]`
- `force_scale in [5.0, 18.0]`
- `stress_scale in [0.5, 2.5]`
- `teacher_frac in [0.05, 0.30]`
- `teacher_force_scale in [0.7, 1.3]`
- `teacher_hebb_boost in [1.0, 2.5]`
- `stiffness_softness in [0.2, 1.4]`
- `basal_contractility in [0.0, 0.20]`
- `activity_decay in [0.80, 0.98]`
- `sensor_threshold in [0.02, 0.25]`
- `teacher_kp in [20.0, 110.0]`
- `teacher_kd in [6.0, 32.0]`
- `chemical_balance in [-0.35, 0.35]`
- `muscle_bias in [0.0, 0.2]`
- `energy_output_scale in [0.6, 1.6]`
- `force_smoothing in [0.0, 0.75]`
- `stress_smoothing in [0.0, 0.75]`

Two decoded values are currently not used in the runtime equations:

- `mech_filter`
- `damage_resilience`

### 3.4 Initial connectivity

Initial directed neighbor coupling is:

```text
diff = mean_g abs(G_i[g] - G_j[g])
adhesion = 1 - abs(G_i[5] - G_j[5])
gap_base = clamp(coupling_base * (0.45 + 0.55 * (1 - diff)) + adhesion_coeff * adhesion, 0.02, 1.0)
gap = gap_base
```

Implication:

- `G[5]` has a direct initialization effect on coupling.
- `G[6]` does not have a direct runtime coupling multiplier.
- phenotype similarity and adhesion are confounded at initialization because both are functions of `G`.

### 3.5 Stress write submodel

Stress is written from cartpole mechanics into the tissue.

External stress before relaxation:

```text
hinge_moment = m*g*l*sin(theta) + m*l*l*theta_ddot
lateral_force = m*l*(theta_ddot*cos(theta) - theta_dot^2*sin(theta))
gravity_load = (M + m)*g

ext_i = stress_scale * (
    -22.0 * hinge_moment * x * (0.30 * hinge + 0.70 * cart_transmit)
    + 9.0 * lateral_force * (0.20 + 0.80 * cart_transmit) * (0.35 + 0.65 * abs(x))
    + 0.12 * gravity_load * (1.0 - y) * (ground ? 1.0 : 0.15)
    + 0.35 * x_dot * (ground ? -x : 0.0)
)
```

Where:

- `x = feature_x_norm`
- `y = feature_y_norm`
- `hinge = feature_hinge_prox`
- `cart_transmit = 0.35 + 0.65 * (1 - y)`

Relaxation step:

```text
stiffness_i = stiffness_softness * (0.35 + 0.65 * (1 - G_i[0])) * (0.75 + 0.25 * (1 - G_i[1]))
blended_i = 0.45 * external_i + 0.55 * neighbor_average_weighted_by_stiffness
stress_i <- stress_smoothing * stress_i + (1 - stress_smoothing) * blended_i
```

Implication:

- stress is not purely exogenous; it is partly reshaped by phenotype through stiffness.
- `G[0]` and `G[1]` confound later mechanosensory effects because they alter the stress field before the cell ever reads it.

### 3.6 Fast cell substep

For each active cell:

```text
I_elec = sum_j gap_ij * (V_j - V_i)
I_chem = sum_j gap_ij * chem_out_j
I_mech = mech_base * (0.50 + 2.20 * G_i[2]) * stress_i

I_total = sigma_i * (I_elec + chem_coeff * I_chem + I_mech + chemical_balance * chem_out_i) * energy_i
```

Voltage and recovery:

```text
tau_v = max(0.35, tau_v_base + tau_v_gain * (G_i[0] - 0.5))
eps   = max(0.015, epsilon_base + epsilon_gain * (G_i[0] - 0.5))

dV = (V - V^3 / 3 - W + I_total) / tau_v
dW = eps * (V + fhn_a - fhn_b * W)
```

Calcium:

```text
Ca <- Ca + dt * (-Ca / tau_ca + alpha_ca * max(0, Vn - v_ca_thresh))
```

Activity smoothing:

```text
activity <- activity_decay * activity_prev + (1 - activity_decay) * abs(Vn)
```

Homeostatic sensitivity:

```text
sigma <- sigma + dt * (sigma_alpha * (sigma_target - sigma) + sigma_beta * (sigma_target - activity))
```

Mechanical output:

```text
a_gate = 0.10 + 0.90 * sigmoid(2.5 * Vn)
mech_contract = max(0, stress_i) * (0.15 + 1.10 * G_i[2])

a_mech = clamp(
  (0.20 + basal_contractility + 1.80 * G_i[1] + muscle_bias * surface_prox_i)
  * (tanh(2.2 * Ca) + 0.70 * mech_contract)
  * a_gate
  * energy
  * energy_output_scale,
  -2.5, 2.5
)
```

Chemical output:

```text
chem_out = (0.15 + G_i[3]) * tanh(1.6 * Ca) * (G_i[4] >= 0.5 ? +1 : -1)
```

Energy:

```text
energy <- clamp(
  energy + dt * (
    energy_replenish
    - cost_vm * |Vn|^2
    - cost_ca * Ca
    - cost_mech * |a_mech|
  ),
  0, 1
)
```

Gap plasticity:

```text
hebb_scale = hebb_rate * (0.35 + 1.2 * G_i[7]) * (teaching_active ? teacher_hebb_boost : 1.0)
corr_ij = clamp(V_i * V_j, -1, 1)
gap_ij <- clamp(gap_ij + dt * (hebb_scale * corr_ij - decay_rate * (gap_ij - gap_base_ij)), 0, 1)
```

### 3.7 Slow update

After each trial:

```text
neighbor_signal = mean_j Ca_j
target_G = clamp(0.55 * G + 0.25 * Ca_self + 0.20 * neighbor_signal, 0, 1)
G <- clamp(G + g_drift_rate * (target_G - G), 0, 1)

fatigue <- clamp(0.97 * fatigue + 0.03 * (1 - energy), 0, 1)
```

Implication:

- phenotype is both a cause and a consequence.
- any attribution that treats `G` as fixed during evaluation is incomplete for multi-trial summaries.

### 3.8 Force readout

Organism force before scaling and clipping:

```text
force_raw = sum_i a_mech_i * force_direction_x_i * mech_advantage_i
```

Current implementation scales it as:

```text
organism_force = 16.0 * force_raw
```

Then trial-level smoothing applies:

```text
force_cmd <- force_smoothing * force_prev + (1 - force_smoothing) * force_cmd
```

Implication:

- even a large `a_mech` change may not appear immediately in the cartpole because geometry and force smoothing mediate it.

### 3.9 Teacher force

During teacher-active trials:

```text
teacher_force = clamp(-(teacher_kp * theta + teacher_kd * theta_dot) * teacher_force_scale, -force_limit, force_limit)
```

Teacher affects learning in two ways:

- directly changes the environment trajectory
- multiplies Hebbian adaptation through `teacher_hebb_boost`

### 3.10 Evolution score

Genome search optimizes:

```text
score =
    survival_ratio
  + 0.18 * sensor_responsiveness
  + 0.18 * signal_connectivity
  + 0.25 * force_correlation
  + 0.12 * clamp(0.5 + 0.5 * adaptation_gain, 0, 1)
  + 0.05 * mean_energy
  - 0.03 * (mean_force_abs / max(1, force_scale))
```

This means a variable can matter to evolution even if its effect on raw survival is indirect or delayed.

## 4. Direct Contribution Register

This section distinguishes direct implementation effects from mediated effects.

### 4.1 Gene-expression channels

`G[0]` excitability-like program:

- direct on `tau_v`
- direct on `eps`
- direct on stress relaxation stiffness
- indirect on all downstream electrical and mechanical behavior

`G[1]` contractility-like program:

- direct `+1.80 * G[1]` in the `a_mech` prefactor
- direct on stress relaxation stiffness
- indirect on final force, energy depletion, and gap evolution

`G[2]` mechanosensitivity-like program:

- direct multiplier `(0.50 + 2.20 * G[2])` on `I_mech`
- direct multiplier `(0.15 + 1.10 * G[2])` on `mech_contract`
- strongest single direct bridge from stress to action

`G[3]` secretion magnitude:

- direct multiplier `(0.15 + G[3])` on `chem_out`

`G[4]` secretion sign:

- direct sign switch on `chem_out`

`G[5]` adhesion class:

- direct on `adhesion = 1 - |G5_i - G5_j|`
- direct on `gap_base`
- no explicit later runtime term beyond initialization

`G[6]` coupling-labelled program:

- no direct dedicated runtime equation in current implementation
- indirect only via phenotype difference used at gap initialization

`G[7]` adaptation-rate program:

- direct on `hebb_scale = hebb_rate * (0.35 + 1.2 * G[7]) * ...`

### 4.2 Fast states

`stress`:

- direct input to `I_mech`
- direct input to `mech_contract`
- but already phenotype-shaped by the relaxation model

`sigma`:

- direct multiplicative gain on all incoming current terms through `I_total`

`energy`:

- direct multiplicative gain on `I_total`
- direct multiplicative gain on `a_mech`
- also a sink state affected by `V`, `Ca`, `a_mech`

`V`:

- direct to `Ca`
- direct to `activity`
- direct to `corr_ij` for gap plasticity

`Ca`:

- direct to `a_mech`
- direct to `chem_out`
- direct to slow `G` drift target

`gap`:

- direct to `I_elec`
- direct to `I_chem`
- adaptive and therefore history-dependent

### 4.3 Geometry and body role

Geometry is not background metadata. It is causal because it directly determines:

- developmental features
- `force_direction_x`
- `mech_advantage`
- `motor`, `hinge`, `ground`
- surface bias in `a_mech`
- stress projection weights

## 5. Confound Structure

### 5.1 Geometry-development-mechanics confound

The same body coordinate influences:

- developmental input features
- stress write weights
- force readout weights

Therefore a cell may appear causally important because:

- it expressed a different program,
- it received a different stress load,
- or it simply had higher mechanical leverage.

These are not separable from current replay telemetry alone.

### 5.2 Endogenous state confound

`stress -> I_mech -> V -> Ca -> a_mech / chem_out -> energy -> future I_total`

This is a closed feedback loop. Any variable inside it is both cause and effect over time.

### 5.3 Plastic-structure confound

Gap strength influences:

- electrical flow
- chemical flow

But gap strength itself is updated from prior `V_i * V_j`.

So “connectivity caused behavior” is always partly confounded by “behavior caused connectivity”.

### 5.4 Teacher confound

Teacher changes:

- action applied to the cartpole
- experienced trajectory
- Hebbian learning rate

Thus early-trial performance and later autonomous performance are not independent.

### 5.5 Cross-trial carryover confound

Trials do not fully reset tissue state. This means later-trial success may reflect:

- current genome,
- accumulated gap reconfiguration,
- accumulated energy/fatigue state,
- and slow `G` drift.

### 5.6 Evaluation-score confound

Evolution does not optimize survival alone. A variable can be selected because it improves:

- signal connectivity
- probe responsiveness
- force correlation
- adaptation gain
- energy economy

without being the most direct lever on survival in a single replay.

## 6. What Is Missing For Gold-Standard Attribution

Current replay is not enough to say exactly “variable X contributed Y% to success” in a defensible way.

The missing telemetry is:

### 6.1 Per-cell current decomposition

Must log per tick and per cell:

- `I_elec`
- `I_chem`
- `I_mech`
- self-chemical term `chemical_balance * chem_out_self`
- `I_total`

Without this, current drive attribution is underidentified.

### 6.2 Stress decomposition

Must log per tick and per cell:

- hinge-moment component
- lateral-force component
- gravity-load component
- cart-velocity component
- relaxed neighbor contribution
- final smoothed `stress`

Without this, “stress caused X” is too coarse.

### 6.3 Mechanical contribution decomposition

Must log per tick and per cell:

- `a_gate`
- `mech_contract`
- contractility prefactor
- final per-cell force contribution `a_mech * force_direction_x * mech_advantage`

Without this, force attribution is mixed across activation, geometry, and smoothing.

### 6.4 Gap plasticity decomposition

Must log per tick and per edge:

- `corr_ij`
- Hebbian increment term
- decay term
- old `gap`
- new `gap`

Without this, structural adaptation cannot be causally interpreted.

### 6.5 Trial context variables

Must log per trial:

- teacher active yes/no
- damage active yes/no
- tissue state snapshot before reset
- tissue state snapshot after reset

Without this, cross-trial carryover remains hidden.

### 6.6 Decoded scalar parameter dump

Must log once per replay:

- all decoded genome scalar parameters
- all `W[g][f]`
- all `bias[g]`

Without this, replay-level attribution cannot be normalized to the true causal parameter set.

## 7. Gold-Standard Intervention Suite

Telemetry alone is not enough. The minimum intervention suite should be:

### 7.1 Single-variable replay ablations

For a fixed initial condition, rerun with one variable clamped or zeroed:

- remove `I_elec`
- remove `I_chem`
- remove `I_mech`
- freeze `sigma`
- freeze `gap`
- freeze `G`
- zero teacher

### 7.2 Gene-channel ablations

For the same genome and same seed, rerun with:

- `G[k]` frozen at baseline
- `G[k]` zeroed
- `G[k]` perturbed by small delta

This is the cleanest way to identify whether a channel is direct, mediated, or redundant.

### 7.3 Per-cell force ablation

For each cell:

- zero only that cell’s `a_mech`
- measure delta in `theta` survival and force correlation

This identifies mechanical importance separately from signaling importance.

### 7.4 Edge ablation

For each edge or edge family:

- zero the gap
- rerun fixed-seed replay

This identifies communication bottlenecks.

## 8. Minimum Gold-Standard Deliverables

To call the analysis gold-standard, the project should provide:

1. An ODD with exact runtime equations and coefficient ranges.
2. A direct-effect table separating direct, mediated, and initialization-only effects.
3. A confound register like Section 5.
4. Replay telemetry extended with the missing decomposition variables in Section 6.
5. An intervention suite like Section 7 on fixed seeds and fixed initial angles.
6. Attribution outputs reported as counterfactual deltas, not just correlations.

## 9. Bottom Line

The current implementation is already strong enough to support a serious causal ODD.

What it does not yet support is gold-standard attribution of:

- exactly how much each variable contributed,
- and how much of that apparent contribution is confounded by geometry, state feedback, plasticity, teaching, and cross-trial carryover.

The path to gold-standard is therefore clear:

- keep the current model,
- add the missing decomposition telemetry,
- run fixed-seed interventions,
- and report direct effects separately from mediated and confounded effects.
