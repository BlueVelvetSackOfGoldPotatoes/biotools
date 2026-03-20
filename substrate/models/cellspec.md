# CellEngine: A General Cell-Based Intelligence Benchmark

## Full Specification for a GPU-Accelerated Evolutionary Cell Simulator

---

# Part I — Strategy, Reasoning, and Scientific Basis

## 1. What We Are Testing

The central claim of deep learning is implicit: a graph of simple units (nodes that compute weighted sums + nonlinearities), optimized end-to-end by gradient descent on a task loss, is sufficient for intelligence. Everything else — the cell biology, the metabolism, the morphology, the homeostasis, the development — is implementation detail that can be abstracted away without losing anything computationally essential.

This project tests the negation: that the structural, topological, morphological, and functional features of real biological cells contain computational principles that are *not* captured by point-neuron abstractions, and that these principles are relevant to adaptive, robust, self-repairing intelligence.

The test is concrete: take a general model of biological cells (not engineered for any task), place those cells in a physical environment with a control problem, evolve genomes that parameterize cell behavior, and measure what emerges. Then compare against standard controllers on the dimensions where cell-like organization should matter: damage recovery, perturbation adaptation, continual learning, energy efficiency.

The key discipline is that the cell model must come from biology, not from the task. If we design cell types that happen to match cartpole components (tilt sensor, wire, motor), we've built a cartpole controller wearing a biology costume. If we derive cell capabilities from what all biological cells actually do, and those capabilities happen to be sufficient for cartpole (and for other tasks), that's evidence that cellular organization contains general computational principles.

## 2. Why One Cell Class, Not Discrete Types

In real biology, all cells share the same genome and the same fundamental machinery. A muscle cell and a neuron and a skin cell all have:

- A membrane potential (voltage across the cell membrane). This is not a neuron-specific feature. Every cell maintains a resting potential through ion pumps and channels. The difference is quantitative: neurons have more excitable membranes (faster, larger voltage swings) because they express more voltage-gated ion channels.

- Calcium signaling. Every cell uses intracellular calcium as a second messenger. Calcium triggers muscle contraction, neurotransmitter release, gene expression changes, cell division, and apoptosis. The difference between a muscle cell and an epithelial cell is how much contractile protein is present to respond to calcium — not whether calcium signaling exists.

- Gap junction coupling. Most cells in a tissue are electrically coupled to their neighbors through gap junctions — protein channels that directly connect the cytoplasm of adjacent cells. Electrical signals (voltage changes) spread through gap junctions at speeds of ~0.1–1 mm/ms. The difference between "neural" and "non-neural" electrical signaling is partly about gap junction density and coupling strength.

- Paracrine (chemical) signaling. Every cell secretes molecules into the extracellular space. Neighbors with appropriate receptors detect those molecules. This is slower than electrical signaling (diffusion-limited, seconds) but carries richer information (different molecules, different concentrations).

- Mechanosensitivity. Every cell has mechanosensitive ion channels that open when the membrane is stretched. This is how cells sense forces, pressure, and deformation. Specialized mechanoreceptors (like hair cells in the ear) are just cells with extremely sensitive versions of channels that all cells possess.

- Contractility. Every cell has a cytoskeleton — a network of protein filaments (actin, myosin, intermediate filaments, microtubules) that gives the cell shape and lets it generate force. Muscle cells have massively upregulated actin-myosin complexes, but even fibroblasts and epithelial cells generate traction forces and can contract.

- Gene expression as phenotype control. Every cell can change what proteins it produces by altering gene expression. This is how cell "types" emerge during development: signaling cascades activate transcription factors that turn on type-specific gene programs. But this is not a one-time event — gene expression continues to shift throughout life in response to signals, stress, and activity.

The consequence for modeling: **cell type is not a categorical variable. It is a point in a continuous phenotype space determined by gene expression.** A "muscle cell" is a cell with high contractility parameters and high calcium sensitivity. A "neuron" is a cell with high excitability and high coupling. A "sensor" is a cell with high mechanosensitivity. These are regions in parameter space, not discrete categories. The genome encodes rules that determine where in this parameter space each cell ends up, based on its developmental history and position.

The model should reflect this. One cell class. One set of update equations. A continuous parameter vector (the gene expression state G) that determines how excitable, how contractile, how secretory, how mechanosensitive, and how adhesive each cell is. The GA evolves genomes that set G-determining rules. "Cell types" emerge from evolution, not from the programmer.

## 3. What Features of Real Cells to Include (and What to Leave Out)

### Include: features that are universal, computationally relevant, and feasible to simulate

**Membrane potential (V_m):** The fastest communication channel. All cells have it. It propagates through gap junctions. It is the primary substrate for fast coordination in neural and non-neural tissues. Without it, the organism can only communicate chemically (too slow for real-time control) or mechanically (too slow and noisy).

**Calcium / second messenger ([Ca]):** The medium-timescale integrator. Calcium waves coordinate tissue-level behavior on the timescale of seconds. Calcium transients trigger contractility, secretion, and gene expression changes. Including it gives the cell a "mood" — a state that modulates all other behaviors and changes more slowly than V_m.

**Gene expression state (G):** The slow phenotype memory. G determines what the cell "is" — its excitability, contractility, secretion profile, adhesion properties. G changes on the slowest timescale (developmental time or across many episodes). Including it allows cells to specialize during development and, potentially, to transdifferentiate in response to sustained signals or damage.

**Energy (E):** The universal metabolic constraint. Every behavior costs energy. Energy replenishment is finite. A cell with zero energy stops functioning. Including it prevents trivial solutions (all cells maximally active all the time) and creates a natural trade-off between performance and efficiency. It also means damage (energy-consuming repair) has a real cost.

**Electrical coupling (gap junctions):** Fast, bidirectional, strength-modifiable. The primary mechanism for fast signal propagation. Coupling strength between two cells depends on both cells' G states (gap junction protein expression) and can be modulated by [Ca] (some gap junctions close when calcium is high, which is a real biological mechanism that compartmentalizes signals).

**Chemical signaling (paracrine):** Slower, diffusible, carries type information. Each cell secretes a signal whose "flavor" (excitatory/inhibitory analog) and magnitude depend on G and [Ca]. The signal diffuses to nearby cells (not just direct neighbors — it reaches 2–3 voxels with attenuation). This provides a second communication channel with different spatiotemporal properties than electrical coupling.

**Mechanosensitivity:** Converts mechanical strain into V_m changes via stretch-activated channels. This is how the organism senses the external physics. Without it, cells have no way to know what's happening in the physical environment. The sensitivity gain is controlled by G.

**Contractility:** Converts [Ca] into mechanical force on neighbors. This is how the organism acts on the external physics. Force magnitude depends on [Ca] level, contractile protein expression (in G), and energy availability.

**Adhesion:** Determines which cells stick to which. Adhesion strength between two cells depends on the compatibility of their adhesion profiles (part of G). In the simplest version, cells with similar G states adhere strongly (homophilic adhesion). This drives tissue organization during development: like cells cluster together, unlike cells separate. It also means damage (removing cells) creates free surfaces that trigger adhesion-driven reorganization.

**Habituation / Sensitization:** The primary within-lifetime adaptation mechanism. Sensitivity (σ, a multiplier on all inputs) drifts based on recent activity. Sustained high activation → σ decreases (habituation). Prolonged low activation → σ increases (sensitization). This is a local, model-free adaptation that every neuron and many non-neural cells exhibit. It is *not* gradient descent. It is *not* reward-based. It is a cellular homeostatic mechanism that happens to produce useful adaptation.

**Hebbian-like coupling changes:** Connection strengths (gap junction density) change based on correlated activity between neighbors. If two cells are frequently active together, their coupling strengthens. If they're uncorrelated, coupling decays. This is local, unsupervised, and biologically grounded (activity-dependent gap junction remodeling is well-documented). It is the main mechanism for within-lifetime "wiring" changes.

### Leave out (for now): features that are important but computationally expensive or add complexity without clear benefit for the first experiment

**Full cell migration:** Cells physically moving through the grid is expensive to simulate (requires solving occupancy conflicts, maintaining connectivity, recomputing neighbors). Save for Phase B.

**Cell division during runtime:** Adding cells mid-episode changes the grid size, invalidates precomputed data structures, and complicates the CUDA kernel. Save for Phase B. Development (division during the build phase before episodes start) is included.

**Detailed gene regulatory networks:** Real gene expression involves networks of transcription factors with complex dynamics. For now, G is a vector of continuous parameters that change slowly in response to signals. A full GRN can be added later if the simple model proves insufficient.

**Detailed ion channel models:** Hodgkin-Huxley-style ion channel dynamics are expensive (multiple gating variables per channel type per cell). Instead, use a simplified excitability model: V_m dynamics are governed by a 2-variable system (V_m + one recovery variable) that captures excitable behavior (threshold, refractory period, bistability) without the full biophysics. The FitzHugh-Nagumo model is the standard choice here.

**Intracellular spatial structure:** Real cells have organelles, cytoskeletal networks, and spatial gradients of molecules. All of this is below the resolution of a voxel model. Each cell is a point with state variables, not a spatially extended object.

## 4. Evolution as the Only Global Optimizer

The genetic algorithm is the only process in the system that has access to fitness (task performance). Nothing inside the organism sees fitness. No cell computes or receives a task-level error signal.

The GA operates at the population level: it creates genomes, evaluates organisms (by running their lifetime in the environment), assigns fitness, and selects/mutates/recombines genomes. The timescale is evolutionary: hundreds to thousands of generations, each involving the birth and death of an entire population.

Inside each organism, cells adapt through local mechanisms (habituation, sensitization, Hebbian coupling changes, calcium-dependent G drift). These mechanisms are parameterized by the genome — the adaptation rates, thresholds, and directions are heritable. But the adaptation itself is driven by cell-level activity, not by any fitness signal. A cell that habituates to sustained input does so whether the organism is balancing well or falling. The GA selects genomes whose cellular adaptation rules *happen to produce useful behavior changes*.

This separation is important because it's the biological reality we're modeling. Evolution sets the genome. The genome sets cell rules. Cell rules + environment produce behavior. Behavior determines fitness. Fitness shapes the next generation of genomes. At no point does fitness directly modify cell behavior within a lifetime.

## 5. Teaching as Environmental Condition

During the first few trials of an organism's lifetime, an external corrective force (a PD controller on the cartpole) can be applied alongside whatever force the organism's cells produce. This "teacher" keeps the pole roughly upright.

From the organism's perspective, nothing special is happening. Its sensor cells experience the stress patterns of a balanced system. Its cells activate in whatever pattern the coupling produces. Its Hebbian mechanisms strengthen whatever connections happen to be active. Its calcium state and gene expression shift in response to the activation history.

When the teacher turns off, the cell states retain traces of the "balanced" experience. If those traces happen to create functional signal paths (because the genome encoded cells whose adaptation rules respond usefully to guided experience), the organism may continue balancing. If not, it falls — and the GA selects against that genome.

The teacher is an environmental condition, like gravity or friction. It provides structured experience, not instruction. No cell knows it's being taught. No cell receives a target or error. The GA evolves genomes whose cells learn from experience — including the experience of being guided.

Whether to include teaching is itself an experimental variable. Runs with and without the teaching phase measure how much structured environmental experience accelerates evolution and improves within-lifetime adaptation.

## 6. What We Expect to Learn

**If the GA evolves functional organisms:** We learn which regions of phenotype space (which G configurations) evolution converges on. Do discrete "types" emerge (clusters in G-space that correspond to sensor-like, conductor-like, muscle-like cells)? Or does the solution use a continuum of cell states? This tells us whether the biological cell-type taxonomy reflects computational necessity or developmental convenience.

**If the organisms show damage recovery:** We learn whether cell-level homeostasis and Hebbian rewiring are sufficient mechanisms for self-repair, or whether more sophisticated mechanisms (global error signals, structural memory) are needed.

**If teaching dramatically accelerates evolution:** We learn that demonstration-based environmental learning (not algorithmic learning) is an important complement to evolution — which is a finding about the relationship between development, learning, and evolution that is relevant to AI architecture.

**If the general cell model fails on cartpole:** We learn which specific cellular feature is missing. Is it directionality (cells need more polarized signaling)? Is it speed (electrical coupling through a voxel grid is too slow)? Is it integration (cells need a more sophisticated transfer function)? Each failure mode points to a specific biological mechanism that might be computationally essential.

---

# Part II — The Cell Model

## 1. One Cell Class

Every cell in every organism in every generation is an instance of the same struct. There are no subclasses, no type enums, no code branches based on cell identity. All behavioral differences come from continuous parameters.

## 2. State Variables

### Fast state (updated every cell tick, ~2.5ms)

**`V_m`** — Membrane potential. Signed float, nominally in [−1, +1] (normalized, not millivolts). This is the cell's primary signaling variable. Dynamics follow a simplified FitzHugh-Nagumo-like system:

```
dV/dt = (V_m - V_m³/3 - w + I_total) / τ_v
dw/dt = ε * (V_m + a - b*w) / τ_w
```

where `w` is a recovery variable (see below), `I_total` is the sum of all input currents (electrical coupling + chemical input + mechanosensory input + noise), and `τ_v`, `ε`, `a`, `b` are parameters determined by the gene expression state G.

This gives excitable dynamics: if I_total exceeds a threshold, V_m spikes. After spiking, w increases (recovery), which suppresses V_m (refractory period). The excitability (how easily V_m spikes) is controlled by G. A cell with low excitability (high threshold) behaves like a passive structural cell — V_m just follows a low-pass filtered version of input. A cell with high excitability fires sharp spikes in response to small inputs — nerve-like.

**`w`** — Recovery variable. Positive float. Part of the FitzHugh-Nagumo dynamics. Provides spike adaptation and refractory behavior. Not directly observable by other cells — it's internal state.

**`a_mech`** — Current mechanical activation (contractile force output). Float, computed from [Ca] and G. This is what the physics engine reads to compute forces. `a_mech = contractility_gain * f([Ca])`, where `contractility_gain` comes from G and `f` is a saturating function (tanh or sigmoid).

### Medium state (updated every ~10 cell ticks)

**`[Ca]`** — Intracellular calcium concentration. Positive float, nominally in [0, 1]. Dynamics:

```
d[Ca]/dt = (-[Ca] / τ_ca) + α_ca * max(0, V_m - V_ca_thresh)
```

Calcium rises when V_m exceeds a threshold (voltage-gated calcium influx) and decays exponentially otherwise. τ_ca and α_ca and V_ca_thresh are determined by G.

Calcium is the medium-timescale integrator. A single V_m spike causes a brief calcium transient. Sustained V_m activity causes sustained elevated calcium. Calcium in turn drives contractility, modulates secretion, affects coupling strength, and (on a slow timescale) influences gene expression.

**`σ`** — Sensitivity (gain multiplier on all inputs). Float in [0.1, 5.0]. Drifts toward a homeostatic target:

```
dσ/dt = α_σ * (σ_target - σ) + β_σ * (σ_target - moving_avg(|V_m|))
```

If the cell has been very active (moving average of |V_m| > σ_target), σ decreases — habituation. If the cell has been quiet, σ increases — sensitization. α_σ, β_σ, σ_target come from G.

This is the primary within-lifetime adaptation. It is purely local and homeostatic: the cell has no information about whether its activity was "useful." It just tries to keep its average activation near a target level. But at the organism level, this produces useful adaptation: if a particular signal path is over-driven, the cells in it desensitize, reducing overshoot. If a path is under-used, it sensitizes, becoming more responsive to weak signals.

**`w_gap[k]`** — Gap junction coupling strength to each occupied neighbor k. Float in [0, 1]. 
Updated by Hebbian-like activity-dependent remodeling:

```
dw_gap[k]/dt = η_hebb * corr(V_m_self, V_m_neighbor_k) - η_decay * w_gap[k] + η_adhesion * adhesion_compatibility(G_self, G_neighbor_k)
```

Coupling strengthens when two cells are frequently co-active (Hebbian term), decays with disuse, and has a baseline set by adhesion compatibility (cells with similar G states couple more strongly). η_hebb, η_decay, η_adhesion come from G.

This is how "wiring" forms within a lifetime. Initially, coupling strengths are set by the development phase (based on G compatibility). During episodes, active signal paths strengthen and inactive ones decay. This is not learning in the ML sense — there is no objective being optimized. It's a local structural change driven by correlated activity.

**`E`** — Energy (ATP analog). Float in [0, 1]. 

```
dE/dt = replenish_rate - cost_vm * |V_m|² - cost_ca * [Ca] - cost_mech * |a_mech|
```

Energy replenishes at a constant rate (from G) and is consumed by electrical activity, calcium maintenance, and mechanical force generation. When E < 1, all outputs are scaled by E (graceful degradation). When E = 0, the cell is inert (no electrical activity, no force, no secretion). Energy recovers when the cell is quiet.

This prevents the trivial strategy of "all cells maximally active all the time." It forces the organism to be efficient — only activating cells that contribute to balance, and only when needed.

### Slow state (updated once per episode or during development)

**`G`** — Gene expression state. A vector of N_g floats (N_g = 8 is the design target), each in [0, 1]. This is the cell's phenotype — it determines all parameters of the cell's behavior:

- `G[0]`: excitability — controls τ_v, ε, threshold in the FHN model. High = neuron-like. Low = passive.
- `G[1]`: contractility — controls the gain from [Ca] to mechanical force. High = muscle-like. Low = structural.
- `G[2]`: mechanosensitivity — controls the gain from mechanical strain to V_m input. High = sensor-like. Low = mechanically insensitive.
- `G[3]`: secretion_magnitude — controls how much chemical signal the cell releases. High = strong signaler.
- `G[4]`: secretion_flavor — controls the sign of the chemical signal. Low = inhibitory, High = excitatory.
- `G[5]`: adhesion_class — determines adhesion compatibility with neighbors. Cells with similar G[5] adhere strongly.
- `G[6]`: coupling_base — base gap junction density. High = strongly coupled to all neighbors.
- `G[7]`: adaptation_rate — controls how fast σ and w_gap change. High = rapidly adapting. Low = stable.

These are not independent — they interact through the update equations. A cell with high excitability AND high contractility is a "myocyte that fires" (like cardiac muscle). A cell with high excitability AND low contractility AND high coupling is "nerve-like." A cell with high mechanosensitivity AND high excitability is "sensory neuron-like." A cell with low everything is "structural."

G changes on the slowest timescale. During development, G is set by the genome's differentiation rules based on position and neighbor context. During runtime (episodes), G drifts very slowly in response to sustained [Ca] levels and neighbor signals:

```
dG/dt = μ_G * f(Ca_sustained, neighbor_signals, G_current)
```

where μ_G is very small (changes of ~1% per episode). This allows long-term specialization shifts but prevents the cell from completely reprogramming itself mid-task.

### Cached / derived (recomputed as needed, not stored as primary state)

**`I_elec`** — Electrical input current from gap junction coupling: `Σ_k w_gap[k] * (V_m_k - V_m_self)`.

**`I_chem`** — Chemical input from neighbor secretion (diffused, attenuated by distance).

**`I_mech`** — Mechanosensory input: `mechanosensitivity * local_strain`.

**`I_total`** — `σ * (I_elec + I_chem + I_mech) * E` — total input, scaled by sensitivity and energy.

**`chem_out`** — Chemical signal output: `secretion_magnitude * f([Ca]) * sign(secretion_flavor - 0.5)`.

## 3. Communication Channels (Three, Operating in Parallel)

### Channel 1: Electrical (fast, local, bidirectional)

Mechanism: gap junction coupling. Each cell's V_m pulls toward its neighbors' V_m, weighted by coupling strength w_gap[k].

Speed: instantaneous within one cell tick (2.5ms). A chain of N well-coupled cells transmits a signal in ~N cell ticks.

Range: direct neighbors only (26-connected voxel neighborhood).

Directionality: inherently bidirectional (gap junctions are symmetric). But the *effective* directionality can be asymmetric if cells have different excitabilities: a signal originating in a highly excitable cell propagates preferentially *away* from it (because it fires strongly and its neighbors fire in response), while a signal from a passive cell barely propagates. This is how biological tissues achieve directional signaling without synapses.

Modification: coupling strengths change via Hebbian remodeling (medium timescale).

### Channel 2: Chemical (medium speed, spatially extended, directed)

Mechanism: paracrine signaling. Each cell secretes a chemical signal proportional to its [Ca] level and secretion parameters (from G). The signal diffuses to nearby cells (not just direct neighbors but also second- and third-order neighbors, with exponential attenuation).

Speed: diffusion-limited. Reaches direct neighbors in ~1 cell tick, 2-voxel range in ~4 ticks, 3-voxel range in ~10 ticks.

Range: ~3 voxels (27-cell extended neighborhood, plus second ring).

Directionality: isotropic (diffuses equally in all directions) by default. But a cell can achieve polarized secretion if its G state and morphological context create asymmetric expression — this can emerge through development (cells at a tissue boundary might polarize).

Information content: richer than electrical (the signal has magnitude AND flavor — excitatory vs inhibitory), but slower.

### Channel 3: Mechanical (slow, long-range through rigid structures, passive)

Mechanism: stress/strain propagation through the structural lattice. When the cartpole physics exerts forces on the organism, those forces propagate through the structure as stress. Cells experience strain (deformation of their local neighborhood). Mechanosensitive cells convert strain into V_m input.

Speed: essentially instantaneous (stress propagates at the speed of sound in the material, which at voxel scale is faster than any simulation timestep). In practice, the stress field is computed once per physics tick as a global solve.

Range: entire structure (a force applied at the pole tip creates stress at the cart base).

Directionality: determined by the structure geometry. Stress from a lateral force propagates laterally. Stress from a bending moment creates compression on one side and tension on the other.

This is the organism's "sensory surface" — it's how the external physics becomes internal cell state. It's also how muscle forces are transmitted to the environment (contractile forces propagate through the structure to the contact surface).

## 4. Homeostasis: The Core Operating Principle

Every cell actively regulates its internal state variables toward target values:

- V_m has a resting potential (the V_m at which dV/dt = 0 with no input). The FHN dynamics naturally return V_m to rest after perturbation.
- [Ca] decays toward zero in the absence of V_m spikes. A cell "wants" low calcium (high calcium is a signal state, not a resting state).
- σ drifts toward σ_target. The cell maintains a preferred sensitivity level.
- E replenishes toward E_max. The cell restores energy when inactive.
- w_gap decays toward a baseline set by adhesion compatibility. Connections revert to a structural default without correlated activity.

The consequence: if you perturb the organism (damage cells, change the environment, apply novel forces), every surviving cell works to restore its own internal state to its homeostatic target. This is *not* the same as restoring the organism to its previous behavior — but it often has that effect, because the previous behavior was a steady state of the cellular dynamics, and the homeostatic targets define that steady state.

This is the mechanism behind damage recovery. When cells are deleted, the remaining cells experience changed inputs (missing neighbor signals, altered stress distribution). Their V_m, [Ca], and σ shift. Their homeostatic mechanisms push back: σ adjusts, coupling strengths change (Hebbian remodeling fills in for missing connections), mechanical loads redistribute. The organism doesn't "know" it was damaged. Each cell individually tries to get back to normal. The collective effect of all cells doing this simultaneously is self-repair.

## 5. The Genome

The genome is a flat vector of floats that parameterizes the cell model. Every cell in one organism shares the same genome but may express different behavior because G differs (determined by position and developmental history).

### Genome structure

**Section 1: FitzHugh-Nagumo parameters** (~10 floats)
Base values and G-sensitivity coefficients for: τ_v, ε, a, b (FHN parameters), V_ca_thresh (calcium threshold), τ_ca, α_ca (calcium dynamics). Each parameter P is computed as:

```
P(G) = P_base + P_G_coeff * dot(P_G_weights, G)
```

where P_base and P_G_coeff are genome values, and P_G_weights is a short weight vector (also in the genome) that determines which G components influence this parameter. This lets the genome create complex mappings from phenotype space to dynamics.

**Section 2: Communication parameters** (~15 floats)
Base gap junction coupling, Hebbian learning rate η_hebb, decay rate η_decay, adhesion coupling coefficient η_adhesion, chemical secretion parameters (magnitude gain, flavor sensitivity), chemical diffusion attenuation rate, mechanosensitivity base and G-coefficient.

**Section 3: Homeostatic parameters** (~10 floats)
σ_target, α_σ, β_σ (sensitivity adaptation), energy replenishment rate, energy costs (V_m, Ca, mechanical), G drift rate μ_G.

**Section 4: Development program** (~40 floats)
Rules for cell differentiation during the development phase (see Part III). These determine what G vector each cell gets based on its position in the body template.

**Section 5: Teaching parameters** (~5 floats)
Whether and how the organism responds to guided trials: teaching phase length (how many of the 100 trials have external guidance), teacher force magnitude, and parameters controlling how strongly the teaching experience imprints on cellular adaptation rates (e.g., a multiplicative boost to η_hebb during teaching trials — the cells adapt faster when being guided, not because they know they're being taught, but because the guided experience produces stronger/more-coherent activation patterns that drive Hebbian changes faster).

**Total genome size: ~80–100 floats.** This is intentionally compact. The genome does not encode per-cell parameters — it encodes rules that determine per-cell parameters. The complexity of the organism comes from the interaction between genome rules and body geometry during development, not from having a large genome.

## 6. Development Phase

Before the first trial, the organism's body is constructed from the genome.

### Phase A (fixed template):

The body shape is pre-defined (a cart-pole voxel template with ~200 occupied positions). Development only assigns G vectors to each cell.

For each voxel position, compute geometric features:
- Normalized position in body (x/L, y/H, z/W)
- Distance from hinge point (normalized)
- Distance from body surface (normalized)
- Whether in "cart region" or "pole region" (binary, from template)
- Local neighbor count (how many occupied neighbors, a proxy for interior vs surface)

The genome's development section encodes a mapping from these ~5 features to the 8 G components. The mapping is a simple linear function with nonlinear activation:

```
G_i = sigmoid(Σ_j W_dev[i][j] * feature_j + bias_dev[i])
```

where W_dev and bias_dev are genome parameters. This is ~48 parameters (8 G components × 5 features + 8 biases).

This produces spatially organized G patterns: cells near the hinge might get high mechanosensitivity (because distance-from-hinge is a feature), cells at the cart edges might get high contractility (because x-position is a feature), cells in the interior might get high coupling (because neighbor count is high). But the specific mapping is evolved, not designed.

After G is assigned, initial coupling strengths w_gap[k] are computed from adhesion compatibility:

```
w_gap_init[k] = coupling_base * exp(-||G_self - G_neighbor_k||² / adhesion_width)
```

Cells with similar G states start more strongly coupled. This creates initial "tissue" organization: regions of similar phenotype are internally well-connected.

### Phase B (evolvable morphology, future):

The genome additionally encodes growth rules. Starting from a seed, cells divide and differentiate over T development steps, producing a body whose shape is genome-determined. This is significantly harder and is not attempted until Phase A demonstrates functional cell-based control.

---

# Part III — The Environment

## 1. CartPole as First Test Environment

The cartpole is chosen because it's simple, well-understood, has clear success criteria, and has extensive baselines. It is NOT the only intended environment — it's the first.

### Physics model: hybrid coupling

**Rigid-body cartpole ODE:**

```
θ̈ = (g sinθ - cosθ (F + m l θ̇² sinθ) / (M + m)) / (l (4/3 - m cos²θ / (M + m)))
ẍ = (F + m l (θ̇² sinθ - θ̈ cosθ)) / (M + m)
```

Standard parameters: M = 1.0 kg (cart), m = 0.1 kg (pole), l = 0.5 m (half-pole-length), g = 9.81 m/s². The ODE is integrated at the physics tick rate (50 Hz, Δt = 0.02s) using RK4.

**Stress projection into the voxel structure:**

At each physics tick, after the ODE step, compute the forces the pole exerts on the structure:

1. Compute the bending moment at the hinge: `M_hinge = m * g * l * sinθ + m * l² * θ̈`. This is the torque trying to rotate the pole.
2. Compute the lateral force on the cart from the pole: `F_lat = m * l * (θ̈ * cosθ - θ̇² * sinθ)`.
3. Compute the gravitational load: `F_grav = (M + m) * g`, distributed vertically.

These forces are applied to the hinge voxels and propagated through the structural lattice using a few (3–5) Gauss-Seidel relaxation iterations on a simple spring model. Each occupied voxel has a stiffness (from G: cells with low excitability and low contractility are stiff; cells with high excitability are soft). Stress propagation follows:

```
for each relaxation iteration:
    for each voxel i:
        stress[i] = external_force[i] + Σ_k stiffness[k] * (stress[k] - stress[i]) / distance
```

This produces a 3D stress field. Each cell reads its local stress as the mechanosensory input `I_mech`.

**Force feedback from organism to ODE:**

The horizontal force F in the ODE is the sum of all cells' mechanical activation projected onto the horizontal axis and transmitted to the ground-contact voxels:

```
F = Σ_i a_mech[i] * force_direction_x[i] * ground_coupling[i]
```

where `force_direction_x[i]` is how much of cell i's contractile force projects onto the horizontal axis (determined by the cell's position in the structure and its G-derived force direction preference), and `ground_coupling[i]` measures how well force from cell i reaches the ground-contact surface (precomputed during development from the structural lattice topology, essentially the mechanical advantage).

### Trial structure

Each organism gets 100 trials:

1. **Trials 1–N_teach (teaching phase, N_teach from genome):** The cartpole starts with θ = random in [−3°, +3°]. A PD controller applies an external correction force F_teacher = Kp*θ + Kd*θ̇ alongside the organism's muscle force. The teacher keeps the pole roughly upright. Cells experience the stress/activation patterns of successful balance.

2. **Trials N_teach+1 through 100 (autonomous phase):** No teacher. The organism must balance on its own. θ = random in [−3°, +3°].

Each trial runs until |θ| > 15° (failure) or 500 physics ticks (10 seconds, success).

Cell state persists across trials: σ, w_gap, E, fatigue carry over. G drifts slightly. Cartpole state (x, θ, ẋ, θ̇) resets each trial.

### Fitness function

```
fitness = total_ticks_survived  (primary, max 50,000)
        + 500 * sensor_responsiveness  
        + 500 * signal_connectivity
        + 1000 * force_correlation
```

Incremental components (same as before) provide gradient in early generations. They are computed from a special probe at the start of evaluation: apply a known impulse, measure sensor activation (responsiveness), trace activation to muscle cells (connectivity), and correlate muscle force with impulse direction (force correlation).

## 2. Future Environments (Not Implemented Now, but the Architecture Supports Them)

The cell model is environment-agnostic. The environment interface is:

```
Input to cells: a stress/strain field over the voxel grid (from physics)
Output from cells: a force field over the voxel grid (from contractility)
```

Other environments that use the same interface:

- **Locomotion on a surface:** a multi-legged or worm-like body that must crawl. Stress from gravity + ground contact. Force from contractility moves the body.
- **Object manipulation:** a body with appendages that must grasp and move an object. Stress from object weight. Force from contractility manipulates the object.
- **Multi-organism interaction:** multiple cell-organisms in a shared environment. Mechanical coupling between organisms.
- **Damage recovery benchmark:** a standardized protocol where cells are deleted at specific times and recovery is measured.

The C++ architecture must support adding new environments by implementing a single interface (compute stress field, read force field, advance physics), without changing the cell model or GA.

---

# Part IV — The Genetic Algorithm

## 1. Overview

The GA is a standard real-valued evolutionary algorithm operating on the genome vector (~100 floats). Each generation:

1. Evaluate all organisms in the population (in parallel on GPU or across CPU threads).
2. Assign fitness.
3. Select parents (tournament selection).
4. Create offspring (mutation + crossover).
5. Replace population.

## 2. Evaluation Pipeline

For each organism:

1. **Decode genome** → compute development parameters.
2. **Run development** → assign G vectors to all cells, compute initial coupling strengths, precompute structural properties (mechanical advantage, neighbor lists).
3. **Run 100 trials** → for each trial, initialize cartpole, run physics+cell simulation loop, record ticks survived.
4. **Compute fitness** → sum ticks + incremental bonuses.

This is the inner loop. It must be fast. It is embarrassingly parallel across organisms.

## 3. GA Parameters

- **Population size:** 500
- **Generations:** 1000–3000 (depending on convergence)
- **Selection:** Tournament, size 5
- **Elitism:** Top 5 genomes survive unchanged
- **Mutation:** Each gene mutated with probability 0.1, Gaussian noise σ = 0.05 (scaled to parameter range)
- **Crossover:** Uniform, probability 0.3 per pair
- **Initial population:** 490 random genomes + 10 hand-seeded "hint" genomes (with development parameters biased toward placing high-mechanosensitivity cells near the hinge and high-contractility cells at cart edges)

## 4. Fitness Landscape Smoothing

The raw fitness landscape (ticks survived) is deceptive — most genomes score near zero, with rare functional organisms scoring high. The incremental components (sensor responsiveness, connectivity, force correlation) provide gradient, but may not be sufficient.

Additional smoothing strategies:

- **Novelty archive (optional):** Maintain an archive of behaviorally diverse organisms (defined by the distribution of G states and activation patterns). Add a novelty bonus to fitness: organisms that differ from the archive get extra fitness. This encourages exploration of the genome space.
- **Multi-objective:** Treat ticks_survived and incremental components as separate objectives and use NSGA-II-style Pareto ranking instead of weighted sum. This prevents the incremental components from being swamped once some organisms start balancing.
- **Speciation (optional):** Group genomes by similarity and enforce diversity across groups (NEAT-style speciation). This protects nascent innovations from being outcompeted by dominant but different strategies.

---

# Part V — Program Architecture

## 1. Module Overview

```
cellengine/
├── core/             # Math, RNG, timing, logging
├── cell/             # The cell model (state, update rules)
├── genome/           # Genome encoding, decoding, development
├── environment/      # Environment interface + cartpole implementation
├── physics/          # Stress computation, structural lattice
├── ga/               # Genetic algorithm (selection, mutation, crossover)
├── engine/           # Simulation orchestration (trial runner, evaluation)
├── gpu/              # CUDA kernels for cell update + GA batching
├── viz/              # Visualization and interpretability tools
├── analysis/         # Post-run analysis (genome comparison, phenotype mapping)
├── app/              # Entry points (headless GA runner, interactive viewer)
└── test/             # Unit tests, regression tests, determinism tests
```

## 2. `core/` — Foundations

### `core/types.h`
Fundamental types used everywhere:

```cpp
struct Vec3 { float x, y, z; };
struct VoxelCoord { int16_t x, y, z; };  // grid position
using CellIndex = uint16_t;              // max 65535 cells per organism
using GenomeFloat = float;               // genome values
constexpr int N_G = 8;                   // gene expression vector dimension
constexpr int MAX_NEIGHBORS = 26;        // 3x3x3 - 1
constexpr int MAX_CELLS = 2048;          // per organism, power of 2 for GPU alignment
```

### `core/rng.h`
Deterministic PRNG. PCG or xoshiro256. Provides:
- `float uniform(float min, float max)`
- `float gaussian(float mean, float stddev)`
- `uint64_t seed_from(uint32_t genome_id, uint32_t generation)`

All randomness in the system comes from this RNG seeded deterministically. Given the same genome_id and generation, the same organism always produces the same fitness.

### `core/timer.h`
High-resolution profiling timer. Measures per-phase time in the simulation loop. Outputs to a structured log for performance analysis.

### `core/config.h`
All compile-time and runtime configuration constants:

```cpp
struct Config {
    // Simulation timing
    float physics_dt = 0.02f;          // 50 Hz physics
    int cell_ticks_per_physics = 8;    // 400 Hz cell updates
    float cell_dt = 0.0025f;           // derived: physics_dt / cell_ticks_per_physics
    int max_physics_ticks = 500;       // 10 seconds max per trial
    int num_trials = 100;

    // GA
    int population_size = 500;
    int num_generations = 2000;
    int tournament_size = 5;
    int elitism_count = 5;
    float mutation_rate = 0.1f;
    float mutation_sigma = 0.05f;
    float crossover_rate = 0.3f;

    // Cell model
    int N_G = 8;
    float sigma_min = 0.1f;
    float sigma_max = 5.0f;
    float energy_max = 1.0f;

    // Environment
    float cart_mass = 1.0f;
    float pole_mass = 0.1f;
    float pole_half_length = 0.5f;
    float gravity = 9.81f;
    float fail_angle_deg = 15.0f;

    // GPU
    int gpu_batch_size = 64;           // organisms evaluated simultaneously on GPU
    bool use_gpu = true;
};
```

## 3. `cell/` — The Cell Model

### `cell/cell_state.h`
SoA (Structure-of-Arrays) storage for all cells in one organism:

```cpp
struct CellPopulation {
    int count;                            // number of active cells (≤ MAX_CELLS)

    // Fast state (per cell tick)
    float V_m[MAX_CELLS];                // membrane potential
    float w_rec[MAX_CELLS];              // recovery variable
    float a_mech[MAX_CELLS];             // mechanical activation (force output)

    // Medium state (per ~10 cell ticks)
    float Ca[MAX_CELLS];                 // calcium
    float sigma[MAX_CELLS];              // sensitivity
    float energy[MAX_CELLS];             // ATP analog
    float w_gap[MAX_CELLS * MAX_NEIGHBORS]; // coupling strengths (flat: cell * MAX_NEIGHBORS + k)

    // Slow state
    float G[MAX_CELLS * N_G];           // gene expression (flat: cell * N_G + g)
    float fatigue[MAX_CELLS];           // cumulative damage

    // Structural (set during development, fixed during trials)
    int16_t neighbor_idx[MAX_CELLS * MAX_NEIGHBORS]; // neighbor cell indices (-1 = no neighbor)
    int8_t  neighbor_count[MAX_CELLS];               // number of occupied neighbors
    float   mech_advantage[MAX_CELLS];               // how much force reaches ground (precomputed)
    float   force_direction_x[MAX_CELLS];            // horizontal force projection (precomputed)

    // Derived (recomputed each tick, not stored persistently)
    float I_total[MAX_CELLS];            // total input current
    float chem_out[MAX_CELLS];           // chemical secretion output
    float stress[MAX_CELLS];             // mechanical stress from physics

    // Double-buffer for V_m (read from one, write to other, swap each tick)
    float V_m_buf[2][MAX_CELLS];
    int active_buf;                      // 0 or 1
};
```

Everything is flat arrays, aligned for SIMD/GPU. No pointers, no indirection, no allocation after construction.

### `cell/cell_update.h` / `cell/cell_update.cpp`
The cell update function. This is the most performance-critical code in the system.

```cpp
// Updates all cells in one organism for one cell tick.
// Reads from V_m_buf[active_buf], writes to V_m_buf[1-active_buf].
void cell_tick(CellPopulation& pop, const CellParams& params, float dt);

// Updates medium state (every 10th cell tick).
void cell_medium_update(CellPopulation& pop, const CellParams& params, float dt_medium);

// Updates slow state (every episode boundary).
void cell_slow_update(CellPopulation& pop, const CellParams& params);
```

`cell_tick` does, for each cell i:

```
1. Compute I_elec = Σ_k w_gap[i*MAX_N+k] * (V_m_read[neighbor_idx[i*MAX_N+k]] - V_m_read[i])
2. Compute I_chem = Σ_k chem_out[neighbor_idx[i*MAX_N+k]] * chem_attenuation (from params)
3. Compute I_mech = G_mechanosensitivity[i] * stress[i]
4. Compute I_total[i] = sigma[i] * (I_elec + I_chem + I_mech) * energy[i]
5. FHN update:
     dV = (V - V³/3 - w_rec + I_total) / tau_v(G)
     dw = eps(G) * (V + a(G) - b(G)*w_rec) / tau_w(G)
     V_m_write[i] = V_m_read[i] + dV * dt
     w_rec[i] += dw * dt
6. Compute a_mech[i] = contractility(G) * tanh(Ca[i]) * energy[i]
7. Compute chem_out[i] = secretion_mag(G) * sigmoid(Ca[i]) * sign(secretion_flavor(G) - 0.5)
```

Steps 1–3 are the expensive part (neighbor reads). Steps 5–7 are pure arithmetic per cell.

### `cell/cell_params.h`
Parameters decoded from the genome for one organism. These are computed once during development and remain constant during trials.

```cpp
struct CellParams {
    // FHN dynamics as function of G (decoded from genome)
    float tau_v_base, tau_v_g_coeff;
    float epsilon_base, epsilon_g_coeff;
    float fhn_a_base, fhn_a_g_coeff;
    float fhn_b_base, fhn_b_g_coeff;

    // Calcium dynamics
    float tau_ca_base, alpha_ca_base, v_ca_thresh;

    // Coupling / communication
    float coupling_base, hebb_rate, decay_rate, adhesion_coeff;
    float chem_attenuation;
    float mechano_base, mechano_g_coeff;

    // Homeostasis
    float sigma_target, sigma_alpha, sigma_beta;
    float energy_replenish, cost_vm, cost_ca, cost_mech;
    float g_drift_rate;

    // Teaching
    int teaching_trials;          // how many trials have teacher force
    float teaching_force_scale;
    float teaching_hebb_boost;    // multiplier on hebb_rate during teaching trials
};
```

## 4. `genome/` — Genome Encoding and Development

### `genome/genome.h`
```cpp
constexpr int GENOME_SIZE = 100;

struct Genome {
    float genes[GENOME_SIZE];

    // Decode sections
    CellParams decode_cell_params() const;
    DevelopmentProgram decode_development() const;
};
```

### `genome/development.h`
```cpp
struct DevelopmentProgram {
    // Linear mapping from 5 geometric features to N_G gene expression components
    float W[N_G][5];   // weights
    float bias[N_G];   // biases

    // Assign G to each cell based on its position in the template
    void assign_G(CellPopulation& pop, const BodyTemplate& body) const;

    // Compute initial coupling strengths from G compatibility
    void init_coupling(CellPopulation& pop, float coupling_base, float adhesion_width) const;

    // Precompute structural properties (mechanical advantage, force direction)
    void precompute_structure(CellPopulation& pop, const BodyTemplate& body) const;
};
```

### `genome/body_template.h`
```cpp
struct BodyTemplate {
    // Voxel occupancy grid (which positions are occupied)
    bool occupied[GRID_X][GRID_Y][GRID_Z];

    // Per-voxel geometric features (precomputed)
    float features[MAX_CELLS][5]; // normalized: x_pos, y_pos, dist_hinge, dist_surface, neighbor_count

    // Hinge voxel mask (these cells have reduced stiffness, not genome-controlled)
    bool is_hinge[MAX_CELLS];

    // Ground contact mask (bottom cart layer, constrained to y=0)
    bool is_ground[MAX_CELLS];

    // Mapping from grid position to cell index and vice versa
    CellIndex grid_to_cell[GRID_X][GRID_Y][GRID_Z]; // -1 if unoccupied
    VoxelCoord cell_to_grid[MAX_CELLS];

    int cell_count;

    // Build the cart-pole template
    static BodyTemplate make_cartpole(int cart_lx=10, int cart_ly=3, int cart_lz=3,
                                       int pole_lx=3, int pole_ly=12, int pole_lz=3);

    // Precompute neighbor lists for a CellPopulation from this template
    void build_neighbor_lists(CellPopulation& pop) const;
};
```

## 5. `environment/` — Environment Interface and CartPole

### `environment/environment.h`
Abstract interface:

```cpp
class Environment {
public:
    virtual ~Environment() = default;

    // Reset environment state for a new trial
    virtual void reset(uint64_t seed) = 0;

    // Advance physics by one tick. Reads force from cells, writes stress to cells.
    virtual void step(CellPopulation& pop, float dt) = 0;

    // Is the trial over?
    virtual bool is_terminal() const = 0;

    // How many ticks has this trial lasted?
    virtual int ticks_elapsed() const = 0;

    // Get state for diagnostics/visualization (environment-specific)
    virtual void get_viz_state(float* out, int max_floats) const = 0;
};
```

### `environment/cartpole.h` / `environment/cartpole.cpp`
```cpp
class CartPoleEnv : public Environment {
    float x, x_dot, theta, theta_dot;   // cartpole state
    float F_organism;                     // force from organism muscles
    float F_teacher;                      // force from teacher (0 if teaching off)
    int ticks;
    bool terminal;

    // Parameters
    float M, m, l, g, fail_angle;
    bool teaching_active;
    float Kp, Kd;                        // PD teacher gains

public:
    void reset(uint64_t seed) override;
    void step(CellPopulation& pop, float dt) override;
    bool is_terminal() const override;
    int ticks_elapsed() const override;
    void get_viz_state(float* out, int max_floats) const override;

    void set_teaching(bool active, float Kp, float Kd);

    // For fitness computation
    float get_theta() const { return theta; }
    float get_F_organism() const { return F_organism; }
};
```

The `step()` method:
1. Sum muscle forces from cells: `F_organism = Σ_i a_mech[i] * force_direction_x[i] * mech_advantage[i]`
2. Compute teacher force (if active): `F_teacher = Kp * theta + Kd * theta_dot`
3. Advance ODE with F = F_organism + F_teacher using RK4.
4. Compute stress field from new (θ, θ̈, ẍ) and write to `pop.stress[]`.
5. Check terminal condition (|θ| > fail_angle).

## 6. `physics/` — Stress Computation

### `physics/stress_solver.h`
```cpp
// Compute the stress field over the voxel structure from external forces.
// Uses Gauss-Seidel relaxation on a spring lattice.
void compute_stress_field(
    CellPopulation& pop,
    const BodyTemplate& body,
    Vec3 hinge_force,       // force at hinge from pole
    Vec3 gravity_load,      // distributed gravity
    int relaxation_iters    // typically 3-5
);
```

This is a small, fast computation (~200 cells, 5 iterations). Not worth GPU-offloading for a single organism; done on CPU even in GPU mode. For batched GPU evaluation, the stress solve can be parallelized across organisms (each organism's stress solve is independent).

## 7. `ga/` — Genetic Algorithm

### `ga/population.h`
```cpp
struct Population {
    int size;
    Genome genomes[MAX_POPULATION];      // current generation
    float fitness[MAX_POPULATION];
    int sorted_indices[MAX_POPULATION];  // sorted by fitness (descending)

    void initialize(const Config& cfg, uint64_t seed);
    void sort_by_fitness();
    Genome tournament_select(RNG& rng, int tournament_size) const;
};
```

### `ga/operators.h`
```cpp
// Gaussian mutation
void mutate(Genome& g, float rate, float sigma, RNG& rng);

// Uniform crossover
Genome crossover(const Genome& a, const Genome& b, float rate, RNG& rng);

// Create next generation
void next_generation(Population& pop, const Config& cfg, RNG& rng);
```

### `ga/seeding.h`
```cpp
// Create hand-designed hint genomes that bias development toward:
// - high mechanosensitivity near hinge
// - high contractility at cart edges
// - high coupling in between
Genome make_hint_genome(int variant);  // variant 0..9
```

## 8. `engine/` — Simulation Orchestration

### `engine/evaluator.h`
```cpp
// Evaluate one organism: run development + 100 trials, return fitness.
float evaluate_organism(const Genome& genome, const BodyTemplate& body,
                        const Config& cfg, uint64_t seed);

// Evaluate a batch of organisms (CPU, multi-threaded).
void evaluate_batch_cpu(Population& pop, const BodyTemplate& body,
                        const Config& cfg, int num_threads);

// Evaluate a batch of organisms (GPU, CUDA).
void evaluate_batch_gpu(Population& pop, const BodyTemplate& body,
                        const Config& cfg);
```

### `engine/trial_runner.h`
```cpp
// Run one trial for one organism. Returns ticks survived.
int run_trial(CellPopulation& pop, CellParams& params,
              Environment& env, const Config& cfg,
              bool teaching_active);

// The inner loop: for each physics tick, run cell ticks then physics step.
// This is the most performance-critical orchestration code.
```

The trial loop:

```
for each physics tick until terminal or max_ticks:
    for each cell tick (8 times):
        cell_tick(pop, params, cell_dt)
        if (tick % 10 == 0):
            cell_medium_update(pop, params, cell_dt * 10)
    env.step(pop, physics_dt)
    ticks++

return ticks
```

### `engine/fitness.h`
```cpp
struct FitnessResult {
    float total_ticks;           // primary fitness
    float sensor_responsiveness; // incremental component
    float signal_connectivity;   // incremental component
    float force_correlation;     // incremental component
    float combined;              // weighted sum
};

FitnessResult compute_fitness(CellPopulation& pop, CellParams& params,
                               Environment& env, const Config& cfg, uint64_t seed);
```

The fitness computation includes the probe: before running the 100 trials, apply a known impulse (brief theta perturbation), measure which sensor cells activate (responsiveness), trace activation propagation to muscle cells (connectivity), and correlate muscle force direction with impulse direction (force correlation).

## 9. `gpu/` — CUDA Kernels

### `gpu/cell_kernel.cu`
The core CUDA kernel that updates all cells in a batch of organisms simultaneously.

```cpp
// One thread per cell per organism.
// Block = one organism (up to MAX_CELLS threads).
// Grid = batch_size blocks.
__global__ void cell_tick_kernel(
    CellPopulationGPU* organisms,  // array of organism data, one per block
    const CellParamsGPU* params,   // array of params, one per block
    float dt,
    int active_buf
);
```

Memory layout for GPU:
- Each organism's CellPopulation is stored contiguously in global memory.
- Neighbor indices and coupling weights are in global memory (too large for shared).
- V_m double-buffer is in global memory, accessed via active_buf index.
- Per-cell arithmetic (FHN update, calcium, energy) is register-only.
- The critical inner loop (neighbor sum for I_elec) accesses 6–26 non-contiguous V_m values per cell. This is the memory bottleneck.

Optimization strategy:
- Sort cells by spatial locality (Hilbert curve ordering on the voxel grid) so that neighbor reads have better cache behavior.
- Use `__ldg()` for cached neighbor V_m reads.
- Each warp processes 32 cells; within a warp, cells are spatially close, so their neighbors overlap → shared memory for V_m of the local neighborhood block.

### `gpu/ga_kernel.cu`
The GA operators (mutation, crossover) can also run on GPU for very large populations, but at population 500 and genome size 100 this is not a bottleneck. Keep GA operators on CPU.

### `gpu/batch_evaluator.cu`
Orchestrates batched evaluation: copies organism data to GPU, launches cell tick kernels + physics (on CPU, synchronized), copies fitness results back.

The evaluation loop for a GPU batch:
```
1. Copy batch of genomes to GPU.
2. Run development (CPU, fast at 200 cells) for each organism in batch.
3. Copy developed CellPopulations to GPU.
4. For each trial (100 times):
     For each physics tick (up to 500):
       Launch cell_tick_kernel 8 times (8 cell ticks per physics tick).
       Copy a_mech back to CPU (or compute force sum on GPU).
       Compute cartpole ODE step on CPU.
       Compute stress field on CPU.
       Copy stress to GPU.
       Check terminal conditions.
5. Copy fitness results back.
```

The bottleneck is the GPU↔CPU synchronization per physics tick (stress solve is CPU, cell update is GPU). For 64 organisms × 200 cells each = 12,800 cells per kernel launch, the kernel is very fast (~10 μs). The synchronization overhead (~5 μs per launch) is significant relative to kernel time. Mitigation: batch multiple cell ticks into one kernel launch (the 8 cell ticks per physics tick can be a single kernel with an internal loop).

## 10. `viz/` — Visualization and Interpretability

This is a separate executable from the headless GA runner. It loads a genome (or a checkpoint from a GA run), constructs an organism, and provides interactive visualization.

### `viz/organism_viewer.h`
The main visualization window. Uses SDL2 + OpenGL (or Vulkan). Renders:

**3D voxel view:** Each cell rendered as a cube. Color-coded by selectable attribute:
- **Cell type (G-space cluster):** Run k-means (k=4 or 5) on the G vectors of all cells. Color each cell by its cluster assignment. This reveals emergent cell types without imposing categories.
- **V_m (membrane potential):** Blue (negative) → white (resting) → red (positive). Shows real-time electrical activity as signals propagate through the organism.
- **[Ca] (calcium):** Dark (low) → bright yellow (high). Shows medium-timescale activation patterns.
- **Energy (E):** Green (full) → dark red (depleted). Shows which cells are working hardest.
- **Sensitivity (σ):** Transparent (low σ, habituated) → opaque (high σ, sensitized). Shows adaptation state.
- **Coupling strength:** Render edges between cells with thickness proportional to w_gap. Shows the "wiring diagram."
- **Mechanical stress:** Color by stress magnitude. Shows what the organism is "feeling."
- **Contractile output:** Color by |a_mech|. Shows which cells are generating force.

Camera: orbit, zoom, pan. Toggle layers on/off. Slice planes to see interior cells.

The cartpole state is rendered alongside: a 2D side view showing cart position and pole angle, with force arrow showing organism's muscle output.

### `viz/timeline.h`
A scrollable timeline that shows, for the current trial:

- **Top track:** Pole angle θ over time. Green band = safe zone, red = failure.
- **Middle track:** Organism's force output F over time. Overlaid with teacher force (if teaching active).
- **Bottom track:** Total organism energy over time. Shows depletion and recovery patterns.
- **Cell tracks (expandable):** Select any cell and see its V_m, [Ca], σ, E over time. This lets you trace signal paths: click a sensor cell, see when it activates, then click the downstream conductor, see the delayed activation, follow it to the muscle.

Playback controls: play/pause, step forward/backward, speed control, jump to specific trial.

### `viz/genome_viewer.h`
Visualizes the genome and its developmental consequences:

- **Genome bar chart:** The raw genome floats as a bar chart, color-coded by section (FHN params, communication, homeostasis, development, teaching). Hovering over a gene shows its decoded parameter name and value.
- **Development map:** A 3D view of the body template colored by G vector (assigned by the development program). Shows how the genome's development rules map spatial features to cell phenotypes. Sliders let you interactively adjust development parameters and see the G assignment change in real time.
- **Parameter space scatter:** Plot all cells in a 2D projection of G-space (PCA or t-SNE of the 8-dimensional G vectors). Color by position in body. This shows how many distinct "phenotypes" the genome produces and where they are in the body.
- **Phenotype profiles:** For each emergent cell type (G-space cluster), show a radar chart of the decoded behavioral parameters: excitability, contractility, mechanosensitivity, secretion, coupling, adaptation rate. This is the "cell type card" — it tells you what each emergent type does.

### `viz/evolution_dashboard.h`
Tracks GA progress across generations:

- **Fitness curve:** Best, mean, and worst fitness per generation. Shows convergence.
- **Diversity metrics:** Standard deviation of genome values across the population. Phenotypic diversity (variance in G-space cluster distributions). Warns if diversity is collapsing (premature convergence).
- **Champion gallery:** The top-5 organisms from each generation, shown as small voxel thumbnails colored by G-cluster. Click one to load it into the organism viewer.
- **Genome phylogeny:** A tree showing the lineage of the champion genome — which ancestors contributed which genes. This reveals whether the GA is refining a single lineage or combining innovations from multiple lineages.
- **Regime test results (if run):** Damage recovery curve, perturbation adaptation curve, plotted alongside baselines.

### `viz/signal_tracer.h`
An interactive tool for tracing signal paths through the organism:

1. Click a cell in the 3D view. The tracer highlights all cells whose activation is correlated with the selected cell's activation (above a threshold) across a time window.
2. The highlighted cells are connected by arrows showing the temporal sequence: which cell activates first, which second, etc.
3. This reveals "circuits" — functional signal paths from sensors to muscles — without requiring the user to manually trace neighbor connections.

The tracer can also run in "probe mode": apply a small impulse to the cartpole and highlight the cascade of cellular responses, from sensor activation through conductor relay to muscle contraction.

### `viz/comparison_view.h`
Side-by-side comparison of two organisms (e.g., before and after damage, or two different evolved solutions):

- Two 3D voxel views synchronized in rotation/zoom.
- Overlaid cartpole views showing both organisms' performance.
- Difference highlighting: cells whose G, σ, or w_gap differ by more than a threshold are highlighted.

## 11. `analysis/` — Post-Run Analysis Tools

### `analysis/phenotype_analysis.h`
After a GA run, analyze the champion organism:

- **Cell type census:** Cluster all cells by G, report cluster centers and sizes. Name clusters by their dominant behavioral parameter (e.g., "excitable-coupled" = high excitability + high coupling → nerve-like).
- **Signal path extraction:** Identify the functional paths from sensor cells to muscle cells by correlation analysis over many trials. Report path length, latency, and reliability.
- **Essentiality analysis:** For each cell (or each cell cluster), simulate removal and measure fitness impact. Report which cells are essential (removal causes >50% fitness drop) and which are redundant.
- **Adaptation profile:** Run the organism through the 100 trials and plot σ, w_gap, and performance over trials. Report whether within-lifetime adaptation improves performance and which mechanisms (habituation, Hebbian, both) contribute.

### `analysis/baseline_comparison.h`
Run standard baselines on the same cartpole task with the same evaluation protocol:

- **PID controller:** Tuned Kp, Kd for the cartpole parameters. Report ticks survived.
- **MLP controller:** 2-layer MLP (state → force), trained with REINFORCE for N episodes. Report ticks survived.
- **MLP + damage:** Delete 15% of MLP weights. Report immediate performance and recovery with fine-tuning.
- **Adaptive PID:** Model-reference adaptive control that adjusts Kp, Kd online. Report performance under pole length change.

All baselines are evaluated with the same 100-trial protocol, same initial conditions (same random seeds for pole angles), and same perturbation schedules for regime tests.

### `analysis/export.h`
Export data for external analysis (Python, Jupyter, etc.):

- Genome values as CSV.
- Per-trial fitness as CSV.
- Per-cell-per-tick state as HDF5 (for detailed analysis of a single organism's behavior).
- Evolution history (fitness per generation, champion genomes) as JSON.

## 12. `app/` — Entry Points

### `app/evolve_main.cpp`
Headless GA runner. Command-line arguments: config file, output directory, random seed, GPU on/off. Runs the full evolutionary loop, saves checkpoints (champion genome + population) every N generations, saves evolution log.

### `app/view_main.cpp`
Interactive viewer. Command-line arguments: genome file (or checkpoint file), config file. Loads a genome, builds an organism, opens the visualization window. Runs trials interactively with all visualization tools.

### `app/benchmark_main.cpp`
Runs the full comparison suite: loads the champion genome from a GA run, runs regime tests (damage, perturbation, continual adaptation), runs baselines, and outputs a comparison report.

## 13. `test/` — Testing

### `test/test_cell_update.cpp`
- FHN dynamics produce excitable spikes when I_total exceeds threshold.
- Calcium rises on V_m spikes and decays otherwise.
- Energy depletes with activity and replenishes when quiet.
- Sensitivity habituates under sustained input and sensitizes under silence.
- Coupling strengthens between co-active cells and decays between uncorrelated cells.

### `test/test_determinism.cpp`
- Run the same genome with the same seed twice. Verify bitwise identical fitness and cell state at every tick. This catches uninitialized memory, non-deterministic floating-point ordering, and RNG misuse.

### `test/test_stress.cpp`
- Apply a known force to the hinge of the template. Verify that stress decreases with distance from hinge. Verify that left-right asymmetry in stress matches left-right force direction.

### `test/test_development.cpp`
- Decode a known genome, run development, verify that G values are reproducible and that the geometric feature mapping produces expected spatial patterns (e.g., high mechanosensitivity near hinge when development weights favor it).

### `test/test_cartpole.cpp`
- With a known constant force, verify that the cartpole ODE produces the expected trajectory (compare against analytical solution for small angles).
- With zero force, verify the pole falls in the expected time from a known initial angle.

### `test/test_gpu_cpu_match.cpp`
- Run the same organism on CPU and GPU. Verify identical (or within floating-point tolerance) fitness and final cell state. This validates the CUDA implementation against the reference CPU implementation.

---

# Part VI — Build and Run

## 1. Dependencies

- **C++20** compiler (GCC 12+, Clang 15+, MSVC 2022+)
- **CUDA Toolkit 12+** (for GPU path; CPU-only build is supported without CUDA)
- **SDL2** (for visualization window)
- **OpenGL 4.5+** (for visualization rendering)
- **Dear ImGui** (for visualization UI — sliders, panels, plots)
- **ImPlot** (ImGui extension for time series and scatter plots)
- **stb_image_write** (for screenshot export)
- **nlohmann/json** (for config files and export)
- **HDF5 C++ API** (optional, for detailed cell state export)

All dependencies are available through vcpkg or system packages.

## 2. Build Targets

CMake build with four targets:

- **`cellengine_core`**: Static library. Sources: core/, cell/, genome/, environment/, physics/, engine/. No GPU, no viz dependencies. Can be tested headless on any platform.
- **`cellengine_gpu`**: Static library. Sources: gpu/. Depends on CUDA. Links with cellengine_core. Optional — excluded if CUDA not found.
- **`cellengine_viz`**: Static library. Sources: viz/. Depends on SDL2, OpenGL, ImGui. Links with cellengine_core. Optional — excluded if SDL2 not found.
- **`evolve`**: Executable. app/evolve_main.cpp. Links cellengine_core + cellengine_gpu (if available).
- **`view`**: Executable. app/view_main.cpp. Links cellengine_core + cellengine_viz.
- **`benchmark`**: Executable. app/benchmark_main.cpp. Links cellengine_core + cellengine_gpu (if available).
- **`tests`**: Executable. All test/*.cpp. Links cellengine_core.

## 3. Typical Workflow

```
# 1. Evolve organisms (headless, GPU-accelerated)
./evolve --config cartpole.json --output runs/run001/ --seed 42 --gpu

# 2. View the champion
./view --genome runs/run001/champion_gen2000.genome --config cartpole.json

# 3. Run benchmarks
./benchmark --genome runs/run001/champion_gen2000.genome --config cartpole.json --output runs/run001/benchmark/
```

## 4. Expected Performance on Target Hardware

**Intel Core Ultra 9 185H (CPU path):**
- Single organism evaluation: ~0.3 seconds
- Full GA run (500 pop × 2000 gen): ~170 hours single-threaded → ~8 hours on 22 threads
- Suitable for development, debugging, and small experiments

**NVIDIA RTX 4090 Laptop (GPU path):**
- Batch of 64 organisms: ~1 second (limited by CPU↔GPU sync for stress computation)
- Full GA run (500 pop × 2000 gen): ~4–6 hours
- Suitable for production runs; multiple experiments per day

---

# Part VII — What Success and Failure Look Like

## Success

The GA evolves organisms that balance the pole for >80% of the maximum ticks. The evolved organisms show emergent cell-type clustering in G-space (distinct sensor-like, conductor-like, muscle-like phenotypes OR novel intermediate phenotypes). Damage recovery exceeds MLP+fine-tuning baseline. The signal tracer reveals functional circuits that were not designed but evolved. Teaching-phase organisms outperform non-teaching organisms, demonstrating that environmental demonstration accelerates cellular adaptation.

## Informative Failure

The GA evolves organisms that balance poorly (<30% of max ticks) despite 2000+ generations. Analysis reveals which cell model features are bottlenecks: if no cells evolve high excitability, the FHN parameter space may need expansion. If signals don't reach muscles, the coupling model may need faster propagation. If force output is always in the wrong direction, the mechanosensitivity→contractility pathway may need a richer nonlinearity. Each failure mode points to a specific biological feature that might be computationally necessary.

## Failure That Isn't Informative

The GA fails because of engineering bugs (non-determinism, fitness landscape artifacts, ODE integration errors) or insufficient search (population too small, not enough generations, premature convergence). This is prevented by the test suite, determinism checks, diversity monitoring, and running multiple independent GA seeds.

---

# Summary

**The cell model** is one class with continuous phenotype parameters (G), not discrete types. Features included are those universal to biological cells: membrane potential (FitzHugh-Nagumo), calcium signaling, electrical coupling (gap junctions), chemical signaling (paracrine), mechanosensitivity, contractility, energy metabolism, homeostasis (habituation/sensitization), and activity-dependent connection remodeling (Hebbian). Specialization (neuron-like, muscle-like, sensor-like) is an emergent consequence of evolved G configurations.

**The environment** is a hybrid cartpole: rigid-body ODE for global dynamics, stress projection for cell sensory input, force summation for cell motor output. The cell model is environment-agnostic: it receives stress and produces force. Swapping environments requires only implementing the Environment interface.

**The GA** is the only global optimizer. It searches over ~100-float genomes that encode cell rules and development programs. Fitness is task performance (ticks balanced) with incremental smoothing components.

**Teaching** is an environmental condition: external guidance during early trials creates activation patterns that cells encode through local adaptation. No cell knows it's being taught.

**Visualization** is a first-class component: real-time 3D organism viewer with multiple color-coding modes, signal tracer, genome viewer, evolution dashboard, and comparison tools. These are essential for understanding what evolution discovers, not optional add-ons.

**The target** is not just "solve cartpole." It is: do general cell-like primitives, evolved and developed without task-specific engineering, produce organisms that balance AND show regime advantages (damage recovery, perturbation adaptation, continual learning) that parameter-matched standard controllers lack? The answer, either way, advances the science.
