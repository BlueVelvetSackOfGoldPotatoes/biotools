/**
 * Mock simulation engine for demo purposes.
 * In production this would communicate with a BETSE C++ backend via WebSocket/REST.
 *
 * All mock results are derived from actual configuration parameters:
 *   - Ion concentrations from the config's ion profile
 *   - Resting Vmem via a simplified Goldman equation, scaled by temperature
 *   - Cell positions from grid_size, world_len, cell_radius, and lattice type
 *   - Gap junction connections from actual neighbor distances
 *   - Channel-dependent action-potential spikes when Nav/Kv are enabled
 */
import {
  SimulationConfig,
  SimulationResults,
  CellData,
  TimeSeriesPoint,
  LogEntry,
  CellLatticeType,
  IonProfileType,
  ChannelType,
} from '../types/simulation';
import { generateId } from './format';

// ---------------------------------------------------------------------------
// Ion profile defaults (match BETSE defaults for each profile type)
// ---------------------------------------------------------------------------

interface IonConcentrations {
  Na_in: number;  Na_out: number;
  K_in: number;   K_out: number;
  Cl_in: number;  Cl_out: number;
  Ca_in: number;  Ca_out: number;
}

/**
 * Return ion concentrations from the config, falling back to BETSE profile
 * defaults when a named profile is selected.
 */
function getIonConcentrations(config: SimulationConfig): IonConcentrations {
  // Always read directly from the config objects -- they already reflect the
  // chosen profile (see presets.ts).  The profile type is only consulted when
  // concentrations haven't been customised, but because the UI binds to the
  // config.ions.{Na,K,...} objects we can just read them.
  return {
    Na_in:  config.ions.Na.c_cell,
    Na_out: config.ions.Na.c_env,
    K_in:   config.ions.K.c_cell,
    K_out:  config.ions.K.c_env,
    Cl_in:  config.ions.Cl.c_cell,
    Cl_out: config.ions.Cl.c_env,
    Ca_in:  config.ions.Ca.c_cell,
    Ca_out: config.ions.Ca.c_env,
  };
}

// ---------------------------------------------------------------------------
// Goldman equation helpers
// ---------------------------------------------------------------------------

/** Relative permeabilities (P_Na / P_K ≈ 0.01, P_Cl / P_K ≈ 0.1). */
const P_K  = 1.0;
const P_Na = 0.01;
const P_Cl = 0.1;

/**
 * Simplified Goldman-Hodgkin-Katz voltage (mV).
 *
 *   Vmem = 26.7 * ln( (P_K*[K]o + P_Na*[Na]o + P_Cl*[Cl]i) /
 *                       (P_K*[K]i + P_Na*[Na]i + P_Cl*[Cl]o) )
 *
 * The result is scaled by (T / 310) to account for the configured temperature.
 * T is read from config.physics.T (in Kelvin).
 */
function computeRestingVmem(config: SimulationConfig): number {
  const ions = getIonConcentrations(config);
  const numerator   = P_K * ions.K_out  + P_Na * ions.Na_out + P_Cl * ions.Cl_in;
  const denominator = P_K * ions.K_in   + P_Na * ions.Na_in  + P_Cl * ions.Cl_out;

  // Avoid log(0) or log(negative)
  if (numerator <= 0 || denominator <= 0) return -70;

  const T = config.physics.T; // Kelvin (default 310 K = 37 C)
  const tempScale = T / 310.0;
  const vmem = 26.7 * Math.log(numerator / denominator) * tempScale;
  return vmem; // mV
}

// ---------------------------------------------------------------------------
// Cell lattice generation (uses actual config geometry)
// ---------------------------------------------------------------------------

/**
 * Generate a lattice of cells using configuration parameters:
 *   - cell_radius, grid_size, world_len for spacing
 *   - cell_lattice_type for hex vs square
 *   - cell_lattice_disorder for positional noise
 *   - Goldman Vmem as the initial membrane potential per cell
 */
export function generateCellLattice(config: SimulationConfig): CellData[] {
  const cells: CellData[] = [];
  const gridSize  = config.space.grid_size;
  const worldLen  = config.space.world_len;
  const isHex     = config.space.cell_lattice_type === CellLatticeType.HEX;
  const disorder  = config.space.cell_lattice_disorder;

  // Cell spacing derived from world length and grid size
  const spacing = worldLen / gridSize;

  // Compute the Goldman resting potential from actual ion concentrations
  const restingVmem = computeRestingVmem(config);

  // Noise amplitude proportional to lattice disorder (more disorder -> more
  // spatial inhomogeneity in the initial voltage).
  const vmemNoiseAmp = 2.0 + disorder * 8.0; // mV

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      let x = (col + 0.5) * spacing;
      let y = (row + 0.5) * spacing;

      // Hex lattice: offset every other row by half a spacing
      if (isHex && row % 2 === 1) {
        x += spacing * 0.5;
      }

      // Apply positional disorder proportional to spacing
      x += (Math.random() - 0.5) * spacing * disorder;
      y += (Math.random() - 0.5) * spacing * disorder;

      // Per-cell Vmem: resting potential + noise scaled by disorder
      const vmem = restingVmem + (Math.random() - 0.5) * vmemNoiseAmp;

      cells.push({ x, y, vmem, neighbors: [] });
    }
  }

  // ---- Assign neighbors based on actual inter-cell distance ----
  // The neighbor cutoff is derived from cell spacing.  For a hex lattice the
  // nearest neighbor distance is ~spacing; we allow up to 1.5x for hex
  // (captures the 6 nearest neighbors) and 1.5x for square (captures the 4
  // nearest + diagonals when disorder is high).
  const neighborCutoff = isHex ? spacing * 1.5 : spacing * 1.5;

  for (let i = 0; i < cells.length; i++) {
    for (let j = i + 1; j < cells.length; j++) {
      const dx = cells[i].x - cells[j].x;
      const dy = cells[i].y - cells[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < neighborCutoff) {
        cells[i].neighbors.push(j);
        cells[j].neighbors.push(i);
      }
    }
  }

  return cells;
}

// ---------------------------------------------------------------------------
// Channel helpers
// ---------------------------------------------------------------------------

/** Check whether a specific channel type is enabled in the config. */
function channelEnabled(config: SimulationConfig, type: ChannelType): boolean {
  return config.channels.some(ch => ch.type === type && ch.enabled);
}

/** Return the half-activation voltage for a given channel type, or a default. */
function channelVHalf(config: SimulationConfig, type: ChannelType, fallback: number): number {
  const ch = config.channels.find(c => c.type === type);
  return ch ? ch.V_half : fallback;
}

/**
 * Boltzmann activation function:  m_inf = 1 / (1 + exp((V_half - V) / V_slope))
 * Returns a value in [0, 1].
 */
function boltzmannActivation(v: number, vHalf: number, vSlope: number): number {
  return 1.0 / (1.0 + Math.exp((vHalf - v) / vSlope));
}

// ---------------------------------------------------------------------------
// Main mock results generator
// ---------------------------------------------------------------------------

/**
 * Generate mock simulation results derived from actual configuration parameters.
 */
export function generateMockResults(config: SimulationConfig): SimulationResults {
  const cells = generateCellLattice(config);
  const totalSteps = Math.round(config.time.sim_time_total / config.time.sim_time_sampling);
  const dt = config.time.sim_time_sampling;
  const totalTime = config.time.sim_time_total;

  const ions = getIonConcentrations(config);
  const restingVmem = computeRestingVmem(config);
  const disorder = config.space.cell_lattice_disorder;

  // Noise amplitude scales with lattice disorder
  const vmemNoise = 0.3 + disorder * 1.5;   // mV per timestep
  const concNoiseFrac = 0.001 + disorder * 0.005; // fraction of concentration

  // --- Detect active voltage-gated channels ---
  const hasNav = channelEnabled(config, ChannelType.NA_V);
  const hasKv  = channelEnabled(config, ChannelType.K_V);
  const hasCav = channelEnabled(config, ChannelType.CA_V);
  const hasExcitableChannels = hasNav && hasKv;

  // Nav / Kv parameters from config
  const navVHalf  = channelVHalf(config, ChannelType.NA_V, -40);
  const kvVHalf   = channelVHalf(config, ChannelType.K_V, -20);
  const navSlope  = (() => {
    const ch = config.channels.find(c => c.type === ChannelType.NA_V);
    return ch ? ch.V_slope : -7;
  })();
  const kvSlope = (() => {
    const ch = config.channels.find(c => c.type === ChannelType.K_V);
    return ch ? ch.V_slope : 10;
  })();

  // -----------------------------------------------------------------
  // Generate Vmem time series for the monitored cell
  // -----------------------------------------------------------------
  const vmemTimeseries: TimeSeriesPoint[] = [];
  let vmem = restingVmem;

  // For excitable channels, schedule action-potential spikes
  // Spikes occur at ~25%, 50%, 75% of the simulation time
  const spikeTimeFractions = hasExcitableChannels ? [0.25, 0.5, 0.75] : [];
  const spikeDuration = Math.max(0.5, totalTime * 0.015); // seconds

  // Gating variables for HH-like dynamics
  let mGate = 0; // Na activation
  let nGate = 0; // K activation

  for (let i = 0; i <= totalSteps; i++) {
    const t = i * dt;

    if (hasExcitableChannels) {
      // Check if we are near a spike time
      let stimulusCurrent = 0;
      for (const frac of spikeTimeFractions) {
        const tSpike = totalTime * frac;
        if (t >= tSpike && t < tSpike + spikeDuration * 0.3) {
          stimulusCurrent = 15; // mV-equivalent depolarising push
        }
      }

      // Simplified HH-style dynamics
      const mInf = boltzmannActivation(vmem, navVHalf, navSlope);
      const nInf = boltzmannActivation(vmem, kvVHalf, kvSlope);

      // Time constants (simplified)
      const tauM = 0.5;  // fast Na
      const tauN = 2.0;  // slower K

      mGate += (mInf - mGate) / (tauM / dt);
      nGate += (nInf - nGate) / (tauN / dt);

      // Na current depolarises, K current repolarises
      const iNa = mGate * (50 - vmem) * 0.4;   // drives toward +50 mV (ENa)
      const iK  = nGate * (-90 - vmem) * 0.3;   // drives toward -90 mV (EK)
      const iLeak = (restingVmem - vmem) * 0.05; // leak drives toward rest

      vmem += iNa + iK + iLeak + stimulusCurrent * dt;
      vmem += (Math.random() - 0.5) * vmemNoise;

      // Clamp to biophysically plausible range
      vmem = Math.max(-100, Math.min(60, vmem));
    } else {
      // No excitable channels: membrane voltage stays near resting with small
      // relaxation dynamics and noise
      vmem += (restingVmem - vmem) * 0.02 + (Math.random() - 0.5) * vmemNoise;
    }

    vmemTimeseries.push({ time: t, value: vmem });
  }

  // -----------------------------------------------------------------
  // Generate ion concentration time series (intracellular)
  // -----------------------------------------------------------------
  const ionConcentrations: Record<string, TimeSeriesPoint[]> = {};

  // Starting concentrations from config
  const ionStarting: Record<string, { cIn: number; cOut: number }> = {
    Na: { cIn: ions.Na_in, cOut: ions.Na_out },
    K:  { cIn: ions.K_in,  cOut: ions.K_out },
    Cl: { cIn: ions.Cl_in, cOut: ions.Cl_out },
    Ca: { cIn: ions.Ca_in, cOut: ions.Ca_out },
  };

  for (const ionName of ['Na', 'K', 'Cl', 'Ca']) {
    const series: TimeSeriesPoint[] = [];
    const { cIn: baseConc } = ionStarting[ionName];
    let conc = baseConc;

    for (let i = 0; i <= totalSteps; i++) {
      const t = i * dt;
      const noise = (Math.random() - 0.5) * Math.abs(conc) * concNoiseFrac;

      if (hasExcitableChannels) {
        // During spike windows, shift concentrations transiently
        let inSpike = false;
        for (const frac of spikeTimeFractions) {
          const tSpike = totalTime * frac;
          if (t >= tSpike && t < tSpike + spikeDuration) {
            inSpike = true;
          }
        }

        if (inSpike) {
          // Na influx during spike
          if (ionName === 'Na') conc += (baseConc * 1.8 - conc) * 0.02;
          // K efflux during spike
          if (ionName === 'K') conc += (baseConc * 0.92 - conc) * 0.015;
          // Ca influx if Cav enabled
          if (ionName === 'Ca' && hasCav) conc += (baseConc * 8 - conc) * 0.02;
          // Cl mostly passive
          if (ionName === 'Cl') conc += (baseConc * 1.02 - conc) * 0.01;
        } else {
          // Recovery toward baseline (pump activity)
          conc += (baseConc - conc) * 0.008;
        }
      } else {
        // Without excitable channels: concentrations are very stable
        // K stays almost constant; Na shows tiny oscillations
        if (ionName === 'Na') {
          conc += (baseConc - conc) * 0.005 + Math.sin(t * 0.5) * baseConc * 0.001;
        } else if (ionName === 'K') {
          // K is tightly regulated -- almost no change
          conc += (baseConc - conc) * 0.01;
        } else {
          conc += (baseConc - conc) * 0.005;
        }
      }

      conc += noise;
      series.push({ time: t, value: Math.max(0, conc) });
    }

    ionConcentrations[ionName] = series;
  }

  // -----------------------------------------------------------------
  // Update final cell Vmem with a spatial pattern centred in the tissue
  // -----------------------------------------------------------------
  const halfWorld = config.space.world_len / 2;
  for (let i = 0; i < cells.length; i++) {
    const dx = cells[i].x - halfWorld;
    const dy = cells[i].y - halfWorld;
    const distFromCenter = Math.sqrt(dx * dx + dy * dy);
    const normalizedDist = distFromCenter / halfWorld;

    // Centre cells are slightly more depolarised (as if a signalling centre)
    const spatialShift = 15 * Math.exp(-normalizedDist * 3);
    cells[i].vmem = restingVmem + spatialShift + (Math.random() - 0.5) * (2 + disorder * 6);
  }

  // -----------------------------------------------------------------
  // Generate current density field (driven by spatial Vmem gradients)
  // -----------------------------------------------------------------
  const spacing = config.space.world_len / config.space.grid_size;
  const currentDensity = cells.map((cell, idx) => {
    // Compute a local gradient from neighbor Vmem differences
    let jx = 0;
    let jy = 0;
    if (cell.neighbors.length > 0) {
      for (const nIdx of cell.neighbors) {
        const neighbor = cells[nIdx];
        const ddx = neighbor.x - cell.x;
        const ddy = neighbor.y - cell.y;
        const dist = Math.sqrt(ddx * ddx + ddy * ddy);
        if (dist > 0) {
          const dV = neighbor.vmem - cell.vmem;
          // Current proportional to voltage difference / distance
          jx += (dV / dist) * (ddx / dist) * 1e-4;
          jy += (dV / dist) * (ddy / dist) * 1e-4;
        }
      }
      jx /= cell.neighbors.length;
      jy /= cell.neighbors.length;
    }
    return { x: cell.x, y: cell.y, jx, jy };
  });

  // -----------------------------------------------------------------
  // Generate gap junction states using actual neighbor distances
  // -----------------------------------------------------------------
  const gapJunctions: { from: number; to: number; conductance: number }[] = [];
  for (let i = 0; i < cells.length; i++) {
    for (const j of cells[i].neighbors) {
      if (i < j) {
        const dx = cells[i].x - cells[j].x;
        const dy = cells[i].y - cells[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Conductance decreases with distance and is modulated by voltage
        // difference (gap junctions close at large dV).
        const dV = Math.abs(cells[i].vmem - cells[j].vmem);
        const distanceFactor = Math.exp(-dist / spacing);         // ~1 for nearest neighbors
        const voltageSensitivity = Math.exp(-dV / 20);            // closes at large dV
        const baseConductance = 1.0e-9;                           // Siemens

        gapJunctions.push({
          from: i,
          to: j,
          conductance: baseConductance * distanceFactor * voltageSensitivity
                       * (0.8 + Math.random() * 0.4), // slight biological variability
        });
      }
    }
  }

  return {
    cells,
    vmem_timeseries: vmemTimeseries,
    ion_concentrations: ionConcentrations,
    current_density: currentDensity,
    gap_junction_states: gapJunctions,
    time_steps_completed: totalSteps,
    total_time_steps: totalSteps,
  };
}

// ---------------------------------------------------------------------------
// Log entry helper
// ---------------------------------------------------------------------------

/**
 * Create a log entry.
 */
export function createLogEntry(
  level: LogEntry['level'],
  message: string,
  source: string = 'BETSE',
): LogEntry {
  return {
    id: generateId(),
    timestamp: new Date(),
    level,
    message,
    source,
  };
}
