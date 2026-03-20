/**
 * Input validation for SimulationConfig fields.
 */
import type { SimulationConfig } from '../types/simulation';

export interface ValidationError {
  path: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

/**
 * Validate an entire SimulationConfig and return all errors found.
 */
export function validateConfig(config: SimulationConfig): ValidationResult {
  const errors: ValidationError[] = [];

  // ---- Time validation ----
  if (config.time.init_time_total <= 0) {
    errors.push({ path: 'time.init_time_total', message: 'Init total time must be > 0' });
  }
  if (config.time.init_time_step <= 0) {
    errors.push({ path: 'time.init_time_step', message: 'Init time step must be > 0' });
  }
  if (config.time.init_time_step >= config.time.init_time_total) {
    errors.push({
      path: 'time.init_time_step',
      message: 'Init time step must be less than init total time',
    });
  }
  if (config.time.init_time_sampling < config.time.init_time_step) {
    errors.push({
      path: 'time.init_time_sampling',
      message: 'Init sampling rate must be >= init time step',
    });
  }
  if (config.time.sim_time_total <= 0) {
    errors.push({ path: 'time.sim_time_total', message: 'Sim total time must be > 0' });
  }
  if (config.time.sim_time_step <= 0) {
    errors.push({ path: 'time.sim_time_step', message: 'Sim time step must be > 0' });
  }
  if (config.time.sim_time_step >= config.time.sim_time_total) {
    errors.push({
      path: 'time.sim_time_step',
      message: 'Sim time step must be less than sim total time',
    });
  }
  if (config.time.sim_time_sampling < config.time.sim_time_step) {
    errors.push({
      path: 'time.sim_time_sampling',
      message: 'Sim sampling rate must be >= sim time step',
    });
  }

  // ---- Space validation ----
  if (config.space.cell_radius <= 0) {
    errors.push({ path: 'space.cell_radius', message: 'Cell radius must be > 0' });
  }
  if (config.space.grid_size < 2) {
    errors.push({ path: 'space.grid_size', message: 'Grid size must be >= 2' });
  }
  if (config.space.world_len <= 0) {
    errors.push({ path: 'space.world_len', message: 'World length must be > 0' });
  }

  // ---- Ion validation ----
  const ionKeys = ['Na', 'K', 'Cl', 'Ca', 'M', 'H'] as const;
  for (const ionKey of ionKeys) {
    const ion = config.ions[ionKey];
    if (ion.c_env < 0) {
      errors.push({
        path: `ions.${ionKey}.c_env`,
        message: `${ion.name} environmental concentration must be >= 0`,
      });
    }
    if (ion.c_cell < 0) {
      errors.push({
        path: `ions.${ionKey}.c_cell`,
        message: `${ion.name} intracellular concentration must be >= 0`,
      });
    }
    if (ion.Dm < 0) {
      errors.push({
        path: `ions.${ionKey}.Dm`,
        message: `${ion.name} membrane diffusion (Dm) must be >= 0`,
      });
    }
    if (ion.Do < 0) {
      errors.push({
        path: `ions.${ionKey}.Do`,
        message: `${ion.name} free diffusion (Do) must be >= 0`,
      });
    }
  }

  // ---- Channel validation ----
  for (let i = 0; i < config.channels.length; i++) {
    const ch = config.channels[i];
    if (ch.enabled) {
      if (ch.max_Dm < 0) {
        errors.push({
          path: `channels[${i}].max_Dm`,
          message: `${ch.name}: max Dm must be >= 0`,
        });
      }
      if (!isFinite(ch.V_half)) {
        errors.push({
          path: `channels[${i}].V_half`,
          message: `${ch.name}: V_half must be a finite number`,
        });
      }
      if (!isFinite(ch.V_slope)) {
        errors.push({
          path: `channels[${i}].V_slope`,
          message: `${ch.name}: V_slope must be a finite number`,
        });
      }
    }
  }

  // ---- Intervention validation ----
  for (let i = 0; i < config.interventions.length; i++) {
    const intv = config.interventions[i];
    if (intv.enabled) {
      if (intv.apply_start < 0) {
        errors.push({
          path: `interventions[${i}].apply_start`,
          message: `${intv.name}: start time must be >= 0`,
        });
      }
      if (intv.apply_end <= intv.apply_start) {
        errors.push({
          path: `interventions[${i}].apply_end`,
          message: `${intv.name}: end time must be > start time`,
        });
      }
    }
  }

  // ---- Physics validation ----
  if (config.physics.T <= 0) {
    errors.push({ path: 'physics.T', message: 'Temperature must be > 0 K' });
  }
  if (config.physics.young_modulus <= 0 && config.physics.is_deformation) {
    errors.push({ path: 'physics.young_modulus', message: "Young's modulus must be > 0 when deformation is enabled" });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate a single field by path and value.
 * Returns an error message string or null if valid.
 */
export function validateField(path: string, value: number): string | null {
  if (!isFinite(value)) {
    return 'Value must be a finite number';
  }

  // Time fields
  if (path.includes('time_total') || path.includes('total_time')) {
    if (value <= 0) return 'Total time must be > 0';
  }
  if (path.includes('time_step')) {
    if (value <= 0) return 'Time step must be > 0';
  }
  if (path.includes('time_sampling') || path.includes('sampling_rate')) {
    if (value <= 0) return 'Sampling rate must be > 0';
  }

  // Space fields
  if (path.includes('cell_radius')) {
    if (value <= 0) return 'Cell radius must be > 0';
  }
  if (path.includes('grid_size')) {
    if (value < 2) return 'Grid size must be >= 2';
    if (!Number.isInteger(value)) return 'Grid size must be an integer';
  }
  if (path.includes('world_len')) {
    if (value <= 0) return 'World length must be > 0';
  }

  // Ion concentrations
  if (path.includes('c_env') || path.includes('c_cell')) {
    if (value < 0) return 'Concentration must be >= 0';
  }

  // Diffusion constants
  if (path.match(/\.Dm$/) || path.match(/\.Do$/) || path.includes('max_Dm') || path.match(/Dm_/)) {
    if (value < 0) return 'Diffusion coefficient must be >= 0';
  }

  // Channel voltages
  if (path.includes('V_half') || path.includes('V_slope')) {
    if (!isFinite(value)) return 'Must be a finite number (no NaN/Infinity)';
  }

  // Temperature
  if (path.endsWith('.T') || path === 'T') {
    if (value <= 0) return 'Temperature must be > 0 K';
  }

  return null;
}
