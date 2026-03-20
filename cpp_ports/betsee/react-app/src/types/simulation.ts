/**
 * Complete type definitions for BETSE simulation configuration.
 * Mirrors the YAML-backed Parameters class from betse.science.parameters.
 */

// ---- Enumerations ----

export enum CellLatticeType {
  HEX = 'hex',
  SQUARE = 'square',
}

export enum IonProfileType {
  BASIC = 'basic',
  BASIC_CA = 'basic_ca',
  MAMMAL = 'mammal',
  AMPHIBIAN = 'amphibian',
  CUSTOM = 'custom',
}

export enum SimmerState {
  UNQUEUED = 'UNQUEUED',
  QUEUED = 'QUEUED',
  MODELLING = 'MODELLING',
  EXPORTING = 'EXPORTING',
  PAUSED = 'PAUSED',
  STOPPING = 'STOPPING',
  FINISHED = 'FINISHED',
}

export enum SimPhaseKind {
  SEED = 'seed',
  INIT = 'init',
  SIM = 'sim',
}

export enum ChannelType {
  NA_V = 'Nav',
  K_V = 'Kv',
  K_IR = 'Kir',
  FUN = 'Fun',
  CA_V = 'Cav',
  NA_LEAK = 'NaLeak',
  K_LEAK = 'KLeak',
  CL_LEAK = 'ClLeak',
  CA_LEAK = 'CaLeak',
}

// ---- Time Configuration ----

export interface TimeConfig {
  init_time_total: number;
  init_time_step: number;
  init_time_sampling: number;
  sim_time_total: number;
  sim_time_step: number;
  sim_time_sampling: number;
}

// ---- Space / Cell Configuration ----

export interface SpaceConfig {
  cell_radius: number;
  cell_lattice_type: CellLatticeType;
  cell_lattice_disorder: number;
  grid_size: number;
  is_ecm: boolean;
  world_len: number;
}

// ---- Ion Species Configuration ----

export interface IonSpecies {
  name: string;
  symbol: string;
  z: number; // valence
  Dm: number; // membrane diffusion constant
  Do: number; // free diffusion constant
  c_env: number; // environmental concentration (mM)
  c_cell: number; // intracellular concentration (mM)
}

export interface IonConfig {
  ion_profile: IonProfileType;
  Na: IonSpecies;
  K: IonSpecies;
  Cl: IonSpecies;
  Ca: IonSpecies;
  M: IonSpecies; // anionic proteins
  H: IonSpecies; // protons
}

// ---- Channel Configuration ----

export interface ChannelConfig {
  type: ChannelType;
  name: string;
  enabled: boolean;
  max_Dm: number;
  channel_class: string;
  apply_to: string[];
  V_half: number;  // half-activation voltage (mV)
  V_slope: number; // slope factor (mV)
  V_tau: number;   // time constant voltage (mV)
  tau_max: number;  // max time constant (s)
}

// ---- Tissue Profile Configuration ----

export interface TissueProfile {
  name: string;
  picker_image_filename: string;
  Dm_Na: number;
  Dm_K: number;
  Dm_Cl: number;
  Dm_Ca: number;
  Dm_M: number;
  Dm_H: number;
  enabled: boolean;
}

export interface TissueConfig {
  default_profile: TissueProfile;
  custom_profiles: TissueProfile[];
}

// ---- Intervention Configuration ----

export interface Intervention {
  name: string;
  enabled: boolean;
  type: 'global' | 'targeted';
  target_tissue: string;
  change_Na_mem: number;
  change_K_mem: number;
  change_Cl_mem: number;
  change_Ca_mem: number;
  apply_start: number;
  apply_end: number;
  change_rate: number;
}

// ---- Network / GRN Configuration ----

export interface GRNConfig {
  enabled: boolean;
  gene_regulatory_network_file: string;
  reaction_network_file: string;
  substances: SubstanceConfig[];
}

export interface SubstanceConfig {
  name: string;
  Dm: number;
  Do: number;
  c_env: number;
  c_cell: number;
  z: number;
  decay_rate: number;
  growth_rate: number;
}

// ---- Physics Configuration ----

export interface PhysicsConfig {
  is_electroosmosis: boolean;
  is_deformation: boolean;
  is_pressure: boolean;
  deform_osmo: boolean;
  deform_electro: boolean;
  p_cells: number; // hydrostatic pressure in cells
  young_modulus: number;
  mu_membrane: number; // membrane viscosity
  fixed_cluster_boundary: boolean;
  T: number; // temperature (K)
}

// ---- Export / Visualization Configuration ----

export interface ExportConfig {
  is_show_cell_indices: boolean;
  single_cell_index: number;
  colormap_diverging_name: string;
  colormap_sequential_name: string;
  colormap_gj_name: string;
  colormap_grn_name: string;
  animations: AnimationExport[];
  plots_cell: PlotExport[];
  plots_cells: PlotExport[];
  csvs: CSVExport[];
}

export interface AnimationExport {
  name: string;
  enabled: boolean;
  type: string;
  colorbar: boolean;
  save_format: string;
  dpi: number;
}

export interface PlotExport {
  name: string;
  enabled: boolean;
  type: string;
  save_format: string;
  dpi: number;
}

export interface CSVExport {
  name: string;
  enabled: boolean;
  type: string;
}

// ---- Path / File Configuration ----

export interface PathConfig {
  seed_pickle_basename: string;
  init_pickle_basename: string;
  init_pickle_dirname_relative: string;
  init_export_dirname_relative: string;
  sim_pickle_basename: string;
  sim_pickle_dirname_relative: string;
  sim_export_dirname_relative: string;
}

// ---- Complete Simulation Configuration ----

export interface SimulationConfig {
  filename: string | null;
  isDirty: boolean;
  time: TimeConfig;
  space: SpaceConfig;
  ions: IonConfig;
  channels: ChannelConfig[];
  tissue: TissueConfig;
  interventions: Intervention[];
  grn: GRNConfig;
  physics: PhysicsConfig;
  exports: ExportConfig;
  paths: PathConfig;
}

// ---- Simulation Results ----

export interface CellData {
  x: number;
  y: number;
  vmem: number;
  neighbors: number[];
}

export interface TimeSeriesPoint {
  time: number;
  value: number;
}

export interface SimulationResults {
  cells: CellData[];
  vmem_timeseries: TimeSeriesPoint[];
  ion_concentrations: Record<string, TimeSeriesPoint[]>;
  current_density: { x: number; y: number; jx: number; jy: number }[];
  gap_junction_states: { from: number; to: number; conductance: number }[];
  time_steps_completed: number;
  total_time_steps: number;
}

// ---- Log Entry ----

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  message: string;
  source: string;
}

// ---- Application State ----

export interface AppState {
  config: SimulationConfig;
  simmerState: SimmerState;
  results: SimulationResults | null;
  logs: LogEntry[];
  progress: number;
  currentPhase: SimPhaseKind | null;
}
