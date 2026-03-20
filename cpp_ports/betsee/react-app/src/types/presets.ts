/**
 * Parameter presets for common simulation scenarios.
 */
import {
  SimulationConfig,
  CellLatticeType,
  IonProfileType,
  ChannelType,
} from './simulation';

export const DEFAULT_ION_NA = {
  name: 'Sodium',
  symbol: 'Na+',
  z: 1,
  Dm: 1.0e-18,
  Do: 1.33e-9,
  c_env: 145.0,
  c_cell: 12.0,
};

export const DEFAULT_ION_K = {
  name: 'Potassium',
  symbol: 'K+',
  z: 1,
  Dm: 1.0e-18,
  Do: 1.96e-9,
  c_env: 5.0,
  c_cell: 140.0,
};

export const DEFAULT_ION_CL = {
  name: 'Chloride',
  symbol: 'Cl-',
  z: -1,
  Dm: 1.0e-18,
  Do: 2.03e-9,
  c_env: 140.0,
  c_cell: 15.0,
};

export const DEFAULT_ION_CA = {
  name: 'Calcium',
  symbol: 'Ca2+',
  z: 2,
  Dm: 1.0e-18,
  Do: 7.92e-10,
  c_env: 2.0,
  c_cell: 0.0001,
};

export const DEFAULT_ION_M = {
  name: 'Anionic Proteins',
  symbol: 'M-',
  z: -1,
  Dm: 0.0,
  Do: 0.0,
  c_env: 10.0,
  c_cell: 135.0,
};

export const DEFAULT_ION_H = {
  name: 'Protons',
  symbol: 'H+',
  z: 1,
  Dm: 1.0e-18,
  Do: 9.31e-9,
  c_env: 0.0000398,
  c_cell: 0.0000631,
};

export function createDefaultConfig(): SimulationConfig {
  return {
    filename: null,
    isDirty: false,
    time: {
      init_time_total: 10.0,
      init_time_step: 1.0e-3,
      init_time_sampling: 0.1,
      sim_time_total: 60.0,
      sim_time_step: 1.0e-3,
      sim_time_sampling: 0.1,
    },
    space: {
      cell_radius: 5.0e-6,
      cell_lattice_type: CellLatticeType.HEX,
      cell_lattice_disorder: 0.4,
      grid_size: 25,
      is_ecm: false,
      world_len: 300.0e-6,
    },
    ions: {
      ion_profile: IonProfileType.BASIC,
      Na: { ...DEFAULT_ION_NA },
      K: { ...DEFAULT_ION_K },
      Cl: { ...DEFAULT_ION_CL },
      Ca: { ...DEFAULT_ION_CA },
      M: { ...DEFAULT_ION_M },
      H: { ...DEFAULT_ION_H },
    },
    channels: [
      {
        type: ChannelType.NA_V,
        name: 'Voltage-gated Na+',
        enabled: false,
        max_Dm: 1.0e-15,
        channel_class: 'Nav1p2',
        apply_to: ['all'],
        V_half: -40.0,
        V_slope: -7.0,
        V_tau: -40.0,
        tau_max: 0.001,
      },
      {
        type: ChannelType.K_V,
        name: 'Voltage-gated K+',
        enabled: false,
        max_Dm: 5.0e-16,
        channel_class: 'Kv1p2',
        apply_to: ['all'],
        V_half: -20.0,
        V_slope: 10.0,
        V_tau: -20.0,
        tau_max: 0.005,
      },
      {
        type: ChannelType.K_IR,
        name: 'Inward-rectifier K+',
        enabled: false,
        max_Dm: 5.0e-16,
        channel_class: 'Kir2p1',
        apply_to: ['all'],
        V_half: -90.0,
        V_slope: -10.0,
        V_tau: -90.0,
        tau_max: 0.01,
      },
      {
        type: ChannelType.FUN,
        name: 'Funny (HCN) channel',
        enabled: false,
        max_Dm: 1.0e-16,
        channel_class: 'HCN2',
        apply_to: ['all'],
        V_half: -80.0,
        V_slope: -10.0,
        V_tau: -80.0,
        tau_max: 0.1,
      },
      {
        type: ChannelType.CA_V,
        name: 'Voltage-gated Ca2+',
        enabled: false,
        max_Dm: 1.0e-16,
        channel_class: 'Cav1p2',
        apply_to: ['all'],
        V_half: -10.0,
        V_slope: -6.0,
        V_tau: -10.0,
        tau_max: 0.002,
      },
      {
        type: ChannelType.NA_LEAK,
        name: 'Na+ leak',
        enabled: true,
        max_Dm: 1.0e-18,
        channel_class: 'NaLeak',
        apply_to: ['all'],
        V_half: 0.0,
        V_slope: 0.0,
        V_tau: 0.0,
        tau_max: 0.0,
      },
      {
        type: ChannelType.K_LEAK,
        name: 'K+ leak',
        enabled: true,
        max_Dm: 2.5e-17,
        channel_class: 'KLeak',
        apply_to: ['all'],
        V_half: 0.0,
        V_slope: 0.0,
        V_tau: 0.0,
        tau_max: 0.0,
      },
    ],
    tissue: {
      default_profile: {
        name: 'Default',
        picker_image_filename: '',
        Dm_Na: 1.0e-18,
        Dm_K: 2.5e-17,
        Dm_Cl: 2.0e-18,
        Dm_Ca: 1.0e-18,
        Dm_M: 0.0,
        Dm_H: 1.0e-18,
        enabled: true,
      },
      custom_profiles: [],
    },
    interventions: [],
    grn: {
      enabled: false,
      gene_regulatory_network_file: '',
      reaction_network_file: '',
      substances: [],
    },
    physics: {
      is_electroosmosis: false,
      is_deformation: false,
      is_pressure: false,
      deform_osmo: false,
      deform_electro: false,
      p_cells: 0.0,
      young_modulus: 50.0,
      mu_membrane: 1.0e-4,
      fixed_cluster_boundary: true,
      T: 310.0,
    },
    exports: {
      is_show_cell_indices: false,
      single_cell_index: 0,
      colormap_diverging_name: 'RdBu_r',
      colormap_sequential_name: 'viridis',
      colormap_gj_name: 'bone',
      colormap_grn_name: 'jet',
      animations: [
        { name: 'Vmem Animation', enabled: true, type: 'Vmem', colorbar: true, save_format: 'png', dpi: 150 },
        { name: 'Ca2+ Animation', enabled: false, type: 'Ca_cell', colorbar: true, save_format: 'png', dpi: 150 },
        { name: 'Current Animation', enabled: false, type: 'I_tot', colorbar: true, save_format: 'png', dpi: 150 },
      ],
      plots_cell: [
        { name: 'Vmem Time Series', enabled: true, type: 'Vmem', save_format: 'png', dpi: 150 },
        { name: 'Ion Concentrations', enabled: false, type: 'ions', save_format: 'png', dpi: 150 },
      ],
      plots_cells: [
        { name: 'Vmem Heatmap', enabled: true, type: 'Vmem_2D', save_format: 'png', dpi: 150 },
        { name: 'Ca2+ Heatmap', enabled: false, type: 'Ca_2D', save_format: 'png', dpi: 150 },
      ],
      csvs: [
        { name: 'Vmem CSV', enabled: true, type: 'Vmem' },
        { name: 'Ion CSV', enabled: false, type: 'ions' },
      ],
    },
    paths: {
      seed_pickle_basename: 'seed.betse.gz',
      init_pickle_basename: 'init.betse.gz',
      init_pickle_dirname_relative: 'init_pickles',
      init_export_dirname_relative: 'init_exports',
      sim_pickle_basename: 'sim.betse.gz',
      sim_pickle_dirname_relative: 'sim_pickles',
      sim_export_dirname_relative: 'sim_exports',
    },
  };
}

export interface Preset {
  name: string;
  description: string;
  config: Partial<SimulationConfig>;
}

export const PRESETS: Preset[] = [
  {
    name: 'Default (Basic)',
    description: 'Basic ion profile with Na+, K+, Cl- and leak channels. Suitable for simple bioelectric simulations.',
    config: {},
  },
  {
    name: 'Mammalian Cell',
    description: 'Mammalian ion concentrations with voltage-gated channels. Produces excitable dynamics.',
    config: {
      ions: {
        ...createDefaultConfig().ions,
        ion_profile: IonProfileType.MAMMAL,
        Na: { ...DEFAULT_ION_NA, c_env: 145.0, c_cell: 12.0 },
        K: { ...DEFAULT_ION_K, c_env: 4.5, c_cell: 150.0 },
        Cl: { ...DEFAULT_ION_CL, c_env: 116.0, c_cell: 10.0 },
      },
    },
  },
  {
    name: 'Amphibian Cell',
    description: 'Amphibian ion concentrations suitable for Xenopus embryo models.',
    config: {
      ions: {
        ...createDefaultConfig().ions,
        ion_profile: IonProfileType.AMPHIBIAN,
        Na: { ...DEFAULT_ION_NA, c_env: 110.0, c_cell: 10.0 },
        K: { ...DEFAULT_ION_K, c_env: 2.5, c_cell: 120.0 },
        Cl: { ...DEFAULT_ION_CL, c_env: 80.0, c_cell: 40.0 },
      },
    },
  },
  {
    name: 'Gap Junction Network',
    description: 'Emphasizes gap junction coupling with a dense tissue lattice. Good for bioelectric pattern propagation.',
    config: {
      space: {
        ...createDefaultConfig().space,
        cell_radius: 5.0e-6,
        cell_lattice_type: CellLatticeType.HEX,
        cell_lattice_disorder: 0.2,
        grid_size: 40,
      },
    },
  },
  {
    name: 'Excitable Tissue',
    description: 'Includes Nav and Kv channels for action-potential-like dynamics across the tissue.',
    config: {
      channels: createDefaultConfig().channels.map(ch => {
        if (ch.type === ChannelType.NA_V || ch.type === ChannelType.K_V) {
          return { ...ch, enabled: true };
        }
        return ch;
      }),
    },
  },
  {
    name: 'Long Simulation',
    description: 'Extended time course of 300s simulation time for studying long-term bioelectric patterns.',
    config: {
      time: {
        init_time_total: 30.0,
        init_time_step: 1.0e-3,
        init_time_sampling: 0.5,
        sim_time_total: 300.0,
        sim_time_step: 1.0e-3,
        sim_time_sampling: 1.0,
      },
    },
  },
  {
    name: 'With Deformation',
    description: 'Enables mechanical deformation driven by osmotic and electroosmotic forces.',
    config: {
      physics: {
        ...createDefaultConfig().physics,
        is_electroosmosis: true,
        is_deformation: true,
        deform_osmo: true,
        deform_electro: true,
      },
    },
  },
];
