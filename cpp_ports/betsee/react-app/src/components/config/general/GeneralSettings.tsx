import React from 'react';
import type { TimeConfig, PathConfig } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup } from '../shared/ConfigWidgets';

interface GeneralSettingsProps {
  timeConfig: TimeConfig;
  pathConfig: PathConfig;
  onChange: (path: string, value: unknown) => void;
}

export const GeneralSettings: React.FC<GeneralSettingsProps> = ({
  timeConfig,
  pathConfig,
  onChange,
}) => {
  return (
    <div>
      <h2 style={styles.pageTitle}>General Settings / Time Configuration</h2>
      <p style={styles.pageDesc}>
        Configure temporal parameters for the initialization and simulation phases,
        as well as file paths for simulation artifacts.
      </p>

      <ConfigSection title="Initialization Phase" description="Time parameters for the initialization (steady-state convergence) phase.">
        <ConfigFieldGroup>
          <ConfigField
            label="Total Time"
            unit="s"
            tooltip="Total duration of the initialization phase in seconds."
          >
            <input
              type="number"
              value={timeConfig.init_time_total}
              onChange={(e) => onChange('time.init_time_total', parseFloat(e.target.value))}
              step={1}
              min={0}
              style={styles.input}
            />
          </ConfigField>
          <ConfigField
            label="Time Step"
            unit="s"
            tooltip="Integration time step for the initialization phase. Smaller values increase accuracy but also computation time."
          >
            <input
              type="number"
              value={timeConfig.init_time_step}
              onChange={(e) => onChange('time.init_time_step', parseFloat(e.target.value))}
              step={0.0001}
              min={0}
              style={styles.input}
            />
          </ConfigField>
          <ConfigField
            label="Sampling Rate"
            unit="s"
            tooltip="How frequently to sample and record data during initialization."
          >
            <input
              type="number"
              value={timeConfig.init_time_sampling}
              onChange={(e) => onChange('time.init_time_sampling', parseFloat(e.target.value))}
              step={0.01}
              min={0}
              style={styles.input}
            />
          </ConfigField>
        </ConfigFieldGroup>
      </ConfigSection>

      <ConfigSection title="Simulation Phase" description="Time parameters for the main simulation phase.">
        <ConfigFieldGroup>
          <ConfigField
            label="Total Time"
            unit="s"
            tooltip="Total duration of the simulation phase in seconds."
          >
            <input
              type="number"
              value={timeConfig.sim_time_total}
              onChange={(e) => onChange('time.sim_time_total', parseFloat(e.target.value))}
              step={1}
              min={0}
              style={styles.input}
            />
          </ConfigField>
          <ConfigField
            label="Time Step"
            unit="s"
            tooltip="Integration time step for the simulation phase."
          >
            <input
              type="number"
              value={timeConfig.sim_time_step}
              onChange={(e) => onChange('time.sim_time_step', parseFloat(e.target.value))}
              step={0.0001}
              min={0}
              style={styles.input}
            />
          </ConfigField>
          <ConfigField
            label="Sampling Rate"
            unit="s"
            tooltip="How frequently to sample and record data during simulation."
          >
            <input
              type="number"
              value={timeConfig.sim_time_sampling}
              onChange={(e) => onChange('time.sim_time_sampling', parseFloat(e.target.value))}
              step={0.01}
              min={0}
              style={styles.input}
            />
          </ConfigField>
        </ConfigFieldGroup>
      </ConfigSection>

      <ConfigSection title="File Paths" description="Paths for simulation data storage. Relative to the configuration file location.">
        <ConfigField label="Seed Pickle Filename" tooltip="Basename for the seed phase output file.">
          <input
            type="text"
            value={pathConfig.seed_pickle_basename}
            onChange={(e) => onChange('paths.seed_pickle_basename', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Init Pickle Filename" tooltip="Basename for the initialization phase output file.">
          <input
            type="text"
            value={pathConfig.init_pickle_basename}
            onChange={(e) => onChange('paths.init_pickle_basename', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Init Pickle Directory" tooltip="Directory for initialization pickle files (relative).">
          <input
            type="text"
            value={pathConfig.init_pickle_dirname_relative}
            onChange={(e) => onChange('paths.init_pickle_dirname_relative', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Init Export Directory" tooltip="Directory for initialization exports (relative).">
          <input
            type="text"
            value={pathConfig.init_export_dirname_relative}
            onChange={(e) => onChange('paths.init_export_dirname_relative', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Sim Pickle Filename" tooltip="Basename for the simulation phase output file.">
          <input
            type="text"
            value={pathConfig.sim_pickle_basename}
            onChange={(e) => onChange('paths.sim_pickle_basename', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Sim Pickle Directory" tooltip="Directory for simulation pickle files (relative).">
          <input
            type="text"
            value={pathConfig.sim_pickle_dirname_relative}
            onChange={(e) => onChange('paths.sim_pickle_dirname_relative', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
        <ConfigField label="Sim Export Directory" tooltip="Directory for simulation exports (relative).">
          <input
            type="text"
            value={pathConfig.sim_export_dirname_relative}
            onChange={(e) => onChange('paths.sim_export_dirname_relative', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          />
        </ConfigField>
      </ConfigSection>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  pageTitle: {
    fontSize: 'var(--font-size-xl)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    marginBottom: 'var(--spacing-sm)',
  },
  pageDesc: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-xl)',
    maxWidth: '680px',
    lineHeight: 1.6,
  },
  input: {
    width: '100%',
  },
};
