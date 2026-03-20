import React from 'react';
import type { PhysicsConfig } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface PhysicsSettingsProps {
  config: PhysicsConfig;
  onChange: (path: string, value: unknown) => void;
}

export const PhysicsSettings: React.FC<PhysicsSettingsProps> = ({ config, onChange }) => {
  return (
    <div>
      <h2 style={styles.pageTitle}>Physics Options</h2>
      <p style={styles.pageDesc}>
        Configure physical phenomena including electroosmosis, mechanical
        deformation, pressure dynamics, and global physical constants.
      </p>

      <ConfigSection title="Electroosmosis" description="Fluid flow driven by electric fields through the extracellular space.">
        <ToggleField
          label="Enable Electroosmosis"
          checked={config.is_electroosmosis}
          onChange={(v) => onChange('is_electroosmosis', v)}
          tooltip="Enable electroosmotic flow calculations. Requires ECM to be enabled in Space settings."
        />
      </ConfigSection>

      <ConfigSection title="Mechanical Deformation" description="Cell cluster deformation driven by osmotic and electrical forces.">
        <ToggleField
          label="Enable Deformation"
          checked={config.is_deformation}
          onChange={(v) => onChange('is_deformation', v)}
          tooltip="Enable mechanical deformation of the cell cluster."
        />

        <ToggleField
          label="Osmotic Deformation"
          checked={config.deform_osmo}
          onChange={(v) => onChange('deform_osmo', v)}
          tooltip="Include osmotic pressure differences as a driving force for deformation."
        />

        <ToggleField
          label="Electrostatic Deformation"
          checked={config.deform_electro}
          onChange={(v) => onChange('deform_electro', v)}
          tooltip="Include electrostatic (Maxwell stress) forces for deformation."
        />

        <ToggleField
          label="Fixed Cluster Boundary"
          checked={config.fixed_cluster_boundary}
          onChange={(v) => onChange('fixed_cluster_boundary', v)}
          tooltip="Fix the outer boundary of the cell cluster (prevent boundary cells from moving)."
        />

        <ConfigFieldGroup columns={2}>
          <ConfigField
            label="Young's Modulus"
            unit="Pa"
            tooltip="Elastic modulus of the cell cluster tissue."
          >
            <input
              type="number"
              value={config.young_modulus}
              onChange={(e) => onChange('young_modulus', parseFloat(e.target.value))}
              step={1}
              min={0}
              style={styles.input}
              disabled={!config.is_deformation}
            />
          </ConfigField>
          <ConfigField
            label="Membrane Viscosity"
            unit="Pa*s"
            tooltip="Viscosity of the cell membrane, affecting deformation dynamics."
          >
            <input
              type="number"
              value={config.mu_membrane}
              onChange={(e) => onChange('mu_membrane', parseFloat(e.target.value))}
              step={1e-5}
              min={0}
              style={styles.input}
              disabled={!config.is_deformation}
            />
            <span style={styles.sci}>{formatScientific(config.mu_membrane)}</span>
          </ConfigField>
        </ConfigFieldGroup>
      </ConfigSection>

      <ConfigSection title="Pressure" description="Hydrostatic pressure within cells.">
        <ToggleField
          label="Enable Pressure Dynamics"
          checked={config.is_pressure}
          onChange={(v) => onChange('is_pressure', v)}
          tooltip="Enable pressure calculations within the cell cluster."
        />

        <ConfigField
          label="Cell Hydrostatic Pressure"
          unit="Pa"
          tooltip="Initial hydrostatic pressure inside cells."
        >
          <input
            type="number"
            value={config.p_cells}
            onChange={(e) => onChange('p_cells', parseFloat(e.target.value))}
            step={0.1}
            style={styles.input}
            disabled={!config.is_pressure}
          />
        </ConfigField>
      </ConfigSection>

      <ConfigSection title="Global Physical Constants" description="Temperature and other fundamental physical parameters.">
        <ConfigField
          label="Temperature"
          unit="K"
          tooltip="Simulation temperature in Kelvin. Affects Nernst potentials and diffusion rates."
        >
          <input
            type="number"
            value={config.T}
            onChange={(e) => onChange('T', parseFloat(e.target.value))}
            step={1}
            min={0}
            style={styles.input}
          />
          <span style={styles.sci}>
            = {(config.T - 273.15).toFixed(1)} deg C
          </span>
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
    fontFamily: 'var(--font-family-mono)',
  },
  sci: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
    display: 'block',
    marginTop: '2px',
  },
};
