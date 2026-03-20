import React from 'react';
import type { IonConfig, IonSpecies, IonProfileType } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface IonSettingsProps {
  config: IonConfig;
  onChange: (path: string, value: unknown) => void;
}

const ION_PROFILES: { value: IonProfileType; label: string }[] = [
  { value: 'basic' as IonProfileType, label: 'Basic' },
  { value: 'basic_ca' as IonProfileType, label: 'Basic + Ca2+' },
  { value: 'mammal' as IonProfileType, label: 'Mammal' },
  { value: 'amphibian' as IonProfileType, label: 'Amphibian' },
  { value: 'custom' as IonProfileType, label: 'Custom' },
];

const IonSpeciesEditor: React.FC<{
  species: IonSpecies;
  ionKey: string;
  onChange: (path: string, value: unknown) => void;
}> = ({ species, ionKey, onChange }) => {
  return (
    <ConfigSection
      title={`${species.name} (${species.symbol})`}
      description={`Valence: ${species.z > 0 ? '+' : ''}${species.z}`}
      collapsible
    >
      <ConfigFieldGroup columns={2}>
        <ConfigField
          label="Membrane Diffusion (Dm)"
          unit="m^2/s"
          tooltip="Membrane diffusion constant for this ion species."
        >
          <input
            type="number"
            value={species.Dm}
            onChange={(e) => onChange(`${ionKey}.Dm`, parseFloat(e.target.value))}
            step={1e-19}
            style={styles.input}
          />
          <span style={styles.sci}>{formatScientific(species.Dm)}</span>
        </ConfigField>
        <ConfigField
          label="Free Diffusion (Do)"
          unit="m^2/s"
          tooltip="Free solution diffusion constant."
        >
          <input
            type="number"
            value={species.Do}
            onChange={(e) => onChange(`${ionKey}.Do`, parseFloat(e.target.value))}
            step={1e-10}
            style={styles.input}
          />
          <span style={styles.sci}>{formatScientific(species.Do)}</span>
        </ConfigField>
        <ConfigField
          label="Environmental Conc."
          unit="mM"
          tooltip="Extracellular / environmental concentration."
        >
          <input
            type="number"
            value={species.c_env}
            onChange={(e) => onChange(`${ionKey}.c_env`, parseFloat(e.target.value))}
            step={0.1}
            min={0}
            style={styles.input}
          />
        </ConfigField>
        <ConfigField
          label="Intracellular Conc."
          unit="mM"
          tooltip="Initial intracellular concentration."
        >
          <input
            type="number"
            value={species.c_cell}
            onChange={(e) => onChange(`${ionKey}.c_cell`, parseFloat(e.target.value))}
            step={0.1}
            min={0}
            style={styles.input}
          />
        </ConfigField>
      </ConfigFieldGroup>
    </ConfigSection>
  );
};

export const IonSettings: React.FC<IonSettingsProps> = ({ config, onChange }) => {
  return (
    <div>
      <h2 style={styles.pageTitle}>Ion Species Configuration</h2>
      <p style={styles.pageDesc}>
        Configure ion species, concentrations, and diffusion constants.
        Select a preset ion profile or customize individual species parameters.
      </p>

      <ConfigSection title="Ion Profile" description="Select a predefined ion concentration profile or configure custom values.">
        <ConfigField label="Profile Type">
          <select
            value={config.ion_profile}
            onChange={(e) => onChange('ion_profile', e.target.value)}
            style={styles.select}
          >
            {ION_PROFILES.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </ConfigField>

        {/* Nernst potential summary */}
        <div style={styles.nernstBox}>
          <h4 style={styles.nernstTitle}>Nernst Potential Estimates (at 37 C)</h4>
          <div style={styles.nernstGrid}>
            {(['Na', 'K', 'Cl', 'Ca'] as const).map((ion) => {
              const species = config[ion];
              const z = species.z;
              const E = z !== 0
                ? (8.314 * 310 / (z * 96485)) * Math.log(species.c_env / Math.max(species.c_cell, 1e-12)) * 1000
                : 0;
              return (
                <div key={ion} style={styles.nernstItem}>
                  <span style={styles.nernstLabel}>E_{species.symbol}</span>
                  <span style={styles.nernstValue}>{E.toFixed(1)} mV</span>
                </div>
              );
            })}
          </div>
        </div>
      </ConfigSection>

      <IonSpeciesEditor species={config.Na} ionKey="Na" onChange={onChange} />
      <IonSpeciesEditor species={config.K} ionKey="K" onChange={onChange} />
      <IonSpeciesEditor species={config.Cl} ionKey="Cl" onChange={onChange} />
      <IonSpeciesEditor species={config.Ca} ionKey="Ca" onChange={onChange} />
      <IonSpeciesEditor species={config.M} ionKey="M" onChange={onChange} />
      <IonSpeciesEditor species={config.H} ionKey="H" onChange={onChange} />
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
  select: {
    width: '100%',
    maxWidth: '320px',
  },
  sci: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
    display: 'block',
    marginTop: '2px',
  },
  nernstBox: {
    marginTop: 'var(--spacing-lg)',
    padding: 'var(--spacing-md)',
    backgroundColor: 'var(--color-bg-tertiary)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--color-border)',
  },
  nernstTitle: {
    fontSize: 'var(--font-size-sm)',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-sm)',
    margin: 0,
  },
  nernstGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 'var(--spacing-md)',
    marginTop: 'var(--spacing-sm)',
  },
  nernstItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '2px',
  },
  nernstLabel: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
  },
  nernstValue: {
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    color: 'var(--color-text-accent)',
    fontFamily: 'var(--font-family-mono)',
  },
};
