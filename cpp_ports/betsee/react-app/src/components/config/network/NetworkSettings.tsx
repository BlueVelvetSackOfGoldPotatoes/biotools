import React from 'react';
import type { GRNConfig, SubstanceConfig } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface NetworkSettingsProps {
  config: GRNConfig;
  onChange: (path: string, value: unknown) => void;
}

export const NetworkSettings: React.FC<NetworkSettingsProps> = ({ config, onChange }) => {
  const addSubstance = () => {
    const newSub: SubstanceConfig = {
      name: `Substance ${config.substances.length + 1}`,
      Dm: 1.0e-18,
      Do: 1.0e-9,
      c_env: 0,
      c_cell: 0,
      z: 0,
      decay_rate: 0,
      growth_rate: 0,
    };
    onChange('substances', [...config.substances, newSub]);
  };

  const removeSubstance = (index: number) => {
    onChange('substances', config.substances.filter((_, i) => i !== index));
  };

  const updateSubstance = (index: number, field: keyof SubstanceConfig, value: unknown) => {
    const updated = config.substances.map((s, i) =>
      i === index ? { ...s, [field]: value } : s,
    );
    onChange('substances', updated);
  };

  return (
    <div>
      <h2 style={styles.pageTitle}>Network / Gene Regulatory Network</h2>
      <p style={styles.pageDesc}>
        Configure gene regulatory networks (GRN) and biochemical reaction networks.
        Define custom substances that interact with the bioelectric simulation.
      </p>

      <ConfigSection title="GRN Settings" description="Enable and configure gene regulatory network integration.">
        <ToggleField
          label="Enable Gene Regulatory Network"
          checked={config.enabled}
          onChange={(v) => onChange('enabled', v)}
          tooltip="Enable GRN integration with the bioelectric simulation."
        />

        <ConfigField
          label="GRN Definition File"
          tooltip="Path to the YAML/CSV file defining the gene regulatory network."
        >
          <input
            type="text"
            value={config.gene_regulatory_network_file}
            onChange={(e) => onChange('gene_regulatory_network_file', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
            placeholder="grn_definition.csv"
            disabled={!config.enabled}
          />
        </ConfigField>

        <ConfigField
          label="Reaction Network File"
          tooltip="Path to the file defining biochemical reactions."
        >
          <input
            type="text"
            value={config.reaction_network_file}
            onChange={(e) => onChange('reaction_network_file', e.target.value)}
            style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
            placeholder="reaction_network.csv"
            disabled={!config.enabled}
          />
        </ConfigField>
      </ConfigSection>

      <ConfigSection title="Custom Substances" description="Define custom biochemical substances for the GRN.">
        {config.substances.map((substance, index) => (
          <div key={index} style={styles.substanceCard}>
            <div style={styles.substanceHeader}>
              <ConfigField label="Name">
                <input
                  type="text"
                  value={substance.name}
                  onChange={(e) => updateSubstance(index, 'name', e.target.value)}
                  style={styles.input}
                  disabled={!config.enabled}
                />
              </ConfigField>
              <button
                onClick={() => removeSubstance(index)}
                style={styles.removeBtn}
                disabled={!config.enabled}
              >
                Remove
              </button>
            </div>

            <ConfigFieldGroup columns={3}>
              <ConfigField label="Dm" unit="m^2/s">
                <input
                  type="number"
                  value={substance.Dm}
                  onChange={(e) => updateSubstance(index, 'Dm', parseFloat(e.target.value))}
                  step={1e-19}
                  style={styles.input}
                  disabled={!config.enabled}
                />
                <span style={styles.sci}>{formatScientific(substance.Dm)}</span>
              </ConfigField>
              <ConfigField label="Do" unit="m^2/s">
                <input
                  type="number"
                  value={substance.Do}
                  onChange={(e) => updateSubstance(index, 'Do', parseFloat(e.target.value))}
                  step={1e-10}
                  style={styles.input}
                  disabled={!config.enabled}
                />
                <span style={styles.sci}>{formatScientific(substance.Do)}</span>
              </ConfigField>
              <ConfigField label="Valence (z)">
                <input
                  type="number"
                  value={substance.z}
                  onChange={(e) => updateSubstance(index, 'z', parseInt(e.target.value, 10))}
                  step={1}
                  style={styles.input}
                  disabled={!config.enabled}
                />
              </ConfigField>
              <ConfigField label="Env. Conc." unit="mM">
                <input
                  type="number"
                  value={substance.c_env}
                  onChange={(e) => updateSubstance(index, 'c_env', parseFloat(e.target.value))}
                  step={0.1}
                  style={styles.input}
                  disabled={!config.enabled}
                />
              </ConfigField>
              <ConfigField label="Cell Conc." unit="mM">
                <input
                  type="number"
                  value={substance.c_cell}
                  onChange={(e) => updateSubstance(index, 'c_cell', parseFloat(e.target.value))}
                  step={0.1}
                  style={styles.input}
                  disabled={!config.enabled}
                />
              </ConfigField>
              <ConfigField label="Decay Rate" unit="1/s">
                <input
                  type="number"
                  value={substance.decay_rate}
                  onChange={(e) => updateSubstance(index, 'decay_rate', parseFloat(e.target.value))}
                  step={0.001}
                  style={styles.input}
                  disabled={!config.enabled}
                />
              </ConfigField>
            </ConfigFieldGroup>
          </div>
        ))}

        <button style={styles.addBtn} onClick={addSubstance} disabled={!config.enabled}>
          + Add Substance
        </button>
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
  substanceCard: {
    padding: 'var(--spacing-md)',
    backgroundColor: 'var(--color-bg-tertiary)',
    borderRadius: 'var(--radius-md)',
    marginBottom: 'var(--spacing-md)',
    border: '1px solid var(--color-border)',
  },
  substanceHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 'var(--spacing-sm)',
  },
  removeBtn: {
    padding: 'var(--spacing-xs) var(--spacing-md)',
    backgroundColor: 'transparent',
    border: '1px solid var(--color-error)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-error)',
    fontSize: 'var(--font-size-xs)',
    cursor: 'pointer',
  },
  addBtn: {
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-tertiary)',
    border: '1px dashed var(--color-border-light)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
    width: '100%',
    textAlign: 'center' as const,
  },
};
