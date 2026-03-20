import React from 'react';
import type { Intervention } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';

interface InterventionSettingsProps {
  interventions: Intervention[];
  onChange: (interventions: Intervention[]) => void;
}

function createDefaultIntervention(index: number): Intervention {
  return {
    name: `Intervention ${index + 1}`,
    enabled: true,
    type: 'global',
    target_tissue: 'all',
    change_Na_mem: 0,
    change_K_mem: 0,
    change_Cl_mem: 0,
    change_Ca_mem: 0,
    apply_start: 10.0,
    apply_end: 30.0,
    change_rate: 1.0,
  };
}

export const InterventionSettings: React.FC<InterventionSettingsProps> = ({
  interventions,
  onChange,
}) => {
  const addIntervention = () => {
    onChange([...interventions, createDefaultIntervention(interventions.length)]);
  };

  const removeIntervention = (index: number) => {
    onChange(interventions.filter((_, i) => i !== index));
  };

  const updateIntervention = (index: number, field: keyof Intervention, value: unknown) => {
    const updated = interventions.map((intv, i) =>
      i === index ? { ...intv, [field]: value } : intv,
    );
    onChange(updated);
  };

  return (
    <div>
      <h2 style={styles.pageTitle}>Intervention Configuration</h2>
      <p style={styles.pageDesc}>
        Configure targeted and global interventions that modify membrane permeability during the
        simulation. Interventions allow you to model experimental manipulations such as drug
        application, voltage clamping, or selective membrane permeability changes.
      </p>

      {interventions.length === 0 && (
        <div style={styles.emptyState}>
          <p style={styles.emptyText}>
            No interventions configured. Add an intervention to modify membrane properties during
            the simulation run.
          </p>
        </div>
      )}

      {interventions.map((intv, index) => (
        <ConfigSection
          key={index}
          title={intv.name}
          description={`Type: ${intv.type} | Target: ${intv.target_tissue} | Time: ${intv.apply_start}s - ${intv.apply_end}s`}
          collapsible
          defaultCollapsed={!intv.enabled}
        >
          <div style={styles.topRow}>
            <ToggleField
              label="Enabled"
              checked={intv.enabled}
              onChange={(v) => updateIntervention(index, 'enabled', v)}
              tooltip="Enable or disable this intervention."
            />
            <button
              style={styles.removeBtn}
              onClick={() => removeIntervention(index)}
              title="Remove this intervention"
            >
              Remove
            </button>
          </div>

          <ConfigFieldGroup columns={2}>
            <ConfigField label="Name" tooltip="A descriptive name for this intervention.">
              <input
                type="text"
                value={intv.name}
                onChange={(e) => updateIntervention(index, 'name', e.target.value)}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField label="Type" tooltip="Global applies everywhere; targeted applies to a specific tissue.">
              <select
                value={intv.type}
                onChange={(e) =>
                  updateIntervention(index, 'type', e.target.value as 'global' | 'targeted')
                }
                style={styles.input}
              >
                <option value="global">Global</option>
                <option value="targeted">Targeted</option>
              </select>
            </ConfigField>
          </ConfigFieldGroup>

          {intv.type === 'targeted' && (
            <ConfigField
              label="Target Tissue"
              tooltip="Name of the tissue profile to target."
            >
              <input
                type="text"
                value={intv.target_tissue}
                onChange={(e) => updateIntervention(index, 'target_tissue', e.target.value)}
                style={styles.input}
              />
            </ConfigField>
          )}

          <h4 style={styles.subhead}>Timing</h4>
          <ConfigFieldGroup columns={3}>
            <ConfigField
              label="Start Time"
              unit="s"
              tooltip="Time at which the intervention begins."
            >
              <input
                type="number"
                value={intv.apply_start}
                onChange={(e) => updateIntervention(index, 'apply_start', parseFloat(e.target.value) || 0)}
                step={1}
                min={0}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField
              label="End Time"
              unit="s"
              tooltip="Time at which the intervention ends."
            >
              <input
                type="number"
                value={intv.apply_end}
                onChange={(e) => updateIntervention(index, 'apply_end', parseFloat(e.target.value) || 0)}
                step={1}
                min={0}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField
              label="Change Rate"
              tooltip="Rate of change application (1.0 = instant at boundaries)."
            >
              <input
                type="number"
                value={intv.change_rate}
                onChange={(e) => updateIntervention(index, 'change_rate', parseFloat(e.target.value) || 0)}
                step={0.1}
                min={0}
                style={styles.input}
              />
            </ConfigField>
          </ConfigFieldGroup>

          <h4 style={styles.subhead}>Membrane Permeability Changes</h4>
          <p style={styles.subDesc}>
            Multiplier applied to the membrane diffusion constant for each ion during the
            intervention window. Values {'>'} 1 increase permeability; values {'<'} 1 decrease it; 0 = no change.
          </p>
          <ConfigFieldGroup columns={2}>
            <ConfigField
              label="Na+ Dm Change"
              unit="m^2/s"
              tooltip="Change in sodium membrane permeability."
            >
              <input
                type="number"
                value={intv.change_Na_mem}
                onChange={(e) => updateIntervention(index, 'change_Na_mem', parseFloat(e.target.value) || 0)}
                step={1e-18}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField
              label="K+ Dm Change"
              unit="m^2/s"
              tooltip="Change in potassium membrane permeability."
            >
              <input
                type="number"
                value={intv.change_K_mem}
                onChange={(e) => updateIntervention(index, 'change_K_mem', parseFloat(e.target.value) || 0)}
                step={1e-18}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField
              label="Cl- Dm Change"
              unit="m^2/s"
              tooltip="Change in chloride membrane permeability."
            >
              <input
                type="number"
                value={intv.change_Cl_mem}
                onChange={(e) => updateIntervention(index, 'change_Cl_mem', parseFloat(e.target.value) || 0)}
                step={1e-18}
                style={styles.input}
              />
            </ConfigField>
            <ConfigField
              label="Ca2+ Dm Change"
              unit="m^2/s"
              tooltip="Change in calcium membrane permeability."
            >
              <input
                type="number"
                value={intv.change_Ca_mem}
                onChange={(e) => updateIntervention(index, 'change_Ca_mem', parseFloat(e.target.value) || 0)}
                step={1e-18}
                style={styles.input}
              />
            </ConfigField>
          </ConfigFieldGroup>
        </ConfigSection>
      ))}

      <button style={styles.addBtn} onClick={addIntervention}>
        + Add Intervention
      </button>
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
  emptyState: {
    padding: 'var(--spacing-xl)',
    textAlign: 'center' as const,
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px dashed var(--color-border)',
    borderRadius: 'var(--radius-md)',
    marginBottom: 'var(--spacing-lg)',
  },
  emptyText: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--font-size-base)',
    margin: 0,
  },
  topRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 'var(--spacing-md)',
  },
  removeBtn: {
    padding: 'var(--spacing-xs) var(--spacing-md)',
    backgroundColor: 'transparent',
    border: '1px solid var(--color-error)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-error)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
  },
  addBtn: {
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-elevated)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-text-primary)',
    fontSize: 'var(--font-size-base)',
    cursor: 'pointer',
    width: '100%',
    textAlign: 'center' as const,
  },
  input: {
    width: '100%',
    fontFamily: 'var(--font-family-mono)',
  },
  subhead: {
    fontSize: 'var(--font-size-sm)',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    marginTop: 'var(--spacing-lg)',
    marginBottom: 'var(--spacing-md)',
  },
  subDesc: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
    marginBottom: 'var(--spacing-md)',
    lineHeight: 1.5,
  },
};
