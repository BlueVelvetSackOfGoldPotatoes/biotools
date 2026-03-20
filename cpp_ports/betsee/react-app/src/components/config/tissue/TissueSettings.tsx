import React, { useState } from 'react';
import type { TissueConfig, TissueProfile } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface TissueSettingsProps {
  config: TissueConfig;
  onChange: (path: string, value: unknown) => void;
}

const TissueProfileEditor: React.FC<{
  profile: TissueProfile;
  pathPrefix: string;
  onChange: (path: string, value: unknown) => void;
  isDefault?: boolean;
  onRemove?: () => void;
}> = ({ profile, pathPrefix, onChange, isDefault = false, onRemove }) => {
  return (
    <ConfigSection
      title={isDefault ? 'Default Tissue Profile' : profile.name}
      description={isDefault ? 'Base tissue properties applied to all cells.' : 'Custom tissue profile with region-specific properties.'}
      collapsible={!isDefault}
    >
      {!isDefault && (
        <div style={styles.profileHeader}>
          <ConfigField label="Profile Name">
            <input
              type="text"
              value={profile.name}
              onChange={(e) => onChange(`${pathPrefix}.name`, e.target.value)}
              style={styles.input}
            />
          </ConfigField>
          <ToggleField
            label="Enabled"
            checked={profile.enabled}
            onChange={(v) => onChange(`${pathPrefix}.enabled`, v)}
          />
          {onRemove && (
            <button onClick={onRemove} style={styles.removeBtn}>
              Remove Profile
            </button>
          )}
        </div>
      )}

      <ConfigField label="Image Mask File" tooltip="Path to the image file defining this tissue region.">
        <input
          type="text"
          value={profile.picker_image_filename}
          onChange={(e) => onChange(`${pathPrefix}.picker_image_filename`, e.target.value)}
          style={{ ...styles.input, fontFamily: 'var(--font-family-mono)' }}
          placeholder="(optional) path/to/mask.png"
        />
      </ConfigField>

      <h4 style={styles.subhead}>Membrane Diffusion Constants (Dm)</h4>
      <p style={styles.subDesc}>
        Membrane diffusion constants control passive ion permeability for each species
        in this tissue region.
      </p>

      <ConfigFieldGroup columns={3}>
        {[
          { key: 'Dm_Na', label: 'Na+ Dm', unit: 'm^2/s' },
          { key: 'Dm_K', label: 'K+ Dm', unit: 'm^2/s' },
          { key: 'Dm_Cl', label: 'Cl- Dm', unit: 'm^2/s' },
          { key: 'Dm_Ca', label: 'Ca2+ Dm', unit: 'm^2/s' },
          { key: 'Dm_M', label: 'M- Dm', unit: 'm^2/s' },
          { key: 'Dm_H', label: 'H+ Dm', unit: 'm^2/s' },
        ].map(({ key, label, unit }) => (
          <ConfigField key={key} label={label} unit={unit}>
            <input
              type="number"
              value={(profile as unknown as Record<string, unknown>)[key] as number}
              onChange={(e) => onChange(`${pathPrefix}.${key}`, parseFloat(e.target.value))}
              step={1e-19}
              style={styles.input}
            />
            <span style={styles.sci}>
              {formatScientific((profile as unknown as Record<string, unknown>)[key] as number)}
            </span>
          </ConfigField>
        ))}
      </ConfigFieldGroup>
    </ConfigSection>
  );
};

export const TissueSettings: React.FC<TissueSettingsProps> = ({ config, onChange }) => {
  const addProfile = () => {
    const newProfile: TissueProfile = {
      name: `Custom Profile ${config.custom_profiles.length + 1}`,
      picker_image_filename: '',
      Dm_Na: 1.0e-18,
      Dm_K: 2.5e-17,
      Dm_Cl: 2.0e-18,
      Dm_Ca: 1.0e-18,
      Dm_M: 0.0,
      Dm_H: 1.0e-18,
      enabled: true,
    };
    onChange('custom_profiles', [...config.custom_profiles, newProfile]);
  };

  const removeProfile = (index: number) => {
    onChange('custom_profiles', config.custom_profiles.filter((_, i) => i !== index));
  };

  return (
    <div>
      <h2 style={styles.pageTitle}>Tissue Profile Configuration</h2>
      <p style={styles.pageDesc}>
        Define tissue heterogeneity by creating multiple tissue profiles with
        different membrane diffusion properties. Each profile can be mapped to
        specific regions using image masks.
      </p>

      <TissueProfileEditor
        profile={config.default_profile}
        pathPrefix="default_profile"
        onChange={onChange}
        isDefault
      />

      {config.custom_profiles.map((profile, index) => (
        <TissueProfileEditor
          key={index}
          profile={profile}
          pathPrefix={`custom_profiles.${index}`}
          onChange={onChange}
          onRemove={() => removeProfile(index)}
        />
      ))}

      <button style={styles.addBtn} onClick={addProfile}>
        + Add Custom Tissue Profile
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
  subhead: {
    fontSize: 'var(--font-size-sm)',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    marginTop: 'var(--spacing-lg)',
    marginBottom: 'var(--spacing-xs)',
  },
  subDesc: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    marginBottom: 'var(--spacing-md)',
  },
  profileHeader: {
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
    marginTop: 'var(--spacing-sm)',
  },
  addBtn: {
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px dashed var(--color-border-light)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
    width: '100%',
    textAlign: 'center' as const,
  },
};
