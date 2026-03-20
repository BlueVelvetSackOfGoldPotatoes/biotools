import React from 'react';
import type { SpaceConfig, CellLatticeType } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';
import { formatScientific } from '../../../utils/format';

interface SpaceSettingsProps {
  config: SpaceConfig;
  onChange: (path: string, value: unknown) => void;
}

export const SpaceSettings: React.FC<SpaceSettingsProps> = ({ config, onChange }) => {
  return (
    <div>
      <h2 style={styles.pageTitle}>Space / Cell Configuration</h2>
      <p style={styles.pageDesc}>
        Configure the spatial domain, cell cluster geometry, lattice type,
        and extracellular space parameters for the simulation.
      </p>

      <ConfigSection title="Intracellular / Cell Cluster" description="Properties of the cell cluster: cell size, arrangement, and lattice geometry.">
        <ConfigFieldGroup columns={2}>
          <ConfigField
            label="Cell Radius"
            unit="m"
            tooltip="Average radius of individual cells. Typically 3-10 um for biological cells."
          >
            <input
              type="number"
              value={config.cell_radius}
              onChange={(e) => onChange('cell_radius', parseFloat(e.target.value))}
              step={1e-7}
              min={0}
              style={styles.input}
            />
            <span style={styles.displayVal}>{formatScientific(config.cell_radius)} m</span>
          </ConfigField>
          <ConfigField
            label="Lattice Disorder"
            tooltip="Random perturbation of cell positions (0 = perfect lattice, 1 = maximum disorder)."
          >
            <input
              type="range"
              value={config.cell_lattice_disorder}
              onChange={(e) => onChange('cell_lattice_disorder', parseFloat(e.target.value))}
              min={0}
              max={1}
              step={0.01}
              style={{ width: '100%' }}
            />
            <span style={styles.displayVal}>{config.cell_lattice_disorder.toFixed(2)}</span>
          </ConfigField>
        </ConfigFieldGroup>

        <ConfigField label="Lattice Type" tooltip="Hexagonal lattices pack cells more densely; square lattices are simpler.">
          <div style={styles.radioGroup}>
            <label style={styles.radioLabel}>
              <input
                type="radio"
                name="lattice_type"
                value="hex"
                checked={config.cell_lattice_type === 'hex'}
                onChange={() => onChange('cell_lattice_type', 'hex' as CellLatticeType)}
              />
              <span>Hexagonal</span>
            </label>
            <label style={styles.radioLabel}>
              <input
                type="radio"
                name="lattice_type"
                value="square"
                checked={config.cell_lattice_type === 'square'}
                onChange={() => onChange('cell_lattice_type', 'square' as CellLatticeType)}
              />
              <span>Square</span>
            </label>
          </div>
        </ConfigField>
      </ConfigSection>

      <ConfigSection title="Extracellular Space" description="Environmental and computational grid parameters.">
        <ConfigFieldGroup columns={2}>
          <ConfigField
            label="Grid Size"
            tooltip="Number of grid points along each axis of the computational domain. Higher values increase resolution and computation time."
          >
            <input
              type="number"
              value={config.grid_size}
              onChange={(e) => onChange('grid_size', parseInt(e.target.value, 10))}
              step={1}
              min={5}
              max={200}
              style={styles.input}
            />
          </ConfigField>
          <ConfigField
            label="World Length"
            unit="m"
            tooltip="Physical length of one side of the square simulation domain."
          >
            <input
              type="number"
              value={config.world_len}
              onChange={(e) => onChange('world_len', parseFloat(e.target.value))}
              step={1e-5}
              min={0}
              style={styles.input}
            />
            <span style={styles.displayVal}>{formatScientific(config.world_len)} m</span>
          </ConfigField>
        </ConfigFieldGroup>

        <ToggleField
          label="Enable Extracellular Matrix (ECM)"
          checked={config.is_ecm}
          onChange={(v) => onChange('is_ecm', v)}
          tooltip="Enable explicit modeling of the extracellular matrix, including extracellular ion diffusion and voltages."
        />
      </ConfigSection>

      {/* Visual preview of cell cluster */}
      <ConfigSection title="Cluster Preview" description="Approximate visualization of the cell lattice geometry.">
        <div style={styles.preview}>
          <CellClusterPreview config={config} />
        </div>
      </ConfigSection>
    </div>
  );
};

/** Simple SVG preview of cell cluster arrangement. */
const CellClusterPreview: React.FC<{ config: SpaceConfig }> = ({ config }) => {
  const size = 300;
  const gridN = Math.min(config.grid_size, 15); // limit for rendering
  const isHex = config.cell_lattice_type === 'hex';
  const spacing = size / (gridN + 1);
  const cellR = Math.max(3, spacing * 0.35);

  const cells: { cx: number; cy: number }[] = [];
  for (let row = 0; row < gridN; row++) {
    for (let col = 0; col < gridN; col++) {
      let cx = (col + 1) * spacing;
      let cy = (row + 1) * spacing;
      if (isHex && row % 2 === 1) {
        cx += spacing * 0.5;
      }
      // disorder
      cx += (Math.sin(row * 7 + col * 13) * spacing * config.cell_lattice_disorder * 0.3);
      cy += (Math.cos(row * 11 + col * 3) * spacing * config.cell_lattice_disorder * 0.3);
      cells.push({ cx, cy });
    }
  }

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ backgroundColor: 'var(--color-bg-primary)', borderRadius: 'var(--radius-md)' }}>
      {cells.map((cell, i) => (
        <circle
          key={i}
          cx={cell.cx}
          cy={cell.cy}
          r={cellR}
          fill="var(--color-accent)"
          fillOpacity={0.6}
          stroke="var(--color-accent)"
          strokeOpacity={0.8}
          strokeWidth={1}
        />
      ))}
    </svg>
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
  displayVal: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
    marginTop: 'var(--spacing-xs)',
    display: 'block',
  },
  radioGroup: {
    display: 'flex',
    gap: 'var(--spacing-lg)',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-xs)',
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-secondary)',
    cursor: 'pointer',
  },
  preview: {
    display: 'flex',
    justifyContent: 'center',
  },
};
