import React from 'react';
import type { ExportConfig, AnimationExport, PlotExport, CSVExport } from '../../../types/simulation';
import { ConfigSection, ConfigField, ConfigFieldGroup, ToggleField } from '../shared/ConfigWidgets';

interface ExportSettingsProps {
  config: ExportConfig;
  onChange: (path: string, value: unknown) => void;
}

const COLORMAPS = [
  'viridis', 'plasma', 'inferno', 'magma', 'cividis',
  'RdBu_r', 'RdYlBu_r', 'coolwarm', 'bwr', 'seismic',
  'bone', 'gray', 'hot', 'cool', 'spring', 'summer',
  'jet', 'rainbow', 'turbo', 'gnuplot', 'gnuplot2',
];

export const ExportSettings: React.FC<ExportSettingsProps> = ({ config, onChange }) => {
  const updateAnimation = (index: number, field: keyof AnimationExport, value: unknown) => {
    const updated = config.animations.map((a, i) =>
      i === index ? { ...a, [field]: value } : a,
    );
    onChange('animations', updated);
  };

  const updatePlotCell = (index: number, field: keyof PlotExport, value: unknown) => {
    const updated = config.plots_cell.map((p, i) =>
      i === index ? { ...p, [field]: value } : p,
    );
    onChange('plots_cell', updated);
  };

  const updatePlotCells = (index: number, field: keyof PlotExport, value: unknown) => {
    const updated = config.plots_cells.map((p, i) =>
      i === index ? { ...p, [field]: value } : p,
    );
    onChange('plots_cells', updated);
  };

  const updateCSV = (index: number, field: keyof CSVExport, value: unknown) => {
    const updated = config.csvs.map((c, i) =>
      i === index ? { ...c, [field]: value } : c,
    );
    onChange('csvs', updated);
  };

  return (
    <div>
      <h2 style={styles.pageTitle}>Export / Visualization Settings</h2>
      <p style={styles.pageDesc}>
        Configure simulation output exports: animations, plots, CSV data files,
        and visualization parameters such as colormaps and cell indices.
      </p>

      <ConfigSection title="General Export Settings" description="Global visualization parameters.">
        <ConfigFieldGroup columns={2}>
          <ToggleField
            label="Show Cell Indices"
            checked={config.is_show_cell_indices}
            onChange={(v) => onChange('is_show_cell_indices', v)}
            tooltip="Overlay cell index numbers on exported visualizations."
          />
          <ConfigField label="Single Cell Index" tooltip="Index of the cell to monitor for single-cell time series plots.">
            <input
              type="number"
              value={config.single_cell_index}
              onChange={(e) => onChange('single_cell_index', parseInt(e.target.value, 10))}
              min={0}
              step={1}
              style={styles.input}
            />
          </ConfigField>
        </ConfigFieldGroup>
      </ConfigSection>

      <ConfigSection title="Colormaps" description="Color mapping schemes for different visualization types.">
        <ConfigFieldGroup columns={2}>
          {[
            { key: 'colormap_diverging_name', label: 'Diverging (Vmem)' },
            { key: 'colormap_sequential_name', label: 'Sequential (concentrations)' },
            { key: 'colormap_gj_name', label: 'Gap Junctions' },
            { key: 'colormap_grn_name', label: 'GRN' },
          ].map(({ key, label }) => (
            <ConfigField key={key} label={label}>
              <select
                value={(config as unknown as Record<string, unknown>)[key] as string}
                onChange={(e) => onChange(key, e.target.value)}
                style={styles.select}
              >
                {COLORMAPS.map((cm) => (
                  <option key={cm} value={cm}>{cm}</option>
                ))}
              </select>
            </ConfigField>
          ))}
        </ConfigFieldGroup>
      </ConfigSection>

      <ConfigSection title="Animation Exports" description="Configure animated visualization outputs." collapsible>
        {config.animations.map((anim, index) => (
          <div key={index} style={styles.exportItem}>
            <div style={styles.exportHeader}>
              <ToggleField
                label={anim.name}
                checked={anim.enabled}
                onChange={(v) => updateAnimation(index, 'enabled', v)}
              />
            </div>
            {anim.enabled && (
              <ConfigFieldGroup columns={3}>
                <ConfigField label="Type">
                  <input
                    type="text"
                    value={anim.type}
                    onChange={(e) => updateAnimation(index, 'type', e.target.value)}
                    style={styles.input}
                  />
                </ConfigField>
                <ConfigField label="Format">
                  <select
                    value={anim.save_format}
                    onChange={(e) => updateAnimation(index, 'save_format', e.target.value)}
                    style={styles.select}
                  >
                    <option value="png">PNG</option>
                    <option value="svg">SVG</option>
                    <option value="pdf">PDF</option>
                  </select>
                </ConfigField>
                <ConfigField label="DPI">
                  <input
                    type="number"
                    value={anim.dpi}
                    onChange={(e) => updateAnimation(index, 'dpi', parseInt(e.target.value, 10))}
                    min={72}
                    max={600}
                    step={10}
                    style={styles.input}
                  />
                </ConfigField>
              </ConfigFieldGroup>
            )}
          </div>
        ))}
      </ConfigSection>

      <ConfigSection title="Single Cell Plots" description="Time series plots for a single monitored cell." collapsible>
        {config.plots_cell.map((plot, index) => (
          <div key={index} style={styles.exportItem}>
            <ToggleField
              label={plot.name}
              checked={plot.enabled}
              onChange={(v) => updatePlotCell(index, 'enabled', v)}
            />
            {plot.enabled && (
              <ConfigFieldGroup columns={3}>
                <ConfigField label="Type">
                  <input type="text" value={plot.type} onChange={(e) => updatePlotCell(index, 'type', e.target.value)} style={styles.input} />
                </ConfigField>
                <ConfigField label="Format">
                  <select value={plot.save_format} onChange={(e) => updatePlotCell(index, 'save_format', e.target.value)} style={styles.select}>
                    <option value="png">PNG</option><option value="svg">SVG</option><option value="pdf">PDF</option>
                  </select>
                </ConfigField>
                <ConfigField label="DPI">
                  <input type="number" value={plot.dpi} onChange={(e) => updatePlotCell(index, 'dpi', parseInt(e.target.value, 10))} min={72} max={600} step={10} style={styles.input} />
                </ConfigField>
              </ConfigFieldGroup>
            )}
          </div>
        ))}
      </ConfigSection>

      <ConfigSection title="Cell Cluster Plots" description="2D heatmap plots of the full cell cluster." collapsible>
        {config.plots_cells.map((plot, index) => (
          <div key={index} style={styles.exportItem}>
            <ToggleField
              label={plot.name}
              checked={plot.enabled}
              onChange={(v) => updatePlotCells(index, 'enabled', v)}
            />
            {plot.enabled && (
              <ConfigFieldGroup columns={3}>
                <ConfigField label="Type">
                  <input type="text" value={plot.type} onChange={(e) => updatePlotCells(index, 'type', e.target.value)} style={styles.input} />
                </ConfigField>
                <ConfigField label="Format">
                  <select value={plot.save_format} onChange={(e) => updatePlotCells(index, 'save_format', e.target.value)} style={styles.select}>
                    <option value="png">PNG</option><option value="svg">SVG</option><option value="pdf">PDF</option>
                  </select>
                </ConfigField>
                <ConfigField label="DPI">
                  <input type="number" value={plot.dpi} onChange={(e) => updatePlotCells(index, 'dpi', parseInt(e.target.value, 10))} min={72} max={600} step={10} style={styles.input} />
                </ConfigField>
              </ConfigFieldGroup>
            )}
          </div>
        ))}
      </ConfigSection>

      <ConfigSection title="CSV Exports" description="Comma-separated value data exports." collapsible>
        {config.csvs.map((csv, index) => (
          <div key={index} style={styles.exportItem}>
            <ToggleField
              label={csv.name}
              checked={csv.enabled}
              onChange={(v) => updateCSV(index, 'enabled', v)}
            />
            {csv.enabled && (
              <ConfigField label="Type">
                <input type="text" value={csv.type} onChange={(e) => updateCSV(index, 'type', e.target.value)} style={styles.input} />
              </ConfigField>
            )}
          </div>
        ))}
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
  select: {
    width: '100%',
  },
  exportItem: {
    padding: 'var(--spacing-sm) 0',
    borderBottom: '1px solid var(--color-border)',
  },
  exportHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
};
