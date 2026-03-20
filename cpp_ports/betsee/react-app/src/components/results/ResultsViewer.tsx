import React, { useState, useMemo } from 'react';
import type { SimulationResults, SimulationConfig } from '../../types/simulation';
import { voltageToColor, formatScientific } from '../../utils/format';

interface ResultsViewerProps {
  results: SimulationResults | null;
  config: SimulationConfig;
}

type ResultsTab = 'heatmap' | 'timeseries' | 'currents' | 'gapjunctions';

export const ResultsViewer: React.FC<ResultsViewerProps> = ({ results, config }) => {
  const [activeTab, setActiveTab] = useState<ResultsTab>('heatmap');
  const [selectedIon, setSelectedIon] = useState<string>('Na');

  if (!results) {
    return (
      <div>
        <h2 style={styles.pageTitle}>Results Viewer</h2>
        <div style={styles.emptyState}>
          <div style={styles.emptyIcon}>V</div>
          <h3 style={styles.emptyTitle}>No Results Available</h3>
          <p style={styles.emptyText}>
            Run a simulation to generate results. Navigate to the Run Control
            panel to start a simulation.
          </p>
        </div>
      </div>
    );
  }

  const tabs: { id: ResultsTab; label: string }[] = [
    { id: 'heatmap', label: 'Vmem Heatmap' },
    { id: 'timeseries', label: 'Ion Concentrations' },
    { id: 'currents', label: 'Current Density' },
    { id: 'gapjunctions', label: 'Gap Junctions' },
  ];

  return (
    <div>
      <h2 style={styles.pageTitle}>Results Viewer</h2>
      <p style={styles.pageDesc}>
        Visualize simulation results including membrane voltage patterns,
        ion concentration dynamics, current density fields, and gap junction states.
      </p>

      {/* Tab bar */}
      <div style={styles.tabBar}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tab,
              ...(activeTab === tab.id ? styles.tabActive : {}),
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={styles.tabContent}>
        {activeTab === 'heatmap' && (
          <VmemHeatmap results={results} config={config} />
        )}
        {activeTab === 'timeseries' && (
          <IonTimeSeries
            results={results}
            selectedIon={selectedIon}
            onIonChange={setSelectedIon}
          />
        )}
        {activeTab === 'currents' && (
          <CurrentDensityView results={results} config={config} />
        )}
        {activeTab === 'gapjunctions' && (
          <GapJunctionView results={results} config={config} />
        )}
      </div>
    </div>
  );
};

/**
 * 2D Voltage Heatmap - displays Vmem across the cell cluster as colored circles.
 */
const VmemHeatmap: React.FC<{ results: SimulationResults; config: SimulationConfig }> = ({
  results,
  config,
}) => {
  const svgSize = 500;

  const { vmin, vmax } = useMemo(() => {
    const voltages = results.cells.map((c) => c.vmem);
    return {
      vmin: Math.min(...voltages),
      vmax: Math.max(...voltages),
    };
  }, [results.cells]);

  const scaledCells = useMemo(() => {
    const worldLen = config.space.world_len;
    const padding = 20;
    const scale = (svgSize - padding * 2) / worldLen;
    return results.cells.map((cell) => ({
      cx: padding + cell.x * scale,
      cy: padding + cell.y * scale,
      r: Math.max(3, config.space.cell_radius * scale * 0.8),
      vmem: cell.vmem,
      color: voltageToColor(cell.vmem, vmin, vmax),
    }));
  }, [results.cells, config, vmin, vmax]);

  return (
    <div>
      <div style={styles.vizHeader}>
        <h3 style={styles.vizTitle}>Membrane Voltage (Vmem) - 2D Cell Cluster</h3>
        <span style={styles.vizStat}>
          Range: {vmin.toFixed(1)} mV to {vmax.toFixed(1)} mV |{' '}
          {results.cells.length} cells
        </span>
      </div>

      <div style={styles.vizContainer}>
        <svg
          width={svgSize}
          height={svgSize}
          viewBox={`0 0 ${svgSize} ${svgSize}`}
          style={styles.svg}
        >
          {/* Cell circles */}
          {scaledCells.map((cell, i) => (
            <circle
              key={i}
              cx={cell.cx}
              cy={cell.cy}
              r={cell.r}
              fill={cell.color}
              stroke="rgba(0,0,0,0.2)"
              strokeWidth={0.5}
            >
              <title>Cell {i}: {cell.vmem.toFixed(2)} mV</title>
            </circle>
          ))}
        </svg>

        {/* Color bar */}
        <div style={styles.colorBar}>
          <span style={styles.colorBarLabel}>{vmax.toFixed(1)} mV</span>
          <div style={styles.colorBarGradient}>
            {Array.from({ length: 20 }, (_, i) => {
              const t = 1 - i / 19;
              const v = vmin + t * (vmax - vmin);
              return (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    backgroundColor: voltageToColor(v, vmin, vmax),
                  }}
                />
              );
            })}
          </div>
          <span style={styles.colorBarLabel}>{vmin.toFixed(1)} mV</span>
        </div>
      </div>
    </div>
  );
};

/**
 * Ion concentration time series plots.
 */
const IonTimeSeries: React.FC<{
  results: SimulationResults;
  selectedIon: string;
  onIonChange: (ion: string) => void;
}> = ({ results, selectedIon, onIonChange }) => {
  const ions = Object.keys(results.ion_concentrations);
  const data = results.ion_concentrations[selectedIon] || [];

  const svgWidth = 600;
  const svgHeight = 300;
  const padding = { top: 20, right: 30, bottom: 40, left: 60 };

  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  const { tMin, tMax, cMin, cMax } = useMemo(() => {
    if (data.length === 0) return { tMin: 0, tMax: 1, cMin: 0, cMax: 1 };
    const times = data.map((d) => d.time);
    const values = data.map((d) => d.value);
    return {
      tMin: Math.min(...times),
      tMax: Math.max(...times),
      cMin: Math.min(...values) * 0.95,
      cMax: Math.max(...values) * 1.05,
    };
  }, [data]);

  const pathD = useMemo(() => {
    if (data.length === 0) return '';
    return data
      .map((d, i) => {
        const x = padding.left + ((d.time - tMin) / (tMax - tMin)) * plotWidth;
        const y = padding.top + (1 - (d.value - cMin) / (cMax - cMin)) * plotHeight;
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  }, [data, tMin, tMax, cMin, cMax, plotWidth, plotHeight]);

  // Also show Vmem time series
  const vmemData = results.vmem_timeseries;
  const vmemBounds = useMemo(() => {
    if (vmemData.length === 0) return { vMin: -80, vMax: 0 };
    const vs = vmemData.map(d => d.value);
    return { vMin: Math.min(...vs) - 5, vMax: Math.max(...vs) + 5 };
  }, [vmemData]);

  const vmemPathD = useMemo(() => {
    if (vmemData.length === 0) return '';
    return vmemData.map((d, i) => {
      const x = padding.left + ((d.time - tMin) / (tMax - tMin)) * plotWidth;
      const y = padding.top + (1 - (d.value - vmemBounds.vMin) / (vmemBounds.vMax - vmemBounds.vMin)) * plotHeight;
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  }, [vmemData, tMin, tMax, vmemBounds, plotWidth, plotHeight]);

  const ION_COLORS: Record<string, string> = {
    Na: '#e06040',
    K: '#5090d0',
    Ca: '#40b070',
    Cl: '#a060c0',
  };

  return (
    <div>
      <div style={styles.vizHeader}>
        <h3 style={styles.vizTitle}>Vmem Time Series (Single Cell)</h3>
      </div>

      <svg width={svgWidth} height={svgHeight} style={styles.svg}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={`hgrid-vmem-${t}`}
            x1={padding.left} y1={padding.top + t * plotHeight}
            x2={svgWidth - padding.right} y2={padding.top + t * plotHeight}
            stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="3,3"
          />
        ))}
        {/* Vmem curve */}
        <path d={vmemPathD} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
        {/* Axes labels */}
        <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}>
          Time (s)
        </text>
        <text x={12} y={svgHeight / 2} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}
          transform={`rotate(-90, 12, ${svgHeight / 2})`}>
          Vmem (mV)
        </text>
        {/* Y-axis tick labels */}
        {[0, 0.5, 1].map((t) => (
          <text key={`ytick-vmem-${t}`}
            x={padding.left - 5} y={padding.top + t * plotHeight + 4}
            textAnchor="end" fill="var(--color-text-muted)" fontSize={10}>
            {(vmemBounds.vMax - t * (vmemBounds.vMax - vmemBounds.vMin)).toFixed(0)}
          </text>
        ))}
        {/* X-axis tick labels */}
        {[0, 0.5, 1].map((t) => (
          <text key={`xtick-${t}`}
            x={padding.left + t * plotWidth} y={svgHeight - padding.bottom + 15}
            textAnchor="middle" fill="var(--color-text-muted)" fontSize={10}>
            {(tMin + t * (tMax - tMin)).toFixed(1)}
          </text>
        ))}
      </svg>

      <div style={{ ...styles.vizHeader, marginTop: 'var(--spacing-xl)' }}>
        <h3 style={styles.vizTitle}>Ion Concentration Time Series</h3>
        <div style={styles.ionSelector}>
          {ions.map((ion) => (
            <button
              key={ion}
              onClick={() => onIonChange(ion)}
              style={{
                ...styles.ionBtn,
                ...(selectedIon === ion ? {
                  backgroundColor: ION_COLORS[ion] || 'var(--color-accent)',
                  color: '#fff',
                } : {}),
              }}
            >
              {ion}+
            </button>
          ))}
        </div>
      </div>

      <svg width={svgWidth} height={svgHeight} style={styles.svg}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={`hgrid-${t}`}
            x1={padding.left} y1={padding.top + t * plotHeight}
            x2={svgWidth - padding.right} y2={padding.top + t * plotHeight}
            stroke="var(--color-border)" strokeWidth={0.5} strokeDasharray="3,3"
          />
        ))}
        {/* Data curve */}
        <path
          d={pathD}
          fill="none"
          stroke={ION_COLORS[selectedIon] || 'var(--color-accent)'}
          strokeWidth={2}
        />
        {/* Axes labels */}
        <text x={svgWidth / 2} y={svgHeight - 5} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}>
          Time (s)
        </text>
        <text x={12} y={svgHeight / 2} textAnchor="middle" fill="var(--color-text-muted)" fontSize={11}
          transform={`rotate(-90, 12, ${svgHeight / 2})`}>
          [{selectedIon}] (mM)
        </text>
        {/* Y-axis tick labels */}
        {[0, 0.5, 1].map((t) => (
          <text key={`ytick-${t}`}
            x={padding.left - 5} y={padding.top + t * plotHeight + 4}
            textAnchor="end" fill="var(--color-text-muted)" fontSize={10}>
            {formatScientific(cMax - t * (cMax - cMin), 3)}
          </text>
        ))}
      </svg>
    </div>
  );
};

/**
 * Current density vector field visualization.
 */
const CurrentDensityView: React.FC<{ results: SimulationResults; config: SimulationConfig }> = ({
  results,
  config,
}) => {
  const svgSize = 500;
  const padding = 20;
  const worldLen = config.space.world_len;
  const scale = (svgSize - padding * 2) / worldLen;

  // Find max current magnitude for scaling arrows
  const maxJ = useMemo(() => {
    return Math.max(
      ...results.current_density.map((c) =>
        Math.sqrt(c.jx * c.jx + c.jy * c.jy),
      ),
    );
  }, [results.current_density]);

  const arrowScale = 30 / Math.max(maxJ, 1e-12);

  return (
    <div>
      <div style={styles.vizHeader}>
        <h3 style={styles.vizTitle}>Current Density Vector Field</h3>
        <span style={styles.vizStat}>
          Max |J| = {formatScientific(maxJ)} A/m^2 | {results.current_density.length} vectors
        </span>
      </div>

      <svg width={svgSize} height={svgSize} viewBox={`0 0 ${svgSize} ${svgSize}`} style={styles.svg}>
        {results.current_density.map((pt, i) => {
          const cx = padding + pt.x * scale;
          const cy = padding + pt.y * scale;
          const dx = pt.jx * arrowScale;
          const dy = pt.jy * arrowScale;
          const mag = Math.sqrt(pt.jx * pt.jx + pt.jy * pt.jy);
          const opacity = Math.min(1, mag / maxJ + 0.2);

          return (
            <g key={i}>
              <line
                x1={cx}
                y1={cy}
                x2={cx + dx}
                y2={cy + dy}
                stroke="var(--color-accent)"
                strokeWidth={1.5}
                opacity={opacity}
                markerEnd="url(#arrowhead)"
              />
            </g>
          );
        })}
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="var(--color-accent)" />
          </marker>
        </defs>
      </svg>
    </div>
  );
};

/**
 * Gap junction state visualization.
 */
const GapJunctionView: React.FC<{ results: SimulationResults; config: SimulationConfig }> = ({
  results,
  config,
}) => {
  const svgSize = 500;
  const padding = 20;
  const worldLen = config.space.world_len;
  const scale = (svgSize - padding * 2) / worldLen;

  const { gMin, gMax } = useMemo(() => {
    const gs = results.gap_junction_states.map((gj) => gj.conductance);
    return {
      gMin: Math.min(...gs),
      gMax: Math.max(...gs),
    };
  }, [results.gap_junction_states]);

  return (
    <div>
      <div style={styles.vizHeader}>
        <h3 style={styles.vizTitle}>Gap Junction Conductance Network</h3>
        <span style={styles.vizStat}>
          {results.gap_junction_states.length} junctions |{' '}
          G range: {formatScientific(gMin)} - {formatScientific(gMax)} S
        </span>
      </div>

      <svg width={svgSize} height={svgSize} viewBox={`0 0 ${svgSize} ${svgSize}`} style={styles.svg}>
        {/* Gap junction connections */}
        {results.gap_junction_states.map((gj, i) => {
          const cellA = results.cells[gj.from];
          const cellB = results.cells[gj.to];
          if (!cellA || !cellB) return null;

          const x1 = padding + cellA.x * scale;
          const y1 = padding + cellA.y * scale;
          const x2 = padding + cellB.x * scale;
          const y2 = padding + cellB.y * scale;

          const normalized = (gj.conductance - gMin) / (gMax - gMin || 1);
          const opacity = 0.1 + normalized * 0.8;
          const width = 0.5 + normalized * 2;

          return (
            <line
              key={i}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="#40b070"
              strokeWidth={width}
              opacity={opacity}
            >
              <title>GJ {gj.from}-{gj.to}: {formatScientific(gj.conductance)} S</title>
            </line>
          );
        })}

        {/* Cell centers */}
        {results.cells.map((cell, i) => (
          <circle
            key={i}
            cx={padding + cell.x * scale}
            cy={padding + cell.y * scale}
            r={2}
            fill="var(--color-text-muted)"
          />
        ))}
      </svg>
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
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 'var(--spacing-xxl)',
    minHeight: '400px',
  },
  emptyIcon: {
    width: '64px',
    height: '64px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-bg-tertiary)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 'var(--font-size-xxl)',
    color: 'var(--color-text-muted)',
    marginBottom: 'var(--spacing-lg)',
  },
  emptyTitle: {
    fontSize: 'var(--font-size-lg)',
    fontWeight: 600,
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-sm)',
  },
  emptyText: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-muted)',
    textAlign: 'center' as const,
    maxWidth: '400px',
  },
  tabBar: {
    display: 'flex',
    gap: '2px',
    marginBottom: 'var(--spacing-lg)',
    backgroundColor: 'var(--color-bg-secondary)',
    borderRadius: 'var(--radius-md)',
    padding: '3px',
    border: '1px solid var(--color-border)',
  },
  tab: {
    padding: 'var(--spacing-sm) var(--spacing-lg)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 500,
    color: 'var(--color-text-secondary)',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
  tabActive: {
    backgroundColor: 'var(--color-bg-elevated)',
    color: 'var(--color-text-primary)',
  },
  tabContent: {
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    padding: 'var(--spacing-xl)',
  },
  vizHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--spacing-md)',
  },
  vizTitle: {
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
  },
  vizStat: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
  },
  vizContainer: {
    display: 'flex',
    gap: 'var(--spacing-lg)',
    alignItems: 'stretch',
  },
  svg: {
    backgroundColor: 'var(--color-bg-primary)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid var(--color-border)',
  },
  colorBar: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: 'var(--spacing-xs)',
    width: '30px',
  },
  colorBarLabel: {
    fontSize: '9px',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-family-mono)',
    whiteSpace: 'nowrap' as const,
  },
  colorBarGradient: {
    flex: 1,
    width: '16px',
    display: 'flex',
    flexDirection: 'column' as const,
    borderRadius: '2px',
    overflow: 'hidden',
    border: '1px solid var(--color-border)',
  },
  ionSelector: {
    display: 'flex',
    gap: 'var(--spacing-xs)',
  },
  ionBtn: {
    padding: 'var(--spacing-xs) var(--spacing-sm)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 'var(--font-size-xs)',
    fontWeight: 600,
    border: '1px solid var(--color-border)',
    backgroundColor: 'var(--color-bg-elevated)',
    color: 'var(--color-text-secondary)',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
};
