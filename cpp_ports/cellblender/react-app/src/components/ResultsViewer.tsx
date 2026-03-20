import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { TimeSeriesData, AggregatedTimeSeries, ReactionOutput, DataModel } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';
import { createDefaultReactionOutput } from '../utils/defaults';

interface Props {
  reactionOutputs: ReactionOutput[];
  moleculeNames: string[];
  reactionNames: string[];
  objectNames: string[];
  actions: DataModelActions;
}

// Generate demo data for visualization
function generateDemoData(moleculeName: string, iterations: number): TimeSeriesData {
  const times: number[] = [];
  const values: number[] = [];
  const dt = 1e-6;
  let val = 100 + Math.random() * 50;
  for (let i = 0; i <= iterations; i += Math.max(1, Math.floor(iterations / 200))) {
    times.push(i * dt);
    val += (Math.random() - 0.5) * 10;
    val = Math.max(0, val);
    values.push(Math.round(val));
  }
  return { label: moleculeName, times, values };
}

function generateAggregatedDemo(moleculeName: string, iterations: number, nSeeds: number): AggregatedTimeSeries {
  const dt = 1e-6;
  const times: number[] = [];
  const means: number[] = [];
  const stddevs: number[] = [];

  let val = 100 + Math.random() * 50;
  for (let i = 0; i <= iterations; i += Math.max(1, Math.floor(iterations / 200))) {
    times.push(i * dt);
    val += (Math.random() - 0.5) * 8;
    val = Math.max(0, val);
    means.push(Math.round(val));
    stddevs.push(5 + Math.random() * 10);
  }

  return { label: moleculeName, times, means, stddevs, numSeeds: nSeeds };
}

// Simple Canvas-based plot renderer
function renderPlot(
  canvas: HTMLCanvasElement,
  datasets: TimeSeriesData[],
  aggregated: AggregatedTimeSeries[],
  showStddev: boolean
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const margin = { top: 30, right: 20, bottom: 50, left: 70 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  // Collect all data ranges
  let minT = Infinity, maxT = -Infinity, minV = Infinity, maxV = -Infinity;
  for (const ds of datasets) {
    for (const t of ds.times) { minT = Math.min(minT, t); maxT = Math.max(maxT, t); }
    for (const v of ds.values) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
  }
  for (const agg of aggregated) {
    for (const t of agg.times) { minT = Math.min(minT, t); maxT = Math.max(maxT, t); }
    for (let i = 0; i < agg.means.length; i++) {
      const lo = showStddev ? agg.means[i] - agg.stddevs[i] : agg.means[i];
      const hi = showStddev ? agg.means[i] + agg.stddevs[i] : agg.means[i];
      minV = Math.min(minV, lo);
      maxV = Math.max(maxV, hi);
    }
  }

  if (!isFinite(minT)) { minT = 0; maxT = 1; minV = 0; maxV = 1; }
  if (minV === maxV) { minV -= 1; maxV += 1; }

  const scaleX = (t: number) => margin.left + ((t - minT) / (maxT - minT)) * plotW;
  const scaleY = (v: number) => margin.top + plotH - ((v - minV) / (maxV - minV)) * plotH;

  // Grid and axes
  ctx.strokeStyle = '#2a2a4e';
  ctx.lineWidth = 0.5;
  const nGridY = 5;
  for (let i = 0; i <= nGridY; i++) {
    const y = margin.top + (i / nGridY) * plotH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = '#555';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Time (s)', margin.left + plotW / 2, h - 8);
  ctx.save();
  ctx.translate(14, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Count', 0, 0);
  ctx.restore();

  // Tick labels
  ctx.textAlign = 'center';
  const nTicksX = 5;
  for (let i = 0; i <= nTicksX; i++) {
    const t = minT + (i / nTicksX) * (maxT - minT);
    const x = scaleX(t);
    ctx.fillText(t.toExponential(1), x, margin.top + plotH + 18);
  }
  ctx.textAlign = 'right';
  for (let i = 0; i <= nGridY; i++) {
    const v = minV + ((nGridY - i) / nGridY) * (maxV - minV);
    const y = margin.top + (i / nGridY) * plotH;
    ctx.fillText(v.toFixed(0), margin.left - 8, y + 4);
  }

  // Plot colors
  const colors = ['#00d4ff', '#ff6b9d', '#c084fc', '#fbbf24', '#34d399', '#f97316', '#e879f9', '#38bdf8'];

  // Plot aggregated data with stddev bands
  aggregated.forEach((agg, idx) => {
    const color = colors[idx % colors.length];
    if (showStddev && agg.stddevs.length > 0) {
      ctx.fillStyle = color + '20';
      ctx.beginPath();
      for (let i = 0; i < agg.times.length; i++) {
        const x = scaleX(agg.times[i]);
        const y = scaleY(agg.means[i] + agg.stddevs[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let i = agg.times.length - 1; i >= 0; i--) {
        const x = scaleX(agg.times[i]);
        const y = scaleY(agg.means[i] - agg.stddevs[i]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < agg.times.length; i++) {
      const x = scaleX(agg.times[i]);
      const y = scaleY(agg.means[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Plot individual datasets
  datasets.forEach((ds, idx) => {
    const ci = (aggregated.length + idx) % colors.length;
    ctx.strokeStyle = colors[ci];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < ds.times.length; i++) {
      const x = scaleX(ds.times[i]);
      const y = scaleY(ds.values[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });

  // Legend
  const allLabels = [...aggregated.map(a => a.label + ' (agg)'), ...datasets.map(d => d.label)];
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  allLabels.forEach((label, i) => {
    const x = margin.left + 10;
    const y = margin.top + 15 + i * 16;
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(x, y - 8, 10, 10);
    ctx.fillStyle = '#ccc';
    ctx.fillText(label, x + 15, y);
  });

  // Title
  ctx.fillStyle = '#eee';
  ctx.font = 'bold 13px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Reaction Data Output', margin.left + plotW / 2, 18);
}

export const ResultsViewer: React.FC<Props> = ({
  reactionOutputs, moleculeNames, reactionNames, objectNames, actions,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [demoData, setDemoData] = useState<TimeSeriesData[]>([]);
  const [demoAgg, setDemoAgg] = useState<AggregatedTimeSeries[]>([]);
  const [showStddev, setShowStddev] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const selected = reactionOutputs.find(r => r.id === selectedId);

  const handleAdd = () => {
    const ro = actions.addReactionOutput();
    setSelectedId(ro.id);
  };

  const handleGenerateDemo = () => {
    const enabledOutputs = reactionOutputs.filter(r => r.plottingEnabled);
    if (enabledOutputs.length === 0) {
      // Generate for all molecules
      const data = moleculeNames.map(n => generateDemoData(n, 1000));
      setDemoData(data);
      setDemoAgg([]);
    } else {
      const aggs = enabledOutputs.map(ro => {
        const name = ro.rxnOrMol === 'Molecule' ? ro.moleculeName : ro.reactionName;
        return generateAggregatedDemo(name || 'output', 1000, 5);
      });
      setDemoAgg(aggs);
      setDemoData([]);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const container = canvas.parentElement;
    if (container) {
      canvas.width = container.clientWidth;
      canvas.height = 350;
    }
    renderPlot(canvas, demoData, demoAgg, showStddev);
  }, [demoData, demoAgg, showStddev]);

  const handleExportCSV = () => {
    const lines: string[] = [];
    if (demoAgg.length > 0) {
      const headers = ['time', ...demoAgg.flatMap(a => [a.label + '_mean', a.label + '_stddev'])];
      lines.push(headers.join(','));
      const n = demoAgg[0]?.times.length || 0;
      for (let i = 0; i < n; i++) {
        const row = [demoAgg[0].times[i].toExponential(6)];
        for (const a of demoAgg) {
          row.push(a.means[i]?.toFixed(2) || '0');
          row.push(a.stddevs[i]?.toFixed(2) || '0');
        }
        lines.push(row.join(','));
      }
    } else if (demoData.length > 0) {
      const headers = ['time', ...demoData.map(d => d.label)];
      lines.push(headers.join(','));
      const n = demoData[0]?.times.length || 0;
      for (let i = 0; i < n; i++) {
        const row = [demoData[0].times[i].toExponential(6)];
        for (const d of demoData) {
          row.push(d.values[i]?.toFixed(2) || '0');
        }
        lines.push(row.join(','));
      }
    }
    if (lines.length > 0) {
      const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'reaction_data.csv';
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="panel">
      <h2>Reaction Data Output / Results</h2>
      <p className="panel-description">
        Configure which molecules and reactions to count, and visualize simulation results
        as time series plots with optional multi-seed aggregation (mean/stddev).
      </p>

      <div className="list-panel">
        <div className="list-header">
          <span>Output Items ({reactionOutputs.length})</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
            <button className="btn btn-sm" onClick={() => {
              // Add all molecules as world counts
              for (const name of moleculeNames) {
                const ro = actions.addReactionOutput();
                actions.updateReactionOutput(ro.id, {
                  rxnOrMol: 'Molecule',
                  moleculeName: name,
                  countLocation: 'World',
                  name: `Count ${name} in World`,
                });
              }
            }}>+ Add All Molecules</button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => { if (selectedId) { actions.removeReactionOutput(selectedId); setSelectedId(null); } }}
              disabled={!selected}
            >
              Remove
            </button>
          </div>
        </div>
        <div className="item-list">
          {reactionOutputs.map(ro => (
            <div
              key={ro.id}
              className={`item-row ${selectedId === ro.id ? 'selected' : ''}`}
              onClick={() => setSelectedId(ro.id)}
            >
              <input
                type="checkbox"
                checked={ro.plottingEnabled}
                onChange={e => {
                  e.stopPropagation();
                  actions.updateReactionOutput(ro.id, { plottingEnabled: e.target.checked });
                }}
              />
              <span className="item-name">
                {ro.rxnOrMol === 'Molecule' ? ro.moleculeName : ro.reactionName}
              </span>
              <span className="item-detail">
                {ro.rxnOrMol} | {ro.countLocation}
              </span>
            </div>
          ))}
          {reactionOutputs.length === 0 && (
            <div className="empty-list">No output items configured.</div>
          )}
        </div>
      </div>

      {selected && (
        <div className="properties-panel">
          <h3>Output Properties</h3>
          <div className="form-grid">
            <label>Type:</label>
            <select value={selected.rxnOrMol}
              onChange={e => actions.updateReactionOutput(selected.id, {
                rxnOrMol: e.target.value as any
              })}>
              <option value="Molecule">Molecule</option>
              <option value="Reaction">Reaction</option>
              <option value="MDLString">MDL String</option>
            </select>

            {selected.rxnOrMol === 'Molecule' && (
              <>
                <label>Molecule:</label>
                <select value={selected.moleculeName}
                  onChange={e => actions.updateReactionOutput(selected.id, { moleculeName: e.target.value })}>
                  <option value="">-- Select --</option>
                  {moleculeNames.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </>
            )}

            {selected.rxnOrMol === 'Reaction' && (
              <>
                <label>Reaction:</label>
                <select value={selected.reactionName}
                  onChange={e => actions.updateReactionOutput(selected.id, { reactionName: e.target.value })}>
                  <option value="">-- Select --</option>
                  {reactionNames.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </>
            )}

            {selected.rxnOrMol === 'MDLString' && (
              <>
                <label>MDL String:</label>
                <input type="text" value={selected.mdlString}
                  onChange={e => actions.updateReactionOutput(selected.id, { mdlString: e.target.value })} />
                <label>File Prefix:</label>
                <input type="text" value={selected.mdlFilePrefix}
                  onChange={e => actions.updateReactionOutput(selected.id, { mdlFilePrefix: e.target.value })} />
              </>
            )}

            <label>Count Location:</label>
            <select value={selected.countLocation}
              onChange={e => actions.updateReactionOutput(selected.id, {
                countLocation: e.target.value as any
              })}>
              <option value="World">World</option>
              <option value="Object">Object</option>
              <option value="Region">Region</option>
            </select>

            {(selected.countLocation === 'Object' || selected.countLocation === 'Region') && (
              <>
                <label>Object:</label>
                <select value={selected.objectName}
                  onChange={e => actions.updateReactionOutput(selected.id, { objectName: e.target.value })}>
                  <option value="">-- Select --</option>
                  {objectNames.map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </>
            )}

            {selected.countLocation === 'Region' && (
              <>
                <label>Region:</label>
                <input type="text" value={selected.regionName}
                  onChange={e => actions.updateReactionOutput(selected.id, { regionName: e.target.value })} />
              </>
            )}

            <label>Plotting Enabled:</label>
            <input type="checkbox" checked={selected.plottingEnabled}
              onChange={e => actions.updateReactionOutput(selected.id, { plottingEnabled: e.target.checked })} />
          </div>
        </div>
      )}

      <div className="plot-section">
        <div className="list-header">
          <span>Results Plot</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleGenerateDemo}>
              Generate Demo Data
            </button>
            <label className="checkbox-label">
              <input type="checkbox" checked={showStddev} onChange={e => setShowStddev(e.target.checked)} />
              Show Std Dev
            </label>
            <button className="btn btn-sm" onClick={handleExportCSV}
              disabled={demoData.length === 0 && demoAgg.length === 0}>
              Export CSV
            </button>
          </div>
        </div>
        <div className="plot-container">
          <canvas ref={canvasRef} className="plot-canvas" />
        </div>
      </div>
    </div>
  );
};
