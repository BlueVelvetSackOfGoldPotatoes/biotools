import React, { useState } from 'react';
import type { SweepParameter, ParameterDef } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  parameters: ParameterDef[];
  sweepParameters: SweepParameter[];
  totalRuns: number;
  actions: DataModelActions;
}

export const ParameterSweep: React.FC<Props> = ({
  parameters, sweepParameters, totalRuns, actions,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = sweepParameters.find(s => s.id === selectedId);

  const handleAdd = () => {
    const sp = actions.addSweepParameter();
    setSelectedId(sp.id);
  };

  // Compute all sweep points for preview
  const sweepPoints: Array<Record<string, number>> = [];
  if (sweepParameters.length > 0 && totalRuns <= 1000) {
    for (let i = 0; i < totalRuns; i++) {
      const point: Record<string, number> = {};
      let idx = i;
      for (let j = sweepParameters.length - 1; j >= 0; j--) {
        const sp = sweepParameters[j];
        const n = Math.max(1, sp.values.length);
        point[sp.name] = sp.values[idx % n] ?? 0;
        idx = Math.floor(idx / n);
      }
      sweepPoints.push(point);
    }
  }

  return (
    <div className="panel">
      <h2>Parameter Sweep</h2>
      <p className="panel-description">
        Configure parameter sweeps to run multiple simulations across a range of parameter values.
        Each parameter's values are combined as a full Cartesian product.
      </p>

      {/* Parameters section */}
      <div className="form-section">
        <h3>Model Parameters</h3>
        <div className="list-panel">
          <div className="list-header">
            <span>Parameters ({parameters.length})</span>
            <div className="btn-group">
              <button className="btn btn-sm btn-primary" onClick={() => actions.addParameter()}>+ Add</button>
            </div>
          </div>
          <div className="item-list compact">
            {parameters.map(p => (
              <div key={p.id} className="item-row param-row">
                <input
                  type="text"
                  className="param-name-input"
                  value={p.name}
                  onChange={e => actions.updateParameter(p.id, { name: e.target.value })}
                  placeholder="Name"
                />
                <span className="param-eq">=</span>
                <input
                  type="text"
                  className="param-expr-input"
                  value={p.expression}
                  onChange={e => actions.updateParameter(p.id, { expression: e.target.value })}
                  placeholder="Expression"
                />
                <span className="param-val">({p.value})</span>
                <button className="btn btn-sm btn-danger"
                  onClick={() => actions.removeParameter(p.id)}>x</button>
              </div>
            ))}
            {parameters.length === 0 && (
              <div className="empty-list">No parameters defined.</div>
            )}
          </div>
        </div>
      </div>

      {/* Sweep Parameters */}
      <div className="form-section">
        <h3>Sweep Parameters</h3>
        <div className="list-panel">
          <div className="list-header">
            <span>Sweep Variables ({sweepParameters.length})</span>
            <div className="btn-group">
              <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
              <button className="btn btn-sm btn-danger"
                onClick={() => { if (selectedId) { actions.removeSweepParameter(selectedId); setSelectedId(null); } }}
                disabled={!selected}>
                Remove
              </button>
            </div>
          </div>
          <div className="item-list">
            {sweepParameters.map(sp => (
              <div
                key={sp.id}
                className={`item-row ${selectedId === sp.id ? 'selected' : ''}`}
                onClick={() => setSelectedId(sp.id)}
              >
                <span className="item-name">{sp.name || '(unnamed)'}</span>
                <span className="item-detail">
                  [{sp.start} .. {sp.end}] step {sp.step} ({sp.values.length} values)
                </span>
              </div>
            ))}
            {sweepParameters.length === 0 && (
              <div className="empty-list">No sweep parameters. Add one to configure parameter sweeps.</div>
            )}
          </div>
        </div>

        {selected && (
          <div className="properties-panel">
            <h3>Sweep Variable: {selected.name || '(unnamed)'}</h3>
            <div className="form-grid">
              <label>Parameter Name:</label>
              <input type="text" value={selected.name}
                onChange={e => actions.updateSweepParameter(selected.id, { name: e.target.value })}
                placeholder="e.g., diffusion_rate" />

              <label>Start:</label>
              <input type="number" value={selected.start} step="any"
                onChange={e => actions.updateSweepParameter(selected.id, { start: parseFloat(e.target.value) || 0 })} />

              <label>End:</label>
              <input type="number" value={selected.end} step="any"
                onChange={e => actions.updateSweepParameter(selected.id, { end: parseFloat(e.target.value) || 1 })} />

              <label>Step:</label>
              <input type="number" value={selected.step} step="any" min="0.0001"
                onChange={e => actions.updateSweepParameter(selected.id, { step: parseFloat(e.target.value) || 0.1 })} />
            </div>

            {selected.values.length > 0 && (
              <div className="sweep-values">
                <strong>Values ({selected.values.length}):</strong>
                <div className="values-list">
                  {selected.values.slice(0, 50).map((v, i) => (
                    <span key={i} className="value-chip">{v}</span>
                  ))}
                  {selected.values.length > 50 && <span className="value-chip">... +{selected.values.length - 50} more</span>}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sweep summary */}
      <div className="form-section">
        <h3>Sweep Summary</h3>
        <div className="sweep-summary">
          <p><strong>Total simulation runs:</strong> {totalRuns}</p>
          {sweepParameters.map(sp => (
            <p key={sp.id}>
              <code>{sp.name}</code>: {sp.values.length} values
            </p>
          ))}
        </div>

        {sweepPoints.length > 0 && sweepPoints.length <= 100 && (
          <div className="sweep-preview">
            <h4>Preview (first {Math.min(sweepPoints.length, 100)} points)</h4>
            <div className="sweep-table-container">
              <table className="sweep-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {sweepParameters.map(sp => <th key={sp.id}>{sp.name}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {sweepPoints.slice(0, 100).map((point, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      {sweepParameters.map(sp => (
                        <td key={sp.id}>{point[sp.name]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {totalRuns > 100 && (
          <p className="hint-box">
            Too many sweep points to preview ({totalRuns} total).
            Showing summary only.
          </p>
        )}
      </div>
    </div>
  );
};
