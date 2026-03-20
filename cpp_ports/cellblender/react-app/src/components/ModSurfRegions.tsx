import React, { useState } from 'react';
import type { ModSurfaceRegion } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  modSurfRegions: ModSurfaceRegion[];
  objectNames: string[];
  surfaceClassNames: string[];
  actions: DataModelActions;
}

export const ModSurfRegions: React.FC<Props> = ({
  modSurfRegions, objectNames, surfaceClassNames, actions,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = modSurfRegions.find(m => m.id === selectedId);

  const handleAdd = () => {
    const msr = actions.addModSurfRegion();
    setSelectedId(msr.id);
  };

  return (
    <div className="panel">
      <h2>Assign Surface Classes</h2>
      <p className="panel-description">
        Assign surface classes to specific regions of geometry objects. This determines
        how molecules interact with specific parts of the simulation geometry.
      </p>

      <div className="list-panel">
        <div className="list-header">
          <span>Surface Region Assignments ({modSurfRegions.length})</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => {
                if (selectedId) { actions.removeModSurfRegion(selectedId); setSelectedId(null); }
              }}
              disabled={!selected}
            >
              Remove
            </button>
          </div>
        </div>
        <div className="item-list">
          {modSurfRegions.map(msr => (
            <div
              key={msr.id}
              className={`item-row ${selectedId === msr.id ? 'selected' : ''}`}
              onClick={() => setSelectedId(msr.id)}
            >
              <span className="item-name">
                {msr.objectName || '?'}[{msr.regionName || '?'}]
              </span>
              <span className="item-detail">
                {msr.surfClassName || '(no class)'}
              </span>
            </div>
          ))}
          {modSurfRegions.length === 0 && (
            <div className="empty-list">No surface region assignments defined.</div>
          )}
        </div>
      </div>

      {selected && (
        <div className="properties-panel">
          <h3>Assignment Properties</h3>
          <div className="form-grid">
            <label>Object:</label>
            <select
              value={selected.objectName}
              onChange={e => actions.updateModSurfRegion(selected.id, { objectName: e.target.value })}
            >
              <option value="">-- Select Object --</option>
              {objectNames.map(n => <option key={n} value={n}>{n}</option>)}
            </select>

            <label>Region:</label>
            <input
              type="text"
              value={selected.regionName}
              onChange={e => actions.updateModSurfRegion(selected.id, { regionName: e.target.value })}
              placeholder="e.g., ALL or region_name"
            />

            <label>Surface Class:</label>
            <select
              value={selected.surfClassName}
              onChange={e => actions.updateModSurfRegion(selected.id, { surfClassName: e.target.value })}
            >
              <option value="">-- Select Surface Class --</option>
              {surfaceClassNames.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
        </div>
      )}
    </div>
  );
};
