import React, { useState } from 'react';
import type {
  SurfaceClass, SurfaceClassProperty,
  SurfaceClassPropertyAffected, SurfaceClassPropertyOrient, SurfaceClassPropertyType,
} from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';
import { createDefaultSurfaceClassProperty } from '../utils/defaults';

interface Props {
  surfaceClasses: SurfaceClass[];
  moleculeNames: string[];
  actions: DataModelActions;
}

const ORIENT_LABELS: Record<SurfaceClassPropertyOrient, string> = {
  TOP_FRONT: "Top/Front",
  BOTTOM_BACK: "Bottom/Back",
  IGNORE: "Ignore",
};

const TYPE_LABELS: Record<SurfaceClassPropertyType, string> = {
  ABSORPTIVE: "Absorptive",
  TRANSPARENT: "Transparent",
  REFLECTIVE: "Reflective",
  CLAMP_CONCENTRATION: "Clamp Concentration",
};

export const SurfaceClassEditor: React.FC<Props> = ({ surfaceClasses, moleculeNames, actions }) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedPropIdx, setSelectedPropIdx] = useState<number>(-1);
  const selected = surfaceClasses.find(s => s.id === selectedId);

  const handleAdd = () => {
    const sc = actions.addSurfaceClass();
    setSelectedId(sc.id);
  };

  const handleAddProperty = () => {
    if (!selected) return;
    const prop = createDefaultSurfaceClassProperty();
    actions.updateSurfaceClass(selected.id, {
      properties: [...selected.properties, prop],
    });
    setSelectedPropIdx(selected.properties.length);
  };

  const handleRemoveProperty = (idx: number) => {
    if (!selected) return;
    const newProps = selected.properties.filter((_, i) => i !== idx);
    actions.updateSurfaceClass(selected.id, { properties: newProps });
    setSelectedPropIdx(-1);
  };

  const handleUpdateProperty = (idx: number, updates: Partial<SurfaceClassProperty>) => {
    if (!selected) return;
    const newProps = selected.properties.map((p, i) =>
      i === idx ? { ...p, ...updates } : p
    );
    actions.updateSurfaceClass(selected.id, { properties: newProps });
  };

  const selectedProp = selected && selectedPropIdx >= 0 ? selected.properties[selectedPropIdx] : null;

  const formatProperty = (p: SurfaceClassProperty): string => {
    const mol = p.affectedMols === 'SINGLE' ? p.molecule : p.affectedMols;
    return `${mol} | ${ORIENT_LABELS[p.orient]} | ${TYPE_LABELS[p.classType]}`;
  };

  return (
    <div className="panel">
      <h2>Surface Classes</h2>
      <p className="panel-description">
        Define surface class properties (absorptive, transparent, reflective, clamp concentration)
        that control how molecules interact with surfaces.
      </p>

      <div className="list-panel">
        <div className="list-header">
          <span>Surface Classes ({surfaceClasses.length})</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => {
                if (selectedId) {
                  actions.removeSurfaceClass(selectedId);
                  setSelectedId(null);
                }
              }}
              disabled={!selected}
            >
              Remove
            </button>
          </div>
        </div>

        <div className="item-list">
          {surfaceClasses.map(sc => (
            <div
              key={sc.id}
              className={`item-row ${selectedId === sc.id ? 'selected' : ''}`}
              onClick={() => { setSelectedId(sc.id); setSelectedPropIdx(-1); }}
            >
              <span className="item-name">{sc.name}</span>
              <span className="item-detail">{sc.properties.length} properties</span>
            </div>
          ))}
          {surfaceClasses.length === 0 && (
            <div className="empty-list">No surface classes defined.</div>
          )}
        </div>
      </div>

      {selected && (
        <div className="properties-panel">
          <h3>Surface Class: {selected.name}</h3>
          <div className="form-grid">
            <label>Name:</label>
            <input
              type="text"
              value={selected.name}
              onChange={e => actions.updateSurfaceClass(selected.id, { name: e.target.value })}
            />
            <label>Description:</label>
            <input
              type="text"
              value={selected.description}
              onChange={e => actions.updateSurfaceClass(selected.id, { description: e.target.value })}
            />
          </div>

          <div className="list-panel" style={{ marginTop: '1rem' }}>
            <div className="list-header">
              <span>Properties ({selected.properties.length})</span>
              <div className="btn-group">
                <button className="btn btn-sm btn-primary" onClick={handleAddProperty}>+ Add</button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => handleRemoveProperty(selectedPropIdx)}
                  disabled={selectedPropIdx < 0}
                >
                  Remove
                </button>
              </div>
            </div>
            <div className="item-list compact">
              {selected.properties.map((prop, idx) => (
                <div
                  key={prop.id}
                  className={`item-row ${selectedPropIdx === idx ? 'selected' : ''}`}
                  onClick={() => setSelectedPropIdx(idx)}
                >
                  <span className="item-name">{formatProperty(prop)}</span>
                </div>
              ))}
            </div>
          </div>

          {selectedProp && (
            <div className="sub-properties">
              <h4>Property Details</h4>
              <div className="form-grid">
                <label>Affected Molecules:</label>
                <select
                  value={selectedProp.affectedMols}
                  onChange={e => handleUpdateProperty(selectedPropIdx, {
                    affectedMols: e.target.value as SurfaceClassPropertyAffected
                  })}
                >
                  <option value="ALL_MOLECULES">All Molecules</option>
                  <option value="ALL_VOLUME_MOLECULES">All Volume Molecules</option>
                  <option value="ALL_SURFACE_MOLECULES">All Surface Molecules</option>
                  <option value="SINGLE">Single Molecule</option>
                </select>

                {selectedProp.affectedMols === 'SINGLE' && (
                  <>
                    <label>Molecule:</label>
                    <select
                      value={selectedProp.molecule}
                      onChange={e => handleUpdateProperty(selectedPropIdx, { molecule: e.target.value })}
                    >
                      <option value="">-- Select --</option>
                      {moleculeNames.map(n => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </>
                )}

                <label>Orientation:</label>
                <select
                  value={selectedProp.orient}
                  onChange={e => handleUpdateProperty(selectedPropIdx, {
                    orient: e.target.value as SurfaceClassPropertyOrient
                  })}
                >
                  <option value="TOP_FRONT">Top/Front</option>
                  <option value="BOTTOM_BACK">Bottom/Back</option>
                  <option value="IGNORE">Ignore</option>
                </select>

                <label>Type:</label>
                <select
                  value={selectedProp.classType}
                  onChange={e => handleUpdateProperty(selectedPropIdx, {
                    classType: e.target.value as SurfaceClassPropertyType
                  })}
                >
                  <option value="ABSORPTIVE">Absorptive</option>
                  <option value="TRANSPARENT">Transparent</option>
                  <option value="REFLECTIVE">Reflective</option>
                  <option value="CLAMP_CONCENTRATION">Clamp Concentration</option>
                </select>

                {selectedProp.classType === 'CLAMP_CONCENTRATION' && (
                  <>
                    <label>Clamp Value:</label>
                    <input
                      type="text"
                      value={selectedProp.clampValue}
                      onChange={e => handleUpdateProperty(selectedPropIdx, { clampValue: e.target.value })}
                    />
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
