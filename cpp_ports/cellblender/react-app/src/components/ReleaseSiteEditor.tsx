import React, { useState } from 'react';
import type {
  ReleaseSite, ReleaseShape, ReleaseOrient, QuantityType,
  ReleasePattern,
} from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  releaseSites: ReleaseSite[];
  releasePatterns: ReleasePattern[];
  moleculeNames: string[];
  objectNames: string[];
  actions: DataModelActions;
}

const SHAPE_LABELS: Record<ReleaseShape, string> = {
  CUBIC: 'Cubic',
  SPHERICAL: 'Spherical',
  SPHERICAL_SHELL: 'Spherical Shell',
  LIST: 'List',
  OBJECT: 'Object/Region',
};

export const ReleaseSiteEditor: React.FC<Props> = ({
  releaseSites, releasePatterns, moleculeNames, objectNames, actions,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showPatterns, setShowPatterns] = useState(false);
  const [selectedPatternId, setSelectedPatternId] = useState<string | null>(null);
  const selected = releaseSites.find(r => r.id === selectedId);
  const selectedPattern = releasePatterns.find(r => r.id === selectedPatternId);

  const handleAdd = () => {
    const rel = actions.addReleaseSite();
    setSelectedId(rel.id);
  };

  return (
    <div className="panel">
      <h2>Molecule Release / Placement</h2>
      <p className="panel-description">
        Define where and how molecules are released into the simulation geometry.
        Specify release shape, quantity, orientation, and optional release patterns.
      </p>

      <div className="tab-bar">
        <button className={`tab ${!showPatterns ? 'active' : ''}`} onClick={() => setShowPatterns(false)}>
          Release Sites
        </button>
        <button className={`tab ${showPatterns ? 'active' : ''}`} onClick={() => setShowPatterns(true)}>
          Release Patterns
        </button>
      </div>

      {!showPatterns ? (
        <>
          <div className="list-panel">
            <div className="list-header">
              <span>Release Sites ({releaseSites.length})</span>
              <div className="btn-group">
                <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => { if (selectedId) { actions.removeReleaseSite(selectedId); setSelectedId(null); } }}
                  disabled={!selected}
                >
                  Remove
                </button>
              </div>
            </div>
            <div className="item-list">
              {releaseSites.map(rel => (
                <div
                  key={rel.id}
                  className={`item-row ${selectedId === rel.id ? 'selected' : ''}`}
                  onClick={() => setSelectedId(rel.id)}
                >
                  <span className="item-name">{rel.name}</span>
                  <span className="item-detail">{rel.molecule || '?'} ({rel.shape})</span>
                </div>
              ))}
              {releaseSites.length === 0 && (
                <div className="empty-list">No release sites defined.</div>
              )}
            </div>
          </div>

          {selected && (
            <div className="properties-panel">
              <h3>Release Site Properties</h3>
              <div className="form-grid">
                <label>Name:</label>
                <input
                  type="text"
                  value={selected.name}
                  onChange={e => actions.updateReleaseSite(selected.id, { name: e.target.value })}
                />

                <label>Molecule:</label>
                <select
                  value={selected.molecule}
                  onChange={e => actions.updateReleaseSite(selected.id, { molecule: e.target.value })}
                >
                  <option value="">-- Select Molecule --</option>
                  {moleculeNames.map(n => <option key={n} value={n}>{n}</option>)}
                </select>

                <label>Shape:</label>
                <select
                  value={selected.shape}
                  onChange={e => actions.updateReleaseSite(selected.id, { shape: e.target.value as ReleaseShape })}
                >
                  {Object.entries(SHAPE_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>{v}</option>
                  ))}
                </select>

                <label>Orientation:</label>
                <select
                  value={selected.orient}
                  onChange={e => actions.updateReleaseSite(selected.id, { orient: e.target.value as ReleaseOrient })}
                >
                  <option value="TOP_FRONT">Top/Front</option>
                  <option value="TOP_BACK">Bottom/Back</option>
                  <option value="MIXED">Mixed</option>
                </select>

                {selected.shape === 'OBJECT' && (
                  <>
                    <label>Object Expression:</label>
                    <input
                      type="text"
                      value={selected.objectExpr}
                      onChange={e => actions.updateReleaseSite(selected.id, { objectExpr: e.target.value })}
                      placeholder="e.g., Cube or Cube[region1]"
                    />
                  </>
                )}

                {['CUBIC', 'SPHERICAL', 'SPHERICAL_SHELL'].includes(selected.shape) && (
                  <>
                    <label>Location X:</label>
                    <input type="text" value={selected.locationX}
                      onChange={e => actions.updateReleaseSite(selected.id, { locationX: e.target.value })} />
                    <label>Location Y:</label>
                    <input type="text" value={selected.locationY}
                      onChange={e => actions.updateReleaseSite(selected.id, { locationY: e.target.value })} />
                    <label>Location Z:</label>
                    <input type="text" value={selected.locationZ}
                      onChange={e => actions.updateReleaseSite(selected.id, { locationZ: e.target.value })} />
                    <label>Site Diameter:</label>
                    <input type="text" value={selected.diameter}
                      onChange={e => actions.updateReleaseSite(selected.id, { diameter: e.target.value })} />
                  </>
                )}

                <label>Quantity Type:</label>
                <select
                  value={selected.quantityType}
                  onChange={e => actions.updateReleaseSite(selected.id, { quantityType: e.target.value as QuantityType })}
                >
                  <option value="NUMBER_TO_RELEASE">Number to Release</option>
                  <option value="GAUSSIAN_RELEASE_NUMBER">Gaussian Release</option>
                  <option value="DENSITY">Density/Concentration</option>
                </select>

                <label>Quantity:</label>
                <input type="text" value={selected.quantity}
                  onChange={e => actions.updateReleaseSite(selected.id, { quantity: e.target.value })} />

                {selected.quantityType === 'GAUSSIAN_RELEASE_NUMBER' && (
                  <>
                    <label>Std. Deviation:</label>
                    <input type="text" value={selected.stddev}
                      onChange={e => actions.updateReleaseSite(selected.id, { stddev: e.target.value })} />
                  </>
                )}

                <label>Probability:</label>
                <input type="text" value={selected.probability}
                  onChange={e => actions.updateReleaseSite(selected.id, { probability: e.target.value })} />

                <label>Release Pattern:</label>
                <select
                  value={selected.pattern}
                  onChange={e => actions.updateReleaseSite(selected.id, { pattern: e.target.value })}
                >
                  <option value="">-- None --</option>
                  {releasePatterns.map(rp => (
                    <option key={rp.id} value={rp.name}>{rp.name}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          <div className="list-panel">
            <div className="list-header">
              <span>Release Patterns ({releasePatterns.length})</span>
              <div className="btn-group">
                <button className="btn btn-sm btn-primary" onClick={() => {
                  const rp = actions.addReleasePattern();
                  setSelectedPatternId(rp.id);
                }}>+ Add</button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => {
                    if (selectedPatternId) {
                      actions.removeReleasePattern(selectedPatternId);
                      setSelectedPatternId(null);
                    }
                  }}
                  disabled={!selectedPattern}
                >
                  Remove
                </button>
              </div>
            </div>
            <div className="item-list">
              {releasePatterns.map(rp => (
                <div
                  key={rp.id}
                  className={`item-row ${selectedPatternId === rp.id ? 'selected' : ''}`}
                  onClick={() => setSelectedPatternId(rp.id)}
                >
                  <span className="item-name">{rp.name}</span>
                </div>
              ))}
              {releasePatterns.length === 0 && (
                <div className="empty-list">No release patterns defined.</div>
              )}
            </div>
          </div>

          {selectedPattern && (
            <div className="properties-panel">
              <h3>Release Pattern Properties</h3>
              <div className="form-grid">
                <label>Name:</label>
                <input type="text" value={selectedPattern.name}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { name: e.target.value })} />
                <label>Delay:</label>
                <input type="text" value={selectedPattern.delay}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { delay: e.target.value })} />
                <label>Release Interval:</label>
                <input type="text" value={selectedPattern.releaseInterval}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { releaseInterval: e.target.value })} />
                <label>Train Duration:</label>
                <input type="text" value={selectedPattern.trainDuration}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { trainDuration: e.target.value })} />
                <label>Train Interval:</label>
                <input type="text" value={selectedPattern.trainInterval}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { trainInterval: e.target.value })} />
                <label>Number of Trains:</label>
                <input type="text" value={selectedPattern.numberOfTrains}
                  onChange={e => actions.updateReleasePattern(selectedPattern.id, { numberOfTrains: e.target.value })} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
