import React, { useState } from 'react';
import type { MoleculeSpecies, MoleculeType } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  molecules: MoleculeSpecies[];
  actions: DataModelActions;
}

export const MoleculeEditor: React.FC<Props> = ({ molecules, actions }) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = molecules.find(m => m.id === selectedId);

  const handleAdd = () => {
    const mol = actions.addMolecule();
    setSelectedId(mol.id);
  };

  const handleDuplicate = () => {
    if (!selected) return;
    const mol = actions.addMolecule();
    actions.updateMolecule(mol.id, {
      ...selected,
      id: mol.id,
      name: selected.name + '_copy',
    });
    setSelectedId(mol.id);
  };

  return (
    <div className="panel">
      <h2>Molecules (Species Definitions)</h2>
      <p className="panel-description">
        Define volume (3D) and surface (2D) molecule species with diffusion constants
        for MCell simulation.
      </p>

      <div className="list-panel">
        <div className="list-header">
          <span>Defined Molecules ({molecules.length})</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
            <button className="btn btn-sm" onClick={handleDuplicate} disabled={!selected}>Duplicate</button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => {
                if (selectedId) {
                  actions.removeMolecule(selectedId);
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
          {molecules.map(mol => (
            <div
              key={mol.id}
              className={`item-row ${selectedId === mol.id ? 'selected' : ''}`}
              onClick={() => setSelectedId(mol.id)}
            >
              <span className={`mol-type-badge ${mol.molType === '2D' ? 'surface' : 'volume'}`}>
                {mol.molType}
              </span>
              <span className="item-name">{mol.name || '(unnamed)'}</span>
              <span className="item-detail">D={mol.diffusionConstant}</span>
            </div>
          ))}
          {molecules.length === 0 && (
            <div className="empty-list">No molecules defined. Click "+ Add" to create one.</div>
          )}
        </div>
      </div>

      {selected && (
        <div className="properties-panel">
          <h3>Molecule Properties</h3>
          <div className="form-grid">
            <label>Name:</label>
            <input
              type="text"
              value={selected.name}
              onChange={e => actions.updateMolecule(selected.id, { name: e.target.value })}
              placeholder="e.g., vol_A"
            />

            <label>Type:</label>
            <select
              value={selected.molType}
              onChange={e => actions.updateMolecule(selected.id, { molType: e.target.value as MoleculeType })}
            >
              <option value="3D">Volume (3D)</option>
              <option value="2D">Surface (2D)</option>
            </select>

            <label>Diffusion Constant:</label>
            <input
              type="text"
              value={selected.diffusionConstant}
              onChange={e => actions.updateMolecule(selected.id, { diffusionConstant: e.target.value })}
              placeholder="e.g., 1e-6"
            />

            <label>Description:</label>
            <input
              type="text"
              value={selected.description}
              onChange={e => actions.updateMolecule(selected.id, { description: e.target.value })}
            />
          </div>

          <details className="advanced-section">
            <summary>Advanced Properties</summary>
            <div className="form-grid">
              <label>Custom Time Step:</label>
              <input
                type="text"
                value={selected.customTimeStep}
                onChange={e => actions.updateMolecule(selected.id, { customTimeStep: e.target.value })}
                placeholder="Leave empty for default"
              />

              <label>Custom Space Step:</label>
              <input
                type="text"
                value={selected.customSpaceStep}
                onChange={e => actions.updateMolecule(selected.id, { customSpaceStep: e.target.value })}
                placeholder="Leave empty for default"
              />

              <label>Maximum Step Length:</label>
              <input
                type="text"
                value={selected.maximumStepLength}
                onChange={e => actions.updateMolecule(selected.id, { maximumStepLength: e.target.value })}
                placeholder="Leave empty for default"
              />

              <label>Target Only:</label>
              <input
                type="checkbox"
                checked={selected.targetOnly}
                onChange={e => actions.updateMolecule(selected.id, { targetOnly: e.target.checked })}
              />

              <label>Export Visualization:</label>
              <input
                type="checkbox"
                checked={selected.exportViz}
                onChange={e => actions.updateMolecule(selected.id, { exportViz: e.target.checked })}
              />

              <label>BNGL Label:</label>
              <input
                type="text"
                value={selected.bnglLabel}
                onChange={e => actions.updateMolecule(selected.id, { bnglLabel: e.target.value })}
              />
            </div>
          </details>
        </div>
      )}
    </div>
  );
};
