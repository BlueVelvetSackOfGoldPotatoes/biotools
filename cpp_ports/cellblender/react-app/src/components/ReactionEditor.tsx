import React, { useState } from 'react';
import type { Reaction, ReactionType } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  reactions: Reaction[];
  moleculeNames: string[];
  actions: DataModelActions;
}

export const ReactionEditor: React.FC<Props> = ({ reactions, moleculeNames, actions }) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = reactions.find(r => r.id === selectedId);

  const handleAdd = () => {
    const rxn = actions.addReaction();
    setSelectedId(rxn.id);
  };

  const formatReaction = (r: Reaction): string => {
    const arrow = r.rxnType === 'reversible' ? '<->' : '->';
    const reactants = r.reactants || '?';
    const products = r.products || '?';
    return `${reactants} ${arrow} ${products}`;
  };

  return (
    <div className="panel">
      <h2>Reactions</h2>
      <p className="panel-description">
        Define chemical reactions with reactants, products, and rate constants.
        Supports irreversible and reversible reactions with optional variable rates.
      </p>

      <div className="list-panel">
        <div className="list-header">
          <span>Defined Reactions ({reactions.length})</span>
          <div className="btn-group">
            <button className="btn btn-sm btn-primary" onClick={handleAdd}>+ Add</button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => {
                if (selectedId) {
                  actions.removeReaction(selectedId);
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
          {reactions.map(rxn => (
            <div
              key={rxn.id}
              className={`item-row ${selectedId === rxn.id ? 'selected' : ''}`}
              onClick={() => setSelectedId(rxn.id)}
            >
              <span className={`rxn-type-badge ${rxn.rxnType}`}>
                {rxn.rxnType === 'reversible' ? 'Rev' : 'Irr'}
              </span>
              <span className="item-name">{formatReaction(rxn)}</span>
              <span className="item-detail">k={rxn.fwdRate}</span>
            </div>
          ))}
          {reactions.length === 0 && (
            <div className="empty-list">No reactions defined. Click "+ Add" to create one.</div>
          )}
        </div>
      </div>

      {selected && (
        <div className="properties-panel">
          <h3>Reaction Properties</h3>
          <div className="form-grid">
            <label>Reactants:</label>
            <input
              type="text"
              value={selected.reactants}
              onChange={e => actions.updateReaction(selected.id, { reactants: e.target.value })}
              placeholder="e.g., A + B or A' + B,"
            />

            <label>Products:</label>
            <input
              type="text"
              value={selected.products}
              onChange={e => actions.updateReaction(selected.id, { products: e.target.value })}
              placeholder="e.g., C or NULL"
            />

            <label>Type:</label>
            <select
              value={selected.rxnType}
              onChange={e => actions.updateReaction(selected.id, { rxnType: e.target.value as ReactionType })}
            >
              <option value="irreversible">Irreversible (->)</option>
              <option value="reversible">Reversible ({'<->'})</option>
            </select>

            <label>Forward Rate:</label>
            <input
              type="text"
              value={selected.fwdRate}
              onChange={e => actions.updateReaction(selected.id, { fwdRate: e.target.value })}
              placeholder="e.g., 1e8"
            />

            {selected.rxnType === 'reversible' && (
              <>
                <label>Backward Rate:</label>
                <input
                  type="text"
                  value={selected.bkwdRate}
                  onChange={e => actions.updateReaction(selected.id, { bkwdRate: e.target.value })}
                  placeholder="e.g., 1e2"
                />
              </>
            )}

            <label>Reaction Name:</label>
            <input
              type="text"
              value={selected.rxnName}
              onChange={e => actions.updateReaction(selected.id, { rxnName: e.target.value })}
              placeholder="Optional name for counting"
            />

            <label>Description:</label>
            <input
              type="text"
              value={selected.description}
              onChange={e => actions.updateReaction(selected.id, { description: e.target.value })}
            />
          </div>

          <details className="advanced-section">
            <summary>Variable Rate Constant</summary>
            <div className="form-grid">
              <label>Enable Variable Rate:</label>
              <input
                type="checkbox"
                checked={selected.variableRateSwitch}
                onChange={e => actions.updateReaction(selected.id, { variableRateSwitch: e.target.checked })}
              />

              {selected.variableRateSwitch && (
                <>
                  <label>Variable Rate Text:</label>
                  <textarea
                    className="code-textarea"
                    value={selected.variableRateText}
                    onChange={e => actions.updateReaction(selected.id, {
                      variableRateText: e.target.value,
                      variableRateValid: e.target.value.trim().length > 0,
                    })}
                    placeholder="time rate_value (one pair per line)"
                    rows={6}
                  />
                  <label>Status:</label>
                  <span className={selected.variableRateValid ? 'status-ok' : 'status-error'}>
                    {selected.variableRateValid ? 'Valid' : 'No variable rate data'}
                  </span>
                </>
              )}
            </div>
          </details>

          {moleculeNames.length > 0 && (
            <div className="hint-box">
              <strong>Available molecules:</strong> {moleculeNames.join(', ')}
            </div>
          )}

          <div className="reaction-preview">
            <strong>Reaction: </strong>
            <code>{formatReaction(selected)}</code>
            {' '}
            <code>
              {selected.rxnType === 'reversible'
                ? `[>${selected.fwdRate}, <${selected.bkwdRate}]`
                : `[${selected.fwdRate}]`
              }
            </code>
          </div>
        </div>
      )}
    </div>
  );
};
