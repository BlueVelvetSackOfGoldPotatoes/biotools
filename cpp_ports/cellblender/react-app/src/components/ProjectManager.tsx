import React, { useRef } from 'react';
import type { DataModel } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';
import { downloadProject, loadProjectFromFile } from '../utils/projectIO';
import { createDefaultDataModel } from '../utils/defaults';

interface Props {
  dataModel: DataModel;
  actions: DataModelActions;
}

export const ProjectManager: React.FC<Props> = ({ dataModel, actions }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSave = () => {
    downloadProject(dataModel);
  };

  const handleLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dm = await loadProjectFromFile(file);
      actions.setDataModel(dm);
    } catch (err) {
      alert(`Error loading project: ${err}`);
    }
    // Reset file input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleNew = () => {
    if (confirm('Create a new project? All unsaved changes will be lost.')) {
      actions.setDataModel(createDefaultDataModel());
    }
  };

  // Summary stats
  const stats = {
    molecules: dataModel.molecules.length,
    reactions: dataModel.reactions.length,
    releaseSites: dataModel.releaseSites.length,
    geometryObjects: dataModel.geometryObjects.length,
    surfaceClasses: dataModel.surfaceClasses.length,
    parameters: dataModel.parameters.length,
    reactionOutputs: dataModel.reactionOutputs.length,
  };

  return (
    <div className="panel">
      <h2>Project</h2>
      <p className="panel-description">
        Save and load CellBlender projects as JSON files. The project file
        contains the complete data model including all molecules, reactions,
        geometry, and simulation settings.
      </p>

      <div className="form-section">
        <h3>Scene</h3>
        <div className="form-grid">
          <label>Scene Name:</label>
          <input
            type="text"
            value={dataModel.sceneName}
            onChange={e => actions.updateDM('sceneName', e.target.value)}
          />
        </div>
      </div>

      <div className="btn-group project-actions">
        <button className="btn btn-primary" onClick={handleNew}>New Project</button>
        <button className="btn btn-primary" onClick={handleSave}>Save Project</button>
        <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()}>
          Load Project
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleLoad}
        />
      </div>

      <div className="form-section">
        <h3>Project Summary</h3>
        <table className="stats-table">
          <tbody>
            <tr><td>Molecules</td><td>{stats.molecules}</td></tr>
            <tr><td>Reactions</td><td>{stats.reactions}</td></tr>
            <tr><td>Release Sites</td><td>{stats.releaseSites}</td></tr>
            <tr><td>Geometry Objects</td><td>{stats.geometryObjects}</td></tr>
            <tr><td>Surface Classes</td><td>{stats.surfaceClasses}</td></tr>
            <tr><td>Parameters</td><td>{stats.parameters}</td></tr>
            <tr><td>Reaction Outputs</td><td>{stats.reactionOutputs}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="form-section">
        <h3>Visualization Output</h3>
        <div className="form-grid">
          <label>Export All Molecules:</label>
          <input
            type="checkbox"
            checked={dataModel.vizOutput.exportAll}
            onChange={e => actions.updateDM('vizOutput', { ...dataModel.vizOutput, exportAll: e.target.checked })}
          />

          <label>All Iterations:</label>
          <input
            type="checkbox"
            checked={dataModel.vizOutput.allIterations}
            onChange={e => actions.updateDM('vizOutput', { ...dataModel.vizOutput, allIterations: e.target.checked })}
          />

          {!dataModel.vizOutput.allIterations && (
            <>
              <label>Start:</label>
              <input type="text" value={dataModel.vizOutput.start}
                onChange={e => actions.updateDM('vizOutput', { ...dataModel.vizOutput, start: e.target.value })} />
              <label>End:</label>
              <input type="text" value={dataModel.vizOutput.endVal}
                onChange={e => actions.updateDM('vizOutput', { ...dataModel.vizOutput, endVal: e.target.value })} />
              <label>Step:</label>
              <input type="text" value={dataModel.vizOutput.step}
                onChange={e => actions.updateDM('vizOutput', { ...dataModel.vizOutput, step: e.target.value })} />
            </>
          )}

          <label>Export All as ASCII:</label>
          <input
            type="checkbox"
            checked={dataModel.initialization.exportAllAscii}
            onChange={e => actions.updateDM('initialization', {
              ...dataModel.initialization, exportAllAscii: e.target.checked
            })}
          />
        </div>
      </div>
    </div>
  );
};
