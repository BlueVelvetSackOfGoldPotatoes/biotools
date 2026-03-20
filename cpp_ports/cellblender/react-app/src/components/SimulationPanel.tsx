import React, { useState, useEffect, useRef } from 'react';
import type { InitializationParams, SimulationControl, SimulationState } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';

interface Props {
  initialization: InitializationParams;
  simControl: SimulationControl;
  simState: SimulationState;
  actions: DataModelActions;
}

export const SimulationPanel: React.FC<Props> = ({ initialization, simControl, simState, actions }) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  const updateInit = (updates: Partial<InitializationParams>) => {
    actions.updateDM('initialization', { ...initialization, ...updates });
  };

  const updateSim = (updates: Partial<SimulationControl>) => {
    actions.updateDM('simControl', { ...simControl, ...updates });
  };

  const handleStart = () => {
    const totalIterations = parseInt(initialization.iterations) || 1000;
    const totalSeeds = simControl.endSeed - simControl.startSeed + 1;

    actions.setSimState({
      status: 'running',
      progress: 0,
      currentSeed: simControl.startSeed,
      totalSeeds,
      currentIteration: 0,
      totalIterations,
      errorMessage: '',
      startTime: Date.now(),
      elapsedTime: 0,
    });
  };

  const handleStop = () => {
    actions.setSimState(prev => ({
      ...prev,
      status: 'idle',
    }));
  };

  // Simulate progress when running (demo mode)
  useEffect(() => {
    if (simState.status === 'running') {
      timerRef.current = setInterval(() => {
        actions.setSimState(prev => {
          const newProgress = Math.min(prev.progress + 0.5 + Math.random() * 1.5, 100);
          const elapsed = prev.startTime ? (Date.now() - prev.startTime) / 1000 : 0;
          const currentIter = Math.floor((newProgress / 100) * prev.totalIterations);
          if (newProgress >= 100) {
            return {
              ...prev,
              status: 'completed',
              progress: 100,
              currentIteration: prev.totalIterations,
              elapsedTime: elapsed,
            };
          }
          return {
            ...prev,
            progress: newProgress,
            currentIteration: currentIter,
            elapsedTime: elapsed,
          };
        });
      }, 200);

      return () => clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [simState.status, actions]);

  const formatTime = (seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="panel">
      <h2>Simulation</h2>
      <p className="panel-description">
        Configure and run MCell simulations. Set iteration count, time step,
        seed range, and other simulation parameters.
      </p>

      {/* Status Banner */}
      {simState.status !== 'idle' && (
        <div className={`status-banner ${simState.status}`}>
          <div className="status-top">
            <span className="status-label">
              {simState.status === 'running' && 'Simulation Running...'}
              {simState.status === 'completed' && 'Simulation Completed'}
              {simState.status === 'error' && `Error: ${simState.errorMessage}`}
              {simState.status === 'preparing' && 'Preparing...'}
            </span>
            <span className="status-time">{formatTime(simState.elapsedTime)}</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar" style={{ width: `${simState.progress}%` }} />
          </div>
          <div className="status-detail">
            Iteration {simState.currentIteration} / {simState.totalIterations}
            {' | '}
            Seed {simState.currentSeed} of {simState.totalSeeds}
            {' | '}
            {simState.progress.toFixed(1)}%
          </div>
        </div>
      )}

      <div className="form-section">
        <h3>Run Settings</h3>
        <div className="form-grid">
          <label>Iterations:</label>
          <input type="text" value={initialization.iterations}
            onChange={e => updateInit({ iterations: e.target.value })} />

          <label>Time Step (s):</label>
          <input type="text" value={initialization.timeStep}
            onChange={e => updateInit({ timeStep: e.target.value })} />

          <label>Start Seed:</label>
          <input type="number" value={simControl.startSeed}
            onChange={e => updateSim({ startSeed: parseInt(e.target.value) || 1 })} min={1} />

          <label>End Seed:</label>
          <input type="number" value={simControl.endSeed}
            onChange={e => updateSim({ endSeed: parseInt(e.target.value) || 1 })} min={1} />

          <label>MCell Binary:</label>
          <input type="text" value={simControl.mcellBinary}
            onChange={e => updateSim({ mcellBinary: e.target.value })} />

          <label>Export Format:</label>
          <select value={simControl.exportFormat}
            onChange={e => updateSim({ exportFormat: e.target.value })}>
            <option value="mcell_mdl_modular">MCell MDL (Modular)</option>
            <option value="mcell_mdl_single">MCell MDL (Single File)</option>
          </select>
        </div>

        <div className="btn-group" style={{ marginTop: '1rem' }}>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleStart}
            disabled={simState.status === 'running'}
          >
            {simState.status === 'running' ? 'Running...' : 'Run Simulation'}
          </button>
          <button
            className="btn btn-danger"
            onClick={handleStop}
            disabled={simState.status !== 'running'}
          >
            Stop
          </button>
        </div>
      </div>

      <div className="form-section">
        <h3>Initialization Parameters</h3>
        <div className="form-grid">
          <label>Vacancy Search Distance:</label>
          <input type="text" value={initialization.vacancySearchDistance}
            onChange={e => updateInit({ vacancySearchDistance: e.target.value })} />

          <label>Surface Grid Density:</label>
          <input type="text" value={initialization.surfaceGridDensity}
            onChange={e => updateInit({ surfaceGridDensity: e.target.value })} />

          <label>Accurate 3D Reactions:</label>
          <input type="checkbox" checked={initialization.accurate3dReactions}
            onChange={e => updateInit({ accurate3dReactions: e.target.checked })} />

          <label>Center Molecules on Grid:</label>
          <input type="checkbox" checked={initialization.centerMoleculesGrid}
            onChange={e => updateInit({ centerMoleculesGrid: e.target.checked })} />

          <label>Microscopic Reversibility:</label>
          <select value={initialization.microscopicReversibility}
            onChange={e => updateInit({ microscopicReversibility: e.target.value })}>
            <option value="ON">ON</option>
            <option value="OFF">OFF</option>
          </select>
        </div>
      </div>

      <details className="advanced-section">
        <summary>Advanced: Time/Space Steps</summary>
        <div className="form-grid">
          <label>Time Step Max:</label>
          <input type="text" value={initialization.timeStepMax}
            onChange={e => updateInit({ timeStepMax: e.target.value })} placeholder="Optional" />
          <label>Space Step:</label>
          <input type="text" value={initialization.spaceStep}
            onChange={e => updateInit({ spaceStep: e.target.value })} placeholder="Optional" />
          <label>Interaction Radius:</label>
          <input type="text" value={initialization.interactionRadius}
            onChange={e => updateInit({ interactionRadius: e.target.value })} placeholder="Optional" />
          <label>Radial Directions:</label>
          <input type="text" value={initialization.radialDirections}
            onChange={e => updateInit({ radialDirections: e.target.value })} placeholder="Optional" />
          <label>Radial Subdivisions:</label>
          <input type="text" value={initialization.radialSubdivisions}
            onChange={e => updateInit({ radialSubdivisions: e.target.value })} placeholder="Optional" />
        </div>
      </details>

      <details className="advanced-section">
        <summary>Partitions</summary>
        <div className="form-grid">
          <label>Include Partitions:</label>
          <input type="checkbox" checked={initialization.partitions.include}
            onChange={e => updateInit({
              partitions: { ...initialization.partitions, include: e.target.checked }
            })} />
          {initialization.partitions.include && (
            <>
              <label>X Start:</label>
              <input type="number" value={initialization.partitions.xStart} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, xStart: parseFloat(e.target.value) || 0 }
                })} />
              <label>X End:</label>
              <input type="number" value={initialization.partitions.xEnd} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, xEnd: parseFloat(e.target.value) || 0 }
                })} />
              <label>X Step:</label>
              <input type="number" value={initialization.partitions.xStep} step="0.001"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, xStep: parseFloat(e.target.value) || 0.05 }
                })} />
              <label>Y Start:</label>
              <input type="number" value={initialization.partitions.yStart} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, yStart: parseFloat(e.target.value) || 0 }
                })} />
              <label>Y End:</label>
              <input type="number" value={initialization.partitions.yEnd} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, yEnd: parseFloat(e.target.value) || 0 }
                })} />
              <label>Y Step:</label>
              <input type="number" value={initialization.partitions.yStep} step="0.001"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, yStep: parseFloat(e.target.value) || 0.05 }
                })} />
              <label>Z Start:</label>
              <input type="number" value={initialization.partitions.zStart} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, zStart: parseFloat(e.target.value) || 0 }
                })} />
              <label>Z End:</label>
              <input type="number" value={initialization.partitions.zEnd} step="0.01"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, zEnd: parseFloat(e.target.value) || 0 }
                })} />
              <label>Z Step:</label>
              <input type="number" value={initialization.partitions.zStep} step="0.001"
                onChange={e => updateInit({
                  partitions: { ...initialization.partitions, zStep: parseFloat(e.target.value) || 0.05 }
                })} />
            </>
          )}
        </div>
      </details>

      <details className="advanced-section">
        <summary>Notifications</summary>
        <div className="form-grid">
          <label>All Notifications:</label>
          <select value={initialization.allNotifications}
            onChange={e => updateInit({ allNotifications: e.target.value })}>
            <option value="INDIVIDUAL">Individual</option>
            <option value="ON">All ON</option>
            <option value="OFF">All OFF</option>
          </select>
          {initialization.allNotifications === 'INDIVIDUAL' && (
            <>
              <label>Probability Report:</label>
              <select value={initialization.probabilityReport}
                onChange={e => updateInit({ probabilityReport: e.target.value })}>
                <option value="ON">ON</option>
                <option value="OFF">OFF</option>
                <option value="THRESHOLD">THRESHOLD</option>
              </select>
              <label>Diffusion Constant Report:</label>
              <select value={initialization.diffusionConstantReport}
                onChange={e => updateInit({ diffusionConstantReport: e.target.value })}>
                <option value="BRIEF">BRIEF</option>
                <option value="ON">ON</option>
                <option value="OFF">OFF</option>
              </select>
              <label>Iteration Report:</label>
              <input type="checkbox" checked={initialization.iterationReport}
                onChange={e => updateInit({ iterationReport: e.target.checked })} />
              <label>Progress Report:</label>
              <input type="checkbox" checked={initialization.progressReport}
                onChange={e => updateInit({ progressReport: e.target.checked })} />
              <label>Release Event Report:</label>
              <input type="checkbox" checked={initialization.releaseEventReport}
                onChange={e => updateInit({ releaseEventReport: e.target.checked })} />
              <label>Final Summary:</label>
              <input type="checkbox" checked={initialization.finalSummary}
                onChange={e => updateInit({ finalSummary: e.target.checked })} />
            </>
          )}
        </div>
      </details>

      <details className="advanced-section">
        <summary>Warnings</summary>
        <div className="form-grid">
          <label>All Warnings:</label>
          <select value={initialization.allWarnings}
            onChange={e => updateInit({ allWarnings: e.target.value })}>
            <option value="INDIVIDUAL">Individual</option>
            <option value="WARNING">All WARNING</option>
            <option value="ERROR">All ERROR</option>
            <option value="IGNORED">All IGNORED</option>
          </select>
          {initialization.allWarnings === 'INDIVIDUAL' && (
            <>
              <label>Degenerate Polygons:</label>
              <select value={initialization.degeneratePolygons}
                onChange={e => updateInit({ degeneratePolygons: e.target.value })}>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="IGNORED">IGNORED</option>
              </select>
              <label>Missing Surface Orientation:</label>
              <select value={initialization.missingSurfaceOrientation}
                onChange={e => updateInit({ missingSurfaceOrientation: e.target.value })}>
                <option value="ERROR">ERROR</option>
                <option value="WARNING">WARNING</option>
                <option value="IGNORED">IGNORED</option>
              </select>
              <label>High Reaction Probability:</label>
              <select value={initialization.highReactionProbability}
                onChange={e => updateInit({ highReactionProbability: e.target.value })}>
                <option value="IGNORED">IGNORED</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
              </select>
              <label>Negative Reaction Rate:</label>
              <select value={initialization.negativeReactionRate}
                onChange={e => updateInit({ negativeReactionRate: e.target.value })}>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="IGNORED">IGNORED</option>
              </select>
              <label>Lifetime Too Short:</label>
              <select value={initialization.lifetimeTooShort}
                onChange={e => updateInit({ lifetimeTooShort: e.target.value })}>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
                <option value="IGNORED">IGNORED</option>
              </select>
              {initialization.lifetimeTooShort === 'WARNING' && (
                <>
                  <label>Lifetime Threshold:</label>
                  <input type="number" value={initialization.lifetimeThreshold}
                    onChange={e => updateInit({ lifetimeThreshold: parseFloat(e.target.value) || 50 })} />
                </>
              )}
            </>
          )}
        </div>
      </details>
    </div>
  );
};
