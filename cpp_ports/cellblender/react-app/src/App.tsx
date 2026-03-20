import React, { useState } from 'react';
import { useDataModel } from './hooks/useDataModel';
import { MoleculeEditor } from './components/MoleculeEditor';
import { ReactionEditor } from './components/ReactionEditor';
import { GeometryViewer } from './components/GeometryViewer';
import { SurfaceClassEditor } from './components/SurfaceClassEditor';
import { ReleaseSiteEditor } from './components/ReleaseSiteEditor';
import { SimulationPanel } from './components/SimulationPanel';
import { ResultsViewer } from './components/ResultsViewer';
import { ParameterSweep } from './components/ParameterSweep';
import { MDLPreview } from './components/MDLPreview';
import { ProjectManager } from './components/ProjectManager';
import { ModSurfRegions } from './components/ModSurfRegions';

type TabId =
  | 'project'
  | 'molecules'
  | 'reactions'
  | 'geometry'
  | 'surface_classes'
  | 'mod_surf_regions'
  | 'release'
  | 'simulation'
  | 'results'
  | 'sweep'
  | 'mdl';

interface TabDef {
  id: TabId;
  label: string;
  icon: string;
  category: string;
}

const TABS: TabDef[] = [
  { id: 'project', label: 'Project', icon: '\u2302', category: 'General' },
  { id: 'molecules', label: 'Molecules', icon: '\u25CF', category: 'Model' },
  { id: 'reactions', label: 'Reactions', icon: '\u21C4', category: 'Model' },
  { id: 'geometry', label: 'Geometry', icon: '\u25A2', category: 'Model' },
  { id: 'surface_classes', label: 'Surface Classes', icon: '\u25A4', category: 'Model' },
  { id: 'mod_surf_regions', label: 'Assign Surfaces', icon: '\u2B12', category: 'Model' },
  { id: 'release', label: 'Release Sites', icon: '\u2727', category: 'Model' },
  { id: 'simulation', label: 'Simulation', icon: '\u25B6', category: 'Run' },
  { id: 'results', label: 'Results', icon: '\u2237', category: 'Run' },
  { id: 'sweep', label: 'Parameter Sweep', icon: '\u2261', category: 'Run' },
  { id: 'mdl', label: 'MDL Preview', icon: '\u2630', category: 'Export' },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('project');
  const dmActions = useDataModel();

  const {
    dataModel, simState, sweepParameters, totalSweepRuns,
  } = dmActions;

  const moleculeNames = dataModel.molecules.map(m => m.name).filter(Boolean);
  const reactionNames = dataModel.reactions
    .map(r => r.rxnName)
    .filter(Boolean);
  const objectNames = dataModel.geometryObjects.map(g => g.name);
  const surfaceClassNames = dataModel.surfaceClasses.map(s => s.name);

  const categories = [...new Set(TABS.map(t => t.category))];

  const renderContent = () => {
    switch (activeTab) {
      case 'project':
        return <ProjectManager dataModel={dataModel} actions={dmActions} />;
      case 'molecules':
        return <MoleculeEditor molecules={dataModel.molecules} actions={dmActions} />;
      case 'reactions':
        return (
          <ReactionEditor
            reactions={dataModel.reactions}
            moleculeNames={moleculeNames}
            actions={dmActions}
          />
        );
      case 'geometry':
        return <GeometryViewer objects={dataModel.geometryObjects} actions={dmActions} />;
      case 'surface_classes':
        return (
          <SurfaceClassEditor
            surfaceClasses={dataModel.surfaceClasses}
            moleculeNames={moleculeNames}
            actions={dmActions}
          />
        );
      case 'mod_surf_regions':
        return (
          <ModSurfRegions
            modSurfRegions={dataModel.modSurfRegions}
            objectNames={objectNames}
            surfaceClassNames={surfaceClassNames}
            actions={dmActions}
          />
        );
      case 'release':
        return (
          <ReleaseSiteEditor
            releaseSites={dataModel.releaseSites}
            releasePatterns={dataModel.releasePatterns}
            moleculeNames={moleculeNames}
            objectNames={objectNames}
            actions={dmActions}
          />
        );
      case 'simulation':
        return (
          <SimulationPanel
            initialization={dataModel.initialization}
            simControl={dataModel.simControl}
            simState={simState}
            actions={dmActions}
          />
        );
      case 'results':
        return (
          <ResultsViewer
            reactionOutputs={dataModel.reactionOutputs}
            moleculeNames={moleculeNames}
            reactionNames={reactionNames}
            objectNames={objectNames}
            actions={dmActions}
          />
        );
      case 'sweep':
        return (
          <ParameterSweep
            parameters={dataModel.parameters}
            sweepParameters={sweepParameters}
            totalRuns={totalSweepRuns}
            actions={dmActions}
          />
        );
      case 'mdl':
        return <MDLPreview dataModel={dataModel} />;
      default:
        return <div className="panel"><h2>Select a panel from the sidebar</h2></div>;
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="logo">
            <svg viewBox="0 0 32 32" width="28" height="28">
              <circle cx="16" cy="16" r="14" fill="none" stroke="#00d4ff" strokeWidth="2" />
              <circle cx="16" cy="16" r="3" fill="#00d4ff" />
              <circle cx="16" cy="6" r="2" fill="#ff6b9d" />
              <circle cx="24.5" cy="21" r="2" fill="#c084fc" />
              <circle cx="7.5" cy="21" r="2" fill="#34d399" />
              <line x1="16" y1="16" x2="16" y2="6" stroke="#ff6b9d" strokeWidth="1" opacity="0.6" />
              <line x1="16" y1="16" x2="24.5" y2="21" stroke="#c084fc" strokeWidth="1" opacity="0.6" />
              <line x1="16" y1="16" x2="7.5" y2="21" stroke="#34d399" strokeWidth="1" opacity="0.6" />
            </svg>
          </div>
          <h1>CellBlender</h1>
          <span className="header-subtitle">MCell Simulation Environment</span>
        </div>
        <div className="header-info">
          <span className="model-stat">{dataModel.molecules.length} mol</span>
          <span className="model-stat">{dataModel.reactions.length} rxn</span>
          <span className="model-stat">{dataModel.geometryObjects.length} obj</span>
          {simState.status === 'running' && (
            <span className="sim-indicator running">Running {simState.progress.toFixed(0)}%</span>
          )}
          {simState.status === 'completed' && (
            <span className="sim-indicator completed">Completed</span>
          )}
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar Navigation */}
        <nav className="sidebar">
          {categories.map(cat => (
            <div key={cat} className="nav-category">
              <div className="nav-category-label">{cat}</div>
              {TABS.filter(t => t.category === cat).map(tab => (
                <button
                  key={tab.id}
                  className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                  title={tab.label}
                >
                  <span className="nav-icon">{tab.icon}</span>
                  <span className="nav-label">{tab.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        {/* Main Content */}
        <main className="content">
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default App;
