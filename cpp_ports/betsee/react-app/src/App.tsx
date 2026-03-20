import React, { useState } from 'react';
import { useSimulation } from './hooks/useSimulation';
import { Sidebar } from './components/sidebar/Sidebar';
import { GeneralSettings } from './components/config/general/GeneralSettings';
import { SpaceSettings } from './components/config/space/SpaceSettings';
import { IonSettings } from './components/config/ions/IonSettings';
import { ChannelSettings } from './components/config/channels/ChannelSettings';
import { TissueSettings } from './components/config/tissue/TissueSettings';
import { NetworkSettings } from './components/config/network/NetworkSettings';
import { PhysicsSettings } from './components/config/physics/PhysicsSettings';
import { ExportSettings } from './components/config/exports/ExportSettings';
import { InterventionSettings } from './components/config/interventions/InterventionSettings';
import { SimulationControl } from './components/simulation/SimulationControl';
import { ResultsViewer } from './components/results/ResultsViewer';
import { LogViewer } from './components/log/LogViewer';
import { FileManager } from './components/file/FileManager';

export type ViewId =
  | 'general'
  | 'space'
  | 'ions'
  | 'channels'
  | 'tissue'
  | 'interventions'
  | 'network'
  | 'physics'
  | 'exports'
  | 'simulation'
  | 'results'
  | 'log';

const App: React.FC = () => {
  const simulation = useSimulation();
  const [activeView, setActiveView] = useState<ViewId>('general');
  const [showFileManager, setShowFileManager] = useState(false);

  const renderView = () => {
    switch (activeView) {
      case 'general':
        return (
          <GeneralSettings
            timeConfig={simulation.config.time}
            pathConfig={simulation.config.paths}
            onChange={(path, value) => simulation.updateNestedConfig(path, value)}
          />
        );
      case 'space':
        return (
          <SpaceSettings
            config={simulation.config.space}
            onChange={(path, value) => simulation.updateNestedConfig(`space.${path}`, value)}
          />
        );
      case 'ions':
        return (
          <IonSettings
            config={simulation.config.ions}
            onChange={(path, value) => simulation.updateNestedConfig(`ions.${path}`, value)}
          />
        );
      case 'channels':
        return (
          <ChannelSettings
            channels={simulation.config.channels}
            onChange={(channels) => simulation.updateConfig('channels', channels)}
          />
        );
      case 'tissue':
        return (
          <TissueSettings
            config={simulation.config.tissue}
            onChange={(path, value) => simulation.updateNestedConfig(`tissue.${path}`, value)}
          />
        );
      case 'interventions':
        return (
          <InterventionSettings
            interventions={simulation.config.interventions}
            onChange={(interventions) => simulation.updateConfig('interventions', interventions)}
          />
        );
      case 'network':
        return (
          <NetworkSettings
            config={simulation.config.grn}
            onChange={(path, value) => simulation.updateNestedConfig(`grn.${path}`, value)}
          />
        );
      case 'physics':
        return (
          <PhysicsSettings
            config={simulation.config.physics}
            onChange={(path, value) => simulation.updateNestedConfig(`physics.${path}`, value)}
          />
        );
      case 'exports':
        return (
          <ExportSettings
            config={simulation.config.exports}
            onChange={(path, value) => simulation.updateNestedConfig(`exports.${path}`, value)}
          />
        );
      case 'simulation':
        return (
          <SimulationControl
            simmerState={simulation.simmerState}
            currentPhase={simulation.currentPhase}
            progress={simulation.progress}
            onStart={simulation.startSimulation}
            onPause={simulation.pauseSimulation}
            onResume={simulation.resumeSimulation}
            onStop={simulation.stopSimulation}
            config={simulation.config}
          />
        );
      case 'results':
        return (
          <ResultsViewer
            results={simulation.results}
            config={simulation.config}
          />
        );
      case 'log':
        return (
          <LogViewer
            logs={simulation.logs}
            onClear={simulation.clearLogs}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div style={styles.container}>
      {/* Header / Title Bar */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.headerBtn} onClick={() => setShowFileManager(true)}>
            File
          </button>
          <span style={styles.headerTitle}>BETSEE</span>
          <span style={styles.headerSubtitle}>
            Bioelectric Tissue Simulation Engine Environment
          </span>
        </div>
        <div style={styles.headerRight}>
          {simulation.config.filename && (
            <span style={styles.headerFilename}>
              {simulation.config.filename}
              {simulation.config.isDirty ? ' *' : ''}
            </span>
          )}
        </div>
      </header>

      <div style={styles.body}>
        {/* Sidebar Navigation */}
        <Sidebar activeView={activeView} onNavigate={setActiveView} />

        {/* Main Content Area */}
        <main style={styles.main}>
          {renderView()}
        </main>
      </div>

      {/* File Manager Modal */}
      {showFileManager && (
        <FileManager
          config={simulation.config}
          onNew={simulation.newConfig}
          onOpen={simulation.openConfig}
          onSave={simulation.saveConfig}
          onLoadPreset={simulation.loadPreset}
          onClose={() => setShowFileManager(false)}
        />
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 'var(--header-height)',
    backgroundColor: 'var(--color-bg-tertiary)',
    borderBottom: '1px solid var(--color-border)',
    padding: '0 var(--spacing-md)',
    flexShrink: 0,
    zIndex: 100,
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
  },
  headerBtn: {
    padding: 'var(--spacing-xs) var(--spacing-md)',
    backgroundColor: 'var(--color-bg-elevated)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-text-primary)',
    fontSize: 'var(--font-size-sm)',
    cursor: 'pointer',
  },
  headerTitle: {
    fontWeight: 700,
    fontSize: 'var(--font-size-lg)',
    color: 'var(--color-text-primary)',
    letterSpacing: '0.5px',
  },
  headerSubtitle: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-muted)',
  },
  headerRight: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
  },
  headerFilename: {
    fontSize: 'var(--font-size-sm)',
    color: 'var(--color-text-secondary)',
    fontFamily: 'var(--font-family-mono)',
  },
  body: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  main: {
    flex: 1,
    overflow: 'auto',
    padding: 'var(--spacing-xl)',
    backgroundColor: 'var(--color-bg-primary)',
  },
};

export default App;
