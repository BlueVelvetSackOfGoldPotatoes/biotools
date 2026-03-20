import React from 'react';
import { SimmerState, SimPhaseKind, SimulationConfig } from '../../types/simulation';
import { formatTime } from '../../utils/format';

interface SimulationControlProps {
  simmerState: SimmerState;
  currentPhase: SimPhaseKind | null;
  progress: number;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  config: SimulationConfig;
}

const PHASE_LABELS: Record<SimPhaseKind, string> = {
  [SimPhaseKind.SEED]: 'Seed',
  [SimPhaseKind.INIT]: 'Initialization',
  [SimPhaseKind.SIM]: 'Simulation',
};

const STATE_LABELS: Record<SimmerState, string> = {
  [SimmerState.UNQUEUED]: 'Ready',
  [SimmerState.QUEUED]: 'Queued',
  [SimmerState.MODELLING]: 'Running',
  [SimmerState.EXPORTING]: 'Exporting',
  [SimmerState.PAUSED]: 'Paused',
  [SimmerState.STOPPING]: 'Stopping...',
  [SimmerState.FINISHED]: 'Finished',
};

const STATE_COLORS: Record<SimmerState, string> = {
  [SimmerState.UNQUEUED]: 'var(--color-text-muted)',
  [SimmerState.QUEUED]: 'var(--color-info)',
  [SimmerState.MODELLING]: 'var(--color-success)',
  [SimmerState.EXPORTING]: 'var(--color-info)',
  [SimmerState.PAUSED]: 'var(--color-warning)',
  [SimmerState.STOPPING]: 'var(--color-warning)',
  [SimmerState.FINISHED]: 'var(--color-success)',
};

export const SimulationControl: React.FC<SimulationControlProps> = ({
  simmerState,
  currentPhase,
  progress,
  onStart,
  onPause,
  onResume,
  onStop,
  config,
}) => {
  const isRunning = simmerState === SimmerState.MODELLING || simmerState === SimmerState.EXPORTING;
  const isPaused = simmerState === SimmerState.PAUSED;
  const isIdle = simmerState === SimmerState.UNQUEUED || simmerState === SimmerState.FINISHED || simmerState === SimmerState.QUEUED;

  return (
    <div>
      <h2 style={styles.pageTitle}>Simulation Control</h2>
      <p style={styles.pageDesc}>
        Control simulation execution: start, pause, resume, or stop the
        BETSE bioelectric tissue simulation phases.
      </p>

      {/* Status Display */}
      <div style={styles.statusCard}>
        <div style={styles.statusRow}>
          <div style={styles.statusIndicator}>
            <div style={{
              ...styles.statusDot,
              backgroundColor: STATE_COLORS[simmerState],
              boxShadow: isRunning ? `0 0 8px ${STATE_COLORS[simmerState]}` : 'none',
              animation: isRunning ? 'pulse 1.5s ease-in-out infinite' : 'none',
            }} />
            <span style={styles.statusLabel}>{STATE_LABELS[simmerState]}</span>
          </div>
          {currentPhase && (
            <span style={styles.phaseLabel}>
              Phase: <strong>{PHASE_LABELS[currentPhase]}</strong>
            </span>
          )}
        </div>

        {/* Progress Bar */}
        <div style={styles.progressContainer}>
          <div style={styles.progressTrack}>
            <div
              style={{
                ...styles.progressBar,
                width: `${progress}%`,
                backgroundColor: isPaused ? 'var(--color-warning)' : 'var(--color-accent)',
              }}
            />
          </div>
          <span style={styles.progressText}>{progress.toFixed(1)}%</span>
        </div>

        {/* Phase Pipeline */}
        <div style={styles.pipeline}>
          {[SimPhaseKind.SEED, SimPhaseKind.INIT, SimPhaseKind.SIM].map((phase) => {
            const isActive = currentPhase === phase;
            const isDone = currentPhase !== null && (
              (phase === SimPhaseKind.SEED && (currentPhase === SimPhaseKind.INIT || currentPhase === SimPhaseKind.SIM)) ||
              (phase === SimPhaseKind.INIT && currentPhase === SimPhaseKind.SIM)
            );
            const isFinished = simmerState === SimmerState.FINISHED;
            return (
              <div key={phase} style={styles.pipelineStep}>
                <div style={{
                  ...styles.pipelineIcon,
                  backgroundColor: (isDone || isFinished) ? 'var(--color-success)' :
                    isActive ? 'var(--color-accent)' : 'var(--color-bg-elevated)',
                  color: (isDone || isActive || isFinished) ? '#fff' : 'var(--color-text-muted)',
                }}>
                  {(isDone || isFinished) ? '\u2713' : PHASE_LABELS[phase][0]}
                </div>
                <span style={{
                  ...styles.pipelineLabel,
                  color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-muted)',
                  fontWeight: isActive ? 600 : 400,
                }}>
                  {PHASE_LABELS[phase]}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Control Buttons */}
      <div style={styles.controls}>
        <button
          style={{
            ...styles.btn,
            ...styles.btnPrimary,
            ...((!isIdle) ? styles.btnDisabled : {}),
          }}
          onClick={onStart}
          disabled={!isIdle}
        >
          {'\u25B6'} Start Simulation
        </button>

        {isRunning && (
          <button style={{ ...styles.btn, ...styles.btnWarning }} onClick={onPause}>
            {'\u23F8'} Pause
          </button>
        )}

        {isPaused && (
          <button style={{ ...styles.btn, ...styles.btnPrimary }} onClick={onResume}>
            {'\u25B6'} Resume
          </button>
        )}

        {(isRunning || isPaused) && (
          <button style={{ ...styles.btn, ...styles.btnDanger }} onClick={onStop}>
            {'\u25A0'} Stop
          </button>
        )}
      </div>

      {/* Configuration Summary */}
      <div style={styles.summaryCard}>
        <h3 style={styles.summaryTitle}>Configuration Summary</h3>
        <div style={styles.summaryGrid}>
          <SummaryItem label="Init Duration" value={formatTime(config.time.init_time_total)} />
          <SummaryItem label="Sim Duration" value={formatTime(config.time.sim_time_total)} />
          <SummaryItem label="Time Step" value={formatTime(config.time.sim_time_step)} />
          <SummaryItem label="Grid Size" value={`${config.space.grid_size} x ${config.space.grid_size}`} />
          <SummaryItem label="Total Cells" value={`~${config.space.grid_size ** 2}`} />
          <SummaryItem label="Lattice" value={config.space.cell_lattice_type.toUpperCase()} />
          <SummaryItem label="ECM" value={config.space.is_ecm ? 'Enabled' : 'Disabled'} />
          <SummaryItem label="Ion Profile" value={config.ions.ion_profile} />
          <SummaryItem label="Active Channels" value={`${config.channels.filter(c => c.enabled).length}`} />
          <SummaryItem label="Deformation" value={config.physics.is_deformation ? 'On' : 'Off'} />
          <SummaryItem label="Temperature" value={`${config.physics.T} K`} />
          <SummaryItem
            label="Est. Time Steps"
            value={`${Math.round(config.time.sim_time_total / config.time.sim_time_step)}`}
          />
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};

const SummaryItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={styles.summaryItem}>
    <span style={styles.summaryLabel}>{label}</span>
    <span style={styles.summaryValue}>{value}</span>
  </div>
);

const styles: Record<string, React.CSSProperties> = {
  pageTitle: {
    fontSize: 'var(--font-size-xl)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    marginBottom: 'var(--spacing-sm)',
  },
  pageDesc: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-secondary)',
    marginBottom: 'var(--spacing-xl)',
    maxWidth: '680px',
    lineHeight: 1.6,
  },
  statusCard: {
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    padding: 'var(--spacing-xl)',
    marginBottom: 'var(--spacing-xl)',
  },
  statusRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 'var(--spacing-lg)',
  },
  statusIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
  },
  statusDot: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
  },
  statusLabel: {
    fontSize: 'var(--font-size-lg)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
  },
  phaseLabel: {
    fontSize: 'var(--font-size-base)',
    color: 'var(--color-text-secondary)',
  },
  progressContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-md)',
    marginBottom: 'var(--spacing-lg)',
  },
  progressTrack: {
    flex: 1,
    height: '8px',
    backgroundColor: 'var(--color-bg-elevated)',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.2s ease',
  },
  progressText: {
    fontSize: 'var(--font-size-sm)',
    fontFamily: 'var(--font-family-mono)',
    color: 'var(--color-text-secondary)',
    minWidth: '50px',
    textAlign: 'right' as const,
  },
  pipeline: {
    display: 'flex',
    justifyContent: 'center',
    gap: 'var(--spacing-xxl)',
  },
  pipelineStep: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: 'var(--spacing-xs)',
  },
  pipelineIcon: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 'var(--font-size-sm)',
    fontWeight: 700,
    transition: 'all 0.2s ease',
  },
  pipelineLabel: {
    fontSize: 'var(--font-size-xs)',
    transition: 'all 0.2s ease',
  },
  controls: {
    display: 'flex',
    gap: 'var(--spacing-md)',
    marginBottom: 'var(--spacing-xl)',
  },
  btn: {
    padding: 'var(--spacing-sm) var(--spacing-xl)',
    borderRadius: 'var(--radius-md)',
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    cursor: 'pointer',
    border: 'none',
    transition: 'all 0.15s ease',
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--spacing-sm)',
  },
  btnPrimary: {
    backgroundColor: 'var(--color-accent)',
    color: '#fff',
  },
  btnWarning: {
    backgroundColor: 'var(--color-warning)',
    color: '#fff',
  },
  btnDanger: {
    backgroundColor: 'var(--color-error)',
    color: '#fff',
  },
  btnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  summaryCard: {
    backgroundColor: 'var(--color-bg-secondary)',
    border: '1px solid var(--color-border)',
    borderRadius: 'var(--radius-lg)',
    padding: 'var(--spacing-lg)',
  },
  summaryTitle: {
    fontSize: 'var(--font-size-md)',
    fontWeight: 600,
    color: 'var(--color-text-primary)',
    marginBottom: 'var(--spacing-md)',
  },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 'var(--spacing-md)',
  },
  summaryItem: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2px',
  },
  summaryLabel: {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--color-text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  summaryValue: {
    fontSize: 'var(--font-size-base)',
    fontWeight: 500,
    color: 'var(--color-text-primary)',
    fontFamily: 'var(--font-family-mono)',
  },
};
