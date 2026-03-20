/**
 * Core simulation state management hook.
 * Manages all application state: config, simulation run state, results, and logs.
 */
import { useState, useCallback, useRef } from 'react';
import {
  SimulationConfig,
  SimulationResults,
  SimmerState,
  SimPhaseKind,
  LogEntry,
} from '../types/simulation';
import { createDefaultConfig, PRESETS } from '../types/presets';
import { generateMockResults, createLogEntry } from '../utils/simulation-engine';
import {
  saveConfigToLocalStorage,
  loadConfigFromLocalStorage,
  serializeConfigToYaml,
  deserializeConfigFromYaml,
} from '../utils/config-persistence';

export interface UseSimulationReturn {
  // Configuration state
  config: SimulationConfig;
  updateConfig: <K extends keyof SimulationConfig>(key: K, value: SimulationConfig[K]) => void;
  updateNestedConfig: (path: string, value: unknown) => void;
  loadPreset: (presetIndex: number) => void;
  newConfig: () => void;
  saveConfig: () => void;
  openConfig: (filename: string) => void;
  exportConfigYaml: () => string;

  // Simulation state
  simmerState: SimmerState;
  currentPhase: SimPhaseKind | null;
  progress: number;
  startSimulation: () => void;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  stopSimulation: () => void;

  // Results
  results: SimulationResults | null;

  // Logs
  logs: LogEntry[];
  clearLogs: () => void;
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const keys = path.split('.');
  const result = { ...obj };
  let current: Record<string, unknown> = result;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    current[key] = { ...(current[key] as Record<string, unknown>) };
    current = current[key] as Record<string, unknown>;
  }

  current[keys[keys.length - 1]] = value;
  return result;
}

export function useSimulation(): UseSimulationReturn {
  const [config, setConfig] = useState<SimulationConfig>(() => loadConfigFromLocalStorage() || createDefaultConfig());
  const [simmerState, setSimmerState] = useState<SimmerState>(SimmerState.UNQUEUED);
  const [currentPhase, setCurrentPhase] = useState<SimPhaseKind | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [results, setResults] = useState<SimulationResults | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([
    createLogEntry('INFO', 'BETSEE React frontend initialized.', 'BETSEE'),
    createLogEntry('INFO', 'Ready. Open or create a simulation configuration to begin.', 'BETSEE'),
  ]);

  const timerRef = useRef<number | null>(null);
  const pausedRef = useRef(false);

  const addLog = useCallback((level: LogEntry['level'], message: string, source: string = 'BETSE') => {
    setLogs(prev => [...prev, createLogEntry(level, message, source)]);
  }, []);

  const updateConfig = useCallback(<K extends keyof SimulationConfig>(key: K, value: SimulationConfig[K]) => {
    setConfig(prev => {
      const next = { ...prev, [key]: value, isDirty: true };
      saveConfigToLocalStorage(next);
      return next;
    });
  }, []);

  const updateNestedConfig = useCallback((path: string, value: unknown) => {
    setConfig(prev => {
      const next = {
        ...setNestedValue(prev as unknown as Record<string, unknown>, path, value) as unknown as SimulationConfig,
        isDirty: true,
      };
      saveConfigToLocalStorage(next);
      return next;
    });
  }, []);

  const loadPreset = useCallback((presetIndex: number) => {
    const preset = PRESETS[presetIndex];
    if (!preset) return;
    const base = createDefaultConfig();
    const merged = { ...base, ...preset.config, isDirty: true };
    setConfig(merged as SimulationConfig);
    addLog('INFO', `Loaded preset: "${preset.name}"`, 'BETSEE');
  }, [addLog]);

  const newConfig = useCallback(() => {
    setConfig(createDefaultConfig());
    setResults(null);
    setSimmerState(SimmerState.UNQUEUED);
    setProgress(0);
    setCurrentPhase(null);
    addLog('INFO', 'Created new simulation configuration with default settings.', 'BETSEE');
  }, [addLog]);

  const saveConfig = useCallback(() => {
    setConfig(prev => {
      const next = { ...prev, isDirty: false };
      saveConfigToLocalStorage(next);
      return next;
    });
    addLog('INFO', `Simulation configuration saved${config.filename ? ` to ${config.filename}` : ''}.`, 'BETSEE');
  }, [addLog, config.filename]);

  const openConfig = useCallback((filename: string) => {
    // Try to parse the filename as YAML content if it contains newlines,
    // otherwise treat it as a filename for a new default config.
    let newConf: SimulationConfig;
    if (filename.includes('\n') || filename.includes(':')) {
      const partial = deserializeConfigFromYaml(filename);
      const base = createDefaultConfig();
      newConf = { ...base, ...partial } as SimulationConfig;
      newConf.filename = (partial.filename as string) || null;
    } else {
      newConf = createDefaultConfig();
      newConf.filename = filename;
    }
    saveConfigToLocalStorage(newConf);
    setConfig(newConf);
    setResults(null);
    addLog('INFO', `Opened simulation configuration: ${newConf.filename || '(untitled)'}`, 'BETSEE');
  }, [addLog]);

  const stopTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startSimulation = useCallback(() => {
    stopTimer();
    pausedRef.current = false;

    setSimmerState(SimmerState.QUEUED);
    setProgress(0);
    addLog('INFO', 'Simulation queued. Starting seed phase...', 'BETSE');

    // Simulate phased execution: SEED -> INIT -> SIM
    const phases: SimPhaseKind[] = [SimPhaseKind.SEED, SimPhaseKind.INIT, SimPhaseKind.SIM];
    let phaseIndex = 0;
    let localProgress = 0;

    const phaseDurations = [5, 30, 65]; // percentage of total

    setCurrentPhase(phases[0]);
    setSimmerState(SimmerState.MODELLING);
    addLog('INFO', 'Starting seed phase: generating cell cluster...', 'BETSE');

    timerRef.current = window.setInterval(() => {
      if (pausedRef.current) return;

      localProgress += 0.5;
      const phaseEnd = phaseDurations.slice(0, phaseIndex + 1).reduce((a, b) => a + b, 0);
      const phaseStart = phaseIndex > 0
        ? phaseDurations.slice(0, phaseIndex).reduce((a, b) => a + b, 0)
        : 0;

      if (localProgress >= phaseEnd && phaseIndex < phases.length - 1) {
        phaseIndex++;
        setCurrentPhase(phases[phaseIndex]);
        const phaseNames = { seed: 'seed', init: 'initialization', sim: 'simulation' };
        addLog('INFO', `Starting ${phaseNames[phases[phaseIndex]]} phase...`, 'BETSE');
      }

      setProgress(Math.min(localProgress, 100));

      if (localProgress >= 100) {
        stopTimer();
        setSimmerState(SimmerState.FINISHED);
        setCurrentPhase(null);
        addLog('INFO', 'Simulation completed successfully.', 'BETSE');

        // Generate mock results
        setResults(prev => {
          // Use current config from closure
          return generateMockResults(config);
        });
      }
    }, 100);
  }, [addLog, stopTimer, config]);

  const pauseSimulation = useCallback(() => {
    pausedRef.current = true;
    setSimmerState(SimmerState.PAUSED);
    addLog('INFO', 'Simulation paused.', 'BETSE');
  }, [addLog]);

  const resumeSimulation = useCallback(() => {
    pausedRef.current = false;
    setSimmerState(SimmerState.MODELLING);
    addLog('INFO', 'Simulation resumed.', 'BETSE');
  }, [addLog]);

  const stopSimulation = useCallback(() => {
    pausedRef.current = false;
    setSimmerState(SimmerState.STOPPING);
    addLog('WARNING', 'Stopping simulation...', 'BETSE');
    stopTimer();

    setTimeout(() => {
      setSimmerState(SimmerState.FINISHED);
      addLog('INFO', 'Simulation stopped by user.', 'BETSE');
    }, 500);
  }, [addLog, stopTimer]);

  const exportConfigYaml = useCallback((): string => {
    return serializeConfigToYaml(config);
  }, [config]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('INFO', 'Log cleared.', 'BETSEE');
  }, [addLog]);

  return {
    config,
    updateConfig,
    updateNestedConfig,
    loadPreset,
    newConfig,
    saveConfig,
    openConfig,
    exportConfigYaml,
    simmerState,
    currentPhase,
    progress,
    startSimulation,
    pauseSimulation,
    resumeSimulation,
    stopSimulation,
    results,
    logs,
    clearLogs,
  };
}
