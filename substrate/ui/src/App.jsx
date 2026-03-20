import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePolling } from "./hooks/usePolling";
import { fetchJson } from "./lib/fetchJson";
import {
  CONTROL_GAME_BENCHMARKS,
  FAMILY_COLORS,
  TASK_BENCHMARK_OPTIONS,
  buildGroupedSeries,
  chooseAutoPreviewColumns,
  confusionMatrix,
  elapsedSince,
  formatCompactNumber,
  formatPct,
  formatSignedPct,
  formatTime,
  listDistinctNumbers,
  listDistinctStrings,
  meanValue,
  medianValue,
  pivotEpoch,
  runStatus,
  runStatusClass,
  runStatusLabel,
  semValue,
  seriesColor,
  stdDev,
  taskStatusClass,
  taskStatusLabel,
  trainingDurationSeconds
} from "./lib/dashboardShared";
import {
  bestRunsByFamily,
  buildEvaluationLandscape,
  buildTrainingLandscape,
  deriveBenchmarkCoverageFromRuns,
  deriveFamilyCoverageFromRuns,
  firstFiniteValue,
  isFiniteNumber,
  makeLatencyHistogram
} from "./lib/appLandscapes";
import { buildViewAllDataset } from "./lib/appViewDataset";
import {
  VIEW_ALL_SPECS,
  decodeBenchmarkSnapshot,
  formatBenchmarkAction,
  outcomeLabel
} from "./lib/appViewMeta";
import { requestJson } from "./lib/requestJson";
import { useAppQueries } from "./hooks/useAppQueries";
import { useAppDerivedMetrics } from "./hooks/useAppDerivedMetrics";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import DashboardOverviewSections from "./components/DashboardOverviewSections";
import DashboardAssetSections from "./components/DashboardAssetSections";

const POLL_MS = 5000;
const MODEL_GROUP_SERIES_LIMIT = 8;
const CellEngineReplayPanel = lazy(() => import("./components/CellEngineReplayPanel"));
const ViewAllChartLazy = lazy(() => import("./components/ViewAllChart"));
const RunDetailPanelLazy = lazy(() => import("./components/RunDetailPanel"));
const TasksPanelLazy = lazy(() => import("./components/TasksPanel"));
const ComparisonPanelLazy = lazy(() => import("./components/ComparisonPanel"));
const TrainingPanelLazy = lazy(() => import("./components/TrainingPanel"));
const EvaluationPanelLazy = lazy(() => import("./components/EvaluationPanel"));
const GameReplayPanelLazy = lazy(() => import("./components/GameReplayPanel"));
const InterpretabilitySuiteLazy = lazy(() =>
  import("./components/ModelArtifactPanels").then((module) => ({ default: module.InterpretabilitySuite }))
);
const AutoCsvPreviewLazy = lazy(() =>
  import("./components/ModelArtifactPanels").then((module) => ({ default: module.AutoCsvPreview }))
);
const CsvModelChartLazy = lazy(() =>
  import("./components/ModelArtifactPanels").then((module) => ({ default: module.CsvModelChart }))
);


export default function App() {
  const [tab, setTab] = useState(() => {
    if (typeof window === "undefined") return "live";
    const params = new URLSearchParams(window.location.search);
    const viewAllQuery = params.get("view_all") || "";
    if (VIEW_ALL_SPECS[viewAllQuery]) return "viewall";
    const value = params.get("tab") || "live";
    const valid = ["live", "tasks", "architectures", "comparison", "viewall", "training", "evaluation", "game", "model", "cellengine", "reports"];
    return valid.includes(value) ? value : "live";
  });
  const [runId, setRunId] = useState(() => {
    if (typeof window === "undefined") return "";
    return new URLSearchParams(window.location.search).get("run_id") || "";
  });
  const [modelCsvPath, setModelCsvPath] = useState("");
  const [benchmarkFilter, setBenchmarkFilter] = useState(() => {
    if (typeof window === "undefined") return "all";
    return new URLSearchParams(window.location.search).get("benchmark") || "all";
  });
  const [viewAllId, setViewAllId] = useState(() => {
    if (typeof window === "undefined") return "";
    const value = new URLSearchParams(window.location.search).get("view_all") || "";
    return VIEW_ALL_SPECS[value] ? value : "";
  });
  const [viewAllReturnTab, setViewAllReturnTab] = useState(() => {
    if (typeof window === "undefined") return "";
    const value = new URLSearchParams(window.location.search).get("view_back") || "";
    const valid = ["live", "tasks", "architectures", "comparison", "training", "evaluation", "game", "model", "cellengine", "reports"];
    return valid.includes(value) ? value : "";
  });
  const [viewAllNonce, setViewAllNonce] = useState(0);
  const [viewAllPayload, setViewAllPayload] = useState({
    loading: false,
    error: "",
    fetchedAt: 0,
    byFamily: {}
  });

  const [expandedRunId, setExpandedRunId] = useState("");

  const [runStateFilter, setRunStateFilter] = useState("all");
  const [familyFilter, setFamilyFilter] = useState("all");
  const [architectureFamily, setArchitectureFamily] = useState("");
  const [bioStudyId, setBioStudyId] = useState("");
  const [bioModel, setBioModel] = useState("");
  const [taskBusyKey, setTaskBusyKey] = useState("");
  const [taskActionError, setTaskActionError] = useState("");
  const [taskActionMessage, setTaskActionMessage] = useState("");

  const [evalSplit, setEvalSplit] = useState("test");
  const [cmEpochChoice, setCmEpochChoice] = useState("latest");
  const [calSplit, setCalSplit] = useState("test");
  const [inferSplit, setInferSplit] = useState("test");

  const [modelPreviewNonce, setModelPreviewNonce] = useState(0);
  const {
    needComparison,
    runsQuery,
    tasksQuery,
    familiesQuery,
    benchmarksQuery,
    bioStudiesQuery,
    runs,
    tasks,
    benchmarks,
    dashboardRuns,
    selectedBenchmark,
    runFamily,
    runBenchmarkId,
    runTaskType,
    bioStudies,
    selectedBioStudy,
    bioAblationQuery,
    needTraining,
    needEvaluation,
    needLandscape,
    needGame,
    needModel,
    needReports,
    needInterpretability,
    needBio,
    needContinuous,
    epochQuery,
    batchQuery,
    gameTraceQuery,
    bioEpochQuery,
    bioLayerQuery,
    continuousEffQuery,
    continuousPhaseQuery,
    hybridExpertQuery,
    cmQuery,
    calQuery,
    inferQuery,
    classMetricsQuery,
    sysQuery,
    modelFilesQuery,
    modelCsvQuery,
    reportsQuery,
    modelFileList,
    modelPreviewMap,
    modelPreviewLoading,
    selectedRun,
    activeRuns
  } = useAppQueries({
    pollMs: POLL_MS,
    tab,
    benchmarkFilter,
    bioStudyId,
    bioModel,
    runId,
    modelCsvPath,
    modelPreviewNonce,
    setBenchmarkFilter,
    setRunId,
    setBioStudyId,
    setBioModel,
    setModelCsvPath
  });

  const {
    activeTaskCount,
    bestFamilyRuns,
    bestFamilyRunsSignature,
    activeViewSpec,
    openViewAllTab,
    families,
    selectedFamilyCoverage,
    modelFamilies,
    selectedModelFamilyCoverage,
    familyOptions,
    runOptionsByFamily,
    modelRunsForFamily,
    visibleRuns,
    jumpToFamilyModelRun,
    epochRows,
    batchRows,
    gameTraceRows,
    epochPivot,
    batchSeries,
    trainingLandscape,
    epochSplits,
    bioEpochRows,
    bioLayerRows,
    continuousEffRows,
    continuousPhaseRows,
    hybridExpertRows,
    lossLines,
    accuracyLines,
    throughputLines,
    bioEpochSeries,
    latestBioEpochRow,
    hasEpochSignViolation,
    hasLayerSignViolation,
    bioLayerMaskSeries,
    bioLayerBioelectricSeries,
    bioLayerSignSeries,
    bioTissueGraph,
    latestBioLayerRows,
    bioEvalLayerRows,
    continuousEffSeries,
    latestContinuousEffRow,
    continuousPhaseLossSeries,
    continuousPhaseAccSeries,
    efficiencyFrontierRows,
    latestHybridExpertRows,
    latestEpoch,
    latestAccuracy,
    latestMetricLabel,
    cmRows,
    calRows,
    inferRows,
    classMetricsRows,
    sysRows,
    cmSplits,
    calSplits,
    inferSplits,
    cmEpochs,
    cmFilteredRows,
    cm,
    calFilteredRows,
    calibrationChartRows,
    inferFilteredRows,
    classProfileRows,
    classRadarRows,
    confidenceOutcomeRows,
    entropyConfidenceScatterRows,
    evaluationLandscape,
    histRows,
    errorRows,
    systemChartRows
  } = useAppDerivedMetrics({
    tab,
    runId,
    viewAllId,
    viewAllReturnTab,
    benchmarkFilter,
    architectureFamily,
    familyFilter,
    runStateFilter,
    evalSplit,
    cmEpochChoice,
    calSplit,
    inferSplit,
    dashboardRuns,
    tasks,
    familiesQuery,
    selectedRun,
    runsQuery,
    tasksQuery,
    benchmarksQuery,
    bioStudiesQuery,
    bioAblationQuery,
    epochQuery,
    batchQuery,
    gameTraceQuery,
    bioEpochQuery,
    bioLayerQuery,
    continuousEffQuery,
    continuousPhaseQuery,
    hybridExpertQuery,
    cmQuery,
    calQuery,
    inferQuery,
    classMetricsQuery,
    sysQuery,
    modelFilesQuery,
    modelCsvQuery,
    reportsQuery,
    setViewAllId,
    setViewAllReturnTab,
    setTab,
    setArchitectureFamily,
    setFamilyFilter,
    setEvalSplit,
    setCalSplit,
    setInferSplit,
    setCmEpochChoice,
    setRunId
  });

  const {
    lastRefresh,
    refreshTaskAndRunQueries,
    createTrainTask,
    createEvaluateTask,
    killTaskAction,
    startTaskAction,
    deleteTaskAction,
    errors,
    refreshCurrent,
    hasViewAllData,
    isCellEngineWorkspace,
    dashboardTabItems,
    switchToCellEngine,
    switchToDashboard
  } = useAppWorkspace({
    tab,
    viewAllId,
    activeViewSpec,
    bestFamilyRuns,
    bestFamilyRunsSignature,
    viewAllNonce,
    setViewAllNonce,
    viewAllPayload,
    setViewAllPayload,
    setViewAllId,
    setTab,
    tasksQuery,
    runsQuery,
    familiesQuery,
    benchmarksQuery,
    bioStudiesQuery,
    bioAblationQuery,
    epochQuery,
    batchQuery,
    gameTraceQuery,
    bioEpochQuery,
    bioLayerQuery,
    continuousEffQuery,
    continuousPhaseQuery,
    hybridExpertQuery,
    cmQuery,
    calQuery,
    inferQuery,
    classMetricsQuery,
    sysQuery,
    modelFilesQuery,
    modelCsvQuery,
    reportsQuery,
    modelPreviewNonce,
    setModelPreviewNonce,
    taskActionError,
    taskActionMessage,
    setTaskActionError,
    setTaskActionMessage,
    setTaskBusyKey,
    requestJson,
    fetchJson
  });

  if (isCellEngineWorkspace) {
    return (
      <div className="app-shell app-shell-cellengine">
        <header className="topbar topbar-cellengine">
          <div>
            <h1>CellEngine Lab</h1>
            <p>
              Dedicated organism-first workspace for the cell controller. The run dashboard is intentionally separated.
            </p>
          </div>
          <div className="live-badge">
            <span className="dot" />
            organism replay, tissue state, connectivity, genome variability, RL comparison
          </div>
        </header>

        <nav className="workspace-nav">
          <button className="workspace-nav-btn" type="button" onClick={switchToDashboard}>
            Research Dashboard
          </button>
          <button className="workspace-nav-btn active" type="button" onClick={switchToCellEngine}>
            CellEngine Lab
          </button>
        </nav>

        <section className="panel workspace-panel">
          <div className="workspace-panel-head">
            <div>
              <h3>Organism Workspace</h3>
              <p>
                This page is isolated from run-level benchmark controls. If you want the general benchmark dashboard, use
                `Research Dashboard` above.
              </p>
            </div>
            <div className="workspace-panel-meta">
              <span className="landscape-pill">entry `?tab=cellengine`</span>
              <span className="landscape-pill">dedicated organism navigation</span>
            </div>
          </div>
        </section>

        <Suspense fallback={<section className="panel"><div className="empty">Loading CellEngine replay UI...</div></section>}>
          <CellEngineReplayPanel />
        </Suspense>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Benchmark Training Control Room</h1>
          <p>
            Live run monitoring, training curves, deployment diagnostics, and architecture-specific visuals.
            {` Scope: ${selectedBenchmark?.benchmark_name || "all benchmarks"}.`}
          </p>
        </div>
        <div className="live-badge">
          <span className="dot" />
          {activeRuns.length} active runs · {activeTaskCount} active tasks
        </div>
      </header>

      <nav className="workspace-nav">
        <button className="workspace-nav-btn active" type="button" onClick={switchToDashboard}>
          Research Dashboard
        </button>
        <button className="workspace-nav-btn" type="button" onClick={switchToCellEngine}>
          CellEngine Lab
        </button>
      </nav>

      <section className="toolbar">
        <label className="run-select">
          Benchmark
          <select value={benchmarkFilter} onChange={(event) => setBenchmarkFilter(event.target.value)}>
            <option value="all">all benchmarks</option>
            {benchmarks.map((row) => (
              <option key={row.benchmark_id} value={row.benchmark_id}>
                {row.benchmark_name} ({row.runs})
              </option>
            ))}
          </select>
        </label>

        <label className="run-select">
          Run
          <select
            value={runId}
            onChange={(event) => setRunId(event.target.value)}
            disabled={!dashboardRuns.length}
          >
            {!dashboardRuns.length && <option value="">no runs</option>}
            {runOptionsByFamily.map(([family, familyRuns]) => (
              <optgroup key={family} label={family}>
                {familyRuns.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id} [{run.model_variant}]
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>

        <button className="refresh-btn" onClick={refreshCurrent} type="button">
          Refresh
        </button>

        <div className="run-meta">
          <span className={`pill ${runStatusClass(runStatus(selectedRun))}`}>
            {runStatusLabel(runStatus(selectedRun))}
          </span>
          <span>{selectedRun?.benchmark_name || selectedBenchmark?.benchmark_name || "unknown benchmark"}</span>
          <span>{selectedRun?.task_type || "unknown task"}</span>
          <span>{selectedRun?.model_family || "unknown"}</span>
          <span>{selectedRun?.model_variant || "unknown"}</span>
          <span>latest epoch: {latestEpoch ?? "n/a"}</span>
          <span>{latestMetricLabel}: {formatPct(latestAccuracy)}</span>
          {runStatus(selectedRun) === "starting" && (
            <span>elapsed: {elapsedSince(selectedRun?.train_start_utc)}</span>
          )}
          <span>last refresh: {formatTime(lastRefresh)}</span>
        </div>
      </section>

      <nav className="tabs">
        {dashboardTabItems.map(([id, label]) => (
          <button
            key={id}
            className={tab === id ? "tab active" : "tab"}
            onClick={() => {
              if (id !== "viewall" && viewAllId) setViewAllId("");
              setTab(id);
            }}
            type="button"
          >
            {label}
          </button>
        ))}
      </nav>
      <DashboardOverviewSections
        tab={tab}
        activeViewSpec={activeViewSpec}
        bestFamilyRuns={bestFamilyRuns}
        viewAllPayload={viewAllPayload}
        hasViewAllData={hasViewAllData}
        setViewAllNonce={setViewAllNonce}
        setViewAllId={setViewAllId}
        setTab={setTab}
        viewAllReturnTab={viewAllReturnTab}
        setRunId={setRunId}
        runStateFilter={runStateFilter}
        setRunStateFilter={setRunStateFilter}
        familyFilter={familyFilter}
        setFamilyFilter={setFamilyFilter}
        familyOptions={familyOptions}
        visibleRuns={visibleRuns}
        expandedRunId={expandedRunId}
        setExpandedRunId={setExpandedRunId}
        runId={runId}
        setArchitectureFamily={setArchitectureFamily}
        runsQuery={runsQuery}
        dashboardRuns={dashboardRuns}
        tasks={tasks}
        tasksQuery={tasksQuery}
        taskBusyKey={taskBusyKey}
        taskActionError={taskActionError}
        taskActionMessage={taskActionMessage}
        refreshTaskAndRunQueries={refreshTaskAndRunQueries}
        createTrainTask={createTrainTask}
        createEvaluateTask={createEvaluateTask}
        killTaskAction={killTaskAction}
        startTaskAction={startTaskAction}
        deleteTaskAction={deleteTaskAction}
        ViewAllChartLazy={ViewAllChartLazy}
        RunDetailPanelLazy={RunDetailPanelLazy}
        TasksPanelLazy={TasksPanelLazy}
      />

      {tab === "architectures" && (
        <section className="panel stack">
          <div className="row controls">
            <label>
              Architecture
              <select
                value={architectureFamily}
                onChange={(event) => setArchitectureFamily(event.target.value)}
                disabled={!families.length}
              >
                {families.map((row) => (
                  <option key={row.family} value={row.family}>
                    {row.family}
                  </option>
                ))}
              </select>
            </label>
            <span className="meta-note">{families.length} model families discovered</span>
          </div>

          {familiesQuery.isLoading && !families.length ? (
            <div className="empty">Loading architecture coverage...</div>
          ) : (
            <>
              <div className="family-grid">
                {families.map((row) => (
                  <button
                    key={row.family}
                    type="button"
                    className={row.family === architectureFamily ? "family-card selected" : "family-card"}
                    onClick={() => setArchitectureFamily(row.family)}
                  >
                    <div className="card-title">
                      <strong>{row.family}</strong>
                      <span className="pill pill-done">{row.runs} runs</span>
                    </div>
                    <div className="card-grid">
                      <span>active</span>
                      <span>{row.active_runs || 0}</span>
                      <span>variants</span>
                      <span>{(row.variants || []).length}</span>
                      <span>learning</span>
                      <span>
                        {row.runs_with_learning || 0}/{row.runs}
                      </span>
                      <span>evaluation</span>
                      <span>
                        {row.runs_with_evaluation || 0}/{row.runs}
                      </span>
                      <span>model-specific</span>
                      <span>
                        {row.runs_with_model_specific || 0}/{row.runs}
                      </span>
                    </div>
                  </button>
                ))}
              </div>

              {selectedFamilyCoverage && (
                <div className="card">
                  <h3>{selectedFamilyCoverage.family} Details</h3>
                  <div className="row controls">
                    <button
                      className="mini-btn"
                      type="button"
                      onClick={() => {
                        if (selectedFamilyCoverage.latest_run_id) {
                          setRunId(selectedFamilyCoverage.latest_run_id);
                          setTab("model");
                        }
                      }}
                      disabled={!selectedFamilyCoverage.latest_run_id}
                    >
                      Open Latest Run
                    </button>
                    <button
                      className="mini-btn"
                      type="button"
                      onClick={() => {
                        setFamilyFilter(selectedFamilyCoverage.family);
                        setTab("live");
                      }}
                    >
                      Filter In Live Runs
                    </button>
                  </div>

                  <div className="card-grid">
                    <span>latest run</span>
                    <span>{selectedFamilyCoverage.latest_run_id || "n/a"}</span>
                    <span>last update</span>
                    <span>{selectedFamilyCoverage.latest_updated_utc || "n/a"}</span>
                    <span>variants</span>
                    <span>{(selectedFamilyCoverage.variants || []).join(", ") || "n/a"}</span>
                    <span>model csv files</span>
                    <span>
                      {(selectedFamilyCoverage.model_specific_files || []).length
                        ? selectedFamilyCoverage.model_specific_files.join(", ")
                        : "n/a"}
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}

      {tab === "comparison" && (
        <Suspense fallback={<div className="empty">Loading comparison workspace...</div>}>
          <ComparisonPanelLazy
            runs={dashboardRuns}
            bioStudies={bioStudies}
            bioStudyId={bioStudyId}
            bioModel={bioModel}
            onBioStudyChange={setBioStudyId}
            onBioModelChange={setBioModel}
            bioAblation={bioAblationQuery.data}
            bioAblationLoading={bioAblationQuery.isLoading || bioStudiesQuery.isLoading}
            bioAblationError={bioAblationQuery.error || bioStudiesQuery.error}
            onOpenRun={(nextRunId) => {
              setRunId(nextRunId);
              setTab("live");
            }}
            onOpenFamily={(family) => {
              setFamilyFilter(family);
              setArchitectureFamily(family);
              setTab("architectures");
            }}
          />
        </Suspense>
      )}

      {tab === "training" && (
        <Suspense fallback={<section className="panel"><div className="empty">Loading training dashboard...</div></section>}>
          <TrainingPanelLazy
            epochQuery={epochQuery}
            batchQuery={batchQuery}
            bioEpochQuery={bioEpochQuery}
            bioLayerQuery={bioLayerQuery}
            continuousEffQuery={continuousEffQuery}
            continuousPhaseQuery={continuousPhaseQuery}
            hybridExpertQuery={hybridExpertQuery}
            trainingLandscape={trainingLandscape}
            openViewAllTab={openViewAllTab}
            batchSeries={batchSeries}
            selectedRun={selectedRun}
            epochPivot={epochPivot}
            lossLines={lossLines}
            accuracyLines={accuracyLines}
            throughputLines={throughputLines}
            latestBioEpochRow={latestBioEpochRow}
            bioEpochSeries={bioEpochSeries}
            bioTissueGraph={bioTissueGraph}
            bioLayerBioelectricSeries={bioLayerBioelectricSeries}
            hasEpochSignViolation={hasEpochSignViolation}
            hasLayerSignViolation={hasLayerSignViolation}
            runFamily={runFamily}
            latestContinuousEffRow={latestContinuousEffRow}
            continuousEffSeries={continuousEffSeries}
            continuousPhaseLossSeries={continuousPhaseLossSeries}
            continuousPhaseAccSeries={continuousPhaseAccSeries}
          />
        </Suspense>
      )}

      {tab === "evaluation" && (
        <Suspense fallback={<section className="panel"><div className="empty">Loading evaluation dashboard...</div></section>}>
          <EvaluationPanelLazy
            evalSplit={evalSplit}
            setEvalSplit={setEvalSplit}
            cmSplits={cmSplits}
            cmEpochChoice={cmEpochChoice}
            setCmEpochChoice={setCmEpochChoice}
            cmEpochs={cmEpochs}
            inferSplit={inferSplit}
            setInferSplit={setInferSplit}
            inferSplits={inferSplits}
            calSplit={calSplit}
            setCalSplit={setCalSplit}
            calSplits={calSplits}
            cmFilteredRows={cmFilteredRows}
            inferFilteredRows={inferFilteredRows}
            cmQuery={cmQuery}
            calQuery={calQuery}
            inferQuery={inferQuery}
            classMetricsQuery={classMetricsQuery}
            sysQuery={sysQuery}
            needContinuous={needContinuous}
            continuousEffQuery={continuousEffQuery}
            continuousPhaseQuery={continuousPhaseQuery}
            hybridExpertQuery={hybridExpertQuery}
            needBio={needBio}
            bioEpochQuery={bioEpochQuery}
            bioLayerQuery={bioLayerQuery}
            evaluationLandscape={evaluationLandscape}
            openViewAllTab={openViewAllTab}
            cm={cm}
            calibrationChartRows={calibrationChartRows}
            histRows={histRows}
            errorRows={errorRows}
            systemChartRows={systemChartRows}
            classProfileRows={classProfileRows}
            confidenceOutcomeRows={confidenceOutcomeRows}
            entropyConfidenceScatterRows={entropyConfidenceScatterRows}
            classRadarRows={classRadarRows}
            runFamily={runFamily}
            efficiencyFrontierRows={efficiencyFrontierRows}
            latestHybridExpertRows={latestHybridExpertRows}
            latestBioEpochRow={latestBioEpochRow}
            bioTissueGraph={bioTissueGraph}
            bioEvalLayerRows={bioEvalLayerRows}
            hasLayerSignViolation={hasLayerSignViolation}
          />
        </Suspense>
      )}
      <DashboardAssetSections
        tab={tab}
        runTaskType={runTaskType}
        selectedRun={selectedRun}
        gameTraceRows={gameTraceRows}
        gameTraceQuery={gameTraceQuery}
        GameReplayPanelLazy={GameReplayPanelLazy}
        architectureFamily={architectureFamily}
        jumpToFamilyModelRun={jumpToFamilyModelRun}
        modelFamilies={modelFamilies}
        selectedModelFamilyCoverage={selectedModelFamilyCoverage}
        runId={runId}
        setRunId={setRunId}
        modelRunsForFamily={modelRunsForFamily}
        modelCsvPath={modelCsvPath}
        setModelCsvPath={setModelCsvPath}
        modelFileList={modelFileList}
        classMetricsQuery={classMetricsQuery}
        inferQuery={inferQuery}
        calQuery={calQuery}
        classProfileRows={classProfileRows}
        classRadarRows={classRadarRows}
        confidenceOutcomeRows={confidenceOutcomeRows}
        entropyConfidenceScatterRows={entropyConfidenceScatterRows}
        calibrationChartRows={calibrationChartRows}
        errorRows={errorRows}
        InterpretabilitySuiteLazy={InterpretabilitySuiteLazy}
        modelFilesQuery={modelFilesQuery}
        modelPreviewLoading={modelPreviewLoading}
        modelPreviewMap={modelPreviewMap}
        AutoCsvPreviewLazy={AutoCsvPreviewLazy}
        modelCsvQuery={modelCsvQuery}
        CsvModelChartLazy={CsvModelChartLazy}
        reportsQuery={reportsQuery}
      />

      {errors.length > 0 && (
        <footer className="error">
          {errors.map((error, idx) => (
            <div key={`${idx}-${error}`}>{error}</div>
          ))}
        </footer>
      )}
    </div>
  );
}
