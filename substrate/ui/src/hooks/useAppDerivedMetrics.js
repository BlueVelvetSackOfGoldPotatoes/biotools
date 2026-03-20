import { useCallback, useEffect, useMemo } from "react";
import {
  buildGroupedSeries,
  confusionMatrix,
  listDistinctStrings,
  pivotEpoch,
  seriesColor,
  topMisclassifications
} from "../lib/dashboardShared";
import {
  bestRunsByFamily,
  buildBioTissueGraphData,
  buildConfidenceOutcomeHistogram,
  buildEvaluationLandscape,
  buildTrainingLandscape,
  deriveFamilyCoverageFromRuns,
  makeLatencyHistogram
} from "../lib/appLandscapes";
import { VIEW_ALL_SPECS, splitSelectValue } from "../lib/appViewMeta";

export function useAppDerivedMetrics({
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
}) {
  const activeTaskCount = useMemo(
    () => tasks.filter((task) => ["queued", "running", "killing"].includes(task.status)).length,
    [tasks]
  );
  const bestFamilyRuns = useMemo(() => bestRunsByFamily(dashboardRuns), [dashboardRuns]);
  const bestFamilyRunsSignature = useMemo(
    () =>
      bestFamilyRuns
        .map(({ family, run }) => `${family}:${run?.run_id || ""}`)
        .sort()
        .join("|"),
    [bestFamilyRuns]
  );
  const activeViewSpec = viewAllId ? VIEW_ALL_SPECS[viewAllId] || null : null;

  const openViewAllTab = useCallback(
    (nextViewId) => {
      const spec = VIEW_ALL_SPECS[nextViewId];
      if (!spec) return;
      setViewAllId(nextViewId);
      setViewAllReturnTab((current) => {
        if (tab && tab !== "viewall") return tab;
        if (current) return current;
        return spec.tab || "training";
      });
      setTab("viewall");
    },
    [setTab, setViewAllId, setViewAllReturnTab, tab]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    params.set("tab", tab);
    if (benchmarkFilter && benchmarkFilter !== "all") params.set("benchmark", benchmarkFilter);
    else params.delete("benchmark");
    if (viewAllId) params.set("view_all", viewAllId);
    else params.delete("view_all");
    if (viewAllId && viewAllReturnTab) params.set("view_back", viewAllReturnTab);
    else params.delete("view_back");
    if (runId) params.set("run_id", runId);
    else params.delete("run_id");

    const nextUrl = `${window.location.pathname}?${params.toString()}${window.location.hash || ""}`;
    window.history.replaceState(null, "", nextUrl);
  }, [tab, benchmarkFilter, viewAllId, viewAllReturnTab, runId]);

  useEffect(() => {
    if (!viewAllId || viewAllReturnTab) return;
    const spec = VIEW_ALL_SPECS[viewAllId];
    if (spec?.tab) setViewAllReturnTab(spec.tab);
  }, [viewAllId, viewAllReturnTab, setViewAllReturnTab]);

  useEffect(() => {
    if (tab === "viewall" && !activeViewSpec) {
      setTab(viewAllReturnTab || "comparison");
    }
  }, [tab, activeViewSpec, viewAllReturnTab, setTab]);

  const families =
    familiesQuery.data?.families?.length
      ? familiesQuery.data.families
      : deriveFamilyCoverageFromRuns(dashboardRuns);

  useEffect(() => {
    if (!families.length) {
      if (architectureFamily) setArchitectureFamily("");
      return;
    }
    if (!architectureFamily || !families.some((row) => row.family === architectureFamily)) {
      setArchitectureFamily(families[0].family);
    }
  }, [architectureFamily, families, setArchitectureFamily]);

  const selectedFamilyCoverage = families.find((row) => row.family === architectureFamily) || null;
  const modelFamilies = useMemo(
    () => families.filter((row) => (row.runs_with_model_specific || 0) > 0),
    [families]
  );
  const selectedModelFamilyCoverage =
    modelFamilies.find((row) => row.family === architectureFamily) || null;

  useEffect(() => {
    if (tab !== "model") return;
    if (!modelFamilies.length) return;
    if (!architectureFamily || !modelFamilies.some((row) => row.family === architectureFamily)) {
      setArchitectureFamily(modelFamilies[0].family);
    }
  }, [tab, modelFamilies, architectureFamily, setArchitectureFamily]);

  const familyOptions = useMemo(() => {
    const unique = new Set(dashboardRuns.map((run) => run.model_family || "unknown"));
    return ["all", ...[...unique].sort()];
  }, [dashboardRuns]);

  useEffect(() => {
    if (familyFilter === "all") return;
    if (!familyOptions.includes(familyFilter)) setFamilyFilter("all");
  }, [familyFilter, familyOptions, setFamilyFilter]);

  const runOptionsByFamily = useMemo(() => {
    const groups = new Map();
    for (const run of dashboardRuns) {
      const family = run.model_family || "unknown";
      if (!groups.has(family)) groups.set(family, []);
      groups.get(family).push(run);
    }
    for (const list of groups.values()) {
      list.sort((a, b) => (a.run_id < b.run_id ? 1 : -1));
    }
    return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [dashboardRuns]);

  const modelRunsForFamily = useMemo(() => {
    if (!architectureFamily) return [];
    return dashboardRuns
      .filter((run) => run.model_family === architectureFamily)
      .sort((a, b) => (a.run_id < b.run_id ? 1 : -1));
  }, [dashboardRuns, architectureFamily]);

  const visibleRuns = useMemo(() => {
    return dashboardRuns.filter((run) => {
      if (runStateFilter === "active" && !run.active) return false;
      if (runStateFilter === "done" && run.active) return false;
      if (familyFilter !== "all" && run.model_family !== familyFilter) return false;
      return true;
    });
  }, [dashboardRuns, runStateFilter, familyFilter]);

  const jumpToFamilyModelRun = useCallback(
    (family) => {
      const row = modelFamilies.find((item) => item.family === family);
      if (!row) return;
      setArchitectureFamily(family);
      const targetRunId = row.latest_model_specific_run_id || row.latest_run_id;
      if (targetRunId) setRunId(targetRunId);
    },
    [modelFamilies, setArchitectureFamily, setRunId]
  );

  const epochRows = epochQuery.data?.rows || [];
  const batchRows = batchQuery.data?.rows || [];
  const gameTraceRows = gameTraceQuery.data?.rows || [];
  const epochPivot = useMemo(() => pivotEpoch(epochRows), [epochRows]);
  const batchSeries = useMemo(
    () =>
      batchRows
        .filter(
          (row) =>
            typeof row.global_step === "number" &&
            Number.isFinite(row.global_step) &&
            row.split === "train_batch"
        )
        .sort((a, b) => a.global_step - b.global_step),
    [batchRows]
  );
  const trainingLandscape = useMemo(
    () => buildTrainingLandscape(epochRows, batchRows),
    [epochRows, batchRows]
  );
  const epochSplits = useMemo(() => listDistinctStrings(epochRows, "split"), [epochRows]);
  const bioEpochRows = bioEpochQuery.data?.rows || [];
  const bioLayerRows = bioLayerQuery.data?.rows || [];
  const continuousEffRows = continuousEffQuery.data?.rows || [];
  const continuousPhaseRows = continuousPhaseQuery.data?.rows || [];
  const hybridExpertRows = hybridExpertQuery.data?.rows || [];

  function metricLines(metricName, offset = 0) {
    return epochSplits
      .filter((split) => epochPivot.some((row) => typeof row[`${split}_${metricName}`] === "number"))
      .map((split, idx) => ({
        label: split,
        key: `${split}_${metricName}`,
        color: seriesColor(idx + offset)
      }));
  }

  const lossLines = metricLines("loss", 0);
  const accuracyLines = metricLines("accuracy", 2);
  const throughputLines = metricLines("samples_per_sec", 4);

  const bioEpochSeries = useMemo(
    () =>
      bioEpochRows
        .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
        .sort((a, b) => a.epoch - b.epoch),
    [bioEpochRows]
  );
  const latestBioEpochRow = bioEpochSeries.length ? bioEpochSeries[bioEpochSeries.length - 1] : null;
  const hasEpochSignViolation = useMemo(
    () =>
      bioEpochSeries.some(
        (row) =>
          typeof row.sign_violation_fraction === "number" &&
          Number.isFinite(row.sign_violation_fraction)
      ),
    [bioEpochSeries]
  );
  const hasLayerSignViolation = useMemo(
    () =>
      bioLayerRows.some(
        (row) =>
          typeof row.sign_violation_fraction === "number" &&
          Number.isFinite(row.sign_violation_fraction)
      ),
    [bioLayerRows]
  );

  const bioLayerMaskSeries = useMemo(
    () => buildGroupedSeries(bioLayerRows, "epoch", "mask_density", "layer", 8),
    [bioLayerRows]
  );
  const bioLayerBioelectricSeries = useMemo(
    () => buildGroupedSeries(bioLayerRows, "epoch", "bioelectric_mean", "layer", 8),
    [bioLayerRows]
  );
  const bioLayerSignSeries = useMemo(
    () => buildGroupedSeries(bioLayerRows, "epoch", "sign_violation_fraction", "layer", 8),
    [bioLayerRows]
  );
  const bioTissueGraph = useMemo(
    () => buildBioTissueGraphData(bioLayerRows, bioEpochSeries),
    [bioLayerRows, bioEpochSeries]
  );
  const latestBioLayerRows = useMemo(() => {
    if (!bioLayerRows.length) return [];
    const targetEpoch =
      latestBioEpochRow && typeof latestBioEpochRow.epoch === "number"
        ? latestBioEpochRow.epoch
        : Math.max(
            ...bioLayerRows
              .map((row) => row.epoch)
              .filter((epoch) => typeof epoch === "number" && Number.isFinite(epoch))
          );
    return bioLayerRows.filter(
      (row) =>
        row.epoch === targetEpoch &&
        typeof row.layer === "string" &&
        row.layer.trim().length > 0
    );
  }, [bioLayerRows, latestBioEpochRow]);
  const bioEvalLayerRows = useMemo(
    () =>
      latestBioLayerRows
        .filter((row) => row.layer !== "none")
        .slice(0, 24)
        .map((row) => ({
          layer: row.layer,
          mask_density: row.mask_density,
          myelin_fraction: row.myelin_fraction,
          mean_delay: row.mean_delay,
          bioelectric_mean: row.bioelectric_mean,
          homeostasis_error: row.homeostasis_error,
          glia_gain: row.glia_gain,
          glia_plasticity: row.glia_plasticity,
          excitatory_fraction: row.excitatory_fraction,
          inhibitory_fraction: row.inhibitory_fraction,
          sign_violation_fraction: row.sign_violation_fraction
        })),
    [latestBioLayerRows]
  );

  const continuousEffSeries = useMemo(
    () =>
      continuousEffRows
        .filter((row) => typeof row.cycle === "number" && Number.isFinite(row.cycle))
        .sort((a, b) => a.cycle - b.cycle),
    [continuousEffRows]
  );
  const latestContinuousEffRow =
    continuousEffSeries.length ? continuousEffSeries[continuousEffSeries.length - 1] : null;
  const continuousPhaseLossSeries = useMemo(
    () => buildGroupedSeries(continuousPhaseRows, "cycle", "loss", "phase", 8),
    [continuousPhaseRows]
  );
  const continuousPhaseAccSeries = useMemo(
    () => buildGroupedSeries(continuousPhaseRows, "cycle", "accuracy", "phase", 8),
    [continuousPhaseRows]
  );
  const efficiencyFrontierRows = useMemo(
    () =>
      continuousEffSeries
        .filter(
          (row) =>
            typeof row.cumulative_samples === "number" &&
            Number.isFinite(row.cumulative_samples) &&
            typeof row.test_accuracy === "number" &&
            Number.isFinite(row.test_accuracy)
        )
        .map((row) => ({
          cycle: row.cycle,
          samples: row.cumulative_samples,
          test_accuracy: row.test_accuracy,
          best_accuracy: row.best_accuracy
        })),
    [continuousEffSeries]
  );
  const latestHybridExpertRows = useMemo(() => {
    if (!hybridExpertRows.length) return [];
    const cycles = hybridExpertRows
      .map((row) => row.cycle)
      .filter((cycle) => typeof cycle === "number" && Number.isFinite(cycle));
    if (!cycles.length) return [];
    const target = Math.max(...cycles);
    return hybridExpertRows.filter((row) => row.cycle === target);
  }, [hybridExpertRows]);

  const latestEpoch =
    epochPivot.length > 0
      ? epochPivot[epochPivot.length - 1].epoch
      : selectedRun?.latest_test_epoch ?? null;
  const latestAccuracy = selectedRun?.latest_test_accuracy ?? null;
  const latestMetricLabel =
    selectedRun?.benchmark_id === "tictactoe" ? "latest non-loss" : "latest test acc";

  const cmRows = cmQuery.data?.rows || [];
  const calRows = calQuery.data?.rows || [];
  const inferRows = inferQuery.data?.rows || [];
  const classMetricsRows = classMetricsQuery.data?.rows || [];
  const sysRows = sysQuery.data?.rows || [];

  const cmSplits = useMemo(() => listDistinctStrings(cmRows, "split"), [cmRows]);
  const calSplits = useMemo(() => listDistinctStrings(calRows, "split"), [calRows]);
  const inferSplits = useMemo(() => listDistinctStrings(inferRows, "split"), [inferRows]);

  useEffect(() => {
    setEvalSplit((current) => splitSelectValue(cmSplits, current, "test"));
  }, [cmSplits, setEvalSplit]);

  useEffect(() => {
    setCalSplit((current) => splitSelectValue(calSplits, current, "test"));
  }, [calSplits, setCalSplit]);

  useEffect(() => {
    setInferSplit((current) => splitSelectValue(inferSplits, current, "test"));
  }, [inferSplits, setInferSplit]);

  const cmEpochs = useMemo(() => {
    const values = [...new Set(cmRows.map((row) => row.epoch).filter((value) => typeof value === "number" && Number.isFinite(value)))];
    return values.sort((a, b) => a - b);
  }, [cmRows]);

  useEffect(() => {
    if (!cmEpochs.length) {
      setCmEpochChoice("latest");
      return;
    }
    if (cmEpochChoice === "latest") return;
    const numericChoice = Number(cmEpochChoice);
    if (!Number.isFinite(numericChoice) || !cmEpochs.includes(numericChoice)) {
      setCmEpochChoice("latest");
    }
  }, [cmEpochChoice, cmEpochs, setCmEpochChoice]);

  const cmFilteredRows = useMemo(() => {
    const splitRows = cmRows.filter((row) => !evalSplit || row.split === evalSplit);
    if (!splitRows.length) return [];
    const targetEpoch = cmEpochChoice === "latest"
      ? Math.max(...splitRows.map((row) => row.epoch).filter((value) => typeof value === "number" && Number.isFinite(value)))
      : Number(cmEpochChoice);
    return splitRows.filter((row) => row.epoch === targetEpoch);
  }, [cmRows, evalSplit, cmEpochChoice]);
  const cm = useMemo(() => confusionMatrix(cmFilteredRows), [cmFilteredRows]);

  const calFilteredRows = useMemo(() => {
    const filtered = calRows.filter((row) => !calSplit || row.split === calSplit);
    return filtered.sort((a, b) => {
      const aCenter = typeof a.center === "number" ? a.center : 0;
      const bCenter = typeof b.center === "number" ? b.center : 0;
      return aCenter - bCenter;
    });
  }, [calRows, calSplit]);

  const calibrationChartRows = useMemo(
    () =>
      calFilteredRows.map((row) => ({
        ...row,
        expected: row.center,
        empirical_acc: row.empirical_acc,
        avg_conf: row.avg_conf
      })),
    [calFilteredRows]
  );

  const inferFilteredRows = useMemo(
    () => inferRows.filter((row) => !inferSplit || row.split === inferSplit),
    [inferRows, inferSplit]
  );
  const classProfileRows = useMemo(() => {
    const latest = classMetricsRows
      .filter((row) => !inferSplit || row.split === inferSplit)
      .sort((a, b) => {
        const ae = typeof a.epoch === "number" ? a.epoch : -Infinity;
        const be = typeof b.epoch === "number" ? b.epoch : -Infinity;
        return be - ae;
      });
    const latestEpochValue = latest.length ? latest[0].epoch : null;
    return latest
      .filter((row) => row.epoch === latestEpochValue)
      .sort((a, b) => (a.class_id ?? 0) - (b.class_id ?? 0));
  }, [classMetricsRows, inferSplit]);
  const classRadarRows = useMemo(
    () =>
      classProfileRows.map((row) => ({
        class: String(row.class_name ?? row.class_id ?? "class"),
        precision: row.precision,
        recall: row.recall,
        f1: row.f1
      })),
    [classProfileRows]
  );
  const confidenceOutcomeRows = useMemo(
    () => buildConfidenceOutcomeHistogram(inferFilteredRows, 16),
    [inferFilteredRows]
  );
  const entropyConfidenceScatterRows = useMemo(() => {
    return inferFilteredRows
      .filter(
        (row) =>
          typeof row.confidence === "number" &&
          Number.isFinite(row.confidence) &&
          typeof row.entropy === "number" &&
          Number.isFinite(row.entropy)
      )
      .slice(-1200)
      .map((row) => ({
        confidence: row.confidence,
        entropy: row.entropy,
        correct: row.is_correct === 1 || row.is_correct === true ? 1 : 0
      }));
  }, [inferFilteredRows]);
  const evaluationLandscape = useMemo(
    () => buildEvaluationLandscape(epochRows, inferFilteredRows),
    [epochRows, inferFilteredRows]
  );

  const histRows = useMemo(() => makeLatencyHistogram(inferFilteredRows), [inferFilteredRows]);
  const errorRows = useMemo(() => topMisclassifications(inferFilteredRows), [inferFilteredRows]);

  const systemChartRows = useMemo(() => {
    const hasSplit = sysRows.some((row) => typeof row.split === "string" && row.split.length);
    const base = hasSplit && inferSplit ? sysRows.filter((row) => row.split === inferSplit) : sysRows;
    return base.map((row, index) => ({
      index,
      qps: row.qps,
      p50_ms: row.p50_ms,
      p95_ms: row.p95_ms,
      p99_ms: row.p99_ms,
      cpu_pct: row.cpu_pct,
      mem_mb: row.mem_mb
    }));
  }, [inferSplit, sysRows]);

  return {
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
  };
}
