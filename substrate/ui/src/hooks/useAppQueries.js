import { useEffect, useMemo, useState } from "react";
import { usePolling } from "./usePolling";
import { fetchJson } from "../lib/fetchJson";
import { deriveBenchmarkCoverageFromRuns } from "../lib/appLandscapes";
import { isMissingCsvError } from "../lib/appViewMeta";

export function useAppQueries({
  pollMs,
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
}) {
  const [modelPreviewMap, setModelPreviewMap] = useState({});
  const [modelPreviewLoading, setModelPreviewLoading] = useState(false);

  const needComparison = tab === "comparison";

  const runsQuery = usePolling(() => fetchJson("/api/runs"), pollMs, []);
  const tasksQuery = usePolling(
    async () => {
      try {
        return await fetchJson("/api/tasks");
      } catch {
        return { tasks: [], active: 0 };
      }
    },
    Math.max(2500, Math.floor(pollMs * 0.8)),
    []
  );
  const benchmarkQuerySuffix =
    benchmarkFilter === "all" ? "" : `?benchmark=${encodeURIComponent(benchmarkFilter)}`;
  const familiesQuery = usePolling(async () => {
    try {
      return await fetchJson(`/api/families${benchmarkQuerySuffix}`);
    } catch {
      return { families: [] };
    }
  }, pollMs * 2, [benchmarkQuerySuffix]);
  const benchmarksQuery = usePolling(async () => {
    try {
      return await fetchJson("/api/benchmarks");
    } catch {
      return { benchmarks: [] };
    }
  }, pollMs * 2, []);
  const bioStudiesQuery = usePolling(
    async () => {
      if (!needComparison) return { studies: [], latest_study_id: null };
      try {
        return await fetchJson("/api/bio/ablation/studies");
      } catch {
        return { studies: [], latest_study_id: null };
      }
    },
    pollMs * 2,
    [needComparison]
  );
  const runs = runsQuery.data?.runs || [];
  const tasks = tasksQuery.data?.tasks || [];
  const benchmarks =
    benchmarksQuery.data?.benchmarks?.length
      ? benchmarksQuery.data.benchmarks
      : deriveBenchmarkCoverageFromRuns(runs);
  const dashboardRuns = useMemo(
    () =>
      benchmarkFilter === "all"
        ? runs
        : runs.filter((run) => (run.benchmark_id || "unknown") === benchmarkFilter),
    [runs, benchmarkFilter]
  );
  const selectedBenchmark = useMemo(
    () =>
      benchmarkFilter === "all"
        ? null
        : benchmarks.find((row) => row.benchmark_id === benchmarkFilter) || null,
    [benchmarkFilter, benchmarks]
  );
  const runFamily = useMemo(
    () => dashboardRuns.find((run) => run.run_id === runId)?.model_family || "",
    [dashboardRuns, runId]
  );
  const runBenchmarkId = useMemo(
    () => dashboardRuns.find((run) => run.run_id === runId)?.benchmark_id || "",
    [dashboardRuns, runId]
  );
  const runTaskType = useMemo(
    () => dashboardRuns.find((run) => run.run_id === runId)?.task_type || "",
    [dashboardRuns, runId]
  );
  const bioStudies = bioStudiesQuery.data?.studies || [];
  const selectedBioStudy = useMemo(
    () => bioStudies.find((study) => study.study_id === bioStudyId) || null,
    [bioStudies, bioStudyId]
  );

  useEffect(() => {
    if (!needComparison) return;
    const latestStudyId = bioStudiesQuery.data?.latest_study_id || bioStudies[0]?.study_id || "";
    if (!latestStudyId) {
      if (bioStudyId) setBioStudyId("");
      return;
    }
    if (!bioStudyId || !bioStudies.some((study) => study.study_id === bioStudyId)) {
      setBioStudyId(latestStudyId);
    }
  }, [needComparison, bioStudyId, bioStudies, bioStudiesQuery.data, setBioStudyId]);

  useEffect(() => {
    if (!needComparison) return;
    const models = selectedBioStudy?.models || [];
    if (!models.length) {
      if (bioModel) setBioModel("");
      return;
    }
    if (!bioModel || !models.includes(bioModel)) {
      setBioModel(models[0]);
    }
  }, [needComparison, selectedBioStudy, bioModel, setBioModel]);

  const bioAblationQuery = usePolling(
    async () => {
      if (!needComparison || !bioStudyId) {
        return { overview: [], model_analysis: null, study: null, feature_order: [] };
      }
      const params = new URLSearchParams({ study: bioStudyId });
      if (bioModel) params.set("model", bioModel);
      return fetchJson(`/api/bio/ablation?${params.toString()}`);
    },
    pollMs * 2,
    [needComparison, bioStudyId, bioModel]
  );

  useEffect(() => {
    if (benchmarkFilter === "all") return;
    if (!benchmarks.some((row) => row.benchmark_id === benchmarkFilter)) {
      setBenchmarkFilter("all");
    }
  }, [benchmarkFilter, benchmarks, setBenchmarkFilter]);

  useEffect(() => {
    if (!dashboardRuns.length) {
      if (runId) setRunId("");
      return;
    }
    if (!dashboardRuns.some((run) => run.run_id === runId)) setRunId(dashboardRuns[0].run_id);
  }, [runId, dashboardRuns, setRunId]);

  const needTraining = tab === "training" && Boolean(runId);
  const needEvaluation = tab === "evaluation" && Boolean(runId);
  const needLandscape = (tab === "training" || tab === "evaluation") && Boolean(runId);
  const needGame = tab === "game" && Boolean(runId) && runTaskType === "control";
  const needModel = tab === "model" && Boolean(runId);
  const needReports = tab === "reports" && Boolean(runId);
  const needInterpretability = (needEvaluation || needModel) && Boolean(runId);
  const needBio = (needTraining || needEvaluation) && Boolean(runId) && Boolean(runFamily);
  const needContinuous = (needTraining || needEvaluation) && Boolean(runId) && Boolean(runFamily);

  const epochQuery = usePolling(
    () =>
      needLandscape
        ? fetchJson(`/api/runs/${runId}/csv?path=learning/epoch_metrics.csv`)
        : Promise.resolve({ rows: [] }),
    pollMs,
    [needLandscape, runId]
  );

  const batchQuery = usePolling(
    () =>
      needLandscape
        ? fetchJson(`/api/runs/${runId}/csv?path=learning/batch_metrics.csv`)
        : Promise.resolve({ rows: [] }),
    pollMs,
    [needLandscape, runId]
  );

  const gameTracePath = runFamily && runBenchmarkId
    ? `model_specific/${runFamily}/${runBenchmarkId}_game_trace.csv`
    : "";
  const gameTraceQuery = usePolling(
    async () => {
      if (!needGame || !gameTracePath) return { rows: [] };
      try {
        return await fetchJson(
          `/api/runs/${runId}/csv?path=${encodeURIComponent(gameTracePath)}&tail=6000`
        );
      } catch (err) {
        if (isMissingCsvError(String(err))) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needGame, runId, gameTracePath]
  );

  const bioEpochPath = runFamily ? `model_specific/${runFamily}/bio_epoch_dynamics.csv` : "";
  const bioLayerPath = runFamily ? `model_specific/${runFamily}/bio_layer_dynamics.csv` : "";
  const continuousEffPath = runFamily ? `model_specific/${runFamily}/continuous_efficiency.csv` : "";
  const continuousPhasePath = runFamily ? `model_specific/${runFamily}/continuous_phase_metrics.csv` : "";
  const hybridExpertPath = runFamily ? `model_specific/${runFamily}/hybrid_expert_metrics.csv` : "";

  const bioEpochQuery = usePolling(
    async () => {
      if (!needBio) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(bioEpochPath)}`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needBio, runId, bioEpochPath]
  );

  const bioLayerQuery = usePolling(
    async () => {
      if (!needBio) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(bioLayerPath)}`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needBio, runId, bioLayerPath]
  );

  const continuousEffQuery = usePolling(
    async () => {
      if (!needContinuous) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(continuousEffPath)}`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needContinuous, runId, continuousEffPath]
  );

  const continuousPhaseQuery = usePolling(
    async () => {
      if (!needContinuous) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(continuousPhasePath)}`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needContinuous, runId, continuousPhasePath]
  );

  const hybridExpertQuery = usePolling(
    async () => {
      if (!needContinuous) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(hybridExpertPath)}`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needContinuous, runId, hybridExpertPath]
  );

  const cmQuery = usePolling(
    async () => {
      if (!needInterpretability) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=learning/confusion_matrix.csv`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needInterpretability, runId]
  );

  const calQuery = usePolling(
    async () => {
      if (!needInterpretability) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=deployment/calibration_bins.csv`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needInterpretability, runId]
  );

  const inferQuery = usePolling(
    async () => {
      if (!needInterpretability) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=deployment/inference_metrics.csv`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needInterpretability, runId]
  );

  const classMetricsQuery = usePolling(
    async () => {
      if (!needInterpretability) return { rows: [] };
      try {
        return await fetchJson(`/api/runs/${runId}/csv?path=learning/class_metrics.csv`);
      } catch (err) {
        if (String(err).includes("404")) return { rows: [] };
        throw err;
      }
    },
    pollMs,
    [needInterpretability, runId]
  );

  const sysQuery = usePolling(
    () =>
      needEvaluation
        ? fetchJson(`/api/runs/${runId}/csv?path=deployment/system_metrics.csv`)
        : Promise.resolve({ rows: [] }),
    pollMs,
    [needEvaluation, runId]
  );

  const modelFilesQuery = usePolling(
    () =>
      needModel
        ? fetchJson(`/api/runs/${runId}/model-specific-files`)
        : Promise.resolve({ files: [] }),
    pollMs,
    [needModel, runId]
  );

  useEffect(() => {
    const files = modelFilesQuery.data?.files || [];
    if (!files.length) {
      if (modelCsvPath) setModelCsvPath("");
      return;
    }
    if (!modelCsvPath || !files.includes(modelCsvPath)) setModelCsvPath(files[0]);
  }, [modelFilesQuery.data, modelCsvPath, setModelCsvPath]);

  const modelCsvQuery = usePolling(
    () =>
      needModel && modelCsvPath
        ? fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(modelCsvPath)}`)
        : Promise.resolve({ rows: [] }),
    pollMs,
    [needModel, modelCsvPath, runId]
  );

  const reportsQuery = usePolling(
    () =>
      needReports
        ? fetchJson(`/api/reports?run_id=${encodeURIComponent(runId)}`)
        : Promise.resolve({ images: [] }),
    pollMs * 2,
    [needReports, runId]
  );

  const modelFileList = modelFilesQuery.data?.files || [];
  const modelFileSignature = modelFileList.join("|");
  useEffect(() => {
    if (!needModel || !runId || !modelFileList.length) {
      setModelPreviewMap((previous) => {
        if (!Object.keys(previous).length) return previous;
        return {};
      });
      setModelPreviewLoading((previous) => (previous ? false : previous));
      return;
    }

    let alive = true;
    setModelPreviewLoading(true);

    Promise.all(
      modelFileList.map(async (filePath) => {
        try {
          const payload = await fetchJson(`/api/runs/${runId}/csv?path=${encodeURIComponent(filePath)}`);
          return [filePath, { rows: payload.rows || [], error: null }];
        } catch (err) {
          return [filePath, { rows: [], error: String(err) }];
        }
      })
    ).then((entries) => {
      if (!alive) return;
      setModelPreviewMap(Object.fromEntries(entries));
      setModelPreviewLoading(false);
    });

    return () => {
      alive = false;
    };
  }, [needModel, runId, modelFileSignature, modelPreviewNonce]);

  const selectedRun = dashboardRuns.find((run) => run.run_id === runId);
  const activeRuns = dashboardRuns.filter((run) => run.active);

  return {
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
  };
}
