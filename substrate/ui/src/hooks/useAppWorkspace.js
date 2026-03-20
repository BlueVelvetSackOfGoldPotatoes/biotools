import { useCallback, useEffect, useRef } from "react";
import { fileLabel } from "../lib/dashboardShared";
import { isMissingCsvError, specPaths } from "../lib/appViewMeta";

export function useAppWorkspace({
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
}) {
  const successTimes = [
    tasksQuery.lastSuccessAt,
    runsQuery.lastSuccessAt,
    familiesQuery.lastSuccessAt,
    benchmarksQuery.lastSuccessAt,
    bioStudiesQuery.lastSuccessAt,
    bioAblationQuery.lastSuccessAt,
    epochQuery.lastSuccessAt,
    batchQuery.lastSuccessAt,
    gameTraceQuery.lastSuccessAt,
    bioEpochQuery.lastSuccessAt,
    bioLayerQuery.lastSuccessAt,
    continuousEffQuery.lastSuccessAt,
    continuousPhaseQuery.lastSuccessAt,
    hybridExpertQuery.lastSuccessAt,
    cmQuery.lastSuccessAt,
    calQuery.lastSuccessAt,
    inferQuery.lastSuccessAt,
    classMetricsQuery.lastSuccessAt,
    sysQuery.lastSuccessAt,
    modelFilesQuery.lastSuccessAt,
    modelCsvQuery.lastSuccessAt,
    reportsQuery.lastSuccessAt
  ].filter((value) => typeof value === "number");
  const lastRefresh = successTimes.length ? Math.max(...successTimes) : null;

  const refreshTaskAndRunQueries = useCallback(() => {
    tasksQuery.refresh();
    runsQuery.refresh();
    familiesQuery.refresh();
    benchmarksQuery.refresh();
  }, [tasksQuery, runsQuery, familiesQuery, benchmarksQuery]);

  const runTaskMutation = useCallback(
    async (busyKey, requestFactory, successMessage) => {
      setTaskBusyKey(busyKey);
      setTaskActionError("");
      try {
        await requestFactory();
        setTaskActionMessage(successMessage);
        refreshTaskAndRunQueries();
      } catch (err) {
        setTaskActionError(String(err));
      } finally {
        setTaskBusyKey("");
      }
    },
    [refreshTaskAndRunQueries, setTaskActionError, setTaskActionMessage, setTaskBusyKey]
  );

  const createTrainTask = useCallback(
    (payload) =>
      runTaskMutation(
        "create-train",
        () => requestJson("/api/tasks/train", { method: "POST", body: payload }),
        "Training task started."
      ),
    [requestJson, runTaskMutation]
  );

  const createEvaluateTask = useCallback(
    (payload) =>
      runTaskMutation(
        "create-eval",
        () => requestJson("/api/tasks/evaluate", { method: "POST", body: payload }),
        "Evaluation task started."
      ),
    [requestJson, runTaskMutation]
  );

  const killTaskAction = useCallback(
    (taskId) =>
      runTaskMutation(
        `kill:${taskId}`,
        () => requestJson(`/api/tasks/${encodeURIComponent(taskId)}/kill`, { method: "POST" }),
        `Kill requested for ${taskId}.`
      ),
    [requestJson, runTaskMutation]
  );

  const startTaskAction = useCallback(
    (taskId) =>
      runTaskMutation(
        `start:${taskId}`,
        () => requestJson(`/api/tasks/${encodeURIComponent(taskId)}/start`, { method: "POST" }),
        `Task restarted from ${taskId}.`
      ),
    [requestJson, runTaskMutation]
  );

  const deleteTaskAction = useCallback(
    (taskId) =>
      runTaskMutation(
        `delete:${taskId}`,
        () => requestJson(`/api/tasks/${encodeURIComponent(taskId)}`, { method: "DELETE" }),
        `Task deleted: ${taskId}.`
      ),
    [requestJson, runTaskMutation]
  );

  useEffect(() => {
    if (!taskActionMessage) return undefined;
    const timer = setTimeout(() => setTaskActionMessage(""), 5000);
    return () => clearTimeout(timer);
  }, [taskActionMessage, setTaskActionMessage]);

  const tabErrors = [runsQuery.error, familiesQuery.error, benchmarksQuery.error];
  if (tab === "tasks") tabErrors.push(tasksQuery.error, taskActionError);
  if (tab === "comparison") {
    tabErrors.push(bioStudiesQuery.error, bioAblationQuery.error);
  }
  if (tab === "training") {
    tabErrors.push(
      epochQuery.error,
      batchQuery.error,
      bioEpochQuery.error,
      bioLayerQuery.error,
      continuousEffQuery.error,
      continuousPhaseQuery.error,
      hybridExpertQuery.error
    );
  }
  if (tab === "evaluation") {
    tabErrors.push(
      epochQuery.error,
      batchQuery.error,
      cmQuery.error,
      calQuery.error,
      inferQuery.error,
      classMetricsQuery.error,
      sysQuery.error,
      bioEpochQuery.error,
      bioLayerQuery.error,
      continuousEffQuery.error,
      continuousPhaseQuery.error,
      hybridExpertQuery.error
    );
  }
  if (tab === "game") {
    tabErrors.push(gameTraceQuery.error);
  }
  if (tab === "model") {
    tabErrors.push(
      modelFilesQuery.error,
      modelCsvQuery.error,
      cmQuery.error,
      calQuery.error,
      inferQuery.error,
      classMetricsQuery.error
    );
  }
  if (tab === "reports") tabErrors.push(reportsQuery.error);
  const errors = [...new Set(tabErrors.filter(Boolean))];

  const refreshCurrent = useCallback(() => {
    tasksQuery.refresh();
    runsQuery.refresh();
    familiesQuery.refresh();
    benchmarksQuery.refresh();

    if (tab === "training") {
      epochQuery.refresh();
      batchQuery.refresh();
      bioEpochQuery.refresh();
      bioLayerQuery.refresh();
      continuousEffQuery.refresh();
      continuousPhaseQuery.refresh();
      hybridExpertQuery.refresh();
    }
    if (tab === "evaluation") {
      epochQuery.refresh();
      batchQuery.refresh();
      cmQuery.refresh();
      calQuery.refresh();
      inferQuery.refresh();
      classMetricsQuery.refresh();
      sysQuery.refresh();
      bioEpochQuery.refresh();
      bioLayerQuery.refresh();
      continuousEffQuery.refresh();
      continuousPhaseQuery.refresh();
      hybridExpertQuery.refresh();
    }
    if (tab === "game") {
      gameTraceQuery.refresh();
    }
    if (tab === "model") {
      modelFilesQuery.refresh();
      modelCsvQuery.refresh();
      cmQuery.refresh();
      calQuery.refresh();
      inferQuery.refresh();
      classMetricsQuery.refresh();
      setModelPreviewNonce((v) => v + 1);
    }
    if (tab === "comparison") {
      bioStudiesQuery.refresh();
      bioAblationQuery.refresh();
    }
    if (tab === "reports") reportsQuery.refresh();
    if (tab === "tasks") setTaskActionError("");
    if (activeViewSpec) setViewAllNonce((value) => value + 1);
  }, [
    activeViewSpec,
    batchQuery,
    benchmarksQuery,
    bioAblationQuery,
    bioEpochQuery,
    bioLayerQuery,
    bioStudiesQuery,
    calQuery,
    classMetricsQuery,
    cmQuery,
    continuousEffQuery,
    continuousPhaseQuery,
    epochQuery,
    familiesQuery,
    gameTraceQuery,
    hybridExpertQuery,
    inferQuery,
    modelCsvQuery,
    modelFilesQuery,
    reportsQuery,
    runsQuery,
    setModelPreviewNonce,
    setTaskActionError,
    setViewAllNonce,
    sysQuery,
    tab,
    tasksQuery
  ]);

  useEffect(() => {
    if (!activeViewSpec) {
      setViewAllPayload((previous) => {
        if (!previous.loading && !previous.error && !Object.keys(previous.byFamily || {}).length) {
          return previous;
        }
        return {
          loading: false,
          error: "",
          fetchedAt: Date.now(),
          byFamily: {}
        };
      });
      return;
    }

    if (!bestFamilyRuns.length) {
      setViewAllPayload((previous) => {
        if (!previous.loading && !previous.error && !Object.keys(previous.byFamily || {}).length) {
          return previous;
        }
        return {
          loading: false,
          error: "",
          fetchedAt: Date.now(),
          byFamily: {}
        };
      });
      return;
    }

    let alive = true;
    setViewAllPayload((previous) => ({
      ...previous,
      loading: true,
      error: ""
    }));

    Promise.all(
      bestFamilyRuns.map(async ({ family, run }) => {
        const paths = specPaths(activeViewSpec, family);
        if (!paths.length) {
          return [
            family,
            {
              family,
              run,
              paths: [],
              byPath: {},
              error: ""
            }
          ];
        }

        const fetches = await Promise.all(
          paths.map(async (path) => {
            try {
              const payload = await fetchJson(
                `/api/runs/${run.run_id}/csv?path=${encodeURIComponent(path)}`
              );
              return [path, { rows: payload.rows || [], error: "" }];
            } catch (err) {
              const message = String(err);
              if (isMissingCsvError(message)) {
                return [path, { rows: [], error: "" }];
              }
              return [path, { rows: [], error: message }];
            }
          })
        );

        const byPath = {};
        const fetchErrors = [];
        for (const [path, payload] of fetches) {
          byPath[path] = payload.rows;
          if (payload.error) {
            fetchErrors.push(`${fileLabel(path)}: ${payload.error}`);
          }
        }

        return [
          family,
          {
            family,
            run,
            paths,
            byPath,
            error: fetchErrors.join(" | ")
          }
        ];
      })
    ).then((entries) => {
      if (!alive) return;
      setViewAllPayload({
        loading: false,
        error: "",
        fetchedAt: Date.now(),
        byFamily: Object.fromEntries(entries)
      });
    }).catch((err) => {
      if (!alive) return;
      setViewAllPayload((previous) => ({
        ...previous,
        loading: false,
        error: String(err),
        fetchedAt: Date.now()
      }));
    });

    return () => {
      alive = false;
    };
  }, [activeViewSpec, bestFamilyRuns, bestFamilyRunsSignature, fetchJson, setViewAllPayload, viewAllNonce]);

  const hasViewAllData = Object.keys(viewAllPayload.byFamily || {}).length > 0;
  const isCellEngineWorkspace = tab === "cellengine";
  const lastDashboardTabRef = useRef("live");

  useEffect(() => {
    if (tab !== "cellengine") {
      lastDashboardTabRef.current = tab;
    }
  }, [tab]);

  const dashboardTabItems = [
    ["live", "Live Runs"],
    ["tasks", "Tasks"],
    ["architectures", "Architectures"],
    ["comparison", "Comparison"],
    ...(viewAllId ? [["viewall", "View-All"]] : []),
    ["training", "Training"],
    ["evaluation", "Evaluation"],
    ["game", "Game Live"],
    ["model", "Model-Specific"],
    ["reports", "Report Gallery"]
  ];

  const switchToCellEngine = useCallback(() => {
    if (viewAllId) setViewAllId("");
    setTab("cellengine");
  }, [setTab, setViewAllId, viewAllId]);

  const switchToDashboard = useCallback(() => {
    if (viewAllId && lastDashboardTabRef.current !== "viewall") setViewAllId("");
    setTab(lastDashboardTabRef.current === "cellengine" ? "live" : lastDashboardTabRef.current || "live");
  }, [setTab, setViewAllId, viewAllId]);

  return {
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
  };
}
