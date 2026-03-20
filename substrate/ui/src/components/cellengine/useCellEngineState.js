import { useEffect, useMemo, useRef, useState } from "react";
import { usePolling } from "../../hooks/usePolling";
import { fetchJson } from "../../lib/fetchJson";
import { requestJson } from "../../lib/requestJson";
import {
  DEFAULT_BODY_VIEW,
  DISCOVERY_JOB_STORAGE_KEY,
  writeCellEngineSectionToUrl,
  GENOME_PREVIEW_LIMIT,
  OVERVIEW_POLL_MS,
  TASK_META,
  recommendedTaskBodyConfig,
  readCellEngineSectionFromUrl
} from "./core";
import { discoveryCandidateSummary, discoveryRunKey } from "./discoveryViews";

export function useCellEngineState() {
  const overviewQuery = usePolling(
    () => fetchJson("/api/cellengine/overview", { dedupe: false }),
    OVERVIEW_POLL_MS,
    []
  );
  const artifactsQuery = usePolling(
    () => fetchJson("/api/cellengine/artifacts", { dedupe: false }),
    OVERVIEW_POLL_MS,
    []
  );
  const activeDiscoveryQuery = usePolling(
    () => fetchJson("/api/cellengine/discover/active", { dedupe: false }),
    4000,
    []
  );
  const [form, setForm] = useState({
    task_name: "cartpole_balance",
    genome_path: "",
    max_ticks: 500,
    damage_tick: "",
    theta_deg: "",
    task_param_a: "",
    task_param_b: "",
    seed: 42,
    rl_algo: "a2c"
  });
  const [discoveryForm, setDiscoveryForm] = useState({
    task_name: "cartpole_balance",
    num_runs: 4,
    seed_start: 42,
    seed_step: 1,
    population_size: 48,
    generations: 50,
    search_trials: 24,
    search_ticks: 250,
    final_trials: 100,
    final_ticks: 500,
    auto_attempts: 3,
    rl_algo: "none",
    odd_enabled: true,
    odd_trials: 24,
    odd_ticks: 500,
    body_mode: "grown3d",
    development_steps: 8,
    development_seed_half_width: 2,
    max_cells: 96,
    body_extent_x: 6,
    body_extent_y: 14,
    body_extent_z: 2,
    chemical_diffusion_steps: 2,
    development_growth_threshold: 0.54,
    chemical_diffusion_rate: 0.32,
    chemical_decay: 0.08,
    evolve_body_mode: false,
    evolve_development_steps: false,
    evolve_development_seed_half_width: false,
    evolve_max_cells: false,
    evolve_body_extent_x: false,
    evolve_body_extent_y: false,
    evolve_body_extent_z: false,
    evolve_chemical_diffusion_steps: false,
    evolve_growth_threshold: false,
    evolve_chemical_diffusion_rate: false,
    evolve_chemical_decay: false,
    worm_goal_distance: 6.0,
    worm_max_backward: 1.5,
    pong_target_hits: 6,
    pong_ball_speed: 1.1,
    pong_paddle_half_height: 0.22
  });
  const [discoveryResult, setDiscoveryResult] = useState(null);
  const [discoveryJob, setDiscoveryJob] = useState(null);
  const [discoveryHistoryByRun, setDiscoveryHistoryByRun] = useState({});
  const [fullBenchmarkResult, setFullBenchmarkResult] = useState(null);
  const [payload, setPayload] = useState(null);
  const [busy, setBusy] = useState(false);
  const [busyMode, setBusyMode] = useState("");
  const [busyStartedAt, setBusyStartedAt] = useState(0);
  const [busyElapsedSec, setBusyElapsedSec] = useState(0);
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState(() => readCellEngineSectionFromUrl());
  const [selectedEvidenceKey, setSelectedEvidenceKey] = useState("active");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [bodyViews, setBodyViews] = useState({
    atlas: { ...DEFAULT_BODY_VIEW },
    genome: { ...DEFAULT_BODY_VIEW },
    matrix: { ...DEFAULT_BODY_VIEW },
    graph: { ...DEFAULT_BODY_VIEW },
    compareLeft: { ...DEFAULT_BODY_VIEW },
    compareRight: { ...DEFAULT_BODY_VIEW }
  });
  const [cellAtlasMode, setCellAtlasMode] = useState("behavior");
  const [cellMetric, setCellMetric] = useState("energy");
  const [genomeMetric, setGenomeMetric] = useState("gene_expr_6");
  const [selectedCellId, setSelectedCellId] = useState(null);
  const [selectedCellViewMode, setSelectedCellViewMode] = useState("facts");
  const [connectivityEnabled, setConnectivityEnabled] = useState(true);
  const [edgeThreshold, setEdgeThreshold] = useState(0.15);
  const [connectivityMode, setConnectivityMode] = useState("all");
  const [genomePreviewState, setGenomePreviewState] = useState({ loading: false, items: [], error: "" });
  const [compareGenomePaths, setCompareGenomePaths] = useState([]);
  const [queuedGenomePaths, setQueuedGenomePaths] = useState([]);
  const [genomeRunStatusByPath, setGenomeRunStatusByPath] = useState({});
  const [galleryLimit, setGalleryLimit] = useState(GENOME_PREVIEW_LIMIT);
  const autoLoadedRef = useRef(false);
  const bodyOrbitDragRef = useRef({});
  const bodyOrbitMovedRef = useRef({});

  const overview = overviewQuery.data || { available_tasks: [], genome_options: [], recent_replays: [], latest_report: null };
  const artifacts = artifactsQuery.data || { benchmark_runs: [], discovery_batches: [] };
  const selectedTaskName = form.task_name || "cartpole_balance";
  const discoveryRunning = Boolean(discoveryJob && ["queued", "running"].includes(discoveryJob.status));
  const activeDiscoveryJob = useMemo(() => {
    if (discoveryJob && ["queued", "running"].includes(discoveryJob.status)) return discoveryJob;
    const job = activeDiscoveryQuery.data?.job || null;
    return job && ["queued", "running"].includes(job.status) ? job : null;
  }, [activeDiscoveryQuery.data?.job, discoveryJob]);
  const discoveryBusy = busyMode === "discover" || busyMode === "full_discover" || discoveryRunning;
  const replayBusy = ["launch", "full_eval", "gallery", "batch", "load"].includes(busyMode);
  const restoredDiscoveryBatch = String(discoveryJob?.job_id || "").startsWith("persisted:");
  const restoredDiscoveryNeedsTelemetryFallback = useMemo(
    () => Array.isArray(discoveryJob?.runs) && discoveryJob.runs.some((run) =>
      Array.isArray(run?.live_population?.candidates) && run.live_population.candidates.some((candidate) =>
        !Number.isFinite(Number(candidate?.summary?.success_rate))
        && !Number.isFinite(Number(candidate?.summary?.survival_ratio))
        && Number.isFinite(Number(candidate?.fitness))
      )
    ),
    [discoveryJob]
  );
  const morphologyPresetActive = Boolean(
    discoveryForm.evolve_body_mode &&
    discoveryForm.evolve_max_cells &&
    discoveryForm.evolve_development_steps &&
    discoveryForm.evolve_development_seed_half_width &&
    discoveryForm.evolve_body_extent_x &&
    discoveryForm.evolve_body_extent_y &&
    discoveryForm.evolve_body_extent_z
  );
  const allOrganismParamsDiscoverableActive = Boolean(
    discoveryForm.evolve_body_mode &&
    discoveryForm.evolve_max_cells &&
    discoveryForm.evolve_development_steps &&
    discoveryForm.evolve_development_seed_half_width &&
    discoveryForm.evolve_body_extent_x &&
    discoveryForm.evolve_body_extent_y &&
    discoveryForm.evolve_body_extent_z &&
    discoveryForm.evolve_chemical_diffusion_steps &&
    discoveryForm.evolve_growth_threshold &&
    discoveryForm.evolve_chemical_diffusion_rate &&
    discoveryForm.evolve_chemical_decay
  );
  const structureFrozenPresetActive = Boolean(
    !discoveryForm.evolve_body_mode &&
    !discoveryForm.evolve_max_cells &&
    !discoveryForm.evolve_development_steps &&
    !discoveryForm.evolve_development_seed_half_width &&
    !discoveryForm.evolve_body_extent_x &&
    !discoveryForm.evolve_body_extent_y &&
    !discoveryForm.evolve_body_extent_z &&
    !discoveryForm.evolve_chemical_diffusion_steps &&
    !discoveryForm.evolve_growth_threshold &&
    !discoveryForm.evolve_chemical_diffusion_rate &&
    !discoveryForm.evolve_chemical_decay
  );
  const showLiveDiscovery = Boolean(
    discoveryJob && (discoveryJob.status === "queued" || discoveryJob.status === "running" || !discoveryResult)
  );
  const taskOptions = Array.isArray(overview.available_tasks) && overview.available_tasks.length
    ? overview.available_tasks
    : Object.keys(TASK_META);
  const taskSpecificGenomeOptions = useMemo(
    () => (overview.genome_options || []).filter((option) => (option.task_name || "cartpole_balance") === selectedTaskName),
    [overview.genome_options, selectedTaskName]
  );
  const taskSpecificRecentReplays = useMemo(
    () => (overview.recent_replays || []).filter((replay) => (replay.task_name || "cartpole_balance") === selectedTaskName),
    [overview.recent_replays, selectedTaskName]
  );
  const galleryGenomeOptions = useMemo(
    () => taskSpecificGenomeOptions.slice(0, galleryLimit),
    [taskSpecificGenomeOptions, galleryLimit]
  );
  const galleryPreviewRequestItems = useMemo(
    () =>
      galleryGenomeOptions.map((option) => ({
        genome_path: option.genome_path,
        summary_dir: option.summary_dir,
        label: option.label
      })),
    [galleryGenomeOptions]
  );
  const galleryPreviewRequestKey = useMemo(
    () => galleryPreviewRequestItems.map((item) => `${item.genome_path}|${item.summary_dir || ""}|${item.label || ""}`).join("||"),
    [galleryPreviewRequestItems]
  );
  const preferredReplay = useMemo(
    () =>
      taskSpecificRecentReplays.find(
        (replay) =>
          replay.success &&
          replay.has_body_cells &&
          replay.has_cell_rows &&
          replay.has_edges &&
          replay.has_genome_variability
      ) ||
      null,
    [taskSpecificRecentReplays]
  );

  useEffect(() => {
    const syncFromUrl = () => {
      setActiveSection(readCellEngineSectionFromUrl());
    };
    syncFromUrl();
    window.addEventListener("popstate", syncFromUrl);
    return () => window.removeEventListener("popstate", syncFromUrl);
  }, []);

  useEffect(() => {
    writeCellEngineSectionToUrl(activeSection, { replace: true });
  }, [activeSection]);

  useEffect(() => {
    if (!galleryPreviewRequestItems.length) {
      setGenomePreviewState({ loading: false, items: [], error: "" });
      return;
    }
    let cancelled = false;
    setGenomePreviewState((current) => ({
      loading: current.items.length === 0,
      items: current.items,
      error: ""
    }));
    requestJson("/api/cellengine/genome-previews", {
      method: "POST",
      body: {
        items: galleryPreviewRequestItems
      }
    })
      .then((payload) => {
        if (cancelled) return;
        setGenomePreviewState({
          loading: false,
          items: Array.isArray(payload?.items) ? payload.items : [],
          error: ""
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setGenomePreviewState({ loading: false, items: [], error: String(err?.message || err) });
      });
    return () => {
      cancelled = true;
    };
  }, [galleryPreviewRequestKey]);

  useEffect(() => {
    if (taskSpecificGenomeOptions.some((option) => option.genome_path === form.genome_path)) return;
    const fallback = taskSpecificGenomeOptions[0]?.genome_path || "";
    if (fallback !== form.genome_path) {
      setForm((current) => ({ ...current, genome_path: fallback }));
    }
  }, [form.genome_path, taskSpecificGenomeOptions]);

  useEffect(() => {
    if (discoveryForm.task_name === selectedTaskName) return;
    const recommendedBody = recommendedTaskBodyConfig(selectedTaskName);
    setDiscoveryForm((current) => ({
      ...current,
      task_name: selectedTaskName,
      body_mode: recommendedBody.body_mode,
      max_cells: recommendedBody.max_cells,
      body_extent_x: recommendedBody.body_extent_x,
      body_extent_y: recommendedBody.body_extent_y,
      body_extent_z: recommendedBody.body_extent_z
    }));
  }, [discoveryForm.task_name, selectedTaskName]);

  useEffect(() => {
    autoLoadedRef.current = false;
  }, [selectedTaskName]);

  useEffect(() => {
    const payloadTaskName = payload?.summary?.task_name || "";
    if (!payloadTaskName || payloadTaskName === selectedTaskName) return;
    setPayload(null);
    setCurrentIndex(0);
    setPlaying(false);
  }, [payload, selectedTaskName]);

  useEffect(() => {
    if (!discoveryJob?.job_id) return undefined;
    if (!["queued", "running"].includes(discoveryJob.status)) return undefined;

    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const nextJob = await fetchJson(`/api/cellengine/discover/status?job_id=${encodeURIComponent(discoveryJob.job_id)}`, { dedupe: false });
        if (cancelled) return;
        setDiscoveryJob(nextJob);
        if (nextJob.status === "completed") {
          setBusy(false);
          setBusyMode("");
          setBusyStartedAt(0);
          if (nextJob.result) {
            setDiscoveryResult(nextJob.result);
            const firstRun = Array.isArray(nextJob.result.runs) ? nextJob.result.runs[0] || null : null;
            if (firstRun?.champion_genome_path) {
              setForm((current) => ({ ...current, genome_path: firstRun.champion_genome_path }));
            }
          }
          overviewQuery.refresh();
          artifactsQuery.refresh();
          return;
        }
        if (nextJob.status === "error") {
          setBusy(false);
          setBusyMode("");
          setBusyStartedAt(0);
          setError(nextJob.error || "Discovery job failed.");
          return;
        }
        timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (cancelled) return;
        setBusy(false);
        setBusyMode("");
        setBusyStartedAt(0);
        setError(String(err?.message || err));
      }
    };

    timer = window.setTimeout(poll, 1200);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [discoveryJob?.job_id, discoveryJob?.status]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (discoveryJob?.job_id && ["queued", "running"].includes(discoveryJob.status)) {
      window.localStorage.setItem(DISCOVERY_JOB_STORAGE_KEY, discoveryJob.job_id);
      return;
    }
    if (!discoveryJob || ["completed", "error"].includes(discoveryJob.status)) {
      window.localStorage.removeItem(DISCOVERY_JOB_STORAGE_KEY);
    }
  }, [discoveryJob]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (discoveryJob?.job_id) return undefined;
    let cancelled = false;

    const restore = async () => {
      const savedJobId = window.localStorage.getItem(DISCOVERY_JOB_STORAGE_KEY) || "";
      const endpoints = savedJobId
        ? [
            `/api/cellengine/discover/status?job_id=${encodeURIComponent(savedJobId)}`,
            "/api/cellengine/discover/active"
          ]
        : ["/api/cellengine/discover/active"];

      for (const endpoint of endpoints) {
        try {
          const payload = await fetchJson(endpoint, { dedupe: false });
          if (cancelled) return;
          const job = payload?.job ? payload.job : payload;
          if (job?.job_id && ["queued", "running"].includes(job.status)) {
            setDiscoveryJob(job);
            setBusyMode("discover");
            setBusyStartedAt(Date.now());
            return;
          }
          // If the saved job completed while we were away, restore it so
          // the UI can show the result instead of appearing blank.
          if (job?.job_id && savedJobId && job.job_id === savedJobId && job.status === "completed") {
            setDiscoveryJob(job);
            if (job.result) {
              setDiscoveryResult(job.result);
              const firstRun = Array.isArray(job.result.runs) ? job.result.runs[0] || null : null;
              if (firstRun?.champion_genome_path) {
                setForm((current) => ({ ...current, genome_path: firstRun.champion_genome_path }));
              }
            }
            window.localStorage.removeItem(DISCOVERY_JOB_STORAGE_KEY);
            return;
          }
        } catch {
          continue;
        }
      }
    };

    restore();
    return () => {
      cancelled = true;
    };
  }, [discoveryJob?.job_id]);

  useEffect(() => {
    setDiscoveryHistoryByRun({});
  }, [discoveryJob?.job_id]);

  useEffect(() => {
    if (!Array.isArray(discoveryJob?.runs) || !discoveryJob.runs.length) return;
    setDiscoveryHistoryByRun((current) => {
      let changed = false;
      const next = { ...current };
      for (const run of discoveryJob.runs) {
        const livePopulation = run?.live_population;
        const candidates = Array.isArray(livePopulation?.candidates) ? livePopulation.candidates : [];
        const generation = Number(livePopulation?.current_generation);
        if (!Number.isFinite(generation) || generation <= 0 || !candidates.length) continue;
        const summaries = candidates
          .map((candidate) => discoveryCandidateSummary(candidate))
          .filter(Boolean);
        if (!summaries.length) continue;
        const successValues = summaries
          .map((summary) => Number(summary?.success_rate))
          .filter((value) => Number.isFinite(value));
        const survivalValues = summaries
          .map((summary) => Number(summary?.survival_ratio))
          .filter((value) => Number.isFinite(value));
        const taskPrimaryValues = summaries
          .map((summary) => Number(summary?.task_primary))
          .filter((value) => Number.isFinite(value));
        const fitnessValues = candidates
          .map((candidate) => Number(candidate?.fitness))
          .filter((value) => Number.isFinite(value));
        if (!successValues.length && !survivalValues.length) continue;
        const bestSummary = candidates
          .map((candidate) => discoveryCandidateSummary(candidate))
          .find(Boolean);
        const key = discoveryRunKey(run);
        const row = {
          generation,
          best_success_rate: successValues.length ? Math.max(...successValues) : null,
          mean_success_rate: successValues.length ? successValues.reduce((sum, value) => sum + value, 0) / successValues.length : null,
          mean_survival_ratio: survivalValues.length ? survivalValues.reduce((sum, value) => sum + value, 0) / survivalValues.length : null,
          best_fitness: fitnessValues.length ? Math.max(...fitnessValues) : null,
          mean_fitness: fitnessValues.length ? fitnessValues.reduce((sum, value) => sum + value, 0) / fitnessValues.length : null,
          worst_fitness: fitnessValues.length ? Math.min(...fitnessValues) : null,
          best_task_primary: bestSummary ? Number(bestSummary.task_primary) : null,
          task_primary_label: bestSummary?.task_primary_label || null
        };
        const previous = current[key] || { seed: run.seed, run_index: run.run_index, rows: [] };
        const filteredRows = previous.rows.filter((entry) => entry.generation !== generation);
        const updatedRows = [...filteredRows, row].sort((a, b) => a.generation - b.generation);
        const previousLast = previous.rows[previous.rows.length - 1];
        if (
          !previousLast ||
          previousLast.generation !== row.generation ||
          previousLast.best_success_rate !== row.best_success_rate ||
          previousLast.mean_success_rate !== row.mean_success_rate ||
          previousLast.mean_survival_ratio !== row.mean_survival_ratio ||
          previousLast.best_fitness !== row.best_fitness ||
          previousLast.mean_fitness !== row.mean_fitness ||
          previousLast.worst_fitness !== row.worst_fitness ||
          previousLast.best_task_primary !== row.best_task_primary ||
          previousLast.task_primary_label !== row.task_primary_label ||
          previous.status !== run.status
        ) {
          next[key] = { ...previous, status: run.status, rows: updatedRows };
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [discoveryJob]);

  useEffect(() => {
    if (autoLoadedRef.current) return;
    if (!preferredReplay) return;
    autoLoadedRef.current = true;
    fetchJson(`/api/cellengine/replay?dir=${encodeURIComponent(preferredReplay.relative_output_dir)}`, { dedupe: false })
      .then((nextPayload) => {
        setPayload(nextPayload);
        setCurrentIndex(0);
        setPlaying(false);
      })
      .catch(() => {
        autoLoadedRef.current = false;
      });
  }, [preferredReplay]);

  useEffect(() => {
    if (!busy || !busyStartedAt) {
      setBusyElapsedSec(0);
      return;
    }
    const updateElapsed = () => {
      setBusyElapsedSec(Math.max(0, Math.floor((Date.now() - busyStartedAt) / 1000)));
    };
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [busy, busyStartedAt]);


  return {
    overviewQuery,
    artifactsQuery,
    activeDiscoveryQuery,
    form,
    setForm,
    discoveryForm,
    setDiscoveryForm,
    discoveryResult,
    setDiscoveryResult,
    discoveryJob,
    setDiscoveryJob,
    discoveryHistoryByRun,
    setDiscoveryHistoryByRun,
    fullBenchmarkResult,
    setFullBenchmarkResult,
    payload,
    setPayload,
    busy,
    setBusy,
    busyMode,
    setBusyMode,
    busyStartedAt,
    setBusyStartedAt,
    busyElapsedSec,
    setBusyElapsedSec,
    error,
    setError,
    activeSection,
    setActiveSection,
    selectedEvidenceKey,
    setSelectedEvidenceKey,
    currentIndex,
    setCurrentIndex,
    playing,
    setPlaying,
    speed,
    setSpeed,
    bodyViews,
    setBodyViews,
    cellAtlasMode,
    setCellAtlasMode,
    cellMetric,
    setCellMetric,
    genomeMetric,
    setGenomeMetric,
    selectedCellId,
    setSelectedCellId,
    selectedCellViewMode,
    setSelectedCellViewMode,
    connectivityEnabled,
    setConnectivityEnabled,
    edgeThreshold,
    setEdgeThreshold,
    connectivityMode,
    setConnectivityMode,
    genomePreviewState,
    setGenomePreviewState,
    compareGenomePaths,
    setCompareGenomePaths,
    queuedGenomePaths,
    setQueuedGenomePaths,
    genomeRunStatusByPath,
    setGenomeRunStatusByPath,
    galleryLimit,
    setGalleryLimit,
    autoLoadedRef,
    bodyOrbitDragRef,
    bodyOrbitMovedRef,
    overview,
    artifacts,
    selectedTaskName,
    discoveryRunning,
    activeDiscoveryJob,
    discoveryBusy,
    replayBusy,
    restoredDiscoveryBatch,
    restoredDiscoveryNeedsTelemetryFallback,
    morphologyPresetActive,
    allOrganismParamsDiscoverableActive,
    structureFrozenPresetActive,
    showLiveDiscovery,
    taskOptions,
    taskSpecificGenomeOptions,
    taskSpecificRecentReplays,
    galleryGenomeOptions,
    galleryPreviewRequestItems,
    galleryPreviewRequestKey,
    preferredReplay
  };
}
