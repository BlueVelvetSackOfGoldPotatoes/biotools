import { fetchJson } from "../../lib/fetchJson";
import { requestJson } from "../../lib/requestJson";
import { writeCellEngineSectionToUrl } from "./core";

export function useCellEngineActions(state) {
  const {
    form,
    setForm,
    discoveryForm,
    setDiscoveryForm,
    selectedTaskName,
    genomePreviewByPath,
    genomePreviewItems,
    setCompareGenomePaths,
    queuedGenomePaths,
    setQueuedGenomePaths,
    setGenomeRunStatusByPath,
    setPayload,
    setCurrentIndex,
    setPlaying,
    setBusy,
    setBusyMode,
    setBusyStartedAt,
    setError,
    setDiscoveryResult,
    setDiscoveryJob,
    setFullBenchmarkResult,
    setSelectedEvidenceKey,
    overviewQuery,
    artifactsQuery,
    setActiveSection
  } = state;

  function toggleCompareGenome(genomePath) {
    setCompareGenomePaths((current) => {
      if (current.includes(genomePath)) {
        return current.filter((path) => path !== genomePath);
      }
      const nextItem = genomePreviewByPath.get(genomePath);
      const nextKey = nextItem?.content_hash || genomePath;
      const filtered = current.filter((path) => {
        const item = genomePreviewByPath.get(path);
        const key = item?.content_hash || path;
        return key !== nextKey;
      });
      if (filtered.length < 2) {
        return [...filtered, genomePath];
      }
      return [filtered[filtered.length - 1], genomePath];
    });
  }

  function setCompareGenomeAt(slotIndex, genomePath) {
    setCompareGenomePaths((current) => {
      const next = [...current];
      if (!genomePath) {
        next.splice(slotIndex, 1);
        return next;
      }
      const candidate = genomePreviewByPath.get(genomePath);
      const candidateKey = candidate?.content_hash || genomePath;
      const otherIndex = slotIndex === 0 ? 1 : 0;
      const otherPath = next[otherIndex];
      const otherItem = otherPath ? genomePreviewByPath.get(otherPath) : null;
      const otherKey = otherItem?.content_hash || otherPath;
      if (otherPath && otherKey === candidateKey) {
        next[slotIndex] = genomePath;
        next[otherIndex] = undefined;
        return next.filter(Boolean);
      }
      next[slotIndex] = genomePath;
      return next.filter(Boolean).slice(0, 2);
    });
  }

  function toggleQueuedGenome(genomePath) {
    setQueuedGenomePaths((current) => (
      current.includes(genomePath)
        ? current.filter((path) => path !== genomePath)
        : [...current, genomePath]
    ));
  }

  function queueVisibleGenomes() {
    setQueuedGenomePaths(genomePreviewItems.map((item) => item.genome_path));
  }

  function clearQueuedGenomes() {
    setQueuedGenomePaths([]);
  }

  function replayRequestBody(genomePath) {
    return {
      task_name: form.task_name,
      genome_path: genomePath,
      max_ticks: Number(form.max_ticks),
      damage_tick: form.damage_tick === "" ? null : Number(form.damage_tick),
      theta_deg: form.theta_deg === "" ? null : Number(form.theta_deg),
      task_param_a: form.task_param_a === "" ? null : Number(form.task_param_a),
      task_param_b: form.task_param_b === "" ? null : Number(form.task_param_b),
      seed: Number(form.seed),
      rl_algo: form.rl_algo
    };
  }

  function discoveryRequestBody() {
    return {
      task_name: discoveryForm.task_name,
      num_runs: Number(discoveryForm.num_runs),
      seed_start: Number(discoveryForm.seed_start),
      seed_step: Number(discoveryForm.seed_step),
      population_size: Number(discoveryForm.population_size),
      generations: Number(discoveryForm.generations),
      search_trials: Number(discoveryForm.search_trials),
      search_ticks: Number(discoveryForm.search_ticks),
      final_trials: Number(discoveryForm.final_trials),
      final_ticks: Number(discoveryForm.final_ticks),
      auto_attempts: Number(discoveryForm.auto_attempts),
      rl_algo: discoveryForm.rl_algo,
      odd_enabled: Boolean(discoveryForm.odd_enabled),
      odd_trials: Number(discoveryForm.odd_trials),
      odd_ticks: Number(discoveryForm.odd_ticks),
      body_mode: discoveryForm.body_mode,
      development_steps: Number(discoveryForm.development_steps),
      development_seed_half_width: Number(discoveryForm.development_seed_half_width),
      max_cells: Number(discoveryForm.max_cells),
      body_extent_x: Number(discoveryForm.body_extent_x),
      body_extent_y: Number(discoveryForm.body_extent_y),
      body_extent_z: Number(discoveryForm.body_extent_z),
      chemical_diffusion_steps: Number(discoveryForm.chemical_diffusion_steps),
      development_growth_threshold: Number(discoveryForm.development_growth_threshold),
      chemical_diffusion_rate: Number(discoveryForm.chemical_diffusion_rate),
      chemical_decay: Number(discoveryForm.chemical_decay),
      evolve_body_mode: Boolean(discoveryForm.evolve_body_mode),
      evolve_development_steps: Boolean(discoveryForm.evolve_development_steps),
      evolve_development_seed_half_width: Boolean(discoveryForm.evolve_development_seed_half_width),
      evolve_max_cells: Boolean(discoveryForm.evolve_max_cells),
      evolve_body_extent_x: Boolean(discoveryForm.evolve_body_extent_x),
      evolve_body_extent_y: Boolean(discoveryForm.evolve_body_extent_y),
      evolve_body_extent_z: Boolean(discoveryForm.evolve_body_extent_z),
      evolve_chemical_diffusion_steps: Boolean(discoveryForm.evolve_chemical_diffusion_steps),
      evolve_growth_threshold: Boolean(discoveryForm.evolve_growth_threshold),
      evolve_chemical_diffusion_rate: Boolean(discoveryForm.evolve_chemical_diffusion_rate),
      evolve_chemical_decay: Boolean(discoveryForm.evolve_chemical_decay),
      worm_goal_distance: Number(discoveryForm.worm_goal_distance),
      worm_max_backward: Number(discoveryForm.worm_max_backward),
      pong_target_hits: Number(discoveryForm.pong_target_hits),
      pong_ball_speed: Number(discoveryForm.pong_ball_speed),
      pong_paddle_half_height: Number(discoveryForm.pong_paddle_half_height)
    };
  }

  function fullBenchmarkRequestBody(genomePath = form.genome_path) {
    return {
      task_name: selectedTaskName,
      genome_path: genomePath,
      seed: Number(form.seed),
      rl_algo: "all",
      body_mode: discoveryForm.body_mode,
      development_steps: Number(discoveryForm.development_steps),
      development_seed_half_width: Number(discoveryForm.development_seed_half_width),
      max_cells: Number(discoveryForm.max_cells),
      body_extent_x: Number(discoveryForm.body_extent_x),
      body_extent_y: Number(discoveryForm.body_extent_y),
      body_extent_z: Number(discoveryForm.body_extent_z),
      chemical_diffusion_steps: Number(discoveryForm.chemical_diffusion_steps),
      development_growth_threshold: Number(discoveryForm.development_growth_threshold),
      chemical_diffusion_rate: Number(discoveryForm.chemical_diffusion_rate),
      chemical_decay: Number(discoveryForm.chemical_decay),
      evolve_body_mode: Boolean(discoveryForm.evolve_body_mode),
      evolve_development_steps: Boolean(discoveryForm.evolve_development_steps),
      evolve_development_seed_half_width: Boolean(discoveryForm.evolve_development_seed_half_width),
      evolve_max_cells: Boolean(discoveryForm.evolve_max_cells),
      evolve_body_extent_x: Boolean(discoveryForm.evolve_body_extent_x),
      evolve_body_extent_y: Boolean(discoveryForm.evolve_body_extent_y),
      evolve_body_extent_z: Boolean(discoveryForm.evolve_body_extent_z),
      evolve_chemical_diffusion_steps: Boolean(discoveryForm.evolve_chemical_diffusion_steps),
      evolve_growth_threshold: Boolean(discoveryForm.evolve_growth_threshold),
      evolve_chemical_diffusion_rate: Boolean(discoveryForm.evolve_chemical_diffusion_rate),
      evolve_chemical_decay: Boolean(discoveryForm.evolve_chemical_decay),
      final_trials: Number(discoveryForm.final_trials),
      final_ticks: Number(discoveryForm.final_ticks),
      damage_trials: 40,
      damage_ticks: Number(discoveryForm.final_ticks),
      odd_enabled: true,
      odd_trials: Number(discoveryForm.odd_trials),
      odd_ticks: Number(discoveryForm.odd_ticks),
      worm_goal_distance: Number(discoveryForm.worm_goal_distance),
      worm_max_backward: Number(discoveryForm.worm_max_backward),
      pong_target_hits: Number(discoveryForm.pong_target_hits),
      pong_ball_speed: Number(discoveryForm.pong_ball_speed),
      pong_paddle_half_height: Number(discoveryForm.pong_paddle_half_height)
    };
  }

  function isReplayPass(nextPayload) {
    if (typeof nextPayload?.summary?.solved === "boolean") return nextPayload.summary.solved;
    const successRate = nextPayload?.summary?.cell_clean?.success_rate;
    if (Number.isFinite(successRate)) return successRate >= 1;
    const maxTheta = nextPayload?.analysis?.max_abs_theta_deg;
    const failTheta = nextPayload?.analysis?.fail_angle_deg || 15;
    if (Number.isFinite(maxTheta)) return maxTheta < failTheta;
    return false;
  }

  async function runReplayForGenome(genomePath, { autoplay = true, refreshOverview = true } = {}) {
    const nextPayload = await requestJson("/api/cellengine/replay", {
      method: "POST",
      body: replayRequestBody(genomePath)
    });
    setPayload(nextPayload);
    setForm((current) => ({
      ...current,
      task_name: nextPayload?.config?.task_name || current.task_name,
      genome_path: genomePath,
      rl_algo: nextPayload?.config?.rl_algo || current.rl_algo,
      task_param_a: nextPayload?.config?.task_param_a ?? current.task_param_a,
      task_param_b: nextPayload?.config?.task_param_b ?? current.task_param_b
    }));
    setCurrentIndex(0);
    setPlaying(Boolean(autoplay));
    if (refreshOverview) {
      overviewQuery.refresh();
    }
    return nextPayload;
  }

  async function launchReplay() {
    setBusyMode("launch");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    try {
      await runReplayForGenome(form.genome_path, { autoplay: true, refreshOverview: true });
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function launchDiscovery() {
    setBusyMode("discover");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    setDiscoveryResult(null);
    setDiscoveryJob(null);
    try {
      const nextPayload = await requestJson("/api/cellengine/discover/start", {
        method: "POST",
        body: discoveryRequestBody()
      });
      setDiscoveryJob(nextPayload);
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    } catch (err) {
      setError(String(err?.message || err));
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function launchSingleFullBenchmarkDiscovery() {
    setBusyMode("full_discover");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    setFullBenchmarkResult(null);
    setDiscoveryResult(null);
    setDiscoveryJob(null);
    try {
      const nextPayload = await requestJson("/api/cellengine/discover/start", {
        method: "POST",
        body: {
          ...discoveryRequestBody(),
          num_runs: 1,
          rl_algo: "all",
          odd_enabled: true
        }
      });
      setDiscoveryJob(nextPayload);
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    } catch (err) {
      setError(String(err?.message || err));
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function launchFullBenchmarkOnSelectedGenome() {
    if (!form.genome_path) return;
    setBusyMode("full_eval");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    setFullBenchmarkResult(null);
    try {
      const nextPayload = await requestJson("/api/cellengine/full-benchmark", {
        method: "POST",
        body: fullBenchmarkRequestBody(form.genome_path)
      });
      setFullBenchmarkResult(nextPayload);
      if (nextPayload?.champion_genome_path) {
        setForm((current) => ({ ...current, genome_path: nextPayload.champion_genome_path }));
      }
      overviewQuery.refresh();
      artifactsQuery.refresh();
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function runReplayFromGallery(genomePath) {
    if (!genomePath) return;
    setBusyMode("gallery");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    setGenomeRunStatusByPath((current) => ({
      ...current,
      [genomePath]: { state: "running", pass: null }
    }));
    try {
      const nextPayload = await runReplayForGenome(genomePath, { autoplay: true, refreshOverview: true });
      setGenomeRunStatusByPath((current) => ({
        ...current,
        [genomePath]: {
          state: "done",
          pass: isReplayPass(nextPayload),
          relative_output_dir: nextPayload?.relative_output_dir || ""
        }
      }));
    } catch (err) {
      const message = String(err?.message || err);
      setGenomeRunStatusByPath((current) => ({
        ...current,
        [genomePath]: { state: "error", pass: false, message }
      }));
      setError(message);
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function runQueuedGenomes() {
    if (!queuedGenomePaths.length) return;
    setBusyMode("batch");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    try {
      for (const genomePath of queuedGenomePaths) {
        setGenomeRunStatusByPath((current) => ({
          ...current,
          [genomePath]: { state: "running", pass: null }
        }));
        try {
          const nextPayload = await runReplayForGenome(genomePath, { autoplay: true, refreshOverview: false });
          setGenomeRunStatusByPath((current) => ({
            ...current,
            [genomePath]: {
              state: "done",
              pass: isReplayPass(nextPayload),
              relative_output_dir: nextPayload?.relative_output_dir || ""
            }
          }));
        } catch (err) {
          setGenomeRunStatusByPath((current) => ({
            ...current,
            [genomePath]: { state: "error", pass: false, message: String(err?.message || err) }
          }));
        }
      }
      overviewQuery.refresh();
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function loadReplay(relativeOutputDir) {
    setBusyMode("load");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    try {
      const nextPayload = await fetchJson(
        `/api/cellengine/replay?dir=${encodeURIComponent(relativeOutputDir)}`,
        { dedupe: false }
      );
      setPayload(nextPayload);
      setForm((current) => ({
        ...current,
        task_name: nextPayload?.summary?.task_name || current.task_name,
        rl_algo: nextPayload?.config?.rl_algo || current.rl_algo
      }));
      setCurrentIndex(0);
      setPlaying(false);
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  async function foregroundBenchmarkRun(run) {
    if (!run) return;
    const taskName = run.task_name || "cartpole_balance";
    setForm((current) => ({
      ...current,
      task_name: taskName,
      genome_path: run.champion_genome_path || current.genome_path
    }));
    setSelectedEvidenceKey(`run:${run.relative_output_dir}`);
    if (run.files?.replay_trace_csv) {
      writeCellEngineSectionToUrl("replay");
      setActiveSection("replay");
      await loadReplay(run.relative_output_dir);
      return;
    }
    writeCellEngineSectionToUrl("reports");
    setActiveSection("reports");
  }

  async function foregroundDiscoveryBatch(batch) {
    if (!batch) return;
    const taskName = batch.task_name || batch.best_run?.summary?.task_name || "cartpole_balance";
    setBusyMode("discover_restore");
    setBusyStartedAt(Date.now());
    setBusy(true);
    setError("");
    try {
      const restored = await fetchJson(
        `/api/cellengine/discover/batch?dir=${encodeURIComponent(batch.relative_output_dir)}`,
        { dedupe: false }
      );
      const nextJob = restored?.job || null;
      const nextResult = restored?.result || null;
      setForm((current) => ({
        ...current,
        task_name: taskName,
        genome_path: batch.best_run?.champion_genome_path || current.genome_path
      }));
      setDiscoveryForm((current) => ({
        ...current,
        ...(nextJob?.config || {}),
        task_name: taskName
      }));
      setDiscoveryJob(nextJob);
      setDiscoveryResult(nextResult);
      setSelectedEvidenceKey(`batch:${batch.relative_output_dir}`);
      writeCellEngineSectionToUrl("discover");
      setActiveSection("discover");
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
      setBusyMode("");
      setBusyStartedAt(0);
    }
  }

  return {
    toggleCompareGenome,
    setCompareGenomeAt,
    toggleQueuedGenome,
    queueVisibleGenomes,
    clearQueuedGenomes,
    replayRequestBody,
    discoveryRequestBody,
    fullBenchmarkRequestBody,
    isReplayPass,
    runReplayForGenome,
    launchReplay,
    launchDiscovery,
    launchSingleFullBenchmarkDiscovery,
    launchFullBenchmarkOnSelectedGenome,
    runReplayFromGallery,
    runQueuedGenomes,
    loadReplay,
    foregroundBenchmarkRun,
    foregroundDiscoveryBatch
  };
}
