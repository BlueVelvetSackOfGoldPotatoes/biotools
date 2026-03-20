import { useEffect, useMemo } from "react";
import {
  DEFAULT_BODY_VIEW,
  PLAYBACK_FPS,
  clamp,
  clampIndex,
  classifyCellFunction,
  formatNumber,
  formatPct,
  formatSigned,
  taskDiscoveryFieldMeta,
  taskLabel,
  taskMeta,
  taskReplayFieldMeta
} from "./core";
import { bodyHasDepth, normalizeBodyView } from "./body";

export function useCellEngineDerived(state) {
  const {
    overview,
    artifacts,
    form,
    selectedTaskName,
    payload,
    currentIndex,
    genomePreviewState,
    setSelectedEvidenceKey,
    selectedEvidenceKey,
    compareGenomePaths,
    queuedGenomePaths,
    genomeRunStatusByPath,
    bodyViews,
    bodyOrbitDragRef,
    bodyOrbitMovedRef,
    setBodyViews,
    setQueuedGenomePaths,
    setCompareGenomePaths,
    playing,
    setPlaying,
    speed,
    setCurrentIndex,
    selectedCellId,
    setSelectedCellId,
    preferredReplay,
    autoLoadedRef
  } = state;
  const frames = payload?.frames || [];
  const replayBodyCells = payload?.body_cells || [];
  const cellRows = payload?.cell_rows || [];
  const edgeRows = payload?.edge_rows || [];
  const rlFrames = payload?.rl_frames || [];
  const analysis = payload?.analysis || null;
  const rlAnalysis = payload?.rl_analysis || null;
  const genomePreviewItems = genomePreviewState.items || [];
  const genomePreviewByPath = useMemo(
    () => new Map(genomePreviewItems.map((item) => [item.genome_path, item])),
    [genomePreviewItems]
  );
  const selectedGenomePreview = useMemo(
    () => genomePreviewByPath.get(form.genome_path) || null,
    [form.genome_path, genomePreviewByPath]
  );
  const activeTaskName = payload?.summary?.task_name || selectedTaskName;
  const activeTaskMeta = taskMeta(activeTaskName);
  const launchTaskFields = taskReplayFieldMeta(selectedTaskName);
  const discoveryTaskFields = taskDiscoveryFieldMeta(selectedTaskName);
  const frame = frames[currentIndex] || null;
  const rlFrame = rlFrames[Math.min(currentIndex, Math.max(0, rlFrames.length - 1))] || null;
  const latestTaskReport = overview.latest_report?.summary?.task_name === selectedTaskName ? overview.latest_report : null;
  const latestSummary = payload?.summary || latestTaskReport?.summary || null;
  const activeSummary = selectedGenomePreview?.summary || payload?.summary || latestTaskReport?.summary || null;
  const activeOdd = activeSummary?.odd || null;
  const activeEvidenceFiles = selectedGenomePreview?.summary
    ? (selectedGenomePreview.files || null)
    : payload?.summary
      ? (payload?.files || null)
      : (latestTaskReport?.files || null);
  const activeOddSourceLabel = selectedGenomePreview?.summary
    ? `selected genome · ${selectedGenomePreview.label?.replace(/^reports\//, "") || "preview"}`
    : payload?.summary
      ? `loaded replay · ${payload.relative_output_dir?.replace(/^reports\//, "") || "current replay"}`
      : latestTaskReport?.relative_output_dir
        ? `latest report · ${latestTaskReport.relative_output_dir.replace(/^reports\//, "")}`
        : "";
  const benchmarkRuns = useMemo(
    () => (artifacts.benchmark_runs || []).filter((run) => (run.task_name || "cartpole_balance") === selectedTaskName),
    [artifacts.benchmark_runs, selectedTaskName]
  );
  const discoveryBatches = useMemo(
    () => (artifacts.discovery_batches || []).filter((batch) => (batch.task_name || "cartpole_balance") === selectedTaskName),
    [artifacts.discovery_batches, selectedTaskName]
  );
  const evidenceSources = useMemo(() => {
    const sources = [
      {
        key: "active",
        label: "Active context",
        note: activeOddSourceLabel || "selected genome preview / loaded replay / latest report",
        summary: activeSummary,
        files: activeEvidenceFiles
      }
    ];
    for (const run of benchmarkRuns) {
      sources.push({
        key: `run:${run.relative_output_dir}`,
        label: run.relative_output_dir.replace(/^reports\//, ""),
        note: `${taskLabel(run.task_name || "cartpole_balance")} · ${run.summary?.body_mode || "n/a"} · ${run.summary?.champion_cell_count ?? "n/a"} cells`,
        summary: run.summary,
        files: run.files || null,
        championGenomePath: run.champion_genome_path || null
      });
    }
    for (const batch of discoveryBatches) {
      if (!batch.best_run?.summary) continue;
      sources.push({
        key: `batch:${batch.relative_output_dir}`,
        label: batch.batch_id,
        note: `best discovery champion · ${batch.best_run.summary?.body_mode || "n/a"} · ${batch.best_run.summary?.champion_cell_count ?? "n/a"} cells`,
        summary: batch.best_run.summary,
        files: batch.best_run.files || null,
        championGenomePath: batch.best_run.champion_genome_path || null
      });
    }
    return sources;
  }, [activeEvidenceFiles, activeOddSourceLabel, activeSummary, benchmarkRuns, discoveryBatches]);
  const selectedEvidenceSource = useMemo(
    () => evidenceSources.find((source) => source.key === selectedEvidenceKey) || evidenceSources[0] || null,
    [evidenceSources, selectedEvidenceKey]
  );
  const selectedEvidencePreview = useMemo(
    () => selectedEvidenceSource?.championGenomePath
      ? (genomePreviewByPath.get(selectedEvidenceSource.championGenomePath) || null)
      : null,
    [genomePreviewByPath, selectedEvidenceSource]
  );
  useEffect(() => {
    if (!evidenceSources.some((source) => source.key === selectedEvidenceKey)) {
      setSelectedEvidenceKey(evidenceSources[0]?.key || "active");
    }
  }, [evidenceSources, selectedEvidenceKey]);
  const missingCellArtifacts = frames.length > 0 && (!replayBodyCells.length || !cellRows.length);
  const bodyCells = useMemo(() => {
    const previewBodyCells = selectedGenomePreview?.body_cells;
    return Array.isArray(previewBodyCells) && previewBodyCells.length ? previewBodyCells : replayBodyCells;
  }, [selectedGenomePreview, replayBodyCells]);
  const selectedEvidenceBodyCells = useMemo(() => {
    const previewBodyCells = selectedEvidencePreview?.body_cells;
    if (Array.isArray(previewBodyCells) && previewBodyCells.length) return previewBodyCells;
    return selectedEvidenceSource?.key === "active" ? bodyCells : [];
  }, [selectedEvidencePreview, bodyCells, selectedEvidenceSource]);
  const bodyFollowsSelectedGenome = Boolean(
    selectedGenomePreview &&
    Array.isArray(selectedGenomePreview.body_cells) &&
    selectedGenomePreview.body_cells.length
  );
  const compareOptions = useMemo(
    () => genomePreviewItems.filter((item) => !item.error),
    [genomePreviewItems]
  );
  const cellRowsByTick = useMemo(() => {
    const byTick = new Map();
    for (const row of cellRows) {
      if (!byTick.has(row.tick)) byTick.set(row.tick, []);
      byTick.get(row.tick).push(row);
    }
    return byTick;
  }, [cellRows]);
  const edgeRowsByTick = useMemo(() => {
    const byTick = new Map();
    for (const row of edgeRows) {
      if (!byTick.has(row.tick)) byTick.set(row.tick, []);
      byTick.get(row.tick).push(row);
    }
    return byTick;
  }, [edgeRows]);
  const currentFrameCellRows = useMemo(
    () => (frame ? cellRowsByTick.get(frame.tick) || [] : []),
    [cellRowsByTick, frame]
  );
  const currentFrameEdges = useMemo(
    () => (frame ? edgeRowsByTick.get(frame.tick) || [] : []),
    [edgeRowsByTick, frame]
  );
  const functionalStateByCellId = useMemo(() => {
    const byCellId = new Map();
    const stateByCellId = new Map(currentFrameCellRows.map((row) => [row.cell_id, row]));
    for (const cell of bodyCells) {
      byCellId.set(cell.cell_id, classifyCellFunction(cell, stateByCellId.get(cell.cell_id) || null));
    }
    return byCellId;
  }, [bodyCells, currentFrameCellRows]);
  const selectedBodyCell = useMemo(
    () =>
      bodyCells.find((cell) => cell.cell_id === selectedCellId) ||
      bodyCells.find((cell) => cell.hinge) ||
      bodyCells[0] ||
      null,
    [bodyCells, selectedCellId]
  );
  const includeDepthCoordinate = useMemo(() => bodyHasDepth(bodyCells), [bodyCells]);
  const organismHasDepth = includeDepthCoordinate;
  const selectedCellState = useMemo(
    () =>
      currentFrameCellRows.find((row) => row.cell_id === selectedBodyCell?.cell_id) || null,
    [currentFrameCellRows, selectedBodyCell]
  );
  const selectedCellFunction = useMemo(
    () => (selectedBodyCell ? functionalStateByCellId.get(selectedBodyCell.cell_id) || null : null),
    [functionalStateByCellId, selectedBodyCell]
  );
  const selectedCellEdges = useMemo(
    () =>
      currentFrameEdges
        .filter((edge) => edge.src_cell_id === selectedBodyCell?.cell_id || edge.dst_cell_id === selectedBodyCell?.cell_id)
        .sort((a, b) => (b.gap_mean || 0) - (a.gap_mean || 0)),
    [currentFrameEdges, selectedBodyCell]
  );
  const atlasBodyView = useMemo(() => normalizeBodyView(bodyViews.atlas), [bodyViews.atlas]);
  const genomeBodyView = useMemo(() => normalizeBodyView(bodyViews.genome), [bodyViews.genome]);
  const matrixBodyView = useMemo(() => normalizeBodyView(bodyViews.matrix), [bodyViews.matrix]);
  const graphBodyView = useMemo(() => normalizeBodyView(bodyViews.graph), [bodyViews.graph]);
  const compareLeftBodyView = useMemo(() => normalizeBodyView(bodyViews.compareLeft), [bodyViews.compareLeft]);
  const compareRightBodyView = useMemo(() => normalizeBodyView(bodyViews.compareRight), [bodyViews.compareRight]);
  const compareItems = useMemo(
    () => compareGenomePaths.map((path) => genomePreviewByPath.get(path)).filter(Boolean),
    [compareGenomePaths, genomePreviewByPath]
  );
  const queuedCount = queuedGenomePaths.length;
  const queuedPassCount = queuedGenomePaths.filter((path) => genomeRunStatusByPath[path]?.pass === true).length;
  const queuedFailCount = queuedGenomePaths.filter((path) => {
    const status = genomeRunStatusByPath[path];
    return status?.state === "error" || status?.pass === false;
  }).length;

  function bodyInteractionProps(viewKey, enabled = organismHasDepth) {
    if (!enabled) return {};
    return {
      onPointerDown: (event) => {
        if (event.button != null && event.button !== 0) return;
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {
          // ignore capture failures
        }
        bodyOrbitDragRef.current[viewKey] = {
          pointerId: event.pointerId,
          x: event.clientX,
          y: event.clientY
        };
        bodyOrbitMovedRef.current[viewKey] = false;
      },
      onPointerMove: (event) => {
        const drag = bodyOrbitDragRef.current[viewKey];
        if (!drag || drag.pointerId !== event.pointerId) return;
        const deltaX = event.clientX - drag.x;
        const deltaY = event.clientY - drag.y;
        bodyOrbitDragRef.current[viewKey] = {
          pointerId: event.pointerId,
          x: event.clientX,
          y: event.clientY
        };
        if (Math.abs(deltaX) > 1 || Math.abs(deltaY) > 1) {
          bodyOrbitMovedRef.current[viewKey] = true;
        }
        setBodyViews((current) => {
          const nextView = normalizeBodyView(current[viewKey]);
          return {
            ...current,
            [viewKey]: {
              ...nextView,
              yaw: nextView.yaw + deltaX * 0.42,
              pitch: clamp(nextView.pitch - deltaY * 0.32, -80, 80)
            }
          };
        });
      },
      onPointerUp: (event) => {
        const drag = bodyOrbitDragRef.current[viewKey];
        if (!drag || drag.pointerId !== event.pointerId) return;
        delete bodyOrbitDragRef.current[viewKey];
        try {
          event.currentTarget.releasePointerCapture(event.pointerId);
        } catch {
          // ignore release failures
        }
      },
      onPointerCancel: () => {
        delete bodyOrbitDragRef.current[viewKey];
      },
      onLostPointerCapture: () => {
        delete bodyOrbitDragRef.current[viewKey];
      },
      onDoubleClick: (event) => {
        event.preventDefault();
        event.stopPropagation();
        delete bodyOrbitDragRef.current[viewKey];
        bodyOrbitMovedRef.current[viewKey] = false;
        setBodyViews((current) => ({
          ...current,
          [viewKey]: { ...DEFAULT_BODY_VIEW }
        }));
      },
      onClickCapture: (event) => {
        if (!bodyOrbitMovedRef.current[viewKey]) return;
        event.preventDefault();
        event.stopPropagation();
        bodyOrbitMovedRef.current[viewKey] = false;
      },
      onWheel: (event) => {
        event.preventDefault();
        event.stopPropagation();
        const direction = event.deltaY > 0 ? -1 : 1;
        setBodyViews((current) => {
          const nextView = normalizeBodyView(current[viewKey]);
          return {
            ...current,
            [viewKey]: {
              ...nextView,
              zoom: clamp(nextView.zoom + direction * 0.08, 0.55, 2.4)
            }
          };
        });
      },
      onWheelCapture: (event) => {
        event.preventDefault();
      },
      style: {
        cursor: bodyOrbitDragRef.current[viewKey] ? "grabbing" : "grab",
        touchAction: "none",
        overscrollBehavior: "contain"
      },
      title: "Drag to rotate, wheel to zoom, double-click to reset"
    };
  }

  useEffect(() => {
    const valid = new Set(genomePreviewItems.map((item) => item.genome_path));
    setQueuedGenomePaths((current) => current.filter((path) => valid.has(path)));
  }, [genomePreviewItems]);

  useEffect(() => {
    if (!compareOptions.length) {
      if (compareGenomePaths.length) setCompareGenomePaths([]);
      return;
    }
    const candidateByPath = new Map(compareOptions.map((item) => [item.genome_path, item]));
    const usedKeys = new Set();
    const next = [];
    for (const path of compareGenomePaths) {
      const item = candidateByPath.get(path);
      if (!item) continue;
      const key = item.content_hash || item.genome_path;
      if (usedKeys.has(key)) continue;
      usedKeys.add(key);
      next.push(path);
      if (next.length >= 2) break;
    }
    for (const item of compareOptions) {
      if (next.length >= 2) break;
      const key = item.content_hash || item.genome_path;
      if (usedKeys.has(key)) continue;
      usedKeys.add(key);
      next.push(item.genome_path);
    }
    if (next.join("|") !== compareGenomePaths.join("|")) {
      setCompareGenomePaths(next);
    }
  }, [compareOptions, compareGenomePaths]);

  useEffect(() => {
    if (!playing || !frames.length) return undefined;
    if (currentIndex >= frames.length - 1) {
      setPlaying(false);
      return undefined;
    }
    const delayMs = Math.max(8, Math.round(1000 / (PLAYBACK_FPS * speed)));
    const timer = setTimeout(() => {
      setCurrentIndex((value) => clampIndex(value + 1, frames.length - 1));
    }, delayMs);
    return () => clearTimeout(timer);
  }, [playing, currentIndex, frames.length, speed]);

  useEffect(() => {
    if (selectedCellId != null && bodyCells.some((cell) => cell.cell_id === selectedCellId)) return;
    const fallback = bodyCells.find((cell) => cell.hinge) || bodyCells[0] || null;
    setSelectedCellId(fallback?.cell_id ?? null);
  }, [bodyCells, selectedCellId]);

  const timelineRows = useMemo(
    () => {
      const byTick = new Map();
      for (const entry of frames) {
        byTick.set(entry.tick, {
          ...entry,
          abs_theta_deg: Math.abs(entry.theta_deg || 0),
          abs_total_force: Math.abs(entry.total_force || 0),
          angle_margin_deg: (analysis?.fail_angle_deg || 15) - Math.abs(entry.theta_deg || 0)
        });
      }
      for (const entry of rlFrames) {
        const row = byTick.get(entry.tick) || { tick: entry.tick };
        row.rl_theta_deg = entry.theta_deg;
        row.rl_theta_rad = entry.theta_rad;
        row.rl_x = entry.x;
        row.rl_task_aux_a = entry.task_aux_a;
        row.rl_task_counter = entry.task_counter;
        row.rl_total_force = entry.total_force;
        row.rl_terminal = entry.terminal;
        byTick.set(entry.tick, row);
      }
      return [...byTick.values()].sort((a, b) => (a.tick || 0) - (b.tick || 0));
    },
    [frames, rlFrames, analysis]
  );

  const phaseRows = useMemo(() => {
    if (!frames.length) return [];
    const stride = Math.max(1, Math.floor(frames.length / 250));
    return frames.filter((_, index) => index % stride === 0 || index === currentIndex);
  }, [frames, currentIndex]);
  const rlPhaseRows = useMemo(() => {
    if (!rlFrames.length) return [];
    const stride = Math.max(1, Math.floor(rlFrames.length / 250));
    return rlFrames.filter((_, index) => index % stride === 0 || index === Math.min(currentIndex, rlFrames.length - 1));
  }, [rlFrames, currentIndex]);

  const summaryCards = useMemo(() => {
    if (!latestSummary) return [];
    const taskName = latestSummary.task_name || activeTaskName;
    const meta = taskMeta(taskName);
    const cards = [
      {
        label: "Outcome",
        value: latestSummary.solved ? (meta.stageFamily === "worm" ? "reaches crawl goal" : meta.stageFamily === "pong" ? "holds the rally" : "balances task") : "failed run",
        note: `${latestSummary.cell_clean?.total_ticks ?? "n/a"} / ${latestSummary.cell_clean?.max_ticks ?? "n/a"} ticks`,
        tone: latestSummary.solved ? "ok" : "warn"
      },
      {
        label: latestSummary.cell_clean?.task_primary_label || meta.primaryLabel,
        value: formatNumber(latestSummary.cell_clean?.task_primary, 2),
        note: `success ${formatPct(latestSummary.cell_clean?.success_rate, 1)}`,
        tone: "info"
      },
      meta.stageFamily === "worm"
        ? {
            label: "Goal Progress",
            value: `${formatNumber(analysis?.goal_progress, 2)} m`,
            note: `mean speed ${formatSigned(analysis?.mean_forward_speed, 2, " m/s")}`,
            tone: "accent"
          }
        : meta.stageFamily === "pong"
          ? {
              label: "Return Count",
              value: `${formatNumber(analysis?.return_count, 1)}`,
              note: `tracking error ${formatNumber(analysis?.mean_tracking_error, 2)}`,
              tone: "accent"
            }
          : {
              label: "Peak Pole Error",
              value: `${formatNumber(analysis?.max_abs_theta_deg, 2)}°`,
              note: `RMS ${formatNumber(analysis?.rms_theta_deg, 2)}°`,
              tone: "accent"
            },
      {
        label: "Actuation",
        value: formatNumber(analysis?.mean_abs_total_force ?? analysis?.mean_abs_force, 2),
        note: `peak ${formatNumber(analysis?.max_abs_total_force ?? analysis?.max_abs_force, 2)}`,
        tone: "accent2"
      },
      {
        label: "Cell Energy",
        value: formatNumber(analysis?.mean_energy, 3),
        note: `min ${formatNumber(analysis?.min_energy, 3)} · stress ${formatNumber(analysis?.mean_stress, 3)}`,
        tone: "ok"
      },
      {
        label: "Damage Recovery",
        value: analysis?.damage_ticks?.length ? `tick ${analysis.damage_ticks[0]}` : "no damage",
        note: analysis?.recovery_tick_after_damage != null ? `recovered by tick ${analysis.recovery_tick_after_damage}` : "no recovery event",
        tone: analysis?.damage_ticks?.length ? "warn" : "default"
      }
    ];
    if (payload?.rl_summary || rlAnalysis) {
      cards.push({
        label: "RL Comparator",
        value: payload?.summary?.rl_algorithm_selected || form.rl_algo || "n/a",
        note: meta.stageFamily === "pong"
          ? `returns ${formatNumber(rlAnalysis?.return_count, 1)} · success ${payload?.rl_summary?.success_rate != null ? formatPct(payload?.rl_summary?.success_rate, 1) : "n/a"}`
          : meta.stageFamily === "worm"
            ? `progress ${formatNumber(rlAnalysis?.goal_progress, 2)} m · success ${payload?.rl_summary?.success_rate != null ? formatPct(payload?.rl_summary?.success_rate, 1) : "n/a"}`
            : `survival ${formatPct(payload?.rl_summary?.survival_ratio, 1)} · peak |theta| ${formatNumber(rlAnalysis?.max_abs_theta_deg, 2)}°`,
        tone: "info"
      });
    }
    return cards;
  }, [latestSummary, analysis, payload, rlAnalysis, form.rl_algo, activeTaskName]);


  return {
    frames,
    replayBodyCells,
    cellRows,
    edgeRows,
    rlFrames,
    analysis,
    rlAnalysis,
    genomePreviewItems,
    genomePreviewByPath,
    selectedGenomePreview,
    activeTaskName,
    activeTaskMeta,
    launchTaskFields,
    discoveryTaskFields,
    frame,
    rlFrame,
    latestTaskReport,
    latestSummary,
    activeSummary,
    activeOdd,
    activeEvidenceFiles,
    activeOddSourceLabel,
    benchmarkRuns,
    discoveryBatches,
    evidenceSources,
    selectedEvidenceSource,
    selectedEvidencePreview,
    missingCellArtifacts,
    bodyCells,
    selectedEvidenceBodyCells,
    bodyFollowsSelectedGenome,
    compareOptions,
    cellRowsByTick,
    edgeRowsByTick,
    currentFrameCellRows,
    currentFrameEdges,
    functionalStateByCellId,
    selectedBodyCell,
    includeDepthCoordinate,
    organismHasDepth,
    selectedCellState,
    selectedCellFunction,
    selectedCellEdges,
    atlasBodyView,
    genomeBodyView,
    matrixBodyView,
    graphBodyView,
    compareLeftBodyView,
    compareRightBodyView,
    compareItems,
    queuedCount,
    queuedPassCount,
    queuedFailCount,
    bodyInteractionProps,
    timelineRows,
    phaseRows,
    rlPhaseRows,
    summaryCards
  };
}
