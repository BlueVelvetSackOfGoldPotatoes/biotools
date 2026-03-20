import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  CELLENGINE_SECTIONS,
  DEFAULT_BODY_VIEW,
  formatNumber,
  formatPct,
  shortHash,
  taskLabel,
  TELEMETRY_QUALITY_META
} from "./core";
import { bodyDepthCount, bodyHasDepth, realizedBodyModeLabel } from "./body";
import { DiscoveryDevelopmentMiniMap } from "./genomeViews";
import { OrganismSurfacePreview, PopulationLandscapePreview } from "./surfacePreviews";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, SectionTitle } from "./uiPrimitives";
export function CellEngineSectionNav({ activeSection, onChange }) {
  return (
    <div className="card cellengine-section-shell">
      <div className="section-head">
        <SectionTitle
          title="CellEngine Sections"
          help="Hard-split internal workspace for discovery, replay, comparison, and evidence. Only the active section renders, so unrelated panels stop competing for state and space."
        />
        <span className="meta-note">URL-backed navigation: `?tab=cellengine&section=...`</span>
      </div>
      <div className="cellengine-section-nav" role="tablist" aria-label="CellEngine sections">
        {CELLENGINE_SECTIONS.map((section) => (
          <button
            key={section.id}
            type="button"
            role="tab"
            aria-selected={activeSection === section.id}
            className={`cellengine-section-btn${activeSection === section.id ? " active" : ""}`}
            onClick={() => onChange(section.id)}
          >
            <strong>{section.label}</strong>
            <span>{section.note}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function seedUnit(seed, salt = 0) {
  const raw = Math.sin((Number(seed) || 0) * 12.9898 + salt * 78.233) * 43758.5453;
  return raw - Math.floor(raw);
}

export function discoveryRunKey(run) {
  return `${Number(run?.run_index) || 0}:${Number(run?.seed) || 0}`;
}

export function discoveryBodyModeColor(bodyMode) {
  if (bodyMode === "grown3d") return "#d97706";
  if (bodyMode === "grown2d") return "#0f766e";
  return "#64748b";
}

export function discoveryCandidateSummary(candidate) {
  return candidate?.summary && typeof candidate.summary === "object" ? candidate.summary : null;
}

export function discoveryTelemetryQuality(value) {
  if (typeof value === "string" && TELEMETRY_QUALITY_META[value]) return value;
  return "none";
}

export function discoveryCandidateTelemetryQuality(candidate) {
  if (candidate?.summary_quality) return discoveryTelemetryQuality(candidate.summary_quality);
  if (discoveryCandidateSummary(candidate)) return "measured";
  if (Number.isFinite(Number(candidate?.fitness))) return "fitness_only";
  return "none";
}

export function discoveryRunTelemetryQuality(run, restoredBatch = false) {
  const summaryQuality = discoveryTelemetryQuality(run?.summary_quality);
  if (summaryQuality !== "none") return summaryQuality;
  const liveSummaryQuality = discoveryTelemetryQuality(run?.live_snapshot?.meta?.best_summary_quality);
  if (liveSummaryQuality !== "none") return restoredBatch && liveSummaryQuality === "measured" ? "restored" : liveSummaryQuality;
  if (run?.summary) return restoredBatch ? "restored" : "measured";
  if (run?.live_snapshot?.meta?.best_summary) return restoredBatch ? "restored" : "measured";
  if (run?.live_snapshot?.meta?.best_search_score != null) return "fitness_only";
  return "none";
}

export function TelemetryQualityBadge({ quality = "none" }) {
  const resolved = discoveryTelemetryQuality(quality);
  const meta = TELEMETRY_QUALITY_META[resolved];
  if (!meta) return null;
  return (
    <span
      className={`cellengine-telemetry-quality-badge ${meta.className}`}
      title={meta.note}
      aria-label={`telemetry quality: ${meta.label}`}
    >
      {meta.label}
    </span>
  );
}

export function compareDiscoveryCandidatesBySuccess(leftCandidate, rightCandidate) {
  const leftSummary = discoveryCandidateSummary(leftCandidate);
  const rightSummary = discoveryCandidateSummary(rightCandidate);
  const leftSuccess = Number(leftSummary?.success_rate);
  const rightSuccess = Number(rightSummary?.success_rate);
  if (Number.isFinite(leftSuccess) || Number.isFinite(rightSuccess)) {
    const delta = (Number.isFinite(rightSuccess) ? rightSuccess : -Infinity) - (Number.isFinite(leftSuccess) ? leftSuccess : -Infinity);
    if (Math.abs(delta) > 1e-9) return delta;
  }
  const leftSurvival = Number(leftSummary?.survival_ratio);
  const rightSurvival = Number(rightSummary?.survival_ratio);
  if (Number.isFinite(leftSurvival) || Number.isFinite(rightSurvival)) {
    const delta = (Number.isFinite(rightSurvival) ? rightSurvival : -Infinity) - (Number.isFinite(leftSurvival) ? leftSurvival : -Infinity);
    if (Math.abs(delta) > 1e-9) return delta;
  }
  const leftTask = Number(leftSummary?.task_primary);
  const rightTask = Number(rightSummary?.task_primary);
  if (Number.isFinite(leftTask) || Number.isFinite(rightTask)) {
    const delta = (Number.isFinite(rightTask) ? rightTask : -Infinity) - (Number.isFinite(leftTask) ? leftTask : -Infinity);
    if (Math.abs(delta) > 1e-9) return delta;
  }
  const leftFitness = Number(leftCandidate?.fitness);
  const rightFitness = Number(rightCandidate?.fitness);
  if (Number.isFinite(leftFitness) || Number.isFinite(rightFitness)) {
    const delta = (Number.isFinite(rightFitness) ? rightFitness : -Infinity) - (Number.isFinite(leftFitness) ? leftFitness : -Infinity);
    if (Math.abs(delta) > 1e-9) return delta;
  }
  const leftCells = Number(leftCandidate?.cell_count);
  const rightCells = Number(rightCandidate?.cell_count);
  if (Number.isFinite(leftCells) || Number.isFinite(rightCells)) {
    const delta = (Number.isFinite(rightCells) ? rightCells : -Infinity) - (Number.isFinite(leftCells) ? leftCells : -Infinity);
    if (Math.abs(delta) > 1e-9) return delta;
  }
  return (Number(leftCandidate?.rank) || 0) - (Number(rightCandidate?.rank) || 0);
}

export function discoveryCandidateTaskLabel(candidate, fallback = "task progress") {
  return discoveryCandidateSummary(candidate)?.task_primary_label || fallback;
}

export function DiscoveryEvolutionField({ job, result, taskName, historyByRun }) {
  const runs = Array.isArray(job?.runs) && job.runs.length ? job.runs : Array.isArray(result?.runs) ? result.runs : [];
  if (!runs.length) {
    return (
      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Search Trajectory"
            help="Generation-by-generation discovery progress for the active run. This shows success and survival instead of abstract search fitness."
          />
          <span className="meta-note">launch a discovery experiment to populate this view</span>
        </div>
        <div className="empty">No discovery batch active yet.</div>
      </div>
    );
  }
  const runRows = runs.map((run) => {
    const key = discoveryRunKey(run);
    const livePopulation = run?.live_population;
    const candidates = Array.isArray(livePopulation?.candidates) ? livePopulation.candidates : [];
    const generation = Number(livePopulation?.current_generation)
      ?? Number(run?.live_snapshot?.meta?.current_generation)
      ?? null;
    const syntheticHistory = run?.summary?.cell_clean && Number.isFinite(generation)
      ? [{
          generation,
          best_success_rate: Number(run.summary.cell_clean.success_rate),
          mean_success_rate: Number(run.summary.cell_clean.success_rate),
          mean_survival_ratio: Number(run.summary.cell_clean.survival_ratio),
          best_task_primary: Number(run.summary.cell_clean.task_primary),
          task_primary_label: run.summary.cell_clean.task_primary_label || null
        }]
      : [];
    const history = (historyByRun?.[key]?.rows || []).length ? (historyByRun?.[key]?.rows || []) : syntheticHistory;
    const latest = history[history.length - 1] || null;
    const liveMeta = run?.live_snapshot?.meta || {};
    const resolvedGeneration = latest?.generation
      ?? generation
      ?? Number(liveMeta?.current_generation)
      ?? null;
    const bodyMode = run?.summary?.body_mode || liveMeta.body_mode || run?.candidate_config?.body_mode || "fixed2d";
    const solved = Boolean(run?.summary?.solved);
    const bestSuccess = latest?.best_success_rate;
    const meanSuccess = latest?.mean_success_rate;
    const meanSurvival = latest?.mean_survival_ratio;
    const bestTaskPrimary = latest?.best_task_primary;
    const taskPrimaryLabel = latest?.task_primary_label
      || candidates.find((candidate) => discoveryCandidateSummary(candidate)?.task_primary_label)?.summary?.task_primary_label
      || run?.summary?.cell_clean?.task_primary_label
      || "task progress";
    return {
      key,
      run,
      history,
      generation: resolvedGeneration,
      bodyMode,
      solved,
      bestSuccess,
      meanSuccess,
      meanSurvival,
      bestTaskPrimary,
      taskPrimaryLabel
    };
  });
  const focusRow = runRows.find((row) => row.run?.status === "running" && row.history.length)
    || runRows.find((row) => row.history.length)
    || runRows[0];
  const focusHistory = focusRow?.history || [];

  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="Search Trajectory"
          help="Live GA trace for the active discovery run. Best and mean success, plus mean survival, show how much of the task the evolving population is actually solving."
        />
        <span className="meta-note">
          {taskLabel(taskName)} · {focusRow?.run?.seed != null ? `seed ${focusRow.run.seed}` : "waiting"} · {focusRow?.generation != null ? `gen ${focusRow.generation}` : "no generations yet"}
        </span>
      </div>
      {focusHistory.length ? (
        <div className="cellengine-trajectory-chart">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={focusHistory} margin={{ top: 12, right: 18, left: 8, bottom: 8 }}>
              <CartesianGrid stroke="#dce7ea" strokeDasharray="3 3" />
              <XAxis dataKey="generation" allowDecimals={false} tickLine={false} axisLine={{ stroke: "#b6c7cf" }} />
              <YAxis domain={[0, 1]} tickLine={false} axisLine={{ stroke: "#b6c7cf" }} width={64} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
              <Tooltip
                formatter={(value, name, row) => {
                  if (name === "best task progress") return formatNumber(value, 3);
                  return formatPct(value, 1);
                }}
                labelFormatter={(label) => `generation ${label}`}
              />
              <Legend />
              <Line type="monotone" dataKey="best_success_rate" name="best success" stroke="#0f766e" strokeWidth={3} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="mean_success_rate" name="mean success" stroke="#2563eb" strokeWidth={2.2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="mean_survival_ratio" name="mean survival" stroke="#c2410c" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="empty">Waiting for the first generation snapshot.</div>
      )}
      <div className="cellengine-trajectory-run-strip">
        {runRows.map((row) => {
          const statusClass = row.run?.status === "running"
            ? "running"
            : row.run?.status === "completed"
              ? row.solved ? "pass" : "fail"
              : "queued";
          return (
            <div key={row.key} className={`cellengine-trajectory-run-chip ${statusClass}`}>
              <div className="cellengine-trajectory-run-chip-head">
                <strong>seed {row.run?.seed}</strong>
                <span style={{ color: discoveryBodyModeColor(row.bodyMode) }}>{row.bodyMode}</span>
              </div>
              <div className="cellengine-trajectory-run-chip-meta">
                <span>{row.generation != null ? `gen ${row.generation}` : "waiting"}</span>
                <span>{Number.isFinite(row.bestSuccess) ? `best success ${formatPct(row.bestSuccess, 1)}` : "best success n/a"}</span>
                <span>{Number.isFinite(row.meanSurvival) ? `mean survival ${formatPct(row.meanSurvival, 1)}` : "mean survival n/a"}</span>
                <span>{row.taskPrimaryLabel} {Number.isFinite(row.bestTaskPrimary) ? formatNumber(row.bestTaskPrimary, 2) : "n/a"}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function DiscoveryPopulationGrid({ run, metricId, historyByRun }) {
  const population = run?.live_population;
  const candidates = Array.isArray(population?.candidates) ? population.candidates : [];
  const sortedCandidates = useMemo(
    () => [...candidates].sort(compareDiscoveryCandidatesBySuccess),
    [candidates]
  );
  const searchHistory = useMemo(() => {
    const key = discoveryRunKey(run);
    return key ? (historyByRun?.[key]?.rows || []) : [];
  }, [historyByRun, run]);
  const generation = population?.current_generation ?? run?.live_snapshot?.meta?.current_generation ?? null;

  if (!candidates.length) {
    return null;
  }

  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="Population Development"
          help="Current generation population for the running search seed. Every candidate is shown at once, each organism can animate its birth-step development trace on hover, and the visible order is sorted by success rate. Important: card colors and surface insets show the selected genome/body metric, not task success."
        />
        <div className="row controls">
          <span className="meta-note">
            seed {run?.seed ?? "n/a"} · gen {generation ?? "n/a"} · {sortedCandidates.length} candidates · sorted by success rate · hover any organism to animate development
          </span>
          <PopulationLandscapePreview
            candidates={sortedCandidates}
            title={`seed ${run?.seed ?? "n/a"} population landscape`}
            note="whole-population shape vs success surface"
          />
        </div>
      </div>
      <div className="meta-note">
        Similar-looking organisms can still perform very differently. These cards are showing morphology plus one selected genome/body metric; success comes from the rollout dynamics, which also depend on the other genome programs, coupling, stress state, and task interaction.
      </div>
      <div className="cellengine-population-grid">
        {sortedCandidates.map((candidate, index) => {
          const bodyCells = Array.isArray(candidate?.body_cells) ? candidate.body_cells : [];
          const cellCount = Number(candidate?.cell_count);
          const realizedBodyLabel = realizedBodyModeLabel(candidate.body_mode, bodyCells);
          const summary = discoveryCandidateSummary(candidate);
          const hasSuccess = Number.isFinite(Number(summary?.success_rate));
          const hasSurvival = Number.isFinite(Number(summary?.survival_ratio));
          const hasTask = Number.isFinite(Number(summary?.task_primary));
          const fallbackFitness = Number(candidate?.fitness);
          const telemetryQuality = discoveryCandidateTelemetryQuality(candidate);
          const displayRank = index + 1;
          return (
            <div key={`candidate-${candidate.rank}-${candidate.candidate_index}`} className={`cellengine-population-card${index === 0 ? " best" : ""}`}>
              <div className="cellengine-population-card-head">
                <strong>#{displayRank}</strong>
                <span>{realizedBodyLabel}</span>
                <TelemetryQualityBadge quality={telemetryQuality} />
              </div>
              <div className="cellengine-organism-card-visual">
                <DiscoveryDevelopmentMiniMap
                  bodyCells={bodyCells}
                  bodyView={DEFAULT_BODY_VIEW}
                  metricId={metricId}
                  footer={false}
                  compact
                  animateOnHover
                />
                <OrganismSurfacePreview
                  bodyCells={bodyCells}
                  metricId={metricId}
                  title={`candidate #${displayRank} spatial field`}
                  note="current generation spatial body field for the selected metric"
                  summary={summary}
                  searchHistory={searchHistory}
                />
              </div>
              <div className="cellengine-population-card-meta">
                <span>{hasSuccess ? `success ${formatPct(summary.success_rate, 1)}` : Number.isFinite(fallbackFitness) ? "success unavailable" : "success n/a"}</span>
                <span>{hasSurvival ? `survival ${formatPct(summary.survival_ratio, 1)}` : Number.isFinite(fallbackFitness) ? "survival unavailable" : "survival n/a"}</span>
                <span>
                  {hasTask
                    ? `${summary.task_primary_label || "task"} ${formatNumber(summary.task_primary, 2)}`
                    : Number.isFinite(fallbackFitness)
                      ? `search fitness ${formatNumber(fallbackFitness, 2)}`
                      : "task n/a"}
                </span>
                <span>{Number.isFinite(cellCount) ? `${cellCount} cells` : "cells n/a"}</span>
                <span>{bodyHasDepth(bodyCells) ? `depth ${bodyDepthCount(bodyCells)}` : "flat body"}</span>
                {Number.isFinite(Number(candidate?.rank)) ? <span>fitness rank {candidate.rank}</span> : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function BenchmarkRunsLibrary({ runs, selectedTaskName, onUseGenome, onSelectEvidence, onForegroundRun }) {
  const filteredRuns = (runs || []).filter((run) => (run.task_name || "cartpole_balance") === selectedTaskName);
  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="Benchmark Runs"
          help="Completed benchmark outputs discovered on disk. These are persisted artifacts, not just the currently loaded replay."
        />
        <span className="meta-note">{filteredRuns.length} visible for {taskLabel(selectedTaskName)}</span>
      </div>
      {filteredRuns.length ? (
        <div className="cellengine-artifact-list">
          {filteredRuns.map((run) => (
            <div key={run.relative_output_dir} className="cellengine-artifact-card">
              <div className="cellengine-artifact-head">
                <strong>{run.relative_output_dir.replace(/^reports\//, "")}</strong>
                <span className={`cellengine-discovery-run-badge ${run.summary?.solved ? "pass" : "fail"}`}>
                  {run.summary?.solved ? "PASS" : "FAIL"}
                </span>
              </div>
              <div className="cellengine-artifact-meta">
                <span>{taskLabel(run.task_name || "cartpole_balance")}</span>
                <span>{run.summary?.body_mode || "n/a"}</span>
                <span>{run.summary?.champion_cell_count != null ? `${run.summary.champion_cell_count} cells` : "n/a"}</span>
                <span>{run.summary?.rl_algorithm_selected ? `RL ${String(run.summary.rl_algorithm_selected).toUpperCase()}` : "RL n/a"}</span>
              </div>
              <div className="cellengine-artifact-stats">
                <span>survival {formatPct(run.summary?.cell_clean?.survival_ratio, 1)}</span>
                <span>success {formatPct(run.summary?.cell_clean?.success_rate, 1)}</span>
                <span>{run.summary?.cell_clean?.task_primary_label || "metric"} {formatNumber(run.summary?.cell_clean?.task_primary, 2)}</span>
              </div>
              <ActionGroups className="cellengine-card-actions">
                <ActionGroup title="Run">
                  <ActionButton
                    type="button"
                    className="refresh-btn"
                    onClick={() => onForegroundRun?.(run)}
                    help={run.files?.replay_trace_csv
                      ? "Open this saved benchmark run in the Replay section and load its saved trace."
                      : "Open this saved benchmark run in the Reports section because it has no replay trace bundle."}
                  >
                    {run.files?.replay_trace_csv ? "Open saved replay" : "Open saved result"}
                  </ActionButton>
                </ActionGroup>
                <ActionGroup title="Select">
                  <ActionButton
                    type="button"
                    className="mini-btn"
                    disabled={!run.champion_genome_path}
                    onClick={() => onUseGenome(run.champion_genome_path || "")}
                    help="Select this run's champion genome as the next replay input. This does not launch anything yet."
                  >
                    Set champion as replay input
                  </ActionButton>
                </ActionGroup>
                <ActionGroup title="Inspect">
                  <ActionButton
                    type="button"
                    className="mini-btn"
                    onClick={() => onSelectEvidence(`run:${run.relative_output_dir}`)}
                    help="Open this run in the Reports / ODD section and make it the active evidence source."
                  >
                    Inspect in Reports
                  </ActionButton>
                  {run.files?.report_md ? (
                    <ActionLink className="mini-btn" href={run.files.report_md} target="_blank" rel="noreferrer" help="Open the saved markdown report for this benchmark run.">
                      Open report file
                    </ActionLink>
                  ) : null}
                  {run.files?.summary_json ? (
                    <ActionLink className="mini-btn" href={run.files.summary_json} target="_blank" rel="noreferrer" help="Open the machine-readable summary JSON for this benchmark run.">
                      Open summary file
                    </ActionLink>
                  ) : null}
                </ActionGroup>
              </ActionGroups>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty">No persisted benchmark runs found for this task yet.</div>
      )}
    </div>
  );
}

export function DiscoveryBatchLibrary({ batches, selectedTaskName, onUseGenome, onSelectEvidence, onForegroundBatch }) {
  const filteredBatches = (batches || []).filter((batch) => (batch.task_name || "cartpole_balance") === selectedTaskName);
  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="Recent Discovery Batches"
          help="Saved discovery batches with aggregate distributions and the best champion from each batch."
        />
        <span className="meta-note">{filteredBatches.length} visible for {taskLabel(selectedTaskName)}</span>
      </div>
      {filteredBatches.length ? (
        <div className="cellengine-artifact-list">
          {filteredBatches.map((batch) => (
            <div key={batch.relative_output_dir} className="cellengine-artifact-card">
              <div className="cellengine-artifact-head">
                <strong>{batch.batch_id}</strong>
                <span className="cellengine-discovery-run-badge running">
                  {batch.aggregate?.solved_runs ?? 0}/{batch.aggregate?.completed_runs ?? batch.config?.num_runs ?? 0} solved
                </span>
              </div>
              <div className="cellengine-artifact-meta">
                <span>{taskLabel(batch.task_name || "cartpole_balance")}</span>
                <span>{batch.config?.num_runs ?? batch.aggregate?.completed_runs ?? 0} runs</span>
                <span>{batch.aggregate?.champion_params?.body_mode_counts ? Object.keys(batch.aggregate.champion_params.body_mode_counts).join(" / ") : "body mix n/a"}</span>
              </div>
              <div className="cellengine-artifact-stats">
                <span>mean survival {formatPct(batch.aggregate?.mean_survival_ratio, 1)}</span>
                <span>mean success {formatPct(batch.aggregate?.mean_success_rate, 1)}</span>
                <span>{batch.aggregate?.task_primary_label || "metric"} {formatNumber(batch.aggregate?.mean_task_primary, 2)}</span>
              </div>
              <div className="meta-note">
                best champion {batch.best_run?.summary?.solved ? "PASS" : "FAIL"} · {batch.best_run?.summary?.body_mode || "n/a"} · {batch.best_run?.summary?.champion_cell_count ?? "n/a"} cells
              </div>
              <ActionGroups className="cellengine-card-actions">
                <ActionGroup title="Run">
                  <ActionButton
                    type="button"
                    className="refresh-btn"
                    onClick={() => onForegroundBatch?.(batch)}
                    help="Restore this saved discovery batch into the Discover section, including the per-seed organism views and saved population snapshots."
                  >
                    Open in Discover
                  </ActionButton>
                </ActionGroup>
                <ActionGroup title="Select">
                  <ActionButton
                    type="button"
                    className="mini-btn"
                    disabled={!batch.best_run?.champion_genome_path}
                    onClick={() => onUseGenome(batch.best_run?.champion_genome_path || "")}
                    help="Select the best champion from this batch as the next replay input. This does not launch anything yet."
                  >
                    Set best genome as replay input
                  </ActionButton>
                </ActionGroup>
                <ActionGroup title="Inspect">
                  <ActionButton
                    type="button"
                    className="mini-btn"
                    onClick={() => onSelectEvidence(`batch:${batch.relative_output_dir}`)}
                    help="Open this saved discovery batch in the Reports / ODD section."
                  >
                    Inspect in Reports
                  </ActionButton>
                  {batch.best_run?.files?.report_md ? (
                    <ActionLink className="mini-btn" href={batch.best_run.files.report_md} target="_blank" rel="noreferrer" help="Open the best run's saved markdown report from this discovery batch.">
                      Open best report
                    </ActionLink>
                  ) : null}
                  {batch.files?.discovery_summary_json ? (
                    <ActionLink className="mini-btn" href={batch.files.discovery_summary_json} target="_blank" rel="noreferrer" help="Open the batch-level discovery summary JSON with aggregate distributions.">
                      Open batch JSON
                    </ActionLink>
                  ) : null}
                </ActionGroup>
              </ActionGroups>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty">No saved discovery batches found for this task yet.</div>
      )}
    </div>
  );
}
