import { useState } from "react";
import { DEFAULT_BODY_VIEW, formatNumber, formatPct, taskLabel } from "../core";
import { bodyDepthCount, bodyHasDepth, realizedBodyModeLabel } from "../body";
import { SharedVisualizationLegend } from "../legends";
import { DiscoveryDevelopmentMiniMap } from "../genomeViews";
import {
  DiscoveryBatchLibrary,
  DiscoveryEvolutionField,
  DiscoveryPopulationGrid,
  TelemetryQualityBadge,
  discoveryRunKey,
  discoveryRunTelemetryQuality
} from "../discoveryViews";
import { OrganismSurfacePreview } from "../surfacePreviews";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, DiscoverableNumberField, DiscoverableSelectField, DiscoveryDropdownSection, FieldLabel, SectionTitle } from "../uiPrimitives";
import { DiscoveryProgressPanel } from "../DiscoveryProgressPanel";
import { DiscoveryReplayViewer } from "../DiscoveryReplayViewer";

export function DiscoverSection({ vm, actions }) {
  const {
    selectedTaskName,
    discoveryBusy,
    discoveryRunning,
    discoveryForm,
    setDiscoveryForm,
    morphologyPresetActive,
    allOrganismParamsDiscoverableActive,
    structureFrozenPresetActive,
    discoveryTaskFields,
    discoveryBatches,
    discoveryJob,
    discoveryResult,
    discoveryHistoryByRun,
    genomeMetric,
    setForm,
    setSelectedEvidenceKey,
    setActiveSection,
    restoredDiscoveryBatch,
    restoredDiscoveryNeedsTelemetryFallback,
    showLiveDiscovery,
    busyStartedAt
  } = vm;
  const {
    launchDiscovery,
    launchSingleFullBenchmarkDiscovery,
    foregroundDiscoveryBatch
  } = actions;

  const [replayTarget, setReplayTarget] = useState(null);

  return (
    <>
        <div className="cellengine-discover-shell">
          <aside className="cellengine-discover-sidebar">
            <div className="card">
              <div className="section-head">
                <SectionTitle
                  title="Discover Genomes"
                  help="Runs the actual evolutionary search from scratch one or more times. This generates new champion genomes under reports/, which then appear in the gallery and compare tools."
                />
                <span className="meta-note">active task: {taskLabel(selectedTaskName)} · separate task from replay: this discovers new genomes instead of re-evaluating an existing one</span>
              </div>
              <div className="cellengine-launchbar">
                <button className="refresh-btn cellengine-run-btn" type="button" disabled={discoveryBusy} onClick={launchDiscovery}>
                  {discoveryBusy ? "Running discovery..." : "Run discovery experiment"}
                </button>
                <button
                  className="refresh-btn cellengine-run-btn secondary"
                  type="button"
                  disabled={discoveryBusy}
                  onClick={launchSingleFullBenchmarkDiscovery}
                >
                  {discoveryBusy ? "Running benchmark..." : "Run one full benchmark from scratch"}
                </button>
                <div className="cellengine-library-preview">
                  <strong>current search plan</strong>
                  <span>{discoveryForm.num_runs} runs · pop {discoveryForm.population_size} · gens {discoveryForm.generations} · task {taskLabel(selectedTaskName)}</span>
                  <span>body {discoveryForm.body_mode} · max cells {discoveryForm.max_cells} · RL {String(discoveryForm.rl_algo || "none").toUpperCase()}</span>
                </div>
              </div>

              <div className="cellengine-discovery-dropdown-stack">
                <DiscoveryDropdownSection
                  title="Search Budget"
                  help="Batch size and GA budget for the discovery run."
                  summaryNote={`${discoveryForm.num_runs} runs · pop ${discoveryForm.population_size} · gens ${discoveryForm.generations}`}
                  defaultOpen
                >
                  <div className="cellengine-form-grid">
                    <label>
                      <FieldLabel label="Runs" help="How many independent evolutionary searches to launch in this batch. Multiple runs give you a genome distribution instead of a single champion." />
                      <input className="task-input" type="number" min="1" max="24" value={discoveryForm.num_runs} onChange={(event) => setDiscoveryForm((current) => ({ ...current, num_runs: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Seed start" help="Base random seed for the first search run." />
                      <input className="task-input" type="number" min="1" value={discoveryForm.seed_start} onChange={(event) => setDiscoveryForm((current) => ({ ...current, seed_start: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Seed step" help="Increment between search seeds in the batch." />
                      <input className="task-input" type="number" min="1" value={discoveryForm.seed_step} onChange={(event) => setDiscoveryForm((current) => ({ ...current, seed_step: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Population" help="GA population size for each search." />
                      <input className="task-input" type="number" min="8" max="512" value={discoveryForm.population_size} onChange={(event) => setDiscoveryForm((current) => ({ ...current, population_size: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Generations" help="GA generations per search." />
                      <input className="task-input" type="number" min="1" max="1000" value={discoveryForm.generations} onChange={(event) => setDiscoveryForm((current) => ({ ...current, generations: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Auto attempts" help="If a run does not solve the task, the benchmark can automatically escalate the search budget up to this many attempts." />
                      <input className="task-input" type="number" min="1" max="16" value={discoveryForm.auto_attempts} onChange={(event) => setDiscoveryForm((current) => ({ ...current, auto_attempts: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Search trials" help="Trial count used during GA fitness search." />
                      <input className="task-input" type="number" min="1" value={discoveryForm.search_trials} onChange={(event) => setDiscoveryForm((current) => ({ ...current, search_trials: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Search ticks" help="Max ticks per trial during GA fitness search." />
                      <input className="task-input" type="number" min="20" value={discoveryForm.search_ticks} onChange={(event) => setDiscoveryForm((current) => ({ ...current, search_ticks: event.target.value }))} />
                    </label>
                  </div>
                </DiscoveryDropdownSection>

                <DiscoveryDropdownSection
                  title="Evaluation Budget"
                  help="Full validation budget for each discovered champion."
                  summaryNote={`${discoveryForm.final_trials} trials · ${discoveryForm.final_ticks} ticks`}
                >
                  <div className="cellengine-form-grid">
                    <label>
                      <FieldLabel label="Final trials" help="Full evaluation trials for each discovered champion." />
                      <input className="task-input" type="number" min="1" value={discoveryForm.final_trials} onChange={(event) => setDiscoveryForm((current) => ({ ...current, final_trials: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="Final ticks" help="Max ticks per trial during final champion evaluation." />
                      <input className="task-input" type="number" min="20" value={discoveryForm.final_ticks} onChange={(event) => setDiscoveryForm((current) => ({ ...current, final_ticks: event.target.value }))} />
                    </label>
                  </div>
                </DiscoveryDropdownSection>

                <DiscoveryDropdownSection
                  title="Organism Structure"
                  help="Morphology and developmental constraints for the discovered organism."
                  summaryNote={`${discoveryForm.body_mode} · max ${discoveryForm.max_cells} cells`}
                >
                  <div className="cellengine-form-grid">
                    <DiscoverableSelectField
                      label="Body mode"
                      help="Base morphology regime for the organism. Keep this fixed to force a regime, or mark it discoverable so evolution can choose between fixed2d, grown2d, and grown3d."
                      value={discoveryForm.body_mode}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, body_mode: event.target.value }))}
                      discoverable={discoveryForm.evolve_body_mode}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_body_mode: event.target.checked }))}
                      options={[
                        { value: "fixed2d", label: "fixed2d" },
                        { value: "grown2d", label: "grown2d" },
                        { value: "grown3d", label: "grown3d" }
                      ]}
                    />
                    <DiscoverableNumberField
                      label="Max cells"
                      help="Upper bound on organism population. If discoverable, evolution can change the actual number of cells it grows toward."
                      value={discoveryForm.max_cells}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, max_cells: event.target.value }))}
                      discoverable={discoveryForm.evolve_max_cells}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_max_cells: event.target.checked }))}
                      min="18"
                      max="512"
                    />
                    <DiscoverableNumberField
                      label="Development steps"
                      help="How many growth/development iterations the organism gets before evaluation starts."
                      value={discoveryForm.development_steps}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, development_steps: event.target.value }))}
                      discoverable={discoveryForm.evolve_development_steps}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_development_steps: event.target.checked }))}
                      min="1"
                      max="64"
                    />
                    <DiscoverableNumberField
                      label="Seed half-width"
                      help="Initial seed width for development before growth expands the body."
                      value={discoveryForm.development_seed_half_width}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, development_seed_half_width: event.target.value }))}
                      discoverable={discoveryForm.evolve_development_seed_half_width}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_development_seed_half_width: event.target.checked }))}
                      min="1"
                      max="16"
                    />
                    <DiscoverableNumberField
                      label="Body extent X"
                      help="Allowed body growth span on the x-axis."
                      value={discoveryForm.body_extent_x}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, body_extent_x: event.target.value }))}
                      discoverable={discoveryForm.evolve_body_extent_x}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_body_extent_x: event.target.checked }))}
                      min="2"
                      max="64"
                    />
                    <DiscoverableNumberField
                      label="Body extent Y"
                      help="Allowed body growth span on the y-axis."
                      value={discoveryForm.body_extent_y}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, body_extent_y: event.target.value }))}
                      discoverable={discoveryForm.evolve_body_extent_y}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_body_extent_y: event.target.checked }))}
                      min="4"
                      max="64"
                    />
                    <DiscoverableNumberField
                      label="Body extent Z"
                      help="Allowed body growth span on the z-axis. Set to zero for strictly flat bodies."
                      value={discoveryForm.body_extent_z}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, body_extent_z: event.target.value }))}
                      discoverable={discoveryForm.evolve_body_extent_z}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_body_extent_z: event.target.checked }))}
                      min="0"
                      max="16"
                    />
                  </div>
                </DiscoveryDropdownSection>

                <DiscoveryDropdownSection
                  title="Growth And Field"
                  help="Development gating and spatial chemical diffusion settings."
                  summaryNote={`diff ${discoveryForm.chemical_diffusion_rate} · decay ${discoveryForm.chemical_decay}`}
                >
                  <div className="cellengine-form-grid">
                    <DiscoverableNumberField
                      label="Chemical diffusion steps"
                      help="How many lattice diffusion iterations run per physics step."
                      value={discoveryForm.chemical_diffusion_steps}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, chemical_diffusion_steps: event.target.value }))}
                      discoverable={discoveryForm.evolve_chemical_diffusion_steps}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_chemical_diffusion_steps: event.target.checked }))}
                      min="1"
                      max="16"
                    />
                    <DiscoverableNumberField
                      label="Growth threshold"
                      help="Development threshold that gates whether new cells get added during growth."
                      value={discoveryForm.development_growth_threshold}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, development_growth_threshold: event.target.value }))}
                      discoverable={discoveryForm.evolve_growth_threshold}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_growth_threshold: event.target.checked }))}
                      min="0.15"
                      max="0.90"
                      step="0.01"
                    />
                    <DiscoverableNumberField
                      label="Chem diffusion rate"
                      help="Spatial chemical field diffusion rate."
                      value={discoveryForm.chemical_diffusion_rate}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, chemical_diffusion_rate: event.target.value }))}
                      discoverable={discoveryForm.evolve_chemical_diffusion_rate}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_chemical_diffusion_rate: event.target.checked }))}
                      min="0.05"
                      max="0.60"
                      step="0.01"
                    />
                    <DiscoverableNumberField
                      label="Chem decay"
                      help="Chemical field decay rate per diffusion cycle."
                      value={discoveryForm.chemical_decay}
                      onChange={(event) => setDiscoveryForm((current) => ({ ...current, chemical_decay: event.target.value }))}
                      discoverable={discoveryForm.evolve_chemical_decay}
                      onToggleDiscoverable={(event) => setDiscoveryForm((current) => ({ ...current, evolve_chemical_decay: event.target.checked }))}
                      min="0"
                      max="0.25"
                      step="0.01"
                    />
                  </div>
                </DiscoveryDropdownSection>

                {discoveryTaskFields ? (
                  <DiscoveryDropdownSection
                    title="Task Controls"
                    help="Task-specific discovery difficulty controls."
                    summaryNote={taskLabel(selectedTaskName)}
                  >
                    <div className="cellengine-form-grid">
                      {discoveryTaskFields?.primaryLabel ? (
                        <label>
                          <FieldLabel label={discoveryTaskFields.primaryLabel} help={discoveryTaskFields.primaryHelp} />
                          <input
                            className="task-input"
                            type="number"
                            step="0.1"
                            value={selectedTaskName === "worm_drag_race" ? discoveryForm.worm_goal_distance : discoveryForm.pong_target_hits}
                            placeholder={discoveryTaskFields.primaryPlaceholder}
                            onChange={(event) => setDiscoveryForm((current) => (
                              selectedTaskName === "worm_drag_race"
                                ? { ...current, worm_goal_distance: event.target.value }
                                : { ...current, pong_target_hits: event.target.value }
                            ))}
                          />
                        </label>
                      ) : null}
                      {discoveryTaskFields?.secondaryLabel ? (
                        <label>
                          <FieldLabel label={discoveryTaskFields.secondaryLabel} help={discoveryTaskFields.secondaryHelp} />
                          <input
                            className="task-input"
                            type="number"
                            step="0.1"
                            value={selectedTaskName === "worm_drag_race" ? discoveryForm.worm_max_backward : discoveryForm.pong_ball_speed}
                            placeholder={discoveryTaskFields.secondaryPlaceholder}
                            onChange={(event) => setDiscoveryForm((current) => (
                              selectedTaskName === "worm_drag_race"
                                ? { ...current, worm_max_backward: event.target.value }
                                : { ...current, pong_ball_speed: event.target.value }
                            ))}
                          />
                        </label>
                      ) : null}
                      {discoveryTaskFields?.tertiaryLabel ? (
                        <label>
                          <FieldLabel label={discoveryTaskFields.tertiaryLabel} help={discoveryTaskFields.tertiaryHelp} />
                          <input
                            className="task-input"
                            type="number"
                            step="0.01"
                            value={discoveryForm.pong_paddle_half_height}
                            placeholder={discoveryTaskFields.tertiaryPlaceholder}
                            onChange={(event) => setDiscoveryForm((current) => ({ ...current, pong_paddle_half_height: event.target.value }))}
                          />
                        </label>
                      ) : null}
                    </div>
                  </DiscoveryDropdownSection>
                ) : null}

                <DiscoveryDropdownSection
                  title="RL And ODD"
                  help="Optional comparator and ODD analysis settings for discovered champions."
                  summaryNote={`RL ${String(discoveryForm.rl_algo || "none").toUpperCase()} · ODD ${discoveryForm.odd_enabled ? "on" : "off"}`}
                >
                  <div className="cellengine-form-grid">
                    <label>
                      <FieldLabel label="RL compare" help="`disabled` runs cell-only discovery. Enable RL only if you want each discovered champion benchmarked against RL during the search batch." />
                      <select className="task-input" value={discoveryForm.rl_algo} onChange={(event) => setDiscoveryForm((current) => ({ ...current, rl_algo: event.target.value }))}>
                        <option value="none">disabled</option>
                        <option value="a2c">A2C</option>
                        <option value="dqn">DQN</option>
                        <option value="all">all</option>
                      </select>
                    </label>
                    <label className="cellengine-toggle">
                      <input type="checkbox" checked={discoveryForm.odd_enabled} onChange={(event) => setDiscoveryForm((current) => ({ ...current, odd_enabled: event.target.checked }))} />
                      <span>run ODD after each champion</span>
                    </label>
                    <label>
                      <FieldLabel label="ODD trials" help="Only used when ODD is enabled." />
                      <input className="task-input" type="number" min="1" value={discoveryForm.odd_trials} onChange={(event) => setDiscoveryForm((current) => ({ ...current, odd_trials: event.target.value }))} />
                    </label>
                    <label>
                      <FieldLabel label="ODD ticks" help="Only used when ODD is enabled." />
                      <input className="task-input" type="number" min="20" value={discoveryForm.odd_ticks} onChange={(event) => setDiscoveryForm((current) => ({ ...current, odd_ticks: event.target.value }))} />
                    </label>
                  </div>
                </DiscoveryDropdownSection>
              </div>

              <ActionGroups className="cellengine-actions-toolbar">
                <ActionGroup title="Run">
                  <ActionButton
                    className="mini-btn"
                    type="button"
                    disabled={discoveryBusy}
                    onClick={launchDiscovery}
                    help="Launches one or more evolutionary searches to discover new genomes. This is the search workflow, not just a replay of an existing genome."
                  >
                    Run search
                  </ActionButton>
                  <ActionButton
                    className="mini-btn"
                    type="button"
                    disabled={discoveryBusy}
                    onClick={launchSingleFullBenchmarkDiscovery}
                    help="Runs one full end-to-end benchmark from scratch for the current task: discover a champion, do final evaluation, RL comparison, and ODD reporting."
                  >
                    Run full benchmark from scratch
                  </ActionButton>
                </ActionGroup>
                <ActionGroup title="Presets">
                  <ActionButton
                    className={`mini-btn${morphologyPresetActive ? " active" : ""}`}
                    type="button"
                    aria-pressed={morphologyPresetActive}
                    disabled={discoveryBusy}
                    help="Preset for sampling a broader genome distribution quickly: more runs, RL off during discovery, ODD on. This changes the form values but does not launch anything by itself."
                    onClick={() =>
                      setDiscoveryForm((current) => ({
                        ...current,
                        num_runs: 8,
                        rl_algo: "none",
                        odd_enabled: true
                      }))
                    }
                  >
                    Apply distribution preset
                  </ActionButton>
                  <ActionButton
                    className="mini-btn"
                    type="button"
                    disabled={discoveryBusy}
                    help="Turns on discoverability for the core morphology fields only: body mode, max cells, development steps, seed width, and body extents. Growth/field chemistry stays fixed."
                    onClick={() =>
                      setDiscoveryForm((current) => ({
                        ...current,
                        evolve_body_mode: true,
                        evolve_max_cells: true,
                        evolve_development_steps: true,
                        evolve_development_seed_half_width: true,
                        evolve_body_extent_x: true,
                        evolve_body_extent_y: true,
                        evolve_body_extent_z: true
                      }))
                    }
                  >
                    Enable morphology search
                  </ActionButton>
                  <ActionButton
                    className={`mini-btn${allOrganismParamsDiscoverableActive ? " active" : ""}`}
                    type="button"
                    aria-pressed={allOrganismParamsDiscoverableActive}
                    disabled={discoveryBusy}
                    help="Turns on discoverability for all organism-level parameters that currently make sense to evolve: morphology plus diffusion and growth-field parameters."
                    onClick={() =>
                      setDiscoveryForm((current) => ({
                        ...current,
                        evolve_body_mode: true,
                        evolve_max_cells: true,
                        evolve_development_steps: true,
                        evolve_development_seed_half_width: true,
                        evolve_body_extent_x: true,
                        evolve_body_extent_y: true,
                        evolve_body_extent_z: true,
                        evolve_chemical_diffusion_steps: true,
                        evolve_growth_threshold: true,
                        evolve_chemical_diffusion_rate: true,
                        evolve_chemical_decay: true
                      }))
                    }
                  >
                    Enable all organism search
                  </ActionButton>
                  <ActionButton
                    className={`mini-btn${structureFrozenPresetActive ? " active" : ""}`}
                    type="button"
                    aria-pressed={structureFrozenPresetActive}
                    disabled={discoveryBusy}
                    help="Freezes all organism-structure and growth-field parameters so evolution only searches over the genome values, not morphology or developmental settings."
                    onClick={() =>
                      setDiscoveryForm((current) => ({
                        ...current,
                        evolve_body_mode: false,
                        evolve_max_cells: false,
                        evolve_development_steps: false,
                        evolve_development_seed_half_width: false,
                        evolve_body_extent_x: false,
                        evolve_body_extent_y: false,
                        evolve_body_extent_z: false,
                        evolve_chemical_diffusion_steps: false,
                        evolve_growth_threshold: false,
                        evolve_chemical_diffusion_rate: false,
                        evolve_chemical_decay: false
                      }))
                    }
                  >
                    Freeze organism structure
                  </ActionButton>
                  <ActionButton
                    className="mini-btn"
                    type="button"
                    disabled={discoveryBusy}
                    help="Applies a short smoke-test search budget for quick end-to-end validation. This only changes the form values."
                    onClick={() =>
                      setDiscoveryForm((current) => ({
                        ...current,
                        num_runs: 2,
                        population_size: 24,
                        generations: 16,
                        search_trials: 8,
                        search_ticks: 120,
                        final_trials: 20,
                        final_ticks: 200,
                        odd_trials: 8,
                        odd_ticks: 200
                      }))
                    }
                  >
                    Apply smoke preset
                  </ActionButton>
                </ActionGroup>
              </ActionGroups>

              {discoveryRunning ? (
                <div className="cellengine-status-note" role="status" aria-live="polite">
                  Running fresh evolutionary searches and streaming partial results as each seed finishes.
                  <strong> elapsed {formatNumber(Math.max(0, (Date.now() - Date.parse(discoveryJob?.started_utc || discoveryJob?.created_utc || new Date().toISOString())) / 1000), 0)}s</strong>
                </div>
              ) : null}
            </div>

            <DiscoveryBatchLibrary
              batches={discoveryBatches}
              selectedTaskName={selectedTaskName}
              onUseGenome={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
              onSelectEvidence={(key) => {
                setSelectedEvidenceKey(key);
                setActiveSection("reports");
              }}
              onForegroundBatch={foregroundDiscoveryBatch}
            />
          </aside>

          <div className="cellengine-discover-main">
            {discoveryRunning ? (
              <DiscoveryProgressPanel
                discoveryJob={discoveryJob}
                discoveryForm={discoveryForm}
                discoveryHistoryByRun={discoveryHistoryByRun}
                selectedTaskName={selectedTaskName}
                busyStartedAt={busyStartedAt}
              />
            ) : null}
            <DiscoveryEvolutionField job={discoveryJob} result={discoveryResult} taskName={selectedTaskName} historyByRun={discoveryHistoryByRun} />
            <SharedVisualizationLegend
              title="Discovery Visual Legend"
              note="shared once for population-development and live-discovery organism previews"
              genomeMetric={genomeMetric}
              showGenome
              showRoles
              showDevelopment
            />
            <DiscoveryPopulationGrid
              run={(discoveryJob?.runs || []).find((run) => run?.status === "running" && run?.live_population?.candidates?.length)
                || (discoveryJob?.runs || []).find((run) => run?.live_population?.candidates?.length)
                || null}
              metricId={genomeMetric}
              historyByRun={discoveryHistoryByRun}
            />
            {showLiveDiscovery ? (
              <div className="cellengine-discovery-summary">
              <div className="section-head">
                <SectionTitle
                  title="Live Discovery"
                  help="Queued, running, and completed search runs for the current discovery batch. This updates while the backend is still evolving genomes."
                />
                <span className="meta-note">
                  {discoveryJob.status} · {discoveryJob.completed_runs || 0} / {discoveryJob.total_runs || 0} complete
                  {Number.isInteger(discoveryJob.current_run_index) ? ` · running seed ${discoveryJob.current_seed}` : ""}
                </span>
              </div>
              {restoredDiscoveryBatch && restoredDiscoveryNeedsTelemetryFallback ? (
                <div className="meta-note">
                  Restored legacy discovery batch: these saved population snapshots predate full per-candidate task telemetry.
                  Best-run cards show restored run metrics; older population candidates fall back to `search fitness` where `success` and `survival` were never written.
                </div>
              ) : null}
              <div className="cellengine-discovery-job-grid">
                {(discoveryJob.runs || []).map((run) => {
                  const solved = run.summary?.solved;
                  const liveBodyCells = run.live_snapshot?.body_cells || [];
                  const liveMeta = run.live_snapshot?.meta || {};
                  const runSearchHistory = discoveryHistoryByRun?.[discoveryRunKey(run)]?.rows || [];
                  const telemetryQuality = discoveryRunTelemetryQuality(run, restoredDiscoveryBatch);
                  const realizedBodyLabel = realizedBodyModeLabel(
                    run.summary?.body_mode || liveMeta.body_mode || run.candidate_config?.body_mode || "fixed2d",
                    liveBodyCells
                  );
                  return (
                    <div
                      key={`discover-run-${run.run_index}-${run.seed}`}
                      className={`cellengine-discovery-run-card ${run.status || "queued"}`}
                    >
                      <div className="cellengine-discovery-run-head">
                        <strong>seed {run.seed}</strong>
                        <div className="cellengine-run-head-badges">
                          <TelemetryQualityBadge quality={telemetryQuality} />
                          <span className={`cellengine-discovery-run-badge ${run.status || "queued"} ${solved === true ? "pass" : solved === false && run.status === "completed" ? "fail" : ""}`}>
                            {run.status === "completed" ? (solved ? "PASS" : "FAIL") : String(run.status || "queued").toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div className="cellengine-discovery-run-meta">
                        <span>run {Number(run.run_index) + 1}</span>
                        <span>{realizedBodyLabel}</span>
                        <span>{run.summary?.champion_cell_count != null ? `${run.summary.champion_cell_count} cells` : liveMeta.cell_count != null ? `${liveMeta.cell_count} cells` : "waiting"}</span>
                        <span>{liveMeta.current_generation != null ? `gen ${liveMeta.current_generation}` : "waiting"}</span>
                        <span>{Number.isFinite(liveMeta.body_depth) ? `depth ${liveMeta.body_depth}` : liveBodyCells.length && bodyHasDepth(liveBodyCells) ? `depth ${bodyDepthCount(liveBodyCells)}` : "flat body"}</span>
                      </div>
                      {liveBodyCells.length ? (
                        <div className="cellengine-discovery-live-organism">
                          <div className="cellengine-organism-card-visual">
                            <DiscoveryDevelopmentMiniMap
                              bodyCells={liveBodyCells}
                              bodyView={DEFAULT_BODY_VIEW}
                              metricId={genomeMetric}
                              animateOnHover
                            />
                            <OrganismSurfacePreview
                              bodyCells={liveBodyCells}
                              metricId={genomeMetric}
                              title={`seed ${run.seed} surface`}
                              note="current best organism field"
                              summary={run.summary || liveMeta.best_summary || null}
                              searchHistory={runSearchHistory}
                            />
                          </div>
                        </div>
                      ) : null}
                      {run.summary ? (
                        <div className="cellengine-discovery-run-stats">
                          <span>survival {formatPct(run.summary?.cell_clean?.survival_ratio, 1)}</span>
                          <span>success {formatPct(run.summary?.cell_clean?.success_rate, 1)}</span>
                          <span>{run.summary?.cell_clean?.task_primary_label || "metric"} {formatNumber(run.summary?.cell_clean?.task_primary, 2)}</span>
                        </div>
                      ) : liveMeta.best_summary ? (
                        <div className="cellengine-discovery-run-stats">
                          <span>success {formatPct(liveMeta.best_summary?.success_rate, 1)}</span>
                          <span>survival {formatPct(liveMeta.best_summary?.survival_ratio, 1)}</span>
                          <span>{liveMeta.best_summary?.task_primary_label || "task"} {formatNumber(liveMeta.best_summary?.task_primary, 2)}</span>
                        </div>
                      ) : liveMeta.best_search_score != null ? (
                        <div className="cellengine-discovery-run-stats">
                          <span>best gen {liveMeta.best_generation ?? "n/a"}</span>
                          <span>{liveMeta.body_depth != null ? `depth ${liveMeta.body_depth}` : "search in progress"}</span>
                          <span>waiting for task metrics</span>
                        </div>
                      ) : (
                        <div className="meta-note">search in progress</div>
                      )}
                      {run.champion_genome_path || (run.status === "completed" && run.relative_output_dir) ? (
                        <button
                          type="button"
                          className="mini-btn cellengine-watch-btn"
                          onClick={() => setReplayTarget({
                            genomePath: run.champion_genome_path || `${run.relative_output_dir}/champion_genome.csv`,
                            taskName: selectedTaskName,
                            label: `seed ${run.seed}`
                          })}
                        >
                          Watch solving task
                        </button>
                      ) : null}
                      {run.relative_output_dir ? (
                        <div className="cellengine-discovery-run-path">{run.relative_output_dir}</div>
                      ) : null}
                      {run.stdout_tail || run.stderr_tail ? (
                        <pre className="cellengine-discovery-log">{(run.stderr_tail || run.stdout_tail || "").trim() || "waiting for log output"}</pre>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
          {discoveryResult ? (
            <div className="cellengine-discovery-summary">
              <div className="cellengine-telemetry-grid">
                <div>
                  <div className="cellengine-telemetry-label">batch</div>
                  <div className="cellengine-telemetry-value">{discoveryResult.batch_id}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">solved</div>
                  <div className="cellengine-telemetry-value">
                    {discoveryResult.aggregate?.solved_runs ?? 0} / {discoveryResult.aggregate?.completed_runs ?? 0}
                  </div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">mean survival</div>
                  <div className="cellengine-telemetry-value">{formatPct(discoveryResult.aggregate?.mean_survival_ratio, 1)}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">mean success</div>
                  <div className="cellengine-telemetry-value">{formatPct(discoveryResult.aggregate?.mean_success_rate, 1)}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">{discoveryResult.aggregate?.task_primary_label || "mean task metric"}</div>
                  <div className="cellengine-telemetry-value">{formatNumber(discoveryResult.aggregate?.mean_task_primary, 2)}</div>
                </div>
              </div>
              <div className="meta-note">output: {discoveryResult.relative_output_dir}. New champions are added to the gallery after refresh.</div>
              {discoveryResult.aggregate?.champion_params ? (
                <div className="cellengine-discovery-genes">
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>body modes</strong> {Object.entries(discoveryResult.aggregate.champion_params.body_mode_counts || {})
                      .map(([mode, count]) => `${mode}:${count}`)
                      .join(" · ") || "n/a"}
                  </span>
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>cells</strong> mean {formatNumber(discoveryResult.aggregate.champion_params.max_cells?.mean, 1)} · std {formatNumber(discoveryResult.aggregate.champion_params.max_cells?.std, 1)}
                  </span>
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>dev steps</strong> mean {formatNumber(discoveryResult.aggregate.champion_params.development_steps?.mean, 1)} · std {formatNumber(discoveryResult.aggregate.champion_params.development_steps?.std, 1)}
                  </span>
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>extent X/Y/Z</strong> {formatNumber(discoveryResult.aggregate.champion_params.body_extent_x?.mean, 1)} / {formatNumber(discoveryResult.aggregate.champion_params.body_extent_y?.mean, 1)} / {formatNumber(discoveryResult.aggregate.champion_params.body_extent_z?.mean, 1)}
                  </span>
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>growth threshold</strong> mean {formatNumber(discoveryResult.aggregate.champion_params.development_growth_threshold?.mean, 3)}
                  </span>
                  <span className="cellengine-selected-all-mini-slot">
                    <strong>chem rate/decay</strong> {formatNumber(discoveryResult.aggregate.champion_params.chemical_diffusion_rate?.mean, 3)} / {formatNumber(discoveryResult.aggregate.champion_params.chemical_decay?.mean, 3)}
                  </span>
                </div>
              ) : null}
              {Array.isArray(discoveryResult.runs) && discoveryResult.runs.length >= 1 ? (
                <ActionGroups className="cellengine-discovery-result-actions">
                  <ActionGroup title="Watch">
                    {discoveryResult.runs.map((resultRun, idx) => resultRun?.champion_genome_path ? (
                      <ActionButton
                        key={`watch-result-${idx}`}
                        className="mini-btn cellengine-watch-btn"
                        type="button"
                        onClick={() => setReplayTarget({
                          genomePath: resultRun.champion_genome_path,
                          taskName: selectedTaskName,
                          label: discoveryResult.runs.length > 1 ? `run ${idx + 1}` : "champion"
                        })}
                        help="Run an on-demand replay to watch this organism solve the task."
                      >
                        {discoveryResult.runs.length > 1 ? `Watch run ${idx + 1}` : "Watch champion solve task"}
                      </ActionButton>
                    ) : null)}
                  </ActionGroup>
                  <ActionGroup title="Select">
                    {discoveryResult.runs[0]?.champion_genome_path ? (
                      <ActionButton
                        className="mini-btn"
                        type="button"
                        onClick={() => setForm((current) => ({ ...current, genome_path: discoveryResult.runs[0].champion_genome_path }))}
                        help="Select the discovered champion as the next replay input without launching it yet."
                      >
                        Set discovered champion as replay input
                      </ActionButton>
                    ) : null}
                  </ActionGroup>
                  <ActionGroup title="Inspect">
                    {discoveryResult.runs[0]?.relative_output_dir ? (
                      <ActionLink
                        className="mini-btn"
                        href={`/${discoveryResult.runs[0].relative_output_dir.replace(/^reports\//, "reports/")}/report.md`}
                        target="_blank"
                        rel="noreferrer"
                        help="Open the saved report for this discovered run."
                      >
                        Open discovered run report
                      </ActionLink>
                    ) : null}
                  </ActionGroup>
                </ActionGroups>
              ) : null}
              {Array.isArray(discoveryResult.aggregate?.genes) && discoveryResult.aggregate.genes.length ? (
                <div className="cellengine-discovery-genes">
                  {[...discoveryResult.aggregate.genes]
                    .sort((a, b) => (b.std || 0) - (a.std || 0))
                    .slice(0, 8)
                    .map((gene) => (
                      <span key={`discover-gene-${gene.gene}`} className="cellengine-selected-all-mini-slot">
                        <strong>g{gene.gene}</strong> std {formatNumber(gene.std, 3)} · mean {formatNumber(gene.mean, 3)}
                      </span>
                    ))}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
      {replayTarget ? (
        <DiscoveryReplayViewer
          genomePath={replayTarget.genomePath}
          taskName={replayTarget.taskName}
          label={replayTarget.label}
          onClose={() => setReplayTarget(null)}
        />
      ) : null}
    </>
  );
}
