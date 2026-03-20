import {
  clampIndex,
  formatNumber,
  formatPct,
  formatSigned,
  isBalanceTask,
  SPEED_OPTIONS,
  taskLabel
} from "../core";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, FieldLabel, SectionTitle } from "../uiPrimitives";

export function ReplayTopGrid({ vm, actions }) {
  const {
    selectedTaskName,
    activeTaskMeta,
    frame,
    rlFrame,
    frames,
    currentIndex,
    setCurrentIndex,
    playing,
    setPlaying,
    speed,
    setSpeed,
    form,
    setForm,
    launchTaskFields,
    replayBusy,
    busyMode,
    busyElapsedSec,
    latestSummary,
    fullBenchmarkResult,
    taskSpecificGenomeOptions,
    payload,
    taskSpecificRecentReplays
  } = vm;
  const { launchReplay, launchFullBenchmarkOnSelectedGenome, loadReplay, runReplayFromGallery } = actions;

  return (
    <div className="grid cards cellengine-replay-top-grid">
      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Launch Replay"
            help="Runs the C++ benchmark in replay mode and writes a fresh trace bundle under reports/cellengine_ui/. This is a new backend run, not a cache load."
          />
          <span className="meta-note">task: {taskLabel(selectedTaskName)} · fresh backend rerun of `bin/benchmark_cellengine` for the selected genome</span>
        </div>
        <div className="cellengine-form-grid">
          <label>
            <FieldLabel label="Genome" help="Champion genome file to instantiate into the organism before the replay starts." />
            <select className="task-input" value={form.genome_path} onChange={(event) => setForm((current) => ({ ...current, genome_path: event.target.value }))}>
              {taskSpecificGenomeOptions.map((option) => (
                <option key={option.genome_path} value={option.genome_path}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            <FieldLabel label="Max ticks" help="Maximum number of physics steps for this replay." />
            <input className="task-input" type="number" min="20" max="4000" value={form.max_ticks} onChange={(event) => setForm((current) => ({ ...current, max_ticks: event.target.value }))} />
          </label>
          <label>
            <FieldLabel label="Damage tick" help="If set, the organism is ablated at this tick so you can inspect recovery and robustness. Leave blank to disable damage." />
            <input className="task-input" type="number" min="0" max={Math.max(0, Number(form.max_ticks) - 1)} value={form.damage_tick} placeholder="disabled" onChange={(event) => setForm((current) => ({ ...current, damage_tick: event.target.value }))} />
          </label>
          <label>
            <FieldLabel label={launchTaskFields.primaryLabel} help={launchTaskFields.primaryHelp} />
            <input
              className="task-input"
              type="number"
              step="0.1"
              value={isBalanceTask(selectedTaskName) ? form.theta_deg : form.task_param_a}
              placeholder={launchTaskFields.primaryPlaceholder}
              onChange={(event) => setForm((current) => (isBalanceTask(selectedTaskName) ? { ...current, theta_deg: event.target.value } : { ...current, task_param_a: event.target.value }))}
            />
          </label>
          {launchTaskFields.secondaryLabel ? (
            <label>
              <FieldLabel label={launchTaskFields.secondaryLabel} help={launchTaskFields.secondaryHelp} />
              <input className="task-input" type="number" step="0.1" value={form.task_param_b} placeholder={launchTaskFields.secondaryPlaceholder} onChange={(event) => setForm((current) => ({ ...current, task_param_b: event.target.value }))} />
            </label>
          ) : null}
          <label>
            <FieldLabel label="Seed" help="Random seed for replay initialization. Reuse it to reproduce the same starting condition." />
            <input className="task-input" type="number" min="1" value={form.seed} onChange={(event) => setForm((current) => ({ ...current, seed: event.target.value }))} />
          </label>
          <label>
            <FieldLabel label="RL comparator" help="RL baseline replayed from the same initial condition so you can compare it against the cell controller." />
            <select className="task-input" value={form.rl_algo} onChange={(event) => setForm((current) => ({ ...current, rl_algo: event.target.value }))}>
              <option value="a2c">A2C</option>
              <option value="dqn">DQN</option>
              <option value="none">disabled</option>
            </select>
          </label>
        </div>
        <ActionGroups className="cellengine-actions-toolbar">
          <ActionGroup title="Run">
            <ActionButton className="refresh-btn cellengine-run-btn" type="button" disabled={replayBusy || !form.genome_path} onClick={launchReplay} help="Runs the selected genome once in replay mode and loads the fresh trace into the viewer. This does not search for a new genome.">
              {replayBusy ? "Running..." : "Run replay"}
            </ActionButton>
            <ActionButton className="refresh-btn cellengine-run-btn secondary" type="button" disabled={replayBusy || !form.genome_path} onClick={launchFullBenchmarkOnSelectedGenome} help="Runs the selected existing genome through the full benchmark protocol: final evaluation, damage evaluation, RL comparison, and ODD. No discovery search is performed.">
              {busyMode === "full_eval" ? "Running full benchmark..." : "Run full benchmark on selected genome"}
            </ActionButton>
          </ActionGroup>
          <ActionGroup title="Presets">
            <ActionButton className="mini-btn" type="button" disabled={replayBusy} onClick={() => setForm((current) => ({ ...current, damage_tick: 150, theta_deg: "" }))} help="Fill the replay form with a mid-run damage event so you can inspect robustness and recovery.">
              Apply damage scenario
            </ActionButton>
            <ActionButton
              className="mini-btn"
              type="button"
              disabled={replayBusy}
              onClick={() => setForm((current) => (isBalanceTask(selectedTaskName) ? { ...current, damage_tick: "", theta_deg: 8 } : selectedTaskName === "worm_drag_race" ? { ...current, damage_tick: "", task_param_a: 0.22 } : { ...current, damage_tick: "", task_param_a: 0.55, task_param_b: 0.75 }))}
              help={isBalanceTask(selectedTaskName) ? "Fill the replay form with a harder initial pole angle." : selectedTaskName === "worm_drag_race" ? "Fill the replay form with a stronger initial crawl bend." : "Fill the replay form with a harder opening serve for the tracking task."}
            >
              {isBalanceTask(selectedTaskName) ? "Apply angle stress test" : selectedTaskName === "worm_drag_race" ? "Apply crawl bias" : "Apply hard serve"}
            </ActionButton>
          </ActionGroup>
          <ActionGroup title="Inspect">
            <ActionButton className="mini-btn" type="button" onClick={() => vm.overviewQuery.refresh()} help="Refresh the saved genomes and replay libraries from disk.">
              Refresh saved artifacts
            </ActionButton>
          </ActionGroup>
        </ActionGroups>
        {latestSummary ? <div className={`cellengine-run-result-inline ${latestSummary.solved ? "pass" : "fail"}`}>{latestSummary.solved ? "PASS" : "FAIL"} · {latestSummary.solved ? (activeTaskMeta.stageFamily === "worm" ? "crawl goal achieved for this run" : activeTaskMeta.stageFamily === "pong" ? "rally target achieved for this run" : "control task held for this run") : "run terminated before target horizon"}</div> : null}
        {busyMode === "launch" ? <div className="cellengine-status-note" role="status" aria-live="polite">Launching the C++ replay backend. The current viewer stays on the previous trace until the new bundle is finished.<strong> elapsed {busyElapsedSec}s</strong></div> : null}
        {busyMode === "full_eval" ? <div className="cellengine-status-note" role="status" aria-live="polite">Running the full benchmark on the selected genome: full evaluation budget, damage evaluation, RL comparison, and ODD.<strong> elapsed {busyElapsedSec}s</strong></div> : null}
        {busyMode === "gallery" ? <div className="cellengine-status-note" role="status" aria-live="polite">Running selected gallery genome and updating the viewer.<strong> elapsed {busyElapsedSec}s</strong></div> : null}
        {busyMode === "batch" ? <div className="cellengine-status-note" role="status" aria-live="polite">Running queued genomes sequentially and recording PASS/FAIL on each card.<strong> elapsed {busyElapsedSec}s</strong></div> : null}
        {busyMode === "load" ? <div className="cellengine-status-note" role="status" aria-live="polite">Loading cached replay artifacts into the viewer.<strong> elapsed {busyElapsedSec}s</strong></div> : null}
        <div className="meta-note cellengine-command-note">`CELLENGINE_MODE=replay`, `CELLENGINE_TASK={form.task_name}`, `CELLENGINE_RL_ALGO=none` (cell-only replay), comparator loads best cached `{form.rl_algo}` trace, output under `reports/cellengine_ui/...`</div>
        {fullBenchmarkResult ? (
          <div className="cellengine-discovery-summary">
            <div className="cellengine-telemetry-grid">
              <div><div className="cellengine-telemetry-label">full benchmark output</div><div className="cellengine-telemetry-value">{fullBenchmarkResult.relative_output_dir?.replace(/^reports\//, "")}</div></div>
              <div><div className="cellengine-telemetry-label">solved</div><div className="cellengine-telemetry-value">{fullBenchmarkResult.summary?.solved ? "yes" : "no"}</div></div>
              <div><div className="cellengine-telemetry-label">survival</div><div className="cellengine-telemetry-value">{formatPct(fullBenchmarkResult.summary?.cell_clean?.survival_ratio, 1)}</div></div>
              <div><div className="cellengine-telemetry-label">success</div><div className="cellengine-telemetry-value">{formatPct(fullBenchmarkResult.summary?.cell_clean?.success_rate, 1)}</div></div>
              <div><div className="cellengine-telemetry-label">RL</div><div className="cellengine-telemetry-value">{String(fullBenchmarkResult.summary?.rl_algorithm_selected || fullBenchmarkResult.config?.rl_algo || "n/a").toUpperCase()}</div></div>
            </div>
            <div className="meta-note">standard full benchmark completed for {taskLabel(fullBenchmarkResult.config?.task_name || selectedTaskName)} · report under {fullBenchmarkResult.relative_output_dir}</div>
            <ActionGroups className="cellengine-full-benchmark-actions">
              <ActionGroup title="Select">
                <ActionButton className="mini-btn" type="button" disabled={!fullBenchmarkResult.champion_genome_path} onClick={() => setForm((current) => ({ ...current, genome_path: fullBenchmarkResult.champion_genome_path || current.genome_path }))} help="Select this benchmark's champion genome as the next replay input without launching it yet.">
                  Set benchmark champion as replay input
                </ActionButton>
              </ActionGroup>
              <ActionGroup title="Inspect">
                {fullBenchmarkResult.files?.report_md ? <ActionLink className="mini-btn" href={fullBenchmarkResult.files.report_md} target="_blank" rel="noreferrer" help="Open the saved markdown report for this full benchmark run.">Open report file</ActionLink> : null}
                {fullBenchmarkResult.files?.summary_json ? <ActionLink className="mini-btn" href={fullBenchmarkResult.files.summary_json} target="_blank" rel="noreferrer" help="Open the machine-readable summary JSON for this full benchmark run.">Open summary file</ActionLink> : null}
              </ActionGroup>
            </ActionGroups>
          </div>
        ) : null}
      </div>

      <div className="card">
        <div className="section-head">
          <SectionTitle title="Playback" help="Timeline controls for the currently loaded replay. This only changes the viewer position in the saved trace." />
          <span className="meta-note">{frames.length ? `${playing ? "running" : "paused"} · ${currentIndex + 1} / ${frames.length} frames` : "idle"}</span>
        </div>
        <div className="row controls cellengine-playback-row">
          <button className="mini-btn" type="button" disabled={!frames.length} onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
          <button className="mini-btn" type="button" disabled={!frames.length} onClick={() => { setCurrentIndex(0); setPlaying(false); }}>Restart</button>
          <button className="mini-btn" type="button" disabled={!frames.length} onClick={() => setCurrentIndex((value) => clampIndex(value - 1, Math.max(0, frames.length - 1)))}>Step -</button>
          <button className="mini-btn" type="button" disabled={!frames.length} onClick={() => setCurrentIndex((value) => clampIndex(value + 1, Math.max(0, frames.length - 1)))}>Step +</button>
          <label>
            <FieldLabel label="Speed" help="Viewer playback multiplier for the loaded frames." />
            <select className="task-input" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              {SPEED_OPTIONS.map((option) => <option key={option} value={option}>{option}x</option>)}
            </select>
          </label>
        </div>
        <input className="cellengine-scrubber" type="range" min="0" max={Math.max(0, frames.length - 1)} value={Math.min(currentIndex, Math.max(0, frames.length - 1))} disabled={!frames.length} onChange={(event) => { setCurrentIndex(Number(event.target.value)); setPlaying(false); }} />
        <div className="cellengine-frame-meta">
          <span>tick {frame?.tick ?? "n/a"}</span>
          <span>{activeTaskMeta.stageFamily === "worm" ? `bend ${formatSigned(frame?.theta_rad, 3)}` : activeTaskMeta.stageFamily === "pong" ? `ball y ${formatSigned(frame?.theta_rad, 2)}` : `theta ${formatSigned(frame?.theta_deg, 2, "°")}`}</span>
          <span>{activeTaskMeta.stageFamily === "pong" ? `paddle y ${formatSigned(frame?.x, 2)}` : `x ${formatSigned(frame?.x, 3, " m")}`}</span>
          <span>force {formatSigned(frame?.total_force, 2)}</span>
          {rlFrame ? <span>RL {String(payload?.summary?.rl_algorithm_selected || form.rl_algo).toUpperCase()} {activeTaskMeta.stageFamily === "worm" ? `progress ${formatSigned(rlFrame?.x, 2, " m")}` : activeTaskMeta.stageFamily === "pong" ? `returns ${rlFrame?.task_counter ?? 0}` : `theta ${formatSigned(rlFrame?.theta_deg, 2, "°")}`}</span> : null}
        </div>
      </div>
    </div>
  );
}
