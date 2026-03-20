import { WhatAmISeeing } from "./cellengine/replayExplainers";
import {
  formatPct,
  taskLabel,
  taskMeta,
  writeCellEngineSectionToUrl
} from "./cellengine/core";
import { CellEngineSectionNav } from "./cellengine/discoveryViews";
import { SectionTitle } from "./cellengine/uiPrimitives";
import { useCellEngineState } from "./cellengine/useCellEngineState";
import { useCellEngineDerived } from "./cellengine/useCellEngineDerived";
import { useCellEngineActions } from "./cellengine/useCellEngineActions";
import { CompareSection } from "./cellengine/sections/CompareSection";
import { DiscoverSection } from "./cellengine/sections/DiscoverSection";
import { ReplaySection } from "./cellengine/sections/ReplaySection";
import { ReportsSection } from "./cellengine/sections/ReportsSection";

export default function CellEngineReplayPanel() {
  const state = useCellEngineState();
  const derived = useCellEngineDerived(state);
  const vm = { ...state, ...derived };
  const actions = useCellEngineActions({
    ...vm,
    writeCellEngineSectionToUrl,
    setActiveSection: state.setActiveSection
  });
  const {
    overview,
    activeSection,
    setActiveSection,
    activeDiscoveryJob,
    form,
    setForm,
    missingCellArtifacts,
    preferredReplay,
    bodyFollowsSelectedGenome,
    organismHasDepth,
    frame,
    analysis,
    latestSummary,
    activeTaskName,
    error,
    selectedTaskName,
    activeTaskMeta
  } = vm;

  return (
    <section className="panel stack cellengine-panel">
      <div className="cellengine-hero">
        <div>
          <div className="cellengine-kicker">CellEngine Replay Lab</div>
          <h2>React viewer for the CellEngine task suite</h2>
          <p>
            Switch tasks in place, launch replays or discovery runs, and keep the organism/genome inspection tools on one page.
          </p>
        </div>
        <div className="cellengine-hero-meta">
          <span className="landscape-pill">source {overview.latest_report?.relative_output_dir || "reports/cellengine_latest"}</span>
          <span className="landscape-pill">latest solved {formatPct(overview.latest_report?.summary?.cell_clean?.survival_ratio, 1)}</span>
          <span className="landscape-pill">RL best {overview.latest_report?.summary?.rl_algorithm_selected || "n/a"}</span>
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Task Switcher"
            help="Choose the active benchmark task here. The launch form, discovery form, replay library filter, and stage panels update in place."
          />
          <span className="meta-note">shared genome and cell tooling, task-specific scene swap</span>
        </div>
        <div className="cellengine-task-strip">
          {vm.taskOptions.map((taskName) => {
            const meta = taskMeta(taskName);
            const active = selectedTaskName === taskName;
            return (
              <button
                key={taskName}
                type="button"
                className={`cellengine-task-pill${active ? " active" : ""}`}
                onClick={() => setForm((current) => ({
                  ...current,
                  task_name: taskName,
                  theta_deg: meta.stageFamily === "balance" ? current.theta_deg : "",
                  task_param_a: meta.stageFamily === "balance" ? "" : current.task_param_a,
                  task_param_b: taskName === "pong_return" ? current.task_param_b : ""
                }))}
              >
                <strong>{meta.label}</strong>
                <span>{meta.short}</span>
              </button>
            );
          })}
        </div>
        <div className="meta-note">{taskMeta(selectedTaskName).description}</div>
      </div>

      <CellEngineSectionNav
        activeSection={activeSection}
        onChange={(nextSection) => {
          writeCellEngineSectionToUrl(nextSection);
          setActiveSection(nextSection);
        }}
      />

      {activeDiscoveryJob ? (
        <div className="card cellengine-running-banner">
          <div className="section-head">
            <SectionTitle
              title="Running Discovery Job"
              help="A discovery batch is currently running on the backend. You can attach this page to it and jump straight to the live organism views."
            />
            <span className="meta-note">
              {activeDiscoveryJob.status} · {activeDiscoveryJob.completed_runs || 0} / {activeDiscoveryJob.total_runs || 0} complete
              {Number.isInteger(activeDiscoveryJob.current_run_index) ? ` · seed ${activeDiscoveryJob.current_seed}` : ""}
            </span>
          </div>
          <div className="row controls">
            <button
              type="button"
              className="refresh-btn"
              onClick={() => {
                state.setDiscoveryJob(activeDiscoveryJob);
                state.setBusyMode("discover");
                state.setBusyStartedAt(Date.now());
                writeCellEngineSectionToUrl("discover");
                setActiveSection("discover");
              }}
            >
              Go To Running Discovery
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={() => {
                state.setDiscoveryJob(activeDiscoveryJob);
                state.setBusyMode("discover");
                state.setBusyStartedAt(Date.now());
              }}
            >
              Attach Viewer
            </button>
            {activeDiscoveryJob.relative_output_dir ? (
              <span className="meta-note">{activeDiscoveryJob.relative_output_dir}</span>
            ) : null}
          </div>
        </div>
      ) : null}

      {activeSection === "replay" ? (
        <>
          <WhatAmISeeing frame={frame} analysis={analysis} latestSummary={latestSummary} taskName={activeTaskName} />
          {missingCellArtifacts ? (
            <div className="error">
              <div>This replay has stage frames but no per-cell artifacts. Click `Run replay` once to generate body, cell, edge, and genome layers.</div>
            </div>
          ) : null}
          {!vm.payload && !preferredReplay ? (
            <div className="error">
              <div>No compatible replay is cached yet for the full cell/genome viewer. Click `Run replay` once.</div>
            </div>
          ) : null}
          {bodyFollowsSelectedGenome ? (
            <div className="meta-note">
              Morphology-aware views follow the currently selected genome immediately. If you changed genomes without rerunning,
              body shape is updated now while live force/cell telemetry still comes from the currently loaded replay.
            </div>
          ) : null}
        </>
      ) : null}

      {activeSection === "compare" && organismHasDepth ? (
        <div className="meta-note">
          3D organism views are direct-manipulation now: drag inside the organism panels to rotate, use the mouse wheel to zoom, and double-click any organism panel to reset.
        </div>
      ) : null}

      {activeSection === "compare" ? <CompareSection vm={vm} actions={actions} /> : null}
      {activeSection === "discover" ? <DiscoverSection vm={vm} actions={actions} /> : null}
      {activeSection === "replay" ? <ReplaySection vm={vm} actions={actions} /> : null}
      {activeSection === "reports" ? <ReportsSection vm={vm} actions={actions} /> : null}

      {error ? (
        <div className="error">
          <div>{error}</div>
        </div>
      ) : null}
    </section>
  );
}
