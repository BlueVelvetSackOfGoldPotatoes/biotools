import { Suspense } from "react";
import {
  FAMILY_COLORS,
  fileLabel,
  formatPct,
  runStatus,
  runStatusClass,
  runStatusLabel
} from "../lib/dashboardShared";
import { buildViewAllDataset } from "../lib/appViewDataset";
import { specPaths } from "../lib/appViewMeta";

export default function DashboardOverviewSections({
  tab,
  activeViewSpec,
  bestFamilyRuns,
  viewAllPayload,
  hasViewAllData,
  setViewAllNonce,
  setViewAllId,
  setTab,
  viewAllReturnTab,
  setRunId,
  runStateFilter,
  setRunStateFilter,
  familyFilter,
  setFamilyFilter,
  familyOptions,
  visibleRuns,
  expandedRunId,
  setExpandedRunId,
  runId,
  setArchitectureFamily,
  runsQuery,
  dashboardRuns,
  tasks,
  tasksQuery,
  taskBusyKey,
  taskActionError,
  taskActionMessage,
  refreshTaskAndRunQueries,
  createTrainTask,
  createEvaluateTask,
  killTaskAction,
  startTaskAction,
  deleteTaskAction,
  ViewAllChartLazy,
  RunDetailPanelLazy,
  TasksPanelLazy
}) {
  return (
    <>
      {tab === "viewall" && activeViewSpec && (
        <section className="panel stack">
          <div className="row controls">
            <strong>View All Architectures: {activeViewSpec.title}</strong>
            <span className="meta-note">
              best run per family ({bestFamilyRuns.length} families)
            </span>
            <button
              className="mini-btn"
              type="button"
              onClick={() => setViewAllNonce((value) => value + 1)}
            >
              Refresh all
            </button>
            <button
              className="mini-btn"
              type="button"
              onClick={() => {
                setViewAllId("");
                setTab(viewAllReturnTab || activeViewSpec.tab || "comparison");
              }}
            >
              Back
            </button>
            {viewAllPayload.fetchedAt ? (
              <span className="meta-note">last update: {new Date(viewAllPayload.fetchedAt).toLocaleTimeString()}</span>
            ) : null}
            {viewAllPayload.loading && hasViewAllData ? (
              <span className="meta-note">refreshing...</span>
            ) : null}
          </div>

          {viewAllPayload.loading && !hasViewAllData ? (
            <div className="empty">Loading cross-architecture charts...</div>
          ) : (
            <div className="all-arch-grid">
              {bestFamilyRuns.map(({ family, run }) => {
                const pack = viewAllPayload.byFamily?.[family] || {
                  family,
                  run,
                  paths: specPaths(activeViewSpec, family),
                  byPath: {},
                  error: ""
                };
                const dataset = buildViewAllDataset(activeViewSpec.id, pack);
                const familyColor = FAMILY_COLORS[family] || "#495057";
                return (
                  <div key={family} className="card">
                    <div className="card-title compact">
                      <div className="card-title-left">
                        <span className="family-dot" style={{ background: familyColor }} />
                        <strong>{family}</strong>
                      </div>
                      <button
                        className="mini-btn"
                        type="button"
                        onClick={() => {
                          setRunId(run.run_id);
                          setViewAllId("");
                          setTab(activeViewSpec.tab || viewAllReturnTab || "live");
                        }}
                      >
                        Open run
                      </button>
                    </div>
                    <div className="meta-note">
                      run: {run.run_id} · {run.benchmark_id === "tictactoe" ? "best non-loss" : "best acc"}: {formatPct(run.latest_test_accuracy)}
                    </div>
                    {pack.paths?.length ? (
                      <div className="meta-note">sources: {pack.paths.map(fileLabel).join(", ")}</div>
                    ) : null}
                    {pack.error ? <div className="meta-note">{pack.error}</div> : null}
                    <Suspense fallback={<div className="empty">Loading chart…</div>}>
                      <ViewAllChartLazy dataset={dataset} />
                    </Suspense>
                  </div>
                );
              })}
            </div>
          )}
          {viewAllPayload.error ? <div className="empty">{viewAllPayload.error}</div> : null}
        </section>
      )}

      {tab === "live" && (
        <section className="panel stack">
          <div className="row controls">
            <label>
              State
              <select value={runStateFilter} onChange={(event) => setRunStateFilter(event.target.value)}>
                <option value="all">all</option>
                <option value="active">active</option>
                <option value="done">done</option>
              </select>
            </label>

            <label>
              Family
              <select value={familyFilter} onChange={(event) => setFamilyFilter(event.target.value)}>
                {familyOptions.map((family) => (
                  <option key={family} value={family}>
                    {family}
                  </option>
                ))}
              </select>
            </label>

            <span className="meta-note">
              showing {visibleRuns.length} runs
              {expandedRunId && " · click a card to expand"}
            </span>
          </div>

          {runsQuery.isLoading ? (
            <div className="empty">Loading runs...</div>
          ) : visibleRuns.length ? (
            <div className="live-run-list">
              {visibleRuns.map((run) => {
                const isExpanded = expandedRunId === run.run_id;
                const familyColor = FAMILY_COLORS[run.model_family] || "#495057";
                return (
                  <div key={run.run_id} className={`live-run-item${isExpanded ? " expanded" : ""}`}>
                    <button
                      className={`card live-card${run.run_id === runId ? " selected" : ""}`}
                      onClick={() => {
                        setRunId(run.run_id);
                        setExpandedRunId(isExpanded ? "" : run.run_id);
                      }}
                      type="button"
                    >
                      <div className="card-title">
                        <div className="card-title-left">
                          <span className="family-dot" style={{ background: familyColor }} />
                          <strong>{run.run_id}</strong>
                        </div>
                        <div className="card-title-right">
                          <span className="acc-badge">{formatPct(run.latest_test_accuracy)}</span>
                          <span className={`pill ${runStatusClass(runStatus(run))}`}>{runStatusLabel(runStatus(run))}</span>
                          <span className={`expand-arrow${isExpanded ? " open" : ""}`}>&#9662;</span>
                        </div>
                      </div>
                      <div className="card-meta-row">
                        <span className="card-meta-chip">{run.benchmark_name || run.benchmark_id || "unknown"}</span>
                        <span className="card-meta-chip">{run.model_family}</span>
                        <span className="card-meta-chip">{run.model_variant}</span>
                        <span className="card-meta-time">{run.updated_utc ? new Date(run.updated_utc).toLocaleTimeString() : "n/a"}</span>
                      </div>
                    </button>
                    {isExpanded && (
                      <Suspense fallback={<div className="detail-loading">Loading run detail...</div>}>
                        <RunDetailPanelLazy
                          runId={run.run_id}
                          onNavigate={(targetTab) => {
                            setRunId(run.run_id);
                            setTab(targetTab);
                            if (targetTab === "model") {
                              setArchitectureFamily(run.model_family || "");
                            }
                          }}
                        />
                      </Suspense>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty">No runs match current filters.</div>
          )}
        </section>
      )}

      {tab === "tasks" && (
        <Suspense fallback={<section className="panel"><div className="empty">Loading task orchestration...</div></section>}>
          <TasksPanelLazy
            tasks={tasks}
            runs={dashboardRuns}
            selectedRunId={runId}
            loading={tasksQuery.isLoading}
            busyKey={taskBusyKey}
            actionError={taskActionError}
            actionMessage={taskActionMessage}
            onRefresh={refreshTaskAndRunQueries}
            onCreateTrain={createTrainTask}
            onCreateEvaluate={createEvaluateTask}
            onKillTask={killTaskAction}
            onStartTask={startTaskAction}
            onDeleteTask={deleteTaskAction}
            onOpenRun={(nextRunId) => {
              setRunId(nextRunId);
              setTab("live");
            }}
          />
        </Suspense>
      )}
    </>
  );
}
