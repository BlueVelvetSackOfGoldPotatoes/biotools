import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  clampIndex,
  cellTypeLabel,
  formatActionLabel,
  formatNumber,
  formatPct,
  formatSigned,
  isBalanceTask,
  SPEED_OPTIONS,
  taskLabel
} from "../core";
import { formatBodyCellPosition } from "../body";
import { SharedVisualizationLegend } from "../legends";
import { TaskReplayScene } from "../replayScenes";
import { ReplayLibrary } from "../libraryViews";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, FieldLabel, ForceBar, SectionTitle, StatCard } from "../uiPrimitives";
import { SelectedAllCellsVisualGrid, SelectedCellVisualInspector } from "../cellViews";
import { ReplayTopGrid } from "./ReplayTopGrid";

export function ReplaySection({ vm, actions }) {
  const {
    selectedTaskName,
    activeTaskName,
    activeTaskMeta,
    frame,
    rlFrame,
    analysis,
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
    overviewQuery,
    latestSummary,
    fullBenchmarkResult,
    summaryCards,
    bodyCells,
    currentFrameCellRows,
    currentFrameEdges,
    functionalStateByCellId,
    selectedBodyCell,
    setSelectedCellId,
    selectedCellState,
    selectedCellFunction,
    selectedCellEdges,
    payload,
    timelineRows,
    phaseRows,
    rlPhaseRows,
    taskSpecificGenomeOptions,
    taskSpecificRecentReplays,
    rlFrames,
    rlAnalysis,
    selectedCellViewMode,
    setSelectedCellViewMode,
    genomeMetric,
    includeDepthCoordinate,
    bodyFollowsSelectedGenome
  } = vm;
  const {
    launchReplay,
    launchFullBenchmarkOnSelectedGenome,
    loadReplay,
    runReplayFromGallery
  } = actions;

  return (
    <>
      <ReplayTopGrid vm={vm} actions={actions} />
      <div className="cellengine-stats-grid">
        {summaryCards.map((card) => (
          <StatCard key={card.label} label={card.label} value={card.value} note={card.note} tone={card.tone} />
        ))}
      </div>

      <SharedVisualizationLegend
        title="Replay Visual Legend"
        note="shared once for replay-stage organism views and per-cell inspectors"
        genomeMetric={genomeMetric}
        showGenome
        showFunction
        showRoles
        showMatrix
        showGraph
      />

      <div className="grid two cellengine-main-grid">
        <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Cell Controller Stage"
              help="Task-specific scene for the cell controller. The organism is always the left-side controlled body; only the task framing changes."
            />
            <span className="meta-note">{activeTaskMeta.description}</span>
          </div>
          <TaskReplayScene
            taskName={activeTaskName}
            frames={frames}
            frameIndex={currentIndex}
            analysis={analysis}
            controllerLabel="organism"
            controllerColor="#a61e4d"
            rightTitle="Cell tissue state"
            rightLines={[
              `active cells ${formatPct(frame?.active_fraction, 1)}`,
              `mean energy ${formatNumber(frame?.mean_energy, 3)}`,
              activeTaskMeta.stageFamily === "worm"
                ? `progress ${formatSigned(frame?.x, 2, " m")}`
                : activeTaskMeta.stageFamily === "pong"
                  ? `returns ${frame?.task_counter ?? 0}`
                  : `mean stress ${formatNumber(frame?.mean_stress, 3)}`
            ]}
            renderOrganismBody
            organismBodyCells={bodyCells}
            organismFrameCellRows={currentFrameCellRows}
            organismFrameEdges={currentFrameEdges}
            organismFunctionalStateByCellId={functionalStateByCellId}
            organismSelectedCellId={selectedBodyCell?.cell_id ?? null}
            onSelectOrganismCell={setSelectedCellId}
          />
        </div>

        <div className="card">
          <div className="section-head">
            <SectionTitle
              title="RL Comparator Stage"
              help="Cached replay of the selected RL policy for reference against the cell controller timeline."
            />
            <span className="meta-note">best cached RL trace for the selected algorithm (no on-the-fly retraining)</span>
          </div>
          {rlFrames.length ? (
            <TaskReplayScene
              taskName={activeTaskName}
              frames={rlFrames}
              frameIndex={Math.min(currentIndex, Math.max(0, rlFrames.length - 1))}
              analysis={rlAnalysis}
              controllerLabel={(payload?.summary?.rl_algorithm_selected || form.rl_algo || "rl").toUpperCase()}
              controllerColor="#155e75"
              rightTitle={`${(payload?.summary?.rl_algorithm_selected || form.rl_algo || "rl").toUpperCase()} state`}
              rightLines={[
                `action ${formatActionLabel(rlFrame?.action)}`,
                `force ${formatSigned(rlFrame?.total_force, 2)}`,
                activeTaskMeta.stageFamily === "worm"
                  ? `progress ${formatSigned(rlFrame?.x, 2, " m")}`
                  : activeTaskMeta.stageFamily === "pong"
                    ? `returns ${rlFrame?.task_counter ?? 0}`
                    : `peak |theta| ${formatNumber(rlAnalysis?.max_abs_theta_deg, 2)}°`,
                `success ${rlAnalysis?.success ? "yes" : "no"}`
              ]}
            />
          ) : (
            <div className="empty">RL comparator disabled for this replay or not available in older cached traces.</div>
          )}
        </div>
      </div>

      <div className="grid two cellengine-main-grid">
        <div className="stack">
          <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Cell Snapshot"
              help="High-level telemetry from the current frame for the cell controller. Labels adapt to the currently selected task."
            />
            <span className="meta-note">live values at the scrubber position</span>
          </div>
          <div className="cellengine-telemetry-grid">
            <div>
              <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "worm" ? "bend / bend rate" : activeTaskMeta.stageFamily === "pong" ? "ball y / ball vy" : "theta / theta dot"}</div>
              <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "balance" ? `${formatSigned(frame?.theta_deg, 2, "°")} · ${formatSigned(frame?.theta_dot, 3)}` : `${formatSigned(frame?.theta_rad, 3)} · ${formatSigned(frame?.theta_dot, 3)}`}</div>
            </div>
            <div>
              <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "pong" ? "paddle y / paddle vy" : "x / x dot"}</div>
              <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "pong" ? `${formatSigned(frame?.x, 3)} · ${formatSigned(frame?.x_dot, 3)}` : `${formatSigned(frame?.x, 3, " m")} · ${formatSigned(frame?.x_dot, 3)}`}</div>
            </div>
            <div>
              <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "pong" ? "ball x / returns" : "x ddot / theta ddot"}</div>
              <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "pong" ? `${formatSigned(frame?.task_aux_a, 3)} · ${frame?.task_counter ?? 0}` : `${formatSigned(frame?.x_ddot, 3)} · ${formatSigned(frame?.theta_ddot, 3)}`}</div>
            </div>
            <div>
              <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "pong" ? "active cells / returns" : "active cells / calcium"}</div>
              <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "pong" ? `${formatPct(frame?.active_fraction, 1)} · ${frame?.task_counter ?? 0}` : `${formatPct(frame?.active_fraction, 1)} · ${formatNumber(frame?.mean_calcium, 3)}`}</div>
            </div>
          </div>
            <div className="cellengine-force-stack">
              <ForceBar label="Organism force" value={frame?.organism_force} maxValue={analysis?.max_abs_total_force} colorClass="organism" />
              <ForceBar label="Teacher force" value={frame?.teacher_force} maxValue={analysis?.max_abs_total_force} colorClass="teacher" />
              <ForceBar label="Total force" value={frame?.total_force} maxValue={analysis?.max_abs_total_force} colorClass="total" />
            </div>
          </div>

          <div className="card">
            <div className="section-head">
              <SectionTitle
                title="Trace Files"
                help="Raw artifacts written by the backend for this replay, including summaries, traces, body maps, and per-cell rows when available."
              />
              <span className="meta-note">saved after every replay</span>
            </div>
            {payload ? (
              <div className="cellengine-file-grid">
                <a href={payload.files?.summary_json} target="_blank" rel="noreferrer">summary.json</a>
                <a href={payload.files?.replay_trace_csv} target="_blank" rel="noreferrer">replay_trace.csv</a>
                <a href={payload.files?.analysis_json} target="_blank" rel="noreferrer">analysis.json</a>
                {payload.files?.replay_body_csv ? <a href={payload.files.replay_body_csv} target="_blank" rel="noreferrer">replay_body.csv</a> : null}
                {payload.files?.replay_cells_csv ? <a href={payload.files.replay_cells_csv} target="_blank" rel="noreferrer">replay_cells.csv</a> : null}
                {payload.files?.replay_edges_csv ? <a href={payload.files.replay_edges_csv} target="_blank" rel="noreferrer">replay_edges.csv</a> : null}
                {payload.files?.rl_replay_trace_csv ? <a href={payload.files.rl_replay_trace_csv} target="_blank" rel="noreferrer">rl_replay_trace.csv</a> : null}
                {payload.files?.rl_analysis_json ? <a href={payload.files.rl_analysis_json} target="_blank" rel="noreferrer">rl_analysis.json</a> : null}
                {payload.files?.report_md ? <a href={payload.files.report_md} target="_blank" rel="noreferrer">report.md</a> : null}
                <div className="meta-note">{payload.relative_output_dir}</div>
              </div>
            ) : (
              <div className="empty">No replay loaded yet.</div>
            )}
          </div>

          {payload?.process ? (
            <div className="card">
              <div className="section-head">
                <SectionTitle
                  title="Process Diagnostics"
                  help="Backend subprocess status for the replay launch. This is useful when a replay fails, times out, or produces partial output."
                />
                <span className="meta-note">useful when a replay fails or times out</span>
              </div>
              <div className="cellengine-telemetry-grid">
                <div>
                  <div className="cellengine-telemetry-label">exit / signal</div>
                  <div className="cellengine-telemetry-value">
                    {payload.process.exit_code ?? "n/a"} / {payload.process.signal || "none"}
                  </div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">timed out</div>
                  <div className="cellengine-telemetry-value">{payload.process.timed_out ? "yes" : "no"}</div>
                </div>
              </div>
              {payload.process.stderr_tail ? (
                <pre className="cellengine-log-tail">{payload.process.stderr_tail}</pre>
              ) : payload.process.stdout_tail ? (
                <pre className="cellengine-log-tail">{payload.process.stdout_tail}</pre>
              ) : (
                <div className="meta-note">No subprocess logs captured.</div>
              )}
            </div>
          ) : null}
        </div>

        <div className="stack">
          <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Comparator Snapshot"
              help="Frame-aligned comparison between the cell controller and the selected RL controller at the current scrubber position."
            />
            <span className="meta-note">same tick, cell controller vs RL comparator</span>
          </div>
          {rlFrame ? (
            <div className="stack">
              <div className="cellengine-telemetry-grid">
                <div>
                  <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "balance" ? "cell theta / force" : activeTaskMeta.stageFamily === "worm" ? "cell progress / force" : "cell returns / force"}</div>
                  <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "balance" ? `${formatSigned(frame?.theta_deg, 2, "°")} · ${formatSigned(frame?.total_force, 2)}` : activeTaskMeta.stageFamily === "worm" ? `${formatSigned(frame?.x, 2, " m")} · ${formatSigned(frame?.total_force, 2)}` : `${frame?.task_counter ?? 0} · ${formatSigned(frame?.total_force, 2)}`}</div>
                </div>
                <div>
                    <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "balance" ? "RL theta / force" : activeTaskMeta.stageFamily === "worm" ? "RL progress / force" : "RL returns / force"}</div>
                    <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "balance" ? `${formatSigned(rlFrame?.theta_deg, 2, "°")} · ${formatSigned(rlFrame?.total_force, 2)}` : activeTaskMeta.stageFamily === "worm" ? `${formatSigned(rlFrame?.x, 2, " m")} · ${formatSigned(rlFrame?.total_force, 2)}` : `${rlFrame?.task_counter ?? 0} · ${formatSigned(rlFrame?.total_force, 2)}`}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "balance" ? "angle gap" : activeTaskMeta.stageFamily === "worm" ? "progress gap" : "return gap"}</div>
                  <div className="cellengine-telemetry-value">{activeTaskMeta.stageFamily === "balance" ? formatSigned((frame?.theta_deg || 0) - (rlFrame?.theta_deg || 0), 2, "°") : activeTaskMeta.stageFamily === "worm" ? formatSigned((frame?.x || 0) - (rlFrame?.x || 0), 2, " m") : formatSigned((frame?.task_counter || 0) - (rlFrame?.task_counter || 0), 0)}</div>
                </div>
                  <div>
                    <div className="cellengine-telemetry-label">RL action / terminal</div>
                    <div className="cellengine-telemetry-value">{formatActionLabel(rlFrame?.action)} · {rlFrame?.terminal ? "yes" : "no"}</div>
                  </div>
                  <div>
                    <div className="cellengine-telemetry-label">{activeTaskMeta.stageFamily === "balance" ? "same starting angle" : activeTaskMeta.stageFamily === "worm" ? "same starting bend" : "same opening state"}</div>
                    <div className="cellengine-telemetry-value">
                      {activeTaskMeta.stageFamily === "balance" ? `${formatSigned(frames[0]?.theta_deg, 2, "°")} / ${formatSigned(rlFrames[0]?.theta_deg, 2, "°")}` : `${formatSigned(frames[0]?.theta_rad, 2)} / ${formatSigned(rlFrames[0]?.theta_rad, 2)}`}
                    </div>
                  </div>
                  <div>
                    <div className="cellengine-telemetry-label">survival so far</div>
                    <div className="cellengine-telemetry-value">
                      cell {frame?.tick ?? 0} · RL {rlFrame?.tick ?? 0}
                    </div>
                  </div>
                </div>
                <div className="cellengine-force-stack">
                  <ForceBar label="RL total force" value={rlFrame?.total_force} maxValue={rlAnalysis?.max_abs_total_force} colorClass="teacher" />
                  <ForceBar label="Cell total force" value={frame?.total_force} maxValue={analysis?.max_abs_total_force} colorClass="organism" />
                </div>
              </div>
            ) : (
              <div className="empty">Run a replay with an RL comparator to populate this panel.</div>
            )}
          </div>
        </div>
      </div>

      <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Selected Cell"
              help="Detailed inspection for the cell currently selected in the atlas or genome map, including live state, role, and local connectivity."
            />
            <div className="cellengine-inline-tools">
              <span className="meta-note">type, location, live state, genome expression, and current neighborhood links</span>
              <div className="cellengine-mode-toggle" role="tablist" aria-label="Selected cell inspector mode">
                <button
                  type="button"
                  className={`cellengine-mode-btn${selectedCellViewMode === "facts" ? " active" : ""}`}
                  onClick={() => setSelectedCellViewMode("facts")}
                >
                  Single Cell Parameters
                </button>
                <button
                  type="button"
                  className={`cellengine-mode-btn${selectedCellViewMode === "all" ? " active" : ""}`}
                  onClick={() => setSelectedCellViewMode("all")}
                >
                  All Cells
                </button>
                <button
                  type="button"
                  className={`cellengine-mode-btn${selectedCellViewMode === "visual" ? " active" : ""}`}
                  onClick={() => setSelectedCellViewMode("visual")}
                >
                  Single Cell Diagrams
                </button>
              </div>
            </div>
          </div>
          {selectedBodyCell && selectedCellState ? (
            <div className="stack">
              <div className="cellengine-selected-head">
                <div>
                  <div className="cellengine-telemetry-label">cell id</div>
                  <div className="cellengine-telemetry-value">{selectedBodyCell.cell_id}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">role</div>
                  <div className="cellengine-telemetry-value">{cellTypeLabel(selectedBodyCell)}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">grid position</div>
                  <div className="cellengine-telemetry-value">{formatBodyCellPosition(selectedBodyCell, includeDepthCoordinate)}</div>
                </div>
              </div>
              <div className="cellengine-chip-row">
                {selectedBodyCell.hinge ? <span className="landscape-pill">hinge sensor</span> : null}
                {selectedBodyCell.motor ? <span className="landscape-pill">motor effector</span> : null}
                {selectedBodyCell.ground ? <span className="landscape-pill">ground support</span> : null}
                {!selectedBodyCell.hinge && !selectedBodyCell.motor && !selectedBodyCell.ground ? <span className="landscape-pill">scaffold</span> : null}
                {selectedCellFunction ? (
                  <span
                    className="cellengine-state-pill"
                    style={{ background: selectedCellFunction.fill, color: selectedCellFunction.ink }}
                  >
                    now: {selectedCellFunction.label}
                  </span>
                ) : null}
              </div>
              {selectedCellViewMode === "all" ? (
                <SelectedAllCellsVisualGrid
                  bodyCells={bodyCells}
                  frameCellRows={currentFrameCellRows}
                  selectedCellId={selectedBodyCell?.cell_id ?? null}
                  onSelectCell={(cellId) => setSelectedCellId(cellId)}
                  functionalStateByCellId={functionalStateByCellId}
                  genomeMetric={genomeMetric}
                />
              ) : selectedCellViewMode === "visual" ? (
                <SelectedCellVisualInspector
                  cell={selectedBodyCell}
                  state={selectedCellState}
                  cellFunction={selectedCellFunction}
                  genomeMetric={genomeMetric}
                  selectedCellEdges={selectedCellEdges}
                  bodyCells={bodyCells}
                  functionalStateByCellId={functionalStateByCellId}
                />
              ) : (
                <>
                  <div className="cellengine-telemetry-grid">
                    <div>
                      <div className="cellengine-telemetry-label">dominant mode / confidence</div>
                      <div className="cellengine-telemetry-value">
                        {selectedCellFunction?.label || "n/a"} · {formatPct(selectedCellFunction?.confidence, 1)}
                      </div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">runner-up mode</div>
                      <div className="cellengine-telemetry-value">{selectedCellFunction?.runnerUpLabel || "n/a"}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">active / voltage</div>
                      <div className="cellengine-telemetry-value">{selectedCellState.active ? "yes" : "no"} · {formatSigned(selectedCellState.V, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">calcium / conductivity</div>
                      <div className="cellengine-telemetry-value">{formatNumber(selectedCellState.Ca, 3)} · {formatNumber(selectedCellState.sigma, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">energy / stress</div>
                      <div className="cellengine-telemetry-value">{formatNumber(selectedCellState.energy, 3)} · {formatNumber(selectedCellState.stress, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">activity / mech output</div>
                      <div className="cellengine-telemetry-value">{formatNumber(selectedCellState.activity, 3)} · {formatSigned(selectedCellState.a_mech, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">chemical output / recovery</div>
                      <div className="cellengine-telemetry-value">{formatSigned(selectedCellState.chem_out, 3)} · {formatNumber(selectedCellState.Wrec, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">mechanical leverage</div>
                      <div className="cellengine-telemetry-value">{formatNumber(selectedBodyCell.mech_advantage, 3)}</div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">coupling gene / adaptation gene</div>
                      <div className="cellengine-telemetry-value">
                        {formatNumber(selectedBodyCell.gene_expr_6, 3)} · {formatNumber(selectedBodyCell.gene_expr_7, 3)}
                      </div>
                    </div>
                    <div>
                      <div className="cellengine-telemetry-label">current genome metric</div>
                      <div className="cellengine-telemetry-value">{formatNumber(selectedBodyCell[genomeMetric], 3)}</div>
                    </div>
                  </div>
                  <div className="stack">
                    <div className="section-head">
                      <SectionTitle
                        title="Neighborhood Links"
                        help="Live gap-junction connections touching the selected cell in the current frame."
                      />
                      <span className="meta-note">current frame connections touching this cell</span>
                    </div>
                    {selectedCellEdges.length ? (
                      <div className="cellengine-link-list">
                        {selectedCellEdges.slice(0, 8).map((edge) => {
                          const neighborId = edge.src_cell_id === selectedBodyCell.cell_id ? edge.dst_cell_id : edge.src_cell_id;
                          return (
                            <div key={`${edge.src_cell_id}-${edge.dst_cell_id}`} className="cellengine-link-item">
                              <strong>cell {neighborId}</strong>
                              <span>
                                gap {formatNumber(edge.gap_mean, 3)} · dir {formatNumber(edge.gap_forward, 3)}/{formatNumber(edge.gap_reverse, 3)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="empty">No active links at this frame.</div>
                    )}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="empty">
              {bodyFollowsSelectedGenome
                ? "Selected genome body is shown, but live per-cell state is still from the loaded replay. Rerun this genome to inspect its actual telemetry."
                : "No per-cell state loaded yet."}
            </div>
          )}
      </div>

      <div className="grid two">
        <div className="card chart">
          <div className="section-head">
            <SectionTitle
              title={activeTaskMeta.stageFamily === "worm" ? "Crawl Timeline" : activeTaskMeta.stageFamily === "pong" ? "Rally Timeline" : "Balance Timeline"}
              help={activeTaskMeta.stageFamily === "worm"
                ? "Time series of forward progress and internal bend across the crawl replay."
                : activeTaskMeta.stageFamily === "pong"
                  ? "Time series of paddle position, ball position, and ball travel across the rally."
                  : "Time series of pole angle and cart displacement across the replay, with the current scrubber frame marked on the chart."}
            />
            <span className="meta-note">{activeTaskMeta.stageFamily === "worm" ? "progress and bend over time" : activeTaskMeta.stageFamily === "pong" ? "paddle, ball, and ball travel over time" : "cell and RL angle plus cart displacement over time"}</span>
          </div>
          {timelineRows.length ? (
            <div className="cellengine-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timelineRows}>
                  <CartesianGrid stroke="#e7ece9" strokeDasharray="3 3" />
                  <XAxis dataKey="tick" />
                  <YAxis yAxisId="angle" domain={activeTaskMeta.stageFamily === "balance" ? [-16, 16] : ["auto", "auto"]} />
                  <YAxis yAxisId="cart" orientation="right" domain={["auto", "auto"]} />
                  <Tooltip />
                  <Legend />
                  {activeTaskMeta.stageFamily === "balance" ? (
                    <>
                      <ReferenceLine yAxisId="angle" y={15} stroke="#c92a2a" strokeDasharray="6 6" />
                      <ReferenceLine yAxisId="angle" y={-15} stroke="#c92a2a" strokeDasharray="6 6" />
                      <ReferenceLine yAxisId="angle" y={2} stroke="#2b8a3e" strokeDasharray="4 4" />
                      <ReferenceLine yAxisId="angle" y={-2} stroke="#2b8a3e" strokeDasharray="4 4" />
                    </>
                  ) : null}
                  {analysis?.damage_ticks?.map((tick) => (
                    <ReferenceLine key={`damage-${tick}`} yAxisId="angle" x={tick} stroke="#f08c00" strokeDasharray="4 4" />
                  ))}
                  <ReferenceLine yAxisId="angle" x={frame?.tick ?? null} stroke="#1f2a30" strokeDasharray="2 6" />
                  {activeTaskMeta.stageFamily === "worm" ? (
                    <>
                      <Line yAxisId="angle" type="monotone" dataKey="theta_rad" name="cell bend" stroke="#d9480f" strokeWidth={2.4} dot={false} isAnimationActive={false} />
                      <Line yAxisId="angle" type="monotone" dataKey="rl_theta_rad" name="RL bend" stroke="#155e75" strokeWidth={2.2} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                      <Line yAxisId="cart" type="monotone" dataKey="x" name="cell progress" stroke="#0c8599" strokeWidth={2.1} dot={false} isAnimationActive={false} />
                      <Line yAxisId="cart" type="monotone" dataKey="rl_x" name="RL progress" stroke="#74c0fc" strokeWidth={1.8} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                    </>
                  ) : activeTaskMeta.stageFamily === "pong" ? (
                    <>
                      <Line yAxisId="angle" type="monotone" dataKey="theta_rad" name="ball y" stroke="#d9480f" strokeWidth={2.4} dot={false} isAnimationActive={false} />
                      <Line yAxisId="angle" type="monotone" dataKey="x" name="paddle y" stroke="#0c8599" strokeWidth={2.1} dot={false} isAnimationActive={false} />
                      <Line yAxisId="cart" type="monotone" dataKey="task_aux_a" name="ball x" stroke="#495057" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                      <Line yAxisId="angle" type="monotone" dataKey="rl_theta_rad" name="RL ball y" stroke="#155e75" strokeWidth={2.2} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                      <Line yAxisId="angle" type="monotone" dataKey="rl_x" name="RL paddle y" stroke="#74c0fc" strokeWidth={1.8} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                    </>
                  ) : (
                    <>
                      <Line yAxisId="angle" type="monotone" dataKey="theta_deg" name="cell theta (deg)" stroke="#d9480f" strokeWidth={2.4} dot={false} isAnimationActive={false} />
                      <Line yAxisId="angle" type="monotone" dataKey="rl_theta_deg" name="RL theta (deg)" stroke="#155e75" strokeWidth={2.2} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                      <Line yAxisId="cart" type="monotone" dataKey="x" name="cell cart x" stroke="#0c8599" strokeWidth={2.1} dot={false} isAnimationActive={false} />
                      <Line yAxisId="cart" type="monotone" dataKey="rl_x" name="RL cart x" stroke="#74c0fc" strokeWidth={1.8} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty">No trace loaded.</div>
          )}
        </div>

        <div className="card chart">
          <div className="section-head">
            <SectionTitle
              title="Actuation And Cell State"
              help="Joint timeline of controller forces and mean cell state so you can relate tissue behavior to actuation."
            />
            <span className="meta-note">{activeTaskMeta.stageFamily === "balance" ? "cell force decomposition with RL total force overlay" : "force, activity, and energy over the active task trace"}</span>
          </div>
          {timelineRows.length ? (
            <div className="cellengine-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineRows}>
                  <CartesianGrid stroke="#e7ece9" strokeDasharray="3 3" />
                  <XAxis dataKey="tick" />
                  <YAxis yAxisId="force" />
                  <YAxis yAxisId="cell" orientation="right" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <ReferenceLine yAxisId="force" x={frame?.tick ?? null} stroke="#1f2a30" strokeDasharray="2 6" />
                  <Area yAxisId="cell" type="monotone" dataKey="mean_energy" name="mean energy" stroke="#2b8a3e" fill="#d3f9d8" isAnimationActive={false} />
                  <Line yAxisId="cell" type="monotone" dataKey="active_fraction" name="active fraction" stroke="#495057" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line yAxisId="force" type="monotone" dataKey="organism_force" name="organism" stroke="#a61e4d" strokeWidth={2.1} dot={false} isAnimationActive={false} />
                  <Line yAxisId="force" type="monotone" dataKey="teacher_force" name="teacher" stroke="#1971c2" strokeWidth={1.8} dot={false} isAnimationActive={false} />
                  <Line yAxisId="force" type="monotone" dataKey="total_force" name="total" stroke="#111827" strokeWidth={2.4} dot={false} isAnimationActive={false} />
                  <Line yAxisId="force" type="monotone" dataKey="rl_total_force" name="RL total" stroke="#155e75" strokeWidth={2.1} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty">No trace loaded.</div>
          )}
        </div>
      </div>

      <div className="grid two">
        <div className="card chart">
          <div className="section-head">
            <SectionTitle
              title={activeTaskMeta.stageFamily === "worm" ? "Crawl Portrait" : activeTaskMeta.stageFamily === "pong" ? "Tracking Portrait" : "Phase Portrait"}
              help={activeTaskMeta.stageFamily === "worm"
                ? "Forward progress versus bend state across the replay."
                : activeTaskMeta.stageFamily === "pong"
                  ? "Paddle position versus ball position across the replay."
                  : "Pole angle versus angular velocity over the replay. Tight clusters indicate stable balance; wide excursions indicate instability."}
            />
            <span className="meta-note">{activeTaskMeta.stageFamily === "worm" ? "progress vs bend" : activeTaskMeta.stageFamily === "pong" ? "paddle vs ball position" : "cell and RL pole angle vs angular velocity"}</span>
          </div>
          {phaseRows.length ? (
            <div className="cellengine-chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid stroke="#e7ece9" strokeDasharray="3 3" />
                  <XAxis type="number" dataKey={activeTaskMeta.stageFamily === "worm" ? "x" : activeTaskMeta.stageFamily === "pong" ? "x" : "theta_deg"} name={activeTaskMeta.stageFamily === "worm" ? "progress" : activeTaskMeta.stageFamily === "pong" ? "paddle y" : "theta"} unit={activeTaskMeta.stageFamily === "balance" ? "deg" : undefined} />
                  <YAxis type="number" dataKey={activeTaskMeta.stageFamily === "worm" ? "theta_rad" : activeTaskMeta.stageFamily === "pong" ? "theta_rad" : "theta_dot"} name={activeTaskMeta.stageFamily === "worm" ? "bend" : activeTaskMeta.stageFamily === "pong" ? "ball y" : "theta dot"} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter data={phaseRows} name="cell" fill="#0c8599" opacity={0.4} isAnimationActive={false} />
                  {rlPhaseRows.length ? <Scatter data={rlPhaseRows} name="rl" fill="#d9480f" opacity={0.28} isAnimationActive={false} /> : null}
                  {frame ? <Scatter data={[frame]} fill="#d9480f" isAnimationActive={false} /> : null}
                  {rlFrame ? <Scatter data={[rlFrame]} fill="#155e75" isAnimationActive={false} /> : null}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty">No trace loaded.</div>
          )}
        </div>

        <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Replay Library"
              help="Quick selectors for saved genomes and cached replays so you can switch inputs without scrolling through long lists."
            />
            <span className="meta-note">load saved traces or switch champion genomes</span>
          </div>
          <ReplayLibrary
            genomeOptions={taskSpecificGenomeOptions}
            recentReplays={taskSpecificRecentReplays}
            loadingReplay={busyMode === "load"}
            runningReplay={replayBusy}
            onLoadReplay={loadReplay}
            onSelectGenome={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
            onRunGenome={runReplayFromGallery}
          />
        </div>
      </div>
    </>
  );
}
