import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { BioTissueGraph3D, LandscapeSurfaceCard } from "./Visualization3D";
import { formatPct } from "../lib/dashboardShared";

export default function EvaluationPanel({
  evalSplit,
  setEvalSplit,
  cmSplits,
  cmEpochChoice,
  setCmEpochChoice,
  cmEpochs,
  inferSplit,
  setInferSplit,
  inferSplits,
  calSplit,
  setCalSplit,
  calSplits,
  cmFilteredRows,
  inferFilteredRows,
  cmQuery,
  calQuery,
  inferQuery,
  classMetricsQuery,
  sysQuery,
  needContinuous,
  continuousEffQuery,
  continuousPhaseQuery,
  hybridExpertQuery,
  needBio,
  bioEpochQuery,
  bioLayerQuery,
  evaluationLandscape,
  openViewAllTab,
  cm,
  calibrationChartRows,
  histRows,
  errorRows,
  systemChartRows,
  classProfileRows,
  confidenceOutcomeRows,
  entropyConfidenceScatterRows,
  classRadarRows,
  runFamily,
  efficiencyFrontierRows,
  latestHybridExpertRows,
  latestBioEpochRow,
  bioTissueGraph,
  bioEvalLayerRows,
  hasLayerSignViolation
}) {
  return (
    <section className="panel stack">
      <div className="row controls">
            <label>
              Confusion split
              <select
                value={evalSplit}
                onChange={(event) => setEvalSplit(event.target.value)}
                disabled={!cmSplits.length}
              >
                {cmSplits.length ? (
                  cmSplits.map((split) => (
                    <option key={split} value={split}>
                      {split}
                    </option>
                  ))
                ) : (
                  <option value="">n/a</option>
                )}
              </select>
            </label>

            <label>
              Epoch
              <select
                value={cmEpochChoice}
                onChange={(event) => setCmEpochChoice(event.target.value)}
                disabled={!cmEpochs.length}
              >
                <option value="latest">latest</option>
                {cmEpochs.map((epoch) => (
                  <option key={epoch} value={String(epoch)}>
                    {epoch}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Inference split
              <select
                value={inferSplit}
                onChange={(event) => setInferSplit(event.target.value)}
                disabled={!inferSplits.length}
              >
                {inferSplits.length ? (
                  inferSplits.map((split) => (
                    <option key={split} value={split}>
                      {split}
                    </option>
                  ))
                ) : (
                  <option value="">n/a</option>
                )}
              </select>
            </label>

            <label>
              Calibration split
              <select
                value={calSplit}
                onChange={(event) => setCalSplit(event.target.value)}
                disabled={!calSplits.length}
              >
                {calSplits.length ? (
                  calSplits.map((split) => (
                    <option key={split} value={split}>
                      {split}
                    </option>
                  ))
                ) : (
                  <option value="">n/a</option>
                )}
              </select>
            </label>

            <span className="meta-note">
              matrix rows: {cmFilteredRows.length || 0}, inferred samples: {inferFilteredRows.length || 0}
            </span>
          </div>

      {cmQuery.isLoading ||
      calQuery.isLoading ||
      inferQuery.isLoading ||
      classMetricsQuery.isLoading ||
      sysQuery.isLoading ||
      (needContinuous &&
        continuousEffQuery.isLoading &&
        continuousPhaseQuery.isLoading &&
        hybridExpertQuery.isLoading) ||
      (needBio && bioEpochQuery.isLoading && bioLayerQuery.isLoading) ? (
        <div className="empty">Loading evaluation metrics...</div>
      ) : (
        <>
          <LandscapeSurfaceCard
            title="3D Generalization Landscape (Evaluation)"
            subtitle="Test-surface basin view from epoch and inference signals."
            landscape={evaluationLandscape}
            emptyMessage="Not enough evaluation points yet to construct the 3D landscape."
            onViewAll={() => openViewAllTab("eval_landscape")}
          />

              <div className="grid two">
                <div className="card">
                  <div className="section-head">
                    <h3>Confusion Matrix</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_confusion_matrix")}>
                      View all architectures
                    </button>
                  </div>
                  {cm.classes.length ? (
                    <div className="matrix-wrap">
                      <table className="matrix">
                        <thead>
                          <tr>
                            <th>t\p</th>
                            {cm.classes.map((cls) => (
                              <th key={`h-${cls}`}>{cls}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {cm.matrix.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              <th>{cm.classes[rowIdx]}</th>
                              {row.map((value, colIdx) => {
                                const alpha = cm.maxValue > 0 ? value / cm.maxValue : 0;
                                const bg =
                                  value === 0
                                    ? "rgba(12,133,153,0.04)"
                                    : `rgba(12,133,153,${Math.max(0.08, alpha)})`;
                                return (
                                  <td key={colIdx} style={{ background: bg }}>
                                    {value}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty">No confusion data for selected split/epoch.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Calibration</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_calibration")}>
                      View all architectures
                    </button>
                  </div>
                  {calibrationChartRows.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={calibrationChartRows}>
                        <CartesianGrid strokeDasharray="4 4" />
                        <XAxis dataKey="center" domain={[0, 1]} type="number" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Legend />
                        <ReferenceLine
                          segment={[
                            { x: 0, y: 0 },
                            { x: 1, y: 1 }
                          ]}
                          stroke="#6c757d"
                          strokeDasharray="4 4"
                        />
                        <Line dataKey="empirical_acc" stroke="#2b8a3e" dot={false} />
                        <Line dataKey="avg_conf" stroke="#a61e4d" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No calibration data for selected split.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Latency Distribution</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_latency")}>
                      View all architectures
                    </button>
                  </div>
                  {histRows.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={histRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="label" hide />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#e67700" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No inference latency data for selected split.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Top Misclassifications</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_misclassifications")}>
                      View all architectures
                    </button>
                  </div>
                  {errorRows.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={errorRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="pair" angle={-35} textAnchor="end" height={70} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#c2255c" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No misclassifications for selected split.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>System Metrics</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_system")}>
                      View all architectures
                    </button>
                  </div>
                  {systemChartRows.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={systemChartRows}>
                        <CartesianGrid strokeDasharray="4 4" />
                        <XAxis dataKey="index" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip />
                        <Legend />
                        <Line yAxisId="left" dataKey="qps" stroke="#0c8599" dot={false} />
                        <Line yAxisId="right" dataKey="p95_ms" stroke="#a61e4d" dot={false} />
                        <Line yAxisId="right" dataKey="p50_ms" stroke="#5c940d" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No system metrics rows found.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Per-Class Precision / Recall / F1</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_interp_class_profile")}>
                      View all architectures
                    </button>
                  </div>
                  {classProfileRows.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={classProfileRows}>
                        <CartesianGrid strokeDasharray="4 4" />
                        <XAxis dataKey="class_id" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Legend />
                        <Line dataKey="precision" stroke="#0c8599" dot={false} />
                        <Line dataKey="recall" stroke="#2b8a3e" dot={false} />
                        <Line dataKey="f1" stroke="#a61e4d" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No class-level metrics rows found.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Confidence vs Error</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_interp_confidence_error")}>
                      View all architectures
                    </button>
                  </div>
                  {confidenceOutcomeRows.length ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={confidenceOutcomeRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="label" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="correct" fill="#2b8a3e" />
                        <Bar dataKey="incorrect" fill="#c92a2a" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No confidence/error rows for selected split.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Entropy vs Confidence</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_interp_confidence_error")}>
                      View all architectures
                    </button>
                  </div>
                  {entropyConfidenceScatterRows.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="4 4" />
                        <XAxis type="number" dataKey="confidence" domain={[0, 1]} name="Confidence" />
                        <YAxis type="number" dataKey="entropy" name="Entropy" />
                        <Tooltip />
                        <Legend />
                        <Scatter
                          data={entropyConfidenceScatterRows.filter((row) => row.correctness === "correct")}
                          fill="#2b8a3e"
                          name="correct"
                        />
                        <Scatter
                          data={entropyConfidenceScatterRows.filter((row) => row.correctness === "incorrect")}
                          fill="#c92a2a"
                          name="incorrect"
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No entropy/confidence rows for selected split.</div>
                  )}
                </div>

                <div className="card">
                  <div className="section-head">
                    <h3>Class Profile Radar (Top Support Classes)</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_interp_class_profile")}>
                      View all architectures
                    </button>
                  </div>
                  {classRadarRows.length ? (
                    <ResponsiveContainer width="100%" height={260}>
                      <RadarChart data={classRadarRows}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="class_label" />
                        <PolarRadiusAxis domain={[0, 1]} />
                        <Legend />
                        <Radar name="precision" dataKey="precision" stroke="#0c8599" fill="#0c859933" />
                        <Radar name="recall" dataKey="recall" stroke="#2b8a3e" fill="#2b8a3e33" />
                        <Radar name="f1" dataKey="f1" stroke="#a61e4d" fill="#a61e4d33" />
                      </RadarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty">No class-profile data for radar view.</div>
                  )}
                </div>
              </div>

              {(runFamily === "continuous" || runFamily === "hybrid" || efficiencyFrontierRows.length > 0) &&
                (efficiencyFrontierRows.length ? (
                  <div className="card">
                    <div className="section-head">
                      <h3>Sample-Efficiency Frontier</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_sample_efficiency")}>
                        View all architectures
                      </button>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={efficiencyFrontierRows}>
                        <CartesianGrid strokeDasharray="4 4" />
                        <XAxis dataKey="samples" type="number" />
                        <YAxis domain={[0, 1]} />
                        <Tooltip />
                        <Legend />
                        <Line dataKey="test_accuracy" stroke="#0c8599" dot={false} />
                        <Line dataKey="best_accuracy" stroke="#2b8a3e" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="empty">No sample-efficiency data found (`continuous_efficiency.csv`).</div>
                ))}

              {(runFamily === "hybrid" || latestHybridExpertRows.length > 0) &&
                (latestHybridExpertRows.length ? (
                  <div className="card">
                    <div className="section-head">
                      <h3>Hybrid Expert Fusion Strength (Latest Cycle)</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_hybrid_expert")}>
                        View all architectures
                      </button>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={latestHybridExpertRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="expert" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="fusion_strength" fill="#a61e4d" />
                        <Bar dataKey="expert_accuracy" fill="#0c8599" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="empty">No hybrid expert metrics found for this run.</div>
                ))}

              <div className="card">
                <h3>Bio Evaluation Snapshot</h3>
                {latestBioEpochRow ? (
                  <div className="card-grid">
                    <span>epoch</span>
                    <span>{latestBioEpochRow.epoch}</span>
                    <span>energy used</span>
                    <span>{Number(latestBioEpochRow.energy_used || 0).toFixed(4)}</span>
                    <span>wiring cost</span>
                    <span>{Number(latestBioEpochRow.wiring_cost || 0).toFixed(4)}</span>
                    <span>mean delay</span>
                    <span>{Number(latestBioEpochRow.mean_delay || 0).toFixed(4)}</span>
                    <span>bioelectric mean</span>
                    <span>{Number(latestBioEpochRow.bioelectric_mean || 0).toFixed(4)}</span>
                    <span>homeostasis error</span>
                    <span>{Number(latestBioEpochRow.homeostasis_error || 0).toFixed(4)}</span>
                    <span>sign violation</span>
                    <span>{Number(latestBioEpochRow.sign_violation_fraction || 0).toFixed(4)}</span>
                    <span>pruned / grown</span>
                    <span>
                      {latestBioEpochRow.total_pruned || 0} / {latestBioEpochRow.total_grown || 0}
                    </span>
                  </div>
              ) : (
                <div className="empty">No bio evaluation telemetry for this run.</div>
              )}
              </div>

              <BioTissueGraph3D
                graph={bioTissueGraph}
                title="3D Bio Tissue Connectivity (Latest Epoch)"
                subtitle="Spatial neuron layout with structural, myelin and bioelectric state overlays."
                onViewAll={() => openViewAllTab("eval_bio_tissue_3d")}
              />

              {bioEvalLayerRows.length ? (
                <div className="grid two">
                  <div className="card">
                    <div className="section-head">
                      <h3>Bio Structure By Layer (Latest Epoch)</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_bio_structure")}>
                        View all architectures
                      </button>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={bioEvalLayerRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="layer" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="mask_density" fill="#2b8a3e" />
                        <Bar dataKey="myelin_fraction" fill="#e67700" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="card">
                    <div className="section-head">
                      <h3>Bio State By Layer (Latest Epoch)</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_bio_state")}>
                        View all architectures
                      </button>
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={bioEvalLayerRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="layer" hide />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="mean_delay" fill="#0c8599" />
                        <Bar dataKey="bioelectric_mean" fill="#c2255c" />
                        <Bar dataKey="homeostasis_error" fill="#5c940d" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="card">
                    <div className="section-head">
                      <h3>E/I Sign Constraint Integrity (Latest Epoch)</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("eval_bio_ei_sign")}>
                        View all architectures
                      </button>
                    </div>
                    {hasLayerSignViolation ? (
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={bioEvalLayerRows}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="layer" hide />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="sign_violation_fraction" fill="#c92a2a" />
                          <Bar dataKey="excitatory_fraction" fill="#2b8a3e" />
                          <Bar dataKey="inhibitory_fraction" fill="#364fc7" />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty">
                        Sign-constraint telemetry not available in this run. Re-run benchmark with updated binaries.
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="empty">No layer-level bio rows available for latest epoch.</div>
              )}
        </>
      )}
    </section>
  );
}
