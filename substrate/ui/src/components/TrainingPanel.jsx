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
import { BioTissueGraph3D, LandscapeSurfaceCard } from "./Visualization3D";
import { elapsedSince, formatPct, runStatus } from "../lib/dashboardShared";

export default function TrainingPanel({
  epochQuery,
  batchQuery,
  bioEpochQuery,
  bioLayerQuery,
  continuousEffQuery,
  continuousPhaseQuery,
  hybridExpertQuery,
  trainingLandscape,
  openViewAllTab,
  batchSeries,
  selectedRun,
  epochPivot,
  lossLines,
  accuracyLines,
  throughputLines,
  latestBioEpochRow,
  bioEpochSeries,
  bioTissueGraph,
  bioLayerBioelectricSeries,
  hasEpochSignViolation,
  hasLayerSignViolation,
  runFamily,
  latestContinuousEffRow,
  continuousEffSeries,
  continuousPhaseLossSeries,
  continuousPhaseAccSeries
}) {
  return (
    <section className="panel stack">
      {epochQuery.isLoading &&
      batchQuery.isLoading &&
      bioEpochQuery.isLoading &&
      bioLayerQuery.isLoading &&
      continuousEffQuery.isLoading &&
      continuousPhaseQuery.isLoading &&
      hybridExpertQuery.isLoading ? (
        <div className="empty">Loading training metrics...</div>
      ) : (
        <>
          <LandscapeSurfaceCard
            title="3D Optimization Landscape (Training)"
            subtitle="Kernel-smoothed loss surface with live learning trajectory and detected local minima."
            landscape={trainingLandscape}
            emptyMessage="Not enough training points yet to construct the 3D landscape."
            onViewAll={() => openViewAllTab("train_landscape")}
          />

              {batchSeries.length ? (
                <div className="chart">
                  <div className="section-head">
                    <h3>Live Batch Loss / Accuracy</h3>
                    <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_live_batch")}>
                      View all architectures
                    </button>
                  </div>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={batchSeries}>
                      <CartesianGrid strokeDasharray="4 4" />
                      <XAxis dataKey="global_step" />
                      <YAxis yAxisId="loss" />
                      <YAxis yAxisId="acc" orientation="right" domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line yAxisId="loss" type="monotone" dataKey="loss" dot={false} stroke="#a61e4d" />
                      <Line yAxisId="acc" type="monotone" dataKey="accuracy" dot={false} stroke="#2b8a3e" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : runStatus(selectedRun) === "starting" ? (
                <div className="empty">
                  Run is starting. Waiting for first batch-metric points...
                </div>
              ) : null}

              {epochPivot.length ? (
                <>
                  <div className="chart">
                    <div className="section-head">
                      <h3>Loss By Epoch</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_epoch_loss")}>
                        View all architectures
                      </button>
                    </div>
                    {lossLines.length ? (
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={epochPivot}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          {lossLines.map((line) => (
                            <Line key={line.key} dataKey={line.key} name={line.label} dot={false} stroke={line.color} />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty">No numeric loss values for this run.</div>
                    )}
                  </div>

                  <div className="chart">
                    <div className="section-head">
                      <h3>Accuracy By Epoch</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_epoch_accuracy")}>
                        View all architectures
                      </button>
                    </div>
                    {accuracyLines.length ? (
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={epochPivot}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          {accuracyLines.map((line) => (
                            <Line key={line.key} dataKey={line.key} name={line.label} dot={false} stroke={line.color} />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty">No numeric accuracy values for this run.</div>
                    )}
                  </div>

                  <div className="chart">
                    <div className="section-head">
                      <h3>Throughput</h3>
                      <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_epoch_throughput")}>
                        View all architectures
                      </button>
                    </div>
                    {throughputLines.length ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={epochPivot}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          {throughputLines.map((line) => (
                            <Line key={line.key} dataKey={line.key} name={line.label} dot={false} stroke={line.color} />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty">No throughput values for this run.</div>
                    )}
                  </div>
                </>
              ) : (
                <div className="empty">
                  {runStatus(selectedRun) === "starting"
                    ? `Run is live and still in first epoch (${elapsedSince(selectedRun?.train_start_utc)} elapsed). Epoch curves appear once epoch 1 completes.`
                    : "No learning/epoch_metrics.csv rows available for this run."}
                </div>
              )}

              <div className="card">
                <h3>Bio Training Snapshot</h3>
                {latestBioEpochRow ? (
                  <div className="card-grid">
                    <span>epoch</span>
                    <span>{latestBioEpochRow.epoch}</span>
                    <span>energy used</span>
                    <span>{Number(latestBioEpochRow.energy_used || 0).toFixed(4)}</span>
                    <span>wiring cost</span>
                    <span>{Number(latestBioEpochRow.wiring_cost || 0).toFixed(4)}</span>
                    <span>mask density</span>
                    <span>{Number(latestBioEpochRow.mask_density || 0).toFixed(4)}</span>
                    <span>myelin fraction</span>
                    <span>{Number(latestBioEpochRow.myelin_fraction || 0).toFixed(4)}</span>
                    <span>mean delay</span>
                    <span>{Number(latestBioEpochRow.mean_delay || 0).toFixed(4)}</span>
                    <span>homeostasis error</span>
                    <span>{Number(latestBioEpochRow.homeostasis_error || 0).toFixed(4)}</span>
                    <span>modulator mean</span>
                    <span>{Number(latestBioEpochRow.modulator_mean || 0).toFixed(4)}</span>
                    <span>sign violation</span>
                    <span>{Number(latestBioEpochRow.sign_violation_fraction || 0).toFixed(4)}</span>
                  </div>
                ) : (
                  <div className="empty">No bio epoch telemetry for this run.</div>
                )}
              </div>

              {bioEpochSeries.length ? (
                <>
                  <BioTissueGraph3D
                    graph={bioTissueGraph}
                    title="3D Bio Tissue Connectivity"
                    subtitle="Neurons are arranged by layer-depth in 3D; hover a node to isolate only its connectivity."
                    onViewAll={() => openViewAllTab("train_bio_tissue_3d")}
                  />

                  <div className="grid two">
                    <div className="card">
                      <div className="section-head">
                        <h3>Bio Energy And Wiring</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_energy")}>
                          View all architectures
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={bioEpochSeries}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line dataKey="energy_used" stroke="#a61e4d" dot={false} />
                          <Line dataKey="wiring_cost" stroke="#0c8599" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>Structural Plasticity</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_structural")}>
                          View all architectures
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={bioEpochSeries}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line dataKey="total_pruned" stroke="#c2255c" dot={false} />
                          <Line dataKey="total_grown" stroke="#2b8a3e" dot={false} />
                          <Line dataKey="mask_density" stroke="#0c8599" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>Myelination And Delay</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_myelin_delay")}>
                          View all architectures
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={bioEpochSeries}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line dataKey="myelin_fraction" stroke="#e67700" dot={false} />
                          <Line dataKey="mean_delay" stroke="#495057" dot={false} />
                          <Line dataKey="wiring_cost" stroke="#0c8599" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>Homeostasis + 3-Factor Plasticity</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_regulation")}>
                          View all architectures
                        </button>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={bioEpochSeries}>
                          <CartesianGrid strokeDasharray="4 4" />
                          <XAxis dataKey="epoch" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line dataKey="homeostasis_error" stroke="#c2255c" dot={false} />
                          <Line dataKey="plasticity_update_mean_abs" stroke="#0b7285" dot={false} />
                          <Line dataKey="modulator_mean" stroke="#5c940d" dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>Bioelectric Dynamics</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_bioelectric")}>
                          View all architectures
                        </button>
                      </div>
                      {bioLayerBioelectricSeries.data.length ? (
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={bioLayerBioelectricSeries.data}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="x" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            {bioLayerBioelectricSeries.lines.map((line) => (
                              <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={bioEpochSeries}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="epoch" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line dataKey="bioelectric_mean" stroke="#d6336c" dot={false} />
                            <Line dataKey="modulator_mean" stroke="#5c940d" dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>E/I Sign Constraint Integrity</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_bio_ei_sign")}>
                          View all architectures
                        </button>
                      </div>
                      {(hasEpochSignViolation || hasLayerSignViolation) ? (
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={bioEpochSeries}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="epoch" />
                            <YAxis domain={[0, 1]} />
                            <Tooltip />
                            <Legend />
                            <Line dataKey="sign_violation_fraction" stroke="#c92a2a" dot={false} />
                            <Line dataKey="homeostasis_error" stroke="#1864ab" dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="empty">
                          Sign-constraint telemetry not available in this run. New runs will include this metric.
                        </div>
                      )}
                    </div>

                    <div className="card">
                      <div className="section-head">
                        <h3>Layer Mask Density</h3>
                        <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_layer_mask_density")}>
                          View all architectures
                        </button>
                      </div>
                      {bioLayerMaskSeries.data.length ? (
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={bioLayerMaskSeries.data}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="x" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            {bioLayerMaskSeries.lines.map((line) => (
                              <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="empty">No layer-level bio mask data available.</div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty">No bio training series found (`bio_epoch_dynamics.csv`).</div>
              )}

              {(runFamily === "continuous" || runFamily === "hybrid" || continuousEffSeries.length > 0) && (
                <>
                  <div className="card">
                    <h3>Continuous Learning Snapshot</h3>
                    {latestContinuousEffRow ? (
                      <div className="card-grid">
                        <span>cycle</span>
                        <span>{latestContinuousEffRow.cycle}</span>
                        <span>test accuracy</span>
                        <span>{formatPct(latestContinuousEffRow.test_accuracy)}</span>
                        <span>best accuracy</span>
                        <span>{formatPct(latestContinuousEffRow.best_accuracy)}</span>
                        <span>cumulative samples</span>
                        <span>{Number(latestContinuousEffRow.cumulative_samples || 0).toLocaleString()}</span>
                        <span>samples / acc-point</span>
                        <span>{Number(latestContinuousEffRow.samples_per_accuracy_point || 0).toFixed(2)}</span>
                        <span>target reached</span>
                        <span>{latestContinuousEffRow.reached_target === 1 ? "yes" : "no"}</span>
                      </div>
                    ) : (
                      <div className="empty">No continuous efficiency telemetry for this run.</div>
                    )}
                  </div>

                  {continuousEffSeries.length ? (
                    <div className="grid two">
                      <div className="card">
                        <div className="section-head">
                          <h3>Continuous Accuracy Trajectory</h3>
                          <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_continuous_accuracy")}>
                            View all architectures
                          </button>
                        </div>
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={continuousEffSeries}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="cycle" />
                            <YAxis domain={[0, 1]} />
                            <Tooltip />
                            <Legend />
                            <Line dataKey="test_accuracy" stroke="#0c8599" dot={false} />
                            <Line dataKey="best_accuracy" stroke="#2b8a3e" dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="card">
                        <div className="section-head">
                          <h3>Samples And Cycle Cost</h3>
                          <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_continuous_samples")}>
                            View all architectures
                          </button>
                        </div>
                        <ResponsiveContainer width="100%" height={260}>
                          <LineChart data={continuousEffSeries}>
                            <CartesianGrid strokeDasharray="4 4" />
                            <XAxis dataKey="cycle" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line dataKey="cumulative_samples" stroke="#a61e4d" dot={false} />
                            <Line dataKey="cycle_time_ms" stroke="#e67700" dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="card">
                        <div className="section-head">
                          <h3>Phase Loss (Active vs Sleep)</h3>
                          <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_phase_loss")}>
                            View all architectures
                          </button>
                        </div>
                        {continuousPhaseLossSeries.data.length ? (
                          <ResponsiveContainer width="100%" height={260}>
                            <LineChart data={continuousPhaseLossSeries.data}>
                              <CartesianGrid strokeDasharray="4 4" />
                              <XAxis dataKey="x" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              {continuousPhaseLossSeries.lines.map((line) => (
                                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="empty">No phase loss rows available.</div>
                        )}
                      </div>

                      <div className="card">
                        <div className="section-head">
                          <h3>Phase Accuracy (Active vs Sleep)</h3>
                          <button className="mini-btn" type="button" onClick={() => openViewAllTab("train_phase_accuracy")}>
                            View all architectures
                          </button>
                        </div>
                        {continuousPhaseAccSeries.data.length ? (
                          <ResponsiveContainer width="100%" height={260}>
                            <LineChart data={continuousPhaseAccSeries.data}>
                              <CartesianGrid strokeDasharray="4 4" />
                              <XAxis dataKey="x" />
                              <YAxis domain={[0, 1]} />
                              <Tooltip />
                              <Legend />
                              {continuousPhaseAccSeries.lines.map((line) => (
                                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="empty">No phase accuracy rows available.</div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="empty">No continuous learning series found (`continuous_efficiency.csv`).</div>
                  )}
                </>
              )}
        </>
      )}
    </section>
  );
}
