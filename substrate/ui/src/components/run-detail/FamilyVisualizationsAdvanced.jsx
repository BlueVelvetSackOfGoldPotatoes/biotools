import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  YAxis,
  ZAxis
} from "recharts";
import {
  FAMILY_COLORS,
  buildGroupedSeries,
  chooseAutoPreviewColumns,
  confusionMatrix,
  formatPct,
  pivotEpoch,
  runStatusClass,
  runStatusLabel,
  seriesColor,
  topMisclassifications
} from "../../lib/dashboardShared";

function ClusteringVisualizations({ data }) {
  const centroid = data.centroid_shift || [];
  const metrics = data.cluster_metrics || [];
  const clusterSizes = data.cluster_sizes || [];


  const shiftSeries = buildGroupedSeries(centroid, "step", "shift_l2", "cluster_id", 8);
  const inertiaSeries = metrics
    .filter((r) => typeof r.step === "number" && typeof r.inertia === "number" && Number.isFinite(r.inertia))
    .sort((a, b) => a.step - b.step);

  // Cluster size distribution - show latest snapshot per algorithm
  const sizesByAlgo = {};
  clusterSizes.forEach((r) => {
    const key = r.algorithm || "unknown";
    if (!sizesByAlgo[key]) sizesByAlgo[key] = [];
    sizesByAlgo[key].push({ cluster: `C${r.cluster_id}`, size: r.size, fraction: r.fraction });
  });
  const firstAlgo = Object.keys(sizesByAlgo)[0];
  const sizeData = firstAlgo ? sizesByAlgo[firstAlgo] : [];

  // Purity over iterations if available
  const puritySeries = metrics
    .filter((r) => typeof r.step === "number" && typeof r.purity === "number" && Number.isFinite(r.purity))
    .sort((a, b) => a.step - b.step);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Centroid Shift by Cluster</h4>
        {shiftSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={shiftSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Step", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: "L2 Shift", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {shiftSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No centroid shift data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Inertia Over Steps</h4>
        {inertiaSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={inertiaSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="inertia" stroke="#862e9c" fill="#862e9c22" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No inertia data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Cluster Size Distribution{firstAlgo ? ` (${firstAlgo})` : ""}</h4>
        {sizeData.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={sizeData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="cluster" />
              <YAxis />
              <Tooltip formatter={(v) => typeof v === "number" ? v.toLocaleString() : v} />
              <Bar dataKey="size" fill="#5c940d">
                {sizeData.map((_, i) => (
                  <Cell key={i} fill={seriesColor(i)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No cluster size data.</div>}
      </div>
      {puritySeries.length > 0 && (
        <div className="detail-viz-card">
          <h4>Cluster Purity</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={puritySeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <ReferenceLine y={1.0} stroke="#888" strokeDasharray="3 3" label="Perfect" />
              <Line dataKey="purity" stroke="#862e9c" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function HebbianVisualizations({ data }) {
  const energy = data.energy_curve || [];
  const hebbUpdate = data.hebb_update || [];
  const weightNorms = data.weight_feature_norms || [];

  const energySeries = energy
    .filter((r) => typeof r.step === "number" && typeof r.energy === "number" && Number.isFinite(r.energy))
    .sort((a, b) => a.step - b.step);

  const updateSeries = buildGroupedSeries(hebbUpdate, "step", "delta_w_l2", "layer", 6);

  // Pixel-level weight norms (learned feature receptive fields) — show latest pass
  const passes = [...new Set(weightNorms.map((r) => r.pass))].sort((a, b) => b - a);
  const latestPass = passes[0];
  const pixelData = weightNorms
    .filter((r) => r.pass === latestPass && typeof r.row === "number" && typeof r.col === "number")
    .map((r) => ({ row: r.row, col: r.col, norm: r.weight_norm }))
    .sort((a, b) => b.norm - a.norm)
    .slice(0, 80);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Hopfield Energy (Pattern Recall)</h4>
        {energySeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={energySeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: "Energy", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Area type="monotone" dataKey="energy" stroke="#c92a2a" fill="#c92a2a22" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No energy data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Weight Update Magnitude (delta_w L2)</h4>
        {updateSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={updateSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Pass", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {updateSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No update data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Learned Feature Map (Weight Norms){latestPass ? ` — Pass ${latestPass}` : ""}</h4>
        {pixelData.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="col" domain={[0, 27]} name="Col" />
              <YAxis type="number" dataKey="row" domain={[0, 27]} name="Row" reversed />
              <ZAxis type="number" dataKey="norm" range={[20, 200]} name="Norm" />
              <Tooltip />
              <Scatter data={pixelData} fill="#c92a2a" fillOpacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        ) : <div className="empty">No weight feature norm data.</div>}
      </div>
    </div>
  );
}

function ActorCriticVisualizations({ data }) {
  const rl = data.rl_train || [];
  const valueStats = data.value_stats || [];

  const series = rl
    .filter((r) => typeof r.step === "number")
    .sort((a, b) => a.step - b.step);

  // Value stats by algorithm
  const valueSeries = buildGroupedSeries(valueStats, "step", "reward_mean", "algorithm", 4);

  // Entropy by algorithm
  const entropySeries = buildGroupedSeries(valueStats, "step", "entropy", "algorithm", 4);

  // Reward vs entropy ratio (exploration-exploitation)
  const ratioData = valueStats
    .filter((r) => typeof r.step === "number" && typeof r.reward_entropy_ratio === "number" && Number.isFinite(r.reward_entropy_ratio))
    .sort((a, b) => a.step - b.step);

  if (!series.length && !valueStats.length) return <div className="empty">No RL training data.</div>;

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Mean Reward Over Steps</h4>
        {series.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Line dataKey="reward_mean" stroke="#1864ab" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        ) : valueSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={valueSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {valueSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No reward data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Policy Entropy</h4>
        {entropySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={entropySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {entropySeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : series.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={series}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="entropy" stroke="#e67700" fill="#e6770022" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No entropy data.</div>}
      </div>
      {ratioData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Reward / Entropy Ratio (Exploitation Score)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={ratioData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <ReferenceLine y={1.0} stroke="#888" strokeDasharray="3 3" label="Balance" />
              <Area type="monotone" dataKey="reward_entropy_ratio" stroke="#2b8a3e" fill="#2b8a3e22" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function DiffusionVisualizations({ data }) {
  const timestepLoss = data.timestep_loss || [];
  const sampleStats = data.sample_stats || [];
  const noiseSchedule = data.noise_schedule || [];
  const forwardProcess = data.forward_process || [];
  const reverseProcess = data.reverse_process || [];
  const perTimestepLoss = data.per_timestep_loss || [];

  // Noise schedule: alpha_bar and SNR vs timestep
  const scheduleSeries = noiseSchedule
    .filter((r) => typeof r.timestep === "number")
    .sort((a, b) => a.timestep - b.timestep);

  // Forward process: pixel stats as image gets noisier (latest step snapshot)
  const fwdSteps = [...new Set(forwardProcess.map((r) => r.step))].sort((a, b) => b - a);
  const latestFwdStep = fwdSteps[0];
  const fwdSeries = forwardProcess
    .filter((r) => r.step === latestFwdStep && typeof r.timestep === "number")
    .sort((a, b) => a.timestep - b.timestep);

  // Reverse process: pixel stats as noise gets denoised (latest step snapshot)
  const revSteps = [...new Set(reverseProcess.map((r) => r.step))].sort((a, b) => b - a);
  const latestRevStep = revSteps[0];
  const revSeries = reverseProcess
    .filter((r) => r.step === latestRevStep && typeof r.timestep === "number")
    .sort((a, b) => a.timestep - b.timestep);

  // Sample quality over training
  const stdSeries = sampleStats
    .filter((r) => typeof r.step === "number" && typeof r.std === "number")
    .sort((a, b) => a.step - b.step);

  // Per-timestep difficulty (SNR curve at latest checkpoint)
  const diffSteps = [...new Set(perTimestepLoss.map((r) => r.step))].sort((a, b) => b - a);
  const latestDiffStep = diffSteps[0];
  const difficultySeries = perTimestepLoss
    .filter((r) => r.step === latestDiffStep && typeof r.timestep === "number")
    .sort((a, b) => a.timestep - b.timestep);

  const lossSeries = buildGroupedSeries(timestepLoss, "timestep", "loss", "step_or_epoch", 6);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Noise Schedule (alpha_bar &amp; SNR)</h4>
        {scheduleSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={scheduleSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="timestep" label={{ value: "Timestep t", position: "insideBottom", offset: -2 }} />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" dataKey="alpha_bar" stroke="#1864ab" dot={false} strokeWidth={2} name="alpha_bar" />
              <Line yAxisId="left" dataKey="noise_level" stroke="#e8590c" dot={false} strokeWidth={2} name="noise_level" />
              <Line yAxisId="right" dataKey="snr" stroke="#5c940d" dot={false} strokeDasharray="5 5" name="SNR" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No noise schedule data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Forward Process (Image → Noise){latestFwdStep ? ` @ step ${latestFwdStep}` : ""}</h4>
        {fwdSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={fwdSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="timestep" label={{ value: "Timestep t", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="pixel_mean" stroke="#1864ab" fill="#1864ab22" name="Mean" />
              <Area type="monotone" dataKey="pixel_std" stroke="#e8590c" fill="#e8590c22" name="Std" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No forward process data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Reverse Process (Noise → Image){latestRevStep ? ` @ step ${latestRevStep}` : ""}</h4>
        {revSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={revSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="timestep" label={{ value: "Denoising step", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="pixel_mean" stroke="#2b8a3e" fill="#2b8a3e22" name="Mean" />
              <Area type="monotone" dataKey="pixel_std" stroke="#862e9c" fill="#862e9c22" name="Std" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No reverse process data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Sample Quality Over Training</h4>
        {stdSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={stdSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" label={{ value: "Training Step", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line dataKey="std" stroke="#e8590c" dot name="Std Dev" />
              <Line dataKey="mean" stroke="#1864ab" dot name="Mean" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No sample stats.</div>}
      </div>
      {difficultySeries.length > 0 && (
        <div className="detail-viz-card">
          <h4>Per-Timestep Difficulty (SNR){latestDiffStep ? ` @ step ${latestDiffStep}` : ""}</h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={difficultySeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="timestep" label={{ value: "Timestep t", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="difficulty" stroke="#c92a2a" fill="#c92a2a22" name="Difficulty (1-ᾱ)" />
              <Area type="monotone" dataKey="snr" stroke="#5c940d" fill="#5c940d22" name="SNR" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      {lossSeries.data.length > 0 && (
        <div className="detail-viz-card">
          <h4>Denoising Loss by Timestep</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lossSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {lossSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function GnnVisualizations({ data }) {
  const msgNorms = data.message_norms || [];
  const embeddings = data.embedding_2d || [];
  const layerStats = data.layer_stats || [];

  const normSeries = buildGroupedSeries(msgNorms, "epoch", "msg_norm_mean", "layer", 6);

  // Layer energy over epochs
  const energySeries = buildGroupedSeries(layerStats, "epoch", "energy", "layer", 6);

  // Layer sparsity over epochs
  const sparsitySeries = buildGroupedSeries(layerStats, "epoch", "sparsity", "layer", 6);

  // 2D embedding scatter (color by class)
  const scatterData = embeddings
    .filter((r) => typeof r.x === "number" && typeof r.y === "number")
    .slice(0, 500);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Message Norm by Layer</h4>
        {normSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={normSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: "Norm", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {normSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No message norm data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Layer Energy Over Training</h4>
        {energySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={energySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {energySeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No layer energy data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Layer Sparsity (fraction near-zero weights)</h4>
        {sparsitySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={sparsitySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {sparsitySeries.lines.map((line, i) => (
                <Area key={line.key} type="monotone" dataKey={line.key} stroke={line.color} fill={line.color + "22"} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No sparsity data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>2D Embedding Space (by class)</h4>
        {scatterData.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="x" name="x" />
              <YAxis type="number" dataKey="y" name="y" />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={scatterData} fill="#087f5b">
                {scatterData.map((entry, i) => (
                  <Cell key={i} fill={seriesColor(entry.class || 0)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        ) : <div className="empty">No embedding data.</div>}
      </div>
    </div>
  );
}

function ForwardForwardVisualizations({ data }) {
  const rows = data.goodness_stats || [];
  if (!rows.length) return <div className="empty">No forward-forward data.</div>;

  const marginSeries = buildGroupedSeries(rows, "epoch", "margin", "layer", 6);
  const posMeanSeries = buildGroupedSeries(rows, "epoch", "pos_mean", "layer", 6);
  const negMeanSeries = buildGroupedSeries(rows, "epoch", "neg_mean", "layer", 6);

  // Threshold line (should be same across layers; take first non-NaN)
  const threshold = rows.find((r) => typeof r.threshold === "number" && Number.isFinite(r.threshold));
  const thresholdVal = threshold ? threshold.threshold : null;

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Goodness Margin by Layer (pos - neg)</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={marginSeries.data}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
            {marginSeries.lines.map((line) => (
              <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} strokeWidth={2} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="detail-viz-card">
        <h4>Positive vs Negative Goodness</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={posMeanSeries.data}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            {thresholdVal !== null && (
              <ReferenceLine y={thresholdVal} stroke="#c92a2a" strokeDasharray="5 5" label={`θ=${thresholdVal}`} />
            )}
            {posMeanSeries.lines.map((line) => (
              <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} name={`Pos ${line.key}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {negMeanSeries.data.length > 0 && (
        <div className="detail-viz-card">
          <h4>Negative Goodness by Layer</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={negMeanSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {thresholdVal !== null && (
                <ReferenceLine y={thresholdVal} stroke="#c92a2a" strokeDasharray="5 5" label="θ" />
              )}
              {negMeanSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export {
  ClusteringVisualizations,
  HebbianVisualizations,
  ActorCriticVisualizations,
  DiffusionVisualizations,
  GnnVisualizations,
  ForwardForwardVisualizations
};
