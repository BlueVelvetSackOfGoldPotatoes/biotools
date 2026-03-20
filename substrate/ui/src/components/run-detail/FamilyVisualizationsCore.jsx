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

function MlpVisualizations({ data }) {
  const neurons = data.neuron_stats || [];
  const layerStats = data.layer_activation_stats || [];
  const pixelImp = data.pixel_importance || [];


  // Weight norms from neuron_stats
  const latestEpoch = neurons.length ? Math.max(...neurons.map((r) => r.epoch).filter((e) => typeof e === "number")) : 0;
  const latest = neurons.filter((r) => r.epoch === latestEpoch);
  const weightData = latest
    .filter((r) => typeof r.weight_l2 === "number" && Number.isFinite(r.weight_l2))
    .map((r) => ({ neuron: r.neuron, weight_l2: r.weight_l2, weight_abs_mean: r.weight_abs_mean }))
    .slice(0, 80);

  // Layer stats over epochs
  const layerSeries = buildGroupedSeries(layerStats, "epoch", "mean_weight_l2", "layer", 4);
  const deadSeries = buildGroupedSeries(layerStats, "epoch", "dead_fraction", "layer", 4);

  // Pixel importance heatmap (28x28 grid)
  const latestPixelEpoch = pixelImp.length ? Math.max(...pixelImp.map((r) => r.epoch).filter((e) => typeof e === "number")) : 0;
  const pixelData = pixelImp
    .filter((r) => r.epoch === latestPixelEpoch && typeof r.weight_norm === "number")
    .sort((a, b) => b.weight_norm - a.weight_norm)
    .slice(0, 50);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Neuron Weight Norms (epoch {latestEpoch || "?"})</h4>
        {weightData.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={weightData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="neuron" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="weight_l2" fill="#0c8599" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No weight norm data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Layer Weight L2 Over Epochs</h4>
        {layerSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={layerSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {layerSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No layer stats.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Dead Neuron Fraction by Layer</h4>
        {deadSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={deadSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {deadSeries.lines.map((line) => (
                <Area key={line.key} type="monotone" dataKey={line.key} stroke={line.color} fill={line.color + "22"} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No dead neuron data.</div>}
      </div>
      {pixelData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Top-50 Most Important Pixels</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="col" name="Column" domain={[0, 27]} />
              <YAxis type="number" dataKey="row" name="Row" domain={[0, 27]} reversed />
              <ZAxis type="number" dataKey="weight_norm" range={[20, 200]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={pixelData} fill="#0c8599" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function TransformerVisualizations({ data }) {
  const entropy = data.attention_entropy || [];
  const headSpec = data.head_specialization || [];

  const entropyByEpoch = new Map();
  for (const r of entropy) {
    if (typeof r.epoch !== "number" || typeof r.entropy_mean !== "number") continue;
    if (!entropyByEpoch.has(r.epoch)) entropyByEpoch.set(r.epoch, { epoch: r.epoch });
    entropyByEpoch.get(r.epoch)[`b${r.block}h${r.head}`] = r.entropy_mean;
  }
  const entropySeries = [...entropyByEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  const headKeys = entropySeries.length
    ? Object.keys(entropySeries[0]).filter((k) => k !== "epoch")
    : [];

  const sparsityByEpoch = buildGroupedSeries(entropy, "epoch", "sparsity", "head", 6);
  const focusSeries = buildGroupedSeries(headSpec, "epoch", "focus_score", "head", 6);

  // Entropy vs Sparsity scatter (latest epoch)
  const latestEpoch = entropy.length ? Math.max(...entropy.map((r) => r.epoch).filter(Number.isFinite)) : 0;
  const scatterData = entropy
    .filter((r) => r.epoch === latestEpoch && typeof r.entropy_mean === "number" && typeof r.sparsity === "number")
    .map((r) => ({ head: `b${r.block}h${r.head}`, entropy: r.entropy_mean, sparsity: r.sparsity }));

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Attention Entropy by Head</h4>
        {entropySeries.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={entropySeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              {headKeys.slice(0, 6).map((k, i) => (
                <Line key={k} dataKey={k} stroke={seriesColor(i)} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No entropy data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Attention Sparsity</h4>
        {sparsityByEpoch.data.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={sparsityByEpoch.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {sparsityByEpoch.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No sparsity data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Head Focus Score</h4>
        {focusSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={focusSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {focusSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No focus data.</div>}
      </div>
      {scatterData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Entropy vs Sparsity (epoch {latestEpoch})</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="entropy" name="Entropy" />
              <YAxis type="number" dataKey="sparsity" name="Sparsity" domain={[0, 1]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={scatterData} fill="#7048e8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function RnnVisualizations({ data }) {
  const rows = data.timestep_stats || [];
  const flowRows = data.gradient_flow || [];

  const gradSeries = buildGroupedSeries(rows, "timestep", "grad_norm", "epoch", 6);
  const hiddenSeries = buildGroupedSeries(rows, "timestep", "hidden_norm", "epoch", 6);

  const flowSeries = flowRows
    .filter((r) => typeof r.epoch === "number" && typeof r.flow_ratio === "number")
    .sort((a, b) => a.epoch - b.epoch);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Gradient Norm by Timestep</h4>
        {gradSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={gradSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Timestep", position: "bottom" }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {gradSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No grad norm data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Hidden State Norm by Timestep</h4>
        {hiddenSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={hiddenSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {hiddenSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No hidden norm data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Gradient Flow Ratio (Vanishing Gradient)</h4>
        {flowSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={flowSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="epoch" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <ReferenceLine y={0.1} stroke="#c92a2a" strokeDasharray="3 3" label="Vanishing" />
              <Area type="monotone" dataKey="flow_ratio" stroke="#5c940d" fill="#5c940d22" name="Flow Ratio" />
              <Area type="monotone" dataKey="vanishing_score" stroke="#c92a2a" fill="#c92a2a22" name="Vanishing Score" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No gradient flow data.</div>}
      </div>
    </div>
  );
}

function LstmVisualizations({ data }) {
  const rows = data.gate_stats || [];
  const cellRows = data.cell_state_range || [];

  const satSeries = buildGroupedSeries(rows, "epoch", "sat_frac", "gate", 6);
  const meanSeries = buildGroupedSeries(rows, "epoch", "mean", "gate", 6);
  const cellSeries = buildGroupedSeries(cellRows, "timestep", "cell_magnitude", "epoch", 6);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Gate Saturation Fraction</h4>
        {satSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={satSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {satSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No gate stats.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Gate Mean Activation</h4>
        {meanSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={meanSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              {meanSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No gate mean data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Cell State Magnitude Over Sequence</h4>
        {cellSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={cellSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Timestep", position: "bottom" }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {cellSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No cell state data.</div>}
      </div>
    </div>
  );
}

function ViTVisualizations({ data }) {
  const clsRows = data.cls_embedding_stats || [];
  const patchRows = data.patch_attention || [];
  const spatialRows = data.spatial_attention_map || [];

  const clsSeries = clsRows
    .filter((r) => typeof r.epoch === "number" && typeof r.mean_norm === "number")
    .sort((a, b) => a.epoch - b.epoch);

  const patchSeries = buildGroupedSeries(patchRows, "patch_idx", "attn", "head", 6);

  // Spatial attention map (latest epoch)
  const latestEpoch = spatialRows.length ? Math.max(...spatialRows.map((r) => r.epoch).filter(Number.isFinite)) : 0;
  const spatialData = spatialRows
    .filter((r) => r.epoch === latestEpoch && typeof r.attn_weight === "number" && r.grid_row >= 0)
    .map((r) => ({ ...r, attn: r.attn_weight }));

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>CLS Token Norm Growth</h4>
        {clsSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={clsSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="mean_norm" stroke="#d6336c" fill="#d6336c22" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No CLS stats.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Per-Head Patch Attention</h4>
        {patchSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={patchSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Patch", position: "bottom" }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {patchSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No patch attention data.</div>}
      </div>
      {spatialData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Spatial Attention Map (epoch {latestEpoch})</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="grid_col" name="Col" domain={[0, 4]} />
              <YAxis type="number" dataKey="grid_row" name="Row" domain={[0, 4]} reversed />
              <ZAxis type="number" dataKey="attn" range={[40, 300]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={spatialData} fill="#d6336c" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function TreeVisualizations({ data }) {
  const growth = data.tree_growth || [];
  const importance = data.split_importance || [];
  const pixelMap = data.pixel_importance_map || [];

  const importantFeatures = importance
    .filter((r) => typeof r.gain_total === "number" && r.gain_total > 0)
    .sort((a, b) => b.gain_total - a.gain_total)
    .slice(0, 30);

  // Pixel importance spatial map (top 100)
  const pixelData = pixelMap
    .filter((r) => typeof r.importance === "number" && r.importance > 0)
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 100);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Tree Structure (Nodes by Depth)</h4>
        {growth.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={growth}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="depth" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="nodes" fill="#2b8a3e" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No tree growth data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Top Feature Importances</h4>
        {importantFeatures.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={importantFeatures}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="feature" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="gain_total" fill="#087f5b" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No feature importance data.</div>}
      </div>
      {pixelData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Pixel Importance Map (28x28)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="col" name="Column" domain={[0, 27]} />
              <YAxis type="number" dataKey="row" name="Row" domain={[0, 27]} reversed />
              <ZAxis type="number" dataKey="importance" range={[15, 180]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={pixelData} fill="#2b8a3e" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ForestVisualizations({ data }) {
  const oob = data.oob_curve || [];
  const agreement = data.tree_agreement || [];
  const featureImp = data.feature_importance || [];

  const oobSeries = oob
    .filter((r) => typeof r.n_trees === "number" && typeof r.oob_acc === "number" && Number.isFinite(r.oob_acc))
    .sort((a, b) => a.n_trees - b.n_trees);

  const agreeHist = [];
  if (agreement.length) {
    const bins = 20;
    const counts = new Array(bins).fill(0);
    for (const r of agreement) {
      if (typeof r.agreement_ratio !== "number") continue;
      const idx = Math.min(bins - 1, Math.floor(r.agreement_ratio * bins));
      counts[idx]++;
    }
    for (let i = 0; i < bins; i++) {
      agreeHist.push({ bin: (i / bins).toFixed(2), count: counts[i] });
    }
  }

  // Spatial feature importance
  const pixelData = featureImp
    .filter((r) => typeof r.importance === "number" && r.importance > 0)
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 80);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>OOB Accuracy vs Number of Trees</h4>
        {oobSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={oobSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="n_trees" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line dataKey="oob_acc" stroke="#0b7285" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No OOB data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Tree Agreement Distribution</h4>
        {agreeHist.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={agreeHist}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="bin" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#2b8a3e" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No agreement data.</div>}
      </div>
      {pixelData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Ensemble Feature Importance Map</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="col" name="Column" domain={[0, 27]} />
              <YAxis type="number" dataKey="row" name="Row" domain={[0, 27]} reversed />
              <ZAxis type="number" dataKey="importance" range={[15, 200]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={pixelData} fill="#0b7285" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function MarkovVisualizations({ data }) {
  const classState = data.class_state_metrics || [];
  const emissions = data.symbol_emissions || [];
  const positionNll = data.position_nll || [];
  const progress = data.training_progress || [];
  const classLikelihoods = data.class_likelihoods || [];

  const progressSeries = progress
    .filter((r) => typeof r.epoch === "number")
    .sort((a, b) => a.epoch - b.epoch);

  const entropySeries = buildGroupedSeries(classState, "epoch", "transition_entropy", "class_id", 10);
  const coverageSeries = buildGroupedSeries(classState, "epoch", "context_coverage", "class_id", 10);

  const emissionEpochs = listDistinctNumbers(emissions, "epoch");
  const latestEmissionEpoch = emissionEpochs.length ? emissionEpochs[emissionEpochs.length - 1] : null;
  const emissionSeries = buildGroupedSeries(
    emissions.filter((r) => latestEmissionEpoch == null || r.epoch === latestEmissionEpoch),
    "bin",
    "fraction",
    "class_id",
    10
  );

  const nllEpochs = listDistinctNumbers(positionNll, "epoch");
  const latestNllEpoch = nllEpochs.length ? nllEpochs[nllEpochs.length - 1] : null;
  const nllCurve = positionNll
    .filter(
      (r) =>
        (latestNllEpoch == null || r.epoch === latestNllEpoch) &&
        typeof r.position === "number" &&
        typeof r.nll === "number" &&
        Number.isFinite(r.nll)
    )
    .sort((a, b) => a.position - b.position);
  const nllMap = nllCurve
    .filter((r) => typeof r.row === "number" && typeof r.col === "number")
    .slice(0, 300);

  const likelihoodData = classLikelihoods
    .filter((r) => typeof r.class_id === "number")
    .sort((a, b) => a.class_id - b.class_id)
    .map((r) => ({
      class_id: `C${r.class_id}`,
      accuracy: r.accuracy,
      avg_true_prob: r.avg_true_prob,
      avg_pred_confidence: r.avg_pred_confidence
    }));

  if (
    !progressSeries.length &&
    !entropySeries.data.length &&
    !coverageSeries.data.length &&
    !emissionSeries.data.length &&
    !nllCurve.length &&
    !likelihoodData.length
  ) {
    return <div className="empty">No Markov diagnostics data.</div>;
  }

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Training Progress</h4>
        {progressSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={progressSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="epoch" />
              <YAxis yAxisId="left" domain={[0, 1]} />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" dataKey="test_accuracy" stroke="#1d4ed8" dot={false} strokeWidth={2} name="Test Acc" />
              <Line yAxisId="right" dataKey="test_loss" stroke="#a61e4d" dot={false} name="Test Loss" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No training progression.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Transition Entropy by Class</h4>
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
        ) : <div className="empty">No transition entropy data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Context Coverage by Class</h4>
        {coverageSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={coverageSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {coverageSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No context coverage data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Symbol Emission Distribution{latestEmissionEpoch != null ? ` (epoch ${latestEmissionEpoch})` : ""}</h4>
        {emissionSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={emissionSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {emissionSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No symbol emissions data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Position-wise NLL{latestNllEpoch != null ? ` (epoch ${latestNllEpoch})` : ""}</h4>
        {nllCurve.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={nllCurve}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="position" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="nll" stroke="#5f3dc4" fill="#5f3dc422" />
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No position NLL data.</div>}
      </div>
      {nllMap.length > 0 && (
        <div className="detail-viz-card">
          <h4>NLL Spatial Map (28x28)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="col" domain={[0, 27]} name="Col" />
              <YAxis type="number" dataKey="row" domain={[0, 27]} name="Row" reversed />
              <ZAxis type="number" dataKey="nll" range={[20, 200]} name="NLL" />
              <Tooltip />
              <Scatter data={nllMap} fill="#1d4ed8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="detail-viz-card">
        <h4>Class Likelihood Summary</h4>
        {likelihoodData.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={likelihoodData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="class_id" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" fill="#1d4ed8" name="Accuracy" />
              <Bar dataKey="avg_true_prob" fill="#2b8a3e" name="Avg True Prob" />
              <Bar dataKey="avg_pred_confidence" fill="#e67700" name="Avg Pred Confidence" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No class likelihood data.</div>}
      </div>
    </div>
  );
}

function CnnVisualizations({ data }) {
  const filterStats = data.filter_stats || [];
  const featuremapStats = data.featuremap_stats || [];
  const activationEnergy = data.activation_energy || [];

  // Filter L2 norms by layer over epochs
  const filterSeries = buildGroupedSeries(filterStats, "epoch", "l2", "layer", 8);

  // Feature map sparsity by layer over epochs
  const sparsitySeries = buildGroupedSeries(featuremapStats, "epoch", "sparsity", "layer", 8);

  // Activation energy by layer over epochs
  const energySeries = buildGroupedSeries(activationEnergy, "epoch", "activation_energy", "layer", 8);

  // Fraction of tiny weights (near-dead filters)
  const tinySeries = buildGroupedSeries(activationEnergy, "epoch", "fraction_tiny", "layer", 8);

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Filter L2 Norms by Layer</h4>
        {filterSeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={filterSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: "L2 Norm", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {filterSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No filter stats data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Feature Map Sparsity</h4>
        {sparsitySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={sparsitySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} label={{ value: "Sparsity", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              {sparsitySeries.lines.map((line) => (
                <Area key={line.key} type="monotone" dataKey={line.key} stroke={line.color} fill={line.color + "22"} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No feature map stats.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Activation Energy by Layer</h4>
        {energySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={energySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {energySeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No activation energy data.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Near-Dead Filter Fraction</h4>
        {tinySeries.data.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={tinySeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Epoch", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {tinySeries.lines.map((line) => (
                <Area key={line.key} type="monotone" dataKey={line.key} stroke={line.color} fill={line.color + "22"} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        ) : <div className="empty">No tiny fraction data.</div>}
      </div>
    </div>
  );
}

export {
  MlpVisualizations,
  TransformerVisualizations,
  RnnVisualizations,
  LstmVisualizations,
  ViTVisualizations,
  TreeVisualizations,
  ForestVisualizations,
  MarkovVisualizations,
  CnnVisualizations
};
