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

function ReinforcementVisualizations({ data }) {
  const search = data.muzero_search || [];
  const algos = data.rl_algorithms || [];

  const searchSeries = search
    .filter((r) => typeof r.step === "number")
    .sort((a, b) => a.step - b.step);

  const algoSeries = algos
    .filter((r) => typeof r.step === "number")
    .sort((a, b) => a.step - b.step);

  // Search depth vs value estimate
  const depthData = search
    .filter((r) => typeof r.step === "number" && typeof r.avg_depth === "number")
    .sort((a, b) => a.step - b.step);

  // Root value convergence
  const valueData = search
    .filter((r) => typeof r.step === "number" && typeof r.root_value === "number" && Number.isFinite(r.root_value))
    .sort((a, b) => a.step - b.step);

  return (
    <div className="detail-viz-grid">
      {searchSeries.length > 0 && (
        <div className="detail-viz-card">
          <h4>MuZero Search Entropy</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={searchSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Line dataKey="root_entropy" stroke="#364fc7" dot={false} strokeWidth={2} name="Entropy" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {valueData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Root Value Estimate</h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={valueData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="root_value" stroke="#2b8a3e" fill="#2b8a3e22" name="Value" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      {depthData.length > 0 && (
        <div className="detail-viz-card">
          <h4>Average Search Depth</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={depthData}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Line dataKey="avg_depth" stroke="#e67700" dot={false} strokeWidth={2} name="Avg Depth" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {algoSeries.length > 0 && (
        <div className="detail-viz-card">
          <h4>RL Algorithm Performance</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={algoSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line dataKey="test_acc" stroke="#364fc7" dot={false} strokeWidth={2} name="Accuracy" />
              <Line dataKey="loss" stroke="#a61e4d" dot={false} name="Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {!searchSeries.length && !algoSeries.length && (
        <div className="empty">No reinforcement learning data.</div>
      )}
    </div>
  );
}

function ContinuousVisualizations({ data }) {
  const phaseMetrics = data.continuous_phase_metrics || [];
  const efficiency = data.continuous_efficiency || [];
  const expertMetrics = data.hybrid_expert_metrics || [];

  // Active vs sleep accuracy by cycle
  const activeSeries = phaseMetrics
    .filter((r) => r.phase === "active" && typeof r.cycle === "number")
    .sort((a, b) => a.cycle - b.cycle);
  const sleepSeries = phaseMetrics
    .filter((r) => r.phase === "sleep" && typeof r.cycle === "number")
    .sort((a, b) => a.cycle - b.cycle);
  const phaseCombined = activeSeries.map((a, i) => ({
    cycle: a.cycle,
    active_acc: a.accuracy,
    sleep_acc: sleepSeries[i] ? sleepSeries[i].accuracy : null,
    active_loss: a.loss,
    sleep_loss: sleepSeries[i] ? sleepSeries[i].loss : null,
  }));

  // Efficiency: test accuracy and replay size over cycles
  const effSeries = efficiency
    .filter((r) => typeof r.cycle === "number")
    .sort((a, b) => a.cycle - b.cycle);

  // Expert fusion strength over cycles (hybrid only)
  const expertSeries = buildGroupedSeries(expertMetrics, "cycle", "fusion_strength", "expert", 6);
  const expertAccSeries = buildGroupedSeries(expertMetrics, "cycle", "expert_accuracy", "expert", 6);

  if (!phaseCombined.length && !effSeries.length && !expertMetrics.length) {
    return <div className="empty">No continuous/hybrid data.</div>;
  }

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Active vs Sleep Accuracy</h4>
        {phaseCombined.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={phaseCombined}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="cycle" label={{ value: "Cycle", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line dataKey="active_acc" stroke="#1864ab" dot={false} strokeWidth={2} name="Active" />
              <Line dataKey="sleep_acc" stroke="#e67700" dot={false} strokeWidth={2} name="Sleep" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No phase metrics.</div>}
      </div>
      <div className="detail-viz-card">
        <h4>Test Accuracy &amp; Replay Size</h4>
        {effSeries.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={effSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="cycle" />
              <YAxis yAxisId="left" domain={[0, 1]} />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" dataKey="test_accuracy" stroke="#2b8a3e" dot={false} strokeWidth={2} name="Test Acc" />
              <Line yAxisId="right" dataKey="replay_size" stroke="#862e9c" dot={false} name="Replay Size" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No efficiency data.</div>}
      </div>
      {expertSeries.data.length > 0 && (
        <div className="detail-viz-card">
          <h4>Expert Fusion Strength</h4>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={expertSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Cycle", position: "insideBottom", offset: -2 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              {expertSeries.lines.map((line) => (
                <Area key={line.key} type="monotone" dataKey={line.key} stroke={line.color} fill={line.color + "22"} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      {expertAccSeries.data.length > 0 && (
        <div className="detail-viz-card">
          <h4>Per-Expert Accuracy</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={expertAccSeries.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" label={{ value: "Cycle", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              {expertAccSeries.lines.map((line) => (
                <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function TicTacToeVisualizations({ data }) {
  const policy = data.tictactoe_policy_metrics || [];
  const deploy = data.tictactoe_deployment_summary || [];
  const training = data.tictactoe_training_tactics || [];
  const bridge = data.benchmark_bridge_diagnostics || [];

  const series = policy
    .filter((r) => typeof r.episode === "number" && Number.isFinite(r.episode))
    .sort((a, b) => a.episode - b.episode);
  const trainingSeries = training
    .filter((r) => typeof r.global_step === "number" && Number.isFinite(r.global_step))
    .sort((a, b) => a.global_step - b.global_step);

  const latestDeploy =
    deploy && deploy.length
      ? deploy.reduce((best, row) => {
          if (!best) return row;
          const be = typeof best.episode === "number" ? best.episode : -1;
          const re = typeof row.episode === "number" ? row.episode : -1;
          return re > be ? row : best;
        }, null)
      : null;

  const latestBridge = bridge && bridge.length ? bridge[bridge.length - 1] : null;

  if (!series.length) {
    return <div className="empty">No TicTacToe policy metrics found.</div>;
  }

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Outcome Rates</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={series}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="episode" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line dataKey="win_rate" stroke="#2b8a3e" dot={false} name="win_rate" />
            <Line dataKey="draw_rate" stroke="#0c8599" dot={false} name="draw_rate" />
            <Line dataKey="loss_rate" stroke="#a61e4d" dot={false} name="loss_rate" />
            <Line dataKey="non_loss_rate" stroke="#364fc7" dot={false} strokeWidth={2} name="non_loss_rate" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="detail-viz-card">
        <h4>Tactical Capture Rates</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={series}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="episode" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line dataKey="win_capture_rate" stroke="#2b8a3e" dot={false} strokeWidth={2} name="win_capture_rate" />
            <Line dataKey="block_capture_rate" stroke="#1864ab" dot={false} strokeWidth={2} name="block_capture_rate" />
            <Line dataKey="missed_win_rate" stroke="#c92a2a" dot={false} name="missed_win_rate" />
            <Line dataKey="missed_block_rate" stroke="#e67700" dot={false} name="missed_block_rate" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {trainingSeries.length > 0 && (
        <div className="detail-viz-card">
          <h4>Training Tactical EMA</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={trainingSeries}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="global_step" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line dataKey="action_match_ema" stroke="#5f3dc4" dot={false} name="action_match_ema" />
              <Line dataKey="missed_win_rate_ema" stroke="#c92a2a" dot={false} name="missed_win_rate_ema" />
              <Line dataKey="missed_block_rate_ema" stroke="#e67700" dot={false} name="missed_block_rate_ema" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="detail-viz-card">
        <h4>Tactical Opportunity Counts</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={series}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="episode" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="win_opportunities" fill="#2b8a3e" name="win_opportunities" />
            <Bar dataKey="missed_wins" fill="#c92a2a" name="missed_wins" />
            <Bar dataKey="block_opportunities" fill="#1864ab" name="block_opportunities" />
            <Bar dataKey="missed_blocks" fill="#e67700" name="missed_blocks" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="detail-viz-card">
        <h4>Policy Hygiene</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={series}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="episode" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line dataKey="action_match_rate" stroke="#5f3dc4" dot={false} name="action_match_rate" />
            <Line dataKey="invalid_rate" stroke="#a61e4d" dot={false} name="invalid_rate" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {(latestDeploy || latestBridge) && (
        <div className="detail-viz-card">
          <h4>Latest Deployment Snapshot</h4>
          <div className="meta-grid">
            {latestDeploy && (
              <>
                <div><span>non_loss</span><span>{formatPct(latestDeploy.non_loss_rate)}</span></div>
                <div><span>win</span><span>{formatPct(latestDeploy.win_rate)}</span></div>
                <div><span>draw</span><span>{formatPct(latestDeploy.draw_rate)}</span></div>
                <div><span>loss</span><span>{formatPct(latestDeploy.loss_rate)}</span></div>
                <div><span>missed_win</span><span>{formatPct(latestDeploy.missed_win_rate)}</span></div>
                <div><span>missed_block</span><span>{formatPct(latestDeploy.missed_block_rate)}</span></div>
                <div><span>p95 latency</span><span>{typeof latestDeploy.p95_ms === "number" ? `${latestDeploy.p95_ms.toFixed(3)} ms` : "n/a"}</span></div>
                <div><span>moves</span><span>{typeof latestDeploy.moves === "number" ? latestDeploy.moves.toLocaleString() : "n/a"}</span></div>
              </>
            )}
            {latestBridge && (
              <>
                <div><span>bridge bits</span><span>{latestBridge.bridge_input_bits ?? "n/a"}</span></div>
                <div><span>bridge width</span><span>{latestBridge.bridge_width ?? "n/a"}</span></div>
                <div><span>backbone input</span><span>{latestBridge.backbone_input ?? "n/a"}</span></div>
                <div><span>logits</span><span>{latestBridge.backbone_logits ?? "n/a"}</span></div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export {
  ReinforcementVisualizations,
  ContinuousVisualizations,
  TicTacToeVisualizations
};
