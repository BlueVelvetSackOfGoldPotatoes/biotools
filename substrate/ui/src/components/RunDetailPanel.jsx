import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis
} from "recharts";
import { usePolling } from "../hooks/usePolling";
import { fetchJson } from "../lib/fetchJson";
import {
  FAMILY_COLORS,
  confusionMatrix,
  formatPct,
  pivotEpoch,
  runStatusClass,
  runStatusLabel,
  topMisclassifications
} from "../lib/dashboardShared";
import FamilyVisualizations from "./run-detail/FamilyVisualizations";

const POLL_MS = 5000;

function RunDetailPanel({ runId, onNavigate }) {
  const {
    data: detail,
    error,
    isLoading
  } = usePolling(
    (signal) => fetchJson(`/api/runs/${runId}/detail`, { signal }),
    Math.max(POLL_MS, 8000),
    [runId]
  );

  if (isLoading && !detail) return <div className="detail-loading">Loading run detail...</div>;
  if (error && !detail) return <div className="detail-error">{error}</div>;
  if (!detail) return null;

  const { manifest, family, epochRows, batchRows, confusionRows, modelSpecific, calibrationRows, inferenceRows } = detail;
  const detailStatus =
    manifest.train_end_utc == null
      ? (epochRows.length || (batchRows || []).length ? "training" : "starting")
      : "finished";
  const detailWaitingText =
    detailStatus === "starting" ? "first epoch in progress..." : "awaiting data";
  const epochPivoted = pivotEpoch(epochRows);
  const liveBatchRows = (batchRows || [])
    .filter(
      (row) =>
        row.split === "train_batch" &&
        typeof row.global_step === "number" &&
        Number.isFinite(row.global_step)
    )
    .sort((a, b) => a.global_step - b.global_step);

  const trainLoss = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.train_loss })).filter((r) => typeof r.v === "number");
  const testLoss = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.test_loss })).filter((r) => typeof r.v === "number");
  const trainAcc = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.train_accuracy })).filter((r) => typeof r.v === "number");
  const testAcc = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.test_accuracy })).filter((r) => typeof r.v === "number");
  const liveLoss = liveBatchRows
    .filter((row) => typeof row.loss === "number" && Number.isFinite(row.loss))
    .map((row) => ({ step: row.global_step, value: row.loss }));
  const liveAcc = liveBatchRows
    .filter((row) => typeof row.accuracy === "number" && Number.isFinite(row.accuracy))
    .map((row) => ({ step: row.global_step, value: row.accuracy }));
  const gradNorm = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.train_grad_norm_mean })).filter((r) => typeof r.v === "number");
  const throughput = epochPivoted.map((r) => ({ epoch: r.epoch, v: r.train_samples_per_sec })).filter((r) => typeof r.v === "number");

  const cm = confusionMatrix(confusionRows);
  const calData = (calibrationRows || [])
    .filter((r) => typeof r.conf_low === "number" && typeof r.conf_high === "number")
    .sort((a, b) => a.conf_low - b.conf_low)
    .map((r) => ({ center: (r.conf_low + r.conf_high) / 2, empirical_acc: r.empirical_acc, avg_conf: r.avg_conf }));
  const topErrors = topMisclassifications(inferenceRows || []);

  const familyColor = FAMILY_COLORS[family] || "#495057";
  const params = manifest.params || {};

  return (
    <div className="detail-panel" style={{ borderColor: `${familyColor}44` }}>
      <div className="detail-header">
        <div className="detail-header-left">
          <span className="detail-family-badge" style={{ background: `${familyColor}18`, color: familyColor, borderColor: `${familyColor}44` }}>
            {family}
          </span>
          <span className="detail-variant">{manifest.model_variant || "default"}</span>
          {manifest.train_end_utc == null && (
            <span className={`pill ${runStatusClass(detailStatus)}`}>{runStatusLabel(detailStatus)}</span>
          )}
        </div>
        <div className="detail-header-right">
          <button className="mini-btn" onClick={() => onNavigate("training")} type="button">Training</button>
          <button className="mini-btn" onClick={() => onNavigate("evaluation")} type="button">Evaluation</button>
          <button className="mini-btn" onClick={() => onNavigate("game")} type="button">Game</button>
          <button className="mini-btn" onClick={() => onNavigate("model")} type="button">Model Data</button>
        </div>
      </div>

      {Object.keys(params).length > 0 && (
        <div className="detail-params">
          {Object.entries(params).map(([k, v]) => (
            <span key={k} className="detail-param-chip">{k}: {String(v)}</span>
          ))}
        </div>
      )}

      <div className="sparkline-grid">
        <div className="sparkline-card">
          <div className="sparkline-label">
            Loss
            {testLoss.length > 0 && <span className="sparkline-value">{testLoss[testLoss.length - 1].v.toFixed(4)}</span>}
          </div>
          {trainLoss.length > 1 || testLoss.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={epochPivoted}>
                <XAxis dataKey="epoch" hide />
                <YAxis hide />
                <Tooltip />
                {trainLoss.length > 0 && <Area type="monotone" dataKey="train_loss" stroke="#a61e4d" fill="#a61e4d18" dot={false} strokeWidth={1.5} />}
                {testLoss.length > 0 && <Area type="monotone" dataKey="test_loss" stroke="#0c8599" fill="#0c859918" dot={false} strokeWidth={1.5} />}
              </AreaChart>
            </ResponsiveContainer>
          ) : liveLoss.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={liveLoss}>
                <XAxis dataKey="step" hide />
                <YAxis hide />
                <Tooltip />
                <Area type="monotone" dataKey="value" stroke="#a61e4d" fill="#a61e4d18" dot={false} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="sparkline-empty">{detailWaitingText}</div>}
        </div>

        <div className="sparkline-card">
          <div className="sparkline-label">
            Accuracy
            {testAcc.length > 0 && <span className="sparkline-value">{(testAcc[testAcc.length - 1].v * 100).toFixed(2)}%</span>}
          </div>
          {trainAcc.length > 1 || testAcc.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={epochPivoted}>
                <XAxis dataKey="epoch" hide />
                <YAxis hide domain={[0, 1]} />
                <Tooltip />
                {trainAcc.length > 0 && <Area type="monotone" dataKey="train_accuracy" stroke="#a61e4d" fill="#a61e4d18" dot={false} strokeWidth={1.5} />}
                {testAcc.length > 0 && <Area type="monotone" dataKey="test_accuracy" stroke="#2b8a3e" fill="#2b8a3e18" dot={false} strokeWidth={1.5} />}
              </AreaChart>
            </ResponsiveContainer>
          ) : liveAcc.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={liveAcc}>
                <XAxis dataKey="step" hide />
                <YAxis hide domain={[0, 1]} />
                <Tooltip />
                <Area type="monotone" dataKey="value" stroke="#2b8a3e" fill="#2b8a3e18" dot={false} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="sparkline-empty">{detailWaitingText}</div>}
        </div>

        <div className="sparkline-card">
          <div className="sparkline-label">
            Grad Norm
            {gradNorm.length > 0 && <span className="sparkline-value">{gradNorm[gradNorm.length - 1].v.toFixed(4)}</span>}
          </div>
          {gradNorm.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={epochPivoted}>
                <XAxis dataKey="epoch" hide />
                <YAxis hide />
                <Tooltip />
                <Area type="monotone" dataKey="train_grad_norm_mean" stroke="#e67700" fill="#e6770018" dot={false} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="sparkline-empty">{detailWaitingText}</div>}
        </div>

        <div className="sparkline-card">
          <div className="sparkline-label">
            Throughput
            {throughput.length > 0 && <span className="sparkline-value">{Math.round(throughput[throughput.length - 1].v)} s/s</span>}
          </div>
          {throughput.length > 1 ? (
            <ResponsiveContainer width="100%" height={80}>
              <AreaChart data={epochPivoted}>
                <XAxis dataKey="epoch" hide />
                <YAxis hide />
                <Tooltip />
                <Area type="monotone" dataKey="train_samples_per_sec" stroke="#5c940d" fill="#5c940d18" dot={false} strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div className="sparkline-empty">{detailWaitingText}</div>}
        </div>
      </div>

      <div className="detail-eval-row">
        {cm.classes.length > 0 && (
          <div className="detail-viz-card">
            <h4>Confusion Matrix</h4>
            <div className="matrix-wrap">
              <table className="matrix mini-matrix">
                <thead>
                  <tr>
                    <th>t\\p</th>
                    {cm.classes.map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {cm.matrix.map((row, ri) => (
                    <tr key={ri}>
                      <th>{cm.classes[ri]}</th>
                      {row.map((v, ci) => {
                        const alpha = cm.maxValue > 0 ? v / cm.maxValue : 0;
                        const bg = v === 0 ? "rgba(12,133,153,0.04)" : `rgba(12,133,153,${Math.max(0.08, alpha)})`;
                        return <td key={ci} style={{ background: bg }}>{v}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {calData.length > 0 && (
          <div className="detail-viz-card">
            <h4>Calibration</h4>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={calData}>
                <CartesianGrid strokeDasharray="4 4" />
                <XAxis dataKey="center" domain={[0, 1]} type="number" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#6c757d" strokeDasharray="4 4" />
                <Area type="monotone" dataKey="empirical_acc" stroke="#2b8a3e" fill="#2b8a3e18" />
                <Area type="monotone" dataKey="avg_conf" stroke="#a61e4d" fill="#a61e4d10" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {topErrors.length > 0 && (
          <div className="detail-viz-card">
            <h4>Top Misclassifications</h4>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={topErrors.slice(0, 8)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="pair" angle={-35} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#c2255c" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <FamilyVisualizations family={family} modelSpecific={modelSpecific} />
    </div>
  );
}

export default RunDetailPanel;
