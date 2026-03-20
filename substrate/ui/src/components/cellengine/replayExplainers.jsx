import { formatNumber, formatPct, formatSigned, taskMeta } from "./core";

export function WhatAmISeeing({ frame, analysis, latestSummary, taskName }) {
  const meta = taskMeta(taskName || latestSummary?.task_name || "cartpole_balance");
  if (!frame || !analysis || !latestSummary) {
    return (
      <div className="card">
        <div className="section-head">
          <strong>What You Are Seeing</strong>
          <span className="meta-note">viewer semantics</span>
        </div>
        <div className="meta-note">Run or load a replay to populate the stage explanation.</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="section-head">
        <strong>What You Are Seeing</strong>
        <span className="meta-note">task semantics, camera, and solve interpretation</span>
      </div>
      <div className="cellengine-telemetry-grid">
        <div>
          <div className="cellengine-telemetry-label">task</div>
          <div className="cellengine-telemetry-value">{meta.label}</div>
        </div>
        <div>
          <div className="cellengine-telemetry-label">solve</div>
          <div className="cellengine-telemetry-value">{latestSummary.solved ? "yes" : "no"}</div>
        </div>
        <div>
          <div className="cellengine-telemetry-label">success</div>
          <div className="cellengine-telemetry-value">{formatPct(latestSummary.cell_clean?.success_rate, 1)}</div>
        </div>
        <div>
          <div className="cellengine-telemetry-label">survival</div>
          <div className="cellengine-telemetry-value">{formatPct(latestSummary.cell_clean?.survival_ratio, 1)}</div>
        </div>
      </div>
      <div className="meta-note">
        {meta.stageFamily === "balance"
          ? `The main stage is a local balance camera. Absolute drift is shown separately below. Current frame: theta ${formatSigned(frame.theta_deg, 2, "°")} · x ${formatSigned(frame.x, 2, " m")} · peak |theta| ${formatNumber(analysis.max_abs_theta_deg, 2)}°.`
          : meta.stageFamily === "worm"
            ? `The stage shows organism crawl behavior. Progress is forward body transport, not pole stability. Current progress ${formatSigned(frame.x, 2, " m")} · mean speed ${formatSigned(analysis.mean_forward_speed, 2, " m/s")}.`
            : `The stage shows paddle/ball tracking. Success is return completion, not spatial symmetry. Current ball/paddle state ${formatSigned(frame.theta_rad, 2)} / ${formatSigned(frame.x, 2)}.`}
      </div>
    </div>
  );
}
