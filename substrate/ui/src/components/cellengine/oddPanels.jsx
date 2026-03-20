import {
  clamp01,
  formatNumber,
  formatPct,
  formatSigned,
  cellTypeLabel,
  cellTypeStroke,
  CELL_METRICS,
  CELL_MATRIX_METRICS,
  GENOME_METRICS,
  genomeProgramMeta,
  formatGenomeMetricLabel,
  taskLabel
} from "./core";
import { projectBodyLayout } from "./body";
import { SectionTitle } from "./uiPrimitives";
export function OddDeltaBar({ value, maxAbs, tone = "mechanism" }) {
  const safeValue = Number.isFinite(value) ? value : 0;
  const safeMax = Math.max(maxAbs || 0, 1e-6);
  const widthPct = (Math.abs(safeValue) / safeMax) * 100;
  return (
    <div className={`cellengine-odd-bar ${tone}`}>
      <div className="cellengine-odd-bar-track">
        <div className="cellengine-odd-bar-fill" style={{ width: `${widthPct}%` }} />
      </div>
      <strong>{formatSigned(safeValue, 3)}</strong>
    </div>
  );
}

export function OddCorrelationBar({ value, maxAbs = 1, tone = "corr" }) {
  const safeValue = Number.isFinite(value) ? value : 0;
  const safeMax = Math.max(Math.abs(maxAbs), 1e-6);
  const widthPct = (Math.abs(safeValue) / safeMax) * 100;
  return (
    <div className={`cellengine-odd-bar ${tone}`}>
      <div className="cellengine-odd-bar-track">
        <div className="cellengine-odd-bar-fill" style={{ width: `${widthPct}%` }} />
      </div>
      <strong>{formatSigned(safeValue, 3)}</strong>
    </div>
  );
}

export function oddSignedDeltaColor(value, maxAbs) {
  const magnitude = clamp01(Math.abs(Number(value) || 0) / Math.max(1e-6, maxAbs || 1));
  if ((value || 0) >= 0) {
    return mixHex("#fee2e2", "#c92a2a", magnitude);
  }
  return mixHex("#dbeafe", "#1d4ed8", magnitude);
}

export function parseOddCellKey(key) {
  const match = String(key || "").match(/^cell_force_(\d+)$/);
  return match ? Number(match[1]) : null;
}

export function parseOddEdgeKey(key) {
  const match = String(key || "").match(/^edge_(\d+)_(\d+)$/);
  return match ? { src: Number(match[1]), dst: Number(match[2]) } : null;
}

export function OddSpatialRankingViewer({ title, note, bodyCells, items, mode }) {
  const cells = Array.isArray(bodyCells) ? bodyCells : [];
  if (!cells.length || !Array.isArray(items) || !items.length) {
    return (
      <div className="card cellengine-odd-card">
        <div className="section-head">
          <h3>{title}</h3>
          <span className="meta-note">{note}</span>
        </div>
        <div className="empty">No body geometry is available for this intervention view.</div>
      </div>
    );
  }

  const layout = projectBodyLayout(cells, 22, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: COMPACT_3D_BODY_VIEW });
  const maxAbs = Math.max(1e-6, ...items.map((item) => Math.abs(item?.delta_survival_ratio || 0)));
  const cellImpactById = new Map();
  const edgeImpact = [];
  if (mode === "cell") {
    for (const item of items) {
      const cellId = parseOddCellKey(item?.key);
      if (cellId == null) continue;
      cellImpactById.set(cellId, item);
    }
  } else {
    for (const item of items) {
      const edge = parseOddEdgeKey(item?.key);
      if (!edge) continue;
      edgeImpact.push({ ...edge, item });
    }
  }

  const width = 420;
  const height = 320;
  const pad = 28;
  const scale = Math.min((width - pad * 2) / Math.max(1, layout.width), (height - pad * 2) / Math.max(1, layout.height));
  const positioned = sortProjectedCells(layout.cells).map((cell) => ({
    ...cell,
    drawX: pad + cell.plotX * scale,
    drawY: pad + cell.plotY * scale
  }));
  const byId = new Map(positioned.map((cell) => [cell.cell_id, cell]));

  return (
    <div className="card cellengine-odd-card">
      <div className="section-head">
        <h3>{title}</h3>
        <span className="meta-note">{note}</span>
      </div>
      <div className="cellengine-odd-spatial-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} className="cellengine-odd-spatial-svg" role="img" aria-label={title}>
          <rect x="0" y="0" width={width} height={height} rx="18" fill="#f8fbfa" stroke="#dbe6df" />
          {mode === "edge" ? edgeImpact.map(({ src, dst, item }) => {
            const left = byId.get(src);
            const right = byId.get(dst);
            if (!left || !right) return null;
            const stroke = oddSignedDeltaColor(item.delta_survival_ratio, maxAbs);
            const strokeWidth = 1.2 + 4.2 * clamp01(Math.abs(item.delta_survival_ratio || 0) / maxAbs);
            return (
              <line
                key={`odd-edge-${src}-${dst}`}
                x1={left.drawX}
                y1={left.drawY}
                x2={right.drawX}
                y2={right.drawY}
                stroke={stroke}
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                opacity="0.86"
              >
                <title>{`${item.label} · delta ${formatSigned(item.delta_survival_ratio, 3)}`}</title>
              </line>
            );
          }) : null}
          {positioned.map((cell) => {
            const item = cellImpactById.get(cell.cell_id) || null;
            const fill = item ? oddSignedDeltaColor(item.delta_survival_ratio, maxAbs) : "#eef3f2";
            const stroke = item ? "#0f172a" : cellTypeStroke(cell);
            const radius = item
              ? 7 + 7 * clamp01(Math.abs(item.delta_survival_ratio || 0) / maxAbs)
              : 6.5 + cell.plotDepth * 1.8;
            return (
              <g key={`odd-cell-${cell.cell_id}`} transform={`translate(${cell.drawX}, ${cell.drawY})`} style={depthShadowStyle(cell.plotDepth, Boolean(item))}>
                <circle cx="0" cy="0" r={radius} fill={fill} stroke={stroke} strokeWidth={item ? 2.2 : 1.2}>
                  <title>
                    {item
                      ? `${item.label} · delta ${formatSigned(item.delta_survival_ratio, 3)}`
                      : `cell ${cell.cell_id}`}
                  </title>
                </circle>
                <text x="0" y="4" textAnchor="middle" className="cellengine-odd-spatial-label">{cell.cell_id}</text>
              </g>
            );
          })}
        </svg>
        <div className="cellengine-odd-spatial-note">
          <span>warm = positive ablation harm</span>
          <span>cool = ablation improved outcome</span>
          <span>{mode === "cell" ? "circle size = |cell delta|" : "line width = |edge delta|"}</span>
        </div>
      </div>
    </div>
  );
}

export function OddResultsPanel({ odd, sourceLabel = "", bodyCells = [] }) {
  if (!odd?.available) {
    return (
      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="ODD Results"
            help="Operational decomposition diagnostics: mechanism ablations, gene-channel ablations, temporal correlations, and structural confounds."
          />
          <span className="meta-note">not available for this summary</span>
        </div>
        <div className="empty">This genome/run does not have ODD output. Rerun with `CELLENGINE_ODD=1`.</div>
      </div>
    );
  }

  const mechanisms = Array.isArray(odd.mechanism_deltas) ? [...odd.mechanism_deltas] : [];
  const genes = Array.isArray(odd.gene_channel_deltas) ? [...odd.gene_channel_deltas] : [];
  const cellForce = Array.isArray(odd.cell_force_deltas) ? [...odd.cell_force_deltas] : [];
  const edges = Array.isArray(odd.edge_deltas) ? [...odd.edge_deltas] : [];
  const temporal = Array.isArray(odd.temporal_correlations) ? [...odd.temporal_correlations] : [];
  const structural = Array.isArray(odd.structural_confounds) ? [...odd.structural_confounds] : [];
  mechanisms.sort((a, b) => (b.relative_importance || 0) - (a.relative_importance || 0));
  genes.sort((a, b) => (b.relative_importance || 0) - (a.relative_importance || 0));
  cellForce.sort((a, b) => (b.relative_importance || 0) - (a.relative_importance || 0));
  edges.sort((a, b) => (b.relative_importance || 0) - (a.relative_importance || 0));
  temporal.sort((a, b) => Math.abs(b.correlation || 0) - Math.abs(a.correlation || 0));
  structural.sort((a, b) => Math.abs(b.correlation || 0) - Math.abs(a.correlation || 0));

  const topMechanism = mechanisms[0] || null;
  const topGene = genes[0] || null;
  const topCellForce = cellForce[0] || null;
  const topEdge = edges[0] || null;
  const topTemporal = temporal[0] || null;
  const topStructural = structural[0] || null;
  const maxMechanismDelta = Math.max(1e-6, ...mechanisms.map((item) => Math.abs(item.delta_survival_ratio || 0)));
  const maxGeneDelta = Math.max(1e-6, ...genes.map((item) => Math.abs(item.delta_survival_ratio || 0)));
  const maxCellForceDelta = Math.max(1e-6, ...cellForce.map((item) => Math.abs(item.delta_survival_ratio || 0)));
  const maxEdgeDelta = Math.max(1e-6, ...edges.map((item) => Math.abs(item.delta_survival_ratio || 0)));
  const budgetTrials = odd.budget?.num_trials;
  const budgetTicks = odd.budget?.max_ticks;

  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="ODD Results"
          help="Operational decomposition diagnostics: positive delta means baseline minus ablation, so larger positive values indicate the removed mechanism/channel mattered more for performance."
        />
        <span className="meta-note">{sourceLabel || "active summary"}{budgetTrials && budgetTicks ? ` · ${budgetTrials} trials x ${budgetTicks} ticks` : ""}</span>
      </div>

      <div className="cellengine-odd-summary-grid">
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">baseline clean</div>
          <div className="cellengine-telemetry-value">{formatPct(odd.baseline_clean?.survival_ratio, 1)}</div>
          <div className="meta-note">success {formatPct(odd.baseline_clean?.success_rate, 1)} · mean ticks {formatNumber(odd.baseline_clean?.mean_ticks, 1)}</div>
        </div>
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">top mechanism</div>
          <div className="cellengine-telemetry-value">{topMechanism?.label || "n/a"}</div>
          <div className="meta-note">delta survival {formatSigned(topMechanism?.delta_survival_ratio, 3)} · importance {formatPct(topMechanism?.relative_importance, 1)}</div>
        </div>
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">top gene channel</div>
          <div className="cellengine-telemetry-value">{topGene?.label || "n/a"}</div>
          <div className="meta-note">delta survival {formatSigned(topGene?.delta_survival_ratio, 3)} · importance {formatPct(topGene?.relative_importance, 1)}</div>
        </div>
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">strongest temporal corr</div>
          <div className="cellengine-telemetry-value">{topTemporal ? `${topTemporal.left} vs ${topTemporal.right}` : "n/a"}</div>
          <div className="meta-note">corr {formatSigned(topTemporal?.correlation, 3)}</div>
        </div>
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">top cell force</div>
          <div className="cellengine-telemetry-value">{topCellForce?.label || "n/a"}</div>
          <div className="meta-note">delta survival {formatSigned(topCellForce?.delta_survival_ratio, 3)} · importance {formatPct(topCellForce?.relative_importance, 1)}</div>
        </div>
        <div className="cellengine-odd-summary-card">
          <div className="cellengine-telemetry-label">top edge</div>
          <div className="cellengine-telemetry-value">{topEdge?.label || "n/a"}</div>
          <div className="meta-note">delta survival {formatSigned(topEdge?.delta_survival_ratio, 3)} · importance {formatPct(topEdge?.relative_importance, 1)}</div>
        </div>
      </div>

      <div className="cellengine-odd-notes">
        <div><strong>How to read deltas:</strong> positive `delta_survival_ratio` means the ablation made performance worse than baseline, so that mechanism or gene channel was contributing positively.</div>
        <div><strong>How to read correlations:</strong> temporal correlations are within a replay trace; structural confounds are across cells. They suggest association, not standalone causality.</div>
        {odd.notes ? <div><strong>Notes:</strong> {odd.notes}</div> : null}
      </div>

      <div className="grid two">
        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Mechanism Ablations</h3>
            <span className="meta-note">which whole mechanisms mattered most</span>
          </div>
          <div className="cellengine-odd-list">
            {mechanisms.map((item) => (
              <div key={item.key} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.label}</strong>
                  <span>importance {formatPct(item.relative_importance, 1)}</span>
                </div>
                <OddDeltaBar value={item.delta_survival_ratio} maxAbs={maxMechanismDelta} tone="mechanism" />
                <div className="meta-note">
                  baseline {formatPct(item.baseline_survival_ratio, 1)} {"->"} ablated {formatPct(item.variant_survival_ratio, 1)} · success delta {formatSigned(item.delta_success_rate, 3)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Gene Channel Ablations</h3>
            <span className="meta-note">which gene programs mattered most</span>
          </div>
          <div className="cellengine-odd-list">
            {genes.map((item) => (
              <div key={item.key} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.label}</strong>
                  <span>importance {formatPct(item.relative_importance, 1)}</span>
                </div>
                <OddDeltaBar value={item.delta_survival_ratio} maxAbs={maxGeneDelta} tone="gene" />
                <div className="meta-note">
                  baseline {formatPct(item.baseline_survival_ratio, 1)} {"->"} neutralized {formatPct(item.variant_survival_ratio, 1)} · success delta {formatSigned(item.delta_success_rate, 3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid two">
        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Per-Cell Force Ranking</h3>
            <span className="meta-note">which single cells matter most for mechanical output</span>
          </div>
          <div className="cellengine-odd-list">
            {cellForce.map((item) => (
              <div key={item.key} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.label}</strong>
                  <span>importance {formatPct(item.relative_importance, 1)}</span>
                </div>
                <OddDeltaBar value={item.delta_survival_ratio} maxAbs={maxCellForceDelta} tone="cell" />
                <div className="meta-note">
                  baseline {formatPct(item.baseline_survival_ratio, 1)} {"->"} ablated {formatPct(item.variant_survival_ratio, 1)} · success delta {formatSigned(item.delta_success_rate, 3)}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Edge Ranking</h3>
            <span className="meta-note">which contact or coupling edges matter most</span>
          </div>
          <div className="cellengine-odd-list">
            {edges.map((item) => (
              <div key={item.key} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.label}</strong>
                  <span>importance {formatPct(item.relative_importance, 1)}</span>
                </div>
                <OddDeltaBar value={item.delta_survival_ratio} maxAbs={maxEdgeDelta} tone="edge" />
                <div className="meta-note">
                  baseline {formatPct(item.baseline_survival_ratio, 1)} {"->"} ablated {formatPct(item.variant_survival_ratio, 1)} · success delta {formatSigned(item.delta_success_rate, 3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid two">
        <OddSpatialRankingViewer
          title="Cell Force Map"
          note="spatial view of per-cell force ablation effects over the organism body"
          bodyCells={bodyCells}
          items={cellForce}
          mode="cell"
        />
        <OddSpatialRankingViewer
          title="Edge Impact Map"
          note="contact-edge ablation effects over the current organism geometry"
          bodyCells={bodyCells}
          items={edges}
          mode="edge"
        />
      </div>

      <div className="grid two">
        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Temporal Correlations</h3>
            <span className="meta-note">within-trace associations</span>
          </div>
          <div className="cellengine-odd-list compact">
            {temporal.map((item) => (
              <div key={`${item.left}-${item.right}`} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.left}</strong>
                  <span>{item.right}</span>
                </div>
                <OddCorrelationBar value={item.correlation} maxAbs={1} tone="corr" />
              </div>
            ))}
          </div>
        </div>

        <div className="card cellengine-odd-card">
          <div className="section-head">
            <h3>Structural Confounds</h3>
            <span className="meta-note">body geometry vs gene-expression coupling</span>
          </div>
          <div className="cellengine-odd-list compact">
            {structural.map((item) => (
              <div key={`${item.left}-${item.right}`} className="cellengine-odd-item">
                <div className="cellengine-odd-item-head">
                  <strong>{item.left}</strong>
                  <span>{item.right}</span>
                </div>
                <OddCorrelationBar value={item.correlation} maxAbs={1} tone="struct" />
              </div>
            ))}
          </div>
          {topStructural ? (
            <div className="meta-note">
              strongest structural confound: {topStructural.left} vs {topStructural.right} ({formatSigned(topStructural.correlation, 3)})
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

