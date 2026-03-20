import { useMemo } from "react";
import {
  CELL_MATRIX_METRICS,
  CELL_METRICS,
  clamp,
  clamp01,
  cellTypeLabel,
  cellTypeStroke,
  classifyCellFunction,
  cellMetricSwatchColor,
  describeCellMetric,
  edgeOpacity,
  edgeStrength,
  edgeStroke,
  formatCellMetricValue,
  formatGenomeMetricLabel,
  formatNumber,
  formatPct,
  formatSigned,
  staticMetricColor,
  surfacePaletteColor
} from "./core";
import { depthShadowStyle, projectBodyLayout, sortProjectedCells } from "./body";
import { FieldLabel } from "./uiPrimitives";
export function CellTissueMap({
  bodyCells,
  bodyView,
  interactionProps,
  frameCellRows,
  frameEdges,
  metricId,
  onMetricChange,
  atlasMode,
  onAtlasModeChange,
  selectedCellId,
  onSelectCell,
  connectivityEnabled,
  onConnectivityEnabledChange,
  edgeThreshold,
  onEdgeThresholdChange,
  connectivityMode,
  onConnectivityModeChange,
  functionalStateByCellId
}) {
  if (!bodyCells?.length || !frameCellRows?.length) {
    return <div className="empty">Run a new replay to populate per-cell tissue data.</div>;
  }

  const stateByCellId = new Map(frameCellRows.map((row) => [row.cell_id, row]));
  const layout = projectBodyLayout(bodyCells, 34, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: bodyView });
  const bodyByCellId = new Map(layout.cells.map((cell) => [cell.cell_id, cell]));
  const size = 34;
  const padding = 28;
  const width = layout.width + padding * 2;
  const height = layout.height + padding * 2;
  const orderedCells = sortProjectedCells(layout.cells);

  return (
    <div className="stack">
      <div className="row controls">
        <label>
          <FieldLabel
            label="Atlas view"
            help="`Function in action` colors cells by the dominant behavior inferred from genome predisposition plus current state. `Single metric` keeps the older raw scalar recoloring mode."
          />
          <select className="task-input" value={atlasMode} onChange={(event) => onAtlasModeChange(event.target.value)}>
            <option value="behavior">function in action</option>
            <option value="metric">single metric</option>
          </select>
        </label>
        <label>
          <FieldLabel
            label="Cell metric"
            help="Only used in `single metric` mode."
          />
          <select className="task-input" value={metricId} onChange={(event) => onMetricChange(event.target.value)} disabled={atlasMode !== "metric"}>
            {CELL_METRICS.map((metric) => (
              <option key={metric.id} value={metric.id}>{metric.label}</option>
            ))}
          </select>
        </label>
        <label>
          Connectivity
          <select className="task-input" value={connectivityMode} onChange={(event) => onConnectivityModeChange(event.target.value)}>
            <option value="all">all links</option>
            <option value="selected">selected cell only</option>
          </select>
        </label>
        <label>
          Edge threshold
          <input
            className="task-input"
            type="range"
            min="0"
            max="0.95"
            step="0.05"
            value={edgeThreshold}
            onChange={(event) => onEdgeThresholdChange(Number(event.target.value))}
          />
        </label>
        <label className="cellengine-toggle">
          <input type="checkbox" checked={connectivityEnabled} onChange={(event) => onConnectivityEnabledChange(event.target.checked)} />
          <span>show live connectivity</span>
        </label>
      </div>
      <div className="meta-note">
        {atlasMode === "behavior"
          ? "Default semantic view: fill color shows the dominant real-time function of each cell; playback pause/step controls above update this live."
          : "Raw scalar view: stroke color marks structural role; fill color shows the selected state variable for the current frame."}
      </div>
      <div className="cellengine-orbit-surface" {...interactionProps}>
      <svg className="cellengine-tissue-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Cell tissue map">
        <rect x="0" y="0" width={width} height={height} rx="24" fill="#fcfcfa" />
        {connectivityEnabled
          ? (frameEdges || []).map((edge) => {
              const src = bodyByCellId.get(edge.src_cell_id);
              const dst = bodyByCellId.get(edge.dst_cell_id);
              if (!src || !dst) return null;
              if (connectivityMode === "selected" && selectedCellId != null && edge.src_cell_id !== selectedCellId && edge.dst_cell_id !== selectedCellId) {
                return null;
              }
              const opacity = edgeOpacity(edge, edgeThreshold, selectedCellId);
              if (opacity <= 0) return null;
              const x1 = padding + src.plotX + (size - 4) / 2;
              const y1 = padding + src.plotY + (size - 4) / 2;
              const x2 = padding + dst.plotX + (size - 4) / 2;
              const y2 = padding + dst.plotY + (size - 4) / 2;
              return (
                <line
                  key={`${edge.src_cell_id}-${edge.dst_cell_id}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={edgeStroke(edge, selectedCellId)}
                  strokeWidth={1 + edgeStrength(edge) * 5}
                  strokeLinecap="round"
                  opacity={opacity}
                />
              );
            })
          : null}
        {orderedCells.map((cell) => {
          const state = stateByCellId.get(cell.cell_id);
          const active = state?.active === 1 || state?.active === true;
          const value = state?.[metricId];
          const functionalState = functionalStateByCellId?.get(cell.cell_id) || null;
          const x = padding + cell.plotX;
          const y = padding + cell.plotY;
          const isSelected = cell.cell_id === selectedCellId;
          const fill = atlasMode === "behavior"
            ? functionalState?.fill || "#e2e8f0"
            : cellMetricColor(metricId, value, active);
          const symbol = atlasMode === "behavior" ? functionalState?.symbol || "?" : null;
          const symbolColor = atlasMode === "behavior" ? functionalState?.ink || "#475569" : null;
          const titleLines = atlasMode === "behavior"
            ? [
                `cell ${cell.cell_id} · ${cellTypeLabel(cell)}`,
                `dominant mode: ${functionalState?.label || "n/a"}`,
                `confidence: ${formatPct(functionalState?.confidence, 1)}`,
                `runner-up: ${functionalState?.runnerUpLabel || "n/a"}`,
                `${functionalState?.explanation || "n/a"}`
              ]
            : [
                `cell ${cell.cell_id} · ${cellTypeLabel(cell)}`,
                `${metricId}: ${formatCellMetricValue(metricId, value)}`
              ];
          return (
            <g
              key={cell.cell_id}
              onClick={() => onSelectCell(cell.cell_id)}
              className="cellengine-tissue-cell"
              style={depthShadowStyle(cell.plotDepth, isSelected)}
            >
              <title>{titleLines.join("\n")}</title>
              <rect
                x={x}
                y={y}
                width={size - 4}
                height={size - 4}
                rx="9"
                fill={fill}
                stroke={isSelected ? "#111827" : cellTypeStroke(cell)}
                strokeWidth={isSelected ? 4 : 2.5}
              />
              {atlasMode === "behavior" ? (
                <text
                  x={x + (size - 4) / 2}
                  y={y + 18}
                  textAnchor="middle"
                  className="cellengine-tissue-symbol"
                  style={{ fill: symbolColor }}
                >
                  {symbol}
                </text>
              ) : null}
              {cell.hinge ? <circle cx={x + 8} cy={y + 8} r="4" fill="#0c8599" /> : null}
              {cell.motor ? <circle cx={x + size - 12} cy={y + 8} r="4" fill="#a61e4d" /> : null}
              {cell.ground ? <rect x={x + 8} y={y + size - 12} width={size - 20} height="4" rx="2" fill="#2b8a3e" /> : null}
              {!active ? (
                <g stroke="#475569" strokeWidth="2.4">
                  <line x1={x + 7} y1={y + 7} x2={x + size - 11} y2={y + size - 11} />
                  <line x1={x + size - 11} y1={y + 7} x2={x + 7} y2={y + size - 11} />
                </g>
              ) : null}
              <text x={x + (size - 4) / 2} y={y + size - 9} textAnchor="middle" className="cellengine-tissue-id">{cell.cell_id}</text>
            </g>
          );
        })}
      </svg>
      </div>
    </div>
  );
}

export function CellConnectivityGraph({
  bodyCells,
  bodyView,
  interactionProps,
  frameCellRows,
  frameEdges,
  selectedCellId,
  onSelectCell,
  functionalStateByCellId,
  connectivityEnabled,
  edgeThreshold,
  connectivityMode
}) {
  if (!bodyCells?.length || !frameCellRows?.length) {
    return <div className="empty">Run a new replay to populate graph nodes and links.</div>;
  }

  const stateByCellId = new Map(frameCellRows.map((row) => [row.cell_id, row]));
  const bodyByCellId = new Map(bodyCells.map((cell) => [cell.cell_id, cell]));
  const projectedLayout = projectBodyLayout(bodyCells, 1, { depthSkewRatio: 0.76, depthLiftRatio: 0.38, camera: bodyView });
  const projectedByCellId = new Map(projectedLayout.cells.map((cell) => [cell.cell_id, cell]));
  const spanX = Math.max(1, projectedLayout.width);
  const spanY = Math.max(1, projectedLayout.height);
  const width = 840;
  const height = 520;
  const padX = 78;
  const padY = 72;
  const graphW = width - padX * 2;
  const graphH = height - padY * 2;

  const nodePosByCellId = new Map();
  for (const cell of bodyCells) {
    const state = stateByCellId.get(cell.cell_id) || null;
    const projected = projectedByCellId.get(cell.cell_id) || { plotX: 0, plotY: 0 };
    const activity = clamp01((state?.activity || 0) / 2.5);
    const mech = Math.max(-1, Math.min(1, (state?.a_mech || 0) / 2.5));
    const chem = Math.max(-1, Math.min(1, (state?.chem_out || 0) / 1.5));
    const baseX = padX + (projected.plotX / spanX) * graphW;
    const baseY = padY + (projected.plotY / spanY) * graphH;
    const x = baseX + mech * 7.5 + chem * 3.4;
    const y = baseY - activity * 4.8 + chem * 1.6;
    nodePosByCellId.set(cell.cell_id, { x, y, activity, depth: projected.plotDepth ?? 0.5 });
  }

  const edgeRows = connectivityEnabled ? (frameEdges || []) : [];
  const visibleEdges = edgeRows.filter((edge) => {
    if (connectivityMode === "selected" && selectedCellId != null && edge.src_cell_id !== selectedCellId && edge.dst_cell_id !== selectedCellId) {
      return false;
    }
    return edgeStrength(edge) >= edgeThreshold;
  });
  const activeCount = bodyCells.filter((cell) => {
    const state = stateByCellId.get(cell.cell_id);
    return state?.active === 1 || state?.active === true;
  }).length;

  return (
    <div className="stack">
      <div className="cellengine-cell-graph-meta">
        <span className="cellengine-cell-graph-chip">{bodyCells.length} nodes</span>
        <span className="cellengine-cell-graph-chip">{visibleEdges.length} edges</span>
        <span className="cellengine-cell-graph-chip">{activeCount} active</span>
        <span className="cellengine-cell-graph-chip">mode {connectivityMode}</span>
        <span className="cellengine-cell-graph-chip">threshold {formatNumber(edgeThreshold, 2)}</span>
      </div>
      <div className="cellengine-orbit-surface" {...interactionProps}>
      <svg className="cellengine-cell-graph" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Live cell connectivity graph">
        <defs>
          <radialGradient id="cellengineGraphBackdrop" cx="50%" cy="35%" r="80%">
            <stop offset="0%" stopColor="#fcfffe" />
            <stop offset="100%" stopColor="#f4f9f8" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width={width} height={height} rx="26" fill="url(#cellengineGraphBackdrop)" />
        <g opacity="0.25">
          {Array.from({ length: 8 }).map((_, idx) => (
            <line
              key={`cell-graph-v-${idx}`}
              x1={padX + (idx / 7) * graphW}
              x2={padX + (idx / 7) * graphW}
              y1={padY - 12}
              y2={height - padY + 12}
              stroke="#c9dcdb"
              strokeDasharray="4 8"
            />
          ))}
          {Array.from({ length: 5 }).map((_, idx) => (
            <line
              key={`cell-graph-h-${idx}`}
              x1={padX - 12}
              x2={width - padX + 12}
              y1={padY + (idx / 4) * graphH}
              y2={padY + (idx / 4) * graphH}
              stroke="#d8e6df"
              strokeDasharray="4 8"
            />
          ))}
        </g>
        {visibleEdges.map((edge) => {
          const src = nodePosByCellId.get(edge.src_cell_id);
          const dst = nodePosByCellId.get(edge.dst_cell_id);
          if (!src || !dst) return null;
          const strength = edgeStrength(edge);
          const isSelected = selectedCellId != null && (edge.src_cell_id === selectedCellId || edge.dst_cell_id === selectedCellId);
          return (
            <line
              key={`cell-graph-edge-${edge.src_cell_id}-${edge.dst_cell_id}`}
              x1={src.x}
              y1={src.y}
              x2={dst.x}
              y2={dst.y}
              stroke={edgeStroke(edge, selectedCellId)}
              strokeWidth={(isSelected ? 1.8 : 1.1) + strength * 4.6}
              opacity={(isSelected ? 0.34 : 0.16) + strength * 0.56}
              strokeLinecap="round"
            />
          );
        })}
        {sortProjectedCells(projectedLayout.cells).map((cell) => {
          const node = nodePosByCellId.get(cell.cell_id);
          const state = stateByCellId.get(cell.cell_id) || null;
          const mode = functionalStateByCellId?.get(cell.cell_id) || FUNCTIONAL_STATE_META.inactive;
          const active = state?.active === 1 || state?.active === true;
          const isSelected = cell.cell_id === selectedCellId;
          const radius = 10 + (node?.activity || 0) * 4.2;
          return (
            <g
              key={`cell-graph-node-${cell.cell_id}`}
              className="cellengine-cell-graph-node"
              onClick={() => onSelectCell?.(cell.cell_id)}
              style={depthShadowStyle(node?.depth ?? 0.5, isSelected)}
            >
              <title>{`cell ${cell.cell_id} · ${cellTypeLabel(cell)} · ${mode.label}`}</title>
              <circle cx={node.x} cy={node.y} r={radius + 4} fill={mode.ink} opacity={active ? 0.18 : 0.06} />
              <circle
                cx={node.x}
                cy={node.y}
                r={radius}
                fill={mode.fill}
                opacity={active ? 1 : 0.55}
                stroke={isSelected ? "#111827" : mode.ink}
                strokeWidth={isSelected ? 3.2 : 1.6}
              />
              <circle
                cx={node.x + radius * 0.4}
                cy={node.y - radius * 0.35}
                r={Math.max(1.7, radius * 0.2)}
                fill={active ? "#2b8a3e" : "#94a3b8"}
                opacity="0.9"
              />
              <text x={node.x} y={node.y + 4} textAnchor="middle" className="cellengine-cell-graph-node-id">{cell.cell_id}</text>
            </g>
          );
        })}
      </svg>
      </div>
      <div className="meta-note">node color = dominant cell function; edge thickness/opacity = current coupling strength; click any node to sync Selected Cell</div>
    </div>
  );
}

export function CellMetricMatrixAtlas({ bodyCells, bodyView, interactionProps, frameCellRows, selectedCellId, onSelectCell }) {
  if (!bodyCells?.length || !frameCellRows?.length) {
    return <div className="empty">Run a new replay to populate per-cell tissue data.</div>;
  }

  const stateByCellId = new Map(frameCellRows.map((row) => [row.cell_id, row]));
  const layout = projectBodyLayout(bodyCells, 62, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: bodyView });
  const size = 62;
  const padding = 32;
  const outerSize = size - 4;
  const inset = 7;
  const gap = 2;
  const subSize = (outerSize - inset * 2 - gap * 2) / 3;
  const width = layout.width + padding * 2;
  const height = layout.height + padding * 2;
  const orderedCells = sortProjectedCells(layout.cells);

  return (
    <div className="stack">
      <div className="meta-note">
        Gold-standard overview for multivariate cell state: every organism cell stays in body position and carries a fixed 3x3 metric matrix.
      </div>
      <div className="cellengine-orbit-surface" {...interactionProps}>
        <svg className="cellengine-matrix-atlas" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-cell metric matrix atlas">
          <rect x="0" y="0" width={width} height={height} rx="26" fill="#fcfcfa" />
          {orderedCells.map((cell) => {
            const state = stateByCellId.get(cell.cell_id) || null;
            const active = state?.active === 1 || state?.active === true;
            const x = padding + cell.plotX;
            const y = padding + cell.plotY;
            const isSelected = cell.cell_id === selectedCellId;
            const titleLines = [
              `cell ${cell.cell_id} · ${cellTypeLabel(cell)}`,
              `active ${active ? "yes" : "no"}`
            ].concat(
              CELL_MATRIX_METRICS.map((metric) => `${metric.label}: ${formatCellMetricValue(metric.id, state?.[metric.id])}`)
            );

            return (
              <g
                key={cell.cell_id}
                onClick={() => onSelectCell(cell.cell_id)}
                className="cellengine-tissue-cell"
                transform={`translate(${x}, ${y})`}
                style={depthShadowStyle(cell.plotDepth, isSelected)}
              >
                <title>{titleLines.join("\n")}</title>
                <rect
                  x="0"
                  y="0"
                  width={outerSize}
                  height={outerSize}
                  rx="14"
                  fill="#ffffff"
                  stroke={isSelected ? "#111827" : cellTypeStroke(cell)}
                  strokeWidth={isSelected ? 4 : 2.6}
                />
                {CELL_MATRIX_METRICS.map((metric, index) => {
                  const col = index % 3;
                  const row = Math.floor(index / 3);
                  const value = state?.[metric.id];
                  const cellX = inset + col * (subSize + gap);
                  const cellY = inset + row * (subSize + gap);
                  return (
                    <rect
                      key={metric.id}
                      x={cellX}
                      y={cellY}
                      width={subSize}
                      height={subSize}
                      rx="4"
                      fill={cellMetricSwatchColor(metric.id, value)}
                      opacity={active ? 1 : 0.45}
                    />
                  );
                })}
                {cell.hinge ? <circle cx="10" cy="10" r="4.2" fill="#0c8599" /> : null}
                {cell.motor ? <circle cx={outerSize - 10} cy="10" r="4.2" fill="#a61e4d" /> : null}
                {cell.ground ? <rect x="8" y={outerSize - 10} width={outerSize - 16} height="4" rx="2" fill="#2b8a3e" /> : null}
                <circle
                  cx={outerSize - 11}
                  cy={outerSize - 11}
                  r="5"
                  fill={active ? "#2b8a3e" : "#94a3b8"}
                  stroke="#ffffff"
                  strokeWidth="2"
                />
                {!active ? (
                  <g stroke="#475569" strokeWidth="2.3" opacity="0.8">
                    <line x1="8" y1="8" x2={outerSize - 8} y2={outerSize - 8} />
                    <line x1={outerSize - 8} y1="8" x2="8" y2={outerSize - 8} />
                  </g>
                ) : null}
                <text x={outerSize / 2} y={outerSize - 8} textAnchor="middle" className="cellengine-matrix-id">{cell.cell_id}</text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export function SelectedCellVisualInspector({
  cell,
  state,
  cellFunction,
  genomeMetric,
  selectedCellEdges,
  bodyCells,
  functionalStateByCellId
}) {
  if (!cell || !state) return null;

  const selectedGenomeMetricLabel = formatGenomeMetricLabel(genomeMetric);
  const includeDepth = bodyHasDepth(bodyCells);
  const bodyById = new Map((bodyCells || []).map((item) => [item.cell_id, item]));
  const active = state.active === 1 || state.active === true;
  const driverPrograms = [
    { id: "gene_expr_0", label: "Excitability" },
    { id: "gene_expr_1", label: "Contractility" },
    { id: "gene_expr_2", label: "Mechanosense" },
    { id: "gene_expr_6", label: "Coupling" },
    { id: "gene_expr_7", label: "Adaptation" }
  ];
  const neighbors = selectedCellEdges.slice(0, 8).map((edge) => {
    const neighborId = edge.src_cell_id === cell.cell_id ? edge.dst_cell_id : edge.src_cell_id;
    const neighborCell = bodyById.get(neighborId) || null;
    const neighborFn = functionalStateByCellId?.get(neighborId) || FUNCTIONAL_STATE_META.inactive;
    return {
      edge,
      neighborId,
      neighborCell,
      neighborFn
    };
  });
  const networkWidth = 320;
  const networkHeight = 250;
  const centerX = networkWidth / 2;
  const centerY = 128;
  const radius = 82;

  return (
    <div className="cellengine-selected-visual-layout">
      <div className="cellengine-selected-visual-panel">
        <div className="cellengine-selected-visual-hero">
          <div className="cellengine-selected-visual-core" style={{ "--accent": cellFunction?.fill || "#dbe4e2", "--ink": cellFunction?.ink || "#24313a" }}>
            <div className="cellengine-selected-visual-core-id">cell {cell.cell_id}</div>
            <div className="cellengine-selected-visual-core-mode">{cellFunction?.label || "inactive"}</div>
            <div className="cellengine-selected-visual-core-role">{cellTypeLabel(cell)}</div>
            <div className={`cellengine-selected-visual-core-status${active ? " active" : ""}`}>{active ? "active now" : "inactive now"}</div>
          </div>
          <div className="cellengine-selected-visual-summary">
            <div className="cellengine-selected-visual-chip">{`grid ${formatBodyCellPosition(cell, includeDepth)}`}</div>
            <div className="cellengine-selected-visual-chip">runner-up {cellFunction?.runnerUpLabel || "n/a"}</div>
            <div className="cellengine-selected-visual-chip">focus {selectedGenomeMetricLabel}</div>
            <div className="cellengine-selected-visual-chip">current {describeCellMetric(genomeMetric, cell?.[genomeMetric])}</div>
          </div>
        </div>

        <div className="cellengine-selected-visual-matrix">
          {CELL_MATRIX_METRICS.map((metric) => (
            <div
              key={`selected-visual-${metric.id}`}
              className="cellengine-selected-visual-tile"
              style={{ background: cellMetricColor(metric.id, state[metric.id], active) }}
              title={`${metric.label}: ${formatCellMetricValue(metric.id, state[metric.id])}`}
            >
              <strong>{metric.short}</strong>
              <span>{metric.label}</span>
              <em>{describeCellMetric(metric.id, state[metric.id])}</em>
            </div>
          ))}
        </div>

        <div className="cellengine-selected-driver-grid">
          {driverPrograms.map((program) => {
            const value = clamp01(cell[program.id] || 0);
            const baseMeta = genomeProgramMeta(program.id);
            return (
              <div key={`driver-${program.id}`} className="cellengine-selected-driver-card" title={`${program.label}: ${formatNumber(cell[program.id], 3)}`}>
                <div className="cellengine-selected-driver-head">
                  <div className="cellengine-selected-driver-title">
                    {baseMeta ? (
                      <span
                        className="cellengine-dna-badge"
                        style={{ "--dna-fill": baseMeta.fill, "--dna-ink": baseMeta.ink, "--dna-pale": baseMeta.pale }}
                      >
                        {baseMeta.base}
                      </span>
                    ) : null}
                    <strong>{program.label}</strong>
                  </div>
                  <span style={{ color: baseMeta?.ink || "#0b7285" }}>{value > 0.72 ? "high" : value > 0.4 ? "mid" : "low"}</span>
                </div>
                <div className="cellengine-selected-driver-track" style={{ background: mixHex("#edf1ee", baseMeta?.pale || "#d9e5ff", 0.8) }}>
                  <div
                    className="cellengine-selected-driver-fill"
                    style={{ width: `${value * 100}%`, background: genomeProgramGradient(program.id) }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="cellengine-selected-visual-panel">
        <div className="section-head">
          <h3>Neighborhood Field</h3>
          <span className="meta-note">link weight by stroke and opacity; neighbor color by current function</span>
        </div>
        {neighbors.length ? (
          <svg className="cellengine-selected-network" viewBox={`0 0 ${networkWidth} ${networkHeight}`} role="img" aria-label="Selected cell neighborhood network">
            <defs>
              <radialGradient id="cellengineSelectedHalo" cx="50%" cy="50%" r="70%">
                <stop offset="0%" stopColor={cellFunction?.fill || "#dbe4e2"} stopOpacity="0.95" />
                <stop offset="100%" stopColor={cellFunction?.fill || "#dbe4e2"} stopOpacity="0.22" />
              </radialGradient>
            </defs>
            <rect x="0" y="0" width={networkWidth} height={networkHeight} rx="24" fill="url(#cellengineSky)" />
            {neighbors.map(({ edge, neighborId, neighborFn }, index) => {
              const angle = (-Math.PI / 2) + (index / Math.max(neighbors.length, 1)) * Math.PI * 2;
              const x = centerX + radius * Math.cos(angle);
              const y = centerY + radius * Math.sin(angle);
              const strength = clamp01(edge.gap_mean || 0);
              return (
                <g key={`selected-neighbor-${neighborId}`}>
                  <line
                    x1={centerX}
                    y1={centerY}
                    x2={x}
                    y2={y}
                    stroke={neighborFn.ink}
                    strokeWidth={1.5 + strength * 6}
                    opacity={0.18 + strength * 0.7}
                    strokeLinecap="round"
                  />
                  <circle cx={x} cy={y} r={16 + strength * 4} fill={neighborFn.fill} stroke={neighborFn.ink} strokeWidth="2.2" />
                  <text x={x} y={y + 4} textAnchor="middle" className="cellengine-selected-network-label">{neighborId}</text>
                  <title>{`cell ${neighborId} · ${neighborFn.label} · gap ${formatNumber(edge.gap_mean, 3)}`}</title>
                </g>
              );
            })}
            <circle cx={centerX} cy={centerY} r="36" fill="url(#cellengineSelectedHalo)" />
            <circle cx={centerX} cy={centerY} r="25" fill={cellFunction?.fill || "#dbe4e2"} stroke={cellFunction?.ink || "#24313a"} strokeWidth="3" />
            <text x={centerX} y={centerY + 5} textAnchor="middle" className="cellengine-selected-network-center">{cell.cell_id}</text>
            <text x={centerX} y="228" textAnchor="middle" className="cellengine-selected-network-note">selected cell centered · hover nodes for exact gap values</text>
          </svg>
        ) : (
          <div className="empty">No active links at this frame.</div>
        )}
      </div>
    </div>
  );
}

export function SelectedAllCellsVisualGrid({
  bodyCells,
  frameCellRows,
  selectedCellId,
  onSelectCell,
  functionalStateByCellId,
  genomeMetric
}) {
  if (!bodyCells?.length) {
    return <div className="empty">No organism frame available for this replay.</div>;
  }

  const stateByCellId = new Map((frameCellRows || []).map((row) => [row.cell_id, row]));
  const selectedGenomeMetricLabel = formatGenomeMetricLabel(genomeMetric);
  const driverPrograms = [
    { id: "gene_expr_0", label: "Excitability" },
    { id: "gene_expr_1", label: "Contractility" },
    { id: "gene_expr_2", label: "Mechanosense" },
    { id: "gene_expr_6", label: "Coupling" },
    { id: "gene_expr_7", label: "Adaptation" }
  ];
  const includeDepth = bodyHasDepth(bodyCells);
  const orderedCells = [...bodyCells].sort((a, b) => {
    if (bodyCoord(a, "y") !== bodyCoord(b, "y")) return bodyCoord(b, "y") - bodyCoord(a, "y");
    if (bodyCoord(a, "z") !== bodyCoord(b, "z")) return bodyCoord(a, "z") - bodyCoord(b, "z");
    return bodyCoord(a, "x") - bodyCoord(b, "x");
  });

  return (
    <div className="cellengine-selected-all-grid-wrap">
      <div className="cellengine-selected-all-head">
        <span className="meta-note">all cells · mini visual cards from the current frame</span>
        {selectedCellId !== null ? <span className="cellengine-selected-all-selected">active focus: cell {selectedCellId}</span> : null}
      </div>
      <div className="cellengine-selected-all-grid">
        {orderedCells.map((cell) => {
          const state = stateByCellId.get(cell.cell_id) || null;
          const active = state?.active === 1 || state?.active === true;
          const mode = functionalStateByCellId?.get(cell.cell_id) || FUNCTIONAL_STATE_META.inactive;
          const isSelected = cell.cell_id === selectedCellId;
          const role = cellTypeLabel(cell);

          return (
            <button
              type="button"
              key={`selected-all-${cell.cell_id}`}
              className={`cellengine-selected-all-card${isSelected ? " active" : ""}`}
              onClick={() => onSelectCell?.(cell.cell_id)}
              title={`cell ${cell.cell_id} · ${role} · ${mode.label}`}
              style={{ "--cell-accent": mode.fill || "#dbe4e2", "--cell-ink": mode.ink || "#24313a" }}
            >
              <div className="cellengine-selected-all-card-head">
                <div className="cellengine-selected-all-card-core">
                  <div className="cellengine-selected-all-card-id">cell {cell.cell_id}</div>
                  <div className="cellengine-selected-all-card-mode">{mode.label}</div>
                  <div className="cellengine-selected-all-card-role">{role}</div>
                </div>
                <div className={`cellengine-selected-all-card-status${active ? " active" : ""}`}>
                  {active ? "active" : "inactive"}
                </div>
              </div>

              <div className="cellengine-selected-all-chip-row">
                <span className="cellengine-selected-all-chip">{formatBodyCellPosition(cell, includeDepth)}</span>
                <span className="cellengine-selected-all-chip">{selectedGenomeMetricLabel}</span>
                <span className="cellengine-selected-all-chip">{describeCellMetric(genomeMetric, cell?.[genomeMetric])}</span>
              </div>

              <div className="cellengine-selected-all-matrix-grid">
                {CELL_MATRIX_METRICS.map((metric) => (
                  <div
                    key={`${cell.cell_id}-${metric.id}`}
                    className="cellengine-selected-all-metric-tile"
                    style={{ background: cellMetricColor(metric.id, state?.[metric.id], active) }}
                    title={`${metric.label}: ${formatCellMetricValue(metric.id, state?.[metric.id])}`}
                  >
                    <strong>{metric.short}</strong>
                    <span>{describeCellMetric(metric.id, state?.[metric.id])}</span>
                  </div>
                ))}
              </div>

              <div className="cellengine-selected-all-driver-strip">
                {driverPrograms.map((program) => {
                  const value = clamp01(cell[program.id] || 0);
                  const baseMeta = genomeProgramMeta(program.id);
                  return (
                    <div key={`${cell.cell_id}-${program.id}`} className="cellengine-selected-all-driver">
                      <div className="cellengine-selected-all-driver-label">
                        <span className="cellengine-selected-all-driver-title">
                          {baseMeta ? (
                            <span
                              className="cellengine-dna-badge mini"
                              style={{ "--dna-fill": baseMeta.fill, "--dna-ink": baseMeta.ink, "--dna-pale": baseMeta.pale }}
                            >
                              {baseMeta.base}
                            </span>
                          ) : null}
                          <span>{program.label}</span>
                        </span>
                        <strong>{value > 0.72 ? "H" : value > 0.4 ? "M" : "L"}</strong>
                      </div>
                      <div className="cellengine-selected-all-driver-track" style={{ background: mixHex("#e6edf2", baseMeta?.pale || "#d9e5ff", 0.9) }}>
                        <div
                          className="cellengine-selected-all-driver-fill"
                          style={{ width: `${value * 100}%`, background: genomeProgramGradient(program.id) }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
