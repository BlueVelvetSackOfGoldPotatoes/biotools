import { useEffect, useMemo, useState } from "react";
import {
  GENOME_METRICS,
  clamp01,
  cellTypeStroke,
  formatGenomeMetricLabel,
  formatNumber,
  genomeProgramGradient,
  genomeProgramMeta,
  lerp,
  staticMetricColor
} from "./core";
import { compactPreviewBodyView, depthShadowStyle, projectBodyLayout, sortProjectedCells } from "./body";
export function GenomeVariabilityMap({ bodyCells, bodyView, interactionProps, metricId, onMetricChange, selectedCellId, onSelectCell, showControls = true }) {
  if (!bodyCells?.length) {
    return <div className="empty">This replay does not include genome/body annotations. Run a fresh replay.</div>;
  }

  const layout = projectBodyLayout(bodyCells, 34, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: bodyView });
  const values = bodyCells.map((cell) => cell[metricId]).filter((value) => typeof value === "number");
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;
  const span = Math.max(1e-6, maxValue - minValue);
  const size = 34;
  const padding = 28;
  const width = layout.width + padding * 2;
  const height = layout.height + padding * 2;
  const orderedCells = sortProjectedCells(layout.cells);

  return (
    <div className="stack">
      {showControls ? (
        <div className="row controls">
          <label>
            Genome metric
            <select className="task-input" value={metricId} onChange={(event) => onMetricChange(event.target.value)}>
              {GENOME_METRICS.map((metric) => (
                <option key={metric.id} value={metric.id}>{formatGenomeMetricLabel(metric.id)}</option>
              ))}
            </select>
          </label>
          <span className="meta-note">All cells share one genome. What varies here is expression strength induced by geometry and local role.</span>
        </div>
      ) : null}
      <div className="cellengine-orbit-surface" {...interactionProps}>
      <svg className="cellengine-tissue-map" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Genome variability map">
        <rect x="0" y="0" width={width} height={height} rx="24" fill="#fcfcfa" />
        {orderedCells.map((cell) => {
          const raw = cell[metricId];
          const normalized = clamp01(((raw || 0) - minValue) / span);
          const x = padding + cell.plotX;
          const y = padding + cell.plotY;
          const isSelected = cell.cell_id === selectedCellId;
          return (
            <g
              key={cell.cell_id}
              onClick={() => onSelectCell(cell.cell_id)}
              className="cellengine-tissue-cell"
              style={depthShadowStyle(cell.plotDepth, isSelected)}
            >
              <rect
                x={x}
                y={y}
                width={size - 4}
                height={size - 4}
                rx="9"
                fill={staticMetricColor(metricId, normalized)}
                stroke={isSelected ? "#111827" : cellTypeStroke(cell)}
                strokeWidth={isSelected ? 4 : 2.5}
              />
              <text x={x + (size - 4) / 2} y={y + 22} textAnchor="middle" className="cellengine-tissue-id">{cell.cell_id}</text>
            </g>
          );
        })}
      </svg>
      </div>
      <div className="meta-note">
        selected metric range {formatNumber(minValue, 3)} to {formatNumber(maxValue, 3)}
      </div>
    </div>
  );
}

export function GenomeMiniMap({ bodyCells, bodyView, metricId, selected = false }) {
  if (!bodyCells?.length) {
    return <div className="empty">preview unavailable</div>;
  }
  const previewView = compactPreviewBodyView(bodyCells, bodyView);
  const layout = projectBodyLayout(bodyCells, 14, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: previewView });
  const values = bodyCells.map((cell) => cell[metricId]).filter((value) => typeof value === "number");
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;
  const span = Math.max(1e-6, maxValue - minValue);
  const size = 14;
  const padding = 14;
  const width = layout.width + padding * 2;
  const height = layout.height + padding * 2;
  return (
    <svg className={`cellengine-mini-map${selected ? " selected" : ""}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Genome preview map">
      <rect x="0" y="0" width={width} height={height} rx="16" fill="#fcfcfa" />
      {sortProjectedCells(layout.cells).map((cell) => {
        const normalized = clamp01((((cell[metricId] || 0) - minValue) / span));
        const drawSize = layout.hasDepth ? Math.max(8.4, size * (0.84 + cell.plotDepth * 0.24)) : size - 2;
        const x = padding + cell.plotX + ((size - 2) - drawSize) / 2;
        const y = padding + cell.plotY + ((size - 2) - drawSize) / 2;
        return (
          <g key={cell.cell_id} style={depthShadowStyle(cell.plotDepth, selected)} opacity={layout.hasDepth ? lerp(0.55, 1, cell.plotDepth) : 1}>
            <rect
              x={x}
              y={y}
              width={drawSize}
              height={drawSize}
              rx={layout.hasDepth ? 4.8 : 4}
              fill={staticMetricColor(metricId, normalized)}
              stroke={cellTypeStroke(cell)}
              strokeWidth={selected ? 1.8 : 1.2}
            />
          </g>
        );
      })}
    </svg>
  );
}

export function DiscoveryDevelopmentMiniMap({
  bodyCells,
  bodyView,
  metricId,
  active = false,
  frameStep = null,
  footer = true,
  compact = false,
  animateOnHover = false
}) {
  const normalizedCells = useMemo(
    () => (Array.isArray(bodyCells) ? bodyCells : []).map((cell) => ({
      ...cell,
      birth_step: Number.isFinite(Number(cell?.birth_step)) ? Number(cell.birth_step) : 0
    })),
    [bodyCells]
  );
  const bodySignature = useMemo(
    () => normalizedCells.map((cell) => `${cell.cell_id}:${cell.birth_step}`).join("|"),
    [normalizedCells]
  );
  const maxBirthStep = useMemo(
    () => normalizedCells.reduce((best, cell) => Math.max(best, cell.birth_step || 0), 0),
    [normalizedCells]
  );
  const hasExternalFrame = Number.isFinite(frameStep);
  const [hovered, setHovered] = useState(false);
  const shouldAnimate = active || (animateOnHover && hovered);
  const [step, setStep] = useState(shouldAnimate ? 0 : maxBirthStep);

  useEffect(() => {
    if (hasExternalFrame) return;
    setStep(shouldAnimate ? 0 : maxBirthStep);
  }, [shouldAnimate, bodySignature, maxBirthStep, hasExternalFrame]);

  useEffect(() => {
    if (hasExternalFrame || !shouldAnimate || maxBirthStep <= 0) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setStep((current) => (current >= maxBirthStep ? 0 : current + 1));
    }, 260);
    return () => window.clearInterval(timer);
  }, [shouldAnimate, maxBirthStep, bodySignature, hasExternalFrame]);

  if (!normalizedCells.length) {
    return <div className="empty">preview unavailable</div>;
  }

  const previewView = compactPreviewBodyView(normalizedCells, bodyView);
  const layout = projectBodyLayout(normalizedCells, 14, { depthSkewRatio: 0.66, depthLiftRatio: 0.34, camera: previewView });
  const values = normalizedCells.map((cell) => cell[metricId]).filter((value) => typeof value === "number");
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;
  const span = Math.max(1e-6, maxValue - minValue);
  const size = 14;
  const padding = 14;
  const width = layout.width + padding * 2;
  const height = layout.height + padding * 2;
  const effectiveStep = hasExternalFrame ? Number(frameStep) : step;
  const animated = (hasExternalFrame || shouldAnimate) && maxBirthStep > 0;

  return (
    <div
      className={`cellengine-development-mini${compact ? " compact" : ""}`}
      onMouseEnter={animateOnHover ? () => setHovered(true) : undefined}
      onMouseLeave={animateOnHover ? () => setHovered(false) : undefined}
    >
      <svg className={`cellengine-mini-map${animated ? " selected" : ""}${compact ? " compact" : ""}`} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Live development preview">
        <rect x="0" y="0" width={width} height={height} rx="16" fill="#fcfcfa" />
        {sortProjectedCells(layout.cells).map((cell) => {
          const normalized = clamp01((((cell[metricId] || 0) - minValue) / span));
          const drawSize = layout.hasDepth ? Math.max(8.4, size * (0.84 + cell.plotDepth * 0.24)) : size - 2;
          const x = padding + cell.plotX + ((size - 2) - drawSize) / 2;
          const y = padding + cell.plotY + ((size - 2) - drawSize) / 2;
          const born = (cell.birth_step || 0) <= effectiveStep || !animated;
          return (
            <g
              key={cell.cell_id}
              style={depthShadowStyle(cell.plotDepth, false)}
              opacity={born ? (layout.hasDepth ? lerp(0.58, 1, cell.plotDepth) : 1) : 0.32}
            >
              <rect
                x={x}
                y={y}
                width={drawSize}
                height={drawSize}
                rx={layout.hasDepth ? 4.8 : 4}
                fill={born ? staticMetricColor(metricId, normalized) : "#eef3f5"}
                stroke={born ? cellTypeStroke(cell) : "#d7e3e7"}
                strokeWidth={animated ? 1.6 : 1.2}
              />
            </g>
          );
        })}
      </svg>
      {footer ? (
        <div className="cellengine-development-step">
          {animated ? `development ${Math.min(effectiveStep, maxBirthStep)} / ${maxBirthStep}` : maxBirthStep > 0 ? `full body developed in ${maxBirthStep} steps` : "single-step body"}
        </div>
      ) : null}
    </div>
  );
}

const SURFACE_DISCLAIMER_ITEMS = [
  "Not the GA loss landscape",
  "Not a training trajectory",
  "Not a control-policy map",
  "Not a success plot",
  "A low point here only means the selected metric is locally low at this body region"
];
