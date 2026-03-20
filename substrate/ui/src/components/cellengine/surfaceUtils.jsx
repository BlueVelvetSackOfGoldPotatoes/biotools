import {
  clamp01,
  formatNumber,
  formatPct,
  lerp,
  surfacePaletteColor,
  taskLabel
} from "./core";
import { bodyCoord, bodyExtents, buildOrganismSurfaceModel } from "./body";

const SURFACE_DISCLAIMER_ITEMS = [
  "Not the GA loss landscape",
  "Not a training trajectory",
  "Not a control-policy map",
  "Not a success plot",
  "A low point here only means the selected metric is locally low at this body region"
];
export function surfaceCarouselStripItems(activeMode) {
  if (activeMode === "ga-fitness") {
    return [
      "Actual GA fitness trace for this search run",
      "Shows best, mean, and worst candidate fitness by generation",
      "Not a spatial body field"
    ];
  }
  if (activeMode === "search") {
    return [
      "Actual search-time trajectory for this run",
      "Shows success and survival across generations",
      "Not a spatial body field"
    ];
  }
  if (activeMode === "control") {
    return [
      "Static body control proxy",
      "Derived from contractility, coupling, leverage, and directional drive",
      "Not the real closed-loop control policy"
    ];
  }
  if (activeMode === "success") {
    return [
      "Actual benchmark outcome summary",
      "Shows solved-task metrics for this organism",
      "Not a spatial body field"
    ];
  }
  return SURFACE_DISCLAIMER_ITEMS;
}

export function normalizePlotRange(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (!finite.length) {
    return { min: 0, max: 1, span: 1 };
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = Math.max(1e-6, max - min);
  return { min, max, span };
}

export function SurfacePointsMiniPlot({ surface }) {
  const width = 240;
  const height = 140;
  const pad = 16;
  const xRange = normalizePlotRange(surface.visibleCells.map((cell) => cell.x));
  const yRange = normalizePlotRange(surface.visibleCells.map((cell) => cell.y));
  return (
    <svg className="cellengine-surface-carousel-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Actual cell metric points">
      <rect x="0" y="0" width={width} height={height} rx="16" fill="#fbfdfc" />
      {surface.visibleCells.map((cell) => {
        const px = pad + ((cell.x - xRange.min) / xRange.span) * (width - pad * 2);
        const py = height - pad - ((cell.y - yRange.min) / yRange.span) * (height - pad * 2);
        return (
          <circle
            key={`surface-point-${cell.cell_id}`}
            cx={px}
            cy={py}
            r={lerp(4, 8, cell.normalizedValue)}
            fill={surfacePaletteColor(cell.normalizedValue)}
            stroke="#ffffff"
            strokeWidth="1.2"
          />
        );
      })}
    </svg>
  );
}

export function SurfaceHeatmapMiniPlot({ surface }) {
  const previewSize = 240;
  const cols = surface.gridX.length;
  const rows = surface.gridY.length;
  const cellSize = previewSize / Math.max(cols, rows);
  return (
    <svg className="cellengine-surface-carousel-plot" viewBox={`0 0 ${previewSize} ${previewSize}`} role="img" aria-label="Interpolated metric field">
      <rect x="0" y="0" width={previewSize} height={previewSize} rx="16" fill="#fbfdfc" />
      {surface.zRows.map((row, yi) => row.map((value, xi) => (
        <rect
          key={`surface-heat-${xi}-${yi}`}
          x={xi * cellSize}
          y={yi * cellSize}
          width={cellSize + 0.4}
          height={cellSize + 0.4}
          fill={surfacePaletteColor(value)}
        />
      )))}
      {surface.minima.map((point, index) => {
        const xi = surface.gridX.findIndex((value) => value >= point.x);
        const yi = surface.gridY.findIndex((value) => value >= point.y);
        const px = (Math.max(0, xi) + 0.5) * cellSize;
        const py = (Math.max(0, yi) + 0.5) * cellSize;
        return <circle key={`surface-heat-min-${index}`} cx={px} cy={py} r="3.2" fill="#d9480f" stroke="#fff5f5" strokeWidth="1" />;
      })}
    </svg>
  );
}

export function SurfaceLayerMiniPlot({ bodyCells, metricId }) {
  const layers = [...new Set((bodyCells || []).map((cell) => bodyCoord(cell, "z")))].sort((left, right) => left - right);
  const stats = layers.map((layer) => {
    const values = (bodyCells || [])
      .filter((cell) => bodyCoord(cell, "z") === layer)
      .map((cell) => Number(cell?.[metricId]))
      .filter((value) => Number.isFinite(value));
    return {
      layer,
      mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
    };
  });
  const width = 240;
  const height = 140;
  const pad = 20;
  const range = normalizePlotRange(stats.map((entry) => entry.mean));
  const barWidth = Math.max(20, (width - pad * 2) / Math.max(1, stats.length) - 10);
  return (
    <svg className="cellengine-surface-carousel-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-layer metric means">
      <rect x="0" y="0" width={width} height={height} rx="16" fill="#fbfdfc" />
      {stats.map((entry, index) => {
        const x = pad + index * ((width - pad * 2) / Math.max(1, stats.length));
        const h = ((entry.mean - range.min) / range.span) * (height - pad * 2);
        const y = height - pad - h;
        return (
          <g key={`surface-layer-bar-${entry.layer}`}>
            <rect x={x} y={y} width={barWidth} height={Math.max(3, h)} rx="8" fill={surfacePaletteColor(clamp01((entry.mean - range.min) / range.span))} />
            <text x={x + barWidth / 2} y={height - 6} textAnchor="middle" className="cellengine-surface-plot-label">z {entry.layer}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function SurfaceHistogramMiniPlot({ surface }) {
  const width = 240;
  const height = 140;
  const pad = 20;
  const bins = Array.from({ length: 6 }, (_, index) => ({
    index,
    count: 0
  }));
  for (const cell of surface.visibleCells) {
    const index = Math.min(5, Math.floor(clamp01(cell.normalizedValue) * 6));
    bins[index].count += 1;
  }
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const barWidth = Math.max(16, (width - pad * 2) / bins.length - 8);
  return (
    <svg className="cellengine-surface-carousel-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Metric distribution histogram">
      <rect x="0" y="0" width={width} height={height} rx="16" fill="#fbfdfc" />
      {bins.map((bin, index) => {
        const x = pad + index * ((width - pad * 2) / bins.length);
        const h = (bin.count / maxCount) * (height - pad * 2);
        const y = height - pad - h;
        const t = (index + 0.5) / bins.length;
        return (
          <g key={`surface-hist-${index}`}>
            <rect x={x} y={y} width={barWidth} height={Math.max(3, h)} rx="8" fill={surfacePaletteColor(t)} />
            <text x={x + barWidth / 2} y={height - 6} textAnchor="middle" className="cellengine-surface-plot-label">{index + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function extractSurfaceSummary(summary) {
  if (!summary || typeof summary !== "object") return null;
  if (summary.cell_clean || summary.cell_damaged) {
    return {
      success_rate: Number(summary.cell_clean?.success_rate),
      survival_ratio: Number(summary.cell_clean?.survival_ratio),
      damage_survival_ratio: Number(summary.cell_damaged?.survival_ratio),
      task_primary: Number(summary.cell_clean?.task_primary),
      task_primary_label: summary.cell_clean?.task_primary_label || "task"
    };
  }
  return {
    success_rate: Number(summary?.success_rate),
    survival_ratio: Number(summary?.survival_ratio),
    damage_survival_ratio: Number(summary?.damage_survival_ratio),
    task_primary: Number(summary?.task_primary),
    task_primary_label: summary?.task_primary_label || "task"
  };
}

export const DEFAULT_SURFACE_COLORSCALE = [
  [0, "#233b63"],
  [0.25, "#335f8a"],
  [0.5, "#4c8f64"],
  [0.75, "#c97a27"],
  [1, "#8d2e63"]
];

export function controlProxyValue(cell) {
  return clamp01(
    clamp01(Number(cell?.gene_expr_1)) * 0.42 +
    clamp01(Number(cell?.gene_expr_6)) * 0.28 +
    clamp01(Number(cell?.mech_advantage)) * 0.18 +
    clamp01(Math.abs(Number(cell?.force_direction_x) || 0)) * 0.12
  );
}

export function buildSeriesLandscape(rows, seriesDefs, { normalizeZ = false } = {}) {
  const normalizedRows = Array.isArray(rows)
    ? rows.filter((row) => Number.isFinite(Number(row?.generation)))
    : [];
  if (!normalizedRows.length || !Array.isArray(seriesDefs) || !seriesDefs.length) return null;
  const x = normalizedRows.map((row) => Number(row.generation));
  const y = seriesDefs.map((_, index) => index);
  const rawZ = seriesDefs.map((series) =>
    normalizedRows.map((row) => {
      const value = Number(row?.[series.key]);
      return Number.isFinite(value) ? value : null;
    })
  );
  const finiteValues = rawZ.flat().filter((value) => Number.isFinite(value));
  if (!finiteValues.length) return null;
  const zRange = normalizePlotRange(finiteValues);
  const zRows = normalizeZ
    ? rawZ.map((row) => row.map((value) => (Number.isFinite(value) ? clamp01((value - zRange.min) / zRange.span) : null)))
    : rawZ;
  const points = [];
  for (let yi = 0; yi < seriesDefs.length; yi += 1) {
    for (let xi = 0; xi < normalizedRows.length; xi += 1) {
      const raw = rawZ[yi]?.[xi];
      const z = zRows[yi]?.[xi];
      if (!Number.isFinite(raw) || !Number.isFinite(z)) continue;
      points.push({
        x: x[xi],
        y: yi,
        z,
        raw,
        label: seriesDefs[yi].label
      });
    }
  }
  return {
    x,
    y,
    yLabels: seriesDefs.map((series) => series.label),
    zRows,
    rawZRows: rawZ,
    points,
    zMin: normalizeZ ? 0 : zRange.min,
    zMax: normalizeZ ? 1 : zRange.max,
    normalized: normalizeZ
  };
}

export function buildMetricStripLandscape(entries) {
  const finiteEntries = (entries || []).filter((entry) => Number.isFinite(Number(entry?.value)));
  if (!finiteEntries.length) return null;
  const x = finiteEntries.map((_, index) => index);
  const y = [0, 1];
  const values = finiteEntries.map((entry) => clamp01(Number(entry.value)));
  const zRows = [
    values,
    values.map((value, index) => clamp01(value * 0.92 + 0.04 * ((index + 1) / Math.max(1, values.length))))
  ];
  const points = finiteEntries.map((entry, index) => ({
    x: index,
    y: 0.5,
    z: clamp01(Number(entry.value)),
    raw: Number(entry.value),
    label: entry.label
  }));
  return {
    x,
    y,
    xLabels: finiteEntries.map((entry) => entry.label),
    zRows,
    points,
    zMin: 0,
    zMax: 1
  };
}

export function buildControlLandscape(bodyCells, layerSelection = "aggregate") {
  const cells = (Array.isArray(bodyCells) ? bodyCells : [])
    .filter((cell) => layerSelection === "aggregate" || bodyCoord(cell, "z") === layerSelection)
    .map((cell) => ({ ...cell, __control_proxy: controlProxyValue(cell) }));
  return buildOrganismSurfaceModel(cells, "__control_proxy", "aggregate");
}

export function buildPopulationLandscape(candidates) {
  const points = (Array.isArray(candidates) ? candidates : []).map((candidate) => {
    const bodyCells = Array.isArray(candidate?.body_cells) ? candidate.body_cells : [];
    if (!bodyCells.length) return null;
    const extents = bodyExtents(bodyCells);
    const summary = candidate?.summary && typeof candidate.summary === "object" ? candidate.summary : null;
    const success = Number(summary?.success_rate);
    if (!Number.isFinite(success)) return null;
    const spread = extents.spanX + 0.45 * extents.spanZ + 1;
    const verticality = (extents.spanY + 1) / Math.max(1, extents.spanX + extents.spanZ + 1);
    return {
      x: spread,
      y: verticality,
      z: clamp01(success),
      label: `#${candidate.rank || "?"}`,
      bodyMode: candidate.body_mode || "body",
      cellCount: Number(candidate?.cell_count) || bodyCells.length
    };
  }).filter(Boolean);
  if (!points.length) return null;
  const xRange = normalizePlotRange(points.map((point) => point.x));
  const yRange = normalizePlotRange(points.map((point) => point.y));
  const gridX = Array.from({ length: 14 }, (_, index) =>
    xRange.min + (xRange.span * index) / Math.max(1, 13)
  );
  const gridY = Array.from({ length: 14 }, (_, index) =>
    yRange.min + (yRange.span * index) / Math.max(1, 13)
  );
  const zRows = gridY.map((gy) => gridX.map((gx) => {
    let numerator = 0;
    let denominator = 0;
    for (const point of points) {
      const dx = (gx - point.x) / xRange.span;
      const dy = (gy - point.y) / yRange.span;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1e-6) return point.z;
      const weight = 1 / (0.03 + dist * dist);
      numerator += point.z * weight;
      denominator += weight;
    }
    return denominator > 0 ? numerator / denominator : 0;
  }));
  return {
    gridX,
    gridY,
    zRows,
    points,
    xLabel: "shape spread",
    yLabel: "verticality",
    zLabel: "success",
    xRange,
    yRange
  };
}
