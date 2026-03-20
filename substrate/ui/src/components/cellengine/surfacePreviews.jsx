import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { formatGenomeMetricLabel, formatNumber, formatPct, lerp, shortHash, surfacePaletteColor } from "./core";
import {
  bodyDepthCount,
  bodyHasDepth,
  buildOrganismSurfaceModel,
  formatBodyCellPosition,
  projectBodyLayout
} from "./body";
import {
  DEFAULT_SURFACE_COLORSCALE,
  SurfaceHeatmapMiniPlot,
  SurfaceHistogramMiniPlot,
  SurfaceLayerMiniPlot,
  SurfacePointsMiniPlot,
  buildControlLandscape,
  buildMetricStripLandscape,
  buildPopulationLandscape,
  buildSeriesLandscape,
  controlProxyValue,
  extractSurfaceSummary,
  surfaceCarouselStripItems
} from "./surfaceUtils";
import { FieldLabel, SectionTitle } from "./uiPrimitives";

const Plot3D = lazy(() => import("../Plot3D"));
export function OrganismSurfacePreview({ bodyCells, metricId, title, note = "", summary = null, searchHistory = [] }) {
  const [open, setOpen] = useState(false);
  const depthLayers = bodyDepthCount(bodyCells);
  const is3dBody = bodyHasDepth(bodyCells);
  const [layerSelection, setLayerSelection] = useState("aggregate");
  const [surfaceBlend, setSurfaceBlend] = useState(0.8);
  const [surfaceSlideIndex, setSurfaceSlideIndex] = useState(0);

  useEffect(() => {
    if (!is3dBody && layerSelection !== "aggregate") {
      setLayerSelection("aggregate");
    }
  }, [is3dBody, layerSelection]);

  const surface = useMemo(() => buildOrganismSurfaceModel(bodyCells, metricId, layerSelection), [bodyCells, metricId, layerSelection]);
  const surfaceSummary = useMemo(() => extractSurfaceSummary(summary), [summary]);
  const validSearchHistory = useMemo(
    () => (Array.isArray(searchHistory) ? searchHistory.filter((row) => Number.isFinite(Number(row?.generation))) : []),
    [searchHistory]
  );
  const gaLandscape = useMemo(
    () => buildSeriesLandscape(validSearchHistory, [
      { key: "best_fitness", label: "best fitness" },
      { key: "mean_fitness", label: "mean fitness" },
      { key: "worst_fitness", label: "worst fitness" }
    ]),
    [validSearchHistory]
  );
  const searchLandscape = useMemo(
    () => buildSeriesLandscape(validSearchHistory, [
      { key: "best_success_rate", label: "best success" },
      { key: "mean_success_rate", label: "mean success" },
      { key: "mean_survival_ratio", label: "mean survival" }
    ], { normalizeZ: true }),
    [validSearchHistory]
  );
  const controlSurface = useMemo(
    () => buildControlLandscape(bodyCells, layerSelection),
    [bodyCells, layerSelection]
  );
  const successLandscape = useMemo(() => {
    if (!surfaceSummary) return null;
    const values = [
      { label: "success", value: Number(surfaceSummary?.success_rate) },
      { label: "survival", value: Number(surfaceSummary?.survival_ratio) }
    ];
    if (Number.isFinite(Number(surfaceSummary?.damage_survival_ratio))) {
      values.push({ label: "damage", value: Number(surfaceSummary.damage_survival_ratio) });
    }
    return buildMetricStripLandscape(values);
  }, [surfaceSummary]);
  const hasSearchHistory = validSearchHistory.length > 0;
  const hasFitnessHistory = Boolean(gaLandscape);
  const hasSummaryPlot = Boolean(successLandscape);
  if (!surface) return null;

  const surfaceSlides = useMemo(() => {
    const slides = [
      {
        id: "field",
        title: "Field Surface",
        note: "Smoothed plane inferred from the visible cell points.",
        mode: "field-surface"
      }
    ];
    if (is3dBody) {
      slides.push({
        id: "voxels",
        title: "Body Voxels",
        note: "Actual 3D cell positions colored by the selected metric.",
        mode: "body-voxels"
      });
    }
    slides.push(
      {
        id: "points",
        title: "Measured Points",
        note: "Only the measured cell positions, without interpolation.",
        mode: "points"
      }
    );
    if (hasFitnessHistory) {
      slides.push({
        id: "ga-fitness",
        title: "GA Fitness Trace",
        note: "Actual generation-by-generation GA fitness for this search run when available.",
        mode: "ga-fitness"
      });
    }
    if (hasSearchHistory) {
      slides.push({
        id: "search",
        title: "Training / Search Trajectory",
        note: "Actual search-time success and survival trace across generations for this run.",
        mode: "search"
      });
    }
    slides.push({
      id: "control",
      title: "Control Proxy Map",
      note: "Static contractility/coupling/leverage proxy over the current body, not the real closed-loop policy.",
      mode: "control"
    });
    if (hasSummaryPlot) {
      slides.push({
        id: "success",
        title: "Success Plot",
        note: "Actual benchmark outcome metrics for this organism, when summary data exists.",
        mode: "success"
      });
    }
    slides.push({
      id: "distribution",
      title: "Metric Distribution",
      note: "Histogram of the selected metric for the currently visible cells.",
      mode: "histogram"
    });
    if (is3dBody) {
      slides.push({
        id: "layers",
        title: "Z-Layer Means",
        note: "Mean selected-metric value by z layer.",
        mode: "layer-means"
      });
    }
    return slides;
  }, [hasFitnessHistory, hasSearchHistory, hasSummaryPlot, is3dBody]);

  useEffect(() => {
    if (surfaceSlideIndex >= surfaceSlides.length) {
      setSurfaceSlideIndex(0);
    }
  }, [surfaceSlideIndex, surfaceSlides.length]);

  const activeSlide = surfaceSlides[surfaceSlideIndex] || surfaceSlides[0];
  const activeMode = activeSlide?.mode || "field-surface";
  const totalSlides = surfaceSlides.length;
  const surface3dModes = new Set(["field-surface", "body-voxels", "ga-fitness", "search", "control", "success"]);
  const isSurface3dMode = surface3dModes.has(activeMode);

  const previewSize = 42;
  const cols = surface.gridX.length;
  const rows = surface.gridY.length;
  const cellSize = previewSize / Math.max(cols, rows);

  return (
    <>
      <button
        type="button"
        className="cellengine-surface-preview"
        onClick={() => setOpen(true)}
        title={`Open spatial body field for ${formatGenomeMetricLabel(metricId)}`}
      >
        <svg viewBox={`0 0 ${previewSize} ${previewSize}`} role="img" aria-label="Organism surface preview">
          <rect x="0" y="0" width={previewSize} height={previewSize} rx="12" fill="#f7fbfa" />
          {surface.zRows.map((row, yi) => row.map((value, xi) => (
            <rect
              key={`surface-${xi}-${yi}`}
              x={xi * cellSize}
              y={yi * cellSize}
              width={cellSize + 0.6}
              height={cellSize + 0.6}
              fill={surfacePaletteColor(value)}
            />
          )))}
          {surface.minima.map((point, index) => {
            const xi = surface.gridX.findIndex((value) => value >= point.x);
            const yi = surface.gridY.findIndex((value) => value >= point.y);
            const px = (Math.max(0, xi) + 0.5) * cellSize;
            const py = (Math.max(0, yi) + 0.5) * cellSize;
            return <circle key={`surface-min-${index}`} cx={px} cy={py} r="1.8" fill="#d9480f" stroke="#fff5f5" strokeWidth="0.9" />;
          })}
        </svg>
        <span aria-hidden="true">S</span>
      </button>
      {open ? (
        <div className="cellengine-surface-modal-backdrop" onClick={() => setOpen(false)}>
          <div className="cellengine-surface-modal" onClick={(event) => event.stopPropagation()}>
            <div className="section-head">
              <SectionTitle
                title={title || "Organism Field Surface"}
                help="Smoothed spatial field over the organism body for the selected metric. X and Y are body coordinates; height and color encode the selected genome/body metric. This is not the GA loss landscape, not a trajectory, and not proof of task success."
              />
              <div className="row controls">
                {note ? <span className="meta-note">{note}</span> : null}
                <button type="button" className="mini-btn" onClick={() => setOpen(false)}>close</button>
              </div>
            </div>
            <div className="meta-note">
              {formatGenomeMetricLabel(metricId)} · red markers are spatial low points in this body field, not training or search minima
            </div>
            <div className="cellengine-surface-toolbar">
              <div className="cellengine-surface-carousel-head">
                <strong>{activeSlide.title}</strong>
                <div className="cellengine-surface-carousel-controls">
                  <button type="button" className="mini-btn" onClick={() => setSurfaceSlideIndex((surfaceSlideIndex - 1 + totalSlides) % totalSlides)}>
                    prev
                  </button>
                  <span>{surfaceSlideIndex + 1} / {totalSlides}</span>
                  <button type="button" className="mini-btn" onClick={() => setSurfaceSlideIndex((surfaceSlideIndex + 1) % totalSlides)}>
                    next
                  </button>
                </div>
              </div>
              {is3dBody ? (
                <label>
                  <FieldLabel
                    label="Layer"
                    help="For 3D bodies, choose whether to view all z layers aggregated together or inspect one z slice at a time."
                  />
                  <select className="task-input" value={String(layerSelection)} onChange={(event) => setLayerSelection(event.target.value === "aggregate" ? "aggregate" : Number(event.target.value))}>
                    <option value="aggregate">all layers</option>
                    {surface.layers.map((layer) => (
                      <option key={`surface-layer-${layer}`} value={String(layer)}>
                        z = {layer}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {activeMode === "field-surface" ? (
                <label>
                  <FieldLabel
                    label="Field blend"
                    help="0 shows only the measured body points. 100 shows the full interpolated surface inferred from those points."
                  />
                  <input
                    className="task-input"
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={Math.round(surfaceBlend * 100)}
                    onChange={(event) => setSurfaceBlend(Number(event.target.value) / 100)}
                  />
                </label>
              ) : null}
            </div>
            <div className="cellengine-surface-stats">
              <span>{surface.visibleCells.length} cells</span>
              <span>{is3dBody ? `${depthLayers} z layers total` : "flat body"}</span>
              <span>metric min {formatNumber(surface.metricMin, 3)}</span>
              <span>metric mean {formatNumber(surface.metricMean, 3)}</span>
              <span>metric max {formatNumber(surface.metricMax, 3)}</span>
              <span>{layerSelection === "aggregate" ? "showing aggregated layers" : `showing z = ${layerSelection}`}</span>
              {activeMode === "field-surface" ? <span>surface {Math.round(surfaceBlend * 100)}%</span> : null}
            </div>
            <div className="cellengine-surface-disclaimer-strip">
              {surfaceCarouselStripItems(activeMode).map((text) => (
                <span key={text}>{text}</span>
              ))}
            </div>
            <div className="cellengine-surface-modal-plot">
              <div className="cellengine-surface-carousel-caption">
                <strong>{activeSlide.title}</strong>
                <span>{activeSlide.note}</span>
              </div>
              <div className={`cellengine-surface-main-plot${isSurface3dMode ? " three-d" : " svg"}`}>
                {activeMode === "field-surface" || activeMode === "body-voxels" || activeMode === "ga-fitness" || activeMode === "search" || activeMode === "control" || activeMode === "success" ? (
                  <Suspense fallback={<div className="empty">Loading 3D surface…</div>}>
                    <Plot3D
                      data={activeMode === "body-voxels"
                      ? [
                          {
                            type: "scatter3d",
                            mode: "markers+text",
                            x: surface.visibleCells.map((cell) => cell.x),
                            y: surface.visibleCells.map((cell) => cell.y),
                            z: surface.visibleCells.map((cell) => cell.z),
                            text: surface.visibleCells.map((cell) => String(cell.cell_id)),
                            textposition: "top center",
                            textfont: { size: 9, color: "#334155" },
                            marker: {
                              size: surface.visibleCells.map((cell) => lerp(6, 12, cell.normalizedValue)),
                              color: surface.visibleCells.map((cell) => cell.normalizedValue),
                              colorscale: [
                                [0, "#233b63"],
                                [0.25, "#335f8a"],
                                [0.5, "#4c8f64"],
                                [0.75, "#c97a27"],
                                [1, "#8d2e63"]
                              ],
                              cmin: 0,
                              cmax: 1,
                              showscale: true,
                              colorbar: { title: { text: formatGenomeMetricLabel(metricId) } },
                              line: { color: "rgba(255,255,255,0.8)", width: 0.8 },
                              opacity: 0.94
                            },
                            name: "body cells",
                            hovertemplate: "cell %{text}<br>x %{x}<br>y %{y}<br>z %{z}<br>metric %{marker.color:.3f}<extra></extra>"
                          }
                        ]
                      : activeMode === "ga-fitness" && gaLandscape
                        ? [
                            {
                              type: "surface",
                              x: gaLandscape.x,
                              y: gaLandscape.y,
                              z: gaLandscape.zRows,
                              colorscale: DEFAULT_SURFACE_COLORSCALE,
                              showscale: true,
                              colorbar: { title: { text: "fitness" } },
                              hovertemplate: "generation %{x}<br>series %{y}<br>fitness %{z:.3f}<extra></extra>"
                            },
                            ...gaLandscape.yLabels.map((label, yi) => ({
                              type: "scatter3d",
                              mode: "lines+markers",
                              x: gaLandscape.points.filter((point) => point.label === label).map((point) => point.x),
                              y: gaLandscape.points.filter((point) => point.label === label).map((point) => point.y),
                              z: gaLandscape.points.filter((point) => point.label === label).map((point) => point.raw),
                              name: label,
                              marker: { size: 3.6 },
                              line: { width: 4 }
                            }))
                          ]
                      : activeMode === "search" && searchLandscape
                        ? [
                            {
                              type: "surface",
                              x: searchLandscape.x,
                              y: searchLandscape.y,
                              z: searchLandscape.zRows,
                              colorscale: DEFAULT_SURFACE_COLORSCALE,
                              showscale: true,
                              colorbar: { title: { text: "ratio" } },
                              hovertemplate: "generation %{x}<br>series %{y}<br>value %{z:.3f}<extra></extra>"
                            },
                            ...searchLandscape.yLabels.map((label, yi) => ({
                              type: "scatter3d",
                              mode: "lines+markers",
                              x: searchLandscape.points.filter((point) => point.label === label).map((point) => point.x),
                              y: searchLandscape.points.filter((point) => point.label === label).map((point) => point.y),
                              z: searchLandscape.points.filter((point) => point.label === label).map((point) => point.z),
                              name: label,
                              marker: { size: 3.6 },
                              line: { width: 4 }
                            }))
                          ]
                      : activeMode === "control" && controlSurface
                        ? [
                            {
                              type: "surface",
                              x: controlSurface.gridX,
                              y: controlSurface.gridY,
                              z: controlSurface.zRows,
                              colorscale: DEFAULT_SURFACE_COLORSCALE,
                              showscale: true,
                              colorbar: { title: { text: "control proxy" } },
                              contours: {
                                z: {
                                  show: true,
                                  usecolormap: false,
                                  highlightcolor: "#ffffff",
                                  project: { z: true }
                                }
                              },
                              hovertemplate: "x %{x:.2f}<br>y %{y:.2f}<br>control proxy %{z:.3f}<extra></extra>"
                            },
                            {
                              type: "scatter3d",
                              mode: "markers",
                              x: controlSurface.visibleCells.map((cell) => cell.x),
                              y: controlSurface.visibleCells.map((cell) => cell.y),
                              z: controlSurface.visibleCells.map((cell) => cell.normalizedValue),
                              marker: {
                                size: controlSurface.visibleCells.map((cell) => lerp(4.4, 8.2, cell.normalizedValue)),
                                color: controlSurface.visibleCells.map((cell) => cell.normalizedValue),
                                colorscale: DEFAULT_SURFACE_COLORSCALE,
                                cmin: 0,
                                cmax: 1,
                                opacity: 0.82,
                                line: { color: "rgba(255,255,255,0.85)", width: 0.7 }
                              },
                              name: "control points",
                              hovertemplate: "cell %{customdata[0]}<br>x %{x}<br>y %{y}<br>body z %{customdata[1]}<br>proxy %{z:.3f}<extra></extra>",
                              customdata: controlSurface.visibleCells.map((cell) => [cell.cell_id, cell.z])
                            }
                          ]
                      : activeMode === "success" && successLandscape
                        ? [
                            {
                              type: "surface",
                              x: successLandscape.x,
                              y: successLandscape.y,
                              z: successLandscape.zRows,
                              colorscale: DEFAULT_SURFACE_COLORSCALE,
                              showscale: true,
                              colorbar: { title: { text: "ratio" } },
                              hovertemplate: "metric %{x}<br>value %{z:.3f}<extra></extra>"
                            },
                            {
                              type: "scatter3d",
                              mode: "markers+text",
                              x: successLandscape.points.map((point) => point.x),
                              y: successLandscape.points.map((point) => point.y),
                              z: successLandscape.points.map((point) => point.z),
                              text: successLandscape.points.map((point) => point.label),
                              textposition: "top center",
                              marker: {
                                size: 6,
                                color: successLandscape.points.map((point) => point.z),
                                colorscale: DEFAULT_SURFACE_COLORSCALE,
                                cmin: 0,
                                cmax: 1,
                                line: { color: "rgba(255,255,255,0.85)", width: 0.8 }
                              },
                              name: "outcome metrics",
                              hovertemplate: "%{text}<br>value %{z:.3f}<extra></extra>"
                            }
                          ]
                      : [
                          {
                            type: "surface",
                            x: surface.gridX,
                            y: surface.gridY,
                            z: surface.zRows,
                            colorscale: [
                              [0, "#233b63"],
                              [0.25, "#335f8a"],
                              [0.5, "#4c8f64"],
                              [0.75, "#c97a27"],
                              [1, "#8d2e63"]
                            ],
                            opacity: surfaceBlend,
                            showscale: true,
                            colorbar: { title: { text: formatGenomeMetricLabel(metricId) } },
                            contours: {
                              z: {
                                show: true,
                                usecolormap: false,
                                highlightcolor: "#ffffff",
                                project: { z: true }
                              }
                            },
                            hovertemplate: "x %{x:.2f}<br>y %{y:.2f}<br>metric %{z:.3f}<extra></extra>"
                          },
                          {
                            type: "scatter3d",
                            mode: "markers",
                            x: surface.visibleCells.map((cell) => cell.x),
                            y: surface.visibleCells.map((cell) => cell.y),
                            z: surface.visibleCells.map((cell) => cell.normalizedValue),
                            marker: {
                              size: surface.visibleCells.map((cell) => lerp(4.4, 8.2, cell.normalizedValue)),
                              color: surface.visibleCells.map((cell) => cell.normalizedValue),
                              colorscale: [
                                [0, "#233b63"],
                                [0.25, "#335f8a"],
                                [0.5, "#4c8f64"],
                                [0.75, "#c97a27"],
                                [1, "#8d2e63"]
                              ],
                              cmin: 0,
                              cmax: 1,
                              opacity: lerp(0.92, 0.58, surfaceBlend),
                              line: { color: "rgba(255,255,255,0.85)", width: 0.7 }
                            },
                            name: "actual cells",
                            hovertemplate: "cell %{customdata[0]}<br>x %{x}<br>y %{y}<br>body z %{customdata[1]}<br>metric %{z:.3f}<extra></extra>",
                            customdata: surface.visibleCells.map((cell) => [cell.cell_id, cell.z])
                          },
                          {
                            type: "scatter3d",
                            mode: "markers",
                            x: surface.minima.map((point) => point.x),
                            y: surface.minima.map((point) => point.y),
                            z: surface.minima.map((point) => point.z),
                            marker: { size: 4.5, color: "#d9480f" },
                            name: "spatial low points",
                            hovertemplate: "spatial low point<br>x %{x:.2f}<br>y %{y:.2f}<br>metric %{z:.3f}<extra></extra>"
                          }
                        ]}
                      layout={{
                        autosize: true,
                        paper_bgcolor: "rgba(0,0,0,0)",
                        plot_bgcolor: "rgba(0,0,0,0)",
                        margin: { l: 0, r: 0, t: 10, b: 0 },
                        scene: {
                          bgcolor: "rgba(0,0,0,0)",
                          xaxis: activeMode === "ga-fitness" || activeMode === "search"
                            ? { title: "generation" }
                            : activeMode === "success"
                              ? { title: "metric index", tickvals: successLandscape?.x || [], ticktext: successLandscape?.xLabels || [] }
                              : { title: "x" },
                          yaxis: activeMode === "ga-fitness"
                            ? { title: "fitness band", tickvals: gaLandscape?.y || [], ticktext: gaLandscape?.yLabels || [] }
                            : activeMode === "search"
                              ? { title: "trajectory band", tickvals: searchLandscape?.y || [], ticktext: searchLandscape?.yLabels || [] }
                              : activeMode === "success"
                                ? { title: "surface strip" }
                                : { title: "y" },
                          zaxis: activeMode === "body-voxels"
                            ? { title: "body z layer" }
                            : activeMode === "control"
                              ? { title: "control proxy", range: [0, 1] }
                              : activeMode === "success"
                                ? { title: "ratio", range: [0, 1] }
                                : activeMode === "search"
                                  ? { title: "ratio", range: [0, 1] }
                                  : activeMode === "ga-fitness"
                                    ? { title: "fitness" }
                                    : { title: "metric value", range: [0, 1] },
                          camera: { eye: { x: 1.45, y: -1.5, z: 0.9 } }
                        },
                        legend: { orientation: "h", x: 0, y: 1.04 }
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      style={{ width: "100%", height: "100%" }}
                      useResizeHandler
                    />
                  </Suspense>
                ) : activeMode === "points" ? (
                  <SurfacePointsMiniPlot surface={surface} />
                ) : activeMode === "layer-means" ? (
                  <SurfaceLayerMiniPlot bodyCells={bodyCells} metricId={metricId} />
                ) : (
                  <SurfaceHistogramMiniPlot surface={surface} />
                )}
              </div>
              <div className="cellengine-surface-carousel-dots" aria-hidden="true">
                {surfaceSlides.map((entry, index) => (
                  <button
                    key={`surface-slide-${entry.id}`}
                    type="button"
                    className={`cellengine-surface-carousel-dot${index === surfaceSlideIndex ? " active" : ""}`}
                    onClick={() => setSurfaceSlideIndex(index)}
                    title={entry.title}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

export function PopulationLandscapePreview({ candidates, title = "Population Shape Landscape", note = "" }) {
  const [open, setOpen] = useState(false);
  const landscape = useMemo(() => buildPopulationLandscape(candidates), [candidates]);
  if (!landscape) return null;
  const previewSize = 44;
  const cols = landscape.gridX.length;
  const rows = landscape.gridY.length;
  const cellSize = previewSize / Math.max(cols, rows);

  return (
    <>
      <button
        type="button"
        className="cellengine-surface-preview cellengine-population-landscape-preview"
        onClick={() => setOpen(true)}
        title="Open population shape-success landscape"
      >
        <svg viewBox={`0 0 ${previewSize} ${previewSize}`} role="img" aria-label="Population shape-success landscape preview">
          <rect x="0" y="0" width={previewSize} height={previewSize} rx="12" fill="#f7fbfa" />
          {landscape.zRows.map((row, yi) => row.map((value, xi) => (
            <rect
              key={`population-landscape-${xi}-${yi}`}
              x={xi * cellSize}
              y={yi * cellSize}
              width={cellSize + 0.6}
              height={cellSize + 0.6}
              fill={surfacePaletteColor(value)}
            />
          )))}
          {landscape.points.map((point, index) => {
            const x = ((point.x - landscape.xRange.min) / landscape.xRange.span) * (previewSize - 8) + 4;
            const y = previewSize - (((point.y - landscape.yRange.min) / landscape.yRange.span) * (previewSize - 8) + 4);
            return (
              <circle
                key={`population-landscape-point-${index}`}
                cx={x}
                cy={y}
                r="1.4"
                fill="#f8fafc"
                stroke="#0f172a"
                strokeWidth="0.6"
              />
            );
          })}
        </svg>
        <span aria-hidden="true">P</span>
      </button>
      {open ? (
        <div className="cellengine-surface-modal-backdrop" onClick={() => setOpen(false)}>
          <div className="cellengine-surface-modal" onClick={(event) => event.stopPropagation()}>
            <div className="section-head">
              <SectionTitle
                title={title}
                help="Interpolated population-level 3D landscape where x is body spread, y is verticality, and height/color are success rate. Points are the actual organisms in the current population."
              />
              <div className="row controls">
                {note ? <span className="meta-note">{note}</span> : null}
                <button type="button" className="mini-btn" onClick={() => setOpen(false)}>close</button>
              </div>
            </div>
            <div className="cellengine-surface-stats">
              <span>{landscape.points.length} organisms</span>
              <span>x = shape spread</span>
              <span>y = verticality</span>
              <span>z = success</span>
            </div>
            <div className="cellengine-surface-disclaimer-strip">
              <span>Population-level shape vs success landscape</span>
              <span>Surface is interpolated from the currently visible organisms</span>
              <span>Points are the actual candidates</span>
            </div>
            <div className="cellengine-surface-modal-plot">
              <div className="cellengine-surface-carousel-caption">
                <strong>Shape vs Success Surface</strong>
                <span>Broader bodies move right, taller bodies move up, higher-success candidates rise higher.</span>
              </div>
              <div className="cellengine-surface-main-plot three-d">
                <Suspense fallback={<div className="empty">Loading 3D surface…</div>}>
                  <Plot3D
                    data={[
                      {
                        type: "surface",
                        x: landscape.gridX,
                        y: landscape.gridY,
                        z: landscape.zRows,
                        colorscale: DEFAULT_SURFACE_COLORSCALE,
                        showscale: true,
                        colorbar: { title: { text: "success" } },
                        contours: {
                          z: {
                            show: true,
                            usecolormap: false,
                            highlightcolor: "#ffffff",
                            project: { z: true }
                          }
                        },
                        hovertemplate: "shape spread %{x:.2f}<br>verticality %{y:.2f}<br>success %{z:.3f}<extra></extra>"
                      },
                      {
                        type: "scatter3d",
                        mode: "markers+text",
                        x: landscape.points.map((point) => point.x),
                        y: landscape.points.map((point) => point.y),
                        z: landscape.points.map((point) => point.z),
                        text: landscape.points.map((point) => point.label),
                        textposition: "top center",
                        textfont: { size: 9, color: "#334155" },
                        marker: {
                          size: landscape.points.map((point) => lerp(5, 10, point.z)),
                          color: landscape.points.map((point) => point.z),
                          colorscale: DEFAULT_SURFACE_COLORSCALE,
                          cmin: 0,
                          cmax: 1,
                          line: { color: "rgba(255,255,255,0.8)", width: 0.8 },
                          opacity: 0.94
                        },
                        name: "population",
                        hovertemplate: "%{text}<br>shape spread %{x:.2f}<br>verticality %{y:.2f}<br>success %{z:.3f}<extra></extra>"
                      }
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      margin: { l: 0, r: 0, t: 10, b: 0 },
                      scene: {
                        bgcolor: "rgba(0,0,0,0)",
                        xaxis: { title: landscape.xLabel },
                        yaxis: { title: landscape.yLabel },
                        zaxis: { title: landscape.zLabel, range: [0, 1] },
                        camera: { eye: { x: 1.45, y: -1.5, z: 0.9 } }
                      },
                      legend: { orientation: "h", x: 0, y: 1.04 }
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                  />
                </Suspense>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
