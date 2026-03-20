import { Suspense, lazy, useEffect, useMemo, useState } from "react";

const Plot3D = lazy(() => import("./Plot3D"));

function clampNumber(value, lo, hi) {
  if (typeof value !== "number" || !Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

export function LandscapeSurfaceCard({ title, subtitle, landscape, emptyMessage, onViewAll }) {
  if (!landscape) {
    return (
      <div className="card landscape-card">
        <div className="section-head">
          <h3>{title}</h3>
          {onViewAll ? (
            <button className="mini-btn" type="button" onClick={onViewAll}>
              View all architectures
            </button>
          ) : null}
        </div>
        {subtitle && <div className="meta-note">{subtitle}</div>}
        <div className="empty">
          {emptyMessage || "Need at least 6 numeric points to build a stable 3D landscape."}
        </div>
      </div>
    );
  }

  const pathX = landscape.path.map((point) => point.x);
  const pathY = landscape.path.map((point) => point.y);
  const pathZ = landscape.path.map((point) => point.z);
  const pathStep = landscape.path.map((point) => point.step);

  const minimaX = landscape.minima.map((point) => point.x);
  const minimaY = landscape.minima.map((point) => point.y);
  const minimaZ = landscape.minima.map((point) => point.z);
  const minimaText = landscape.minima.map((point, index) => `min #${index + 1} (step ${point.step})`);

  return (
    <div className="card landscape-card">
      <div className="section-head">
        <h3>{title}</h3>
        {onViewAll ? (
          <button className="mini-btn" type="button" onClick={onViewAll}>
            View all architectures
          </button>
        ) : null}
      </div>
      {subtitle && <div className="meta-note">{subtitle}</div>}
      <div className="landscape-plot-wrap">
        <Suspense fallback={<div className="empty">Loading 3D renderer...</div>}>
          <Plot3D
            data={[
              {
                type: "surface",
                x: landscape.xAxis,
                y: landscape.yAxis,
                z: landscape.zGrid,
                colorscale: [
                  [0.0, "#16324f"],
                  [0.25, "#236192"],
                  [0.5, "#2b8a3e"],
                  [0.75, "#e67700"],
                  [1.0, "#a61e4d"]
                ],
                opacity: 0.92,
                showscale: true,
                contours: { z: { show: true, usecolormap: false, color: "rgba(255,255,255,0.25)", width: 1 } },
                hovertemplate: "x=%{x:.3f}<br>y=%{y:.3f}<br>loss=%{z:.3f}<extra>landscape</extra>"
              },
              {
                type: "scatter3d",
                mode: "lines+markers",
                x: pathX,
                y: pathY,
                z: pathZ,
                text: pathStep.map((step) => `step ${step}`),
                hovertemplate: "%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>loss=%{z:.3f}<extra>trajectory</extra>",
                line: { width: 6, color: "#ffe066" },
                marker: { size: 3, color: "#fff3bf", opacity: 0.85 },
                name: "learning path"
              },
              {
                type: "scatter3d",
                mode: "markers+text",
                x: minimaX,
                y: minimaY,
                z: minimaZ,
                text: minimaText,
                textposition: "top center",
                hovertemplate: "%{text}<br>loss=%{z:.3f}<extra>local optimum</extra>",
                marker: {
                  size: 5,
                  color: "#f03e3e",
                  line: { width: 1, color: "#fff" }
                },
                name: "local minima"
              }
            ]}
            layout={{
              autosize: true,
              margin: { l: 0, r: 0, t: 8, b: 0 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              legend: { orientation: "h", x: 0, y: 1.02 },
              scene: {
                xaxis: { title: "training progression", backgroundcolor: "rgba(245,248,250,0.5)" },
                yaxis: { title: "optimization direction", backgroundcolor: "rgba(245,248,250,0.5)" },
                zaxis: { title: "relative loss", backgroundcolor: "rgba(245,248,250,0.5)" },
                aspectratio: { x: 1.2, y: 1, z: 0.7 },
                camera: { eye: { x: 1.5, y: 1.35, z: 0.85 } }
              },
              uirevision: "landscape-live"
            }}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            config={{
              responsive: true,
              displaylogo: false,
              scrollZoom: true,
              modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"]
            }}
          />
        </Suspense>
      </div>
      <div className="landscape-meta">
        <span className="landscape-pill">{landscape.pointCount} data points</span>
        <span className="landscape-pill">{landscape.minima.length} local minima</span>
        <span className="landscape-pill">live-updating trajectory</span>
      </div>
    </div>
  );
}

function buildEdgeTraceCoordinates(edges) {
  const x = [];
  const y = [];
  const z = [];
  const text = [];
  for (const edge of edges) {
    x.push(edge.src.x, edge.dst.x, null);
    y.push(edge.src.y, edge.dst.y, null);
    z.push(edge.src.z, edge.dst.z, null);
    text.push(
      `${edge.src.layer} -> ${edge.dst.layer}<br>strength=${Number(edge.strength || 0).toFixed(3)}<br>delay=${Number(edge.delay || 0).toFixed(3)}<br>myelin=${Number(edge.myelin || 0).toFixed(3)}`,
      `${edge.src.layer} -> ${edge.dst.layer}<br>strength=${Number(edge.strength || 0).toFixed(3)}<br>delay=${Number(edge.delay || 0).toFixed(3)}<br>myelin=${Number(edge.myelin || 0).toFixed(3)}`,
      ""
    );
  }
  return { x, y, z, text };
}

export function BioTissueGraph3D({
  graph,
  title = "3D Bio Tissue Connectivity",
  subtitle = "Hover a neuron to isolate only its incident connections.",
  height = 340,
  compact = false,
  onViewAll = null
}) {
  const [focusNodeId, setFocusNodeId] = useState(null);

  useEffect(() => {
    setFocusNodeId(null);
  }, [graph?.epoch, graph?.nodeCount, graph?.edgeCount]);

  const focusEdges = useMemo(() => {
    if (!graph?.edges?.length || !focusNodeId) return graph?.edges || [];
    return graph.edges.filter((edge) => edge.srcId === focusNodeId || edge.dstId === focusNodeId);
  }, [graph, focusNodeId]);

  if (!graph || !graph.nodes?.length) {
    return (
      <div className="card">
        <h3>{title}</h3>
        <div className="empty">No layer-level bio data to build a 3D tissue graph.</div>
      </div>
    );
  }

  const edgeCoords = buildEdgeTraceCoordinates(focusEdges);
  const layerColor = (layerIndex) => {
    const hue = (layerIndex * 43) % 360;
    return `hsl(${hue} 68% 46%)`;
  };

  const nodeSizes = graph.nodes.map((node) => 5 + clampNumber(node.gliaPlasticity, 0.1, 2.2) * 2.4);
  const nodeColors = graph.nodes.map((node) => layerColor(node.layerIndex));
  const nodeHover = graph.nodes.map((node) => {
    const degree = graph.connectionCount[node.id] || 0;
    const signText = Number.isFinite(node.signViolationFraction)
      ? Number(node.signViolationFraction).toFixed(4)
      : "n/a";
    return `layer=${node.layer}<br>node=${node.id.split("::")[1]}<br>degree=${degree}<br>bioelectric=${Number(node.bioelectric || 0).toFixed(4)}<br>homeostasis=${Number(node.homeostasis || 0).toFixed(4)}<br>sign_violation=${signText}`;
  });

  const focusedNode = focusNodeId ? graph.nodes.find((node) => node.id === focusNodeId) || null : null;
  const focusDegree = focusedNode ? graph.connectionCount[focusedNode.id] || 0 : null;

  return (
    <div className="card">
      <div className="section-head">
        <h3>{title}</h3>
        <div className="row">
          {onViewAll ? (
            <button className="mini-btn" type="button" onClick={onViewAll}>
              View all architectures
            </button>
          ) : null}
          {focusNodeId ? (
            <button className="mini-btn" type="button" onClick={() => setFocusNodeId(null)}>
              Clear focus
            </button>
          ) : null}
        </div>
      </div>
      {subtitle ? <div className="meta-note">{subtitle}</div> : null}
      <div className="landscape-plot-wrap" style={{ height }}>
        <Suspense fallback={<div className="empty">Loading 3D renderer...</div>}>
          <Plot3D
            data={[
              {
                type: "scatter3d",
                mode: "lines",
                x: edgeCoords.x,
                y: edgeCoords.y,
                z: edgeCoords.z,
                text: edgeCoords.text,
                hovertemplate: "%{text}<extra>connection</extra>",
                line: { width: focusNodeId ? 4 : 2, color: focusNodeId ? "#e67700" : "rgba(12,133,153,0.42)" },
                opacity: focusNodeId ? 0.95 : 0.5,
                name: focusNodeId ? "focused edges" : "all edges"
              },
              {
                type: "scatter3d",
                mode: "markers",
                x: graph.nodes.map((node) => node.x),
                y: graph.nodes.map((node) => node.y),
                z: graph.nodes.map((node) => node.z),
                customdata: graph.nodes.map((node) => node.id),
                text: nodeHover,
                hovertemplate: "%{text}<extra>neuron</extra>",
                marker: {
                  size: nodeSizes,
                  color: nodeColors,
                  opacity: 0.95,
                  line: { width: 0.5, color: "#1f2a30" }
                },
                name: "neurons"
              }
            ]}
            layout={{
              autosize: true,
              margin: { l: 0, r: 0, t: 8, b: 0 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              scene: {
                xaxis: { title: "x" },
                yaxis: { title: "y" },
                zaxis: { title: "layer depth" },
                aspectratio: { x: 1.05, y: 1.05, z: 0.75 },
                camera: { eye: { x: 1.55, y: 1.35, z: 0.92 } }
              },
              showlegend: !compact,
              legend: { orientation: "h", x: 0, y: 1.02 },
              uirevision: `bio-tissue-${graph.epoch}`
            }}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
            config={{
              responsive: true,
              displaylogo: false,
              scrollZoom: true,
              modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"]
            }}
            onHover={(event) => {
              const point = event?.points?.[0];
              if (!point || point?.data?.name !== "neurons") return;
              const id = point.customdata;
              if (typeof id === "string" && id.length) setFocusNodeId(id);
            }}
            onUnhover={() => {
              // Keep focused state until user clears or hovers another node.
            }}
          />
        </Suspense>
      </div>
      <div className="landscape-meta">
        <span className="landscape-pill">epoch {graph.epoch}</span>
        <span className="landscape-pill">{graph.layerCount} layers</span>
        <span className="landscape-pill">{graph.nodeCount} neurons</span>
        <span className="landscape-pill">{graph.edgeCount} connections</span>
        {focusedNode ? (
          <span className="landscape-pill">
            focus {focusedNode.layer}::{focusedNode.id.split("::")[1]} ({focusDegree} edges)
          </span>
        ) : (
          <span className="landscape-pill">hover neuron to isolate connectivity</span>
        )}
      </div>
    </div>
  );
}

// ─── Per-family visualization renderers ───────────────────────────

