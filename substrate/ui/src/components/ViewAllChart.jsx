import { Suspense, lazy } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const Plot3D = lazy(() => import("./Plot3D"));
const BioTissueGraph3D = lazy(() =>
  import("./Visualization3D").then((module) => ({ default: module.BioTissueGraph3D }))
);

const DEFAULT_SURFACE_COLORSCALE = [
  [0.0, "#16324f"],
  [0.25, "#236192"],
  [0.5, "#2b8a3e"],
  [0.75, "#e67700"],
  [1.0, "#a61e4d"]
];

export default function ViewAllChart({ dataset }) {
  if (!dataset || dataset.kind === "empty") {
    return <div className="empty">No chartable rows for this run.</div>;
  }

  if (dataset.kind === "landscape") {
    const landscape = dataset.landscape;
    if (!landscape) return <div className="empty">No chartable rows for this run.</div>;

    const pathX = landscape.path.map((point) => point.x);
    const pathY = landscape.path.map((point) => point.y);
    const pathZ = landscape.path.map((point) => point.z);

    return (
      <div className="landscape-plot-wrap" style={{ height: 250 }}>
        <Suspense fallback={<div className="empty">Loading 3D renderer...</div>}>
          <Plot3D
            data={[
              {
                type: "surface",
                x: landscape.xAxis,
                y: landscape.yAxis,
                z: landscape.zGrid,
                colorscale: DEFAULT_SURFACE_COLORSCALE,
                opacity: 0.92,
                showscale: false
              },
              {
                type: "scatter3d",
                mode: "lines",
                x: pathX,
                y: pathY,
                z: pathZ,
                line: { width: 4, color: "#ffe066" },
                name: "path"
              }
            ]}
            layout={{
              autosize: true,
              margin: { l: 0, r: 0, t: 6, b: 0 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              scene: {
                xaxis: { title: "progress" },
                yaxis: { title: "direction" },
                zaxis: { title: "loss" },
                aspectratio: { x: 1.15, y: 1, z: 0.65 }
              },
              uirevision: "viewall-landscape"
            }}
            style={{ width: "100%", height: "100%" }}
          />
        </Suspense>
      </div>
    );
  }

  if (dataset.kind === "matrix") {
    if (!dataset.classes?.length) {
      return <div className="empty">No chartable rows for this run.</div>;
    }
    return (
      <div className="matrix-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>t\p</th>
              {dataset.classes.map((cls) => (
                <th key={`view-all-h-${cls}`}>{cls}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataset.matrix.map((row, rowIdx) => (
              <tr key={`view-all-r-${rowIdx}`}>
                <th>{dataset.classes[rowIdx]}</th>
                {row.map((value, colIdx) => {
                  const alpha = dataset.maxValue > 0 ? value / dataset.maxValue : 0;
                  const bg =
                    value === 0
                      ? "rgba(12,133,153,0.04)"
                      : `rgba(12,133,153,${Math.max(0.08, alpha)})`;
                  return (
                    <td key={`view-all-c-${rowIdx}-${colIdx}`} style={{ background: bg }}>
                      {value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (dataset.kind === "bio_tissue_3d") {
    return (
      <Suspense fallback={<div className="empty">Loading 3D renderer...</div>}>
        <BioTissueGraph3D
          graph={dataset.graph}
          title="3D Bio Tissue Connectivity"
          subtitle="Hover nodes to isolate local pathways."
          height={260}
          compact
        />
      </Suspense>
    );
  }

  if (!dataset.data?.length) {
    return <div className="empty">No chartable rows for this run.</div>;
  }

  if (dataset.kind === "bar") {
    return (
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={dataset.data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={dataset.xKey} hide={dataset.hideXAxis || false} />
          <YAxis domain={dataset.yDomain || undefined} />
          <Tooltip />
          <Bar dataKey={dataset.barKey} fill={dataset.color || "#0c8599"} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (dataset.kind === "bar_multi") {
    return (
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={dataset.data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={dataset.xKey} hide={dataset.hideXAxis || false} />
          <YAxis domain={dataset.yDomain || undefined} />
          <Tooltip />
          <Legend />
          {dataset.bars.map((bar) => (
            <Bar key={bar.key} dataKey={bar.key} name={bar.label} fill={bar.color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  const xType = dataset.xType || (dataset.xKey === "center" ? "number" : "category");
  const xDomain = xType === "number"
    ? dataset.xDomain || (dataset.xKey === "center" ? [0, 1] : ["auto", "auto"])
    : undefined;

  if (dataset.kind === "line_dual") {
    return (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={dataset.data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={dataset.xKey} type={xType} domain={xDomain} />
          <YAxis yAxisId="left" domain={dataset.yDomain || undefined} />
          <YAxis yAxisId="right" orientation="right" domain={dataset.yRightDomain || [0, 1]} />
          <Tooltip />
          <Legend />
          {dataset.lines.map((line) => (
            <Line
              key={line.key}
              yAxisId={line.yAxis || "left"}
              dataKey={line.key}
              name={line.label}
              stroke={line.color}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={dataset.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={dataset.xKey} type={xType} domain={xDomain} />
        <YAxis domain={dataset.yDomain || undefined} />
        <Tooltip />
        <Legend />
        {dataset.addDiag ? (
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 }
            ]}
            stroke="#6c757d"
            strokeDasharray="4 4"
          />
        ) : null}
        {dataset.lines.map((line) => (
          <Line key={line.key} dataKey={line.key} name={line.label} stroke={line.color} dot={false} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
