import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine
} from "recharts";
import {
  buildGroupedSeries,
  chooseAutoPreviewColumns,
  fileLabel
} from "../lib/dashboardShared";

const MODEL_GROUP_SERIES_LIMIT = 8;
const MODEL_PREVIEW_SERIES_LIMIT = 6;

export function InterpretabilitySuite({
  classProfileRows,
  classRadarRows,
  confidenceOutcomeRows,
  entropyConfidenceScatterRows,
  calibrationChartRows,
  errorRows
}) {
  const hasAny =
    classProfileRows.length ||
    classRadarRows.length ||
    confidenceOutcomeRows.length ||
    entropyConfidenceScatterRows.length ||
    calibrationChartRows.length ||
    errorRows.length;
  if (!hasAny) return <div className="empty">No interpretability metrics available for this run.</div>;

  return (
    <div className="detail-viz-grid">
      <div className="detail-viz-card">
        <h4>Per-Class Precision / Recall / F1</h4>
        {classProfileRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={classProfileRows}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="class_id" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line dataKey="precision" stroke="#0c8599" dot={false} />
              <Line dataKey="recall" stroke="#2b8a3e" dot={false} />
              <Line dataKey="f1" stroke="#a61e4d" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No class-level metrics.</div>}
      </div>

      <div className="detail-viz-card">
        <h4>Confidence vs Error Bins</h4>
        {confidenceOutcomeRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={confidenceOutcomeRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" hide />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="correct" fill="#2b8a3e" />
              <Bar dataKey="incorrect" fill="#c92a2a" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No confidence/error rows.</div>}
      </div>

      <div className="detail-viz-card">
        <h4>Entropy vs Confidence</h4>
        {entropyConfidenceScatterRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis type="number" dataKey="confidence" domain={[0, 1]} />
              <YAxis type="number" dataKey="entropy" />
              <Tooltip />
              <Legend />
              <Scatter
                data={entropyConfidenceScatterRows.filter((row) => row.correctness === "correct")}
                fill="#2b8a3e"
                name="correct"
              />
              <Scatter
                data={entropyConfidenceScatterRows.filter((row) => row.correctness === "incorrect")}
                fill="#c92a2a"
                name="incorrect"
              />
            </ScatterChart>
          </ResponsiveContainer>
        ) : <div className="empty">No entropy-confidence pairs.</div>}
      </div>

      <div className="detail-viz-card">
        <h4>Calibration</h4>
        {calibrationChartRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={calibrationChartRows}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="center" domain={[0, 1]} type="number" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 }
                ]}
                stroke="#6c757d"
                strokeDasharray="4 4"
              />
              <Line dataKey="empirical_acc" stroke="#2b8a3e" dot={false} />
              <Line dataKey="avg_conf" stroke="#a61e4d" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div className="empty">No calibration bins.</div>}
      </div>

      <div className="detail-viz-card">
        <h4>Top Misclassifications</h4>
        {errorRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={errorRows.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="pair" angle={-35} textAnchor="end" height={64} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#c2255c" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No errors available.</div>}
      </div>

      <div className="detail-viz-card">
        <h4>Class Profile Radar (Top Support Classes)</h4>
        {classRadarRows.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={classRadarRows}>
              <PolarGrid />
              <PolarAngleAxis dataKey="class_label" />
              <PolarRadiusAxis domain={[0, 1]} />
              <Legend />
              <Radar name="precision" dataKey="precision" stroke="#0c8599" fill="#0c859933" />
              <Radar name="recall" dataKey="recall" stroke="#2b8a3e" fill="#2b8a3e33" />
              <Radar name="f1" dataKey="f1" stroke="#a61e4d" fill="#a61e4d33" />
            </RadarChart>
          </ResponsiveContainer>
        ) : <div className="empty">No radar-ready class rows.</div>}
      </div>
    </div>
  );
}

// ─── Expandable Run Detail Panel ─────────────────────────────────


export function AutoCsvPreview({ filePath, rows, error, onOpenDetail }) {
  const columns = useMemo(() => chooseAutoPreviewColumns(rows || []), [rows]);
  const series = useMemo(() => {
    if (!columns) return { data: [], lines: [], hiddenGroups: 0 };
    return buildGroupedSeries(rows || [], columns.x, columns.y, columns.group, MODEL_PREVIEW_SERIES_LIMIT);
  }, [rows, columns]);

  return (
    <div className="card">
      <div className="card-title compact">
        <strong>{fileLabel(filePath)}</strong>
        <button className="mini-btn" onClick={onOpenDetail} type="button">
          Open
        </button>
      </div>

      {error ? (
        <div className="empty">{error}</div>
      ) : !rows?.length ? (
        <div className="empty">No rows</div>
      ) : !columns ? (
        <div className="empty">No numeric columns to preview</div>
      ) : !series.data.length ? (
        <div className="empty">No plottable values</div>
      ) : (
        <>
          <div className="meta-note">
            {columns.x} vs {columns.y}
            {columns.group ? ` grouped by ${columns.group}` : ""}
            {series.hiddenGroups > 0 ? ` (top ${MODEL_PREVIEW_SERIES_LIMIT} groups)` : ""}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={series.data}>
              <CartesianGrid strokeDasharray="4 4" />
              <XAxis dataKey="x" type="number" />
              <YAxis />
              <Tooltip />
              <Legend />
              {series.lines.map((line) => (
                <Line
                  key={line.key}
                  dataKey={line.key}
                  type="monotone"
                  dot={false}
                  stroke={line.color}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

export function CsvModelChart({ rows }) {
  const [x, setX] = useState("");
  const [y, setY] = useState("");
  const [group, setGroup] = useState("");

  const numericColumns = useMemo(() => {
    if (!rows || !rows.length) return [];
    const keys = Object.keys(rows[0]);
    return keys.filter((key) => rows.some((row) => typeof row[key] === "number" && Number.isFinite(row[key])));
  }, [rows]);

  const allColumns = useMemo(() => {
    if (!rows || !rows.length) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  useEffect(() => {
    if (!numericColumns.length) {
      if (x) setX("");
      if (y) setY("");
      return;
    }

    if (!x || !numericColumns.includes(x)) setX(numericColumns[0]);
    if (!y || !numericColumns.includes(y)) {
      setY(numericColumns.length > 1 ? numericColumns[1] : numericColumns[0]);
    }
  }, [numericColumns, x, y]);

  useEffect(() => {
    if (group && !allColumns.includes(group)) setGroup("");
  }, [allColumns, group]);

  const chartModel = useMemo(() => {
    return buildGroupedSeries(rows || [], x, y, group, MODEL_GROUP_SERIES_LIMIT);
  }, [rows, x, y, group]);

  if (!rows || !rows.length) return <div className="empty">No model-specific rows yet.</div>;
  if (!numericColumns.length) return <div className="empty">CSV has no numeric columns to chart.</div>;

  return (
    <div className="stack">
      <div className="row controls">
        <label>
          X
          <select value={x} onChange={(event) => setX(event.target.value)}>
            {numericColumns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label>
          Y
          <select value={y} onChange={(event) => setY(event.target.value)}>
            {numericColumns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label>
          Group
          <select value={group} onChange={(event) => setGroup(event.target.value)}>
            <option value="">none</option>
            {allColumns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        {chartModel.hiddenGroups > 0 && (
          <span className="meta-note">showing top {MODEL_GROUP_SERIES_LIMIT} groups by row count</span>
        )}
      </div>

      <div className="chart">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartModel.data}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="x" type="number" allowDuplicatedCategory={false} />
            <YAxis />
            <Tooltip />
            <Legend />
            {chartModel.lines.map((line) => (
              <Line
                key={line.key}
                dataKey={line.key}
                type="monotone"
                name={line.label}
                dot={false}
                stroke={line.color}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {Object.keys(rows[0]).map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 150).map((row, rowIdx) => (
              <tr key={rowIdx}>
                {Object.keys(rows[0]).map((column) => (
                  <td key={column}>{String(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
