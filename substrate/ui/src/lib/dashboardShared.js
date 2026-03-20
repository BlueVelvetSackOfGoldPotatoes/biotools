export const TASK_BENCHMARK_OPTIONS = [
  { id: "mnist", label: "MNIST" },
  { id: "tictactoe", label: "TicTacToe" },
  { id: "connect_four", label: "Connect Four" },
  { id: "battleship", label: "Battleship" },
  { id: "go", label: "Go" },
  { id: "chess", label: "Chess" },
  { id: "cartpole", label: "CartPole" },
  { id: "continuous", label: "Continuous" }
];

export const CONTROL_GAME_BENCHMARKS = new Set(["connect_four", "battleship", "go", "chess", "cartpole"]);

export const FAMILY_COLORS = {
  mlp: "#0c8599", cnn: "#a61e4d", rnn: "#5c940d", lstm: "#e67700",
  transformer: "#7048e8", vit: "#d6336c", trees: "#2b8a3e", forests: "#0b7285",
  clustering: "#862e9c", hebbian: "#c92a2a", actor_critic: "#1864ab",
  diffusion: "#e8590c", gnn: "#087f5b", forward_forward: "#5f3dc4",
  reinforcement: "#364fc7", continuous: "#495057", hybrid: "#d9480f",
  markov: "#1d4ed8"
};

export function formatPct(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
}

export function formatSignedPct(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}pp`;
}

export function formatTime(ts) {
  if (!ts) return "n/a";
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return "n/a";
  }
}

export function runStatus(run) {
  if (!run) return "finished";
  if (run.status === "starting" || run.status === "training" || run.status === "finished" || run.status === "stale") {
    return run.status;
  }
  return run.active ? "training" : "finished";
}

export function runStatusLabel(status) {
  if (status === "starting") return "starting";
  if (status === "training") return "training";
  if (status === "stale") return "stale";
  return "finished";
}

export function runStatusClass(status) {
  if (status === "starting") return "pill-starting";
  if (status === "training") return "pill-live";
  if (status === "stale") return "pill-stale";
  return "pill-done";
}

export function taskStatusClass(status) {
  if (status === "running" || status === "queued") return "pill-live";
  if (status === "killing") return "pill-starting";
  if (status === "failed") return "pill-stale";
  return "pill-done";
}

export function taskStatusLabel(status) {
  if (!status) return "unknown";
  return String(status).toLowerCase();
}

export function elapsedSince(utcTs) {
  if (!utcTs) return "n/a";
  const t = Date.parse(utcTs);
  if (!Number.isFinite(t)) return "n/a";
  const secs = Math.max(0, Math.floor((Date.now() - t) / 1000));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  return `${m}m ${s}s`;
}

export function listDistinctStrings(rows, key) {
  const values = new Set();
  for (const row of rows || []) {
    if (typeof row[key] === "string" && row[key].trim().length) values.add(row[key]);
  }
  return [...values].sort();
}

export function listDistinctNumbers(rows, key) {
  const values = new Set();
  for (const row of rows || []) {
    if (typeof row[key] === "number" && Number.isFinite(row[key])) values.add(row[key]);
  }
  return [...values].sort((a, b) => a - b);
}

export function seriesColor(index) {
  return `hsl(${(index * 61) % 360} 72% 42%)`;
}

export function pivotEpoch(rows) {
  const byEpoch = new Map();
  for (const row of rows || []) {
    if (typeof row.epoch !== "number" || !Number.isFinite(row.epoch)) continue;
    if (!byEpoch.has(row.epoch)) byEpoch.set(row.epoch, { epoch: row.epoch });

    const current = byEpoch.get(row.epoch);
    const split = typeof row.split === "string" && row.split ? row.split : "unknown";
    for (const key of ["loss", "accuracy", "samples_per_sec", "grad_norm_mean", "param_norm_mean"]) {
      if (typeof row[key] === "number" && Number.isFinite(row[key])) {
        current[`${split}_${key}`] = row[key];
      }
    }
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

function aggregateSeries(rows, xKey, yKey) {
  const bins = new Map();
  for (const row of rows) {
    const x = row[xKey];
    const y = row[yKey];
    if (typeof x !== "number" || !Number.isFinite(x)) continue;
    if (typeof y !== "number" || !Number.isFinite(y)) continue;
    const key = String(x);
    if (!bins.has(key)) bins.set(key, { x, sum: 0, count: 0 });
    const entry = bins.get(key);
    entry.sum += y;
    entry.count += 1;
  }
  return [...bins.values()].map((entry) => ({ x: entry.x, value: entry.count ? entry.sum / entry.count : null })).sort((a, b) => a.x - b.x);
}

export function buildGroupedSeries(rows, xKey, yKey, groupKey, maxGroups) {
  if (!rows.length || !xKey || !yKey) return { data: [], lines: [], hiddenGroups: 0 };
  if (!groupKey) {
    return {
      data: aggregateSeries(rows, xKey, yKey),
      lines: [{ key: "value", label: yKey, color: seriesColor(0) }],
      hiddenGroups: 0
    };
  }

  const cleanRows = rows.filter((row) => {
    const xv = row[xKey];
    const yv = row[yKey];
    return typeof xv === "number" && Number.isFinite(xv) && typeof yv === "number" && Number.isFinite(yv);
  });

  const groupCounts = new Map();
  for (const row of cleanRows) {
    const label = String(row[groupKey] ?? "null");
    groupCounts.set(label, (groupCounts.get(label) || 0) + 1);
  }

  const topGroups = [...groupCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, maxGroups).map(([label]) => label);
  const allowed = new Set(topGroups);
  const byGroupAndX = new Map();
  for (const row of cleanRows) {
    const label = String(row[groupKey] ?? "null");
    if (!allowed.has(label)) continue;
    const xv = row[xKey];
    const yv = row[yKey];
    const key = `${label}\u241F${xv}`;
    if (!byGroupAndX.has(key)) {
      byGroupAndX.set(key, { label, x: xv, sum: 0, count: 0 });
    }
    const entry = byGroupAndX.get(key);
    entry.sum += yv;
    entry.count += 1;
  }

  const points = new Map();
  for (const entry of byGroupAndX.values()) {
    const pointKey = String(entry.x);
    if (!points.has(pointKey)) points.set(pointKey, { x: entry.x });
    points.get(pointKey)[entry.label] = entry.count ? entry.sum / entry.count : null;
  }

  return {
    data: [...points.values()].sort((a, b) => a.x - b.x),
    lines: topGroups.map((label, idx) => ({ key: label, label, color: seriesColor(idx) })),
    hiddenGroups: Math.max(0, groupCounts.size - topGroups.length)
  };
}

function preferColumn(columns, preferred) {
  for (const key of preferred) {
    if (columns.includes(key)) return key;
  }
  return "";
}

export function chooseAutoPreviewColumns(rows) {
  if (!rows.length) return null;
  const keys = Object.keys(rows[0]);
  const numericColumns = keys.filter((key) => rows.some((row) => typeof row[key] === "number" && Number.isFinite(row[key])));
  if (numericColumns.length < 2) return null;
  const x = preferColumn(numericColumns, ["epoch","step","timestep","t","round","iter","iteration","batch","bin_id","tree_id","layer","head","index"]) || numericColumns[0];
  const yCandidates = numericColumns.filter((key) => key !== x);
  const y = preferColumn(yCandidates, ["loss","accuracy","reward","value","entropy","count","score","q_value","policy","norm","magnitude"]) || yCandidates[0];
  const group = preferColumn(keys, ["split", "class", "cluster", "agent", "policy", "tree", "layer", "head"]) || "";
  return { x, y, group };
}

export function fileLabel(pathValue) {
  const parts = String(pathValue || "").split("/");
  return parts.slice(-1)[0] || pathValue;
}

export function confusionMatrix(rows) {
  const classes = [...new Set((rows || []).flatMap((row) => [row.true_class, row.pred_class]).filter((v) => typeof v === "number"))].sort((a, b) => a - b);
  const indexByClass = new Map(classes.map((value, index) => [value, index]));
  const matrix = Array.from({ length: classes.length }, () => Array.from({ length: classes.length }, () => 0));
  let maxValue = 0;
  for (const row of rows || []) {
    const i = indexByClass.get(row.true_class);
    const j = indexByClass.get(row.pred_class);
    if (i == null || j == null || typeof row.count !== "number") continue;
    matrix[i][j] += row.count;
    maxValue = Math.max(maxValue, matrix[i][j]);
  }
  return { classes, matrix, maxValue };
}

export function topMisclassifications(rows, limit = 12) {
  const counts = new Map();
  for (const row of rows || []) {
    const isCorrect = row.is_correct === 1 || row.is_correct === true;
    if (isCorrect) continue;
    if (typeof row.true_class !== "number" || typeof row.pred_class !== "number") continue;
    const key = `${row.true_class}→${row.pred_class}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].map(([pair, count]) => ({ pair, count })).sort((a, b) => b.count - a.count).slice(0, limit);
}

export function meanValue(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function medianValue(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

export function stdDev(values) {
  if (!values.length) return null;
  if (values.length === 1) return 0;
  const mean = meanValue(values);
  const variance = values.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) / (values.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

export function semValue(values) {
  if (!values.length) return null;
  const std = stdDev(values);
  if (std == null) return null;
  return std / Math.sqrt(values.length);
}

export function formatCompactNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value ?? "");
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${Math.round(value / 1000)}k`;
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (abs >= 100) return Math.round(value).toString();
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

export function buildLogTicks(minValue, maxValue, maxTicks = 8) {
  if (typeof minValue !== "number" || typeof maxValue !== "number" || !Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue <= 0 || maxValue <= 0 || maxValue <= minValue) {
    return [];
  }
  const minPow = Math.floor(Math.log10(minValue));
  const maxPow = Math.ceil(Math.log10(maxValue));
  const fullTicks = [];
  for (let pow = minPow; pow <= maxPow; pow += 1) fullTicks.push(10 ** pow);
  if (fullTicks.length <= maxTicks) return fullTicks;
  const step = Math.ceil(fullTicks.length / maxTicks);
  const reduced = fullTicks.filter((_, index) => index % step === 0);
  const last = fullTicks[fullTicks.length - 1];
  if (reduced[reduced.length - 1] !== last) reduced.push(last);
  return reduced;
}

export function trainingDurationSeconds(run) {
  if (!run || !run.train_start_utc || !run.train_end_utc) return null;
  const startMs = Date.parse(run.train_start_utc);
  const endMs = Date.parse(run.train_end_utc);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) return null;
  return (endMs - startMs) / 1000;
}
