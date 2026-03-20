import { seriesColor } from "./dashboardShared";

export function splitSelectValue(splits, current, fallback = "test") {
  if (!splits.length) return "";
  if (current && splits.includes(current)) return current;
  if (splits.includes(fallback)) return fallback;
  return splits[0];
}

export const VIEW_ALL_SPECS = {
  train_landscape: {
    id: "train_landscape",
    tab: "training",
    title: "3D Optimization Landscape (Training)",
    paths: ["learning/epoch_metrics.csv", "learning/batch_metrics.csv"]
  },
  train_live_batch: {
    id: "train_live_batch",
    tab: "training",
    title: "Live Batch Loss / Accuracy",
    paths: ["learning/batch_metrics.csv"]
  },
  train_epoch_loss: {
    id: "train_epoch_loss",
    tab: "training",
    title: "Loss By Epoch",
    paths: ["learning/epoch_metrics.csv"]
  },
  train_epoch_accuracy: {
    id: "train_epoch_accuracy",
    tab: "training",
    title: "Accuracy By Epoch",
    paths: ["learning/epoch_metrics.csv"]
  },
  train_epoch_throughput: {
    id: "train_epoch_throughput",
    tab: "training",
    title: "Throughput",
    paths: ["learning/epoch_metrics.csv"]
  },
  train_bio_energy: {
    id: "train_bio_energy",
    tab: "training",
    title: "Bio Energy And Wiring",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv"]
  },
  train_bio_constraints: {
    id: "train_bio_constraints",
    tab: "training",
    title: "Bio Constraints",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv"]
  },
  train_bio_regulation: {
    id: "train_bio_regulation",
    tab: "training",
    title: "Bio Regulation",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv"]
  },
  train_bio_structural: {
    id: "train_bio_structural",
    tab: "training",
    title: "Bio Structural Plasticity",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv"]
  },
  train_bio_myelin_delay: {
    id: "train_bio_myelin_delay",
    tab: "training",
    title: "Bio Myelination And Delay",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv", "model_specific/{family}/bio_layer_dynamics.csv"]
  },
  train_bio_bioelectric: {
    id: "train_bio_bioelectric",
    tab: "training",
    title: "Bioelectric Dynamics",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv", "model_specific/{family}/bio_layer_dynamics.csv"]
  },
  train_bio_ei_sign: {
    id: "train_bio_ei_sign",
    tab: "training",
    title: "E/I Sign Constraint Integrity",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv", "model_specific/{family}/bio_layer_dynamics.csv"]
  },
  train_bio_tissue_3d: {
    id: "train_bio_tissue_3d",
    tab: "training",
    title: "3D Bio Tissue Connectivity",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv", "model_specific/{family}/bio_layer_dynamics.csv"]
  },
  train_layer_mask_density: {
    id: "train_layer_mask_density",
    tab: "training",
    title: "Layer Mask Density",
    paths: ["model_specific/{family}/bio_layer_dynamics.csv"]
  },
  train_continuous_accuracy: {
    id: "train_continuous_accuracy",
    tab: "training",
    title: "Continuous Accuracy Trajectory",
    paths: ["model_specific/{family}/continuous_efficiency.csv"]
  },
  train_continuous_samples: {
    id: "train_continuous_samples",
    tab: "training",
    title: "Samples And Cycle Cost",
    paths: ["model_specific/{family}/continuous_efficiency.csv"]
  },
  train_phase_loss: {
    id: "train_phase_loss",
    tab: "training",
    title: "Phase Loss (Active vs Sleep)",
    paths: ["model_specific/{family}/continuous_phase_metrics.csv"]
  },
  train_phase_accuracy: {
    id: "train_phase_accuracy",
    tab: "training",
    title: "Phase Accuracy (Active vs Sleep)",
    paths: ["model_specific/{family}/continuous_phase_metrics.csv"]
  },
  eval_landscape: {
    id: "eval_landscape",
    tab: "evaluation",
    title: "3D Generalization Landscape (Evaluation)",
    paths: ["learning/epoch_metrics.csv", "deployment/inference_metrics.csv"]
  },
  eval_confusion_matrix: {
    id: "eval_confusion_matrix",
    tab: "evaluation",
    title: "Confusion Matrix",
    paths: ["learning/confusion_matrix.csv"]
  },
  eval_calibration: {
    id: "eval_calibration",
    tab: "evaluation",
    title: "Calibration",
    paths: ["deployment/calibration_bins.csv"]
  },
  eval_latency: {
    id: "eval_latency",
    tab: "evaluation",
    title: "Latency Distribution",
    paths: ["deployment/inference_metrics.csv"]
  },
  eval_misclassifications: {
    id: "eval_misclassifications",
    tab: "evaluation",
    title: "Top Misclassifications",
    paths: ["deployment/inference_metrics.csv"]
  },
  eval_system: {
    id: "eval_system",
    tab: "evaluation",
    title: "System Metrics",
    paths: ["deployment/system_metrics.csv"]
  },
  eval_sample_efficiency: {
    id: "eval_sample_efficiency",
    tab: "evaluation",
    title: "Sample-Efficiency Frontier",
    paths: ["model_specific/{family}/continuous_efficiency.csv"]
  },
  eval_hybrid_expert: {
    id: "eval_hybrid_expert",
    tab: "evaluation",
    title: "Hybrid Expert Fusion Strength (Latest Cycle)",
    paths: ["model_specific/{family}/hybrid_expert_metrics.csv"]
  },
  eval_bio_structure: {
    id: "eval_bio_structure",
    tab: "evaluation",
    title: "Bio Structure By Layer (Latest Epoch)",
    paths: ["model_specific/{family}/bio_layer_dynamics.csv"]
  },
  eval_bio_state: {
    id: "eval_bio_state",
    tab: "evaluation",
    title: "Bio State By Layer (Latest Epoch)",
    paths: ["model_specific/{family}/bio_layer_dynamics.csv"]
  },
  eval_bio_tissue_3d: {
    id: "eval_bio_tissue_3d",
    tab: "evaluation",
    title: "3D Bio Tissue Connectivity (Latest Epoch)",
    paths: ["model_specific/{family}/bio_epoch_dynamics.csv", "model_specific/{family}/bio_layer_dynamics.csv"]
  },
  eval_bio_ei_sign: {
    id: "eval_bio_ei_sign",
    tab: "evaluation",
    title: "E/I Sign Constraint Integrity (Latest Epoch)",
    paths: ["model_specific/{family}/bio_layer_dynamics.csv"]
  },
  eval_interp_class_profile: {
    id: "eval_interp_class_profile",
    tab: "evaluation",
    title: "Per-Class Precision/Recall/F1",
    paths: ["learning/class_metrics.csv"]
  },
  eval_interp_confidence_error: {
    id: "eval_interp_confidence_error",
    tab: "evaluation",
    title: "Confidence vs Error Diagnostics",
    paths: ["deployment/inference_metrics.csv"]
  }
};

export function resolveSpecPath(pathTemplate, family = "") {
  return String(pathTemplate || "").replace(/\{family\}/g, family || "unknown");
}

export function specPaths(spec, family = "") {
  const base =
    Array.isArray(spec?.paths) && spec.paths.length
      ? spec.paths
      : spec?.path
        ? [spec.path]
        : [];
  return base
    .map((path) => resolveSpecPath(path, family))
    .filter((path) => typeof path === "string" && path.length);
}

export function sourceRows(source, pathTemplate) {
  const family = source?.family || source?.run?.model_family || "";
  const resolvedPath = resolveSpecPath(pathTemplate, family);

  if (source?.byPath && Array.isArray(source.byPath[resolvedPath])) {
    return source.byPath[resolvedPath];
  }

  if (Array.isArray(source?.rows) && (!source.path || source.path === resolvedPath)) {
    return source.rows;
  }

  return [];
}

export function isMissingCsvError(errorText) {
  const msg = String(errorText || "").toLowerCase();
  return (
    msg.includes("404") ||
    msg.includes("enoent") ||
    msg.includes("no such file") ||
    msg.includes("not found")
  );
}

export function parseActiveBitIndices(value) {
  if (Array.isArray(value)) {
    return value
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v) && v >= 0)
      .map((v) => Math.floor(v));
  }
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return [Math.floor(value)];
  }
  const text = String(value || "").trim();
  if (!text) return [];
  return text
    .split(/[;,|]/)
    .map((token) => Number(token))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .map((v) => Math.floor(v));
}

export function decodeOneHotCells(activeIndices, cellCount, classes) {
  const cells = Array.from({ length: cellCount }, () => 0);
  if (classes <= 0 || cellCount <= 0) return cells;
  for (const idx of activeIndices) {
    const cell = Math.floor(idx / classes);
    const cls = idx % classes;
    if (cell >= 0 && cell < cellCount && cls >= 0 && cls < classes) {
      cells[cell] = cls;
    }
  }
  return cells;
}

export function inferSquareBoardSize(obsBits, classes) {
  if (typeof obsBits !== "number" || !Number.isFinite(obsBits) || obsBits <= 0 || classes <= 0) return null;
  const cells = obsBits / classes;
  const side = Math.sqrt(cells);
  if (!Number.isFinite(side)) return null;
  const rounded = Math.round(side);
  if (rounded * rounded !== Math.round(cells)) return null;
  return rounded;
}

export function decodeBenchmarkSnapshot(benchmarkId, obsBits, activeIndices) {
  const id = String(benchmarkId || "").toLowerCase();
  if (id === "tictactoe") {
    const cells = decodeOneHotCells(activeIndices, 9, 3);
    return { kind: "grid", rows: 3, cols: 3, cells, labels: [".", "X", "O"] };
  }
  if (id === "connect_four" || id === "connect4") {
    const cells = decodeOneHotCells(activeIndices, 6 * 7, 3);
    return { kind: "grid", rows: 6, cols: 7, cells, labels: [".", "R", "Y"] };
  }
  if (id === "go") {
    const side = inferSquareBoardSize(obsBits, 3) || 5;
    const cells = decodeOneHotCells(activeIndices, side * side, 3);
    return { kind: "grid", rows: side, cols: side, cells, labels: [".", "B", "W"] };
  }
  if (id === "battleship") {
    const side = inferSquareBoardSize(obsBits, 3) || 8;
    const cells = decodeOneHotCells(activeIndices, side * side, 3);
    return { kind: "grid", rows: side, cols: side, cells, labels: [".", "o", "x"] };
  }
  if (id === "chess") {
    const cells = decodeOneHotCells(activeIndices, 64, 13);
    return {
      kind: "grid",
      rows: 8,
      cols: 8,
      cells,
      labels: [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    };
  }
  if (id === "cartpole") {
    const bins = typeof obsBits === "number" && obsBits >= 8 ? Math.max(8, Math.round(obsBits / 4)) : 32;
    const dims = [
      { key: "x", bin: 0 },
      { key: "x_dot", bin: 0 },
      { key: "theta", bin: 0 },
      { key: "theta_dot", bin: 0 }
    ];
    for (const idx of activeIndices) {
      const dim = Math.floor(idx / bins);
      const bin = idx % bins;
      if (dim >= 0 && dim < dims.length && bin >= 0 && bin < bins) dims[dim].bin = bin;
    }
    return { kind: "cartpole", bins, dims };
  }
  return { kind: "raw", activeIndices };
}

export function chessSquareName(index) {
  if (!Number.isFinite(index) || index < 0 || index >= 64) return "n/a";
  const file = String.fromCharCode(97 + (index % 8));
  const rank = 8 - Math.floor(index / 8);
  return `${file}${rank}`;
}

export function formatBenchmarkAction(benchmarkId, action, obsBits) {
  if (typeof action !== "number" || !Number.isFinite(action) || action < 0) return "n/a";
  const id = String(benchmarkId || "").toLowerCase();
  if (id === "connect_four" || id === "connect4") return `col ${action + 1}`;
  if (id === "cartpole") return action === 1 ? "push right" : "push left";
  if (id === "chess") {
    const from = Math.floor(action / 64);
    const to = action % 64;
    return `${chessSquareName(from)} -> ${chessSquareName(to)}`;
  }
  if (id === "tictactoe" || id === "go" || id === "battleship") {
    const side = inferSquareBoardSize(obsBits, 3);
    if (!side) return String(action);
    const row = Math.floor(action / side);
    const col = action % side;
    return `r${row + 1} c${col + 1}`;
  }
  return String(action);
}

export function outcomeLabel(done, outcome) {
  if (!done) return "ongoing";
  if (typeof outcome === "number" && outcome > 0) return "win";
  if (typeof outcome === "number" && outcome < 0) return "loss";
  return "draw";
}

export function splitPrefixesForMetric(pivotRows, metricName) {
  const suffix = `_${metricName}`;
  const splits = new Set();
  for (const row of pivotRows) {
    for (const key of Object.keys(row)) {
      if (key.endsWith(suffix) && typeof row[key] === "number" && Number.isFinite(row[key])) {
        splits.add(key.slice(0, -suffix.length));
      }
    }
  }
  return [...splits].sort();
}

export function linesForPivotMetric(pivotRows, metricName) {
  return splitPrefixesForMetric(pivotRows, metricName).map((split, index) => ({
    key: `${split}_${metricName}`,
    label: split,
    color: seriesColor(index)
  }));
}

export function pickSplitRows(rows, preferred = "test") {
  const clean = rows || [];
  const hasSplit = clean.some((row) => typeof row.split === "string" && row.split.length);
  if (!hasSplit) return clean;
  const preferredRows = clean.filter((row) => row.split === preferred);
  if (preferredRows.length) return preferredRows;
  return clean;
}

