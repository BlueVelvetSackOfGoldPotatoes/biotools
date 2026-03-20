export const OVERVIEW_POLL_MS = 60000;
export const PLAYBACK_FPS = 30;
export const SPEED_OPTIONS = [0.5, 1, 2, 4];
export const DEFAULT_BODY_VIEW = { yaw: -32, pitch: 18, zoom: 1 };
export const COMPACT_3D_BODY_VIEW = { yaw: -58, pitch: 28, zoom: 1.14 };
export const GENOME_PREVIEW_LIMIT = 24;
export const GENOME_PREVIEW_STEP = 24;
export const DISCOVERY_JOB_STORAGE_KEY = "cellengine_discovery_job_id";
export const CELL_METRICS = [
  { id: "energy", label: "Energy" },
  { id: "stress", label: "Stress" },
  { id: "activity", label: "Activity" },
  { id: "Ca", label: "Calcium" },
  { id: "V", label: "Voltage" },
  { id: "a_mech", label: "Mechanical output" },
  { id: "chem_out", label: "Chemical output" },
  { id: "sigma", label: "Conductivity" }
];
export const CELL_MATRIX_METRICS = [
  { id: "energy", label: "Energy", short: "EN" },
  { id: "stress", label: "Stress", short: "ST" },
  { id: "activity", label: "Activity", short: "AC" },
  { id: "Ca", label: "Calcium", short: "CA" },
  { id: "V", label: "Voltage", short: "VO" },
  { id: "sigma", label: "Conductivity", short: "SG" },
  { id: "a_mech", label: "Mechanical output", short: "ME" },
  { id: "chem_out", label: "Chemical output", short: "CH" },
  { id: "Wrec", label: "Recovery memory", short: "RC" }
];
export const GENOME_METRICS = [
  { id: "gene_expr_0", label: "Program 0 excitability" },
  { id: "gene_expr_1", label: "Program 1 contractility" },
  { id: "gene_expr_2", label: "Program 2 mechanosensitivity" },
  { id: "gene_expr_3", label: "Program 3 secretion magnitude" },
  { id: "gene_expr_4", label: "Program 4 secretion sign" },
  { id: "gene_expr_5", label: "Program 5 adhesion class" },
  { id: "gene_expr_6", label: "Program 6 coupling" },
  { id: "gene_expr_7", label: "Program 7 adaptation" },
  { id: "feature_hinge_prox", label: "Feature hinge proximity" },
  { id: "feature_surface_prox", label: "Feature surface exposure" },
  { id: "feature_neighbor_density", label: "Feature neighbor density" }
];
export const DNA_BASE_META = {
  A: { base: "A", label: "adenine", fill: "#d95543", ink: "#7c241c", pale: "#f6d8d3" },
  C: { base: "C", label: "cytosine", fill: "#4b7cf2", ink: "#214caa", pale: "#d9e5ff" },
  G: { base: "G", label: "guanine", fill: "#e1b92f", ink: "#8b6b0f", pale: "#f8efc6" },
  T: { base: "T", label: "thymine", fill: "#58b861", ink: "#237737", pale: "#d9f0dc" }
};
export const GENE_PROGRAM_BASE = {
  gene_expr_0: "A",
  gene_expr_1: "C",
  gene_expr_2: "G",
  gene_expr_3: "T",
  gene_expr_4: "A",
  gene_expr_5: "C",
  gene_expr_6: "G",
  gene_expr_7: "T"
};
export const TASK_META = {
  cartpole_balance: {
    id: "cartpole_balance",
    label: "Cartpole",
    short: "pole balance",
    description: "Classic balance control: keep the pole upright while the organism replaces the cart body.",
    stageFamily: "balance",
    primaryLabel: "balance ratio",
    secondaryLabel: "angle stability"
  },
  mass_spring_balance: {
    id: "mass_spring_balance",
    label: "Mass-Spring",
    short: "spring stabilize",
    description: "Stabilize an equivalent spring-mass target using the same cell controller and RL baselines.",
    stageFamily: "balance",
    primaryLabel: "spring stability",
    secondaryLabel: "displacement"
  },
  worm_drag_race: {
    id: "worm_drag_race",
    label: "Worm Drag",
    short: "crawl forward",
    description: "Sea-cucumber style crawl: drag the body forward across the track by coordinated contraction.",
    stageFamily: "worm",
    primaryLabel: "goal progress",
    secondaryLabel: "forward velocity"
  },
  pong_return: {
    id: "pong_return",
    label: "Pong Return",
    short: "return the ball",
    description: "Move a single paddle to return repeated incoming balls using the same two-action control interface.",
    stageFamily: "pong",
    primaryLabel: "rally completion",
    secondaryLabel: "return count"
  }
};
export const CELLENGINE_SECTIONS = [
  { id: "discover", label: "Discover", note: "search, live batches, and distributions" },
  { id: "replay", label: "Replay", note: "run one genome and inspect the trace" },
  { id: "compare", label: "Compare", note: "static morphology and genome inspection" },
  { id: "reports", label: "Reports / ODD", note: "benchmark outputs and evidence" }
];
export const TELEMETRY_QUALITY_META = {
  measured: {
    label: "measured",
    note: "Task metrics were recorded directly from an evaluated candidate or run.",
    className: "measured"
  },
  restored: {
    label: "restored",
    note: "Task metrics were restored from a saved run summary, not from per-candidate telemetry.",
    className: "restored"
  },
  fitness_only: {
    label: "fitness-only",
    note: "Only search fitness was persisted for this candidate; success and survival were never written.",
    className: "fitness-only"
  },
  none: {
    label: "no telemetry",
    note: "No task telemetry is available for this item.",
    className: "none"
  }
};

export function normalizeCellEngineSection(value) {
  const next = String(value || "").trim().toLowerCase();
  return CELLENGINE_SECTIONS.some((section) => section.id === next) ? next : "discover";
}

export function readCellEngineSectionFromUrl() {
  if (typeof window === "undefined") return "discover";
  const params = new URLSearchParams(window.location.search);
  return normalizeCellEngineSection(params.get("section"));
}

export function writeCellEngineSectionToUrl(sectionId, { replace = false } = {}) {
  if (typeof window === "undefined") return;
  const nextSection = normalizeCellEngineSection(sectionId);
  const url = new URL(window.location.href);
  url.searchParams.set("section", nextSection);
  const nextHref = `${url.pathname}${url.search}${url.hash}`;
  if (replace) {
    window.history.replaceState({}, "", nextHref);
  } else {
    window.history.pushState({}, "", nextHref);
  }
}

export function formatNumber(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

export function formatPct(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
}

export function taskMeta(taskName) {
  return TASK_META[taskName] || TASK_META.cartpole_balance;
}

export function taskLabel(taskName) {
  return taskMeta(taskName).label;
}

export function isBalanceTask(taskName) {
  return taskMeta(taskName).stageFamily === "balance";
}

export function taskReplayFieldMeta(taskName) {
  const family = taskMeta(taskName).stageFamily;
  if (family === "worm") {
    return {
      primaryLabel: "Initial bend",
      primaryHelp: "Initial body bend for the crawl task. Blank lets the backend sample a small bend.",
      primaryPlaceholder: "random small bend",
      secondaryLabel: null,
      secondaryHelp: null,
      secondaryPlaceholder: ""
    };
  }
  if (family === "pong") {
    return {
      primaryLabel: "Serve height",
      primaryHelp: "Initial ball height for the next rally. Blank lets the backend sample one.",
      primaryPlaceholder: "random serve height",
      secondaryLabel: "Ball vertical speed",
      secondaryHelp: "Initial vertical drift of the ball. Blank keeps the cached/default serve.",
      secondaryPlaceholder: "default serve drift"
    };
  }
  return {
    primaryLabel: taskName === "mass_spring_balance" ? "Initial displacement proxy" : "Initial angle (deg)",
    primaryHelp: taskName === "mass_spring_balance"
      ? "Starting spring displacement proxy. Blank means a small random initial displacement."
      : "Starting pole angle. Blank means the backend samples a small random initial tilt.",
    primaryPlaceholder: taskName === "mass_spring_balance" ? "random displacement" : "random small angle",
    secondaryLabel: null,
    secondaryHelp: null,
    secondaryPlaceholder: ""
  };
}

export function taskDiscoveryFieldMeta(taskName) {
  const family = taskMeta(taskName).stageFamily;
  if (family === "worm") {
    return {
      primaryLabel: "Goal distance",
      primaryHelp: "Forward distance required for a successful crawl trial.",
      primaryPlaceholder: "6.0",
      secondaryLabel: "Max backward slip",
      secondaryHelp: "How far backward the body can slide before the crawl trial terminates.",
      secondaryPlaceholder: "1.5",
      tertiaryLabel: null,
      tertiaryHelp: null,
      tertiaryPlaceholder: ""
    };
  }
  if (family === "pong") {
    return {
      primaryLabel: "Target returns",
      primaryHelp: "How many successful returns the controller must complete before the trial counts as solved.",
      primaryPlaceholder: "6",
      secondaryLabel: "Ball speed",
      secondaryHelp: "Horizontal ball speed during discovery and final evaluation.",
      secondaryPlaceholder: "1.1",
      tertiaryLabel: "Paddle half-height",
      tertiaryHelp: "Paddle half-height in arena units. Smaller means less forgiving returns.",
      tertiaryPlaceholder: "0.22"
    };
  }
  return null;
}

export function recommendedTaskBodyConfig(taskName) {
  switch (taskName) {
    case "worm_drag_race":
      return { body_mode: "grown2d", max_cells: 96, body_extent_x: 10, body_extent_y: 10, body_extent_z: 1 };
    case "pong_return":
      return { body_mode: "grown2d", max_cells: 96, body_extent_x: 8, body_extent_y: 12, body_extent_z: 1 };
    case "mass_spring_balance":
      return { body_mode: "grown3d", max_cells: 96, body_extent_x: 6, body_extent_y: 14, body_extent_z: 2 };
    case "cartpole_balance":
    default:
      return { body_mode: "grown3d", max_cells: 96, body_extent_x: 6, body_extent_y: 14, body_extent_z: 2 };
  }
}

export function formatSigned(value, digits = 2, suffix = "") {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}${suffix}`;
}

export function shortHash(value) {
  return typeof value === "string" && value ? value.slice(0, 8) : "n/a";
}

export function clampIndex(value, max) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(max, value));
}

export function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.max(min, Math.min(max, value));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function degToRad(value) {
  return (value * Math.PI) / 180;
}

export function hexToRgb(hex) {
  const normalized = String(hex || "").replace("#", "");
  if (normalized.length !== 6) return { r: 219, g: 228, b: 226 };
  return {
    r: Number.parseInt(normalized.slice(0, 2), 16),
    g: Number.parseInt(normalized.slice(2, 4), 16),
    b: Number.parseInt(normalized.slice(4, 6), 16)
  };
}

export function mixHex(a, b, t) {
  const left = hexToRgb(a);
  const right = hexToRgb(b);
  const clamped = clamp01(t);
  const r = Math.round(lerp(left.r, right.r, clamped));
  const g = Math.round(lerp(left.g, right.g, clamped));
  const bch = Math.round(lerp(left.b, right.b, clamped));
  return `rgb(${r}, ${g}, ${bch})`;
}

export function surfacePaletteColor(value) {
  const t = clamp01(value);
  if (t < 0.25) return mixHex("#233b63", "#335f8a", t / 0.25);
  if (t < 0.5) return mixHex("#335f8a", "#4c8f64", (t - 0.25) / 0.25);
  if (t < 0.75) return mixHex("#4c8f64", "#c97a27", (t - 0.5) / 0.25);
  return mixHex("#c97a27", "#8d2e63", (t - 0.75) / 0.25);
}

export function genomeProgramMeta(metricId) {
  const base = GENE_PROGRAM_BASE[metricId];
  return base ? DNA_BASE_META[base] : null;
}

export function formatGenomeMetricLabel(metricId) {
  const metric = GENOME_METRICS.find((item) => item.id === metricId);
  const meta = genomeProgramMeta(metricId);
  if (!metric) return metricId;
  return meta ? `${meta.base} · ${metric.label}` : metric.label;
}

export function genomeProgramFill(metricId, normalizedValue) {
  const meta = genomeProgramMeta(metricId);
  if (!meta) return null;
  return mixHex(meta.pale, meta.fill, clamp01(normalizedValue));
}

export function genomeProgramGradient(metricId) {
  const meta = genomeProgramMeta(metricId);
  if (!meta) return "linear-gradient(90deg, #7db6df 0%, #3c8dbc 100%)";
  return `linear-gradient(90deg, ${mixHex("#ffffff", meta.fill, 0.36)} 0%, ${meta.fill} 100%)`;
}

export function normalizeMetric(metricId, value) {
  if (!Number.isFinite(value)) return 0.5;
  if (metricId === "energy") return clamp01(value);
  if (metricId === "stress") return clamp01(value / 3);
  if (metricId === "activity") return clamp01(value / 2.5);
  if (metricId === "Ca") return clamp01(value / 2.5);
  if (metricId === "Wrec") return clamp01(value);
  if (metricId === "active") return value ? 1 : 0;
  if (metricId === "sigma") return clamp01((value - 0.1) / 4.9);
  if (metricId === "V") return clamp01((value + 2.5) / 5);
  if (metricId === "a_mech") return clamp01((value + 2.5) / 5);
  if (metricId === "chem_out") return clamp01((value + 1.5) / 3);
  return 0.5;
}

export function cellMetricSwatchColor(metricId, value) {
  const t = normalizeMetric(metricId, value);
  if (metricId === "energy") return `hsl(${lerp(18, 135, t)} 62% ${lerp(86, 38, t)}%)`;
  if (metricId === "stress") return `hsl(${lerp(58, 4, t)} 84% ${lerp(88, 47, t)}%)`;
  if (metricId === "activity") return `hsl(${lerp(210, 270, t)} 68% ${lerp(84, 44, t)}%)`;
  if (metricId === "Ca") return `hsl(${lerp(190, 232, t)} 72% ${lerp(88, 44, t)}%)`;
  if (metricId === "Wrec") return `hsl(${lerp(24, 178, t)} 58% ${lerp(88, 39, t)}%)`;
  if (metricId === "sigma") return `hsl(${lerp(160, 120, t)} 55% ${lerp(88, 36, t)}%)`;
  if (metricId === "V") return value >= 0
    ? `hsl(${lerp(48, 6, clamp01(value / 2.5))} 88% ${lerp(84, 45, clamp01(value / 2.5))}%)`
    : `hsl(${lerp(195, 230, clamp01(Math.abs(value) / 2.5))} 78% ${lerp(86, 45, clamp01(Math.abs(value) / 2.5))}%)`;
  if (metricId === "a_mech") return value >= 0
    ? `hsl(${lerp(22, 342, clamp01(value / 2.5))} 78% ${lerp(84, 46, clamp01(value / 2.5))}%)`
    : `hsl(${lerp(210, 255, clamp01(Math.abs(value) / 2.5))} 76% ${lerp(86, 44, clamp01(Math.abs(value) / 2.5))}%)`;
  if (metricId === "chem_out") return value >= 0
    ? `hsl(${lerp(95, 128, clamp01(value / 1.5))} 60% ${lerp(84, 42, clamp01(value / 1.5))}%)`
    : `hsl(${lerp(274, 316, clamp01(Math.abs(value) / 1.5))} 66% ${lerp(86, 46, clamp01(Math.abs(value) / 1.5))}%)`;
  return "#dbe4e2";
}

export function cellMetricColor(metricId, value, active) {
  if (!active) return "#cbd5e1";
  return cellMetricSwatchColor(metricId, value);
}

export function formatCellMetricValue(metricId, value) {
  if (metricId === "V" || metricId === "a_mech" || metricId === "chem_out") {
    return formatSigned(value, 3);
  }
  return formatNumber(value, 3);
}

export function describeCellMetric(metricId, value) {
  if (!Number.isFinite(value)) return "unknown";
  if (metricId === "energy") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.8) return "charged";
    if (t > 0.45) return "stable";
    return "drained";
  }
  if (metricId === "stress") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.78) return "strained";
    if (t > 0.45) return "loaded";
    return "relaxed";
  }
  if (metricId === "activity") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.72) return "surging";
    if (t > 0.35) return "active";
    return "quiet";
  }
  if (metricId === "Ca") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.72) return "flooded";
    if (t > 0.35) return "primed";
    return "low";
  }
  if (metricId === "Wrec") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.72) return "recovering";
    if (t > 0.35) return "settling";
    return "reset";
  }
  if (metricId === "sigma") {
    const t = normalizeMetric(metricId, value);
    if (t > 0.72) return "conductive";
    if (t > 0.35) return "open";
    return "damped";
  }
  if (metricId === "V") {
    if (value > 0.35) return "depolarized";
    if (value < -0.35) return "hyperpolarized";
    return "balanced";
  }
  if (metricId === "a_mech") {
    if (value > 0.2) return "push right";
    if (value < -0.2) return "push left";
    return "idle";
  }
  if (metricId === "chem_out") {
    if (value > 0.15) return "release";
    if (value < -0.15) return "uptake";
    return "neutral";
  }
  return "mixed";
}

export function cellTypeLabel(cell) {
  if (!cell) return "unknown";
  if (cell.motor && cell.hinge) return "motor hinge";
  if (cell.motor) return "motor effector";
  if (cell.hinge) return "hinge sensor";
  if (cell.ground) return "ground support";
  return "internal scaffold";
}

export function cellTypeStroke(cell) {
  if (!cell) return "#94a3b8";
  if (cell.motor) return "#a61e4d";
  if (cell.hinge) return "#0c8599";
  if (cell.ground) return "#2b8a3e";
  return "#64748b";
}

export const FUNCTIONAL_STATE_META = {
  drive: { label: "mechanical drive", fill: "#f8bbd0", ink: "#8d1c45", symbol: ">" },
  excite: { label: "electrical spike", fill: "#c3e8ff", ink: "#155e75", symbol: "~" },
  signal: { label: "chemical signaling", fill: "#c8f2d0", ink: "#2b8a3e", symbol: "+" },
  couple: { label: "coordination", fill: "#c9f3f5", ink: "#0b7285", symbol: "C" },
  repair: { label: "stress recovery", fill: "#ffe3b3", ink: "#c2410c", symbol: "R" },
  support: { label: "support/sensing", fill: "#dde3ea", ink: "#495057", symbol: "#" },
  reserve: { label: "energy buffer", fill: "#f4e3b2", ink: "#8f5b00", symbol: "E" },
  inactive: { label: "inactive", fill: "#e2e8f0", ink: "#475569", symbol: "x" }
};

export const FUNCTIONAL_STATE_LEGEND = [
  { id: "drive", note: "dominant mechanical actuation" },
  { id: "excite", note: "voltage and calcium dynamics dominate" },
  { id: "signal", note: "chemical output dominates" },
  { id: "couple", note: "gap-coupling and coordination dominate" },
  { id: "repair", note: "stress and recovery dominate" },
  { id: "support", note: "structural sensing/support dominates" },
  { id: "reserve", note: "holding energy with low outward action" },
  { id: "inactive", note: "cell is currently inactive" }
];
export const CELL_ROLE_LEGEND = [
  { id: "motor", label: "motor effector", stroke: "#a61e4d", note: "mechanical actuator cells" },
  { id: "hinge", label: "hinge sensor", stroke: "#0c8599", note: "hinge-adjacent sensing cells" },
  { id: "ground", label: "ground support", stroke: "#2b8a3e", note: "ground-contact support cells" },
  { id: "scaffold", label: "internal scaffold", stroke: "#64748b", note: "structural interior cells" }
];

export function structuralSupportScore(cell) {
  if (!cell) return 0;
  return clamp01(
    (cell.ground ? 0.42 : 0) +
    (cell.hinge ? 0.24 : 0) +
    clamp01(cell.feature_surface_prox || 0) * 0.18 +
    clamp01(cell.feature_hinge_prox || 0) * 0.2
  );
}

export function classifyCellFunction(cell, state) {
  if (!cell || !state) {
    return {
      id: "inactive",
      label: FUNCTIONAL_STATE_META.inactive.label,
      fill: FUNCTIONAL_STATE_META.inactive.fill,
      ink: FUNCTIONAL_STATE_META.inactive.ink,
      symbol: FUNCTIONAL_STATE_META.inactive.symbol,
      confidence: 0,
      runnerUpLabel: "n/a",
      explanation: "no state loaded",
      scores: {}
    };
  }

  const active = state.active === 1 || state.active === true;
  if (!active) {
    return {
      id: "inactive",
      label: FUNCTIONAL_STATE_META.inactive.label,
      fill: FUNCTIONAL_STATE_META.inactive.fill,
      ink: FUNCTIONAL_STATE_META.inactive.ink,
      symbol: FUNCTIONAL_STATE_META.inactive.symbol,
      confidence: 1,
      runnerUpLabel: "n/a",
      explanation: "inactive at this frame",
      scores: { inactive: 1 }
    };
  }

  const activity = normalizeMetric("activity", state.activity);
  const stress = normalizeMetric("stress", state.stress);
  const recovery = normalizeMetric("Wrec", state.Wrec);
  const voltage = clamp01(Math.abs(state.V || 0) / 2.5);
  const calcium = normalizeMetric("Ca", state.Ca);
  const conductivity = normalizeMetric("sigma", state.sigma);
  const mech = clamp01(Math.abs(state.a_mech || 0) / 2.5);
  const chem = clamp01(Math.abs(state.chem_out || 0) / 1.5);
  const energy = normalizeMetric("energy", state.energy);
  const driveGene = clamp01(cell.gene_expr_1 || 0);
  const exciteGene = clamp01(cell.gene_expr_0 || 0);
  const chemGene = clamp01(cell.gene_expr_3 || 0);
  const coupleGene = clamp01(cell.gene_expr_6 || 0);
  const adaptGene = clamp01(cell.gene_expr_7 || 0);
  const support = structuralSupportScore(cell);

  const scores = {
    drive: mech * 0.5 + driveGene * 0.2 + activity * 0.15 + clamp01(cell.mech_advantage || 0) * 0.1 + (cell.motor ? 0.12 : 0),
    excite: voltage * 0.35 + calcium * 0.22 + activity * 0.18 + exciteGene * 0.17 + (cell.hinge ? 0.08 : 0),
    signal: chem * 0.5 + chemGene * 0.2 + calcium * 0.1 + activity * 0.1 + clamp01(cell.feature_surface_prox || 0) * 0.1,
    couple: conductivity * 0.44 + coupleGene * 0.28 + clamp01(cell.feature_neighbor_density || 0) * 0.18 + activity * 0.08,
    repair: stress * 0.42 + recovery * 0.28 + adaptGene * 0.18 + (1 - energy) * 0.08 + (cell.ground ? 0.04 : 0),
    support: support * 0.72 + energy * 0.08 + (1 - activity) * 0.08 + (cell.ground || cell.hinge ? 0.12 : 0),
    reserve: energy * 0.48 + (1 - activity) * 0.22 + (1 - stress) * 0.14 + (!cell.motor && !cell.hinge ? 0.08 : 0) + (cell.ground ? 0.04 : 0)
  };

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [dominantId, dominantScore] = ranked[0];
  const [runnerUpId, runnerUpScore] = ranked[1] || ["inactive", 0];
  const meta = FUNCTIONAL_STATE_META[dominantId] || FUNCTIONAL_STATE_META.inactive;
  let symbol = meta.symbol;
  let label = meta.label;

  if (dominantId === "drive") {
    symbol = state.a_mech >= 0 ? ">" : "<";
    label = state.a_mech >= 0 ? "drive right" : "drive left";
  } else if (dominantId === "signal") {
    symbol = state.chem_out >= 0 ? "+" : "-";
    label = state.chem_out >= 0 ? "chemical release" : "chemical uptake";
  }

  return {
    id: dominantId,
    label,
    fill: meta.fill,
    ink: meta.ink,
    symbol,
    confidence: dominantScore / Math.max(1e-6, dominantScore + runnerUpScore),
    runnerUpLabel: FUNCTIONAL_STATE_META[runnerUpId]?.label || runnerUpId,
    explanation: `${label} dominates over ${FUNCTIONAL_STATE_META[runnerUpId]?.label || runnerUpId}`,
    scores
  };
}

export function edgeStrength(edge) {
  if (!edge) return 0;
  return clamp01(edge.gap_mean || 0);
}

export function staticMetricColor(metricId, value) {
  const t = clamp01(Number.isFinite(value) ? value : 0);
  if (metricId.startsWith("gene_expr_")) return genomeProgramFill(metricId, t) || "#dbe4e2";
  if (metricId === "feature_hinge_prox") return `hsl(${lerp(190, 350, t)} 68% ${lerp(88, 46, t)}%)`;
  if (metricId === "feature_surface_prox") return `hsl(${lerp(85, 30, t)} 70% ${lerp(88, 44, t)}%)`;
  if (metricId === "feature_neighbor_density") return `hsl(${lerp(265, 160, t)} 64% ${lerp(86, 42, t)}%)`;
  return "#dbe4e2";
}

export function edgeStroke(edge, selectedCellId) {
  const asymmetry = Math.abs((edge?.gap_forward || 0) - (edge?.gap_reverse || 0));
  if (!edge?.src_active || !edge?.dst_active) return "#cbd5e1";
  if (selectedCellId != null && (edge.src_cell_id === selectedCellId || edge.dst_cell_id === selectedCellId)) {
    return asymmetry > 0.14 ? "#a61e4d" : "#0c8599";
  }
  return asymmetry > 0.14 ? "#d9480f" : "#7cc4ce";
}

export function edgeOpacity(edge, threshold, selectedCellId) {
  const strength = edgeStrength(edge);
  if (strength < threshold) return 0;
  const focused = selectedCellId != null && (edge.src_cell_id === selectedCellId || edge.dst_cell_id === selectedCellId);
  return focused ? clamp01(0.35 + strength * 0.8) : clamp01(0.12 + strength * 0.55);
}

export function formatActionLabel(action) {
  if (action === 0) return "push left";
  if (action === 1) return "push right";
  return "warmup";
}

async function requestJson(url, { method = "GET", body } = {}) {
  const response = await fetch(url, {
    method,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
      Pragma: "no-cache"
    },
    body: body == null ? undefined : JSON.stringify(body)
  });

  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { raw_text: text };
    }
  }
  if (!response.ok) {
    throw new Error(payload?.error || payload?.raw_text || `${response.status} ${response.statusText}`);
  }
  return payload;
}
