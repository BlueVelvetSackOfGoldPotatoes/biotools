import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { createHash, randomUUID } from "crypto";
import { parse } from "csv-parse/sync";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..", "..");
const RUNS_DIR = path.join(ROOT_DIR, "runs");
const REPORTS_DIR = path.join(ROOT_DIR, "reports");
const TASKS_DIR = path.join(RUNS_DIR, "_tasks");
const TASKS_FILE = path.join(TASKS_DIR, "tasks.json");
const TASK_LOGS_DIR = path.join(TASKS_DIR, "logs");
const CELLENGINE_LATEST_DIR = path.join(REPORTS_DIR, "cellengine_latest");
const CELLENGINE_PREVIEWS_DIR = path.join(REPORTS_DIR, "cellengine_previews");
const CELLENGINE_FAIL_ANGLE_DEG = 15;
const CELLENGINE_API_REPLAY_TIMEOUT_MS = Number(process.env.CELLENGINE_API_REPLAY_TIMEOUT_MS || 180000);
const CELLENGINE_API_DISCOVER_TIMEOUT_MS = Number(process.env.CELLENGINE_API_DISCOVER_TIMEOUT_MS || 3600000);
const PORT = Number(process.env.PORT || 8787);
const RUN_SUMMARY_CACHE_TTL_MS = Number(process.env.RUN_SUMMARY_CACHE_TTL_MS || 4000);
const RUN_ACTIVE_STALE_MS = Number(process.env.RUN_ACTIVE_STALE_MS || 15 * 60 * 1000);
const REPORTS_CACHE_TTL_MS = 4000;
const FAMILY_CACHE_TTL_MS = 4000;
const BENCHMARK_CACHE_TTL_MS = 4000;
const BIO_ABLATION_CACHE_TTL_MS = 4000;
const CSV_CACHE_MAX_FILES = Number(process.env.CSV_CACHE_MAX_FILES || 256);
const TASK_MAX_RECENT = Number(process.env.TASK_MAX_RECENT || 1200);
const TASK_KILL_GRACE_MS = Number(process.env.TASK_KILL_GRACE_MS || 5000);
const BIO_FEATURES = [
  { short: "s", key: "BIO_ENABLE_STRUCTURAL_PLASTICITY", name: "structural_plasticity" },
  { short: "h", key: "BIO_ENABLE_HOMEOSTASIS", name: "homeostasis" },
  { short: "t", key: "BIO_ENABLE_THREE_FACTOR", name: "three_factor" },
  { short: "m", key: "BIO_ENABLE_MYELINATION", name: "myelination" },
  { short: "b", key: "BIO_ENABLE_BIOELECTRIC", name: "bioelectric" },
  { short: "e", key: "BIO_ENABLE_EI_SIGN_CONSTRAINTS", name: "ei_sign_constraints" }
];
const KNOWN_MODEL_FAMILIES = new Set([
  "mlp",
  "cnn",
  "rnn",
  "gru",
  "lstm",
  "transformer",
  "vit",
  "hebbian",
  "actor_critic",
  "markov",
  "reinforcement",
  "gnn",
  "forward_forward",
  "diffusion",
  "trees",
  "forests",
  "clustering",
  "continuous",
  "hybrid"
]);
const MNIST_MODEL_BINARIES = {
  mlp: "bin/benchmark_mlp",
  cnn: "bin/benchmark_cnn",
  rnn: "bin/benchmark_rnn",
  gru: "bin/benchmark_rnn",
  lstm: "bin/benchmark_lstm",
  transformer: "bin/benchmark_transformer",
  vit: "bin/benchmark_vit",
  trees: "bin/benchmark_trees",
  forests: "bin/benchmark_forests",
  clustering: "bin/benchmark_clustering",
  hebbian: "bin/benchmark_hebbian",
  actor_critic: "bin/benchmark_actor_critic",
  diffusion: "bin/benchmark_diffusion",
  gnn: "bin/benchmark_gnn",
  forward_forward: "bin/benchmark_forward_forward",
  reinforcement: "bin/benchmark_reinforcement",
  markov: "bin/benchmark_markov",
  continuous: "bin/benchmark_continuous"
};
const CONTROL_GAME_BENCHMARKS = new Set(["connect_four", "battleship", "go", "chess", "cartpole"]);

const app = express();
app.disable("etag");
app.use(express.json({ limit: "1mb" }));

app.use("/api", (_req, res, next) => {
  res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  res.setHeader("Surrogate-Control", "no-store");
  next();
});

const runSummaryCache = new Map();
const csvFileCache = new Map();
const jsonFileCache = new Map();
let reportImageCache = { scannedAtMs: 0, images: [] };
let familyCoverageCache = { scannedAtMs: 0, families: [] };
let benchmarkCoverageCache = { scannedAtMs: 0, benchmarks: [] };
let bioStudyCache = { scannedAtMs: 0, studies: [] };
const taskStore = new Map();
const taskProcesses = new Map();

function isSafePath(base, candidate) {
  const resolvedBase = path.resolve(base);
  const resolvedCandidate = path.resolve(candidate);
  return (
    resolvedCandidate === resolvedBase ||
    resolvedCandidate.startsWith(resolvedBase + path.sep)
  );
}

function parseJsonSafe(filePath, fallback = {}) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return fallback;
  }
}

function readJsonCached(filePath, fallback = null) {
  const mtimeMs = fileMtimeMs(filePath);
  const size = fileSizeBytes(filePath);
  const cached = jsonFileCache.get(filePath);
  if (cached && cached.mtimeMs === mtimeMs && cached.size === size) {
    return cached.value;
  }
  const value = parseJsonSafe(filePath, fallback);
  jsonFileCache.set(filePath, { mtimeMs, size, value });
  if (jsonFileCache.size > CSV_CACHE_MAX_FILES) {
    const oldestKey = jsonFileCache.keys().next().value;
    if (oldestKey) jsonFileCache.delete(oldestKey);
  }
  return value;
}

function fileMtimeMs(filePath) {
  try {
    return fs.statSync(filePath).mtimeMs;
  } catch {
    return 0;
  }
}

function fileSizeBytes(filePath) {
  try {
    return fs.statSync(filePath).size;
  } catch {
    return 0;
  }
}

function fileContentSha1(filePath) {
  try {
    const h = createHash("sha1");
    h.update(fs.readFileSync(filePath));
    return h.digest("hex").slice(0, 12);
  } catch {
    return null;
  }
}

function parseUtcMs(value) {
  if (typeof value !== "string" || !value.trim()) return 0;
  const ts = Date.parse(value);
  return Number.isFinite(ts) ? ts : 0;
}

function lowerTrim(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function parseManifestParams(manifest) {
  const raw = manifest?.params;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw;
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch {
      return {};
    }
  }
  return {};
}

function normalizeModelSpecToFamily(value) {
  const spec = lowerTrim(value);
  if (!spec) return "";
  if (spec.startsWith("hybrid:")) return "hybrid";
  return spec;
}

function inferFamilyFromVariant(variantRaw) {
  const variant = lowerTrim(variantRaw);
  if (!variant) return "";

  if (variant.startsWith("tictactoe_deep_q_")) {
    const suffix = variant.slice("tictactoe_deep_q_".length);
    if (suffix.startsWith("hybrid_") || suffix.startsWith("hybrid:")) return "hybrid";
    if (KNOWN_MODEL_FAMILIES.has(suffix)) return suffix;
    return "";
  }

  if (variant.startsWith("bit_bridge_")) {
    let suffix = variant.slice("bit_bridge_".length);
    if (suffix.endsWith("_deep_q")) {
      suffix = suffix.slice(0, suffix.length - "_deep_q".length);
    }
    if (suffix.startsWith("hybrid_") || suffix.startsWith("hybrid:")) return "hybrid";
    for (const known of KNOWN_MODEL_FAMILIES) {
      if (suffix === known || suffix.startsWith(`${known}_`)) return known;
    }
  }

  return "";
}

function inferModelFamily(manifest, benchmarkId = "") {
  const rawFamily =
    typeof manifest.model_family === "string" && manifest.model_family.trim()
      ? manifest.model_family.trim()
      : "unknown";
  const rawLower = lowerTrim(rawFamily);
  const params = parseManifestParams(manifest);
  const modelSpecFamily = normalizeModelSpecToFamily(params.model);
  const variantFamily = inferFamilyFromVariant(manifest.model_variant);
  const algo = lowerTrim(params.algorithm);
  const looksBridgeOrBenchmarkFamily =
    rawLower === "tictactoe" ||
    rawLower.startsWith("tictactoe_") ||
    rawLower === "bit_bridge" ||
    rawLower.startsWith("bit_bridge");

  // Keep benchmark and model identity separate: for TicTacToe, always prefer
  // declared model spec/variant over benchmark-like family labels.
  if (benchmarkId === "tictactoe") {
    if (modelSpecFamily) return modelSpecFamily;
    if (variantFamily) return variantFamily;
    if (
      lowerTrim(manifest.model_variant).includes("tabular") ||
      lowerTrim(manifest.model_variant).startsWith("q_learning") ||
      lowerTrim(manifest.model_variant).startsWith("sarsa") ||
      ["q_learning", "sarsa", "dqn", "ddqn", "muzero_lite", "deep_q"].includes(algo)
    ) {
      return "reinforcement";
    }
    if (looksBridgeOrBenchmarkFamily) return "unknown";
  }

  if ((looksBridgeOrBenchmarkFamily || rawLower === "unknown") && modelSpecFamily) {
    return modelSpecFamily;
  }
  if ((looksBridgeOrBenchmarkFamily || rawLower === "unknown") && variantFamily) {
    return variantFamily;
  }

  if (
    lowerTrim(manifest.model_variant).includes("tabular") ||
    lowerTrim(manifest.model_variant).startsWith("q_learning") ||
    lowerTrim(manifest.model_variant).startsWith("sarsa") ||
    ["q_learning", "sarsa", "dqn", "ddqn", "muzero_lite", "deep_q"].includes(algo)
  ) {
    return "reinforcement";
  }

  return rawFamily;
}

function inferBenchmarkId(manifest) {
  const explicit = typeof manifest.benchmark_id === "string" ? manifest.benchmark_id.trim() : "";
  if (explicit) return explicit;
  const dataVersion =
    typeof manifest.data_version === "string" ? manifest.data_version.toLowerCase().trim() : "";
  if (dataVersion.includes("mnist")) return "mnist";
  if (dataVersion.includes("tictactoe")) return "tictactoe";
  if (!dataVersion.length) return "unknown";
  const dash = dataVersion.indexOf("-");
  return dash >= 0 ? dataVersion.slice(0, dash) : dataVersion;
}

function inferBenchmarkName(manifest, benchmarkId) {
  const explicit = typeof manifest.benchmark_name === "string" ? manifest.benchmark_name.trim() : "";
  if (explicit) return explicit;
  if (benchmarkId === "mnist") return "MNIST";
  if (benchmarkId === "tictactoe") return "TicTacToe";
  if (!benchmarkId || benchmarkId === "unknown") return "Unknown";
  return benchmarkId;
}

function inferTaskType(manifest) {
  const explicit = typeof manifest.task_type === "string" ? manifest.task_type.trim() : "";
  if (explicit) return explicit;
  const benchmarkId = inferBenchmarkId(manifest);
  const family = inferModelFamily(manifest, benchmarkId).toLowerCase();
  if (family === "reinforcement" || family === "actor_critic") {
    return "control";
  }
  return "classification";
}

function latestRunEventMs(runDir) {
  const candidates = [
    runDir,
    path.join(runDir, "manifest.json"),
    path.join(runDir, "learning", "epoch_metrics.csv"),
    path.join(runDir, "learning", "batch_metrics.csv"),
    path.join(runDir, "learning", "confusion_matrix.csv"),
    path.join(runDir, "deployment", "calibration_bins.csv"),
    path.join(runDir, "deployment", "inference_metrics.csv"),
    path.join(runDir, "deployment", "system_metrics.csv"),
    path.join(runDir, "runtime", "heartbeat.json")
  ];

  let latestMs = 0;
  for (const filePath of candidates) latestMs = Math.max(latestMs, fileMtimeMs(filePath));
  return latestMs;
}

function isRunLikelyActive(manifest, latestEventMs, heartbeat = null, nowMs = Date.now()) {
  const hasExplicitEnd =
    typeof manifest.train_end_utc === "string" && manifest.train_end_utc.trim().length > 0;
  if (hasExplicitEnd) return false;

  if (heartbeat) {
    const hbStatus = lowerTrim(heartbeat.status);
    if (hbStatus === "finished" || hbStatus === "done") return false;
    const hbMs = parseUtcMs(heartbeat.last_update_utc ?? "");
    if (hbMs && nowMs - hbMs <= RUN_ACTIVE_STALE_MS) {
      if (typeof heartbeat.pid !== "number") return true;
      return isPidAlive(heartbeat.pid);
    }
  }

  const startMs = parseUtcMs(manifest.train_start_utc ?? "");
  if (!startMs) return false;
  if (!latestEventMs) return false;

  return nowMs - latestEventMs <= RUN_ACTIVE_STALE_MS;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function coerceValue(v) {
  if (v == null) return null;
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!t.length) return null;
  if (t.toLowerCase() === "nan") return null;
  if (t.toLowerCase() === "null") return null;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(t)) return Number(t);
  return t;
}

function readCsvRows(filePath) {
  const mtimeMs = fileMtimeMs(filePath);
  const size = fileSizeBytes(filePath);
  const cached = csvFileCache.get(filePath);
  if (cached && cached.mtimeMs === mtimeMs && cached.size === size) {
    return cached.rows;
  }

  const raw = fs.readFileSync(filePath, "utf-8");
  const rows = parse(raw, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  }).map((row) => {
    const out = {};
    for (const [k, v] of Object.entries(row)) out[k] = coerceValue(v);
    return out;
  });

  csvFileCache.set(filePath, { mtimeMs, size, rows });
  if (csvFileCache.size > CSV_CACHE_MAX_FILES) {
    const oldestKey = csvFileCache.keys().next().value;
    if (oldestKey) csvFileCache.delete(oldestKey);
  }
  return rows;
}

function csvHeaderHasColumn(filePath, columnName) {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const header = raw.split(/\r?\n/, 1)[0] || "";
    return header.split(",").map((part) => part.trim()).includes(columnName);
  } catch {
    return false;
  }
}

function readHeartbeat(runDir) {
  const hbPath = path.join(runDir, "runtime", "heartbeat.json");
  if (!fs.existsSync(hbPath)) return null;
  const hb = parseJsonSafe(hbPath, null);
  return hb && typeof hb === "object" ? hb : null;
}

function isPidAlive(pid) {
  if (typeof pid !== "number" || !Number.isFinite(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function utcNowIso() {
  return new Date().toISOString();
}

function ensureTaskLayout() {
  fs.mkdirSync(TASK_LOGS_DIR, { recursive: true });
}

function parseExtraEnv(extraEnvRaw) {
  const out = {};
  if (!extraEnvRaw) return out;
  if (typeof extraEnvRaw === "object" && !Array.isArray(extraEnvRaw)) {
    for (const [k, v] of Object.entries(extraEnvRaw)) {
      const key = String(k || "").trim();
      if (!/^[A-Za-z_][A-Za-z0-9_]{0,79}$/.test(key)) continue;
      out[key] = String(v ?? "");
    }
    return out;
  }
  if (typeof extraEnvRaw !== "string") return out;
  const pairs = extraEnvRaw
    .split(/[,\n;]/)
    .map((token) => token.trim())
    .filter(Boolean);
  for (const pair of pairs) {
    const eq = pair.indexOf("=");
    if (eq <= 0) continue;
    const key = pair.slice(0, eq).trim();
    const value = pair.slice(eq + 1).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,79}$/.test(key)) continue;
    out[key] = value;
  }
  return out;
}

function setNumericEnvIfPresent(env, body, bodyKey, envKey, min = 1) {
  const raw = body?.[bodyKey];
  if (raw == null || raw === "") return;
  const value = Number(raw);
  if (!Number.isFinite(value)) return;
  env[envKey] = String(Math.max(min, Math.floor(value)));
}

function findRunsByTrialUuid(trialUuid) {
  if (!trialUuid || !fs.existsSync(RUNS_DIR)) return [];
  const out = [];
  const dirs = fs
    .readdirSync(RUNS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith("_"))
    .map((d) => d.name);
  for (const runId of dirs) {
    const manifestPath = path.join(RUNS_DIR, runId, "manifest.json");
    if (!fs.existsSync(manifestPath)) continue;
    const manifest = parseJsonSafe(manifestPath, {});
    if (String(manifest?.trial_uuid || "") === String(trialUuid)) out.push(runId);
  }
  out.sort();
  return out;
}

function taskIsTerminal(task) {
  return task && ["completed", "failed", "killed"].includes(task.status);
}

function persistTaskStore() {
  ensureTaskLayout();
  const tasks = [...taskStore.values()].sort((a, b) => {
    const at = parseUtcMs(a.created_utc || "") || 0;
    const bt = parseUtcMs(b.created_utc || "") || 0;
    return bt - at;
  });
  if (tasks.length > TASK_MAX_RECENT) {
    const keep = tasks.slice(0, TASK_MAX_RECENT);
    const keepIds = new Set(keep.map((t) => t.task_id));
    for (const [taskId, task] of taskStore.entries()) {
      if (keepIds.has(taskId)) continue;
      const logAbs = path.join(RUNS_DIR, task.log_path || "");
      if (isSafePath(RUNS_DIR, logAbs) && fs.existsSync(logAbs)) {
        try { fs.unlinkSync(logAbs); } catch { /* ignore */ }
      }
      taskStore.delete(taskId);
    }
  }
  const payload = {
    updated_utc: utcNowIso(),
    tasks: [...taskStore.values()].sort((a, b) => {
      const at = parseUtcMs(a.created_utc || "") || 0;
      const bt = parseUtcMs(b.created_utc || "") || 0;
      return bt - at;
    })
  };
  fs.writeFileSync(TASKS_FILE, JSON.stringify(payload, null, 2), "utf-8");
}

function loadTaskStore() {
  ensureTaskLayout();
  if (!fs.existsSync(TASKS_FILE)) return;
  const payload = parseJsonSafe(TASKS_FILE, {});
  const tasks = Array.isArray(payload?.tasks) ? payload.tasks : [];
  for (const raw of tasks) {
    if (!raw || typeof raw !== "object") continue;
    const taskId = String(raw.task_id || "").trim();
    if (!taskId) continue;
    const task = {
      task_id: taskId,
      kind: String(raw.kind || "train"),
      status: String(raw.status || "failed"),
      benchmark_id: raw.benchmark_id ? String(raw.benchmark_id) : null,
      model_family: raw.model_family ? String(raw.model_family) : null,
      target_run_id: raw.target_run_id ? String(raw.target_run_id) : null,
      created_utc: raw.created_utc || utcNowIso(),
      started_utc: raw.started_utc || null,
      ended_utc: raw.ended_utc || null,
      pid: typeof raw.pid === "number" ? raw.pid : null,
      exit_code: typeof raw.exit_code === "number" ? raw.exit_code : null,
      signal: raw.signal ? String(raw.signal) : null,
      error: raw.error ? String(raw.error) : null,
      command: String(raw.command || ""),
      args: Array.isArray(raw.args) ? raw.args.map((v) => String(v)) : [],
      command_display: String(raw.command_display || ""),
      env_preview:
        raw.env_preview && typeof raw.env_preview === "object" && !Array.isArray(raw.env_preview)
          ? raw.env_preview
          : {},
      trial_uuid: raw.trial_uuid ? String(raw.trial_uuid) : "",
      run_ids: Array.isArray(raw.run_ids) ? raw.run_ids.map((v) => String(v)) : [],
      log_path: raw.log_path ? String(raw.log_path) : `_tasks/logs/${taskId}.log`,
      kill_requested_utc: raw.kill_requested_utc ? String(raw.kill_requested_utc) : null
    };
    taskStore.set(taskId, task);
  }
}

function refreshTaskLiveness() {
  let changed = false;
  for (const task of taskStore.values()) {
    if (taskIsTerminal(task)) continue;
    if (!task.pid || !isPidAlive(task.pid)) {
      if (!task.ended_utc) task.ended_utc = utcNowIso();
      if (!task.status || task.status === "running" || task.status === "queued") {
        task.status = "failed";
      } else if (task.status === "killing") {
        task.status = "killed";
      }
      if (!task.run_ids?.length && task.trial_uuid) {
        task.run_ids = findRunsByTrialUuid(task.trial_uuid);
      }
      changed = true;
    }
  }
  if (changed) persistTaskStore();
}

function taskView(task) {
  return {
    task_id: task.task_id,
    kind: task.kind,
    status: task.status,
    benchmark_id: task.benchmark_id,
    model_family: task.model_family,
    target_run_id: task.target_run_id,
    created_utc: task.created_utc,
    started_utc: task.started_utc,
    ended_utc: task.ended_utc,
    pid: task.pid,
    exit_code: task.exit_code,
    signal: task.signal,
    error: task.error,
    command_display: task.command_display,
    env_preview: task.env_preview,
    trial_uuid: task.trial_uuid || null,
    run_ids: task.run_ids || [],
    log_path: task.log_path
  };
}

function buildTrainLaunchSpec(body = {}) {
  const benchmarkId = lowerTrim(body.benchmark_id || body.benchmark || "mnist") || "mnist";
  const modelFamily = lowerTrim(body.model_family || body.model || "mlp") || "mlp";
  const env = {};
  let binaryRel = "";

  if (benchmarkId === "mnist") {
    if (!(modelFamily in MNIST_MODEL_BINARIES)) {
      throw new Error(`Unsupported MNIST model family '${modelFamily}'.`);
    }
    binaryRel = MNIST_MODEL_BINARIES[modelFamily];
    if (modelFamily === "gru") env.RNN_VARIANT = "gru";
  } else if (benchmarkId === "tictactoe") {
    binaryRel = "bin/benchmark_tictactoe";
    env.TICTACTOE_MODEL = modelFamily;
    setNumericEnvIfPresent(env, body, "episodes", "TICTACTOE_EPISODES", 10);
    setNumericEnvIfPresent(env, body, "eval_every", "TICTACTOE_EVAL_EVERY", 1);
    setNumericEnvIfPresent(env, body, "eval_episodes", "TICTACTOE_EVAL_EPISODES", 1);
    setNumericEnvIfPresent(env, body, "max_steps", "TICTACTOE_MAX_STEPS", 1);
  } else if (CONTROL_GAME_BENCHMARKS.has(benchmarkId)) {
    binaryRel = "bin/benchmark_games";
    env.GAME_BENCHMARK = benchmarkId;
    env.GAME_MODEL = modelFamily;
    const controlMode = lowerTrim(body.control_mode || "full_side");
    env.GAME_CONTROL_MODE = controlMode === "per_piece" ? "per_piece" : "full_side";
    if (typeof body.piece_models === "string" && body.piece_models.trim()) {
      env.GAME_PIECE_MODELS = body.piece_models.trim();
    }
    setNumericEnvIfPresent(env, body, "episodes", "GAME_EPISODES", 10);
    setNumericEnvIfPresent(env, body, "eval_every", "GAME_EVAL_EVERY", 1);
    setNumericEnvIfPresent(env, body, "eval_episodes", "GAME_EVAL_EPISODES", 1);
    setNumericEnvIfPresent(env, body, "max_steps", "GAME_MAX_STEPS", 1);
  } else if (benchmarkId === "continuous") {
    binaryRel = "bin/benchmark_continuous";
    if (modelFamily) env.ONLINE_MODEL = modelFamily;
  } else {
    throw new Error(
      `Unsupported benchmark '${benchmarkId}'. Supported: mnist, tictactoe, connect_four, battleship, go, chess, cartpole, continuous.`
    );
  }

  const command = path.join(ROOT_DIR, binaryRel);
  if (!fs.existsSync(command)) {
    throw new Error(`Binary not found: ${binaryRel}. Build it first.`);
  }
  return {
    kind: "train",
    benchmark_id: benchmarkId,
    model_family: modelFamily,
    command,
    args: [],
    env
  };
}

function buildEvaluateLaunchSpec(body = {}) {
  const runId = String(body.run_id || "").trim();
  if (!runId) {
    throw new Error("run_id is required for evaluate tasks.");
  }
  const runDir = path.join(RUNS_DIR, runId);
  if (!isSafePath(RUNS_DIR, runDir) || !fs.existsSync(runDir)) {
    throw new Error(`Run not found: ${runId}`);
  }
  const script = path.join(ROOT_DIR, "analytics", "plots", "common", "plot_bundle.py");
  if (!fs.existsSync(script)) {
    throw new Error("analytics/plots/common/plot_bundle.py not found.");
  }
  const format = String(body.format || "png").trim().toLowerCase() || "png";
  const outDir = path.join(REPORTS_DIR, runId);
  const pythonBin = String(process.env.PYTHON_BIN || "python3");
  return {
    kind: "evaluate",
    benchmark_id: null,
    model_family: null,
    target_run_id: runId,
    command: pythonBin,
    args: [script, "--run_dir", runDir, "--out_dir", outDir, "--format", format],
    env: {}
  };
}

function launchTask(launchSpec, body = {}) {
  ensureTaskLayout();
  const taskId = `task_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const trialUuid = `ui_${launchSpec.kind}_${randomUUID().slice(0, 12)}`;
  const extraEnv = parseExtraEnv(body.extra_env);
  const env = { ...process.env, ...launchSpec.env, ...extraEnv };
  env.TRIAL_UUID = trialUuid;
  env.JOB_ORIGIN = String(env.JOB_ORIGIN || "ui_task");
  if (body.batch_log_every != null && body.batch_log_every !== "") {
    const value = Number(body.batch_log_every);
    if (Number.isFinite(value)) env.BATCH_LOG_EVERY = String(Math.max(1, Math.floor(value)));
  }

  const relativeCommand = isSafePath(ROOT_DIR, launchSpec.command)
    ? path.relative(ROOT_DIR, launchSpec.command) || launchSpec.command
    : launchSpec.command;
  const commandDisplay = [relativeCommand, ...(launchSpec.args || [])].join(" ").trim();

  const task = {
    task_id: taskId,
    kind: launchSpec.kind,
    status: "queued",
    benchmark_id: launchSpec.benchmark_id || null,
    model_family: launchSpec.model_family || null,
    target_run_id: launchSpec.target_run_id || null,
    created_utc: utcNowIso(),
    started_utc: null,
    ended_utc: null,
    pid: null,
    exit_code: null,
    signal: null,
    error: null,
    command: launchSpec.command,
    args: launchSpec.args || [],
    command_display: commandDisplay,
    env_preview: { ...launchSpec.env, ...extraEnv, TRIAL_UUID: env.TRIAL_UUID, JOB_ORIGIN: env.JOB_ORIGIN },
    trial_uuid: env.TRIAL_UUID,
    run_ids: [],
    log_path: `_tasks/logs/${taskId}.log`,
    kill_requested_utc: null
  };

  taskStore.set(taskId, task);
  persistTaskStore();

  const logAbs = path.join(RUNS_DIR, task.log_path);
  const logStream = fs.createWriteStream(logAbs, { flags: "a" });

  let child = null;
  try {
    child = spawn(task.command, task.args, {
      cwd: ROOT_DIR,
      env,
      stdio: ["ignore", "pipe", "pipe"]
    });
  } catch (err) {
    task.status = "failed";
    task.error = String(err);
    task.ended_utc = utcNowIso();
    persistTaskStore();
    logStream.end();
    return taskView(task);
  }

  task.pid = typeof child.pid === "number" ? child.pid : null;
  task.status = "running";
  task.started_utc = utcNowIso();
  persistTaskStore();
  taskProcesses.set(taskId, child);

  if (child.stdout) child.stdout.on("data", (chunk) => logStream.write(chunk));
  if (child.stderr) child.stderr.on("data", (chunk) => logStream.write(chunk));

  child.on("error", (err) => {
    const current = taskStore.get(taskId);
    if (!current || taskIsTerminal(current)) return;
    current.status = current.status === "killing" ? "killed" : "failed";
    current.error = String(err);
    current.ended_utc = utcNowIso();
    current.run_ids = current.run_ids?.length ? current.run_ids : findRunsByTrialUuid(current.trial_uuid);
    taskProcesses.delete(taskId);
    persistTaskStore();
    logStream.end();
  });

  child.on("close", (code, signal) => {
    const current = taskStore.get(taskId);
    if (!current) {
      logStream.end();
      taskProcesses.delete(taskId);
      return;
    }
    current.exit_code = typeof code === "number" ? code : null;
    current.signal = signal ? String(signal) : null;
    current.ended_utc = utcNowIso();
    current.run_ids = current.run_ids?.length ? current.run_ids : findRunsByTrialUuid(current.trial_uuid);
    if (current.status === "killing" || signal === "SIGTERM" || signal === "SIGKILL") {
      current.status = "killed";
    } else if (code === 0) {
      current.status = "completed";
    } else {
      current.status = "failed";
    }
    taskProcesses.delete(taskId);
    persistTaskStore();
    logStream.end();
  });

  return taskView(task);
}

function killTask(taskId) {
  const task = taskStore.get(taskId);
  if (!task) throw new Error("task not found");
  if (taskIsTerminal(task)) return taskView(task);
  if (!task.pid || !isPidAlive(task.pid)) {
    task.status = "failed";
    task.ended_utc = task.ended_utc || utcNowIso();
    task.run_ids = task.run_ids?.length ? task.run_ids : findRunsByTrialUuid(task.trial_uuid);
    persistTaskStore();
    return taskView(task);
  }
  task.status = "killing";
  task.kill_requested_utc = utcNowIso();
  persistTaskStore();
  try {
    process.kill(task.pid, "SIGTERM");
  } catch (err) {
    task.status = "failed";
    task.error = String(err);
    task.ended_utc = utcNowIso();
    persistTaskStore();
    return taskView(task);
  }
  setTimeout(() => {
    const current = taskStore.get(taskId);
    if (!current || taskIsTerminal(current)) return;
    if (!current.pid || !isPidAlive(current.pid)) return;
    try {
      process.kill(current.pid, "SIGKILL");
    } catch {
      /* ignore */
    }
  }, TASK_KILL_GRACE_MS);
  return taskView(task);
}

function deleteTask(taskId) {
  const task = taskStore.get(taskId);
  if (!task) throw new Error("task not found");
  if (!taskIsTerminal(task)) {
    throw new Error("cannot delete a running task; kill it first");
  }
  const logAbs = path.join(RUNS_DIR, task.log_path || "");
  if (isSafePath(RUNS_DIR, logAbs) && fs.existsSync(logAbs)) {
    try { fs.unlinkSync(logAbs); } catch { /* ignore */ }
  }
  taskStore.delete(taskId);
  persistTaskStore();
}

function sortTasksNewest(tasks) {
  return [...tasks].sort((a, b) => {
    const at = parseUtcMs(a.created_utc || "") || 0;
    const bt = parseUtcMs(b.created_utc || "") || 0;
    return bt - at;
  });
}

function latestByEpoch(rows, split = "test") {
  const subset = rows.filter((r) => r.split === split && typeof r.epoch === "number");
  if (!subset.length) return null;
  subset.sort((a, b) => a.epoch - b.epoch);
  return subset[subset.length - 1];
}

function summarizeRun(runId) {
  const runDir = path.join(RUNS_DIR, runId);
  const manifestPath = path.join(runDir, "manifest.json");
  const epochPath = path.join(runDir, "learning", "epoch_metrics.csv");
  const batchPath = path.join(runDir, "learning", "batch_metrics.csv");
  const heartbeatPath = path.join(runDir, "runtime", "heartbeat.json");
  const latestEventMs = latestRunEventMs(runDir);
  const signature =
    `${fileMtimeMs(manifestPath)}:${fileMtimeMs(epochPath)}:${fileMtimeMs(batchPath)}:${fileMtimeMs(heartbeatPath)}`;
  const nowMs = Date.now();

  const cached = runSummaryCache.get(runId);
  if (
    cached &&
    cached.signature === signature &&
    nowMs - cached.cachedAtMs <= RUN_SUMMARY_CACHE_TTL_MS
  ) {
    return cached.summary;
  }

  const manifest = fs.existsSync(manifestPath) ? parseJsonSafe(manifestPath, {}) : {};
  const heartbeat = readHeartbeat(runDir);
  let latestTest = null;
  let latestTrain = null;
  let latestEpoch = null;
  let hasEpochRows = false;
  let hasBatchRows = false;
  let latestBatchEpoch = null;
  if (fs.existsSync(epochPath)) {
    try {
      const rows = readCsvRows(epochPath);
      const numericEpochs = rows
        .map((row) => row.epoch)
        .filter((epoch) => typeof epoch === "number" && Number.isFinite(epoch));
      hasEpochRows = numericEpochs.length > 0;
      latestEpoch = hasEpochRows ? Math.max(...numericEpochs) : null;
      latestTest = latestByEpoch(rows, "test");
      latestTrain = latestByEpoch(rows, "train");
    } catch {
      latestTest = null;
      latestTrain = null;
      latestEpoch = null;
      hasEpochRows = false;
    }
  }
  if (fs.existsSync(batchPath)) {
    try {
      const rows = readCsvRows(batchPath);
      hasBatchRows = rows.some((row) => typeof row.global_step === "number" && Number.isFinite(row.global_step));
      const batchEpochs = rows
        .map((row) => row.epoch)
        .filter((epoch) => typeof epoch === "number" && Number.isFinite(epoch));
      latestBatchEpoch = batchEpochs.length ? Math.max(...batchEpochs) : null;
    } catch {
      hasBatchRows = false;
      latestBatchEpoch = null;
    }
  }

  const hasExplicitEnd =
    typeof manifest.train_end_utc === "string" && manifest.train_end_utc.trim().length > 0;
  const active = isRunLikelyActive(manifest, latestEventMs, heartbeat, nowMs);
  const heartbeatStatus = heartbeat ? lowerTrim(heartbeat.status) : "";
  let status = "finished";
  if (!hasExplicitEnd && active) {
    if (heartbeatStatus === "starting") status = "starting";
    else status = (hasEpochRows || hasBatchRows) ? "training" : "starting";
  }
  else if (!hasExplicitEnd && !active) status = "stale";
  const benchmarkId = inferBenchmarkId(manifest);
  const benchmarkName = inferBenchmarkName(manifest, benchmarkId);
  const modelFamilyRaw = manifest.model_family ?? "unknown";
  const modelFamily = inferModelFamily(manifest, benchmarkId);

  const summary = {
    benchmark_id: benchmarkId,
    benchmark_name: benchmarkName,
    task_type: inferTaskType(manifest),
    model_family: modelFamily,
    model_family_raw: modelFamilyRaw,
    model_variant: manifest.model_variant ?? "unknown",
    train_start_utc: manifest.train_start_utc ?? null,
    train_end_utc: manifest.train_end_utc ?? null,
    active,
    status,
    has_epoch_rows: hasEpochRows,
    has_batch_rows: hasBatchRows,
    heartbeat_status: heartbeatStatus || null,
    latest_test_accuracy:
      latestTest && typeof latestTest.accuracy === "number" ? latestTest.accuracy : null,
    latest_test_epoch:
      latestTest?.epoch ?? latestTrain?.epoch ?? latestEpoch ?? latestBatchEpoch ?? null
  };

  runSummaryCache.set(runId, { signature, summary, cachedAtMs: nowMs });
  return summary;
}

function updatedUtcForRun(runId) {
  const runDir = path.join(RUNS_DIR, runId);
  const latestMs = latestRunEventMs(runDir);
  if (!latestMs) return null;
  return new Date(latestMs).toISOString();
}

function listRuns() {
  if (!fs.existsSync(RUNS_DIR)) return [];
  const dirs = fs
    .readdirSync(RUNS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .filter((d) => !d.name.startsWith("_"))
    .filter((d) => fs.existsSync(path.join(RUNS_DIR, d.name, "manifest.json")))
    .map((d) => d.name)
    .sort()
    .reverse();

  const liveSet = new Set(dirs);
  for (const cachedRunId of runSummaryCache.keys()) {
    if (!liveSet.has(cachedRunId)) runSummaryCache.delete(cachedRunId);
  }

  return dirs.map((runId) => {
    const summary = summarizeRun(runId);
    return {
      run_id: runId,
      ...summary,
      updated_utc: updatedUtcForRun(runId)
    };
  });
}

function buildFamilyCoverageFromRuns(runs) {
  const byFamily = new Map();
  for (const run of runs) {
    const runId = run.run_id;
    const runDir = path.join(RUNS_DIR, runId);
    const family = run.model_family || "unknown";
    const rawFamily = run.model_family_raw || family;
    const variant = run.model_variant || "unknown";
    const updated = run.updated_utc || null;

    if (!byFamily.has(family)) {
      byFamily.set(family, {
        family,
        runs: 0,
        active_runs: 0,
        variants: new Set(),
        runs_with_learning: 0,
        runs_with_evaluation: 0,
        runs_with_model_specific: 0,
        model_specific_files: new Set(),
        latest_run_id: null,
        latest_updated_utc: null,
        latest_model_specific_run_id: null,
        latest_model_specific_updated_utc: null
      });
    }

    const row = byFamily.get(family);
    row.runs += 1;
    if (run.active) row.active_runs += 1;
    row.variants.add(variant);

    const hasLearning = fs.existsSync(path.join(runDir, "learning", "epoch_metrics.csv"));
    if (hasLearning) row.runs_with_learning += 1;

    const hasEvaluation =
      fs.existsSync(path.join(runDir, "learning", "confusion_matrix.csv")) ||
      fs.existsSync(path.join(runDir, "deployment", "calibration_bins.csv")) ||
      fs.existsSync(path.join(runDir, "deployment", "inference_metrics.csv"));
    if (hasEvaluation) row.runs_with_evaluation += 1;

    const modelSpecificCandidates = [...new Set([family, rawFamily].filter(Boolean))];
    let hasModelSpecific = false;
    for (const msFamily of modelSpecificCandidates) {
      const modelDir = path.join(runDir, "model_specific", msFamily);
      if (!fs.existsSync(modelDir)) continue;
      const files = listCsvFilesRecursive(modelDir, `model_specific/${msFamily}`);
      if (!files.length) continue;
      hasModelSpecific = true;
      for (const file of files) row.model_specific_files.add(file);
    }
    if (hasModelSpecific) {
      row.runs_with_model_specific += 1;
      if (
        updated &&
        (!row.latest_model_specific_updated_utc || updated > row.latest_model_specific_updated_utc)
      ) {
        row.latest_model_specific_updated_utc = updated;
        row.latest_model_specific_run_id = runId;
      }
    }

    if (updated && (!row.latest_updated_utc || updated > row.latest_updated_utc)) {
      row.latest_updated_utc = updated;
      row.latest_run_id = runId;
    }
  }

  const families = [...byFamily.values()]
    .map((row) => ({
      family: row.family,
      runs: row.runs,
      active_runs: row.active_runs,
      variants: [...row.variants].sort(),
      runs_with_learning: row.runs_with_learning,
      runs_with_evaluation: row.runs_with_evaluation,
      runs_with_model_specific: row.runs_with_model_specific,
      model_specific_files: [...row.model_specific_files].sort(),
      latest_run_id: row.latest_run_id,
      latest_updated_utc: row.latest_updated_utc,
      latest_model_specific_run_id: row.latest_model_specific_run_id,
      latest_model_specific_updated_utc: row.latest_model_specific_updated_utc
    }))
    .sort((a, b) => a.family.localeCompare(b.family));

  return families;
}

function listFamilyCoverage(benchmarkFilter = "") {
  if (!fs.existsSync(RUNS_DIR)) return [];

  const now = Date.now();
  if (!benchmarkFilter && now - familyCoverageCache.scannedAtMs < FAMILY_CACHE_TTL_MS) {
    return familyCoverageCache.families;
  }

  const allRuns = listRuns();
  const scopedRuns = benchmarkFilter
    ? allRuns.filter((run) => run.benchmark_id === benchmarkFilter)
    : allRuns;
  const families = buildFamilyCoverageFromRuns(scopedRuns);

  if (!benchmarkFilter) {
    familyCoverageCache = { scannedAtMs: now, families };
  }
  return families;
}

function listBenchmarkCoverage() {
  if (!fs.existsSync(RUNS_DIR)) return [];

  const now = Date.now();
  if (now - benchmarkCoverageCache.scannedAtMs < BENCHMARK_CACHE_TTL_MS) {
    return benchmarkCoverageCache.benchmarks;
  }

  const runs = listRuns();
  const byBenchmark = new Map();
  for (const run of runs) {
    const benchmarkId = run.benchmark_id || "unknown";
    const benchmarkName = run.benchmark_name || benchmarkId;
    if (!byBenchmark.has(benchmarkId)) {
      byBenchmark.set(benchmarkId, {
        benchmark_id: benchmarkId,
        benchmark_name: benchmarkName,
        task_types: new Set(),
        runs: 0,
        active_runs: 0,
        families: new Set(),
        latest_run_id: null,
        latest_updated_utc: null
      });
    }
    const row = byBenchmark.get(benchmarkId);
    row.runs += 1;
    if (run.active) row.active_runs += 1;
    row.families.add(run.model_family || "unknown");
    row.task_types.add(run.task_type || "unknown");
    if (run.updated_utc && (!row.latest_updated_utc || run.updated_utc > row.latest_updated_utc)) {
      row.latest_updated_utc = run.updated_utc;
      row.latest_run_id = run.run_id;
    }
  }

  const benchmarks = [...byBenchmark.values()]
    .map((row) => ({
      benchmark_id: row.benchmark_id,
      benchmark_name: row.benchmark_name,
      task_types: [...row.task_types].sort(),
      runs: row.runs,
      active_runs: row.active_runs,
      families: [...row.families].sort(),
      latest_run_id: row.latest_run_id,
      latest_updated_utc: row.latest_updated_utc
    }))
    .sort((a, b) => a.benchmark_name.localeCompare(b.benchmark_name));

  benchmarkCoverageCache = { scannedAtMs: now, benchmarks };
  return benchmarks;
}

function listCsvFilesRecursive(baseDir, prefix) {
  const csvs = [];
  const stack = [baseDir];

  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(cur, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && entry.name.endsWith(".csv")) {
        const rel = path.relative(baseDir, full).split(path.sep).join("/");
        csvs.push(`${prefix}/${rel}`);
      }
    }
  }

  csvs.sort();
  return csvs;
}

function listModelSpecificFiles(runId) {
  const runDir = path.join(RUNS_DIR, runId);
  const manifest = parseJsonSafe(path.join(runDir, "manifest.json"), {});
  const benchmarkId = inferBenchmarkId(manifest);
  const inferredFamily = inferModelFamily(manifest, benchmarkId);
  const rawFamily = typeof manifest.model_family === "string" ? manifest.model_family.trim() : "";
  const familyCandidates = [...new Set([inferredFamily, rawFamily].filter((v) => typeof v === "string" && v.trim().length))];
  if (!familyCandidates.length) return [];

  const files = new Set();
  for (const family of familyCandidates) {
    const base = path.join(runDir, "model_specific", family);
    if (!fs.existsSync(base)) continue;
    for (const rel of listCsvFilesRecursive(base, `model_specific/${family}`)) files.add(rel);
  }
  return [...files].sort();
}

function reportMatchesRun(relPath, runId) {
  if (!runId) return true;
  const segments = relPath.split("/");
  if (segments.includes(runId)) return true;
  const boundary = new RegExp(`(^|[^A-Za-z0-9_])${escapeRegExp(runId)}([^A-Za-z0-9_]|$)`);
  return boundary.test(relPath);
}

function scanReportImages() {
  if (!fs.existsSync(REPORTS_DIR)) return [];

  const now = Date.now();
  if (now - reportImageCache.scannedAtMs < REPORTS_CACHE_TTL_MS) {
    return reportImageCache.images;
  }

  const images = [];
  const stack = [REPORTS_DIR];
  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(cur, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (/\.(png|jpg|jpeg|svg)$/i.test(e.name)) {
        const rel = path.relative(REPORTS_DIR, full).split(path.sep).join("/");
        images.push(`/reports/${rel}`);
      }
    }
  }

  images.sort();
  reportImageCache = { scannedAtMs: now, images };
  return images;
}

function relFromRoot(absPath) {
  return path.relative(ROOT_DIR, absPath).split(path.sep).join("/");
}

function relFromReports(absPath) {
  return path.relative(REPORTS_DIR, absPath).split(path.sep).join("/");
}

function safeResolveFromRoot(rawPath) {
  const raw = String(rawPath || "").trim();
  if (!raw) throw new Error("path is required");
  const candidate = path.isAbsolute(raw) ? raw : path.join(ROOT_DIR, raw);
  if (!isSafePath(ROOT_DIR, candidate)) {
    throw new Error(`Path escapes repository root: ${raw}`);
  }
  return candidate;
}

function safeResolveReportDir(rawPath) {
  const raw = String(rawPath || "").trim();
  if (!raw) throw new Error("report directory is required");
  const trimmed = raw.replace(/^reports[\\/]/, "");
  const candidate = path.join(REPORTS_DIR, trimmed);
  if (!isSafePath(REPORTS_DIR, candidate)) {
    throw new Error(`Report directory escapes reports/: ${raw}`);
  }
  return candidate;
}

function walkReportFiles(matchName) {
  if (!fs.existsSync(REPORTS_DIR)) return [];
  const matches = [];
  const stack = [REPORTS_DIR];
  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(cur, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.name === matchName) {
        matches.push(full);
      }
    }
  }
  return matches;
}

function numericMean(values) {
  const clean = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((acc, value) => acc + value, 0) / clean.length;
}

function numericMaxAbs(values) {
  const clean = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  return Math.max(...clean.map((value) => Math.abs(value)));
}

function numericMin(values) {
  const clean = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  return Math.min(...clean);
}

function numericMax(values) {
  const clean = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  return Math.max(...clean);
}

function numericStd(values) {
  const clean = (values || []).filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  const mean = numericMean(clean);
  const variance = clean.reduce((acc, value) => acc + (value - mean) * (value - mean), 0) / clean.length;
  return Math.sqrt(variance);
}

function longestStreak(frames, predicate) {
  let best = 0;
  let current = 0;
  for (const frame of frames || []) {
    if (predicate(frame)) {
      current += 1;
      best = Math.max(best, current);
    } else {
      current = 0;
    }
  }
  return best;
}

function findWindowStart(frames, startIndex, predicate, width) {
  if (!Array.isArray(frames) || !frames.length) return null;
  const windowWidth = Math.max(1, Math.floor(width));
  for (let i = Math.max(0, startIndex); i + windowWidth <= frames.length; i += 1) {
    let ok = true;
    for (let j = 0; j < windowWidth; j += 1) {
      if (!predicate(frames[i + j])) {
        ok = false;
        break;
      }
    }
    if (ok) return frames[i]?.tick ?? i;
  }
  return null;
}

function buildCellEngineReplayAnalysis(frames, summary = null) {
  const taskName = lowerTrim(summary?.task_name || "cartpole_balance");
  if (taskName === "worm_drag_race") {
    const progress = (frames || []).map((frame) => frame.x).filter((value) => typeof value === "number");
    const speed = (frames || []).map((frame) => frame.x_dot).filter((value) => typeof value === "number");
    const strain = (frames || []).map((frame) => frame.theta_rad).filter((value) => typeof value === "number");
    const damageTicks = (frames || [])
      .filter((frame) => frame.damage_event === 1 || frame.damage_event === true)
      .map((frame) => frame.tick)
      .filter((tick) => typeof tick === "number");
    const terminalFrame = [...(frames || [])].reverse().find((frame) => frame.terminal === 1 || frame.terminal === true) || null;
    return {
      task_name: taskName,
      frame_count: Array.isArray(frames) ? frames.length : 0,
      terminal_tick: terminalFrame?.tick ?? null,
      success: summary?.cell_clean?.success_rate === 1 || summary?.solved === true,
      total_ticks: summary?.cell_clean?.total_ticks ?? null,
      goal_progress: numericMax(progress),
      mean_forward_speed: numericMean(speed),
      peak_abs_strain: numericMaxAbs(strain),
      damage_ticks: damageTicks,
      mean_abs_force: numericMean((frames || []).map((frame) => Math.abs(frame.organism_force || 0))),
      max_abs_force: numericMaxAbs((frames || []).map((frame) => frame.organism_force)),
      mean_energy: numericMean((frames || []).map((frame) => frame.mean_energy).filter((value) => typeof value === "number")),
      min_energy: numericMin((frames || []).map((frame) => frame.mean_energy).filter((value) => typeof value === "number")),
      mean_stress: numericMean((frames || []).map((frame) => frame.mean_stress).filter((value) => typeof value === "number")),
      max_stress: numericMax((frames || []).map((frame) => frame.mean_stress).filter((value) => typeof value === "number"))
    };
  }
  if (taskName === "pong_return") {
    const paddleY = (frames || []).map((frame) => frame.x).filter((value) => typeof value === "number");
    const ballY = (frames || []).map((frame) => frame.theta_rad).filter((value) => typeof value === "number");
    const ballX = (frames || []).map((frame) => frame.task_aux_a).filter((value) => typeof value === "number");
    const hitCounts = (frames || []).map((frame) => frame.task_counter).filter((value) => typeof value === "number");
    const trackingError = (frames || [])
      .map((frame) => (typeof frame.theta_rad === "number" && typeof frame.x === "number" ? Math.abs(frame.theta_rad - frame.x) : null))
      .filter((value) => typeof value === "number");
    const damageTicks = (frames || [])
      .filter((frame) => frame.damage_event === 1 || frame.damage_event === true)
      .map((frame) => frame.tick)
      .filter((tick) => typeof tick === "number");
    const terminalFrame = [...(frames || [])].reverse().find((frame) => frame.terminal === 1 || frame.terminal === true) || null;
    return {
      task_name: taskName,
      frame_count: Array.isArray(frames) ? frames.length : 0,
      terminal_tick: terminalFrame?.tick ?? null,
      success: summary?.cell_clean?.success_rate === 1 || summary?.solved === true,
      total_ticks: summary?.cell_clean?.total_ticks ?? null,
      return_count: numericMax(hitCounts),
      mean_tracking_error: numericMean(trackingError),
      max_ball_x: numericMax(ballX),
      paddle_span: numericMaxAbs(paddleY),
      ball_span: numericMaxAbs(ballY),
      damage_ticks: damageTicks,
      mean_abs_force: numericMean((frames || []).map((frame) => Math.abs(frame.organism_force || 0))),
      max_abs_force: numericMaxAbs((frames || []).map((frame) => frame.organism_force)),
      mean_energy: numericMean((frames || []).map((frame) => frame.mean_energy).filter((value) => typeof value === "number")),
      min_energy: numericMin((frames || []).map((frame) => frame.mean_energy).filter((value) => typeof value === "number")),
      mean_stress: numericMean((frames || []).map((frame) => frame.mean_stress).filter((value) => typeof value === "number")),
      max_stress: numericMax((frames || []).map((frame) => frame.mean_stress).filter((value) => typeof value === "number"))
    };
  }
  const tickCount = Array.isArray(frames) ? frames.length : 0;
  const thetaDeg = (frames || []).map((frame) => frame.theta_deg).filter((value) => typeof value === "number");
  const energy = (frames || []).map((frame) => frame.mean_energy).filter((value) => typeof value === "number");
  const stress = (frames || []).map((frame) => frame.mean_stress).filter((value) => typeof value === "number");
  const activity = (frames || []).map((frame) => frame.mean_activity).filter((value) => typeof value === "number");
  const activeFraction = (frames || []).map((frame) => frame.active_fraction).filter((value) => typeof value === "number");
  const damageTicks = (frames || [])
    .filter((frame) => frame.damage_event === 1 || frame.damage_event === true)
    .map((frame) => frame.tick)
    .filter((tick) => typeof tick === "number");
  const stablePredicate = (frame) => Math.abs(frame?.theta_deg || 0) <= 2.0;
  const stableFrames = (frames || []).filter(stablePredicate).length;
  const warningFrames = (frames || []).filter((frame) => Math.abs(frame?.theta_deg || 0) >= 10.0).length;
  const terminalFrame = [...(frames || [])].reverse().find((frame) => frame.terminal === 1 || frame.terminal === true) || null;
  const damageStartIndex =
    damageTicks.length > 0
      ? Math.max(
          0,
          (frames || []).findIndex((frame) => frame.tick >= damageTicks[0])
        )
      : 0;
  const rmsThetaDeg =
    thetaDeg.length > 0
      ? Math.sqrt(thetaDeg.reduce((acc, value) => acc + value * value, 0) / thetaDeg.length)
      : null;

  return {
    fail_angle_deg: CELLENGINE_FAIL_ANGLE_DEG,
    frame_count: tickCount,
    terminal_tick: terminalFrame?.tick ?? null,
    success: summary?.cell_clean?.success_rate === 1 || summary?.solved === true,
    total_ticks: summary?.cell_clean?.total_ticks ?? null,
    max_abs_theta_deg: numericMaxAbs(thetaDeg),
    rms_theta_deg: rmsThetaDeg,
    stable_fraction: tickCount > 0 ? stableFrames / tickCount : null,
    warning_fraction: tickCount > 0 ? warningFrames / tickCount : null,
    longest_balanced_streak: longestStreak(frames, stablePredicate),
    settle_tick: findWindowStart(frames, 0, stablePredicate, 25),
    damage_ticks: damageTicks,
    recovery_tick_after_damage: damageTicks.length
      ? findWindowStart(frames, damageStartIndex, stablePredicate, 25)
      : null,
    max_abs_x: numericMaxAbs((frames || []).map((frame) => frame.x)),
    mean_abs_force: numericMean((frames || []).map((frame) => Math.abs(frame.organism_force || 0))),
    max_abs_force: numericMaxAbs((frames || []).map((frame) => frame.organism_force)),
    mean_abs_total_force: numericMean((frames || []).map((frame) => Math.abs(frame.total_force || 0))),
    max_abs_total_force: numericMaxAbs((frames || []).map((frame) => frame.total_force)),
    mean_abs_teacher_force: numericMean((frames || []).map((frame) => Math.abs(frame.teacher_force || 0))),
    mean_energy: numericMean(energy),
    min_energy: numericMin(energy),
    max_energy: numericMax(energy),
    mean_stress: numericMean(stress),
    max_stress: numericMax(stress),
    mean_activity: numericMean(activity),
    min_active_fraction: numericMin(activeFraction)
  };
}

function parseCellEngineSummaryLine(line) {
  const match =
    /^([a-z0-9_]+): survival=([^\s]+) mean_ticks=([^\s]+) success=([^\s]+) force=([^\s]+) energy=([^\s]+)$/i.exec(
      String(line || "").trim()
    );
  if (!match) return null;
  return {
    key: match[1],
    value: {
      survival_ratio: coerceValue(match[2]),
      mean_ticks: coerceValue(match[3]),
      success_rate: coerceValue(match[4]),
      mean_force_abs: coerceValue(match[5]),
      mean_energy: coerceValue(match[6]),
      total_ticks: null,
      max_ticks: null
    }
  };
}

function parseCellEngineReportMarkdown(reportPath) {
  try {
    const raw = fs.readFileSync(reportPath, "utf-8");
    const blockMatch = /```text\s*([\s\S]*?)```/m.exec(raw);
    const text = blockMatch ? blockMatch[1] : raw;
    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const out = {};
    for (const line of lines) {
      const summaryLine = parseCellEngineSummaryLine(line);
      if (summaryLine) {
        out[summaryLine.key] = summaryLine.value;
        continue;
      }
      const algo = /^rl_best_algorithm=(.+)$/.exec(line);
      if (algo) {
        out.rl_algorithm_selected = algo[1].trim();
        continue;
      }
      const header = /^attempt=([^\s]+)\s+champion_generation=([^\s]+)\s+search_score=([^\s]+)\s+solved=(yes|no)$/i.exec(line);
      if (header) {
        out.attempt = coerceValue(header[1]);
        out.champion_generation = coerceValue(header[2]);
        out.champion_search_score = coerceValue(header[3]);
        out.solved = header[4].toLowerCase() === "yes";
      }
    }
    return Object.keys(out).length ? out : {};
  } catch {
    return {};
  }
}

function loadCellEngineSummaryFromDir(dirPath) {
  const summaryPath = path.join(dirPath, "summary.json");
  const reportPath = path.join(dirPath, "report.md");
  const parsed = fs.existsSync(summaryPath) ? parseJsonSafe(summaryPath, null) : null;
  if (parsed && typeof parsed === "object" && parsed.cell_clean) return parsed;
  if (fs.existsSync(reportPath)) {
    const fallback = parseCellEngineReportMarkdown(reportPath);
    if (fallback && typeof fallback === "object" && Object.keys(fallback).length) return fallback;
  }
  return parsed && typeof parsed === "object" ? parsed : {};
}

function writeCellEngineSummaryToDir(dirPath, summary) {
  const summaryPath = path.join(dirPath, "summary.json");
  fs.writeFileSync(summaryPath, JSON.stringify(summary || {}, null, 2));
}

function selectRlSummary(summary, rlAlgo) {
  if (!summary || typeof summary !== "object") return null;
  if (rlAlgo === "a2c") {
    return summary.rl_a2c_clean || (summary.rl_algorithm_selected === "a2c" ? summary.rl_best_clean : null) || null;
  }
  if (rlAlgo === "dqn") {
    return summary.rl_dqn_clean || (summary.rl_algorithm_selected === "dqn" ? summary.rl_best_clean : null) || null;
  }
  return null;
}

function bestCachedRlReplay(rlAlgo, { excludeReplayDir = null, taskName = null } = {}) {
  if (!["a2c", "dqn"].includes(rlAlgo)) return null;
  const candidates = walkReportFiles("rl_replay_trace.csv")
    .map((rlTracePath) => {
      const replayDir = path.dirname(rlTracePath);
      if (excludeReplayDir && path.resolve(replayDir) === path.resolve(excludeReplayDir)) return null;
      const summary = loadCellEngineSummaryFromDir(replayDir);
      if (taskName && lowerTrim(summary?.task_name || "cartpole_balance") !== lowerTrim(taskName)) return null;
      const rlSummary = selectRlSummary(summary, rlAlgo);
      if (!rlSummary) return null;
      const success = Number(rlSummary.success_rate);
      const survival = Number(rlSummary.survival_ratio);
      const scoreSuccess = Number.isFinite(success) ? success : -1;
      const scoreSurvival = Number.isFinite(survival) ? survival : -1;
      const updatedMs = fileMtimeMs(rlTracePath) || fileMtimeMs(path.join(replayDir, "summary.json")) || 0;
      return {
        replayDir,
        rlTracePath,
        rlAnalysisPath: cellEngineAnalysisPath(replayDir, "rl_analysis.json"),
        summary: rlSummary,
        scoreSuccess,
        scoreSurvival,
        updatedMs
      };
    })
    .filter(Boolean)
    .sort((a, b) => {
      if (b.scoreSuccess !== a.scoreSuccess) return b.scoreSuccess - a.scoreSuccess;
      if (b.scoreSurvival !== a.scoreSurvival) return b.scoreSurvival - a.scoreSurvival;
      return b.updatedMs - a.updatedMs;
    });
  return candidates[0] || null;
}

function attachCachedRlComparator(replayDir, rlAlgo, taskName = null) {
  const best = bestCachedRlReplay(rlAlgo, { excludeReplayDir: replayDir, taskName });
  if (!best) return null;

  const dstRlTrace = path.join(replayDir, "rl_replay_trace.csv");
  fs.copyFileSync(best.rlTracePath, dstRlTrace);
  if (fs.existsSync(best.rlAnalysisPath)) {
    fs.copyFileSync(best.rlAnalysisPath, cellEngineAnalysisPath(replayDir, "rl_analysis.json"));
  }

  const summary = loadCellEngineSummaryFromDir(replayDir);
  const patched = {
    ...summary,
    rl_algorithm_selected: rlAlgo,
    rl_algorithm: rlAlgo,
    rl_best_clean: best.summary
  };
  if (rlAlgo === "a2c") patched.rl_a2c_clean = best.summary;
  if (rlAlgo === "dqn") patched.rl_dqn_clean = best.summary;
  const notePrefix = typeof summary?.notes === "string" && summary.notes.trim() ? `${summary.notes} | ` : "";
  patched.notes = `${notePrefix}rl_replay_source=reports/${relFromReports(best.replayDir)}`;
  writeCellEngineSummaryToDir(replayDir, patched);

  return {
    source_replay_dir: `reports/${relFromReports(best.replayDir)}`,
    summary: best.summary
  };
}

function cellEngineAnalysisPath(replayDir, fileName = "analysis.json") {
  return path.join(replayDir, fileName);
}

function readCellEngineAnalysisFromDir(replayDir, fileName = "analysis.json") {
  const analysisPath = cellEngineAnalysisPath(replayDir, fileName);
  if (!fs.existsSync(analysisPath)) return null;
  const parsed = parseJsonSafe(analysisPath, null);
  return parsed && typeof parsed === "object" ? parsed : null;
}

function writeCellEngineAnalysisToDir(replayDir, analysis, fileName = "analysis.json") {
  try {
    fs.writeFileSync(cellEngineAnalysisPath(replayDir, fileName), JSON.stringify(analysis, null, 2));
    return true;
  } catch {
    return false;
  }
}

function loadCellEngineReplayPayload(replayDir, { includeFrames = true } = {}) {
  const summaryPath = path.join(replayDir, "summary.json");
  const tracePath = path.join(replayDir, "replay_trace.csv");
  const bodyPath = path.join(replayDir, "replay_body.csv");
  const cellRowsPath = path.join(replayDir, "replay_cells.csv");
  const edgesPath = path.join(replayDir, "replay_edges.csv");
  const rlTracePath = path.join(replayDir, "rl_replay_trace.csv");
  const reportPath = path.join(replayDir, "report.md");
  if (!fs.existsSync(summaryPath)) {
    throw new Error(`Missing summary.json in ${relFromRoot(replayDir)}`);
  }
  if (!fs.existsSync(tracePath)) {
    throw new Error(`Missing replay_trace.csv in ${relFromRoot(replayDir)}`);
  }

  const summary = loadCellEngineSummaryFromDir(replayDir);
  const frames = includeFrames ? readCsvRows(tracePath) : undefined;
  const body_cells = includeFrames && fs.existsSync(bodyPath) ? readCsvRows(bodyPath) : undefined;
  const cell_rows = includeFrames && fs.existsSync(cellRowsPath) ? readCsvRows(cellRowsPath) : undefined;
  const edge_rows = includeFrames && fs.existsSync(edgesPath) ? readCsvRows(edgesPath) : undefined;
  const rl_frames = includeFrames && fs.existsSync(rlTracePath) ? readCsvRows(rlTracePath) : undefined;
  let analysis = includeFrames ? null : readCellEngineAnalysisFromDir(replayDir, "analysis.json");
  if (!analysis) {
    const analysisFrames = includeFrames ? frames : readCsvRows(tracePath);
    analysis = buildCellEngineReplayAnalysis(analysisFrames, summary);
    writeCellEngineAnalysisToDir(replayDir, analysis, "analysis.json");
  }
  const rlSummary =
    summary?.rl_best_clean ||
    (summary?.rl_algorithm_selected === "dqn" ? summary?.rl_dqn_clean : summary?.rl_a2c_clean) ||
    null;
  let rl_analysis =
    rl_frames || fs.existsSync(rlTracePath)
      ? (includeFrames ? null : readCellEngineAnalysisFromDir(replayDir, "rl_analysis.json"))
      : null;
  if (!rl_analysis && fs.existsSync(rlTracePath)) {
    const analysisFrames = includeFrames ? rl_frames : readCsvRows(rlTracePath);
    rl_analysis = buildCellEngineReplayAnalysis(analysisFrames, {
      cell_clean: rlSummary,
      solved: rlSummary?.success_rate === 1
    });
    writeCellEngineAnalysisToDir(replayDir, rl_analysis, "rl_analysis.json");
  }
  const relativeOutputDir = `reports/${relFromReports(replayDir)}`;

  return {
    replay_id: relFromReports(replayDir).replace(/[\\/]/g, "__"),
    relative_output_dir: relativeOutputDir,
    updated_utc: new Date(fileMtimeMs(tracePath) || fileMtimeMs(summaryPath) || Date.now()).toISOString(),
    files: {
      summary_json: `/reports/${relFromReports(summaryPath)}`,
      replay_trace_csv: `/reports/${relFromReports(tracePath)}`,
      analysis_json: `/reports/${relFromReports(cellEngineAnalysisPath(replayDir, "analysis.json"))}`,
      replay_body_csv: fs.existsSync(bodyPath) ? `/reports/${relFromReports(bodyPath)}` : null,
      replay_cells_csv: fs.existsSync(cellRowsPath) ? `/reports/${relFromReports(cellRowsPath)}` : null,
      replay_edges_csv: fs.existsSync(edgesPath) ? `/reports/${relFromReports(edgesPath)}` : null,
      rl_replay_trace_csv: fs.existsSync(rlTracePath) ? `/reports/${relFromReports(rlTracePath)}` : null,
      rl_analysis_json: fs.existsSync(cellEngineAnalysisPath(replayDir, "rl_analysis.json"))
        ? `/reports/${relFromReports(cellEngineAnalysisPath(replayDir, "rl_analysis.json"))}`
        : null,
      report_md: fs.existsSync(reportPath) ? `/reports/${relFromReports(reportPath)}` : null
    },
    summary,
    analysis,
    rl_summary: rlSummary,
    rl_analysis,
    frames: includeFrames ? frames : undefined,
    body_cells,
    cell_rows,
    edge_rows,
    rl_frames
  };
}

function cellEnginePreviewCacheKey(genomePath) {
  const h = createHash("sha1");
  h.update(relFromRoot(genomePath));
  h.update(`:${fileMtimeMs(genomePath)}`);
  h.update(`:${fileSizeBytes(genomePath)}`);
  return h.digest("hex").slice(0, 16);
}

function cellEnginePreviewCacheDir(genomePath) {
  return path.join(CELLENGINE_PREVIEWS_DIR, cellEnginePreviewCacheKey(genomePath));
}

function cellEnginePreviewBodyPath(previewDir) {
  return path.join(previewDir, "preview_body.csv");
}

function readCellEnginePreviewBody(previewDir) {
  const bodyPath = cellEnginePreviewBodyPath(previewDir);
  if (!fs.existsSync(bodyPath)) return [];
  return readCsvRows(bodyPath);
}

async function ensureCellEngineGenomePreview(genomePath) {
  const binary = path.join(ROOT_DIR, "bin", "benchmark_cellengine");
  if (!fs.existsSync(binary)) {
    throw new Error("Binary not found: bin/benchmark_cellengine. Build it with `make benchmark_cellengine`.");
  }
  const previewDir = cellEnginePreviewCacheDir(genomePath);
  const previewBodyPath = cellEnginePreviewBodyPath(previewDir);
  if (fs.existsSync(previewBodyPath) && csvHeaderHasColumn(previewBodyPath, "gene_expr_0")) {
    return previewDir;
  }

  fs.mkdirSync(previewDir, { recursive: true });
  const env = {
    ...process.env,
    CELLENGINE_MODE: "preview",
    CELLENGINE_RL_ALGO: "none",
    CELLENGINE_LOAD_GENOME: genomePath,
    CELLENGINE_OUT: previewDir,
    CELLENGINE_REPLAY_STDOUT: "0"
  };

  const exitInfo = await new Promise((resolve, reject) => {
    let child;
    let timedOut = false;
    let timeout = null;
    try {
      child = spawn(binary, [], { cwd: ROOT_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
    } catch (err) {
      reject(err);
      return;
    }
    timeout = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
    }, Math.max(1000, Math.floor(CELLENGINE_API_REPLAY_TIMEOUT_MS / 2)));
    child.on("error", (err) => {
      if (timeout) clearTimeout(timeout);
      reject(err);
    });
    child.on("close", (code, signal) => {
      if (timeout) clearTimeout(timeout);
      resolve({ code, signal, timed_out: timedOut });
    });
  });

  if (!fs.existsSync(previewBodyPath)) {
    throw new Error(
      `CellEngine preview failed for ${relFromRoot(genomePath)} (exit=${exitInfo.code ?? "null"}${exitInfo.signal ? ` signal=${exitInfo.signal}` : ""}${exitInfo.timed_out ? " timeout=true" : ""}).`
    );
  }
  return previewDir;
}

async function loadCellEngineGenomePreview(item = {}) {
  const genomePath = safeResolveFromRoot(item.genome_path || "");
  if (!fs.existsSync(genomePath)) {
    throw new Error(`Genome file not found: ${item.genome_path || genomePath}`);
  }
  const previewDir = await ensureCellEngineGenomePreview(genomePath);
  const summaryDir = item.summary_dir ? safeResolveFromRoot(item.summary_dir) : path.dirname(genomePath);
  const files = isSafePath(REPORTS_DIR, summaryDir) ? buildCellEngineRunFiles(summaryDir) : null;
  const summary = loadCellEngineSummaryFromDir(summaryDir);
  return {
    genome_path: relFromRoot(genomePath),
    label: item.label || relFromRoot(genomePath),
    content_hash: fileContentSha1(genomePath),
    summary_dir: isSafePath(ROOT_DIR, summaryDir) ? relFromRoot(summaryDir) : null,
    preview_dir: `reports/${relFromReports(previewDir)}`,
    preview_body_csv: `/reports/${relFromReports(cellEnginePreviewBodyPath(previewDir))}`,
    updated_utc: new Date(fileMtimeMs(cellEnginePreviewBodyPath(previewDir)) || Date.now()).toISOString(),
    body_cells: readCellEnginePreviewBody(previewDir),
    summary,
    files
  };
}

function loadCellEngineLiveDiscoverySnapshot(runDir) {
  const bodyPath = path.join(runDir, "live_body.csv");
  const metaPath = path.join(runDir, "live_meta.json");
  if (!fs.existsSync(bodyPath)) return null;
  const bodyCells = readCsvRows(bodyPath);
  const meta = fs.existsSync(metaPath) ? readJsonCached(metaPath, {}) : {};
  return {
    body_cells: Array.isArray(bodyCells) ? bodyCells : [],
    meta: meta && typeof meta === "object" ? meta : {}
  };
}

function loadCellEngineLiveDiscoveryPopulation(runDir) {
  const populationPath = path.join(runDir, "live_population.json");
  if (!fs.existsSync(populationPath)) return null;
  const parsed = readJsonCached(populationPath, null);
  if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.candidates)) return null;
  return {
    ...parsed,
    candidates: parsed.candidates.map((candidate) => ({
      ...candidate,
      summary_quality:
        candidate?.summary && typeof candidate.summary === "object"
          ? "measured"
          : (Number.isFinite(Number(candidate?.fitness)) ? "fitness_only" : "none")
    }))
  };
}

function listCellEngineGenomeOptions(limit = 120) {
  const matches = walkReportFiles("champion_genome.csv");
  return matches
    .map((genomePath) => {
      const dir = path.dirname(genomePath);
      const relDir = relFromReports(dir);
      if (relDir.startsWith("cellengine_previews/") || relDir.startsWith("cellengine_ui/")) {
        return null;
      }
      const summaryPath = path.join(dir, "summary.json");
      const summary = loadCellEngineSummaryFromDir(dir);
      const updatedMs = Math.max(fileMtimeMs(genomePath), fileMtimeMs(summaryPath));
      return {
        label: relDir,
        genome_path: relFromRoot(genomePath),
        content_hash: fileContentSha1(genomePath),
        summary_dir: `reports/${relDir}`,
        task_name: summary?.task_name || "cartpole_balance",
        updated_utc: updatedMs ? new Date(updatedMs).toISOString() : null,
        solved: Boolean(summary?.solved),
        cell_clean_survival: summary?.cell_clean?.survival_ratio ?? null,
        cell_clean_success: summary?.cell_clean?.success_rate ?? null,
        task_primary: summary?.cell_clean?.task_primary ?? null,
        task_primary_label: summary?.cell_clean?.task_primary_label ?? null,
        rl_algorithm_selected: summary?.rl_algorithm_selected ?? null
      };
    })
    .filter(Boolean)
    .sort((a, b) => Date.parse(b.updated_utc || 0) - Date.parse(a.updated_utc || 0))
    .slice(0, limit);
}

function listCellEngineReplaySummaries(limit = 10) {
  const matches = walkReportFiles("replay_trace.csv")
    .map((tracePath) => ({ tracePath, mtimeMs: fileMtimeMs(tracePath) }))
    .sort((a, b) => b.mtimeMs - a.mtimeMs)
    .slice(0, Math.max(1, limit * 2));
  return matches
    .map(({ tracePath }) => {
      const replayDir = path.dirname(tracePath);
      try {
        const payload = loadCellEngineReplayPayload(replayDir, { includeFrames: false });
        const bodyPath = path.join(replayDir, "replay_body.csv");
        return {
          replay_id: payload.replay_id,
          relative_output_dir: payload.relative_output_dir,
          updated_utc: payload.updated_utc,
          task_name: payload.summary?.task_name || "cartpole_balance",
          success: payload.analysis.success,
          has_body_cells: Boolean(payload.files?.replay_body_csv),
          has_cell_rows: Boolean(payload.files?.replay_cells_csv),
          has_edges: Boolean(payload.files?.replay_edges_csv),
          has_rl_trace: Boolean(payload.files?.rl_replay_trace_csv),
          has_genome_variability: csvHeaderHasColumn(bodyPath, "gene_expr_0"),
          total_ticks: payload.summary?.cell_clean?.total_ticks ?? null,
          max_ticks: payload.summary?.cell_clean?.max_ticks ?? null,
          max_abs_theta_deg: payload.analysis.max_abs_theta_deg,
          task_primary: payload.summary?.cell_clean?.task_primary ?? null,
          task_primary_label: payload.summary?.cell_clean?.task_primary_label ?? null,
          damage_ticks: payload.analysis.damage_ticks,
          genome_file: payload.summary?.notes ?? null
        };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a, b) => Date.parse(b.updated_utc || 0) - Date.parse(a.updated_utc || 0))
    .slice(0, limit);
}

function isCellEngineArtifactRunDir(dir) {
  const relDir = relFromReports(dir);
  if (
    relDir.startsWith("cellengine_previews/") ||
    relDir.includes("/cellengine_previews/")
  ) {
    return false;
  }
  if (
    relDir.startsWith("cellengine_searches/") ||
    relDir.includes("/cellengine_searches/")
  ) {
    return false;
  }
  if (
    relDir.startsWith("cellengine_ui/replay_") ||
    relDir === "cellengine_ui/replay_compatible_latest" ||
    relDir.includes("/cellengine_ui/replay_")
  ) {
    return false;
  }
  if (fs.existsSync(path.join(dir, "replay_trace.csv"))) return false;
  return true;
}

function buildCellEngineRunFiles(dir) {
  const summaryPath = path.join(dir, "summary.json");
  const reportPath = path.join(dir, "report.md");
  const genomePath = path.join(dir, "champion_genome.csv");
  const replayTracePath = path.join(dir, "replay_trace.csv");
  const replayBodyPath = path.join(dir, "replay_body.csv");
  const replayCellsPath = path.join(dir, "replay_cells.csv");
  const replayEdgesPath = path.join(dir, "replay_edges.csv");
  const rlReplayTracePath = path.join(dir, "rl_replay_trace.csv");
  const cleanTeacherContextPath = path.join(dir, "clean_teacher_trial_context.json");
  const cleanTeacherPrePath = path.join(dir, "clean_teacher_trial_pre_reset_cells.csv");
  const cleanTeacherPostPath = path.join(dir, "clean_teacher_trial_post_reset_cells.csv");
  const cleanAutonomousContextPath = path.join(dir, "clean_autonomous_trial_context.json");
  const cleanAutonomousPrePath = path.join(dir, "clean_autonomous_trial_pre_reset_cells.csv");
  const cleanAutonomousPostPath = path.join(dir, "clean_autonomous_trial_post_reset_cells.csv");
  const damagedAutonomousContextPath = path.join(dir, "damaged_autonomous_trial_context.json");
  const damagedAutonomousPrePath = path.join(dir, "damaged_autonomous_trial_pre_reset_cells.csv");
  const damagedAutonomousPostPath = path.join(dir, "damaged_autonomous_trial_post_reset_cells.csv");
  return {
    summary_json: fs.existsSync(summaryPath) ? `/reports/${relFromReports(summaryPath)}` : null,
    report_md: fs.existsSync(reportPath) ? `/reports/${relFromReports(reportPath)}` : null,
    champion_genome_csv: fs.existsSync(genomePath) ? `/reports/${relFromReports(genomePath)}` : null,
    replay_trace_csv: fs.existsSync(replayTracePath) ? `/reports/${relFromReports(replayTracePath)}` : null,
    replay_body_csv: fs.existsSync(replayBodyPath) ? `/reports/${relFromReports(replayBodyPath)}` : null,
    replay_cells_csv: fs.existsSync(replayCellsPath) ? `/reports/${relFromReports(replayCellsPath)}` : null,
    replay_edges_csv: fs.existsSync(replayEdgesPath) ? `/reports/${relFromReports(replayEdgesPath)}` : null,
    rl_replay_trace_csv: fs.existsSync(rlReplayTracePath) ? `/reports/${relFromReports(rlReplayTracePath)}` : null,
    clean_teacher_trial_context_json: fs.existsSync(cleanTeacherContextPath) ? `/reports/${relFromReports(cleanTeacherContextPath)}` : null,
    clean_teacher_trial_pre_reset_cells_csv: fs.existsSync(cleanTeacherPrePath) ? `/reports/${relFromReports(cleanTeacherPrePath)}` : null,
    clean_teacher_trial_post_reset_cells_csv: fs.existsSync(cleanTeacherPostPath) ? `/reports/${relFromReports(cleanTeacherPostPath)}` : null,
    clean_autonomous_trial_context_json: fs.existsSync(cleanAutonomousContextPath) ? `/reports/${relFromReports(cleanAutonomousContextPath)}` : null,
    clean_autonomous_trial_pre_reset_cells_csv: fs.existsSync(cleanAutonomousPrePath) ? `/reports/${relFromReports(cleanAutonomousPrePath)}` : null,
    clean_autonomous_trial_post_reset_cells_csv: fs.existsSync(cleanAutonomousPostPath) ? `/reports/${relFromReports(cleanAutonomousPostPath)}` : null,
    damaged_autonomous_trial_context_json: fs.existsSync(damagedAutonomousContextPath) ? `/reports/${relFromReports(damagedAutonomousContextPath)}` : null,
    damaged_autonomous_trial_pre_reset_cells_csv: fs.existsSync(damagedAutonomousPrePath) ? `/reports/${relFromReports(damagedAutonomousPrePath)}` : null,
    damaged_autonomous_trial_post_reset_cells_csv: fs.existsSync(damagedAutonomousPostPath) ? `/reports/${relFromReports(damagedAutonomousPostPath)}` : null
  };
}

function buildCellEngineRunLibraryEntry(dir) {
  if (!isCellEngineArtifactRunDir(dir)) return null;
  const summary = loadCellEngineSummaryFromDir(dir);
  if (!summary || typeof summary !== "object" || !Object.keys(summary).length) return null;
  const summaryPath = path.join(dir, "summary.json");
  const reportPath = path.join(dir, "report.md");
  const genomePath = path.join(dir, "champion_genome.csv");
  const updatedMs = Math.max(
    fileMtimeMs(summaryPath),
    fileMtimeMs(reportPath),
    fileMtimeMs(genomePath)
  );
  return {
    relative_output_dir: `reports/${relFromReports(dir)}`,
    task_name: summary?.task_name || "cartpole_balance",
    updated_utc: updatedMs ? new Date(updatedMs).toISOString() : null,
    source_kind: relFromReports(dir).startsWith("cellengine_searches/") ? "discovery_run" : "benchmark_run",
    summary,
    has_odd: Boolean(summary?.odd?.available),
    champion_genome_path: fs.existsSync(genomePath) ? relFromRoot(genomePath) : null,
    files: buildCellEngineRunFiles(dir)
  };
}

function listCellEngineBenchmarkRuns(limit = 48) {
  const dirs = [...new Set(walkReportFiles("summary.json").map((summaryPath) => path.dirname(summaryPath)))];
  return dirs
    .map((dir) => buildCellEngineRunLibraryEntry(dir))
    .filter(Boolean)
    .sort((a, b) => Date.parse(b.updated_utc || 0) - Date.parse(a.updated_utc || 0))
    .slice(0, limit);
}

function rankDiscoveryRun(run) {
  const solved = run?.summary?.solved ? 1 : 0;
  const survival = Number(run?.summary?.cell_clean?.survival_ratio);
  const success = Number(run?.summary?.cell_clean?.success_rate);
  return [
    solved,
    Number.isFinite(success) ? success : -1,
    Number.isFinite(survival) ? survival : -1
  ];
}

function compareRankTuples(a, b) {
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const av = a[index] ?? -Infinity;
    const bv = b[index] ?? -Infinity;
    if (bv !== av) return bv - av;
  }
  return 0;
}

function listCellEngineDiscoveryBatches(limit = 24) {
  return walkReportFiles("discovery_summary.json")
    .map((summaryPath) => {
      const batchDir = path.dirname(summaryPath);
      const parsed = parseJsonSafe(summaryPath, null);
      if (!parsed || typeof parsed !== "object") return null;
      const runs = Array.isArray(parsed.runs) ? parsed.runs : [];
      const sortedRuns = [...runs].sort((left, right) => compareRankTuples(rankDiscoveryRun(left), rankDiscoveryRun(right)));
      const bestRun = sortedRuns[0] || null;
      const bestRunDir = bestRun?.relative_output_dir
        ? safeResolveReportDir(bestRun.relative_output_dir)
        : null;
      const runUpdatedMs = runs.map((run) => {
        if (!run?.relative_output_dir) return 0;
        try {
          return fileMtimeMs(path.join(safeResolveReportDir(run.relative_output_dir), "summary.json"));
        } catch {
          return 0;
        }
      });
      const updatedMs = Math.max(fileMtimeMs(summaryPath), ...runUpdatedMs);
      return {
        batch_id: parsed.batch_id || path.basename(batchDir),
        relative_output_dir: `reports/${relFromReports(batchDir)}`,
        updated_utc: updatedMs ? new Date(updatedMs).toISOString() : null,
        task_name: parsed?.config?.task_name || bestRun?.summary?.task_name || "cartpole_balance",
        config: parsed.config || {},
        aggregate: parsed.aggregate || {},
        files: {
          discovery_summary_json: `/reports/${relFromReports(summaryPath)}`
        },
        best_run: bestRun
          ? {
              relative_output_dir: bestRun.relative_output_dir || null,
              champion_genome_path: bestRun.champion_genome_path || null,
              summary: bestRun.summary || null,
              files: bestRunDir ? buildCellEngineRunFiles(bestRunDir) : {}
            }
          : null
      };
    })
    .filter(Boolean)
    .sort((a, b) => Date.parse(b.updated_utc || 0) - Date.parse(a.updated_utc || 0))
    .slice(0, limit);
}

function loadCellEnginePersistedDiscoveryBatch(relativeOutputDir) {
  const batchDir = safeResolveReportDir(relativeOutputDir);
  const summaryPath = path.join(batchDir, "discovery_summary.json");
  const parsed = parseJsonSafe(summaryPath, null);
  if (!parsed || typeof parsed !== "object") {
    throw new Error(`Missing or invalid discovery_summary.json in ${relativeOutputDir}`);
  }
  const summaryMtime = fileMtimeMs(summaryPath);
  const rawRuns = Array.isArray(parsed.runs) ? parsed.runs : [];
  const runs = rawRuns.map((run, index) => {
    const runRelativeOutputDir = run?.relative_output_dir || null;
    const runDir = runRelativeOutputDir ? safeResolveReportDir(runRelativeOutputDir) : null;
    const summary = runDir ? loadCellEngineSummaryFromDir(runDir) : (run?.summary || null);
    const summaryCellClean = summary?.cell_clean || null;
    const rawPopulation = runDir ? loadCellEngineLiveDiscoveryPopulation(runDir) : null;
    const livePopulation = rawPopulation && Array.isArray(rawPopulation.candidates)
      ? {
          ...rawPopulation,
          candidates: rawPopulation.candidates.map((candidate) => {
            if (candidate?.summary && typeof candidate.summary === "object") {
              return {
                ...candidate,
                summary_quality: "measured"
              };
            }
            if (candidate?.rank === 1 && summaryCellClean) {
              return {
                ...candidate,
                summary: {
                  success_rate: summaryCellClean.success_rate ?? null,
                  survival_ratio: summaryCellClean.survival_ratio ?? null,
                  task_primary: summaryCellClean.task_primary ?? null,
                  task_primary_label: summaryCellClean.task_primary_label || "task progress"
                },
                summary_quality: "restored"
              };
            }
            if (Number.isFinite(Number(candidate?.fitness))) {
              return {
                ...candidate,
                summary: {
                  success_rate: null,
                  survival_ratio: null,
                  task_primary: Number(candidate.fitness),
                  task_primary_label: "search_fitness"
                },
                summary_quality: "fitness_only"
              };
            }
            return {
              ...candidate,
              summary_quality: "none"
            };
          })
        }
      : rawPopulation;
    const championGenomePath = run?.champion_genome_path
      || (runDir && fs.existsSync(path.join(runDir, "champion_genome.csv")) ? relFromRoot(path.join(runDir, "champion_genome.csv")) : null);
    return {
      ...(runDir
        ? {
            live_snapshot: (() => {
              const snapshot = loadCellEngineLiveDiscoverySnapshot(runDir);
              if (!snapshot) return null;
              const bestSummaryQuality =
                snapshot?.meta?.best_summary && typeof snapshot.meta.best_summary === "object"
                  ? "measured"
                  : (summaryCellClean ? "restored" : "none");
              return {
                ...snapshot,
                meta: {
                  ...snapshot.meta,
                  best_summary_quality: bestSummaryQuality
                }
              };
            })(),
            live_population: livePopulation
          }
        : { live_snapshot: null, live_population: null }),
      run_index: Number.isInteger(run?.run_index) ? run.run_index : index,
      seed: Number.isFinite(run?.seed) ? run.seed : null,
      status: "completed",
      relative_output_dir: runRelativeOutputDir,
      summary,
      champion_genome_path: championGenomePath,
      candidate_config: run?.candidate_config || null,
      process: run?.process || null,
      stdout_tail: run?.stdout_tail || "",
      stderr_tail: run?.stderr_tail || ""
    };
  });
  const completedRuns = runs.filter((run) => run.status === "completed");
  const result = {
    ...parsed,
    relative_output_dir: `reports/${relFromReports(batchDir)}`,
    runs: runs.map((run) => ({
      run_index: run.run_index,
      seed: run.seed,
      status: run.status,
      relative_output_dir: run.relative_output_dir,
      summary: run.summary,
      champion_genome_path: run.champion_genome_path,
      candidate_config: run.candidate_config
    }))
  };
  const job = {
    job_id: `persisted:${relFromReports(batchDir)}`,
    status: "completed",
    created_utc: summaryMtime ? new Date(summaryMtime).toISOString() : null,
    started_utc: summaryMtime ? new Date(summaryMtime).toISOString() : null,
    finished_utc: summaryMtime ? new Date(summaryMtime).toISOString() : null,
    batch_id: parsed.batch_id || path.basename(batchDir),
    relative_output_dir: `reports/${relFromReports(batchDir)}`,
    config: parsed.config || {},
    total_runs: runs.length,
    completed_runs: completedRuns.length,
    current_run_index: null,
    current_seed: null,
    aggregate: parsed.aggregate || summarizeDiscoveryRuns(completedRuns),
    runs,
    error: "",
    result
  };
  return { job, result };
}

function clampIntegerInput(value, fallback, lo, hi) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, Math.round(n)));
}

function clampOptionalNumber(value, lo, hi) {
  if (value == null || value === "") return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.max(lo, Math.min(hi, n));
}

async function runCellEngineReplay(body = {}) {
  const binary = path.join(ROOT_DIR, "bin", "benchmark_cellengine");
  if (!fs.existsSync(binary)) {
    throw new Error("Binary not found: bin/benchmark_cellengine. Build it with `make benchmark_cellengine`.");
  }

  const defaultGenome = path.join(CELLENGINE_LATEST_DIR, "champion_genome.csv");
  const genomePath = safeResolveFromRoot(body.genome_path || relFromRoot(defaultGenome));
  if (!fs.existsSync(genomePath)) {
    throw new Error(`Genome file not found: ${relFromRoot(genomePath)}`);
  }

  const maxTicks = clampIntegerInput(body.max_ticks, 500, 20, 4000);
  const seed = clampIntegerInput(body.seed, 42, 1, 2147483647);
  const taskNameRaw = lowerTrim(body.task_name || "cartpole_balance");
  const taskName = ["cartpole_balance", "mass_spring_balance", "worm_drag_race", "pong_return"].includes(taskNameRaw)
    ? taskNameRaw
    : "cartpole_balance";
  const damageTickRaw = clampOptionalNumber(body.damage_tick, 0, maxTicks - 1);
  const thetaDeg = clampOptionalNumber(body.theta_deg, -14.0, 14.0);
  const taskParamA = clampOptionalNumber(body.task_param_a, -10.0, 10.0);
  const taskParamB = clampOptionalNumber(body.task_param_b, -10.0, 10.0);
  const rlAlgoRaw = lowerTrim(body.rl_algo || "a2c");
  const rlAlgo = ["a2c", "dqn", "none"].includes(rlAlgoRaw) ? rlAlgoRaw : "a2c";
  const replayId = `replay_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const replayDir = path.join(REPORTS_DIR, "cellengine_ui", replayId);
  fs.mkdirSync(replayDir, { recursive: true });

  const env = {
    ...process.env,
    CELLENGINE_MODE: "replay",
    CELLENGINE_TASK: taskName,
    CELLENGINE_LOAD_GENOME: genomePath,
    CELLENGINE_OUT: replayDir,
    CELLENGINE_SEED: String(seed),
    CELLENGINE_RL_ALGO: "none",
    CELLENGINE_FINAL_TICKS: String(maxTicks),
    CELLENGINE_FINAL_TRIALS: "1",
    CELLENGINE_REPLAY_SLEEP_MS: "0",
    CELLENGINE_REPLAY_FRAME_STRIDE: "1",
    CELLENGINE_REPLAY_CLEAR: "0",
    CELLENGINE_REPLAY_STDOUT: "0"
  };
  if (damageTickRaw != null) env.CELLENGINE_REPLAY_DAMAGE_TICK = String(damageTickRaw);
  if (thetaDeg != null) env.CELLENGINE_REPLAY_THETA_DEG = String(thetaDeg);
  if (taskParamA != null) env.CELLENGINE_REPLAY_TASK_A = String(taskParamA);
  if (taskParamB != null) env.CELLENGINE_REPLAY_TASK_B = String(taskParamB);

  const stdoutChunks = [];
  const stderrChunks = [];
  let stdoutBytes = 0;
  let stderrBytes = 0;
  const exitInfo = await new Promise((resolve, reject) => {
    let child;
    let timedOut = false;
    let timeout = null;
    try {
      child = spawn(binary, [], { cwd: ROOT_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
    } catch (err) {
      reject(err);
      return;
    }
    timeout = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
    }, Math.max(1000, CELLENGINE_API_REPLAY_TIMEOUT_MS));

    child.stdout?.on("data", (chunk) => {
      const text = String(chunk);
      if (stdoutBytes < 32000) {
        stdoutChunks.push(text);
        stdoutBytes += text.length;
      }
    });
    child.stderr?.on("data", (chunk) => {
      const text = String(chunk);
      if (stderrBytes < 32000) {
        stderrChunks.push(text);
        stderrBytes += text.length;
      }
    });
    child.on("error", (err) => {
      if (timeout) clearTimeout(timeout);
      reject(err);
    });
    child.on("close", (code, signal) => {
      if (timeout) clearTimeout(timeout);
      resolve({ code, signal, timed_out: timedOut });
    });
  });

  const summaryPath = path.join(replayDir, "summary.json");
  const tracePath = path.join(replayDir, "replay_trace.csv");
  if (!fs.existsSync(summaryPath) || !fs.existsSync(tracePath)) {
    const stderr = stderrChunks.join("").trim();
    const stdout = stdoutChunks.join("").trim();
    const details = [stderr, stdout].filter(Boolean).join("\n").slice(0, 1200);
    throw new Error(
      `CellEngine replay failed to produce trace artifacts (exit=${exitInfo.code ?? "null"}${exitInfo.signal ? ` signal=${exitInfo.signal}` : ""}${exitInfo.timed_out ? " timeout=true" : ""}).${details ? ` ${details}` : ""}`
    );
  }

  let rlCacheMeta = null;
  if (rlAlgo !== "none") {
    rlCacheMeta = attachCachedRlComparator(replayDir, rlAlgo, taskName);
  }

  const payload = loadCellEngineReplayPayload(replayDir, { includeFrames: true });
  return {
    ...payload,
    config: {
      genome_path: relFromRoot(genomePath),
      max_ticks: maxTicks,
      task_name: taskName,
      damage_tick: damageTickRaw,
      theta_deg: thetaDeg,
      task_param_a: taskParamA,
      task_param_b: taskParamB,
      seed,
      rl_algo: rlAlgo,
      rl_source: rlAlgo === "none" ? "disabled" : (rlCacheMeta ? "cached_best" : "missing_cache"),
      rl_source_replay_dir: rlCacheMeta?.source_replay_dir || null
    },
    process: {
      exit_code: exitInfo.code,
      signal: exitInfo.signal || null,
      timed_out: Boolean(exitInfo.timed_out),
      stdout_tail: stdoutChunks.join("").slice(-4000),
      stderr_tail: stderrChunks.join("").slice(-4000)
    }
  };
}

function readChampionGenomeVector(genomeCsvPath) {
  if (!fs.existsSync(genomeCsvPath)) return [];
  return readCsvRows(genomeCsvPath)
    .map((row) => ({
      gene: Number(row.gene),
      value: Number(row.value)
    }))
    .filter((row) => Number.isFinite(row.gene) && row.gene >= 0 && row.gene < 96 && Number.isFinite(row.value))
    .sort((a, b) => a.gene - b.gene)
    .map((row) => row.value);
}

function summarizeDiscoveryRuns(runs = []) {
  const completed = runs.filter((run) => run && run.summary && typeof run.summary === "object" && Object.keys(run.summary).length);
  const genomes = completed
    .map((run) => run.champion_genes)
    .filter((genes) => Array.isArray(genes) && genes.length);
  const geneCount = genomes[0]?.length || 0;
  const genes = [];
  for (let geneIndex = 0; geneIndex < geneCount; geneIndex += 1) {
    const values = genomes
      .map((row) => row[geneIndex])
      .filter((value) => Number.isFinite(value));
    genes.push({
      gene: geneIndex,
      mean: numericMean(values),
      std: numericStd(values),
      min: numericMin(values),
      max: numericMax(values)
    });
  }

  const survival = completed
    .map((run) => run.summary?.cell_clean?.survival_ratio)
    .filter((value) => Number.isFinite(value));
  const success = completed
    .map((run) => run.summary?.cell_clean?.success_rate)
    .filter((value) => Number.isFinite(value));
  const taskPrimary = completed
    .map((run) => run.summary?.cell_clean?.task_primary)
    .filter((value) => Number.isFinite(value));
  const solvedRuns = completed.filter((run) => run.summary?.solved === true).length;
  const championParams = completed
    .map((run) => run.summary?.champion_params)
    .filter((params) => params && typeof params === "object");
  const numericParamSummary = (key) => {
    const values = championParams
      .map((params) => Number(params?.[key]))
      .filter((value) => Number.isFinite(value));
    return {
      mean: numericMean(values),
      std: numericStd(values),
      min: numericMin(values),
      max: numericMax(values)
    };
  };
  const bodyModeCounts = championParams.reduce((acc, params) => {
    const key = String(params?.body_mode || "unknown");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  return {
    task_name: completed[0]?.summary?.task_name || null,
    task_primary_label: completed[0]?.summary?.cell_clean?.task_primary_label || null,
    completed_runs: completed.length,
    solved_runs: solvedRuns,
    solved_fraction: completed.length ? solvedRuns / completed.length : null,
    mean_survival_ratio: numericMean(survival),
    survival_std: numericStd(survival),
    mean_success_rate: numericMean(success),
    success_std: numericStd(success),
    mean_task_primary: numericMean(taskPrimary),
    task_primary_std: numericStd(taskPrimary),
    champion_params: {
      body_mode_counts: bodyModeCounts,
      development_steps: numericParamSummary("development_steps"),
      development_seed_half_width: numericParamSummary("development_seed_half_width"),
      max_cells: numericParamSummary("max_cells"),
      body_extent_x: numericParamSummary("body_extent_x"),
      body_extent_y: numericParamSummary("body_extent_y"),
      body_extent_z: numericParamSummary("body_extent_z"),
      chemical_diffusion_steps: numericParamSummary("chemical_diffusion_steps"),
      development_growth_threshold: numericParamSummary("development_growth_threshold"),
      chemical_diffusion_rate: numericParamSummary("chemical_diffusion_rate"),
      chemical_decay: numericParamSummary("chemical_decay")
    },
    genes
  };
}

function recommendedCellEngineTaskConfig(taskName) {
  switch (lowerTrim(taskName || "")) {
    case "worm_drag_race":
      return {
        body_mode: "grown2d",
        max_cells: 96,
        body_extent_x: 10,
        body_extent_y: 10,
        body_extent_z: 1
      };
    case "pong_return":
      return {
        body_mode: "grown2d",
        max_cells: 96,
        body_extent_x: 8,
        body_extent_y: 12,
        body_extent_z: 1
      };
    case "mass_spring_balance":
      return {
        body_mode: "grown3d",
        max_cells: 96,
        body_extent_x: 6,
        body_extent_y: 14,
        body_extent_z: 2
      };
    case "cartpole_balance":
    default:
      return {
        body_mode: "grown3d",
        max_cells: 96,
        body_extent_x: 6,
        body_extent_y: 14,
        body_extent_z: 2
      };
  }
}

function applyCellEngineCandidateParamEnv(env, body = {}, taskDefaults = {}) {
  const bodyModeRaw = lowerTrim(body.body_mode || taskDefaults.body_mode || "grown3d");
  const bodyMode = ["fixed2d", "grown2d", "grown3d"].includes(bodyModeRaw) ? bodyModeRaw : (taskDefaults.body_mode || "grown3d");
  const maxCells = clampIntegerInput(body.max_cells, taskDefaults.max_cells || 96, 18, 512);
  const requestedDevelopmentSteps = clampIntegerInput(body.development_steps, bodyMode === "fixed2d" ? 1 : 8, 1, 64);
  const seedHalfWidth = clampIntegerInput(body.development_seed_half_width, 2, 1, 16);
  const bodyExtentX = clampIntegerInput(body.body_extent_x, taskDefaults.body_extent_x || 6, 2, 64);
  const bodyExtentY = clampIntegerInput(body.body_extent_y, taskDefaults.body_extent_y || 14, 4, 64);
  const requestedBodyExtentZ = clampIntegerInput(body.body_extent_z, taskDefaults.body_extent_z || 0, 0, 16);
  const chemicalDiffusionSteps = clampIntegerInput(body.chemical_diffusion_steps, 2, 1, 16);
  const growthThreshold = clampOptionalNumber(body.development_growth_threshold, 0.15, 0.90);
  const chemicalDiffusionRate = clampOptionalNumber(body.chemical_diffusion_rate, 0.05, 0.60);
  const chemicalDecay = clampOptionalNumber(body.chemical_decay, 0.0, 0.25);
  const developmentSteps = bodyMode === "fixed2d" ? 1 : requestedDevelopmentSteps;
  const bodyExtentZ = bodyMode === "grown3d" ? Math.max(1, requestedBodyExtentZ) : requestedBodyExtentZ;

  env.CELLENGINE_BODY_MODE = bodyMode;
  env.CELLENGINE_MAX_CELLS = String(maxCells);
  env.CELLENGINE_DEVELOP_STEPS = String(developmentSteps);
  env.CELLENGINE_SEED_HALF_WIDTH = String(seedHalfWidth);
  env.CELLENGINE_BODY_EXTENT_X = String(bodyExtentX);
  env.CELLENGINE_BODY_EXTENT_Y = String(bodyExtentY);
  env.CELLENGINE_BODY_EXTENT_Z = String(bodyExtentZ);
  env.CELLENGINE_CHEM_DIFF_STEPS = String(chemicalDiffusionSteps);
  if (growthThreshold != null) env.CELLENGINE_GROWTH_THRESHOLD = String(growthThreshold);
  if (chemicalDiffusionRate != null) env.CELLENGINE_CHEM_DIFF_RATE = String(chemicalDiffusionRate);
  if (chemicalDecay != null) env.CELLENGINE_CHEM_DECAY = String(chemicalDecay);

  env.CELLENGINE_EVOLVE_BODY_MODE = parseBoolish(body.evolve_body_mode) ? "1" : "0";
  env.CELLENGINE_EVOLVE_DEVELOP_STEPS = parseBoolish(body.evolve_development_steps) ? "1" : "0";
  env.CELLENGINE_EVOLVE_SEED_HALF_WIDTH = parseBoolish(body.evolve_development_seed_half_width) ? "1" : "0";
  env.CELLENGINE_EVOLVE_MAX_CELLS = parseBoolish(body.evolve_max_cells) ? "1" : "0";
  env.CELLENGINE_EVOLVE_BODY_EXTENT_X = parseBoolish(body.evolve_body_extent_x) ? "1" : "0";
  env.CELLENGINE_EVOLVE_BODY_EXTENT_Y = parseBoolish(body.evolve_body_extent_y) ? "1" : "0";
  env.CELLENGINE_EVOLVE_BODY_EXTENT_Z = parseBoolish(body.evolve_body_extent_z) ? "1" : "0";
  env.CELLENGINE_EVOLVE_CHEM_STEPS = parseBoolish(body.evolve_chemical_diffusion_steps) ? "1" : "0";
  env.CELLENGINE_EVOLVE_GROWTH_THRESHOLD = parseBoolish(body.evolve_growth_threshold) ? "1" : "0";
  env.CELLENGINE_EVOLVE_CHEM_RATE = parseBoolish(body.evolve_chemical_diffusion_rate) ? "1" : "0";
  env.CELLENGINE_EVOLVE_CHEM_DECAY = parseBoolish(body.evolve_chemical_decay) ? "1" : "0";

  return {
    body_mode: bodyMode,
    max_cells: maxCells,
    development_steps: developmentSteps,
    development_seed_half_width: seedHalfWidth,
    body_extent_x: bodyExtentX,
    body_extent_y: bodyExtentY,
    body_extent_z: bodyExtentZ,
    chemical_diffusion_steps: chemicalDiffusionSteps,
    development_growth_threshold: growthThreshold,
    chemical_diffusion_rate: chemicalDiffusionRate,
    chemical_decay: chemicalDecay,
    evolve_body_mode: parseBoolish(body.evolve_body_mode),
    evolve_development_steps: parseBoolish(body.evolve_development_steps),
    evolve_development_seed_half_width: parseBoolish(body.evolve_development_seed_half_width),
    evolve_max_cells: parseBoolish(body.evolve_max_cells),
    evolve_body_extent_x: parseBoolish(body.evolve_body_extent_x),
    evolve_body_extent_y: parseBoolish(body.evolve_body_extent_y),
    evolve_body_extent_z: parseBoolish(body.evolve_body_extent_z),
    evolve_chemical_diffusion_steps: parseBoolish(body.evolve_chemical_diffusion_steps),
    evolve_growth_threshold: parseBoolish(body.evolve_growth_threshold),
    evolve_chemical_diffusion_rate: parseBoolish(body.evolve_chemical_diffusion_rate),
    evolve_chemical_decay: parseBoolish(body.evolve_chemical_decay)
  };
}

async function runCellEngineFullBenchmark(body = {}) {
  const binary = path.join(ROOT_DIR, "bin", "benchmark_cellengine");
  if (!fs.existsSync(binary)) {
    throw new Error("Binary not found: bin/benchmark_cellengine. Build it with `make benchmark_cellengine`.");
  }

  const defaultGenome = path.join(CELLENGINE_LATEST_DIR, "champion_genome.csv");
  const genomePath = safeResolveFromRoot(body.genome_path || relFromRoot(defaultGenome));
  if (!fs.existsSync(genomePath)) {
    throw new Error(`Genome file not found: ${relFromRoot(genomePath)}`);
  }

  const seed = clampIntegerInput(body.seed, 42, 1, 2147483647);
  const taskNameRaw = lowerTrim(body.task_name || "cartpole_balance");
  const taskName = ["cartpole_balance", "mass_spring_balance", "worm_drag_race", "pong_return"].includes(taskNameRaw)
    ? taskNameRaw
    : "cartpole_balance";
  const finalTrials = clampIntegerInput(body.final_trials, 100, 1, 4000);
  const finalTicks = clampIntegerInput(body.final_ticks, 500, 20, 4000);
  const damageTrials = clampIntegerInput(body.damage_trials, 40, 0, 4000);
  const damageTicks = clampIntegerInput(body.damage_ticks, finalTicks, 20, 4000);
  const oddEnabled = !(body.odd_enabled === false || body.odd === false);
  const oddTrials = clampIntegerInput(body.odd_trials, 24, 1, 4000);
  const oddTicks = clampIntegerInput(body.odd_ticks, finalTicks, 20, 4000);
  const rlAlgoRaw = lowerTrim(body.rl_algo || "all");
  const rlAlgo = ["none", "dqn", "a2c", "all"].includes(rlAlgoRaw) ? rlAlgoRaw : "all";
  const mode = rlAlgo === "none" ? "cell_only" : "full";
  const taskDefaults = recommendedCellEngineTaskConfig(taskName);
  const wormGoalDistance = clampOptionalNumber(body.worm_goal_distance, 1.0, 50.0);
  const wormMaxBackward = clampOptionalNumber(body.worm_max_backward, 0.25, 20.0);
  const pongTargetHits = clampOptionalNumber(body.pong_target_hits, 1, 100);
  const pongBallSpeed = clampOptionalNumber(body.pong_ball_speed, 0.45, 1.60);
  const pongPaddleHalf = clampOptionalNumber(body.pong_paddle_half_height, 0.08, 0.45);

  const runId = `full_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const runDir = path.join(REPORTS_DIR, "cellengine_ui", runId);
  fs.mkdirSync(runDir, { recursive: true });

  const env = {
    ...process.env,
    CELLENGINE_MODE: mode,
    CELLENGINE_TASK: taskName,
    CELLENGINE_LOAD_GENOME: genomePath,
    CELLENGINE_OUT: runDir,
    CELLENGINE_SEED: String(seed),
    CELLENGINE_RL_ALGO: rlAlgo,
    CELLENGINE_FINAL_TRIALS: String(finalTrials),
    CELLENGINE_FINAL_TICKS: String(finalTicks),
    CELLENGINE_DAMAGE_TRIALS: String(damageTrials),
    CELLENGINE_DAMAGE_TICKS: String(damageTicks),
    CELLENGINE_ODD: oddEnabled ? "1" : "0",
    CELLENGINE_ODD_TRIALS: String(oddTrials),
    CELLENGINE_ODD_TICKS: String(oddTicks),
    CELLENGINE_REPLAY_STDOUT: "0"
  };
  const candidateConfig = applyCellEngineCandidateParamEnv(env, body, taskDefaults);
  if (wormGoalDistance != null) env.CELLENGINE_WORM_GOAL_DISTANCE = String(wormGoalDistance);
  if (wormMaxBackward != null) env.CELLENGINE_WORM_MAX_BACKWARD = String(wormMaxBackward);
  if (pongTargetHits != null) env.CELLENGINE_PONG_TARGET_HITS = String(Math.round(pongTargetHits));
  if (pongBallSpeed != null) env.CELLENGINE_PONG_BALL_SPEED = String(pongBallSpeed);
  if (pongPaddleHalf != null) env.CELLENGINE_PONG_PADDLE_HALF = String(pongPaddleHalf);

  const stdoutChunks = [];
  const stderrChunks = [];
  let stdoutBytes = 0;
  let stderrBytes = 0;
  const exitInfo = await new Promise((resolve, reject) => {
    let child;
    let timedOut = false;
    let timeout = null;
    try {
      child = spawn(binary, [], { cwd: ROOT_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
    } catch (err) {
      reject(err);
      return;
    }

    timeout = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGTERM");
      } catch {
        // ignore
      }
    }, Math.max(30000, CELLENGINE_API_DISCOVER_TIMEOUT_MS));

    child.stdout?.on("data", (chunk) => {
      const text = String(chunk);
      if (stdoutBytes < 64000) {
        stdoutChunks.push(text);
        stdoutBytes += text.length;
      }
    });
    child.stderr?.on("data", (chunk) => {
      const text = String(chunk);
      if (stderrBytes < 64000) {
        stderrChunks.push(text);
        stderrBytes += text.length;
      }
    });
    child.on("error", (err) => {
      if (timeout) clearTimeout(timeout);
      reject(err);
    });
    child.on("close", (code, signal) => {
      if (timeout) clearTimeout(timeout);
      resolve({ code, signal, timed_out: timedOut });
    });
  });

  const summaryPath = path.join(runDir, "summary.json");
  if (!fs.existsSync(summaryPath) && !fs.existsSync(path.join(runDir, "report.md"))) {
    const stderr = stderrChunks.join("").trim();
    const stdout = stdoutChunks.join("").trim();
    const details = [stderr, stdout].filter(Boolean).join("\n").slice(0, 1200);
    throw new Error(
      `CellEngine full benchmark failed to produce summary artifacts (exit=${exitInfo.code ?? "null"}${exitInfo.signal ? ` signal=${exitInfo.signal}` : ""}${exitInfo.timed_out ? " timeout=true" : ""}).${details ? ` ${details}` : ""}`
    );
  }

  const championGenomePath = path.join(runDir, "champion_genome.csv");
  const summary = loadCellEngineSummaryFromDir(runDir);
  return {
    relative_output_dir: `reports/${relFromReports(runDir)}`,
    summary,
    champion_genome_path: fs.existsSync(championGenomePath) ? relFromRoot(championGenomePath) : relFromRoot(genomePath),
    files: {
      summary_json: fs.existsSync(summaryPath) ? `/reports/${relFromReports(summaryPath)}` : null,
      report_md: fs.existsSync(path.join(runDir, "report.md")) ? `/reports/${relFromReports(path.join(runDir, "report.md"))}` : null,
      champion_genome_csv: fs.existsSync(championGenomePath) ? `/reports/${relFromReports(championGenomePath)}` : null
    },
    config: {
      task_name: taskName,
      genome_path: relFromRoot(genomePath),
      seed,
      rl_algo: rlAlgo,
      ...candidateConfig,
      final_trials: finalTrials,
      final_ticks: finalTicks,
      damage_trials: damageTrials,
      damage_ticks: damageTicks,
      odd_enabled: oddEnabled,
      odd_trials: oddTrials,
      odd_ticks: oddTicks
    },
    process: {
      exit_code: exitInfo.code,
      signal: exitInfo.signal || null,
      timed_out: Boolean(exitInfo.timed_out),
      stdout_tail: stdoutChunks.join("").slice(-4000),
      stderr_tail: stderrChunks.join("").slice(-4000)
    }
  };
}

async function runCellEngineDiscovery(body = {}, progress = null) {
  const binary = path.join(ROOT_DIR, "bin", "benchmark_cellengine");
  if (!fs.existsSync(binary)) {
    throw new Error("Binary not found: bin/benchmark_cellengine. Build it with `make benchmark_cellengine`.");
  }

  const numRuns = clampIntegerInput(body.num_runs, 4, 1, 24);
  const seedStart = clampIntegerInput(body.seed_start ?? body.seed, 42, 1, 2147483647);
  const seedStep = clampIntegerInput(body.seed_step, 1, 1, 1000000);
  const populationSize = clampIntegerInput(body.population_size, 48, 8, 512);
  const generations = clampIntegerInput(body.generations, 50, 1, 1000);
  const searchTrials = clampIntegerInput(body.search_trials, 24, 1, 2000);
  const searchTicks = clampIntegerInput(body.search_ticks, 250, 20, 4000);
  const finalTrials = clampIntegerInput(body.final_trials, 100, 1, 4000);
  const finalTicks = clampIntegerInput(body.final_ticks, 500, 20, 4000);
  const autoAttempts = clampIntegerInput(body.auto_attempts, 3, 1, 16);
  const oddEnabled = !(body.odd_enabled === false || body.odd === false);
  const oddTrials = clampIntegerInput(body.odd_trials, 24, 1, 4000);
  const oddTicks = clampIntegerInput(body.odd_ticks, finalTicks, 20, 4000);
  const taskNameRaw = lowerTrim(body.task_name || "cartpole_balance");
  const taskName = ["cartpole_balance", "mass_spring_balance", "worm_drag_race", "pong_return"].includes(taskNameRaw)
    ? taskNameRaw
    : "cartpole_balance";
  const wormGoalDistance = clampOptionalNumber(body.worm_goal_distance, 1.0, 50.0);
  const wormMaxBackward = clampOptionalNumber(body.worm_max_backward, 0.25, 20.0);
  const pongTargetHits = clampOptionalNumber(body.pong_target_hits, 1, 100);
  const pongBallSpeed = clampOptionalNumber(body.pong_ball_speed, 0.45, 1.60);
  const pongPaddleHalf = clampOptionalNumber(body.pong_paddle_half_height, 0.08, 0.45);
  const rlAlgoRaw = lowerTrim(body.rl_algo || "none");
  const rlAlgo = ["none", "dqn", "a2c", "all"].includes(rlAlgoRaw) ? rlAlgoRaw : "none";
  const mode = rlAlgo === "none" ? "cell_only" : "full";
  const taskDefaults = recommendedCellEngineTaskConfig(taskName);

  const batchId = `discovery_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const batchDir = path.join(REPORTS_DIR, "cellengine_searches", batchId);
  fs.mkdirSync(batchDir, { recursive: true });
  const batchRelativeDir = `reports/${relFromReports(batchDir)}`;
  progress?.onBatchStart?.({
    batch_id: batchId,
    relative_output_dir: batchRelativeDir,
    config: {
      num_runs: numRuns,
      seed_start: seedStart,
      seed_step: seedStep,
      population_size: populationSize,
      generations,
      search_trials: searchTrials,
      search_ticks: searchTicks,
      final_trials: finalTrials,
      final_ticks: finalTicks,
      auto_attempts: autoAttempts,
      rl_algo: rlAlgo,
      odd_enabled: oddEnabled,
      odd_trials: oddTrials,
      odd_ticks: oddTicks,
      ...applyCellEngineCandidateParamEnv({}, body, taskDefaults)
    }
  });

  const runs = [];
  for (let runIndex = 0; runIndex < numRuns; runIndex += 1) {
    const seed = seedStart + runIndex * seedStep;
    const runDir = path.join(batchDir, `seed_${seed}`);
    fs.mkdirSync(runDir, { recursive: true });

    const env = {
      ...process.env,
      CELLENGINE_MODE: mode,
      CELLENGINE_TASK: taskName,
      CELLENGINE_RL_ALGO: rlAlgo,
      CELLENGINE_OUT: runDir,
      CELLENGINE_SEED: String(seed),
      CELLENGINE_POP: String(populationSize),
      CELLENGINE_GENS: String(generations),
      CELLENGINE_SEARCH_TRIALS: String(searchTrials),
      CELLENGINE_SEARCH_TICKS: String(searchTicks),
      CELLENGINE_FINAL_TRIALS: String(finalTrials),
      CELLENGINE_FINAL_TICKS: String(finalTicks),
      CELLENGINE_AUTO_ATTEMPTS: String(autoAttempts),
      CELLENGINE_ODD: oddEnabled ? "1" : "0",
      CELLENGINE_ODD_TRIALS: String(oddTrials),
      CELLENGINE_ODD_TICKS: String(oddTicks),
      CELLENGINE_REPLAY_STDOUT: "0"
    };
    const candidateConfig = applyCellEngineCandidateParamEnv(env, body, taskDefaults);
    progress?.onRunStart?.({
      run_index: runIndex,
      seed,
      relative_output_dir: `reports/${relFromReports(runDir)}`,
      candidate_config: candidateConfig
    });
    if (wormGoalDistance != null) env.CELLENGINE_WORM_GOAL_DISTANCE = String(wormGoalDistance);
    if (wormMaxBackward != null) env.CELLENGINE_WORM_MAX_BACKWARD = String(wormMaxBackward);
    if (pongTargetHits != null) env.CELLENGINE_PONG_TARGET_HITS = String(Math.round(pongTargetHits));
    if (pongBallSpeed != null) env.CELLENGINE_PONG_BALL_SPEED = String(pongBallSpeed);
    if (pongPaddleHalf != null) env.CELLENGINE_PONG_PADDLE_HALF = String(pongPaddleHalf);

    const stdoutChunks = [];
    const stderrChunks = [];
    let stdoutBytes = 0;
    let stderrBytes = 0;

    const exitInfo = await new Promise((resolve, reject) => {
      let child;
      let timedOut = false;
      let timeout = null;
      try {
        child = spawn(binary, [], { cwd: ROOT_DIR, env, stdio: ["ignore", "pipe", "pipe"] });
      } catch (err) {
        reject(err);
        return;
      }

      timeout = setTimeout(() => {
        timedOut = true;
        try {
          child.kill("SIGTERM");
        } catch {
          // ignore
        }
      }, Math.max(30000, CELLENGINE_API_DISCOVER_TIMEOUT_MS));

      child.stdout?.on("data", (chunk) => {
        const text = String(chunk);
        if (stdoutBytes < 64000) {
          stdoutChunks.push(text);
          stdoutBytes += text.length;
        }
        progress?.onRunOutput?.({
          run_index: runIndex,
          stdout_tail: stdoutChunks.join("").slice(-4000),
          stderr_tail: stderrChunks.join("").slice(-4000)
        });
      });
      child.stderr?.on("data", (chunk) => {
        const text = String(chunk);
        if (stderrBytes < 64000) {
          stderrChunks.push(text);
          stderrBytes += text.length;
        }
        progress?.onRunOutput?.({
          run_index: runIndex,
          stdout_tail: stdoutChunks.join("").slice(-4000),
          stderr_tail: stderrChunks.join("").slice(-4000)
        });
      });
      child.on("error", (err) => {
        if (timeout) clearTimeout(timeout);
        reject(err);
      });
      child.on("close", (code, signal) => {
        if (timeout) clearTimeout(timeout);
        resolve({ code, signal, timed_out: timedOut });
      });
    });

    const summary = loadCellEngineSummaryFromDir(runDir);
    const genomePath = path.join(runDir, "champion_genome.csv");
    const runPayload = {
      run_index: runIndex,
      seed,
      mode,
      rl_algo: rlAlgo,
      relative_output_dir: `reports/${relFromReports(runDir)}`,
      summary,
      candidate_config: candidateConfig,
      champion_genome_path: fs.existsSync(genomePath) ? relFromRoot(genomePath) : null,
      champion_genes: readChampionGenomeVector(genomePath),
      process: {
        exit_code: exitInfo.code,
        signal: exitInfo.signal || null,
        timed_out: Boolean(exitInfo.timed_out),
        stdout_tail: stdoutChunks.join("").slice(-4000),
        stderr_tail: stderrChunks.join("").slice(-4000)
      }
    };
    runs.push(runPayload);
    progress?.onRunComplete?.(runPayload);
  }

  const payload = {
    batch_id: batchId,
    relative_output_dir: batchRelativeDir,
    config: {
      num_runs: numRuns,
      seed_start: seedStart,
      seed_step: seedStep,
      population_size: populationSize,
      generations,
      search_trials: searchTrials,
      search_ticks: searchTicks,
      final_trials: finalTrials,
      final_ticks: finalTicks,
      auto_attempts: autoAttempts,
      rl_algo: rlAlgo,
      ...applyCellEngineCandidateParamEnv({}, body, taskDefaults),
      odd_enabled: oddEnabled,
      odd_trials: oddTrials,
      odd_ticks: oddTicks
    },
    aggregate: summarizeDiscoveryRuns(runs),
    runs
  };
  fs.writeFileSync(path.join(batchDir, "discovery_summary.json"), JSON.stringify(payload, null, 2));
  return payload;
}

const cellEngineDiscoveryJobs = new Map();
const CELLENGINE_JOBS_FILE = path.join(REPORTS_DIR, "_cellengine_discovery_jobs.json");

function persistCellEngineDiscoveryJobs() {
  try {
    const serializable = [...cellEngineDiscoveryJobs.values()].map((job) => ({
      job_id: job.job_id,
      status: job.status,
      created_utc: job.created_utc,
      started_utc: job.started_utc || null,
      finished_utc: job.finished_utc || null,
      batch_id: job.batch_id || null,
      relative_output_dir: job.relative_output_dir || null,
      config: job.config || null,
      total_runs: job.total_runs || 0,
      completed_runs: job.completed_runs || 0,
      current_run_index: job.current_run_index ?? null,
      current_seed: job.current_seed ?? null,
      error: job.error || "",
      runs: (job.runs || []).map((run) => ({
        run_index: run.run_index,
        seed: run.seed,
        status: run.status,
        relative_output_dir: run.relative_output_dir || null,
        summary: run.summary || null,
        champion_genome_path: run.champion_genome_path || null,
        candidate_config: run.candidate_config || null
      }))
    }));
    fs.writeFileSync(CELLENGINE_JOBS_FILE, JSON.stringify(serializable, null, 2));
  } catch {
    // best-effort persistence
  }
}

function restoreCellEngineDiscoveryJobs() {
  try {
    if (!fs.existsSync(CELLENGINE_JOBS_FILE)) return;
    const raw = JSON.parse(fs.readFileSync(CELLENGINE_JOBS_FILE, "utf-8"));
    if (!Array.isArray(raw)) return;
    for (const entry of raw) {
      if (!entry?.job_id) continue;
      // If a job was "running" at persist time, the process is gone now.
      // Check if it actually completed (results on disk) or mark as error.
      if (entry.status === "running" || entry.status === "queued") {
        const runs = Array.isArray(entry.runs) ? entry.runs : [];
        let allDone = runs.length > 0;
        for (const run of runs) {
          if (run.relative_output_dir) {
            const summaryPath = path.join(ROOT_DIR, run.relative_output_dir, "summary.json");
            if (fs.existsSync(summaryPath)) {
              run.status = "completed";
              try {
                run.summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
              } catch { /* ignore */ }
              const genomePath = path.join(ROOT_DIR, run.relative_output_dir, "champion_genome.csv");
              if (fs.existsSync(genomePath)) {
                run.champion_genome_path = path.relative(ROOT_DIR, genomePath);
              }
            } else {
              allDone = false;
            }
          } else if (run.status !== "completed") {
            allDone = false;
          }
        }
        if (allDone && runs.length > 0) {
          entry.status = "completed";
          entry.finished_utc = entry.finished_utc || new Date().toISOString();
          entry.completed_runs = runs.filter((r) => r.status === "completed").length;
          entry.result = summarizeDiscoveryRuns(runs.filter((r) => r.status === "completed"));
        } else {
          entry.status = "error";
          entry.error = entry.error || "Server restarted while job was running. Check disk for partial results.";
          entry.finished_utc = entry.finished_utc || new Date().toISOString();
        }
      }
      // Keep stdout/stderr empty on restore (not persisted)
      for (const run of entry.runs || []) {
        run.stdout_tail = "";
        run.stderr_tail = "";
      }
      cellEngineDiscoveryJobs.set(entry.job_id, entry);
    }
  } catch {
    // best-effort restore
  }
}

restoreCellEngineDiscoveryJobs();

function snapshotCellEngineDiscoveryJob(job) {
  const sourceRuns = Array.isArray(job.runs) ? job.runs : [];
  const runs = sourceRuns.map((run) => ({
    ...(run.relative_output_dir
      ? (() => {
          try {
            const runDir = safeResolveReportDir(run.relative_output_dir);
            return {
              live_snapshot: loadCellEngineLiveDiscoverySnapshot(runDir),
              live_population: loadCellEngineLiveDiscoveryPopulation(runDir)
            };
          } catch {
            return { live_snapshot: null, live_population: null };
          }
        })()
      : { live_snapshot: null, live_population: null }),
    run_index: run.run_index,
    seed: run.seed,
    status: run.status,
    relative_output_dir: run.relative_output_dir || null,
    summary: run.summary || null,
    champion_genome_path: run.champion_genome_path || null,
    candidate_config: run.candidate_config || null,
    process: run.process || null,
    stdout_tail: run.stdout_tail || "",
    stderr_tail: run.stderr_tail || ""
  }));
  const completedRuns = runs.filter((run) => run.status === "completed");
  return {
    job_id: job.job_id,
    status: job.status,
    created_utc: job.created_utc,
    started_utc: job.started_utc || null,
    finished_utc: job.finished_utc || null,
    batch_id: job.batch_id || null,
    relative_output_dir: job.relative_output_dir || null,
    config: job.config || null,
    total_runs: job.total_runs || runs.length,
    completed_runs: completedRuns.length,
    current_run_index: Number.isInteger(job.current_run_index) ? job.current_run_index : null,
    current_seed: Number.isFinite(job.current_seed) ? job.current_seed : null,
    aggregate: summarizeDiscoveryRuns(sourceRuns.filter((run) => run.status === "completed")),
    runs,
    error: job.error || "",
    result: job.status === "completed" ? job.result || null : null
  };
}

function trimDiscoveryJobCache(maxEntries = 8) {
  if (cellEngineDiscoveryJobs.size <= maxEntries) return;
  const entries = [...cellEngineDiscoveryJobs.values()].sort((a, b) => Date.parse(a.created_utc || 0) - Date.parse(b.created_utc || 0));
  while (entries.length > maxEntries) {
    const oldest = entries.shift();
    if (oldest) cellEngineDiscoveryJobs.delete(oldest.job_id);
  }
}

function latestActiveCellEngineDiscoveryJob() {
  const active = [...cellEngineDiscoveryJobs.values()]
    .filter((job) => job && (job.status === "queued" || job.status === "running"))
    .sort((a, b) => Date.parse(b.started_utc || b.created_utc || 0) - Date.parse(a.started_utc || a.created_utc || 0));
  return active[0] || null;
}

function startCellEngineDiscoveryJob(body = {}) {
  const jobId = `cellengine_discovery_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const numRuns = clampIntegerInput(body.num_runs, 4, 1, 24);
  const seedStart = clampIntegerInput(body.seed_start ?? body.seed, 42, 1, 2147483647);
  const seedStep = clampIntegerInput(body.seed_step, 1, 1, 1000000);
  const runs = Array.from({ length: numRuns }, (_, runIndex) => ({
    run_index: runIndex,
    seed: seedStart + runIndex * seedStep,
    status: "queued",
    stdout_tail: "",
    stderr_tail: ""
  }));
  const job = {
    job_id: jobId,
    status: "queued",
    created_utc: new Date().toISOString(),
    started_utc: null,
    finished_utc: null,
    batch_id: null,
    relative_output_dir: null,
    config: null,
    total_runs: numRuns,
    completed_runs: 0,
    current_run_index: null,
    current_seed: null,
    runs,
    error: "",
    result: null
  };
  cellEngineDiscoveryJobs.set(jobId, job);
  trimDiscoveryJobCache();
  persistCellEngineDiscoveryJobs();

  Promise.resolve().then(async () => {
    job.status = "running";
    job.started_utc = new Date().toISOString();
    persistCellEngineDiscoveryJobs();
    const result = await runCellEngineDiscovery(body, {
      onBatchStart(meta) {
        job.batch_id = meta.batch_id || null;
        job.relative_output_dir = meta.relative_output_dir || null;
        job.config = meta.config || null;
        persistCellEngineDiscoveryJobs();
      },
      onRunStart(meta) {
        job.current_run_index = meta.run_index;
        job.current_seed = meta.seed;
        if (job.runs[meta.run_index]) {
          job.runs[meta.run_index] = {
            ...job.runs[meta.run_index],
            status: "running",
            relative_output_dir: meta.relative_output_dir || null,
            candidate_config: meta.candidate_config || null
          };
        }
        persistCellEngineDiscoveryJobs();
      },
      onRunOutput(meta) {
        if (job.runs[meta.run_index]) {
          job.runs[meta.run_index] = {
            ...job.runs[meta.run_index],
            stdout_tail: meta.stdout_tail || "",
            stderr_tail: meta.stderr_tail || ""
          };
        }
      },
      onRunComplete(runPayload) {
        job.current_run_index = null;
        job.current_seed = null;
        job.completed_runs += 1;
        job.runs[runPayload.run_index] = {
          ...job.runs[runPayload.run_index],
          ...runPayload,
          status: "completed"
        };
        persistCellEngineDiscoveryJobs();
      }
    });
    job.status = "completed";
    job.finished_utc = new Date().toISOString();
    job.result = result;
    persistCellEngineDiscoveryJobs();
  }).catch((err) => {
    job.status = "error";
    job.finished_utc = new Date().toISOString();
    job.error = String(err?.message || err);
    persistCellEngineDiscoveryJobs();
  });

  return snapshotCellEngineDiscoveryJob(job);
}

function meanValueFinite(values) {
  const clean = (values || []).filter((v) => typeof v === "number" && Number.isFinite(v));
  if (!clean.length) return null;
  return clean.reduce((acc, v) => acc + v, 0) / clean.length;
}

function stdDevFinite(values) {
  const clean = (values || []).filter((v) => typeof v === "number" && Number.isFinite(v));
  if (clean.length < 2) return 0;
  const mu = meanValueFinite(clean);
  const variance = clean.reduce((acc, v) => acc + (v - mu) ** 2, 0) / clean.length;
  return Math.sqrt(variance);
}

function parseBoolish(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value !== "string") return false;
  const t = value.trim().toLowerCase();
  return t === "1" || t === "true" || t === "yes" || t === "on";
}

function parseEnvJson(raw) {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) return raw;
  if (typeof raw !== "string" || !raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
  } catch {
    return {};
  }
  return {};
}

function parseBitsFromComboId(comboId) {
  if (typeof comboId !== "string" || !comboId.startsWith("bio_on_")) return null;
  const bits = [];
  for (const feature of BIO_FEATURES) {
    const match = comboId.match(new RegExp(`${feature.short}([01])`));
    if (!match) return null;
    bits.push(match[1] === "1" ? "1" : "0");
  }
  return bits.join("");
}

function parseBitsFromEnv(env) {
  const bits = [];
  let foundAny = false;
  for (const feature of BIO_FEATURES) {
    if (Object.prototype.hasOwnProperty.call(env, feature.key)) foundAny = true;
    bits.push(parseBoolish(env[feature.key]) ? "1" : "0");
  }
  return foundAny ? bits.join("") : null;
}

function parseScore(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function normalizeModelName(value) {
  const model = String(value || "").trim().toLowerCase();
  return model || "unknown";
}

function parseBioSummaryRow(row) {
  const comboId = String(row.combo_id || "").trim();
  const env = parseEnvJson(row.env_json);
  const hasBioEnv =
    Object.prototype.hasOwnProperty.call(env, "BIO_ACTIVE") ||
    BIO_FEATURES.some((feature) => Object.prototype.hasOwnProperty.call(env, feature.key));
  const isBioCombo = comboId.startsWith("bio_on_") || comboId === "bio_observe_baseline";
  if (!isBioCombo && !hasBioEnv) return null;

  const model = normalizeModelName(row.model);
  const isBaseline = comboId === "bio_observe_baseline" || !parseBoolish(env.BIO_ACTIVE);
  const bitsFromEnv = parseBitsFromEnv(env);
  const bitsFromCombo = parseBitsFromComboId(comboId);
  const bits = isBaseline ? null : (bitsFromEnv || bitsFromCombo);
  const score = parseScore(row.score);
  const returncode = parseScore(row.returncode);

  return {
    model,
    combo_id: comboId || (isBaseline ? "bio_observe_baseline" : ""),
    bits,
    is_baseline: isBaseline,
    score,
    returncode,
    start_ts: typeof row.start_ts === "string" ? row.start_ts : null,
    end_ts: typeof row.end_ts === "string" ? row.end_ts : null
  };
}

function listBioStudyDirs() {
  if (!fs.existsSync(REPORTS_DIR)) return [];
  const dirs = new Set();
  const stack = [REPORTS_DIR];

  while (stack.length) {
    const cur = stack.pop();
    const entries = fs.readdirSync(cur, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(cur, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && entry.name === "summary.csv") {
        dirs.add(path.dirname(full));
      }
    }
  }

  return [...dirs];
}

function scanBioStudies() {
  const studies = [];
  for (const dir of listBioStudyDirs()) {
    const summaryPath = path.join(dir, "summary.csv");
    if (!fs.existsSync(summaryPath)) continue;

    let rows = [];
    try {
      rows = readCsvRows(summaryPath);
    } catch {
      continue;
    }
    const parsed = rows.map(parseBioSummaryRow).filter(Boolean);
    if (!parsed.length) continue;

    const metaPath = path.join(dir, "meta.json");
    const meta = parseJsonSafe(metaPath, {});
    const relId = path.relative(REPORTS_DIR, dir).split(path.sep).join("/");
    const models = [...new Set(parsed.map((row) => row.model))].sort();
    const updatedMs = Math.max(fileMtimeMs(summaryPath), fileMtimeMs(metaPath));
    const generatedAt = typeof meta.generated_at === "string" && meta.generated_at.trim()
      ? meta.generated_at
      : (updatedMs ? new Date(updatedMs).toISOString() : null);
    const finiteScoreCount = parsed.filter((row) => typeof row.score === "number").length;

    studies.push({
      study_id: relId,
      summary_path: summaryPath,
      generated_at: generatedAt,
      updated_utc: updatedMs ? new Date(updatedMs).toISOString() : null,
      combo_mode: typeof meta.combo_mode === "string" ? meta.combo_mode : null,
      combo_order: typeof meta.combo_order === "string" ? meta.combo_order : null,
      profile: typeof meta.profile === "string" ? meta.profile : null,
      models,
      total_rows: parsed.length,
      scored_rows: finiteScoreCount
    });
  }

  studies.sort((a, b) => {
    const ta = parseUtcMs(a.generated_at || "") || parseUtcMs(a.updated_utc || "");
    const tb = parseUtcMs(b.generated_at || "") || parseUtcMs(b.updated_utc || "");
    return tb - ta;
  });
  return studies;
}

function listBioAblationStudies() {
  const now = Date.now();
  if (now - bioStudyCache.scannedAtMs < BIO_ABLATION_CACHE_TTL_MS) {
    return bioStudyCache.studies;
  }
  const studies = scanBioStudies();
  bioStudyCache = { scannedAtMs: now, studies };
  return studies;
}

function getBioStudyById(studyId) {
  const studies = listBioAblationStudies();
  if (!studies.length) return null;
  if (!studyId) return studies[0];
  return studies.find((study) => study.study_id === studyId) || null;
}

function statsFromScores(values) {
  const clean = (values || []).filter((v) => typeof v === "number" && Number.isFinite(v));
  return {
    n: clean.length,
    mean: meanValueFinite(clean),
    std: stdDevFinite(clean),
    values: clean
  };
}

function analyzeBioRowsForModel(rows, model) {
  const modelRows = rows.filter((row) => row.model === model);
  if (!modelRows.length) return null;

  const comboMap = new Map();
  const byBits = new Map();
  const activeRows = [];
  const baselineScores = [];

  for (const row of modelRows) {
    if (typeof row.score !== "number" || !Number.isFinite(row.score)) continue;

    const comboKey = row.combo_id || (row.is_baseline ? "bio_observe_baseline" : `bits:${row.bits || "unknown"}`);
    if (!comboMap.has(comboKey)) {
      comboMap.set(comboKey, {
        combo_id: comboKey,
        bits: row.bits,
        is_baseline: row.is_baseline,
        scores: []
      });
    }
    comboMap.get(comboKey).scores.push(row.score);

    if (row.is_baseline) {
      baselineScores.push(row.score);
    } else if (typeof row.bits === "string" && row.bits.length === BIO_FEATURES.length) {
      activeRows.push(row);
      if (!byBits.has(row.bits)) byBits.set(row.bits, []);
      byBits.get(row.bits).push(row.score);
    }
  }

  const baseline = statsFromScores(baselineScores);
  const allOnBits = "1".repeat(BIO_FEATURES.length);
  const allOn = statsFromScores(byBits.get(allOnBits) || []);
  const baselineMean = baseline.mean;
  const allOnMean = allOn.mean;

  const combos = [...comboMap.values()]
    .map((entry) => {
      const stats = statsFromScores(entry.scores);
      return {
        combo_id: entry.combo_id,
        bits: entry.bits,
        is_baseline: entry.is_baseline,
        n: stats.n,
        mean: stats.mean,
        std: stats.std,
        delta_vs_baseline:
          typeof stats.mean === "number" && typeof baselineMean === "number"
            ? stats.mean - baselineMean
            : null
      };
    })
    .sort((a, b) => {
      const am = typeof a.mean === "number" ? a.mean : -Infinity;
      const bm = typeof b.mean === "number" ? b.mean : -Infinity;
      return bm - am;
    });

  const activeCombos = combos.filter((combo) => !combo.is_baseline);
  const bestCombo = activeCombos.length ? activeCombos[0] : null;

  const leaveOne = BIO_FEATURES.map((feature, idx) => {
    const bitsArr = Array(BIO_FEATURES.length).fill("1");
    bitsArr[idx] = "0";
    const bits = bitsArr.join("");
    const stats = statsFromScores(byBits.get(bits) || []);
    const mean = stats.mean;
    return {
      feature_short: feature.short,
      feature_name: feature.name,
      bits,
      n: stats.n,
      mean,
      std: stats.std,
      delta_vs_all_on:
        typeof mean === "number" && typeof allOnMean === "number"
          ? mean - allOnMean
          : null,
      delta_vs_baseline:
        typeof mean === "number" && typeof baselineMean === "number"
          ? mean - baselineMean
          : null
    };
  });

  const marginal = BIO_FEATURES.map((feature, idx) => {
    const on = [];
    const off = [];
    for (const row of activeRows) {
      if (row.bits[idx] === "1") on.push(row.score);
      else off.push(row.score);
    }
    const onStats = statsFromScores(on);
    const offStats = statsFromScores(off);
    return {
      feature_short: feature.short,
      feature_name: feature.name,
      on_n: onStats.n,
      off_n: offStats.n,
      on_mean: onStats.mean,
      off_mean: offStats.mean,
      delta:
        typeof onStats.mean === "number" && typeof offStats.mean === "number"
          ? onStats.mean - offStats.mean
          : null
    };
  });

  return {
    model,
    feature_order: BIO_FEATURES,
    counts: {
      rows: modelRows.length,
      scored_rows: modelRows.filter((row) => typeof row.score === "number").length,
      combos_observed: activeCombos.length + (baseline.n ? 1 : 0)
    },
    baseline: { n: baseline.n, mean: baseline.mean, std: baseline.std },
    all_on: {
      bits: allOnBits,
      n: allOn.n,
      mean: allOn.mean,
      std: allOn.std,
      delta_vs_baseline:
        typeof allOn.mean === "number" && typeof baselineMean === "number"
          ? allOn.mean - baselineMean
          : null
    },
    best_combo: bestCombo
      ? {
          ...bestCombo,
          delta_vs_all_on:
            typeof bestCombo.mean === "number" && typeof allOnMean === "number"
              ? bestCombo.mean - allOnMean
              : null
        }
      : null,
    leave_one: leaveOne,
    marginal,
    combos
  };
}

app.get("/api/health", (_, res) => {
  res.json({ ok: true, root: ROOT_DIR });
});

app.get("/api/cellengine/overview", (_req, res) => {
  const latestSummaryPath = path.join(CELLENGINE_LATEST_DIR, "summary.json");
  const latestGenomePath = path.join(CELLENGINE_LATEST_DIR, "champion_genome.csv");
  const latestReport = fs.existsSync(latestSummaryPath) || fs.existsSync(path.join(CELLENGINE_LATEST_DIR, "report.md"))
    ? {
        relative_output_dir: "reports/cellengine_latest",
        summary: loadCellEngineSummaryFromDir(CELLENGINE_LATEST_DIR),
        files: {
          summary_json: "/reports/cellengine_latest/summary.json",
          champion_genome_csv: fs.existsSync(latestGenomePath) ? "/reports/cellengine_latest/champion_genome.csv" : null,
          report_md: fs.existsSync(path.join(CELLENGINE_LATEST_DIR, "report.md"))
            ? "/reports/cellengine_latest/report.md"
            : null
        }
      }
    : null;

  res.json({
    available_tasks: [
      "cartpole_balance",
      "mass_spring_balance",
      "worm_drag_race",
      "pong_return"
    ],
    latest_report: latestReport,
    default_genome_path: fs.existsSync(latestGenomePath) ? relFromRoot(latestGenomePath) : null,
    genome_options: listCellEngineGenomeOptions(),
    recent_replays: listCellEngineReplaySummaries()
  });
});

app.get("/api/cellengine/artifacts", (_req, res) => {
  res.json({
    benchmark_runs: listCellEngineBenchmarkRuns(),
    discovery_batches: listCellEngineDiscoveryBatches()
  });
});

app.get("/api/cellengine/replay", (req, res) => {
  const relativeDir = String(req.query.dir || "").trim();
  if (!relativeDir) {
    return res.status(400).json({ error: "dir query parameter is required." });
  }
  try {
    const replayDir = safeResolveReportDir(relativeDir);
    const payload = loadCellEngineReplayPayload(replayDir, { includeFrames: true });
    return res.json(payload);
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/cellengine/genome-previews", async (req, res) => {
  try {
    const rawItems = Array.isArray(req.body?.items) ? req.body.items : [];
    const items = rawItems.slice(0, 64);
    const previews = [];
    for (const item of items) {
      try {
        previews.push(await loadCellEngineGenomePreview(item));
      } catch (err) {
        previews.push({
          genome_path: item?.genome_path || "",
          label: item?.label || item?.genome_path || "unknown",
          error: String(err?.message || err)
        });
      }
    }
    return res.json({ items: previews });
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/cellengine/replay", async (req, res) => {
  try {
    const payload = await runCellEngineReplay(req.body || {});
    return res.status(201).json(payload);
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/cellengine/full-benchmark", async (req, res) => {
  try {
    const payload = await runCellEngineFullBenchmark(req.body || {});
    return res.status(201).json(payload);
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/cellengine/discover", async (req, res) => {
  try {
    const payload = await runCellEngineDiscovery(req.body || {});
    return res.status(201).json(payload);
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/cellengine/discover/start", (req, res) => {
  try {
    const payload = startCellEngineDiscoveryJob(req.body || {});
    return res.status(201).json(payload);
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.get("/api/cellengine/discover/status", (req, res) => {
  const jobId = String(req.query.job_id || "").trim();
  if (!jobId) {
    return res.status(400).json({ error: "job_id query parameter is required." });
  }
  const job = cellEngineDiscoveryJobs.get(jobId);
  if (!job) {
    return res.status(404).json({ error: `Unknown discovery job: ${jobId}` });
  }
  return res.json(snapshotCellEngineDiscoveryJob(job));
});

app.get("/api/cellengine/discover/batch", (req, res) => {
  const relativeDir = String(req.query.dir || "").trim();
  if (!relativeDir) {
    return res.status(400).json({ error: "dir query parameter is required." });
  }
  try {
    return res.json(loadCellEnginePersistedDiscoveryBatch(relativeDir));
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.get("/api/cellengine/discover/active", (_req, res) => {
  const job = latestActiveCellEngineDiscoveryJob();
  if (!job) {
    return res.json({ job: null });
  }
  return res.json({ job: snapshotCellEngineDiscoveryJob(job) });
});

app.get("/api/tasks", (_req, res) => {
  refreshTaskLiveness();
  const tasks = sortTasksNewest([...taskStore.values()]).map(taskView);
  const active = tasks.filter((task) => ["queued", "running", "killing"].includes(task.status)).length;
  return res.json({ tasks, active });
});

app.post("/api/tasks/train", (req, res) => {
  refreshTaskLiveness();
  try {
    const launchSpec = buildTrainLaunchSpec(req.body || {});
    const task = launchTask(launchSpec, req.body || {});
    return res.status(201).json({ task });
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/tasks/evaluate", (req, res) => {
  refreshTaskLiveness();
  try {
    const launchSpec = buildEvaluateLaunchSpec(req.body || {});
    const task = launchTask(launchSpec, req.body || {});
    return res.status(201).json({ task });
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/tasks/:taskId/start", (req, res) => {
  refreshTaskLiveness();
  const taskId = String(req.params.taskId || "").trim();
  const source = taskStore.get(taskId);
  if (!source) return res.status(404).json({ error: "task not found" });
  if (!taskIsTerminal(source)) {
    return res.status(409).json({ error: "task is already active; kill or wait before starting again" });
  }
  try {
    const launchSpec = {
      kind: source.kind,
      benchmark_id: source.benchmark_id || null,
      model_family: source.model_family || null,
      target_run_id: source.target_run_id || null,
      command: source.command,
      args: Array.isArray(source.args) ? source.args : [],
      env: source.env_preview && typeof source.env_preview === "object" ? source.env_preview : {}
    };
    const task = launchTask(launchSpec, req.body || {});
    return res.status(201).json({ task, restarted_from: taskId });
  } catch (err) {
    return res.status(400).json({ error: String(err?.message || err) });
  }
});

app.post("/api/tasks/:taskId/kill", (req, res) => {
  refreshTaskLiveness();
  const taskId = String(req.params.taskId || "").trim();
  try {
    const task = killTask(taskId);
    return res.json({ task });
  } catch (err) {
    const message = String(err?.message || err);
    const status = message.includes("not found") ? 404 : 400;
    return res.status(status).json({ error: message });
  }
});

app.delete("/api/tasks/:taskId", (req, res) => {
  refreshTaskLiveness();
  const taskId = String(req.params.taskId || "").trim();
  try {
    deleteTask(taskId);
    return res.json({ ok: true, task_id: taskId });
  } catch (err) {
    const message = String(err?.message || err);
    const status = message.includes("not found") ? 404 : 400;
    return res.status(status).json({ error: message });
  }
});

app.get("/api/bio/ablation/studies", (_req, res) => {
  const studies = listBioAblationStudies();
  res.json({
    studies,
    latest_study_id: studies[0]?.study_id || null,
    feature_order: BIO_FEATURES
  });
});

app.get("/api/bio/ablation", (req, res) => {
  const studyId = String(req.query.study || "").trim();
  const study = getBioStudyById(studyId);
  if (!study) {
    return res.json({
      studies: [],
      study: null,
      feature_order: BIO_FEATURES,
      overview: [],
      model: null,
      model_analysis: null
    });
  }

  let rows = [];
  try {
    rows = readCsvRows(study.summary_path).map(parseBioSummaryRow).filter(Boolean);
  } catch (err) {
    return res.status(500).json({ error: "failed to parse bio summary", details: String(err) });
  }

  const models = [...new Set(rows.map((row) => row.model))].sort();
  const overview = models
    .map((model) => analyzeBioRowsForModel(rows, model))
    .filter(Boolean)
    .map((analysis) => ({
      model: analysis.model,
      baseline_mean: analysis.baseline.mean,
      all_on_mean: analysis.all_on.mean,
      all_on_delta_vs_baseline: analysis.all_on.delta_vs_baseline,
      best_combo_id: analysis.best_combo?.combo_id ?? null,
      best_combo_bits: analysis.best_combo?.bits ?? null,
      best_combo_mean: analysis.best_combo?.mean ?? null,
      best_combo_delta_vs_baseline: analysis.best_combo?.delta_vs_baseline ?? null,
      combos_observed: analysis.counts.combos_observed,
      scored_rows: analysis.counts.scored_rows
    }))
    .sort((a, b) => {
      const am = typeof a.best_combo_mean === "number" ? a.best_combo_mean : -Infinity;
      const bm = typeof b.best_combo_mean === "number" ? b.best_combo_mean : -Infinity;
      return bm - am;
    });

  const requestedModel = normalizeModelName(req.query.model || "");
  const modelForDetail = requestedModel && models.includes(requestedModel)
    ? requestedModel
    : (overview[0]?.model || null);
  const modelAnalysis = modelForDetail ? analyzeBioRowsForModel(rows, modelForDetail) : null;

  return res.json({
    studies: listBioAblationStudies(),
    study,
    feature_order: BIO_FEATURES,
    overview,
    model: modelForDetail,
    model_analysis: modelAnalysis
  });
});

app.get("/api/runs", (req, res) => {
  const benchmarkFilter = String(req.query.benchmark || "").trim();
  let runs = listRuns();
  if (benchmarkFilter) {
    runs = runs.filter((run) => run.benchmark_id === benchmarkFilter);
  }
  res.json({ runs });
});

app.get("/api/families", (req, res) => {
  const benchmarkFilter = String(req.query.benchmark || "").trim();
  res.json({ families: listFamilyCoverage(benchmarkFilter) });
});

app.get("/api/benchmarks", (_, res) => {
  res.json({ benchmarks: listBenchmarkCoverage() });
});

app.get("/api/runs/:runId/manifest", (req, res) => {
  const runDir = path.join(RUNS_DIR, req.params.runId);
  if (!isSafePath(RUNS_DIR, runDir) || !fs.existsSync(runDir)) {
    return res.status(404).json({ error: "run not found" });
  }
  const manifestPath = path.join(runDir, "manifest.json");
  if (!fs.existsSync(manifestPath)) return res.status(404).json({ error: "manifest missing" });
  return res.json(parseJsonSafe(manifestPath, {}));
});

app.get("/api/runs/:runId/model-specific-files", (req, res) => {
  const runDir = path.join(RUNS_DIR, req.params.runId);
  if (!isSafePath(RUNS_DIR, runDir) || !fs.existsSync(runDir)) {
    return res.status(404).json({ error: "run not found" });
  }
  return res.json({ files: listModelSpecificFiles(req.params.runId) });
});

app.get("/api/runs/:runId/csv", (req, res) => {
  const rel = String(req.query.path || "");
  if (!rel.endsWith(".csv")) {
    return res.status(400).json({ error: "path must be a csv" });
  }

  const runDir = path.join(RUNS_DIR, req.params.runId);
  if (!isSafePath(RUNS_DIR, runDir) || !fs.existsSync(runDir)) {
    return res.status(404).json({ error: "run not found" });
  }
  const filePath = path.resolve(runDir, rel);
  if (!isSafePath(runDir, filePath) || !fs.existsSync(filePath)) {
    return res.status(404).json({ error: "csv not found" });
  }

  try {
    let rows = readCsvRows(filePath);
    const offset = Math.max(0, Number(req.query.offset || 0) || 0);
    const limitRaw = Number(req.query.limit || 0) || 0;
    const limit = limitRaw > 0 ? Math.max(1, limitRaw) : 0;
    const tailRaw = Number(req.query.tail || 0) || 0;
    const tail = tailRaw > 0 ? Math.min(20000, Math.max(1, tailRaw)) : 0;
    const fromEpoch = Number(req.query.from_epoch);
    const fromStep = Number(req.query.from_step);
    const downsample = Math.max(1, Number(req.query.downsample || 1) || 1);

    if (Number.isFinite(fromEpoch)) {
      rows = rows.filter((row) => typeof row.epoch === "number" && row.epoch >= fromEpoch);
    }
    if (Number.isFinite(fromStep)) {
      rows = rows.filter((row) => typeof row.global_step === "number" && row.global_step >= fromStep);
    }
    if (downsample > 1) {
      rows = rows.filter((_, idx) => idx % downsample === 0);
    }
    if (tail > 0 && rows.length > tail) {
      rows = rows.slice(rows.length - tail);
    }
    if (offset > 0 || limit > 0) {
      rows = limit > 0 ? rows.slice(offset, offset + limit) : rows.slice(offset);
    }
    return res.json({ path: rel, rows });
  } catch (err) {
    return res.status(500).json({ error: "failed to parse csv", details: String(err) });
  }
});

app.get("/api/runs/:runId/detail", (req, res) => {
  const runDir = path.join(RUNS_DIR, req.params.runId);
  if (!isSafePath(RUNS_DIR, runDir) || !fs.existsSync(runDir)) {
    return res.status(404).json({ error: "run not found" });
  }

  const manifest = parseJsonSafe(path.join(runDir, "manifest.json"), {});
  const benchmarkId = inferBenchmarkId(manifest);
  const family = inferModelFamily(manifest, benchmarkId);
  const rawFamily = manifest.model_family ?? "unknown";
  const includeRaw = String(req.query.include || "").trim();
  const includeSet = includeRaw
    ? new Set(includeRaw.split(",").map((v) => v.trim().toLowerCase()).filter(Boolean))
    : null;
  const include = (key) => !includeSet || includeSet.has(key);

  // Epoch metrics
  let epochRows = [];
  const epochPath = path.join(runDir, "learning", "epoch_metrics.csv");
  if (include("epoch") && fs.existsSync(epochPath)) {
    try { epochRows = readCsvRows(epochPath); } catch { /* skip */ }
  }

  // Batch metrics (for real-time training curves)
  let batchRows = [];
  const batchPath = path.join(runDir, "learning", "batch_metrics.csv");
  if (include("batch") && fs.existsSync(batchPath)) {
    try { batchRows = readCsvRows(batchPath); } catch { /* skip */ }
  }

  // Confusion matrix (latest epoch only)
  let confusionRows = [];
  const cmPath = path.join(runDir, "learning", "confusion_matrix.csv");
  if (include("confusion") && fs.existsSync(cmPath)) {
    try {
      const allCm = readCsvRows(cmPath);
      const testCm = allCm.filter((r) => r.split === "test");
      if (testCm.length) {
        const maxEpoch = Math.max(...testCm.map((r) => r.epoch).filter((e) => typeof e === "number"));
        confusionRows = testCm.filter((r) => r.epoch === maxEpoch);
      }
    } catch { /* skip */ }
  }

  // Model-specific CSVs
  const modelSpecific = {};
  if (include("model_specific")) {
    const modelSpecificCandidates = [...new Set([family, rawFamily].filter(Boolean))];
    const seenPaths = new Set();
    for (const msFamily of modelSpecificCandidates) {
      const modelDir = path.join(runDir, "model_specific", msFamily);
      if (!fs.existsSync(modelDir)) continue;
      const csvFiles = listCsvFilesRecursive(modelDir, `model_specific/${msFamily}`);
      for (const relPath of csvFiles) {
        if (seenPaths.has(relPath)) continue;
        seenPaths.add(relPath);
        const absPath = path.resolve(runDir, relPath);
        if (!isSafePath(runDir, absPath) || !fs.existsSync(absPath)) continue;
        // Skip bio files (already in training tab)
        if (relPath.includes("bio_epoch") || relPath.includes("bio_layer")) continue;
        if (relPath.includes("continuous_")) continue;
        try {
          const baseName = path.basename(relPath, ".csv");
          modelSpecific[baseName] = readCsvRows(absPath);
        } catch { /* skip */ }
      }
    }
  }

  // Calibration data
  let calibrationRows = [];
  const calPath = path.join(runDir, "deployment", "calibration_bins.csv");
  if (include("calibration") && fs.existsSync(calPath)) {
    try { calibrationRows = readCsvRows(calPath).filter((r) => r.split === "test"); } catch { /* skip */ }
  }

  // Inference metrics
  let inferenceRows = [];
  const inferPath = path.join(runDir, "deployment", "inference_metrics.csv");
  if (include("inference") && fs.existsSync(inferPath)) {
    try { inferenceRows = readCsvRows(inferPath).filter((r) => r.split === "test"); } catch { /* skip */ }
  }

  return res.json({
    manifest,
    family,
    epochRows,
    batchRows,
    confusionRows,
    modelSpecific,
    calibrationRows,
    inferenceRows
  });
});

app.get("/api/reports", (req, res) => {
  const runId = String(req.query.run_id || "").trim();
  const images = scanReportImages().filter((src) =>
    reportMatchesRun(src.replace(/^\/reports\//, ""), runId)
  );
  res.json({ images });
});

app.use("/reports", express.static(REPORTS_DIR, { maxAge: 0, etag: false }));

loadTaskStore();
refreshTaskLiveness();
const taskLivenessTimer = setInterval(refreshTaskLiveness, 15000);
if (typeof taskLivenessTimer.unref === "function") taskLivenessTimer.unref();

app.listen(PORT, () => {
  console.log(`Substrate API listening on http://localhost:${PORT}`);
});
