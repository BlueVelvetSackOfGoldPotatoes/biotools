import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const UI_DIR = path.resolve(__dirname, '..');
const ROOT_DIR = path.resolve(UI_DIR, '..');
const REPORTS_DIR = path.join(ROOT_DIR, 'reports');
const API_BASE = process.env.CELLENGINE_API_BASE_URL || 'http://127.0.0.1:8787';
const UI_BASE = process.env.CELLENGINE_UI_BASE_URL || 'http://127.0.0.1:5173';
const POLL_MS = Number(process.env.CELLENGINE_BURNIN_POLL_MS || 5000);
const DISCOVERY_TIMEOUT_MS = Number(process.env.CELLENGINE_BURNIN_DISCOVERY_TIMEOUT_MS || 45 * 60 * 1000);
const RUN_TIMEOUT_MS = Number(process.env.CELLENGINE_BURNIN_RUN_TIMEOUT_MS || 45 * 60 * 1000);

function assert(condition, message, extra = null) {
  if (!condition) {
    const err = new Error(message);
    if (extra != null) err.extra = extra;
    throw err;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function requestJson(endpoint, { method = 'GET', body = null, expectedStatus = null } = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined
  });
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { raw: text };
  }
  if (expectedStatus != null) {
    assert(response.status === expectedStatus, `Unexpected status for ${endpoint}: ${response.status}`, payload);
  } else {
    assert(response.ok, `HTTP ${response.status} for ${endpoint}`, payload);
  }
  return payload;
}

async function runNodeScript(relPath, args = []) {
  return await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(UI_DIR, relPath), ...args], {
      cwd: UI_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
      process.stdout.write(String(chunk));
    });
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
      process.stderr.write(String(chunk));
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`Script ${relPath} exited with code ${code}\n${stderr || stdout}`));
    });
  });
}

function taskPreset(taskName) {
  switch (taskName) {
    case 'worm_drag_race':
      return {
        body_mode: 'grown2d',
        max_cells: 96,
        body_extent_x: 10,
        body_extent_y: 10,
        body_extent_z: 1,
        development_steps: 8,
        development_seed_half_width: 2,
        chemical_diffusion_steps: 2,
        development_growth_threshold: 0.54,
        chemical_diffusion_rate: 0.32,
        chemical_decay: 0.08,
        worm_goal_distance: 6.0,
        worm_max_backward: 1.5,
        replay: { task_param_a: 0.22, task_param_b: null }
      };
    case 'pong_return':
      return {
        body_mode: 'grown2d',
        max_cells: 96,
        body_extent_x: 8,
        body_extent_y: 12,
        body_extent_z: 1,
        development_steps: 8,
        development_seed_half_width: 2,
        chemical_diffusion_steps: 2,
        development_growth_threshold: 0.54,
        chemical_diffusion_rate: 0.32,
        chemical_decay: 0.08,
        pong_target_hits: 6,
        pong_ball_speed: 1.1,
        pong_paddle_half_height: 0.22,
        replay: { task_param_a: 0.55, task_param_b: 0.75 }
      };
    case 'mass_spring_balance':
      return {
        body_mode: 'grown3d',
        max_cells: 96,
        body_extent_x: 6,
        body_extent_y: 14,
        body_extent_z: 2,
        development_steps: 8,
        development_seed_half_width: 2,
        chemical_diffusion_steps: 2,
        development_growth_threshold: 0.54,
        chemical_diffusion_rate: 0.32,
        chemical_decay: 0.08,
        replay: { theta_deg: 8, task_param_a: null, task_param_b: null }
      };
    case 'cartpole_balance':
    default:
      return {
        body_mode: 'grown3d',
        max_cells: 96,
        body_extent_x: 6,
        body_extent_y: 14,
        body_extent_z: 2,
        development_steps: 8,
        development_seed_half_width: 2,
        chemical_diffusion_steps: 2,
        development_growth_threshold: 0.54,
        chemical_diffusion_rate: 0.32,
        chemical_decay: 0.08,
        replay: { theta_deg: 8, task_param_a: null, task_param_b: null }
      };
  }
}

function discoveryBody(taskName, seedBase) {
  const preset = taskPreset(taskName);
  return {
    task_name: taskName,
    num_runs: 2,
    seed_start: seedBase,
    seed_step: 1,
    population_size: 20,
    generations: 10,
    search_trials: 8,
    search_ticks: 120,
    final_trials: 16,
    final_ticks: 180,
    auto_attempts: 2,
    rl_algo: 'none',
    odd_enabled: true,
    odd_trials: 8,
    odd_ticks: 180,
    ...preset
  };
}

function fullBenchmarkBody(taskName, genomePath) {
  const preset = taskPreset(taskName);
  return {
    task_name: taskName,
    genome_path: genomePath,
    seed: 7000 + Math.floor(Math.random() * 100000),
    rl_algo: 'all',
    final_trials: 24,
    final_ticks: 220,
    damage_trials: 8,
    damage_ticks: 220,
    odd_enabled: true,
    odd_trials: 8,
    odd_ticks: 220,
    ...preset
  };
}

function replayBody(taskName, genomePath) {
  const preset = taskPreset(taskName);
  return {
    task_name: taskName,
    genome_path: genomePath,
    max_ticks: 220,
    damage_tick: null,
    seed: 9000 + Math.floor(Math.random() * 100000),
    rl_algo: 'a2c',
    theta_deg: preset.replay.theta_deg ?? null,
    task_param_a: preset.replay.task_param_a ?? null,
    task_param_b: preset.replay.task_param_b ?? null
  };
}

function relToAbsReport(relativeOutputDir) {
  const rel = String(relativeOutputDir || '').replace(/^reports\//, '');
  return path.join(REPORTS_DIR, rel);
}

function ensureFileExists(filePath, label) {
  assert(filePath && fs.existsSync(filePath), `Missing ${label}: ${filePath}`);
}

function bestRun(runs) {
  return [...runs].sort((a, b) => {
    const as = Number(a?.summary?.cell_clean?.success_rate ?? -1);
    const bs = Number(b?.summary?.cell_clean?.success_rate ?? -1);
    if (bs !== as) return bs - as;
    const av = Number(a?.summary?.cell_clean?.survival_ratio ?? -1);
    const bv = Number(b?.summary?.cell_clean?.survival_ratio ?? -1);
    if (bv !== av) return bv - av;
    const ap = Number(a?.summary?.cell_clean?.task_primary ?? -1);
    const bp = Number(b?.summary?.cell_clean?.task_primary ?? -1);
    return bp - ap;
  })[0] || null;
}

function validateSummary(taskName, summary, expectedBodyMode) {
  assert(summary && typeof summary === 'object', `Missing summary for ${taskName}`);
  assert(summary.task_name === taskName, `Task mismatch in summary for ${taskName}`, summary);
  assert(summary.body_mode === expectedBodyMode, `Body mode mismatch for ${taskName}`, summary);
  assert(Number.isFinite(Number(summary.champion_cell_count)) && Number(summary.champion_cell_count) > 0, `Invalid champion_cell_count for ${taskName}`, summary);
  assert(Number.isFinite(Number(summary.champion_body_depth)) && Number(summary.champion_body_depth) > 0, `Invalid champion_body_depth for ${taskName}`, summary);
  assert(summary.cell_clean && typeof summary.cell_clean === 'object', `Missing cell_clean block for ${taskName}`, summary);
  assert(Number.isFinite(Number(summary.cell_clean.success_rate)), `Missing success_rate for ${taskName}`, summary.cell_clean);
  assert(Number.isFinite(Number(summary.cell_clean.survival_ratio)), `Missing survival_ratio for ${taskName}`, summary.cell_clean);
}

async function pollDiscovery(jobId, taskName) {
  const startedAt = Date.now();
  let previous = '';
  let sawLiveSnapshot = false;
  let sawLivePopulation = false;
  for (;;) {
    const payload = await requestJson(`/api/cellengine/discover/status?job_id=${encodeURIComponent(jobId)}`);
    const marker = `${payload.status}:${payload.completed_runs}:${payload.current_seed}`;
    if (marker !== previous) {
      console.log(`[${taskName}] discovery status=${payload.status} completed=${payload.completed_runs}/${payload.total_runs} current_seed=${payload.current_seed ?? '-'} batch=${payload.batch_id ?? '-'}`);
      previous = marker;
    }
    for (const run of payload.runs || []) {
      if (run.live_snapshot?.body_cells?.length) sawLiveSnapshot = true;
      if (run.live_population?.candidates?.length) sawLivePopulation = true;
      if (run.stdout_tail) {
        const tail = run.stdout_tail.trim().split('\n').slice(-3).join(' | ');
        if (tail) console.log(`[${taskName}] seed=${run.seed} tail=${tail}`);
      }
    }
    if (payload.status === 'completed') {
      assert(sawLiveSnapshot, `No live discovery snapshot surfaced for ${taskName}`, payload);
      assert(sawLivePopulation, `No live discovery population surfaced for ${taskName}`, payload);
      return payload;
    }
    if (payload.status === 'error') {
      throw new Error(`Discovery job failed for ${taskName}: ${payload.error || 'unknown error'}`);
    }
    if (Date.now() - startedAt > DISCOVERY_TIMEOUT_MS) {
      throw new Error(`Discovery job timed out for ${taskName}`);
    }
    await sleep(POLL_MS);
  }
}

async function validateDiscovery(taskName, discoveryStatus, expectedBodyMode) {
  assert(discoveryStatus.aggregate.completed_runs === discoveryStatus.total_runs, `Discovery aggregate incomplete for ${taskName}`, discoveryStatus.aggregate);
  assert(Array.isArray(discoveryStatus.runs) && discoveryStatus.runs.length === discoveryStatus.total_runs, `Run count mismatch for ${taskName}`, discoveryStatus);
  assert((discoveryStatus.aggregate.genes || []).length === 96, `Gene distribution missing from live discovery aggregate for ${taskName}`, discoveryStatus.aggregate);
  for (const run of discoveryStatus.runs) {
    const runDir = relToAbsReport(run.relative_output_dir);
    ensureFileExists(path.join(runDir, 'summary.json'), `${taskName} run summary.json`);
    ensureFileExists(path.join(runDir, 'champion_genome.csv'), `${taskName} champion_genome.csv`);
    ensureFileExists(path.join(runDir, 'live_population.json'), `${taskName} live_population.json`);
    ensureFileExists(path.join(runDir, 'live_meta.json'), `${taskName} live_meta.json`);
    validateSummary(taskName, run.summary, expectedBodyMode);
  }
  const persisted = await requestJson(`/api/cellengine/discover/batch?dir=${encodeURIComponent(discoveryStatus.relative_output_dir)}`);
  assert(persisted.job.status === 'completed', `Persisted batch not completed for ${taskName}`, persisted);
  assert((persisted.job.aggregate.genes || []).length === 96, `Persisted discovery gene aggregate missing for ${taskName}`, persisted.job.aggregate);
  const artifacts = await requestJson('/api/cellengine/artifacts');
  assert(
    (artifacts.discovery_batches || []).some((batch) => batch.relative_output_dir === discoveryStatus.relative_output_dir),
    `Discovery batch not indexed in artifacts for ${taskName}`,
    { relative_output_dir: discoveryStatus.relative_output_dir, artifacts }
  );
  return persisted;
}

async function runFullBenchmark(taskName, genomePath, expectedBodyMode) {
  const payload = await requestJson('/api/cellengine/full-benchmark', {
    method: 'POST',
    body: fullBenchmarkBody(taskName, genomePath),
    expectedStatus: 201
  });
  console.log(`[${taskName}] full benchmark dir=${payload.relative_output_dir}`);
  validateSummary(taskName, payload.summary, expectedBodyMode);
  const runDir = relToAbsReport(payload.relative_output_dir);
  ensureFileExists(path.join(runDir, 'summary.json'), `${taskName} full summary.json`);
  ensureFileExists(path.join(runDir, 'report.md'), `${taskName} full report.md`);
  ensureFileExists(path.join(runDir, 'champion_genome.csv'), `${taskName} full champion_genome.csv`);
  assert(payload.summary.odd && typeof payload.summary.odd === 'object', `ODD missing from full benchmark summary for ${taskName}`, payload.summary);
  const artifacts = await requestJson('/api/cellengine/artifacts');
  assert(
    (artifacts.benchmark_runs || []).some((run) => run.relative_output_dir === payload.relative_output_dir),
    `Full benchmark run not indexed in artifacts for ${taskName}`,
    { relative_output_dir: payload.relative_output_dir, artifacts }
  );
  return payload;
}

async function runReplay(taskName, genomePath, expectedBodyMode) {
  const payload = await requestJson('/api/cellengine/replay', {
    method: 'POST',
    body: replayBody(taskName, genomePath),
    expectedStatus: 201
  });
  console.log(`[${taskName}] replay dir=${payload.relative_output_dir}`);
  validateSummary(taskName, payload.summary, expectedBodyMode);
  assert(Array.isArray(payload.frames) && payload.frames.length > 0, `Replay frames missing for ${taskName}`, payload);
  assert(Array.isArray(payload.body_cells) && payload.body_cells.length > 0, `Replay body cells missing for ${taskName}`, payload);
  assert(Array.isArray(payload.cell_rows) && payload.cell_rows.length > 0, `Replay cell rows missing for ${taskName}`, payload);
  assert(Array.isArray(payload.edge_rows) && payload.edge_rows.length > 0, `Replay edge_rows missing for ${taskName}`, payload);
  const roundTrip = await requestJson(`/api/cellengine/replay?dir=${encodeURIComponent(payload.relative_output_dir)}`);
  assert(roundTrip.replay_id === payload.replay_id, `Replay round-trip mismatch for ${taskName}`, { payload, roundTrip });
  const runDir = relToAbsReport(payload.relative_output_dir);
  ensureFileExists(path.join(runDir, 'replay_trace.csv'), `${taskName} replay_trace.csv`);
  ensureFileExists(path.join(runDir, 'replay_body.csv'), `${taskName} replay_body.csv`);
  ensureFileExists(path.join(runDir, 'replay_cells.csv'), `${taskName} replay_cells.csv`);
  ensureFileExists(path.join(runDir, 'replay_edges.csv'), `${taskName} replay_edges.csv`);
  return payload;
}

async function validateOverviewAndArtifacts() {
  const overview = await requestJson('/api/cellengine/overview');
  const artifacts = await requestJson('/api/cellengine/artifacts');
  assert(Array.isArray(overview.available_tasks) && overview.available_tasks.length === 4, 'overview.available_tasks invalid', overview);
  assert(Array.isArray(overview.genome_options) && overview.genome_options.length > 0, 'overview.genome_options empty', overview);
  assert(Array.isArray(overview.recent_replays), 'overview.recent_replays invalid', overview);
  assert(Array.isArray(artifacts.benchmark_runs) && artifacts.benchmark_runs.length > 0, 'artifacts.benchmark_runs empty', artifacts);
  assert(Array.isArray(artifacts.discovery_batches) && artifacts.discovery_batches.length > 0, 'artifacts.discovery_batches empty', artifacts);
  const previewItems = overview.genome_options.slice(0, 2).map((item) => ({ genome_path: item.genome_path, label: item.label }));
  const previews = await requestJson('/api/cellengine/genome-previews', {
    method: 'POST',
    body: { items: previewItems },
    expectedStatus: 200
  });
  assert(Array.isArray(previews.items) && previews.items.length === previewItems.length, 'genome-previews response invalid', previews);
  for (const item of previews.items) {
    assert(!item.error, `Genome preview returned error for ${item.genome_path}`, item);
    assert(Array.isArray(item.body_cells) && item.body_cells.length > 0, `Genome preview missing body cells for ${item.genome_path}`, item);
  }
  return { overview, artifacts };
}

async function main() {
  console.log(`CellEngine burn-in starting against UI=${UI_BASE} API=${API_BASE}`);
  await runNodeScript('scripts/cellengine_runtime_harness.mjs');
  await validateOverviewAndArtifacts();

  const tasks = [
    ['cartpole_balance', 5201],
    ['mass_spring_balance', 6201],
    ['worm_drag_race', 7201],
    ['pong_return', 8201]
  ];
  const results = [];

  for (const [taskName, seedBase] of tasks) {
    const preset = taskPreset(taskName);
    console.log(`\n=== ${taskName} ===`);
    const active = await requestJson('/api/cellengine/discover/active');
    assert(active.job == null, `Unexpected active discovery job before starting ${taskName}`, active);
    const start = await requestJson('/api/cellengine/discover/start', {
      method: 'POST',
      body: discoveryBody(taskName, seedBase),
      expectedStatus: 201
    });
    assert(start.job_id, `Missing discovery job_id for ${taskName}`, start);
    const discoveryStatus = await pollDiscovery(start.job_id, taskName);
    const persisted = await validateDiscovery(taskName, discoveryStatus, preset.body_mode);
    const winner = bestRun(persisted.job.runs || []);
    assert(winner?.champion_genome_path, `Missing best champion genome for ${taskName}`, persisted.job.runs);
    const full = await runFullBenchmark(taskName, winner.champion_genome_path, preset.body_mode);
    const replay = await runReplay(taskName, winner.champion_genome_path, preset.body_mode);
    results.push({ taskName, discovery: discoveryStatus.relative_output_dir, full: full.relative_output_dir, replay: replay.relative_output_dir });
    await validateOverviewAndArtifacts();
    await runNodeScript('scripts/cellengine_runtime_harness.mjs');
  }

  console.log(JSON.stringify({ ok: true, results }, null, 2));
}

main().catch((err) => {
  console.error(JSON.stringify({ ok: false, error: String(err?.message || err), extra: err?.extra || null }, null, 2));
  process.exit(1);
});
