import { useEffect, useMemo, useState, lazy, Suspense } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { formatNumber, formatPct, taskLabel, clamp01, surfacePaletteColor } from "./core";
import { SectionTitle } from "./uiPrimitives";

const Plot3D = lazy(() => import("../Plot3D"));

function formatDuration(totalSeconds) {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "n/a";
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function computePopulationGeneMeans(candidates) {
  if (!Array.isArray(candidates) || !candidates.length) return [];
  return candidates.map((candidate) => {
    const cells = Array.isArray(candidate?.body_cells) ? candidate.body_cells : [];
    if (!cells.length) return null;
    const means = new Array(8).fill(0);
    for (const cell of cells) {
      for (let g = 0; g < 8; g++) {
        means[g] += Number(cell[`gene_expr_${g}`]) || 0;
      }
    }
    for (let g = 0; g < 8; g++) means[g] /= cells.length;
    const fitness = Number(candidate?.fitness);
    const summary = candidate?.summary;
    const success = Number(summary?.success_rate);
    const survival = Number(summary?.survival_ratio);
    return {
      means,
      fitness: Number.isFinite(fitness) ? fitness : 0,
      success: Number.isFinite(success) ? success : null,
      survival: Number.isFinite(survival) ? survival : null,
      cellCount: cells.length,
      rank: Number(candidate?.rank) || 0,
      bodyMode: candidate?.body_mode || "fixed2d"
    };
  }).filter(Boolean);
}

function pca3Axes(points) {
  if (points.length < 3) return null;
  const n = points.length;
  const dim = points[0].means.length;
  const mean = new Array(dim).fill(0);
  for (const p of points) {
    for (let d = 0; d < dim; d++) mean[d] += p.means[d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= n;

  const centered = points.map((p) => p.means.map((v, d) => v - mean[d]));

  const variance = new Array(dim).fill(0);
  for (const row of centered) {
    for (let d = 0; d < dim; d++) variance[d] += row[d] * row[d];
  }
  for (let d = 0; d < dim; d++) variance[d] /= n;

  const ranked = variance.map((v, i) => ({ index: i, variance: v })).sort((a, b) => b.variance - a.variance);
  const axes = ranked.slice(0, 3).map((r) => r.index);

  return points.map((p, i) => ({
    x: centered[i][axes[0]],
    y: centered[i][axes[1]],
    z: centered[i][axes[2]],
    fitness: p.fitness,
    success: p.success,
    survival: p.survival,
    cellCount: p.cellCount,
    rank: p.rank,
    bodyMode: p.bodyMode
  }));
}

function fitnessDistributionBins(candidates, binCount = 12) {
  if (!Array.isArray(candidates) || !candidates.length) return [];
  const values = candidates.map((c) => Number(c?.fitness)).filter(Number.isFinite);
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max - min < 1e-9) {
    return [{ label: formatNumber(min, 2), count: values.length, lo: min, hi: max }];
  }
  const bins = [];
  const step = (max - min) / binCount;
  for (let i = 0; i < binCount; i++) {
    const lo = min + i * step;
    const hi = lo + step;
    bins.push({ label: formatNumber(lo + step / 2, 2), count: 0, lo, hi });
  }
  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.floor((v - min) / step));
    bins[idx].count++;
  }
  return bins;
}

export function DiscoveryProgressPanel({
  discoveryJob,
  discoveryForm,
  discoveryHistoryByRun,
  selectedTaskName,
  busyStartedAt
}) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const runningRun = useMemo(() => {
    const runs = Array.isArray(discoveryJob?.runs) ? discoveryJob.runs : [];
    return runs.find((r) => r.status === "running" && r.live_snapshot?.meta) || runs.find((r) => r.live_snapshot?.meta) || null;
  }, [discoveryJob]);

  const liveMeta = runningRun?.live_snapshot?.meta || {};
  const livePopulation = runningRun?.live_population;
  const candidates = useMemo(
    () => Array.isArray(livePopulation?.candidates) ? livePopulation.candidates : [],
    [livePopulation]
  );

  const currentGen = Number(liveMeta.current_generation);
  const bestGen = Number(liveMeta.best_generation);
  const totalGens = Number(discoveryForm?.generations) || 50;
  const popSize = Number(discoveryForm?.population_size) || 48;
  const totalRuns = Number(discoveryJob?.total_runs) || 1;
  const completedRuns = Number(discoveryJob?.completed_runs) || 0;

  const hasGeneration = Number.isFinite(currentGen) && currentGen >= 0;
  const genProgress = hasGeneration ? Math.min(1, currentGen / Math.max(1, totalGens)) : 0;
  const runProgress = completedRuns / Math.max(1, totalRuns);
  const overallProgress = (completedRuns + genProgress) / Math.max(1, totalRuns);

  const jobStartMs = discoveryJob?.started_utc ? Date.parse(discoveryJob.started_utc) : busyStartedAt;
  // eslint-disable-next-line no-unused-vars
  const _tick = tick; // force recalc every second
  const elapsedSec = jobStartMs ? Math.max(0, (Date.now() - jobStartMs) / 1000) : 0;
  const etaSec = useMemo(() => {
    if (overallProgress < 0.01 || elapsedSec < 3) return null;
    const totalEstimate = elapsedSec / overallProgress;
    return Math.max(0, totalEstimate - elapsedSec);
  }, [overallProgress, elapsedSec, tick]);

  const totalOrganismsEvaluated = useMemo(() => {
    const fromCompletedRuns = completedRuns * totalGens * popSize;
    const fromCurrentRun = hasGeneration ? currentGen * popSize : 0;
    return fromCompletedRuns + fromCurrentRun;
  }, [completedRuns, totalGens, popSize, hasGeneration, currentGen]);

  const fitnessStats = useMemo(() => {
    if (!candidates.length) return null;
    const values = candidates.map((c) => Number(c?.fitness)).filter(Number.isFinite);
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return {
      best: sorted[sorted.length - 1],
      worst: sorted[0],
      mean: values.reduce((s, v) => s + v, 0) / values.length,
      median: sorted[Math.floor(sorted.length / 2)],
      std: Math.sqrt(values.reduce((s, v) => s + (v - values.reduce((ss, vv) => ss + vv, 0) / values.length) ** 2, 0) / values.length),
      count: values.length
    };
  }, [candidates]);

  const fitnessBins = useMemo(() => fitnessDistributionBins(candidates), [candidates]);

  const allHistory = useMemo(() => {
    if (!discoveryHistoryByRun) return [];
    const allRows = [];
    for (const entry of Object.values(discoveryHistoryByRun)) {
      if (Array.isArray(entry.rows)) {
        for (const row of entry.rows) {
          const existing = allRows.find((r) => r.generation === row.generation);
          if (existing) {
            if (Number.isFinite(row.best_fitness) && (!Number.isFinite(existing.best_fitness) || row.best_fitness > existing.best_fitness)) {
              existing.best_fitness = row.best_fitness;
            }
            if (Number.isFinite(row.mean_fitness) && Number.isFinite(existing.mean_fitness)) {
              existing.mean_fitness = (existing.mean_fitness + row.mean_fitness) / 2;
            }
            if (Number.isFinite(row.worst_fitness) && Number.isFinite(existing.worst_fitness)) {
              existing.worst_fitness = Math.min(existing.worst_fitness, row.worst_fitness);
            }
          } else {
            allRows.push({ ...row });
          }
        }
      }
    }
    return allRows.sort((a, b) => a.generation - b.generation);
  }, [discoveryHistoryByRun]);

  const geneMeans = useMemo(() => computePopulationGeneMeans(candidates), [candidates]);
  const pcaPoints = useMemo(() => pca3Axes(geneMeans), [geneMeans]);

  const bestScore = liveMeta.best_search_score;
  const bestSummary = liveMeta.best_summary || {};

  return (
    <div className="cellengine-progress-panel">
      <div className="section-head">
        <SectionTitle
          title="Evolution Progress"
          help="Live progress for the running discovery search. Shows generation progress, estimated time remaining, organism evaluation count, fitness distribution, and genome convergence landscape."
        />
        <span className="meta-note">
          {taskLabel(selectedTaskName)} · run {completedRuns + 1} of {totalRuns} · {hasGeneration ? `gen ${currentGen} / ${totalGens}` : "initializing"}
        </span>
      </div>

      <div className="cellengine-progress-bars">
        <div className="cellengine-progress-bar-row">
          <div className="cellengine-progress-bar-label">
            <span>Overall</span>
            <span>{Math.round(overallProgress * 100)}%</span>
          </div>
          <div className="cellengine-progress-bar-track">
            <div className="cellengine-progress-bar-fill overall" style={{ width: `${Math.round(overallProgress * 100)}%` }} />
          </div>
        </div>
        <div className="cellengine-progress-bar-row">
          <div className="cellengine-progress-bar-label">
            <span>Current run ({completedRuns + 1}/{totalRuns})</span>
            <span>{hasGeneration ? `gen ${currentGen}/${totalGens}` : "waiting"}</span>
          </div>
          <div className="cellengine-progress-bar-track">
            <div className="cellengine-progress-bar-fill current" style={{ width: `${Math.round(genProgress * 100)}%` }} />
          </div>
        </div>
      </div>

      <div className="cellengine-progress-stats">
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{formatDuration(elapsedSec)}</div>
          <div className="cellengine-progress-stat-label">elapsed</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{etaSec != null ? formatDuration(etaSec) : "calculating"}</div>
          <div className="cellengine-progress-stat-label">ETA</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{totalOrganismsEvaluated.toLocaleString()}</div>
          <div className="cellengine-progress-stat-label">organisms evaluated</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{Number.isFinite(bestScore) ? formatNumber(bestScore, 3) : "n/a"}</div>
          <div className="cellengine-progress-stat-label">best score</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{Number.isFinite(bestGen) ? bestGen : "n/a"}</div>
          <div className="cellengine-progress-stat-label">best gen</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{formatPct(bestSummary.success_rate, 1)}</div>
          <div className="cellengine-progress-stat-label">best success</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{formatPct(bestSummary.survival_ratio, 1)}</div>
          <div className="cellengine-progress-stat-label">best survival</div>
        </div>
        <div className="cellengine-progress-stat">
          <div className="cellengine-progress-stat-value">{fitnessStats ? formatNumber(fitnessStats.std, 3) : "n/a"}</div>
          <div className="cellengine-progress-stat-label">pop diversity (std)</div>
        </div>
      </div>

      <div className="cellengine-progress-charts">
        {allHistory.length > 1 ? (
          <div className="cellengine-progress-chart-card">
            <div className="section-head">
              <SectionTitle
                title="Fitness Over Generations"
                help="Best, mean, and worst fitness across the population at each generation. The shaded area between best and worst shows population spread. Narrowing spread indicates convergence."
              />
            </div>
            <div className="cellengine-progress-chart-container">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={allHistory} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                  <defs>
                    <linearGradient id="fitnessBand" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0f766e" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="#0f766e" stopOpacity={0.04} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#dce7ea" strokeDasharray="3 3" />
                  <XAxis dataKey="generation" allowDecimals={false} tickLine={false} axisLine={{ stroke: "#b6c7cf" }} />
                  <YAxis tickLine={false} axisLine={{ stroke: "#b6c7cf" }} width={56} />
                  <Tooltip
                    formatter={(value, name) => [formatNumber(value, 3), name]}
                    labelFormatter={(label) => `generation ${label}`}
                  />
                  <Legend />
                  <Area type="monotone" dataKey="worst_fitness" stackId="band" stroke="none" fill="url(#fitnessBand)" name="fitness range" />
                  <Line type="monotone" dataKey="best_fitness" name="best fitness" stroke="#0f766e" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="mean_fitness" name="mean fitness" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="worst_fitness" name="worst fitness" stroke="#c2410c" strokeWidth={1.5} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : null}

        {fitnessBins.length > 1 ? (
          <div className="cellengine-progress-chart-card">
            <div className="section-head">
              <SectionTitle
                title="Current Population Fitness Distribution"
                help="Histogram of fitness values for all candidates in the current generation. A tight cluster means the population is converging; a wide spread means diversity remains."
              />
              <span className="meta-note">{candidates.length} candidates</span>
            </div>
            <div className="cellengine-progress-chart-container">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={fitnessBins} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                  <CartesianGrid stroke="#dce7ea" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: "#b6c7cf" }} tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={{ stroke: "#b6c7cf" }} width={32} />
                  <Tooltip
                    formatter={(value) => [value, "organisms"]}
                    labelFormatter={(label) => `fitness ~ ${label}`}
                  />
                  <Bar dataKey="count" name="count" isAnimationActive={false}>
                    {fitnessBins.map((bin, index) => {
                      const t = fitnessBins.length > 1
                        ? index / (fitnessBins.length - 1)
                        : 0.5;
                      return <Cell key={`bin-${index}`} fill={surfacePaletteColor(t)} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : null}

        {pcaPoints && pcaPoints.length >= 3 ? (
          <div className="cellengine-progress-chart-card three-d">
            <div className="section-head">
              <SectionTitle
                title="Genome Convergence Landscape"
                help="3D projection of each organism's mean gene expression (8 gene programs averaged across cells) onto the top 3 axes of variance. Color maps to fitness. A tight cluster means the population is converging to a similar genome strategy; scattered points mean diversity. Hover for details."
              />
              <span className="meta-note">{pcaPoints.length} organisms · PCA of mean gene programs</span>
            </div>
            <div className="cellengine-progress-3d-container">
              <Suspense fallback={<div className="empty">Loading 3D landscape…</div>}>
                <Plot3D
                  data={[
                    {
                      type: "scatter3d",
                      mode: "markers",
                      x: pcaPoints.map((p) => p.x),
                      y: pcaPoints.map((p) => p.y),
                      z: pcaPoints.map((p) => p.z),
                      marker: {
                        size: pcaPoints.map((p) => {
                          const base = p.success != null ? p.success : clamp01(p.fitness / Math.max(1, Math.max(...pcaPoints.map((pp) => pp.fitness))));
                          return 4 + base * 8;
                        }),
                        color: pcaPoints.map((p) => p.fitness),
                        colorscale: [
                          [0, "#233b63"],
                          [0.25, "#335f8a"],
                          [0.5, "#4c8f64"],
                          [0.75, "#c97a27"],
                          [1, "#8d2e63"]
                        ],
                        showscale: true,
                        colorbar: { title: { text: "fitness" }, thickness: 14, len: 0.6 },
                        opacity: 0.88,
                        line: { color: "rgba(255,255,255,0.7)", width: 0.5 }
                      },
                      customdata: pcaPoints.map((p) => [
                        `rank ${p.rank}`,
                        `${p.cellCount} cells`,
                        p.bodyMode,
                        p.success != null ? `success ${(p.success * 100).toFixed(1)}%` : "success n/a",
                        p.survival != null ? `survival ${(p.survival * 100).toFixed(1)}%` : "survival n/a"
                      ]),
                      hovertemplate:
                        "%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<br>" +
                        "%{customdata[3]}<br>%{customdata[4]}<br>" +
                        "fitness %{marker.color:.3f}<br>" +
                        "PC1 %{x:.3f}<br>PC2 %{y:.3f}<br>PC3 %{z:.3f}<extra></extra>",
                      name: "population"
                    }
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    margin: { l: 0, r: 0, t: 10, b: 0 },
                    scene: {
                      bgcolor: "rgba(0,0,0,0)",
                      xaxis: { title: "PC1 (gene axis 1)" },
                      yaxis: { title: "PC2 (gene axis 2)" },
                      zaxis: { title: "PC3 (gene axis 3)" },
                      camera: { eye: { x: 1.6, y: -1.5, z: 0.8 } }
                    },
                    legend: { orientation: "h", x: 0, y: 1.02 }
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </Suspense>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
