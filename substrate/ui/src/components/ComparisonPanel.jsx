import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  FAMILY_COLORS,
  buildLogTicks,
  formatCompactNumber,
  formatPct,
  formatSignedPct,
  meanValue,
  medianValue,
  semValue,
  stdDev,
  trainingDurationSeconds
} from "../lib/dashboardShared";
import ComparisonTables from "./comparison/ComparisonTables";

export default function ComparisonPanel({
  runs,
  onOpenRun,
  onOpenFamily,
  bioStudies,
  bioStudyId,
  bioModel,
  onBioStudyChange,
  onBioModelChange,
  bioAblation,
  bioAblationLoading,
  bioAblationError
}) {
  const [epochAxisMode, setEpochAxisMode] = useState("auto");
  const benchmarkIds = useMemo(
    () =>
      [...new Set((runs || []).map((run) => String(run?.benchmark_id || "").toLowerCase()).filter(Boolean))].sort(),
    [runs]
  );
  const isTicTacToeOnly = benchmarkIds.length === 1 && benchmarkIds[0] === "tictactoe";
  const scoreTitle = isTicTacToeOnly ? "Non-Loss" : "Accuracy";
  const scoreLower = isTicTacToeOnly ? "non-loss" : "accuracy";
  const scoreAbbr = isTicTacToeOnly ? "non-loss" : "acc";
  const bioOverviewRows = bioAblation?.overview || [];
  const bioStudy = bioAblation?.study || bioStudies?.find((study) => study.study_id === bioStudyId) || null;
  const bioFeatureOrder = bioAblation?.feature_order || [];
  const bioModelAnalysis = bioAblation?.model_analysis || null;
  const bioSelectableModels = useMemo(() => {
    const fromStudy = bioStudy?.models || [];
    if (fromStudy.length) return fromStudy;
    const fromOverview = bioOverviewRows.map((row) => row.model).filter(Boolean);
    return [...new Set(fromOverview)].sort();
  }, [bioStudy, bioOverviewRows]);
  const bioLeaveOneRows = useMemo(
    () =>
      (bioModelAnalysis?.leave_one || []).map((row) => ({
        feature: row.feature_short,
        feature_name: row.feature_name,
        score: row.mean,
        n: row.n,
        delta_vs_all_on: row.delta_vs_all_on,
        delta_vs_baseline: row.delta_vs_baseline
      })),
    [bioModelAnalysis]
  );
  const bioMarginalRows = useMemo(
    () =>
      (bioModelAnalysis?.marginal || []).map((row) => ({
        feature: row.feature_short,
        feature_name: row.feature_name,
        delta: row.delta,
        on_mean: row.on_mean,
        off_mean: row.off_mean,
        on_n: row.on_n,
        off_n: row.off_n
      })),
    [bioModelAnalysis]
  );
  const bioTopCombos = useMemo(
    () =>
      (bioModelAnalysis?.combos || [])
        .filter((row) => !row.is_baseline)
        .slice(0, 12),
    [bioModelAnalysis]
  );

  const summary = useMemo(() => {
    const accuracyValues = runs
      .map((run) => run.latest_test_accuracy)
      .filter((value) => typeof value === "number" && Number.isFinite(value));
    let bestRun = null;
    for (const run of runs) {
      if (typeof run.latest_test_accuracy !== "number" || !Number.isFinite(run.latest_test_accuracy)) continue;
      if (!bestRun || run.latest_test_accuracy > bestRun.latest_test_accuracy) bestRun = run;
    }
    return {
      runs: runs.length,
      families: new Set(runs.map((run) => run.model_family || "unknown")).size,
      variants: new Set(runs.map((run) => `${run.model_family || "unknown"}:${run.model_variant || "unknown"}`)).size,
      activeRuns: runs.filter((run) => run.active).length,
      accuracyCoverage: runs.length ? accuracyValues.length / runs.length : null,
      meanAccuracy: meanValue(accuracyValues),
      bestRun
    };
  }, [runs]);

  const familyRows = useMemo(() => {
    const byFamily = new Map();
    for (const run of runs) {
      const family = run.model_family || "unknown";
      if (!byFamily.has(family)) {
        byFamily.set(family, {
          family,
          runs: 0,
          active_runs: 0,
          variants: new Set(),
          accuracy_values: [],
          epoch_values: [],
          duration_values: [],
          best_run_id: null,
          best_acc: null
        });
      }
      const row = byFamily.get(family);
      row.runs += 1;
      if (run.active) row.active_runs += 1;
      row.variants.add(run.model_variant || "unknown");

      if (typeof run.latest_test_accuracy === "number" && Number.isFinite(run.latest_test_accuracy)) {
        row.accuracy_values.push(run.latest_test_accuracy);
        if (row.best_acc == null || run.latest_test_accuracy > row.best_acc) {
          row.best_acc = run.latest_test_accuracy;
          row.best_run_id = run.run_id;
        }
      }
      if (typeof run.latest_test_epoch === "number" && Number.isFinite(run.latest_test_epoch)) {
        row.epoch_values.push(run.latest_test_epoch);
      }
      const durationSec = trainingDurationSeconds(run);
      if (typeof durationSec === "number" && Number.isFinite(durationSec)) {
        row.duration_values.push(durationSec);
      }
    }

    return [...byFamily.values()]
      .map((row) => {
        const accuracy_count = row.accuracy_values.length;
        const epoch_count = row.epoch_values.length;
        const train_count = row.duration_values.length;
        const acc_std = stdDev(row.accuracy_values);
        const epoch_std = stdDev(row.epoch_values);
        const train_std = stdDev(row.duration_values);
        return {
          family: row.family,
          runs: row.runs,
          active_runs: row.active_runs,
          done_runs: row.runs - row.active_runs,
          variants: row.variants.size,
          accuracy_count,
          accuracy_coverage: row.runs ? accuracy_count / row.runs : null,
          best_acc: row.best_acc,
          mean_acc: meanValue(row.accuracy_values),
          median_acc: medianValue(row.accuracy_values),
          acc_std,
          acc_sem: semValue(row.accuracy_values),
          mean_epoch: meanValue(row.epoch_values),
          epoch_std,
          epoch_sem: semValue(row.epoch_values),
          epoch_count,
          mean_train_s: meanValue(row.duration_values),
          train_std,
          train_sem: semValue(row.duration_values),
          train_count,
          best_run_id: row.best_run_id
        };
      })
      .sort((a, b) => {
        const aBest = a.best_acc ?? -1;
        const bBest = b.best_acc ?? -1;
        if (bBest !== aBest) return bBest - aBest;
        return a.family.localeCompare(b.family);
      });
  }, [runs]);

  const variantRows = useMemo(() => {
    const byVariant = new Map();
    for (const run of runs) {
      const family = run.model_family || "unknown";
      const variant = run.model_variant || "unknown";
      const key = `${family}\u241F${variant}`;
      if (!byVariant.has(key)) {
        byVariant.set(key, {
          family,
          variant,
          runs: 0,
          active_runs: 0,
          accuracy_values: [],
          epoch_values: [],
          duration_values: [],
          best_run_id: null,
          best_acc: null
        });
      }
      const row = byVariant.get(key);
      row.runs += 1;
      if (run.active) row.active_runs += 1;

      if (typeof run.latest_test_accuracy === "number" && Number.isFinite(run.latest_test_accuracy)) {
        row.accuracy_values.push(run.latest_test_accuracy);
        if (row.best_acc == null || run.latest_test_accuracy > row.best_acc) {
          row.best_acc = run.latest_test_accuracy;
          row.best_run_id = run.run_id;
        }
      }
      if (typeof run.latest_test_epoch === "number" && Number.isFinite(run.latest_test_epoch)) {
        row.epoch_values.push(run.latest_test_epoch);
      }
      const durationSec = trainingDurationSeconds(run);
      if (typeof durationSec === "number" && Number.isFinite(durationSec)) {
        row.duration_values.push(durationSec);
      }
    }

    return [...byVariant.values()]
      .map((row) => ({
        family: row.family,
        variant: row.variant,
        runs: row.runs,
        active_runs: row.active_runs,
        done_runs: row.runs - row.active_runs,
        accuracy_count: row.accuracy_values.length,
        best_acc: row.best_acc,
        mean_acc: meanValue(row.accuracy_values),
        acc_std: stdDev(row.accuracy_values),
        acc_sem: semValue(row.accuracy_values),
        mean_epoch: meanValue(row.epoch_values),
        epoch_std: stdDev(row.epoch_values),
        epoch_sem: semValue(row.epoch_values),
        mean_train_s: meanValue(row.duration_values),
        train_std: stdDev(row.duration_values),
        train_sem: semValue(row.duration_values),
        best_run_id: row.best_run_id
      }))
      .sort((a, b) => {
        const aBest = a.best_acc ?? -1;
        const bBest = b.best_acc ?? -1;
        if (bBest !== aBest) return bBest - aBest;
        if (b.runs !== a.runs) return b.runs - a.runs;
        return `${a.family}:${a.variant}`.localeCompare(`${b.family}:${b.variant}`);
      });
  }, [runs]);

  const familyAccuracyChartRows = useMemo(
    () => familyRows.filter((row) => row.best_acc != null).slice(0, 24),
    [familyRows]
  );

  const familyStatusChartRows = useMemo(
    () =>
      familyRows.map((row) => ({
        family: row.family,
        done_runs: row.done_runs,
        active_runs: row.active_runs,
        with_accuracy: row.accuracy_count,
        without_accuracy: row.runs - row.accuracy_count
      })),
    [familyRows]
  );

  const familyAccuracyUncertaintyRows = useMemo(
    () =>
      familyRows
        .filter((row) => row.mean_acc != null)
        .sort((a, b) => (b.mean_acc ?? -1) - (a.mean_acc ?? -1))
        .slice(0, 20)
        .map((row) => ({
          family: row.family,
          mean_acc: row.mean_acc,
          median_acc: row.median_acc,
          best_acc: row.best_acc,
          acc_sem: row.acc_sem ?? 0,
          acc_std: row.acc_std ?? 0
        })),
    [familyRows]
  );

  const familyEfficiencyUncertaintyRows = useMemo(
    () =>
      familyRows
        .filter((row) => row.mean_epoch != null || row.mean_train_s != null)
        .sort((a, b) => (a.mean_epoch ?? Number.POSITIVE_INFINITY) - (b.mean_epoch ?? Number.POSITIVE_INFINITY))
        .slice(0, 20)
        .map((row) => ({
          family: row.family,
          mean_epoch: row.mean_epoch,
          mean_train_min: row.mean_train_s != null ? row.mean_train_s / 60 : null,
          epoch_sem: row.epoch_sem ?? 0,
          train_sem_min: row.train_sem != null ? row.train_sem / 60 : 0
        })),
    [familyRows]
  );

  const variantAccuracyUncertaintyRows = useMemo(
    () =>
      variantRows
        .filter((row) => row.mean_acc != null)
        .sort((a, b) => (b.mean_acc ?? -1) - (a.mean_acc ?? -1))
        .slice(0, 24)
        .map((row) => ({
          label: `${row.family}:${row.variant}`,
          mean_acc: row.mean_acc,
          best_acc: row.best_acc,
          acc_sem: row.acc_sem ?? 0
        })),
    [variantRows]
  );

  const accuracyScatterByFamily = useMemo(() => {
    const grouped = new Map();
    for (const run of runs) {
      if (typeof run.latest_test_accuracy !== "number" || !Number.isFinite(run.latest_test_accuracy)) continue;
      if (typeof run.latest_test_epoch !== "number" || !Number.isFinite(run.latest_test_epoch)) continue;
      const family = run.model_family || "unknown";
      if (!grouped.has(family)) grouped.set(family, []);
      const epochRaw = run.latest_test_epoch;
      grouped.get(family).push({
        run_id: run.run_id,
        variant: run.model_variant || "unknown",
        epoch_raw: epochRaw,
        epoch_plot: Math.max(1, epochRaw),
        accuracy: run.latest_test_accuracy
      });
    }

    return [...grouped.entries()]
      .map(([family, rows]) => ({
        family,
        color: FAMILY_COLORS[family] || "#495057",
        rows
      }))
      .sort((a, b) => b.rows.length - a.rows.length);
  }, [runs]);

  const epochScatterMeta = useMemo(() => {
    const positiveEpochs = [];
    let clampedCount = 0;

    for (const group of accuracyScatterByFamily) {
      for (const row of group.rows) {
        const epoch = row.epoch_raw;
        if (typeof epoch !== "number" || !Number.isFinite(epoch)) continue;
        if (epoch > 0) positiveEpochs.push(epoch);
        else clampedCount += 1;
      }
    }

    if (!positiveEpochs.length) {
      return {
        minPositiveEpoch: null,
        maxPositiveEpoch: null,
        dynamicRange: null,
        canUseLog: false,
        recommendLog: false,
        logTicks: [],
        logDomainMin: 1,
        logDomainMax: 10,
        clampedCount
      };
    }

    const minPositiveEpoch = Math.min(...positiveEpochs);
    const maxPositiveEpoch = Math.max(...positiveEpochs);
    const dynamicRange = maxPositiveEpoch / Math.max(minPositiveEpoch, 1e-12);
    const canUseLog = maxPositiveEpoch > minPositiveEpoch && minPositiveEpoch > 0;
    const recommendLog = canUseLog && (dynamicRange >= 30 || maxPositiveEpoch >= 1000);
    const logDomainMin = 10 ** Math.floor(Math.log10(minPositiveEpoch));
    const logDomainMax = 10 ** Math.ceil(Math.log10(maxPositiveEpoch));

    return {
      minPositiveEpoch,
      maxPositiveEpoch,
      dynamicRange,
      canUseLog,
      recommendLog,
      logTicks: buildLogTicks(minPositiveEpoch, maxPositiveEpoch),
      logDomainMin,
      logDomainMax,
      clampedCount
    };
  }, [accuracyScatterByFamily]);

  const effectiveEpochAxisMode =
    epochAxisMode === "auto"
      ? epochScatterMeta.recommendLog
        ? "log"
        : "linear"
      : epochAxisMode;
  const useLogEpochAxis = effectiveEpochAxisMode === "log" && epochScatterMeta.canUseLog;
  const epochAxisKey = useLogEpochAxis ? "epoch_plot" : "epoch_raw";

  return (
    <section className="panel stack">
      <div className="grid cards">
        <div className="card">
          <h3>Corpus</h3>
          <div className="card-grid">
            <span>runs</span>
            <span>{summary.runs}</span>
            <span>families</span>
            <span>{summary.families}</span>
            <span>variants</span>
            <span>{summary.variants}</span>
            <span>active</span>
            <span>{summary.activeRuns}</span>
          </div>
        </div>
        <div className="card">
          <h3>{scoreTitle} Coverage</h3>
          <div className="card-grid">
            <span>coverage</span>
            <span>{formatPct(summary.accuracyCoverage)}</span>
            <span>mean latest {scoreLower}</span>
            <span>{formatPct(summary.meanAccuracy)}</span>
            <span>best run</span>
            <span>{summary.bestRun?.run_id || "n/a"}</span>
            <span>best {scoreAbbr}</span>
            <span>{formatPct(summary.bestRun?.latest_test_accuracy ?? null)}</span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <h3>Bio Feature Impact Vs Baseline (Per Architecture)</h3>
          <span className="meta-note">centralized ablation summary</span>
        </div>
        <div className="row controls">
          <label>
            study{" "}
            <select value={bioStudyId || ""} onChange={(event) => onBioStudyChange(event.target.value)}>
              {!bioStudies?.length && <option value="">no bio studies</option>}
              {(bioStudies || []).map((study) => (
                <option key={study.study_id} value={study.study_id}>
                  {study.study_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            architecture{" "}
            <select
              value={bioModel || ""}
              onChange={(event) => onBioModelChange(event.target.value)}
              disabled={!bioSelectableModels.length}
            >
              {!bioSelectableModels.length && <option value="">n/a</option>}
              {bioSelectableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
          {bioStudy && (
            <span className="meta-note">
              {bioStudy.combo_mode || "unknown"} / {bioStudy.combo_order || "default"} / {bioStudy.profile || "unknown"}{" "}
              · rows {bioStudy.scored_rows ?? 0}/{bioStudy.total_rows ?? 0}
            </span>
          )}
        </div>

        {bioAblationLoading ? (
          <div className="empty">Loading bio ablation comparison...</div>
        ) : bioAblationError ? (
          <div className="empty">{String(bioAblationError)}</div>
        ) : !bioStudy ? (
          <div className="empty">No bio ablation studies found in `reports/*/summary.csv` yet.</div>
        ) : (
          <>
            <div className="grid cards">
              <div className="card">
                <h3>Selected Architecture</h3>
                <div className="card-grid">
                  <span>model</span>
                  <span>{bioModelAnalysis?.model || "n/a"}</span>
                  <span>baseline</span>
                  <span>{formatPct(bioModelAnalysis?.baseline?.mean)}</span>
                  <span>all bio on</span>
                  <span>{formatPct(bioModelAnalysis?.all_on?.mean)}</span>
                  <span>all-on vs baseline</span>
                  <span>{formatSignedPct(bioModelAnalysis?.all_on?.delta_vs_baseline)}</span>
                  <span>best combo</span>
                  <span>{bioModelAnalysis?.best_combo?.combo_id || "n/a"}</span>
                  <span>best combo score</span>
                  <span>{formatPct(bioModelAnalysis?.best_combo?.mean)}</span>
                  <span>best vs baseline</span>
                  <span>{formatSignedPct(bioModelAnalysis?.best_combo?.delta_vs_baseline)}</span>
                </div>
              </div>
              <div className="card">
                <h3>Coverage</h3>
                <div className="card-grid">
                  <span>combos observed</span>
                  <span>{bioModelAnalysis?.counts?.combos_observed ?? 0}</span>
                  <span>scored rows</span>
                  <span>{bioModelAnalysis?.counts?.scored_rows ?? 0}</span>
                  <span>feature count</span>
                  <span>{bioFeatureOrder.length}</span>
                  <span>baseline samples</span>
                  <span>{bioModelAnalysis?.baseline?.n ?? 0}</span>
                  <span>all-on samples</span>
                  <span>{bioModelAnalysis?.all_on?.n ?? 0}</span>
                </div>
              </div>
            </div>

            <div className="grid two">
              <div className="card">
                <h3>Leave-One-Out Effect (vs all bio on)</h3>
                {bioLeaveOneRows.length ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={bioLeaveOneRows}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="feature" />
                      <YAxis />
                      <Tooltip formatter={(value) => formatSignedPct(value)} />
                      <ReferenceLine y={0} stroke="#495057" />
                      <Bar dataKey="delta_vs_all_on" fill="#d6336c" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty">No leave-one-out rows for this architecture.</div>
                )}
              </div>
              <div className="card">
                <h3>Marginal Feature Effect (on mean - off mean)</h3>
                {bioMarginalRows.length ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={bioMarginalRows}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="feature" />
                      <YAxis />
                      <Tooltip formatter={(value) => formatSignedPct(value)} />
                      <ReferenceLine y={0} stroke="#495057" />
                      <Bar dataKey="delta" fill="#0b7285" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty">No marginal rows for this architecture.</div>
                )}
              </div>
            </div>

            <div className="grid two">
              <div className="card">
                <h3>Architecture Overview In Study</h3>
                {bioOverviewRows.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>model</th>
                          <th>baseline</th>
                          <th>all-on</th>
                          <th>all-on delta</th>
                          <th>best combo</th>
                          <th>best score</th>
                          <th>best delta</th>
                          <th>combos</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bioOverviewRows.map((row) => (
                          <tr key={row.model}>
                            <td>
                              <button className="mini-btn" type="button" onClick={() => onBioModelChange(row.model)}>
                                {row.model}
                              </button>
                            </td>
                            <td>{formatPct(row.baseline_mean)}</td>
                            <td>{formatPct(row.all_on_mean)}</td>
                            <td>{formatSignedPct(row.all_on_delta_vs_baseline)}</td>
                            <td>{row.best_combo_id || "n/a"}</td>
                            <td>{formatPct(row.best_combo_mean)}</td>
                            <td>{formatSignedPct(row.best_combo_delta_vs_baseline)}</td>
                            <td>{row.combos_observed ?? 0}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty">No architecture-level ablation overview available.</div>
                )}
              </div>

              <div className="card">
                <h3>Top Bio Combos ({bioModelAnalysis?.model || "n/a"})</h3>
                {bioTopCombos.length ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>combo</th>
                          <th>bits</th>
                          <th>score</th>
                          <th>delta vs baseline</th>
                          <th>n</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bioTopCombos.map((row) => (
                          <tr key={row.combo_id}>
                            <td>{row.combo_id}</td>
                            <td>{row.bits || "n/a"}</td>
                            <td>{formatPct(row.mean)}</td>
                            <td>{formatSignedPct(row.delta_vs_baseline)}</td>
                            <td>{row.n ?? 0}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="empty">No combo rows yet for selected architecture.</div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Family {scoreTitle} Leaderboard</h3>
          {familyAccuracyChartRows.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={familyAccuracyChartRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="family" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="best_acc" name="best" fill="#0c8599" />
                <Bar dataKey="mean_acc" name="mean" fill="#2b8a3e" />
                <Bar dataKey="median_acc" name="median" fill="#a61e4d" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">No family {scoreLower} data yet.</div>
          )}
        </div>

        <div className="card">
          <h3>Run Status By Family</h3>
          {familyStatusChartRows.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={familyStatusChartRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="family" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="done_runs" stackId="status" fill="#0c8599" name="done" />
                <Bar dataKey="active_runs" stackId="status" fill="#a61e4d" name="active" />
                <Bar
                  dataKey="with_accuracy"
                  stackId="coverage"
                  fill="#2b8a3e"
                  name={isTicTacToeOnly ? "with non-loss" : "with acc"}
                />
                <Bar
                  dataKey="without_accuracy"
                  stackId="coverage"
                  fill="#adb5bd"
                  name={isTicTacToeOnly ? "without non-loss" : "without acc"}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">No status data yet.</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <h3>{scoreTitle} Vs Training Epoch (All Runs)</h3>
          <div className="row">
            <span className="meta-note">x-scale</span>
            <button className="mini-btn" type="button" onClick={() => setEpochAxisMode("auto")}>
              auto
            </button>
            <button
              className="mini-btn"
              type="button"
              onClick={() => setEpochAxisMode("log")}
              disabled={!epochScatterMeta.canUseLog}
            >
              log
            </button>
            <button className="mini-btn" type="button" onClick={() => setEpochAxisMode("linear")}>
              linear
            </button>
          </div>
        </div>
        <div className="meta-note">
          mode: {useLogEpochAxis ? "log" : "linear"}
          {epochScatterMeta.dynamicRange != null
            ? ` · epoch range: ${formatCompactNumber(epochScatterMeta.minPositiveEpoch)}-${formatCompactNumber(epochScatterMeta.maxPositiveEpoch)}`
            : ""}
          {epochScatterMeta.dynamicRange != null
            ? ` (${formatCompactNumber(epochScatterMeta.dynamicRange)}x)`
            : ""}
          {useLogEpochAxis && epochScatterMeta.clampedCount > 0
            ? ` · ${epochScatterMeta.clampedCount} non-positive epoch points clamped to 1`
            : ""}
        </div>
        {accuracyScatterByFamily.length ? (
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey={epochAxisKey}
                name="epoch"
                scale={useLogEpochAxis ? "log" : "linear"}
                domain={
                  useLogEpochAxis
                    ? [epochScatterMeta.logDomainMin, epochScatterMeta.logDomainMax]
                    : ["auto", "auto"]
                }
                ticks={useLogEpochAxis ? epochScatterMeta.logTicks : undefined}
                tickFormatter={(value) => formatCompactNumber(value)}
              />
              <YAxis type="number" dataKey="accuracy" name={scoreLower} domain={[0, 1]} />
              <Tooltip
                labelFormatter={(_label, payload) => {
                  const runId = payload?.[0]?.payload?.run_id;
                  const variant = payload?.[0]?.payload?.variant;
                  if (runId) return `run: ${runId}${variant ? ` [${variant}]` : ""}`;
                  return "";
                }}
                formatter={(value, name, item) => {
                  if (name === "epoch" || name === "epoch_plot" || name === "epoch_raw") {
                    const raw = item?.payload?.epoch_raw;
                    return [raw, "epoch"];
                  }
                  if (name === "accuracy" && typeof value === "number") {
                    return [value.toFixed(4), scoreLower];
                  }
                  return [value, name];
                }}
              />
              <Legend />
              {accuracyScatterByFamily.slice(0, 18).map((group) => (
                <Scatter key={group.family} name={group.family} data={group.rows} fill={group.color} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="empty">No run-level epoch+{scoreLower} pairs available.</div>
        )}
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Family {scoreTitle} Line (With Error Bars)</h3>
          {familyAccuracyUncertaintyRows.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={familyAccuracyUncertaintyRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="family" angle={-30} textAnchor="end" height={70} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="mean_acc" stroke="#0c8599" dot={{ r: 2 }} name="mean">
                  <ErrorBar dataKey="acc_sem" width={4} stroke="#0c8599" />
                </Line>
                <Line type="monotone" dataKey="median_acc" stroke="#2b8a3e" dot={false} name="median" />
                <Line type="monotone" dataKey="best_acc" stroke="#a61e4d" dot={false} name="best" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">Not enough family {scoreLower} rows for uncertainty plot.</div>
          )}
        </div>

        <div className="card">
          <h3>Family Efficiency Line (Epoch + Time Error Bars)</h3>
          {familyEfficiencyUncertaintyRows.length ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={familyEfficiencyUncertaintyRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="family" angle={-30} textAnchor="end" height={70} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="mean_epoch" stroke="#5c940d" dot={{ r: 2 }} name="mean epoch">
                  <ErrorBar dataKey="epoch_sem" width={4} stroke="#5c940d" />
                </Line>
                <Line yAxisId="right" type="monotone" dataKey="mean_train_min" stroke="#e67700" dot={{ r: 2 }} name="mean train (min)">
                  <ErrorBar dataKey="train_sem_min" width={4} stroke="#e67700" />
                </Line>
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty">No epoch/time rows with uncertainty available.</div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Top Variants Line (With Error Bars)</h3>
        {variantAccuracyUncertaintyRows.length ? (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={variantAccuracyUncertaintyRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" angle={-35} textAnchor="end" height={90} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="mean_acc" stroke="#364fc7" dot={{ r: 2 }} name="variant mean">
                <ErrorBar dataKey="acc_sem" width={4} stroke="#364fc7" />
              </Line>
              <Line type="monotone" dataKey="best_acc" stroke="#d6336c" dot={false} name="variant best" />
            </LineChart>
          </ResponsiveContainer>
          ) : (
          <div className="empty">No variant {scoreLower} rows for uncertainty line chart.</div>
        )}
      </div>

      <ComparisonTables
        familyRows={familyRows}
        variantRows={variantRows}
        scoreLower={scoreLower}
        scoreAbbr={scoreAbbr}
        onOpenFamily={onOpenFamily}
        onOpenRun={onOpenRun}
      />
    </section>
  );
}
