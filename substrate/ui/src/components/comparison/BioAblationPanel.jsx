import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { formatPct, formatSignedPct } from "../../lib/dashboardShared";

export default function BioAblationPanel({
  bioStudies,
  bioStudyId,
  bioModel,
  onBioStudyChange,
  onBioModelChange,
  bioSelectableModels,
  bioStudy,
  bioAblationLoading,
  bioAblationError,
  bioModelAnalysis,
  bioFeatureOrder,
  bioLeaveOneRows,
  bioMarginalRows,
  bioOverviewRows,
  bioTopCombos
}) {
  return (
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
  );
}
