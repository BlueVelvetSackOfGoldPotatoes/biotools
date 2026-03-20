import { formatPct } from "../core";
import { OddResultsPanel } from "../oddPanels";
import { BenchmarkRunsLibrary, DiscoveryBatchLibrary } from "../discoveryViews";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, FieldLabel, SectionTitle } from "../uiPrimitives";

export function ReportsSection({ vm, actions }) {
  const {
    selectedEvidenceSource,
    setSelectedEvidenceKey,
    setForm,
    selectedEvidenceBodyCells,
    activeOdd,
    activeOddSourceLabel,
    evidenceSources,
    benchmarkRuns,
    discoveryBatches,
    selectedTaskName
  } = vm;
  const { foregroundBenchmarkRun, foregroundDiscoveryBatch } = actions;

  return (
        <div className="stack">
          <div className="card">
            <div className="section-head">
              <SectionTitle
                title="Evidence Source"
                help="Pick which persisted run or discovery champion should drive the ODD panel and report links below."
              />
              <span className="meta-note">switch between active replay context and saved benchmark artifacts</span>
            </div>
            <label>
              <FieldLabel
                label="ODD source"
                help="Choose the summary whose ODD and benchmark files you want to inspect."
              />
              <select
                className="task-input"
                value={selectedEvidenceSource?.key || "active"}
                onChange={(event) => setSelectedEvidenceKey(event.target.value)}
              >
                {evidenceSources.map((source) => (
                  <option key={source.key} value={source.key}>
                    {source.label}
                  </option>
                ))}
              </select>
            </label>
            {selectedEvidenceSource ? (
              <div className="cellengine-library-preview">
                <strong>{selectedEvidenceSource.label}</strong>
                <span>{selectedEvidenceSource.note}</span>
              </div>
            ) : null}
            {selectedEvidenceSource?.summary ? (
              <div className="cellengine-telemetry-grid">
                <div>
                  <div className="cellengine-telemetry-label">solved</div>
                  <div className="cellengine-telemetry-value">{selectedEvidenceSource.summary?.solved ? "yes" : "no"}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">survival</div>
                  <div className="cellengine-telemetry-value">{formatPct(selectedEvidenceSource.summary?.cell_clean?.survival_ratio, 1)}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">success</div>
                  <div className="cellengine-telemetry-value">{formatPct(selectedEvidenceSource.summary?.cell_clean?.success_rate, 1)}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">body mode / cells</div>
                  <div className="cellengine-telemetry-value">{selectedEvidenceSource.summary?.body_mode || "n/a"} / {selectedEvidenceSource.summary?.champion_cell_count ?? "n/a"}</div>
                </div>
                <div>
                  <div className="cellengine-telemetry-label">RL baseline</div>
                  <div className="cellengine-telemetry-value">{String(selectedEvidenceSource.summary?.rl_algorithm_selected || "n/a").toUpperCase()}</div>
                </div>
              </div>
            ) : (
              <div className="empty">No persisted evidence source available yet.</div>
            )}
            <ActionGroups className="cellengine-evidence-actions">
              <ActionGroup title="Select">
                <ActionButton
                  type="button"
                  className="mini-btn"
                  disabled={!selectedEvidenceSource?.championGenomePath}
                  onClick={() => setForm((current) => ({ ...current, genome_path: selectedEvidenceSource?.championGenomePath || current.genome_path }))}
                  help="Select this evidence source's champion genome as the next replay input without launching it yet."
                >
                  Set as replay input
                </ActionButton>
              </ActionGroup>
              <ActionGroup title="Inspect">
                {selectedEvidenceSource?.files?.report_md ? (
                  <ActionLink className="mini-btn" href={selectedEvidenceSource.files.report_md} target="_blank" rel="noreferrer" help="Open the saved markdown report for this evidence source.">
                    Open report file
                  </ActionLink>
                ) : null}
                {selectedEvidenceSource?.files?.summary_json ? (
                  <ActionLink className="mini-btn" href={selectedEvidenceSource.files.summary_json} target="_blank" rel="noreferrer" help="Open the machine-readable summary JSON for this evidence source.">
                    Open summary file
                  </ActionLink>
                ) : null}
              </ActionGroup>
            </ActionGroups>
            {selectedEvidenceSource?.files ? (
              <div className="cellengine-file-grid">
                {selectedEvidenceSource.files.champion_genome_csv ? <a href={selectedEvidenceSource.files.champion_genome_csv} target="_blank" rel="noreferrer">champion_genome.csv</a> : null}
                {selectedEvidenceSource.files.clean_teacher_trial_context_json ? <a href={selectedEvidenceSource.files.clean_teacher_trial_context_json} target="_blank" rel="noreferrer">clean_teacher_trial_context.json</a> : null}
                {selectedEvidenceSource.files.clean_teacher_trial_pre_reset_cells_csv ? <a href={selectedEvidenceSource.files.clean_teacher_trial_pre_reset_cells_csv} target="_blank" rel="noreferrer">clean_teacher_trial_pre_reset_cells.csv</a> : null}
                {selectedEvidenceSource.files.clean_teacher_trial_post_reset_cells_csv ? <a href={selectedEvidenceSource.files.clean_teacher_trial_post_reset_cells_csv} target="_blank" rel="noreferrer">clean_teacher_trial_post_reset_cells.csv</a> : null}
                {selectedEvidenceSource.files.clean_autonomous_trial_context_json ? <a href={selectedEvidenceSource.files.clean_autonomous_trial_context_json} target="_blank" rel="noreferrer">clean_autonomous_trial_context.json</a> : null}
                {selectedEvidenceSource.files.clean_autonomous_trial_pre_reset_cells_csv ? <a href={selectedEvidenceSource.files.clean_autonomous_trial_pre_reset_cells_csv} target="_blank" rel="noreferrer">clean_autonomous_trial_pre_reset_cells.csv</a> : null}
                {selectedEvidenceSource.files.clean_autonomous_trial_post_reset_cells_csv ? <a href={selectedEvidenceSource.files.clean_autonomous_trial_post_reset_cells_csv} target="_blank" rel="noreferrer">clean_autonomous_trial_post_reset_cells.csv</a> : null}
                {selectedEvidenceSource.files.damaged_autonomous_trial_context_json ? <a href={selectedEvidenceSource.files.damaged_autonomous_trial_context_json} target="_blank" rel="noreferrer">damaged_autonomous_trial_context.json</a> : null}
                {selectedEvidenceSource.files.damaged_autonomous_trial_pre_reset_cells_csv ? <a href={selectedEvidenceSource.files.damaged_autonomous_trial_pre_reset_cells_csv} target="_blank" rel="noreferrer">damaged_autonomous_trial_pre_reset_cells.csv</a> : null}
                {selectedEvidenceSource.files.damaged_autonomous_trial_post_reset_cells_csv ? <a href={selectedEvidenceSource.files.damaged_autonomous_trial_post_reset_cells_csv} target="_blank" rel="noreferrer">damaged_autonomous_trial_post_reset_cells.csv</a> : null}
              </div>
            ) : null}
          </div>

          <OddResultsPanel
            odd={selectedEvidenceSource?.summary?.odd || activeOdd}
            sourceLabel={selectedEvidenceSource?.note || activeOddSourceLabel}
            bodyCells={selectedEvidenceBodyCells}
          />

          <div className="grid two">
            <BenchmarkRunsLibrary
              runs={benchmarkRuns}
              selectedTaskName={selectedTaskName}
              onUseGenome={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
              onSelectEvidence={setSelectedEvidenceKey}
              onForegroundRun={foregroundBenchmarkRun}
            />
            <DiscoveryBatchLibrary
              batches={discoveryBatches}
              selectedTaskName={selectedTaskName}
              onUseGenome={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
              onSelectEvidence={setSelectedEvidenceKey}
              onForegroundBatch={foregroundDiscoveryBatch}
            />
          </div>
        </div>
  );
}
