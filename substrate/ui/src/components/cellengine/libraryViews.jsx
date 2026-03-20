import { useEffect, useMemo, useState } from "react";
import { DEFAULT_BODY_VIEW, GENOME_METRICS, formatGenomeMetricLabel, formatNumber, formatPct, shortHash, taskLabel, taskMeta } from "./core";
import { bodyDepthCount, bodyHasDepth, realizedBodyModeLabel } from "./body";
import { GenomeMiniMap, GenomeVariabilityMap } from "./genomeViews";
import { OrganismSurfacePreview } from "./surfacePreviews";
import { ActionButton, ActionGroup, ActionGroups, ActionLink, FieldLabel, SectionTitle } from "./uiPrimitives";
export function GenomeGalleryCard({
  item,
  bodyView,
  metricId,
  isSelected,
  isQueued,
  runStatus,
  onToggleSelect,
  onToggleQueue,
  onRunReplay,
  onUseForReplay,
  disabled
}) {
  if (!item) return null;
  const statusLabel = runStatus?.state === "running"
    ? "RUNNING"
    : runStatus?.state === "done"
      ? (runStatus?.pass ? "PASS" : "FAIL")
      : runStatus?.state === "error"
        ? "FAIL"
        : "";
  const statusClass = runStatus?.state === "running"
    ? "running"
    : runStatus?.pass
      ? "pass"
      : runStatus?.state
        ? "fail"
        : "";
  return (
    <div
      className={`cellengine-genome-card${isSelected ? " selected" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onToggleSelect(item.genome_path)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggleSelect(item.genome_path);
        }
      }}
    >
      <div className="cellengine-genome-card-top">
        <strong>{item.label?.replace(/^reports\//, "") || item.genome_path}</strong>
        <span>{formatPct(item.summary?.cell_clean?.survival_ratio, 1)}</span>
      </div>
      {statusLabel ? (
        <div className={`cellengine-run-result-badge ${statusClass}`}>
          {statusLabel}
        </div>
      ) : null}
      {item.error ? (
        <div className="meta-note">{item.error}</div>
      ) : (
        <div className="cellengine-organism-card-visual">
          <GenomeMiniMap bodyCells={item.body_cells} bodyView={bodyView} metricId={metricId} selected={isSelected} />
          <OrganismSurfacePreview
            bodyCells={item.body_cells}
            metricId={metricId}
            title={`${item.label?.replace(/^reports\//, "") || "genome"} surface`}
            note="static genome/body field preview"
            summary={item.summary}
          />
        </div>
      )}
      <div className="cellengine-genome-card-meta">
        <span>success {formatPct(item.summary?.cell_clean?.success_rate, 1)}</span>
        <span>{item.summary?.rl_algorithm_selected || "cell only"}</span>
        <span>{bodyHasDepth(item.body_cells) ? `depth ${bodyDepthCount(item.body_cells)}` : "flat body"}</span>
      </div>
      <ActionGroups className="cellengine-genome-card-actions">
        <ActionGroup title="Run">
          <ActionButton
            type="button"
            className="refresh-btn"
            disabled={disabled}
            help="Run this saved genome now and load the resulting replay into the Replay section."
            onClick={(event) => {
              event.stopPropagation();
              onRunReplay(item.genome_path);
            }}
          >
            Run now
          </ActionButton>
        </ActionGroup>
        <ActionGroup title="Select">
          <ActionButton
            type="button"
            className="mini-btn"
            disabled={disabled}
            help="Make this genome the selected replay input without launching anything yet."
            onClick={(event) => {
              event.stopPropagation();
              onUseForReplay(item.genome_path);
            }}
          >
            Set as replay input
          </ActionButton>
          <ActionButton
            type="button"
            className={`mini-btn${isQueued ? " active" : ""}`}
            disabled={disabled}
            help={isQueued
              ? "This genome is already in the batch replay queue. Click again to remove it."
              : "Add this genome to the batch replay queue so you can run several saved genomes in sequence."}
            onClick={(event) => {
              event.stopPropagation();
              onToggleQueue(item.genome_path);
            }}
          >
            {isQueued ? "In queue" : "Add to queue"}
          </ActionButton>
        </ActionGroup>
      </ActionGroups>
    </div>
  );
}

export function GenomeComparisonPanel({
  leftItem,
  rightItem,
  leftBodyView,
  rightBodyView,
  leftBodyInteractionProps,
  rightBodyInteractionProps,
  metricId,
  onMetricChange,
  compareOptions,
  leftPath,
  rightPath,
  onSetComparePath,
  onRunGenome,
  onUseGenome,
  disabled
}) {
  const identicalGenome = Boolean(
    leftItem &&
    rightItem &&
    leftItem.content_hash &&
    rightItem.content_hash &&
    leftItem.content_hash === rightItem.content_hash
  );

  return (
    <div className="card">
      <div className="section-head">
        <SectionTitle
          title="Genome Compare Bench"
          help="Static side-by-side body view for up to two saved genomes. This compares their tissue layout and expression maps, not their time-varying replay dynamics."
        />
        <span className="meta-note">select up to two genomes from the gallery for static organism comparison</span>
      </div>
      <div className="row controls">
        <label>
          <FieldLabel
            label="Compare metric"
            help="Chooses which genome-derived program or body feature colors the two organism maps."
          />
          <select className="task-input" value={metricId} onChange={(event) => onMetricChange(event.target.value)}>
            {GENOME_METRICS.map((metric) => (
              <option key={metric.id} value={metric.id}>{formatGenomeMetricLabel(metric.id)}</option>
            ))}
          </select>
        </label>
        <span className="meta-note">These views are body/genome programs, not time-varying replay states.</span>
      </div>
      <div className="grid two">
        <label className="cellengine-library-field">
          <FieldLabel
            label="Genome A"
            help="Explicit left compare slot. Options with the same content hash as Genome B are disabled so the two panes stay meaningfully distinct."
          />
          <select className="task-input" value={leftPath || ""} onChange={(event) => onSetComparePath(0, event.target.value)}>
            <option value="">select genome</option>
            {compareOptions.map((item) => {
              const duplicateOfOther = rightItem && item.content_hash && rightItem.content_hash && item.content_hash === rightItem.content_hash;
              return (
                <option key={`compare-left-${item.genome_path}`} value={item.genome_path} disabled={duplicateOfOther}>
                  {item.label?.replace(/^reports\//, "")} · {shortHash(item.content_hash)}
                </option>
              );
            })}
          </select>
        </label>
        <label className="cellengine-library-field">
          <FieldLabel
            label="Genome B"
            help="Explicit right compare slot. Options with the same content hash as Genome A are disabled so the two panes stay meaningfully distinct."
          />
          <select className="task-input" value={rightPath || ""} onChange={(event) => onSetComparePath(1, event.target.value)}>
            <option value="">select genome</option>
            {compareOptions.map((item) => {
              const duplicateOfOther = leftItem && item.content_hash && leftItem.content_hash && item.content_hash === leftItem.content_hash;
              return (
                <option key={`compare-right-${item.genome_path}`} value={item.genome_path} disabled={duplicateOfOther}>
                  {item.label?.replace(/^reports\//, "")} · {shortHash(item.content_hash)}
                </option>
              );
            })}
          </select>
        </label>
      </div>
      {identicalGenome ? (
        <div className="error">
          <div>Genome A and Genome B have the same genome content hash. This compare view will be identical until you choose a distinct genome.</div>
        </div>
      ) : null}
      <div className="grid two">
        {[leftItem, rightItem].map((item, index) => (
          <div key={item?.genome_path || `empty-${index}`} className="card cellengine-compare-card">
            {item ? (
              <div className="stack">
                <div className="section-head">
                  <h3>{index === 0 ? "Genome A" : "Genome B"}</h3>
                  <span className="meta-note">{item.label?.replace(/^reports\//, "")}</span>
                </div>
                <div className="cellengine-compare-meta">
                  <span>hash {shortHash(item.content_hash)}</span>
                  <span>{item.summary_dir?.replace(/^reports\//, "") || "no summary dir"}</span>
                </div>
                <ActionGroups className="cellengine-compare-actions">
                  <ActionGroup title="Run">
                    <ActionButton
                      type="button"
                      className="refresh-btn"
                      disabled={disabled}
                      help="Run this saved genome now and open the resulting replay."
                      onClick={() => onRunGenome?.(item.genome_path)}
                    >
                      Run now
                    </ActionButton>
                  </ActionGroup>
                  <ActionGroup title="Select">
                    <ActionButton
                      type="button"
                      className="mini-btn"
                      disabled={disabled}
                      help="Select this genome as the next replay input without launching it yet."
                      onClick={() => onUseGenome?.(item.genome_path)}
                    >
                      Set as replay input
                    </ActionButton>
                  </ActionGroup>
                </ActionGroups>
                <div className="cellengine-organism-card-visual cellengine-organism-card-visual-wide">
                  <GenomeVariabilityMap
                    bodyCells={item.body_cells || []}
                    bodyView={index === 0 ? leftBodyView : rightBodyView}
                    interactionProps={index === 0 ? leftBodyInteractionProps : rightBodyInteractionProps}
                    metricId={metricId}
                    onMetricChange={onMetricChange}
                    selectedCellId={null}
                    onSelectCell={() => {}}
                    showControls={false}
                  />
                  <OrganismSurfacePreview
                    bodyCells={item.body_cells || []}
                    metricId={metricId}
                    title={`${index === 0 ? "Genome A" : "Genome B"} surface`}
                    note="static genome/body field preview"
                    summary={item.summary}
                  />
                </div>
                <div className="cellengine-telemetry-grid">
                  <div>
                    <div className="cellengine-telemetry-label">survival</div>
                    <div className="cellengine-telemetry-value">{formatPct(item.summary?.cell_clean?.survival_ratio, 1)}</div>
                  </div>
                  <div>
                    <div className="cellengine-telemetry-label">success rate</div>
                    <div className="cellengine-telemetry-value">{formatPct(item.summary?.cell_clean?.success_rate, 1)}</div>
                  </div>
                  <div>
                    <div className="cellengine-telemetry-label">damage survival</div>
                    <div className="cellengine-telemetry-value">{formatPct(item.summary?.cell_damaged?.survival_ratio, 1)}</div>
                  </div>
                  <div>
                    <div className="cellengine-telemetry-label">RL selected</div>
                    <div className="cellengine-telemetry-value">{item.summary?.rl_algorithm_selected || "n/a"}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty">Select a genome card below.</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ReplayLibrary({
  genomeOptions,
  recentReplays,
  onLoadReplay,
  onSelectGenome,
  onRunGenome,
  loadingReplay,
  runningReplay
}) {
  const [selectedGenomePath, setSelectedGenomePath] = useState(genomeOptions[0]?.genome_path || "");
  const [selectedReplayDir, setSelectedReplayDir] = useState(recentReplays[0]?.relative_output_dir || "");

  useEffect(() => {
    if (!genomeOptions.length) {
      setSelectedGenomePath("");
      return;
    }
    if (!genomeOptions.some((option) => option.genome_path === selectedGenomePath)) {
      setSelectedGenomePath(genomeOptions[0].genome_path);
    }
  }, [genomeOptions, selectedGenomePath]);

  useEffect(() => {
    if (!recentReplays.length) {
      setSelectedReplayDir("");
      return;
    }
    if (!recentReplays.some((replay) => replay.relative_output_dir === selectedReplayDir)) {
      setSelectedReplayDir(recentReplays[0].relative_output_dir);
    }
  }, [recentReplays, selectedReplayDir]);

  const selectedGenome = useMemo(
    () => genomeOptions.find((option) => option.genome_path === selectedGenomePath) || null,
    [genomeOptions, selectedGenomePath]
  );
  const selectedReplay = useMemo(
    () => recentReplays.find((replay) => replay.relative_output_dir === selectedReplayDir) || null,
    [recentReplays, selectedReplayDir]
  );
  const replayHasFullArtifacts = Boolean(
    selectedReplay?.has_body_cells &&
    selectedReplay?.has_cell_rows &&
    selectedReplay?.has_edges &&
    selectedReplay?.has_genome_variability
  );

  return (
    <div className="grid two cellengine-library-grid">
      <div className="card cellengine-library-card">
        <div className="section-head">
          <SectionTitle
            title="Genome Presets"
            help="Saved champion genomes discovered under reports/. Choose one here to point the replay launcher, atlas, and comparison tools at that genome."
          />
          <span className="meta-note">{genomeOptions.length} found</span>
        </div>
        {genomeOptions.length ? (
          <div className="cellengine-library-stack">
            <label className="cellengine-library-field">
              <FieldLabel
                label="Genome preset"
                help="This selects which saved champion genome to use the next time you launch a replay."
              />
              <select
                className="task-input cellengine-library-select"
                value={selectedGenomePath}
                onChange={(event) => setSelectedGenomePath(event.target.value)}
              >
                {genomeOptions.map((option) => (
                  <option key={option.genome_path} value={option.genome_path}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            {selectedGenome ? (
              <div className="cellengine-library-preview">
                <strong>{selectedGenome.label}</strong>
                <span>
                  {taskMeta(selectedGenome.task_name || "cartpole_balance").label} · {selectedGenome.task_primary_label || "survival"} {selectedGenome.task_primary != null ? formatNumber(selectedGenome.task_primary, 2) : formatPct(selectedGenome.cell_clean_survival, 1)} · success {formatPct(selectedGenome.cell_clean_success, 1)}
                </span>
                <span className="code-lite">{selectedGenome.genome_path}</span>
              </div>
            ) : null}
            <ActionGroups className="cellengine-library-actions">
              <ActionGroup title="Run">
                <ActionButton
                  type="button"
                  className="refresh-btn"
                  disabled={!selectedGenomePath || runningReplay}
                  onClick={() => onRunGenome?.(selectedGenomePath)}
                  help="Run the selected saved genome now and open the resulting replay."
                >
                  {runningReplay ? "Running..." : "Run now"}
                </ActionButton>
              </ActionGroup>
              <ActionGroup title="Select">
                <ActionButton
                  type="button"
                  className="mini-btn"
                  disabled={!selectedGenomePath}
                  onClick={() => onSelectGenome(selectedGenomePath)}
                  help="Select this saved genome as the next replay input without launching it yet."
                >
                  Set as replay input
                </ActionButton>
              </ActionGroup>
            </ActionGroups>
          </div>
        ) : (
          <div className="empty">No `champion_genome.csv` files found under `reports/`.</div>
        )}
      </div>

      <div className="card cellengine-library-card">
        <div className="section-head">
          <SectionTitle
            title="Recent Replays"
            help="Cached browser replays already rendered by the C++ backend. Loading one is instant because it reuses saved trace files."
          />
          <span className="meta-note">{recentReplays.length} cached</span>
        </div>
        {recentReplays.length ? (
          <div className="cellengine-library-stack">
            <label className="cellengine-library-field">
              <FieldLabel
                label="Replay cache"
                help="Each saved replay bundles trace files from a previous run. Load one here instead of launching a new backend process."
              />
              <select
                className="task-input cellengine-library-select"
                value={selectedReplayDir}
                onChange={(event) => setSelectedReplayDir(event.target.value)}
              >
                {recentReplays.map((replay) => (
                  <option key={replay.relative_output_dir} value={replay.relative_output_dir}>
                    {replay.relative_output_dir.replace(/^reports\//, "")}
                  </option>
                ))}
              </select>
            </label>
            {selectedReplay ? (
              <div className="cellengine-library-preview">
                <strong>{selectedReplay.relative_output_dir.replace(/^reports\//, "")}</strong>
                <span>
                  {taskMeta(selectedReplay.task_name || "cartpole_balance").label} · {selectedReplay.task_primary_label || "ticks"} {selectedReplay.task_primary != null ? formatNumber(selectedReplay.task_primary, 2) : selectedReplay.total_ticks ?? "n/a"}{selectedReplay.max_abs_theta_deg != null ? ` · max |theta| ${formatNumber(selectedReplay.max_abs_theta_deg, 2)}°` : ""}
                </span>
                <span>full cell viewer {replayHasFullArtifacts ? "ready" : "partial"}</span>
              </div>
            ) : null}
            <ActionGroups className="cellengine-library-actions">
              <ActionGroup title="Run">
                <ActionButton
                  type="button"
                  className="refresh-btn"
                  disabled={loadingReplay || !selectedReplayDir}
                  onClick={() => onLoadReplay(selectedReplayDir)}
                  help="Open this saved replay trace in the Replay section without launching a new backend run."
                >
                  {loadingReplay ? "Opening..." : "Open saved replay"}
                </ActionButton>
              </ActionGroup>
            </ActionGroups>
          </div>
        ) : (
          <div className="empty">No saved browser replays yet. Launch one above.</div>
        )}
      </div>
    </div>
  );
}
