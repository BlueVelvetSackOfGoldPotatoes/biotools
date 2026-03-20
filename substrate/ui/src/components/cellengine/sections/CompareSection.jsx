import { DEFAULT_BODY_VIEW, GENOME_PREVIEW_STEP } from "../core";
import { SharedVisualizationLegend } from "../legends";
import { CellMetricMatrixAtlas, CellTissueMap, CellConnectivityGraph } from "../cellViews";
import { GenomeVariabilityMap } from "../genomeViews";
import { GenomeComparisonPanel, GenomeGalleryCard } from "../libraryViews";
import { ActionButton, ActionGroup, ActionGroups, SectionTitle } from "../uiPrimitives";

export function CompareSection({ vm, actions }) {
  const {
    genomeMetric,
    setGenomeMetric,
    bodyCells,
    atlasBodyView,
    genomeBodyView,
    matrixBodyView,
    graphBodyView,
    bodyInteractionProps,
    currentFrameCellRows,
    currentFrameEdges,
    cellMetric,
    setCellMetric,
    cellAtlasMode,
    setCellAtlasMode,
    selectedBodyCell,
    setSelectedCellId,
    connectivityEnabled,
    setConnectivityEnabled,
    edgeThreshold,
    setEdgeThreshold,
    connectivityMode,
    setConnectivityMode,
    functionalStateByCellId,
    compareItems,
    compareLeftBodyView,
    compareRightBodyView,
    compareOptions,
    compareGenomePaths,
    replayBusy,
    genomePreviewState,
    genomePreviewItems,
    queuedCount,
    queuedPassCount,
    queuedFailCount,
    queuedGenomePaths,
    genomeRunStatusByPath,
    galleryLimit,
    setGalleryLimit,
    taskSpecificGenomeOptions,
    setForm
  } = vm;
  const {
    runReplayFromGallery,
    toggleCompareGenome,
    toggleQueuedGenome,
    runQueuedGenomes,
    queueVisibleGenomes,
    clearQueuedGenomes,
    setCompareGenomeAt
  } = actions;

  return (
        <>
      <SharedVisualizationLegend
        title="Compare Legend"
        note="shared once for all compare maps below"
        genomeMetric={genomeMetric}
        showGenome
        showFunction
        showRoles
        showMatrix
        showGraph
      />
      <div className="grid two">
        <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Organism Atlas"
              help="This is the actual 29-cell body. By default it colors each cell by the dominant real-time function inferred from genome predisposition plus current state; raw metric mode is still available."
            />
            <span className="meta-note">the actual cell body, with live function, state, and connectivity front and center</span>
          </div>
          <CellTissueMap
            bodyCells={bodyCells}
            bodyView={atlasBodyView}
            interactionProps={bodyInteractionProps("atlas")}
            frameCellRows={currentFrameCellRows}
            frameEdges={currentFrameEdges}
            metricId={cellMetric}
            onMetricChange={setCellMetric}
            atlasMode={cellAtlasMode}
            onAtlasModeChange={setCellAtlasMode}
            selectedCellId={selectedBodyCell?.cell_id ?? null}
            onSelectCell={setSelectedCellId}
            connectivityEnabled={connectivityEnabled}
            onConnectivityEnabledChange={setConnectivityEnabled}
            edgeThreshold={edgeThreshold}
            onEdgeThresholdChange={setEdgeThreshold}
            connectivityMode={connectivityMode}
            onConnectivityModeChange={setConnectivityMode}
            functionalStateByCellId={functionalStateByCellId}
          />
        </div>

        <div className="card">
          <div className="section-head">
            <SectionTitle
              title="Genome Variability"
              help="All cells share one genome, but they express it differently depending on body role and position. This map shows that per-cell variation."
            />
            <span className="meta-note">shared genome, position-dependent expression variability across the tissue</span>
          </div>
          <GenomeVariabilityMap
            bodyCells={bodyCells}
            bodyView={genomeBodyView}
            interactionProps={bodyInteractionProps("genome")}
            metricId={genomeMetric}
            onMetricChange={setGenomeMetric}
            selectedCellId={selectedBodyCell?.cell_id ?? null}
            onSelectCell={setSelectedCellId}
          />
        </div>
      </div>

      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Per-Cell Metric Matrix"
            help="Each organism cell contains the full multivariate state as a fixed 3x3 micro-heatmap. This is the dense whole-organism view for comparing all cell states at once."
          />
          <span className="meta-note">all continuous cell metrics per cell, in the actual organism geometry</span>
        </div>
        <CellMetricMatrixAtlas
          bodyCells={bodyCells}
          bodyView={matrixBodyView}
          interactionProps={bodyInteractionProps("matrix")}
          frameCellRows={currentFrameCellRows}
          selectedCellId={selectedBodyCell?.cell_id ?? null}
          onSelectCell={setSelectedCellId}
        />
      </div>

      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Live Cell Graph"
            help="Graph view of the current replay frame: each node is one cell and each edge is a live coupling link."
          />
          <span className="meta-note">real-time graph: node = cell, edge = current coupling strength</span>
        </div>
        <CellConnectivityGraph
          bodyCells={bodyCells}
          bodyView={graphBodyView}
          interactionProps={bodyInteractionProps("graph")}
          frameCellRows={currentFrameCellRows}
          frameEdges={currentFrameEdges}
          selectedCellId={selectedBodyCell?.cell_id ?? null}
          onSelectCell={setSelectedCellId}
          functionalStateByCellId={functionalStateByCellId}
          connectivityEnabled={connectivityEnabled}
          edgeThreshold={edgeThreshold}
          connectivityMode={connectivityMode}
        />
      </div>

      <GenomeComparisonPanel
        leftItem={compareItems[0] || null}
        rightItem={compareItems[1] || null}
        leftBodyView={compareLeftBodyView}
        rightBodyView={compareRightBodyView}
        leftBodyInteractionProps={bodyInteractionProps("compareLeft")}
        rightBodyInteractionProps={bodyInteractionProps("compareRight")}
        metricId={genomeMetric}
        onMetricChange={setGenomeMetric}
        compareOptions={compareOptions}
        leftPath={compareGenomePaths[0] || ""}
        rightPath={compareGenomePaths[1] || ""}
        onSetComparePath={setCompareGenomeAt}
        onRunGenome={runReplayFromGallery}
        onUseGenome={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
        disabled={replayBusy}
      />

      <div className="card">
        <div className="section-head">
          <SectionTitle
            title="Genome Gallery"
            help="Browse many saved genomes at once. Each tile can run a replay directly, be queued for batch reruns, or be selected for static comparison."
          />
          <span className="meta-note">run any genome from here, or queue many and batch replay with PASS/FAIL badges</span>
        </div>
        {genomePreviewState.loading && !genomePreviewItems.length ? (
          <div className="empty">Loading genome previews...</div>
        ) : genomePreviewState.error ? (
          <div className="error"><div>{genomePreviewState.error}</div></div>
        ) : genomePreviewItems.length ? (
          <div className="stack">
            {genomePreviewState.loading ? (
              <div className="meta-note">Refreshing genome previews in the background...</div>
            ) : null}
            <ActionGroups className="cellengine-gallery-toolbar">
              <ActionGroup title="Run">
                <ActionButton
                  className="mini-btn"
                  type="button"
                  disabled={replayBusy || !queuedCount}
                  onClick={runQueuedGenomes}
                  help="Run every genome currently in the queue, one after another, and mark each result PASS or FAIL."
                >
                  Run queued genomes ({queuedCount})
                </ActionButton>
              </ActionGroup>
              <ActionGroup title="Select">
                <ActionButton
                  className="mini-btn"
                  type="button"
                  disabled={replayBusy || !genomePreviewItems.length}
                  onClick={queueVisibleGenomes}
                  help="Add all currently visible genome cards to the replay queue."
                >
                  Queue all visible
                </ActionButton>
                <ActionButton
                  className="mini-btn"
                  type="button"
                  disabled={replayBusy || !queuedCount}
                  onClick={clearQueuedGenomes}
                  help="Clear the current replay queue."
                >
                  Clear queue
                </ActionButton>
              </ActionGroup>
              <ActionGroup title="Inspect">
                <span className="meta-note">queued results: {queuedPassCount} pass · {queuedFailCount} fail</span>
              </ActionGroup>
            </ActionGroups>
            <div className="cellengine-genome-gallery">
              {genomePreviewItems.map((item) => (
                <GenomeGalleryCard
                  key={item.genome_path || item.label}
                  item={item}
                  bodyView={DEFAULT_BODY_VIEW}
                  metricId={genomeMetric}
                  isSelected={compareGenomePaths.includes(item.genome_path)}
                  isQueued={queuedGenomePaths.includes(item.genome_path)}
                  runStatus={genomeRunStatusByPath[item.genome_path] || null}
                  onToggleSelect={toggleCompareGenome}
                  onToggleQueue={toggleQueuedGenome}
                  onRunReplay={runReplayFromGallery}
                  onUseForReplay={(genomePath) => setForm((current) => ({ ...current, genome_path: genomePath }))}
                  disabled={replayBusy}
                />
              ))}
            </div>
            {taskSpecificGenomeOptions.length > galleryLimit ? (
              <div className="row controls">
                <button className="mini-btn" type="button" onClick={() => setGalleryLimit((value) => value + GENOME_PREVIEW_STEP)}>
                  show more genomes
                </button>
                <span className="meta-note">
                  showing {genomePreviewItems.length} of {taskSpecificGenomeOptions.length}
                </span>
              </div>
            ) : (
              <div className="meta-note">showing all {genomePreviewItems.length} available genome previews</div>
            )}
          </div>
        ) : (
          <div className="empty">No genome previews available yet.</div>
        )}
      </div>
        </>
  );
}
