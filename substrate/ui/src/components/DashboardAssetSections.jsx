import { Suspense } from "react";
import { runStatus, runStatusLabel } from "../lib/dashboardShared";

export default function DashboardAssetSections({
  tab,
  runTaskType,
  selectedRun,
  gameTraceRows,
  gameTraceQuery,
  GameReplayPanelLazy,
  architectureFamily,
  jumpToFamilyModelRun,
  modelFamilies,
  selectedModelFamilyCoverage,
  runId,
  setRunId,
  modelRunsForFamily,
  modelCsvPath,
  setModelCsvPath,
  modelFileList,
  classMetricsQuery,
  inferQuery,
  calQuery,
  classProfileRows,
  classRadarRows,
  confidenceOutcomeRows,
  entropyConfidenceScatterRows,
  calibrationChartRows,
  errorRows,
  InterpretabilitySuiteLazy,
  modelFilesQuery,
  modelPreviewLoading,
  modelPreviewMap,
  AutoCsvPreviewLazy,
  modelCsvQuery,
  CsvModelChartLazy,
  reportsQuery
}) {
  return (
    <>
      {tab === "game" && (
        runTaskType !== "control" ? (
          <section className="panel stack">
            <div className="empty">
              Game replay is available for control benchmarks. Current run task type: `{runTaskType || "unknown"}`.
            </div>
          </section>
        ) : (
          <Suspense fallback={<div className="empty">Loading game replay...</div>}>
            <GameReplayPanelLazy
              run={selectedRun}
              rows={gameTraceRows}
              isLoading={gameTraceQuery.isLoading}
              error={gameTraceQuery.error}
              onRefresh={() => gameTraceQuery.refresh()}
            />
          </Suspense>
        )
      )}

      {tab === "model" && (
        <section className="panel stack">
          <div className="row controls">
            <label>
              Model Family
              <select
                value={architectureFamily}
                onChange={(event) => jumpToFamilyModelRun(event.target.value)}
                disabled={!modelFamilies.length}
              >
                {modelFamilies.length ? (
                  modelFamilies.map((row) => (
                    <option key={row.family} value={row.family}>
                      {row.family}
                    </option>
                  ))
                ) : (
                  <option value="">n/a</option>
                )}
              </select>
            </label>

            <button
              className="mini-btn"
              type="button"
              onClick={() => {
                if (selectedModelFamilyCoverage) {
                  jumpToFamilyModelRun(selectedModelFamilyCoverage.family);
                }
              }}
              disabled={!selectedModelFamilyCoverage?.latest_model_specific_run_id}
            >
              Load Latest Family Run
            </button>

            <label>
              Family Run
              <select
                value={runId}
                onChange={(event) => setRunId(event.target.value)}
                disabled={!modelRunsForFamily.length}
              >
                {modelRunsForFamily.length ? (
                  modelRunsForFamily.map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {run.run_id}
                      {runStatus(run) === "training" || runStatus(run) === "starting"
                        ? ` (${runStatusLabel(runStatus(run))})`
                        : ""}
                    </option>
                  ))
                ) : (
                  <option value="">n/a</option>
                )}
              </select>
            </label>

            <label>
              Model CSV
              <select
                value={modelCsvPath}
                onChange={(event) => setModelCsvPath(event.target.value)}
                disabled={!modelFileList.length}
              >
                {modelFileList.map((file) => (
                  <option key={file} value={file}>
                    {file}
                  </option>
                ))}
              </select>
            </label>
            <span className="meta-note">
              {modelFileList.length} csv files in run {runId || "n/a"}. Family catalog: {(selectedModelFamilyCoverage?.model_specific_files || []).length} files.
            </span>
          </div>

          {selectedModelFamilyCoverage &&
            selectedModelFamilyCoverage.latest_model_specific_run_id &&
            runId !== selectedModelFamilyCoverage.latest_model_specific_run_id && (
              <div className="empty">
                Selected run is `{runId || "n/a"}`. Latest `{selectedModelFamilyCoverage.family}` run with
                model-specific files is `{selectedModelFamilyCoverage.latest_model_specific_run_id}`.
                {" "}
                <button
                  className="mini-btn"
                  type="button"
                  onClick={() => jumpToFamilyModelRun(selectedModelFamilyCoverage.family)}
                >
                  Switch
                </button>
              </div>
            )}

          <div>
            <h3>Classic Interpretability Suite</h3>
            {classMetricsQuery.isLoading || inferQuery.isLoading || calQuery.isLoading ? (
              <div className="empty">Loading interpretability metrics...</div>
            ) : (
              <Suspense fallback={<div className="empty">Loading interpretability suite...</div>}>
                <InterpretabilitySuiteLazy
                  classProfileRows={classProfileRows}
                  classRadarRows={classRadarRows}
                  confidenceOutcomeRows={confidenceOutcomeRows}
                  entropyConfidenceScatterRows={entropyConfidenceScatterRows}
                  calibrationChartRows={calibrationChartRows}
                  errorRows={errorRows}
                />
              </Suspense>
            )}
          </div>

          <div>
            <h3>Architecture Previews</h3>
            {modelFilesQuery.isLoading || modelPreviewLoading ? (
              <div className="empty">Loading model-specific previews...</div>
            ) : modelFileList.length ? (
              <div className="preview-grid">
                {modelFileList.map((filePath) => (
                  <Suspense key={filePath} fallback={<div className="empty">Loading preview...</div>}>
                    <AutoCsvPreviewLazy
                      filePath={filePath}
                      rows={modelPreviewMap[filePath]?.rows || []}
                      error={modelPreviewMap[filePath]?.error || null}
                      onOpenDetail={() => setModelCsvPath(filePath)}
                    />
                  </Suspense>
                ))}
              </div>
            ) : (
              <div className="empty">No model-specific CSV files found for this run.</div>
            )}
          </div>

          <div>
            <h3>Detailed CSV Explorer</h3>
            {modelFilesQuery.isLoading || modelCsvQuery.isLoading ? (
              <div className="empty">Loading selected CSV...</div>
            ) : modelFileList.length ? (
              <Suspense fallback={<div className="empty">Loading CSV explorer...</div>}>
                <CsvModelChartLazy rows={modelCsvQuery.data?.rows || []} />
              </Suspense>
            ) : (
              <div className="empty">No model-specific CSV files found for this run.</div>
            )}
          </div>
        </section>
      )}

      {tab === "reports" && (
        <section className="panel stack">
          <h3>Generated Plot Gallery</h3>
          {reportsQuery.isLoading ? (
            <div className="empty">Loading report images...</div>
          ) : (reportsQuery.data?.images || []).length ? (
            <>
              <div className="meta-note">{reportsQuery.data.images.length} images</div>
              <div className="gallery">
                {reportsQuery.data.images.map((src) => (
                  <figure key={src} className="shot">
                    <img src={src} alt={src} loading="lazy" />
                    <figcaption>{src.split("/").slice(-2).join("/")}</figcaption>
                  </figure>
                ))}
              </div>
            </>
          ) : (
            <div className="empty">
              No report images found for this run. Generate with `analytics/plots/...` scripts.
            </div>
          )}
        </section>
      )}
    </>
  );
}
