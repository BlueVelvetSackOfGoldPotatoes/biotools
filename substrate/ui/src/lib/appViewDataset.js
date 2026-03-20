import { buildGroupedSeries, listDistinctNumbers, pivotEpoch, topMisclassifications } from "./dashboardShared";
import { buildBioTissueGraphData, buildConfidenceOutcomeHistogram, buildEvaluationLandscape, buildTrainingLandscape, makeLatencyHistogram } from "./appLandscapes";
import { linesForPivotMetric, pickSplitRows, sourceRows } from "./appViewMeta";

export function buildViewAllDataset(viewId, source) {
  if (!source) return { kind: "empty", data: [] };

  if (viewId === "train_landscape") {
    const landscape = buildTrainingLandscape(
      sourceRows(source, "learning/epoch_metrics.csv"),
      sourceRows(source, "learning/batch_metrics.csv")
    );
    return landscape ? { kind: "landscape", landscape } : { kind: "empty", data: [] };
  }

  if (viewId === "train_live_batch") {
    const rows = sourceRows(source, "learning/batch_metrics.csv");
    const data = rows
      .filter(
        (row) =>
          (row.split === "train_batch" || row.split === "train" || !row.split) &&
          typeof row.global_step === "number" &&
          Number.isFinite(row.global_step)
      )
      .sort((a, b) => a.global_step - b.global_step);
    return {
      kind: data.length ? "line_dual" : "empty",
      data,
      lines: [
        { key: "loss", label: "loss", color: "#a61e4d", yAxis: "left" },
        { key: "accuracy", label: "accuracy", color: "#2b8a3e", yAxis: "right" }
      ],
      xKey: "global_step",
      xType: "number",
      yRightDomain: [0, 1]
    };
  }

  if (
    viewId === "train_epoch_loss" ||
    viewId === "train_epoch_accuracy" ||
    viewId === "train_epoch_throughput"
  ) {
    const rows = sourceRows(source, "learning/epoch_metrics.csv");
    const pivot = pivotEpoch(rows);
    const metric =
      viewId === "train_epoch_loss"
        ? "loss"
        : viewId === "train_epoch_accuracy"
          ? "accuracy"
          : "samples_per_sec";
    const lines = linesForPivotMetric(pivot, metric);
    return {
      kind: lines.length ? "line_multi" : "empty",
      data: pivot,
      lines,
      xKey: "epoch",
      xType: "number",
      yDomain: viewId === "train_epoch_accuracy" ? [0, 1] : null
    };
  }

  if (viewId === "train_bio_energy") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "energy_used", label: "energy_used", color: "#a61e4d" },
        { key: "wiring_cost", label: "wiring_cost", color: "#0c8599" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_constraints") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "mask_density", label: "mask_density", color: "#2b8a3e" },
        { key: "myelin_fraction", label: "myelin_fraction", color: "#e67700" },
        { key: "mean_delay", label: "mean_delay", color: "#495057" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_regulation") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "homeostasis_error", label: "homeostasis_error", color: "#c2255c" },
        { key: "plasticity_update_mean_abs", label: "plasticity_update_mean_abs", color: "#0b7285" },
        { key: "modulator_mean", label: "modulator_mean", color: "#5c940d" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_structural") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "total_pruned", label: "total_pruned", color: "#c2255c" },
        { key: "total_grown", label: "total_grown", color: "#2b8a3e" },
        { key: "mask_density", label: "mask_density", color: "#0c8599" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_myelin_delay") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "myelin_fraction", label: "myelin_fraction", color: "#e67700" },
        { key: "mean_delay", label: "mean_delay", color: "#495057" },
        { key: "wiring_cost", label: "wiring_cost", color: "#0c8599" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_bioelectric") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "bioelectric_mean", label: "bioelectric_mean", color: "#d6336c" },
        { key: "modulator_mean", label: "modulator_mean", color: "#5c940d" },
        { key: "plasticity_update_mean_abs", label: "plasticity_update_mean_abs", color: "#0b7285" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_ei_sign") {
    const rows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv")
      .filter((row) => typeof row.epoch === "number" && Number.isFinite(row.epoch))
      .sort((a, b) => a.epoch - b.epoch);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "sign_violation_fraction", label: "sign_violation_fraction", color: "#c92a2a" },
        { key: "homeostasis_error", label: "homeostasis_error", color: "#1864ab" }
      ],
      xKey: "epoch",
      xType: "number"
    };
  }

  if (viewId === "train_bio_tissue_3d") {
    const layerRows = sourceRows(source, "model_specific/{family}/bio_layer_dynamics.csv");
    const epochRows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv");
    const graph = buildBioTissueGraphData(layerRows, epochRows);
    return graph ? { kind: "bio_tissue_3d", graph } : { kind: "empty", data: [] };
  }

  if (viewId === "train_layer_mask_density") {
    const grouped = buildGroupedSeries(
      sourceRows(source, "model_specific/{family}/bio_layer_dynamics.csv"),
      "epoch",
      "mask_density",
      "layer",
      MODEL_GROUP_SERIES_LIMIT
    );
    return {
      kind: grouped.data.length ? "line_multi" : "empty",
      data: grouped.data,
      lines: grouped.lines,
      xKey: "x",
      xType: "number"
    };
  }

  if (viewId === "train_continuous_accuracy") {
    const rows = sourceRows(source, "model_specific/{family}/continuous_efficiency.csv")
      .filter((row) => typeof row.cycle === "number" && Number.isFinite(row.cycle))
      .sort((a, b) => a.cycle - b.cycle);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "test_accuracy", label: "test_accuracy", color: "#0c8599" },
        { key: "best_accuracy", label: "best_accuracy", color: "#2b8a3e" }
      ],
      xKey: "cycle",
      xType: "number",
      yDomain: [0, 1]
    };
  }

  if (viewId === "train_continuous_samples") {
    const rows = sourceRows(source, "model_specific/{family}/continuous_efficiency.csv")
      .filter((row) => typeof row.cycle === "number" && Number.isFinite(row.cycle))
      .sort((a, b) => a.cycle - b.cycle);
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "cumulative_samples", label: "cumulative_samples", color: "#a61e4d" },
        { key: "cycle_time_ms", label: "cycle_time_ms", color: "#e67700" }
      ],
      xKey: "cycle",
      xType: "number"
    };
  }

  if (viewId === "train_phase_loss" || viewId === "train_phase_accuracy") {
    const grouped = buildGroupedSeries(
      sourceRows(source, "model_specific/{family}/continuous_phase_metrics.csv"),
      "cycle",
      viewId === "train_phase_loss" ? "loss" : "accuracy",
      "phase",
      MODEL_GROUP_SERIES_LIMIT
    );
    return {
      kind: grouped.data.length ? "line_multi" : "empty",
      data: grouped.data,
      lines: grouped.lines,
      xKey: "x",
      xType: "number",
      yDomain: viewId === "train_phase_accuracy" ? [0, 1] : null
    };
  }

  if (viewId === "eval_landscape") {
    const epochRows = sourceRows(source, "learning/epoch_metrics.csv");
    const inferRows = pickSplitRows(sourceRows(source, "deployment/inference_metrics.csv"), "test");
    const landscape = buildEvaluationLandscape(epochRows, inferRows);
    return landscape ? { kind: "landscape", landscape } : { kind: "empty", data: [] };
  }

  if (viewId === "eval_confusion_matrix") {
    const rows = sourceRows(source, "learning/confusion_matrix.csv");
    const splitRows = pickSplitRows(rows, "test");
    const epochs = listDistinctNumbers(splitRows, "epoch");
    const targetEpoch = epochs.length ? epochs[epochs.length - 1] : null;
    const subset =
      targetEpoch == null
        ? splitRows
        : splitRows.filter((row) => row.epoch === targetEpoch);
    const cm = confusionMatrix(subset);
    return cm.classes.length ? { kind: "matrix", ...cm } : { kind: "empty", data: [] };
  }

  if (viewId === "eval_calibration") {
    const rows = sourceRows(source, "deployment/calibration_bins.csv");
    const subset = pickSplitRows(rows, "test")
      .filter(
        (row) =>
          typeof row.conf_low === "number" &&
          typeof row.conf_high === "number" &&
          Number.isFinite(row.conf_low) &&
          Number.isFinite(row.conf_high)
      )
      .sort((a, b) => {
        const aKey = typeof a.bin_id === "number" ? a.bin_id : a.conf_low;
        const bKey = typeof b.bin_id === "number" ? b.bin_id : b.conf_low;
        return aKey - bKey;
      })
      .map((row) => ({
        center: (row.conf_low + row.conf_high) / 2,
        empirical_acc: row.empirical_acc,
        avg_conf: row.avg_conf
      }));
    return {
      kind: subset.length ? "line_multi" : "empty",
      data: subset,
      lines: [
        { key: "empirical_acc", label: "empirical_acc", color: "#2b8a3e" },
        { key: "avg_conf", label: "avg_conf", color: "#a61e4d" }
      ],
      xKey: "center",
      xType: "number",
      xDomain: [0, 1],
      yDomain: [0, 1],
      addDiag: true
    };
  }

  if (viewId === "eval_latency") {
    const subset = pickSplitRows(sourceRows(source, "deployment/inference_metrics.csv"), "test");
    const data = makeLatencyHistogram(subset, 20);
    return {
      kind: data.length ? "bar" : "empty",
      data,
      xKey: "label",
      barKey: "count",
      color: "#e67700",
      hideXAxis: true
    };
  }

  if (viewId === "eval_misclassifications") {
    const subset = pickSplitRows(sourceRows(source, "deployment/inference_metrics.csv"), "test");
    const data = topMisclassifications(subset);
    return {
      kind: data.length ? "bar" : "empty",
      data,
      xKey: "pair",
      barKey: "count",
      color: "#c2255c"
    };
  }

  if (viewId === "eval_system") {
    const rows = sourceRows(source, "deployment/system_metrics.csv");
    const subset = pickSplitRows(rows, "test").map((row, index) => ({
      index,
      qps: row.qps,
      p95_ms: row.p95_ms,
      p50_ms: row.p50_ms
    }));
    return {
      kind: subset.length ? "line_multi" : "empty",
      data: subset,
      lines: [
        { key: "qps", label: "qps", color: "#0c8599" },
        { key: "p95_ms", label: "p95_ms", color: "#a61e4d" },
        { key: "p50_ms", label: "p50_ms", color: "#5c940d" }
      ],
      xKey: "index",
      xType: "number"
    };
  }

  if (viewId === "eval_sample_efficiency") {
    const rows = sourceRows(source, "model_specific/{family}/continuous_efficiency.csv")
      .filter(
        (row) =>
          typeof row.cumulative_samples === "number" &&
          Number.isFinite(row.cumulative_samples) &&
          typeof row.test_accuracy === "number" &&
          Number.isFinite(row.test_accuracy)
      )
      .sort((a, b) => a.cumulative_samples - b.cumulative_samples)
      .map((row) => ({
        samples: row.cumulative_samples,
        test_accuracy: row.test_accuracy,
        best_accuracy: row.best_accuracy
      }));
    return {
      kind: rows.length ? "line_multi" : "empty",
      data: rows,
      lines: [
        { key: "test_accuracy", label: "test_accuracy", color: "#0c8599" },
        { key: "best_accuracy", label: "best_accuracy", color: "#2b8a3e" }
      ],
      xKey: "samples",
      xType: "number",
      yDomain: [0, 1]
    };
  }

  if (viewId === "eval_hybrid_expert") {
    const rows = sourceRows(source, "model_specific/{family}/hybrid_expert_metrics.csv");
    const cycles = listDistinctNumbers(rows, "cycle");
    if (!cycles.length) return { kind: "empty", data: [] };
    const targetCycle = cycles[cycles.length - 1];
    const data = rows
      .filter((row) => row.cycle === targetCycle)
      .map((row) => ({
        expert: String(row.expert ?? "expert"),
        fusion_strength: row.fusion_strength,
        expert_accuracy: row.expert_accuracy
      }));
    return {
      kind: data.length ? "bar_multi" : "empty",
      data,
      xKey: "expert",
      bars: [
        { key: "fusion_strength", label: "fusion_strength", color: "#a61e4d" },
        { key: "expert_accuracy", label: "expert_accuracy", color: "#0c8599" }
      ]
    };
  }

  if (viewId === "eval_bio_structure" || viewId === "eval_bio_state") {
    const rows = sourceRows(source, "model_specific/{family}/bio_layer_dynamics.csv");
    const epochs = listDistinctNumbers(rows, "epoch");
    if (!epochs.length) return { kind: "empty", data: [] };
    const targetEpoch = epochs[epochs.length - 1];
    const latestRows = rows
      .filter(
        (row) =>
          row.epoch === targetEpoch &&
          typeof row.layer === "string" &&
          row.layer.trim().length > 0 &&
          row.layer !== "none"
      )
      .slice(0, 24)
      .map((row) => ({
        layer: row.layer,
        mask_density: row.mask_density,
        myelin_fraction: row.myelin_fraction,
        mean_delay: row.mean_delay,
        bioelectric_mean: row.bioelectric_mean,
        homeostasis_error: row.homeostasis_error
      }));
    if (!latestRows.length) return { kind: "empty", data: [] };

    if (viewId === "eval_bio_structure") {
      return {
        kind: "bar_multi",
        data: latestRows,
        xKey: "layer",
        hideXAxis: true,
        bars: [
          { key: "mask_density", label: "mask_density", color: "#2b8a3e" },
          { key: "myelin_fraction", label: "myelin_fraction", color: "#e67700" }
        ]
      };
    }

    return {
      kind: "bar_multi",
      data: latestRows,
      xKey: "layer",
      hideXAxis: true,
      bars: [
        { key: "mean_delay", label: "mean_delay", color: "#0c8599" },
        { key: "bioelectric_mean", label: "bioelectric_mean", color: "#c2255c" },
        { key: "homeostasis_error", label: "homeostasis_error", color: "#5c940d" }
      ]
    };
  }

  if (viewId === "eval_bio_tissue_3d") {
    const layerRows = sourceRows(source, "model_specific/{family}/bio_layer_dynamics.csv");
    const epochRows = sourceRows(source, "model_specific/{family}/bio_epoch_dynamics.csv");
    const graph = buildBioTissueGraphData(layerRows, epochRows);
    return graph ? { kind: "bio_tissue_3d", graph } : { kind: "empty", data: [] };
  }

  if (viewId === "eval_bio_ei_sign") {
    const rows = sourceRows(source, "model_specific/{family}/bio_layer_dynamics.csv");
    const epochs = listDistinctNumbers(rows, "epoch");
    if (!epochs.length) return { kind: "empty", data: [] };
    const targetEpoch = epochs[epochs.length - 1];
    const latest = rows
      .filter((row) => row.epoch === targetEpoch && typeof row.layer === "string" && row.layer !== "none")
      .slice(0, 24)
      .map((row) => ({
        layer: row.layer,
        sign_violation_fraction: row.sign_violation_fraction,
        excitatory_fraction: row.excitatory_fraction,
        inhibitory_fraction: row.inhibitory_fraction
      }));
    return {
      kind: latest.length ? "bar_multi" : "empty",
      data: latest,
      xKey: "layer",
      hideXAxis: true,
      bars: [
        { key: "sign_violation_fraction", label: "sign_violation_fraction", color: "#c92a2a" },
        { key: "excitatory_fraction", label: "excitatory_fraction", color: "#2b8a3e" },
        { key: "inhibitory_fraction", label: "inhibitory_fraction", color: "#364fc7" }
      ]
    };
  }

  if (viewId === "eval_interp_class_profile") {
    const rows = sourceRows(source, "learning/class_metrics.csv");
    const splitRows = pickSplitRows(rows, "test");
    const epochs = listDistinctNumbers(splitRows, "epoch");
    if (!epochs.length) return { kind: "empty", data: [] };
    const targetEpoch = epochs[epochs.length - 1];
    const latest = splitRows
      .filter((row) => row.epoch === targetEpoch && typeof row.class_id === "number")
      .map((row) => ({
        class_id: row.class_id,
        f1: row.f1,
        precision: row.precision,
        recall: row.recall
      }))
      .sort((a, b) => a.class_id - b.class_id);
    return {
      kind: latest.length ? "line_multi" : "empty",
      data: latest,
      lines: [
        { key: "precision", label: "precision", color: "#0c8599" },
        { key: "recall", label: "recall", color: "#2b8a3e" },
        { key: "f1", label: "f1", color: "#a61e4d" }
      ],
      xKey: "class_id",
      xType: "number",
      yDomain: [0, 1]
    };
  }

  if (viewId === "eval_interp_confidence_error") {
    const infer = pickSplitRows(sourceRows(source, "deployment/inference_metrics.csv"), "test");
    const data = buildConfidenceOutcomeHistogram(infer, 16);
    return {
      kind: data.length ? "bar_multi" : "empty",
      data,
      xKey: "label",
      bars: [
        { key: "correct", label: "correct", color: "#2b8a3e" },
        { key: "incorrect", label: "incorrect", color: "#c92a2a" }
      ]
    };
  }

  return { kind: "empty", data: [] };
}

