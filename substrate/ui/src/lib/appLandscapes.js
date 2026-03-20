export function makeLatencyHistogram(rows, bins = 24) {
  const values = (rows || [])
    .map((row) => row.latency_ms)
    .filter((value) => typeof value === "number" && Number.isFinite(value));

  if (!values.length) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = Math.max((max - min) / bins, 1e-9);
  const hist = Array.from({ length: bins }, (_, idx) => {
    const start = min + idx * width;
    const end = start + width;
    return {
      index: idx,
      label: `${start.toFixed(3)}-${end.toFixed(3)}`,
      count: 0
    };
  });

  for (const value of values) {
    const idx = Math.min(bins - 1, Math.floor((value - min) / width));
    hist[idx].count += 1;
  }

  return hist;
}

export function clampNumber(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

export function stableHash(text) {
  const s = String(text ?? "");
  let h = 2166136261;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function layerSortKey(layerName) {
  const layer = String(layerName || "");
  const match = layer.match(/(\d+)(?!.*\d)/);
  if (match) return Number(match[1]);
  return stableHash(layer) % 997;
}

export function buildBioTissueGraphData(layerRows, epochRows = []) {
  const usableRows = (layerRows || []).filter(
    (row) =>
      typeof row.epoch === "number" &&
      Number.isFinite(row.epoch) &&
      typeof row.layer === "string" &&
      row.layer.trim() &&
      row.layer !== "none"
  );
  if (!usableRows.length) return null;

  const latestEpoch =
    usableRows.length > 0
      ? Math.max(...usableRows.map((row) => row.epoch))
      : Math.max(...(epochRows || []).map((row) => row.epoch).filter(Number.isFinite));
  const layers = usableRows
    .filter((row) => row.epoch === latestEpoch)
    .sort((a, b) => layerSortKey(a.layer) - layerSortKey(b.layer));
  if (!layers.length) return null;

  const layerNodes = [];
  const nodes = [];
  for (let li = 0; li < layers.length; li += 1) {
    const layer = layers[li];
    const density = clampNumber(Number(layer.mask_density ?? 0.7), 0, 1);
    const myelin = clampNumber(Number(layer.myelin_fraction ?? 0), 0, 1);
    const sizeFactor = clampNumber(Number(layer.mean_abs_weight ?? 0), 0, 3);
    const count = clampNumber(Math.round(6 + density * 8 + myelin * 4), 6, 20);
    const phase = (stableHash(layer.layer) % 360) * (Math.PI / 180);
    const z = li * 3.2;
    const ring = [];

    for (let ni = 0; ni < count; ni += 1) {
      const theta = phase + (2 * Math.PI * ni) / count;
      const radialJitter = (((stableHash(`${layer.layer}:${ni}`) % 1000) / 1000) - 0.5) * 0.42;
      const radius = 1.8 + 0.6 * Math.sin(theta * 2.3) + radialJitter;
      const id = `${layer.layer}::${ni}`;
      const node = {
        id,
        layer: layer.layer,
        layerIndex: li,
        x: Math.cos(theta) * radius * (1 + sizeFactor * 0.08),
        y: Math.sin(theta) * radius * (1 + sizeFactor * 0.08),
        z,
        myelin,
        bioelectric: Number(layer.bioelectric_mean ?? 0),
        homeostasis: Number(layer.homeostasis_error ?? 0),
        gliaGain: Number(layer.glia_gain ?? 1),
        gliaPlasticity: Number(layer.glia_plasticity ?? 1),
        excitatoryFraction: Number(layer.excitatory_fraction ?? Number.NaN),
        signViolationFraction: Number(layer.sign_violation_fraction ?? Number.NaN)
      };
      nodes.push(node);
      ring.push(node);
    }
    layerNodes.push(ring);
  }

  const edgeMap = new Map();
  const addEdge = (src, dst, strength) => {
    const key = `${src.id}->${dst.id}`;
    if (edgeMap.has(key)) return;
    edgeMap.set(key, {
      key,
      srcId: src.id,
      dstId: dst.id,
      src,
      dst,
      myelin: clampNumber((src.myelin + dst.myelin) * 0.5, 0, 1),
      delay: clampNumber((Math.abs(src.z - dst.z) + 1) / (1 + 2.5 * ((src.myelin + dst.myelin) * 0.5)), 0.2, 6),
      strength
    });
  };

  for (let li = 0; li < layerNodes.length - 1; li += 1) {
    const srcNodes = layerNodes[li];
    const dstNodes = layerNodes[li + 1];
    const layer = layers[li];
    const density = clampNumber(Number(layer.mask_density ?? 0.7), 0, 1);
    const fanout = clampNumber(Math.round(2 + density * 5), 1, 9);
    for (let i = 0; i < srcNodes.length; i += 1) {
      const src = srcNodes[i];
      const offset = stableHash(`${src.id}:${layers[li + 1].layer}`) % dstNodes.length;
      for (let k = 0; k < fanout; k += 1) {
        const j = (offset + k * 3 + i) % dstNodes.length;
        const dst = dstNodes[j];
        const strength = clampNumber(
          0.15 + density * 0.65 + clampNumber(Number(layer.mean_abs_weight ?? 0), 0, 2) * 0.1,
          0.05,
          1.0
        );
        addEdge(src, dst, strength);
      }
    }

    // Sparse long-range skip tract.
    if (li + 2 < layerNodes.length && density > 0.35) {
      const skipDst = layerNodes[li + 2];
      const longFanout = clampNumber(Math.round(1 + density * 2), 1, 3);
      for (let i = 0; i < Math.min(srcNodes.length, 10); i += 1) {
        const src = srcNodes[i];
        const offset = stableHash(`${src.id}:skip`) % skipDst.length;
        for (let k = 0; k < longFanout; k += 1) {
          const dst = skipDst[(offset + k * 5) % skipDst.length];
          addEdge(src, dst, clampNumber(0.08 + density * 0.35, 0.05, 0.6));
        }
      }
    }
  }

  const edges = [...edgeMap.values()];
  const connectionCount = {};
  for (const node of nodes) connectionCount[node.id] = 0;
  for (const edge of edges) {
    connectionCount[edge.srcId] += 1;
    connectionCount[edge.dstId] += 1;
  }

  return {
    epoch: latestEpoch,
    layers,
    nodes,
    edges,
    connectionCount,
    layerCount: layers.length,
    nodeCount: nodes.length,
    edgeCount: edges.length
  };
}

export function buildConfidenceOutcomeHistogram(rows, bins = 16) {
  const clean = (rows || []).filter(
    (row) => typeof row.confidence === "number" && Number.isFinite(row.confidence)
  );
  if (!clean.length) return [];
  const step = 1 / bins;
  const out = Array.from({ length: bins }, (_, idx) => ({
    bucket: idx,
    label: `${(idx * step).toFixed(2)}-${((idx + 1) * step).toFixed(2)}`,
    correct: 0,
    incorrect: 0
  }));
  for (const row of clean) {
    const conf = clampNumber(row.confidence, 0, 1);
    const idx = Math.min(bins - 1, Math.floor(conf / step));
    const correct = row.is_correct === 1 || row.is_correct === true;
    if (correct) out[idx].correct += 1;
    else out[idx].incorrect += 1;
  }
  return out.map((row) => ({
    ...row,
    total: row.correct + row.incorrect,
    error_rate: row.correct + row.incorrect > 0 ? row.incorrect / (row.correct + row.incorrect) : 0
  }));
}

export function splitSelectValue(splits, current, fallback = "test") {
  if (!splits.length) return "";
  if (current && splits.includes(current)) return current;
  if (splits.includes(fallback)) return fallback;
  return splits[0];
}

export function bestRunsByFamily(runs) {
  const best = new Map();
  for (const run of runs || []) {
    const family = run.model_family || "unknown";
    const score = Number.isFinite(run.latest_test_accuracy) ? run.latest_test_accuracy : -Infinity;
    const updated = typeof run.updated_utc === "string" ? run.updated_utc : "";
    const current = best.get(family);
    if (!current) {
      best.set(family, { run, score, updated });
      continue;
    }
    if (score > current.score) {
      best.set(family, { run, score, updated });
      continue;
    }
    if (score === current.score && updated > current.updated) {
      best.set(family, { run, score, updated });
    }
  }
  return [...best.entries()]
    .map(([family, payload]) => ({ family, run: payload.run }))
    .sort((a, b) => a.family.localeCompare(b.family));
}

export function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function firstFiniteValue(row, keys) {
  for (const key of keys) {
    const value = row?.[key];
    if (isFiniteNumber(value)) return value;
  }
  return null;
}

export function normalizeToSignedRange(values) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (!Number.isFinite(range) || range <= 1e-12) return values.map(() => 0);
  return values.map((value) => ((value - min) / range) * 2 - 1);
}

export function normalizeToUnitRange(values) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (!Number.isFinite(range) || range <= 1e-12) return values.map(() => 0.5);
  return values.map((value) => (value - min) / range);
}

export function rollingAverage(values, radius = 2) {
  if (!values.length) return [];
  const out = new Array(values.length).fill(0);
  for (let i = 0; i < values.length; i += 1) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(values.length - 1, i + radius); j += 1) {
      sum += values[j];
      count += 1;
    }
    out[i] = count ? sum / count : values[i];
  }
  return out;
}

export function downsampleByStride(rows, maxPoints = 900) {
  if (!rows.length || rows.length <= maxPoints) return rows;
  const stride = Math.max(1, Math.floor(rows.length / maxPoints));
  return rows.filter((_, index) => index % stride === 0 || index === rows.length - 1);
}

export function buildLandscapeFromSamples(samples, config = {}) {
  const sorted = [...samples]
    .filter((row) => isFiniteNumber(row.xRaw) && isFiniteNumber(row.zRaw))
    .sort((a, b) => a.xRaw - b.xRaw);

  if (sorted.length < 6) return null;

  const trimmed = downsampleByStride(sorted, config.maxPoints || 900).map((row, index) => ({
    ...row,
    order: index
  }));

  const xRaw = trimmed.map((row, index) => row.xRaw + index * 1e-7);
  const zRaw = trimmed.map((row) => row.zRaw);
  const yRaw = trimmed.map((row) => (isFiniteNumber(row.yRaw) ? row.yRaw : null));

  const xNorm = normalizeToSignedRange(xRaw);
  const zNorm = normalizeToUnitRange(zRaw);

  const usableY = yRaw.filter(isFiniteNumber);
  const yVar =
    usableY.length > 1 ? Math.max(...usableY) - Math.min(...usableY) : 0;

  let yNorm;
  if (usableY.length >= 4 && yVar > 1e-9) {
    const fallbackY = rollingAverage(
      zNorm.map((value, index) => {
        if (index === 0) return 0;
        const dx = xNorm[index] - xNorm[index - 1];
        if (Math.abs(dx) < 1e-9) return 0;
        return (zNorm[index] - zNorm[index - 1]) / dx;
      }),
      2
    );
    const mixed = yRaw.map((value, index) =>
      isFiniteNumber(value) ? value : fallbackY[index]
    );
    yNorm = normalizeToSignedRange(mixed);
  } else {
    const slope = rollingAverage(
      zNorm.map((value, index) => {
        if (index === 0) return 0;
        const dx = xNorm[index] - xNorm[index - 1];
        if (Math.abs(dx) < 1e-9) return 0;
        return (zNorm[index] - zNorm[index - 1]) / dx;
      }),
      2
    );
    yNorm = normalizeToSignedRange(slope);
  }

  const points = trimmed.map((row, index) => ({
    step: row.order,
    x: xNorm[index],
    y: yNorm[index],
    z: zNorm[index],
    source: row.source || "unknown"
  }));

  const gridSize = config.gridSize || 35;
  const xAxis = Array.from({ length: gridSize }, (_, index) => -1 + (2 * index) / (gridSize - 1));
  const yAxis = Array.from({ length: gridSize }, (_, index) => -1 + (2 * index) / (gridSize - 1));
  const sigmaX = config.sigmaX || 0.22;
  const sigmaY = config.sigmaY || 0.22;

  const zGrid = yAxis.map((gy) =>
    xAxis.map((gx) => {
      let weightedZ = 0;
      let totalW = 0;
      let nearest = Number.POSITIVE_INFINITY;
      let nearestZ = 0.5;

      for (const point of points) {
        const dx = (gx - point.x) / sigmaX;
        const dy = (gy - point.y) / sigmaY;
        const dist2 = dx * dx + dy * dy;
        const w = Math.exp(-0.5 * dist2);
        weightedZ += point.z * w;
        totalW += w;

        const directDist2 = (gx - point.x) ** 2 + (gy - point.y) ** 2;
        if (directDist2 < nearest) {
          nearest = directDist2;
          nearestZ = point.z;
        }
      }

      const bowl = 0.11 * (gx * gx + 0.78 * gy * gy);
      if (totalW < 1e-7) return nearestZ + bowl;
      return weightedZ / totalW + bowl * 0.18;
    })
  );

  const minima = [];
  for (let yi = 1; yi < gridSize - 1; yi += 1) {
    for (let xi = 1; xi < gridSize - 1; xi += 1) {
      const z = zGrid[yi][xi];
      let lowerThanNeighbors = true;
      for (let oy = -1; oy <= 1 && lowerThanNeighbors; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (ox === 0 && oy === 0) continue;
          if (zGrid[yi + oy][xi + ox] <= z) {
            lowerThanNeighbors = false;
            break;
          }
        }
      }
      if (!lowerThanNeighbors) continue;

      let nearestStep = 0;
      let nearestDist = Number.POSITIVE_INFINITY;
      for (const point of points) {
        const d2 = (point.x - xAxis[xi]) ** 2 + (point.y - yAxis[yi]) ** 2;
        if (d2 < nearestDist) {
          nearestDist = d2;
          nearestStep = point.step;
        }
      }

      minima.push({
        x: xAxis[xi],
        y: yAxis[yi],
        z,
        step: nearestStep,
        ix: xi,
        iy: yi
      });
    }
  }

  minima.sort((a, b) => a.z - b.z);
  const selectedMinima = [];
  for (const candidate of minima) {
    const separated = selectedMinima.every(
      (item) => (item.ix - candidate.ix) ** 2 + (item.iy - candidate.iy) ** 2 >= 16
    );
    if (!separated) continue;
    selectedMinima.push(candidate);
    if (selectedMinima.length >= 6) break;
  }

  const pathStride = Math.max(1, Math.floor(points.length / 220));
  const path = points.filter(
    (_, index) => index % pathStride === 0 || index === points.length - 1
  );

  return {
    xAxis,
    yAxis,
    zGrid,
    path,
    minima: selectedMinima,
    pointCount: points.length
  };
}

export function buildTrainingLandscape(epochRows, batchRows) {
  const lossKeys = ["loss", "q_loss_ema", "train_loss", "td_error_ema", "mse"];
  const yKeys = [
    "grad_norm_mean",
    "grad_norm",
    "param_norm_mean",
    "td_error_ema",
    "accuracy",
    "action_match_ema",
    "reward_ema",
    "lr"
  ];

  const batchSamples = (batchRows || [])
    .filter((row) => !row.split || row.split === "train_batch" || row.split === "train")
    .map((row) => ({
      xRaw: firstFiniteValue(row, ["global_step", "step", "batch", "epoch"]),
      yRaw: firstFiniteValue(row, yKeys),
      zRaw: firstFiniteValue(row, lossKeys),
      source: "batch"
    }))
    .filter((row) => isFiniteNumber(row.xRaw) && isFiniteNumber(row.zRaw));

  if (batchSamples.length >= 6) {
    return buildLandscapeFromSamples(batchSamples, { maxPoints: 1000, gridSize: 37 });
  }

  const epochSamples = (epochRows || [])
    .filter((row) => !row.split || row.split === "train")
    .map((row) => ({
      xRaw: firstFiniteValue(row, ["epoch", "step"]),
      yRaw: firstFiniteValue(row, [
        "accuracy",
        "grad_norm_mean",
        "param_norm_mean",
        "samples_per_sec",
        "f1_macro"
      ]),
      zRaw: firstFiniteValue(row, lossKeys),
      source: "epoch"
    }))
    .filter((row) => isFiniteNumber(row.xRaw) && isFiniteNumber(row.zRaw));

  return buildLandscapeFromSamples(epochSamples, { maxPoints: 400, gridSize: 35 });
}

export function buildEvaluationLandscape(epochRows, inferenceRows) {
  const testEpochSamples = (epochRows || [])
    .filter((row) => row.split === "test")
    .map((row) => {
      const acc = firstFiniteValue(row, ["accuracy", "f1_macro", "recall_macro", "precision_macro"]);
      const loss = firstFiniteValue(row, ["loss", "test_loss"]);
      return {
        xRaw: firstFiniteValue(row, ["epoch", "step"]),
        yRaw: acc,
        zRaw: isFiniteNumber(loss) ? loss : (isFiniteNumber(acc) ? 1 - acc : null),
        source: "test_epoch"
      };
    })
    .filter((row) => isFiniteNumber(row.xRaw) && isFiniteNumber(row.zRaw));

  if (testEpochSamples.length >= 6) {
    return buildLandscapeFromSamples(testEpochSamples, {
      maxPoints: 350,
      gridSize: 31,
      sigmaX: 0.3,
      sigmaY: 0.3
    });
  }

  const inferSamples = (inferenceRows || [])
    .map((row, index) => {
      const correct = row.is_correct === 1 || row.is_correct === true;
      const confidence = firstFiniteValue(row, ["confidence", "avg_conf"]);
      const entropy = firstFiniteValue(row, ["entropy", "top2_margin"]);
      const errorSignal =
        (correct ? 0 : 1) +
        (isFiniteNumber(entropy) ? Math.max(0, entropy) * 0.2 : 0);

      return {
        xRaw: firstFiniteValue(row, ["sample_id"]) ?? index,
        yRaw: confidence,
        zRaw: errorSignal,
        source: "inference"
      };
    })
    .filter((row) => isFiniteNumber(row.xRaw) && isFiniteNumber(row.zRaw));

  return buildLandscapeFromSamples(inferSamples, { maxPoints: 700, gridSize: 35 });
}

export function deriveFamilyCoverageFromRuns(runs) {
  const byFamily = new Map();
  for (const run of runs || []) {
    const family = run.model_family || "unknown";
    if (!byFamily.has(family)) {
      byFamily.set(family, {
        family,
        runs: 0,
        active_runs: 0,
        variants: new Set(),
        runs_with_learning: 0,
        runs_with_evaluation: 0,
        runs_with_model_specific: 0,
        model_specific_files: [],
        latest_run_id: run.run_id,
        latest_updated_utc: run.updated_utc || null,
        latest_model_specific_run_id: null,
        latest_model_specific_updated_utc: null
      });
    }

    const row = byFamily.get(family);
    row.runs += 1;
    if (run.active) row.active_runs += 1;
    row.variants.add(run.model_variant || "unknown");

    if (run.updated_utc && (!row.latest_updated_utc || run.updated_utc > row.latest_updated_utc)) {
      row.latest_updated_utc = run.updated_utc;
      row.latest_run_id = run.run_id;
    }
  }

  return [...byFamily.values()]
    .map((row) => ({
      ...row,
      variants: [...row.variants].sort()
    }))
    .sort((a, b) => a.family.localeCompare(b.family));
}

export function deriveBenchmarkCoverageFromRuns(runs) {
  const byBenchmark = new Map();
  for (const run of runs || []) {
    const benchmark_id = run.benchmark_id || "unknown";
    const benchmark_name = run.benchmark_name || benchmark_id;
    if (!byBenchmark.has(benchmark_id)) {
      byBenchmark.set(benchmark_id, {
        benchmark_id,
        benchmark_name,
        task_types: new Set(),
        runs: 0,
        active_runs: 0,
        families: new Set(),
        latest_run_id: null,
        latest_updated_utc: null
      });
    }

    const row = byBenchmark.get(benchmark_id);
    row.runs += 1;
    if (run.active) row.active_runs += 1;
    row.task_types.add(run.task_type || "unknown");
    row.families.add(run.model_family || "unknown");
    if (run.updated_utc && (!row.latest_updated_utc || run.updated_utc > row.latest_updated_utc)) {
      row.latest_updated_utc = run.updated_utc;
      row.latest_run_id = run.run_id;
    }
  }

  return [...byBenchmark.values()]
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
}

