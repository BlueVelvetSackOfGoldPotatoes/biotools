import {
  COMPACT_3D_BODY_VIEW,
  DEFAULT_BODY_VIEW,
  clamp,
  clamp01,
  degToRad,
  lerp,
  normalizeMetric,
  staticMetricColor,
  surfacePaletteColor
} from "./core";
export function replayTipTrail(frames, currentIndex, pivotX, poleLengthPx, sceneHeight, trailLength) {
  if (!frames.length) return [];
  const start = Math.max(0, currentIndex - trailLength);
  return frames.slice(start, currentIndex + 1).map((frame) => {
    const pivotY = sceneHeight - 152;
    const tipX = pivotX + poleLengthPx * Math.sin(frame.theta_rad || 0);
    const tipY = pivotY - poleLengthPx * Math.cos(frame.theta_rad || 0);
    return { tipX, tipY, damage: frame.damage_event === 1 || frame.damage_event === true };
  });
}

export function bodyCoord(cell, key) {
  const value = cell?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function bodyExtents(bodyCells) {
  if (!bodyCells?.length) {
    return {
      minX: 0,
      maxX: 0,
      minY: 0,
      maxY: 0,
      minZ: 0,
      maxZ: 0,
      spanX: 0,
      spanY: 0,
      spanZ: 0
    };
  }
  const xs = bodyCells.map((cell) => bodyCoord(cell, "x"));
  const ys = bodyCells.map((cell) => bodyCoord(cell, "y"));
  const zs = bodyCells.map((cell) => bodyCoord(cell, "z"));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  return {
    minX,
    maxX,
    minY,
    maxY,
    minZ,
    maxZ,
    spanX: Math.max(0, maxX - minX),
    spanY: Math.max(0, maxY - minY),
    spanZ: Math.max(0, maxZ - minZ)
  };
}

export function bodyHasDepth(bodyCells) {
  return bodyExtents(bodyCells).spanZ > 0;
}

export function bodyDepthCount(bodyCells) {
  const extents = bodyExtents(bodyCells);
  return extents.spanZ > 0 ? extents.spanZ + 1 : 1;
}

export function realizedBodyModeLabel(requestedBodyMode, bodyCells) {
  const requested = requestedBodyMode || "fixed2d";
  const hasDepth = bodyHasDepth(bodyCells);
  if (requested === "grown3d") {
    return hasDepth ? "grown3d" : "grown3d (flat realization)";
  }
  if (requested === "grown2d") {
    return "grown2d";
  }
  return "fixed2d";
}

export function normalizeBodyView(bodyView) {
  return {
    yaw: Number.isFinite(bodyView?.yaw) ? bodyView.yaw : DEFAULT_BODY_VIEW.yaw,
    pitch: clamp(Number.isFinite(bodyView?.pitch) ? bodyView.pitch : DEFAULT_BODY_VIEW.pitch, -80, 80),
    zoom: clamp(Number.isFinite(bodyView?.zoom) ? bodyView.zoom : DEFAULT_BODY_VIEW.zoom, 0.55, 2.4)
  };
}

export function compactPreviewBodyView(bodyCells, bodyView) {
  const view = normalizeBodyView(bodyView);
  if (!bodyHasDepth(bodyCells)) {
    return view;
  }
  const looksDefault =
    Math.abs(view.yaw - DEFAULT_BODY_VIEW.yaw) < 1e-6 &&
    Math.abs(view.pitch - DEFAULT_BODY_VIEW.pitch) < 1e-6 &&
    Math.abs(view.zoom - DEFAULT_BODY_VIEW.zoom) < 1e-6;
  if (looksDefault) {
    return COMPACT_3D_BODY_VIEW;
  }
  return {
    yaw: view.yaw,
    pitch: Math.max(view.pitch, COMPACT_3D_BODY_VIEW.pitch),
    zoom: Math.max(view.zoom, COMPACT_3D_BODY_VIEW.zoom)
  };
}

export function rotateBodyPoint(x, y, z, bodyView, center) {
  const view = normalizeBodyView(bodyView);
  const yaw = degToRad(view.yaw);
  const pitch = degToRad(view.pitch);
  const relX = x - center.x;
  const relY = y - center.y;
  const relZ = z - center.z;

  const yawX = relX * Math.cos(yaw) + relZ * Math.sin(yaw);
  const yawZ = -relX * Math.sin(yaw) + relZ * Math.cos(yaw);
  const pitchY = relY * Math.cos(pitch) - yawZ * Math.sin(pitch);
  const depth = relY * Math.sin(pitch) + yawZ * Math.cos(pitch);

  return {
    x: yawX,
    y: pitchY,
    depth
  };
}

export function sortProjectedCells(cells) {
  return [...(cells || [])].sort((left, right) => {
    const depthDelta = (left.plotDepthOrder ?? 0) - (right.plotDepthOrder ?? 0);
    if (Math.abs(depthDelta) > 1e-6) return depthDelta;
    const yDelta = (left.plotY ?? 0) - (right.plotY ?? 0);
    if (Math.abs(yDelta) > 1e-6) return yDelta;
    return (left.plotX ?? 0) - (right.plotX ?? 0);
  });
}

export function depthShadowStyle(depth, selected = false) {
  const d = clamp01(depth);
  const offsetY = lerp(1.1, 5.8, d);
  const blur = lerp(1.6, 6.8, d);
  const alpha = selected ? lerp(0.18, 0.34, d) : lerp(0.08, 0.22, d);
  return {
    filter: `drop-shadow(0px ${offsetY.toFixed(2)}px ${blur.toFixed(2)}px rgba(15, 23, 42, ${alpha.toFixed(3)}))`
  };
}

export function projectBodyLayout(bodyCells, pitch, options = {}) {
  const cells = bodyCells || [];
  const extents = bodyExtents(cells);
  const bodyView = extents.spanZ > 0 && options.camera ? normalizeBodyView(options.camera) : null;
  const depthSkew = !bodyView && extents.spanZ > 0 ? pitch * (options.depthSkewRatio ?? 0.62) : 0;
  const depthLift = !bodyView && extents.spanZ > 0 ? pitch * (options.depthLiftRatio ?? 0.34) : 0;
  const center = {
    x: (extents.minX + extents.maxX) / 2,
    y: (extents.minY + extents.maxY) / 2,
    z: (extents.minZ + extents.maxZ) / 2
  };
  const projected = cells.map((cell) => {
    let plotX;
    let plotY;
    let plotDepthOrder;
    if (bodyView) {
      const rotated = rotateBodyPoint(bodyCoord(cell, "x"), bodyCoord(cell, "y"), bodyCoord(cell, "z"), bodyView, center);
      plotX = rotated.x * pitch * bodyView.zoom;
      plotY = -rotated.y * pitch * bodyView.zoom;
      plotDepthOrder = rotated.depth;
    } else {
      plotX =
        (bodyCoord(cell, "x") - extents.minX) * pitch +
        (bodyCoord(cell, "z") - extents.minZ) * depthSkew;
      plotY =
        (extents.maxY - bodyCoord(cell, "y")) * pitch -
        (bodyCoord(cell, "z") - extents.minZ) * depthLift;
      plotDepthOrder = bodyCoord(cell, "z") - extents.minZ;
    }
    return {
      ...cell,
      plotX,
      plotY,
      plotDepthOrder
    };
  });
  const xs = projected.map((cell) => cell.plotX);
  const ys = projected.map((cell) => cell.plotY);
  const depths = projected.map((cell) => cell.plotDepthOrder);
  const minPlotX = xs.length ? Math.min(...xs) : 0;
  const maxPlotX = xs.length ? Math.max(...xs) : 0;
  const minPlotY = ys.length ? Math.min(...ys) : 0;
  const maxPlotY = ys.length ? Math.max(...ys) : 0;
  const minDepth = depths.length ? Math.min(...depths) : 0;
  const maxDepth = depths.length ? Math.max(...depths) : 1;
  const depthSpan = Math.max(1e-6, maxDepth - minDepth);
  const cellFootprint = pitch * (bodyView?.zoom || 1);

  return {
    cells: projected.map((cell) => ({
      ...cell,
      plotX: cell.plotX - minPlotX,
      plotY: cell.plotY - minPlotY,
      plotDepth: clamp01((cell.plotDepthOrder - minDepth) / depthSpan)
    })),
    extents,
    hasDepth: extents.spanZ > 0,
    width: (maxPlotX - minPlotX) + cellFootprint,
    height: (maxPlotY - minPlotY) + cellFootprint
  };
}

export function buildOrganismSurfaceModel(bodyCells, metricId, layerSelection = "aggregate") {
  const allCells = Array.isArray(bodyCells) ? bodyCells : [];
  if (!allCells.length) return null;
  const layers = [...new Set(allCells.map((cell) => bodyCoord(cell, "z")))].sort((left, right) => left - right);
  const cells = layerSelection === "aggregate"
    ? allCells
    : allCells.filter((cell) => bodyCoord(cell, "z") === layerSelection);
  if (!cells.length) return null;
  const xs = cells.map((cell) => bodyCoord(cell, "x"));
  const ys = cells.map((cell) => bodyCoord(cell, "y"));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const metricValues = cells.map((cell) => Number(cell?.[metricId])).filter((value) => Number.isFinite(value));
  const metricMin = metricValues.length ? Math.min(...metricValues) : 0;
  const metricMax = metricValues.length ? Math.max(...metricValues) : 1;
  const metricSpan = Math.max(1e-6, metricMax - metricMin);
  const metricMean = metricValues.length ? metricValues.reduce((sum, value) => sum + value, 0) / metricValues.length : 0;
  const gridSize = 18;
  const pad = 0.6;
  const gridX = Array.from({ length: gridSize }, (_, idx) => lerp(minX - pad, maxX + pad, idx / Math.max(1, gridSize - 1)));
  const gridY = Array.from({ length: gridSize }, (_, idx) => lerp(minY - pad, maxY + pad, idx / Math.max(1, gridSize - 1)));
  const sigma = Math.max(0.85, Math.max(maxX - minX, maxY - minY) / 4);
  const zRows = gridY.map((y) => gridX.map((x) => {
    let weightSum = 0;
    let valueSum = 0;
    for (const cell of cells) {
      const dx = x - bodyCoord(cell, "x");
      const dy = y - bodyCoord(cell, "y");
      const dist2 = dx * dx + dy * dy;
      const weight = Math.exp(-dist2 / (2 * sigma * sigma));
      const raw = Number(cell?.[metricId]);
      const normalized = Number.isFinite(raw) ? clamp01((raw - metricMin) / metricSpan) : 0.5;
      weightSum += weight;
      valueSum += normalized * weight;
    }
    return weightSum > 0 ? valueSum / weightSum : 0.5;
  }));
  const minima = [];
  for (let yi = 1; yi < gridSize - 1; yi += 1) {
    for (let xi = 1; xi < gridSize - 1; xi += 1) {
      const value = zRows[yi][xi];
      let isMinimum = true;
      for (let oy = -1; oy <= 1 && isMinimum; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (ox === 0 && oy === 0) continue;
          if (zRows[yi + oy][xi + ox] <= value) {
            isMinimum = false;
            break;
          }
        }
      }
      if (isMinimum) {
        minima.push({ x: gridX[xi], y: gridY[yi], z: value });
      }
    }
  }
  minima.sort((a, b) => a.z - b.z);
  return {
    gridX,
    gridY,
    zRows,
    minima: minima.slice(0, 4),
    metricId,
    layerSelection,
    layers,
    visibleCells: cells.map((cell) => {
      const raw = Number(cell?.[metricId]);
      return {
        cell_id: cell.cell_id,
        x: bodyCoord(cell, "x"),
        y: bodyCoord(cell, "y"),
        z: bodyCoord(cell, "z"),
        rawValue: Number.isFinite(raw) ? raw : null,
        normalizedValue: Number.isFinite(raw) ? clamp01((raw - metricMin) / metricSpan) : 0.5
      };
    }),
    metricMin,
    metricMax,
    metricMean
  };
}

export function formatBodyCellPosition(cell, includeDepth = false) {
  if (!cell) return "(n/a)";
  const x = bodyCoord(cell, "x");
  const y = bodyCoord(cell, "y");
  const z = bodyCoord(cell, "z");
  return includeDepth ? `(${x}, ${y}, ${z})` : `(${x}, ${y})`;
}
