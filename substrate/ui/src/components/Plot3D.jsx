import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_FOCAL = 2.8;
const DEFAULT_MARGIN = 26;
const DEFAULT_EYE = { x: 1.45, y: 1.35, z: 0.9 };
const DEFAULT_SCALE = [
  [0.0, "#16324f"],
  [0.25, "#236192"],
  [0.5, "#2b8a3e"],
  [0.75, "#e67700"],
  [1.0, "#a61e4d"]
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function norm(vec) {
  const mag = Math.hypot(vec.x, vec.y, vec.z) || 1;
  return { x: vec.x / mag, y: vec.y / mag, z: vec.z / mag };
}

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function axisTitle(axis) {
  if (!axis) return "";
  if (typeof axis.title === "string") return axis.title;
  if (axis.title && typeof axis.title.text === "string") return axis.title.text;
  return "";
}

function parseColor(color) {
  const value = String(color || "").trim();
  if (/^#[0-9a-f]{6}$/i.test(value)) {
    return {
      r: parseInt(value.slice(1, 3), 16),
      g: parseInt(value.slice(3, 5), 16),
      b: parseInt(value.slice(5, 7), 16),
      a: 1
    };
  }
  if (/^#[0-9a-f]{3}$/i.test(value)) {
    return {
      r: parseInt(value[1] + value[1], 16),
      g: parseInt(value[2] + value[2], 16),
      b: parseInt(value[3] + value[3], 16),
      a: 1
    };
  }
  const rgbaMatch = value.match(/^rgba?\(([^)]+)\)$/i);
  if (rgbaMatch) {
    const parts = rgbaMatch[1].split(",").map((part) => Number(part.trim()));
    return {
      r: clamp(parts[0] || 0, 0, 255),
      g: clamp(parts[1] || 0, 0, 255),
      b: clamp(parts[2] || 0, 0, 255),
      a: parts.length > 3 ? clamp(parts[3], 0, 1) : 1
    };
  }
  return { r: 86, g: 102, b: 119, a: 1 };
}

function rgbaString(color, alpha = color.a) {
  return `rgba(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)}, ${alpha})`;
}

function interpolateColor(scale, value) {
  const normalized = clamp(value, 0, 1);
  if (!Array.isArray(scale) || !scale.length) return rgbaString(parseColor(DEFAULT_SCALE[0][1]));
  const sorted = [...scale].sort((a, b) => a[0] - b[0]);
  if (normalized <= sorted[0][0]) return rgbaString(parseColor(sorted[0][1]));
  if (normalized >= sorted[sorted.length - 1][0]) return rgbaString(parseColor(sorted[sorted.length - 1][1]));
  for (let i = 1; i < sorted.length; i += 1) {
    const left = sorted[i - 1];
    const right = sorted[i];
    if (normalized > right[0]) continue;
    const t = (normalized - left[0]) / Math.max(1e-9, right[0] - left[0]);
    const a = parseColor(left[1]);
    const b = parseColor(right[1]);
    return rgbaString({
      r: a.r + (b.r - a.r) * t,
      g: a.g + (b.g - a.g) * t,
      b: a.b + (b.b - a.b) * t,
      a: a.a + (b.a - a.a) * t
    });
  }
  return rgbaString(parseColor(sorted[sorted.length - 1][1]));
}

function traceColorScale(trace) {
  return trace.colorscale || trace.marker?.colorscale || DEFAULT_SCALE;
}

function numericExtent(values) {
  const clean = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return { min: 0, max: 1, span: 1 };
  const min = Math.min(...clean);
  const max = Math.max(...clean);
  const span = Math.max(1e-6, max - min);
  return { min, max, span };
}

function buildBounds(data) {
  const xs = [];
  const ys = [];
  const zs = [];
  for (const trace of data || []) {
    if (trace.type === "surface") {
      xs.push(...(trace.x || []));
      ys.push(...(trace.y || []));
      for (const row of trace.z || []) zs.push(...row);
      continue;
    }
    xs.push(...(trace.x || []));
    ys.push(...(trace.y || []));
    zs.push(...(trace.z || []));
  }
  const x = numericExtent(xs);
  const y = numericExtent(ys);
  const z = numericExtent(zs);
  const maxSpan = Math.max(x.span, y.span, z.span, 1e-6);
  return {
    x,
    y,
    z,
    center: {
      x: (x.min + x.max) * 0.5,
      y: (y.min + y.max) * 0.5,
      z: (z.min + z.max) * 0.5
    },
    maxSpan
  };
}

function normalizePoint(point, bounds) {
  return {
    x: ((point.x - bounds.center.x) / bounds.maxSpan) * 2,
    y: ((point.y - bounds.center.y) / bounds.maxSpan) * 2,
    z: ((point.z - bounds.center.z) / bounds.maxSpan) * 2
  };
}

function cameraFromLayout(layout, orbit) {
  const scene = layout?.scene || {};
  const eye = scene.camera?.eye || DEFAULT_EYE;
  const yaw = orbit?.yaw ?? Math.atan2(eye.y, eye.x);
  const pitch = orbit?.pitch ?? Math.atan2(eye.z, Math.hypot(eye.x, eye.y));
  const radius = orbit?.radius ?? Math.max(2.2, Math.hypot(eye.x, eye.y, eye.z));
  return {
    position: {
      x: Math.cos(yaw) * Math.cos(pitch) * radius,
      y: Math.sin(yaw) * Math.cos(pitch) * radius,
      z: Math.sin(pitch) * radius
    },
    yaw,
    pitch,
    radius
  };
}

function projectPoint(point, camera, width, height) {
  const eye = camera.position;
  const forward = norm({ x: -eye.x, y: -eye.y, z: -eye.z });
  let right = cross(forward, { x: 0, y: 0, z: 1 });
  if (Math.hypot(right.x, right.y, right.z) < 1e-6) right = { x: 1, y: 0, z: 0 };
  right = norm(right);
  const up = norm(cross(right, forward));
  const rel = sub(point, eye);
  const cx = dot(rel, right);
  const cy = dot(rel, up);
  const cz = Math.max(0.05, dot(rel, forward));
  const scale = (Math.min(width, height) * 0.35 * (camera.radius / 2.6) * DEFAULT_FOCAL) / (DEFAULT_FOCAL + cz);
  return {
    x: width * 0.5 + cx * scale,
    y: height * 0.52 - cy * scale,
    depth: cz,
    scale
  };
}

function buildRenderModel(data, layout, width, height, orbit) {
  const bounds = buildBounds(data);
  const camera = cameraFromLayout(layout, orbit);
  const items = [];
  const legends = [];
  const colorbars = [];
  let legendSeen = new Set();

  const pushLegend = (name, color) => {
    if (!name || legendSeen.has(name)) return;
    legendSeen.add(name);
    legends.push({ name, color });
  };

  for (const trace of data || []) {
    if (trace.name) {
      const legendColor = typeof trace.line?.color === "string"
        ? trace.line.color
        : Array.isArray(trace.marker?.color)
          ? interpolateColor(traceColorScale(trace), 0.6)
          : typeof trace.marker?.color === "string"
            ? trace.marker.color
            : interpolateColor(traceColorScale(trace), 0.6);
      pushLegend(trace.name, legendColor);
    }

    if (trace.type === "surface") {
      const x = trace.x || [];
      const y = trace.y || [];
      const zRows = trace.z || [];
      const zExtent = numericExtent(zRows.flat());
      if (trace.showscale) {
        colorbars.push({
          key: `surface-${colorbars.length}`,
          title: axisTitle(trace.colorbar) || axisTitle(trace.colorbar?.title) || axisTitle(layout?.scene?.zaxis) || "value",
          scale: traceColorScale(trace),
          min: trace.cmin ?? zExtent.min,
          max: trace.cmax ?? zExtent.max
        });
      }
      for (let yi = 0; yi < y.length - 1; yi += 1) {
        for (let xi = 0; xi < x.length - 1; xi += 1) {
          const quad = [
            { x: x[xi], y: y[yi], z: zRows[yi]?.[xi] },
            { x: x[xi + 1], y: y[yi], z: zRows[yi]?.[xi + 1] },
            { x: x[xi + 1], y: y[yi + 1], z: zRows[yi + 1]?.[xi + 1] },
            { x: x[xi], y: y[yi + 1], z: zRows[yi + 1]?.[xi] }
          ];
          if (quad.some((point) => typeof point.z !== "number" || !Number.isFinite(point.z))) continue;
          const normalized = quad.map((point) => normalizePoint(point, bounds));
          const projected = normalized.map((point) => projectPoint(point, camera, width, height));
          const avgDepth = projected.reduce((sum, point) => sum + point.depth, 0) / projected.length;
          const avgZ = quad.reduce((sum, point) => sum + point.z, 0) / quad.length;
          const colorValue = clamp(((avgZ - (trace.cmin ?? zExtent.min)) / Math.max(1e-6, (trace.cmax ?? zExtent.max) - (trace.cmin ?? zExtent.min))), 0, 1);
          items.push({
            kind: "polygon",
            depth: avgDepth,
            points: projected,
            fill: interpolateColor(traceColorScale(trace), colorValue),
            opacity: trace.opacity ?? 0.95,
            stroke: "rgba(255,255,255,0.18)",
            strokeWidth: trace.contours?.z?.show ? 0.6 : 0.25,
            title: `x=${quad[0].x.toFixed?.(2) ?? quad[0].x} y=${quad[0].y.toFixed?.(2) ?? quad[0].y} z=${avgZ.toFixed(3)}`
          });
        }
      }
      continue;
    }

    if (trace.type !== "scatter3d") continue;
    const mode = trace.mode || "markers";
    const xs = trace.x || [];
    const ys = trace.y || [];
    const zs = trace.z || [];
    const texts = Array.isArray(trace.text) ? trace.text : [];
    const customdata = Array.isArray(trace.customdata) ? trace.customdata : [];
    const markerSize = trace.marker?.size;
    const markerColor = trace.marker?.color;
    const markerLine = trace.marker?.line || {};
    const lineColor = trace.line?.color || "rgba(60,72,88,0.65)";
    const lineWidth = trace.line?.width || 2;
    const scatterZExtent = numericExtent(zs);

    const projectedPoints = xs.map((xv, index) => {
      if (xv == null || ys[index] == null || zs[index] == null) return null;
      const raw = { x: Number(xv), y: Number(ys[index]), z: Number(zs[index]) };
      const projected = projectPoint(normalizePoint(raw, bounds), camera, width, height);
      const size = Array.isArray(markerSize)
        ? Number(markerSize[index] ?? 5)
        : Number(markerSize ?? 5);
      let fill = typeof markerColor === "string" ? markerColor : "#4c6ef5";
      if (Array.isArray(markerColor)) {
        const value = markerColor[index];
        if (typeof value === "string") fill = value;
        else if (typeof value === "number" && Number.isFinite(value)) {
          const min = trace.marker?.cmin ?? scatterZExtent.min;
          const max = trace.marker?.cmax ?? scatterZExtent.max;
          fill = interpolateColor(traceColorScale(trace), (value - min) / Math.max(1e-6, max - min));
        }
      }
      return {
        raw,
        projected,
        size: clamp(size, 1.5, 22),
        fill,
        lineColor: markerLine.color || "rgba(255,255,255,0.75)",
        lineWidth: markerLine.width || 0.8,
        text: texts[index] == null ? "" : String(texts[index]),
        customdata: customdata[index],
        trace
      };
    });

    if (trace.marker?.showscale) {
      colorbars.push({
        key: `marker-${colorbars.length}`,
        title: axisTitle(trace.marker?.colorbar?.title) || axisTitle(trace.marker?.colorbar) || trace.name || "value",
        scale: traceColorScale(trace),
        min: trace.marker?.cmin ?? scatterZExtent.min,
        max: trace.marker?.cmax ?? scatterZExtent.max
      });
    }

    if (mode.includes("lines")) {
      let segment = [];
      const flush = () => {
        if (segment.length < 2) {
          segment = [];
          return;
        }
        const avgDepth = segment.reduce((sum, point) => sum + point.projected.depth, 0) / segment.length;
        items.push({
          kind: "polyline",
          depth: avgDepth,
          points: segment.map((point) => point.projected),
          stroke: lineColor,
          strokeWidth: lineWidth,
          opacity: trace.opacity ?? 0.92,
          title: trace.name || "line"
        });
        segment = [];
      };
      for (const point of projectedPoints) {
        if (!point) {
          flush();
          continue;
        }
        segment.push(point);
      }
      flush();
    }

    if (mode.includes("markers")) {
      for (const point of projectedPoints) {
        if (!point) continue;
        items.push({
          kind: "marker",
          depth: point.projected.depth,
          point,
          opacity: trace.marker?.opacity ?? trace.opacity ?? 0.95,
          title: point.text || trace.name || "point"
        });
      }
    }
  }

  items.sort((a, b) => b.depth - a.depth);
  return { items, legends, colorbars, axisTitles: {
    x: axisTitle(layout?.scene?.xaxis),
    y: axisTitle(layout?.scene?.yaxis),
    z: axisTitle(layout?.scene?.zaxis)
  }};
}

function ColorBar({ entry, top = 14 }) {
  const gradientId = `gradient-${entry.key}`;
  const minLabel = Number.isFinite(entry.min) ? Number(entry.min).toFixed(2) : "n/a";
  const maxLabel = Number.isFinite(entry.max) ? Number(entry.max).toFixed(2) : "n/a";
  return (
    <div className="plot3d-colorbar" style={{ top }}>
      <svg width="18" height="120" viewBox="0 0 18 120" aria-hidden="true">
        <defs>
          <linearGradient id={gradientId} x1="0" y1="1" x2="0" y2="0">
            {[...entry.scale].sort((a, b) => a[0] - b[0]).map(([stop, color]) => (
              <stop key={`${gradientId}-${stop}`} offset={`${stop * 100}%`} stopColor={color} />
            ))}
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="18" height="120" rx="8" fill={`url(#${gradientId})`} />
      </svg>
      <div className="plot3d-colorbar-labels">
        <strong>{entry.title}</strong>
        <span>{maxLabel}</span>
        <span>{minLabel}</span>
      </div>
    </div>
  );
}

export default function Plot3D({
  data = [],
  layout = {},
  style,
  onHover,
  onUnhover
}) {
  const hostRef = useRef(null);
  const [size, setSize] = useState({ width: 640, height: 360 });
  const [orbit, setOrbit] = useState(null);
  const dragRef = useRef(null);

  useEffect(() => {
    if (!hostRef.current || typeof ResizeObserver === "undefined") return undefined;
    const node = hostRef.current;
    const updateSize = () => {
      const nextWidth = Math.max(240, Math.round(node.clientWidth || 640));
      const nextHeight = Math.max(180, Math.round(node.clientHeight || 360));
      setSize((current) => current.width === nextWidth && current.height === nextHeight ? current : { width: nextWidth, height: nextHeight });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const eye = layout?.scene?.camera?.eye || DEFAULT_EYE;
    setOrbit({
      yaw: Math.atan2(eye.y, eye.x),
      pitch: Math.atan2(eye.z, Math.hypot(eye.x, eye.y)),
      radius: Math.max(2.2, Math.hypot(eye.x, eye.y, eye.z))
    });
  }, [layout?.uirevision, layout?.scene?.camera?.eye?.x, layout?.scene?.camera?.eye?.y, layout?.scene?.camera?.eye?.z]);

  const renderModel = useMemo(
    () => buildRenderModel(data, layout, size.width - DEFAULT_MARGIN * 2, size.height - DEFAULT_MARGIN * 2, orbit),
    [data, layout, orbit, size.height, size.width]
  );

  const startDrag = (event) => {
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      yaw: orbit?.yaw ?? 0,
      pitch: orbit?.pitch ?? 0
    };
  };

  const moveDrag = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (event.cancelable) event.preventDefault();
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    setOrbit((current) => current ? {
      ...current,
      yaw: drag.yaw - dx * 0.01,
      pitch: clamp(drag.pitch + dy * 0.008, -1.35, 1.35)
    } : current);
  };

  const endDrag = (event = null) => {
    if (event && event.currentTarget && dragRef.current?.pointerId === event.pointerId) {
      try {
        event.currentTarget.releasePointerCapture(event.pointerId);
      } catch {
        // ignore release failures
      }
    }
    dragRef.current = null;
  };

  useEffect(() => {
    const node = hostRef.current;
    if (!node) return undefined;
    const handleWheel = (event) => {
      event.preventDefault();
      event.stopPropagation();
      setOrbit((current) => current ? {
        ...current,
        radius: clamp(current.radius + Math.sign(event.deltaY) * 0.18, 1.6, 5.4)
      } : current);
    };
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      node.removeEventListener("wheel", handleWheel);
    };
  }, []);

  return (
    <div
      ref={hostRef}
      className="plot3d-host"
      style={{ width: "100%", height: "100%", minHeight: 180, position: "relative", touchAction: "none", overscrollBehavior: "contain", ...style }}
      onPointerDown={(event) => {
        if (event.button != null && event.pointerType === "mouse" && event.button !== 0) return;
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {
          // ignore capture failures
        }
        startDrag(event);
      }}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
      onMouseLeave={() => {
        endDrag();
        onUnhover?.();
      }}
    >
      <svg width="100%" height="100%" viewBox={`0 0 ${size.width} ${size.height}`} role="img" aria-label="3D plot">
        <rect x="0" y="0" width={size.width} height={size.height} rx="18" fill="rgba(247,251,250,0.58)" />
        <g transform={`translate(${DEFAULT_MARGIN}, ${DEFAULT_MARGIN})`}>
          {renderModel.items.map((item, index) => {
            if (item.kind === "polygon") {
              return (
                <polygon
                  key={`poly-${index}`}
                  points={item.points.map((point) => `${point.x},${point.y}`).join(" ")}
                  fill={item.fill}
                  fillOpacity={item.opacity}
                  stroke={item.stroke}
                  strokeWidth={item.strokeWidth}
                >
                  <title>{item.title}</title>
                </polygon>
              );
            }
            if (item.kind === "polyline") {
              return (
                <polyline
                  key={`line-${index}`}
                  points={item.points.map((point) => `${point.x},${point.y}`).join(" ")}
                  fill="none"
                  stroke={item.stroke}
                  strokeOpacity={item.opacity}
                  strokeWidth={item.strokeWidth}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <title>{item.title}</title>
                </polyline>
              );
            }
            const marker = item.point;
            return (
              <g
                key={`marker-${index}`}
                onMouseEnter={() => onHover?.({ points: [{ data: marker.trace, customdata: marker.customdata, x: marker.raw.x, y: marker.raw.y, z: marker.raw.z, text: marker.text }] })}
                onMouseLeave={() => onUnhover?.()}
              >
                <circle
                  cx={marker.projected.x}
                  cy={marker.projected.y}
                  r={marker.size * 0.5}
                  fill={marker.fill}
                  fillOpacity={item.opacity}
                  stroke={marker.lineColor}
                  strokeWidth={marker.lineWidth}
                >
                  <title>{marker.text || marker.trace.name || "point"}</title>
                </circle>
                {marker.text ? (
                  <text
                    x={marker.projected.x}
                    y={marker.projected.y - marker.size * 0.7}
                    fontSize="9"
                    fill="#334155"
                    textAnchor="middle"
                  >
                    {marker.text}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>
      {renderModel.legends.length ? (
        <div className="plot3d-legend">
          {renderModel.legends.map((entry) => (
            <span key={entry.name} className="plot3d-legend-item">
              <i style={{ background: entry.color }} />
              {entry.name}
            </span>
          ))}
        </div>
      ) : null}
      <div className="plot3d-axis-label plot3d-axis-label-x">{renderModel.axisTitles.x}</div>
      <div className="plot3d-axis-label plot3d-axis-label-y">{renderModel.axisTitles.y}</div>
      <div className="plot3d-axis-label plot3d-axis-label-z">{renderModel.axisTitles.z}</div>
      {renderModel.colorbars.map((entry, index) => (
        <ColorBar key={entry.key} entry={entry} top={18 + index * 132} />
      ))}
    </div>
  );
}
