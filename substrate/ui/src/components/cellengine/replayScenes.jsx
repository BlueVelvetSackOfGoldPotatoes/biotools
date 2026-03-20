import {
  clamp,
  clamp01,
  FUNCTIONAL_STATE_META,
  cellTypeStroke,
  edgeStrength,
  edgeStroke,
  formatActionLabel,
  formatNumber,
  formatPct,
  formatSigned,
  taskMeta
} from "./core";
import {
  bodyCoord,
  bodyExtents,
  compactPreviewBodyView,
  depthShadowStyle,
  projectBodyLayout,
  replayTipTrail,
  sortProjectedCells
} from "./body";
export function SceneOrganismOverlay({
  x,
  y,
  width,
  height,
  bodyCells,
  bodyView,
  functionalStateByCellId,
  selectedCellId,
  onSelectCell,
  footerLines = []
}) {
  if (!bodyCells?.length) {
    return (
      <g className="cellengine-overlay-card">
        <rect x={x} y={y} width={width} height={height} rx="20" fill="#ffffffdd" stroke="#a61e4d" opacity="0.92" />
        <text x={x + 20} y={y + 30} className="cellengine-overlay-title">Selected Genome Live Body</text>
        <text x={x + 20} y={y + 58} className="cellengine-overlay-line">No body cells available in this replay.</text>
      </g>
    );
  }

  const topPad = 38;
  const sidePad = 16;
  const bottomPad = Math.max(24, 10 + footerLines.length * 18);
  const rawLayout = projectBodyLayout(bodyCells, 18, { depthSkewRatio: 0.68, depthLiftRatio: 0.36, camera: bodyView });
  const fitScale = Math.min(
    (width - sidePad * 2) / Math.max(1, rawLayout.width),
    (height - topPad - bottomPad) / Math.max(1, rawLayout.height)
  );
  const cellSize = Math.max(8.5, fitScale * 15.4);
  const gridWidth = rawLayout.width * fitScale;
  const gridHeight = rawLayout.height * fitScale;
  const gridX = x + (width - gridWidth) / 2;
  const gridY = y + topPad;
  const sortedCells = sortProjectedCells(rawLayout.cells);

  return (
    <g className="cellengine-overlay-card">
      <rect x={x} y={y} width={width} height={height} rx="20" fill="#ffffffdd" stroke="#a61e4d" opacity="0.92" />
      <text x={x + 20} y={y + 28} className="cellengine-overlay-title">Selected Genome Live Body</text>
      <rect x={gridX - 6} y={gridY - 6} width={gridWidth + 12} height={gridHeight + 12} rx="14" fill="#f8fbfa" stroke="#dbe6df" />
      {sortedCells.map((cell) => {
        const mode = functionalStateByCellId?.get(cell.cell_id) || FUNCTIONAL_STATE_META.inactive;
        const isSelected = cell.cell_id === selectedCellId;
        const cellX = gridX + cell.plotX * fitScale;
        const cellY = gridY + cell.plotY * fitScale;
        const drawSize = Math.max(7, cellSize * (0.92 + cell.plotDepth * 0.1));
        return (
          <g
            key={`scene-body-${cell.cell_id}`}
            transform={`translate(${cellX}, ${cellY})`}
            className={onSelectCell ? "cellengine-tissue-cell" : undefined}
            onClick={onSelectCell ? () => onSelectCell(cell.cell_id) : undefined}
            style={depthShadowStyle(cell.plotDepth, isSelected)}
          >
            <title>{`cell ${cell.cell_id} · ${mode.label}`}</title>
            <rect
              x="0"
              y="0"
              width={drawSize - 1.2}
              height={drawSize - 1.2}
              rx={Math.max(3, drawSize * 0.18)}
              fill={mode.fill}
              stroke={isSelected ? "#111827" : cellTypeStroke(cell)}
              strokeWidth={isSelected ? 2.2 : 1.2}
            />
            {drawSize >= 11 ? (
              <text
                x={(drawSize - 1.2) / 2}
                y={drawSize * 0.63}
                textAnchor="middle"
                className="cellengine-scene-cell-symbol"
                style={{ fill: mode.ink }}
              >
                {mode.symbol}
              </text>
            ) : null}
          </g>
        );
      })}
      {footerLines.map((line, index) => (
        <text key={`body-footer-${index}`} x={x + 20} y={y + height - 10 - (footerLines.length - 1 - index) * 18} className="cellengine-scene-detail">
          {line}
        </text>
      ))}
    </g>
  );
}

export function SceneOrganismBody({
  pivotX,
  pivotY,
  theta,
  bodyCells,
  frameCellRows,
  frameEdges,
  functionalStateByCellId,
  selectedCellId,
  onSelectCell
}) {
  if (!bodyCells?.length) return null;

  const stateByCellId = new Map((frameCellRows || []).map((row) => [row.cell_id, row]));
  const bodyById = new Map(bodyCells.map((cell) => [cell.cell_id, cell]));
  const extents = bodyExtents(bodyCells);
  const hingeCells = bodyCells.filter((cell) => cell.hinge);
  const pivotBodyX = hingeCells.length
    ? hingeCells.reduce((sum, cell) => sum + bodyCoord(cell, "x"), 0) / hingeCells.length
    : 0;
  const pivotBodyY = hingeCells.length
    ? Math.max(...hingeCells.map((cell) => bodyCoord(cell, "y")))
    : 0;
  const pivotBodyZ = hingeCells.length
    ? hingeCells.reduce((sum, cell) => sum + bodyCoord(cell, "z"), 0) / hingeCells.length
    : 0;
  const xFootprint = extents.spanX + extents.spanZ * 0.72 + 1.4;
  const yFootprint = extents.spanY + extents.spanZ * 0.36 + 1.3;
  const colPitch = Math.max(9.5, Math.min(18, 176 / Math.max(1, xFootprint)));
  const rowPitch = Math.max(10.5, Math.min(19, 188 / Math.max(1, yFootprint)));
  const depthSkew = extents.spanZ > 0 ? colPitch * 0.68 : 0;
  const depthLift = extents.spanZ > 0 ? rowPitch * 0.34 : 0;
  const cellSize = Math.max(8.2, Math.min(13.5, Math.min(colPitch, rowPitch) * 0.88));
  const centers = new Map();
  for (const cell of bodyCells) {
    const state = stateByCellId.get(cell.cell_id) || null;
    const active = state?.active === 1 || state?.active === true;
    const activity = clamp01((state?.activity || 0) / 2.5);
    const zNorm = extents.spanZ > 0
      ? (bodyCoord(cell, "z") - extents.minZ) / Math.max(1, extents.spanZ)
      : 0.5;
    let localDx = Math.max(-2.8, Math.min(2.8, (state?.a_mech || 0) * 1.1 + (state?.chem_out || 0) * 0.55));
    let localDy = Math.max(-2.4, Math.min(2.4, -(state?.activity || 0) * 0.6 + (state?.V || 0) * 0.22));
    const scale = 0.82 + activity * 0.2 + (active ? 0.04 : 0) + zNorm * 0.08;
    const baseSize = cellSize - 1.4;
    const scaledSize = baseSize * scale;
    if (cell.ground) {
      localDx *= 0.15;
      localDy = 0;
    }
    const localX =
      (bodyCoord(cell, "x") - pivotBodyX) * colPitch +
      (bodyCoord(cell, "z") - pivotBodyZ) * depthSkew +
      localDx;
    const localY =
      (pivotBodyY - bodyCoord(cell, "y")) * rowPitch -
      (bodyCoord(cell, "z") - pivotBodyZ) * depthLift +
      localDy;
    const centerX = cell.ground
      ? pivotX + localX
      : pivotX + localX * Math.cos(theta) - localY * Math.sin(theta);
    const centerY = cell.ground
      ? pivotY + localY
      : pivotY + localX * Math.sin(theta) + localY * Math.cos(theta);
    centers.set(cell.cell_id, {
      centerX,
      centerY,
      scaledSize,
      state,
      active,
      activity,
      zNorm,
      z: bodyCoord(cell, "z")
    });
  }
  const sortedCells = [...bodyCells].sort((left, right) => {
    const zDelta = bodyCoord(left, "z") - bodyCoord(right, "z");
    if (zDelta !== 0) return zDelta;
    return bodyCoord(left, "y") - bodyCoord(right, "y");
  });
  const sortedEdges = [...(frameEdges || [])].sort((left, right) => {
    const leftDepth = (bodyCoord(bodyById.get(left.src_cell_id), "z") + bodyCoord(bodyById.get(left.dst_cell_id), "z")) / 2;
    const rightDepth = (bodyCoord(bodyById.get(right.src_cell_id), "z") + bodyCoord(bodyById.get(right.dst_cell_id), "z")) / 2;
    return leftDepth - rightDepth;
  });

  return (
    <g className="cellengine-scene-organism">
      {sortedEdges.map((edge) => {
        const srcCell = bodyById.get(edge.src_cell_id);
        const dstCell = bodyById.get(edge.dst_cell_id);
        const src = centers.get(edge.src_cell_id);
        const dst = centers.get(edge.dst_cell_id);
        if (!srcCell || !dstCell || !src || !dst) return null;
        const strength = edgeStrength(edge);
        if (strength <= 0.04) return null;
        return (
          <line
            key={`scene-link-${edge.src_cell_id}-${edge.dst_cell_id}`}
            x1={src.centerX}
            y1={src.centerY}
            x2={dst.centerX}
            y2={dst.centerY}
            stroke={edgeStroke(edge, selectedCellId)}
            strokeWidth={0.8 + strength * 3.1}
            opacity={0.18 + strength * 0.45}
            strokeLinecap="round"
          />
        );
      })}
      {sortedCells.map((cell) => {
        const mode = functionalStateByCellId?.get(cell.cell_id) || FUNCTIONAL_STATE_META.inactive;
        const { centerX, centerY, scaledSize, active, activity, zNorm } = centers.get(cell.cell_id);
        const r = scaledSize / 2;
        const isSelected = cell.cell_id === selectedCellId;
        return (
          <g
            key={`scene-body-live-${cell.cell_id}`}
            className={onSelectCell ? "cellengine-tissue-cell" : undefined}
            onClick={onSelectCell ? () => onSelectCell(cell.cell_id) : undefined}
            style={depthShadowStyle(zNorm, isSelected)}
          >
            <title>{`cell ${cell.cell_id} · ${mode.label}`}</title>
            <circle
              cx={centerX}
              cy={centerY}
              r={r + 1.8}
              fill={mode.ink}
              opacity={active ? 0.12 + activity * 0.16 + zNorm * 0.06 : 0.04}
            />
            <circle
              cx={centerX}
              cy={centerY}
              r={r}
              fill={mode.fill}
              opacity={active ? 1 : 0.48}
              stroke={isSelected ? "#111827" : cellTypeStroke(cell)}
              strokeWidth={isSelected ? 2.1 : 1.05 + zNorm * 0.35}
            />
            <circle
              cx={centerX}
              cy={centerY}
              r={Math.max(1.6, r * 0.32)}
              fill={mode.ink}
              opacity={0.88}
            />
            {cell.motor ? (
              <circle cx={centerX + r * 0.52} cy={centerY - r * 0.42} r={Math.max(1.2, r * 0.18)} fill="#a61e4d" />
            ) : null}
            {cell.hinge ? (
              <circle cx={centerX - r * 0.52} cy={centerY - r * 0.42} r={Math.max(1.2, r * 0.18)} fill="#0c8599" />
            ) : null}
            {cell.ground ? (
              <rect x={centerX - r * 0.5} y={centerY + r * 0.46} width={r} height={Math.max(1.6, r * 0.18)} rx="2" fill="#2b8a3e" />
            ) : null}
            {!active ? (
              <g stroke="#475569" strokeWidth="1.3" opacity="0.72">
                <line x1={centerX - r * 0.58} y1={centerY - r * 0.58} x2={centerX + r * 0.58} y2={centerY + r * 0.58} />
                <line x1={centerX + r * 0.58} y1={centerY - r * 0.58} x2={centerX - r * 0.58} y2={centerY + r * 0.58} />
              </g>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

export function ControllerReplayScene({
  frames,
  frameIndex,
  analysis,
  controllerLabel,
  controllerColor,
  rightTitle,
  rightLines,
  driftNote,
  renderOrganismBody = false,
  organismBodyCells = null,
  organismFrameCellRows = null,
  organismFrameEdges = null,
  organismFunctionalStateByCellId = null,
  organismSelectedCellId = null,
  onSelectOrganismCell = null
}) {
  const frame = frames[frameIndex] || null;
  if (!frame) {
    return <div className="empty">Run a replay to populate the viewer.</div>;
  }

  const sceneWidth = 960;
  const sceneHeight = 540;
  const poleLengthPx = 170;
  const trackHalfSpan = Math.max(
    2.4,
    ...frames.map((entry) =>
      typeof entry.x === "number" && Number.isFinite(entry.x) ? Math.abs(entry.x) + 0.45 : 0
    )
  );
  const cartX = sceneWidth / 2;
  const railY = sceneHeight - 110;
  const cartWidth = 124;
  const cartHeight = 46;
  const pivotX = cartX;
  const pivotY = railY - 32;
  const theta = frame.theta_rad || 0;
  const tipX = pivotX + poleLengthPx * Math.sin(theta);
  const tipY = pivotY - poleLengthPx * Math.cos(theta);
  const trail = replayTipTrail(frames, frameIndex, pivotX, poleLengthPx, sceneHeight, 80);
  const trailPath = trail
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.tipX.toFixed(1)} ${point.tipY.toFixed(1)}`)
    .join(" ");
  const failAngleRad = ((analysis?.fail_angle_deg || 15) * Math.PI) / 180;
  const failLeftX = pivotX + poleLengthPx * Math.sin(-failAngleRad);
  const failRightX = pivotX + poleLengthPx * Math.sin(failAngleRad);
  const failTopY = pivotY - poleLengthPx * Math.cos(failAngleRad);
  const organismArrow = Math.max(-150, Math.min(150, (frame.organism_force || 0) * 12));
  const teacherArrow = Math.max(-150, Math.min(150, (frame.teacher_force || 0) * 12));
  const rightBoxHeight = 56 + Math.max(1, (rightLines || []).length) * 24;
  const forceAnchorX = pivotX;

  return (
    <svg className="cellengine-scene" viewBox={`0 0 ${sceneWidth} ${sceneHeight}`} role="img" aria-label="CellEngine cartpole replay">
      <defs>
        <linearGradient id="cellengineSky" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#fffaf0" />
          <stop offset="55%" stopColor="#eef9f8" />
          <stop offset="100%" stopColor="#fdeef3" />
        </linearGradient>
        <linearGradient id="cellengineTrack" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#10454f" />
          <stop offset="50%" stopColor="#0c8599" />
          <stop offset="100%" stopColor="#10454f" />
        </linearGradient>
        <linearGradient id="cellengineCart" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#1f2937" />
          <stop offset="100%" stopColor="#475569" />
        </linearGradient>
        <filter id="cellengineGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="6" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect x="0" y="0" width={sceneWidth} height={sceneHeight} rx="30" fill="url(#cellengineSky)" />

      <g opacity="0.22">
        {Array.from({ length: 9 }).map((_, idx) => (
          <line key={`v-${idx}`} x1={120 + idx * 90} x2={120 + idx * 90} y1="70" y2={railY + 20} stroke="#7cc4ce" strokeDasharray="4 10" />
        ))}
        {Array.from({ length: 4 }).map((_, idx) => (
          <line key={`h-${idx}`} x1="80" x2="880" y1={110 + idx * 90} y2={110 + idx * 90} stroke="#d6d9cc" strokeDasharray="5 11" />
        ))}
      </g>

      {renderOrganismBody ? (
        <g>
          <text x="96" y={railY + 52} className="cellengine-scene-label">{controllerLabel} replay, centered for readability</text>
        </g>
      ) : (
        <g>
          <rect x="76" y={railY + 10} width="808" height="18" rx="9" fill="#dce7e5" />
          <rect x="92" y={railY + 2} width="776" height="14" rx="7" fill="url(#cellengineTrack)" />
          <text x="96" y={railY + 52} className="cellengine-scene-label">{controllerLabel} replay, cart-centered for readability</text>
        </g>
      )}

      <g opacity="0.55">
        <line x1={pivotX} y1={pivotY} x2={failLeftX} y2={failTopY} stroke="#d9485f" strokeWidth="3" strokeDasharray="8 8" />
        <line x1={pivotX} y1={pivotY} x2={failRightX} y2={failTopY} stroke="#d9485f" strokeWidth="3" strokeDasharray="8 8" />
        <text x={pivotX + 12} y={failTopY - 8} className="cellengine-scene-alert">±{analysis?.fail_angle_deg || 15}° fail boundary</text>
      </g>

      {trailPath ? (
        <path d={trailPath} fill="none" stroke="#f08c00" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" opacity="0.6" filter="url(#cellengineGlow)" />
      ) : null}
      {trail.filter((point) => point.damage).map((point, idx) => (
        <circle key={`damage-${idx}`} cx={point.tipX} cy={point.tipY} r="7" fill="#c92a2a" opacity="0.85" />
      ))}

      {renderOrganismBody ? (
        <g>
          <circle cx={tipX} cy={tipY} r="18" fill="#f08c00" />
          <circle cx={tipX} cy={tipY} r="9" fill="#fff7e6" />
        </g>
      ) : (
        <g>
          <line x1={pivotX} y1={pivotY} x2={tipX} y2={tipY} stroke="#2d3748" strokeWidth="12" strokeLinecap="round" filter="url(#cellengineGlow)" />
          <circle cx={tipX} cy={tipY} r="18" fill="#f08c00" />
          <circle cx={tipX} cy={tipY} r="9" fill="#fff7e6" />
        </g>
      )}

      <g>
        {renderOrganismBody ? (
          <SceneOrganismBody
            pivotX={pivotX}
            pivotY={pivotY}
            theta={theta}
            bodyCells={organismBodyCells}
            frameCellRows={organismFrameCellRows}
            frameEdges={organismFrameEdges}
            functionalStateByCellId={organismFunctionalStateByCellId}
            selectedCellId={organismSelectedCellId}
            onSelectCell={onSelectOrganismCell}
          />
        ) : (
          <>
            <rect x={cartX - cartWidth / 2} y={railY - cartHeight} width={cartWidth} height={cartHeight} rx="16" fill="url(#cellengineCart)" />
            <rect x={cartX - cartWidth / 2 + 8} y={railY - cartHeight + 8} width={cartWidth - 16} height="12" rx="6" fill="#cbd5e1" opacity="0.55" />
            <circle cx={cartX - 34} cy={railY + 6} r="14" fill="#0f172a" />
            <circle cx={cartX + 34} cy={railY + 6} r="14" fill="#0f172a" />
            <circle cx={pivotX} cy={pivotY} r="12" fill="#fff" stroke="#0c8599" strokeWidth="6" />
          </>
        )}
      </g>

      <g opacity="0.9">
        <line x1={forceAnchorX} y1={railY - 72} x2={forceAnchorX + organismArrow} y2={railY - 72} stroke={controllerColor} strokeWidth="10" strokeLinecap="round" />
        <line x1={forceAnchorX + organismArrow} y1={railY - 72} x2={forceAnchorX + organismArrow - Math.sign(organismArrow || 1) * 18} y2={railY - 84} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <line x1={forceAnchorX + organismArrow} y1={railY - 72} x2={forceAnchorX + organismArrow - Math.sign(organismArrow || 1) * 18} y2={railY - 60} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <text
          x={forceAnchorX + organismArrow + (organismArrow >= 0 ? 10 : -10)}
          y={railY - 86}
          textAnchor={organismArrow >= 0 ? "start" : "end"}
          className="cellengine-scene-force"
          style={{ fill: controllerColor }}
        >
          {controllerLabel}
        </text>
      </g>

      <g opacity={Math.abs(teacherArrow) > 2 ? 0.9 : 0.25}>
        <line x1={forceAnchorX} y1={railY - 102} x2={forceAnchorX + teacherArrow} y2={railY - 102} stroke="#1971c2" strokeWidth="7" strokeLinecap="round" />
        <line x1={forceAnchorX + teacherArrow} y1={railY - 102} x2={forceAnchorX + teacherArrow - Math.sign(teacherArrow || 1) * 16} y2={railY - 112} stroke="#1971c2" strokeWidth="6" strokeLinecap="round" />
        <line x1={forceAnchorX + teacherArrow} y1={railY - 102} x2={forceAnchorX + teacherArrow - Math.sign(teacherArrow || 1) * 16} y2={railY - 92} stroke="#1971c2" strokeWidth="6" strokeLinecap="round" />
        <text x={forceAnchorX + teacherArrow + (teacherArrow >= 0 ? 10 : -10)} y={railY - 116} textAnchor={teacherArrow >= 0 ? "start" : "end"} className="cellengine-scene-force teacher">teacher</text>
      </g>

      <g className="cellengine-overlay-card">
        <rect x="34" y="28" width="238" height="122" rx="20" fill="#ffffffdd" stroke="#d8e7e2" />
        <text x="54" y="56" className="cellengine-overlay-title">Frame telemetry</text>
        <text x="54" y="84" className="cellengine-overlay-line">tick {frame.tick} / {frames[frames.length - 1]?.tick ?? "n/a"}</text>
        <text x="54" y="108" className="cellengine-overlay-line">theta {formatSigned(frame.theta_deg, 2, "°")}</text>
        <text x="54" y="132" className="cellengine-overlay-line">cart x {formatSigned(frame.x, 3, " m")}</text>
      </g>

      <g className="cellengine-overlay-card">
        <rect x="688" y="28" width="238" height={rightBoxHeight} rx="20" fill="#ffffffdd" stroke={controllerColor} opacity="0.9" />
        <text x="708" y="56" className="cellengine-overlay-title">{rightTitle}</text>
        {(rightLines || []).map((line, index) => (
          <text key={`${index}-${line}`} x="708" y={84 + index * 24} className="cellengine-overlay-line">{line}</text>
        ))}
      </g>

      {frame.damage_event === 1 || frame.damage_event === true ? (
        <g>
          <rect x="360" y="34" width="240" height="44" rx="22" fill="#fff2f0" stroke="#f08c7b" />
          <text x="480" y="61" textAnchor="middle" className="cellengine-scene-alert">damage event injected</text>
        </g>
      ) : null}
      {frame.terminal === 1 || frame.terminal === true ? (
        <g>
          <rect x="356" y="88" width="248" height="52" rx="24" fill="#7f1d1dcc" />
          <text x="480" y="121" textAnchor="middle" className="cellengine-scene-terminal">terminal state reached</text>
        </g>
      ) : null}

      <g>
        <rect x="166" y="470" width="628" height="18" rx="9" fill="#e7ece9" />
        <rect x="184" y="476" width="592" height="6" rx="3" fill="#9fb3b8" />
        <line x1="480" y1="466" x2="480" y2="494" stroke="#475569" strokeDasharray="3 3" />
        <circle
          cx={184 + clamp01((frame.x + trackHalfSpan) / (2 * trackHalfSpan)) * 592}
          cy="479"
          r="8"
          fill="#d9480f"
        />
        <text x="166" y="512" className="cellengine-scene-label">
          absolute cart drift {formatSigned(frame.x, 2, " m")} across ±{formatNumber(trackHalfSpan, 2)} m observed span
        </text>
        <text x="166" y="532" className="cellengine-scene-detail">
          {driftNote}
        </text>
      </g>
    </svg>
  );
}

export function springCoilPath(startX, endX, y, amplitude = 16, coils = 8) {
  const span = Math.max(12, endX - startX);
  const step = span / Math.max(1, coils * 2);
  let path = `M ${startX.toFixed(1)} ${y.toFixed(1)}`;
  for (let index = 1; index < coils * 2; index += 1) {
    const x = startX + step * index;
    const dy = index % 2 === 0 ? -amplitude : amplitude;
    path += ` L ${x.toFixed(1)} ${(y + dy).toFixed(1)}`;
  }
  path += ` L ${endX.toFixed(1)} ${y.toFixed(1)}`;
  return path;
}

export function MassSpringReplayScene({
  frames,
  frameIndex,
  analysis,
  controllerLabel,
  controllerColor,
  rightTitle,
  rightLines,
  renderOrganismBody = false,
  organismBodyCells = null,
  organismFrameCellRows = null,
  organismFrameEdges = null,
  organismFunctionalStateByCellId = null,
  organismSelectedCellId = null,
  onSelectOrganismCell = null
}) {
  const frame = frames[frameIndex] || null;
  if (!frame) return <div className="empty">Run a replay to populate the viewer.</div>;

  const sceneWidth = 960;
  const sceneHeight = 540;
  const railY = sceneHeight - 120;
  const equilibriumX = 510;
  const anchorX = 168;
  const massWidth = 128;
  const massHeight = 62;
  const displacementSpan = Math.max(
    0.5,
    ...frames.map((entry) =>
      typeof entry.x === "number" && Number.isFinite(entry.x) ? Math.abs(entry.x) + 0.12 : 0
    ),
    2.4
  );
  const normalizedX = clamp(frame.x / displacementSpan, -1, 1);
  const massX = equilibriumX + normalizedX * 248;
  const springStartX = anchorX + 18;
  const springEndX = massX - massWidth / 2 + 6;
  const springY = railY - massHeight / 2 + 4;
  const springPath = springCoilPath(springStartX, springEndX, springY);
  const forceArrow = Math.max(-150, Math.min(150, (frame.organism_force || 0) * 14));
  const trail = frames
    .slice(Math.max(0, frameIndex - 50), frameIndex + 1)
    .map((entry) => equilibriumX + clamp(entry.x / displacementSpan, -1, 1) * 248);

  return (
    <svg className="cellengine-scene" viewBox={`0 0 ${sceneWidth} ${sceneHeight}`} role="img" aria-label="CellEngine mass spring replay">
      <defs>
        <linearGradient id="springSky" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#fff8ef" />
          <stop offset="60%" stopColor="#eef8fb" />
          <stop offset="100%" stopColor="#f7eef7" />
        </linearGradient>
        <linearGradient id="springMass" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#1f2937" />
          <stop offset="100%" stopColor="#64748b" />
        </linearGradient>
      </defs>

      <rect x="0" y="0" width={sceneWidth} height={sceneHeight} rx="30" fill="url(#springSky)" />
      <g opacity="0.2">
        {Array.from({ length: 9 }).map((_, idx) => (
          <line key={`spring-v-${idx}`} x1={120 + idx * 90} x2={120 + idx * 90} y1="74" y2={railY + 30} stroke="#7cc4ce" strokeDasharray="4 10" />
        ))}
      </g>
      <rect x="76" y={railY + 10} width="808" height="18" rx="9" fill="#dce7e5" />
      <rect x="92" y={railY + 2} width="776" height="14" rx="7" fill="url(#cellengineTrack)" />

      <g>
        <rect x={anchorX - 20} y={railY - 104} width="20" height="128" rx="6" fill="#475569" />
        <line x1={equilibriumX} y1={railY - 120} x2={equilibriumX} y2={railY + 26} stroke="#d9485f" strokeDasharray="7 8" strokeWidth="3" opacity="0.48" />
        <text x={equilibriumX + 10} y={railY - 126} className="cellengine-scene-alert">equilibrium</text>
        {trail.map((x, index) => (
          <circle
            key={`spring-trail-${index}`}
            cx={x}
            cy={railY - 20}
            r={Math.max(1.5, 5 - (trail.length - index) * 0.07)}
            fill="#f08c00"
            opacity={0.08 + index / Math.max(1, trail.length) * 0.28}
          />
        ))}
        <path d={springPath} fill="none" stroke="#8b5e34" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
      </g>

      {renderOrganismBody ? (
        <SceneOrganismBody
          pivotX={massX}
          pivotY={railY - 4}
          theta={0}
          bodyCells={organismBodyCells}
          frameCellRows={organismFrameCellRows}
          frameEdges={organismFrameEdges}
          functionalStateByCellId={organismFunctionalStateByCellId}
          selectedCellId={organismSelectedCellId}
          onSelectCell={onSelectOrganismCell}
        />
      ) : (
        <g>
          <rect x={massX - massWidth / 2} y={railY - massHeight} width={massWidth} height={massHeight} rx="18" fill="url(#springMass)" />
          <rect x={massX - massWidth / 2 + 10} y={railY - massHeight + 10} width={massWidth - 20} height="14" rx="7" fill="#cbd5e1" opacity="0.55" />
        </g>
      )}

      <g opacity="0.9">
        <line x1={massX} y1={railY - 104} x2={massX + forceArrow} y2={railY - 104} stroke={controllerColor} strokeWidth="10" strokeLinecap="round" />
        <line x1={massX + forceArrow} y1={railY - 104} x2={massX + forceArrow - Math.sign(forceArrow || 1) * 18} y2={railY - 116} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <line x1={massX + forceArrow} y1={railY - 104} x2={massX + forceArrow - Math.sign(forceArrow || 1) * 18} y2={railY - 92} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <text
          x={massX + forceArrow + (forceArrow >= 0 ? 10 : -10)}
          y={railY - 120}
          textAnchor={forceArrow >= 0 ? "start" : "end"}
          className="cellengine-scene-force"
          style={{ fill: controllerColor }}
        >
          {controllerLabel}
        </text>
      </g>

      <g className="cellengine-overlay-card">
        <rect x="34" y="28" width="250" height="122" rx="20" fill="#ffffffdd" stroke="#d8e7e2" />
        <text x="54" y="56" className="cellengine-overlay-title">Frame telemetry</text>
        <text x="54" y="84" className="cellengine-overlay-line">tick {frame.tick} / {frames[frames.length - 1]?.tick ?? "n/a"}</text>
        <text x="54" y="108" className="cellengine-overlay-line">displacement {formatSigned(frame.x, 3, " m")}</text>
        <text x="54" y="132" className="cellengine-overlay-line">velocity {formatSigned(frame.x_dot, 3, " m/s")}</text>
      </g>

      <g className="cellengine-overlay-card">
        <rect x="688" y="28" width="238" height={56 + Math.max(1, (rightLines || []).length) * 24} rx="20" fill="#ffffffdd" stroke={controllerColor} opacity="0.9" />
        <text x="708" y="56" className="cellengine-overlay-title">{rightTitle}</text>
        {(rightLines || []).map((line, index) => (
          <text key={`${index}-${line}`} x="708" y={84 + index * 24} className="cellengine-overlay-line">{line}</text>
        ))}
      </g>

      {(frame.terminal === 1 || frame.terminal === true) ? (
        <g>
          <rect x="340" y="84" width="280" height="52" rx="24" fill="#7f1d1dcc" />
          <text x="480" y="117" textAnchor="middle" className="cellengine-scene-terminal">spring state destabilized</text>
        </g>
      ) : null}

      <g>
        <rect x="166" y="470" width="628" height="18" rx="9" fill="#e7ece9" />
        <rect x="184" y="476" width="592" height="6" rx="3" fill="#9fb3b8" />
        <line x1="480" y1="466" x2="480" y2="494" stroke="#475569" strokeDasharray="3 3" />
        <circle
          cx={184 + clamp01((frame.x + displacementSpan) / (2 * displacementSpan)) * 592}
          cy="479"
          r="8"
          fill="#d9480f"
        />
        <text x="166" y="512" className="cellengine-scene-label">
          displacement {formatSigned(frame.x, 2, " m")} across ±{formatNumber(displacementSpan, 2)} m observed span
        </text>
        <text x="166" y="532" className="cellengine-scene-detail">
          mass-spring task: the controlled body is the moving mass, and success means damping displacement without diverging
        </text>
      </g>
    </svg>
  );
}

export function WormReplayScene({
  frames,
  frameIndex,
  analysis,
  controllerLabel,
  controllerColor,
  rightTitle,
  rightLines,
  renderOrganismBody = false,
  organismBodyCells = null,
  organismFrameCellRows = null,
  organismFrameEdges = null,
  organismFunctionalStateByCellId = null,
  organismSelectedCellId = null,
  onSelectOrganismCell = null
}) {
  const frame = frames[frameIndex] || null;
  if (!frame) return <div className="empty">Run a replay to populate the viewer.</div>;
  const sceneWidth = 960;
  const sceneHeight = 540;
  const railY = sceneHeight - 110;
  const progress = typeof frame.x === "number" ? frame.x : 0;
  const progressSpan = Math.max(6, Math.abs(analysis?.goal_progress || progress || 0) + 2);
  const bodyX = 180 + clamp01((progress + 1.5) / Math.max(2, progressSpan + 1.5)) * 560;
  const bodyY = railY - 44;
  const strain = typeof frame.theta_rad === "number" ? frame.theta_rad : 0;
  const forceArrow = Math.max(-150, Math.min(150, (frame.organism_force || 0) * 14));
  return (
    <svg className="cellengine-scene" viewBox={`0 0 ${sceneWidth} ${sceneHeight}`} role="img" aria-label="CellEngine worm drag replay">
      <defs>
        <linearGradient id="wormSky" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#fff8ef" />
          <stop offset="60%" stopColor="#eef9f5" />
          <stop offset="100%" stopColor="#f8eef8" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={sceneWidth} height={sceneHeight} rx="30" fill="url(#wormSky)" />
      <rect x="76" y={railY + 10} width="808" height="18" rx="9" fill="#dce7e5" />
      <rect x="92" y={railY + 2} width="776" height="14" rx="7" fill="url(#cellengineTrack)" />
      <line x1="120" x2="120" y1="90" y2={railY + 20} stroke="#b8d2d1" strokeDasharray="5 8" />
      <line x1="810" x2="810" y1="90" y2={railY + 20} stroke="#d9485f" strokeDasharray="5 8" />
      <text x="120" y={railY + 52} className="cellengine-scene-label">start</text>
      <text x="780" y={railY + 52} className="cellengine-scene-alert">goal</text>
      {renderOrganismBody ? (
        <SceneOrganismBody
          pivotX={bodyX}
          pivotY={bodyY}
          theta={strain}
          bodyCells={organismBodyCells}
          frameCellRows={organismFrameCellRows}
          frameEdges={organismFrameEdges}
          functionalStateByCellId={organismFunctionalStateByCellId}
          selectedCellId={organismSelectedCellId}
          onSelectCell={onSelectOrganismCell}
        />
      ) : (
        <g>
          <rect x={bodyX - 58} y={bodyY - 28} width="116" height="48" rx="24" fill="#334155" />
          <circle cx={bodyX - 28} cy={bodyY - 4} r="12" fill="#0f172a" />
          <circle cx={bodyX} cy={bodyY - 2} r="11" fill="#334155" opacity="0.8" />
          <circle cx={bodyX + 28} cy={bodyY + 2} r="10" fill="#64748b" />
        </g>
      )}
      <g opacity="0.9">
        <line x1={bodyX} y1={bodyY - 86} x2={bodyX + forceArrow} y2={bodyY - 86} stroke={controllerColor} strokeWidth="10" strokeLinecap="round" />
        <line x1={bodyX + forceArrow} y1={bodyY - 86} x2={bodyX + forceArrow - Math.sign(forceArrow || 1) * 18} y2={bodyY - 98} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <line x1={bodyX + forceArrow} y1={bodyY - 86} x2={bodyX + forceArrow - Math.sign(forceArrow || 1) * 18} y2={bodyY - 74} stroke={controllerColor} strokeWidth="8" strokeLinecap="round" />
        <text x={bodyX + forceArrow + (forceArrow >= 0 ? 10 : -10)} y={bodyY - 102} textAnchor={forceArrow >= 0 ? "start" : "end"} className="cellengine-scene-force" style={{ fill: controllerColor }}>
          {controllerLabel}
        </text>
      </g>
      <g className="cellengine-overlay-card">
        <rect x="34" y="28" width="250" height="122" rx="20" fill="#ffffffdd" stroke="#d8e7e2" />
        <text x="54" y="56" className="cellengine-overlay-title">Frame telemetry</text>
        <text x="54" y="84" className="cellengine-overlay-line">tick {frame.tick} / {frames[frames.length - 1]?.tick ?? "n/a"}</text>
        <text x="54" y="108" className="cellengine-overlay-line">progress {formatSigned(frame.x, 2, " m")}</text>
        <text x="54" y="132" className="cellengine-overlay-line">bend {formatSigned(frame.theta_rad, 3)}</text>
      </g>
      <g className="cellengine-overlay-card">
        <rect x="688" y="28" width="238" height={56 + Math.max(1, (rightLines || []).length) * 24} rx="20" fill="#ffffffdd" stroke={controllerColor} opacity="0.9" />
        <text x="708" y="56" className="cellengine-overlay-title">{rightTitle}</text>
        {(rightLines || []).map((line, index) => (
          <text key={`${index}-${line}`} x="708" y={84 + index * 24} className="cellengine-overlay-line">{line}</text>
        ))}
      </g>
      <g>
        <rect x="166" y="470" width="628" height="18" rx="9" fill="#e7ece9" />
        <rect x="184" y="476" width="592" height="6" rx="3" fill="#9fb3b8" />
        <circle cx={184 + clamp01((progress + 1.5) / Math.max(2, progressSpan + 1.5)) * 592} cy="479" r="8" fill="#d9480f" />
        <text x="166" y="512" className="cellengine-scene-label">
          goal progress {formatSigned(progress, 2, " m")} · mean speed {formatSigned(frame.x_dot, 2, " m/s")}
        </text>
        <text x="166" y="532" className="cellengine-scene-detail">
          sea-cucumber crawl task: forward progress matters; body bend is the internal locomotion state
        </text>
      </g>
    </svg>
  );
}

export function PongReplayScene({
  frames,
  frameIndex,
  controllerLabel,
  controllerColor,
  rightTitle,
  rightLines,
  renderOrganismBody = false,
  organismBodyCells = null,
  organismFrameCellRows = null,
  organismFrameEdges = null,
  organismFunctionalStateByCellId = null,
  organismSelectedCellId = null,
  onSelectOrganismCell = null
}) {
  const frame = frames[frameIndex] || null;
  if (!frame) return <div className="empty">Run a replay to populate the viewer.</div>;
  const sceneWidth = 960;
  const sceneHeight = 540;
  const arena = { x: 120, y: 72, w: 720, h: 360 };
  // Physics coordinate → scene coordinate conversion.
  // The physics state uses s.x ∈ [-1.05, 1.05] for the paddle and
  // s.theta ∈ [-1.0, 1.0] for the ball (both y-axis).
  // task_aux_a ∈ [-1.15, 1.15] is the ball x-position (left wall → right paddle).
  const coordScale = arena.h / 2.2;
  const paddleY = arena.y + arena.h * 0.5 - clamp(frame.x, -1.05, 1.05) * coordScale;
  const ballY = arena.y + arena.h * 0.5 - clamp(frame.theta_rad || 0, -1.0, 1.0) * coordScale;
  // Ball x: travels from left wall (task_aux_a = -1.15) to right paddle (task_aux_a = 1.15).
  // Place the paddle/organism at a fixed x near the right edge. The ball travels all the way there.
  const paddleX = arena.x + arena.w - 70;
  const leftWallX = arena.x + 36;
  const ballT = clamp(((frame.task_aux_a || -1.15) + 1.15) / 2.3, 0, 1);
  const ballX = leftWallX + ballT * (paddleX - leftWallX);
  // Collision zone: paddle_half_height (default 0.22) mapped to scene coordinates.
  // This is the zone around the paddle y where a return succeeds.
  const paddleHalfPhysics = 0.22;
  const collisionHalfScene = paddleHalfPhysics * coordScale;
  // Contact detection: ball is at the paddle x and within the collision zone.
  const ballAtPaddle = ballT > 0.92;
  const miss = Math.abs((frame.theta_rad || 0) - frame.x);
  const isContact = ballAtPaddle && miss <= paddleHalfPhysics;
  // Ball trail: show recent positions.
  const trailLength = 8;
  const trailStart = Math.max(0, frameIndex - trailLength);
  const trailFrames = frames.slice(trailStart, frameIndex);
  return (
    <svg className="cellengine-scene" viewBox={`0 0 ${sceneWidth} ${sceneHeight}`} role="img" aria-label="CellEngine pong return replay">
      <defs>
        <radialGradient id="pongContactFlash" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#fef3c7" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect x="0" y="0" width={sceneWidth} height={sceneHeight} rx="30" fill="url(#cellengineSky)" />
      <rect x={arena.x} y={arena.y} width={arena.w} height={arena.h} rx="28" fill="#fcfaf3" stroke="#dbe4e2" strokeWidth="4" />
      <line x1={arena.x + arena.w / 2} x2={arena.x + arena.w / 2} y1={arena.y + 16} y2={arena.y + arena.h - 16} stroke="#d0d8d4" strokeDasharray="6 10" />
      {/* Collision zone indicator — the physics return window around the paddle */}
      <rect
        x={paddleX - 20}
        y={paddleY - collisionHalfScene}
        width="40"
        height={collisionHalfScene * 2}
        rx="6"
        fill={isContact ? "#10b98144" : "#0f766e18"}
        stroke={isContact ? "#10b981" : "#0f766e"}
        strokeWidth={isContact ? 2.5 : 1}
        strokeDasharray={isContact ? "0" : "4 3"}
      />
      {/* Organism body or fallback paddle */}
      {renderOrganismBody ? (
        <SceneOrganismBody
          pivotX={paddleX}
          pivotY={paddleY}
          theta={0}
          bodyCells={organismBodyCells}
          frameCellRows={organismFrameCellRows}
          frameEdges={organismFrameEdges}
          functionalStateByCellId={organismFunctionalStateByCellId}
          selectedCellId={organismSelectedCellId}
          onSelectCell={onSelectOrganismCell}
        />
      ) : (
        <rect x={paddleX - 8} y={paddleY - 54} width="16" height="108" rx="8" fill="#334155" />
      )}
      {/* Ball trail */}
      {trailFrames.map((tf, idx) => {
        const tBallT = clamp(((tf.task_aux_a || -1.15) + 1.15) / 2.3, 0, 1);
        const tBallX = leftWallX + tBallT * (paddleX - leftWallX);
        const tBallY = arena.y + arena.h * 0.5 - clamp(tf.theta_rad || 0, -1.0, 1.0) * coordScale;
        const alpha = 0.06 + 0.12 * (idx / Math.max(1, trailFrames.length));
        return (
          <circle
            key={`trail-${trailStart + idx}`}
            cx={tBallX}
            cy={tBallY}
            r={4 + 6 * (idx / Math.max(1, trailFrames.length))}
            fill="#f08c00"
            opacity={alpha}
          />
        );
      })}
      {/* Contact flash */}
      {isContact ? (
        <circle cx={ballX} cy={ballY} r="42" fill="url(#pongContactFlash)" opacity="0.85" />
      ) : null}
      {/* Ball */}
      <circle cx={ballX} cy={ballY} r="14" fill="#f08c00" stroke="#fff7e6" strokeWidth="4" />
      {/* Miss indicator — when ball is at paddle x but outside collision zone */}
      {ballAtPaddle && !isContact && !(frame.terminal === 1 || frame.terminal === true) ? (
        <line
          x1={paddleX - 16}
          y1={paddleY - collisionHalfScene}
          x2={paddleX - 16}
          y2={ballY}
          stroke="#dc262688"
          strokeWidth="2"
          strokeDasharray="4 3"
        />
      ) : null}
      <g className="cellengine-overlay-card">
        <rect x="34" y="28" width="250" height="146" rx="20" fill="#ffffffdd" stroke="#d8e7e2" />
        <text x="54" y="56" className="cellengine-overlay-title">Frame telemetry</text>
        <text x="54" y="84" className="cellengine-overlay-line">tick {frame.tick} / {frames[frames.length - 1]?.tick ?? "n/a"}</text>
        <text x="54" y="108" className="cellengine-overlay-line">returns {frame.task_counter ?? 0}</text>
        <text x="54" y="132" className="cellengine-overlay-line">paddle/ball {formatSigned(frame.x, 2)} / {formatSigned(frame.theta_rad, 2)}</text>
        <text x="54" y="156" className="cellengine-overlay-line">gap {formatNumber(miss, 3)} {miss <= paddleHalfPhysics ? "· in zone" : "· outside"}</text>
      </g>
      <g className="cellengine-overlay-card">
        <rect x="688" y="28" width="238" height={56 + Math.max(1, (rightLines || []).length) * 24} rx="20" fill="#ffffffdd" stroke={controllerColor} opacity="0.9" />
        <text x="708" y="56" className="cellengine-overlay-title">{rightTitle}</text>
        {(rightLines || []).map((line, index) => (
          <text key={`${index}-${line}`} x="708" y={84 + index * 24} className="cellengine-overlay-line">{line}</text>
        ))}
      </g>
      {(frame.terminal === 1 || frame.terminal === true) ? (
        <g>
          <rect x="356" y="440" width="248" height="52" rx="24" fill="#7f1d1dcc" />
          <text x="480" y="473" textAnchor="middle" className="cellengine-scene-terminal">missed return</text>
        </g>
      ) : null}
      <text x="166" y="496" className="cellengine-scene-label">
        paddle y {formatSigned(frame.x, 2)} · ball y {formatSigned(frame.theta_rad, 2)} · gap {formatNumber(miss, 3)} · returns {frame.task_counter ?? 0}
      </text>
      <text x="166" y="520" className="cellengine-scene-detail">
        pong return: dashed box = collision zone (paddle ± {paddleHalfPhysics}). ball returns if it enters the zone.
      </text>
    </svg>
  );
}

export function TaskReplayScene(props) {
  if (props.taskName === "mass_spring_balance") return <MassSpringReplayScene {...props} />;
  const family = taskMeta(props.taskName).stageFamily;
  if (family === "worm") return <WormReplayScene {...props} />;
  if (family === "pong") return <PongReplayScene {...props} />;
  return <ControllerReplayScene {...props} />;
}
