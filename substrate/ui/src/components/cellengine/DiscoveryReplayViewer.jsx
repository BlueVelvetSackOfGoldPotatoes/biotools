import { useCallback, useEffect, useRef, useState } from "react";
import { formatNumber, formatPct, formatSigned, PLAYBACK_FPS, SPEED_OPTIONS, taskLabel, taskMeta } from "./core";
import { TaskReplayScene } from "./replayScenes";
import { SectionTitle } from "./uiPrimitives";
import { requestJson } from "../../lib/requestJson";

const replayCache = new Map();

export function DiscoveryReplayViewer({
  genomePath,
  taskName,
  label,
  onClose
}) {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const animRef = useRef(null);
  const lastTickRef = useRef(0);

  useEffect(() => {
    if (!genomePath) return;
    const cacheKey = `${genomePath}::${taskName}`;
    if (replayCache.has(cacheKey)) {
      setPayload(replayCache.get(cacheKey));
      setCurrentIndex(0);
      setPlaying(true);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    requestJson("/api/cellengine/replay", {
      method: "POST",
      body: {
        task_name: taskName,
        genome_path: genomePath,
        max_ticks: 500,
        seed: 42,
        rl_algo: "a2c"
      }
    })
      .then((result) => {
        if (cancelled) return;
        replayCache.set(cacheKey, result);
        setPayload(result);
        setCurrentIndex(0);
        setPlaying(true);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(String(err?.message || err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [genomePath, taskName]);

  const frames = payload?.frames || [];
  const rlFrames = payload?.rl_frames || [];
  const bodyCells = payload?.body_cells || [];
  const analysis = payload?.analysis || null;
  const rlAnalysis = payload?.rl_analysis || null;
  const frame = frames[currentIndex] || null;
  const rlFrame = rlFrames[Math.min(currentIndex, Math.max(0, rlFrames.length - 1))] || null;
  const maxIndex = Math.max(0, frames.length - 1);
  const meta = taskMeta(taskName);

  useEffect(() => {
    if (!playing || !frames.length) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }
    const intervalMs = 1000 / (PLAYBACK_FPS * speed);
    const tick = (ts) => {
      if (ts - lastTickRef.current >= intervalMs) {
        lastTickRef.current = ts;
        setCurrentIndex((prev) => {
          const next = prev + 1;
          if (next > maxIndex) {
            setPlaying(false);
            return maxIndex;
          }
          return next;
        });
      }
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [playing, frames.length, maxIndex, speed]);

  const togglePlay = useCallback(() => {
    if (!frames.length) return;
    if (currentIndex >= maxIndex) {
      setCurrentIndex(0);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }, [currentIndex, maxIndex, frames.length]);

  return (
    <div className="cellengine-replay-viewer-backdrop" onClick={onClose}>
      <div className="cellengine-replay-viewer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cellengine-replay-viewer-header">
          <div>
            <SectionTitle
              title="Organism Replay"
              help="On-demand replay of the selected genome solving the task. The scene shows the organism controller on the left and the RL comparator on the right."
            />
            <span className="meta-note">{label || genomePath} · {taskLabel(taskName)}</span>
          </div>
          <button type="button" className="mini-btn" onClick={onClose}>close</button>
        </div>

        {loading ? (
          <div className="cellengine-replay-viewer-loading">
            <div className="cellengine-replay-viewer-spinner" />
            <span>Running replay for {taskLabel(taskName)}...</span>
            <span className="meta-note">generating trace from genome · this takes a few seconds</span>
          </div>
        ) : error ? (
          <div className="cellengine-replay-viewer-error">
            <strong>Replay failed</strong>
            <span>{error}</span>
          </div>
        ) : frames.length ? (
          <>
            <div className="cellengine-replay-viewer-scenes">
              <div className="cellengine-replay-viewer-scene-card">
                <strong>Cell Controller</strong>
                <TaskReplayScene
                  taskName={taskName}
                  frames={frames}
                  frameIndex={currentIndex}
                  analysis={analysis}
                  controllerLabel="organism"
                  controllerColor="#a61e4d"
                  rightTitle="Cell tissue"
                  rightLines={[
                    `active ${formatPct(frame?.active_fraction, 1)}`,
                    `energy ${formatNumber(frame?.mean_energy, 3)}`,
                    meta.stageFamily === "worm"
                      ? `progress ${formatSigned(frame?.x, 2, " m")}`
                      : meta.stageFamily === "pong"
                        ? `returns ${frame?.task_counter ?? 0}`
                        : `stress ${formatNumber(frame?.mean_stress, 3)}`
                  ]}
                  renderOrganismBody
                  organismBodyCells={bodyCells}
                />
              </div>
              {rlFrames.length ? (
                <div className="cellengine-replay-viewer-scene-card">
                  <strong>RL Comparator ({(payload?.summary?.rl_algorithm_selected || "a2c").toUpperCase()})</strong>
                  <TaskReplayScene
                    taskName={taskName}
                    frames={rlFrames}
                    frameIndex={Math.min(currentIndex, Math.max(0, rlFrames.length - 1))}
                    analysis={rlAnalysis}
                    controllerLabel={(payload?.summary?.rl_algorithm_selected || "rl").toUpperCase()}
                    controllerColor="#155e75"
                    rightTitle="RL state"
                    rightLines={[
                      `action ${rlFrame?.action ?? "n/a"}`,
                      `force ${formatSigned(rlFrame?.total_force, 2)}`,
                      meta.stageFamily === "worm"
                        ? `progress ${formatSigned(rlFrame?.x, 2, " m")}`
                        : meta.stageFamily === "pong"
                          ? `returns ${rlFrame?.task_counter ?? 0}`
                          : `theta ${formatNumber(rlFrame?.theta_deg, 2)}°`
                    ]}
                  />
                </div>
              ) : null}
            </div>

            <div className="cellengine-replay-viewer-controls">
              <button type="button" className="mini-btn" onClick={() => { setCurrentIndex(0); setPlaying(false); }}>|&lt;</button>
              <button type="button" className="mini-btn" onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}>&lt;</button>
              <button type="button" className="mini-btn replay-play-btn" onClick={togglePlay}>{playing ? "pause" : currentIndex >= maxIndex ? "replay" : "play"}</button>
              <button type="button" className="mini-btn" onClick={() => setCurrentIndex((i) => Math.min(maxIndex, i + 1))}>&gt;</button>
              <button type="button" className="mini-btn" onClick={() => { setCurrentIndex(maxIndex); setPlaying(false); }}>&gt;|</button>
              <input
                type="range"
                className="cellengine-replay-viewer-scrubber"
                min={0}
                max={maxIndex}
                value={currentIndex}
                onChange={(e) => { setCurrentIndex(Number(e.target.value)); setPlaying(false); }}
              />
              <span className="cellengine-replay-viewer-tick">tick {currentIndex} / {maxIndex}</span>
              <div className="cellengine-replay-viewer-speed">
                {SPEED_OPTIONS.map((s) => (
                  <button
                    key={`speed-${s}`}
                    type="button"
                    className={`mini-btn${speed === s ? " active" : ""}`}
                    onClick={() => setSpeed(s)}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>

            <div className="cellengine-replay-viewer-summary">
              <span>success: <strong>{payload?.summary?.solved ? "PASS" : "FAIL"}</strong></span>
              <span>survival: <strong>{formatPct(payload?.summary?.cell_clean?.survival_ratio ?? payload?.analysis?.survival_ratio, 1)}</strong></span>
              <span>cells: <strong>{bodyCells.length}</strong></span>
              <span>body: <strong>{payload?.summary?.body_mode || "n/a"}</strong></span>
              {analysis?.max_abs_theta_deg != null ? (
                <span>peak angle: <strong>{formatNumber(analysis.max_abs_theta_deg, 2)}°</strong></span>
              ) : null}
            </div>
          </>
        ) : (
          <div className="empty">No replay data available.</div>
        )}
      </div>
    </div>
  );
}
