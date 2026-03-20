import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function clampNumber(value, lo, hi) {
  if (typeof value !== "number" || !Number.isFinite(value)) return lo;
  return Math.max(lo, Math.min(hi, value));
}

function parseActiveBitIndices(value) {
  if (Array.isArray(value)) {
    return value
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v) && v >= 0)
      .map((v) => Math.floor(v));
  }
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return [Math.floor(value)];
  }
  const text = String(value || "").trim();
  if (!text) return [];
  return text
    .split(/[;,|]/)
    .map((token) => Number(token))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .map((v) => Math.floor(v));
}

function decodeOneHotCells(activeIndices, cellCount, classes) {
  const cells = Array.from({ length: cellCount }, () => 0);
  if (classes <= 0 || cellCount <= 0) return cells;
  for (const idx of activeIndices) {
    const cell = Math.floor(idx / classes);
    const cls = idx % classes;
    if (cell >= 0 && cell < cellCount && cls >= 0 && cls < classes) {
      cells[cell] = cls;
    }
  }
  return cells;
}

function inferSquareBoardSize(obsBits, classes) {
  if (typeof obsBits !== "number" || !Number.isFinite(obsBits) || obsBits <= 0 || classes <= 0) return null;
  const cells = obsBits / classes;
  const side = Math.sqrt(cells);
  if (!Number.isFinite(side)) return null;
  const rounded = Math.round(side);
  if (rounded * rounded !== Math.round(cells)) return null;
  return rounded;
}

function decodeBenchmarkSnapshot(benchmarkId, obsBits, activeIndices) {
  const id = String(benchmarkId || "").toLowerCase();
  if (id === "tictactoe") {
    const cells = decodeOneHotCells(activeIndices, 9, 3);
    return { kind: "grid", rows: 3, cols: 3, cells, labels: [".", "X", "O"] };
  }
  if (id === "connect_four" || id === "connect4") {
    const cells = decodeOneHotCells(activeIndices, 6 * 7, 3);
    return { kind: "grid", rows: 6, cols: 7, cells, labels: [".", "R", "Y"] };
  }
  if (id === "go") {
    const side = inferSquareBoardSize(obsBits, 3) || 5;
    const cells = decodeOneHotCells(activeIndices, side * side, 3);
    return { kind: "grid", rows: side, cols: side, cells, labels: [".", "B", "W"] };
  }
  if (id === "battleship") {
    const side = inferSquareBoardSize(obsBits, 3) || 8;
    const cells = decodeOneHotCells(activeIndices, side * side, 3);
    return { kind: "grid", rows: side, cols: side, cells, labels: [".", "o", "x"] };
  }
  if (id === "chess") {
    const cells = decodeOneHotCells(activeIndices, 64, 13);
    return {
      kind: "grid",
      rows: 8,
      cols: 8,
      cells,
      labels: [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    };
  }
  if (id === "cartpole") {
    const bins = typeof obsBits === "number" && obsBits >= 8 ? Math.max(8, Math.round(obsBits / 4)) : 32;
    const dims = [
      { key: "x", bin: 0 },
      { key: "x_dot", bin: 0 },
      { key: "theta", bin: 0 },
      { key: "theta_dot", bin: 0 }
    ];
    for (const idx of activeIndices) {
      const dim = Math.floor(idx / bins);
      const bin = idx % bins;
      if (dim >= 0 && dim < dims.length && bin >= 0 && bin < bins) dims[dim].bin = bin;
    }
    return { kind: "cartpole", bins, dims };
  }
  return { kind: "raw", activeIndices };
}

function chessSquareName(index) {
  if (!Number.isFinite(index) || index < 0 || index >= 64) return "n/a";
  const file = String.fromCharCode(97 + (index % 8));
  const rank = 8 - Math.floor(index / 8);
  return `${file}${rank}`;
}

function formatBenchmarkAction(benchmarkId, action, obsBits) {
  if (typeof action !== "number" || !Number.isFinite(action) || action < 0) return "n/a";
  const id = String(benchmarkId || "").toLowerCase();
  if (id === "connect_four" || id === "connect4") return `col ${action + 1}`;
  if (id === "cartpole") return action === 1 ? "push right" : "push left";
  if (id === "chess") {
    const from = Math.floor(action / 64);
    const to = action % 64;
    return `${chessSquareName(from)} -> ${chessSquareName(to)}`;
  }
  if (id === "tictactoe" || id === "go" || id === "battleship") {
    const side = inferSquareBoardSize(obsBits, 3);
    if (!side) return String(action);
    const row = Math.floor(action / side);
    const col = action % side;
    return `r${row + 1} c${col + 1}`;
  }
  return String(action);
}

function outcomeLabel(done, outcome) {
  if (!done) return "ongoing";
  if (typeof outcome === "number" && outcome > 0) return "win";
  if (typeof outcome === "number" && outcome < 0) return "loss";
  return "draw";
}

export default function GameReplayPanel({ run, rows, isLoading, error, onRefresh }) {
  const [episodeChoice, setEpisodeChoice] = useState("latest");
  const [speedMode, setSpeedMode] = useState("realtime");
  const [humanDelayMs, setHumanDelayMs] = useState(700);
  const [playing, setPlaying] = useState(true);
  const [cursor, setCursor] = useState(0);
  const lastRunRef = useRef("");

  useEffect(() => {
    const nextRunId = run?.run_id || "";
    if (lastRunRef.current === nextRunId) return;
    lastRunRef.current = nextRunId;
    setEpisodeChoice("latest");
    setSpeedMode("realtime");
    setHumanDelayMs(700);
    setPlaying(true);
    setCursor(0);
  }, [run?.run_id]);

  const sortedRows = useMemo(() => {
    const source = Array.isArray(rows) ? rows : [];
    return [...source]
      .filter(
        (row) =>
          typeof row.episode === "number" &&
          Number.isFinite(row.episode) &&
          typeof row.step === "number" &&
          Number.isFinite(row.step)
      )
      .sort((a, b) => {
        if (a.episode !== b.episode) return a.episode - b.episode;
        if (a.step !== b.step) return a.step - b.step;
        const ag = typeof a.global_step === "number" ? a.global_step : 0;
        const bg = typeof b.global_step === "number" ? b.global_step : 0;
        return ag - bg;
      });
  }, [rows]);

  const episodes = useMemo(() => {
    const set = new Set();
    for (const row of sortedRows) {
      if (typeof row.episode === "number" && Number.isFinite(row.episode)) set.add(row.episode);
    }
    return [...set].sort((a, b) => a - b);
  }, [sortedRows]);

  useEffect(() => {
    if (episodeChoice === "latest") return;
    const numeric = Number(episodeChoice);
    if (!Number.isFinite(numeric) || !episodes.includes(numeric)) {
      setEpisodeChoice("latest");
    }
  }, [episodeChoice, episodes]);

  const selectedEpisode = useMemo(() => {
    if (!episodes.length) return null;
    if (episodeChoice === "latest") return episodes[episodes.length - 1];
    const numeric = Number(episodeChoice);
    return Number.isFinite(numeric) ? numeric : episodes[episodes.length - 1];
  }, [episodeChoice, episodes]);

  const episodeRows = useMemo(() => {
    if (selectedEpisode == null) return [];
    return sortedRows.filter((row) => row.episode === selectedEpisode);
  }, [sortedRows, selectedEpisode]);

  useEffect(() => {
    setCursor(0);
    if (speedMode !== "manual") setPlaying(true);
  }, [selectedEpisode]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (speedMode === "manual") setPlaying(false);
  }, [speedMode]);

  useEffect(() => {
    const maxIndex = Math.max(0, episodeRows.length - 1);
    setCursor((value) => Math.min(value, maxIndex));
  }, [episodeRows.length]);

  useEffect(() => {
    if (!episodeRows.length) return undefined;
    if (speedMode === "manual" || !playing) return undefined;
    if (cursor >= episodeRows.length - 1) return undefined;

    const current = episodeRows[cursor];
    const next = episodeRows[cursor + 1];
    let delayMs = clampNumber(humanDelayMs, 120, 4000);
    if (speedMode === "realtime") {
      const currT = typeof current.elapsed_ms === "number" ? current.elapsed_ms : null;
      const nextT = typeof next.elapsed_ms === "number" ? next.elapsed_ms : null;
      const dt = currT != null && nextT != null ? nextT - currT : 120;
      delayMs = clampNumber(Number.isFinite(dt) ? dt : 120, 20, 2000);
    }
    const timer = setTimeout(() => {
      setCursor((value) => Math.min(value + 1, episodeRows.length - 1));
    }, delayMs);
    return () => clearTimeout(timer);
  }, [cursor, episodeRows, speedMode, playing, humanDelayMs]);

  const current = episodeRows[cursor] || null;
  const benchmarkId = run?.benchmark_id || "";
  const obsBits =
    current && typeof current.obs_bits === "number" && Number.isFinite(current.obs_bits)
      ? current.obs_bits
      : null;
  const preState = useMemo(
    () => decodeBenchmarkSnapshot(benchmarkId, obsBits, parseActiveBitIndices(current?.obs_pre_active)),
    [benchmarkId, obsBits, current?.obs_pre_active]
  );
  const postState = useMemo(
    () => decodeBenchmarkSnapshot(benchmarkId, obsBits, parseActiveBitIndices(current?.obs_post_active)),
    [benchmarkId, obsBits, current?.obs_post_active]
  );

  const changedCells = useMemo(() => {
    if (preState?.kind !== "grid" || postState?.kind !== "grid") return new Set();
    if (!Array.isArray(preState.cells) || !Array.isArray(postState.cells)) return new Set();
    if (preState.cells.length !== postState.cells.length) return new Set();
    const changed = new Set();
    for (let i = 0; i < postState.cells.length; i += 1) {
      if (preState.cells[i] !== postState.cells[i]) changed.add(i);
    }
    return changed;
  }, [preState, postState]);

  const timelineRows = useMemo(
    () =>
      episodeRows.map((row, idx) => ({
        move: idx + 1,
        reward: typeof row.reward === "number" ? row.reward : null,
        epsilon: typeof row.epsilon === "number" ? row.epsilon : null
      })),
    [episodeRows]
  );

  const canStepBack = cursor > 0;
  const canStepForward = cursor < episodeRows.length - 1;

  return (
    <section className="panel stack">
      <div className="row controls">
        <label>
          Episode
          <select value={String(episodeChoice)} onChange={(event) => setEpisodeChoice(event.target.value)}>
            <option value="latest">latest</option>
            {episodes.map((episode) => (
              <option key={episode} value={String(episode)}>
                {episode}
              </option>
            ))}
          </select>
        </label>

        <label>
          Speed
          <select value={speedMode} onChange={(event) => setSpeedMode(event.target.value)}>
            <option value="realtime">real-time (training pace)</option>
            <option value="human">human</option>
            <option value="manual">free hand</option>
          </select>
        </label>

        {speedMode === "human" && (
          <label>
            Human Delay
            <select
              value={String(humanDelayMs)}
              onChange={(event) => setHumanDelayMs(Math.max(120, Number(event.target.value) || 700))}
            >
              {[250, 400, 700, 1000, 1500].map((v) => (
                <option key={v} value={String(v)}>
                  {v} ms
                </option>
              ))}
            </select>
          </label>
        )}

        <div className="row controls game-controls">
          <button className="mini-btn" type="button" onClick={() => setCursor(0)} disabled={!episodeRows.length || !canStepBack}>
            {"|<"}
          </button>
          <button
            className="mini-btn"
            type="button"
            onClick={() => {
              setSpeedMode("manual");
              setCursor((value) => Math.max(0, value - 1));
            }}
            disabled={!episodeRows.length || !canStepBack}
          >
            {"<"}
          </button>
          {speedMode !== "manual" && (
            <button className="mini-btn" type="button" onClick={() => setPlaying((value) => !value)} disabled={!episodeRows.length}>
              {playing ? "Pause" : "Play"}
            </button>
          )}
          <button
            className="mini-btn"
            type="button"
            onClick={() => {
              setSpeedMode("manual");
              setCursor((value) => Math.min(value + 1, episodeRows.length - 1));
            }}
            disabled={!episodeRows.length || !canStepForward}
          >
            {">"}
          </button>
          <button
            className="mini-btn"
            type="button"
            onClick={() => setCursor(Math.max(0, episodeRows.length - 1))}
            disabled={!episodeRows.length || !canStepForward}
          >
            {">|"}
          </button>
        </div>

        <button className="mini-btn" type="button" onClick={onRefresh}>
          Refresh trace
        </button>
        <span className="meta-note">
          moves: {episodeRows.length} · frame: {episodeRows.length ? cursor + 1 : 0}/{episodeRows.length}
        </span>
      </div>

      {error && <div className="empty">{error}</div>}
      {isLoading && !rows?.length ? (
        <div className="empty">Loading game trace...</div>
      ) : !episodeRows.length ? (
        <div className="empty">
          No game trace rows found for this run. Start a new run to generate `*_game_trace.csv`.
        </div>
      ) : (
        <>
          <div className="card stack">
            <div className="row controls">
              <span className="meta-note">episode {selectedEpisode}</span>
              <span className="meta-note">
                step {current?.step ?? "n/a"} · global step {current?.global_step ?? "n/a"}
              </span>
              <span className="meta-note">action: {formatBenchmarkAction(benchmarkId, current?.action, obsBits)}</span>
              <span className="meta-note">
                teacher: {formatBenchmarkAction(benchmarkId, current?.teacher_action, obsBits)}
              </span>
              <span className="meta-note">
                immediate win: {formatBenchmarkAction(benchmarkId, current?.immediate_win_action, obsBits)}
              </span>
              <span className="meta-note">
                immediate block: {formatBenchmarkAction(benchmarkId, current?.immediate_block_action, obsBits)}
              </span>
              <span className="meta-note">
                reward: {typeof current?.reward === "number" ? current.reward.toFixed(4) : "n/a"}
              </span>
              <span className="meta-note">outcome: {outcomeLabel(current?.done === 1 || current?.done === true, current?.outcome)}</span>
            </div>

            {postState?.kind === "grid" ? (
              <div
                className="game-grid"
                style={{ gridTemplateColumns: `repeat(${postState.cols}, minmax(30px, 1fr))` }}
              >
                {postState.cells.map((value, idx) => {
                  const label = postState.labels?.[value] ?? String(value);
                  return (
                    <div
                      key={`cell-${idx}`}
                      className={`game-cell game-v-${value}${changedCells.has(idx) ? " changed" : ""}`}
                      title={`index ${idx}`}
                    >
                      {label}
                    </div>
                  );
                })}
              </div>
            ) : postState?.kind === "cartpole" ? (
              <div className="stack">
                {postState.dims.map((dim) => {
                  const norm = postState.bins > 1 ? dim.bin / (postState.bins - 1) : 0;
                  return (
                    <div key={dim.key} className="cartpole-row">
                      <span className="cartpole-key">{dim.key}</span>
                      <div className="cartpole-bar">
                        <div className="cartpole-fill" style={{ width: `${Math.max(2, norm * 100)}%` }} />
                      </div>
                      <span className="cartpole-val">
                        {dim.bin}/{postState.bins - 1}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty">No board renderer for benchmark `{benchmarkId}`.</div>
            )}
          </div>

          <div className="chart">
            <h3>Move Timeline</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineRows}>
                <CartesianGrid strokeDasharray="4 4" />
                <XAxis dataKey="move" />
                <YAxis yAxisId="reward" />
                <YAxis yAxisId="epsilon" orientation="right" domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <ReferenceLine x={cursor + 1} stroke="#495057" strokeDasharray="3 3" />
                <Line yAxisId="reward" type="monotone" dataKey="reward" stroke="#a61e4d" dot={false} name="reward" />
                <Line yAxisId="epsilon" type="monotone" dataKey="epsilon" stroke="#0c8599" dot={false} name="epsilon" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </section>
  );
}

