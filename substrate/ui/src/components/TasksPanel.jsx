import { useEffect, useMemo, useState } from "react";
import {
  CONTROL_GAME_BENCHMARKS,
  TASK_BENCHMARK_OPTIONS,
  formatTime,
  taskStatusClass,
  taskStatusLabel
} from "../lib/dashboardShared";

export default function TasksPanel({
  tasks,
  runs,
  selectedRunId,
  loading,
  busyKey,
  actionError,
  actionMessage,
  onRefresh,
  onCreateTrain,
  onCreateEvaluate,
  onKillTask,
  onStartTask,
  onDeleteTask,
  onOpenRun
}) {
  const [trainBenchmark, setTrainBenchmark] = useState("mnist");
  const [trainModelFamily, setTrainModelFamily] = useState("mlp");
  const [episodes, setEpisodes] = useState("");
  const [evalEvery, setEvalEvery] = useState("");
  const [evalEpisodes, setEvalEpisodes] = useState("");
  const [maxSteps, setMaxSteps] = useState("");
  const [controlMode, setControlMode] = useState("full_side");
  const [pieceModels, setPieceModels] = useState("");
  const [evaluateRunId, setEvaluateRunId] = useState(selectedRunId || "");
  const [evaluateFormat, setEvaluateFormat] = useState("png");
  const [kindFilter, setKindFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  const modelFamilies = useMemo(() => {
    const merged = new Set([
      "mlp",
      "cnn",
      "rnn",
      "gru",
      "lstm",
      "transformer",
      "vit",
      "hebbian",
      "actor_critic",
      "reinforcement",
      "markov",
      "gnn",
      "forward_forward",
      "diffusion",
      "trees",
      "forests",
      "clustering",
      "continuous",
      "hybrid"
    ]);
    for (const run of runs || []) {
      if (typeof run.model_family === "string" && run.model_family.trim()) {
        merged.add(run.model_family.trim());
      }
    }
    return [...merged].sort();
  }, [runs]);

  const sortedRuns = useMemo(
    () =>
      [...(runs || [])].sort((a, b) => {
        if (a.run_id < b.run_id) return 1;
        if (a.run_id > b.run_id) return -1;
        return 0;
      }),
    [runs]
  );

  const sortedTasks = useMemo(
    () =>
      [...(tasks || [])].sort((a, b) => {
        const at = Date.parse(a.created_utc || "") || 0;
        const bt = Date.parse(b.created_utc || "") || 0;
        return bt - at;
      }),
    [tasks]
  );

  const activeTasks = useMemo(
    () => sortedTasks.filter((task) => ["queued", "running", "killing"].includes(task.status)),
    [sortedTasks]
  );

  const visibleTasks = useMemo(
    () =>
      sortedTasks.filter((task) => {
        if (kindFilter !== "all" && task.kind !== kindFilter) return false;
        if (statusFilter !== "all" && task.status !== statusFilter) return false;
        return true;
      }),
    [sortedTasks, kindFilter, statusFilter]
  );

  useEffect(() => {
    if (!modelFamilies.length) return;
    if (!modelFamilies.includes(trainModelFamily)) {
      setTrainModelFamily(modelFamilies[0]);
    }
  }, [modelFamilies, trainModelFamily]);

  useEffect(() => {
    if (!sortedRuns.length) {
      if (evaluateRunId) setEvaluateRunId("");
      return;
    }
    if (!evaluateRunId) {
      setEvaluateRunId(selectedRunId || sortedRuns[0].run_id);
      return;
    }
    if (!sortedRuns.some((run) => run.run_id === evaluateRunId)) {
      setEvaluateRunId(selectedRunId || sortedRuns[0].run_id);
    }
  }, [sortedRuns, evaluateRunId, selectedRunId]);

  const isControlBenchmark = CONTROL_GAME_BENCHMARKS.has(trainBenchmark);
  const requiresEpisodeParams = trainBenchmark === "tictactoe" || isControlBenchmark;
  const hasTrainBusy = busyKey === "create-train";
  const hasEvalBusy = busyKey === "create-eval";

  const startTrain = () => {
    const payload = {
      benchmark_id: trainBenchmark,
      model_family: trainModelFamily
    };
    if (requiresEpisodeParams) {
      if (episodes.trim()) payload.episodes = Number(episodes);
      if (evalEvery.trim()) payload.eval_every = Number(evalEvery);
      if (evalEpisodes.trim()) payload.eval_episodes = Number(evalEpisodes);
      if (maxSteps.trim()) payload.max_steps = Number(maxSteps);
    }
    if (isControlBenchmark) {
      payload.control_mode = controlMode;
      if (pieceModels.trim()) payload.piece_models = pieceModels.trim();
    }
    onCreateTrain(payload);
  };

  const startEvaluate = () => {
    if (!evaluateRunId) return;
    onCreateEvaluate({
      run_id: evaluateRunId,
      format: evaluateFormat
    });
  };

  return (
    <section className="panel stack">
      <div className="row controls">
        <strong>Task Orchestration</strong>
        <span className="meta-note">{activeTasks.length} active tasks</span>
        <button className="mini-btn" type="button" onClick={onRefresh}>
          Refresh tasks
        </button>
      </div>

      {actionError ? <div className="error"><div>{actionError}</div></div> : null}
      {actionMessage ? <div className="empty">{actionMessage}</div> : null}

      <div className="grid two">
        <div className="card">
          <h3>Start Training Task</h3>
          <div className="card-grid">
            <span>benchmark</span>
            <select value={trainBenchmark} onChange={(event) => setTrainBenchmark(event.target.value)}>
              {TASK_BENCHMARK_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>{option.label}</option>
              ))}
            </select>

            <span>model family</span>
            <select value={trainModelFamily} onChange={(event) => setTrainModelFamily(event.target.value)}>
              {modelFamilies.map((family) => (
                <option key={family} value={family}>{family}</option>
              ))}
            </select>

            {requiresEpisodeParams ? (
              <>
                <span>episodes</span>
                <input
                  className="task-input"
                  value={episodes}
                  onChange={(event) => setEpisodes(event.target.value)}
                  placeholder="optional"
                />

                <span>eval every</span>
                <input
                  className="task-input"
                  value={evalEvery}
                  onChange={(event) => setEvalEvery(event.target.value)}
                  placeholder="optional"
                />

                <span>eval episodes</span>
                <input
                  className="task-input"
                  value={evalEpisodes}
                  onChange={(event) => setEvalEpisodes(event.target.value)}
                  placeholder="optional"
                />

                <span>max steps</span>
                <input
                  className="task-input"
                  value={maxSteps}
                  onChange={(event) => setMaxSteps(event.target.value)}
                  placeholder="optional"
                />
              </>
            ) : null}

            {isControlBenchmark ? (
              <>
                <span>control mode</span>
                <select value={controlMode} onChange={(event) => setControlMode(event.target.value)}>
                  <option value="full_side">full_side</option>
                  <option value="per_piece">per_piece</option>
                </select>

                <span>piece models</span>
                <input
                  className="task-input"
                  value={pieceModels}
                  onChange={(event) => setPieceModels(event.target.value)}
                  placeholder="king=transformer,queen=cnn"
                />
              </>
            ) : null}
          </div>
          <div className="row controls">
            <button className="mini-btn" type="button" onClick={startTrain} disabled={hasTrainBusy}>
              {hasTrainBusy ? "Starting..." : "Start Train"}
            </button>
          </div>
        </div>

        <div className="card">
          <h3>Start Evaluation Task</h3>
          <div className="card-grid">
            <span>run</span>
            <select
              value={evaluateRunId}
              onChange={(event) => setEvaluateRunId(event.target.value)}
              disabled={!sortedRuns.length}
            >
              {!sortedRuns.length && <option value="">no runs</option>}
              {sortedRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} [{run.model_family}]
                </option>
              ))}
            </select>

            <span>format</span>
            <select value={evaluateFormat} onChange={(event) => setEvaluateFormat(event.target.value)}>
              <option value="png">png</option>
              <option value="svg">svg</option>
            </select>
          </div>
          <div className="row controls">
            <button
              className="mini-btn"
              type="button"
              onClick={startEvaluate}
              disabled={hasEvalBusy || !evaluateRunId}
            >
              {hasEvalBusy ? "Starting..." : "Start Evaluate"}
            </button>
            {evaluateRunId ? (
              <button className="mini-btn" type="button" onClick={() => onOpenRun(evaluateRunId)}>
                Open run
              </button>
            ) : null}
          </div>
        </div>
      </div>

      <div className="row controls">
        <label>
          kind
          <select value={kindFilter} onChange={(event) => setKindFilter(event.target.value)}>
            <option value="all">all</option>
            <option value="train">train</option>
            <option value="evaluate">evaluate</option>
          </select>
        </label>
        <label>
          status
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
            <option value="all">all</option>
            <option value="queued">queued</option>
            <option value="running">running</option>
            <option value="killing">killing</option>
            <option value="completed">completed</option>
            <option value="failed">failed</option>
            <option value="killed">killed</option>
          </select>
        </label>
        <span className="meta-note">showing {visibleTasks.length} tasks</span>
      </div>

      {loading && !sortedTasks.length ? (
        <div className="empty">Loading tasks...</div>
      ) : visibleTasks.length ? (
        <div className="table-wrap task-table-wrap">
          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>kind</th>
                <th>status</th>
                <th>benchmark</th>
                <th>family</th>
                <th>run(s)</th>
                <th>created</th>
                <th>ended</th>
                <th>command</th>
                <th>actions</th>
              </tr>
            </thead>
            <tbody>
              {visibleTasks.map((task) => {
                const isTerminal = ["completed", "failed", "killed"].includes(task.status);
                const isKilling = task.status === "killing";
                const actionKeyStart = `start:${task.task_id}`;
                const actionKeyKill = `kill:${task.task_id}`;
                const actionKeyDelete = `delete:${task.task_id}`;
                return (
                  <tr key={task.task_id}>
                    <td className="code-lite">{task.task_id}</td>
                    <td>{task.kind || "n/a"}</td>
                    <td>
                      <span className={`pill ${taskStatusClass(task.status)}`}>
                        {taskStatusLabel(task.status)}
                      </span>
                    </td>
                    <td>{task.benchmark_id || "n/a"}</td>
                    <td>{task.model_family || "n/a"}</td>
                    <td>
                      {(task.run_ids || []).length ? (
                        <div className="task-actions">
                          {task.run_ids.map((run) => (
                            <button key={run} className="mini-btn" type="button" onClick={() => onOpenRun(run)}>
                              {run}
                            </button>
                          ))}
                        </div>
                      ) : (
                        task.target_run_id || "n/a"
                      )}
                    </td>
                    <td>{formatTime(task.created_utc)}</td>
                    <td>{formatTime(task.ended_utc)}</td>
                    <td className="code-lite">{task.command_display || "n/a"}</td>
                    <td>
                      <div className="task-actions">
                        {!isTerminal ? (
                          <button
                            className="mini-btn"
                            type="button"
                            disabled={isKilling || busyKey === actionKeyKill}
                            onClick={() => onKillTask(task.task_id)}
                          >
                            {busyKey === actionKeyKill || isKilling ? "Killing..." : "Kill"}
                          </button>
                        ) : (
                          <>
                            <button
                              className="mini-btn"
                              type="button"
                              disabled={busyKey === actionKeyStart}
                              onClick={() => onStartTask(task.task_id)}
                            >
                              {busyKey === actionKeyStart ? "Starting..." : "Start"}
                            </button>
                            <button
                              className="mini-btn"
                              type="button"
                              disabled={busyKey === actionKeyDelete}
                              onClick={() => onDeleteTask(task.task_id)}
                            >
                              {busyKey === actionKeyDelete ? "Deleting..." : "Delete"}
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">No tasks yet. Start a training or evaluation task above.</div>
      )}
    </section>
  );
}
