import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { buildGroupedSeries, chooseAutoPreviewColumns } from "../../lib/dashboardShared";
import {
  MlpVisualizations,
  TransformerVisualizations,
  RnnVisualizations,
  LstmVisualizations,
  ViTVisualizations,
  TreeVisualizations,
  ForestVisualizations,
  MarkovVisualizations,
  CnnVisualizations
} from "./FamilyVisualizationsCore";
import {
  ClusteringVisualizations,
  HebbianVisualizations,
  ActorCriticVisualizations,
  DiffusionVisualizations,
  GnnVisualizations,
  ForwardForwardVisualizations
} from "./FamilyVisualizationsAdvanced";
import {
  ReinforcementVisualizations,
  ContinuousVisualizations,
  TicTacToeVisualizations
} from "./FamilyVisualizationsGames";

function FamilyVisualizations({ family, modelSpecific }) {
  if (!modelSpecific || !Object.keys(modelSpecific).length) return null;

  const isTicTacToeRun =
    Array.isArray(modelSpecific.tictactoe_policy_metrics) &&
    modelSpecific.tictactoe_policy_metrics.length > 0;
  if (isTicTacToeRun) {
    return <TicTacToeVisualizations data={modelSpecific} />;
  }

  const vizMap = {
    mlp: MlpVisualizations,
    cnn: CnnVisualizations,
    rnn: RnnVisualizations,
    lstm: LstmVisualizations,
    transformer: TransformerVisualizations,
    vit: ViTVisualizations,
    trees: TreeVisualizations,
    forests: ForestVisualizations,
    markov: MarkovVisualizations,
    clustering: ClusteringVisualizations,
    hebbian: HebbianVisualizations,
    actor_critic: ActorCriticVisualizations,
    diffusion: DiffusionVisualizations,
    gnn: GnnVisualizations,
    forward_forward: ForwardForwardVisualizations,
    reinforcement: ReinforcementVisualizations,
    continuous: ContinuousVisualizations,
    hybrid: ContinuousVisualizations
  };

  const Viz = vizMap[family];
  if (!Viz) {
    const keys = Object.keys(modelSpecific);
    return (
      <div className="detail-viz-grid">
        {keys.slice(0, 4).map((key) => {
          const rows = modelSpecific[key] || [];
          const cols = chooseAutoPreviewColumns(rows);
          if (!cols || !rows.length) return null;
          const series = buildGroupedSeries(rows, cols.x, cols.y, cols.group, 4);
          return (
            <div key={key} className="detail-viz-card">
              <h4>{key}</h4>
              {series.data.length ? (
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={series.data}>
                    <CartesianGrid strokeDasharray="4 4" />
                    <XAxis dataKey="x" type="number" />
                    <YAxis />
                    <Tooltip />
                    {series.lines.map((line) => (
                      <Line key={line.key} dataKey={line.key} stroke={line.color} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : <div className="empty">No data</div>}
            </div>
          );
        })}
      </div>
    );
  }

  return <Viz data={modelSpecific} />;
}

export default FamilyVisualizations;
