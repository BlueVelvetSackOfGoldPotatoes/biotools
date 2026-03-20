import {
  CELL_ROLE_LEGEND,
  CELL_MATRIX_METRICS,
  DNA_BASE_META,
  FUNCTIONAL_STATE_LEGEND,
  FUNCTIONAL_STATE_META,
  GENE_PROGRAM_BASE,
  GENOME_METRICS,
  cellTypeStroke,
  staticMetricColor,
  genomeProgramFill,
  genomeProgramGradient,
  genomeProgramMeta,
  formatGenomeMetricLabel
} from "./core";
import { SectionTitle } from "./uiPrimitives";
export function GenomeDnaLegend({ activeMetricId = "" }) {
  const activeBase = genomeProgramMeta(activeMetricId)?.base || "";
  return (
    <div className="cellengine-dna-legend" aria-label="DNA-style genome palette legend">
      {Object.values(DNA_BASE_META).map((item) => (
        <div
          key={item.base}
          className={`cellengine-dna-chip${activeBase === item.base ? " active" : ""}`}
          style={{ "--dna-fill": item.fill, "--dna-ink": item.ink, "--dna-pale": item.pale }}
          title={`${item.base} = ${item.label}`}
        >
          <strong>{item.base}</strong>
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
}

export function CellRoleLegend() {
  return (
    <div className="cellengine-role-legend" aria-label="Cell structural role legend">
      {CELL_ROLE_LEGEND.map((item) => (
        <div key={item.id} className="cellengine-role-chip">
          <span className="cellengine-role-swatch" style={{ "--role-stroke": item.stroke }} />
          <div>
            <strong>{item.label}</strong>
            <span>{item.note}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function FunctionalStateLegend() {
  return (
    <div className="cellengine-functional-legend" aria-label="Cell function legend">
      {FUNCTIONAL_STATE_LEGEND.map((item) => {
        const meta = FUNCTIONAL_STATE_META[item.id];
        return (
          <div key={item.id} className="cellengine-functional-chip">
            <span className="cellengine-functional-chip-swatch" style={{ background: meta.fill, color: meta.ink }}>
              {meta.symbol}
            </span>
            <div>
              <strong>{meta.label}</strong>
              <span>{item.note}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function CellMatrixLegend() {
  return (
    <div className="cellengine-legend-stack">
      <div className="cellengine-matrix-slot-grid">
        {CELL_MATRIX_METRICS.map((metric) => (
          <div key={metric.id} className="cellengine-matrix-slot">
            <strong>{metric.short}</strong>
            <span>{metric.label}</span>
          </div>
        ))}
      </div>
      <div className="cellengine-matrix-legend-notes">
        <div><strong>Border</strong><span>cell role: motor, hinge, ground, scaffold</span></div>
        <div><strong>Bottom-right dot</strong><span>green = active now, gray = inactive now</span></div>
        <div><strong>Diagonal cross</strong><span>inactive cell at this frame</span></div>
        <div><strong>Warm vs cool</strong><span>signed channels use warm positive and cool negative colors</span></div>
      </div>
    </div>
  );
}

export function GenomeMetricScale({ metricId }) {
  const start = staticMetricColor(metricId, 0);
  const mid = staticMetricColor(metricId, 0.5);
  const end = staticMetricColor(metricId, 1);
  return (
    <div className="cellengine-legend-scale">
      <div
        className="cellengine-legend-scale-bar"
        style={{ background: `linear-gradient(90deg, ${start} 0%, ${mid} 50%, ${end} 100%)` }}
      />
      <div className="cellengine-legend-scale-labels">
        <span>low</span>
        <strong>{formatGenomeMetricLabel(metricId)}</strong>
        <span>high</span>
      </div>
    </div>
  );
}

export function SharedVisualizationLegend({
  title,
  note,
  genomeMetric,
  showGenome = false,
  showFunction = false,
  showRoles = false,
  showMatrix = false,
  showGraph = false,
  showDevelopment = false
}) {
  return (
    <div className="card cellengine-legend-card">
      <div className="section-head">
        <SectionTitle
          title={title}
          help="Shared reading rules for the visualizations in this section. The same legend is not repeated inside every repeated organism or cell view."
        />
        {note ? <span className="meta-note">{note}</span> : null}
      </div>
      <div className="cellengine-legend-grid">
        {showGenome ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Genome fill</div>
            <GenomeMetricScale metricId={genomeMetric} />
            {genomeProgramMeta(genomeMetric) ? <GenomeDnaLegend activeMetricId={genomeMetric} /> : null}
            <div className="meta-note">fill colors encode the selected genome program or body feature for every cell preview below</div>
          </div>
        ) : null}
        {showRoles ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Structural role</div>
            <CellRoleLegend />
            <div className="meta-note">border or stroke color marks the structural class of each cell</div>
          </div>
        ) : null}
        {showFunction ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Function in action</div>
            <FunctionalStateLegend />
          </div>
        ) : null}
        {showMatrix ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Metric matrix</div>
            <CellMatrixLegend />
          </div>
        ) : null}
        {showGraph ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Connectivity graph</div>
            <div className="cellengine-matrix-legend-notes">
              <div><strong>Node color</strong><span>dominant cell function at the current replay frame</span></div>
              <div><strong>Node size and halo</strong><span>activity and active/inactive state</span></div>
              <div><strong>Edge thickness / opacity</strong><span>current gap-junction strength</span></div>
              <div><strong>Orange edge tint</strong><span>asymmetric directed coupling</span></div>
            </div>
          </div>
        ) : null}
        {showDevelopment ? (
          <div className="cellengine-legend-panel">
            <div className="cellengine-matrix-legend-title">Development animation</div>
            <div className="cellengine-matrix-legend-notes">
              <div><strong>Cell appearance order</strong><span>cells are revealed by `birth_step` during development playback</span></div>
              <div><strong>Population grid</strong><span>each tile is one candidate from the current generation</span></div>
              <div><strong>Live discovery cards</strong><span>show the current best organism per seed as the search evolves</span></div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
