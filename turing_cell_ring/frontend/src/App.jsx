import React, { useEffect, useMemo, useState } from 'react'
import { analyze, batch, fetchPresets, simulate } from './api'
import ControlPanel from './components/ControlPanel'
import HeatmapGrid from './components/HeatmapGrid'
import LineChart from './components/LineChart'
import ModeBarChart from './components/ModeBarChart'
import ReferenceTable from './components/ReferenceTable'

const SERIES_COLORS = ['#0b6b62', '#c65b2c', '#2c5cc5', '#8f7b34', '#b21f44']

const DEFAULT_FORM = {
  preset: 'paper-quick',
  familyId: 'section10_example1',
  engine: 'reduced_xy',
  analysisMode: 'stationary_two_species',
  executionMode: 'modern',
  executionProfileId: 'modern_default',
  speciesOrder: ['X', 'Y'],
  familyParameters: {},
  seed: 1,
  dt: 0.01,
  totalTime: 80,
  enableNoise: true,
  noiseScale: 1,
  captureStride: 20,
  maxMode: 10,
  incipientCaptureMode: 'gamma_threshold',
  incipientCaptureGamma: 1 / 16,
  incipientCaptureStartTime: 0,
  incipientMode234Threshold: 0.18,
  replicates: 128
}

function buildPayload(form) {
  return {
    preset: form.preset,
    familyId: form.familyId,
    engine: form.engine,
    analysisMode: form.analysisMode,
    executionMode: form.executionMode,
    executionProfileId: form.executionProfileId,
    speciesOrder: form.speciesOrder,
    familyParameters: form.familyParameters,
    seed: form.seed,
    dt: form.dt,
    totalTime: form.totalTime,
    enableNoise: form.enableNoise,
    noiseScale: form.noiseScale,
    captureStride: form.captureStride,
    maxMode: form.maxMode,
    incipientCaptureMode: form.incipientCaptureMode,
    incipientCaptureGamma: form.incipientCaptureGamma,
    incipientCaptureStartTime: form.incipientCaptureStartTime,
    incipientMode234Threshold: form.incipientMode234Threshold
  }
}

function formPatchFromPreset(preset, family, current = DEFAULT_FORM) {
  const supportsFullChemistry = Boolean(family?.supportsFullChemistry)
  const nextEngine = supportsFullChemistry && current.engine === 'full_chemistry'
    ? 'full_chemistry'
    : (supportsFullChemistry ? preset.config.engine : 'reduced_xy')
  const supportsHistorical = Boolean(family?.supportsHistoricalExecution) && nextEngine !== 'full_chemistry'
  const nextExecutionMode = supportsHistorical && current.executionMode === 'historical_paper_constrained'
    ? 'historical_paper_constrained'
    : 'modern'

  return {
    ...current,
    preset: preset.id,
    familyId: preset.familyId,
    engine: nextEngine,
    analysisMode: preset.config.analysisMode,
    executionMode: nextExecutionMode,
    executionProfileId: nextExecutionMode === 'historical_paper_constrained'
      ? 'historic_1952_baseline'
      : 'modern_default',
    speciesOrder: preset.config.speciesOrder || family?.speciesOrder || [],
    familyParameters: preset.config.familyParameters || {},
    dt: preset.config.dt,
    totalTime: preset.config.totalTime,
    enableNoise: preset.config.enableNoise,
    noiseScale: preset.config.noiseScale,
    captureStride: preset.config.captureStride,
    maxMode: preset.config.maxMode,
    incipientCaptureMode: preset.config.incipientCaptureMode,
    incipientCaptureGamma: preset.config.incipientCaptureGamma,
    incipientCaptureStartTime: preset.config.incipientCaptureStartTime,
    incipientMode234Threshold: preset.config.incipientMode234Threshold
  }
}

function snapshotSeries(snapshot) {
  if (!snapshot) {
    return []
  }
  if (snapshot.species?.length) {
    return snapshot.species
  }
  const fallback = []
  if (snapshot.x?.length) {
    fallback.push({ name: 'X', values: snapshot.x })
  }
  if (snapshot.y?.length) {
    fallback.push({ name: 'Y', values: snapshot.y })
  }
  return fallback
}

function snapshotLineSeries(snapshot) {
  return snapshotSeries(snapshot).map((series, index) => ({
    label: series.name,
    values: series.values,
    stroke: SERIES_COLORS[index % SERIES_COLORS.length]
  }))
}

function equilibriumLabel(analysis) {
  if (!analysis?.equilibrium?.speciesOrder?.length) {
    return ''
  }
  return analysis.equilibrium.speciesOrder
    .map((name, index) => `${name}=${analysis.equilibrium.speciesValues[index].toFixed(3)}`)
    .join(' ')
}

function maxDifference(seriesA = [], seriesB = []) {
  return seriesA.reduce((best, value, index) => {
    const delta = Math.abs(value - (seriesB[index] ?? 0))
    return Math.max(best, delta)
  }, 0)
}

function familyComparisonEligible(form, family) {
  return Boolean(family?.supportsHistoricalExecution) && form.engine !== 'full_chemistry'
}

function SummaryMetrics({ simulation, family }) {
  if (!simulation) {
    return null
  }

  const metrics = simulation.metrics
  const analysis = simulation.incipientAnalysis
  const isSection10 = family?.id === 'section10_example1'
  const isExample2 = family?.id === 'example2_table2'
  const executionProfileId = simulation.config.executionProfileId || 'modern_default'
  const example2F = simulation.config.familyParameters?.example2F
  const example2K = Number.isFinite(example2F) ? 16 - example2F : null

  return (
    <section className="metric-grid">
      <div className="metric-card">
        <span className="metric-label">Family</span>
        <strong>{family?.name || analysis.familyId}</strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Engine / execution</span>
        <strong>{simulation.config.engine}</strong>
        <small>{simulation.config.executionMode}</small>
      </div>
      <div className="metric-card">
        <span className="metric-label">Execution profile</span>
        <strong>{executionProfileId}</strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Predicted mode</span>
        <strong>s = {analysis.dominantMode}</strong>
        <small>{analysis.dominantClassification}</small>
      </div>
      <div className="metric-card">
        <span className="metric-label">Observed mode</span>
        <strong>s = {metrics.dominantMode}</strong>
        <small>{metrics.dominantClassification}</small>
      </div>
      <div className="metric-card">
        <span className="metric-label">Growth / regularity</span>
        <strong>{analysis.dominantGrowth.toFixed(5)}</strong>
        <small>{metrics.regularityIndex.toFixed(3)}</small>
      </div>
      <div className="metric-card">
        <span className="metric-label">Frequency / speed</span>
        <strong>{metrics.dominantFrequencyCyclesPerTime.toFixed(4)}</strong>
        <small>{metrics.dominantPhaseVelocityCellsPerTime.toFixed(4)} cells per t</small>
      </div>
      <div className="metric-card">
        <span className="metric-label">Neighbour phase offset</span>
        <strong>{metrics.neighborPhaseOffsetRadians.toFixed(3)}</strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Travelling consistency</span>
        <strong>{metrics.travellingConsistency.toFixed(3)}</strong>
      </div>
      <div className="metric-card">
        <span className="metric-label">Equilibrium</span>
        <strong>{equilibriumLabel(analysis)}</strong>
      </div>
      {isExample2 ? (
        <div className="metric-card">
          <span className="metric-label">Example 2 parameters</span>
          <strong>{Number.isFinite(example2F) ? `f=${example2F.toFixed(1)}` : 'preset default'}</strong>
          <small>{Number.isFinite(example2K) ? `k=${example2K.toFixed(1)}` : 'k unavailable'}</small>
        </div>
      ) : null}
      {isSection10 ? (
        <>
          <div className="metric-card">
            <span className="metric-label">Mode 3 / Mode 4</span>
            <strong>{metrics.mode3ToMode4Ratio.toFixed(3)}</strong>
          </div>
          <div className="metric-card">
            <span className="metric-label">Mode 3 - Mode 4 gap</span>
            <strong>{analysis.mode34Gap.toFixed(5)}</strong>
          </div>
          <div className="metric-card">
            <span className="metric-label">Threshold gamma</span>
            <strong>{analysis.thresholdGamma.toFixed(5)}</strong>
            <small>{analysis.mode34EFoldTimeHours.toFixed(1)} h e-fold lead</small>
          </div>
          <div className="metric-card">
            <span className="metric-label">Arrest</span>
            <strong>
              {simulation.arrest.observed ? simulation.arrest.firstYZeroTime.toFixed(2) : 'not reached'}
            </strong>
            <small>{simulation.arrest.finalZeroCellCount} final zero-Y cells</small>
          </div>
        </>
      ) : null}
    </section>
  )
}

function BatchSummary({ batchResult, family }) {
  if (!batchResult) {
    return null
  }

  const histogram = batchResult.dominantModeHistogram.slice(1)
  const maxValue = Math.max(...histogram, 1)

  return (
    <section className="card">
      <h3>Batch summary</h3>
      <div className="metric-grid compact">
        <div className="metric-card">
          <span className="metric-label">Family</span>
          <strong>{family?.name || batchResult.config.familyId}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Engine / execution</span>
          <strong>{batchResult.config.engine}</strong>
          <small>{batchResult.config.executionMode}</small>
        </div>
        <div className="metric-card">
          <span className="metric-label">Replicates</span>
          <strong>{batchResult.replicates}</strong>
        </div>
        <div className="metric-card">
          <span className="metric-label">Share mode 3</span>
          <strong>{batchResult.shareMode3.toFixed(3)}</strong>
          <small>
            {batchResult.shareMode3CiLow.toFixed(3)}-{batchResult.shareMode3CiHigh.toFixed(3)}
          </small>
        </div>
        <div className="metric-card">
          <span className="metric-label">Share mode 4</span>
          <strong>{batchResult.shareMode4.toFixed(3)}</strong>
          <small>
            {batchResult.shareMode4CiLow.toFixed(3)}-{batchResult.shareMode4CiHigh.toFixed(3)}
          </small>
        </div>
        <div className="metric-card">
          <span className="metric-label">Mean regularity</span>
          <strong>{batchResult.meanRegularityIndex.toFixed(3)}</strong>
          <small>{batchResult.meanMode3ToMode4Ratio.toFixed(3)} mode 3 / mode 4</small>
        </div>
      </div>

      <div className="mode-bars">
        {histogram.map((count, index) => (
          <div
            key={index + 1}
            className={`mode-bar ${index + 1 === 3 || index + 1 === 4 ? 'highlight' : ''}`}
          >
            <div
              className="mode-fill"
              style={{ height: `${(count / maxValue) * 100}%` }}
            />
            <div className="mode-label">s={index + 1}</div>
            <div className="mode-value">{count}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function HistoricalComparison({ primary, companion, family }) {
  if (!primary || !companion || !family) {
    return null
  }

  const primarySeries = snapshotSeries(primary.finalSnapshot)
  const companionSeries = snapshotSeries(companion.finalSnapshot)
  const panels = primarySeries
    .map((series) => {
      const other = companionSeries.find((entry) => entry.name === series.name)
      if (!other) {
        return null
      }
      return {
        name: series.name,
        series: [
          { label: `${primary.config.executionMode} ${series.name}`, values: series.values, stroke: '#0b6b62' },
          { label: `${companion.config.executionMode} ${series.name}`, values: other.values, stroke: '#c65b2c' }
        ],
        delta: maxDifference(series.values, other.values)
      }
    })
    .filter(Boolean)

  return (
    <section className="card">
      <h3>Modern vs historical comparison</h3>
      <p className="section-copy">
        The historical branch is a paper-constrained reconstruction with fixed-step Euler updates, decimal rounding,
        and a simple seeded LCG. It is a sensitivity branch, not a claim of exact Manchester-machine execution.
      </p>
      <div className="metric-grid compact">
        <div className="metric-card">
          <span className="metric-label">Primary profile</span>
          <strong>{primary.config.executionMode}</strong>
          <small>{primary.config.executionProfileId}</small>
        </div>
        <div className="metric-card">
          <span className="metric-label">Comparison profile</span>
          <strong>{companion.config.executionMode}</strong>
          <small>{companion.config.executionProfileId}</small>
        </div>
        {panels.map((panel) => (
          <div key={panel.name} className="metric-card">
            <span className="metric-label">{panel.name} max absolute delta</span>
            <strong>{panel.delta.toFixed(4)}</strong>
          </div>
        ))}
      </div>
      <div className="two-column">
        {panels.slice(0, 2).map((panel) => (
          <LineChart key={panel.name} title={`${panel.name} final profile`} series={panel.series} />
        ))}
      </div>
    </section>
  )
}

export default function App() {
  const [families, setFamilies] = useState([])
  const [presets, setPresets] = useState([])
  const [form, setForm] = useState(DEFAULT_FORM)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [simulationResult, setSimulationResult] = useState(null)
  const [comparisonResult, setComparisonResult] = useState(null)
  const [batchResult, setBatchResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('Loading presets...')

  useEffect(() => {
    fetchPresets()
      .then((data) => {
        setFamilies(data.families || [])
        setPresets(data.presets || [])

        const quick = (data.presets || []).find((preset) => preset.id === 'paper-quick')
        const quickFamily = (data.families || []).find((family) => family.id === quick?.familyId)
        if (quick && quickFamily) {
          setForm((current) => ({ ...formPatchFromPreset(quick, quickFamily, current) }))
        }
        setStatus('Ready')
      })
      .catch((caught) => {
        setError(caught.message)
        setStatus('Failed to load presets')
      })
  }, [])

  const activePreset = useMemo(
    () => presets.find((preset) => preset.id === form.preset),
    [presets, form.preset]
  )
  const activeFamily = useMemo(
    () => families.find((family) => family.id === form.familyId),
    [families, form.familyId]
  )

  useEffect(() => {
    if (!activeFamily) {
      return
    }
    if (!activeFamily.supportsFullChemistry && form.engine !== 'reduced_xy') {
      setForm((current) => ({
        ...current,
        engine: 'reduced_xy',
        executionMode: 'modern',
        executionProfileId: 'modern_default'
      }))
      return
    }
    if ((!activeFamily.supportsHistoricalExecution || form.engine === 'full_chemistry')
      && form.executionMode !== 'modern') {
      setForm((current) => ({
        ...current,
        executionMode: 'modern',
        executionProfileId: 'modern_default'
      }))
    }
  }, [activeFamily, form.engine, form.executionMode])

  const selectPreset = (presetId) => {
    const preset = presets.find((entry) => entry.id === presetId)
    const family = families.find((entry) => entry.id === preset?.familyId)
    if (!preset || !family) {
      return
    }
    setForm((current) => formPatchFromPreset(preset, family, current))
  }

  const selectFamily = (familyId) => {
    const preset = presets.find((entry) => entry.familyId === familyId)
    if (preset) {
      selectPreset(preset.id)
    }
  }

  const run = async (kind) => {
    setBusy(true)
    setError('')
    setStatus(`Running ${kind}...`)

    try {
      const payload = buildPayload(form)
      if (kind === 'analysis') {
        const result = await analyze(payload)
        setAnalysisResult(result)
        setStatus('Analysis complete')
        return
      }

      if (kind === 'single') {
        const result = await simulate(payload)
        setSimulationResult(result)

        if (familyComparisonEligible(form, activeFamily)) {
          const comparisonMode =
            form.executionMode === 'historical_paper_constrained'
              ? 'modern'
              : 'historical_paper_constrained'
          const comparison = await simulate({
            ...payload,
            executionMode: comparisonMode,
            executionProfileId:
              comparisonMode === 'historical_paper_constrained'
                ? 'historic_1952_baseline'
                : 'modern_default'
          })
          setComparisonResult(comparison)
          setStatus('Single specimen and comparison complete')
        } else {
          setComparisonResult(null)
          setStatus('Single specimen complete')
        }
        return
      }

      if (kind === 'batch') {
        const result = await batch({
          ...payload,
          replicates: form.replicates
        })
        setBatchResult(result)
        setStatus('Batch complete')
      }
    } catch (caught) {
      setError(caught.message)
      setStatus('Request failed')
    } finally {
      setBusy(false)
    }
  }

  const analysisSeries = useMemo(() => {
    if (!analysisResult?.modes?.length) {
      return []
    }
    return [
      { label: 'real growth', values: analysisResult.modes.map((mode) => mode.real), stroke: '#0b6b62' },
      { label: 'imaginary part', values: analysisResult.modes.map((mode) => mode.imag), stroke: '#c65b2c' }
    ]
  }, [analysisResult])

  const modeTraceSeries = useMemo(() => {
    if (!simulationResult?.modeTrace?.length) {
      return []
    }
    const dominantMode = simulationResult.metrics.dominantMode || simulationResult.incipientAnalysis.dominantMode || 1
    return [
      {
        label: `mode ${dominantMode}`,
        values: simulationResult.modeTrace.map((sample) => sample.amplitudes[dominantMode] || 0),
        stroke: '#0b6b62'
      },
      {
        label: 'phase',
        values: simulationResult.modeTrace.map((sample) => sample.dominantPhaseRadians || 0),
        stroke: '#c65b2c'
      },
      {
        label: 'gamma',
        values: simulationResult.modeTrace.map((sample) => sample.gamma || 0),
        stroke: '#53687a'
      }
    ]
  }, [simulationResult])

  const heatmaps = useMemo(() => {
    if (!simulationResult?.snapshotTrace?.length) {
      return []
    }
    const series = snapshotSeries(simulationResult.finalSnapshot)
    return series.map((entry) => ({
      name: entry.name,
      matrix: simulationResult.snapshotTrace.map((snapshot) => {
        const current = snapshotSeries(snapshot).find((seriesEntry) => seriesEntry.name === entry.name)
        return current?.values || []
      })
    }))
  }, [simulationResult])

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Paper-example platform</p>
          <h1>Turing ring morphogenesis laboratory</h1>
          <p>
            One interface for Section 10 stationary rings, the second chemical example and Table 2,
            the three-morphogen oscillatory cases, and paper-constrained historical sensitivity runs.
          </p>
        </div>
        <div className="status-pill">{status}</div>
      </header>

      <ControlPanel
        families={families}
        presets={presets}
        form={form}
        setForm={setForm}
        onSelectFamily={selectFamily}
        onSelectPreset={selectPreset}
        onAnalyze={() => run('analysis')}
        onSimulate={() => run('single')}
        onBatch={() => run('batch')}
        busy={busy}
      />

      {error ? <section className="error-banner">{error}</section> : null}

      {analysisResult ? (
        <section className="card">
          <h3>Linear stability analysis</h3>
          <div className="metric-grid compact">
            <div className="metric-card">
              <span className="metric-label">Dominant mode</span>
              <strong>s = {analysisResult.dominantMode}</strong>
              <small>{analysisResult.dominantClassification}</small>
            </div>
            <div className="metric-card">
              <span className="metric-label">Growth / frequency</span>
              <strong>{analysisResult.dominantGrowth.toFixed(5)}</strong>
              <small>{analysisResult.dominantFrequencyCyclesPerTime.toFixed(4)} cycles per t</small>
            </div>
            <div className="metric-card">
              <span className="metric-label">Phase velocity</span>
              <strong>{analysisResult.dominantPhaseVelocityCellsPerTime.toFixed(4)}</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Equilibrium</span>
              <strong>{equilibriumLabel(analysisResult)}</strong>
            </div>
          </div>
          <div className="two-column">
            <LineChart title="Mode spectrum: real and imaginary parts" series={analysisSeries} />
            <ModeBarChart
              title="Primary-mode amplitudes at final snapshot"
              amplitudes={simulationResult?.finalSnapshot?.primaryModeAmplitudes || [0]}
              highlight={[analysisResult.dominantMode]}
            />
          </div>
        </section>
      ) : null}

      {simulationResult ? (
        <>
          <SummaryMetrics simulation={simulationResult} family={activeFamily} />

          <section className="two-column">
            <LineChart title="Initial profile" series={snapshotLineSeries(simulationResult.initialSnapshot)} />
            <LineChart title="Incipient profile" series={snapshotLineSeries(simulationResult.incipientSnapshot)} />
          </section>

          <section className="two-column">
            <LineChart title="Final profile" series={snapshotLineSeries(simulationResult.finalSnapshot)} />
            <LineChart title="Mode dynamics and phase" series={modeTraceSeries} />
          </section>

          {heatmaps.length ? (
            <section className="card">
              <h3>Species-by-cell heatmaps over time</h3>
              <p className="section-copy">
                Each column is a captured time slice and each row is a ring cell. This view is especially useful for
                the oscillatory presets where phase transport is easier to spot as diagonal structure.
              </p>
              <div className="two-column heatmap-grid">
                {heatmaps.map((heatmap) => (
                  <HeatmapGrid
                    key={heatmap.name}
                    title={`${heatmap.name} heatmap`}
                    matrix={heatmap.matrix}
                    caption={`${heatmap.matrix.length} captured times by ${heatmap.matrix[0]?.length || 0} cells`}
                  />
                ))}
              </div>
            </section>
          ) : null}

          <ReferenceTable comparisons={simulationResult.metrics.comparisons} />
          <HistoricalComparison
            primary={simulationResult}
            companion={comparisonResult}
            family={familyComparisonEligible(form, activeFamily) ? activeFamily : null}
          />
        </>
      ) : null}

      <BatchSummary batchResult={batchResult} family={activeFamily} />
    </main>
  )
}
