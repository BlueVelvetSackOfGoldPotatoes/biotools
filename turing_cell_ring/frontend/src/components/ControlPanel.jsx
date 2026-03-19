import React from 'react'

export default function ControlPanel({
  families,
  presets,
  form,
  setForm,
  onSelectFamily,
  onSelectPreset,
  onAnalyze,
  onSimulate,
  onBatch,
  busy
}) {
  const activePreset = presets.find((preset) => preset.id === form.preset)
  const activeFamily = families.find((family) => family.id === form.familyId)
  const familyPresets = presets.filter((preset) => preset.familyId === form.familyId)
  const allowFullChemistry = Boolean(activeFamily?.supportsFullChemistry)
  const allowHistorical = Boolean(activeFamily?.supportsHistoricalExecution) && form.engine !== 'full_chemistry'

  const update = (key, value) => {
    setForm((current) => ({ ...current, [key]: value }))
  }

  return (
    <section className="card control-panel">
      <div className="panel-header">
        <div>
          <h2>Experiment controls</h2>
          <p>
            Switch across the paper’s worked families, execution branches, and validation presets from one surface.
          </p>
        </div>
      </div>

      <div className="form-grid">
        <label>
          <span>Family</span>
          <select
            value={form.familyId}
            onChange={(event) => onSelectFamily(event.target.value)}
          >
            {families.map((family) => (
              <option key={family.id} value={family.id}>
                {family.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Preset</span>
          <select
            value={form.preset}
            onChange={(event) => onSelectPreset(event.target.value)}
          >
            {familyPresets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Engine</span>
          <select
            value={form.engine}
            onChange={(event) => update('engine', event.target.value)}
          >
            <option value="reduced_xy">Reduced / generic engine</option>
            <option value="full_chemistry" disabled={!allowFullChemistry}>
              Full chemistry
            </option>
          </select>
        </label>

        <label>
          <span>Execution mode</span>
          <select
            value={form.executionMode}
            onChange={(event) => {
              const value = event.target.value
              update('executionMode', value)
              update(
                'executionProfileId',
                value === 'historical_paper_constrained'
                  ? 'historic_1952_baseline'
                  : 'modern_default'
              )
            }}
          >
            <option value="modern">Modern</option>
            <option value="historical_paper_constrained" disabled={!allowHistorical}>
              Historical paper-constrained
            </option>
          </select>
        </label>

        {form.executionMode === 'historical_paper_constrained' ? (
          <label>
            <span>Historical profile</span>
            <select
              value={form.executionProfileId}
              onChange={(event) => update('executionProfileId', event.target.value)}
            >
              <option value="historic_1952_baseline">Baseline</option>
              <option value="historic_1952_coarse_rounding">Coarse rounding</option>
              <option value="historic_1952_fine_rounding">Fine rounding</option>
              <option value="historic_1952_synchronous">Synchronous update</option>
            </select>
          </label>
        ) : null}

        <label>
          <span>Seed</span>
          <input
            type="number"
            value={form.seed}
            onChange={(event) => update('seed', Number(event.target.value))}
          />
        </label>

        <label>
          <span>dt</span>
          <input
            type="number"
            step="0.001"
            value={form.dt}
            onChange={(event) => update('dt', Number(event.target.value))}
          />
        </label>

        <label>
          <span>Total time</span>
          <input
            type="number"
            step="0.1"
            value={form.totalTime}
            onChange={(event) => update('totalTime', Number(event.target.value))}
          />
        </label>

        <label>
          <span>Capture stride</span>
          <input
            type="number"
            value={form.captureStride}
            onChange={(event) => update('captureStride', Number(event.target.value))}
          />
        </label>

        <label>
          <span>Batch replicates</span>
          <input
            type="number"
            value={form.replicates}
            onChange={(event) => update('replicates', Number(event.target.value))}
          />
        </label>

        <label>
          <span>Noise scale</span>
          <input
            type="number"
            step="0.1"
            value={form.noiseScale}
            onChange={(event) => update('noiseScale', Number(event.target.value))}
          />
        </label>

        {form.familyId === 'example2_table2' ? (
          <label>
            <span>Example 2 f</span>
            <input
              type="number"
              step="0.1"
              value={form.familyParameters.example2F}
              onChange={(event) =>
                update('familyParameters', {
                  ...form.familyParameters,
                  example2F: Number(event.target.value)
                })}
            />
          </label>
        ) : null}

        {form.familyId === 'section10_example1' ? (
          <>
            <label>
              <span>Incipient capture</span>
              <select
                value={form.incipientCaptureMode}
                onChange={(event) => update('incipientCaptureMode', event.target.value)}
              >
                <option value="gamma_threshold">Gamma threshold</option>
                <option value="mode234_threshold">Mode 2/3/4 threshold</option>
              </select>
            </label>

            {form.incipientCaptureMode === 'gamma_threshold' ? (
              <label>
                <span>Incipient gamma</span>
                <input
                  type="number"
                  step="0.0001"
                  value={form.incipientCaptureGamma}
                  onChange={(event) => update('incipientCaptureGamma', Number(event.target.value))}
                />
              </label>
            ) : (
              <>
                <label>
                  <span>Mode 2/3/4 start</span>
                  <input
                    type="number"
                    step="0.1"
                    value={form.incipientCaptureStartTime}
                    onChange={(event) => update('incipientCaptureStartTime', Number(event.target.value))}
                  />
                </label>

                <label>
                  <span>Mode 2/3/4 threshold</span>
                  <input
                    type="number"
                    step="0.01"
                    value={form.incipientMode234Threshold}
                    onChange={(event) => update('incipientMode234Threshold', Number(event.target.value))}
                  />
                </label>
              </>
            )}
          </>
        ) : null}

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={form.enableNoise}
            onChange={(event) => update('enableNoise', event.target.checked)}
          />
          <span>Enable stochastic disturbances</span>
        </label>
      </div>

      {activePreset && activeFamily ? (
        <div className="preset-summary">
          <div>
            <span className="eyebrow">Family</span>
            <h3>{activeFamily.name}</h3>
            <p>{activeFamily.description}</p>
          </div>
          <div>
            <span className="eyebrow">Preset</span>
            <h3>{activePreset.name}</h3>
            <p>{activePreset.description}</p>
          </div>
          <div>
            <span className="eyebrow">Hypothesis</span>
            <p>{activePreset.hypothesis}</p>
          </div>
          <div>
            <span className="eyebrow">Capabilities</span>
            <p>
              {allowFullChemistry ? 'Full chemistry available. ' : 'Generic engine only. '}
              {activeFamily.supportsHistoricalExecution
                ? 'Historical comparison available.'
                : 'Historical comparison not available.'}
            </p>
          </div>
        </div>
      ) : null}

      <div className="button-row">
        <button disabled={busy} onClick={onAnalyze}>
          Analyze
        </button>
        <button disabled={busy} onClick={onSimulate}>
          Run single specimen
        </button>
        <button disabled={busy} onClick={onBatch}>
          Run Monte Carlo batch
        </button>
      </div>
    </section>
  )
}
