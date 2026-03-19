import React from 'react'

export default function ModeBarChart({ title, amplitudes, highlight = [] }) {
  const bars = amplitudes.slice(1)
  const maxValue = Math.max(...bars, 1e-9)

  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      <div className="mode-bars">
        {bars.map((value, index) => {
          const mode = index + 1
          const active = highlight.includes(mode)
          return (
            <div key={mode} className={`mode-bar ${active ? 'highlight' : ''}`}>
              <div
                className="mode-fill"
                style={{ height: `${(value / maxValue) * 100}%` }}
              />
              <div className="mode-label">s={mode}</div>
              <div className="mode-value">{value.toFixed(3)}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
