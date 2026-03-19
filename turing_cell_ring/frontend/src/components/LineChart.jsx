import React from 'react'

function toPolyline(values, width, height, minValue, maxValue) {
  if (!values.length) {
    return ''
  }
  const span = Math.max(maxValue - minValue, 1e-9)
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width
      const y = height - ((value - minValue) / span) * height
      return `${x},${y}`
    })
    .join(' ')
}

export default function LineChart({ title, series, height = 180 }) {
  const allValues = series.flatMap((item) => item.values)
  const minValue = Math.min(...allValues, 0)
  const maxValue = Math.max(...allValues, 1)
  const width = 640

  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart">
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} className="axis-line" />
        {series.map((item) => (
          <polyline
            key={item.label}
            fill="none"
            stroke={item.stroke}
            strokeWidth="2.5"
            points={toPolyline(item.values, width, height - 8, minValue, maxValue)}
          />
        ))}
      </svg>
      <div className="legend">
        {series.map((item) => (
          <span key={item.label} className="legend-item">
            <span className="legend-swatch" style={{ backgroundColor: item.stroke }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  )
}
