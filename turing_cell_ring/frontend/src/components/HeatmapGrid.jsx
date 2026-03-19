import React from 'react'

function mix(start, end, alpha) {
  return Math.round(start + (end - start) * alpha)
}

function colorFor(value, minValue, maxValue) {
  const span = Math.max(maxValue - minValue, 1e-9)
  const alpha = Math.min(Math.max((value - minValue) / span, 0), 1)
  const low = [245, 239, 227]
  const high = [11, 107, 98]
  return `rgb(${mix(low[0], high[0], alpha)}, ${mix(low[1], high[1], alpha)}, ${mix(low[2], high[2], alpha)})`
}

export default function HeatmapGrid({ title, matrix, caption }) {
  const width = matrix.length || 1
  const height = matrix[0]?.length || 1
  const flat = matrix.flat()
  const minValue = flat.length ? Math.min(...flat) : 0
  const maxValue = flat.length ? Math.max(...flat) : 1

  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="heatmap-chart"
      >
        {matrix.map((column, timeIndex) =>
          column.map((value, cellIndex) => (
            <rect
              key={`${timeIndex}-${cellIndex}`}
              x={timeIndex}
              y={cellIndex}
              width="1"
              height="1"
              fill={colorFor(value, minValue, maxValue)}
            />
          ))
        )}
      </svg>
      <div className="heatmap-scale">
        <span>{minValue.toFixed(3)}</span>
        <span>{maxValue.toFixed(3)}</span>
      </div>
      {caption ? <p className="chart-caption">{caption}</p> : null}
    </div>
  )
}
