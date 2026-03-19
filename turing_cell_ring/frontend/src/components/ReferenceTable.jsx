import React from 'react'

export default function ReferenceTable({ comparisons }) {
  if (!comparisons?.length) {
    return (
      <div className="card">
        <h3>Reference comparisons</h3>
        <p className="section-copy">No embedded reference profile is attached to this preset.</p>
      </div>
    )
  }

  return (
    <div className="card">
      <h3>Reference comparisons</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Reference</th>
              <th>Shift</th>
              <th>RMSE X</th>
              <th>RMSE Y</th>
              <th>Combined</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((row) => (
              <tr key={row.referenceId + row.label}>
                <td>{row.label}</td>
                <td>{row.bestShift}</td>
                <td>{row.rmseX >= 0 ? row.rmseX.toFixed(4) : '—'}</td>
                <td>{row.rmseY >= 0 ? row.rmseY.toFixed(4) : '—'}</td>
                <td>{row.rmseCombined >= 0 ? row.rmseCombined.toFixed(4) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
