import { formatPct } from "../../lib/dashboardShared";

export default function ComparisonTables({
  familyRows,
  variantRows,
  scoreLower,
  scoreAbbr,
  onOpenFamily,
  onOpenRun
}) {
  return (
    <>
      <div className="card">
        <h3>Family Comparison Table</h3>
        {familyRows.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>family</th>
                  <th>runs</th>
                  <th>active</th>
                  <th>variants</th>
                  <th>{scoreLower} coverage</th>
                  <th>best {scoreAbbr}</th>
                  <th>mean {scoreAbbr}</th>
                  <th>median {scoreAbbr}</th>
                  <th>mean epoch</th>
                  <th>mean train (s)</th>
                  <th>best run</th>
                </tr>
              </thead>
              <tbody>
                {familyRows.map((row) => (
                  <tr key={row.family}>
                    <td>
                      <button className="mini-btn" type="button" onClick={() => onOpenFamily(row.family)}>
                        {row.family}
                      </button>
                    </td>
                    <td>{row.runs}</td>
                    <td>{row.active_runs}</td>
                    <td>{row.variants}</td>
                    <td>{formatPct(row.accuracy_coverage)}</td>
                    <td>{formatPct(row.best_acc)}</td>
                    <td>{formatPct(row.mean_acc)}</td>
                    <td>{formatPct(row.median_acc)}</td>
                    <td>{typeof row.mean_epoch === "number" ? row.mean_epoch.toFixed(1) : "n/a"}</td>
                    <td>{typeof row.mean_train_s === "number" ? row.mean_train_s.toFixed(1) : "n/a"}</td>
                    <td>
                      {row.best_run_id ? (
                        <button className="mini-btn" type="button" onClick={() => onOpenRun(row.best_run_id)}>
                          {row.best_run_id}
                        </button>
                      ) : (
                        "n/a"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">No runs available.</div>
        )}
      </div>

      <div className="card">
        <h3>Variant Comparison Table</h3>
        {variantRows.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>family</th>
                  <th>variant</th>
                  <th>runs</th>
                  <th>active</th>
                  <th>best {scoreAbbr}</th>
                  <th>mean {scoreAbbr}</th>
                  <th>mean epoch</th>
                  <th>mean train (s)</th>
                  <th>best run</th>
                </tr>
              </thead>
              <tbody>
                {variantRows.slice(0, 120).map((row) => (
                  <tr key={`${row.family}:${row.variant}`}>
                    <td>{row.family}</td>
                    <td>{row.variant}</td>
                    <td>{row.runs}</td>
                    <td>{row.active_runs}</td>
                    <td>{formatPct(row.best_acc)}</td>
                    <td>{formatPct(row.mean_acc)}</td>
                    <td>{typeof row.mean_epoch === "number" ? row.mean_epoch.toFixed(1) : "n/a"}</td>
                    <td>{typeof row.mean_train_s === "number" ? row.mean_train_s.toFixed(1) : "n/a"}</td>
                    <td>
                      {row.best_run_id ? (
                        <button className="mini-btn" type="button" onClick={() => onOpenRun(row.best_run_id)}>
                          {row.best_run_id}
                        </button>
                      ) : (
                        "n/a"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">No variant data available.</div>
        )}
      </div>
    </>
  );
}
