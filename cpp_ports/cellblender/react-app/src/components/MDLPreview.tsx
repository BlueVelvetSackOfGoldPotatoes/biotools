import React, { useState, useMemo } from 'react';
import type { DataModel } from '../types/dataModel';
import { generateMDL } from '../utils/mdlGenerator';
import { downloadMDL } from '../utils/projectIO';

interface Props {
  dataModel: DataModel;
}

export const MDLPreview: React.FC<Props> = ({ dataModel }) => {
  const [wrapLines, setWrapLines] = useState(false);
  const [fontSize, setFontSize] = useState(13);

  const mdlText = useMemo(() => generateMDL(dataModel), [dataModel]);
  const lineCount = mdlText.split('\n').length;

  const handleCopy = () => {
    navigator.clipboard.writeText(mdlText).catch(() => {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = mdlText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    });
  };

  return (
    <div className="panel">
      <h2>MDL Preview</h2>
      <p className="panel-description">
        Preview the generated MCell MDL (Model Description Language) text
        based on your current model configuration. This is the input file
        format that MCell uses to run simulations.
      </p>

      <div className="mdl-toolbar">
        <div className="btn-group">
          <button className="btn btn-sm btn-primary" onClick={handleCopy}>
            Copy to Clipboard
          </button>
          <button className="btn btn-sm" onClick={() => downloadMDL(mdlText)}>
            Download MDL
          </button>
        </div>
        <div className="mdl-options">
          <label className="checkbox-label">
            <input type="checkbox" checked={wrapLines} onChange={e => setWrapLines(e.target.checked)} />
            Wrap Lines
          </label>
          <label>
            Font Size:
            <input type="range" min={9} max={18} value={fontSize}
              onChange={e => setFontSize(parseInt(e.target.value))} />
            {fontSize}px
          </label>
          <span className="line-count">{lineCount} lines</span>
        </div>
      </div>

      <div className="mdl-preview-container">
        <pre
          className="mdl-code"
          style={{
            fontSize: `${fontSize}px`,
            whiteSpace: wrapLines ? 'pre-wrap' : 'pre',
          }}
        >
          {mdlText.split('\n').map((line, i) => (
            <div key={i} className="mdl-line">
              <span className="line-number">{i + 1}</span>
              <span className="line-content">{line}</span>
            </div>
          ))}
        </pre>
      </div>
    </div>
  );
};
