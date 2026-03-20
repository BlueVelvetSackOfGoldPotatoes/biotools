import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { GeometryObject, Vec3 } from '../types/dataModel';
import type { DataModelActions } from '../hooks/useDataModel';
import { computeMeshStats, computeBoundingBox } from '../utils/geometry';
import { makeCube, makeIcosphere } from '../utils/defaults';

interface Props {
  objects: GeometryObject[];
  actions: DataModelActions;
}

// Simple Canvas-based 3D wireframe renderer
function renderWireframe(
  canvas: HTMLCanvasElement,
  objects: GeometryObject[],
  rotation: { rx: number; ry: number },
  zoom: number,
  selectedName: string | null
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const scale = Math.min(w, h) * 0.3 * zoom;

  const cosRx = Math.cos(rotation.rx);
  const sinRx = Math.sin(rotation.rx);
  const cosRy = Math.cos(rotation.ry);
  const sinRy = Math.sin(rotation.ry);

  const project = (v: Vec3, loc: Vec3): { x: number; y: number; z: number } => {
    let x = v.x + loc.x;
    let y = v.y + loc.y;
    let z = v.z + loc.z;
    // Rotate around Y
    const x1 = x * cosRy - z * sinRy;
    const z1 = x * sinRy + z * cosRy;
    // Rotate around X
    const y1 = y * cosRx - z1 * sinRx;
    const z2 = y * sinRx + z1 * cosRx;
    return { x: cx + x1 * scale, y: cy - y1 * scale, z: z2 };
  };

  // Draw grid
  ctx.strokeStyle = '#2a2a4e';
  ctx.lineWidth = 0.5;
  for (let i = -5; i <= 5; i++) {
    const p1 = project({ x: i * 0.2, y: 0, z: -1 }, { x: 0, y: 0, z: 0 });
    const p2 = project({ x: i * 0.2, y: 0, z: 1 }, { x: 0, y: 0, z: 0 });
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    const p3 = project({ x: -1, y: 0, z: i * 0.2 }, { x: 0, y: 0, z: 0 });
    const p4 = project({ x: 1, y: 0, z: i * 0.2 }, { x: 0, y: 0, z: 0 });
    ctx.beginPath();
    ctx.moveTo(p3.x, p3.y);
    ctx.lineTo(p4.x, p4.y);
    ctx.stroke();
  }

  // Draw axes
  const origin = project({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 });
  const axisLen = 0.3;
  const colors = ['#ff4444', '#44ff44', '#4444ff'];
  const axes: Vec3[] = [
    { x: axisLen, y: 0, z: 0 },
    { x: 0, y: axisLen, z: 0 },
    { x: 0, y: 0, z: axisLen },
  ];
  const axisLabels = ['X', 'Y', 'Z'];
  axes.forEach((a, i) => {
    const p = project(a, { x: 0, y: 0, z: 0 });
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.fillStyle = colors[i];
    ctx.font = '12px monospace';
    ctx.fillText(axisLabels[i], p.x + 4, p.y - 4);
  });

  // Draw objects
  const objectColors = ['#00d4ff', '#ff6b9d', '#c084fc', '#fbbf24', '#34d399', '#f97316'];
  objects.forEach((obj, objIdx) => {
    const isSelected = obj.name === selectedName;
    const color = isSelected ? '#ffffff' : objectColors[objIdx % objectColors.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 1.5 : 0.8;
    ctx.globalAlpha = isSelected ? 1.0 : 0.7;

    for (const f of obj.faces) {
      const p0 = project(obj.vertices[f.v0], obj.location);
      const p1 = project(obj.vertices[f.v1], obj.location);
      const p2 = project(obj.vertices[f.v2], obj.location);
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.closePath();
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;

    // Label
    if (obj.vertices.length > 0) {
      const center = obj.vertices.reduce(
        (acc, v) => ({
          x: acc.x + (v.x + obj.location.x) / obj.vertices.length,
          y: acc.y + (v.y + obj.location.y) / obj.vertices.length,
          z: acc.z + (v.z + obj.location.z) / obj.vertices.length,
        }),
        { x: 0, y: 0, z: 0 }
      );
      const pc = project(center, { x: 0, y: 0, z: 0 });
      ctx.fillStyle = color;
      ctx.font = isSelected ? 'bold 13px sans-serif' : '11px sans-serif';
      ctx.fillText(obj.name, pc.x + 5, pc.y - 5);
    }
  });
}

export const GeometryViewer: React.FC<Props> = ({ objects, actions }) => {
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [rotation, setRotation] = useState({ rx: 0.4, ry: 0.6 });
  const [zoom, setZoom] = useState(1.0);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const selected = objects.find(o => o.name === selectedName);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.parentElement?.getBoundingClientRect();
    if (rect) {
      canvas.width = rect.width;
      canvas.height = Math.max(350, rect.height);
    }
    renderWireframe(canvas, objects, rotation, zoom, selectedName);
  }, [objects, rotation, zoom, selectedName]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setRotation(prev => ({
      rx: prev.rx + dy * 0.005,
      ry: prev.ry + dx * 0.005,
    }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, [isDragging, lastMouse]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.1, Math.min(10, prev - e.deltaY * 0.001)));
  }, []);

  const handleAddPrimitive = (type: 'cube' | 'icosphere') => {
    const baseName = type === 'cube' ? 'Cube' : 'Icosphere';
    let name = baseName;
    let idx = 1;
    while (objects.some(o => o.name === name)) {
      name = `${baseName}_${idx++}`;
    }
    const obj = type === 'cube' ? makeCube(name) : makeIcosphere(name, 1.0, 2);
    actions.addGeometryObject(obj);
    setSelectedName(name);
  };

  const stats = selected ? computeMeshStats(selected) : null;

  return (
    <div className="panel">
      <h2>Geometry / Model Objects</h2>
      <p className="panel-description">
        Define and visualize triangulated mesh objects for MCell simulation geometry.
        Drag to rotate, scroll to zoom.
      </p>

      <div className="geometry-layout">
        <div className="viewport-container">
          <canvas
            ref={canvasRef}
            className="viewport-canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onWheel={handleWheel}
          />
          <div className="viewport-controls">
            <button className="btn btn-sm" onClick={() => setRotation({ rx: 0, ry: 0 })}>Front</button>
            <button className="btn btn-sm" onClick={() => setRotation({ rx: -Math.PI / 2, ry: 0 })}>Top</button>
            <button className="btn btn-sm" onClick={() => setRotation({ rx: 0, ry: Math.PI / 2 })}>Right</button>
            <button className="btn btn-sm" onClick={() => setRotation({ rx: 0.4, ry: 0.6 })}>Perspective</button>
            <button className="btn btn-sm" onClick={() => setZoom(1.0)}>Reset Zoom</button>
          </div>
        </div>

        <div className="geometry-sidebar">
          <div className="list-panel">
            <div className="list-header">
              <span>Objects ({objects.length})</span>
              <div className="btn-group">
                <button className="btn btn-sm btn-primary" onClick={() => handleAddPrimitive('cube')}>+ Cube</button>
                <button className="btn btn-sm btn-primary" onClick={() => handleAddPrimitive('icosphere')}>+ Sphere</button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => {
                    if (selectedName) {
                      actions.removeGeometryObject(selectedName);
                      setSelectedName(null);
                    }
                  }}
                  disabled={!selected}
                >
                  Remove
                </button>
              </div>
            </div>
            <div className="item-list compact">
              {objects.map(obj => (
                <div
                  key={obj.name}
                  className={`item-row ${selectedName === obj.name ? 'selected' : ''}`}
                  onClick={() => setSelectedName(obj.name)}
                >
                  <span className="item-name">{obj.name}</span>
                  <span className="item-detail">
                    V:{obj.vertices.length} F:{obj.faces.length}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {selected && stats && (
            <div className="properties-panel compact">
              <h3>Mesh Analysis: {selected.name}</h3>
              <table className="stats-table">
                <tbody>
                  <tr><td>Vertices</td><td>{stats.vertices}</td></tr>
                  <tr><td>Edges</td><td>{stats.edges}</td></tr>
                  <tr><td>Faces</td><td>{stats.faces}</td></tr>
                  <tr><td>Surface Area</td><td>{stats.area.toFixed(6)}</td></tr>
                  <tr><td>Volume</td><td>{stats.volume.toFixed(6)}</td></tr>
                  <tr><td>Euler Char.</td><td>{stats.eulerCharacteristic}</td></tr>
                  <tr><td>Genus</td><td>{stats.genus}</td></tr>
                  <tr><td>Watertight</td><td>{stats.watertight ? 'Yes' : 'No'}</td></tr>
                </tbody>
              </table>

              <h4>Location</h4>
              <div className="form-grid compact">
                <label>X:</label>
                <input
                  type="number"
                  value={selected.location.x}
                  onChange={e => actions.updateGeometryObject(selected.name, {
                    location: { ...selected.location, x: parseFloat(e.target.value) || 0 }
                  })}
                  step="0.1"
                />
                <label>Y:</label>
                <input
                  type="number"
                  value={selected.location.y}
                  onChange={e => actions.updateGeometryObject(selected.name, {
                    location: { ...selected.location, y: parseFloat(e.target.value) || 0 }
                  })}
                  step="0.1"
                />
                <label>Z:</label>
                <input
                  type="number"
                  value={selected.location.z}
                  onChange={e => actions.updateGeometryObject(selected.name, {
                    location: { ...selected.location, z: parseFloat(e.target.value) || 0 }
                  })}
                  step="0.1"
                />
              </div>

              {selected.regions.length > 0 && (
                <>
                  <h4>Surface Regions</h4>
                  <div className="region-list">
                    {selected.regions.map((reg, i) => (
                      <div key={i} className="region-item">
                        <span>{reg.name}</span>
                        <span className="item-detail">{reg.faceIndices.length} faces</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
