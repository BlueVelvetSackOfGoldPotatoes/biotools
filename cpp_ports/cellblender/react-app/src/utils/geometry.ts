// Geometry utility functions for mesh analysis
// Mirrors GeometryObject methods from cellblender.h

import type { GeometryObject, Vec3, Face } from '../types/dataModel';

export function computeSurfaceArea(obj: GeometryObject): number {
  let area = 0;
  for (const f of obj.faces) {
    const a = obj.vertices[f.v0];
    const b = obj.vertices[f.v1];
    const c = obj.vertices[f.v2];
    const e1 = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
    const e2 = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
    const cross = {
      x: e1.y * e2.z - e1.z * e2.y,
      y: e1.z * e2.x - e1.x * e2.z,
      z: e1.x * e2.y - e1.y * e2.x,
    };
    area += Math.sqrt(cross.x ** 2 + cross.y ** 2 + cross.z ** 2) * 0.5;
  }
  return area;
}

export function computeSignedVolume(obj: GeometryObject): number {
  let vol = 0;
  for (const f of obj.faces) {
    const a = obj.vertices[f.v0];
    const b = obj.vertices[f.v1];
    const c = obj.vertices[f.v2];
    const cross = {
      x: b.y * c.z - b.z * c.y,
      y: b.z * c.x - b.x * c.z,
      z: b.x * c.y - b.y * c.x,
    };
    vol += (a.x * cross.x + a.y * cross.y + a.z * cross.z) / 6.0;
  }
  return vol;
}

export function computeEdgeCount(obj: GeometryObject): number {
  const edgeSet = new Set<string>();
  for (const f of obj.faces) {
    const addEdge = (a: number, b: number) => {
      const key = `${Math.min(a, b)}_${Math.max(a, b)}`;
      edgeSet.add(key);
    };
    addEdge(f.v0, f.v1);
    addEdge(f.v1, f.v2);
    addEdge(f.v2, f.v0);
  }
  return edgeSet.size;
}

export function computeEulerCharacteristic(obj: GeometryObject): number {
  return obj.vertices.length - computeEdgeCount(obj) + obj.faces.length;
}

export function computeGenus(obj: GeometryObject): number {
  return 1 - computeEulerCharacteristic(obj) / 2;
}

export function isWatertight(obj: GeometryObject): boolean {
  const edgeCount = new Map<string, number>();
  for (const f of obj.faces) {
    const addEdge = (a: number, b: number) => {
      const key = `${Math.min(a, b)}_${Math.max(a, b)}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    };
    addEdge(f.v0, f.v1);
    addEdge(f.v1, f.v2);
    addEdge(f.v2, f.v0);
  }
  for (const count of edgeCount.values()) {
    if (count !== 2) return false;
  }
  return true;
}

export function computeBoundingBox(obj: GeometryObject): { min: Vec3; max: Vec3 } {
  if (obj.vertices.length === 0) {
    return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
  }
  const mn = { ...obj.vertices[0] };
  const mx = { ...obj.vertices[0] };
  for (const v of obj.vertices) {
    mn.x = Math.min(mn.x, v.x);
    mn.y = Math.min(mn.y, v.y);
    mn.z = Math.min(mn.z, v.z);
    mx.x = Math.max(mx.x, v.x);
    mx.y = Math.max(mx.y, v.y);
    mx.z = Math.max(mx.z, v.z);
  }
  return { min: mn, max: mx };
}

export interface MeshStats {
  vertices: number;
  edges: number;
  faces: number;
  area: number;
  volume: number;
  eulerCharacteristic: number;
  genus: number;
  watertight: boolean;
}

export function computeMeshStats(obj: GeometryObject): MeshStats {
  return {
    vertices: obj.vertices.length,
    edges: computeEdgeCount(obj),
    faces: obj.faces.length,
    area: computeSurfaceArea(obj),
    volume: Math.abs(computeSignedVolume(obj)),
    eulerCharacteristic: computeEulerCharacteristic(obj),
    genus: computeGenus(obj),
    watertight: isWatertight(obj),
  };
}
