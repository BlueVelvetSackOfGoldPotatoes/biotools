// Project I/O - Save/Load CellBlender projects as JSON
// Mirrors C++ DataModelIO from cellblender.h

import type { DataModel } from '../types/dataModel';

export function saveProject(dm: DataModel): string {
  return JSON.stringify(dm, null, 2);
}

export function loadProject(json: string): DataModel {
  return JSON.parse(json) as DataModel;
}

export function downloadProject(dm: DataModel, filename: string = 'cellblender_project.json') {
  const blob = new Blob([saveProject(dm)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadMDL(mdlText: string, filename: string = 'Scene.main.mdl') {
  const blob = new Blob([mdlText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function loadProjectFromFile(file: File): Promise<DataModel> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = loadProject(e.target?.result as string);
        resolve(data);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}
