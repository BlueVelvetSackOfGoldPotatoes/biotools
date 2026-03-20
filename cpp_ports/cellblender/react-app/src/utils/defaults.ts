// Default values for the CellBlender data model

import type {
  DataModel,
  InitializationParams,
  VizOutput,
  SimulationControl,
  Partition,
  SimulationState,
  MoleculeSpecies,
  Reaction,
  ReleaseSite,
  SurfaceClass,
  SurfaceClassProperty,
  ReleasePattern,
  ReactionOutput,
  GeometryObject,
  ModSurfaceRegion,
  ParameterDef,
  SweepParameter,
  Vec3,
  Face,
} from '../types/dataModel';

let nextId = 1;
export function generateId(): string {
  return `cb_${Date.now()}_${nextId++}`;
}

export function createDefaultPartition(): Partition {
  return {
    include: false,
    xStart: -1, xEnd: 1, xStep: 0.05,
    yStart: -1, yEnd: 1, yStep: 0.05,
    zStart: -1, zEnd: 1, zStep: 0.05,
  };
}

export function createDefaultInitialization(): InitializationParams {
  return {
    iterations: '1000',
    timeStep: '1e-6',
    vacancySearchDistance: '10',
    timeStepMax: '',
    spaceStep: '',
    interactionRadius: '',
    radialDirections: '',
    radialSubdivisions: '',
    surfaceGridDensity: '10000',
    accurate3dReactions: true,
    centerMoleculesGrid: false,
    microscopicReversibility: 'ON',
    exportAllAscii: false,
    allNotifications: 'INDIVIDUAL',
    probabilityReport: 'ON',
    probabilityReportThreshold: 0,
    diffusionConstantReport: 'BRIEF',
    fileOutputReport: false,
    finalSummary: true,
    iterationReport: true,
    partitionLocationReport: false,
    varyingProbabilityReport: true,
    progressReport: true,
    releaseEventReport: true,
    moleculeCollisionReport: false,
    allWarnings: 'INDIVIDUAL',
    degeneratePolygons: 'WARNING',
    negativeDiffusionConstant: 'WARNING',
    missingSurfaceOrientation: 'ERROR',
    negativeReactionRate: 'WARNING',
    uselessVolumeOrientation: 'WARNING',
    highReactionProbability: 'IGNORED',
    lifetimeTooShort: 'WARNING',
    lifetimeThreshold: 50,
    missedReactions: 'WARNING',
    missedReactionThreshold: 0.001,
    partitions: createDefaultPartition(),
  };
}

export function createDefaultVizOutput(): VizOutput {
  return {
    exportAll: true,
    allIterations: true,
    start: '0',
    endVal: '1',
    step: '1',
  };
}

export function createDefaultSimControl(): SimulationControl {
  return {
    startSeed: 1,
    endSeed: 1,
    exportFormat: 'mcell_mdl_modular',
    mcellBinary: 'mcell',
  };
}

export function createDefaultDataModel(): DataModel {
  return {
    parameters: [],
    molecules: [],
    reactions: [],
    releaseSites: [],
    releasePatterns: [],
    surfaceClasses: [],
    modSurfRegions: [],
    geometryObjects: [],
    reactionOutputs: [],
    initialization: createDefaultInitialization(),
    vizOutput: createDefaultVizOutput(),
    simControl: createDefaultSimControl(),
    sceneName: 'Scene',
  };
}

export function createDefaultSimState(): SimulationState {
  return {
    status: 'idle',
    progress: 0,
    currentSeed: 0,
    totalSeeds: 0,
    currentIteration: 0,
    totalIterations: 0,
    errorMessage: '',
    startTime: null,
    elapsedTime: 0,
  };
}

export function createDefaultMolecule(): MoleculeSpecies {
  return {
    id: generateId(),
    name: '',
    description: '',
    molType: '3D',
    diffusionConstant: '1e-6',
    targetOnly: false,
    customTimeStep: '',
    customSpaceStep: '',
    maximumStepLength: '',
    exportViz: true,
    bnglLabel: '',
    spatialStructure: 'None',
    components: [],
  };
}

export function createDefaultReaction(): Reaction {
  return {
    id: generateId(),
    name: '',
    description: '',
    rxnName: '',
    reactants: '',
    products: '',
    rxnType: 'irreversible',
    fwdRate: '0',
    bkwdRate: '',
    variableRateSwitch: false,
    variableRate: '',
    variableRateText: '',
    variableRateValid: false,
  };
}

export function createDefaultReleaseSite(): ReleaseSite {
  return {
    id: generateId(),
    name: 'Release_Site',
    description: '',
    molecule: '',
    shape: 'SPHERICAL',
    orient: 'TOP_FRONT',
    objectExpr: '',
    locationX: '0',
    locationY: '0',
    locationZ: '0',
    diameter: '0',
    probability: '1',
    quantityType: 'NUMBER_TO_RELEASE',
    quantity: '100',
    stddev: '0',
    pattern: '',
    pointsList: [],
  };
}

export function createDefaultReleasePattern(): ReleasePattern {
  return {
    id: generateId(),
    name: 'Release_Pattern',
    description: '',
    delay: '0',
    releaseInterval: '',
    trainDuration: '',
    trainInterval: '0',
    numberOfTrains: '1',
  };
}

export function createDefaultSurfaceClass(): SurfaceClass {
  return {
    id: generateId(),
    name: 'Surface_Class',
    description: '',
    properties: [],
  };
}

export function createDefaultSurfaceClassProperty(): SurfaceClassProperty {
  return {
    id: generateId(),
    affectedMols: 'ALL_MOLECULES',
    molecule: '',
    orient: 'IGNORE',
    classType: 'TRANSPARENT',
    clampValue: '0',
  };
}

export function createDefaultReactionOutput(): ReactionOutput {
  return {
    id: generateId(),
    name: '',
    rxnOrMol: 'Molecule',
    moleculeName: '',
    reactionName: '',
    objectName: '',
    regionName: '',
    countLocation: 'World',
    mdlString: '',
    mdlFilePrefix: '',
    dataFileName: '',
    plottingEnabled: true,
  };
}

export function createDefaultModSurfRegion(): ModSurfaceRegion {
  return {
    id: generateId(),
    name: '',
    objectName: '',
    regionName: '',
    surfClassName: '',
  };
}

export function createDefaultParameter(): ParameterDef {
  return {
    id: generateId(),
    name: '',
    expression: '',
    value: 0,
  };
}

export function createDefaultSweepParameter(): SweepParameter {
  return {
    id: generateId(),
    name: '',
    start: 0,
    end: 1,
    step: 0.1,
    values: [],
  };
}

// Primitive geometry factories (matching C++ geometry namespace)
export function makeCube(name: string, halfSize: number = 1.0): GeometryObject {
  const s = halfSize;
  return {
    name,
    vertices: [
      { x: -s, y: -s, z: -s }, { x: s, y: -s, z: -s },
      { x: s, y: s, z: -s },   { x: -s, y: s, z: -s },
      { x: -s, y: -s, z: s },  { x: s, y: -s, z: s },
      { x: s, y: s, z: s },    { x: -s, y: s, z: s },
    ],
    faces: [
      { v0: 0, v1: 1, v2: 2 }, { v0: 0, v1: 2, v2: 3 },
      { v0: 4, v1: 6, v2: 5 }, { v0: 4, v1: 7, v2: 6 },
      { v0: 0, v1: 4, v2: 5 }, { v0: 0, v1: 5, v2: 1 },
      { v0: 2, v1: 6, v2: 7 }, { v0: 2, v1: 7, v2: 3 },
      { v0: 0, v1: 3, v2: 7 }, { v0: 0, v1: 7, v2: 4 },
      { v0: 1, v1: 5, v2: 6 }, { v0: 1, v1: 6, v2: 2 },
    ],
    regions: [],
    location: { x: 0, y: 0, z: 0 },
    parentObject: '',
    membraneName: '',
    dynamic: false,
  };
}

export function makeIcosphere(name: string, radius: number = 1.0, subdivisions: number = 1): GeometryObject {
  const phi = (1.0 + Math.sqrt(5.0)) / 2.0;
  const vertices: Vec3[] = [];
  const addV = (x: number, y: number, z: number): number => {
    const l = Math.sqrt(x * x + y * y + z * z);
    vertices.push({ x: (x / l) * radius, y: (y / l) * radius, z: (z / l) * radius });
    return vertices.length - 1;
  };

  addV(-1, phi, 0); addV(1, phi, 0); addV(-1, -phi, 0); addV(1, -phi, 0);
  addV(0, -1, phi); addV(0, 1, phi); addV(0, -1, -phi); addV(0, 1, -phi);
  addV(phi, 0, -1); addV(phi, 0, 1); addV(-phi, 0, -1); addV(-phi, 0, 1);

  let faces: Face[] = [
    { v0: 0, v1: 11, v2: 5 }, { v0: 0, v1: 5, v2: 1 }, { v0: 0, v1: 1, v2: 7 },
    { v0: 0, v1: 7, v2: 10 }, { v0: 0, v1: 10, v2: 11 }, { v0: 1, v1: 5, v2: 9 },
    { v0: 5, v1: 11, v2: 4 }, { v0: 11, v1: 10, v2: 2 }, { v0: 10, v1: 7, v2: 6 },
    { v0: 7, v1: 1, v2: 8 }, { v0: 3, v1: 9, v2: 4 }, { v0: 3, v1: 4, v2: 2 },
    { v0: 3, v1: 2, v2: 6 }, { v0: 3, v1: 6, v2: 8 }, { v0: 3, v1: 8, v2: 9 },
    { v0: 4, v1: 9, v2: 5 }, { v0: 2, v1: 4, v2: 11 }, { v0: 6, v1: 2, v2: 10 },
    { v0: 8, v1: 6, v2: 7 }, { v0: 9, v1: 8, v2: 1 },
  ];

  for (let s = 0; s < subdivisions; s++) {
    const cache = new Map<string, number>();
    const newFaces: Face[] = [];
    const mid = (a: number, b: number): number => {
      const key = `${Math.min(a, b)}_${Math.max(a, b)}`;
      if (cache.has(key)) return cache.get(key)!;
      const m = {
        x: (vertices[a].x + vertices[b].x) / 2,
        y: (vertices[a].y + vertices[b].y) / 2,
        z: (vertices[a].z + vertices[b].z) / 2,
      };
      const idx = addV(m.x, m.y, m.z);
      cache.set(key, idx);
      return idx;
    };
    for (const f of faces) {
      const a = mid(f.v0, f.v1);
      const b = mid(f.v1, f.v2);
      const c = mid(f.v2, f.v0);
      newFaces.push({ v0: f.v0, v1: a, v2: c });
      newFaces.push({ v0: f.v1, v1: b, v2: a });
      newFaces.push({ v0: f.v2, v1: c, v2: b });
      newFaces.push({ v0: a, v1: b, v2: c });
    }
    faces = newFaces;
  }

  return {
    name,
    vertices,
    faces,
    regions: [],
    location: { x: 0, y: 0, z: 0 },
    parentObject: '',
    membraneName: '',
    dynamic: false,
  };
}
