// CellBlender Data Model Types
// Mirrors the C++ DataModel structs from cellblender.h

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Face {
  v0: number;
  v1: number;
  v2: number;
}

export interface SurfaceRegion {
  name: string;
  faceIndices: number[];
}

export interface GeometryObject {
  name: string;
  vertices: Vec3[];
  faces: Face[];
  regions: SurfaceRegion[];
  location: Vec3;
  parentObject: string;
  membraneName: string;
  dynamic: boolean;
}

export interface MoleculeComponent {
  name: string;
  states: string[];
  isKey: boolean;
  location: Vec3;
  rotationAxis: Vec3;
  rotationAngle: number;
  rotIndex: number;
}

export type MoleculeType = '3D' | '2D';

export interface MoleculeSpecies {
  id: string;
  name: string;
  description: string;
  molType: MoleculeType;
  diffusionConstant: string;
  targetOnly: boolean;
  customTimeStep: string;
  customSpaceStep: string;
  maximumStepLength: string;
  exportViz: boolean;
  bnglLabel: string;
  spatialStructure: string;
  components: MoleculeComponent[];
}

export type ReactionType = 'irreversible' | 'reversible';

export interface Reaction {
  id: string;
  name: string;
  description: string;
  rxnName: string;
  reactants: string;
  products: string;
  rxnType: ReactionType;
  fwdRate: string;
  bkwdRate: string;
  variableRateSwitch: boolean;
  variableRate: string;
  variableRateText: string;
  variableRateValid: boolean;
}

export type SurfaceClassPropertyAffected =
  | 'ALL_MOLECULES'
  | 'ALL_VOLUME_MOLECULES'
  | 'ALL_SURFACE_MOLECULES'
  | 'SINGLE';

export type SurfaceClassPropertyOrient = 'TOP_FRONT' | 'BOTTOM_BACK' | 'IGNORE';

export type SurfaceClassPropertyType =
  | 'ABSORPTIVE'
  | 'TRANSPARENT'
  | 'REFLECTIVE'
  | 'CLAMP_CONCENTRATION';

export interface SurfaceClassProperty {
  id: string;
  affectedMols: SurfaceClassPropertyAffected;
  molecule: string;
  orient: SurfaceClassPropertyOrient;
  classType: SurfaceClassPropertyType;
  clampValue: string;
}

export interface SurfaceClass {
  id: string;
  name: string;
  description: string;
  properties: SurfaceClassProperty[];
}

export interface ModSurfaceRegion {
  id: string;
  name: string;
  objectName: string;
  regionName: string;
  surfClassName: string;
}

export interface ReleasePattern {
  id: string;
  name: string;
  description: string;
  delay: string;
  releaseInterval: string;
  trainDuration: string;
  trainInterval: string;
  numberOfTrains: string;
}

export type ReleaseShape = 'CUBIC' | 'SPHERICAL' | 'SPHERICAL_SHELL' | 'LIST' | 'OBJECT';
export type ReleaseOrient = 'TOP_FRONT' | 'TOP_BACK' | 'MIXED';
export type QuantityType = 'NUMBER_TO_RELEASE' | 'GAUSSIAN_RELEASE_NUMBER' | 'DENSITY';

export interface ReleaseSite {
  id: string;
  name: string;
  description: string;
  molecule: string;
  shape: ReleaseShape;
  orient: ReleaseOrient;
  objectExpr: string;
  locationX: string;
  locationY: string;
  locationZ: string;
  diameter: string;
  probability: string;
  quantityType: QuantityType;
  quantity: string;
  stddev: string;
  pattern: string;
  pointsList: Vec3[];
}

export type RxnOrMol = 'Molecule' | 'Reaction' | 'MDLString' | 'FileOutput';
export type CountLocation = 'World' | 'Object' | 'Region';

export interface ReactionOutput {
  id: string;
  name: string;
  rxnOrMol: RxnOrMol;
  moleculeName: string;
  reactionName: string;
  objectName: string;
  regionName: string;
  countLocation: CountLocation;
  mdlString: string;
  mdlFilePrefix: string;
  dataFileName: string;
  plottingEnabled: boolean;
}

export interface Partition {
  include: boolean;
  xStart: number;
  xEnd: number;
  xStep: number;
  yStart: number;
  yEnd: number;
  yStep: number;
  zStart: number;
  zEnd: number;
  zStep: number;
}

export interface VizOutput {
  exportAll: boolean;
  allIterations: boolean;
  start: string;
  endVal: string;
  step: string;
}

export interface InitializationParams {
  iterations: string;
  timeStep: string;
  vacancySearchDistance: string;
  timeStepMax: string;
  spaceStep: string;
  interactionRadius: string;
  radialDirections: string;
  radialSubdivisions: string;
  surfaceGridDensity: string;
  accurate3dReactions: boolean;
  centerMoleculesGrid: boolean;
  microscopicReversibility: string;
  exportAllAscii: boolean;
  // Notifications
  allNotifications: string;
  probabilityReport: string;
  probabilityReportThreshold: number;
  diffusionConstantReport: string;
  fileOutputReport: boolean;
  finalSummary: boolean;
  iterationReport: boolean;
  partitionLocationReport: boolean;
  varyingProbabilityReport: boolean;
  progressReport: boolean;
  releaseEventReport: boolean;
  moleculeCollisionReport: boolean;
  // Warnings
  allWarnings: string;
  degeneratePolygons: string;
  negativeDiffusionConstant: string;
  missingSurfaceOrientation: string;
  negativeReactionRate: string;
  uselessVolumeOrientation: string;
  highReactionProbability: string;
  lifetimeTooShort: string;
  lifetimeThreshold: number;
  missedReactions: string;
  missedReactionThreshold: number;
  // Partitions
  partitions: Partition;
}

export interface SimulationControl {
  startSeed: number;
  endSeed: number;
  exportFormat: string;
  mcellBinary: string;
}

export interface ParameterDef {
  id: string;
  name: string;
  expression: string;
  value: number;
}

export interface SweepParameter {
  id: string;
  name: string;
  start: number;
  end: number;
  step: number;
  values: number[];
}

export interface TimeSeriesData {
  label: string;
  times: number[];
  values: number[];
}

export interface AggregatedTimeSeries {
  label: string;
  times: number[];
  means: number[];
  stddevs: number[];
  numSeeds: number;
}

// Top-level Data Model
export interface DataModel {
  parameters: ParameterDef[];
  molecules: MoleculeSpecies[];
  reactions: Reaction[];
  releaseSites: ReleaseSite[];
  releasePatterns: ReleasePattern[];
  surfaceClasses: SurfaceClass[];
  modSurfRegions: ModSurfaceRegion[];
  geometryObjects: GeometryObject[];
  reactionOutputs: ReactionOutput[];
  initialization: InitializationParams;
  vizOutput: VizOutput;
  simControl: SimulationControl;
  sceneName: string;
}

export type SimulationStatus = 'idle' | 'preparing' | 'running' | 'completed' | 'error';

export interface SimulationState {
  status: SimulationStatus;
  progress: number;
  currentSeed: number;
  totalSeeds: number;
  currentIteration: number;
  totalIterations: number;
  errorMessage: string;
  startTime: number | null;
  elapsedTime: number;
}
