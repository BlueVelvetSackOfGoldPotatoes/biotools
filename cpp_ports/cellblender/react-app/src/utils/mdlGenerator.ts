// MDL Generator - mirrors C++ MDLWriter from cellblender.h
// Generates MCell MDL text from the DataModel

import type { DataModel, MoleculeSpecies, ReleaseSite, ReactionOutput } from '../types/dataModel';

function orientStr(orient: string): string {
  switch (orient) {
    case 'TOP_FRONT': return "'";
    case 'BOTTOM_BACK': case 'TOP_BACK': return ',';
    default: return ';';
  }
}

export function generateMDL(dm: DataModel): string {
  const lines: string[] = [];

  // Parameters
  if (dm.parameters.length > 0) {
    lines.push('/* Parameters */');
    for (const p of dm.parameters) {
      if (p.expression) lines.push(`${p.name} = ${p.expression}`);
    }
    lines.push('');
  }

  // Initialization
  lines.push(`ITERATIONS = ${dm.initialization.iterations}`);
  lines.push(`TIME_STEP = ${dm.initialization.timeStep}`);
  if (dm.initialization.vacancySearchDistance)
    lines.push(`VACANCY_SEARCH_DISTANCE = ${dm.initialization.vacancySearchDistance}`);
  lines.push('');

  if (dm.initialization.timeStepMax)
    lines.push(`TIME_STEP_MAX = ${dm.initialization.timeStepMax}`);
  if (dm.initialization.spaceStep)
    lines.push(`SPACE_STEP = ${dm.initialization.spaceStep}`);
  if (dm.initialization.interactionRadius)
    lines.push(`INTERACTION_RADIUS = ${dm.initialization.interactionRadius}`);
  if (dm.initialization.radialDirections)
    lines.push(`RADIAL_DIRECTIONS = ${dm.initialization.radialDirections}`);
  if (dm.initialization.radialSubdivisions)
    lines.push(`RADIAL_SUBDIVISIONS = ${dm.initialization.radialSubdivisions}`);
  lines.push(`SURFACE_GRID_DENSITY = ${dm.initialization.surfaceGridDensity}`);
  lines.push(`ACCURATE_3D_REACTIONS = ${dm.initialization.accurate3dReactions ? 'TRUE' : 'FALSE'}`);
  lines.push(`CENTER_MOLECULES_ON_GRID = ${dm.initialization.centerMoleculesGrid ? 'TRUE' : 'FALSE'}`);
  lines.push(`MICROSCOPIC_REVERSIBILITY = ${dm.initialization.microscopicReversibility}`);
  lines.push('');

  // Notifications
  lines.push('NOTIFICATIONS');
  lines.push('{');
  if (dm.initialization.allNotifications === 'INDIVIDUAL') {
    if (dm.initialization.probabilityReport === 'THRESHOLD')
      lines.push(`   PROBABILITY_REPORT_THRESHOLD = ${dm.initialization.probabilityReportThreshold}`);
    else
      lines.push(`   PROBABILITY_REPORT = ${dm.initialization.probabilityReport}`);
    lines.push(`   DIFFUSION_CONSTANT_REPORT = ${dm.initialization.diffusionConstantReport}`);
    lines.push(`   FILE_OUTPUT_REPORT = ${dm.initialization.fileOutputReport ? 'ON' : 'OFF'}`);
    lines.push(`   FINAL_SUMMARY = ${dm.initialization.finalSummary ? 'ON' : 'OFF'}`);
    lines.push(`   ITERATION_REPORT = ${dm.initialization.iterationReport ? 'ON' : 'OFF'}`);
    lines.push(`   PARTITION_LOCATION_REPORT = ${dm.initialization.partitionLocationReport ? 'ON' : 'OFF'}`);
    lines.push(`   VARYING_PROBABILITY_REPORT = ${dm.initialization.varyingProbabilityReport ? 'ON' : 'OFF'}`);
    lines.push(`   PROGRESS_REPORT = ${dm.initialization.progressReport ? 'ON' : 'OFF'}`);
    lines.push(`   RELEASE_EVENT_REPORT = ${dm.initialization.releaseEventReport ? 'ON' : 'OFF'}`);
    lines.push(`   MOLECULE_COLLISION_REPORT = ${dm.initialization.moleculeCollisionReport ? 'ON' : 'OFF'}`);
  } else {
    lines.push(`   ALL_NOTIFICATIONS = ${dm.initialization.allNotifications}`);
  }
  lines.push('}');
  lines.push('');

  // Warnings
  lines.push('WARNINGS');
  lines.push('{');
  if (dm.initialization.allWarnings === 'INDIVIDUAL') {
    lines.push(`   DEGENERATE_POLYGONS = ${dm.initialization.degeneratePolygons}`);
    lines.push(`   NEGATIVE_DIFFUSION_CONSTANT = ${dm.initialization.negativeDiffusionConstant}`);
    lines.push(`   MISSING_SURFACE_ORIENTATION = ${dm.initialization.missingSurfaceOrientation}`);
    lines.push(`   NEGATIVE_REACTION_RATE = ${dm.initialization.negativeReactionRate}`);
    lines.push(`   USELESS_VOLUME_ORIENTATION = ${dm.initialization.uselessVolumeOrientation}`);
    lines.push(`   HIGH_REACTION_PROBABILITY = ${dm.initialization.highReactionProbability}`);
    lines.push(`   LIFETIME_TOO_SHORT = ${dm.initialization.lifetimeTooShort}`);
    if (dm.initialization.lifetimeTooShort === 'WARNING')
      lines.push(`   LIFETIME_THRESHOLD = ${dm.initialization.lifetimeThreshold}`);
    lines.push(`   MISSED_REACTIONS = ${dm.initialization.missedReactions}`);
    if (dm.initialization.missedReactions === 'WARNING')
      lines.push(`   MISSED_REACTION_THRESHOLD = ${dm.initialization.missedReactionThreshold}`);
  } else {
    lines.push(`   ALL_WARNINGS = ${dm.initialization.allWarnings}`);
  }
  lines.push('}');
  lines.push('');

  // Partitions
  if (dm.initialization.partitions.include) {
    const p = dm.initialization.partitions;
    lines.push(`PARTITION_X = [[${p.xStart} TO ${p.xEnd} STEP ${p.xStep}]]`);
    lines.push(`PARTITION_Y = [[${p.yStart} TO ${p.yEnd} STEP ${p.yStep}]]`);
    lines.push(`PARTITION_Z = [[${p.zStart} TO ${p.zEnd} STEP ${p.zStep}]]`);
    lines.push('');
  }

  // Molecules
  if (dm.molecules.length > 0) {
    lines.push('DEFINE_MOLECULES');
    lines.push('{');
    for (const m of dm.molecules) {
      lines.push(`  ${m.name}`);
      lines.push('  {');
      const dc = m.molType === '2D' ? 'DIFFUSION_CONSTANT_2D' : 'DIFFUSION_CONSTANT_3D';
      lines.push(`    ${dc} = ${m.diffusionConstant}`);
      if (m.customTimeStep) lines.push(`    CUSTOM_TIME_STEP = ${m.customTimeStep}`);
      if (m.customSpaceStep) lines.push(`    CUSTOM_SPACE_STEP = ${m.customSpaceStep}`);
      if (m.maximumStepLength) lines.push(`    MAXIMUM_STEP_LENGTH = ${m.maximumStepLength}`);
      if (m.targetOnly) lines.push('    TARGET_ONLY');
      lines.push('  }');
    }
    lines.push('}');
    lines.push('');
  }

  // Surface Classes
  if (dm.surfaceClasses.length > 0) {
    lines.push('DEFINE_SURFACE_CLASSES');
    lines.push('{');
    for (const sc of dm.surfaceClasses) {
      lines.push(`  ${sc.name}`);
      lines.push('  {');
      for (const prop of sc.properties) {
        let molStr: string;
        if (prop.affectedMols === 'SINGLE')
          molStr = prop.molecule + orientStr(prop.orient);
        else
          molStr = prop.affectedMols + orientStr(prop.orient);
        let line = `    ${prop.classType} = ${molStr}`;
        if (prop.classType === 'CLAMP_CONCENTRATION')
          line += ` = ${prop.clampValue}`;
        lines.push(line);
      }
      lines.push('  }');
    }
    lines.push('}');
    lines.push('');
  }

  // Reactions
  if (dm.reactions.length > 0) {
    lines.push('DEFINE_REACTIONS');
    lines.push('{');
    for (const r of dm.reactions) {
      const arrow = r.rxnType === 'reversible' ? '<->' : '->';
      const rxnExpr = `${r.reactants} ${arrow} ${r.products}`;
      let rateStr: string;
      if (r.rxnType === 'reversible') {
        rateStr = `[>${r.fwdRate}, <${r.bkwdRate}]`;
      } else {
        if (r.variableRateSwitch && r.variableRateValid)
          rateStr = `["${r.variableRate}"]`;
        else
          rateStr = `[${r.fwdRate}]`;
      }
      let line = `  ${rxnExpr} ${rateStr}`;
      if (r.rxnName) line += ` : ${r.rxnName}`;
      lines.push(line);
    }
    lines.push('}');
    lines.push('');
  }

  // Geometry
  if (dm.geometryObjects.length > 0) {
    for (const obj of dm.geometryObjects) {
      lines.push(`${obj.name} POLYGON_LIST`);
      lines.push('{');
      lines.push('  VERTEX_LIST');
      lines.push('  {');
      for (const v of obj.vertices) {
        lines.push(`    [ ${v.x + obj.location.x}, ${v.y + obj.location.y}, ${v.z + obj.location.z} ]`);
      }
      lines.push('  }');
      lines.push('  ELEMENT_CONNECTIONS');
      lines.push('  {');
      for (const f of obj.faces) {
        lines.push(`    [ ${f.v0}, ${f.v1}, ${f.v2} ]`);
      }
      lines.push('  }');
      if (obj.regions.length > 0) {
        lines.push('  DEFINE_SURFACE_REGIONS');
        lines.push('  {');
        for (const reg of obj.regions) {
          lines.push(`    ${reg.name}`);
          lines.push('    {');
          lines.push(`      ELEMENT_LIST = [${reg.faceIndices.join(', ')}]`);
          lines.push('    }');
        }
        lines.push('  }');
      }
      lines.push('}');
      lines.push('');
    }
  }

  // Modify Surface Regions
  if (dm.modSurfRegions.length > 0) {
    lines.push('MODIFY_SURFACE_REGIONS');
    lines.push('{');
    for (const msr of dm.modSurfRegions) {
      lines.push(`  ${msr.objectName}[${msr.regionName}]`);
      lines.push('  {');
      lines.push(`    SURFACE_CLASS = ${msr.surfClassName}`);
      lines.push('  }');
    }
    lines.push('}');
    lines.push('');
  }

  // Release Patterns
  if (dm.releasePatterns.length > 0) {
    for (const rp of dm.releasePatterns) {
      lines.push(`DEFINE_RELEASE_PATTERN ${rp.name}`);
      lines.push('{');
      lines.push(`  DELAY = ${rp.delay}`);
      if (rp.releaseInterval) lines.push(`  RELEASE_INTERVAL = ${rp.releaseInterval}`);
      if (rp.trainDuration) lines.push(`  TRAIN_DURATION = ${rp.trainDuration}`);
      if (rp.trainInterval) lines.push(`  TRAIN_INTERVAL = ${rp.trainInterval}`);
      lines.push(`  NUMBER_OF_TRAINS = ${rp.numberOfTrains}`);
      lines.push('}');
      lines.push('');
    }
  }

  // Instantiation
  const hasObjects = dm.geometryObjects.length > 0;
  const hasReleases = dm.releaseSites.length > 0;
  if (hasObjects || hasReleases) {
    lines.push(`INSTANTIATE ${dm.sceneName} OBJECT`);
    lines.push('{');
    for (const obj of dm.geometryObjects) {
      lines.push(`  ${obj.name} OBJECT ${obj.name} {}`);
    }
    for (const rel of dm.releaseSites) {
      lines.push(`  ${rel.name} RELEASE_SITE`);
      lines.push('  {');
      if (['CUBIC', 'SPHERICAL', 'SPHERICAL_SHELL', 'LIST'].includes(rel.shape)) {
        lines.push(`   SHAPE = ${rel.shape}`);
        if (rel.shape !== 'LIST')
          lines.push(`   LOCATION = [${rel.locationX}, ${rel.locationY}, ${rel.locationZ}]`);
        lines.push(`   SITE_DIAMETER = ${rel.diameter}`);
      }
      if (rel.shape === 'OBJECT')
        lines.push(`   SHAPE = ${dm.sceneName}.${rel.objectExpr}`);

      let molSpec = rel.molecule;
      const molDef = dm.molecules.find(m => m.name === rel.molecule);
      if (molDef && molDef.molType === '2D') {
        molSpec = rel.molecule + orientStr(rel.orient);
      }

      if (rel.shape === 'LIST') {
        lines.push('   MOLECULE_POSITIONS');
        lines.push('   {');
        for (const p of rel.pointsList) {
          lines.push(`     ${molSpec} [${p.x}, ${p.y}, ${p.z}]`);
        }
        lines.push('   }');
      } else {
        lines.push(`   MOLECULE = ${molSpec}`);
        if (rel.quantityType === 'NUMBER_TO_RELEASE') {
          lines.push(`   NUMBER_TO_RELEASE = ${rel.quantity}`);
        } else if (rel.quantityType === 'GAUSSIAN_RELEASE_NUMBER') {
          lines.push('   GAUSSIAN_RELEASE_NUMBER');
          lines.push('   {');
          lines.push(`        MEAN_NUMBER = ${rel.quantity}`);
          lines.push(`        STANDARD_DEVIATION = ${rel.stddev}`);
          lines.push('   }');
        } else if (rel.quantityType === 'DENSITY') {
          const is2d = molDef?.molType === '2D';
          lines.push(`   ${is2d ? 'DENSITY' : 'CONCENTRATION'} = ${rel.quantity}`);
        }
      }
      lines.push(`   RELEASE_PROBABILITY = ${rel.probability}`);
      if (rel.pattern) lines.push(`   RELEASE_PATTERN = ${rel.pattern}`);
      lines.push('  }');
    }
    lines.push('}');
    lines.push('');
  }

  lines.push('sprintf(seed,"%05g",SEED)');
  lines.push('');

  // Viz Output
  let molListStr = '';
  if (dm.vizOutput.exportAll) {
    molListStr = 'ALL_MOLECULES';
  } else {
    molListStr = dm.molecules.filter(m => m.exportViz).map(m => m.name).join(' ');
  }
  if (molListStr) {
    lines.push('VIZ_OUTPUT');
    lines.push('{');
    lines.push(`  MODE = ${dm.initialization.exportAllAscii ? 'ASCII' : 'CELLBLENDER'}`);
    lines.push(`  FILENAME = "./viz_data/seed_" & seed & "/${dm.sceneName}"`);
    lines.push('  MOLECULES');
    lines.push('  {');
    lines.push(`    NAME_LIST {${molListStr}}`);
    if (dm.vizOutput.allIterations)
      lines.push('    ITERATION_NUMBERS {ALL_DATA @ ALL_ITERATIONS}');
    else
      lines.push(`    ITERATION_NUMBERS {ALL_DATA @ [[${dm.vizOutput.start} TO ${dm.vizOutput.endVal} STEP ${dm.vizOutput.step}]]}`);
    lines.push('  }');
    lines.push('}');
    lines.push('');
  }

  // Reaction Data Output
  if (dm.reactionOutputs.length > 0) {
    lines.push('REACTION_DATA_OUTPUT');
    lines.push('{');
    lines.push(`  STEP = ${dm.initialization.timeStep}`);
    for (const ro of dm.reactionOutputs) {
      let countExpr = '';
      let fileName = '';
      if (ro.rxnOrMol === 'Molecule') {
        if (ro.countLocation === 'World') {
          countExpr = `COUNT[${ro.moleculeName},WORLD]`;
          fileName = `${ro.moleculeName}.World.dat`;
        } else if (ro.countLocation === 'Object') {
          countExpr = `COUNT[${ro.moleculeName},${dm.sceneName}.${ro.objectName}]`;
          fileName = `${ro.moleculeName}.${ro.objectName}.dat`;
        } else {
          countExpr = `COUNT[${ro.moleculeName},${dm.sceneName}.${ro.objectName}[${ro.regionName}]]`;
          fileName = `${ro.moleculeName}.${ro.objectName}.${ro.regionName}.dat`;
        }
      } else if (ro.rxnOrMol === 'Reaction') {
        countExpr = `COUNT[${ro.reactionName},WORLD]`;
        fileName = `${ro.reactionName}.World.dat`;
      } else if (ro.rxnOrMol === 'MDLString') {
        countExpr = ro.mdlString;
        fileName = `${ro.mdlFilePrefix}_MDLString.dat`;
      }
      if (countExpr && fileName) {
        lines.push(`  { ${countExpr} } => "./react_data/seed_" & seed & "/${fileName}"`);
      }
    }
    lines.push('}');
    lines.push('');
  }

  return lines.join('\n');
}
