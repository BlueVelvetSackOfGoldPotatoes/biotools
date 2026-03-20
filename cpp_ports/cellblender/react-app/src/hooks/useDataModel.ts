// Central state management hook for the CellBlender data model

import { useState, useCallback } from 'react';
import type {
  DataModel, MoleculeSpecies, Reaction, ReleaseSite,
  SurfaceClass, ReleasePattern, ReactionOutput, GeometryObject,
  ModSurfaceRegion, ParameterDef, SweepParameter, SimulationState,
} from '../types/dataModel';
import {
  createDefaultDataModel, createDefaultSimState,
  createDefaultMolecule, createDefaultReaction,
  createDefaultReleaseSite, createDefaultReleasePattern,
  createDefaultSurfaceClass, createDefaultReactionOutput,
  createDefaultModSurfRegion, createDefaultParameter,
  createDefaultSweepParameter,
} from '../utils/defaults';

export function useDataModel() {
  const [dataModel, setDataModel] = useState<DataModel>(createDefaultDataModel());
  const [simState, setSimState] = useState<SimulationState>(createDefaultSimState());
  const [sweepParameters, setSweepParameters] = useState<SweepParameter[]>([]);

  // Helper to update nested state
  const updateDM = useCallback(<K extends keyof DataModel>(key: K, value: DataModel[K]) => {
    setDataModel(prev => ({ ...prev, [key]: value }));
  }, []);

  // === Molecules ===
  const addMolecule = useCallback(() => {
    const mol = createDefaultMolecule();
    mol.name = `Molecule_${dataModel.molecules.length + 1}`;
    setDataModel(prev => ({ ...prev, molecules: [...prev.molecules, mol] }));
    return mol;
  }, [dataModel.molecules.length]);

  const updateMolecule = useCallback((id: string, updates: Partial<MoleculeSpecies>) => {
    setDataModel(prev => ({
      ...prev,
      molecules: prev.molecules.map(m => m.id === id ? { ...m, ...updates } : m),
    }));
  }, []);

  const removeMolecule = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      molecules: prev.molecules.filter(m => m.id !== id),
    }));
  }, []);

  // === Reactions ===
  const addReaction = useCallback(() => {
    const rxn = createDefaultReaction();
    setDataModel(prev => ({ ...prev, reactions: [...prev.reactions, rxn] }));
    return rxn;
  }, []);

  const updateReaction = useCallback((id: string, updates: Partial<Reaction>) => {
    setDataModel(prev => ({
      ...prev,
      reactions: prev.reactions.map(r => {
        if (r.id !== id) return r;
        const updated = { ...r, ...updates };
        const arrow = updated.rxnType === 'reversible' ? '<->' : '->';
        updated.name = `${updated.reactants} ${arrow} ${updated.products}`;
        return updated;
      }),
    }));
  }, []);

  const removeReaction = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      reactions: prev.reactions.filter(r => r.id !== id),
    }));
  }, []);

  // === Release Sites ===
  const addReleaseSite = useCallback(() => {
    const rel = createDefaultReleaseSite();
    rel.name = `Release_Site_${dataModel.releaseSites.length + 1}`;
    setDataModel(prev => ({ ...prev, releaseSites: [...prev.releaseSites, rel] }));
    return rel;
  }, [dataModel.releaseSites.length]);

  const updateReleaseSite = useCallback((id: string, updates: Partial<ReleaseSite>) => {
    setDataModel(prev => ({
      ...prev,
      releaseSites: prev.releaseSites.map(r => r.id === id ? { ...r, ...updates } : r),
    }));
  }, []);

  const removeReleaseSite = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      releaseSites: prev.releaseSites.filter(r => r.id !== id),
    }));
  }, []);

  // === Release Patterns ===
  const addReleasePattern = useCallback(() => {
    const rp = createDefaultReleasePattern();
    rp.name = `Release_Pattern_${dataModel.releasePatterns.length + 1}`;
    setDataModel(prev => ({ ...prev, releasePatterns: [...prev.releasePatterns, rp] }));
    return rp;
  }, [dataModel.releasePatterns.length]);

  const updateReleasePattern = useCallback((id: string, updates: Partial<ReleasePattern>) => {
    setDataModel(prev => ({
      ...prev,
      releasePatterns: prev.releasePatterns.map(r => r.id === id ? { ...r, ...updates } : r),
    }));
  }, []);

  const removeReleasePattern = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      releasePatterns: prev.releasePatterns.filter(r => r.id !== id),
    }));
  }, []);

  // === Surface Classes ===
  const addSurfaceClass = useCallback(() => {
    const sc = createDefaultSurfaceClass();
    sc.name = `Surface_Class_${dataModel.surfaceClasses.length + 1}`;
    setDataModel(prev => ({ ...prev, surfaceClasses: [...prev.surfaceClasses, sc] }));
    return sc;
  }, [dataModel.surfaceClasses.length]);

  const updateSurfaceClass = useCallback((id: string, updates: Partial<SurfaceClass>) => {
    setDataModel(prev => ({
      ...prev,
      surfaceClasses: prev.surfaceClasses.map(s => s.id === id ? { ...s, ...updates } : s),
    }));
  }, []);

  const removeSurfaceClass = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      surfaceClasses: prev.surfaceClasses.filter(s => s.id !== id),
    }));
  }, []);

  // === Geometry Objects ===
  const addGeometryObject = useCallback((obj: GeometryObject) => {
    setDataModel(prev => ({ ...prev, geometryObjects: [...prev.geometryObjects, obj] }));
  }, []);

  const updateGeometryObject = useCallback((name: string, updates: Partial<GeometryObject>) => {
    setDataModel(prev => ({
      ...prev,
      geometryObjects: prev.geometryObjects.map(g => g.name === name ? { ...g, ...updates } : g),
    }));
  }, []);

  const removeGeometryObject = useCallback((name: string) => {
    setDataModel(prev => ({
      ...prev,
      geometryObjects: prev.geometryObjects.filter(g => g.name !== name),
    }));
  }, []);

  // === Reaction Outputs ===
  const addReactionOutput = useCallback(() => {
    const ro = createDefaultReactionOutput();
    setDataModel(prev => ({ ...prev, reactionOutputs: [...prev.reactionOutputs, ro] }));
    return ro;
  }, []);

  const updateReactionOutput = useCallback((id: string, updates: Partial<ReactionOutput>) => {
    setDataModel(prev => ({
      ...prev,
      reactionOutputs: prev.reactionOutputs.map(r => r.id === id ? { ...r, ...updates } : r),
    }));
  }, []);

  const removeReactionOutput = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      reactionOutputs: prev.reactionOutputs.filter(r => r.id !== id),
    }));
  }, []);

  // === Mod Surface Regions ===
  const addModSurfRegion = useCallback(() => {
    const msr = createDefaultModSurfRegion();
    setDataModel(prev => ({ ...prev, modSurfRegions: [...prev.modSurfRegions, msr] }));
    return msr;
  }, []);

  const updateModSurfRegion = useCallback((id: string, updates: Partial<ModSurfaceRegion>) => {
    setDataModel(prev => ({
      ...prev,
      modSurfRegions: prev.modSurfRegions.map(m => m.id === id ? { ...m, ...updates } : m),
    }));
  }, []);

  const removeModSurfRegion = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      modSurfRegions: prev.modSurfRegions.filter(m => m.id !== id),
    }));
  }, []);

  // === Parameters ===
  const addParameter = useCallback(() => {
    const p = createDefaultParameter();
    setDataModel(prev => ({ ...prev, parameters: [...prev.parameters, p] }));
    return p;
  }, []);

  const updateParameter = useCallback((id: string, updates: Partial<ParameterDef>) => {
    setDataModel(prev => ({
      ...prev,
      parameters: prev.parameters.map(p => {
        if (p.id !== id) return p;
        const updated = { ...p, ...updates };
        try { updated.value = parseFloat(updated.expression) || 0; } catch { updated.value = 0; }
        return updated;
      }),
    }));
  }, []);

  const removeParameter = useCallback((id: string) => {
    setDataModel(prev => ({
      ...prev,
      parameters: prev.parameters.filter(p => p.id !== id),
    }));
  }, []);

  // === Sweep Parameters ===
  const addSweepParameter = useCallback(() => {
    const sp = createDefaultSweepParameter();
    setSweepParameters(prev => [...prev, sp]);
    return sp;
  }, []);

  const updateSweepParameter = useCallback((id: string, updates: Partial<SweepParameter>) => {
    setSweepParameters(prev => prev.map(s => {
      if (s.id !== id) return s;
      const updated = { ...s, ...updates };
      // Recompute values array from start/end/step
      if (updated.step > 0 && updated.end >= updated.start) {
        const vals: number[] = [];
        for (let v = updated.start; v <= updated.end + updated.step * 0.001; v += updated.step) {
          vals.push(parseFloat(v.toPrecision(10)));
        }
        updated.values = vals;
      }
      return updated;
    }));
  }, []);

  const removeSweepParameter = useCallback((id: string) => {
    setSweepParameters(prev => prev.filter(s => s.id !== id));
  }, []);

  const totalSweepRuns = sweepParameters.reduce(
    (total, p) => total * Math.max(1, p.values.length), 1
  );

  return {
    dataModel,
    setDataModel,
    simState,
    setSimState,
    sweepParameters,
    setSweepParameters,
    updateDM,
    // Molecules
    addMolecule, updateMolecule, removeMolecule,
    // Reactions
    addReaction, updateReaction, removeReaction,
    // Release Sites
    addReleaseSite, updateReleaseSite, removeReleaseSite,
    // Release Patterns
    addReleasePattern, updateReleasePattern, removeReleasePattern,
    // Surface Classes
    addSurfaceClass, updateSurfaceClass, removeSurfaceClass,
    // Geometry
    addGeometryObject, updateGeometryObject, removeGeometryObject,
    // Reaction Outputs
    addReactionOutput, updateReactionOutput, removeReactionOutput,
    // Mod Surface Regions
    addModSurfRegion, updateModSurfRegion, removeModSurfRegion,
    // Parameters
    addParameter, updateParameter, removeParameter,
    // Sweep Parameters
    addSweepParameter, updateSweepParameter, removeSweepParameter,
    totalSweepRuns,
  };
}

export type DataModelActions = ReturnType<typeof useDataModel>;
