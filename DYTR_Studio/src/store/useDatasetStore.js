// ==============================================================================
// DYTR Studio Data Mining & Processing Store (Zustand)
// Multi-Dataset Workspace Registry, CRISP-DM 5-Phase State, Undo/Redo & Merging
// Zero Emojis Policy - High Rigor Educational & Continual Learning Architecture
// ==============================================================================

import { create } from 'zustand';
import { generateArabicSentiment, sampleDatasetsCatalog, getSampleDatasetById } from '../lib/data/synthetic';
import { profileDataset } from '../lib/data/processor';
import { useStudioStore } from './studioStore';

const STORAGE_KEY = 'dytr_saved_datasets_library';

function loadSavedLibrary() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.warn('Could not read saved datasets from localStorage', e);
  }
  return [];
}

function persistLibrary(library) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(library));
  } catch (e) {
    console.warn('Could not persist dataset library', e);
  }
}

// Initial dataset loaded by default so the user immediately sees working data
const initialData = generateArabicSentiment(50);
const initialDatasetId = 'ds_default_sample';
const initialDatasetEntry = {
  id: initialDatasetId,
  name: 'عينة_مشاعر_عربية.csv',
  data: initialData,
  sourceType: 'sample',
  sourceMetadata: { origin: 'DyTr Native Generator' },
  rowsCount: initialData.length,
  colsCount: Object.keys(initialData[0] || {}).length,
  loadedAt: new Date().toLocaleTimeString(),
  updatedAt: new Date().toLocaleTimeString(),
};

export const useDatasetStore = create((set, get) => ({
  // Multi-Dataset Project Workspace Registry
  loadedDatasets: [initialDatasetEntry],
  activeDatasetId: initialDatasetId,

  // Current Active Working Dataset (mirrored for instant access across all tabs)
  currentDataset: initialData,
  datasetName: 'عينة_مشاعر_عربية.csv',
  sourceType: 'sample', // 'upload' | 'scrape' | 'search' | 'synthetic' | 'sample' | 'merged'
  sourceMetadata: { origin: 'DyTr Native Generator' },

  // CRISP-DM Phases
  activePhase: 1, // 1: Acquisition, 2: Inspection, 3: Preprocessing, 4: Transformation, 5: Export
  setActivePhase: (phase) => set({ activePhase: phase }),

  // Provenance & Undo / Redo Stacks (Max 20 steps)
  pastHistory: [],
  futureHistory: [],
  operationsLog: [
    {
      id: 'op_init',
      timestamp: new Date().toLocaleTimeString(),
      type: 'load',
      title_ar: 'تحميل مجموعة بيانات أولية لتحليل المشاعر',
      title_en: 'Initial Arabic sentiment dataset loaded',
      rowsCount: initialData.length,
      colsCount: Object.keys(initialData[0] || {}).length,
    },
  ],

  // Saved Datasets Library (Persistent in localStorage)
  savedLibrary: loadSavedLibrary(),

  // Add & Set New Dataset in the Project Workspace (preserves existing datasets)
  setDataset: (newData, name = 'dataset.csv', sourceType = 'upload', meta = {}) => {
    if (!newData || !Array.isArray(newData) || newData.length === 0) return;
    const { loadedDatasets, activeDatasetId, currentDataset, pastHistory, operationsLog } = get();
    const cols = Object.keys(newData[0] || {}).length;

    // Generate unique ID
    const newId = `ds_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    // Avoid duplicate names if user imports same file again
    let finalName = name;
    const existingWithSameName = (loadedDatasets || []).find((ds) => ds.name === finalName && ds.id !== activeDatasetId);
    if (existingWithSameName && existingWithSameName.data !== newData) {
      const extMatch = finalName.match(/\.[^/.]+$/);
      const ext = extMatch ? extMatch[0] : '';
      const base = finalName.replace(ext, '');
      finalName = `${base}_${(loadedDatasets || []).length + 1}${ext}`;
    }

    const newEntry = {
      id: newId,
      name: finalName,
      data: newData,
      sourceType,
      sourceMetadata: meta,
      rowsCount: newData.length,
      colsCount: cols,
      loadedAt: new Date().toLocaleTimeString(),
      updatedAt: new Date().toLocaleTimeString(),
    };

    // Keep all previous datasets and append the new one
    const updatedLoaded = [...(loadedDatasets || []), newEntry];

    set({
      loadedDatasets: updatedLoaded,
      activeDatasetId: newId,
      currentDataset: newData,
      datasetName: finalName,
      sourceType,
      sourceMetadata: meta,
      pastHistory: [...pastHistory.slice(-15), { data: currentDataset, name: get().datasetName }],
      futureHistory: [],
      operationsLog: [
        ...operationsLog,
        {
          id: `op_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'load',
          title_ar: `إضافة جدول بيانات جديد: ${finalName}`,
          title_en: `Imported dataset: ${finalName}`,
          rowsCount: newData.length,
          colsCount: cols,
        },
      ],
      activePhase: 2, // Auto-advance to Inspection/EDA
    });

    try {
      useStudioStore.getState()?.syncDatasetToModel(newData, finalName);
    } catch (e) {}
  },

  // Switch Active Dataset in Workspace
  switchActiveDataset: (id) => {
    const { loadedDatasets, activeDatasetId, currentDataset, datasetName, sourceType, sourceMetadata, operationsLog } = get();
    if (id === activeDatasetId) return;

    // Sync active dataset data back into list first
    const synced = (loadedDatasets || []).map((ds) =>
      ds.id === activeDatasetId
        ? {
            ...ds,
            data: currentDataset,
            name: datasetName,
            sourceType,
            sourceMetadata,
            rowsCount: currentDataset.length,
            colsCount: Object.keys(currentDataset[0] || {}).length,
            updatedAt: new Date().toLocaleTimeString(),
          }
        : ds
    );

    const target = synced.find((ds) => ds.id === id);
    if (!target) return;

    const cols = Object.keys(target.data[0] || {}).length;

    set({
      loadedDatasets: synced,
      activeDatasetId: id,
      currentDataset: target.data,
      datasetName: target.name,
      sourceType: target.sourceType || 'sample',
      sourceMetadata: target.sourceMetadata || {},
      pastHistory: [],
      futureHistory: [],
      operationsLog: [
        ...operationsLog,
        {
          id: `op_switch_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'switch',
          title_ar: `التبديل إلى جدول البيانات النشط: ${target.name}`,
          title_en: `Switched active dataset to: ${target.name}`,
          rowsCount: target.data.length,
          colsCount: cols,
        },
      ],
    });

    try {
      useStudioStore.getState()?.syncDatasetToModel(target.data, target.name);
    } catch (e) {}
  },

  // Remove a Dataset from the Workspace
  removeDataset: (id) => {
    const { loadedDatasets, activeDatasetId } = get();
    const remaining = (loadedDatasets || []).filter((ds) => ds.id !== id);

    if (remaining.length === 0) {
      const sample = generateArabicSentiment(50);
      const sampleId = 'ds_default_' + Date.now();
      const sampleEntry = {
        id: sampleId,
        name: 'عينة_مشاعر_عربية.csv',
        data: sample,
        sourceType: 'sample',
        sourceMetadata: { origin: 'DyTr Native Generator' },
        rowsCount: sample.length,
        colsCount: Object.keys(sample[0] || {}).length,
        loadedAt: new Date().toLocaleTimeString(),
        updatedAt: new Date().toLocaleTimeString(),
      };
      set({
        loadedDatasets: [sampleEntry],
        activeDatasetId: sampleId,
        currentDataset: sample,
        datasetName: sampleEntry.name,
        sourceType: 'sample',
        sourceMetadata: sampleEntry.sourceMetadata,
        pastHistory: [],
        futureHistory: [],
      });
      return;
    }

    if (activeDatasetId === id) {
      const nextActive = remaining[0];
      set({
        loadedDatasets: remaining,
        activeDatasetId: nextActive.id,
        currentDataset: nextActive.data,
        datasetName: nextActive.name,
        sourceType: nextActive.sourceType || 'sample',
        sourceMetadata: nextActive.sourceMetadata || {},
        pastHistory: [],
        futureHistory: [],
      });
    } else {
      set({ loadedDatasets: remaining });
    }
  },

  // Rename a Dataset
  renameDataset: (id, newName) => {
    if (!newName || !newName.trim()) return;
    const { loadedDatasets, activeDatasetId } = get();
    const updated = (loadedDatasets || []).map((ds) =>
      ds.id === id ? { ...ds, name: newName.trim(), updatedAt: new Date().toLocaleTimeString() } : ds
    );
    set({
      loadedDatasets: updated,
      datasetName: id === activeDatasetId ? newName.trim() : get().datasetName,
    });
  },

  // Duplicate a Dataset in Workspace
  duplicateDataset: (id) => {
    const { loadedDatasets } = get();
    const target = (loadedDatasets || []).find((ds) => ds.id === id);
    if (!target) return;

    const extMatch = target.name.match(/\.[^/.]+$/);
    const ext = extMatch ? extMatch[0] : '';
    const base = target.name.replace(ext, '');
    const newName = `${base}_نسخة${ext}`;
    const newId = `ds_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    const cloned = {
      ...target,
      id: newId,
      name: newName,
      data: JSON.parse(JSON.stringify(target.data)),
      loadedAt: new Date().toLocaleTimeString(),
      updatedAt: new Date().toLocaleTimeString(),
    };

    set({
      loadedDatasets: [...loadedDatasets, cloned],
      activeDatasetId: newId,
      currentDataset: cloned.data,
      datasetName: newName,
    });
  },

  // Merge / Join Multiple Datasets into a New Combined Dataset
  mergeDatasets: ({ sourceIdA, sourceIdB, mergeType = 'concat_rows', joinKeyA = '', joinKeyB = '', customName = '' }) => {
    const { loadedDatasets } = get();
    const dsA = (loadedDatasets || []).find((ds) => ds.id === sourceIdA);
    const dsB = (loadedDatasets || []).find((ds) => ds.id === sourceIdB);
    if (!dsA || !dsB) return { success: false, error: 'One or both datasets not found' };

    let mergedData = [];
    const nameA = dsA.name.replace(/\.[^/.]+$/, '');
    const nameB = dsB.name.replace(/\.[^/.]+$/, '');
    const defaultName = customName.trim() || `دمج_${nameA}_مع_${nameB}.csv`;

    if (mergeType === 'concat_rows') {
      // 1. Row Concatenation (Union of rows)
      const colsA = Object.keys(dsA.data[0] || {});
      const colsB = Object.keys(dsB.data[0] || {});
      const allCols = Array.from(new Set([...colsA, ...colsB]));

      mergedData = [
        ...dsA.data.map((r) => {
          const row = {};
          allCols.forEach((c) => { row[c] = r[c] !== undefined ? r[c] : null; });
          return row;
        }),
        ...dsB.data.map((r) => {
          const row = {};
          allCols.forEach((c) => { row[c] = r[c] !== undefined ? r[c] : null; });
          return row;
        }),
      ];
    } else if (mergeType === 'join_columns') {
      // 2. Key-based Join (Left/Inner join on matching key)
      if (!joinKeyA || !joinKeyB) {
        return { success: false, error: 'Join keys must be specified' };
      }

      const mapB = new Map();
      dsB.data.forEach((row) => {
        const val = String(row[joinKeyB] ?? '').trim().toLowerCase();
        if (val) mapB.set(val, row);
      });

      mergedData = dsA.data.map((rowA) => {
        const keyVal = String(rowA[joinKeyA] ?? '').trim().toLowerCase();
        const rowB = mapB.get(keyVal) || {};
        const combined = { ...rowA };
        Object.entries(rowB).forEach(([kB, vB]) => {
          if (kB === joinKeyB) return;
          const finalKey = combined[kB] !== undefined ? `${nameB}_${kB}` : kB;
          combined[finalKey] = vB;
        });
        return combined;
      });
    }

    if (mergedData.length === 0) {
      return { success: false, error: 'Merge resulted in 0 rows' };
    }

    const newId = `ds_merged_${Date.now()}`;
    const newEntry = {
      id: newId,
      name: defaultName,
      data: mergedData,
      sourceType: 'merged',
      sourceMetadata: {
        mergedFrom: [dsA.name, dsB.name],
        mergeType,
        joinKeys: mergeType === 'join_columns' ? [joinKeyA, joinKeyB] : null,
      },
      rowsCount: mergedData.length,
      colsCount: Object.keys(mergedData[0] || {}).length,
      loadedAt: new Date().toLocaleTimeString(),
      updatedAt: new Date().toLocaleTimeString(),
    };

    set({
      loadedDatasets: [...loadedDatasets, newEntry],
      activeDatasetId: newId,
      currentDataset: mergedData,
      datasetName: defaultName,
      sourceType: 'merged',
      sourceMetadata: newEntry.sourceMetadata,
      activePhase: 2, // Auto-advance to Inspection to explore the merged data
      operationsLog: [
        ...get().operationsLog,
        {
          id: `op_merge_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'merge',
          title_ar: `دمج جدولين: ${dsA.name} مع ${dsB.name} (${mergeType === 'concat_rows' ? 'دمج صفوف' : 'دمج أعمدة بالمفتاح'})`,
          title_en: `Merged datasets: ${dsA.name} with ${dsB.name}`,
          rowsCount: mergedData.length,
          colsCount: Object.keys(mergedData[0] || {}).length,
        },
      ],
    });

    try {
      useStudioStore.getState()?.syncDatasetToModel(mergedData, defaultName);
    } catch (e) {}

    return { success: true, newId, name: defaultName, rowsCount: mergedData.length };
  },

  // Apply a Transformation with Immutable History Snapshot
  applyOperation: (op) => {
    const { currentDataset, pastHistory, operationsLog, datasetName, loadedDatasets, activeDatasetId } = get();
    if (!op || !op.transformFn) return;

    try {
      const nextData = op.transformFn(currentDataset);
      const cols = Object.keys(nextData[0] || {}).length;

      const updatedLoaded = (loadedDatasets || []).map((ds) =>
        ds.id === activeDatasetId
          ? {
              ...ds,
              data: nextData,
              rowsCount: nextData.length,
              colsCount: cols,
              updatedAt: new Date().toLocaleTimeString(),
            }
          : ds
      );

      set({
        currentDataset: nextData,
        loadedDatasets: updatedLoaded,
        pastHistory: [...pastHistory.slice(-15), { data: currentDataset, name: datasetName }],
        futureHistory: [],
        operationsLog: [
          ...operationsLog,
          {
            id: `op_${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            type: op.type || 'transform',
            title_ar: op.title_ar || 'تطبيق معالجة على البيانات',
            title_en: op.title_en || 'Applied transformation',
            ...op,
            rowsCount: nextData.length,
            colsCount: cols,
          },
        ],
      });

      try {
        useStudioStore.getState()?.syncDatasetToModel(nextData, datasetName);
      } catch (e) {}
    } catch (err) {
      console.error('Failed to apply dataset operation', err);
    }
  },

  // Undo Last Action
  undo: () => {
    const { pastHistory, currentDataset, futureHistory, datasetName, operationsLog, loadedDatasets, activeDatasetId } = get();
    if (pastHistory.length === 0) return;

    const previous = pastHistory[pastHistory.length - 1];
    const newPast = pastHistory.slice(0, pastHistory.length - 1);
    const cols = Object.keys(previous.data[0] || {}).length;

    const updatedLoaded = (loadedDatasets || []).map((ds) =>
      ds.id === activeDatasetId
        ? { ...ds, data: previous.data, name: previous.name, rowsCount: previous.data.length, colsCount: cols }
        : ds
    );

    set({
      currentDataset: previous.data,
      datasetName: previous.name,
      loadedDatasets: updatedLoaded,
      pastHistory: newPast,
      futureHistory: [{ data: currentDataset, name: datasetName }, ...futureHistory],
      operationsLog: [
        ...operationsLog,
        {
          id: `op_undo_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'undo',
          title_ar: 'تراجع عن آخر عملية (Undo)',
          title_en: 'Undo last transformation',
          rowsCount: previous.data.length,
          colsCount: cols,
        },
      ],
    });

    try {
      useStudioStore.getState()?.syncDatasetToModel(previous.data, previous.name);
    } catch (e) {}
  },

  // Redo Action
  redo: () => {
    const { futureHistory, currentDataset, pastHistory, datasetName, operationsLog, loadedDatasets, activeDatasetId } = get();
    if (futureHistory.length === 0) return;

    const next = futureHistory[0];
    const newFuture = futureHistory.slice(1);
    const cols = Object.keys(next.data[0] || {}).length;

    const updatedLoaded = (loadedDatasets || []).map((ds) =>
      ds.id === activeDatasetId
        ? { ...ds, data: next.data, name: next.name, rowsCount: next.data.length, colsCount: cols }
        : ds
    );

    set({
      currentDataset: next.data,
      datasetName: next.name,
      loadedDatasets: updatedLoaded,
      pastHistory: [...pastHistory, { data: currentDataset, name: datasetName }],
      futureHistory: newFuture,
      operationsLog: [
        ...operationsLog,
        {
          id: `op_redo_${Date.now()}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'redo',
          title_ar: 'إعادة تطبيق العملية (Redo)',
          title_en: 'Redo transformation',
          rowsCount: next.data.length,
          colsCount: cols,
        },
      ],
    });

    try {
      useStudioStore.getState()?.syncDatasetToModel(next.data, next.name);
    } catch (e) {}
  },

  // Save Current Dataset to Studio Library
  saveToLibrary: (meta = {}) => {
    const { currentDataset, datasetName, savedLibrary } = get();
    const profile = profileDataset(currentDataset);

    const newEntry = {
      id: `ds_${Date.now()}`,
      name: meta.name || datasetName,
      description: meta.description || 'مجموعة بيانات معالجة وجاهزة للتدريب في DYTR Studio',
      savedAt: new Date().toISOString(),
      rows: currentDataset.length,
      cols: profile.totalCols,
      qualityScore: profile.qualityScore,
      data: currentDataset,
      tags: meta.tags || ['dytr', 'clean'],
    };

    const updated = [newEntry, ...savedLibrary];
    set({ savedLibrary: updated });
    persistLibrary(updated);
    return newEntry;
  },

  // Load Dataset from Library
  loadFromLibrary: (id) => {
    const { savedLibrary, setDataset } = get();
    const item = savedLibrary.find((x) => x.id === id);
    if (item && item.data) {
      setDataset(item.data, item.name, 'library', { id: item.id });
    }
  },

  // Delete Dataset from Library
  deleteFromLibrary: (id) => {
    const { savedLibrary } = get();
    const updated = savedLibrary.filter((x) => x.id !== id);
    set({ savedLibrary: updated });
    persistLibrary(updated);
  },

  // Reset to sample dataset
  resetToSample: (sampleId = 'arabic_sentiment') => {
    const sample = getSampleDatasetById(sampleId);
    const catalogItem = sampleDatasetsCatalog.find((x) => x.id === sampleId);
    get().setDataset(sample, catalogItem ? `${catalogItem.name_en}.csv` : 'sample.csv', 'sample');
  },
}));
