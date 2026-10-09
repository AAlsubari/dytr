// ==============================================================================
// DyTrStudio Data Mining - Active Dataset Workspace Bar & Multi-Table Manager
// Displays active dataset, multi-table tab switcher, and dataset management actions
// Zero Emojis Policy - High Rigor Arabic & English Educational Guidance
// ==============================================================================

import React, { useState } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import { Database, Plus, GitMerge, Trash2, Copy, Edit2, Check, X, Layers } from 'lucide-react';

export default function ActiveDatasetBar({ onOpenMerge }) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const {
    loadedDatasets,
    activeDatasetId,
    datasetName,
    currentDataset,
    switchActiveDataset,
    removeDataset,
    renameDataset,
    duplicateDataset,
    setActivePhase,
  } = useDatasetStore();

  const [editingId, setEditingId] = useState(null);
  const [editingName, setEditingName] = useState('');

  const activeDs = (loadedDatasets || []).find((ds) => ds.id === activeDatasetId) || {
    id: activeDatasetId,
    name: datasetName,
    rowsCount: currentDataset.length,
    colsCount: Object.keys(currentDataset[0] || {}).length,
  };

  const handleStartRename = (ds, e) => {
    e.stopPropagation();
    setEditingId(ds.id);
    setEditingName(ds.name);
  };

  const handleSaveRename = (id, e) => {
    e.stopPropagation();
    if (editingName.trim()) {
      renameDataset(id, editingName.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const getSourceBadge = (type) => {
    switch (type) {
      case 'kaggle':
      case 'search':
        return { label_ar: 'كاجل / بحث', label_en: 'Kaggle / Search', color: 'badge-teal' };
      case 'upload':
        return { label_ar: 'ملف محلي', label_en: 'Local File', color: 'badge-blue' };
      case 'scrape':
        return { label_ar: 'ويب مباشر', label_en: 'Web Scrape', color: 'badge-purple' };
      case 'synthetic':
        return { label_ar: 'بيانات مصنعة', label_en: 'Synthetic', color: 'badge-amber' };
      case 'merged':
        return { label_ar: 'جدول مدمج', label_en: 'Merged Table', color: 'badge-teal' };
      case 'sample':
      default:
        return { label_ar: 'عينة معيارية', label_en: 'Sample Benchmark', color: 'badge-neutral' };
    }
  };

  return (
    <div
      style={{
        borderRadius: 'var(--radius-lg, 12px)',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)',
        padding: '0.85rem 1.15rem',
        marginBottom: '1.5rem',
        boxShadow: 'var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.06))',
      }}
    >
      {/* Top Meta Line: Active Dataset Info & Fast Action Buttons */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginBottom: '0.75rem',
          paddingBottom: '0.75rem',
          borderBottom: '1px dashed var(--border-subtle)',
        }}
      >
        {/* Active Indicator & Name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span
              style={{
                width: '9px',
                height: '9px',
                borderRadius: '50%',
                background: 'var(--accent-teal)',
                boxShadow: '0 0 8px var(--accent-teal)',
                display: 'inline-block',
              }}
            />
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              {isAr ? 'جدول البيانات النشط حالياً:' : 'Active Working Dataset:'}
            </span>
          </div>

          <strong
            style={{
              fontSize: '0.92rem',
              color: 'var(--accent-teal)',
              fontFamily: 'var(--font-mono)',
              background: 'rgba(20, 184, 166, 0.08)',
              padding: '0.2rem 0.65rem',
              borderRadius: 'var(--radius-sm, 4px)',
              border: '1px solid rgba(20, 184, 166, 0.25)',
            }}
          >
            {activeDs.name}
          </strong>

          <div style={{ display: 'flex', gap: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
            <span className="badge badge-teal">
              {currentDataset.length} {isAr ? 'صفاً' : 'rows'}
            </span>
            <span className="badge badge-neutral">
              {Object.keys(currentDataset[0] || {}).length} {isAr ? 'أعمدة' : 'cols'}
            </span>
            <span className={`badge ${getSourceBadge(activeDs.sourceType).color}`}>
              {isAr ? getSourceBadge(activeDs.sourceType).label_ar : getSourceBadge(activeDs.sourceType).label_en}
            </span>
          </div>
        </div>

        {/* Global Dataset Toolbar Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => setActivePhase(1)}
            className="btn btn-secondary btn-sm"
            style={{ fontSize: '0.76rem', padding: '0.25rem 0.6rem', gap: '0.35rem' }}
            title={isAr ? 'الانتقال لمرحلة جلب البيانات لإضافة قاعدة بيانات أخرى' : 'Add another dataset'}
          >
            <Plus size={13} />
            <span>{isAr ? 'إضافة قاعدة بيانات' : 'Add Dataset'}</span>
          </button>

          <button
            onClick={onOpenMerge}
            disabled={(loadedDatasets || []).length < 2}
            className="btn btn-secondary btn-sm"
            style={{
              fontSize: '0.76rem',
              padding: '0.25rem 0.6rem',
              gap: '0.35rem',
              color: (loadedDatasets || []).length >= 2 ? 'var(--accent-teal)' : 'var(--text-muted)',
              borderColor: (loadedDatasets || []).length >= 2 ? 'rgba(20, 184, 166, 0.35)' : 'var(--border-subtle)',
            }}
            title={
              (loadedDatasets || []).length < 2
                ? (isAr ? 'يلزم وجود جدولين على الأقل لتفعيل ميزة الدمج' : 'Need at least 2 datasets to merge')
                : (isAr ? 'دمج جدولين أو ربطهما بمفتاح مشترك' : 'Merge or join two datasets')
            }
          >
            <GitMerge size={13} />
            <span>{isAr ? 'دمج الجداول (Merge / Join)' : 'Merge Tables'}</span>
          </button>

          <button
            onClick={() => duplicateDataset(activeDatasetId)}
            className="btn btn-secondary btn-sm"
            style={{ fontSize: '0.76rem', padding: '0.25rem 0.55rem', gap: '0.3rem' }}
            title={isAr ? 'إنشاء نسخة مطابقة من هذا الجدول للتجربة عليها' : 'Duplicate current dataset'}
          >
            <Copy size={12} />
            <span>{isAr ? 'نسخ' : 'Clone'}</span>
          </button>
        </div>
      </div>

      {/* Bottom Tabs Row: Loaded Datasets in Workspace */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.45rem' }}>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            {isAr
              ? `قواعد البيانات المفتوحة في بيئة العمل (${(loadedDatasets || []).length}): انقر للتبديل الفوري بينها`
              : `Open Workspace Datasets (${(loadedDatasets || []).length}): Click any tab to switch`}
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {isAr ? 'كافة العمليات والفحص والتنظيف ستطبق على الجدول النشط' : 'All operations apply to active table'}
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            overflowX: 'auto',
            paddingBottom: '0.3rem',
          }}
        >
          {(loadedDatasets || []).map((ds) => {
            const isActive = ds.id === activeDatasetId;
            const badge = getSourceBadge(ds.sourceType);
            const isEditing = editingId === ds.id;

            return (
              <div
                key={ds.id}
                onClick={() => !isEditing && switchActiveDataset(ds.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.45rem 0.75rem',
                  borderRadius: 'var(--radius-md, 8px)',
                  background: isActive ? 'var(--bg-primary)' : 'rgba(255, 255, 255, 0.03)',
                  border: isActive ? '1.5px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  boxShadow: isActive ? '0 0 10px rgba(20, 184, 166, 0.15)' : 'none',
                  cursor: isActive ? 'default' : 'pointer',
                  transition: 'all 0.15s ease',
                  flexShrink: 0,
                  maxWidth: '260px',
                }}
                title={isAr ? `انقر لتفعيل جدول: ${ds.name}` : `Click to activate: ${ds.name}`}
              >
                {/* Source Icon */}
                <div
                  style={{
                    color: isActive ? 'var(--accent-teal)' : 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <Database size={14} />
                </div>

                {/* Name / Inline Rename Input */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  {isEditing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }} onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveRename(ds.id, e);
                          if (e.key === 'Escape') handleCancelRename(e);
                        }}
                        autoFocus
                        style={{
                          fontSize: '0.78rem',
                          padding: '0.15rem 0.35rem',
                          background: 'var(--bg-secondary)',
                          border: '1px solid var(--accent-teal)',
                          borderRadius: '4px',
                          color: 'var(--text-primary)',
                          width: '120px',
                        }}
                      />
                      <button
                        onClick={(e) => handleSaveRename(ds.id, e)}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-teal)', cursor: 'pointer', padding: 0 }}
                      >
                        <Check size={13} />
                      </button>
                      <button
                        onClick={handleCancelRename}
                        style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 0 }}
                      >
                        <X size={13} />
                      </button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span
                          style={{
                            fontSize: '0.8rem',
                            fontWeight: isActive ? 700 : 500,
                            color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            maxWidth: '130px',
                          }}
                          title={ds.name}
                        >
                          {ds.name}
                        </span>
                        {isActive && (
                          <span
                            style={{
                              fontSize: '0.65rem',
                              padding: '0.1rem 0.35rem',
                              borderRadius: '3px',
                              background: 'var(--accent-teal)',
                              color: '#042f2e',
                              fontWeight: 700,
                            }}
                          >
                            {isAr ? 'النشط' : 'Active'}
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {ds.rowsCount || (ds.data ? ds.data.length : 0)} {isAr ? 'صفاً' : 'rows'}
                      </span>
                    </div>
                  )}
                </div>

                {/* Edit & Close Actions on Tab */}
                {!isEditing && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                    <button
                      onClick={(e) => handleStartRename(ds, e)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '0.15rem',
                        borderRadius: '3px',
                        display: 'flex',
                        alignItems: 'center',
                        opacity: 0.65,
                      }}
                      title={isAr ? 'إعادة التسمية' : 'Rename'}
                    >
                      <Edit2 size={11} />
                    </button>

                    {(loadedDatasets || []).length > 1 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeDataset(ds.id);
                        }}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-muted)',
                          cursor: 'pointer',
                          padding: '0.15rem',
                          borderRadius: '3px',
                          display: 'flex',
                          alignItems: 'center',
                          opacity: 0.65,
                        }}
                        title={isAr ? 'إغلاق هذا الجدول من بيئة العمل' : 'Remove from workspace'}
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
