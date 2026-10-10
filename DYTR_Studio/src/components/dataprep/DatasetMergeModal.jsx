// ==============================================================================
// DyTrStudio Data Mining - Multi-Dataset Merge & Join Studio Modal
// Merges multiple tables via Row Concatenation (Union) or Key-Based Join
// Zero Emojis Policy - High Rigor Arabic & English Educational Guidance
// ==============================================================================

import React, { useState, useMemo } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import { GitMerge, X, Database, ArrowRight, Check, AlertCircle, Layers } from 'lucide-react';

export default function DatasetMergeModal({ isOpen, onClose }) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const { loadedDatasets, activeDatasetId, mergeDatasets } = useDatasetStore();

  const datasets = loadedDatasets || [];

  // Default selection: Table A is active, Table B is first other dataset
  const [sourceIdA, setSourceIdA] = useState(activeDatasetId);
  const [sourceIdB, setSourceIdB] = useState(() => {
    const other = datasets.find((d) => d.id !== activeDatasetId);
    return other ? other.id : '';
  });

  const [mergeType, setMergeType] = useState('concat_rows'); // 'concat_rows' | 'join_columns'
  const [joinKeyA, setJoinKeyA] = useState('');
  const [joinKeyB, setJoinKeyB] = useState('');
  const [customName, setCustomName] = useState('');
  const [errorMsg, setErrorMsg] = useState(null);

  const dsA = datasets.find((d) => d.id === sourceIdA) || datasets[0];
  const dsB = datasets.find((d) => d.id === sourceIdB);

  const colsA = useMemo(() => Object.keys(dsA?.data?.[0] || {}), [dsA]);
  const colsB = useMemo(() => Object.keys(dsB?.data?.[0] || {}), [dsB]);

  // Set default keys if available
  React.useEffect(() => {
    if (colsA.length > 0 && !joinKeyA) {
      const match = colsA.find((c) => ['id', 'vin', 'model', 'name', 'code'].includes(c.toLowerCase()));
      setJoinKeyA(match || colsA[0]);
    }
  }, [colsA, joinKeyA]);

  React.useEffect(() => {
    if (colsB.length > 0 && !joinKeyB) {
      const match = colsB.find((c) => ['id', 'vin', 'model', 'name', 'code'].includes(c.toLowerCase()));
      setJoinKeyB(match || colsB[0]);
    }
  }, [colsB, joinKeyB]);

  // Update default name when sources change
  const defaultMergeName = useMemo(() => {
    if (!dsA || !dsB) return 'جدول_مدمج.csv';
    const nameA = dsA.name.replace(/\.[^/.]+$/, '');
    const nameB = dsB.name.replace(/\.[^/.]+$/, '');
    return `دمج_${nameA}_مع_${nameB}.csv`;
  }, [dsA, dsB]);

  // Live simulation of dimensions
  const estimatedRows = useMemo(() => {
    if (!dsA || !dsB) return 0;
    if (mergeType === 'concat_rows') {
      return (dsA.data?.length || 0) + (dsB.data?.length || 0);
    } else {
      return dsA.data?.length || 0;
    }
  }, [dsA, dsB, mergeType]);

  const estimatedCols = useMemo(() => {
    if (!dsA || !dsB) return 0;
    if (mergeType === 'concat_rows') {
      const set = new Set([...colsA, ...colsB]);
      return set.size;
    } else {
      // join: colsA + (colsB without joinKey)
      return colsA.length + colsB.length - (colsB.includes(joinKeyB) ? 1 : 0);
    }
  }, [dsA, dsB, colsA, colsB, mergeType, joinKeyB]);

  const handleExecuteMerge = () => {
    if (!dsA || !dsB) {
      setErrorMsg(isAr ? 'يرجى اختيار جدولين صالحين للدمج' : 'Please select two valid datasets');
      return;
    }
    if (dsA.id === dsB.id) {
      setErrorMsg(isAr ? 'لا يمكن دمج الجدول مع نفسه، يرجى اختيار جدولين مختلفين' : 'Cannot merge a dataset with itself');
      return;
    }
    if (mergeType === 'join_columns' && (!joinKeyA || !joinKeyB)) {
      setErrorMsg(isAr ? 'يرجى تحديد مفتاح الربط لكلا الجدولين' : 'Please select join keys for both datasets');
      return;
    }

    setErrorMsg(null);
    const result = mergeDatasets({
      sourceIdA: dsA.id,
      sourceIdB: dsB.id,
      mergeType,
      joinKeyA,
      joinKeyB,
      customName: customName.trim() || defaultMergeName,
    });

    if (result && result.success) {
      onClose();
    } else {
      setErrorMsg(result?.error || (isAr ? 'فشلت عملية الدمج' : 'Merge operation failed'));
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.72)',
        backdropFilter: 'blur(5px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          borderRadius: 'var(--radius-lg, 12px)',
          border: '1px solid var(--border-subtle)',
          boxShadow: 'var(--shadow-xl, 0 20px 25px -5px rgba(0,0,0,0.4))',
          width: '100%',
          maxWidth: '680px',
          maxHeight: '90vh',
          overflowY: 'auto',
          padding: '1.75rem',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                background: 'rgba(20, 184, 166, 0.12)',
                border: '1px solid rgba(20, 184, 166, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-teal)',
              }}
            >
              <GitMerge size={20} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                {isAr ? 'دمج وتكامل قواعد البيانات (Multi-Dataset Merge)' : 'Multi-Dataset Merge & Join Studio'}
              </h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0' }}>
                {isAr
                  ? 'دمج جدولين مفتوحين في بيئة العمل بأسلوب جمع الصفوف أو ربط الأعمدة بمفتاح مشترك'
                  : 'Combine two workspace tables via row concatenation (union) or key-based column join.'}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.35rem',
              borderRadius: '6px',
            }}
          >
            <X size={18} />
          </button>
        </div>

        {errorMsg && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-md, 8px)',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: 'var(--accent-red, #ef4444)',
              fontSize: '0.84rem',
              marginBottom: '1.25rem',
            }}
          >
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Step 1: Select Tables */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
          {/* Table A */}
          <div
            style={{
              padding: '1rem',
              borderRadius: 'var(--radius-md, 8px)',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.45rem' }}>
              {isAr ? 'الجدول الأساسي (الجدول أ):' : 'Primary Table (Table A):'}
            </label>
            <select
              value={sourceIdA}
              onChange={(e) => setSourceIdA(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)',
                fontSize: '0.82rem',
              }}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.rowsCount || d.data?.length || 0} {isAr ? 'صف' : 'rows'})
                </option>
              ))}
            </select>
            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {colsA.length} {isAr ? 'أعمدة متاحة' : 'columns available'}
            </div>
          </div>

          {/* Table B */}
          <div
            style={{
              padding: '1rem',
              borderRadius: 'var(--radius-md, 8px)',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.45rem' }}>
              {isAr ? 'الجدول الثانوي (الجدول ب):' : 'Secondary Table (Table B):'}
            </label>
            <select
              value={sourceIdB}
              onChange={(e) => setSourceIdB(e.target.value)}
              style={{
                width: '100%',
                padding: '0.5rem',
                borderRadius: '6px',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-primary)',
                fontSize: '0.82rem',
              }}
            >
              <option value="" disabled>
                {isAr ? '-- اختر الجدول الثاني --' : '-- Select secondary table --'}
              </option>
              {datasets
                .filter((d) => d.id !== sourceIdA)
                .map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.rowsCount || d.data?.length || 0} {isAr ? 'صف' : 'rows'})
                  </option>
                ))}
            </select>
            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {colsB.length} {isAr ? 'أعمدة متاحة' : 'columns available'}
            </div>
          </div>
        </div>

        {/* Step 2: Merge Method Selection */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            {isAr ? 'استراتيجية وطريقة الدمج:' : 'Merge Strategy:'}
          </label>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            {/* Mode 1: Concat Rows */}
            <div
              onClick={() => setMergeType('concat_rows')}
              style={{
                padding: '0.85rem',
                borderRadius: 'var(--radius-md, 8px)',
                background: mergeType === 'concat_rows' ? 'rgba(20, 184, 166, 0.08)' : 'var(--bg-secondary)',
                border: mergeType === 'concat_rows' ? '1.5px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <strong style={{ fontSize: '0.85rem', color: mergeType === 'concat_rows' ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                  {isAr ? '1. دمج الصفوف (Append / Union)' : '1. Row Append / Union'}
                </strong>
                {mergeType === 'concat_rows' && <Check size={14} color="var(--accent-teal)" />}
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
                {isAr
                  ? 'جمع صفوف الجدول الأول مع الجدول الثاني وتوحيد الأعمدة المتشابهة (مناسب لتجميع مبيعات سنوية أو محادثات متعددة).'
                  : 'Stacks rows of Table A and Table B vertically, unifying common columns.'}
              </p>
            </div>

            {/* Mode 2: Key-based Join */}
            <div
              onClick={() => setMergeType('join_columns')}
              style={{
                padding: '0.85rem',
                borderRadius: 'var(--radius-md, 8px)',
                background: mergeType === 'join_columns' ? 'rgba(20, 184, 166, 0.08)' : 'var(--bg-secondary)',
                border: mergeType === 'join_columns' ? '1.5px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                <strong style={{ fontSize: '0.85rem', color: mergeType === 'join_columns' ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                  {isAr ? '2. ربط الأعمدة بمفتاح (Key Join)' : '2. Key-based Column Join'}
                </strong>
                {mergeType === 'join_columns' && <Check size={14} color="var(--accent-teal)" />}
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
                {isAr
                  ? 'إثراء كل صف في الجدول أ بخصائص الجدول ب اعتماداً على عمود ربط متطابق (مثل ربط أسعار السيارات بمواصفات المحرك).'
                  : 'Enriches rows of Table A with matching columns from Table B using a common key.'}
              </p>
            </div>
          </div>
        </div>

        {/* Step 3: Keys Configuration (Only if Join) */}
        {mergeType === 'join_columns' && (
          <div
            style={{
              padding: '1rem',
              borderRadius: 'var(--radius-md, 8px)',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              marginBottom: '1.25rem',
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  {isAr ? 'مفتاح الربط في الجدول أ:' : 'Join Key in Table A:'}
                </label>
                <select
                  value={joinKeyA}
                  onChange={(e) => setJoinKeyA(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.45rem',
                    borderRadius: '6px',
                    background: 'var(--bg-primary)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.8rem',
                  }}
                >
                  {colsA.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  {isAr ? 'مفتاح الربط في الجدول ب:' : 'Join Key in Table B:'}
                </label>
                <select
                  value={joinKeyB}
                  onChange={(e) => setJoinKeyB(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.45rem',
                    borderRadius: '6px',
                    background: 'var(--bg-primary)',
                    border: '1px solid var(--border-subtle)',
                    color: 'var(--text-primary)',
                    fontSize: '0.8rem',
                  }}
                >
                  {colsB.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Output Name & Simulation Preview */}
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.45rem' }}>
            {isAr ? 'اسم الجدول المدمج الناتج:' : 'Resulting Merged Dataset Name:'}
          </label>
          <input
            type="text"
            value={customName}
            placeholder={defaultMergeName}
            onChange={(e) => setCustomName(e.target.value)}
            style={{
              width: '100%',
              padding: '0.55rem 0.85rem',
              borderRadius: '6px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.85rem',
              marginBottom: '0.85rem',
            }}
          />

          {/* Result Dimension Preview Badge */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.65rem 1rem',
              borderRadius: 'var(--radius-md, 8px)',
              background: 'rgba(20, 184, 166, 0.05)',
              border: '1px solid rgba(20, 184, 166, 0.2)',
              fontSize: '0.82rem',
            }}
          >
            <span style={{ color: 'var(--text-muted)' }}>
              {isAr ? 'أبعاد الجدول المدمج المتوقعة:' : 'Estimated Result Dimensions:'}
            </span>
            <div style={{ display: 'flex', gap: '0.45rem', fontFamily: 'var(--font-mono)' }}>
              <span className="badge badge-teal">
                {estimatedRows} {isAr ? 'صفاً متوقعاً' : 'estimated rows'}
              </span>
              <span className="badge badge-neutral">
                {estimatedCols} {isAr ? 'أعمدة' : 'columns'}
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
          <button onClick={onClose} className="btn btn-secondary btn-md">
            {isAr ? 'إلغاء' : 'Cancel'}
          </button>

          <button
            onClick={handleExecuteMerge}
            disabled={!dsA || !dsB || dsA.id === dsB.id}
            className="btn btn-primary btn-md"
            style={{ gap: '0.45rem' }}
          >
            <GitMerge size={15} />
            <span>{isAr ? 'تنفيذ الدمج وإضافته كجدول نشط' : 'Execute Merge & Activate Table'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
