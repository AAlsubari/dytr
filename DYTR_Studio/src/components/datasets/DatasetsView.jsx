// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Screen 2: Data Engine Studio (استوديو البيانات)
// Industrial Swiss Design, Zero Emojis, Drag & Drop, Subword Tokenizer Inspector

import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../utils/api';

export default function DatasetsView() {
  const { lang } = useApp();
  const isAr = lang === 'ar';

  const [datasets, setDatasets] = useState([]);
  const [activeDataset, setActiveDataset] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const [customFile, setCustomFile] = useState(null);

  // Subword Tokenization Inspector State
  const [sampleText, setSampleText] = useState(
    isAr
      ? 'نموذج DyTR للتعلم المستمر يحافظ على المعارف السابقة بكفاءة عالية'
      : 'DyTR continual learning framework eliminates catastrophic forgetting efficiently.'
  );
  const [tokenResult, setTokenResult] = useState(null);
  const [isTokenizing, setIsTokenizing] = useState(false);

  useEffect(() => {
    api.getPrebuiltDatasets().then((res) => {
      if (res && res.length > 0) {
        setDatasets(res);
        setActiveDataset(res[0]);
      }
    });
  }, []);

  // Run tokenization inspection when sampleText changes
  useEffect(() => {
    let active = true;
    setIsTokenizing(true);
    api.postTokenizeAlign({
      task_name: activeDataset?.task_name || 'task_sentiment',
      text: sampleText,
      max_seq_len: 32,
    }).then((res) => {
      if (active && res) {
        setTokenResult(res);
        setIsTokenizing(false);
      }
    }).catch(() => {
      if (active) setIsTokenizing(false);
    });

    return () => { active = false; };
  }, [sampleText, activeDataset]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomFile({
        name: file.name,
        size: `${(file.size / 1024).toFixed(1)} KB`,
        type: file.type || 'text/csv',
      });
    }
  };

  const filteredRows = (activeDataset?.sample_rows || []).filter((row) => {
    const matchesSearch = row.text.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = categoryFilter === 'ALL' || String(row.label) === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  return (
    <section>
      {/* Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
          {isAr ? 'استوديو البيانات والتجهيز العصبي' : 'Data Engine Studio & Ingestion Pipeline'}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {isAr
            ? 'فحص وتحضير البيانات المرجعية، استعراض التقطيع المجهري (Subword Tokenization)، وتطبيق أقنعة الخسارة (-100).'
            : 'Inspect benchmark datasets, verify subword token alignment, and analyze loss masking schemas.'}
        </p>
      </div>

      {/* Top Grid: Prebuilt Benchmarks + Custom Upload Dropzone */}
      <div className="grid-2" style={{ marginBottom: '1.5rem', alignItems: 'stretch' }}>
        {/* Prebuilt Datasets */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: '700', marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
            {isAr ? 'حزم البيانات المرجعية المدمجة' : 'Benchmark Datasets Catalog'}
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
            {datasets.map((d) => {
              const isSelected = activeDataset?.id === d.id;
              return (
                <div
                  key={d.id}
                  onClick={() => setActiveDataset(d)}
                  className="glass-card"
                  style={{
                    cursor: 'pointer',
                    borderColor: isSelected ? 'var(--accent-steel)' : 'var(--border-subtle)',
                    background: isSelected ? 'var(--accent-steel-bg)' : 'var(--bg-card)',
                    padding: '0.9rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span className="badge badge-steel" style={{ fontSize: '0.72rem' }}>{d.task_type}</span>
                    <span className="mono-font" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {d.num_samples.toLocaleString()}
                    </span>
                  </div>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--text-primary)' }}>{d.name}</h4>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px', lineClamp: 2 }}>
                    {d.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Drag & Drop File Upload Zone */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: '700', marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
            {isAr ? 'استيراد بيانات مخصصة (CSV / JSON)' : 'Custom Dataset Ingestion (CSV / JSON)'}
          </h3>
          <label
            style={{
              flex: 1,
              border: '2px dashed var(--border-strong)',
              borderRadius: '10px',
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              background: 'var(--bg-tertiary)',
              textAlign: 'center',
              transition: 'all 0.2s ease',
            }}
          >
            <input type="file" accept=".csv,.json" onChange={handleFileUpload} style={{ display: 'none' }} />
            <span style={{ fontWeight: '600', fontSize: '0.92rem', color: 'var(--text-primary)', marginBottom: '4px' }}>
              {customFile ? customFile.name : (isAr ? 'انقر أو اسحب ملف البيانات هنا' : 'Click or Drag CSV/JSON dataset here')}
            </span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {customFile ? `${customFile.size} · ${customFile.type}` : (isAr ? 'يدعم أعمدة (text, label, task_name)' : 'Supports columns: text, label, task_name')}
            </span>
          </label>
        </div>
      </div>

      {/* Subword Tokenizer & Loss Masking (-100) Inspector */}
      <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)' }}>
              {isAr ? 'مختبر تقطيع الكلمات المجهري ومطابقة التدرج (Subword Tokenization & Alignment)' : 'Microscopic Subword Token Alignment & Loss Masking (-100)'}
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isAr
                ? 'تقطيع النص إلى رقاقات (Subwords) وتحديد رقاقات القناع (-100) لمنع تشويه تدرجات الكلمات المتفرعة:'
                : 'Fragment words into sub-tokens and assign -100 to subsequent sub-tokens to suppress redundant loss gradients:'}
            </p>
          </div>
          <span className="badge badge-amber" style={{ fontSize: '0.75rem' }}>
            {isTokenizing ? (isAr ? 'جاري المعالجة...' : 'Processing...') : `Tokens: ${tokenResult?.tokens?.length || 0}`}
          </span>
        </div>

        <input
          type="text"
          value={sampleText}
          onChange={(e) => setSampleText(e.target.value)}
          placeholder={isAr ? 'أدخل نصاً لاختبار التقطيع العصبوني المباشر...' : 'Type text to test live subword token alignment...'}
          style={{ marginBottom: '0.85rem' }}
        />

        {tokenResult && (
          <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-medium)' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.75rem' }}>
              {tokenResult.tokens.map((token, idx) => {
                const label = tokenResult.aligned_labels?.[idx];
                const isMasked = label === -100;
                return (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      background: isMasked ? 'var(--accent-terracotta-bg)' : 'var(--accent-steel-bg)',
                      border: `1px solid ${isMasked ? 'var(--accent-terracotta)' : 'var(--accent-steel)'}`,
                      borderRadius: '6px',
                      padding: '0.35rem 0.6rem',
                    }}
                  >
                    <span className="mono-font" style={{ fontSize: '0.85rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                      {token}
                    </span>
                    <span className="mono-font" style={{ fontSize: '0.7rem', color: isMasked ? 'var(--accent-terracotta)' : 'var(--text-muted)', marginTop: '2px' }}>
                      {isMasked ? 'mask: -100' : `id: ${tokenResult.token_ids?.[idx] || idx}`}
                    </span>
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '1.25rem' }}>
              <span>• <strong style={{ color: 'var(--accent-steel)' }}>{isAr ? 'أزرق فولاذي:' : 'Steel Blue:'}</strong> {isAr ? 'جذر الكلمة المحسوب' : 'Root Active Token'}</span>
              <span>• <strong style={{ color: 'var(--accent-terracotta)' }}>{isAr ? 'أحمر طوبي:' : 'Terracotta:'}</strong> {isAr ? 'قناع الخسارة (-100 Loss Mask)' : '-100 Ignored Gradient'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Dataset Records Table with Search & Category Filter */}
      {activeDataset && (
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
              {isAr ? `معاينة سجلات: ${activeDataset.name}` : `Dataset Records: ${activeDataset.name}`}
            </h3>
            
            <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={isAr ? 'بحث في النصوص...' : 'Search records...'}
                style={{ width: '180px', padding: '0.45rem 0.75rem', fontSize: '0.85rem' }}
              />
              <span className="badge badge-steel">{filteredRows.length} {isAr ? 'عنصر' : 'rows'}</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', maxHeight: '420px', overflowY: 'auto' }}>
            {filteredRows.map((row, idx) => (
              <div
                key={idx}
                style={{
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '1rem',
                }}
              >
                <div style={{ flex: 1 }}>
                  <p style={{ fontSize: '0.88rem', color: 'var(--text-primary)', lineHeight: '1.5' }}>"{row.text}"</p>
                  {row.annotated_preview && (
                    <p style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', marginTop: '4px' }}>
                      {isAr ? 'الكيانات المستخرجة:' : 'Extracted Entities:'} {row.annotated_preview}
                    </p>
                  )}
                </div>
                <div>
                  <span className="badge badge-pine">
                    {row.tags ? `Tags: ${row.tags}` : `${isAr ? 'الفئة:' : 'Class:'} ${row.label}`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
