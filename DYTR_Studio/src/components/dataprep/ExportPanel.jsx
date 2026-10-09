// ==============================================================================
// DyTrStudio Data Preparation: Phase 5 - Export & Reproducibility Pipeline
// Multi-Format Export, Studio Library Management, Deterministic Python Codegen
// Aligned with Pineau et al. (2021) Reproducibility & FAIR Principles (2016)
// ==============================================================================

import React, { useState, useEffect, useMemo } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import TermTooltip from '../common/TermTooltip';
import SectionCard from '../common/SectionCard';
import { generatePythonCode } from '../../lib/data/processor';
import { CheckCircle } from 'lucide-react';

export default function ExportPanel() {
  const { language, setActiveModule } = useStudioStore();
  const isAr = language === 'ar';

  const {
    currentDataset,
    datasetName,
    operationsLog,
    saveToLibrary,
  } = useDatasetStore();

  const [exportFilename, setExportFilename] = useState(
    datasetName ? datasetName.replace(/\.[^/.]+$/, '') : 'dytr_processed_corpus'
  );

  // Library Save Form State
  const [libName, setLibName] = useState(datasetName || (isAr ? 'مجموعة_بيانات_مجهزة' : 'processed_corpus'));
  const [libDesc, setLibDesc] = useState(
    isAr
      ? 'مجموعة بيانات نظيفة ومعالجة جاهزة للتدريب في استوديو المحولات الديناميكية DYTR Studio.'
      : 'Cleaned and preprocessed corpus ready for training in DYTR Studio.'
  );
  const [libTags, setLibTags] = useState('nlp, transformer, arabic, dytr');
  const [saveToast, setSaveToast] = useState(false);
  const [copyToast, setCopyToast] = useState(false);

  // Dynamically sync export filename and library name when switching active datasets
  useEffect(() => {
    if (datasetName) {
      const cleanBase = datasetName.replace(/\.[^/.]+$/, '');
      setExportFilename(cleanBase);
      setLibName(datasetName);
    }
  }, [datasetName]);

  // Python Code Generation - strictly bound to currently active dataset and export target
  const generatedCode = useMemo(() => {
    return generatePythonCode(
      operationsLog,
      datasetName || 'dataset.csv',
      `${exportFilename || 'dytr_processed_corpus'}.csv`
    );
  }, [operationsLog, datasetName, exportFilename]);

  // Matrix dimensions
  const rowCount = currentDataset?.length || 0;
  const colCount = currentDataset && currentDataset.length > 0 ? Object.keys(currentDataset[0]).length : 0;
  const approxSizeKB = useMemo(() => {
    if (!currentDataset || currentDataset.length === 0) return 0;
    return (JSON.stringify(currentDataset).length / 1024).toFixed(1);
  }, [currentDataset]);

  // --- Download Handlers ---
  const downloadBlob = (content, filename, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportCSV = () => {
    if (!currentDataset || currentDataset.length === 0) return;
    const headers = Object.keys(currentDataset[0] || {}).filter((k) => !k.endsWith('_original'));
    const rows = currentDataset.map((row) =>
      headers
        .map((h) => {
          const val = row[h];
          if (val === null || val === undefined) return '';
          const str = String(val);
          return str.includes(',') || str.includes('\n') || str.includes('"')
            ? `"${str.replace(/"/g, '""')}"`
            : str;
        })
        .join(',')
    );
    const csvContent = '\uFEFF' + [headers.join(','), ...rows].join('\n');
    downloadBlob(csvContent, `${exportFilename || 'dytr_corpus'}.csv`, 'text/csv;charset=utf-8;');
  };

  const handleExportJSON = () => {
    if (!currentDataset || currentDataset.length === 0) return;
    const cleanData = currentDataset.map((row) => {
      const copy = { ...row };
      Object.keys(copy).forEach((k) => {
        if (k.endsWith('_original')) delete copy[k];
      });
      return copy;
    });
    const jsonContent = JSON.stringify(cleanData, null, 2);
    downloadBlob(jsonContent, `${exportFilename || 'dytr_corpus'}.json`, 'application/json');
  };

  const handleExportTSV = () => {
    if (!currentDataset || currentDataset.length === 0) return;
    const headers = Object.keys(currentDataset[0] || {}).filter((k) => !k.endsWith('_original'));
    const rows = currentDataset.map((row) =>
      headers.map((h) => String(row[h] || '').replace(/\t/g, ' ')).join('\t')
    );
    const tsvContent = '\uFEFF' + [headers.join('\t'), ...rows].join('\n');
    downloadBlob(tsvContent, `${exportFilename || 'dytr_corpus'}.tsv`, 'text/tab-separated-values;charset=utf-8;');
  };

  const handleExportScript = () => {
    downloadBlob(generatedCode, `${exportFilename || 'dytr_pipeline'}_pipeline.py`, 'text/x-python');
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopyToast(true);
    setTimeout(() => setCopyToast(false), 2500);
  };

  const handleSaveToLibrary = () => {
    if (!libName.trim()) return;

    saveToLibrary({
      name: libName.trim(),
      description: libDesc.trim(),
      tags: libTags.split(',').map((t) => t.trim()).filter(Boolean),
    });

    setSaveToast(true);
    setTimeout(() => setSaveToast(false), 3000);
  };

  const handleDownloadAuditLog = () => {
    const report = [
      `# DyTrStudio Data Provenance & Reproducibility Audit Log`,
      `**Dataset Identifier:** ${datasetName || 'In-Memory Matrix'}`,
      `**Total Matrix Dimensions:** ${rowCount} rows × ${colCount} columns`,
      `**Estimated Size:** ${approxSizeKB} KB`,
      `**Generated At:** ${new Date().toISOString()}`,
      `\n## Transformation History & Lineage Operations:\n`,
      ...(operationsLog && operationsLog.length > 0
        ? operationsLog.map(
            (op, i) =>
              `### ${i + 1}. [${op.timestamp || 'N/A'}] ${op.title_en || op.title_ar || op.type}\n` +
              `- **Operation Type:** \`${op.type}\`\n` +
              `- **Target Column:** \`${op.column || 'Global Matrix'}\`\n` +
              `- **Shape After Step:** ${op.rowsCount || rowCount} rows × ${op.colsCount || colCount} columns\n`
          )
        : ['*No transformation operations recorded. Raw dataset exported directly.*']),
      `\n## Academic Standards & Citations:`,
      `1. Pineau, J. et al. (2021). Improving Reproducibility in Machine Learning Research. Journal of Machine Learning Research (JMLR).`,
      `2. Wilkinson, M. D. et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data, 3:160018.`,
    ].join('\n');

    downloadBlob(report, `${exportFilename || 'provenance'}_provenance_audit.md`, 'text/markdown');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header Banner */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.12) 0%, rgba(13, 148, 136, 0.04) 100%)',
          border: '1px solid var(--border-medium)',
          borderRadius: '12px',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '30px',
                  height: '30px',
                  borderRadius: '50%',
                  background: 'var(--accent-teal, #14b8a6)',
                  color: '#ffffff',
                  fontWeight: '700',
                  fontSize: '0.9rem',
                }}
              >
                5
              </span>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '700', margin: 0, color: 'var(--text-primary)' }}>
                {isAr ? 'المرحلة 5 — التصدير وحفظ الاستوديو والشفافية الأكاديمية' : 'Phase 5 — Export, Studio Library & FAIR Provenance'}
              </h2>
            </div>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '880px', lineHeight: 1.6 }}>
              {isAr
                ? 'صدّر بياناتك المعالجة بصيغ متعددة، احفظها في مكتبة الاستوديو لإعادة استخدامها، وولّد كود بايثون يوثّق كل خطوة لضمان الشفافية العلمية.'
                : 'Export preprocessed datasets across standard formats, persist to studio local storage for training, and generate deterministic Python scripts with audit trails aligned with FAIR research standards.'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <TermTooltip termKey="reproducibility" />
            <TermTooltip termKey="provenance" />
          </div>
        </div>

        {/* Live Corpus Metrics Summary */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
            gap: '0.85rem',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '8px',
            padding: '0.85rem 1rem',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
              {isAr ? 'مجموعة البيانات الحالية' : 'Current Dataset'}
            </span>
            <span
              style={{
                fontSize: '0.84rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                wordBreak: 'break-word',
                display: 'block',
                lineHeight: 1.35,
              }}
            >
              <bdi dir="ltr">{datasetName || (isAr ? 'بدون عنوان' : 'Untitled Dataset')}</bdi>
            </span>
          </div>
          <div style={{ minWidth: 0 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
              {isAr ? 'الأبعاد' : 'Dimensions'}
            </span>
            <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--accent-teal, #14b8a6)', display: 'block', wordBreak: 'break-word' }}>
              {rowCount.toLocaleString()} {isAr ? 'صفاً' : 'rows'} × {colCount} {isAr ? 'أعمدة' : 'cols'}
            </span>
          </div>
          <div style={{ minWidth: 0 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
              {isAr ? 'حجم البيانات التقريبي' : 'Approx Memory Size'}
            </span>
            <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)', display: 'block' }}>
              {approxSizeKB} KB
            </span>
          </div>
          <div style={{ minWidth: 0 }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
              {isAr ? 'سجل العمليات المنفذة' : 'Executed Operations'}
            </span>
            <span style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)', display: 'block' }}>
              {operationsLog?.length || 0} {isAr ? 'عمليات موثقة' : 'operations logged'}
            </span>
          </div>
        </div>
      </div>

      {/* Grid: Download Formats & Save to Library */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {/* Section 1: File Download Formats */}
        <SectionCard
          title={isAr ? 'تصدير وتنزيل الملفات' : 'Direct File Downloads'}
          badge={isAr ? 'صيغ قياسية مفتوحة' : 'Open Standards'}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                {isAr ? 'اسم ملف التصدير (بدون الامتداد)' : 'Export Filename (without extension)'}
              </label>
              <input
                type="text"
                value={exportFilename}
                onChange={(e) => setExportFilename(e.target.value)}
                placeholder="dytr_processed_corpus"
                style={{
                  width: '100%',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
              {/* CSV Button */}
              <button
                onClick={handleExportCSV}
                disabled={rowCount === 0}
                style={{
                  padding: '1rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  textAlign: 'start',
                  cursor: rowCount === 0 ? 'not-allowed' : 'pointer',
                  opacity: rowCount === 0 ? 0.6 : 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-teal, #14b8a6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                  </svg>
                  <span style={{ fontWeight: '700', fontSize: '0.95rem', color: 'var(--accent-teal, #14b8a6)' }}>
                    CSV (.csv)
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'الصيغة القياسية لـ Excel و Pandas' : 'Universal standard for Pandas & Excel'}
                </span>
              </button>

              {/* JSON Button */}
              <button
                onClick={handleExportJSON}
                disabled={rowCount === 0}
                style={{
                  padding: '1rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  textAlign: 'start',
                  cursor: rowCount === 0 ? 'not-allowed' : 'pointer',
                  opacity: rowCount === 0 ? 0.6 : 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-teal, #14b8a6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  <span style={{ fontWeight: '700', fontSize: '0.95rem', color: 'var(--accent-teal, #14b8a6)' }}>
                    JSON (.json)
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'صيغة كائنات JSON لـ Web APIs' : 'Structured array for API endpoints'}
                </span>
              </button>

              {/* TSV Button */}
              <button
                onClick={handleExportTSV}
                disabled={rowCount === 0}
                style={{
                  padding: '1rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  textAlign: 'start',
                  cursor: rowCount === 0 ? 'not-allowed' : 'pointer',
                  opacity: rowCount === 0 ? 0.6 : 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-teal, #14b8a6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <line x1="9" y1="3" x2="9" y2="21" />
                  </svg>
                  <span style={{ fontWeight: '700', fontSize: '0.95rem', color: 'var(--accent-teal, #14b8a6)' }}>
                    TSV (.tsv)
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'مفصول بمسافات لمهام معالجة اللغة' : 'Tab-delimited format for NLP'}
                </span>
              </button>

              {/* Audit Log Button */}
              <button
                onClick={handleDownloadAuditLog}
                disabled={rowCount === 0}
                style={{
                  padding: '1rem 0.85rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  textAlign: 'start',
                  cursor: rowCount === 0 ? 'not-allowed' : 'pointer',
                  opacity: rowCount === 0 ? 0.6 : 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-teal, #14b8a6)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                  <span style={{ fontWeight: '700', fontSize: '0.95rem', color: 'var(--accent-teal, #14b8a6)' }}>
                    Audit Log (.md)
                  </span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'تقرير سجل العمليات والأصول العلمية' : 'FAIR Provenance report in Markdown'}
                </span>
              </button>
            </div>
          </div>
        </SectionCard>

        {/* Section 2: Save to Studio Library */}
        <SectionCard
          title={isAr ? 'الحفظ الدائم في مكتبة الاستوديو' : 'Save to Studio Datasets Library'}
          badge={isAr ? 'تخزين محلي' : 'Local Storage'}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                {isAr ? 'اسم المجموعة في المكتبة' : 'Dataset Library Name'}
              </label>
              <input
                type="text"
                value={libName}
                onChange={(e) => setLibName(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                {isAr ? 'الوصف التوثيقي' : 'Academic Description'}
              </label>
              <textarea
                rows={2}
                value={libDesc}
                onChange={(e) => setLibDesc(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                  resize: 'none',
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                {isAr ? 'الوسوم التصنيفية (مفصولة بفواصل)' : 'Tags (comma-separated)'}
              </label>
              <input
                type="text"
                value={libTags}
                onChange={(e) => setLibTags(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.25rem' }}>
              <button
                onClick={handleSaveToLibrary}
                disabled={rowCount === 0 || !libName.trim()}
                style={{
                  padding: '0.65rem 1.25rem',
                  borderRadius: '6px',
                  border: 'none',
                  background: 'var(--accent-teal, #14b8a6)',
                  color: '#ffffff',
                  fontSize: '0.85rem',
                  fontWeight: '700',
                  cursor: rowCount === 0 || !libName.trim() ? 'not-allowed' : 'pointer',
                  opacity: rowCount === 0 || !libName.trim() ? 0.6 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                  <polyline points="17 21 17 13 7 13 7 21" />
                  <polyline points="7 3 7 8 15 8" />
                </svg>
                {isAr ? 'حفظ في مكتبة الاستوديو' : 'Save to Studio Library'}
              </button>
              {saveToast && (
                <span style={{ fontSize: '0.82rem', color: 'var(--color-success, #22c55e)', fontWeight: '700', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                  <CheckCircle size={14} />
                  <span>{isAr ? 'تم الحفظ في المكتبة بنجاح! ستجدها جاهزة للتحميل في لوحة [جلب البيانات 01.6].' : 'Saved to Library! Available now in [Data Acquisition 01.6].'}</span>
                </span>
              )}
            </div>
          </div>
        </SectionCard>
      </div>

      {/* Section 3: Generated Python Reproducibility Script - Apple Developer IDE & Antigravity Syntax */}
      <SectionCard
        title={isAr ? 'توليد كود بايثون قابل لإعادة الإنتاج (Python Pipeline)' : 'Reproducible Python Pipeline'}
        badge={isAr ? 'كود قطعي' : 'FAIR Principles'}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '750px', lineHeight: 1.5 }}>
              {isAr
                ? 'كل عملية تنظيف أو تحويل قمت بها في الاستوديو يتم تحويلها تلقائياً إلى كود بايثون يمكنك تشغيله لاحقاً لإعادة إنتاج نفس النتائج:'
                : 'Every cleaning or transformation step you performed in the studio is automatically converted to a Python script you can run to reproduce the same results:'}
            </p>
          </div>

          {/* Apple IDE Container */}
          <div
            style={{
              background: '#0a0d14',
              borderRadius: '12px',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              boxShadow: '0 14px 38px -6px rgba(0, 0, 0, 0.75), 0 0 0 1px rgba(255, 255, 255, 0.05)',
              overflow: 'hidden',
              direction: 'ltr',
              textAlign: 'left',
            }}
          >
            {/* Apple macOS Window Header */}
            <div
              style={{
                background: 'linear-gradient(180deg, #161d2b 0%, #101520 100%)',
                padding: '0.55rem 1rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '1rem',
                flexWrap: 'wrap',
              }}
            >
              {/* Apple Traffic Light Controls */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
                <span
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: '#FF5F56',
                    border: '1px solid #E0443E',
                    display: 'inline-block',
                    boxShadow: '0 1px 3px rgba(255, 95, 86, 0.35)',
                  }}
                />
                <span
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: '#FFBD2E',
                    border: '1px solid #DEA123',
                    display: 'inline-block',
                    boxShadow: '0 1px 3px rgba(255, 189, 46, 0.35)',
                  }}
                />
                <span
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: '#27C93F',
                    border: '1px solid #1AAB29',
                    display: 'inline-block',
                    boxShadow: '0 1px 3px rgba(39, 201, 63, 0.35)',
                  }}
                />
              </div>

              {/* macOS Tab / Active Document Badge */}
              <div
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  padding: '0.22rem 0.85rem',
                  borderRadius: '6px',
                  fontFamily: '"SF Mono", "JetBrains Mono", "Fira Code", monospace',
                  fontSize: '0.74rem',
                  color: '#e2e8f0',
                  letterSpacing: '0.02em',
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                <span style={{ fontWeight: 600 }}>{exportFilename || 'dytr_pipeline'}.py</span>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#38bdf8', opacity: 0.8 }} />
              </div>

              {/* Action Buttons (Copy + Download) */}
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <button
                  type="button"
                  onClick={handleCopyCode}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    padding: '0.28rem 0.75rem',
                    borderRadius: '6px',
                    background: copyToast ? 'rgba(34, 197, 94, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                    color: copyToast ? '#4ade80' : '#cbd5e1',
                    border: copyToast ? '1px solid rgba(74, 222, 128, 0.35)' : '1px solid rgba(255, 255, 255, 0.15)',
                    cursor: 'pointer',
                    transition: 'all 0.18s ease-in-out',
                  }}
                >
                  {copyToast ? (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>{isAr ? 'تم النسخ!' : 'Copied!'}</span>
                    </>
                  ) : (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                      </svg>
                      <span>{isAr ? 'نسخ الكود' : 'Copy Code'}</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleExportScript}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    padding: '0.28rem 0.75rem',
                    borderRadius: '6px',
                    background: 'linear-gradient(135deg, var(--accent-teal) 0%, var(--accent-cyan) 100%)',
                    color: '#ffffff',
                    border: 'none',
                    cursor: 'pointer',
                    boxShadow: '0 2px 10px var(--accent-teal-glow)',
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>{isAr ? 'تنزيل (.py)' : 'Download .py'}</span>
                </button>
              </div>
            </div>

            {/* Code Editor Body with Line Numbers & Antigravity Syntax Palette */}
            <div
              style={{
                padding: '0.85rem 0',
                maxHeight: '340px',
                overflowX: 'auto',
                overflowY: 'auto',
                fontFamily: '"SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace',
                fontSize: '0.78rem',
                lineHeight: 1.6,
              }}
            >
              {(generatedCode || '').split('\n').map((line, lineIdx) => {
                const hashIndex = line.indexOf('#');
                const codePart = hashIndex !== -1 ? line.slice(0, hashIndex) : line;
                const commentPart = hashIndex !== -1 ? line.slice(hashIndex) : '';

                const tokenRegex = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|r'(?:\\.|[^'\\])*'|f'(?:\\.|[^'\\])*'|\b(?:from|import|def|class|return|if|else|elif|for|in|as|with|pass|lambda|sum|print|try|except|finally)\b|\b(?:True|False|None)\b|\b(?:pd|np|re|DataFrame|Series|Pipeline|Transformer|StandardScaler|LabelEncoder|Dataset|DyTr)\b|\b\d+(?:\.\d+)?\b|[a-zA-Z_]\w*(?=\()|[a-zA-Z_]\w*(?==)|[a-zA-Z_]\w*|[^\s\w])/g;

                const tokens = [];
                let match;
                let lastIndex = 0;

                while ((match = tokenRegex.exec(codePart)) !== null) {
                  if (match.index > lastIndex) {
                    tokens.push({ type: 'text', value: codePart.slice(lastIndex, match.index) });
                  }
                  const val = match[0];
                  let type = 'text';

                  if (val.startsWith('"') || val.startsWith("'") || val.startsWith("r'") || val.startsWith("f'")) {
                    type = 'string';
                  } else if (/^(from|import|def|class|return|if|else|elif|for|in|as|with|pass|lambda|sum|print|try|except|finally)$/.test(val)) {
                    type = 'keyword';
                  } else if (/^(True|False|None)$/.test(val)) {
                    type = 'boolean';
                  } else if (/^(pd|np|re|DataFrame|Series|Pipeline|Transformer|StandardScaler|LabelEncoder|Dataset|DyTr)$/.test(val)) {
                    type = 'class';
                  } else if (/^\d+(\.\d+)?$/.test(val)) {
                    type = 'number';
                  } else if (codePart[match.index + val.length] === '(') {
                    type = 'function';
                  } else if (codePart[match.index + val.length] === '=') {
                    type = 'parameter';
                  }

                  tokens.push({ type, value: val });
                  lastIndex = tokenRegex.lastIndex;
                }

                if (lastIndex < codePart.length) {
                  tokens.push({ type: 'text', value: codePart.slice(lastIndex) });
                }

                const getTokenColor = (tokenType) => {
                  switch (tokenType) {
                    case 'keyword': return '#38bdf8'; // Antigravity electric cyan
                    case 'class': return '#f43f5e'; // Antigravity vivid coral/rose
                    case 'function': return '#c084fc'; // Antigravity purple/violet
                    case 'parameter': return '#7dd3fc'; // Soft sky blue
                    case 'string': return '#4ade80'; // Emerald lime
                    case 'number':
                    case 'boolean': return '#fbbf24'; // Amber gold
                    default: return '#e2e8f0'; // Crisp off-white
                  }
                };

                return (
                  <div key={lineIdx} style={{ display: 'flex', minHeight: '1.45rem', alignItems: 'center' }}>
                    {/* Line Number Gutter */}
                    <span
                      style={{
                        width: '3.2rem',
                        flexShrink: 0,
                        textAlign: 'right',
                        paddingRight: '1rem',
                        color: '#475569',
                        userSelect: 'none',
                        fontSize: '0.72rem',
                        borderRight: '1px solid rgba(255, 255, 255, 0.06)',
                      }}
                    >
                      {lineIdx + 1}
                    </span>
                    {/* Code Line */}
                    <span style={{ flex: 1, paddingLeft: '1rem', whiteSpace: 'pre', overflowX: 'visible' }}>
                      {tokens.map((tok, tokIdx) => (
                        <span
                          key={tokIdx}
                          style={{
                            color: getTokenColor(tok.type),
                            fontWeight: tok.type === 'keyword' || tok.type === 'class' ? 600 : 400,
                          }}
                        >
                          {tok.value}
                        </span>
                      ))}
                      {commentPart && (
                        <span style={{ color: '#64748b', fontStyle: 'italic' }}>
                          {commentPart}
                        </span>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Apple IDE Footer / Status Bar */}
            <div
              style={{
                background: '#0d121c',
                borderTop: '1px solid rgba(255, 255, 255, 0.06)',
                padding: '0.35rem 1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.68rem',
                color: '#64748b',
                fontFamily: '"SF Mono", "JetBrains Mono", monospace',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <span style={{ color: '#38bdf8', fontWeight: 600 }}>Python 3.10+</span>
                <span>•</span>
                <span>pandas 2.2</span>
                <span>•</span>
                <span>FAIR Reproducibility</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <span>UTF-8</span>
                <span>•</span>
                <span>Spaces: 4</span>
                <span>•</span>
                <span style={{ color: '#4ade80' }}>● Deterministic Pipeline</span>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

    </div>
  );
}
