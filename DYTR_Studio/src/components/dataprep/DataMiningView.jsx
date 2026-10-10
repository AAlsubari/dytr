// ==============================================================================
// DyTrStudio Data Mining & Preparation Studio - Master Container
// 5-Phase CRISP-DM Architecture with Provenance Tracking & 3-Layer Academic Engine
// Zero Emojis Policy - Bilingual RTL/LTR - Dual Theme (Dark / Light) Compatible
// ==============================================================================

import React, { useState, useEffect, useMemo } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';
import { academicGlossary, glossaryCategories } from '../../lib/data/glossary';

import TermTooltip from '../common/TermTooltip';
import StepProgressBar from '../common/StepProgressBar';
import SectionCard from '../common/SectionCard';
import AcquisitionPanel from './AcquisitionPanel';
import InspectionPanel from './InspectionPanel';
import PreprocessingPanel from './PreprocessingPanel';
import TransformationPanel from './TransformationPanel';
import ExportPanel from './ExportPanel';
import ActiveDatasetBar from './ActiveDatasetBar';
import DatasetMergeModal from './DatasetMergeModal';
import { ArrowLeft, ArrowRight } from 'lucide-react';

export default function DataMiningView() {
  const { language, theme, isSidebarCollapsed, setActiveModule } = useStudioStore();
  const isAr = language === 'ar';
  const sidebarOffset = isSidebarCollapsed ? 68 : 280;

  const {
    currentDataset,
    datasetName,
    loadedDatasets,
    activeDatasetId,
    switchActiveDataset,
    activePhase,
    setActivePhase,
    pastHistory,
    futureHistory,
    undo,
    redo,
    operationsLog,
    saveToLibrary,
  } = useDatasetStore();

  const [previewOpen, setPreviewOpen] = useState(false);
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [glossaryModalOpen, setGlossaryModalOpen] = useState(false);
  const [glossarySearch, setGlossarySearch] = useState('');
  const [selectedGlossaryCategory, setSelectedGlossaryCategory] = useState('all');
  const [activeGlossaryTerm, setActiveGlossaryTerm] = useState(null);
  const [saveToast, setSaveToast] = useState(false);

  // Pagination for sticky bottom preview
  const [previewPage, setPreviewPage] = useState(1);
  const [previewSearch, setPreviewSearch] = useState('');
  const pageSize = 8;

  // Keyboard shortcut listener for Ctrl+Z (Undo) and Ctrl+Shift+Z / Ctrl+Y (Redo)
  const [previewHeight, setPreviewHeight] = useState(360);
  const isResizingRef = React.useRef(false);

  const handleDragResize = (e) => {
    e.preventDefault();
    isResizingRef.current = true;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'ns-resize';

    const onMouseMove = (moveEvent) => {
      if (!isResizingRef.current) return;
      const newHeight = window.innerHeight - moveEvent.clientY;
      const clampedHeight = Math.max(140, Math.min(window.innerHeight * 0.85, newHeight));
      setPreviewHeight(clampedHeight);
    };

    const onMouseUp = () => {
      isResizingRef.current = false;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  // Keyboard shortcut listener for Ctrl+Z (Undo) and Ctrl+Shift+Z / Ctrl+Y (Redo)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        if (e.shiftKey) {
          e.preventDefault();
          redo();
        } else {
          e.preventDefault();
          undo();
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  // Phase Definitions matching CRISP-DM methodology
  const phases = [
    {
      id: 1,
      code: '02-A',
      name_ar: 'جلب البيانات',
      name_en: 'Data Acquisition',
      desc_ar: 'جمع البيانات من الملفات، الويب، البث أو العينات الاصطناعية لبدء التجربة',
      desc_en: 'Upload, scrape, search & synthetic dataset generation',
    },
    {
      id: 2,
      code: '02-B',
      name_ar: 'فحص البيانات',
      name_en: 'Data Inspection',
      desc_ar: 'تشخيص سلامة البيانات، رصد القيم المفقودة، وتقييم مؤشر الجودة الشامل',
      desc_en: 'Audit completeness, quality gauge & feature distributions',
    },
    {
      id: 3,
      code: '02-C',
      name_ar: 'تنظيف البيانات',
      name_en: 'Data Cleaning',
      desc_ar: 'معالجة النقص والتكرار وتوحيد المقاييس لاستقرار تدريب الذكاء الاصطناعي',
      desc_en: 'Impute missing values, remove duplicates & scale features',
    },
    {
      id: 4,
      code: '02-D',
      name_ar: 'تحويل البيانات',
      name_en: 'Data Transformation',
      desc_ar: 'فلترة الصفوف واشتقاق خصائص جديدة وتجميع البيانات للحصول على أفضل النتائج',
      desc_en: 'Filter logic, engineer new features & aggregate datasets',
    },
    {
      id: 5,
      code: '02-E',
      name_ar: 'تصدير البيانات',
      name_en: 'Data Export',
      desc_ar: 'تنزيل النتائج بصيغة CSV أو حفظ البيانات في مكتبة الاستوديو وتوليد الكود',
      desc_en: 'Download CSV, save to studio library & generate Python scripts',
    },
  ];

  const stepLabels = phases.map((p) => (isAr ? p.name_ar : p.name_en));

  // Filtered Preview Data
  const filteredPreviewData = useMemo(() => {
    if (!previewSearch.trim()) return currentDataset;
    const q = previewSearch.toLowerCase();
    return currentDataset.filter((row) =>
      Object.values(row).some((val) => String(val).toLowerCase().includes(q))
    );
  }, [currentDataset, previewSearch]);

  const totalPreviewPages = Math.ceil(filteredPreviewData.length / pageSize) || 1;
  const currentPreviewSlice = useMemo(() => {
    const start = (previewPage - 1) * pageSize;
    return filteredPreviewData.slice(start, start + pageSize);
  }, [filteredPreviewData, previewPage, pageSize]);

  const columns = useMemo(
    () => Object.keys(currentDataset[0] || {}).filter((k) => !k.endsWith('_original')),
    [currentDataset]
  );

  // Filtered Glossary for Encyclopedia Modal
  const filteredGlossaryList = useMemo(() => {
    return Object.entries(academicGlossary).filter(([key, term]) => {
      const matchCat =
        selectedGlossaryCategory === 'all' || term.category === selectedGlossaryCategory;
      const matchQuery =
        !glossarySearch.trim() ||
        key.toLowerCase().includes(glossarySearch.toLowerCase()) ||
        (term.title_ar && term.title_ar.includes(glossarySearch)) ||
        (term.title_en && term.title_en.toLowerCase().includes(glossarySearch.toLowerCase())) ||
        (term.layer1_ar && term.layer1_ar.includes(glossarySearch));
      return matchCat && matchQuery;
    });
  }, [glossarySearch, selectedGlossaryCategory]);

  const handleQuickSave = () => {
    saveToLibrary({ name: datasetName });
    setSaveToast(true);
    setTimeout(() => setSaveToast(false), 2500);
  };

  const currentPhaseInfo = phases[activePhase - 1] || phases[0];

  return (
    <div style={{ background: 'var(--bg-primary)', minHeight: 'calc(100vh - 120px)', paddingBottom: previewOpen ? '290px' : '80px' }}>
      {/* Academy-Style 900px Centered Container */}
      <div className="container" style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem 1.5rem' }}>
        
        {/* Page Header Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.75rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.35rem' }}>
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: 'var(--accent-teal)',
                  boxShadow: '0 0 8px var(--accent-teal)',
                }}
              />
              <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                {isAr ? 'استوديو تحضير وتنقيب البيانات' : 'Data Preparation & Mining Studio'}
              </h1>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: 0, lineHeight: 1.6 }}>
              {isAr
                ? 'رحلة تفاعلية مبسطة من 5 خطوات لجلب البيانات وفحصها وتنظيفها وتحويلها وتصديرها لبدء تدريب النماذج'
                : 'A clean 5-step guided workflow to acquire, inspect, clean, transform and export datasets for AI model training.'}
            </p>
          </div>

          {/* Top Quick Actions */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={undo}
              disabled={pastHistory.length === 0}
              className="btn btn-secondary btn-sm"
              style={{
                color: pastHistory.length > 0 ? 'var(--text-primary)' : 'var(--text-muted)',
                opacity: pastHistory.length > 0 ? 1 : 0.45,
                gap: '0.35rem',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="1 4 1 10 7 10"></polyline>
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
              </svg>
              <span>{isAr ? 'تراجع' : 'Undo'}</span>
            </button>

            <button
              onClick={redo}
              disabled={futureHistory.length === 0}
              className="btn btn-secondary btn-sm"
              style={{
                color: futureHistory.length > 0 ? 'var(--text-primary)' : 'var(--text-muted)',
                opacity: futureHistory.length > 0 ? 1 : 0.45,
                gap: '0.35rem',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
              <span>{isAr ? 'إعادة' : 'Redo'}</span>
            </button>

            <button
              onClick={handleQuickSave}
              className="btn btn-secondary btn-sm"
              style={{ gap: '0.35rem' }}
            >
              <span>{saveToast ? (isAr ? 'تم الحفظ!' : 'Saved!') : (isAr ? 'حفظ سريع' : 'Quick Save')}</span>
            </button>

            <button
              onClick={() => setGlossaryModalOpen(true)}
              className="btn btn-primary btn-sm"
              style={{ gap: '0.35rem' }}
            >
              <span>{isAr ? 'موسوعة المفاهيم' : 'Encyclopedia'}</span>
            </button>
          </div>
        </div>

        {/* Multi-Dataset Workspace Bar (Tabs & Active Dataset Indicator) */}
        <ActiveDatasetBar onOpenMerge={() => setMergeModalOpen(true)} />

        {/* Step Progress Bar (Matches Academy View) */}
        <StepProgressBar
          currentStep={activePhase}
          totalSteps={5}
          labels={stepLabels}
          onStepClick={(s) => setActivePhase(s)}
          language={language}
        />

        {/* Active Step Content Wrapped in SectionCard with Persistent Component Mounting */}
        <SectionCard
          sectionCode={currentPhaseInfo.code}
          title={isAr ? currentPhaseInfo.name_ar : currentPhaseInfo.name_en}
          subtitle={isAr ? currentPhaseInfo.desc_ar : currentPhaseInfo.desc_en}
        >
          <div style={{ display: activePhase === 1 ? 'block' : 'none' }}>
            <AcquisitionPanel />
          </div>
          <div style={{ display: activePhase === 2 ? 'block' : 'none' }}>
            <InspectionPanel />
          </div>
          <div style={{ display: activePhase === 3 ? 'block' : 'none' }}>
            <PreprocessingPanel />
          </div>
          <div style={{ display: activePhase === 4 ? 'block' : 'none' }}>
            <TransformationPanel />
          </div>
          <div style={{ display: activePhase === 5 ? 'block' : 'none' }}>
            <ExportPanel />
          </div>
        </SectionCard>

        {/* Unified Studio Stage & Module Navigation Footer */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
            marginTop: '2.5rem',
            padding: '1.25rem 0 0.5rem 0',
            borderTop: '1px solid var(--border-subtle)',
            flexWrap: 'wrap',
          }}
        >
          {/* Previous Button (Back to phase - 1, or to Problem Definition if at phase 1) */}
          <button
            onClick={() => {
              if (activePhase > 1) {
                setActivePhase(activePhase - 1);
              } else {
                setActiveModule('problem');
              }
            }}
            className="btn btn-secondary"
            style={{
              padding: '0.65rem 1.35rem',
              cursor: 'pointer',
              fontSize: '0.84rem',
              fontWeight: 700,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
          >
            <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            <span>
              {activePhase === 1
                ? (isAr ? 'العودة إلى تحديد المشكلة (01)' : 'Back to Problem Framing (01)')
                : (isAr ? 'الخطوة السابقة' : 'Previous Step')}
            </span>
          </button>

          {/* Current Step Indicator Badge */}
          <div
            style={{
              fontSize: '0.84rem',
              fontWeight: 700,
              color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)',
              background: 'var(--bg-tertiary)',
              padding: '0.4rem 1rem',
              borderRadius: 'var(--radius-sm, 6px)',
              border: '1px solid var(--border-subtle)',
              textAlign: 'center',
            }}
          >
            {isAr
              ? `المرحلة ${activePhase} من 5: ${phases[activePhase - 1]?.name_ar || ''}`
              : `Phase ${activePhase} of 5: ${phases[activePhase - 1]?.name_en || ''}`}
          </div>

          {/* Next Button (Advance to next phase, or transition to Architecture Builder if at phase 5) */}
          <button
            onClick={() => {
              if (activePhase < 5) {
                setActivePhase(activePhase + 1);
              } else {
                setActiveModule('builder');
              }
            }}
            className="btn btn-primary"
            style={{
              padding: '0.65rem 1.45rem',
              cursor: 'pointer',
              fontSize: '0.84rem',
              fontWeight: 700,
              background: activePhase === 5 ? 'var(--accent-teal, #14b8a6)' : undefined,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              boxShadow: activePhase === 5 ? '0 4px 14px rgba(20, 184, 166, 0.3)' : undefined,
            }}
          >
            <span>
              {activePhase === 5
                ? (isAr ? 'المتابعة إلى بناء المعمارية (03)' : 'Proceed to Architecture Builder (03)')
                : (isAr ? 'الخطوة التالية' : 'Next Step')}
            </span>
            <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          </button>
        </div>
      </div>

      {/* Sticky Bottom Resizable Collapsible Live Data Preview (IDE/Windows Drag Resizer) */}
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          right: isAr ? `${sidebarOffset}px` : 0,
          left: isAr ? 0 : `${sidebarOffset}px`,
          background: 'var(--bg-secondary)',
          borderTop: '2px solid var(--accent-teal)',
          zIndex: 40,
          boxShadow: '0 -4px 20px rgba(0,0,0,0.25)',
          height: previewOpen ? `${previewHeight}px` : '46px',
          maxHeight: '85vh',
          minHeight: previewOpen ? '140px' : '46px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          transition: isResizingRef.current ? 'none' : 'height 0.2s cubic-bezier(0.4, 0, 0.2, 1), right 0.25s cubic-bezier(0.16, 1, 0.3, 1), left 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Windows / IDE Style Drag-to-Resize Top Handle Bar */}
        {previewOpen && (
          <div
            onMouseDown={handleDragResize}
            title={isAr ? 'اسحب للأعلى أو الأسفل لتغيير الارتفاع مثل أنظمة التشغيل' : 'Drag up or down to resize height'}
            style={{
              height: '8px',
              background: 'var(--bg-tertiary)',
              borderBottom: '1px solid var(--border-subtle)',
              cursor: 'ns-resize',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              userSelect: 'none',
              flexShrink: 0,
            }}
          >
            <div
              style={{
                width: '48px',
                height: '3px',
                borderRadius: '2px',
                background: 'var(--accent-teal)',
                opacity: 0.8,
              }}
            />
          </div>
        )}

        {/* Preview Bar Handle Header */}
        <div
          onClick={() => setPreviewOpen(!previewOpen)}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0.55rem 1.25rem',
            background: 'var(--bg-elevated)',
            cursor: 'pointer',
            borderBottom: previewOpen ? '1px solid var(--border-medium)' : 'none',
            fontSize: '0.82rem',
            fontWeight: '700',
            userSelect: 'none',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--accent-teal)', display: 'inline-flex', alignItems: 'center' }}>
              {previewOpen ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="18 15 12 9 6 15"></polyline>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              )}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap', minWidth: 0 }}>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', flexShrink: 0 }}>
                {isAr ? 'معاينة جدول البيانات النشط:' : 'Active Dataset Preview:'}
              </span>
              <strong style={{ color: 'var(--text-primary)', fontWeight: 800, fontSize: '0.88rem', fontFamily: 'var(--font-mono)', wordBreak: 'break-word' }}>
                <bdi dir="ltr">{datasetName}</bdi>
              </strong>
            </div>

            {(loadedDatasets || []).length > 1 && (
              <select
                value={activeDatasetId}
                onChange={(e) => {
                  e.stopPropagation();
                  switchActiveDataset(e.target.value);
                }}
                onClick={(e) => e.stopPropagation()}
                style={{
                  padding: '0.2rem 0.55rem',
                  borderRadius: 'var(--radius-sm, 4px)',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--accent-teal)',
                  color: 'var(--accent-teal)',
                  fontSize: '0.74rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {loadedDatasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.rowsCount || ds.data?.length || 0} {isAr ? 'صف' : 'rows'})
                  </option>
                ))}
              </select>
            )}

            <span className="badge badge-teal" style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
              {currentDataset.length} {isAr ? 'صف' : 'rows'} · {columns.length} {isAr ? 'عمود' : 'cols'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }} onClick={(e) => e.stopPropagation()}>
            {previewOpen && (
              <input
                type="text"
                placeholder={isAr ? 'بحث فوري في الخلايا...' : 'Search preview cells...'}
                value={previewSearch}
                onChange={(e) => {
                  setPreviewSearch(e.target.value);
                  setPreviewPage(1);
                }}
                style={{
                  padding: '0.3rem 0.75rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.76rem',
                  width: '180px',
                }}
              />
            )}
            <button
              onClick={() => setPreviewOpen(!previewOpen)}
              className="btn btn-ghost btn-sm"
              style={{ color: 'var(--accent-teal)', fontSize: '0.78rem', fontWeight: 800, padding: '0.2rem 0.5rem' }}
            >
              {previewOpen ? (isAr ? 'طي المعاينة ▲' : 'Collapse ▲') : (isAr ? 'توسيع المعاينة ▼' : 'Expand ▼')}
            </button>
          </div>
        </div>

        {/* Preview Table Content */}
        {previewOpen && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-secondary)' }}>
            <div style={{ flex: 1, overflow: 'auto', background: 'var(--bg-secondary)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'start' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-tertiary)', borderBottom: '1px solid var(--border-medium)', position: 'sticky', top: 0, zIndex: 10 }}>
                    <th style={{ padding: '0.55rem 0.75rem', color: 'var(--text-muted)', width: '50px', textAlign: 'center' }}>#</th>
                    {columns.map((c) => (
                      <th key={c} style={{ padding: '0.55rem 0.75rem', color: 'var(--accent-teal)', fontWeight: '800', fontFamily: 'var(--font-mono)' }}>
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {currentPreviewSlice.map((row, idx) => (
                    <tr
                      key={idx}
                      style={{
                        borderBottom: '1px solid var(--border-subtle)',
                        background: idx % 2 === 0 ? 'var(--bg-secondary)' : 'var(--bg-tertiary)',
                      }}
                    >
                      <td style={{ padding: '0.5rem 0.75rem', color: 'var(--text-muted)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                        {(previewPage - 1) * pageSize + idx + 1}
                      </td>
                      {columns.map((c) => {
                        const valStr = row[c] !== null && row[c] !== undefined ? String(row[c]) : '—';
                        const origVal = row[`${c}_original`];
                        const cellTooltip =
                          origVal !== undefined
                            ? (isAr ? `النص الأصلي للخلية: "${origVal}" | قيمة المتجه: ${valStr}` : `Original raw value: "${origVal}" | Vector value: ${valStr}`)
                            : valStr;

                        return (
                          <td
                            key={c}
                            title={cellTooltip}
                            style={{
                              padding: '0.5rem 0.75rem',
                              color: origVal !== undefined ? 'var(--accent-teal)' : 'var(--text-primary)',
                              fontWeight: origVal !== undefined ? 700 : 400,
                              maxWidth: '320px',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              fontFamily: valStr.startsWith('[') ? 'var(--font-mono)' : 'inherit',
                            }}
                          >
                            {valStr}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0.45rem 1.25rem',
                borderTop: '1px solid var(--border-medium)',
                background: 'var(--bg-tertiary)',
                fontSize: '0.75rem',
              }}
            >
              <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
                {isAr
                  ? `عرض الصفحة ${previewPage} من ${totalPreviewPages} (${filteredPreviewData.length} صفاً)`
                  : `Showing page ${previewPage} of ${totalPreviewPages} (${filteredPreviewData.length} rows)`}
              </span>

              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  onClick={() => setPreviewPage(Math.max(1, previewPage - 1))}
                  disabled={previewPage === 1}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '0.25rem 0.75rem', fontSize: '0.75rem', fontWeight: 700 }}
                >
                  {isAr ? 'السابق' : 'Prev'}
                </button>
                <button
                  onClick={() => setPreviewPage(Math.min(totalPreviewPages, previewPage + 1))}
                  disabled={previewPage >= totalPreviewPages}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '0.25rem 0.75rem', fontSize: '0.75rem', fontWeight: 700 }}
                >
                  {isAr ? 'التالي' : 'Next'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Academic Encyclopedia Modal (Dialog) */}
      {glossaryModalOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.65)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
            padding: '1.5rem',
          }}
          onClick={() => {
            setGlossaryModalOpen(false);
            setActiveGlossaryTerm(null);
          }}
        >
          <div
            style={{
              background: 'var(--bg-secondary)',
              borderRadius: 'var(--radius-lg, 12px)',
              border: '1px solid var(--border-medium)',
              width: '100%',
              maxWidth: '900px',
              maxHeight: '85vh',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: 'var(--shadow-lg)',
              overflow: 'hidden',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                padding: '1.25rem 1.5rem',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'var(--bg-tertiary)',
              }}
            >
              <div>
                <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: '800', color: 'var(--text-primary)' }}>
                  {isAr ? 'موسوعة المفاهيم الأكاديمية لتنقيب البيانات' : 'Academic Encyclopedia for Data Mining'}
                </h3>
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {isAr
                    ? 'شرح ثلاثي الطبقات مع الصيغ الرياضية والمراجع الأكاديمية المعتمدة (APA 7th)'
                    : '3-Layer explanations with formal mathematical formulations and APA 7th citations.'}
                </p>
              </div>
              <button
                onClick={() => {
                  setGlossaryModalOpen(false);
                  setActiveGlossaryTerm(null);
                }}
                className="btn btn-secondary btn-sm"
                style={{ padding: '0.35rem 0.6rem' }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>

            {/* Modal Filters & Search */}
            <div
              style={{
                padding: '0.85rem 1.5rem',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                gap: '0.75rem',
                flexWrap: 'wrap',
                alignItems: 'center',
                background: 'var(--bg-secondary)',
              }}
            >
              <input
                type="text"
                placeholder={isAr ? 'ابحث في أكثر من 50 مفهوماً أكاديمياً...' : 'Search over 50 academic concepts...'}
                value={glossarySearch}
                onChange={(e) => setGlossarySearch(e.target.value)}
                style={{
                  flex: 1,
                  minWidth: '220px',
                  padding: '0.45rem 0.75rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: '1px solid var(--border-medium)',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                }}
              />

              <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto' }}>
                {glossaryCategories.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setSelectedGlossaryCategory(cat.id)}
                    style={{
                      padding: '0.35rem 0.65rem',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: 'none',
                      background: selectedGlossaryCategory === cat.id ? 'var(--accent-teal)' : 'var(--bg-tertiary)',
                      color: selectedGlossaryCategory === cat.id ? '#ffffff' : 'var(--text-secondary)',
                      fontSize: '0.75rem',
                      fontWeight: '700',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {isAr ? cat.name_ar : cat.name_en}
                  </button>
                ))}
              </div>
            </div>

            {/* Modal Body: Term List or Detail View */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem', background: 'var(--bg-secondary)' }}>
              {activeGlossaryTerm ? (
                // Detailed 3-Layer View
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <button
                    onClick={() => setActiveGlossaryTerm(null)}
                    className="btn btn-secondary btn-sm"
                    style={{ alignSelf: 'flex-start', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                  >
                    <ArrowLeft size={14} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                    <span>{isAr ? 'العودة إلى قائمة المفاهيم' : 'Back to Concept Index'}</span>
                  </button>

                  <div>
                    <h2 style={{ fontSize: '1.3rem', fontWeight: '800', color: 'var(--text-primary)', margin: '0 0 0.35rem 0' }}>
                      {isAr ? activeGlossaryTerm.title_ar : activeGlossaryTerm.title_en}
                    </h2>
                    <span style={{ fontSize: '0.82rem', color: 'var(--accent-teal)', fontWeight: '700' }}>
                      {activeGlossaryTerm.title_en}
                    </span>
                  </div>

                  {/* Layer 1 */}
                  <div style={{ padding: '0.85rem', borderRadius: 'var(--radius-md, 8px)', background: 'var(--accent-teal-glow)', border: '1px solid var(--accent-teal)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--accent-teal)', marginBottom: '0.25rem' }}>
                      {isAr ? 'الطبقة 1: الملخص السريع' : 'Layer 1: Executive Tooltip Summary'}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                      {isAr ? activeGlossaryTerm.layer1_ar : activeGlossaryTerm.layer1_en}
                    </div>
                  </div>

                  {/* Layer 2 */}
                  <div style={{ padding: '1rem', borderRadius: 'var(--radius-md, 8px)', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                      {isAr ? 'الطبقة 2: التفسير الأكاديمي الموسع' : 'Layer 2: Elaborated Conceptual Context'}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
                      {isAr ? activeGlossaryTerm.layer2_ar : activeGlossaryTerm.layer2_en}
                    </div>
                  </div>

                  {/* Mathematical Formulation (if available) */}
                  {activeGlossaryTerm.layer3_ar?.formula && (
                    <div style={{ padding: '1rem', borderRadius: 'var(--radius-md, 8px)', background: 'var(--bg-elevated)', color: 'var(--accent-teal)', border: '1px solid var(--border-medium)', fontFamily: 'var(--font-mono)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                        {isAr ? 'الصيغة الرياضية / الخوارزمية:' : 'Mathematical Formulation:'}
                      </div>
                      <div style={{ fontSize: '0.95rem', fontWeight: '700', direction: 'ltr', textAlign: 'left' }}>
                        {activeGlossaryTerm.layer3_ar.formula}
                      </div>
                    </div>
                  )}

                  {/* Academic Citation */}
                  <div style={{ padding: '0.85rem', borderRadius: 'var(--radius-md, 8px)', background: 'var(--bg-tertiary)', border: '1px dashed var(--border-medium)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                      {isAr ? 'المرجع الأكاديمي المعتمد (APA 7th):' : 'Academic Citation (APA 7th):'}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic', direction: 'ltr', textAlign: 'left' }}>
                      {activeGlossaryTerm.citation || 'CRISP-DM Standard Reference Guide (Chapman et al., 2000).'}
                    </div>
                  </div>
                </div>
              ) : (
                // Index Grid View
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.85rem' }}>
                  {filteredGlossaryList.map(([key, term]) => (
                    <div
                      key={key}
                      onClick={() => setActiveGlossaryTerm(term)}
                      style={{
                        padding: '0.85rem',
                        borderRadius: 'var(--radius-md, 8px)',
                        border: '1px solid var(--border-subtle)',
                        background: 'var(--bg-tertiary)',
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = 'var(--accent-teal)';
                        e.currentTarget.style.transform = 'translateY(-2px)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = 'var(--border-subtle)';
                        e.currentTarget.style.transform = 'translateY(0)';
                      }}
                    >
                      <div style={{ fontWeight: '800', fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                        {isAr ? term.title_ar : term.title_en}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--accent-teal)', marginBottom: '0.35rem', fontWeight: '700' }}>
                        {term.title_en}
                      </div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {isAr ? term.layer1_ar : term.layer1_en}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Dataset Merge & Integration Modal */}
      <DatasetMergeModal isOpen={mergeModalOpen} onClose={() => setMergeModalOpen(false)} />
    </div>
  );
}
