// ==============================================================================
// DyTrStudio Feature Selection & Feature Engineering Suite
// Simplified, modern, pedagogical interface for non-specialists & learners
// Zero Emojis Policy - High Academic Rigor - Bilingual RTL/LTR
// ==============================================================================

import React, { useState, useMemo } from 'react';
import TermTooltip from './TermTooltip';
import { buildCompositeFeatureString } from '../../utils/featureEngineering';
import { Check, Plus, X, ArrowRight } from 'lucide-react';

export default function FeatureEngineeringSuite({
  language = 'ar',
  datasetCols = [],
  sampleRow = null,
  datasetAnalysis = {},
  selectedTextCol = '',
  selectedLabelCol = '',
  featureMode = 'single', // 'single' | 'multiple' | 'selection' | 'engineering'
  featureColumns = [],
  featureEngineering = {
    strategy: 'delimiter',
    delimiter: ' [SEP] ',
    template: '',
    excluded_columns: [],
    derived_features: [],
  },
  onChangeConfig = () => {},
  readOnly = false,
}) {
  const isAr = language === 'ar';

  // Active sub-tab
  const [activeTab, setActiveTab] = useState(
    featureMode === 'engineering'
      ? 'engineering'
      : featureMode === 'selection'
      ? 'selection'
      : featureMode === 'multiple' || (featureColumns && featureColumns.length > 1)
      ? 'multiple'
      : 'single'
  );

  const activeCols = useMemo(() => {
    if (Array.isArray(featureColumns) && featureColumns.length > 0) {
      const valid = featureColumns.filter((c) => datasetCols.includes(c));
      if (valid.length > 0) return valid;
    }
    if (selectedTextCol && datasetCols.includes(selectedTextCol)) {
      return [selectedTextCol];
    }
    return datasetCols.slice(0, 1);
  }, [featureColumns, selectedTextCol, datasetCols]);

  const excludedCols = useMemo(() => {
    return Array.isArray(featureEngineering?.excluded_columns)
      ? featureEngineering.excluded_columns
      : [];
  }, [featureEngineering?.excluded_columns]);

  // Handle Mode Switch
  const handleTabChange = (newTab) => {
    setActiveTab(newTab);
    const updatedFE = { ...(featureEngineering || {}) };

    if (newTab === 'single') {
      const primaryCol = activeCols[0] || datasetCols[0] || 'text';
      onChangeConfig({
        feature_mode: 'single',
        text_column: primaryCol,
        feature_columns: [primaryCol],
        feature_engineering: {
          ...updatedFE,
          strategy: 'delimiter',
        },
      });
    } else if (newTab === 'multiple') {
      const cols = activeCols.length > 1 ? activeCols : datasetCols.slice(0, Math.min(3, datasetCols.length));
      onChangeConfig({
        feature_mode: 'multiple',
        text_column: cols.join(' + '),
        feature_columns: cols,
        feature_engineering: {
          ...updatedFE,
          strategy: updatedFE.strategy || 'delimiter',
          delimiter: updatedFE.delimiter || ' [SEP] ',
        },
      });
    } else if (newTab === 'selection') {
      onChangeConfig({
        feature_mode: 'multiple',
        text_column: activeCols.join(' + '),
        feature_columns: activeCols,
        feature_engineering: {
          ...updatedFE,
          strategy: 'delimiter',
          excluded_columns: excludedCols,
        },
      });
    } else if (newTab === 'engineering') {
      const template = updatedFE.template || activeCols.map((c) => `{${c}}`).join(' [SEP] ');
      onChangeConfig({
        feature_mode: 'multiple',
        text_column: activeCols.join(' + '),
        feature_columns: activeCols,
        feature_engineering: {
          ...updatedFE,
          strategy: updatedFE.strategy === 'derived' ? 'derived' : 'custom_template',
          template: template,
        },
      });
    }
  };

  // Toggle Column Selection in Multi-Feature
  const toggleColumnSelection = (col) => {
    if (readOnly) return;
    let nextCols;
    if (activeCols.includes(col)) {
      if (activeCols.length <= 1) return; // Keep at least one
      nextCols = activeCols.filter((c) => c !== col);
    } else {
      nextCols = [...activeCols, col];
    }

    const updatedFE = { ...(featureEngineering || {}) };
    onChangeConfig({
      feature_mode: nextCols.length > 1 ? 'multiple' : 'single',
      text_column: nextCols.join(' + '),
      feature_columns: nextCols,
      feature_engineering: {
        ...updatedFE,
        template: nextCols.map((c) => `{${c}}`).join(updatedFE.delimiter || ' [SEP] '),
      },
    });
  };

  // Toggle Column Exclusion
  const toggleNoiseExclusion = (col) => {
    if (readOnly) return;
    let nextExcluded;
    if (excludedCols.includes(col)) {
      nextExcluded = excludedCols.filter((c) => c !== col);
    } else {
      nextExcluded = [...excludedCols, col];
    }

    onChangeConfig({
      feature_engineering: {
        ...(featureEngineering || {}),
        excluded_columns: nextExcluded,
      },
    });
  };

  // Delimiter change
  const handleDelimiterChange = (delim) => {
    const updatedFE = {
      ...(featureEngineering || {}),
      delimiter: delim,
      template: activeCols.map((c) => `{${c}}`).join(delim),
    };
    onChangeConfig({
      feature_engineering: updatedFE,
    });
  };

  // Live Composite String for Preview
  const previewCompositeText = useMemo(() => {
    if (!sampleRow) return '';
    const raw = sampleRow.rawRow || sampleRow;
    const taskCfg = {
      feature_mode: activeTab === 'single' ? 'single' : 'multiple',
      feature_columns: activeCols,
      text_column: activeCols.join(' + '),
      feature_engineering: {
        ...(featureEngineering || {}),
        strategy: activeTab === 'engineering' ? (featureEngineering?.strategy || 'custom_template') : 'delimiter',
        excluded_columns: excludedCols,
      },
    };
    return buildCompositeFeatureString(raw, taskCfg);
  }, [sampleRow, activeTab, activeCols, featureEngineering, excludedCols]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
      {/* 1. Mode Selector Pills */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
            {isAr ? 'أسلوب إدخال البيانات للنموذج:' : 'Input Mode Strategy:'}
          </span>
          <TermTooltip termKey="column_mapping" size="sm" hideLabel={true} />
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
            gap: '0.45rem',
            background: 'var(--bg-secondary)',
            padding: '0.3rem',
            borderRadius: 'var(--radius-sm, 6px)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          {[
            {
              id: 'single',
              titleAr: 'عمود واحد (افتراضي)',
              titleEn: 'Single Column (Standard)',
              descAr: 'نص مباشر كتعليق أو مقال',
              descEn: 'Single text field',
            },
            {
              id: 'multiple',
              titleAr: 'دمج عدة أعمدة',
              titleEn: 'Multiple Columns',
              descAr: 'دمج نصوص من أكثر من عمود',
              descEn: 'Combine multi fields',
            },
            {
              id: 'selection',
              titleAr: 'تصفية واستبعاد أعمدة',
              titleEn: 'Prune Columns',
              descAr: 'إلغاء الأعمدة غير المفيدة',
              descEn: 'Exclude noise columns',
            },
            {
              id: 'engineering',
              titleAr: 'تنسيق متقدم',
              titleEn: 'Custom Formatting',
              descAr: 'تنسيق مخصص وفواصل',
              descEn: 'Advanced templates',
            },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => handleTabChange(tab.id)}
                style={{
                  background: isActive ? 'var(--accent-teal)' : 'transparent',
                  color: isActive ? '#ffffff' : 'var(--text-primary)',
                  border: 'none',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  textAlign: isAr ? 'right' : 'left',
                  transition: 'all 0.15s ease',
                  boxShadow: isActive ? '0 2px 8px var(--accent-teal-glow)' : 'none',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.15rem',
                }}
              >
                <span style={{ fontSize: '0.8rem', fontWeight: isActive ? 800 : 700 }}>
                  {isAr ? tab.titleAr : tab.titleEn}
                </span>
                <span
                  style={{
                    fontSize: '0.68rem',
                    color: isActive ? 'rgba(255, 255, 255, 0.85)' : 'var(--text-muted)',
                  }}
                >
                  {isAr ? tab.descAr : tab.descEn}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Clear Configuration Form */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm, 6px)',
          padding: '1.1rem',
        }}
      >
        {/* Mode 1: Single Feature */}
        {activeTab === 'single' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem' }}>
            <div>
              <label style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block', marginBottom: '0.35rem' }}>
                {isAr ? '1. نص المدخل (ما سيقرأه النموذج):' : '1. Input Text (What Model Reads):'}
              </label>
              <select
                value={activeCols[0] || datasetCols[0] || ''}
                onChange={(e) => {
                  const col = e.target.value;
                  onChangeConfig({
                    feature_mode: 'single',
                    text_column: col,
                    feature_columns: [col],
                  });
                }}
                className="form-select"
                style={{
                  width: '100%',
                  fontSize: '0.88rem',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '6px',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-subtle)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {datasetCols.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem', display: 'block' }}>
                {isAr ? 'العمود الذي يحتوي على النص المراد تحليله وفهمه' : 'Column containing the primary raw text'}
              </span>
            </div>

            <div>
              <label style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block', marginBottom: '0.35rem' }}>
                {isAr ? '2. المخرج المستهدف (ما سيتوقعه النموذج):' : '2. Target Output (What Model Predicts):'}
              </label>
              <select
                value={selectedLabelCol}
                onChange={(e) => {
                  onChangeConfig({
                    label_column: e.target.value,
                  });
                }}
                className="form-select"
                style={{
                  width: '100%',
                  fontSize: '0.88rem',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '6px',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-subtle)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {datasetCols.map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.3rem', display: 'block' }}>
                {isAr ? 'التصنيف أو القيمة التي سيتعلم النموذج استنتاجها' : 'Target classification or predicted outcome'}
              </span>
            </div>
          </div>
        )}

        {/* Mode 2: Multi Features */}
        {activeTab === 'multiple' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block', marginBottom: '0.45rem' }}>
                {isAr ? 'انقر لاختيار الأعمدة المطلوب دمجها في مدخل النموذج:' : 'Select Columns to Combine:'}
              </label>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                {datasetCols.map((col) => {
                  const isSelected = activeCols.includes(col);
                  return (
                    <button
                      key={col}
                      type="button"
                      onClick={() => toggleColumnSelection(col)}
                      style={{
                        background: isSelected ? 'var(--accent-teal)' : 'var(--bg-elevated)',
                        color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                        border: isSelected ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                        padding: '0.35rem 0.7rem',
                        borderRadius: '6px',
                        fontSize: '0.78rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        fontFamily: 'var(--font-mono)',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {isSelected ? <Check size={13} /> : <Plus size={13} />}
                      <span>{col}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                {isAr ? 'الفاصل بين الأعمدة:' : 'Delimiter:'}
              </span>
              <div style={{ display: 'flex', gap: '0.35rem' }}>
                {[
                  { label: '[SEP]', val: ' [SEP] ' },
                  { label: '· نقطة', val: ' · ' },
                  { label: '| خط', val: ' | ' },
                  { label: 'فاصلة', val: ', ' },
                ].map((del) => (
                  <button
                    key={del.val}
                    type="button"
                    onClick={() => handleDelimiterChange(del.val)}
                    style={{
                      background: (featureEngineering?.delimiter || ' [SEP] ') === del.val ? 'var(--accent-teal)' : 'var(--bg-elevated)',
                      color: (featureEngineering?.delimiter || ' [SEP] ') === del.val ? '#ffffff' : 'var(--text-secondary)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '4px',
                      padding: '0.2rem 0.55rem',
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                    }}
                  >
                    {del.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Mode 3: Selection */}
        {activeTab === 'selection' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <label style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block' }}>
              {isAr ? 'انقر على الأعمدة غير المفيدة لاستبعادها من التدريب:' : 'Exclude Uninformative Columns:'}
            </label>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
              {datasetCols.map((col) => {
                const isExcluded = excludedCols.includes(col);
                return (
                  <button
                    key={col}
                    type="button"
                    onClick={() => toggleNoiseExclusion(col)}
                    style={{
                      background: isExcluded ? 'rgba(239, 68, 68, 0.12)' : 'var(--bg-elevated)',
                      border: isExcluded ? '1px solid var(--accent-rose, #ef4444)' : '1px solid var(--border-subtle)',
                      color: isExcluded ? 'var(--accent-rose, #ef4444)' : 'var(--text-primary)',
                      padding: '0.35rem 0.7rem',
                      borderRadius: '6px',
                      fontSize: '0.78rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                      {isExcluded ? <X size={13} /> : <Check size={13} />}
                      <span>{isExcluded ? (isAr ? 'مُستبعد' : 'Excluded') : (isAr ? 'مُضمّن' : 'Included')}</span>
                    </span>
                    <span>{col}</span>
                  </button>
                );
              })}
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              {isAr ? 'يساعد استبعاد الأعمدة العشوائية (كالأرقام التسلسلية) على تحسين دقة النموذج' : 'Excluding random identifiers helps improve accuracy'}
            </span>
          </div>
        )}

        {/* Mode 4: Engineering */}
        {activeTab === 'engineering' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <label style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', display: 'block' }}>
              {isAr ? 'قالب تنسيق المدخلات المخصص:' : 'Custom Input Template:'}
            </label>

            <input
              type="text"
              dir="ltr"
              value={featureEngineering?.template || activeCols.map((c) => `{${c}}`).join(featureEngineering?.delimiter || ' [SEP] ')}
              onChange={(e) => {
                onChangeConfig({
                  feature_engineering: {
                    ...(featureEngineering || {}),
                    strategy: 'custom_template',
                    template: e.target.value,
                  },
                });
              }}
              style={{
                width: '100%',
                padding: '0.55rem 0.75rem',
                borderRadius: '6px',
                border: '1px solid var(--border-subtle)',
                background: 'var(--bg-elevated)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.84rem',
              }}
              placeholder="{col1} [SEP] {col2}"
            />
          </div>
        )}
      </div>

      {/* 3. Sleek Live Pipeline Preview */}
      {sampleRow && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm, 6px)',
            padding: '0.85rem 1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
              {isAr ? 'المعاينة الحية لمدخلات ومخرجات النموذج:' : 'Live Input / Output Preview:'}
            </span>
            <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
              {activeTab === 'single' ? (isAr ? 'عمود منفرد' : 'Single Column') : `${activeCols.length} ${isAr ? 'أعمدة' : 'Columns'}`}
            </span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr auto 1fr',
              gap: '0.75rem',
              alignItems: 'center',
              background: 'var(--bg-elevated)',
              padding: '0.8rem 1rem',
              borderRadius: '6px',
              border: '1px solid var(--border-subtle)',
            }}
          >
            {/* Input Preview */}
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 700, marginBottom: '0.2rem' }}>
                {isAr ? 'المدخل الفعلي للنموذج:' : 'Model Input Text:'}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', lineHeight: 1.45, fontStyle: 'italic', wordBreak: 'break-word' }}>
                «{previewCompositeText ? (previewCompositeText.length > 80 ? previewCompositeText.slice(0, 80) + '...' : previewCompositeText) : (isAr ? 'نص العينة' : 'Sample')}»
              </div>
            </div>

            {/* Inference Direction */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0 0.4rem' }}>
              <span style={{ fontSize: '0.65rem', fontWeight: 800, color: 'var(--accent-teal)', marginBottom: '0.1rem' }}>
                {isAr ? 'استنتاج' : 'Infer'}
              </span>
              <span style={{ color: 'var(--accent-teal)', display: 'inline-flex', alignItems: 'center' }}>
                <ArrowRight size={18} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
              </span>
            </div>

            {/* Target Output Preview */}
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 700, marginBottom: '0.2rem' }}>
                {isAr ? `النتيجة المطلوبة (${selectedLabelCol || 'الهدف'}):` : `Target (${selectedLabelCol || 'Label'}):`}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span className="badge badge-teal" style={{ fontSize: '0.76rem', fontWeight: 800, padding: '0.2rem 0.55rem' }}>
                  {sampleRow.label || (sampleRow.rawRow && sampleRow.rawRow[selectedLabelCol]) || (isAr ? 'الفئة المستهدفة' : 'Target')}
                </span>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                  ({isAr ? `${datasetAnalysis?.uniqueClasses?.length || 2} فئات` : `${datasetAnalysis?.uniqueClasses?.length || 2} classes`})
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
