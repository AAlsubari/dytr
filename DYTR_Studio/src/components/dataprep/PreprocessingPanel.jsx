// ==============================================================================
// DyTrStudio Data Mining - Phase 3: Preprocessing & Cleaning
// Missing Values, Deduplication, Type Alignment, IQR Outliers, Scaling & Arabic Normalization
// Zero Emojis Policy - High Contrast Dual Theme - Interactive Before/After Diff Preview
// ==============================================================================

import React, { useState, useMemo } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import {
  profileDataset,
  handleMissingValues,
  removeDuplicates,
  handleOutliers,
  normalizeColumn,
  encodeCategorical,
  cleanArabicText,
} from '../../lib/data/processor';
import TermTooltip from '../common/TermTooltip';
import OptionTipBox from '../common/OptionTipBox';
import { Check, X, ArrowRight } from 'lucide-react';

export default function PreprocessingPanel() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';
  const { currentDataset, applyOperation } = useDatasetStore();

  const [activeTool, setActiveTool] = useState('missing'); // 'missing' | 'dedup' | 'outlier' | 'normalize' | 'encode' | 'arabic'
  const [selectedCol, setSelectedCol] = useState('');
  const [imputeStrategy, setImputeStrategy] = useState('median');
  const [customFill, setCustomFill] = useState('0');
  const [outlierAction, setOutlierAction] = useState('clip');
  const [normMethod, setNormMethod] = useState('minmax');
  const [encodeMethod, setEncodeMethod] = useState('onehot');
  const [actionToast, setActionToast] = useState(null);

  const triggerToast = (msg_ar, msg_en) => {
    setActionToast(isAr ? msg_ar : msg_en);
    setTimeout(() => setActionToast(null), 4500);
  };

  const profile = profileDataset(currentDataset);
  const columns = profile.columns;

  // Set default selected column if unset or invalid
  React.useEffect(() => {
    if (columns.length > 0) {
      const exists = columns.some((c) => c.key === selectedCol);
      if (!selectedCol || !exists) {
        setSelectedCol(columns[0].key);
      }
    }
  }, [columns, selectedCol]);

  // Handler for Missing Values
  const handleApplyImputation = () => {
    if (!selectedCol) return;
    applyOperation({
      type: 'missing_fill',
      column: selectedCol,
      strategy: imputeStrategy,
      value: imputeStrategy === 'custom' ? customFill : imputeStrategy,
      title_ar: `معالجة القيم المفقودة في [${selectedCol}] باستراتيجية: ${imputeStrategy}`,
      title_en: `Imputed missing values in [${selectedCol}] using ${imputeStrategy}`,
      transformFn: (data) => handleMissingValues(data, selectedCol, imputeStrategy, customFill),
    });
    triggerToast(
      `تم تطبيق سد القيم المفقودة بنجاح على العمود [${selectedCol}] باستراتيجية: ${imputeStrategy}.`,
      `Successfully imputed missing values in column [${selectedCol}] using ${imputeStrategy}.`
    );
  };

  // Handler for Deduplication
  const handleApplyDedup = () => {
    applyOperation({
      type: 'dedup',
      title_ar: 'إزالة كافة الصفوف المكررة من مجموعة البيانات',
      title_en: 'Removed duplicate rows from corpus',
      transformFn: (data) => removeDuplicates(data),
    });
    triggerToast(
      'تم فحص وإزالة الصفوف المكررة بنجاح من جدول البيانات الحالي.',
      'Successfully scanned and removed duplicate rows from current dataset.'
    );
  };

  // Handler for Outliers
  const handleApplyOutliers = () => {
    if (!selectedCol) return;
    applyOperation({
      type: 'outlier_iqr',
      column: selectedCol,
      action: outlierAction,
      value: outlierAction,
      title_ar: `معالجة القيم الشاذة في [${selectedCol}] بنطاق IQR (${outlierAction})`,
      title_en: `Handled outliers in [${selectedCol}] using IQR (${outlierAction})`,
      transformFn: (data) => handleOutliers(data, selectedCol, outlierAction),
    });
    triggerToast(
      `تم تطبيق معالجة القيم الشاذة بنجاح على العمود [${selectedCol}] بأسلوب IQR: ${outlierAction}.`,
      `Successfully processed outliers in column [${selectedCol}] using IQR: ${outlierAction}.`
    );
  };

  // Handler for Normalization
  const handleApplyNormalization = () => {
    if (!selectedCol) return;
    applyOperation({
      type: normMethod === 'minmax' ? 'normalize_minmax' : normMethod === 'robust' ? 'normalize_robust' : 'normalize_zscore',
      normMethod,
      column: selectedCol,
      title_ar: `تطبيع وتقييس العمود [${selectedCol}] عبر: ${normMethod}`,
      title_en: `Normalized [${selectedCol}] using ${normMethod}`,
      transformFn: (data) => normalizeColumn(data, selectedCol, normMethod),
    });
    triggerToast(
      `تم تطبيق تقييس وتطبيع المقياس بنجاح على العمود [${selectedCol}] عبر طريقة ${normMethod}.`,
      `Successfully normalized feature column [${selectedCol}] using ${normMethod}.`
    );
  };

  // Handler for Categorical Encoding
  const handleApplyEncoding = () => {
    if (!selectedCol) return;
    applyOperation({
      type: encodeMethod === 'onehot' ? 'onehot' : 'label_encode',
      encodeMethod,
      column: selectedCol,
      title_ar: `ترميز المتغير الفئوي [${selectedCol}] عبر: ${encodeMethod}`,
      title_en: `Encoded categorical [${selectedCol}] via ${encodeMethod}`,
      transformFn: (data) => encodeCategorical(data, selectedCol, encodeMethod),
    });
    triggerToast(
      `تم تطبيق الترميز بنجاح على العمود الفئوي [${selectedCol}] عبر طريقة ${encodeMethod}.`,
      `Successfully encoded categorical column [${selectedCol}] via ${encodeMethod}.`
    );
  };

  // Handler for Arabic Text Normalization
  const handleApplyArabicClean = () => {
    if (!selectedCol) return;
    applyOperation({
      type: 'arabic_clean',
      column: selectedCol,
      title_ar: `تطبيع وتجريد النصوص العربية في العمود [${selectedCol}]`,
      title_en: `Normalized Arabic text tokens in [${selectedCol}]`,
      transformFn: (data) => cleanArabicText(data, selectedCol, { removeDiacritics: true, normalizeLetters: true, stripExtraSpaces: true }),
    });
    triggerToast(
      `تم تطبيع وتجريد النصوص العربية بنجاح في العمود [${selectedCol}].`,
      `Successfully normalized Arabic text tokens in column [${selectedCol}].`
    );
  };

  // Live Simulated Diff Preview
  const previewDiff = useMemo(() => {
    if (!currentDataset || currentDataset.length === 0 || !selectedCol) return { before: [], after: [] };
    const sampleRows = currentDataset.slice(0, 5);

    let transformedSample = sampleRows;
    if (activeTool === 'missing') {
      transformedSample = handleMissingValues(sampleRows, selectedCol, imputeStrategy, customFill);
    } else if (activeTool === 'outlier') {
      transformedSample = handleOutliers(sampleRows, selectedCol, outlierAction);
    } else if (activeTool === 'normalize') {
      transformedSample = normalizeColumn(sampleRows, selectedCol, normMethod);
    } else if (activeTool === 'encode') {
      transformedSample = encodeCategorical(currentDataset, selectedCol, encodeMethod).slice(0, 5);
    } else if (activeTool === 'arabic') {
      transformedSample = cleanArabicText(sampleRows, selectedCol, { removeDiacritics: true, normalizeLetters: true, stripExtraSpaces: true });
    }

    return {
      before: sampleRows,
      after: transformedSample,
    };
  }, [currentDataset, selectedCol, activeTool, imputeStrategy, customFill, outlierAction, normMethod, encodeMethod]);

  const tools = [
    { id: 'missing', code: '03.1', title: isAr ? 'القيم المفقودة' : 'Missing Values', badge: isAr ? 'ليتل وروبين' : 'Imputation' },
    { id: 'dedup', code: '03.2', title: isAr ? 'إزالة التكرار' : 'Deduplication', badge: isAr ? 'تنقية تامة' : 'Deduplication' },
    { id: 'outlier', code: '03.3', title: isAr ? 'كشف الشواذ (IQR)' : 'Outlier Fences', badge: 'Tukey' },
    { id: 'normalize', code: '03.4', title: isAr ? 'التطبيع والتقييس' : 'Normalization', badge: 'Scaling' },
    { id: 'encode', code: '03.5', title: isAr ? 'الترميز الفئوي' : 'Encoding', badge: 'One-Hot' },
    { id: 'arabic', code: '03.6', title: isAr ? 'تنقية النصوص العربية' : 'Arabic NLP Clean', badge: 'NLP' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Action Toast Banner */}
      {actionToast && (
        <div
          style={{
            padding: '0.85rem 1.25rem',
            borderRadius: 'var(--radius-md, 8px)',
            background: 'var(--accent-teal-glow)',
            border: '1px solid var(--accent-teal)',
            color: 'var(--accent-teal)',
            fontSize: '0.88rem',
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: 'var(--shadow-sm)',
            animation: 'fadeIn 0.2s ease',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span className="badge badge-teal" style={{ fontSize: '0.72rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <Check size={13} />
              <span>{isAr ? 'تم تطبيق المعالجة بنجاح' : 'Processed Successfully'}</span>
            </span>
            <span>{actionToast}</span>
          </div>
          <button
            onClick={() => setActionToast(null)}
            style={{ background: 'none', border: 'none', color: 'var(--accent-teal)', cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Introduction Header */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.25rem 1.5rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
          <span className="badge badge-teal">{isAr ? 'المرحلة 3 من 5' : 'Phase 3 of 5'}</span>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            <TermTooltip termKey="mcar" customLabel={isAr ? 'معالجة العيوب الإحصائية' : 'Statistical Data Cleansing'} />
          </span>
        </div>
        <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {isAr
            ? 'المعالجة الأولية هي مرحلة تصحيح الشوائب والعيوب: سد الفراغات بالمتوسط أو الوسيط، عزل القيم الشاذة بنطاق توكي الربيعي، وتطبيع المقاييس الرقمية لضمان استقرار انحدار التدرج في تدريب المحولات.'
            : 'Preprocessing addresses empirical data defects: robust imputation, Tukey outlier isolation, feature normalization, and Arabic NLP token standardization to stabilize neural loss convergence.'}
        </p>
      </div>

      {/* Tool Selector Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: '0.75rem',
        }}
      >
        {tools.map((tool) => {
          const isActive = activeTool === tool.id;
          return (
            <button
              key={tool.id}
              onClick={() => setActiveTool(tool.id)}
              style={{
                background: isActive ? 'var(--accent-teal-glow)' : 'var(--bg-secondary)',
                border: isActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md, 8px)',
                padding: '0.85rem 1rem',
                textAlign: isAr ? 'right' : 'left',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: '700' }}>
                  [{tool.code}]
                </span>
                <span className={`badge ${isActive ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.65rem' }}>
                  {tool.badge}
                </span>
              </div>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: isActive ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                {tool.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Tool Configuration & Live Diff Box */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* Tool 1: Missing Values */}
        {activeTool === 'missing' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {isAr ? 'معالجة وسد القيم المفقودة (Missing Value Imputation)' : 'Missing Value Imputation'}
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'اختر الإستراتيجية الأنسب حسب توزيع المتغير ونسبة الفراغات' : 'Choose robust imputation strategies based on feature distribution'}
                </span>
              </div>
              <TermTooltip termKey="mcar" customLabel={isAr ? 'نظرية ليتل وروبين (MCAR/MAR)' : 'Little & Rubin Taxonomy'} />
            </div>

            {/* Pedagogical Explanation Card */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', lineHeight: 1.55 }}>
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'إرشاد علاجي:' : 'Guidance:'}</strong>{' '}
              {isAr
                ? 'إذا كان العمود يحتوي قيماً شاذة أو توزيعاً ملتوياً، استخدم الوسيط (Median) لأنه أكثر متانة. للمتغيرات الفئوية والنصية استخدم المنوال (Mode). إذا كانت نسبة المفقود قليلة جداً (<1%) يمكنك حذف الصفوف بأمان.'
                : 'For skewed numeric features with outliers, prefer Median over Mean. For categorical features, use Mode. For minimal missingness (<1%), listwise deletion is safe.'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'اختر العمود المراد معالجته:' : 'Select Target Column:'}
                </label>
                <select
                  value={selectedCol}
                  onChange={(e) => setSelectedCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c.key} value={c.key}>
                      {c.key} ({c.missingPercent.toFixed(1)}% {isAr ? 'مفقود' : 'missing'})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'إستراتيجية المعالجة:' : 'Imputation Strategy:'}
                </label>
                <select
                  value={imputeStrategy}
                  onChange={(e) => setImputeStrategy(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="median">{isAr ? 'التعويض بالوسيط (Median - مقاوم للشواذ)' : 'Impute Median (Robust)'}</option>
                  <option value="mean">{isAr ? 'التعويض بالمتوسط الحسابي (Mean)' : 'Impute Mean'}</option>
                  <option value="mode">{isAr ? 'التعويض بالمنوال (Mode - للفئات والنصوص)' : 'Impute Mode (Most frequent)'}</option>
                  <option value="drop_rows">{isAr ? 'حذف الصفوف التي تحوي فراغاً (Drop Rows)' : 'Listwise Deletion (Drop Rows)'}</option>
                  <option value="drop_column">{isAr ? 'حذف العمود بالكامل' : 'Drop Entire Column'}</option>
                  <option value="custom">{isAr ? 'ملء بقيمة ثابتة مخصصة' : 'Constant Value'}</option>
                </select>
              </div>
            </div>

            {imputeStrategy === 'custom' && (
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'القيمة الثابتة للتعويض:' : 'Constant Replacement Value:'}
                </label>
                <input
                  type="text"
                  value={customFill}
                  onChange={(e) => setCustomFill(e.target.value)}
                  style={{
                    width: '100%',
                    maxWidth: '300px',
                    padding: '0.5rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                />
              </div>
            )}

            <button onClick={handleApplyImputation} className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              <span>{isAr ? 'تطبيق المعالجة وحفظ السجل' : 'Apply Imputation & Record'}</span>
              <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        )}

        {/* Tool 2: Deduplication */}
        {activeTool === 'dedup' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'إزالة السجلات المكررة (Corpus Deduplication)' : 'Corpus Deduplication'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? `البيانات الحالية تحتوي على ${profile.duplicateCount} صفاً مكرراً. إزالة التكرار ضرورية لأن وجود نسخ مكررة يجعل النموذج ينحاز بشدة لتلك العينات أثناء تحديث الأوزان والتدرجات.`
                  : `Current dataset contains ${profile.duplicateCount} duplicate records. Deduplication prevents synthetic gradient over-weighting and memorization artifacts.`}
              </p>
            </div>

            <button
              onClick={handleApplyDedup}
              disabled={profile.duplicateCount === 0}
              className="btn btn-primary"
              style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800 }}
            >
              {isAr ? `حذف كافة الصفوف المكررة (${profile.duplicateCount} صف)` : `Remove All Duplicates (${profile.duplicateCount} rows)`}
            </button>
          </div>
        )}

        {/* Tool 3: Outlier Handling */}
        {activeTool === 'outlier' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {isAr ? 'كشف ومعالجة القيم الشاذة (Tukey IQR Fences)' : 'Tukey IQR Outlier Fences'}
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'عزل القيم الخارجة عن نطاق [Q1 - 1.5×IQR , Q3 + 1.5×IQR]' : 'Isolate values outside Tukey boundary fences'}
                </span>
              </div>
              <TermTooltip termKey="iqr" customLabel={isAr ? 'معيار توكي للربيعات' : 'Tukey IQR Standard'} />
            </div>

            {/* Pedagogical Explanation */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', lineHeight: 1.55 }}>
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'ما هو الاقتطاع (Winsorize / Clip)؟' : 'What is Clipping?'}</strong>{' '}
              {isAr
                ? 'بدلاً من حذف السجل الشاذ بالكامل وفقدان معلومات الأعمدة الأخرى، يقوم الاقتطاع بضبط القيمة المتطرفة عند أقرب حد طبيعي مسموح به (Upper/Lower Fence)، مما يحافظ على استقرار تدريب النموذج دون خسارة صفوف.'
                : 'Clipping caps extreme anomalies to the boundary fences rather than discarding the entire row, preserving sample volume while eliminating gradient explosion.'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'اختر المتغير الرقمي:' : 'Numeric Feature:'}
                </label>
                <select
                  value={selectedCol}
                  onChange={(e) => setSelectedCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns
                    .filter((c) => c.type === 'numeric')
                    .map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.key} (IQR: {c.stats?.iqr || 0})
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'الإجراء المطلوب عند رصد شذوذ:' : 'Outlier Action:'}
                </label>
                <select
                  value={outlierAction}
                  onChange={(e) => setOutlierAction(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="clip">{isAr ? 'اقتطاع عند الحدود (Winsorize / Clip - موصى به)' : 'Clip at Fences (Winsorize - Recommended)'}</option>
                  <option value="drop_rows">{isAr ? 'حذف السجلات الشاذة بالكامل' : 'Drop Outlier Rows'}</option>
                  <option value="null">{isAr ? 'تحويل الشواذ إلى قيم فارغة (NULL)' : 'Set to NULL'}</option>
                </select>

                <OptionTipBox optionKey={`outlier_action_${outlierAction}`} isAr={isAr} />
              </div>
            </div>

            <button onClick={handleApplyOutliers} className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              <span>{isAr ? 'تطبيق معالجة الشواذ' : 'Process Outliers'}</span>
              <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        )}

        {/* Tool 4: Normalization */}
        {activeTool === 'normalize' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {isAr ? 'التطبيع والتقييس الإحصائي (Feature Scaling)' : 'Feature Scaling & Standardization'}
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'توحيد مقاييس الأرقام لمنع طغيان المتغيرات الكبيرة على تدرجات التعلم' : 'Scale numeric continuous variables to ensure equal gradient updates'}
                </span>
              </div>
              <TermTooltip termKey="normalization" customLabel={isAr ? 'استقرار التدرجات' : 'Gradient Stability'} />
            </div>

            {/* Pedagogical Explanation */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', lineHeight: 1.55 }}>
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'الفرق بين Min-Max و Z-Score:' : 'Min-Max vs Z-Score:'}</strong>{' '}
              {isAr
                ? 'يحصر Min-Max القيم بدقة بين [0 و 1] وهو ممتاز للشبكات العصبية. بينما Z-Score يحول المتوسط إلى 0 والانحراف إلى 1، وهو الأفضل للبيانات التي تتبع توزيعاً طبيعياً.'
                : 'Min-Max compresses continuous ranges into [0, 1]. Z-Score shifts mean to 0 and variance to 1, preserving negative deviations.'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'العمود الرقمي:' : 'Numeric Column:'}
                </label>
                <select
                  value={selectedCol}
                  onChange={(e) => setSelectedCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns
                    .filter((c) => c.type === 'numeric')
                    .map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.key}
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'معادلة التطبيع:' : 'Normalization Method:'}
                </label>
                <select
                  value={normMethod}
                  onChange={(e) => setNormMethod(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="minmax">{isAr ? 'Min-Max Scaling [0, 1] (موصى به للشبكات العصبية)' : 'Min-Max Scaling [0, 1]'}</option>
                  <option value="zscore">{isAr ? 'Z-Score Standardization (μ=0, σ=1)' : 'Z-Score Standardization'}</option>
                  <option value="robust">{isAr ? 'Robust Scaling (باستخدام Median و IQR)' : 'Robust Scaling (IQR)'}</option>
                </select>

                <OptionTipBox optionKey={`scale_${normMethod}`} isAr={isAr} />
              </div>
            </div>

            <button onClick={handleApplyNormalization} className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              <span>{isAr ? 'تطبيق التقييس الرياضي' : 'Apply Scaling'}</span>
              <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        )}

        {/* Tool 5: Categorical Encoding */}
        {activeTool === 'encode' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {isAr ? 'الترميز الفئوي (Categorical Encoding)' : 'Categorical Encoding'}
                </h3>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'تحويل الفئات النصية إلى متجهات رقمية تفهمها المصفوفات العصبية' : 'Convert text classes into numerical binary vectors'}
                </span>
              </div>
              <TermTooltip termKey="one_hot_encoding" customLabel={isAr ? 'الترميز أحادي الساخن' : 'One-Hot Orthogonality'} />
            </div>

            {/* Pedagogical Explanation */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', lineHeight: 1.55 }}>
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? 'لماذا نستخدم One-Hot بدلاً من مجرد أرقام 1,2,3؟' : 'Why One-Hot?'}</strong>{' '}
              {isAr
                ? 'إذا رقمت الفئات (أحمر=1, أخضر=2, أزرق=3)، سيعتقد النموذج أن "أزرق" أكبر من "أحمر" أو أنه متوسط أخضر وأحمر! ترميز One-Hot ينشئ عموداً ثنائياً منفصلاً لكل فئة (0 أو 1) ليضمن التعامد والاستقلالية التامة.'
                : 'Label encoding creates unintended arithmetic hierarchy. One-Hot encoding guarantees mathematical orthogonality across nominal classes.'}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'المتغير الفئوي المراد ترميزه:' : 'Target Categorical Column:'}
                </label>
                <select
                  value={selectedCol}
                  onChange={(e) => setSelectedCol(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  {columns.map((c) => (
                    <option key={c.key} value={c.key}>
                      {c.key} ({c.uniqueCount} {isAr ? 'فئات' : 'levels'})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                  {isAr ? 'نوع الترميز:' : 'Encoding Scheme:'}
                </label>
                <select
                  value={encodeMethod}
                  onChange={(e) => setEncodeMethod(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.75rem',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: 'var(--radius-sm, 6px)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                  }}
                >
                  <option value="onehot">{isAr ? 'أحادي الساخن (One-Hot Binary Vectors - موصى به)' : 'One-Hot Encoding (Recommended)'}</option>
                  <option value="label">{isAr ? 'الترميز العددي الترتيبي (Ordinal / Label)' : 'Label / Ordinal Encoding'}</option>
                </select>

                <OptionTipBox optionKey={`encode_${encodeMethod === 'onehot' ? 'onehot' : 'label'}`} isAr={isAr} />
              </div>
            </div>

            <button onClick={handleApplyEncoding} className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              <span>{isAr ? 'تطبيق الترميز الهندسي' : 'Apply Encoding'}</span>
              <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        )}

        {/* Tool 6: Arabic NLP Cleaning */}
        {activeTool === 'arabic' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                {isAr ? 'تنقية وتطبيع النصوص العربية (Arabic Text Normalization)' : 'Arabic Text Normalization'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                {isAr
                  ? 'إزالة التشكيل والحركات، حذف التطويل (المدة)، توحيد أشكال الهمزات (إ/أ/آ إلى ا)، والياء والألف المقصورة (ى إلى ي) والتاء المربوطة (ة إلى ه) لمنع تشتت قواميس التوكن في المحول.'
                  : 'Strip Arabic diacritics, remove tatweel, unify morphological letter forms, and standardize spacing for optimal subword tokenization.'}
              </p>
            </div>

            <div style={{ maxWidth: '320px' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                {isAr ? 'عمود النصوص العربية:' : 'Arabic Text Column:'}
              </label>
              <select
                value={selectedCol}
                onChange={(e) => setSelectedCol(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-medium)',
                  borderRadius: 'var(--radius-sm, 6px)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                }}
              >
                {columns.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.key}
                  </option>
                ))}
              </select>
            </div>

            <button onClick={handleApplyArabicClean} className="btn btn-primary" style={{ alignSelf: 'flex-start', padding: '0.65rem 1.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
              <span>{isAr ? 'تنقية وتوحيد النصوص العربية' : 'Normalize Arabic Tokens'}</span>
              <ArrowRight size={15} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        )}

        {/* Live Side-by-Side Before / After Diff Preview (Compact & Space-Efficient) */}
        {selectedCol && previewDiff.before.length > 0 && (
          <div
            style={{
              background: 'var(--bg-tertiary)',
              borderRadius: 'var(--radius-md, 8px)',
              border: '1px solid var(--border-medium)',
              padding: '0.75rem 1rem',
              marginTop: '0.5rem',
            }}
          >
            <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              {isAr ? `معاينة فورية للأثر على أول 5 عينات في عمود [${selectedCol}]:` : `Interactive Live Impact on first 5 samples [${selectedCol}]:`}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '0.65rem' }}>
              {/* Before Box */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.55rem 0.65rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)', minWidth: 0, overflow: 'hidden' }}>
                <span className="badge badge-neutral" style={{ marginBottom: '0.35rem', display: 'inline-block', fontSize: '0.7rem' }}>
                  {isAr ? 'الحالة الحالية (قبل)' : 'Current (Before)'}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.74rem', maxHeight: '140px', overflowY: 'auto' }}>
                  {previewDiff.before.map((r, i) => {
                    const origVal = r[`${selectedCol}_original`] !== undefined ? r[`${selectedCol}_original`] : r[selectedCol];
                    const strVal = origVal !== null && origVal !== undefined ? String(origVal) : null;
                    return (
                      <div
                        key={i}
                        title={strVal || 'NULL'}
                        style={{
                          padding: '0.25rem 0.45rem',
                          background: 'var(--bg-tertiary)',
                          borderRadius: '4px',
                          lineHeight: '1.35',
                          border: '1px solid var(--border-subtle)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          wordBreak: 'break-word',
                        }}
                      >
                        {strVal !== null ? strVal : <span style={{ color: 'var(--accent-rose)', fontWeight: 700 }}>NULL</span>}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* After Box */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.55rem 0.65rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--accent-teal)', minWidth: 0, overflow: 'hidden' }}>
                <span className="badge badge-teal" style={{ marginBottom: '0.35rem', display: 'inline-block', fontSize: '0.7rem' }}>
                  {isAr ? 'النتيجة المتوقعة (بعد)' : 'Simulated (After)'}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', fontSize: '0.74rem', maxHeight: '140px', overflowY: 'auto' }}>
                  {previewDiff.after.map((r, i) => {
                    let strVal = '';
                    if (activeTool === 'encode') {
                      if (encodeMethod === 'label') {
                        const code = r[`${selectedCol}_code`];
                        strVal = String(code !== undefined ? code : (r[selectedCol] ?? '0'));
                      } else {
                        // One-Hot Encoding vector
                        strVal = String(r[selectedCol] ?? '');
                      }
                    } else {
                      const val = r[selectedCol];
                      strVal = val !== null && val !== undefined ? String(val) : 'NULL';
                    }
                    return (
                      <div
                        key={i}
                        title={strVal}
                        style={{
                          padding: '0.25rem 0.45rem',
                          background: 'var(--accent-teal-glow)',
                          borderRadius: '4px',
                          color: 'var(--accent-teal)',
                          fontWeight: 600,
                          lineHeight: '1.35',
                          border: '1px solid var(--border-medium)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          wordBreak: 'break-word',
                        }}
                      >
                        {strVal}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
