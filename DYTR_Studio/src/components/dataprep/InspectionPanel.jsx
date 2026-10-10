// ==============================================================================
// DyTrStudio Data Mining - Phase 2: Initial Inspection & EDA
// Five-Number Summary, Composite Quality Gauge, Column Profiler & Diagnostics
// Zero Emojis Policy - High Contrast Dual Theme - Pedagogical Clarity
// ==============================================================================

import React, { useState } from 'react';
import { useDatasetStore } from '../../store/useDatasetStore';
import { useStudioStore } from '../../store/studioStore';
import { profileDataset, removeDuplicates } from '../../lib/data/processor';
import TermTooltip from '../common/TermTooltip';
import { ArrowRight } from 'lucide-react';

export default function InspectionPanel() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';
  const { currentDataset, applyOperation, setActivePhase } = useDatasetStore();

  const [expandedCol, setExpandedCol] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 8;
  const [colPage, setColPage] = useState(0);
  const colPageSize = 5;

  // Quality Issues pagination & category filtering
  const [selectedIssueCategory, setSelectedIssueCategory] = useState('all');
  const [issuePage, setIssuePage] = useState(0);
  const [issuePageSize, setIssuePageSize] = useState(5);

  const profile = profileDataset(currentDataset);

  // Categorize issues
  const allIssues = profile.issues || [];
  const missingIssues = allIssues.filter((i) => i.category === 'missing');
  const duplicateIssues = allIssues.filter((i) => i.category === 'duplicate');
  const outlierIssues = allIssues.filter((i) => i.category === 'outlier');
  const structureIssues = allIssues.filter((i) => i.category === 'structure');

  const issueCategories = [
    { id: 'all', label_ar: 'جميع الملاحظات', label_en: 'All Diagnostics', count: allIssues.length },
    { id: 'missing', label_ar: 'القيم المفقودة', label_en: 'Missing Values', count: missingIssues.length },
    { id: 'duplicate', label_ar: 'الصفوف المكررة', label_en: 'Duplicates', count: duplicateIssues.length },
    { id: 'outlier', label_ar: 'القيم الشاذة والالتواء', label_en: 'Outliers & Skewness', count: outlierIssues.length },
    { id: 'structure', label_ar: 'هيكل الأعمدة والتباين', label_en: 'Structure & Variance', count: structureIssues.length },
  ];

  const filteredIssues = allIssues.filter((i) => {
    if (selectedIssueCategory === 'all') return true;
    return i.category === selectedIssueCategory;
  });

  const totalIssuePages = Math.ceil(filteredIssues.length / issuePageSize) || 1;
  const safeIssuePage = Math.min(issuePage, Math.max(0, totalIssuePages - 1));
  const pagedIssues = filteredIssues.slice(safeIssuePage * issuePageSize, (safeIssuePage + 1) * issuePageSize);

  const getQualityColor = (score) => {
    if (score >= 75) return 'var(--accent-emerald)';
    if (score >= 45) return 'var(--accent-amber)';
    return 'var(--accent-rose)';
  };

  const getQualityLabel = (score) => {
    if (score >= 75) return isAr ? 'ممتاز (جاهز للتدريب العصبي)' : 'Pristine (Training Ready)';
    if (score >= 45) return isAr ? 'متوسط (يحتاج تنقية ومعالجة)' : 'Moderate (Needs Cleansing)';
    return isAr ? 'حرج (عيوب هيكلية تتطلب تدخلاً)' : 'Critical (Defects Detected)';
  };

  // Filtered rows for preview
  const filteredData = currentDataset.filter((row) => {
    if (!searchTerm) return true;
    return Object.values(row).some((val) =>
      String(val).toLowerCase().includes(searchTerm.toLowerCase())
    );
  });

  const pageData = filteredData.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filteredData.length / pageSize) || 1;

  const totalColPages = Math.ceil((profile.columns?.length || 0) / colPageSize) || 1;
  const safeColPage = Math.min(colPage, Math.max(0, totalColPages - 1));
  const paginatedColumns = profile.columns.slice(safeColPage * colPageSize, (safeColPage + 1) * colPageSize);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Introduction & Pedagogical Header */}
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
          <span className="badge badge-teal">{isAr ? 'المرحلة 2 من 5' : 'Phase 2 of 5'}</span>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            <TermTooltip termKey="eda" customLabel={isAr ? 'منهجية توكي للتحليل الاستكشافي (EDA)' : 'Tukey Exploratory Data Analysis'} />
          </span>
        </div>
        <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {isAr
            ? 'الفحص والاستكشاف (EDA) هو نظرة الطبيب التشخيصية قبل التدخل الجراحي: نفحص سلامة البيانات، نكشف القيم المفقودة، ونلخص التوزيع الإحصائي (الملخص الخماسي) لضمان عدم وجود تشوهات في فضاء الخصائص.'
            : 'Exploratory Data Analysis (EDA) conducts an empirical diagnostic before model training: inspecting completeness, identifying null densities, and computing Tukey five-number summaries.'}
        </p>
      </div>

      {/* Overview Statistics Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '0.75rem',
        }}
      >
        <div className="stat-box">
          <div className="stat-label">{isAr ? 'إجمالي السجلات' : 'Total Rows'}</div>
          <div className="stat-value">{profile.totalRows}</div>
          <div className="stat-desc">{isAr ? 'عينة نشطة' : 'Active Corpus'}</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">{isAr ? 'عدد الأعمدة' : 'Columns'}</div>
          <div className="stat-value" style={{ color: 'var(--accent-teal)' }}>{profile.totalCols}</div>
          <div className="stat-desc">{isAr ? 'أبعاد الخصائص' : 'Feature Dimensions'}</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">{isAr ? 'حجم الذاكرة' : 'Memory'}</div>
          <div className="stat-value" style={{ color: 'var(--accent-cyan)' }}>{profile.memoryKB} KB</div>
          <div className="stat-desc">In-Memory Cache</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">{isAr ? 'الصفوف المكررة' : 'Duplicates'}</div>
          <div
            className="stat-value"
            style={{
              color: profile.duplicateCount > 0 ? 'var(--accent-amber)' : 'var(--accent-emerald)',
            }}
          >
            {profile.duplicateCount}
          </div>
          <div className="stat-desc">
            {((profile.duplicateCount / (profile.totalRows || 1)) * 100).toFixed(1)}% {isAr ? 'من الإجمالي' : 'of total'}
          </div>
        </div>
      </div>

      {/* Composite Quality Gauge Card */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.5rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ flex: 1, minWidth: '260px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span className="badge badge-teal">
              <TermTooltip termKey="eda" customLabel={isAr ? 'مؤشر الجودة الشامل' : 'Composite Quality Index'} />
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              (0 - 100)
            </span>
          </div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0, marginBottom: '0.35rem' }}>
            {getQualityLabel(profile.qualityScore)}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.55 }}>
            {isAr
              ? 'مؤشر علمي مركب يزن 4 محاور: الاكتمال والخلو من الفراغات (40%)، عدم التكرار (30%)، اتساق وتجانس الأنواع (20%)، وحجم العينة (10%).'
              : 'Weighted metric evaluating 4 pillars: matrix completeness (40%), record uniqueness (30%), type consistency (20%), and corpus volume (10%).'}
          </p>
        </div>

        {/* Quality Score Circle */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '105px',
            height: '105px',
            borderRadius: '50%',
            background: 'var(--bg-tertiary)',
            border: `4px solid ${getQualityColor(profile.qualityScore)}`,
            boxShadow: `0 0 16px ${getQualityColor(profile.qualityScore)}33`,
            flexShrink: 0,
          }}
        >
          <span style={{ fontSize: '1.75rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
            {profile.qualityScore}
          </span>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 700 }}>/ 100</span>
        </div>
      </div>

      {/* Quality Issues List - Categorized Tabs & Pagination (Default 5 items) */}
      {allIssues.length > 0 && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-lg, 12px)',
            padding: '1.25rem 1.5rem',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          {/* Header & Badges */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '0.75rem',
              marginBottom: '1rem',
              paddingBottom: '0.75rem',
              borderBottom: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
              <div style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {isAr ? 'الملاحظات والعيوب المرصودة آلياً:' : 'Diagnostic Issues & Anomalies:'}
              </div>
              <span
                style={{
                  fontSize: '0.75rem',
                  padding: '0.2rem 0.6rem',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '12px',
                  fontWeight: 700,
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                {isAr ? `${allIssues.length} عيب مرصود` : `${allIssues.length} detected`}
              </span>
              {allIssues.some((x) => x.severity === 'critical') && (
                <span
                  style={{
                    fontSize: '0.72rem',
                    padding: '0.15rem 0.55rem',
                    background: 'rgba(244, 63, 94, 0.12)',
                    color: 'var(--accent-rose)',
                    borderRadius: '10px',
                    fontWeight: 700,
                    border: '1px solid rgba(244, 63, 94, 0.25)',
                  }}
                >
                  {isAr
                    ? `${allIssues.filter((x) => x.severity === 'critical').length} حرجة`
                    : `${allIssues.filter((x) => x.severity === 'critical').length} critical`}
                </span>
              )}
            </div>

            {/* Page Size Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
              <span>{isAr ? 'عرض في الصفحة:' : 'Items per page:'}</span>
              {[5, 10, 20].map((sz) => (
                <button
                  key={sz}
                  type="button"
                  onClick={() => {
                    setIssuePageSize(sz);
                    setIssuePage(0);
                  }}
                  style={{
                    padding: '0.18rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.72rem',
                    fontWeight: issuePageSize === sz ? 800 : 500,
                    background: issuePageSize === sz ? 'var(--accent-cyan)' : 'var(--bg-tertiary)',
                    color: issuePageSize === sz ? '#000' : 'var(--text-secondary)',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                >
                  {sz}
                </button>
              ))}
            </div>
          </div>

          {/* Category Tabs (فئات المشاكل المبوبة) */}
          <div
            style={{
              display: 'flex',
              gap: '0.4rem',
              flexWrap: 'wrap',
              marginBottom: '1rem',
              paddingBottom: '0.4rem',
            }}
          >
            {issueCategories.map((cat) => {
              const isSelected = selectedIssueCategory === cat.id;
              const hasItems = cat.count > 0;
              return (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => {
                    setSelectedIssueCategory(cat.id);
                    setIssuePage(0);
                  }}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '0.35rem 0.75rem',
                    borderRadius: '6px',
                    fontSize: '0.78rem',
                    fontWeight: isSelected ? 800 : 600,
                    cursor: hasItems ? 'pointer' : 'default',
                    border: isSelected ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
                    background: isSelected
                      ? 'rgba(6, 182, 212, 0.15)'
                      : hasItems
                      ? 'var(--bg-tertiary)'
                      : 'var(--bg-primary)',
                    color: isSelected
                      ? 'var(--accent-cyan)'
                      : hasItems
                      ? 'var(--text-primary)'
                      : 'var(--text-muted)',
                    opacity: hasItems ? 1 : 0.45,
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span>{isAr ? cat.label_ar : cat.label_en}</span>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      padding: '0.08rem 0.35rem',
                      borderRadius: '8px',
                      background: isSelected ? 'var(--accent-cyan)' : 'var(--border-subtle)',
                      color: isSelected ? '#000' : 'var(--text-secondary)',
                      fontWeight: 700,
                    }}
                  >
                    {cat.count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Issues List (Paged) */}
          {pagedIssues.length === 0 ? (
            <div
              style={{
                padding: '1.5rem',
                textAlign: 'center',
                background: 'var(--bg-tertiary)',
                borderRadius: '8px',
                color: 'var(--text-muted)',
                fontSize: '0.82rem',
              }}
            >
              {isAr
                ? 'لا توجد عيوب مرصودة في هذه الفئة المحددة (البيانات سليمة من هذا الجانب).'
                : 'No defects detected in this specific category.'}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
              {pagedIssues.map((issue, idx) => {
                const borderSide = isAr ? 'borderRight' : 'borderLeft';
                const severityColor =
                  issue.severity === 'critical'
                    ? 'var(--accent-rose)'
                    : issue.severity === 'warning'
                    ? 'var(--accent-amber)'
                    : 'var(--accent-cyan)';

                return (
                  <div
                    key={issue.id || idx}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '0.75rem 1rem',
                      background: 'var(--bg-tertiary)',
                      borderRadius: 'var(--radius-sm, 6px)',
                      border: '1px solid var(--border-subtle)',
                      [borderSide]: `4px solid ${severityColor}`,
                      flexWrap: 'wrap',
                      gap: '0.6rem',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: '240px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
                        <span
                          style={{
                            fontSize: '0.65rem',
                            padding: '0.1rem 0.4rem',
                            borderRadius: '4px',
                            fontWeight: 700,
                            background:
                              issue.category === 'missing'
                                ? 'rgba(244, 63, 94, 0.12)'
                                : issue.category === 'duplicate'
                                ? 'rgba(245, 158, 11, 0.12)'
                                : issue.category === 'outlier'
                                ? 'rgba(168, 85, 247, 0.12)'
                                : 'rgba(6, 182, 212, 0.12)',
                            color:
                              issue.category === 'missing'
                                ? 'var(--accent-rose)'
                                : issue.category === 'duplicate'
                                ? 'var(--accent-amber)'
                                : issue.category === 'outlier'
                                ? 'var(--accent-purple, #a855f7)'
                                : 'var(--accent-cyan)',
                          }}
                        >
                          {isAr ? issue.category_ar || 'ملاحظة' : issue.category_en || 'Issue'}
                        </span>
                        <div style={{ fontWeight: 800, fontSize: '0.84rem', color: 'var(--text-primary)' }}>
                          {isAr ? issue.title_ar : issue.title_en}
                        </div>
                      </div>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: '0.25rem', lineHeight: 1.4 }}>
                        {isAr ? issue.desc_ar : issue.desc_en}
                      </div>
                    </div>

                    {/* Contextual Quick Actions */}
                    <div>
                      {issue.action === 'dedup' && (
                        <button
                          onClick={() => {
                            applyOperation({
                              type: 'dedup',
                              title_ar: 'إزالة الصفوف المكررة من الفحص السريع',
                              title_en: 'Removed duplicate records from quick audit',
                              transformFn: (data) => removeDuplicates(data),
                            });
                          }}
                          className="btn btn-secondary btn-sm"
                          style={{ fontSize: '0.75rem', fontWeight: 700 }}
                        >
                          {isAr ? 'إزالة التكرار فوراً' : 'Remove Duplicates'}
                        </button>
                      )}
                      {issue.action === 'impute' && (
                        <button
                          onClick={() => setActivePhase(3)}
                          className="btn btn-primary btn-sm"
                          style={{ fontSize: '0.75rem', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                        >
                          <span>{isAr ? 'معالجة المفقود في المرحلة 3' : 'Fix in Preprocessing'}</span>
                          <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                        </button>
                      )}
                      {(issue.action === 'transform' || issue.action === 'drop') && (
                        <button
                          onClick={() => setActivePhase(3)}
                          className="btn btn-secondary btn-sm"
                          style={{ fontSize: '0.75rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                        >
                          <span>{isAr ? 'إجراء المعالجة في المرحلة 3' : 'Resolve in Phase 3'}</span>
                          <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Pagination Controls Bar */}
          {filteredIssues.length > issuePageSize && (
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '0.75rem',
                marginTop: '1rem',
                paddingTop: '0.75rem',
                borderTop: '1px solid var(--border-subtle)',
                fontSize: '0.76rem',
                color: 'var(--text-muted)',
              }}
            >
              <div>
                {isAr
                  ? `عرض ${safeIssuePage * issuePageSize + 1} - ${Math.min((safeIssuePage + 1) * issuePageSize, filteredIssues.length)} من أصل ${filteredIssues.length} ملاحظة`
                  : `Showing ${safeIssuePage * issuePageSize + 1} - ${Math.min((safeIssuePage + 1) * issuePageSize, filteredIssues.length)} of ${filteredIssues.length} issues`}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <button
                  type="button"
                  onClick={() => setIssuePage(Math.max(0, safeIssuePage - 1))}
                  disabled={safeIssuePage === 0}
                  className="btn btn-secondary btn-sm"
                  style={{
                    padding: '0.2rem 0.6rem',
                    fontSize: '0.72rem',
                    opacity: safeIssuePage === 0 ? 0.4 : 1,
                    cursor: safeIssuePage === 0 ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isAr ? 'السابق' : 'Previous'}
                </button>

                {Array.from({ length: totalIssuePages }, (_, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setIssuePage(i)}
                    style={{
                      width: '26px',
                      height: '26px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '4px',
                      fontSize: '0.72rem',
                      fontWeight: safeIssuePage === i ? 800 : 500,
                      background: safeIssuePage === i ? 'var(--accent-cyan)' : 'var(--bg-tertiary)',
                      color: safeIssuePage === i ? '#000' : 'var(--text-secondary)',
                      border: '1px solid var(--border-subtle)',
                      cursor: 'pointer',
                    }}
                  >
                    {i + 1}
                  </button>
                ))}

                <button
                  type="button"
                  onClick={() => setIssuePage(Math.min(totalIssuePages - 1, safeIssuePage + 1))}
                  disabled={safeIssuePage >= totalIssuePages - 1}
                  className="btn btn-secondary btn-sm"
                  style={{
                    padding: '0.2rem 0.6rem',
                    fontSize: '0.72rem',
                    opacity: safeIssuePage >= totalIssuePages - 1 ? 0.4 : 1,
                    cursor: safeIssuePage >= totalIssuePages - 1 ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isAr ? 'التالي' : 'Next'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Column Profiler Cards */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.25rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h4 style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'ملفات تعريف الأعمدة الإحصائية (Column Profiles):' : 'Column Profiles & Distributions:'}
            </h4>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              {isAr
                ? `عرض الأعمدة ${safeColPage * colPageSize + 1} - ${Math.min((safeColPage + 1) * colPageSize, profile.columns.length)} من إجمالي ${profile.columns.length} (انقر للتفاصيل)`
                : `Showing columns ${safeColPage * colPageSize + 1} - ${Math.min((safeColPage + 1) * colPageSize, profile.columns.length)} of ${profile.columns.length} (click to expand)`}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <TermTooltip termKey="five_number_summary" customLabel={isAr ? 'الملخص الخماسي لتوكي' : 'Tukey Five-Number Summary'} />
            {totalColPages > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', background: 'var(--bg-tertiary)', padding: '0.2rem 0.4rem', borderRadius: 'var(--radius-md, 8px)', border: '1px solid var(--border-subtle)' }}>
                <button
                  onClick={() => setColPage((p) => Math.max(0, p - 1))}
                  disabled={safeColPage === 0}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '0.2rem 0.55rem', fontSize: '0.75rem' }}
                >
                  {isAr ? 'السابق' : 'Prev'}
                </button>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 700, padding: '0 0.4rem' }}>
                  {safeColPage + 1} / {totalColPages}
                </span>
                <button
                  onClick={() => setColPage((p) => Math.min(totalColPages - 1, p + 1))}
                  disabled={safeColPage >= totalColPages - 1}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '0.2rem 0.55rem', fontSize: '0.75rem' }}
                >
                  {isAr ? 'التالي' : 'Next'}
                </button>
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          {paginatedColumns.map((col) => {
            const isExpanded = expandedCol === col.key;
            return (
              <div
                key={col.key}
                style={{
                  background: 'var(--bg-tertiary)',
                  border: isExpanded ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md, 8px)',
                  padding: '0.85rem 1.15rem',
                  transition: 'all 0.15s ease',
                  boxShadow: 'var(--shadow-sm)',
                }}
              >
                <div
                  onClick={() => setExpandedCol(isExpanded ? null : col.key)}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    cursor: 'pointer',
                    userSelect: 'none',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ color: 'var(--accent-teal)', display: 'inline-flex', alignItems: 'center' }}>
                      {isExpanded ? (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ transform: isAr ? 'rotate(90deg)' : 'rotate(-90deg)' }}>
                          <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                      )}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                      {col.key}
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                      {col.type}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.78rem' }}>
                    <span style={{ color: col.nullCount > 0 ? 'var(--accent-amber)' : 'var(--text-muted)', fontWeight: 600 }}>
                      {col.missingPercent.toFixed(1)}% {isAr ? 'مفقود' : 'missing'}
                    </span>
                    <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                      {col.uniqueCount} {isAr ? 'قيمة فريدة' : 'unique'}
                    </span>
                  </div>
                </div>

                {/* Expanded Column Details */}
                {isExpanded && (
                  <div style={{ marginTop: '1rem', paddingTop: '0.85rem', borderTop: '1px solid var(--border-subtle)' }}>
                    {col.type === 'numeric' && col.stats && (
                      <div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(95px, 1fr))', gap: '0.5rem', textAlign: 'center' }}>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Min (الصغرى)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.min}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Q1 (الربيع 1)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.q1}</div>
                          </div>
                          <div style={{ background: 'var(--accent-teal-glow)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--accent-teal)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--accent-teal)', fontWeight: 800 }}>Median (الوسيط)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>{col.stats.median}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Q3 (الربيع 3)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.q3}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Max (العظمى)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.max}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Mean (المتوسط)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.mean.toFixed(2)}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>Std (الانحراف)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.std.toFixed(2)}</div>
                          </div>
                          <div style={{ background: 'var(--bg-secondary)', padding: '0.5rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
                            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>IQR (المدى)</div>
                            <div style={{ fontSize: '0.88rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{col.stats.iqr}</div>
                          </div>
                        </div>

                        <div style={{ marginTop: '0.75rem', fontSize: '0.76rem', color: 'var(--text-secondary)', background: 'var(--bg-secondary)', padding: '0.6rem 0.85rem', borderRadius: '6px', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                          <span style={{ color: 'var(--accent-teal, #14b8a6)', fontWeight: 700 }}>{isAr ? 'ملاحظة تعليمية:' : 'Pedagogical Note:'}</span>
                          <span>
                            {isAr
                              ? 'الوسيط (Median) هو القيمة التي تقسم البيانات لنصفين متساويين، وهو أكثر متانة من المتوسط (Mean) لأنه لا يتأثر بالقيم الشاذة والمتطرفة.'
                              : 'The Median divides sorted data into equal halves and is resistant to extreme outliers, whereas Mean is sensitive to skewness.'}
                          </span>
                        </div>
                      </div>
                    )}

                    {col.type !== 'numeric' && col.categoryCounts && (
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 700 }}>
                          {isAr ? 'أعلى الفئات تكراراً ونسبها:' : 'Category Frequencies & Levels:'}
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                          {Object.entries(col.categoryCounts)
                            .slice(0, 8)
                            .map(([val, count]) => (
                              <span
                                key={val}
                                style={{
                                  background: 'var(--bg-secondary)',
                                  border: '1px solid var(--border-subtle)',
                                  padding: '0.25rem 0.65rem',
                                  borderRadius: 'var(--radius-sm, 4px)',
                                  fontSize: '0.76rem',
                                  fontFamily: 'var(--font-mono)',
                                  color: 'var(--text-primary)',
                                }}
                              >
                                {val}: <strong style={{ color: 'var(--accent-teal)' }}>{count}</strong> ({((count / profile.totalRows) * 100).toFixed(1)}%)
                              </span>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Bottom Pagination Controls for Column Cards */}
        {totalColPages > 1 && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              {isAr ? `صفحة الأعمدة ${safeColPage + 1} من ${totalColPages}` : `Column Page ${safeColPage + 1} of ${totalColPages}`}
            </span>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button
                onClick={() => setColPage((p) => Math.max(0, p - 1))}
                disabled={safeColPage === 0}
                className="btn btn-secondary btn-sm"
                style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
              >
                {isAr ? 'السابق' : 'Prev'}
              </button>
              <button
                onClick={() => setColPage((p) => Math.min(totalColPages - 1, p + 1))}
                disabled={safeColPage >= totalColPages - 1}
                className="btn btn-secondary btn-sm"
                style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
              >
                {isAr ? 'التالي' : 'Next'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Paginated Data Preview Table */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-lg, 12px)',
          padding: '1.25rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <span style={{ fontSize: '0.92rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {isAr ? `معاينة السجلات الحالية (${filteredData.length} سجل)` : `Record Preview (${filteredData.length} rows)`}
            </span>
          </div>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setPage(0);
            }}
            placeholder={isAr ? 'بحث سريع في الجدول...' : 'Quick filter table...'}
            style={{
              padding: '0.35rem 0.75rem',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-sm, 6px)',
              color: 'var(--text-primary)',
              fontSize: '0.8rem',
            }}
          />
        </div>

        <div style={{ overflowX: 'auto', maxHeight: '420px', overflowY: 'auto', marginBottom: '1rem', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm, 6px)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: isAr ? 'right' : 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-medium)', background: 'var(--bg-tertiary)', position: 'sticky', top: 0, zIndex: 10 }}>
                {profile.columns.map((col) => (
                  <th
                    key={col.key}
                    style={{
                      padding: '0.6rem 0.75rem',
                      color: 'var(--accent-teal)',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 800,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {col.key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageData.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)', background: i % 2 === 0 ? 'var(--bg-secondary)' : 'var(--bg-tertiary)' }}>
                  {profile.columns.map((col) => {
                    const val = row[col.key];
                    const valStr = val !== null && val !== undefined ? String(val) : 'NULL';
                    return (
                      <td
                        key={col.key}
                        title={valStr}
                        style={{
                          padding: '0.45rem 0.75rem',
                          color: val === null || val === undefined ? 'var(--accent-rose)' : 'var(--text-primary)',
                          whiteSpace: 'nowrap',
                          maxWidth: '280px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          fontFamily: typeof val === 'number' ? 'var(--font-mono)' : 'inherit',
                        }}
                      >
                        {val !== null && val !== undefined ? (
                          valStr
                        ) : (
                          <span style={{ color: 'var(--accent-rose)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>NULL</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            {isAr ? `صفحة ${page + 1} من ${totalPages || 1}` : `Page ${page + 1} of ${totalPages || 1}`}
          </span>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="btn btn-secondary btn-sm"
              style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
            >
              {isAr ? 'السابق' : 'Prev'}
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="btn btn-secondary btn-sm"
              style={{ padding: '0.25rem 0.65rem', fontSize: '0.75rem' }}
            >
              {isAr ? 'التالي' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
