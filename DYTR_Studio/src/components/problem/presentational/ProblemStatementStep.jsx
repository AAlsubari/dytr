// ==============================================================================
// Step 1: Problem Statement Presentation Component (Pure Presentational / Dumb)
// Zero hardcoded domain arrays - Receives state & callbacks via props
// ==============================================================================

import React from 'react';
import SectionCard from '../../common/SectionCard';
import TermTooltip from '../../common/TermTooltip';

function ProblemStatementStep({
  isAr,
  statement,
  scenarios = [],
  onStatementChange,
  onApplyScenario,
}) {
  return (
    <div className="problem-step-content">
      <SectionCard
        sectionCode="01.1"
        title={isAr ? 'ما المشكلة التي تريد حلها؟' : 'What Problem Are You Solving?'}
        subtitle={
          isAr
            ? 'صياغة المشكلة بدقة بلغة طبيعية هي نصف حلها (Polya, 1945). صف الهدف النهائي من بناء النموذج.'
            : 'Formulating the problem precisely in natural language is half the solution (Polya, 1945).'
        }
      >
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '1.25rem', lineHeight: 1.6 }}>
          {isAr
            ? 'اكتب جملة أو فقرة واضحة تصف المشكلة أو التطبيق الذي تستهدفه. يساعد ذلك المنصة على استنتاج المعمارية والبيانات المناسبة.'
            : 'Write a concise statement defining the core business or research challenge you aim to address.'}
        </p>

        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
            {isAr ? 'وصف المشكلة المستهدفة:' : 'Problem Description:'}
          </label>
          <textarea
            value={statement}
            onChange={(e) => onStatementChange(e.target.value)}
            rows={4}
            maxLength={500}
            placeholder={
              isAr
                ? 'مثال: أريد تصنيف مراجعات وتعليقات العملاء في المتجر الإلكتروني إلى إيجابية وسلبية لتحسين تجربة المستخدم...'
                : 'Example: I want to classify incoming user feedback into positive and negative sentiments to prioritize support...'
            }
            style={{
              width: '100%',
              padding: '0.85rem',
              borderRadius: '8px',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-card)',
              color: 'var(--text-primary)',
              fontSize: '0.95rem',
              fontFamily: 'inherit',
              lineHeight: 1.5,
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.35rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            <span>{isAr ? 'الحد الأدنى الموصى به: 10 أحرف' : 'Minimum recommended: 10 chars'}</span>
            <span>{statement.length} / 500</span>
          </div>
        </div>

        {/* Dynamically Loaded Academic Sample Templates */}
        {scenarios.length > 0 && (
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.6rem' }}>
              {isAr ? 'أمثلة أكاديمية شائعة جاهزة للاختيار:' : 'Common Pre-Formulated Scientific Scenarios:'}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {scenarios.map((sc) => (
                <button
                  key={sc.id}
                  type="button"
                  onClick={() => onApplyScenario(sc)}
                  className="btn btn-secondary btn-sm"
                >
                  {isAr ? sc.title_ar : sc.title_en}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* 3-Layer Methodological Helper */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-elevated)', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
          <TermTooltip
            termKey="problem_framing"
            customLabel={isAr ? 'تأطير المشكلة (Problem Framing)' : 'Problem Framing'}
          />
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr ? 'لماذا صياغة المشكلة مهمة علميا؟ انقر على أيقونة ? للاطلاع على المرجع المعرفي الموسع.' : 'Why is scientific framing critical? Click ? tooltip for full academic reference.'}
          </span>
        </div>
      </SectionCard>
    </div>
  );
}

export default React.memo(ProblemStatementStep);
