// ==============================================================================
// Step 2: AI Necessity Diagnostic Presentation Component (Pure Presentational / Dumb)
// Zero hardcoded quiz structures - Renders dynamic questions fetched from API
// ==============================================================================

import React from 'react';
import SectionCard from '../../common/SectionCard';
import TermTooltip from '../../common/TermTooltip';

function DiagnosticQuizStep({
  isAr,
  questions = [],
  answers = {},
  diagnosticResult = 'ai_needed',
  onAnswerChange,
}) {
  const isRuleSufficient = answers.fixedRules === 'yes' && answers.complexPattern === 'simple';

  return (
    <div className="problem-step-content">
      <SectionCard
        sectionCode="01.2"
        title={isAr ? 'التشخيص الذاتي: هل تحتاج ذكاء اصطناعيا فعلا؟' : 'AI Necessity & Feasibility Diagnostic'}
        subtitle={
          isAr ? (
            <>
              60% من المشاريع تفشل لاختيار الذكاء الاصطناعي في مسائل تكفيها قواعد شرطية بسيطة. اطلع على{' '}
              <TermTooltip termKey="ai_necessity" customLabel="تشخيص جدوى الذكاء الاصطناعي" />
            </>
          ) : (
            <>
              Empirical diagnostic under <TermTooltip termKey="ai_necessity" customLabel="AI Feasibility Standard" />
            </>
          )
        }
      >
        <div style={{ display: 'grid', gap: '1.25rem', marginBottom: '1.5rem' }}>
          {questions.map((q) => (
            <div
              key={q.id}
              style={{
                background: 'var(--bg-card)',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.6rem', color: 'var(--text-primary)' }}>
                {q.order}. {isAr ? q.question_ar : q.question_en}
              </div>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {q.options.map((opt) => (
                  <label
                    key={opt.value}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      fontSize: '0.86rem',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="radio"
                      name={q.id}
                      checked={answers[q.id] === opt.value}
                      onChange={() => onAnswerChange(q.id, opt.value)}
                    />
                    <span>{isAr ? opt.label_ar : opt.label_en}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Diagnostic Result Card */}
        <div
          style={{
            padding: '1.25rem',
            borderRadius: '8px',
            border: '1px solid var(--border-active)',
            background: 'var(--bg-elevated)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <span className="badge badge-teal">{isAr ? 'تشخيص المنصة الأكاديمي' : 'System Diagnostic'}</span>
            <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--accent-teal)' }}>
              {isRuleSufficient
                ? isAr
                  ? 'قاعدة برمجية if-else تكفي'
                  : 'Heuristic Rules Sufficient'
                : isAr
                ? 'المسألة تتطلب تدريب شبكة عصبية عميقة (Transformer)'
                : 'Transformer Neural Network Recommended'}
            </span>
          </div>

          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
            {isRuleSufficient
              ? isAr
                ? 'بما أن القواعد محددة بدقة والنمط بسيط، كتابة شرط برمجي أو جدول بحث يوفر الموارد والوقت بدلا من تدريب نموذج مكلف. يمكنك مع ذلك المتابعة إن رغبت.'
                : 'Deterministic rules will yield higher precision and zero latency compared to neural modeling.'
              : isAr
              ? 'تتطلب مسألتك معالجة لغوية سياقية عميقة لا يمكن حصرها بقواعد يدوية. مكتبة dytr توفر المعمارية المثالية لحل هذه المسألة مع الحفاظ على الذاكرة.'
              : 'Implicit semantics and variable context require transformer attention mechanisms.'}
          </p>
        </div>
      </SectionCard>
    </div>
  );
}

export default React.memo(DiagnosticQuizStep);
