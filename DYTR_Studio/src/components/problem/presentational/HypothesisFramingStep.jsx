// ==============================================================================
// Step 4: Hypothesis Framing Presentation Component (Pure Presentational / Dumb)
// Zero hardcoded model catalogs - Renders dynamic foundation models from API
// ==============================================================================

import React from 'react';
import SectionCard from '../../common/SectionCard';
import TermTooltip from '../../common/TermTooltip';

function HypothesisFramingStep({
  isAr,
  hypothesis = {},
  foundationModels = [],
  onHypothesisChange,
}) {
  const handleModelChange = (e) => {
    onHypothesisChange({ ...hypothesis, foundationModel: e.target.value });
  };

  const handleAccuracyChange = (e) => {
    onHypothesisChange({ ...hypothesis, targetAccuracy: Number(e.target.value) });
  };

  const handleSamplesChange = (e) => {
    onHypothesisChange({ ...hypothesis, minSamples: Number(e.target.value) });
  };

  return (
    <div className="problem-step-content">
      <SectionCard
        sectionCode="01.4"
        title={isAr ? 'صياغة الفرضية الأكاديمية القابلة للاختبار' : 'Scientific Hypothesis Formulation'}
        subtitle={
          isAr ? (
            <>
              البحث العلمي يبدأ بـ <TermTooltip termKey="popper_hypothesis" customLabel="فرضية قابلة للإبطال" /> ومحددة بـ <TermTooltip termKey="baseline_metric" customLabel="مقاييس مرجعية" /> قبل البدء بالتدريب.
            </>
          ) : (
            <>
              Formulating falsifiable claims under <TermTooltip termKey="popper_hypothesis" customLabel="Popperian Standard" /> with quantitative <TermTooltip termKey="baseline_metric" customLabel="Baseline Metrics" />.
            </>
          )
        }
      >
        <div style={{ background: 'var(--bg-card)', padding: '1.5rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem', fontWeight: 600 }}>
            {isAr ? 'صيغة الفرضية الرياضية للمشروع:' : 'Hypothesis Template:'}
          </div>

          <div
            style={{
              fontSize: '1.05rem',
              lineHeight: 2,
              color: 'var(--text-primary)',
              background: 'var(--bg-elevated)',
              padding: '1.25rem',
              borderRadius: '8px',
              border: '1px solid var(--border-active)',
            }}
          >
            {isAr ? (
              <>
                نفترض أن نموذج المحولات الديناميكي المبني على معمارية{' '}
                <select
                  value={hypothesis.foundationModel || 'custom_dytr'}
                  onChange={handleModelChange}
                  style={{
                    padding: '0.2rem 0.5rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    fontWeight: 700,
                  }}
                >
                  {foundationModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name_ar || m.name_en || m.id}
                    </option>
                  ))}
                </select>{' '}
                قادر على تحقيق دقة تصنيف لا تقل عن{' '}
                <input
                  type="number"
                  min="50"
                  max="99"
                  value={hypothesis.targetAccuracy ?? 85}
                  onChange={handleAccuracyChange}
                  style={{
                    width: '65px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--accent-teal)',
                    fontWeight: 700,
                    textAlign: 'center',
                  }}
                />
                % عند تدريبه على حجم عينة يبلغ{' '}
                <input
                  type="number"
                  step="500"
                  min="1000"
                  max="100000"
                  value={hypothesis.minSamples ?? 5000}
                  onChange={handleSamplesChange}
                  style={{
                    width: '90px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--accent-teal)',
                    fontWeight: 700,
                    textAlign: 'center',
                  }}
                />{' '}
                {hypothesis.sampleUnit || 'عينة تدريبية'}.
              </>
            ) : (
              <>
                We hypothesize that a dynamic transformer utilizing backbone{' '}
                <select
                  value={hypothesis.foundationModel || 'custom_dytr'}
                  onChange={handleModelChange}
                  style={{
                    padding: '0.2rem 0.5rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    fontWeight: 700,
                  }}
                >
                  {foundationModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name_en || m.name_ar || m.id}
                    </option>
                  ))}
                </select>{' '}
                can achieve a validation accuracy of at least{' '}
                <input
                  type="number"
                  min="50"
                  max="99"
                  value={hypothesis.targetAccuracy ?? 85}
                  onChange={handleAccuracyChange}
                  style={{
                    width: '65px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--accent-teal)',
                    fontWeight: 700,
                    textAlign: 'center',
                  }}
                />
                % when trained on a dataset of{' '}
                <input
                  type="number"
                  step="500"
                  min="1000"
                  max="100000"
                  value={hypothesis.minSamples ?? 5000}
                  onChange={handleSamplesChange}
                  style={{
                    width: '90px',
                    padding: '0.2rem 0.4rem',
                    borderRadius: '4px',
                    border: '1px solid var(--accent-teal)',
                    background: 'var(--bg-card)',
                    color: 'var(--accent-teal)',
                    fontWeight: 700,
                    textAlign: 'center',
                  }}
                />{' '}
                samples.
              </>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'سيتم حفظ هذه الفرضية ومقارنتها تلقائيا في مختبر التقييم (Module 09) بعد اكتمال التدريب.'
              : 'This hypothesis will be tracked and validated in Module 09 (Evaluation Lab).'}
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

export default React.memo(HypothesisFramingStep);
