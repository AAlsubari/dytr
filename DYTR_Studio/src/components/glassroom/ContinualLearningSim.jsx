// ==============================================================================
// DyTrStudio Continual Learning & Catastrophic Forgetting Interactive Simulator
// Matching Academy Design Language: Trajectory Curves, Bilingual RTL/LTR, Dual Mode (Day/Night)
// Zero Emojis Policy - DyTR EWC & Experience Replay Buffer Mechanics - 3-Layer TermTooltips
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function ContinualLearningSim() {
  const { language, theme } = useStudioStore();
  const isAr = language === 'ar';

  const [useEWC, setUseEWC] = useState(true);
  const [ewcLambda, setEwcLambda] = useState(1000);
  const [useReplay, setUseReplay] = useState(false);
  const [simData, setSimData] = useState(null);

  useEffect(() => {
    let isMounted = true;
    api
      .getContinualSim({
        use_ewc: useEWC,
        ewc_lambda: ewcLambda,
        use_replay: useReplay,
        num_steps: 30,
      })
      .then((data) => {
        if (isMounted && data) setSimData(data);
      });
    return () => {
      isMounted = false;
    };
  }, [useEWC, ewcLambda, useReplay]);

  const isProtected = useEWC || useReplay;

  return (
    <div
      style={{
        background: 'var(--bg-secondary)',
        borderRadius: 'var(--radius-lg, 12px)',
        padding: '1.5rem',
        border: '1px solid var(--border-subtle)',
        boxShadow: 'var(--shadow-sm, 0 2px 8px rgba(0,0,0,0.06))',
      }}
    >
      {/* Simulator Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1.25rem',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div>
          <h3
            style={{
              fontSize: '1.15rem',
              fontWeight: 800,
              color: 'var(--text-primary)',
              marginBottom: '0.25rem',
            }}
          >
            {isAr ? 'درع التعلم المستمر وتثبيت الأوزان (EWC & Replay)' : 'Continual Learning: Elastic Weight Consolidation (EWC)'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'تدريب متتابع: مرحلة 1 (المشاعر) ثم مرحلة 2 (الموضوعات). شاهد كيف تحمي مصفوفة فيشر وذاكرة الاسترجاع الأوزان من النسيان الكارثي.'
              : 'Sequential Training: Task 1 (Sentiment) then Task 2 (Topic). Inspect how EWC Fisher penalty and replay buffers prevent catastrophic forgetting.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span className={`badge ${isProtected ? 'badge-emerald' : 'badge-rose'}`}>
            {isProtected
              ? (isAr ? 'حماية التعلم المستمر مفعلة' : 'Continual Shields Active')
              : (isAr ? 'غير محمي (نسيان كارثي)' : 'Unprotected (Catastrophic)')}
          </span>
          <span className="badge badge-teal">
            dytr.memory (EWC & Replay)
          </span>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '1.5rem',
          alignItems: 'stretch',
        }}
      >
        {/* Interactive Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* EWC Toggle */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              padding: '1rem',
              borderRadius: 'var(--radius-md, 8px)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              boxShadow: 'var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.05))',
            }}
          >
            <div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                }}
              >
                <span>{isAr ? 'درع تثبيت الأوزان (EWC)' : 'Elastic Weight Consolidation (EWC)'}</span>
                <TermTooltip
                  title={isAr ? 'درع تثبيت الأوزان (EWC)' : 'Elastic Weight Consolidation (EWC)'}
                  explanation={
                    isAr
                      ? 'تقنية رياضية في dytr.memory.ewc تحسب مصفوفة معلومات فيشر (Fisher Information Matrix) لتحديد الأوزان الحرجة للمهام السابقة، وتفرض عقوبة تربيعية على تعديلها لحمايتها من النسيان الكارثي.'
                      : 'Computes diagonal Fisher Information matrix to identify parameters critical to past tasks and penalizes modification using quadratic penalty loss.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </div>
              <p
                style={{
                  fontSize: '0.78rem',
                  color: 'var(--text-muted)',
                  marginTop: '0.25rem',
                  fontFamily: 'var(--font-mono)',
                  direction: 'ltr',
                  textAlign: isAr ? 'right' : 'left',
                }}
              >
                L = L_task + (λ / 2) · Σ F_i (θ_i - θ_i*)^2
              </p>
            </div>
            <button
              type="button"
              onClick={() => setUseEWC(!useEWC)}
              className={`btn btn-sm ${useEWC ? 'btn-primary' : 'btn-secondary'}`}
              style={{ minWidth: '90px' }}
            >
              {useEWC ? (isAr ? 'مفعّل' : 'Enabled') : (isAr ? 'معطّل' : 'Disabled')}
            </button>
          </div>

          {/* EWC Lambda Regularization Slider */}
          {useEWC && (
            <div
              className="form-group"
              style={{
                background: 'var(--bg-tertiary)',
                padding: '1rem',
                borderRadius: 'var(--radius-md, 8px)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div
                className="form-label"
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '0.6rem',
                  fontSize: '0.88rem',
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  <span>{isAr ? 'قوة عقوبة فيشر (Regularization λ):' : 'EWC Regularization Penalty (λ):'}</span>
                  <TermTooltip
                    title={isAr ? 'معامل عقوبة فيشر (λ)' : 'Fisher Regularization Penalty (λ)'}
                    explanation={
                      isAr
                        ? 'معامل العقوبة λ في مصفوفة فيشر؛ القيمة الأعلى تحافظ بقوة أكبر على أداء ومفردات المهمة السابقة، بينما القيمة المنخفضة تمنح النموذج مرونة أسرع لاستيعاب المهمة الجديدة.'
                        : 'Penalty coefficient balancing stability of earlier tasks against plasticity for new tasks.'
                    }
                    size="sm"
                    hideLabel={true}
                  />
                </span>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: '6px',
                    padding: '0.2rem 0.5rem',
                  }}
                >
                  <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-muted)' }}>λ =</span>
                  <input
                    type="number"
                    min="100"
                    max="5000"
                    step="100"
                    value={ewcLambda}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) setEwcLambda(Math.min(5000, Math.max(100, val)));
                    }}
                    style={{
                      width: '60px',
                      background: 'transparent',
                      border: 'none',
                      outline: 'none',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 800,
                      color: 'var(--accent-teal)',
                      padding: 0,
                    }}
                  />
                </div>
              </div>
              <input
                type="range"
                min="100"
                max="5000"
                step="100"
                value={ewcLambda}
                onChange={(e) => setEwcLambda(parseFloat(e.target.value))}
                className="slider-input"
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Experience Replay Buffer Toggle */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              padding: '1rem',
              borderRadius: 'var(--radius-md, 8px)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              boxShadow: 'var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.05))',
            }}
          >
            <div>
              <div
                style={{
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                }}
              >
                <span>{isAr ? 'ذاكرة الاسترجاع (Replay Buffer)' : 'Experience Replay Buffer'}</span>
                <TermTooltip
                  title={isAr ? 'ذاكرة الاسترجاع (Replay Buffer)' : 'Experience Replay Buffer'}
                  explanation={
                    isAr
                      ? 'مخزن ذاكرة عصبية (dytr.memory.replay.ReplayBuffer) يحتفظ بنماذج وعينات تمثيلية من البيانات السابقة، ويدمجها بتناوب دوري مع بيانات المهمة الجديدة لإنعاش ذاكرة النموذج.'
                      : 'Maintains exemplar memory from earlier datasets and replays them interleaved during subsequent training steps.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                {isAr ? 'سعة المخزن: 1,000 عينة مسترجعة' : 'Buffer capacity: 1,000 rehearsal exemplars'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setUseReplay(!useReplay)}
              className={`btn btn-sm ${useReplay ? 'btn-primary' : 'btn-secondary'}`}
              style={{ minWidth: '90px' }}
            >
              {useReplay ? (isAr ? 'مفعّل' : 'Enabled') : (isAr ? 'معطّل' : 'Disabled')}
            </button>
          </div>

          {/* Metric Summary Comparison */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '0.85rem',
            }}
          >
            <div
              style={{
                padding: '1rem',
                borderRadius: 'var(--radius-md, 8px)',
                background: isProtected ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
                border: `1px solid ${isProtected ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'}`,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div
                style={{
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  color: isProtected ? 'var(--accent-emerald, #10b981)' : 'var(--accent-rose, #f43f5e)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  marginBottom: '0.4rem',
                }}
              >
                <span>{isAr ? 'استبقاء المهمة 1:' : 'Task 1 Retention:'}</span>
                <TermTooltip
                  title={isAr ? 'استبقاء المعرفة السابقة' : 'Memory Retention'}
                  explanation={
                    isAr
                      ? 'نسبة دقة المهمة الأولى بعد اكتمال تدريب المهمة الثانية تماماً. تشير النسبة المرتفعة إلى ثبات المعرفة ونجاح الحماية.'
                      : 'Task 1 accuracy measured after sequential training on Task 2 concludes.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </div>
              <div
                style={{
                  fontSize: '1.45rem',
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  color: isProtected ? 'var(--accent-emerald, #10b981)' : 'var(--accent-rose, #f43f5e)',
                }}
              >
                {simData?.final_metrics?.task1_final ?? (isProtected ? 92.4 : 18.2)}%
              </div>
            </div>

            <div
              style={{
                padding: '1rem',
                borderRadius: 'var(--radius-md, 8px)',
                background: (simData?.final_metrics?.bwt || 0) >= -8 || isProtected ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
                border: `1px solid ${(simData?.final_metrics?.bwt || 0) >= -8 || isProtected ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'}`,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div
                style={{
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  color: (simData?.final_metrics?.bwt || 0) >= -8 || isProtected ? 'var(--accent-emerald, #10b981)' : 'var(--accent-rose, #f43f5e)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  marginBottom: '0.4rem',
                }}
              >
                <span>{isAr ? 'النقل الرجعي (BWT):' : 'Backward Transfer:'}</span>
                <TermTooltip
                  title={isAr ? 'معامل النقل الرجعي (BWT)' : 'Backward Transfer (BWT)'}
                  explanation={
                    isAr
                      ? 'مقياس أكاديمي لحساب فرق الأداء في المهام السابقة بعد تدريب مهام جديدة. القيم القريبة من 0 تعني صمود المعرفة السابقة دون تراجع.'
                      : 'Standard academic metric calculating the delta in prior task accuracy after completing subsequent training.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </div>
              <div
                style={{
                  fontSize: '1.45rem',
                  fontWeight: 800,
                  fontFamily: 'var(--font-mono)',
                  color: (simData?.final_metrics?.bwt || 0) >= -8 || isProtected ? 'var(--accent-emerald, #10b981)' : 'var(--accent-rose, #f43f5e)',
                }}
              >
                {simData?.final_metrics?.bwt ?? (isProtected ? -2.6 : -76.8)}%
              </div>
            </div>
          </div>
        </div>

        {/* Dynamic Trajectory Visual Graph */}
        <div
          style={{
            background: 'var(--bg-tertiary)',
            borderRadius: 'var(--radius-md, 8px)',
            padding: '1.25rem',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '0.85rem',
              fontSize: '0.8rem',
              fontFamily: 'var(--font-mono)',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <span style={{ color: 'var(--accent-teal, #14b8a6)', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ width: '12px', height: '3px', background: 'var(--accent-teal)', display: 'inline-block', borderRadius: '2px' }} />
              <span>{isAr ? 'دقة المهمة 1 (المشاعر)' : 'Task 1 (Sentiment)'}</span>
            </span>
            <span style={{ color: 'var(--accent-amber, #f59e0b)', fontWeight: 800, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ width: '12px', height: '3px', background: 'var(--accent-amber)', display: 'inline-block', borderRadius: '2px' }} />
              <span>{isAr ? 'دقة المهمة 2 (الموضوعات)' : 'Task 2 (Topic)'}</span>
            </span>
          </div>

          <div
            style={{
              background: 'var(--bg-elevated)',
              borderRadius: '8px',
              border: '1px solid var(--border-medium)',
              padding: '0.75rem',
            }}
          >
            <svg width="100%" height="150" viewBox="0 0 300 120" style={{ overflow: 'visible', display: 'block' }}>
              {/* Background Horizontal Guide Lines (Light & Dark Compatible) */}
              <line x1="0" y1="20" x2="300" y2="20" stroke="var(--border-subtle, rgba(148, 163, 184, 0.15))" strokeDasharray="3 3" />
              <line x1="0" y1="60" x2="300" y2="60" stroke="var(--border-subtle, rgba(148, 163, 184, 0.15))" strokeDasharray="3 3" />
              <line x1="0" y1="100" x2="300" y2="100" stroke="var(--border-subtle, rgba(148, 163, 184, 0.15))" strokeDasharray="3 3" />

              {/* Stage Divider Line - High Contrast Dual Theme */}
              <line
                x1="150"
                y1="0"
                x2="150"
                y2="120"
                stroke="var(--border-medium, rgba(148, 163, 184, 0.4))"
                strokeDasharray="4 4"
                strokeWidth="1.5"
              />

              <text x="75" y="14" fill="var(--text-muted)" fontSize="8.5" fontWeight="700" textAnchor="middle" fontFamily="var(--font-mono)">
                {isAr ? 'المرحلة 1: تدريب المهمة 1' : 'STAGE 1: TASK 1'}
              </text>
              <text x="225" y="14" fill="var(--text-muted)" fontSize="8.5" fontWeight="700" textAnchor="middle" fontFamily="var(--font-mono)">
                {isAr ? 'المرحلة 2: تدريب المهمة 2' : 'STAGE 2: TASK 2'}
              </text>

              {/* Task 1 Accuracy Polyline */}
              {simData && (
                <polyline
                  fill="none"
                  stroke="var(--accent-teal, #14b8a6)"
                  strokeWidth="2.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={simData.task1_accuracy.map((val, idx) => `${idx * 10},${110 - (val * 0.9)}`).join(' ')}
                />
              )}

              {/* Task 2 Accuracy Polyline */}
              {simData && (
                <polyline
                  fill="none"
                  stroke="var(--accent-amber, #f59e0b)"
                  strokeWidth="2.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={simData.task2_accuracy.map((val, idx) => `${idx * 10},${110 - (val * 0.9)}`).join(' ')}
                />
              )}
            </svg>
          </div>

          <p
            style={{
              fontSize: '0.8rem',
              color: 'var(--text-muted)',
              marginTop: '0.85rem',
              textAlign: 'center',
              lineHeight: 1.55,
            }}
          >
            {isProtected
              ? (isAr ? 'بفضل درع EWC وذاكرة الاسترجاع، تبقى معرفة المهمة 1 محفوظة ومحمية أثناء استيعاب المهمة 2.' : 'With dytr EWC & Replay protection, Task 1 memory is retained while Task 2 is learned.')
              : (isAr ? 'بدون حماية، تنهار دقة المهمة 1 لحظياً بمجرد بدء تدريب المهمة 2 (ظاهرة النسيان الكارثي).' : 'Without protection, Task 1 knowledge collapses immediately upon training Task 2.')}
          </p>
        </div>
      </div>
    </div>
  );
}

