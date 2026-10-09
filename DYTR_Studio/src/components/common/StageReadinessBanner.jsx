// ==============================================================================
// StageReadinessBanner.jsx - Academic Pipeline Prerequisites & Safety Guardrails
// Enforces CRISP-DM / ISO 9241-210 rigor across training, evaluation, export & inference
// Zero Emojis Policy - Responsive RTL/LTR - Dual Theme - Educational Action Prompts
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';

export default function StageReadinessBanner({ stageCode, stageName, requiredStages = [] }) {
  const {
    language,
    modelConfig,
    trainingState,
    evaluationState,
    setActiveModule,
  } = useStudioStore();

  const { currentDataset } = useDatasetStore();
  const isAr = language === 'ar';

  const [bypassed, setBypassed] = useState(false);

  // Evaluate prerequisite conditions
  const checks = [];

  // Check 1: Data Preparation
  if (requiredStages.includes('dataprep') || requiredStages.includes('02')) {
    const hasData = Boolean(currentDataset && currentDataset.length > 0);
    checks.push({
      id: 'data',
      stageId: 'dataprep',
      stageCode: '02',
      name_ar: 'تجهيز وتقسيم البيانات',
      name_en: 'Dataset Preparation',
      status: hasData,
      requirement_ar: 'يجب اختيار مجموعة بيانات والتحقق من تقسيمها أولاً.',
      requirement_en: 'Dataset must be selected and split before training.',
      action_ar: 'الانتقال إلى تجهيز البيانات [02]',
      action_en: 'Go to Data Prep [02]',
    });
  }

  // Check 2: Architecture Construction
  if (requiredStages.includes('builder') || requiredStages.includes('03')) {
    const hasTasks = Boolean(modelConfig?.tasks && modelConfig.tasks.length > 0);
    checks.push({
      id: 'arch',
      stageId: 'builder',
      stageCode: '03',
      name_ar: 'بناء معمارية النموذج ورؤوس المهام',
      name_en: 'Architecture Definition',
      status: hasTasks,
      requirement_ar: 'يجب تحديد رؤوس التصنيف وأبعاد المحول التوليدي.',
      requirement_en: 'Transformer dimensions and task heads must be configured.',
      action_ar: 'الانتقال إلى بناء المعمارية [03]',
      action_en: 'Go to Architecture [03]',
    });
  }

  // Check 3: Model Training
  if (requiredStages.includes('arena') || requiredStages.includes('04')) {
    const isTrained = Boolean(
      trainingState?.is_trained ||
      trainingState?.is_training ||
      (trainingState?.step > 0) ||
      (trainingState?.accHistory && trainingState.accHistory.length > 1) ||
      trainingState?.accuracy > 60
    );
    checks.push({
      id: 'train',
      stageId: 'arena',
      stageCode: '04',
      name_ar: 'تدريب النموذج وتقارب الأوزان',
      name_en: 'Model Training Convergence',
      status: isTrained,
      requirement_ar: 'يتطلب تشغيل دورة تدريبية لضبط الأوزان العصبية قبل القياس أو التصدير.',
      requirement_en: 'Model weights must be optimized through training loop before evaluation/export.',
      action_ar: 'الانتقال إلى التدريب وبناء و بناء النماذج [04]',
      action_en: 'Go to Training Arena [04]',
    });
  }

  // Check 4: Evaluation Metrics
  if (requiredStages.includes('evaluation') || requiredStages.includes('06')) {
    const isEvaluated = Boolean(evaluationState?.metrics?.accuracy > 0);
    checks.push({
      id: 'eval',
      stageId: 'evaluation',
      stageCode: '06',
      name_ar: 'تقييم الدقة ومصفوفة الارتباك',
      name_en: 'Accuracy & Confusion Matrix Evaluation',
      status: isEvaluated,
      requirement_ar: 'يجب فحص دقة النموذج على حزمة الاختبار المستقلة لضمان جودة الأداء.',
      requirement_en: 'Evaluation benchmark and confusion matrix must be audited.',
      action_ar: 'الانتقال إلى التقييم والتشخيص [06]',
      action_en: 'Go to Evaluation Lab [06]',
    });
  }

  const failedChecks = checks.filter((c) => !c.status);
  const allPassed = failedChecks.length === 0;

  if (allPassed || bypassed) 
    {
    return (
      <div
        // style={{
        //   background: 'rgba(20, 184, 166, 0.06)',
        //   border: '1px solid rgba(20, 184, 166, 0.3)',
        //   borderInlineStart: '4px solid var(--accent-teal)',
        //   borderRadius: 'var(--radius-md, 8px)',
        //   padding: '0.75rem 1.15rem',
        //   marginBottom: '1.5rem',
        //   display: 'flex',
        //   justifyContent: 'space-between',
        //   alignItems: 'center',
        //   flexWrap: 'wrap',
        //   gap: '0.65rem',
        //   fontSize: '0.8rem',
        // }}
      >
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {/* <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: 'var(--accent-teal)',
              boxShadow: '0 0 8px var(--accent-teal)',
            }}
          />
           */}
          {/* <strong style={{ color: 'var(--accent-teal)' }}>
            {isAr ? 'بوابة التحقق الأكاديمي: مكتملة وجاهزة' : 'Academic Guardrail: Verified & Ready'}
          </strong> */}
          {/* <span style={{ color: 'var(--text-secondary)', fontSize: '0.76rem' }}>
            {isAr
              ? `تم استيفاء كافة المتطلبات المسبقة لمرحلة [${stageCode}] بنجاح.`
              : `All pipeline prerequisites for stage [${stageCode}] are verified.`}
          </span> */}
        </div>

        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
        {
            /* 
          checks.map((c) => (
            <span
              key={c.id}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.7rem',
                padding: '0.15rem 0.45rem',
                borderRadius: '4px',
                background: 'rgba(20, 184, 166, 0.12)',
                color: 'var(--accent-teal)',
                border: '1px solid rgba(20, 184, 166, 0.25)',
              }}
            >
              [{c.stageCode}] {c.id.toUpperCase()}: OK
            </span>
          ))*/
        }
            
        </div>
      </div>
    );
  }

  // Failed Prerequisites Alert Card
  const primaryFailed = failedChecks[0];

  return (
    <div
      style={{
        background: 'rgba(245, 158, 11, 0.07)',
        border: '1px solid var(--accent-amber)',
        borderInlineStart: '5px solid var(--accent-amber)',
        borderRadius: 'var(--radius-md, 8px)',
        padding: '1.15rem 1.35rem',
        marginBottom: '1.75rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem',
        boxShadow: '0 4px 16px rgba(245, 158, 11, 0.08)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.65rem' }}>
          <div
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '6px',
              background: 'rgba(245, 158, 11, 0.18)',
              color: 'var(--accent-amber)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 900,
              fontSize: '0.9rem',
              fontFamily: 'var(--font-mono)',
              flexShrink: 0,
            }}
          >
            !
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
              {isAr ? `تنبيه التحقق الأكاديمي لمرحلة [${stageCode}] ${stageName || ''}` : `Prerequisite Guardrail for Stage [${stageCode}]`}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.45 }}>
              {isAr ? primaryFailed.requirement_ar : primaryFailed.requirement_en}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={() => setActiveModule(primaryFailed.stageId)}
            className="btn btn-primary btn-sm"
            style={{ fontSize: '0.78rem', fontWeight: 700, padding: '0.45rem 0.85rem' }}
          >
            {isAr ? primaryFailed.action_ar : primaryFailed.action_en}
          </button>

          <button
            type="button"
            onClick={() => setBypassed(true)}
            className="btn btn-secondary btn-sm"
            style={{ fontSize: '0.74rem', padding: '0.45rem 0.65rem' }}
            title={isAr ? 'المتابعة بنمط المحاكاة التفاعلية دون حظر' : 'Bypass and continue in preview simulation mode'}
          >
            {isAr ? 'متابعة كمعاينة استكشافية' : 'Preview Mode'}
          </button>
        </div>
      </div>

      {/* Checklist summary */}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', borderTop: '1px dashed var(--border-subtle)', paddingTop: '0.65rem' }}>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {isAr ? 'سلسلة الاعتماد الأكاديمي:' : 'Pipeline Dependencies:'}
        </span>
        {checks.map((c) => (
          <span
            key={c.id}
            style={{
              fontSize: '0.72rem',
              color: c.status ? 'var(--accent-teal)' : 'var(--accent-amber)',
              fontWeight: 700,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <span>[{c.stageCode}] {isAr ? c.name_ar : c.name_en}:</span>
            <span>{c.status ? (isAr ? 'مكتمل' : 'Passed') : (isAr ? 'مطلوب' : 'Pending')}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
