// ==============================================================================
// DyTrStudio Training Arena Studio (Stage 04)
// Zero Emojis Policy - Visual Animated Optimization Loop - Bilingual RTL/LTR
// Complete Section 04.3 Upgrade: Seek Sliders + Direct Numeric Inputs + Standard Presets
// Responsive Zero-Overflow Layout with Deep Academic Tooltips (TermTooltip)
// ==============================================================================

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';
import { academicDictionary } from '../../utils/academicDictionary';
import TermTooltip from '../common/TermTooltip';
import SectionCard from '../common/SectionCard';
import StageReadinessBanner from '../common/StageReadinessBanner';
import NeuralTrainingLoopAnimator from './NeuralTrainingLoopAnimator';
import { buildCompositeFeatureString } from '../../utils/featureEngineering';
import { ArrowLeft, ArrowRight } from 'lucide-react';

export default function TrainingArenaView() {
  const {
    language,
    modelConfig,
    trainingConfig,
    updateTrainingConfig,
    capabilities,
    setActiveModule,
  } = useStudioStore();

  const { currentDataset, datasetName } = useDatasetStore();

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  // --- Hyperparameters State from Persistent Studio Store ---
  const learningRate = trainingConfig?.learningRate ?? 0.0003;
  const batchSize = trainingConfig?.batchSize ?? 16;
  const epochs = trainingConfig?.epochs ?? 4;
  const weightDecay = trainingConfig?.weightDecay ?? 0.01;
  const optimizerType = trainingConfig?.optimizerType ?? 'adamw'; // 'adamw' | 'sgd'
  const schedulerType = trainingConfig?.schedulerType ?? 'cosine'; // 'cosine' | 'linear' | 'constant'
  const activePreset = trainingConfig?.activePreset ?? 'balanced'; // 'balanced' | 'low_vram' | 'high_precision' | 'classic_sgd'
  const trainingRegime = trainingConfig?.trainingRegime ?? 'finetuning'; // 'finetuning' | 'scratch'
  const trainRatio = trainingConfig?.trainRatio ?? 80;
  const valRatio = trainingConfig?.valRatio ?? 10;
  const saveBestModel = trainingConfig?.saveBestModel ?? true;
  const saveMetric = trainingConfig?.saveMetric ?? 'val_loss';
  const checkpointFrequency = trainingConfig?.checkpointFrequency ?? 1;
  const earlyStopping = trainingConfig?.earlyStopping ?? true;
  const earlyStoppingPatience = trainingConfig?.earlyStoppingPatience ?? 3;

  const selectedDevice = trainingConfig?.selectedDevice ?? 'auto'; // 'auto' | 'cuda' | 'cpu'
  const isCudaAvailable = Boolean(capabilities?.cuda_available);
  const effectiveDevice = selectedDevice === 'cuda'
    ? 'cuda'
    : selectedDevice === 'cpu'
    ? 'cpu'
    : (isCudaAvailable ? 'cuda' : 'cpu');

  const setSelectedDevice = (dev) => updateTrainingConfig({ selectedDevice: dev });
  const setLearningRate = (v) => updateTrainingConfig({ learningRate: v });
  const setBatchSize = (v) => updateTrainingConfig({ batchSize: v });
  const setEpochs = (v) => updateTrainingConfig({ epochs: v });
  const setWeightDecay = (v) => updateTrainingConfig({ weightDecay: v });
  const setOptimizerType = (v) => updateTrainingConfig({ optimizerType: v });
  const setSchedulerType = (v) => updateTrainingConfig({ schedulerType: v });
  const setActivePreset = (v) => updateTrainingConfig({ activePreset: v });
  const setTrainingRegime = (v) => updateTrainingConfig({ trainingRegime: v });
  const setTrainRatio = (v) => updateTrainingConfig({ trainRatio: v });
  const setValRatio = (v) => updateTrainingConfig({ valRatio: v });
  const setSaveBestModel = (v) => updateTrainingConfig({ saveBestModel: v });
  const setSaveMetric = (v) => updateTrainingConfig({ saveMetric: v });
  const setCheckpointFrequency = (v) => updateTrainingConfig({ checkpointFrequency: v });
  const setEarlyStopping = (v) => updateTrainingConfig({ earlyStopping: v });
  const setEarlyStoppingPatience = (v) => updateTrainingConfig({ earlyStoppingPatience: v });
  const testRatio = Math.max(0, 100 - trainRatio - valRatio);

  const totalDatasetRows = currentDataset?.length || 50;
  const trainCount = Math.round((totalDatasetRows * trainRatio) / 100);
  const valCount = Math.round((totalDatasetRows * valRatio) / 100);
  const testCount = Math.max(0, totalDatasetRows - trainCount - valCount);
  const stepsPerEpoch = Math.max(1, Math.ceil(trainCount / batchSize));
  const totalTrainingSteps = epochs * stepsPerEpoch;

  // --- Standard Recognized Presets ---
  const presets = [
    {
      id: 'balanced',
      name_ar: 'النموذج القياسي المتوازن (موصى به للمحولات)',
      name_en: 'Standard Balanced (Recommended for Transformers)',
      tag_ar: 'الأفضل لمعظم المهام',
      tag_en: 'Sweet Spot',
      lr: 0.0003,
      batch: 16,
      epochs: trainingRegime === 'scratch' ? Math.max(30, Math.ceil(120 / stepsPerEpoch)) : 4,
      decay: 0.01,
      opt: 'adamw',
      sched: 'cosine',
      desc_ar: trainingRegime === 'scratch'
        ? 'تم تكييف عدد الحقب تلقائياً لتعويض قلة عينات البيانات وتحقيق ~120 خطوة تحديث لتعديل الأوزان العشوائية.'
        : 'التوليفة القياسية العالمية المثبتة في أبحاث المحولات (BERT/T5): تحقق سرعة تقارب مثالية واستقراراً عالياً دون تشتت الأوزان.',
      desc_en: 'The empirical standard in transformer literature (BERT/T5): optimal convergence speed and stability.',
    },
    {
      id: 'low_vram',
      name_ar: 'النمط الاقتصادي للأجهزة المحدودة',
      name_en: 'Low VRAM / Resource-Constrained',
      tag_ar: 'اقتصادي وسريع',
      tag_en: 'Fast & Light',
      lr: 0.0005,
      batch: 8,
      epochs: trainingRegime === 'scratch' ? Math.max(25, Math.ceil(100 / stepsPerEpoch)) : 3,
      decay: 0.01,
      opt: 'adamw',
      sched: 'linear',
      desc_ar: 'حجم دفعة صغير (8) يخفض استهلاك ذاكرة الفيديو والرام إلى النصف، مع رفع طفيف لسرعة التعلم لتعويض تباين التدرجات.',
      desc_en: 'Batch size 8 cuts memory consumption in half while slightly raising LR to balance gradient variance.',
    },
    {
      id: 'high_precision',
      name_ar: 'الضبط الدقيق عالي الحساسية',
      name_en: 'High-Precision Fine-Tuning',
      tag_ar: 'دقة قصوى ومحافظة',
      tag_en: 'Maximum Precision',
      lr: 0.0001,
      batch: 32,
      epochs: trainingRegime === 'scratch' ? Math.max(40, Math.ceil(150 / stepsPerEpoch)) : 6,
      decay: 0.02,
      opt: 'adamw',
      sched: 'cosine',
      desc_ar: 'خطوات تعلم فائقة الدقة مع دفعة عريضة (32) لمنع الانجراف الحاد في متجهات التضمين اللغوية الحساسة وحفظ التعميم.',
      desc_en: 'Conservative step sizes with large batch (32) preventing aggressive parameter drift on sensitive embeddings.',
    },
    {
      id: 'classic_sgd',
      name_ar: 'نمط الانحدار التقليدي (SGD + Momentum)',
      name_en: 'Classical SGD with Momentum',
      tag_ar: 'نمط كلاسيكي مقارن',
      tag_en: 'Classical Baseline',
      lr: 0.01,
      batch: 16,
      epochs: trainingRegime === 'scratch' ? Math.max(50, Math.ceil(180 / stepsPerEpoch)) : 5,
      decay: 0.0001,
      opt: 'sgd',
      sched: 'constant',
      desc_ar: 'انحدار التدرج العشوائي الكلاسيكي مع زخم حركي: مفيد للمقارنة الأكاديمية بين المحسنات التكيفية والتقليدية.',
      desc_en: 'Standard Stochastic Gradient Descent with momentum for baseline comparisons.',
    },
  ];

  const applyPreset = (preset) => {
    setActivePreset(preset.id);
    setLearningRate(preset.lr);
    setBatchSize(preset.batch);
    setEpochs(preset.epochs);
    setWeightDecay(preset.decay);
    setOptimizerType(preset.opt);
    setSchedulerType(preset.sched);
  };

  const handleRegimeChange = (newRegime) => {
    setTrainingRegime(newRegime);
    const targetPreset = presets.find((p) => p.id === activePreset) || presets[0];
    const newEpochs = newRegime === 'scratch'
      ? Math.max(30, Math.ceil(120 / stepsPerEpoch))
      : (targetPreset.id === 'high_precision' ? 6 : targetPreset.id === 'low_vram' ? 3 : targetPreset.id === 'classic_sgd' ? 5 : 4);
    setEpochs(newEpochs);
  };

  // Sample sentence from active dataset for tokenization preview (respecting active feature engineering and text_column)
  const activeTask = modelConfig.tasks?.[0] || {};
  const activeTextCol = activeTask.text_column || 'content';
  const activeLabelCol = activeTask.label_column || 'category';
  const sampleSentence = currentDataset && currentDataset.length > 0
    ? buildCompositeFeatureString(currentDataset[0], activeTask) || (currentDataset[0][activeTextCol] || currentDataset[0].content || currentDataset[0].headline || currentDataset[0].text || 'كشفت المؤتمرات العالمية عن معالجات عصبية دقيقة تقلل استهلاك الطاقة بنسبة 40% في مراكز البيانات.')
    : 'كشفت المؤتمرات العالمية عن معالجات عصبية دقيقة تقلل استهلاك الطاقة بنسبة 40% في مراكز البيانات.';

  const getLRCommentary = () => {
    if (learningRate < 0.00008) {
      return isAr
        ? 'معدل حذر جداً: تقارب فائق الاستقرار لكنه يتطلب حقباً تدريبية أطول.'
        : 'Ultra-cautious: highly stable convergence but requires longer training horizon.';
    }
    if (learningRate <= 0.0006) {
      return isAr
        ? 'معدل متوازن وموصى به (0.0001 - 0.0005): المدى المعياري لنماذج المحولات (BERT/T5/LLaMA).'
        : 'Optimal transformer standard: optimal tradeoff between convergence speed and stability.';
    }
    return isAr
      ? 'معدل سريع وهجومي: قد يقفز فوق النقطة الصغرى لدالة الخسارة ويسبب تشتت الأوزان.'
      : 'Aggressive step size: risks overshooting minimal loss landscape.';
  };

  return (
    <div className="container" style={{ maxWidth: '1080px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Studio Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.4rem' }}>
          <span className="badge badge-teal">[04] {d.module_arena}</span>
          <span className="badge badge-neutral">
            {isAr ? 'دورة التعلم والتحسين البصري' : 'Visual Optimization Engine'}
          </span>
        </div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
          {isAr ? 'التدريب وبناء وتحديث الأوزان' : 'Training Arena: Interactive Weight Optimization'}
        </h1>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
          {isAr
            ? 'محاكاة مرئية متدفقة: انسياب الإشارات، واحتساب الخطأ، وتعديل مصفوفات الأوزان خطوة بخطوة.'
            : 'Visual training stream: forward activations, loss feedback, and step-by-step weight calibration.'}
        </p>
      </div>

      {/* Academic Prerequisite Guardrail */}
      <StageReadinessBanner
        stageCode="04"
        stageName={d.module_arena}
        requiredStages={['dataprep', 'builder']}
      />

      {/* Pipeline Handshake Connection Banner */}
      <div
        style={{
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border-subtle)',
          borderInlineStart: '4px solid var(--accent-teal)',
          borderRadius: 'var(--radius-md, 8px)',
          padding: '0.9rem 1.2rem',
          marginBottom: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.65rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
          <span className="badge badge-teal" style={{ fontSize: '0.72rem', fontWeight: 800 }}>
            {isAr ? 'حزمة التدريب المعتمدة' : 'Verified Package'}
          </span>
          <strong style={{ color: 'var(--text-primary)', fontSize: '0.9rem', fontFamily: 'var(--font-mono)' }}>
            {datasetName || 'dataset.csv'}
          </strong>
          <div
            dir="ltr"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: 'var(--bg-secondary)',
              padding: '0.15rem 0.55rem',
              borderRadius: '4px',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <span style={{ fontSize: '0.72rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
              {activeTextCol}
            </span>
            <svg width="12" height="8" viewBox="0 0 14 10" fill="none" style={{ color: 'var(--accent-teal)', flexShrink: 0 }}>
              <path d="M1 5H13M13 5L9 1M13 5L9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span style={{ fontSize: '0.72rem', color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
              {activeLabelCol}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          {totalDatasetRows} {isAr ? 'صفاً' : 'rows'} · {modelConfig.num_layers || 4}L · {modelConfig.embed_dim || 768}d · {modelConfig.tasks?.[0]?.task_type || 'sentence_classification'}
        </div>
      </div>

      {/* Section 04.1: Data Partitioning & Animated Token Stream */}
      <SectionCard
        sectionCode="04.1"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span>{isAr ? 'تقسيم البيانات وتشكيل حزم الدفعات' : 'Data Partitioning & Mini-Batch Slicing'}</span>
            <TermTooltip termKey="data_partitioning" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'عزل جزء للاختبار النهائي لضمان الفهم الحقيقي وعدم الحفظ الصم، مع معاينة حية لانسياب التوكنات.'
            : 'Isolating test partitions prevents overfitting and provides unbiased generalization metrics.'
        }
      >
        {/* Split Distribution Bar */}
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {isAr ? 'نسب التوزيع الحالية:' : 'Current Distribution:'}
            </span>
            <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.78rem' }}>
              <span style={{ color: 'var(--accent-teal)', fontWeight: 700 }}>
                {isAr ? 'تدريب' : 'Train'}: {trainRatio}% ({trainCount})
              </span>
              <span style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>
                {isAr ? 'تحقق' : 'Val'}: {valRatio}% ({valCount})
              </span>
              <span style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>
                {isAr ? 'اختبار' : 'Test'}: {testRatio}% ({testCount})
              </span>
            </div>
          </div>

          <div
            style={{
              height: '12px',
              borderRadius: '6px',
              overflow: 'hidden',
              display: 'flex',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ width: `${trainRatio}%`, background: 'var(--accent-teal)', transition: 'width 0.3s ease' }} title={`Train: ${trainRatio}%`} />
            <div style={{ width: `${valRatio}%`, background: 'var(--accent-amber)', transition: 'width 0.3s ease' }} title={`Val: ${valRatio}%`} />
            <div style={{ width: `${testRatio}%`, background: 'var(--accent-emerald)', transition: 'width 0.3s ease' }} title={`Test: ${testRatio}%`} />
          </div>

          {/* Dual Seek Slider + Direct Numeric Inputs Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            {/* Train Split */}
            <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {isAr ? 'بيانات التدريب (Train)' : 'Training Partition'}
                </span>
                <span className="badge badge-teal" style={{ fontSize: '0.7rem' }}>Train</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <input
                  type="range"
                  dir="ltr"
                  min="50"
                  max="90"
                  step="5"
                  value={trainRatio}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setTrainRatio(val);
                    if (val + valRatio > 95) setValRatio(95 - val);
                  }}
                  className="slider-input"
                  style={{ flex: 1, margin: 0, direction: 'ltr' }}
                />
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.45rem', flexShrink: 0 }}>
                  <input
                    type="number"
                    min="50"
                    max="90"
                    step="1"
                    value={trainRatio}
                    onChange={(e) => {
                      const val = Math.min(90, Math.max(50, parseInt(e.target.value) || 50));
                      setTrainRatio(val);
                      if (val + valRatio > 95) setValRatio(95 - val);
                    }}
                    style={{
                      width: '45px',
                      background: 'transparent',
                      border: 'none',
                      outline: 'none',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.82rem',
                      fontWeight: 700,
                      color: 'var(--accent-teal)',
                      padding: 0,
                    }}
                  />
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>%</span>
                </div>
              </div>
            </div>

            {/* Validation Split */}
            <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {isAr ? 'بيانات التحقق (Validation)' : 'Validation Partition'}
                </span>
                <span className="badge badge-neutral" style={{ fontSize: '0.7rem' }}>Val</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <input
                  type="range"
                  dir="ltr"
                  min="5"
                  max="30"
                  step="5"
                  value={valRatio}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (trainRatio + val <= 95) setValRatio(val);
                  }}
                  className="slider-input"
                  style={{ flex: 1, margin: 0, direction: 'ltr' }}
                />
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.45rem', flexShrink: 0 }}>
                  <input
                    type="number"
                    min="5"
                    max="30"
                    step="1"
                    value={valRatio}
                    onChange={(e) => {
                      const val = Math.min(30, Math.max(5, parseInt(e.target.value) || 5));
                      if (trainRatio + val <= 95) setValRatio(val);
                    }}
                    style={{
                      width: '45px',
                      background: 'transparent',
                      border: 'none',
                      outline: 'none',
                      textAlign: 'center',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.82rem',
                      fontWeight: 700,
                      color: 'var(--accent-amber)',
                      padding: 0,
                    }}
                  />
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Section 04.2: Upgraded Optimization Hyperparameters & Presets */}
      <SectionCard
        sectionCode="04.2"
        title={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span>{isAr ? 'معلمات التدريب والمحسن الرياضي (Hyperparameters Engine)' : 'Hyperparameters & Optimization Engine'}</span>
            <TermTooltip termKey="optimizer_adamw" size="sm" hideLabel={true} />
          </span>
        }
        subtitle={
          isAr
            ? 'اختر من التوليفات المعيارية الموصى بها علمياً، أو خصص معدل التعلم وحجم الدفعة بدقة عبر أشرطة السحب والإدخال المباشر.'
            : 'Select from empirical standard presets or fine-tune parameters directly via dual sliders and numeric inputs.'
        }
      >
        {/* Training Strategy Regime Switcher */}
        <div
          style={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-md, 8px)',
            padding: '0.9rem 1.1rem',
            marginBottom: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {isAr ? 'استراتيجية التدريب وطبيعة الأوزان:' : 'Training Paradigm & Weight Nature:'}
            </span>
            <div style={{ display: 'inline-flex', background: 'var(--bg-secondary)', padding: '0.2rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <button
                type="button"
                onClick={() => handleRegimeChange('finetuning')}
                style={{
                  padding: '0.35rem 0.75rem',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  borderRadius: '5px',
                  border: 'none',
                  cursor: 'pointer',
                  background: trainingRegime === 'finetuning' ? 'var(--accent-teal)' : 'transparent',
                  color: trainingRegime === 'finetuning' ? '#ffffff' : 'var(--text-secondary)',
                  transition: 'all 0.15s ease',
                }}
              >
                {isAr ? 'الضبط الدقيق (Fine-Tuning: 3-6 حقب)' : 'Fine-Tuning (3-6 Epochs)'}
              </button>
              <button
                type="button"
                onClick={() => handleRegimeChange('scratch')}
                style={{
                  padding: '0.35rem 0.75rem',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  borderRadius: '5px',
                  border: 'none',
                  cursor: 'pointer',
                  background: trainingRegime === 'scratch' ? 'var(--accent-teal)' : 'transparent',
                  color: trainingRegime === 'scratch' ? '#ffffff' : 'var(--text-secondary)',
                  transition: 'all 0.15s ease',
                }}
              >
                {isAr ? 'تدريب من الصفر / بيانات صغيرة (Scratch: 30-100 حقبة)' : 'From Scratch / Small Data (30-100 Epochs)'}
              </button>
            </div>
          </div>
          <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {trainingRegime === 'finetuning'
              ? (isAr
                ? 'الضبط الدقيق (Transfer Learning): النموذج يمتلك مسبقاً مليارات العلاقات اللغوية، لذا نحتاج فقط 3-6 حقب لتكييف الرأس النهائي وتفادي التدمير الكارثي للأوزان (Catastrophic Forgetting).'
                : 'Fine-Tuning: Leverages pretrained representations; 3-6 epochs suffice to align task heads without catastrophic forgetting.')
              : (isAr
                ? 'التدريب من الصفر (Random Init): الأوزان تبدأ عشوائية تماماً. في البيانات الصغيرة (50 صفاً مثلاً = 3 خطوات/حقبة)، 4 حقب تعطي 12 خطوة فقط وهو غير كافٍ للتقارب، لذا نرفع الحقب إلى 30-100 لتحقيق خطوات تعلم كافية.'
                : 'Training From Scratch: Weights start randomly. Small datasets yield few steps per epoch, requiring 30-100 epochs to accumulate sufficient optimization steps.')}
          </div>
        </div>

        {/* Recommended Standard Presets Grid */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ fontSize: '0.84rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.65rem' }}>
            {isAr ? 'القوالب والتوليفات القياسية الموصى بها (Standard Archetypes):' : 'Recommended Optimization Presets:'}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '0.75rem' }}>
            {presets.map((preset) => {
              const isSelected = activePreset === preset.id;
              return (
                <div
                  key={preset.id}
                  onClick={() => applyPreset(preset)}
                  style={{
                    background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                    border: `1px solid ${isSelected ? 'var(--accent-teal)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-md, 8px)',
                    padding: '0.85rem',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: isSelected ? '0 0 0 1px var(--accent-teal), 0 4px 12px rgba(14, 165, 233, 0.1)' : 'none',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.35rem', marginBottom: '0.4rem' }}>
                      <strong style={{ fontSize: '0.84rem', color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                        {isAr ? preset.name_ar : preset.name_en}
                      </strong>
                      <span className={`badge ${isSelected ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.65rem', whiteSpace: 'nowrap' }}>
                        {isAr ? preset.tag_ar : preset.tag_en}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', lineHeight: 1.45, marginBottom: '0.65rem' }}>
                      {isAr ? preset.desc_ar : preset.desc_en}
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.45rem', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    <span>lr: {preset.lr}</span>
                    <span>· B: {preset.batch}</span>
                    <span>· E: {preset.epochs}</span>
                    <span>· {preset.opt.toUpperCase()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dual Input Controls Grid (Seek Sliders + Direct Numeric Inputs with Zero Overflow) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.75rem', marginBottom: '1rem', width: '100%', boxSizing: 'border-box' }}>
          {/* 1. Learning Rate (η) */}
          <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0, minWidth: 0, boxSizing: 'border-box' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'معدل التعلم' : 'Learning Rate'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(η)</bdi>
                <TermTooltip termKey="learning_rate" size="sm" hideLabel={true} />
              </span>
              <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>η</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', minWidth: 0 }}>
              <input
                type="range"
                dir="ltr"
                min="0.00001"
                max="0.002"
                step="0.00001"
                value={learningRate}
                onChange={(e) => {
                  setLearningRate(parseFloat(e.target.value));
                  setActivePreset('custom');
                }}
                className="slider-input"
                style={{ flex: 1, minWidth: '50px', margin: 0, direction: 'ltr' }}
              />
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.15rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.35rem', flexShrink: 0 }}>
                <span dir="ltr" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>η=</span>
                <input
                  type="number"
                  min="0.00001"
                  max="0.002"
                  step="0.00005"
                  value={learningRate}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value) || 0.0001;
                    setLearningRate(val);
                    setActivePreset('custom');
                  }}
                  style={{
                    width: '68px',
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: 'var(--accent-teal)',
                    padding: 0,
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--accent-amber)', lineHeight: 1.4 }}>
              {getLRCommentary()}
            </div>
          </div>

          {/* 2. Batch Size (B) */}
          <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0, minWidth: 0, boxSizing: 'border-box' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'حجم الدفعة' : 'Batch Size'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(Batch)</bdi>
                <TermTooltip termKey="batch_size" size="sm" hideLabel={true} />
              </span>
              <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>B</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', minWidth: 0 }}>
              <input
                type="range"
                dir="ltr"
                min="4"
                max="64"
                step="4"
                value={batchSize}
                onChange={(e) => {
                  setBatchSize(parseInt(e.target.value));
                  setActivePreset('custom');
                }}
                className="slider-input"
                style={{ flex: 1, minWidth: '50px', margin: 0, direction: 'ltr' }}
              />
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.15rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.35rem', flexShrink: 0 }}>
                <span dir="ltr" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>B=</span>
                <input
                  type="number"
                  min="4"
                  max="64"
                  step="4"
                  value={batchSize}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 16;
                    setBatchSize(val);
                    setActivePreset('custom');
                  }}
                  style={{
                    width: '45px',
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: 'var(--accent-teal)',
                    padding: 0,
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--text-muted)' }}>
              {isAr ? `ينتج ${stepsPerEpoch} خطوات تحديث للأوزان في كل حقبة` : `Yields ${stepsPerEpoch} update steps per epoch`}
            </div>
          </div>

          {/* 3. Epochs (E) */}
          <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0, minWidth: 0, boxSizing: 'border-box' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'الحقب التدريبية' : 'Epochs'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(Epochs)</bdi>
                <TermTooltip termKey="epochs" size="sm" hideLabel={true} />
              </span>
              <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>E</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', minWidth: 0 }}>
              <input
                type="range"
                dir="ltr"
                min="1"
                max={trainingRegime === 'scratch' ? 100 : 50}
                step="1"
                value={epochs}
                onChange={(e) => {
                  setEpochs(parseInt(e.target.value));
                  setActivePreset('custom');
                }}
                className="slider-input"
                style={{ flex: 1, minWidth: '50px', margin: 0, direction: 'ltr' }}
              />
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.15rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.35rem', flexShrink: 0 }}>
                <span dir="ltr" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>E=</span>
                <input
                  type="number"
                  min="1"
                  max="200"
                  step="1"
                  value={epochs}
                  onChange={(e) => {
                    const val = parseInt(e.target.value) || 4;
                    setEpochs(val);
                    setActivePreset('custom');
                  }}
                  style={{
                    width: '45px',
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: 'var(--accent-teal)',
                    padding: 0,
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--text-muted)' }}>
              {isAr ? `إجمالي خطوات التدريب: ${totalTrainingSteps} خطوة` : `Total steps: ${totalTrainingSteps}`}
            </div>
          </div>

          {/* 4. Weight Decay Regularization (λ) */}
          <div className="form-group" style={{ background: 'var(--bg-secondary)', padding: '0.85rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: 0, minWidth: 0, boxSizing: 'border-box' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'تخفيف الوزن' : 'Weight Decay'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(Weight Decay)</bdi>
                <TermTooltip termKey="weight_update" size="sm" hideLabel={true} />
              </span>
              <span className="badge badge-neutral" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>λ</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', minWidth: 0 }}>
              <input
                type="range"
                dir="ltr"
                min="0.0"
                max="0.1"
                step="0.005"
                value={weightDecay}
                onChange={(e) => {
                  setWeightDecay(parseFloat(e.target.value));
                  setActivePreset('custom');
                }}
                className="slider-input"
                style={{ flex: 1, minWidth: '50px', margin: 0, direction: 'ltr' }}
              />
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.15rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.2rem 0.35rem', flexShrink: 0 }}>
                <span dir="ltr" style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>λ=</span>
                <input
                  type="number"
                  min="0.0"
                  max="0.1"
                  step="0.005"
                  value={weightDecay}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value) || 0.01;
                    setWeightDecay(val);
                    setActivePreset('custom');
                  }}
                  style={{
                    width: '55px',
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: 'var(--accent-teal)',
                    padding: 0,
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--text-muted)' }}>
              {isAr ? 'يمنع تضخم الأوزان ويحمي النموذج من فرط التخصيص' : 'Penalizes large weights to prevent overfitting'}
            </div>
          </div>
        </div>

        {/* Optimizer & LR Scheduler Selector Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.85rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', width: '100%', boxSizing: 'border-box' }}>
          {/* Optimizer Type */}
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'المحسن الرياضي' : 'Optimizer:'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(Optimizer)</bdi>
                <TermTooltip termKey="optimizer_adamw" size="sm" hideLabel={true} />
              </span>
              <span className="badge badge-teal" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                {optimizerType.toUpperCase()}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem' }}>
              <button
                type="button"
                onClick={() => {
                  setOptimizerType('adamw');
                  setActivePreset('custom');
                }}
                className={`btn btn-sm ${optimizerType === 'adamw' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ textAlign: 'center', fontSize: '0.74rem', padding: '0.35rem 0.25rem' }}
              >
                AdamW (موصى به)
              </button>
              <button
                type="button"
                onClick={() => {
                  setOptimizerType('sgd');
                  setActivePreset('custom');
                }}
                className={`btn btn-sm ${optimizerType === 'sgd' ? 'btn-primary' : 'btn-secondary'}`}
                style={{ textAlign: 'center', fontSize: '0.74rem', padding: '0.35rem 0.25rem' }}
              >
                SGD + Momentum
              </button>
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--text-muted)', marginTop: '0.4rem', lineHeight: 1.45 }}>
              {optimizerType === 'adamw'
                ? (isAr ? 'يضبط سرعة التعلم لكل وزن تلقائياً ويمنع تضخم الأوزان لضمان تعميم قوي.' : 'Adaptive momentum with decoupled weight decay regularization.')
                : (isAr ? 'انحدار التدرج الكلاسيكي مع الزخم الحركي لتسريع الهبوط.' : 'Classical gradient descent with momentum acceleration.')}
            </div>
          </div>

          {/* Learning Rate Scheduler */}
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
                <span>{isAr ? 'مجدول معدل التعلم' : 'LR Schedule Curve:'}</span>
                <bdi dir="ltr" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>(LR Scheduler)</bdi>
              </span>
              <span className="badge badge-neutral" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                {schedulerType.toUpperCase()}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.35rem' }}>
              {[
                { id: 'cosine', label_ar: 'Cosine (موصى به)', label_en: 'Cosine' },
                { id: 'linear', label_ar: 'Linear Warmup', label_en: 'Linear' },
                { id: 'constant', label_ar: 'ثابت Constant', label_en: 'Constant' },
              ].map((sch) => (
                <button
                  key={sch.id}
                  type="button"
                  onClick={() => setSchedulerType(sch.id)}
                  className={`btn btn-sm ${schedulerType === sch.id ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ fontSize: '0.7rem', padding: '0.35rem 0.15rem', textAlign: 'center' }}
                >
                  {isAr ? sch.label_ar : sch.label_en}
                </button>
              ))}
            </div>
            <div style={{ fontSize: '0.71rem', color: 'var(--text-muted)', marginTop: '0.4rem', lineHeight: 1.45 }}>
              {schedulerType === 'cosine'
                ? (isAr ? 'يبدأ بسرعة مثالية ثم يخفض السرعة بنعومة في نهاية التدريب لاستقرار التوقع النهائي.' : 'Smooth cosine annealing decay toward final parameter optimum.')
                : schedulerType === 'linear'
                ? (isAr ? 'تسخين تدريجي لمعدل التعلم في البداية ثم انخفاض خطي مستمر.' : 'Linear warmup followed by linear decay.')
                : (isAr ? 'سرعة تعلم ثابتة طوال الحقب التدريبية.' : 'Constant step size across all iterations.')}
            </div>
          </div>
        </div>

        {/* Checkpointing & Best Model Strategy Panel */}
        <div
          style={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-medium)',
            borderInlineStart: '4px solid var(--accent-emerald)',
            borderRadius: 'var(--radius-md, 8px)',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            marginTop: '0.5rem',
            width: '100%',
            boxSizing: 'border-box',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.84rem', fontWeight: 800, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
              <span>{isAr ? 'استراتيجية حفظ أفضل نموذج ونقاط الفحص' : 'Model Checkpointing & Best Model Strategy:'}</span>
              <bdi dir="ltr" style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>(Model Checkpointing)</bdi>
              <TermTooltip termKey="model_checkpointing" size="sm" hideLabel={true} />
            </span>
            <span className="badge badge-emerald" style={{ fontSize: '0.68rem' }}>
              {saveBestModel ? (isAr ? 'حفظ الأفضل مفعّل' : 'Best Model Auto-Save') : (isAr ? 'حفظ دوري فقط' : 'Periodic Only')}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem', width: '100%', boxSizing: 'border-box' }}>
            {/* 1. Best Model Save Mode */}
            <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 0.85rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0, boxSizing: 'border-box' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {isAr ? 'معيار أفضل نموذج' : 'Best Model Criterion:'}
                </span>
                <input
                  type="checkbox"
                  checked={saveBestModel}
                  onChange={(e) => setSaveBestModel(e.target.checked)}
                  style={{ accentColor: 'var(--accent-emerald)', cursor: 'pointer' }}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.35rem' }}>
                <button
                  type="button"
                  disabled={!saveBestModel}
                  onClick={() => setSaveMetric('val_loss')}
                  className={`btn btn-sm ${saveMetric === 'val_loss' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ fontSize: '0.68rem', padding: '0.3rem 0.2rem', textAlign: 'center', whiteSpace: 'nowrap' }}
                >
                  {isAr ? 'أقل خطأ (Min Loss)' : 'Min Val Loss'}
                </button>
                <button
                  type="button"
                  disabled={!saveBestModel}
                  onClick={() => setSaveMetric('val_acc')}
                  className={`btn btn-sm ${saveMetric === 'val_acc' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ fontSize: '0.68rem', padding: '0.3rem 0.2rem', textAlign: 'center', whiteSpace: 'nowrap' }}
                >
                  {isAr ? 'أعلى دقة (Max Acc)' : 'Max Val Acc'}
                </button>
              </div>
            </div>

            {/* 2. Checkpoint Frequency */}
            <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 0.85rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0, boxSizing: 'border-box' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {isAr ? 'تكرار الحفظ الدوري:' : 'Save Frequency:'}
                </span>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.35rem' }}>
                  <input
                    type="number"
                    min="1"
                    max={Math.max(1, Math.min(20, epochs))}
                    step="1"
                    value={Math.min(checkpointFrequency, epochs)}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (!isNaN(val)) setCheckpointFrequency(Math.min(epochs, Math.max(1, val)));
                    }}
                    style={{ width: '35px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', fontWeight: 700, color: 'var(--accent-teal)', padding: 0 }}
                  />
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'حقبة' : 'E'}</span>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: 0 }}>
                <input
                  type="range"
                  dir="ltr"
                  min="1"
                  max={Math.max(1, Math.min(20, epochs))}
                  step="1"
                  value={Math.min(checkpointFrequency, epochs)}
                  onChange={(e) => setCheckpointFrequency(parseInt(e.target.value) || 1)}
                  className="slider-input"
                  style={{ flex: 1, minWidth: '40px', margin: 0, direction: 'ltr' }}
                />
              </div>
            </div>

            {/* 3. Early Stopping */}
            <div style={{ background: 'var(--bg-secondary)', padding: '0.75rem 0.85rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0, boxSizing: 'border-box' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {isAr ? 'التوقف المبكر' : 'Early Stopping:'}
                  <bdi dir="ltr" style={{ marginInlineStart: '0.25rem', fontSize: '0.7rem' }}>(Early Stopping)</bdi>
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  {earlyStopping && (
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.35rem' }}>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        step="1"
                        disabled={!earlyStopping}
                        value={earlyStoppingPatience}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (!isNaN(val)) setEarlyStoppingPatience(Math.min(10, Math.max(1, val)));
                        }}
                        style={{ width: '35px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', fontWeight: 700, color: 'var(--accent-teal)', padding: 0 }}
                      />
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'صبر' : 'patience'}</span>
                    </div>
                  )}
                  <input
                    type="checkbox"
                    checked={earlyStopping}
                    onChange={(e) => setEarlyStopping(e.target.checked)}
                    style={{ accentColor: 'var(--accent-teal)', cursor: 'pointer' }}
                  />
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', opacity: earlyStopping ? 1 : 0.45, minWidth: 0 }}>
                <input
                  type="range"
                  dir="ltr"
                  min="1"
                  max="10"
                  step="1"
                  disabled={!earlyStopping}
                  value={earlyStoppingPatience}
                  onChange={(e) => setEarlyStoppingPatience(parseInt(e.target.value) || 1)}
                  className="slider-input"
                  style={{ flex: 1, minWidth: '40px', margin: 0, direction: 'ltr' }}
                />
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* 3. SECTION 04.3: Final Training Matrix & Hardware Acceleration Plan */}
      <SectionCard
        sectionCode="04.3"
        title={isAr ? 'ملخص خطة التدريب والإطلاق النهائي' : 'Executive Training Plan & Launch Briefing'}
        subtitle={isAr ? 'المراجعة الشاملة لجميع محددات الجلسة والعتاد الحسابي قبل بدء التدريب في المرصد' : 'Comprehensive review of session hyper-parameters and compute hardware prior to execution'}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', width: '100%', boxSizing: 'border-box' }}>
          {/* Executive Matrix Grid (7 Metrics) */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
              gap: '0.75rem',
              width: '100%',
              boxSizing: 'border-box',
            }}
          >
            {/* 1. Partitions */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'تقسيم العينات' : 'Data Partitioning'}
                <bdi dir="ltr" style={{ marginInlineStart: '0.25rem', fontSize: '0.68rem' }}>(Data Slices)</bdi>
              </span>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                {trainCount} {isAr ? 'تدريب' : 'train'} ({trainRatio}%)
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {valCount} {isAr ? 'تحقق' : 'val'} · {testCount} {isAr ? 'اختبار' : 'test'}
              </div>
            </div>

            {/* 2. Batch & Epochs */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'الدفعة والحقب' : 'Batch & Epochs'}
                <bdi dir="ltr" style={{ marginInlineStart: '0.25rem', fontSize: '0.68rem' }}>(Batch & Epochs)</bdi>
              </span>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                B = {batchSize} · E = {epochs}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {stepsPerEpoch} {isAr ? 'خطوة/حقبة' : 'steps/epoch'}
              </div>
            </div>

            {/* 3. Total Steps */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box', border: `1px solid ${totalTrainingSteps < 20 ? 'var(--accent-amber)' : 'var(--accent-teal)'}` }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'إجمالي خطوات التحديث' : 'Total Steps'}
                <bdi dir="ltr" style={{ marginInlineStart: '0.25rem', fontSize: '0.68rem' }}>(Total Steps)</bdi>
              </span>
              <div style={{ fontSize: '0.95rem', fontWeight: 800, color: totalTrainingSteps < 20 ? 'var(--accent-amber)' : 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                {totalTrainingSteps} {isAr ? 'خطوة تحديث' : 'steps'}
              </div>
              <div style={{ fontSize: '0.72rem', color: totalTrainingSteps < 20 ? 'var(--accent-amber)' : 'var(--accent-emerald)', marginTop: '0.2rem' }}>
                {totalTrainingSteps < 20 ? (isAr ? 'خطوات محدودة' : 'Limited steps') : (isAr ? 'أفق تدريبي مثالي' : 'Optimal horizon')}
              </div>
            </div>

            {/* 4. Learning Rate & Decay */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'سرعة التعلم والانتظام' : 'Learning Rate & Regularization'}
              </span>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                η = {learningRate}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                λ = {weightDecay} · {schedulerType.toUpperCase()}
              </div>
            </div>

            {/* 5. Checkpointing Strategy */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'حفظ النموذج ونقاط الفحص' : 'Model Checkpointing'}
              </span>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                {saveBestModel ? (isAr ? `أفضل نموذج (${saveMetric === 'val_loss' ? 'أقل خطأ' : 'أعلى دقة'})` : `Best (${saveMetric})`) : (isAr ? 'حفظ دوري' : 'Periodic')}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {isAr ? `حفظ كل E${checkpointFrequency}` : `Every ${checkpointFrequency}E`} · {earlyStopping ? (isAr ? `صبر ${earlyStoppingPatience}E` : `Patience ${earlyStoppingPatience}`) : (isAr ? 'بلا توقف' : 'No ES')}
              </div>
            </div>

            {/* 6. Optimizer & Regime */}
            <div className="diagram-box" style={{ padding: '0.85rem', textAlign: 'center', alignItems: 'center', minWidth: 0, width: '100%', boxSizing: 'border-box' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                {isAr ? 'المحسن ونمط التدريب' : 'Optimizer & Paradigm'}
              </span>
              <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {optimizerType.toUpperCase()}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {trainingRegime === 'finetuning' ? (isAr ? 'ضبط دقيق (Fine-Tuning)' : 'Fine-Tuning') : (isAr ? 'تدريب من البداية (Scratch)' : 'Scratch')}
              </div>
            </div>

            {/* 7. Hardware & Time */}
            <div
              className="diagram-box"
              style={{
                padding: '0.85rem',
                textAlign: 'center',
                alignItems: 'center',
                justifyContent: 'space-between',
                minWidth: 0,
                width: '100%',
                boxSizing: 'border-box',
                overflow: 'hidden',
                border: effectiveDevice === 'cuda' ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: '0.3rem', gap: '0.25rem', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span>{isAr ? 'العتاد والزمن المتوقع' : 'Hardware & Time'}</span>
                  <TermTooltip termKey="hardware_compute" size="sm" hideLabel={true} />
                </span>
                <span className={`badge ${effectiveDevice === 'cuda' ? 'badge-emerald' : 'badge-amber'}`} style={{ fontSize: '0.62rem', whiteSpace: 'nowrap' }}>
                  {effectiveDevice === 'cuda' ? (isAr ? 'GPU مفعّل' : 'GPU Active') : (isAr ? 'نمط CPU' : 'CPU Mode')}
                </span>
              </div>
              <div
                style={{
                  fontSize: '0.74rem',
                  fontWeight: 800,
                  color: effectiveDevice === 'cuda' ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                  maxWidth: '100%',
                  width: '100%',
                  overflowWrap: 'anywhere',
                  wordBreak: 'break-word',
                  lineHeight: 1.3,
                  textAlign: 'center',
                }}
              >
                {effectiveDevice === 'cuda' ? (isCudaAvailable ? (capabilities?.device_name || 'NVIDIA CUDA GPU') : 'GPU Tensor Engine') : 'CPU Host Neural Engine'}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', marginTop: '0.25rem' }}>
                {modelConfig.num_layers || 4}L · ~{effectiveDevice === 'cuda' ? Math.max(2, Math.round(totalTrainingSteps * 0.04)) : Math.max(5, Math.round(totalTrainingSteps * 0.35))}s {isAr ? 'متوقع' : 'est.'}
              </div>
            </div>
          </div>

          {/* Dedicated Hardware Diagnostics & Manual GPU / CPU Selector Banner */}
          <div
            style={{
              background: effectiveDevice === 'cuda' ? 'rgba(16, 185, 129, 0.05)' : 'rgba(245, 158, 11, 0.06)',
              border: effectiveDevice === 'cuda' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(245, 158, 11, 0.35)',
              borderInlineStart: effectiveDevice === 'cuda' ? '4px solid var(--accent-emerald)' : '4px solid var(--accent-amber)',
              borderRadius: 'var(--radius-md, 8px)',
              padding: '0.95rem 1.15rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.85rem',
              width: '100%',
              boxSizing: 'border-box',
            }}
          >
            {/* Header: Status + Device Toggle Switcher + 3-Level Explanation Button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.65rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
                <span
                  style={{
                    width: '9px',
                    height: '9px',
                    borderRadius: '50%',
                    background: effectiveDevice === 'cuda' ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                    display: 'inline-block',
                    flexShrink: 0,
                  }}
                />
                <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
                  <span>{isAr ? 'العتاد الحسابي المعتمد:' : 'Compute Hardware:'}</span>
                  <span style={{ color: effectiveDevice === 'cuda' ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
                    {effectiveDevice === 'cuda' ? (isAr ? 'تسريع الرسوميات' : 'GPU Accelerated') : (isAr ? 'المعالج المركزي' : 'Host CPU')}
                  </span>
                  <bdi dir="ltr" style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    ({effectiveDevice === 'cuda' ? (isCudaAvailable ? (capabilities?.device_name || 'NVIDIA CUDA') : 'GPU Tensor Engine') : 'CPU Mode'})
                  </bdi>
                </strong>
              </div>

              {/* User Device Switcher (Auto | GPU | CPU) + Explainer Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                {/* 3-Way Device Toggle */}
                <div style={{ display: 'inline-flex', background: 'var(--bg-secondary)', padding: '0.2rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
                  <button
                    type="button"
                    onClick={() => setSelectedDevice('auto')}
                    style={{
                      padding: '0.25rem 0.55rem',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      borderRadius: '4px',
                      border: 'none',
                      cursor: 'pointer',
                      background: selectedDevice === 'auto' ? 'var(--accent-teal)' : 'transparent',
                      color: selectedDevice === 'auto' ? '#ffffff' : 'var(--text-secondary)',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {isAr ? 'تلقائي (Auto)' : 'Auto'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedDevice('cuda')}
                    style={{
                      padding: '0.25rem 0.55rem',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      borderRadius: '4px',
                      border: 'none',
                      cursor: 'pointer',
                      background: selectedDevice === 'cuda' ? 'var(--accent-emerald)' : 'transparent',
                      color: selectedDevice === 'cuda' ? '#ffffff' : 'var(--text-secondary)',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    GPU (CUDA)
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedDevice('cpu')}
                    style={{
                      padding: '0.25rem 0.55rem',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      borderRadius: '4px',
                      border: 'none',
                      cursor: 'pointer',
                      background: selectedDevice === 'cpu' ? 'var(--accent-amber)' : 'transparent',
                      color: selectedDevice === 'cpu' ? '#ffffff' : 'var(--text-secondary)',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    CPU
                  </button>
                </div>

                {/* 3-Level Explanation Button */}
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                  <TermTooltip termKey="hardware_compute" customLabel={isAr ? 'دليل الفروق' : 'Guide'} size="sm" />
                </div>
              </div>
            </div>

            {/* Divided 3-Pillar Technical Highlights Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.65rem', width: '100%', boxSizing: 'border-box' }}>
              {/* 1. Operational Mode */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.6rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0 }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                  {isAr ? '1. النمط التشغيلي:' : '1. Operational Mode:'}
                </span>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, color: effectiveDevice === 'cuda' ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
                  {effectiveDevice === 'cuda'
                    ? (isAr ? 'تسريع متوازي عبر الأنوية (Parallel SIMD)' : 'Parallel SIMD Acceleration')
                    : (isAr ? 'معالجة تسلسلية عامة (Host CPU)' : 'Sequential Host CPU Execution')}
                </div>
              </div>

              {/* 2. Target Hardware */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.6rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0 }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                  {isAr ? '2. وحدة المعالجة المستخدة:' : '2. Target Processor:'}
                </span>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)', wordBreak: 'break-word' }}>
                  {effectiveDevice === 'cuda'
                    ? (isCudaAvailable ? (capabilities?.device_name || 'NVIDIA CUDA GPU') : (isAr ? 'محرك GPU التنسوري' : 'GPU Tensor Engine'))
                    : (isAr ? 'أنوية المعالج المركزي (Multi-Core)' : 'Host Multi-Core CPU')}
                </div>
              </div>

              {/* 3. Performance Throughput */}
              <div style={{ background: 'var(--bg-secondary)', padding: '0.6rem 0.75rem', borderRadius: '6px', border: '1px solid var(--border-subtle)', minWidth: 0 }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem' }}>
                  {isAr ? '3. كفاءة التدريب المتوقعة:' : '3. Training Throughput:'}
                </span>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, color: effectiveDevice === 'cuda' ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                  {effectiveDevice === 'cuda'
                    ? (isAr ? 'تسريع ~10x-50x لضرب المصفوفات' : '~10x-50x Matrix Contraction Speedup')
                    : (isAr ? 'سرعة قياسية معيارية' : 'Standard Baseline Speed')}
                </div>
              </div>
            </div>

            {/* Concise Epistemological Footnote */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.15rem' }}>
              <span style={{ fontSize: '0.73rem', color: 'var(--text-secondary)' }}>
                {isAr
                  ? 'الشروحات المعمارية العميقة، وقوانين أمدال (Amdahl) و Roofline لضرب المصفوفات، متاحة بالكامل عبر علامة الاستفهام (؟).'
                  : 'Deep architectural proofs, Amdahl speedup, and Roofline formulations are available in the 3-level guide (?)'}
              </span>
              <TermTooltip termKey="gpu_vs_cpu" customLabel={isAr ? 'مقارنة GPU مقابل CPU ؟' : 'GPU vs CPU ?'} size="sm" />
            </div>
          </div>

          {/* Unified Prominent Action Button */}
          <button
            type="button"
            onClick={() => setActiveModule('observatory')}
            className="btn btn-primary"
            style={{
              width: '100%',
              padding: '1.1rem',
              fontSize: '1.05rem',
              fontWeight: 800,
              borderRadius: 'var(--radius-md, 8px)',
              background: 'linear-gradient(135deg, var(--accent-teal) 0%, var(--accent-cyan) 100%)',
              boxShadow: '0 4px 18px var(--accent-teal-glow)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.65rem',
              cursor: 'pointer',
              border: 'none',
              color: '#ffffff',
            }}
          >
            <span>
              {isAr
                ? 'الانتقال إلى مراجعة الخطة وبدء التدريب في المرصد (المرحلة 05)'
                : 'Proceed to Pre-Flight Briefing & Launch Observatory (Stage 05)'}
            </span>
            <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          </button>
        </div>
      </SectionCard>

      {/* Module Navigation Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2.5rem', gap: '1rem', flexWrap: 'wrap' }}>
        <button
          onClick={() => setActiveModule('builder')}
          className="btn btn-secondary"
          style={{ padding: '0.75rem 1.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          <span>{isAr ? 'العودة إلى معمارية النموذج (المرحلة 03)' : 'Back to Architecture (Stage 03)'}</span>
        </button>

        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {isAr ? 'المرحلة 04 من 09' : 'Stage 04 of 09'}
        </span>

        <button
          onClick={() => setActiveModule('observatory')}
          className="btn btn-secondary"
          style={{ padding: '0.75rem 1.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <span>{isAr ? 'مرصد القياسات الحية (المرحلة 05)' : 'Next: Telemetry Observatory (Stage 05)'}</span>
          <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
        </button>
      </div>
    </div>
  );
}
