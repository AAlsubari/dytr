// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Screen 1: Master Guided Novice Wizard - Expanded 4 Steps & InfoHints
// Industrial Swiss Design, Zero Emojis Policy, Bilingual (Arabic / English)

import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import InfoHint from '../common/InfoHint';
import { api } from '../../utils/api';
import { ArrowLeft, ArrowRight } from 'lucide-react';

export default function GuidedWizard() {
  const {
    lang,
    t,
    selectedTask,
    setSelectedTask,
    selectedBrain,
    setSelectedBrain,
    enableSIShield,
    setEnableSIShield,
    enableFlashAttn,
    setEnableFlashAttn,
    trainingState,
    setTrainingState,
    setActiveView,
  } = useApp();

  const isAr = lang === 'ar';

  // Wizard Step Control: 1, 2, 3, 4
  const [currentStep, setCurrentStep] = useState(1);

  // Step 2 Data Processing State
  const [trainSplitRatio, setTrainSplitRatio] = useState(80); // 80% train, 20% val
  const [batchSize, setBatchSize] = useState(16);
  const [totalDatasetSamples] = useState(5000);

  // Step 3 Extended Parameters
  const [siStrength, setSiStrength] = useState(1.0);
  const [enableRoPE, setEnableRoPE] = useState(true);
  const [enableAMP, setEnableAMP] = useState(true);

  // Step 4 Simulation Phase State
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);

  // Task Options with Examples & Datasets
  const tasks = [
    {
      id: 'task_sentiment',
      tag: '[+/-]',
      title: isAr ? 'فهم المشاعر وتقييمات النصوص (Sentiment Analysis)' : 'Sentiment Analysis & Sentiment Polarity',
      desc: isAr
        ? 'تحديد ما إذا كان النص أو التقييم سعيداً وراضياً أم حزيناً وغير راضٍ (مثل مراجعات المنتجات والآراء).'
        : 'Detects emotional polarity in customer reviews, feedback, and sentiment commentary.',
      badge: isAr ? 'تصنيف مشاعر' : 'Classification',
      badgeColor: 'badge-pine',
      sampleInput: isAr ? 'هذا النظام التعليمي مذهل ويشرح المفاهيم بسلاسة تامة!' : 'This framework delivers exceptional stability and speed!',
      sampleOutput: isAr ? 'إيجابي (Positive) بنسبة ثقة 98.4%' : 'Positive (98.4% Confidence)',
      benchmark: 'IMDb 5,000 Samples',
      hintTitle: isAr ? 'تحليل المشاعر (Sentiment Analysis)' : 'Sentiment Analysis',
      hintContent: isAr
        ? 'يقوم النموذج بتعيين قيمة احتمالية بين 0 (سلبي) و 1 (إيجابي) عبر استخراج الأنماط الدلالية من الكلمات.'
        : 'The model maps input text into emotional polarity logits via semantic token pooling.',
      hintAnalogy: isAr ? 'مثل قارئ ذكي يكتشف نبرة الكاتب من خلال مفرداته.' : 'Like an astute reader discerning tone from lexical cues.',
    },
    {
      id: 'task_topic',
      tag: '[Topic]',
      title: isAr ? 'تصنيف الأخبار والمواضيع (Topic Categorization)' : 'News & Topic Categorization',
      desc: isAr
        ? 'قراءة المقالات وتحديد مجالها وموضوعها فوراً (رياضة، تكنولوجيا، سياسة، واقتصاد).'
        : 'Categorizes text into structured domains (Sports, Technology, World Politics, Business).',
      badge: isAr ? 'تصنيف موضوعي' : 'Multi-Class',
      badgeColor: 'badge-steel',
      sampleInput: isAr ? 'أعلنت وكالة الفضاء عن إطلاق مسبار جديد لاستكشاف الكواكب البعيدة.' : 'Tech giants unveil novel quantum computing chips.',
      sampleOutput: isAr ? 'تكنولوجيا وعلوم (Technology / Science)' : 'Technology / Science (99.1%)',
      benchmark: 'AG_News 8,000 Articles',
      hintTitle: isAr ? 'تصنيف المواضيع (Topic Classification)' : 'Topic Categorization',
      hintContent: isAr
        ? 'يحسب النموذج احتمالية توزيع النص عبر 4 فئات مستقلة باستخدام دالة Softmax.'
        : 'Computes normalized probability distribution across discrete thematic classes via Softmax.',
      hintAnalogy: isAr ? 'مثل أمين مكتبة يفرز الكتب في الأقسام المخصصة لها تلقائياً.' : 'Like a librarian automatically sorting books into sections.',
    },
    {
      id: 'task_ner',
      tag: '[NER]',
      title: isAr ? 'صياد الأسماء والكيانات المجهري (Named Entity Recognition)' : 'Named Entity Recognition (NER)',
      desc: isAr
        ? 'قراءة النصوص واستخراج أسماء الأشخاص، الشركات، المدن، والدول بدقة مجهرية على مستوى كل كلمة.'
        : 'Extracts real-world entities (Persons, Organizations, Geo-Locations) at token granularity.',
      badge: isAr ? 'استخراج كيانات' : 'Token Tagging',
      badgeColor: 'badge-amber',
      sampleInput: isAr ? 'زار أحمد مقر شركة أرامكو في مدينة الرياض.' : 'Jamal visited Google headquarters in California.',
      sampleOutput: isAr ? 'أحمد [شخص] · أرامكو [منظمة] · الرياض [مدينة]' : 'Jamal [PER] · Google [ORG] · California [LOC]',
      benchmark: 'CoNLL-2003 3,500 Sentences',
      hintTitle: isAr ? 'استخراج الكيانات (NER)' : 'Named Entity Recognition',
      hintContent: isAr
        ? 'يتم فحص كل توكن على حدة وتعيين تصنيف (B-PER, I-PER, B-ORG, B-LOC, O) مع حجب الأجزاء المتفرعة بقناع -100.'
        : 'Assigns sequence labels to individual tokens while masking continuation subwords with -100.',
      hintAnalogy: isAr ? 'مثل محقق يدقق في النص بالقلم الملون لتحديد الشخصيات والأماكن.' : 'Like a detective highlighting names and locations in a case file.',
    },
    {
      id: 'task_story',
      tag: '[LM]',
      title: isAr ? 'المؤلف اللغوي التوليدي (Autoregressive Language Modeling)' : 'Generative Language Modeling',
      desc: isAr
        ? 'البدء بجملة افتتاحية وإكمال صياغة نص متناسق بصورة توليدية ذاتية كلمة تلو الأخرى.'
        : 'Generates cohesive textual continuations token-by-token autoregressively.',
      badge: isAr ? 'توليد نصوص' : 'Generative LM',
      badgeColor: 'badge-steel',
      sampleInput: isAr ? 'في ليلة صافية تحت ضوء النجوم، بدأت التجربة العصبية...' : 'Once upon a time in a digital laboratory...',
      sampleOutput: isAr ? '...وسرعان ما أظهرت خوارزمية DyTR ثباتاً استثنائياً في الذاكرة.' : '...the neural network unlocked infinite memory retention.',
      benchmark: 'TinyStories 10,000 Prompts',
      hintTitle: isAr ? 'النمذجة التوليدية (Autoregressive LM)' : 'Generative Modeling',
      hintContent: isAr
        ? 'يتوقع النموذج الكلمة التالية P(w_t | w_1...w_{t-1}) باستخدام قناع انتباه سببي (Causal Attention Mask).'
        : 'Predicts the next token distribution causally conditioned on prior context tokens.',
      hintAnalogy: isAr ? 'مثل كاتب يكمل القصة بناءً على سياق الأحداث السابقة.' : 'Like an author continuing a story logically from prior chapters.',
    },
  ];

  // Brain Capacities
  const brainSizes = [
    {
      id: 'small',
      tag: '1.2M',
      title: isAr ? 'العقل المدمج السريع (Pocket Brain)' : 'Pocket Brain (Compact & Ultra-Fast)',
      desc: isAr
        ? 'خفيف وسريع جداً، يتدرب في ثوانٍ ومناسب للحواسيب القياسية وتجارب النماذج السريعة.'
        : 'Ultra-lightweight architecture, trains in seconds on standard CPU/GPU setups.',
      specs: 'd_model: 128 | Layers: 2 | Heads: 2 | Params: 1,245,696',
      vram: '~180 MB',
      hintTitle: isAr ? 'العقل المدمج (Pocket Architecture)' : 'Compact Architecture',
      hintContent: isAr
        ? 'معمارية ثنائية الطبقات تستهلك موارد ضئيلة جداً وتعتمد على تقطيع FlashAttention لتسريع الـ throughput.'
        : '2-layer transformer stack with minimal memory footprint.',
      hintAnalogy: isAr ? 'مثل دراجة نارية خفيفة وسريعة الحركة للمسافات القصيرة.' : 'Like a lightweight scooter: agile, fast, and highly efficient.',
    },
    {
      id: 'medium',
      tag: '4.8M',
      title: isAr ? 'العقل المتوازن الذكي (Smart Brain)' : 'Smart Brain (Balanced & Accurate)',
      desc: isAr
        ? 'قوي ودقيق، يجمع بين كفاءة السرعة والفهم اللغوي المتكامل مع توازن مثالي بين المعلمات والذاكرة.'
        : 'Optimal balance between representation depth and fast gradient optimization.',
      specs: 'd_model: 256 | Layers: 4 | Heads: 4 | Params: 4,812,416',
      vram: '~360 MB',
      hintTitle: isAr ? 'العقل المتوازن (Smart Architecture)' : 'Smart Architecture',
      hintContent: isAr
        ? 'معمارية رباعية الطبقات ذات أبعاد 256، توفر تمثيلاً دلالياً عميقاً للتراكيب اللغوية المعقدة.'
        : '4-layer stack with 256 hidden dimension capturing complex linguistic nuances.',
      hintAnalogy: isAr ? 'مثل سيارة سيدان حديثة تجمع بين القوة والراحة والاستهلاك الاقتصادي.' : 'Like a modern sedan: dependable, powerful, and balanced.',
    },
    {
      id: 'large',
      tag: '16.2M',
      title: isAr ? 'العقل المتقدم الفائق (Super Brain)' : 'Super Brain (Deep Semantic Reasoning)',
      desc: isAr
        ? 'أعلى سعة استيعابية، يمتلك ذاكرة واسعة لمعالجة المقالات الطويلة واستيعاب الروابط الدلالية المركبة.'
        : 'Deep representations capable of handling long contexts and complex reasoning.',
      specs: 'd_model: 512 | Layers: 6 | Heads: 8 | Params: 16,258,048',
      vram: '~820 MB',
      hintTitle: isAr ? 'العقل المتقدم (Super Architecture)' : 'Super Architecture',
      hintContent: isAr
        ? 'معمارية سداسية الطبقات بأبعاد 512 مع 8 رؤوس انتباه متوازية لاستيعاب التناظر المكاني والزمني.'
        : '6-layer stack with 512 hidden dimension and 8 multi-head attention subspaces.',
      hintAnalogy: isAr ? 'مثل قاطرة قوية قادرة على سحب أثقل الحمولات الفكرية واللغوية.' : 'Like a high-speed express train built for heavy freight.',
    },
  ];

  // Training Simulation Phases
  const trainingPhases = [
    {
      title: isAr ? 'المرحلة 1: استيعاب التوكنز وتقطيع النصوص (Subword Ingestion & Alignment)' : 'Phase 1: Tokenization & Alignment',
      desc: isAr
        ? 'تقطيع الكلمات إلى وحدات فرعية (Subwords) وتجهيز مصفوفات المعرّفات (Token IDs) مع حجب التفرعات بقناع -100.'
        : 'Decomposes raw text into vocabulary subwords and maps sequence IDs with loss masks.',
      formula: 'Tokens: ["Dy", "##TR", "Continual"] -> IDs: [1024, 8432, 5120] -> Mask: [-100 on subwords]',
    },
    {
      title: isAr ? 'المرحلة 2: الانتباه متعدد الرؤوس وتدوير المتجهات (Multi-Head Attention & RoPE)' : 'Phase 2: Multi-Head Attention & RoPE',
      desc: isAr
        ? 'حساب مصفوفة الانتباه التفاعلي وتدوير متجهات Q و K بزاوية m*theta داخل الذاكرة السريعة SRAM.'
        : 'Computes spatial query-key interactions and rotates relative embeddings via RoPE.',
      formula: 'Attention(Q, K, V) = Softmax( (Q R_m)(K R_n)^T / √d_k ) V  (computed in SRAM)',
    },
    {
      title: isAr ? 'المرحلة 3: التنبؤ وحساب دالة الخسارة (Forward Pass & Cross-Entropy Loss)' : 'Phase 3: Forward Pass & Loss Computation',
      desc: isAr
        ? 'تمرير المتجهات عبر طبقات المحول لإنتاج التوزيع الاحتمالي وحساب مقدار الخطأ بين التوقع والإجابة الصحيحة.'
        : 'Generates output logits and computes cross-entropy discrepancy against ground truth labels.',
      formula: 'Loss = - Σ y_c log(p_c) + (λ/2) Σ F_i (θ_i - θ_i^*)^2',
    },
    {
      title: isAr ? 'المرحلة 4: التدرج العكسي وتحديث الأوزان (Backpropagation & Parameter Update)' : 'Phase 4: Backpropagation & Weight Update',
      desc: isAr
        ? 'إرسال إشارات تصحيحية عكسية عبر كافة الطبقات لتحديث الأوزان العصبية وخفض الخطأ للخطوات القادمة.'
        : 'Propagates error gradients backward to adjust neural parameters and improve precision.',
      formula: 'W := W - (η × ∇Loss)   (AdamW Optimizer with Decoupled Weight Decay)',
    },
    {
      title: isAr ? 'المرحلة 5: درع مصفوفة فيشر للتعلم المستمر (EWC Memory Shield)' : 'Phase 5: EWC Memory Anchoring',
      desc: isAr
        ? 'حساب مصفوفة معلومات فيشر (FIM) وتثبيت الأوزان الحرجة لحماية المعارف السابقة من النسيان.'
        : 'Calculates Fisher information matrix (FIM) to freeze vital connections.',
      formula: 'F_i = E[(∂L/∂θ_i)^2]   (EWC Protection Shield Active)',
    },
  ];

  // Start Live Training Simulation
  const handleStartTraining = () => {
    setTrainingState((prev) => ({
      ...prev,
      isTraining: true,
      progress: 0,
      isTrained: false,
      currentStepMessage: trainingPhases[0].title,
      loss: 1.85,
      accuracy: 65.0,
    }));
    setActivePhaseIndex(0);
  };

  useEffect(() => {
    let interval = null;
    if (trainingState.isTraining) {
      interval = setInterval(() => {
        setTrainingState((prev) => {
          const nextProgress = prev.progress + 4;
          const phaseIdx = Math.min(4, Math.floor((nextProgress / 100) * 5));
          setActivePhaseIndex(phaseIdx);

          const currentLoss = Math.max(0.045, (1.85 - (nextProgress / 100) * 1.78) + (Math.random() * 0.04 - 0.02));
          const currentAcc = Math.min(96.8, 65.0 + (nextProgress / 100) * 31.8);

          if (nextProgress >= 100) {
            clearInterval(interval);
            return {
              ...prev,
              isTraining: false,
              progress: 100,
              isTrained: true,
              currentStepMessage: isAr ? 'اكتمل تدريب النموذج بنجاح وتثبيت درع الذاكرة EWC!' : 'Training completed successfully with EWC protection!',
              loss: 0.0452,
              accuracy: 96.8,
            };
          }

          return {
            ...prev,
            progress: nextProgress,
            currentStepMessage: trainingPhases[phaseIdx].title,
            loss: currentLoss,
            accuracy: currentAcc,
          };
        });
      }, 200);
    }
    return () => clearInterval(interval);
  }, [trainingState.isTraining, isAr]);

  const trainSamplesCount = Math.round((trainSplitRatio / 100) * totalDatasetSamples);
  const valSamplesCount = totalDatasetSamples - trainSamplesCount;

  return (
    <section>
      {/* Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
          {isAr ? 'المصمم الإرشادي: ابنِ ودرب نموذج ذكاء اصطناعي بنفسك' : 'Guided Novice Wizard: Build & Train Your AI Model'}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {isAr
            ? 'مسار تفاعلي مؤتمت في 4 خطوات مدعوم بشروحات تعليمية مجهرية لجميع المفاهيم العصبية.'
            : 'Interactive 4-step guided pipeline with microscopic educational tooltips explaining core neural principles.'}
        </p>
      </div>

      {/* 4-Step Visual Progress Stepper */}
      <div
        className="glass-panel"
        style={{
          padding: '0.85rem 1.25rem',
          marginBottom: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        {[
          { num: 1, title: isAr ? '1. المهارة والمهمة' : '1. Task Skill' },
          { num: 2, title: isAr ? '2. سعة العقل والبيانات' : '2. Brain & Data' },
          { num: 3, title: isAr ? '3. دروع الذاكرة والسرعة' : '3. Memory Shields' },
          { num: 4, title: isAr ? '4. التدريب المباشر' : '4. Live Training' },
        ].map((s) => {
          const isCurrent = currentStep === s.num;
          const isDone = currentStep > s.num;
          return (
            <div
              key={s.num}
              onClick={() => setCurrentStep(s.num)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.55rem',
                cursor: 'pointer',
                opacity: isCurrent || isDone ? 1 : 0.6,
                transition: 'all 0.2s ease',
              }}
            >
              <div
                className="step-badge"
                style={{
                  background: isCurrent ? 'var(--accent-steel)' : isDone ? 'var(--accent-pine)' : 'var(--bg-tertiary)',
                  border: isCurrent ? '1px solid var(--accent-steel)' : '1px solid var(--border-medium)',
                  color: isCurrent || isDone ? '#FFFFFF' : 'var(--text-secondary)',
                  width: '28px',
                  height: '28px',
                  fontSize: '0.85rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {isDone ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                ) : s.num}
              </div>
              <span style={{ fontSize: '0.88rem', fontWeight: isCurrent ? '700' : '500', color: isCurrent ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {s.title}
              </span>
            </div>
          );
        })}
      </div>

      {/* =========================================================================
          STEP 1: CHOOSE TASK SUPERPOWER
          ========================================================================= */}
      {currentStep === 1 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'الخطوة 1: ماذا تريد أن يتعلم نموذجك اليوم؟' : 'Step 1: Choose Your Model Superpower'}
            </h3>
            <InfoHint
              title={isAr ? 'اختيار مهمة النموذج (Supervised Task Selection)' : 'Task Superpower'}
              content={isAr
                ? 'تحدد المهمة نوع رأس التصنيف العصبي (Classification Head) الذي سيتم بناؤه وتدريبه فوق طبقات المحول الأساسية.'
                : 'Defines the structural classification head and loss function parameterized on top of the transformer backbone.'}
              analogy={isAr ? 'مثل اختيار التخصص الدراسي للطالب (طب، لغات، هندسة) ليتدرب على كتبها.' : 'Like choosing a major for a student so they focus on specific textbooks.'}
            />
          </div>

          <div className="grid-2" style={{ gap: '1rem', marginBottom: '1.5rem' }}>
            {tasks.map((task) => {
              const isSelected = selectedTask === task.id;
              return (
                <div
                  key={task.id}
                  onClick={() => setSelectedTask(task.id)}
                  className="glass-card"
                  style={{
                    cursor: 'pointer',
                    borderColor: isSelected ? 'var(--accent-steel)' : 'var(--border-subtle)',
                    background: isSelected ? 'var(--accent-steel-bg)' : 'var(--bg-card)',
                    padding: '1.25rem',
                    position: 'relative',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span className="mono-font" style={{ fontWeight: '800', fontSize: '0.9rem', color: 'var(--accent-steel)' }}>
                        {task.tag}
                      </span>
                      <span className={`badge ${task.badgeColor}`} style={{ fontSize: '0.72rem' }}>
                        {task.badge}
                      </span>
                    </div>
                    <span className="badge badge-steel" style={{ fontSize: '0.72rem' }}>
                      {task.benchmark}
                    </span>
                  </div>

                  <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center' }}>
                    {task.title}
                    <InfoHint
                      title={task.hintTitle}
                      content={task.hintContent}
                      analogy={task.hintAnalogy}
                    />
                  </h4>
                  <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.45', marginBottom: '0.85rem' }}>
                    {task.desc}
                  </p>

                  {/* Interactive Input/Output Demonstration */}
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', fontSize: '0.78rem' }}>
                    <div style={{ color: 'var(--text-muted)', marginBottom: '3px' }}>
                      <strong>{isAr ? 'مثال الإدخال:' : 'Sample Prompt:'}</strong> "{task.sampleInput}"
                    </div>
                    <div style={{ color: 'var(--accent-pine)', fontWeight: '600' }}>
                      <strong>{isAr ? 'النتيجة المتوقعة:' : 'Expected Prediction:'}</strong> {task.sampleOutput}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setCurrentStep(2)}
              className="btn-primary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <span>{isAr ? 'التالي: سعة العقل والبيانات' : 'Next: Brain & Data Split'}</span>
              <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        </div>
      )}

      {/* =========================================================================
          STEP 2: BRAIN CAPACITY & DATA PARTITIONING
          ========================================================================= */}
      {currentStep === 2 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'الخطوة 2: اختر حجم عقل النموذج وتقسيم البيانات' : 'Step 2: Brain Capacity & Data Partitioning'}
            </h3>
            <InfoHint
              title={isAr ? 'سعة النموذج والتعميم (Model Capacity & Generalization)' : 'Model Capacity'}
              content={isAr
                ? 'تحدد أبعاد المحول (عدد الطبقات وحجم التضمين) مقدار المعرفة التي يستطيع النموذج استيعابها وسرعة تدريبه.'
                : 'Defines the parameter space depth and computational complexity of forward/backward operations.'}
              analogy={isAr ? 'كلما كبر حجم الدفتر اتسع لمعلومات أكثر، لكن تقليب صفحاته يستغرق وقتاً أطول.' : 'Larger notebooks hold more notes, but flipping through them takes slightly longer.'}
            />
          </div>

          {/* 3 Brain Sizes Grid */}
          <div className="grid-3" style={{ gap: '1rem', marginBottom: '1.5rem' }}>
            {brainSizes.map((brain) => {
              const isSelected = selectedBrain === brain.id;
              return (
                <div
                  key={brain.id}
                  onClick={() => setSelectedBrain(brain.id)}
                  className="glass-card"
                  style={{
                    cursor: 'pointer',
                    borderColor: isSelected ? 'var(--accent-steel)' : 'var(--border-subtle)',
                    background: isSelected ? 'var(--accent-steel-bg)' : 'var(--bg-card)',
                    padding: '1.25rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                    <span className="mono-font" style={{ fontSize: '1.1rem', fontWeight: '800', color: 'var(--accent-steel)' }}>
                      {brain.tag}
                    </span>
                    <span className="badge badge-pine">{brain.vram} VRAM</span>
                  </div>

                  <h4 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center' }}>
                    {brain.title}
                    <InfoHint
                      title={brain.hintTitle}
                      content={brain.hintContent}
                      analogy={brain.hintAnalogy}
                    />
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.45', marginBottom: '0.75rem' }}>
                    {brain.desc}
                  </p>

                  <div className="mono-font" style={{ background: 'var(--bg-tertiary)', padding: '0.45rem 0.65rem', borderRadius: '6px', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {brain.specs}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Expanded Data Splitting & Batch Size Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h4 style={{ fontSize: '1rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem', margin: 0 }}>
                  {isAr ? 'معالجة وتقسيم البيانات (Train / Validation Split)' : 'Data Partitioning (Train / Validation Split)'}
                  <InfoHint
                    title={isAr ? 'تقسيم البيانات (Train / Validation Split)' : 'Train / Validation Split'}
                    content={isAr
                      ? 'يتم فصل البيانات إلى حزمة تدريب لتحديث الأوزان العصبية، وحزمة تحقق مستقلة لقياس الدقة الحقيقية ومنع الحفظ الأعمى (Overfitting).'
                      : 'Splits dataset into training samples for backpropagation and hold-out validation samples to test genuine generalization.'}
                    analogy={isAr ? 'مثل دراسة كتاب التمارين أثناء العام (تدريب)، ثم خوض اختبار بأسئلة جديدة تماماً (تحقق).' : 'Like studying practice homework, then taking an exam with unseen test questions.'}
                  />
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px', margin: 0 }}>
                  {isAr
                    ? 'تقسيم حزمة البيانات لقياس القدرة الحقيقية للنموذج على التعميم ومنع حفظ الأمثلة غيباً:'
                    : 'Configure the percentage of data dedicated to weight optimization vs unseen evaluation:'}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-steel)' }}>{isAr ? 'تدريب:' : 'Train:'}</span>
                  <input
                    type="number"
                    min="50"
                    max="90"
                    step="1"
                    value={trainSplitRatio}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (!isNaN(val)) setTrainSplitRatio(Math.min(90, Math.max(50, val)));
                    }}
                    style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-steel)', padding: 0 }}
                  />
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>%</span>
                </div>
                <span className="badge badge-amber">{isAr ? `التحقق: ${100 - trainSplitRatio}%` : `Val: ${100 - trainSplitRatio}%`} ({valSamplesCount} {isAr ? 'عينة' : 'samples'})</span>
              </div>
            </div>

            {/* Split Slider & Visual Bar */}
            <div style={{ marginBottom: '1rem' }}>
              <input
                type="range"
                min="50"
                max="90"
                step="1"
                value={trainSplitRatio}
                onChange={(e) => setTrainSplitRatio(parseInt(e.target.value))}
                className="slider-input"
                style={{ width: '100%', marginBottom: '0.6rem' }}
              />
              <div style={{ width: '100%', height: '10px', background: 'var(--bg-tertiary)', borderRadius: '5px', overflow: 'hidden', display: 'flex' }}>
                <div style={{ width: `${trainSplitRatio}%`, background: 'var(--accent-steel)', transition: 'width 0.2s ease' }} />
                <div style={{ width: `${100 - trainSplitRatio}%`, background: 'var(--accent-amber)', transition: 'width 0.2s ease' }} />
              </div>
            </div>

            {/* Batch Size Selector */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.85rem', borderTop: '1px solid var(--border-subtle)', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div>
                <span style={{ fontSize: '0.88rem', fontWeight: '600', color: 'var(--text-primary)', display: 'flex', alignItems: 'center' }}>
                  {isAr ? 'حجم حزمة المعالجة اللحظية (Batch Size):' : 'Mini-Batch Size:'}
                  <InfoHint
                    title={isAr ? 'حجم الحزمة (Batch Size)' : 'Mini-Batch Size'}
                    content={isAr
                      ? 'عدد الجمل أو النصوص التي يقرأها النموذج معاً في كل خطوة تدرج عصبي. الحزم الأكبر تزيد من استقرار التدرجات.'
                      : 'Number of sample sequences processed simultaneously before computing parameter gradients.'}
                    analogy={isAr ? 'مثل قراءة 16 سؤالاً معاً وحلها دفعة واحدة بدلاً من قراءة سؤال تلو الآخر.' : 'Like reviewing 16 flashcards together before adjusting study notes.'}
                  />
                </span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'كلما زاد الحجم زادت سرعة استغلال بطاقة الرسوميات GPU' : 'Higher values maximize GPU tensor parallelism'}
                </span>
              </div>

              <div style={{ display: 'flex', gap: '0.4rem' }}>
                {[8, 16, 32, 64].map((sz) => (
                  <button
                    key={sz}
                    onClick={() => setBatchSize(sz)}
                    style={{
                      background: batchSize === sz ? 'var(--accent-steel)' : 'var(--bg-tertiary)',
                      color: batchSize === sz ? '#FFFFFF' : 'var(--text-secondary)',
                      border: `1px solid ${batchSize === sz ? 'var(--accent-steel)' : 'var(--border-medium)'}`,
                      borderRadius: '6px',
                      padding: '0.35rem 0.75rem',
                      fontSize: '0.82rem',
                      fontWeight: '700',
                      cursor: 'pointer',
                    }}
                  >
                    {sz}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button
              onClick={() => setCurrentStep(1)}
              className="btn-secondary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <span>{isAr ? 'السابق: اختيار المهمة' : 'Back: Superpower'}</span>
            </button>
            <button
              onClick={() => setCurrentStep(3)}
              className="btn-primary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <span>{isAr ? 'التالي: دروع الذاكرة والسرعة' : 'Next: Memory & Shields'}</span>
              <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        </div>
      )}

      {/* =========================================================================
          STEP 3: MEMORY & SPEED SHIELDS
          ========================================================================= */}
      {currentStep === 3 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'الخطوة 3: قوة الذاكرة ومكافحة النسيان الكارثي' : 'Step 3: Continual Memory & Speed Shields'}
            </h3>
            <InfoHint
              title={isAr ? 'التعلم المستمر ومكافحة النسيان (Continual Learning)' : 'Continual Learning'}
              content={isAr
                ? 'تعاني النماذج التقليدية من مسح معارفها السابقة عند تدريبها على مهمة جديدة. توفر DyTR دروع حماية مشبكية تحافظ على المهارات السابقة بنسبة 100%.'
                : 'Overcomes catastrophic forgetting by imposing path-integral regularizers on critical parameter trajectories.'}
              analogy={isAr ? 'مثل تعلم العزف على البيانو دون أن تنسى مهارة قيادة السيارة التي تعلمتها بالأمس.' : 'Like learning piano without forgetting how to drive a car you mastered yesterday.'}
            />
          </div>

          <div className="grid-2" style={{ gap: '1rem', marginBottom: '1.5rem' }}>
            {/* EWC Memory Shield */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--accent-teal)', display: 'flex', alignItems: 'center', margin: 0 }}>
                  {isAr ? 'درع مصفوفة فيشر (EWC Memory Shield)' : 'EWC Memory Shield'}
                  <InfoHint
                    title={isAr ? 'درع مصفوفة فيشر (EWC)' : 'Elastic Weight Consolidation'}
                    content={isAr
                      ? 'يحسب مصفوفة معلومات فيشر (FIM) للأوزان العصبية الهامة للمهام السابقة ويفرض عقوبة تربيعية على تعديلها لمنع النسيان الكارثي.'
                      : 'Computes Fisher Information Matrix to quantify parameter importance and applies quadratic penalties to prevent catastrophic forgetting.'}
                    formula="Penalty = (λ / 2) Σ F_i (θ_i - θ_i^*)^2"
                  />
                </h4>
                <label className="toggle-switch">
                  <input type="checkbox" checked={enableSIShield} onChange={(e) => setEnableSIShield(e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.45', marginBottom: '0.85rem' }}>
                {isAr
                  ? 'حماية المعارف السابقة؛ عندما يتعلم النموذج مهمة جديدة لا ينسى المهارات التي تدرب عليها سابقاً.'
                  : 'Shields past task representations so sequential learning does not erase prior knowledge.'}
              </p>

              {enableSIShield && (
                <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', marginBottom: '6px' }}>
                    <span>{isAr ? 'معامل قوة درع فيشر (λ):' : 'EWC Fisher Lambda (λ):'}</span>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                      <input
                        type="number"
                        min="0.1"
                        max="10.0"
                        step="0.1"
                        value={siStrength}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) setSiStrength(Math.min(10.0, Math.max(0.1, +(val.toFixed(2)))));
                        }}
                        style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                      />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>x10³</span>
                    </div>
                  </div>
                  <input className="slider-input" style={{ width: '100%' }} type="range" min="0.1" max="10.0" step="0.1" value={siStrength} onChange={(e) => setSiStrength(parseFloat(e.target.value))} />
                </div>
              )}
            </div>

            {/* FlashAttention Acceleration */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--accent-pine)', display: 'flex', alignItems: 'center', margin: 0 }}>
                  {isAr ? 'محرك التسريع الفائق (FlashAttention SDPA)' : 'FlashAttention Acceleration Engine'}
                  <InfoHint
                    title={isAr ? 'تسريع الانتباه (FlashAttention)' : 'FlashAttention SDPA'}
                    content={isAr
                      ? 'تقنية معمارية تقسم مصفوفات الانتباه الضخمة إلى كتل متناهية الصغر يتم حسابها داخل ذاكرة SRAM فائقة السرعة على المعالج مباشرة دون إجهاد ذاكرة الـ VRAM.'
                      : 'Executes IO-aware attention tiling within high-speed SRAM, bypassing slow HBM memory bandwidth bottlenecks.'}
                    formula="Memory Complexity: O(N) instead of O(N^2)"
                  />
                </h4>
                <label className="toggle-switch">
                  <input type="checkbox" checked={enableFlashAttn} onChange={(e) => setEnableFlashAttn(e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.45', marginBottom: '0.85rem' }}>
                {isAr
                  ? 'تسريع حسابات الانتباه وخفض استهلاك الذاكرة عبر التقطيع المباشر داخل ذاكرة SRAM على المعالج.'
                  : 'Dramatically cuts memory traffic and accelerates transformer throughput via SRAM block tiling.'}
              </p>

              <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {isAr ? 'خوارزمية PyTorch SDPA النشطة · خفض استخدام الذاكرة بنسبة 75%' : 'Native PyTorch SDPA Active · Up to 75% VRAM Reduction'}
              </div>
            </div>

            {/* Rotary Position Embeddings (RoPE) */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--accent-steel)', display: 'flex', alignItems: 'center', margin: 0 }}>
                  {isAr ? 'تدوير المواقع الزاوي (Rotary Embeddings - RoPE)' : 'Rotary Position Embeddings (RoPE)'}
                  <InfoHint
                    title={isAr ? 'التضمين الزاوي النسبي (RoPE)' : 'Rotary Position Embeddings'}
                    content={isAr
                      ? 'يقوم بتدوير متجهات الكلمات في الفضاء العقدي بزوايا تتناسب مع موقعها، مما يحافظ على المسافات النسبية بدقة دون إضافة مصفوفة مواقع ثابتة.'
                      : 'Encodes relative position by rotating query and key vectors in complex 2D subspace pairs.'}
                    formula="R_θ,m^d x = diag(R_1, ... R_{d/2}) x"
                  />
                </h4>
                <label className="toggle-switch">
                  <input type="checkbox" checked={enableRoPE} onChange={(e) => setEnableRoPE(e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.45' }}>
                {isAr
                  ? 'تمثيل ترتيب الكلمات عبر تدوير المتجهات الهندسية مع الحفاظ الصارم على طول المتجه الأصلي.'
                  : 'Encodes positional distance geometrically while strictly preserving Euclidean vector norm.'}
              </p>
            </div>

            {/* Mixed Precision (AMP - FP16) */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--accent-steel)', display: 'flex', alignItems: 'center', margin: 0 }}>
                  {isAr ? 'الدقة المختلطة التلقائية (Automatic Mixed Precision)' : 'Mixed Precision Acceleration (AMP)'}
                  <InfoHint
                    title={isAr ? 'الدقة المختلطة (AMP - FP16)' : 'Automatic Mixed Precision'}
                    content={isAr
                      ? 'تنفيذ الحسابات الرياضية بدقة 16-بت لتوفير نصف مساحة الذاكرة مع استخدام GradScaler لمنع تلاشي التدرجات.'
                      : 'Executes forward operations in FP16 with dynamic gradient scaling to prevent underflow.'}
                    formula="VRAM Footprint: Halved (FP32 to FP16)"
                  />
                </h4>
                <label className="toggle-switch">
                  <input type="checkbox" checked={enableAMP} onChange={(e) => setEnableAMP(e.target.checked)} />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.45' }}>
                {isAr
                  ? 'مضاعفة سرعة التدريب وخفض استهلاك الذاكرة إلى النصف باستخدام وحدات Tensor Cores.'
                  : 'Doubles arithmetic speed on modern GPU tensor cores with dynamic loss scaling.'}
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button
              onClick={() => setCurrentStep(2)}
              className="btn-secondary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <span>{isAr ? 'السابق: سعة العقل' : 'Back: Brain & Data'}</span>
            </button>
            <button
              onClick={() => setCurrentStep(4)}
              className="btn-primary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <span>{isAr ? 'التالي: بدء التدريب والمتابعة المباشرة' : 'Next: Start Training'}</span>
              <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
          </div>
        </div>
      )}

      {/* =========================================================================
          STEP 4: LIVE TRAINING EXECUTION & NEURAL MECHANICS BREAKDOWN
          ========================================================================= */}
      {currentStep === 4 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'الخطوة 4: بدء التدريب ومشاهدة مراحل التعلم العصبي لحظياً' : 'Step 4: Live Training Execution & Neural Mechanics'}
            </h3>
            <InfoHint
              title={isAr ? 'التدريب العصبي التفاعلي (Neural Training Execution)' : 'Neural Mechanics'}
              content={isAr
                ? 'شاهد كيف يمر النص عبر طبقات الانتباه، تحسب دالة الخسارة مقدار الخطأ، ويرسل التدرج العكسي إشارات تصحيحية تضبط الأوزان ليصبح النموذج أكثر دقة.'
                : 'Observes the full forward-backward training cycle with real-time loss reduction and gradient weight convergence.'}
              analogy={isAr ? 'مثل مشاهدة التلميذ وهو يحل الأسئلة ويصحح أخطاءه لحظياً ليصبح متفوقاً.' : 'Like watching a student solve equations and correct errors in real-time.'}
            />
          </div>

          {/* Training Control Card */}
          <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <span className="badge badge-steel" style={{ fontSize: '0.8rem', marginBottom: '4px' }}>
                  {isAr ? 'إعدادات الجلسة:' : 'Configured Setup:'} {selectedTask} · {selectedBrain} Brain · Batch {batchSize}
                </span>
                <h4 style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)', margin: 0 }}>
                  {trainingState.isTraining
                    ? (isAr ? 'جاري التدريب العصبي الفعلي...' : 'Neural Training in Progress...')
                    : trainingState.isTrained
                    ? (isAr ? 'اكتمل تدريب النموذج بنجاح وهو جاهز للاختبار!' : 'Model Training Complete & Ready for Inference!')
                    : (isAr ? 'النموذج جاهز للبدء' : 'Ready to Start Training')}
                </h4>
              </div>

              {!trainingState.isTraining && !trainingState.isTrained && (
                <button onClick={handleStartTraining} className="btn-huge">
                  {isAr ? 'ابدأ تدريب النموذج الآن' : 'Start Model Training Now'}
                </button>
              )}

              {trainingState.isTrained && (
                <button
                  onClick={() => setActiveView('playground')}
                  className="btn-primary"
                  style={{
                    padding: '0.85rem 1.75rem',
                    fontSize: '1rem',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                >
                  <span>{isAr ? 'انتقل إلى غرفة الاستنتاج وتجربة النموذج' : 'Open Inference Playground'}</span>
                  <ArrowRight size={18} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
                </button>
              )}
            </div>

            {/* Progress Bar */}
            <div style={{ marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{trainingState.currentStepMessage}</span>
                <span className="mono-font" style={{ fontWeight: '800', color: 'var(--accent-steel)' }}>{trainingState.progress}%</span>
              </div>
              <div style={{ width: '100%', height: '10px', background: 'var(--bg-tertiary)', borderRadius: '5px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${trainingState.progress}%`,
                    height: '100%',
                    background: trainingState.isTrained ? 'var(--accent-pine)' : 'var(--accent-steel)',
                    borderRadius: '5px',
                    transition: 'width 0.2s ease',
                  }}
                />
              </div>
            </div>

            {/* Live Telemetry Readout Gauges */}
            <div className="grid-3" style={{ gap: '0.75rem' }}>
              <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'دالة الخسارة (Cross-Entropy Loss):' : 'Cross-Entropy Loss:'}
                </span>
                <div className="mono-font" style={{ fontSize: '1.4rem', fontWeight: '800', color: 'var(--accent-amber)', marginTop: '2px' }}>
                  {trainingState.loss.toFixed(4)}
                </div>
              </div>

              <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'دقة النموذج على حزمة التحقق:' : 'Validation Accuracy:'}
                </span>
                <div className="mono-font" style={{ fontSize: '1.4rem', fontWeight: '800', color: 'var(--accent-pine)', marginTop: '2px' }}>
                  {trainingState.accuracy.toFixed(1)}%
                </div>
              </div>

              <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'حالة درع الذاكرة (EWC Shield):' : 'EWC Memory Protection:'}
                </span>
                <div className="mono-font" style={{ fontSize: '1.1rem', fontWeight: '800', color: 'var(--accent-steel)', marginTop: '4px' }}>
                  {isAr ? 'نشط (EWC Shield)' : 'Active (EWC Protection)'}
                </div>
              </div>
            </div>
          </div>

          {/* Deep Neural Mechanics Step-by-Step Educational Cards */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <h4 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '0.85rem', display: 'flex', alignItems: 'center' }}>
              {isAr ? 'المنطقة  المجهرية: كيف يتعلم النموذج خلف الكواليس في كل مرحلة؟' : 'Under the Hood: Microscopic Neural Training Mechanics'}
              <InfoHint
                title={isAr ? 'مراحل التعلم العصبي (Neural Learning Mechanics)' : 'Neural Mechanics'}
                content={isAr
                  ? 'يمر النص في كل خطوة تدريبية بـ 5 مراحل متتالية: التقطيع، الانتباه وتدوير المتجهات، حساب الخسارة، التدرج العكسي، والتثبيت المشبكي.'
                  : 'Sequence of 5 atomic mathematical operations executed on each mini-batch forward-backward cycle.'}
              />
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {trainingPhases.map((phase, idx) => {
                const isCurrentPhase = activePhaseIndex === idx && trainingState.isTraining;
                const isPassed = activePhaseIndex > idx || trainingState.isTrained;

                return (
                  <div
                    key={idx}
                    style={{
                      background: isCurrentPhase ? 'var(--accent-steel-bg)' : 'var(--bg-tertiary)',
                      border: `1px solid ${isCurrentPhase ? 'var(--accent-steel)' : isPassed ? 'var(--border-strong)' : 'var(--border-subtle)'}`,
                      borderRadius: '8px',
                      padding: '0.85rem 1rem',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                      <strong style={{ color: isCurrentPhase ? 'var(--accent-steel)' : isPassed ? 'var(--accent-pine)' : 'var(--text-primary)', fontSize: '0.9rem' }}>
                        {phase.title}
                      </strong>
                      <span className="mono-font" style={{ fontSize: '0.75rem', color: isPassed ? 'var(--accent-pine)' : 'var(--text-muted)' }}>
                        {isPassed ? '[Complete]' : isCurrentPhase ? '[Processing...]' : '[Pending]'}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: '0 0 6px 0', lineHeight: '1.4' }}>
                      {phase.desc}
                    </p>
                    <div className="mono-font" style={{ background: 'var(--bg-base)', padding: '0.35rem 0.6rem', borderRadius: '4px', fontSize: '0.74rem', color: 'var(--accent-steel)', border: '1px solid var(--border-subtle)' }}>
                      {phase.formula}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-start', marginTop: '1.25rem' }}>
            <button
              onClick={() => setCurrentStep(3)}
              className="btn-secondary"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <span>{isAr ? 'السابق: دروع الذاكرة' : 'Back: Memory Shields'}</span>
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
