// ==============================================================================
// AcademyWizardView - Complete 10-Step Educational Journey
// From "What is a Machine?" to "LoRA & Getting Started"
// Zero Emojis Policy - Bilingual RTL/LTR - Connected to dytr via FastAPI
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { academicDictionary } from '../../utils/academicDictionary';
import TermTooltip from '../common/TermTooltip';
import StepProgressBar from '../common/StepProgressBar';
import SectionCard from '../common/SectionCard';
import { api } from '../../utils/api';
import { ArrowRight, ArrowLeft } from 'lucide-react';

const TOTAL_STEPS = 10;

export default function AcademyWizardView() {
  const { language, setActiveModule } = useStudioStore();
  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';
  const [currentStep, setCurrentStep] = useState(1);

  // Interactive state for Step 5 (Tokenization)
  const [tokenInput, setTokenInput] = useState('');
  const [tokenResult, setTokenResult] = useState(null);

  // Interactive state for Step 9 (Continual Learning)
  const [clProtection, setClProtection] = useState(true);
  const [clData, setClData] = useState(null);

  // Interactive state for Step 7 (Self-Attention)
  const [selectedWordIdx, setSelectedWordIdx] = useState(0);

  // Interactive state for Step 10 (LoRA Rank)
  const [loraRank, setLoraRank] = useState(4);

  const stepLabels = [
    d.step1_title, d.step2_title, d.step3_title, d.step4_title,
    d.step5_title, d.step6_title, d.step7_title, d.step8_title,
    d.step9_title, d.step10_title,
  ];

  // Fetch tokenization data for Step 5
  const handleTokenize = async () => 
    {
    if (!tokenInput.trim()) return;
    try {
      const res = await api.simulateTokenization(tokenInput);
      if (res) {
        setTokenResult(res);
        return;
      }
    } catch (e) { /* fallback below */ }
    // Local fallback
    const words = tokenInput.trim().split(/\s+/);
    const tokens = words.map((w, i) => ({
      token_id: i,
      token_piece: w,
      parent_word: w,
      is_subword_continuation: false,
    }));
    setTokenResult({ tokens, total_tokens: tokens.length, word_count: words.length });
  };

  // Fetch continual learning data for Step 9
  useEffect(() => {
    const fetchCL = async () => {
      try {
        const res = await api.simulateContinualLearning(clProtection);
        if (res) { setClData(res); return; }
      } catch (e) { /* fallback */ }
      // Simple simulation fallback
      const steps = Array.from({ length: 30 }, (_, i) => i + 1);
      const half = 15;
      const t1 = steps.map((t) => {
        if (t <= half) return +(50 + 45 * (1 - Math.exp(-3 * (t / half)))).toFixed(1);
        const p = (t - half) / half;
        return clProtection
          ? +(95 * (1 - 0.05 * p)).toFixed(1)
          : +(95 * Math.exp(-4.5 * p)).toFixed(1);
      });
      const t2 = steps.map((t) => {
        if (t <= half) return +(25 + 5 * Math.sin(t / half)).toFixed(1);
        return +(25 + 65 * (1 - Math.exp(-3 * ((t - half) / half)))).toFixed(1);
      });
      setClData({ task1_accuracy: t1, task2_accuracy: t2, steps });
    };
    if (currentStep === 9) fetchCL();
  }, [currentStep, clProtection]);

  const goNext = () => { if (currentStep < TOTAL_STEPS) setCurrentStep(currentStep + 1); };
  const goPrev = () => { if (currentStep > 1) setCurrentStep(currentStep - 1); };
  const goToDataPrep = () => setActiveModule('dataprep');

  // =========================================================================
  // Step Renderers
  // =========================================================================

  const renderStep1 = () => (
    <div className="academy-step-content">
      <SectionCard sectionCode="01" title={d.step1_title} subtitle={d.step1_why} termKey="machine_learning" tooltipText={d.step1_why}>
        <p>{d.step1_content_1}</p>
        <p>{d.step1_content_2}</p>
        <p>{d.step1_content_3}</p>

        {/* Machine Diagram: Input --> Machine --> Output */}
        <div className="diagram-container">
          <div className="diagram-box">
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.step1_diagram_input}</span>
          </div>
          <span className="diagram-arrow"><ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} /></span>
          <div className="diagram-box highlight">
            <span>{d.step1_diagram_machine}</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{d.step1_diagram_rules}</span>
          </div>
          <span className="diagram-arrow"><ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} /></span>
          <div className="diagram-box">
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.step1_diagram_output}</span>
          </div>
        </div>

        {/* Traditional vs Smart comparison */}
        <div className="comparison-grid">
          <div className="comparison-column">
            <div className="comparison-column-title" style={{ color: 'var(--accent-amber)' }}>
              {d.step1_traditional}
            </div>
            <p className="branch-card-desc">{d.step1_traditional_desc}</p>
          </div>
          <div className="comparison-column">
            <div className="comparison-column-title" style={{ color: 'var(--accent-teal)' }}>
              {d.step1_smart}
            </div>
            <p className="branch-card-desc">{d.step1_smart_desc}</p>
          </div>
        </div>

        <p style={{ marginTop: '1rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
          {d.step1_content_4}
        </p>
      </SectionCard>
    </div>
  );

  const renderStep2 = () => {
    const branches = [
      { key: 'machine_learning', title: d.step2_branch_ml, desc: d.step2_branch_ml_desc, tip: d.step2_branch_ml_tooltip },
      { key: 'deep_learning', title: d.step2_branch_dl, desc: d.step2_branch_dl_desc, tip: d.step2_branch_dl_tooltip },
      { key: 'nlp', title: d.step2_branch_nlp, desc: d.step2_branch_nlp_desc, tip: d.step2_branch_nlp_tooltip },
      { key: 'computer_vision', title: d.step2_branch_cv, desc: d.step2_branch_cv_desc, tip: d.step2_branch_cv_tooltip },
      { key: 'reinforcement_learning', title: d.step2_branch_rl, desc: d.step2_branch_rl_desc, tip: d.step2_branch_rl_tooltip },
    ];
    return (
      <div className="academy-step-content">
        <SectionCard sectionCode="02" title={d.step2_title} subtitle={d.step2_why} termKey="deep_learning" tooltipText={d.step2_why}>
          <p>{d.step2_content_1}</p>
          <div className="branch-grid">
            {branches.map((b) => (
              <div key={b.key} className="branch-card">
                <div className="branch-card-title">
                  <span>{b.title}</span>
                  <TermTooltip termKey={b.key} explanation={b.tip} title={b.title} hideLabel={true} />
                </div>
                <div className="branch-card-desc">{b.desc}</div>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>
    );
  };

  const renderStep3 = () => {
    const cycles = [
      { key: 'forward_pass', num: 1, title: d.step3_cycle_forward, desc: d.step3_cycle_forward_desc, tip: d.step3_cycle_forward_tooltip },
      { key: 'loss_function', num: 2, title: d.step3_cycle_loss, desc: d.step3_cycle_loss_desc, tip: d.step3_cycle_loss_tooltip },
      { key: 'backpropagation', num: 3, title: d.step3_cycle_backprop, desc: d.step3_cycle_backprop_desc, tip: d.step3_cycle_backprop_tooltip },
      { key: 'optimizer_update', num: 4, title: d.step3_cycle_update, desc: d.step3_cycle_update_desc, tip: d.step3_cycle_update_tooltip },
    ];
    return (
      <div className="academy-step-content">
        <SectionCard sectionCode="03" title={d.step3_title} subtitle={d.step3_why} termKey="loss_function" tooltipText={d.step3_why}>
          <p>{d.step3_content_1}</p>
          <p>{d.step3_content_2}</p>
          <p>{d.step3_content_3}</p>

          <div className="training-cycle">
            {cycles.map((c) => (
              <div key={c.num} className="cycle-step">
                <div className="cycle-step-number">{c.num}</div>
                <div className="cycle-step-title">
                  <span>{c.title}</span>
                  <TermTooltip termKey={c.key} explanation={c.tip} title={c.title} hideLabel={true} size="sm" />
                </div>
                <div className="cycle-step-desc">{c.desc}</div>
              </div>
            ))}
          </div>

          <p style={{ textAlign: 'center', color: 'var(--accent-teal)', fontWeight: 600, marginTop: '1rem' }}>
            {d.step3_training_label}: {d.step3_training_desc}
          </p>
        </SectionCard>
      </div>
    );
  };

  const renderStep4 = () => (
    <div className="academy-step-content">
      <SectionCard sectionCode="04" title={d.step4_title} subtitle={d.step4_why} termKey="parameters_weights" tooltipText={d.step4_why}>
        <p>{d.step4_content_1}</p>
        <p>{d.step4_content_2}</p>

        {/* Factory Diagram: Data x --> Factory f --> Answer y */}
        <div className="diagram-container">
          <div className="diagram-box">
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)' }}>x</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.step4_diagram_input}</span>
          </div>
          <span className="diagram-arrow"><ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} /></span>
          <div className="diagram-box highlight">
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>f</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.step4_diagram_function}</span>
          </div>
          <span className="diagram-arrow"><ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} /></span>
          <div className="diagram-box">
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>y</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.step4_diagram_output}</span>
          </div>
        </div>

        {/* Parameters & Architecture */}
        <div className="comparison-grid">
          <div className="comparison-column">
            <div className="comparison-column-title" style={{ color: 'var(--accent-teal)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>{d.step4_params_title}</span>
              <TermTooltip termKey="parameters_weights" explanation={d.step4_params_tooltip} title={d.step4_params_title} hideLabel={true} />
            </div>
            <p className="branch-card-desc">{d.step4_params_desc}</p>
          </div>
          <div className="comparison-column">
            <div className="comparison-column-title" style={{ color: 'var(--accent-amber)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>{d.step4_arch_title}</span>
              <TermTooltip termKey="neural_architecture" explanation={d.step4_arch_tooltip} title={d.step4_arch_title} hideLabel={true} />
            </div>
            <p className="branch-card-desc">{d.step4_arch_desc}</p>
          </div>
        </div>

        {/* Foundation Model & Fine-tuning */}
        <div className="branch-grid" style={{ marginTop: '1rem' }}>
          <div className="branch-card">
            <div className="branch-card-title">
              <span>{d.step4_foundation}</span>
              <TermTooltip termKey="foundation_model" explanation={d.step4_foundation_tooltip} title={d.step4_foundation} hideLabel={true} />
            </div>
            <div className="branch-card-desc">{d.step4_foundation_desc}</div>
          </div>
          <div className="branch-card">
            <div className="branch-card-title">
              <span>{d.step4_finetuning}</span>
              <TermTooltip termKey="fine_tuning" explanation={d.step4_finetuning_tooltip} title={d.step4_finetuning} hideLabel={true} />
            </div>
            <div className="branch-card-desc">{d.step4_finetuning_desc}</div>
          </div>
        </div>
      </SectionCard>
    </div>
  );

  const renderStep5 = () => (
    <div className="academy-step-content">
      <SectionCard sectionCode="05" title={d.step5_title} subtitle={d.step5_why} termKey="tokenization" tooltipText={d.step5_why}>
        <p>{d.step5_content_1}</p>
        <p>{d.step5_content_2}</p>
        <p>{d.step5_content_3}</p>

        {/* Interactive tokenizer */}
        <div style={{
          margin: '1.5rem 0', padding: '1.5rem',
          background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--border-subtle)',
        }}>
          <p style={{ fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
            {d.step5_try_it}
          </p>
          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
            <input
              type="text"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleTokenize()}
              placeholder={d.step5_input_placeholder}
              style={{
                flex: 1, padding: '0.75rem 1rem',
                background: 'var(--bg-secondary)', border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)',
                fontSize: '0.9rem',
              }}
            />
            <button onClick={handleTokenize} className="btn btn-primary" style={{ padding: '0.75rem 1.5rem' }}>
              {language === 'ar' ? 'جزّئ' : 'Tokenize'}
            </button>
          </div>

          {tokenResult && tokenResult.tokens && (
            <div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                {d.step5_parent_word}: {tokenResult.word_count} | {d.step5_token_piece}: {tokenResult.total_tokens}
              </div>
              <div className="token-grid">
                {tokenResult.tokens.map((t, idx) => (
                  <div key={idx} className="token-block">
                    <span className="token-block-id">{d.step5_token_id}: {t.token_id}</span>
                    <span className="token-block-piece">{t.token_piece}</span>
                    <span className="token-block-parent">{t.parent_word}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );

  const renderStep6 = () => (
    <div className="academy-step-content">
      <SectionCard
        sectionCode="06"
        title={d.step6_title}
        subtitle={d.step6_why}
        termKey="embedding"
        tooltipText={d.step6_tooltip}
      >
        <p>{d.step6_content_1}</p>
        <p>{d.step6_content_2}</p>
        <p>{d.step6_content_3}</p>

        {/* Simple embedding visualization */}
        <div className="diagram-container" style={{ flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center' }}>
            {language === 'ar'
              ? 'مثال: كلمة "ملك" تتحول الى متجه من 4 ارقام (مبسط):'
              : 'Example: the word "king" becomes a vector of 4 numbers (simplified):'}
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', fontFamily: 'var(--font-mono)' }}>
            {[0.82, -0.15, 0.64, 0.33].map((v, i) => (
              <div key={i} style={{
                padding: '0.5rem 0.75rem', borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-surface)', border: '1px solid var(--border-medium)',
                color: v >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                fontSize: '0.85rem', fontWeight: 600,
              }}>
                {v > 0 ? '+' : ''}{v.toFixed(2)}
              </div>
            ))}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
            {language === 'ar'
              ? '"ملكة" ستكون قريبة: [+0.80, -0.12, +0.61, +0.35]'
              : '"queen" would be close: [+0.80, -0.12, +0.61, +0.35]'}
          </div>
        </div>
      </SectionCard>
    </div>
  );

  const renderStep7 = () => {
    const arabicWords = ['الطالب', 'ذهب', 'إلى', 'المدرسة', 'لأنه', 'يحب', 'العلم'];
    const englishWords = ['The', 'student', 'went', 'to', 'school', 'because', 'he', 'loves', 'learning'];
    const words = language === 'ar' ? arabicWords : englishWords;

    const attentionMatrix = language === 'ar' ? [
      [0.15, 0.35, 0.05, 0.25, 0.08, 0.07, 0.05],
      [0.32, 0.12, 0.18, 0.28, 0.04, 0.03, 0.03],
      [0.05, 0.20, 0.10, 0.60, 0.02, 0.01, 0.02],
      [0.22, 0.30, 0.25, 0.15, 0.03, 0.02, 0.03],
      [0.45, 0.05, 0.02, 0.08, 0.15, 0.20, 0.05],
      [0.25, 0.04, 0.02, 0.06, 0.18, 0.15, 0.30],
      [0.12, 0.03, 0.01, 0.14, 0.05, 0.40, 0.25],
    ] : [
      [0.10, 0.60, 0.10, 0.05, 0.05, 0.04, 0.02, 0.02, 0.02],
      [0.15, 0.20, 0.35, 0.05, 0.15, 0.03, 0.02, 0.03, 0.02],
      [0.05, 0.30, 0.15, 0.20, 0.25, 0.02, 0.01, 0.01, 0.01],
      [0.02, 0.05, 0.20, 0.10, 0.60, 0.01, 0.01, 0.01, 0.00],
      [0.10, 0.15, 0.30, 0.30, 0.10, 0.02, 0.01, 0.01, 0.01],
      [0.30, 0.25, 0.05, 0.05, 0.05, 0.10, 0.15, 0.03, 0.02],
      [0.05, 0.40, 0.05, 0.02, 0.08, 0.12, 0.08, 0.15, 0.05],
      [0.10, 0.20, 0.05, 0.02, 0.08, 0.05, 0.15, 0.10, 0.25],
      [0.05, 0.15, 0.02, 0.05, 0.10, 0.03, 0.05, 0.35, 0.20],
    ];

    const currentScores = attentionMatrix[selectedWordIdx % words.length] || attentionMatrix[0];

    return (
      <div className="academy-step-content">
        <SectionCard
          sectionCode="07"
          title={d.step7_title}
          subtitle={d.step7_why}
          termKey="self_attention"
          tooltipText={d.step7_tooltip}
        >
          <p>{d.step7_content_1}</p>
          <p>{d.step7_content_2}</p>
          <p>{d.step7_content_3}</p>

          <div style={{
            margin: '1.5rem 0', padding: '1.5rem',
            background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-subtle)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {language === 'ar'
                  ? 'انقر على أي كلمة لاختيارها كـ Query ورؤية درجات الانتباه المتبادل لباقي الكلمات:'
                  : 'Click any word as Query to inspect dynamic attention affinities:'}
              </div>
              <span className="badge badge-teal" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                Softmax(Q·K^T / √d)
              </span>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1.25rem' }}>
              {words.map((w, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedWordIdx(idx)}
                  className={`btn btn-sm ${selectedWordIdx === idx ? 'btn-primary' : 'btn-secondary'}`}
                  style={{
                    padding: '0.45rem 0.85rem',
                    borderRadius: '6px',
                    fontWeight: selectedWordIdx === idx ? 800 : 500,
                  }}
                >
                  {w} {selectedWordIdx === idx ? (isAr ? '(المستفهم Q)' : '(Query)') : ''}
                </button>
              ))}
            </div>

            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
              {isAr
                ? `درجة انتباه كلمة "${words[selectedWordIdx % words.length]}" بالكلمات الأخرى:`
                : `Attention weights of "${words[selectedWordIdx % words.length]}" across the sequence:`}
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {words.map((w, i) => {
                const score = currentScores[i] || 0.05;
                const isCurrent = i === (selectedWordIdx % words.length);
                return (
                  <div
                    key={i}
                    style={{
                      flex: '1 1 90px',
                      padding: '0.75rem 0.5rem',
                      borderRadius: 'var(--radius-sm)',
                      border: isCurrent ? '2px solid var(--accent-teal)' : '1px solid var(--border-medium)',
                      background: `rgba(45, 212, 191, ${Math.max(0.12, score * 1.5)})`,
                      color: score > 0.25 ? '#ffffff' : 'var(--text-primary)',
                      textAlign: 'center',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <div style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.25rem' }}>{w}</div>
                    <div style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
                      {(score * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </SectionCard>
      </div>
    );
  };

  const renderStep8 = () => (
    <div className="academy-step-content">
      <SectionCard sectionCode="08" title={d.step8_title} subtitle={d.step8_why} termKey="multi_task_learning" tooltipText={d.step8_why}>
        <p>{d.step8_content_1}</p>
        <p>{d.step8_content_2}</p>
        <p>{d.step8_content_3}</p>

        {/* Tree diagram: Backbone + Task Heads */}
        <div className="diagram-container" style={{ flexDirection: 'column', gap: '1rem' }}>
          {/* Task Heads */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            {[
              language === 'ar' ? 'تحليل المشاعر' : 'Sentiment',
              language === 'ar' ? 'تصنيف الموضوعات' : 'Topic',
              language === 'ar' ? 'استخراج الكيانات' : 'NER',
            ].map((head, i) => (
              <div key={i} className="diagram-box" style={{ minWidth: '120px', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem' }}>
                  <span>{d.step8_task_head} {i + 1}</span>
                  <TermTooltip termKey="multi_task_learning" explanation={d.step8_task_head_tooltip} title={d.step8_task_head} hideLabel={true} size="sm" />
                </div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{head}</span>
              </div>
            ))}
          </div>
          {/* Arrows up */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            {[1, 2, 3].map((_, i) => (
              <span key={i} style={{ color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', transform: 'rotate(-90deg)', fontSize: '1.2rem' }}>
                --&gt;
              </span>
            ))}
          </div>
          {/* Backbone */}
          <div className="diagram-box highlight" style={{ width: '80%', maxWidth: '400px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.35rem' }}>
              <span style={{ fontWeight: 700 }}>{d.step8_backbone}</span>
              <TermTooltip termKey="neural_architecture" explanation={d.step8_backbone_tooltip} title={d.step8_backbone} hideLabel={true} size="sm" />
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Transformer Encoder
            </span>
          </div>
        </div>
      </SectionCard>
    </div>
  );

  const renderStep9 = () => (
    <div className="academy-step-content">
      <SectionCard sectionCode="09" title={d.step9_title} subtitle={d.step9_why} termKey="catastrophic_forgetting" tooltipText={d.step9_why}>
        <p>{d.step9_content_1}</p>
        <p style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
          <span>{d.step9_content_2}</span>
          <TermTooltip termKey="elastic_weight_consolidation" explanation={d.step9_ewc_tooltip} title={language === 'ar' ? 'مصفوفة فيشر وتنظيم EWC' : 'EWC Fisher Regularization'} hideLabel={true} />
        </p>
        <p style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexWrap: 'wrap' }}>
          <span>{d.step9_content_3}</span>
          <TermTooltip termKey="experience_replay" explanation={d.step9_replay_tooltip} title={language === 'ar' ? 'إعادة تشغيل الذاكرة' : 'Experience Replay'} hideLabel={true} />
        </p>

        {/* Toggle protection */}
        <div style={{
          margin: '1.5rem 0', padding: '1.25rem',
          background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--border-subtle)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <label style={{ fontSize: '0.9rem', color: 'var(--text-primary)', fontWeight: 600 }}>
              {d.step9_toggle_label}
            </label>
            <button
              onClick={() => setClProtection(!clProtection)}
              style={{
                padding: '0.4rem 1rem', borderRadius: 'var(--radius-sm)',
                border: `1px solid ${clProtection ? 'var(--accent-emerald)' : 'var(--accent-rose)'}`,
                background: clProtection ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                color: clProtection ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer',
              }}
            >
              {clProtection ? d.step9_with_protection : d.step9_without_protection}
            </button>
          </div>

          {/* Simple bar chart of final accuracies */}
          {clData && (
            <div className="comparison-grid">
              <div className={`comparison-column ${clProtection ? 'good' : 'bad'}`}>
                <div className="comparison-column-title">
                  {language === 'ar' ? 'المهمة الاولى (المشاعر)' : 'Task 1 (Sentiment)'}
                </div>
                <div style={{
                  height: '8px', borderRadius: '4px',
                  background: 'var(--bg-surface)', overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%', borderRadius: '4px',
                    width: `${clData.task1_accuracy[clData.task1_accuracy.length - 1]}%`,
                    background: clProtection ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                    transition: 'width 0.5s ease',
                  }} />
                </div>
                <div style={{ fontSize: '0.82rem', marginTop: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {clData.task1_accuracy[clData.task1_accuracy.length - 1]}%
                </div>
              </div>
              <div className="comparison-column good">
                <div className="comparison-column-title" style={{ color: 'var(--accent-teal)' }}>
                  {language === 'ar' ? 'المهمة الثانية (الموضوعات)' : 'Task 2 (Topic)'}
                </div>
                <div style={{
                  height: '8px', borderRadius: '4px',
                  background: 'var(--bg-surface)', overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%', borderRadius: '4px',
                    width: `${clData.task2_accuracy[clData.task2_accuracy.length - 1]}%`,
                    background: 'var(--accent-teal)',
                    transition: 'width 0.5s ease',
                  }} />
                </div>
                <div style={{ fontSize: '0.82rem', marginTop: '0.5rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {clData.task2_accuracy[clData.task2_accuracy.length - 1]}%
                </div>
              </div>
            </div>
          )}
        </div>
      </SectionCard>
    </div>
  );

  const renderStep10 = () => {
    const dModel = 512;
    const baseParams = dModel * dModel;
    const loraParams = 2 * dModel * loraRank;
    const reductionPercent = ((1 - loraParams / baseParams) * 100).toFixed(2);

    return (
      <div className="academy-step-content">
        <SectionCard sectionCode="10" title={d.step10_title} subtitle={d.step10_why} termKey="multi_task_lora" tooltipText={d.step10_why}>
          <p>{d.step10_content_1}</p>
          <p>{d.step10_content_2}</p>
          <p>{d.step10_content_3}</p>

          {/* Interactive LoRA Rank Controller */}
          <div style={{
            background: 'var(--bg-tertiary)',
            padding: '1.25rem',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-subtle)',
            marginBottom: '1.5rem',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div>
                <label style={{ fontSize: '0.9rem', color: 'var(--text-primary)', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>{isAr ? 'رتبة التكيف المنخفضة (LoRA Rank - r):' : 'Low-Rank Adaptation Rank (r):'}</span>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>r =</span>
                    <input
                      type="number"
                      min="1"
                      max="16"
                      step="1"
                      value={loraRank}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        if (!isNaN(val)) setLoraRank(Math.min(16, Math.max(1, val)));
                      }}
                      style={{ width: '35px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                    />
                  </div>
                </label>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {isAr ? 'الرتبة تحدد عنق الزجاجة لضغط الأوزان (d x r و r x d)' : 'Rank specifies the bottleneck dimension for low-rank matrix decomposition'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                {[1, 2, 4, 8, 16].map((rk) => (
                  <button
                    key={rk}
                    onClick={() => setLoraRank(rk)}
                    style={{
                      padding: '0.3rem 0.75rem',
                      borderRadius: 'var(--radius-sm)',
                      border: loraRank === rk ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                      background: loraRank === rk ? 'rgba(20,184,166,0.15)' : 'var(--bg-surface)',
                      color: loraRank === rk ? 'var(--accent-teal)' : 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                    }}
                  >
                    r={rk}
                  </button>
                ))}
              </div>
            </div>

            <input
              type="range"
              min="1"
              max="16"
              step="1"
              value={loraRank}
              onChange={(e) => setLoraRank(parseInt(e.target.value))}
              className="slider-input"
              style={{ width: '100%', accentColor: 'var(--accent-teal)', cursor: 'pointer', marginBottom: '1rem' }}
            />

            {/* Parameter & Reduction Statistics */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '0.75rem',
              marginTop: '0.5rem',
            }}>
              <div style={{ background: 'var(--bg-surface)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'معاملات الأساس المجمدة (W₀)' : 'Base Frozen Params (W₀)'}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {baseParams.toLocaleString()} ({dModel}×{dModel})
                </div>
              </div>

              <div style={{ background: 'rgba(20,184,166,0.08)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid rgba(20,184,166,0.3)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--accent-teal)' }}>{isAr ? 'معاملات LoRA القابلة للتدريب (A, B)' : 'Trainable LoRA Params (A, B)'}</div>
                <div style={{ fontSize: '1rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>
                  {loraParams.toLocaleString()} (2×{dModel}×{loraRank})
                </div>
              </div>

              <div style={{ background: 'rgba(16,185,129,0.08)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16,185,129,0.3)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--accent-emerald)' }}>{isAr ? 'وفر استهلاك المعاملات والذاكرة' : 'Parameter & Memory Savings'}</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                  {reductionPercent}%
                </div>
              </div>
            </div>
          </div>

          {/* LoRA Dynamic Matrix Decomposition Diagram */}
          <div className="diagram-container" style={{ flexDirection: 'column', gap: '1.5rem', padding: '1.75rem', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap', direction: 'ltr' }}>
              {/* Frozen Base Matrix W0 */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{
                  width: '120px', height: '120px',
                  background: 'var(--bg-surface)', border: '2px solid var(--border-medium)',
                  borderRadius: 'var(--radius-md)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexDirection: 'column', gap: '0.3rem',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                }}>
                  <span style={{ fontSize: '1.35rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>W₀</span>
                  <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', background: 'var(--bg-tertiary)', padding: '0.15rem 0.45rem', borderRadius: '4px' }}>
                    {dModel} × {dModel}
                  </span>
                </div>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)' }}>{d.step10_frozen}</span>
              </div>

              <span style={{ fontSize: '1.8rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>+</span>

              {/* LoRA Adapters B x A */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', background: 'rgba(20, 184, 166, 0.05)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px dashed rgba(20, 184, 166, 0.3)' }}>
                {/* Matrix B */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{
                    width: `${Math.max(34, loraRank * 6)}px`, height: '120px',
                    background: 'rgba(20,184,166,0.18)', border: '2px solid var(--accent-teal)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexDirection: 'column', gap: '0.2rem',
                    transition: 'width 0.3s ease',
                    boxShadow: '0 4px 12px rgba(20,184,166,0.15)',
                  }}>
                    <span style={{ fontSize: '0.95rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>B</span>
                    <span style={{ fontSize: '0.62rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>{dModel}×{loraRank}</span>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)', fontWeight: 700 }}>{isAr ? 'مصفوفة B' : 'Matrix B'}</span>
                </div>

                <span style={{ fontSize: '1.2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>×</span>

                {/* Matrix A */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{
                    width: '120px', height: `${Math.max(34, loraRank * 6)}px`,
                    background: 'rgba(20,184,166,0.18)', border: '2px solid var(--accent-teal)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexDirection: 'column', gap: '0.2rem',
                    transition: 'height 0.3s ease',
                    boxShadow: '0 4px 12px rgba(20,184,166,0.15)',
                  }}>
                    <span style={{ fontSize: '0.95rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>A</span>
                    <span style={{ fontSize: '0.62rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>{loraRank}×{dModel}</span>
                  </div>
                  <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)', fontWeight: 700 }}>{isAr ? 'مصفوفة A' : 'Matrix A'}</span>
                </div>
              </div>

              <span style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>=</span>

              {/* Low Rank Result Delta W */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{
                  width: '120px', height: '120px',
                  background: 'rgba(16,185,129,0.12)', border: '2px dashed var(--accent-emerald)',
                  borderRadius: 'var(--radius-md)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexDirection: 'column', gap: '0.25rem',
                  boxShadow: '0 4px 14px rgba(16,185,129,0.15)',
                }}>
                  <span style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>ΔW</span>
                  <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)', fontWeight: 700 }}>Rank-{loraRank} Update</span>
                </div>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>{isAr ? 'تعديل منخفض الرتبة' : 'Low-Rank Delta'}</span>
              </div>
            </div>

            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textAlign: 'center', maxWidth: '700px', margin: '0 auto', lineHeight: 1.6 }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ fontWeight: 700, color: 'var(--accent-teal)' }}>{d.step10_trainable}</span>
                <TermTooltip termKey="multi_task_lora" explanation={d.step10_trainable_tooltip} title={d.step10_trainable} hideLabel={true} size="sm" />
              </div>
              {' — '}
              {isAr
                ? `عند رتبة r = ${loraRank}، يتم تدريب ${loraParams.toLocaleString()} وزناً فقط بدلاً من ${baseParams.toLocaleString()}، مما يوفر ${reductionPercent}% من عبء الحساب والذاكرة دون تراجع في الجودة.`
                : `At rank r = ${loraRank}, only ${loraParams.toLocaleString()} weights are tuned instead of ${baseParams.toLocaleString()}, delivering ${reductionPercent}% reduction in computation & storage without precision degradation.`}
            </div>
          </div>
        </SectionCard>

        {/* Transition to next module */}
        <button
          className="btn btn-primary"
          style={{
            width: '100%',
            padding: '1rem',
            fontSize: '0.98rem',
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
            marginTop: '1.25rem',
          }}
          onClick={() => setActiveModule('problem')}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem' }}>
            <span>{isAr ? 'انتقل إلى الخطوة التالية: تحديد وصياغة المشكلة (المرحلة 01)' : 'Proceed to Next Step: Problem Framing (Stage 01)'}</span>
            <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          </span>
        </button>
      </div>
    );
  };

  const stepRenderers = [
    renderStep1, renderStep2, renderStep3, renderStep4, renderStep5,
    renderStep6, renderStep7, renderStep8, renderStep9, renderStep10,
  ];

  return (
    <div className="container" style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Page Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-teal">[00] {d.module_academy || (isAr ? 'الأكاديمية التعليمية' : 'Academy Wizard')}</span>
        </div>
        <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          {d.academy_title}
        </h1>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          {d.academy_subtitle}
        </p>
      </div>

      {/* Progress Bar */}
      <StepProgressBar
        currentStep={currentStep}
        totalSteps={TOTAL_STEPS}
        labels={stepLabels}
        onStepClick={setCurrentStep}
        language={language}
      />

      {/* Step Content */}
      <div style={{ marginBottom: '2rem' }}>
        {stepRenderers[currentStep - 1]()}
      </div>

      {/* Navigation Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
        <button
          onClick={goPrev}
          disabled={currentStep === 1}
          className="btn btn-secondary"
          style={{
            padding: '0.75rem 1.5rem',
            opacity: currentStep === 1 ? 0.3 : 1,
            cursor: currentStep === 1 ? 'not-allowed' : 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          <span>{d.btn_prev_step}</span>
        </button>

        <span style={{
          fontSize: '0.85rem', color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          alignSelf: 'center',
        }}>
          {currentStep} {d.step_of} {TOTAL_STEPS}
        </span>

        <button
          onClick={goNext}
          disabled={currentStep === TOTAL_STEPS}
          className="btn btn-primary"
          style={{
            padding: '0.75rem 1.5rem',
            opacity: currentStep === TOTAL_STEPS ? 0.3 : 1,
            cursor: currentStep === TOTAL_STEPS ? 'not-allowed' : 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <span>{d.btn_next_step}</span>
          <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
        </button>
      </div>
    </div>
  );
}
