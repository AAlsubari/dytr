// ==============================================================================
// DyTrStudio Visual Model Explainer (المفسر البصري التفاعلي لبناء النماذج)
// Step-by-Step Pedagogical Transformer Architecture Visualizer
// Zero Emojis Policy - High Rigor Arabic & English Educational Guidance
// Integrated with 3-Layer Epistemological TermTooltip System
// ==============================================================================

import React, { useState } from 'react';
import TermTooltip from '../common/TermTooltip';
import { ArrowRight } from 'lucide-react';

export default function VisualModelExplainer({ isAr }) {
  const [activeStep, setActiveStep] = useState(1);
  const [viewMode, setViewMode] = useState('simple'); // 'simple' | 'technical'

  const steps = [
    {
      id: 1,
      stepNumber: '01',
      termKey: 'embedding_params',
      title_ar: '1. بوابة الاستقبال وتشفير الكلمات (Tokenization & Embeddings)',
      title_en: '1. Input Ingestion & Token Embeddings',
      short_ar: 'تقطيع وتشفير الكلمات',
      short_en: 'Tokenization & Vectors',
      analogy_ar: 'مثل تقطيع الجملة إلى قطع ليجو رقمية، وإعطاء كل كلمة بطاقة إحداثيات تعبر عن معناها في فضاء اللغة.',
      analogy_en: 'Like breaking a sentence into digital Lego bricks and giving each word spatial coordinates representing its meaning.',
      technical_ar: 'يتم تجزئة النص إلى وحدات لغوية (Tokens) ثم تحويلها عبر مصفوفة التضمين إلى متجهات رقمية متصلة ذات أبعاد d_model، مع حقن التشفير الموضعي (RoPE) لحفظ ترتيب الكلمات.',
      technical_en: 'Text is split into sub-word tokens, mapped via embedding table into d_model dimensional vectors, and combined with Rotary Position Embeddings (RoPE).',
      highlight_tags: [
        { label: 'd_model (أبعاد التضمين)', termKey: 'd_model' },
        { label: 'RoPE (التضمين الدوراني)', termKey: 'rope' },
        { label: 'Embedding Matrix', termKey: 'embedding_params' },
      ],
      diagramType: 'input_embedding',
    },
    {
      id: 2,
      stepNumber: '02',
      termKey: 'num_heads',
      title_ar: '2. ورشة التفكير والاهتمام المشترك (Transformer Layers & Attention)',
      title_en: '2. Thinking Engine & Multi-Head Self-Attention',
      short_ar: 'فهم السياق المشترك',
      short_en: 'Context & Attention',
      analogy_ar: 'مثل فريق من المحللين الأذكياء ينظرون إلى جميع الكلمات في وقت واحد، ليفهموا علاقة كل كلمة بما حولها بدقة.',
      analogy_en: 'Like a team of expert readers analyzing all words simultaneously to discover semantic links and contextual nuances.',
      technical_ar: 'تتكون من طبقات متتابعة من الاهتمام الذاتي متعدد الرؤوس (Multi-Head Attention) والشبكات الأمامية (FFN)، حيث تحسب مصفوفات الاستعلام والمفتاح والقيمة (Q, K, V) الأوزان النسبية للترابط السياقي.',
      technical_en: 'Stacked blocks containing Multi-Head Attention and Feed-Forward Networks computing Query-Key-Value interactions to construct contextualized representations.',
      highlight_tags: [
        { label: 'Multi-Head Attention', termKey: 'num_heads' },
        { label: 'Transformer Layers', termKey: 'num_layers' },
        { label: 'FlashAttention IO', termKey: 'flash_attention' },
      ],
      diagramType: 'attention_layers',
    },
    {
      id: 3,
      stepNumber: '03',
      termKey: 'ewc',
      title_ar: '3. دروع الذاكرة وحماية المعرفة (Continual Learning & Adapters)',
      title_en: '3. Memory Shields & Continual Adaptation',
      short_ar: 'حماية الذاكرة ومنع النسيان',
      short_en: 'Memory Protection',
      analogy_ar: 'درع واقٍ يقفل على المعلومات والمهارات السابقة حتى لا ينساها النموذج عندما يتعلم مهام جديدة ومستمرة.',
      analogy_en: 'A protective shield locking critical past knowledge so the model continuously learns new tasks without catastrophic forgetting.',
      technical_ar: 'تطبيق خوارزمية تجميع الأوزان المرنة (EWC) عبر مصفوفة فيشر لحساب حساسية المعاملات، مع دمج أغلفة LoRA منخفضة الرتبة وذاكرة استرجاع العينات (Experience Replay).',
      technical_en: 'Elastic Weight Consolidation (EWC) penalizes critical parameter shifts alongside Low-Rank Adaptation (LoRA) and experience rehearsal replay buffers.',
      highlight_tags: [
        { label: 'EWC Shield (درع فيشر)', termKey: 'ewc' },
        { label: 'LoRA Adapters (المواءمة)', termKey: 'lora' },
        { label: 'Experience Replay', termKey: 'experience_replay' },
      ],
      diagramType: 'memory_shield',
    },
    {
      id: 4,
      stepNumber: '04',
      termKey: 'task_heads',
      title_ar: '4. مكاتب اتخاذ القرار والمهام المتخصصة (Multi-Task Prediction Heads)',
      title_en: '4. Task Heads & Decision Projections',
      short_ar: 'اتخاذ القرارات المتعددة',
      short_en: 'Task Decisions',
      analogy_ar: 'مثل عدة إدارات متخصصة تستلم نفس التقرير الشامل: إدارة تحلل المشاعر، وإدارة تستخرج الأسماء، وإدارة تصيغ الإجابات.',
      analogy_en: 'Like specialized departments receiving the same intelligence briefing: one classifies sentiment, another extracts names, and a third generates answers.',
      technical_ar: 'طبقات إسقاط خطية مستقلة (Linear Output Heads) متصلة بالعمود الفقري المشترك، تمكن النموذج من إنجاز مهام تصنيف الجمل، تصنيف الكلمات (NER)، والتوليد اللغوي.',
      technical_en: 'Independent linear output projection heads branching from the shared backbone to execute classification, entity recognition, and text generation.',
      highlight_tags: [
        { label: 'Multi-Task Routing', termKey: 'task_heads' },
        { label: 'Task Framing', termKey: 'task_type' },
        { label: 'Column Mapping', termKey: 'column_mapping' },
      ],
      diagramType: 'task_heads',
    },
  ];


  const curr = steps.find((s) => s.id === activeStep) || steps[0];

  return (
    <div
      style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border-medium)',
        borderRadius: 'var(--radius-md, 8px)',
        padding: '1.25rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        marginBottom: '1.75rem',
      }}
    >
      {/* Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className="badge badge-teal" style={{ fontSize: '0.72rem', fontWeight: 800 }}>
              {isAr ? 'دليل الفهم المرئي' : 'Visual Learning Guide'}
            </span>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
              {isAr ? 'كيف يتم بناء النموذج والمحول الذكي خطوة بخطوة؟' : 'How Are Neural Transformers & AI Models Constructed?'}
            </h3>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem', margin: 0 }}>
            {isAr
              ? 'رحلة البيانات من النص البسيط حتى اتخاذ القرارات الذكية وتعدد المهام عبر 4 مراحل هندسية واضحة.'
              : 'The step-by-step journey from raw text into deep contextual understanding and multi-task predictions.'}
          </p>
        </div>

        {/* View Mode Toggle */}
        <div style={{ display: 'inline-flex', background: 'var(--bg-secondary)', padding: '0.2rem', borderRadius: 'var(--radius-sm, 6px)', border: '1px solid var(--border-subtle)' }}>
          <button
            type="button"
            onClick={() => setViewMode('simple')}
            style={{
              padding: '0.3rem 0.75rem',
              borderRadius: 'var(--radius-sm, 4px)',
              border: 'none',
              background: viewMode === 'simple' ? 'var(--accent-teal)' : 'transparent',
              color: viewMode === 'simple' ? '#ffffff' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isAr ? 'الشرح المبسط' : 'Intuitive Analogy'}
          </button>
          <button
            type="button"
            onClick={() => setViewMode('technical')}
            style={{
              padding: '0.3rem 0.75rem',
              borderRadius: 'var(--radius-sm, 4px)',
              border: 'none',
              background: viewMode === 'technical' ? 'var(--accent-teal)' : 'transparent',
              color: viewMode === 'technical' ? '#ffffff' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isAr ? 'الشرح الهندسي الرياضي' : 'Technical Architecture'}
          </button>
        </div>
      </div>

      {/* 4 Interactive Step Selector Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.65rem' }}>
        {steps.map((st) => {
          const isActive = st.id === activeStep;
          return (
            <button
              key={st.id}
              type="button"
              onClick={() => setActiveStep(st.id)}
              style={{
                textAlign: isAr ? 'right' : 'left',
                padding: '0.75rem 0.85rem',
                borderRadius: 'var(--radius-sm, 6px)',
                background: isActive ? 'rgba(20, 184, 166, 0.12)' : 'var(--bg-tertiary)',
                border: isActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.3rem',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span
                  style={{
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 800,
                    color: isActive ? 'var(--accent-teal)' : 'var(--text-muted)',
                  }}
                >
                  {st.stepNumber}
                </span>
                {isActive && (
                  <span className="badge badge-teal" style={{ fontSize: '0.62rem', padding: '0.05rem 0.35rem' }}>
                    {isAr ? 'الطور النشط' : 'Active'}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '0.82rem', fontWeight: 800, color: isActive ? 'var(--accent-teal)' : 'var(--text-primary)' }}>
                {isAr ? st.short_ar : st.short_en}
              </div>
            </button>
          );
        })}
      </div>

      {/* Active Step Detailed Content Card */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm, 6px)',
          padding: '1.1rem 1.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.85rem',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--accent-teal)', margin: 0 }}>
              {isAr ? curr.title_ar : curr.title_en}
            </h4>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-primary)', marginTop: '0.45rem', lineHeight: 1.6, margin: '0.45rem 0 0 0' }}>
              {viewMode === 'simple' ? (isAr ? curr.analogy_ar : curr.analogy_en) : (isAr ? curr.technical_ar : curr.technical_en)}
            </p>
          </div>
        </div>

        {/* Feature Tags for this phase */}
        <div style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap', alignItems: 'center', paddingTop: '0.45rem', borderTop: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--text-muted)' }}>
            {isAr ? 'المفاهيم الهندسية ومراجعها الأكاديمية (انقر للشرح ثلاثي المستويات):' : 'Architectural Concepts & References (3-Layer Explanation):'}
          </span>
          {curr.highlight_tags.map((tg, idx) => (
            <div key={idx} style={{ display: 'inline-flex', alignItems: 'center', background: 'var(--bg-tertiary)', padding: '0.15rem 0.45rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
              <TermTooltip termKey={tg.termKey || tg} customLabel={tg.label || tg} size="sm" />
            </div>
          ))}
        </div>

        {/* Dynamic Interactive SVG Diagram for each step */}
        <div
          style={{
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm, 6px)',
            padding: '1rem',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '130px',
            overflowX: 'auto',
          }}
        >
          {curr.diagramType === 'input_embedding' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              <div style={{ padding: '0.5rem 0.85rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-medium)', borderRadius: '6px', fontSize: '0.8rem', textAlign: 'center' }}>
                <span style={{ color: 'var(--text-secondary)', display: 'block', fontSize: '0.7rem' }}>{isAr ? 'النص المدخل' : 'Raw Text'}</span>
                <strong style={{ color: 'var(--text-primary)' }}>&ldquo;الذكاء الاصطناعي&rdquo;</strong>
              </div>
              <ArrowRight size={18} style={{ color: 'var(--accent-teal)', transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <div style={{ display: 'flex', gap: '0.3rem' }}>
                {['الـ', 'ذكاء', 'الـ', 'اصطناعي'].map((t, idx) => (
                  <div key={idx} style={{ padding: '0.35rem 0.55rem', background: 'rgba(20, 184, 166, 0.1)', border: '1px solid var(--accent-teal)', borderRadius: '4px', fontSize: '0.75rem', color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                    {t}
                  </div>
                ))}
              </div>
              <ArrowRight size={18} style={{ color: 'var(--accent-teal)', transform: isAr ? 'scaleX(-1)' : 'none' }} />
              <div style={{ padding: '0.5rem 0.85rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-medium)', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                [0.84, -0.12, 0.45, ... <strong>d_model</strong>]
              </div>
            </div>
          )}

          {curr.diagramType === 'attention_layers' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: '100%', maxWidth: '480px', textAlign: 'center' }}>
              <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', gap: '0.5rem' }}>
                {['Head 1 (دلالي)', 'Head 2 (نحوي)', 'Head 3 (مكاني)', 'Head 4 (توليدي)'].map((h, hIdx) => (
                  <div key={hIdx} style={{ padding: '0.35rem 0.5rem', background: 'var(--bg-secondary)', border: '1px solid var(--accent-teal)', borderRadius: '4px', fontSize: '0.7rem', color: 'var(--accent-teal)', fontWeight: 700 }}>
                    {isAr ? h : `Head ${hIdx + 1}`}
                  </div>
                ))}
              </div>
              <div style={{ padding: '0.6rem', background: 'rgba(20, 184, 166, 0.1)', border: '1px dashed var(--accent-teal)', borderRadius: '6px', fontSize: '0.78rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                {isAr ? 'حساب مصفوفات الاهتمام الذاتي والترابط السياقي (Softmax(QK^T / √d_k) · V)' : 'Attention Map & Context Projection'}
              </div>
            </div>
          )}

          {curr.diagramType === 'memory_shield' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              <div style={{ padding: '0.6rem 0.9rem', background: 'rgba(16, 185, 129, 0.12)', border: '1px solid var(--accent-emerald)', borderRadius: '6px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--accent-emerald)', display: 'block', fontWeight: 700 }}>{isAr ? 'العمود الفقري المحمي' : 'Shielded Weights'}</span>
                <strong style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>EWC Fisher Matrix Shield</strong>
              </div>
              <span style={{ color: 'var(--text-muted)', fontWeight: 800 }}>+</span>
              <div style={{ padding: '0.6rem 0.9rem', background: 'rgba(20, 184, 166, 0.12)', border: '1px solid var(--accent-teal)', borderRadius: '6px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--accent-teal)', display: 'block', fontWeight: 700 }}>{isAr ? 'مصفوفات التكيف السريعة' : 'Low-Rank Adaptation'}</span>
                <strong style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>LoRA Task Adapters</strong>
              </div>
            </div>
          )}

          {curr.diagramType === 'task_heads' && (
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              {[
                { name_ar: 'رأس المشاعر', name_en: 'Sentiment Head', badge: '2 Classes' },
                { name_ar: 'رأس الموضوعات', name_en: 'Topic Head', badge: '4 Classes' },
                { name_ar: 'رأس استخراج الكيانات', name_en: 'NER Head', badge: '5 Classes' },
                { name_ar: 'رأس التوليد اللغوي', name_en: 'LM Head', badge: 'Vocab Size' },
              ].map((th, thIdx) => (
                <div key={thIdx} style={{ padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-medium)', borderRadius: '6px', textAlign: 'center', minWidth: '110px' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--text-primary)' }}>{isAr ? th.name_ar : th.name_en}</div>
                  <span className="badge badge-teal" style={{ fontSize: '0.65rem', marginTop: '0.2rem' }}>{th.badge}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
