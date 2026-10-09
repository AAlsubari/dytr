// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Screen 3: Neural Architecture Studio (استوديو المعمارية)
// Industrial Swiss Design, Zero Emojis, Hyperparameter Sliders, Live Memory Telemetry

import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../utils/api';

export default function ModelBuilderView() {
  const { lang } = useApp();
  const isAr = lang === 'ar';

  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState(null);

  // Model Hyperparameters State
  const [vocabSize, setVocabSize] = useState(30522);
  const [embedDim, setEmbedDim] = useState(256);
  const [numLayers, setNumLayers] = useState(4);
  const [numHeads, setNumHeads] = useState(4);
  const [ffMult, setFfMult] = useState(4);
  const [maxSeqLen, setMaxSeqLen] = useState(256);
  const [dropout, setDropout] = useState(0.1);

  // Hardware Switches
  const [useFlashAttn, setUseFlashAttn] = useState(true);
  const [useRoPE, setUseRoPE] = useState(true);
  const [useAMP, setUseAMP] = useState(true);
  const [useCheckpointing, setUseCheckpointing] = useState(false);

  // Memory & PEFT (EWC & LoRA)
  const [useEWC, setUseEWC] = useState(true);
  const [ewcLambda, setEwcLambda] = useState(5.0);
  const [useLoRA, setUseLoRA] = useState(false);
  const [loraRank, setLoraRank] = useState(8);

  // Estimation & Code Output
  const [estimation, setEstimation] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);

  useEffect(() => {
    api.getModelPresets().then((res) => {
      if (res && res.length > 0) {
        setPresets(res);
        setSelectedPreset(res[1]?.id || res[0]?.id);
      }
    });
  }, []);

  const handleApplyPreset = (preset) => {
    setSelectedPreset(preset.id);
    const cfg = preset.config;
    if (cfg.embed_dim) setEmbedDim(cfg.embed_dim);
    if (cfg.num_layers) setNumLayers(cfg.num_layers);
    if (cfg.num_heads) setNumHeads(cfg.num_heads);
    if (cfg.max_seq_len) setMaxSeqLen(cfg.max_seq_len);
    if (cfg.use_flash_attention !== undefined) setUseFlashAttn(cfg.use_flash_attention);
    if (cfg.use_rotary_embedding !== undefined) setUseRoPE(cfg.use_rotary_embedding);
    if (cfg.use_ewc !== undefined) setUseEWC(cfg.use_ewc);
    else if (cfg.use_si !== undefined) setUseEWC(cfg.use_si);
    if (cfg.use_lora !== undefined) setUseLoRA(cfg.use_lora);
    else if (cfg.use_adalora !== undefined) setUseLoRA(cfg.use_adalora);
  };

  useEffect(() => {
    const payload = {
      vocab_size: vocabSize,
      embed_dim: embedDim,
      num_layers: numLayers,
      num_heads: numHeads,
      ff_mult: ffMult,
      max_seq_len: maxSeqLen,
      dropout: dropout,
      use_flash_attention: useFlashAttn,
      use_rotary_embedding: useRoPE,
      gradient_checkpointing: useCheckpointing,
      use_amp: useAMP,
      use_ewc: useEWC,
      ewc_lambda: ewcLambda * 1000,
      use_lora: useLoRA,
      lora_rank: loraRank,
      tasks: [
        { task_name: 'task_sentiment', task_type: 'sentence_classification', num_classes: 2, max_len: 64 },
        { task_name: 'task_topic', task_type: 'sentence_classification', num_classes: 4, max_len: 64 },
        { task_name: 'task_ner', task_type: 'token_classification', num_classes: 5, max_len: 64 },
      ],
    };

    api.estimateModel(payload).then((res) => {
      if (res) setEstimation(res);
    });
  }, [vocabSize, embedDim, numLayers, numHeads, ffMult, maxSeqLen, dropout, useFlashAttn, useRoPE, useAMP, useCheckpointing, useEWC, ewcLambda, useLoRA, loraRank]);

  const handleCopyCode = () => {
    if (estimation?.python_code) {
      navigator.clipboard.writeText(estimation.python_code);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  return (
    <section>
      {/* Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
          {isAr ? 'استوديو المعمارية العصبية وهندسة المعلمات' : 'Neural Architecture Studio & Parametric Tuning'}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {isAr
            ? 'مختبر تحكم دقيق لضبط أبعاد المحول الديناميكي (Dynamic Transformer)، حساب حجم الـ VRAM، وتوليد كود التدريب.'
            : 'Granular laboratory for parametric configuration, VRAM estimation, and automated DyTR script generation.'}
        </p>
      </div>

      {/* Preset Archetypes Grid */}
      <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
        {presets.map((p) => {
          const isSelected = selectedPreset === p.id;
          return (
            <div
              key={p.id}
              onClick={() => handleApplyPreset(p)}
              className="glass-card"
              style={{
                cursor: 'pointer',
                borderColor: isSelected ? 'var(--accent-steel)' : 'var(--border-subtle)',
                background: isSelected ? 'var(--accent-steel-bg)' : 'var(--bg-card)',
                padding: '1rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span className="badge badge-steel" style={{ fontSize: '0.72rem' }}>{p.badge}</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{p.category}</span>
              </div>
              <h4 style={{ fontSize: '0.95rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '2px' }}>{p.name}</h4>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{p.description}</p>
            </div>
          );
        })}
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Left Column: Sliders & Switches */}
        <div className="glass-panel" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)' }}>
            {isAr ? 'أبعاد المعمارية العصبية (Hyperparameters)' : 'Architecture Dimensions'}
          </h3>

          {/* Embedding Dimension */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span>{isAr ? 'أبعاد التضمين (d_model):' : 'Embedding Dimension (d_model):'}</span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <input
                  type="number"
                  min="64"
                  max="1024"
                  step="64"
                  value={embedDim}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setEmbedDim(Math.min(1024, Math.max(64, val)));
                  }}
                  style={{ width: '55px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-steel)', padding: 0 }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>dim</span>
              </div>
            </div>
            <input className="slider-input" style={{ width: '100%' }} type="range" min="64" max="1024" step="64" value={embedDim} onChange={(e) => setEmbedDim(parseInt(e.target.value))} />
          </div>

          {/* Transformer Layers */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span>{isAr ? 'عدد طبقات المحول (Layers):' : 'Transformer Layers:'}</span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <input
                  type="number"
                  min="1"
                  max="12"
                  step="1"
                  value={numLayers}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setNumLayers(Math.min(12, Math.max(1, val)));
                  }}
                  style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-steel)', padding: 0 }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'طبقات' : 'layers'}</span>
              </div>
            </div>
            <input className="slider-input" style={{ width: '100%' }} type="range" min="1" max="12" value={numLayers} onChange={(e) => setNumLayers(parseInt(e.target.value))} />
          </div>

          {/* Attention Heads */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span>{isAr ? 'رؤوس الانتباه (Attention Heads):' : 'Attention Heads:'}</span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <input
                  type="number"
                  min="1"
                  max="16"
                  step="1"
                  value={numHeads}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setNumHeads(Math.min(16, Math.max(1, val)));
                  }}
                  style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-steel)', padding: 0 }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>heads ({Math.floor(embedDim / (numHeads || 1))} d_h)</span>
              </div>
            </div>
            <input className="slider-input" style={{ width: '100%' }} type="range" min="1" max="16" value={numHeads} onChange={(e) => setNumHeads(parseInt(e.target.value))} />
          </div>

          {/* Feed-Forward Multiplier */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span>{isAr ? 'مضاعف الشبكة الأمامية (ff_mult):' : 'Feed-Forward Multiplier:'}</span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <input
                  type="number"
                  min="2"
                  max="8"
                  step="1"
                  value={ffMult}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setFfMult(Math.min(8, Math.max(2, val)));
                  }}
                  style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-steel)', padding: 0 }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>x ({embedDim * ffMult} dim)</span>
              </div>
            </div>
            <input className="slider-input" style={{ width: '100%' }} type="range" min="2" max="8" value={ffMult} onChange={(e) => setFfMult(parseInt(e.target.value))} />
          </div>

          {/* Max Sequence Length */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', fontSize: '0.88rem' }}>
              <span>{isAr ? 'الحد الأقصى للسياق (Max Context):' : 'Max Context Window:'}</span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <input
                  type="number"
                  min="64"
                  max="2048"
                  step="64"
                  value={maxSeqLen}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setMaxSeqLen(Math.min(2048, Math.max(64, val)));
                  }}
                  style={{ width: '55px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-steel)', padding: 0 }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>tokens</span>
              </div>
            </div>
            <input className="slider-input" style={{ width: '100%' }} type="range" min="64" max="2048" step="64" value={maxSeqLen} onChange={(e) => setMaxSeqLen(parseInt(e.target.value))} />
          </div>

          <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)', marginTop: '0.5rem' }}>
            {isAr ? 'تقنيات الذاكرة والتسريع الفيزيائي' : 'Hardware & Continual Learning Toggles'}
          </h3>

          <div className="toggle-container">
            <div>
              <div style={{ fontWeight: '600', fontSize: '0.86rem' }}>FlashAttention (SDPA Tiling)</div>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                {isAr ? 'تسريع الانتباه عبر SRAM المباشر وتوفير الذاكرة O(N)' : 'IO-aware memory tiling within high-speed SRAM'}
              </p>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={useFlashAttn} onChange={(e) => setUseFlashAttn(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="toggle-container">
            <div>
              <div style={{ fontWeight: '600', fontSize: '0.86rem' }}>Rotary Embeddings (RoPE)</div>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                {isAr ? 'تدوير المتجهات دون إضافة مصفوفة مواقع ثابتة' : 'Vector rotation preserving exact euclidean norm'}
              </p>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={useRoPE} onChange={(e) => setUseRoPE(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="toggle-container">
            <div>
              <div style={{ fontWeight: '600', fontSize: '0.86rem' }}>Elastic Weight Consolidation (EWC Shield)</div>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                {isAr ? 'درع مصفوفة فيشر لمكافحة النسيان الكارثي' : 'Fisher information matrix quadratic regularization'}
              </p>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={useEWC} onChange={(e) => setUseEWC(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {useEWC && (
            <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', marginBottom: '6px' }}>
                <span>{isAr ? 'معامل قوة درع فيشر (EWC λ):' : 'EWC Fisher Lambda (λ):'}</span>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                  <input
                    type="number"
                    min="0.1"
                    max="10.0"
                    step="0.1"
                    value={ewcLambda}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val)) setEwcLambda(Math.min(10.0, Math.max(0.1, +(val.toFixed(2)))));
                    }}
                    style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-teal)', padding: 0 }}
                  />
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>x10³</span>
                </div>
              </div>
              <input className="slider-input" style={{ width: '100%' }} type="range" min="0.1" max="10.0" step="0.1" value={ewcLambda} onChange={(e) => setEwcLambda(parseFloat(e.target.value))} />
            </div>
          )}

          <div className="toggle-container">
            <div>
              <div style={{ fontWeight: '600', fontSize: '0.86rem' }}>Multi-Task LoRA Adaptation</div>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                {isAr ? 'أغلفة تكيف منخفضة الرتبة لكل مهمة مع تجميد العمود الفقري' : 'Low-rank adaptation modules for parameter-efficient training'}
              </p>
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={useLoRA} onChange={(e) => setUseLoRA(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {useLoRA && (
            <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', marginBottom: '6px' }}>
                <span>{isAr ? 'رتبة LoRA المستهدفة (r):' : 'LoRA Target Rank (r):'}</span>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                  <input
                    type="number"
                    min="2"
                    max="64"
                    step="2"
                    value={loraRank}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (!isNaN(val)) setLoraRank(Math.min(64, Math.max(2, val)));
                    }}
                    style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: '700', color: 'var(--accent-teal)', padding: 0 }}
                  />
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>rank</span>
                </div>
              </div>
              <input className="slider-input" style={{ width: '100%' }} type="range" min="2" max="64" step="2" value={loraRank} onChange={(e) => setLoraRank(parseInt(e.target.value))} />
            </div>
          )}
        </div>

        {/* Right Column: Telemetry Footprint & Python Code Generator */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Hardware Readout Gauge Cards */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: '700', marginBottom: '0.85rem', color: 'var(--text-primary)' }}>
              {isAr ? 'المقاييس الحسابية وحجم البصمة الذاكرية' : 'Model Footprint & Memory Telemetry'}
            </h3>
            <div className="grid-2" style={{ gap: '0.85rem' }}>
              <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'إجمالي المعلمات (Parameters)' : 'Total Model Parameters'}
                </span>
                <div className="mono-font" style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-steel)', marginTop: '4px' }}>
                  {estimation?.total_parameters_formatted || '0'}
                </div>
              </div>

              <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  {isAr ? 'استهلاك الـ VRAM المتوقع (FP16)' : 'VRAM Consumption (FP16 AMP)'}
                </span>
                <div className="mono-font" style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--accent-pine)', marginTop: '4px' }}>
                  {estimation?.memory_footprint?.fp16_amp_mb || 0} MB
                </div>
              </div>
            </div>
          </div>

          {/* Generated Python Code Script */}
          <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                {isAr ? 'كود تهيئة بايثون الجاهز (DyTR Script)' : 'Generated DyTR Python Code'}
              </h3>
              <button onClick={handleCopyCode} className="btn-primary" style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}>
                {copySuccess ? (isAr ? 'تم النسخ!' : 'Copied!') : (isAr ? 'نسخ كود البايثون' : 'Copy Python Code')}
              </button>
            </div>
            <pre
              className="mono-font"
              style={{
                flex: 1,
                background: 'var(--bg-tertiary)',
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid var(--border-medium)',
                color: 'var(--text-primary)',
                fontSize: '0.8rem',
                lineHeight: '1.5',
                overflowX: 'auto',
                maxHeight: '380px',
              }}
            >
              <code>{estimation?.python_code || '# Generating code from parameters...'}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
