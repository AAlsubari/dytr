// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// FlashAttention SRAM Tiling vs HBM Simulator
// Matching Academy Design Language: Comparison Grid, Bilingual, Dual Mode
// 3-Layer Academic Explanation (TermTooltip), Day/Night Adaptive Contrast
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function FlashAttentionSim() {
  const { language, connectionMode } = useStudioStore();
  const isAr = language === 'ar';

  const [seqLen, setSeqLen] = useState(2048);
  const [hiddenDim, setHiddenDim] = useState(512);
  const [numHeads, setNumHeads] = useState(8);
  const [blockSize, setBlockSize] = useState(64);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    let isMounted = true;
    api
      .getFlashAttentionSim({
        seq_len: seqLen,
        hidden_dim: hiddenDim,
        num_heads: numHeads,
        block_size: blockSize,
      })
      .then((data) => {
        if (isMounted && data) setStats(data);
      });
    return () => {
      isMounted = false;
    };
  }, [seqLen, hiddenDim, numHeads, blockSize]);

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'تجزئة الذاكرة السريعة (FlashAttention SRAM Tiling)' : 'FlashAttention: SRAM Tiling vs HBM Memory Wall'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'خوارزمية ذكية تتفادى جدار الذاكرة البطيئة HBM عبر حوسبة Softmax التراكمي بشرائح صغيرة داخل ذاكرة المعالج فائقة السرعة SRAM.'
              : 'IO-Aware tiling computing online softmax in fast on-chip SRAM, avoiding quadratic HBM memory wall.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-emerald">
            {stats?.hbm_memory_traffic_reduction_percent || 80}% {isAr ? 'وفر في نقل الذاكرة' : 'HBM IO Saved'}
          </span>
          <span className="badge badge-neutral">
            {connectionMode === 'library' ? 'dytr SDPA' : (isAr ? 'محاكاة رياضية' : 'Simulation Engine')}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
        {/* Interactive Parameter Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'طول التسلسل (Sequence Length N):' : 'Sequence Length (N tokens):'}</span>
                <TermTooltip
                  termKey="flash_attention"
                  title={isAr ? 'طول التسلسل (Sequence Length N)' : 'Sequence Length (N tokens)'}
                  explanation={
                    isAr
                      ? 'عدد الكلمات المعالجة دفعة واحدة؛ يتضاعف استهلاك الذاكرة تربيعياً O(N²) في الانتباه التقليدي، بينما يخفضه FlashAttention لتعقيد خطي O(N).'
                      : 'Total sequence length. Standard attention requires O(N²) memory bandwidth, whereas FlashAttention scales linearly with IO-aware tiling.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>N =</span>
                <input
                  type="number"
                  min="256"
                  max="4096"
                  step="256"
                  value={seqLen}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setSeqLen(Math.min(4096, Math.max(256, val)));
                  }}
                  style={{ width: '55px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Sequence Length"
              min="256"
              max="4096"
              step="256"
              value={seqLen}
              onChange={(e) => setSeqLen(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'أبعاد النموذج (Hidden Dim d_model):' : 'Model Dimension (d_model):'}</span>
                <TermTooltip
                  title={isAr ? 'أبعاد النموذج (Hidden Dim d_model)' : 'Model Dimension (d_model)'}
                  explanation={
                    isAr
                      ? 'البعد الخفي لمتجهات العمود الفقري وحجم تمثيل كل توكن في الفضاء الدلالي.'
                      : 'Hidden representation dimension across transformer layers.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>d =</span>
                <input
                  type="number"
                  min="256"
                  max="1024"
                  step="128"
                  value={hiddenDim}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setHiddenDim(Math.min(1024, Math.max(256, val)));
                  }}
                  style={{ width: '50px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Model Dimension"
              min="256"
              max="1024"
              step="128"
              value={hiddenDim}
              onChange={(e) => setHiddenDim(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'حجم شريحة الذاكرة السريعة (SRAM Block B):' : 'SRAM Block Size (Br / Bc):'}</span>
                <TermTooltip
                  title={isAr ? 'حجم شريحة الذاكرة السريعة (SRAM Block Size B)' : 'SRAM Block Size (Br / Bc)'}
                  explanation={
                    isAr
                      ? 'حجم الشريحة المحملة في ذاكرة SRAM الفائقة السرعة داخل رقاقة GPU؛ يوازن بين سعة التخزين وحجم النطاق الترددي.'
                      : 'Tile chunk size loaded into GPU on-chip SRAM cache for fast fused kernel computation.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>B =</span>
                <input
                  type="number"
                  min="32"
                  max="128"
                  step="32"
                  value={blockSize}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setBlockSize(Math.min(128, Math.max(32, val)));
                  }}
                  style={{ width: '45px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-amber)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="SRAM Tiling Block Size"
              min="32"
              max="128"
              step="32"
              value={blockSize}
              onChange={(e) => setBlockSize(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>
        </div>

        {/* Comparative Architecture Blueprint */}
        <div className="comparison-grid" style={{ margin: 0 }}>
          <div className="comparison-column bad" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-medium)', borderRadius: 'var(--radius-md)', padding: '1.25rem' }}>
            <div className="comparison-column-title" style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent-rose)', fontWeight: 700, marginBottom: '0.75rem' }}>
              <span>{isAr ? 'الاهتمام التقليدي (Naive HBM)' : 'Standard Attention (Naive)'}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>O(N²)</span>
            </div>
            <p className="branch-card-desc" style={{ marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {isAr
                ? 'يكتب ويقرأ مصفوفة الاهتمام الضخمة N×N بالكامل في ذاكرة HBM البطيئة.'
                : 'Materializes full N×N attention score matrix into high-latency HBM memory.'}
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>{isAr ? 'حجم المصفوفة:' : 'Matrix Footprint:'}</span>
              <strong style={{ color: 'var(--accent-rose)', fontFamily: 'var(--font-mono)' }}>
                {stats?.standard_attention?.matrix_footprint_mb || 32} MB
              </strong>
            </div>
          </div>

          <div className="comparison-column good" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--accent-teal)', borderRadius: 'var(--radius-md)', padding: '1.25rem' }}>
            <div className="comparison-column-title" style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent-teal)', fontWeight: 700, marginBottom: '0.75rem' }}>
              <span>{isAr ? 'تقنية FlashAttention (dytr)' : 'FlashAttention (dytr Tiled)'}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>O(N)</span>
            </div>
            <p className="branch-card-desc" style={{ marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {isAr
                ? 'حوسبة الشرائح وتجميع Softmax مباشرة في SRAM السريعة دون حفظ المصفوفة الوسيطة.'
                : 'Online tiled softmax fused in fast GPU SRAM without materializing intermediate scores.'}
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>{isAr ? 'استهلاك ذاكرة SRAM:' : 'On-Chip SRAM:'}</span>
              <strong style={{ color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                {stats?.flash_attention?.sram_footprint_kb || 48} KB
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>{isAr ? 'معامل التسريع التقريبي:' : 'Estimated Speedup:'}</span>
              <strong style={{ color: 'var(--accent-emerald)', fontFamily: 'var(--font-mono)' }}>
                ~{stats?.estimated_speedup_factor || 3.8}x {isAr ? 'أسرع' : 'faster'}
              </strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
