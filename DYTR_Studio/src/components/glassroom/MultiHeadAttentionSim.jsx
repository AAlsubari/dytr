// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// MultiHeadAttentionSim.jsx - Multi-Head Attention & Scaled Dot-Product Simulator
// Faithfully matches dytr.core.layers.MultiHeadAttention
// Matching Academy Design Language: SectionCards, Bilingual RTL/LTR, Zero Emojis
// 3-Layer Academic Explanation (TermTooltip), Dual Day/Night Contrast
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';
import TermTooltip from '../common/TermTooltip';

export default function MultiHeadAttentionSim() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const [activeHead, setActiveHead] = useState(1);
  const [numHeads, setNumHeads] = useState(4);
  const [embedDim, setEmbedDim] = useState(256);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState(0);

  const tokens = isAr
    ? ['المحولات', 'الديناميكية', 'تتعلم', 'باستمرار', 'دون', 'نسيان']
    : ['Dynamic', 'transformers', 'adapt', 'continually', 'without', 'forgetting'];

  const headDim = embedDim / numHeads;
  const scaleFactor = (1 / Math.sqrt(headDim)).toFixed(4);

  // Synthesize realistic attention weight matrix based on selected head and tokens
  // Head 1: Syntactic adjacency; Head 2: Semantic focus on "transformers/adapt"; Head 3: Global diffuse; Head 4: Positional recency
  const getAttentionWeights = (head) => {
    const N = tokens.length;
    const matrix = [];
    for (let i = 0; i < N; i++) {
      const row = [];
      let sum = 0;
      for (let j = 0; j < N; j++) {
        let score = 0;
        if (head === 1) {
          // Adjacency / local syntax
          score = Math.exp(-Math.abs(i - j) * 1.4);
        } else if (head === 2) {
          // Subject-Verb attention (tokens 0 & 1 & 2)
          score = (i <= 2 && j <= 2) ? 2.5 : Math.exp(-Math.abs(i - j) * 0.8);
        } else if (head === 3) {
          // Global broad attention
          score = 1.0 + Math.sin(i * j + head) * 0.4;
        } else {
          // Backward contextual anchoring
          score = j <= i ? 1.8 / (i - j + 1) : 0.2;
        }
        row.push(score);
        sum += score;
      }
      matrix.push(row.map((val) => Number((val / sum).toFixed(3))));
    }
    return matrix;
  };

  const currentMatrix = getAttentionWeights(activeHead);
  const activeTokenWeights = currentMatrix[selectedTokenIdx] || currentMatrix[0];

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'الانتباه متعدد الرؤوس وحوسبة درجات التوافق (Multi-Head Attention)' : 'Multi-Head Attention & Scaled Dot-Product Engine'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'القلب النابض لمكتبة dytr: تفكيك متجهات الاستعلام Q، والمفاتيح K، والقيم V، وحساب مصفوفة الانتباه المقاسة.'
              : 'Core dytr layer: projects input tensors into Query, Key, and Value spaces with scaled dot-product routing.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-teal">
            dytr.core.layers.MultiHeadAttention
          </span>
        </div>
      </div>

      {/* Control Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Parameters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="form-group" style={{ margin: 0 }}>
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'عدد رؤوس الانتباه (num_heads):' : 'Attention Heads (h):'}</span>
                <TermTooltip
                  termKey="attention"
                  title={isAr ? 'عدد رؤوس الانتباه (num_heads)' : 'Attention Heads (h)'}
                  explanation={
                    isAr
                      ? 'عدد الرؤوس المتوازية؛ يُمكّن النموذج من التركيز في فضاءات إسقاط متعددة في آن واحد (أنماط نحوية، دلالية، ومكانية).'
                      : 'Parallel linear projection subspaces allowing the model to jointly attend to information from different representation positions.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <span className="badge badge-teal" style={{ fontFamily: 'var(--font-mono)' }}>
                h = {numHeads}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {[2, 4, 8].map((h) => (
                <button
                  key={h}
                  onClick={() => {
                    setNumHeads(h);
                    if (activeHead > h) setActiveHead(1);
                  }}
                  className={`btn btn-sm ${numHeads === h ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1 }}
                >
                  {h} {isAr ? 'رؤوس' : 'Heads'}
                </button>
              ))}
            </div>
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'أبعاد التضمين (embed_dim):' : 'Embedding Dimension (d_model):'}</span>
                <TermTooltip
                  title={isAr ? 'أبعاد التضمين (embed_dim)' : 'Embedding Dimension (d_model)'}
                  explanation={
                    isAr
                      ? 'البعد الكلي لمتجه الكلمة، ويقسم بالتساوي على عدد الرؤوس (d_k = d_model / h).'
                      : 'Total hidden representation dimension, split evenly across heads to maintain constant compute complexity.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <span className="badge badge-neutral" style={{ fontFamily: 'var(--font-mono)' }}>
                {embedDim}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {[128, 256, 512].map((dim) => (
                <button
                  key={dim}
                  onClick={() => setEmbedDim(dim)}
                  className={`btn btn-sm ${embedDim === dim ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1 }}
                >
                  {dim}
                </button>
              ))}
            </div>
          </div>

          {/* Mathematical Projections Info Box */}
          <div style={{ padding: '1rem', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem' }}>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              {isAr ? 'المعادلة الرياضية المطبقة في dytr' : 'Mathematical Formulation in dytr'}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', marginBottom: '0.5rem', direction: 'ltr', textAlign: 'left' }}>
              Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', color: 'var(--text-secondary)' }}>
              <div>{isAr ? 'بعد الرأس الواحد (head_dim):' : 'Head Dimension (d_k):'} <strong style={{ color: 'var(--text-primary)' }}>{headDim}</strong></div>
              <div>{isAr ? 'معامل التدريج (1/√d_k):' : 'Scale Factor:'} <strong style={{ color: 'var(--text-primary)' }}>{scaleFactor}</strong></div>
            </div>
          </div>
        </div>

        {/* Head Selector & Interactive Sentence Tokens */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span>{isAr ? 'اختر الرأس المراد فحصه:' : 'Active Inspected Head:'}</span>
              <TermTooltip
                title={isAr ? 'الرأس المفحوص (Active Inspected Head)' : 'Active Inspected Head'}
                explanation={
                  isAr
                    ? 'لكل رأس مصفوفة أوزان خاصة به (W_Q, W_K, W_V)، فيتعلم كل رأس التركيز على علاقات لغوية مستقلة.'
                    : 'Each head projects embeddings with independent weight matrices, learning complementary syntactic and semantic relationships.'
                }
                size="sm"
                hideLabel={true}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {Array.from({ length: numHeads }, (_, i) => i + 1).map((h) => (
                <button
                  key={h}
                  onClick={() => setActiveHead(h)}
                  className={`btn btn-sm ${activeHead === h ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1 }}
                >
                  {isAr ? `رأس ${h}` : `Head ${h}`}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
              {isAr ? 'انقر على كلمة لمشاهدة توزيع اهتمامها:' : 'Click token to inspect its attention distribution:'}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {tokens.map((tok, idx) => (
                <button
                  key={idx}
                  onClick={() => setSelectedTokenIdx(idx)}
                  style={{
                    padding: '0.4rem 0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    border: selectedTokenIdx === idx ? '2px solid var(--accent-teal)' : '1px solid var(--border-medium)',
                    background: selectedTokenIdx === idx ? 'var(--accent-teal-subtle)' : 'var(--bg-tertiary)',
                    color: selectedTokenIdx === idx ? 'var(--accent-teal)' : 'var(--text-primary)',
                    fontWeight: selectedTokenIdx === idx ? 800 : 500,
                    fontSize: '0.85rem',
                    cursor: 'pointer',
                    transition: 'var(--transition)',
                  }}
                >
                  {tok}
                </button>
              ))}
            </div>
          </div>

          {/* Attention distribution bar for selected token */}
          <div style={{ padding: '0.75rem', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
              {isAr
                ? `درجات اهتمام [${tokens[selectedTokenIdx]}] بالكلمات الأخرى (الرأس ${activeHead}):`
                : `Attention from [${tokens[selectedTokenIdx]}] across sequence (Head ${activeHead}):`}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {tokens.map((tok, j) => {
                const weight = activeTokenWeights[j] || 0;
                return (
                  <div key={j} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem' }}>
                    <span style={{ width: '85px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)' }}>
                      {tok}
                    </span>
                    <div style={{ flex: 1, height: '8px', background: 'var(--bg-elevated)', borderRadius: '4px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${weight * 100}%`,
                          background: j === selectedTokenIdx ? 'var(--accent-teal)' : 'var(--accent-emerald)',
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                    <span style={{ width: '40px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', textAlign: 'right', color: 'var(--text-secondary)' }}>
                      {(weight * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* 2D Attention Heatmap Grid */}
      <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.25rem' }}>
        <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>{isAr ? `مصفوفة الانتباه الكاملة N × N (الرأس ${activeHead})` : `Full Attention Weight Heatmap N × N (Head ${activeHead})`}</span>
          <TermTooltip
            termKey="attention"
            title={isAr ? 'مصفوفة الانتباه المقاسة (Attention Matrix N × N)' : 'Scaled Dot-Product Attention Matrix'}
            explanation={
              isAr
                ? 'مصفوفة ناتجة عن ضرب متجهات الاستعلام بالمفاتيح (Q · Kᵀ) مقسومة على جذر البعد √d_k ومطبقة عليها دالة Softmax، بحيث يكون مجموع كل صف 1.0.'
                : 'Computed via softmax(Q·Kᵀ / √d_k), quantifying pairwise alignment weights between every query and key token in the sequence.'
            }
            size="sm"
            hideLabel={true}
          />
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'center' }}>
            <thead>
              <tr>
                <th style={{ padding: '0.4rem', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  Q \ K
                </th>
                {tokens.map((t, idx) => (
                  <th key={idx} style={{ padding: '0.4rem', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-primary)', fontWeight: 700 }}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tokens.map((qTok, rIdx) => (
                <tr key={rIdx} style={{ background: rIdx === selectedTokenIdx ? 'var(--accent-teal-subtle)' : 'transparent' }}>
                  <td style={{ padding: '0.4rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: isAr ? 'right' : 'left' }}>
                    {qTok}
                  </td>
                  {tokens.map((_, cIdx) => {
                    const val = currentMatrix[rIdx][cIdx];
                    const bgAlpha = Math.min(0.85, Math.max(0.08, val * 1.5));
                    return (
                      <td
                        key={cIdx}
                        style={{
                          padding: '0.4rem',
                          background: `rgba(14, 165, 233, ${bgAlpha})`,
                          color: bgAlpha > 0.42 ? '#FFFFFF' : 'var(--text-primary)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.75rem',
                          fontWeight: val > 0.3 ? 800 : 500,
                          border: '1px solid var(--border-subtle)',
                        }}
                      >
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
