// ==============================================================================
// DyTrStudio Multi-Task LoRA Parameter-Efficient Adaptation Simulator
// Matching Academy Design Language: Low-Rank Decomposition, Bilingual RTL/LTR
// Zero Emojis Policy - DyTR MultiTaskLoRAWrapper Simulation
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function MultiTaskLoRASim() {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const [rank, setRank] = useState(8);
  const [alpha, setAlpha] = useState(16.0);
  const [selectedTaskIdx, setSelectedTaskIdx] = useState(0);
  const [simData, setSimData] = useState(null);

  useEffect(() => {
    let isMounted = true;
    api
      .getAdaLoRASim({
        rank,
        alpha,
        d_in: 256,
        d_out: 256,
        num_tasks: 2,
      })
      .then((data) => {
        if (isMounted && data) setSimData(data);
      })
      .catch(() => {
        // Fallback calculations matching dytr.training.lora
        if (isMounted) {
          const d_in = 256;
          const d_out = 256;
          const dense = d_in * d_out;
          const loraPerTask = rank * (d_in + d_out);
          setSimData({
            rank,
            alpha,
            scaling: Number((alpha / rank).toFixed(2)),
            d_in,
            d_out,
            dense_params: dense,
            lora_params_per_task: loraPerTask,
            param_reduction_vs_dense_percent: Number(((1.0 - (loraPerTask / dense)) * 100).toFixed(2)),
            task_adapters: [
              {
                task_id: 1,
                task_name: 'task_sentiment',
                matrix_a_shape: `(${rank}, ${d_in})`,
                matrix_b_shape: `(${d_out}, ${rank})`,
                params_count: loraPerTask,
                scaling_factor: Number((alpha / rank).toFixed(2)),
              },
              {
                task_id: 2,
                task_name: 'task_topic',
                matrix_a_shape: `(${rank}, ${d_in})`,
                matrix_b_shape: `(${d_out}, ${rank})`,
                params_count: loraPerTask,
                scaling_factor: Number((alpha / rank).toFixed(2)),
              },
            ],
          });
        }
      });

    return () => {
      isMounted = false;
    };
  }, [rank, alpha]);

  const denseParams = simData?.dense_params || 65536;
  const loraParams = simData?.lora_params_per_task || rank * 512;
  const reductionPercent = simData?.param_reduction_vs_dense_percent || 93.75;
  const scaling = (alpha / rank).toFixed(2);

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'أغلفة التكيف منخفضة الرتبة (Multi-Task LoRA Adapters)' : 'Parameter-Efficient Multi-Task LoRA Adapters'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'تجميد أوزان النموذج الأساسي W_0 وتدريب مصفوفات منخفضة الرتبة (A و B) لكل مهمة؛ مما يقلل المعاملات المدربة بأكثر من 93% ويمنع التداخل.'
              : 'Freezes base backbone weights W_0 and trains dedicated low-rank matrices (A and B) per task, cutting trainable parameters by >93%.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-emerald">
            {reductionPercent}% {isAr ? 'توفير في المعاملات' : 'Param Reduction'}
          </span>
          <span className="badge badge-teal">
            dytr MultiTaskLoRAWrapper
          </span>
        </div>
      </div>

      {/* Interactive Controls & Diagram */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Sliders */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Rank Slider */}
          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'رتبة التكيف (Rank r):' : 'LoRA Rank (r):'}</span>
                <TermTooltip
                  termKey="lora"
                  title={isAr ? 'رتبة التكيف المنخفضة (Rank r)' : 'LoRA Rank (r)'}
                  explanation={
                    isAr
                      ? 'البعد الداخلي لمصفوفات التكيف (A و B)؛ تُمثل الرتبة المنخفضة اختزالاً ذكياً لتحديثات الأوزان، مما يقلل الذاكرة والمعاملات بأكثر من 98%.'
                      : 'The bottleneck rank of low-rank matrices A and B, constraining weight updates to an intrinsic manifold to slash trainable parameters.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>r =</span>
                <input
                  type="number"
                  min="2"
                  max="32"
                  step="2"
                  value={rank}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setRank(Math.min(32, Math.max(2, val)));
                  }}
                  style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              min="2"
              max="32"
              step="2"
              value={rank}
              onChange={(e) => setRank(parseInt(e.target.value))}
              className="slider-input"
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
              {isAr ? 'رتبة أصغر تعني توفيراً أعلى في الذاكرة والحوسبة' : 'Lower rank drastically cuts trainable parameter memory'}
            </div>
          </div>

          {/* Alpha Slider */}
          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'معامل التكبير (LoRA Alpha α):' : 'Scaling Factor Alpha (α):'}</span>
                <TermTooltip
                  title={isAr ? 'معامل التكبير (LoRA Alpha α)' : 'LoRA Scaling Alpha (α)'}
                  explanation={
                    isAr
                      ? 'معامل قياس حجم التحديث العصبي؛ النسبة (α / r) تُحافظ على ثبات مقياس التدرجات حتى عند تغيير قيمة الرتبة r أثناء التجارب.'
                      : 'Scaling factor normalizing the magnitude of adaptation updates (alpha / r) to preserve stable optimization dynamics across varying ranks.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>α =</span>
                <input
                  type="number"
                  min="4"
                  max="64"
                  step="4"
                  value={alpha}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) setAlpha(Math.min(64, Math.max(4, val)));
                  }}
                  style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              min="4"
              max="64"
              step="4"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="slider-input"
            />
            <div style={{ fontSize: '0.75rem', color: 'var(--accent-teal)', marginTop: '0.3rem' }}>
              {isAr ? `معامل القياس الفعلي: α / r = ${scaling}` : `Effective adapter multiplier: α / r = ${scaling}`}
            </div>
          </div>

          {/* Active Task Adapter Switcher */}
          <div className="form-group">
            <label className="form-label" style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)' }}>
              <span>{isAr ? 'المهمة المستهدفة (Task Adapter):' : 'Active Multi-Task Adapter:'}</span>
              <TermTooltip
                title={isAr ? 'أغلفة التكيف متعدد المهام (Multi-Task LoRA)' : 'Multi-Task LoRA Wrappers'}
                explanation={
                  isAr
                    ? 'في dytr MultiTaskLoRAWrapper، يمتلك كل رأس مهمة غلاف LoRA مخصصاً له، ويتبادل النموذج هذه الأغلفة تلقائياً بحسب نوع الدفعة.'
                    : 'Each task head is bound to its own low-rank adapter, dynamically swapped during multi-task batch collation.'
                }
                size="sm"
                hideLabel={true}
              />
            </label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                type="button"
                onClick={() => setSelectedTaskIdx(0)}
                className={`btn btn-sm ${selectedTaskIdx === 0 ? 'btn-primary' : 'btn-secondary'}`}
                style={{ flex: 1 }}
              >
                {isAr ? 'مهمة المشاعر (A1, B1)' : 'Task 1: Sentiment (A1, B1)'}
              </button>
              <button
                type="button"
                onClick={() => setSelectedTaskIdx(1)}
                className={`btn btn-sm ${selectedTaskIdx === 1 ? 'btn-primary' : 'btn-secondary'}`}
                style={{ flex: 1 }}
              >
                {isAr ? 'مهمة الموضوعات (A2, B2)' : 'Task 2: Topic (A2, B2)'}
              </button>
            </div>
          </div>
        </div>

        {/* Mathematical Tensor Decomposition Card */}
        <div
          style={{
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-md)',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-teal)', marginBottom: '0.5rem' }}>
              {isAr ? 'المعادلة الرياضية لتفكيك الأوزان في dytr:' : 'Mathematical Tensor Decomposition in dytr:'}
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                background: 'var(--bg-surface)',
                padding: '0.75rem',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                border: '1px solid var(--border-subtle)',
                marginBottom: '1rem',
                color: 'var(--text-primary)',
                direction: 'ltr',
                textAlign: 'center',
              }}
            >
              h = x · W_0^T + (α / r) · (x · A_t^T) · B_t^T
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{isAr ? 'العمود الفقري الأساسي W_0 (مجمد):' : 'Base Dense Layer W_0 (Frozen):'}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>256 × 256 = {denseParams.toLocaleString()} {isAr ? 'معامل' : 'params'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--accent-teal)' }}>{isAr ? `مصفوفة A_${selectedTaskIdx + 1} (Kaiming):` : `Adapter Matrix A_${selectedTaskIdx + 1}:`}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{rank} × 256 = {(rank * 256).toLocaleString()}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.35rem' }}>
                <span style={{ color: 'var(--accent-amber)' }}>{isAr ? `مصفوفة B_${selectedTaskIdx + 1} (أصفار):` : `Adapter Matrix B_${selectedTaskIdx + 1}:`}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>256 × {rank} = {(256 * rank).toLocaleString()}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: '0.2rem' }}>
                <span style={{ fontWeight: 700, color: 'var(--accent-emerald)' }}>{isAr ? 'إجمالي المعاملات المدربة للمهمة:' : 'Trainable Parameters per Task:'}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                  {loraParams.toLocaleString()} ({isAr ? 'فقط' : 'only'})
                </span>
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            {isAr
              ? 'يتم تجميد W_0 بالكامل؛ ولذا لا يمكن لأي تدريب جديد أن يمحو ميزات التمثيل اللغوي الأساسي.'
              : 'W_0 remains strictly frozen, mathematically immunizing shared core representations from drift.'}
          </div>
        </div>
      </div>
    </div>
  );
}
