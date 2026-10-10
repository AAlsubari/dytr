// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// AdaLoRA SVD Matrix Decomposition & Budget Pruning Simulator
// Matching Academy Design Language: Singular Value Spectrum, Bilingual, Dual Mode
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function AdaLoRASim() {
  const { language, connectionMode } = useStudioStore();
  const isAr = language === 'ar';

  const [initRank, setInitRank] = useState(16);
  const [targetRank, setTargetRank] = useState(4);
  const [simData, setSimData] = useState(null);

  useEffect(() => {
    let isMounted = true;
    api
      .getAdaLoRASim({
        init_rank: initRank,
        target_rank: targetRank,
        d_in: 256,
        d_out: 256,
      })
      .then((data) => {
        if (isMounted && data) setSimData(data);
      });
    return () => {
      isMounted = false;
    };
  }, [initRank, targetRank]);

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'تقليص الرتبة التكيفي (AdaLoRA Adaptive SVD)' : 'AdaLoRA: Adaptive SVD Rank Parameterization'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'تفكيك مصفوفة التحديث ΔW = P·Λ·Q إلى قيم مفردة، وحذف الأبعاد غير المؤثرة لتقليل المعاملات القابلة للتدريب بأكثر من 95%.'
              : 'Dynamic singular value decomposition ΔW = P·Λ·Q pruning unimportant directions to cut trainable parameters by >95%.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-teal">
            {simData?.param_reduction_vs_dense_percent || 96.8}% {isAr ? 'توفير في المعاملات' : 'Param Reduction'}
          </span>
          <span className="badge badge-neutral">
            {connectionMode === 'library' ? 'dytr AdaLoRA' : (isAr ? 'محاكاة تفكيك المصفوفات' : 'Simulation Engine')}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
        {/* Sliders and Configuration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'الرتبة الأولية المخصصة (Initial Rank r_init):' : 'Initial Allocated Rank (r_init):'}</span>
                <TermTooltip
                  title={isAr ? 'الرتبة الأولية المخصصة (r_init)' : 'Initial Allocated Rank (r_init)'}
                  explanation={
                    isAr
                      ? 'الحد الأقصى للرتبة المعطاة لمصفوفات التكيف قبل بدء عملية تقليم المعاملات غير المؤثرة بناءً على تحليل القيم المفردة SVD.'
                      : 'Initial rank ceiling allocated to dynamic adapter matrices prior to singular value importance pruning.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>r_init =</span>
                <input
                  type="number"
                  min="8"
                  max="32"
                  step="4"
                  value={initRank}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) {
                      const clamped = Math.min(32, Math.max(8, val));
                      setInitRank(clamped);
                      if (targetRank > clamped) setTargetRank(clamped);
                    }
                  }}
                  style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Initial Allocated Rank"
              min="8"
              max="32"
              step="4"
              value={initRank}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setInitRank(val);
                if (targetRank > val) setTargetRank(val);
              }}
              className="slider-input"
            />
          </div>

          <div className="form-group">
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'الرتبة النهائية بعد التقليم (Target Rank r_final):' : 'Target Pruned Rank (r_final):'}</span>
                <TermTooltip
                  title={isAr ? 'الرتبة المستهدفة بعد التقليم (r_final)' : 'Target Pruned Rank (r_final)'}
                  explanation={
                    isAr
                      ? 'الرتبة المضغوطة المستهدفة بعد حذف المتجهات الأقل أهمية واحتجاز أبعاد التكيف ذات الطاقة الأكبر فقط.'
                      : 'Final compressed rank retaining only the most critical singular vectors with maximum energy.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>r_final =</span>
                <input
                  type="number"
                  min="1"
                  max={initRank}
                  step="1"
                  value={targetRank}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setTargetRank(Math.min(initRank, Math.max(1, val)));
                  }}
                  style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Target Pruned Rank"
              min="1"
              max={initRank}
              value={targetRank}
              onChange={(e) => setTargetRank(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>

          {/* Metric comparison */}
          <div className="comparison-grid" style={{ margin: 0 }}>
            <div className="comparison-column" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md, 8px)', padding: '0.85rem' }}>
              <div className="comparison-column-title" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {isAr ? 'معاملات الطبقة الكثيفة الأصلية:' : 'Dense Full Weight Params:'}
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                {(simData?.dense_params || 65536).toLocaleString()}
              </div>
            </div>
            <div className="comparison-column good" style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--accent-teal)', borderRadius: 'var(--radius-md, 8px)', padding: '0.85rem' }}>
              <div className="comparison-column-title" style={{ fontSize: '0.8rem', color: 'var(--accent-teal)' }}>
                {isAr ? 'معاملات AdaLoRA القابلة للتدريب:' : 'AdaLoRA Trainable Params:'}
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', marginTop: '0.25rem' }}>
                {(simData?.pruned_lora_params || 2052).toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        {/* Visual Rank SVD Spectrum */}
        <div style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md, 8px)', padding: '1.25rem', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span>{isAr ? 'طيف أهمية المتجهات المفردة (Singular Values Spectrum Λ):' : 'Singular Values Importance Spectrum (Λ):'}</span>
            <TermTooltip
              title={isAr ? 'طيف القيم المفردة (Singular Spectrum Λ)' : 'Singular Values Spectrum (Λ)'}
              explanation={
                isAr
                  ? 'تفكيك مصفوفة التكيف ΔW = P·Λ·Q إلى قيم مفردة قطرية؛ تُمثل كل قيمة درجة الأهمية النسبية لاتجاه التحديث العصبي.'
                  : 'Diagonal singular values in SVD decomposition representing the relative energy and importance of each adaptation direction.'
              }
              size="sm"
              hideLabel={true}
            />
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
            {isAr
              ? 'الأعمدة باللون الأزرق محتفظ بها كأبعاد حيوية، والأعمدة الشفافة يتم تقليمها وتصفيرها.'
              : 'Highlighted bars represent retained singular directions; dim bars are pruned from gradient computation.'}
          </p>

          <div style={{ display: 'flex', gap: '5px', height: '110px', alignItems: 'flex-end', padding: '0.5rem 0', background: 'var(--bg-elevated)', borderRadius: '6px', border: '1px solid var(--border-subtle)', paddingInline: '0.5rem' }}>
            {simData?.ranks?.map((r) => {
              const isRetained = r.status === 'retained';
              const heightPercent = Math.max(14, r.importance_score * 100);
              return (
                <div
                  key={r.rank_index}
                  style={{
                    flex: 1,
                    height: `${heightPercent}%`,
                    background: isRetained ? 'var(--accent-teal)' : 'var(--border-medium)',
                    opacity: isRetained ? 1 : 0.45,
                    borderRadius: '3px 3px 0 0',
                    transition: 'all 0.3s ease',
                  }}
                  title={`Rank ${r.rank_index}: Score=${r.importance_score} (${r.status})`}
                />
              );
            })}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontFamily: 'var(--font-mono)' }}>
            <span>{isAr ? 'رتبة 1 (أعلى أهمية)' : 'Rank 1 (Highest)'}</span>
            <span>{isAr ? `رتبة ${initRank} (مقلّمة)` : `Rank ${initRank} (Pruned)`}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
