// ==============================================================================
// DyTrStudio Architecture Summary Card Component
// Zero Emojis Policy - Bilingual RTL/LTR
// Integrated with 3-Layer Epistemological TermTooltip System
// ==============================================================================

import React from 'react';
import TermTooltip from '../common/TermTooltip';

export default function ArchitectureSummaryCard({
  modelConfig,
  modelEstimate,
  language,
}) {
  const isAr = language === 'ar';

  return (
    <div className="grid-4" style={{ marginBottom: '2rem' }}>
      <div className="stat-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="stat-label">
            {isAr ? 'إجمالي المعلمات' : 'Total Parameters'}
          </div>
          <TermTooltip termKey="total_parameters" size="sm" hideLabel={true} />
        </div>
        <div className="stat-value" style={{ color: 'var(--accent-teal)' }}>
          {modelEstimate.total_parameters_formatted || '12.5M'}
        </div>
        <div className="stat-desc">
          {isAr ? 'أوزان قابلة للتدريب' : 'Trainable weights'}
        </div>
      </div>

      <div className="stat-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="stat-label">
            {isAr ? 'معلمات التضمين' : 'Embedding Weights'}
          </div>
          <TermTooltip termKey="embedding_params" size="sm" hideLabel={true} />
        </div>
        <div className="stat-value">
          {modelEstimate.embedding_params
            ? `${(modelEstimate.embedding_params / 1e6).toFixed(2)}M`
            : '7.81M'}
        </div>
        <div className="stat-desc">
          {isAr ? `حجم القاموس (${modelConfig.vocab_size})` : `Vocab Size (${modelConfig.vocab_size})`}
        </div>
      </div>

      <div className="stat-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="stat-label">
            {isAr ? 'استهلاك الذاكرة' : 'Memory Footprint'}
          </div>
          <TermTooltip termKey="memory_footprint" size="sm" hideLabel={true} />
        </div>
        <div className="stat-value" style={{ color: 'var(--accent-emerald)' }}>
          {modelEstimate.memory_footprint
            ? `${modelEstimate.memory_footprint.fp16_amp_mb} MB`
            : '24.0 MB'}
        </div>
        <div className="stat-desc">
          {isAr ? 'دقة نصفية AMP FP16' : 'AMP Half Precision'}
        </div>
      </div>

      <div className="stat-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="stat-label">
            {isAr ? 'رؤوس المهام النشطة' : 'Active Task Heads'}
          </div>
          <TermTooltip termKey="task_heads" size="sm" hideLabel={true} />
        </div>
        <div className="stat-value" style={{ color: 'var(--accent-amber)' }}>
          {modelConfig.tasks.length}
        </div>
        <div className="stat-desc">
          {isAr ? 'معمارية متعددة المهام' : 'Multi-Task Architecture'}
        </div>
      </div>
    </div>
  );
}

