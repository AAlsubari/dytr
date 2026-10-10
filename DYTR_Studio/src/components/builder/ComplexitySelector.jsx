// ==============================================================================
// DyTrStudio Complexity Level Selector Component
// Zero Emojis Policy - Bilingual RTL/LTR
// Integrated with 3-Layer Epistemological TermTooltip System
// ==============================================================================

import React from 'react';
import TermTooltip from '../common/TermTooltip';

export default function ComplexitySelector({
  activeLevel,
  onSelectLevel,
  language,
}) {
  const isAr = language === 'ar';

  const levels = [
    {
      id: 'preset',
      code: 'A',
      termKey: 'transformer_archetypes',
      title: isAr ? 'المستوى التلقائي (الموصى به)' : 'Automatic Presets (Recommended)',
      subtitle: isAr
        ? 'نماذج معدة مسبقا تناسب احتياجاتك دون الحاجة لخبرة تقنية'
        : 'Pre-configured models ready for use without engineering overhead',
    },
    {
      id: 'guided',
      code: 'B',
      termKey: 'anti_overfitting',
      title: isAr ? 'المستوى الإرشادي' : 'Guided Configuration',
      subtitle: isAr
        ? 'تحكم بسيط وسهل عبر مؤشرات عامة تشرح الأثر على السرعة والذاكرة'
        : 'Intuitive controls with human explanations of speed and capacity tradeoffs',
    },
    {
      id: 'advanced',
      code: 'C',
      termKey: 'd_model',
      title: isAr ? 'المستوى المتقدم (للمهندسين)' : 'Advanced Engineering',
      subtitle: isAr
        ? 'تحكم كامل في كافة معاملات شبكة المحولات الديناميكية ورؤوس المهام'
        : 'Full granular control over transformer backbone, heads, and PEFT',
    },
  ];

  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.85rem' }}>
        <span style={{ fontSize: '0.88rem', fontWeight: '700', color: 'var(--text-secondary)' }}>
          {isAr ? 'اختر مستوى التحكم بالمعمارية:' : 'Select Architecture Control Level:'}
        </span>
        <TermTooltip termKey="complexity_levels" size="sm" hideLabel={true} />
      </div>

      <div className="grid-3" style={{ gap: '1rem' }}>
        {levels.map((lvl) => {
          const isSelected = activeLevel === lvl.id;
          return (
            <div
              key={lvl.id}
              onClick={() => onSelectLevel(lvl.id)}
              style={{
                background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                border: isSelected ? '2px solid var(--accent-teal)' : '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                padding: '1.15rem 1.25rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: isSelected ? '0 4px 14px rgba(20, 184, 166, 0.12)' : 'none',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: '700',
                    fontFamily: 'var(--font-mono)',
                    color: isSelected ? 'var(--accent-teal)' : 'var(--text-muted)',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '4px',
                    background: 'var(--bg-tertiary)',
                  }}
                >
                  {lvl.code}
                </span>
                <TermTooltip termKey={lvl.termKey} size="sm" hideLabel={true} />
              </div>

              <div
                style={{
                  fontSize: '1rem',
                  fontWeight: '700',
                  color: isSelected ? 'var(--accent-teal)' : 'var(--text-primary)',
                  marginBottom: '0.35rem',
                }}
              >
                {lvl.title}
              </div>

              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                {lvl.subtitle}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

