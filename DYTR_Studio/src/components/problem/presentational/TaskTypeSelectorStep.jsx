// ==============================================================================
// Step 3: Task Type Selection Presentation Component (Pure Presentational / Dumb)
// Zero hardcoded task cards - Renders dynamic DyTR architectures from API
// ==============================================================================

import React from 'react';
import SectionCard from '../../common/SectionCard';
import TermTooltip from '../../common/TermTooltip';

function TaskTypeSelectorStep({
  isAr,
  taskTypes = [],
  selectedTask,
  onSelectTask,
}) {
  return (
    <div className="problem-step-content">
      <SectionCard
        sectionCode="01.3"
        title={isAr ? 'اختر استراتيجية ومهمة التعلم (dytr Strategy)' : 'Select dytr Training Strategy'}
        subtitle={
          isAr ? (
            <>
              تطابق معمارية النموذج مع <TermTooltip termKey="task_type" customLabel="نوع المهمة العصبية" /> المغطية لكافة فروع معالجة اللغة الطبيعية.
            </>
          ) : (
            <>
              Map architecture with native <TermTooltip termKey="task_type" customLabel="Neural Task Framing" />.
            </>
          )
        }
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1rem',
            marginBottom: '1.5rem',
          }}
        >
          {taskTypes.map((card) => {
            const isSelected = selectedTask === card.id;
            return (
              <div
                key={card.id}
                onClick={() => onSelectTask(card.id)}
                style={{
                  padding: '1.25rem',
                  borderRadius: '8px',
                  border: isSelected ? '2px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-card)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span className="badge badge-neutral" style={{ fontSize: '0.75rem' }}>
                      {isAr ? card.badge_ar : card.badge_en}
                    </span>
                    {isSelected && <span className="badge badge-teal">{isAr ? 'محدد' : 'Selected'}</span>}
                  </div>
                  <div style={{ fontWeight: 700, fontSize: '0.98rem', color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
                    {isAr ? card.title_ar : card.title_en}
                  </div>
                  <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.45, marginBottom: '0.75rem' }}>
                    {isAr ? card.desc_ar : card.desc_en}
                  </p>
                </div>
                <div
                  style={{
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                    background: 'var(--bg-secondary)',
                    padding: '0.4rem 0.6rem',
                    borderRadius: '4px',
                    color: 'var(--accent-teal)',
                  }}
                >
                  {isAr ? card.example_ar : card.example_en}
                </div>
              </div>
            );
          })}
        </div>
      </SectionCard>
    </div>
  );
}

export default React.memo(TaskTypeSelectorStep);
