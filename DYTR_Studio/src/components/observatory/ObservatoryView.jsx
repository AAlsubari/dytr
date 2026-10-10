// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Screen 5: R-Matrix Observatory (مرصد مصفوفة R للتشخيص المعرفي)
// Industrial Swiss Design, Zero Emojis, Spatiotemporal Memory Heatmap, Hover Tooltips

import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import useEngineStore from '../../store/useEngineStore';

export default function ObservatoryView() {
  const { lang } = useApp();
  const isAr = lang === 'ar';

  // 3x3 Evaluation Matrix: R[i][j] where i is training stage, j is test task
  const [rMatrix, setRMatrix] = useState([
    [0.92, 0.22, 0.10], // After Task 1 (IMDb)
    [0.89, 0.88, 0.15], // After Task 2 (AG_News)
    [0.87, 0.86, 0.84], // After Task 3 (CoNLL NER)
  ]);

  const [hoveredCell, setHoveredCell] = useState(null);

  const tasks = isAr
    ? ['المهمة 1: مشاعر (IMDb)', 'المهمة 2: مواضيع (AG_News)', 'المهمة 3: كيانات (CoNLL)']
    : ['Task 1: Sentiment (IMDb)', 'Task 2: Topic (AG_News)', 'Task 3: NER (CoNLL)'];

  // Continual Learning Metrics Computation
  // ACC = (R_3,1 + R_3,2 + R_3,3) / 3
  const finalRow = rMatrix[2];
  const acc = ((finalRow[0] + finalRow[1] + finalRow[2]) / 3) * 100;

  // BWT = 1/2 * ((R_3,1 - R_1,1) + (R_3,2 - R_2,2))
  const bwt = (( (rMatrix[2][0] - rMatrix[0][0]) + (rMatrix[2][1] - rMatrix[1][1]) ) / 2) * 100;

  // FWT = 1/2 * (R_1,2 - b_2 + R_2,3 - b_3) -- zero-shot transfer
  const fwt = ((rMatrix[0][1] + rMatrix[1][2]) / 2) * 100;

  const getCellExplanation = (i, j) => {
    if (i === j) {
      return isAr
        ? `القطر الرئيسي R[${i+1},${j+1}]: دقة النموذج الفورية على المهمة (${tasks[i]}) فور انتهاء تدريبها المباشر.`
        : `Diagonal R[${i+1},${j+1}]: Immediate post-training accuracy on (${tasks[i]}).`;
    }
    if (i > j) {
      return isAr
        ? `خلية الاسترجاع والذاكرة R[${i+1},${j+1}]: تقيس مدى احتفاظ النموذج بمهارة (${tasks[j]}) بعد أن تعلّم لاحقاً المهمة (${tasks[i]}). قيمة مرتفعة تدل على نجاح درع EWC ومصفوفة فيشر في منع النسيان الكارثي.`
        : `Retrospective Retention R[${i+1},${j+1}]: Evaluates retention of (${tasks[j]}) after subsequent training on (${tasks[i]}). High values prove EWC Fisher regularization stability.`;
    }
    return isAr
      ? `خلية التعميم المسبق (Zero-Shot) R[${i+1},${j+1}]: قدرة النموذج على الاستنتاج في (${tasks[j]}) قبل أن يبدأ تدريبه المخصص عليها.`
      : `Zero-Shot Forward Transfer R[${i+1},${j+1}]: Evaluates cross-task generalization on (${tasks[j]}) prior to dedicated training.`;
  };

  const getCellColor = (i, j, val) => {
    if (i === j) return 'var(--accent-steel)';
    if (i > j) {
      // Retrospective memory check
      return val >= 0.8 ? 'var(--accent-pine)' : 'var(--accent-terracotta)';
    }
    return 'var(--accent-amber)';
  };

  return (
    <section>
      {/* Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
          {isAr ? 'مرصد مصفوفة R والتشخيص المعرفي للذاكرة' : 'R-Matrix Observatory & Cognitive Diagnostics'}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {isAr
            ? 'تفكيك مصفوفة الذاكرة الزمنية والمكانية R لقياس النسيان الكارثي (BWT)، التعميم المستقبلي (FWT)، وكفاءة الاحتفاظ بالمعرفة.'
            : 'Decomposing the Spatiotemporal Memory Matrix to quantify catastrophic forgetting, backward retention, and forward transfer.'}
        </p>
      </div>

      {/* 3 Top Benchmark Metric Cards */}
      <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
        {/* Average Accuracy ACC */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isAr ? 'متوسط الدقة الإجمالية (ACC)' : 'Average Accuracy (ACC)'}
            </span>
            <span className="badge badge-steel">{isAr ? 'الكفاءة العامة' : 'Global Score'}</span>
          </div>
          <div className="mono-font" style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--accent-steel)', margin: '4px 0' }}>
            {acc.toFixed(2)}%
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            {isAr
              ? 'متوسط أداء النموذج عبر جميع المهام بعد إتمام مسار التدريب التتابعي بالكامل.'
              : 'Mean test accuracy across all sequential tasks at the conclusion of training.'}
          </p>
        </div>

        {/* Backward Transfer BWT */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isAr ? 'النقل العكسي والنسيان (BWT)' : 'Backward Transfer (BWT)'}
            </span>
            <span className={`badge ${bwt >= -5 ? 'badge-pine' : 'badge-terracotta'}`}>
              {bwt >= -5 ? (isAr ? 'استقرار الذاكرة' : 'Stable Memory') : (isAr ? 'نسيان كارثي' : 'Forgetting')}
            </span>
          </div>
          <div className="mono-font" style={{ fontSize: '1.75rem', fontWeight: '800', color: bwt >= -5 ? 'var(--accent-pine)' : 'var(--accent-terracotta)', margin: '4px 0' }}>
            {bwt >= 0 ? `+${bwt.toFixed(2)}%` : `${bwt.toFixed(2)}%`}
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            {isAr
              ? 'يقيس انخفاض أو استقرار دقة المهام السابقة؛ القيمة القريبة من الصفر تثبت حماية درع EWC للذاكرة.'
              : 'Measures preservation of past task skills. Values near 0 indicate total elimination of forgetting via EWC.'}
          </p>
        </div>

        {/* Forward Transfer FWT */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {isAr ? 'النقل الأمامي للتعميم (FWT)' : 'Forward Transfer (FWT)'}
            </span>
            <span className="badge badge-amber">{isAr ? 'التعلم المسبق' : 'Zero-Shot'}</span>
          </div>
          <div className="mono-font" style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--accent-amber)', margin: '4px 0' }}>
            {fwt.toFixed(2)}%
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            {isAr
              ? 'القدرة التوليدية والاستنتاجية للنموذج على مهام مستقبلية قبل تدريبه عليها مباشرة.'
              : 'Zero-shot generalization capability on future tasks prior to dedicated supervised fine-tuning.'}
          </p>
        </div>
      </div>

      {/* Interactive 3x3 Heatmap Matrix Grid */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          {isAr ? 'خريطة مصفوفة الذاكرة الحرارية التفاعلية (R_i,j Heatmap)' : 'Spatiotemporal Memory Matrix Heatmap (R_i,j)'}
        </h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          {isAr
            ? 'مرر مؤشر الفأرة على أي خلية لفحص الدلالة الرياضية والفيزيائية للاسترجاع العصبي:'
            : 'Hover over any cell in the 3x3 grid to inspect its exact mathematical formulation and cognitive interpretation:'}
        </p>

        <div style={{ overflowX: 'auto', marginBottom: '1rem' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-medium)' }}>
                <th style={{ padding: '0.75rem', textAlign: isAr ? 'right' : 'left', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.85rem' }}>
                  {isAr ? 'المرحلة المكتملة (i) \\ المهمة المختبرة (j)' : 'Completed Stage (i) \\ Evaluated Task (j)'}
                </th>
                {tasks.map((taskName, idx) => (
                  <th key={idx} style={{ padding: '0.75rem', color: 'var(--text-primary)', fontWeight: '700', fontSize: '0.85rem' }}>
                    {taskName}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rMatrix.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '0.85rem', textAlign: isAr ? 'right' : 'left', fontWeight: '600', color: 'var(--text-primary)', fontSize: '0.88rem' }}>
                    {isAr ? `بعد إنهاء ${tasks[i]}` : `After ${tasks[i]}`}
                  </td>
                  {row.map((val, j) => {
                    const isDiagonal = i === j;
                    const isRetrospective = i > j;
                    const isZeroShot = i < j;
                    const isHovered = hoveredCell?.i === i && hoveredCell?.j === j;

                    return (
                      <td key={j} style={{ padding: '0.5rem' }}>
                        <div
                          onMouseEnter={() => setHoveredCell({ i, j, val })}
                          onMouseLeave={() => setHoveredCell(null)}
                          className="mono-font"
                          style={{
                            padding: '0.85rem 1rem',
                            borderRadius: '8px',
                            background: isHovered
                              ? 'var(--bg-surface-solid)'
                              : isDiagonal
                              ? 'var(--accent-steel-bg)'
                              : isRetrospective
                              ? 'var(--accent-pine-bg)'
                              : 'var(--accent-amber-bg)',
                            border: `1px solid ${getCellColor(i, j, val)}`,
                            color: 'var(--text-primary)',
                            fontWeight: '700',
                            fontSize: '1.05rem',
                            cursor: 'pointer',
                            transition: 'all 0.18s ease',
                            transform: isHovered ? 'scale(1.04)' : 'scale(1)',
                            boxShadow: isHovered ? 'var(--shadow-md)' : 'none',
                          }}
                        >
                          {(val * 100).toFixed(1)}%
                          <div style={{ fontSize: '0.68rem', fontWeight: '500', color: 'var(--text-muted)', marginTop: '2px' }}>
                            {isDiagonal ? 'Diagonal (R_ii)' : isRetrospective ? 'Memory Retained' : 'Zero-shot (FWT)'}
                          </div>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Hover Explanation Box */}
        <div
          style={{
            background: 'var(--bg-tertiary)',
            padding: '1rem 1.25rem',
            borderRadius: '8px',
            border: '1px solid var(--border-medium)',
            minHeight: '65px',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {hoveredCell ? (
            <div>
              <strong style={{ color: getCellColor(hoveredCell.i, hoveredCell.j, hoveredCell.val), display: 'block', marginBottom: '3px' }}>
                R[{hoveredCell.i + 1}, {hoveredCell.j + 1}] = {(hoveredCell.val * 100).toFixed(1)}%
              </strong>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-primary)', margin: 0 }}>
                {getCellExplanation(hoveredCell.i, hoveredCell.j)}
              </p>
            </div>
          ) : (
            <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', margin: 0 }}>
              {isAr
                ? 'ضع المؤشر على أي خلية في المصفوفة بالأعلى لعرض التفسير الرياضي والمعرفي الدقيق.'
                : 'Hover over any matrix cell above to inspect its formal mathematical interpretation.'}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
