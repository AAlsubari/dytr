// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// Token Alignment & -100 Subword Loss Masking Visualizer
// Matching Academy Design Language: Token Chips, Bilingual, Dual Mode
// 3-Layer Academic Explanation (TermTooltip), Day/Night Adaptive Contrast
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function TokenAlignmentSim() {
  const { language, connectionMode } = useStudioStore();
  const isAr = language === 'ar';

  const [inputText, setInputText] = useState(
    isAr
      ? 'الباحث الأكاديمي الدكتور اكرم زار جامعة كامبريدج في المملكة المتحدة'
      : 'The researcher Dr. Akram visited the University of Cambridge in the United Kingdom'
  );
  const [data, setData] = useState(null);

  useEffect(() => {
    let isMounted = true;
    api.postTokenizeAlign({ text: inputText, task_type: 'token_classification' }).then((res) => {
      if (isMounted && res) setData(res);
    });
    return () => {
      isMounted = false;
    };
  }, [inputText]);

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'محاذاة التوكن وتكميم الخسارة (-100 Loss Masking)' : 'Subword Token Alignment & -100 Loss Masking'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'تجزئة الكلمات إلى مقاطع فرعية (Subwords) وتعيين قيمة الخسارة -100 للواحق حتى يتجاهلها PyTorch ولا يشوش على دقة التصنيف.'
              : 'Microscopic subword segmentation with gradient suppression (label = -100) for subsequent word fragments in PyTorch.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-teal">
            {data?.total_tokens || 0} {isAr ? 'توكن فرعي' : 'Subword Tokens'}
          </span>
          <span className="badge badge-neutral">
            {connectionMode === 'library' ? 'dytr Tokenizer' : (isAr ? 'محاكاة التجزئة' : 'Simulation Engine')}
          </span>
        </div>
      </div>

      <div className="form-group" style={{ marginBottom: '1.25rem' }}>
        <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
            <span>{isAr ? 'النص المدخل للمعاينة والتجزئة الحية:' : 'Input Sentence for Live Token Alignment:'}</span>
            <TermTooltip
              title={isAr ? 'تجزئة المقاطع وتكميم الخسارة (-100 Loss Masking)' : 'Subword Token Alignment & -100 Loss Masking'}
              explanation={
                isAr
                  ? 'تقنية تستخدم في تصنيف الكلمات (NER) حيث يُقسم الـ Tokenizer الكلمة لمقاطع؛ ويتم إسناد -100 للمقاطع اللاحقة حتى يتجاهلها PyTorch في حساب دالة الخسارة (CrossEntropyLoss) ولا يشوش على دقة التصنيف.'
                  : 'In PyTorch token classification, subwords following the initial word fragment receive label -100, which CrossEntropyLoss automatically ignores to avoid penalizing partial tokens.'
              }
              size="sm"
              hideLabel={true}
            />
          </span>
        </div>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder={isAr ? 'أدخل جملة لاختبار التجزئة والمحاذاة...' : 'Enter sentence to inspect subword fragmentation...'}
          className="form-input"
          style={{ width: '100%', padding: '0.75rem 1rem' }}
        />
      </div>

      {/* Visual Token Chips Grid */}
      <div style={{ background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: '1.25rem', border: '1px solid var(--border-subtle)' }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
          {isAr ? 'المقاطع المستخرجة وأهداف التدريب المعينة:' : 'Extracted Token Fragments & Loss Targets:'}
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
          {data?.tokens?.map((tok) => {
            const isIgnored = tok.assigned_label === '-100';
            return (
              <div
                key={tok.token_id}
                style={{
                  background: isIgnored ? 'rgba(239, 68, 68, 0.08)' : 'var(--bg-elevated)',
                  border: isIgnored ? '1px dashed rgba(239, 68, 68, 0.45)' : '1px solid var(--border-medium)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.5rem 0.85rem',
                  fontSize: '0.85rem',
                  boxShadow: 'var(--shadow-xs)',
                  transition: 'var(--transition)',
                }}
              >
                <div style={{ fontWeight: 700, color: isIgnored ? 'var(--text-muted)' : 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>
                  {tok.token_piece}
                </div>
                <div style={{ fontSize: '0.72rem', marginTop: '3px', color: isIgnored ? 'var(--accent-rose)' : 'var(--accent-emerald)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {isAr ? 'الهدف:' : 'Target:'} {tok.assigned_label}
                </div>
              </div>
            );
          })}
        </div>

        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem', lineHeight: 1.5 }}>
          {isAr
            ? 'ملاحظة علمية: القطعة الأولى من الكلمة تحمل وسم التصنيف الحقيقي (مثل B-PER أو B-LOC)، بينما اللواحق التابعة تمنح القيمة -100 حتى تتجاهلها دالة CrossEntropyLoss في PyTorch تلقائيا أثناء التحديث العكسي للتدرجات.'
            : 'Scientific Note: First token piece receives the true label, while continuation pieces receive target -100 so PyTorch CrossEntropyLoss ignores them during gradient updates.'}
        </p>
      </div>
    </div>
  );
}
