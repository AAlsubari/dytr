// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Screen 7: Inference Playground (غرفة الاستنتاج)
// Industrial Swiss Design, Zero Emojis, Task Head Selector, Confidence Bars, Export Weights

import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../utils/api';

export default function PlaygroundView() {
  const { lang, t, selectedTask } = useApp();
  const isAr = lang === 'ar';

  const [activeTaskHead, setActiveTaskHead] = useState(selectedTask || 'task_sentiment');
  const [inputText, setInputText] = useState(
    isAr
      ? 'هذا الإطار العصبي يقدم استقراراً ملحوظاً وأداءً استثنائياً في معالجة البيانات.'
      : 'This new transformer framework delivers remarkable stability and exceptional throughput.'
  );
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [exportNotice, setExportNotice] = useState(null);

  const handlePredict = async () => {
    setIsLoading(true);
    const res = await api.predict({
      task_name: activeTaskHead,
      text: inputText,
    });
    setResult(res);
    setIsLoading(false);
  };

  const handleDownloadWeights = async () => {
    try {
      setExportNotice(isAr ? 'جاري تجهيز وتحميل ملف الأوزان الفعلي (.pt)...' : 'Preparing and downloading binary model weights (.pt)...');
      const response = await fetch('http://localhost:8000/api/v1/models/download/checkpoint?format=pt');
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dytr_${activeTaskHead}_checkpoint.pt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        const sizeMb = (blob.size / (1024 * 1024)).toFixed(2);
        setExportNotice(isAr ? `تم تحميل ملف الأوزان الفعلي (.pt) بنجاح (${sizeMb} ميجابايت)` : `Binary model weights (.pt) downloaded successfully (${sizeMb} MB)`);
      } else {
        throw new Error('Server returned status: ' + response.status);
      }
    } catch (err) {
      console.warn('Fallback direct download:', err);
      window.open('http://localhost:8000/api/v1/models/download/checkpoint?format=pt', '_blank');
      setExportNotice(isAr ? 'جاري بدء تحميل ملف الأوزان الثنائي عبر الخادم...' : 'Starting binary checkpoint download via backend server...');
    }
    setTimeout(() => setExportNotice(null), 4000);
  };

  const handleExportConfig = () => {
    const configData = JSON.stringify({
      framework: 'DyTR',
      version: '0.1.3',
      task_name: activeTaskHead,
      embed_dim: 256,
      num_layers: 4,
      num_heads: 4,
      use_flash_attention: true,
      use_rotary_embedding: true,
      use_si: true,
      exported_at: new Date().toISOString(),
    }, null, 2);

    const blob = new Blob([configData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dytr_${activeTaskHead}_config.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportNotice(isAr ? 'تم تصدير ملف الإعدادات (.json) بنجاح' : 'Model config (.json) exported successfully');
    setTimeout(() => setExportNotice(null), 3000);
  };

  return (
    <section>
      {/* Header */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
          {isAr ? 'غرفة الاستنتاج واختبار استجابة النموذج' : 'Inference Playground & Model Deployment'}
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
          {isAr
            ? 'اختبار النموذج المُدرب عبر رؤوس المهام المتعددة، فحص نسب الثقة الاحتمالية، وتصدير الأوزان (.pt).'
            : 'Evaluate real-time forward predictions across multi-task heads, inspect confidence distributions, and export checkpoints.'}
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Left Column: Task Head Selector & Prompt Input */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '0.85rem' }}>
            {isAr ? 'اختيار رأس المهمة والمدخلات' : 'Target Task Head & Prompt Input'}
          </h3>

          {/* Task Head Dropdown */}
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
              {isAr ? 'رأس المهمة المستهدفة (Task Head):' : 'Active Task Head:'}
            </label>
            <select
              value={activeTaskHead}
              onChange={(e) => {
                const head = e.target.value;
                setActiveTaskHead(head);
                if (head === 'task_ner') {
                  setInputText(isAr ? 'زار أحمد مقر شركة أرامكو في مدينة الرياض.' : 'Barack Obama visited Google headquarters in New York yesterday.');
                } else if (head === 'task_topic') {
                  setInputText(isAr ? 'أعلن البنك المركزي خفض أسعار الفائدة لدعم النمو الاقتصادي.' : 'Central bank raises benchmark interest rates amid surging inflation report.');
                } else {
                  setInputText(isAr ? 'هذا الإطار العصبي يقدم استقراراً ملحوظاً وأداءً استثنائياً.' : 'This new transformer framework delivers remarkable stability and exceptional throughput.');
                }
              }}
            >
              <option value="task_sentiment">{isAr ? 'المهمة 1: تحليل المشاعر (Sentiment IMDb)' : 'Task 1: Sentiment Analysis (IMDb)'}</option>
              <option value="task_topic">{isAr ? 'المهمة 2: تصنيف الأخبار والمواضيع (Topic AG_News)' : 'Task 2: Topic Classification (AG_News)'}</option>
              <option value="task_ner">{isAr ? 'المهمة 3: صياد الأسماء والكيانات (NER CoNLL)' : 'Task 3: Named Entity Recognition (CoNLL)'}</option>
            </select>
          </div>

          {/* Prompt Textarea */}
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>
              {isAr ? 'نص التجربة والاستنتاج:' : 'Evaluation Test Prompt:'}
            </label>
            <textarea
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={isAr ? 'اكتب جملة لاختبار استجابة النموذج...' : 'Type a test sentence to evaluate...'}
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handlePredict}
            disabled={isLoading}
            className="btn-primary"
            style={{ width: '100%', padding: '0.85rem', fontSize: '0.98rem' }}
          >
            {isLoading
              ? (isAr ? 'جاري تنفيذ التمرير الأمامي للشبكة...' : 'Processing Neural Forward Pass...')
              : (isAr ? 'اختبار استجابة النموذج الآن' : 'Run Model Inference')}
          </button>

          {/* Export Checkpoint Actions */}
          <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              {isAr ? 'تصدير ونشر النموذج (Model Deployment):' : 'Export & Deploy Model Artifacts:'}
            </div>
            <div style={{ display: 'flex', gap: '0.6rem' }}>
              <button onClick={handleDownloadWeights} className="btn-secondary" style={{ flex: 1, fontSize: '0.8rem', padding: '0.45rem' }}>
                {isAr ? 'تحميل الأوزان (.pt)' : 'Download Weights (.pt)'}
              </button>
              <button onClick={handleExportConfig} className="btn-secondary" style={{ flex: 1, fontSize: '0.8rem', padding: '0.45rem' }}>
                {isAr ? 'تصدير الإعدادات (.json)' : 'Export Config (.json)'}
              </button>
            </div>
            {exportNotice && (
              <div style={{ fontSize: '0.78rem', color: 'var(--accent-pine)', marginTop: '6px', textAlign: 'center' }}>
                {exportNotice}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Prediction Results & Confidence Distribution */}
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '0.85rem' }}>
            {isAr ? 'نتائج الاستنتاج والتوزيع الاحتمالي' : 'Prediction Result & Probability Distribution'}
          </h3>

          {result ? (
            <div>
              {result.task_type === 'sentence_classification' && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', background: 'var(--bg-tertiary)', padding: '0.85rem 1rem', borderRadius: '8px' }}>
                    <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                      {isAr ? 'الفئة المتوقعة:' : 'Predicted Class:'}
                    </span>
                    <span className="badge badge-pine" style={{ fontSize: '0.95rem', padding: '0.35rem 0.85rem' }}>
                      {result.prediction}
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                    {result.probabilities?.map((p, idx) => {
                      const probPercent = (p.prob * 100).toFixed(1);
                      return (
                        <div key={idx}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                            <span style={{ color: 'var(--text-primary)', fontWeight: '600' }}>{p.class_name}</span>
                            <span className="mono-font" style={{ color: 'var(--accent-steel)', fontWeight: '700' }}>{probPercent}%</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', background: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${probPercent}%`,
                                height: '100%',
                                background: 'var(--accent-steel)',
                                borderRadius: '4px',
                                transition: 'width 0.4s ease',
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {result.task_type === 'token_classification' && (
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    {isAr ? 'الكيانات والأسماء المستخرجة (Named Entities):' : 'Extracted Named Entity Tokens:'}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {result.tokens?.map((t, idx) => (
                      <span
                        key={idx}
                        style={{
                          background: t.is_entity ? 'var(--accent-amber-bg)' : 'var(--bg-tertiary)',
                          border: `1px solid ${t.is_entity ? 'var(--accent-amber)' : 'var(--border-subtle)'}`,
                          borderRadius: '6px',
                          padding: '0.4rem 0.65rem',
                          fontSize: '0.85rem',
                        }}
                      >
                        <strong>{t.word}</strong>
                        {t.is_entity && (
                          <span className="mono-font" style={{ color: 'var(--accent-amber)', marginInlineStart: '6px', fontSize: '0.75rem' }}>
                            [{t.tag}]
                          </span>
                        )}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '3.5rem 1rem', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '0.9rem' }}>
                {isAr
                  ? 'انقر على "اختبار استجابة النموذج الآن" لتنفيذ الاستنتاج المباشر وعرض أشرطة الاحتمالات.'
                  : 'Click "Run Model Inference" to execute live forward pass and view confidence bars.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
