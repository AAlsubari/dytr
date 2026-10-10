// ==============================================================================
// DyTrStudio Visual Glassroom Studio
// Matching Academy Design Language: SectionCards, Tensor Mechanics, Dual Mode
// Zero Emojis Policy - RoPE Compass, Multi-Head Attention, EWC Memory Shield, LoRA Adapters & Alignment
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { academicDictionary } from '../../utils/academicDictionary';
import SectionCard from '../common/SectionCard';

import TokenAlignmentSim from './TokenAlignmentSim';
import RoPECompass from './RoPECompass';
import MultiHeadAttentionSim from './MultiHeadAttentionSim';
import ContinualLearningSim from './ContinualLearningSim';
import MultiTaskLoRASim from './MultiTaskLoRASim';
import { ArrowLeft, ArrowRight } from 'lucide-react';

export default function VisualGlassroomView() {
  const { language, setActiveModule } = useStudioStore();
  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  const [activeTab, setActiveTab] = useState('rope'); // 'token_align', 'rope', 'multihead_attn', 'continual', 'lora'

  const simTabs = [
    {
      id: 'token_align',
      code: '06.1',
      title: isAr ? 'معالجة الكلمات وتكميم الخسارة' : 'Token Alignment & Loss Masking',
      subtitle: isAr ? 'تقطيع الكلمات إلى مقاطع رقمية وحجب تدرجات اللواحق (-100)' : 'Subword tokenization and -100 loss masking for suffixes',
      badge: isAr ? 'معالجة المدخلات' : 'Input Preprocessing',
    },
    {
      id: 'rope',
      code: '06.2',
      title: isAr ? 'بوصلة التدوير الموضعي (RoPE)' : 'RoPE Rotation Compass',
      subtitle: isAr ? 'تدوير متجهات الكلمات رياضياً لحفظ ترتيبها ومسافاتها النسبية' : 'Rotary embeddings preserving relative positional distances',
      badge: isAr ? 'الترتيب المكاني' : 'Positional Encoding',
    },
    {
      id: 'multihead_attn',
      code: '06.3',
      title: isAr ? 'الانتباه متعدد الرؤوس (Q, K, V)' : 'Multi-Head Attention (Q, K, V)',
      subtitle: isAr ? 'كيف تبحث الكلمات عن بعضها وتفهم السياق المشترك عبر عدة رؤوس متوازية' : 'Scaled dot-product attention mapping across specialized subspace heads',
      badge: isAr ? 'محرك الفهم' : 'Attention Core',
    },
    {
      id: 'continual',
      code: '06.4',
      title: isAr ? 'حماية الذاكرة من النسيان (EWC)' : 'EWC Memory Shield',
      subtitle: isAr ? 'تثبيت الأوزان الذكية المهمة لمنع مسح المعرفة السابقة عند تعلم مهمة جديدة' : 'Fisher information penalty preserving critical weights across tasks',
      badge: isAr ? 'استدامة المعرفة' : 'Continual Stability',
    },
    {
      id: 'lora',
      code: '06.5',
      title: isAr ? 'أغلفة التكيف السريع (Multi-Task LoRA)' : 'Multi-Task LoRA Adaptation',
      subtitle: isAr ? 'تجميد أوزان النموذج وتدريب أغلفة خفيفة لكل مهمة لتوفير الذاكرة والوقت' : 'Freeze base backbone and train lightweight task-specific low-rank matrices',
      badge: isAr ? 'كفاءة المعاملات' : 'Parameter Efficiency',
    },
  ];

  const currentTab = simTabs.find((t) => t.id === activeTab) || simTabs[0];

  return (
    <div className="container" style={{ maxWidth: '1000px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Studio Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-teal">[06] {isAr ? 'تشريح آليات المحولات' : (d.module_glassroom || 'Neural Mechanics Lab')}</span>
          <span className="badge badge-neutral">
            {isAr ? 'رؤية بصرية تفاعلية لباطن المحول العصبي' : 'Transformer Neural Anatomy & Tensor Mechanics'}
          </span>
        </div>
        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
          {isAr ? 'تشريح وتفكيك آليات النموذج العصبي (Transformer Mechanics)' : 'Transformer Neural Anatomy & Mechanics'}
        </h1>
        <p style={{ fontSize: '0.92rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          {isAr
            ? 'تفكيك الصندوق الأسود: تفاعل مباشرة وببساطة مع آليات التدوير الموضعي (RoPE)، وانتباه الكلمات لبعضها (Q/K/V)، وحماية الذاكرة من النسيان (EWC).'
            : 'Deconstruct the neural black box: interact directly with RoPE positional math, Multi-Head Attention, and continual stability shields.'}
        </p>
      </div>

      {/* Simulator Navigation Bar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.75rem',
          marginBottom: '2rem',
        }}
      >
        {simTabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                background: isActive ? 'var(--bg-elevated)' : 'var(--bg-secondary)',
                border: isActive ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '0.85rem 1rem',
                textAlign: isAr ? 'right' : 'left',
                cursor: 'pointer',
                transition: 'var(--transition)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)' }}>
                  [{tab.code}]
                </span>
                <span className={`badge ${isActive ? 'badge-teal' : 'badge-neutral'}`} style={{ fontSize: '0.65rem' }}>
                  {tab.badge}
                </span>
              </div>
              <div style={{ fontSize: '0.85rem', fontWeight: 700, color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {tab.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Interactive Section Card for Active Simulator */}
      <SectionCard
        sectionCode={currentTab.code}
        title={currentTab.title}
        subtitle={
          isAr
            ? `لماذا هذه الآلية مهمة؟ ${currentTab.subtitle}. يمكنك تغيير المعلمات في الأسفل لمشاهدة التغير الحسابي لحظيا.`
            : `Why this mechanic matters: ${currentTab.subtitle}. Adjust parameters below to inspect instantaneous mathematical feedback.`
        }
      >
        <div style={{ minHeight: '340px' }}>
          {activeTab === 'token_align' && <TokenAlignmentSim />}
          {activeTab === 'rope' && <RoPECompass />}
          {activeTab === 'multihead_attn' && <MultiHeadAttentionSim />}
          {activeTab === 'continual' && <ContinualLearningSim />}
          {(activeTab === 'lora' || activeTab === 'adalora') && <MultiTaskLoRASim />}
        </div>
      </SectionCard>

      {/* Module Navigation Row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2.5rem', gap: '1rem', flexWrap: 'wrap' }}>
        <button
          onClick={() => setActiveModule('observatory')}
          className="btn btn-secondary"
          style={{ padding: '0.75rem 1.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          <span>{isAr ? 'العودة إلى مرصد القياسات (المرحلة 05)' : 'Back to Observatory (Stage 05)'}</span>
        </button>

        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {isAr ? 'المرحلة 06 من 11' : 'Stage 06 of 11'}
        </span>

        <button
          onClick={() => setActiveModule('evaluation')}
          className="btn btn-primary"
          style={{ padding: '0.75rem 1.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <span>{isAr ? 'الانتقال إلى مختبر التقييم (المرحلة 07)' : 'Next: Evaluation Lab (Stage 07)'}</span>
          <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
        </button>
      </div>
    </div>
  );
}
