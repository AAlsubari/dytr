// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Screen 6: The Educational Glassroom Parent Container - Industrial Swiss Design
// Zero Emojis Policy, 5 Granular Visual Simulators

import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import RoPECompass from './RoPECompass';
import FlashAttentionSim from './FlashAttentionSim';
import ContinualLearningSim from './ContinualLearningSim';
import AdaLoRASim from './AdaLoRASim';
import TokenAlignmentSim from './TokenAlignmentSim';

export default function GlassroomView() {
  const { lang, t } = useApp();
  const isAr = lang === 'ar';
  const [activeSim, setActiveSim] = useState('all');

  const simTabs = [
    { id: 'all', label: isAr ? 'جميع التجارب المعرفية' : 'All Experiments' },
    { id: 'rope', label: '1. RoPE Compass' },
    { id: 'flash', label: '2. FlashAttention' },
    { id: 'continual', label: isAr ? '3. درع الذاكرة EWC' : '3. EWC & Replay' },
    { id: 'adalora', label: isAr ? '4. أغلفة LoRA' : '4. Multi-Task LoRA' },
    { id: 'token', label: '5. Token Masking' },
  ];

  return (
    <section>
      {/* Header & Filter Pills */}
      <div style={{ marginBottom: '1.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '4px' }}>
            {t('glassroom_title')}
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
            {t('glassroom_subtitle')}
          </p>
        </div>

        {/* Filter Tabs */}
        <div style={{ display: 'flex', gap: '0.35rem', background: 'var(--bg-surface)', padding: '0.3rem', borderRadius: '10px', border: '1px solid var(--border-medium)', flexWrap: 'wrap' }}>
          {simTabs.map((tab) => {
            const isActive = activeSim === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSim(tab.id)}
                style={{
                  background: isActive ? 'var(--accent-steel)' : 'transparent',
                  color: isActive ? '#FFFFFF' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '0.4rem 0.85rem',
                  fontSize: '0.82rem',
                  fontWeight: isActive ? '700' : '500',
                  cursor: 'pointer',
                  transition: 'all 0.18s ease',
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        {/* 1. RoPE Compass */}
        {(activeSim === 'all' || activeSim === 'rope') && (
          <div>
            <div style={{ background: 'var(--accent-steel-bg)', border: '1px solid var(--accent-steel)', borderRadius: '10px', padding: '1rem', marginBottom: '0.85rem' }}>
              <h4 style={{ color: 'var(--accent-steel)', marginBottom: '3px', fontSize: '1.05rem', fontWeight: '700' }}>
                {t('sim_rope_title')}
              </h4>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: 0 }}>
                {t('sim_rope_simple')}
              </p>
            </div>
            <RoPECompass />
          </div>
        )}

        {/* 2. FlashAttention */}
        {(activeSim === 'all' || activeSim === 'flash') && (
          <div>
            <div style={{ background: 'var(--accent-pine-bg)', border: '1px solid var(--accent-pine)', borderRadius: '10px', padding: '1rem', marginBottom: '0.85rem' }}>
              <h4 style={{ color: 'var(--accent-pine)', marginBottom: '3px', fontSize: '1.05rem', fontWeight: '700' }}>
                {t('sim_flash_title')}
              </h4>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: 0 }}>
                {t('sim_flash_simple')}
              </p>
            </div>
            <FlashAttentionSim />
          </div>
        )}

        {/* 3. Continual Learning (EWC) */}
        {(activeSim === 'all' || activeSim === 'continual') && (
          <div>
            <div style={{ background: 'var(--accent-amber-bg)', border: '1px solid var(--accent-amber)', borderRadius: '10px', padding: '1rem', marginBottom: '0.85rem' }}>
              <h4 style={{ color: 'var(--accent-amber)', marginBottom: '3px', fontSize: '1.05rem', fontWeight: '700' }}>
                {t('sim_cl_title')}
              </h4>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: 0 }}>
                {t('sim_cl_simple')}
              </p>
            </div>
            <ContinualLearningSim />
          </div>
        )}

        {/* 4. AdaLoRA SVD Explorer */}
        {(activeSim === 'all' || activeSim === 'adalora') && (
          <div>
            <div style={{ background: 'var(--accent-steel-bg)', border: '1px solid var(--accent-steel)', borderRadius: '10px', padding: '1rem', marginBottom: '0.85rem' }}>
              <h4 style={{ color: 'var(--accent-steel)', marginBottom: '3px', fontSize: '1.05rem', fontWeight: '700' }}>
                {t('sim_adalora_title')}
              </h4>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: 0 }}>
                {t('sim_adalora_simple')}
              </p>
            </div>
            <AdaLoRASim />
          </div>
        )}

        {/* 5. Subword Tokenizer & Loss Masking (-100) */}
        {(activeSim === 'all' || activeSim === 'token') && (
          <div>
            <div style={{ background: 'var(--accent-terracotta-bg)', border: '1px solid var(--accent-terracotta)', borderRadius: '10px', padding: '1rem', marginBottom: '0.85rem' }}>
              <h4 style={{ color: 'var(--accent-terracotta)', marginBottom: '3px', fontSize: '1.05rem', fontWeight: '700' }}>
                {t('sim_token_title')}
              </h4>
              <p style={{ fontSize: '0.86rem', color: 'var(--text-primary)', margin: 0 }}>
                {t('sim_token_simple')}
              </p>
            </div>
            <TokenAlignmentSim />
          </div>
        )}
      </div>
    </section>
  );
}
