// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Beginner Guide Modal Component

import React from 'react';
import { useApp } from '../../context/AppContext';

export default function SimpleGuideModal() {
  const { isGuideOpen, setIsGuideOpen, t } = useApp();

  if (!isGuideOpen) return null;

  return (
    <div className="modal-overlay" onClick={() => setIsGuideOpen(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.4rem' }}>{t('guide_title')}</h3>
          <button
            onClick={() => setIsGuideOpen(false)}
            aria-label="Close"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '4px',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
          <p>{t('guide_p1')}</p>
          <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
            <p style={{ fontWeight: '600', color: 'var(--accent-teal)', marginBottom: '4px' }}>{t('guide_p2')}</p>
            <p style={{ fontWeight: '600', color: 'var(--accent-amber)', marginBottom: '4px' }}>{t('guide_p3')}</p>
            <p style={{ fontWeight: '600', color: 'var(--accent-emerald)' }}>{t('guide_p4')}</p>
          </div>
        </div>

        <div style={{ marginTop: '1.75rem', textAlign: 'center' }}>
          <button onClick={() => setIsGuideOpen(false)} className="btn-huge" style={{ padding: '0.8rem 2rem', fontSize: '1rem' }}>
            {t('btn_close')}
          </button>
        </div>
      </div>
    </div>
  );
}
