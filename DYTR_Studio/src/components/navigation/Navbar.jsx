// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// DyTrStudio Master Navigation Shell - Industrial Swiss Design & Zero Emojis Policy

import React from 'react';
import { Sun, Moon, HelpCircle, Activity, RefreshCw, Cpu, CheckCircle2 } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import useEngineStore from '../../store/useEngineStore';
import { ENGINE_MODE } from '../../communication/config';

export default function Navbar() {
  const {
    theme,
    toggleTheme,
    lang,
    toggleLanguage,
    t,
    activeView,
    setActiveView,
    setIsGuideOpen,
    capabilities,
  } = useApp();

  const engineMode = useEngineStore((state) => state.engineMode);
  const isReconnecting = useEngineStore((state) => state.isReconnecting);
  const reconnect = useEngineStore((state) => state.reconnect);

  const isLive = engineMode === ENGINE_MODE.LIVE || engineMode === ENGINE_MODE.LIVE_ENGINE;

  const navTabs = [
    { id: 'wizard', labelKey: 'mode_wizard' },
    { id: 'datasets', labelKey: 'mode_datasets' },
    { id: 'builder', labelKey: 'mode_builder' },
    { id: 'arena', labelKey: 'mode_arena' },
    { id: 'observatory', labelKey: 'mode_observatory' },
    { id: 'glassroom', labelKey: 'mode_glassroom' },
    { id: 'playground', labelKey: 'mode_playground' },
  ];

  return (
    <header className="glass-panel" style={{ margin: '1.25rem 0 2rem 0', padding: '1.25rem 1.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.25rem' }}>
        {/* Brand & Platform Architecture */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontSize: '1.65rem', fontWeight: '800', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                {t('app_title')}
              </span>
              <span className="badge badge-steel" style={{ fontSize: '0.72rem', padding: '0.15rem 0.5rem' }}>
                v0.1.3 · DyTR
              </span>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              {t('app_subtitle')}
            </p>
          </div>
        </div>

        {/* Global Central Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* Dual Mode Engine Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span
              className={`badge ${isLive ? 'badge-pine' : 'badge-amber'}`}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.45rem',
                fontSize: '0.78rem',
                padding: '0.4rem 0.75rem',
              }}
              title={isLive ? 'Connected to FastAPI DyTR backend' : 'Client-side simulation fallback'}
            >
              <span
                style={{
                  width: '7px',
                  height: '7px',
                  borderRadius: '50%',
                  background: 'currentColor',
                }}
              />
              {isLive ? t('engine_live') : t('engine_simulation')}
            </span>

            {!isLive && (
              <button
                onClick={() => reconnect()}
                disabled={isReconnecting}
                className="btn-secondary"
                style={{
                  padding: '0.35rem 0.7rem',
                  fontSize: '0.75rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  cursor: isReconnecting ? 'wait' : 'pointer',
                }}
              >
                <RefreshCw size={12} style={{ animation: isReconnecting ? 'spin 1s linear infinite' : 'none' }} />
                {isReconnecting ? t('engine_connecting') : t('engine_reconnect')}
              </button>
            )}
          </div>

          {/* Theme Toggle (Dark / Light) */}
          <button
            onClick={toggleTheme}
            className="btn-secondary"
            style={{
              padding: '0.45rem 0.85rem',
              fontSize: '0.82rem',
              fontWeight: '600',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
            title={theme === 'dark' ? t('theme_light') : t('theme_dark')}
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
            {theme === 'dark' ? t('theme_light') : t('theme_dark')}
          </button>

          {/* Language Switcher */}
          <button
            onClick={toggleLanguage}
            className="btn-secondary"
            style={{
              padding: '0.45rem 0.85rem',
              fontSize: '0.82rem',
              fontWeight: '700',
            }}
          >
            {t('lang_name')}
          </button>

          {/* Beginner Guide Trigger */}
          <button
            onClick={() => setIsGuideOpen(true)}
            className="btn-secondary"
            style={{
              padding: '0.45rem 0.85rem',
              fontSize: '0.82rem',
              borderColor: 'var(--accent-amber)',
              color: 'var(--accent-amber)',
              background: 'var(--accent-amber-bg)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
            }}
          >
            <HelpCircle size={14} />
            {t('help_button')}
          </button>

          {/* Hardware Capability Status */}
          <span className={`badge ${capabilities?.cuda_available ? 'badge-pine' : 'badge-steel'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <Cpu size={13} />
            {capabilities?.device_name || t('system_status')}
          </span>
        </div>
      </div>

      {/* 7 Central Navigation Tabs */}
      <nav
        style={{
          display: 'flex',
          gap: '0.5rem',
          marginTop: '1.25rem',
          overflowX: 'auto',
          paddingBottom: '0.25rem',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '0.9rem',
        }}
      >
        {navTabs.map((tab) => {
          const isActive = activeView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              style={{
                background: isActive ? 'var(--accent-steel)' : 'transparent',
                color: isActive ? '#FFFFFF' : 'var(--text-secondary)',
                border: isActive ? '1px solid var(--accent-steel)' : '1px solid transparent',
                borderRadius: '8px',
                padding: '0.55rem 1.15rem',
                fontSize: '0.88rem',
                fontWeight: isActive ? '700' : '500',
                cursor: 'pointer',
                transition: 'all 0.18s ease',
                whiteSpace: 'nowrap',
              }}
            >
              {t(tab.labelKey)}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
