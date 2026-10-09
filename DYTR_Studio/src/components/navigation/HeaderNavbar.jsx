// ==============================================================================
// DYTR Studio Academic Header Navbar Component
// Modern Glassmorphism Architecture - Zero Emojis Policy - Responsive RTL/LTR
// Streamlined Header: Brand, Navigation Tabs, and Unified Settings Trigger
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { academicDictionary } from '../../utils/academicDictionary';
import SystemSettingsModal from './SystemSettingsModal';
import DytrLogo from '../common/DytrLogo';

export default function HeaderNavbar() {
  const {
    activeModule,
    setActiveModule,
    language,
  } = useStudioStore();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  const navItems = [
    { id: 'academy', label: d.module_academy || (isAr ? 'الأكاديمية التعليمية' : 'Academy Wizard'), code: '00' },
    { id: 'problem', label: d.module_problem || (isAr ? 'تحديد المشكلة' : 'Problem Definition'), code: '01' },
    { id: 'dataprep', label: d.module_dataprep || (isAr ? 'تجهيز البيانات' : 'Data Prep'), code: '02' },
    { id: 'builder', label: d.module_builder || (isAr ? 'بناء المعمارية' : 'Architecture'), code: '03' },
    { id: 'arena', label: d.module_arena || (isAr ? 'التدريب وبناء وتحديث الأوزان' : 'Training Arena'), code: '04' },
    { id: 'observatory', label: d.module_observatory || (isAr ? 'مرصد القياسات' : 'Observatory'), code: '05' },
    { id: 'evaluation', label: d.module_evaluation || (isAr ? 'التقييم والتشخيص' : 'Evaluation Lab'), code: '06' },
    { id: 'playground', label: d.module_playground || (isAr ? 'مختبر الاستنتاج' : 'Playground'), code: '07' },
    { id: 'export', label: d.module_export || (isAr ? 'تصدير الأكواد' : 'Code Export'), code: '08' },
    { id: 'continual', label: d.module_continual || (isAr ? 'التعلم المستمر' : 'Continual Learning'), code: '09' },
  ];

  return (
    <header
      style={{
        background: 'var(--bg-secondary)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border-subtle)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.3s ease',
      }}
    >
      <div className="container" style={{ padding: '0.75rem 1.5rem 0.5rem 1.5rem' }}>
        {/* Top Tier: Brand & Unified Settings Trigger */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
          }}
        >
          {/* Brand & Academic Subtitle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', cursor: 'pointer' }} onClick={() => setActiveModule('showcase')}>
            <DytrLogo variant="icon" size={38} glow />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontWeight: '900', fontSize: '1.2rem', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                  DYTR Studio
                </span>
                <span
                  style={{
                    fontSize: '0.68rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    padding: '0.1rem 0.45rem',
                    borderRadius: '4px',
                    background: 'rgba(20, 184, 166, 0.12)',
                    color: 'var(--accent-teal)',
                    border: '1px solid rgba(20, 184, 166, 0.3)',
                  }}
                >
                  v0.1.3 Core
                </span>
              </div>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '1px' }}>
                {isAr ? 'منصة المحولات الديناميكية والتعلم المستمر' : 'Dynamic Transformers & Continual Learning Platform'}
              </div>
            </div>
          </div>

          {/* Top Control Bar: Only the Unified Settings Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => setIsSettingsOpen(true)}
              className="btn btn-secondary btn-sm"
              title={isAr ? 'إعدادات المنصة والمحرك والمظهر' : 'Platform, Engine & Theme Settings'}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.45rem 0.9rem',
                borderRadius: '8px',
                fontWeight: 700,
                fontSize: '0.8rem',
                border: '1px solid var(--border-medium)',
                background: 'var(--bg-elevated)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
              <span>{isAr ? 'الإعدادات' : 'Settings'}</span>
            </button>
          </div>
        </div>

        {/* Lower Tier: Interactive 10-Stage Academic Navigation Tabs */}
        <nav
          style={{
            display: 'flex',
            gap: '0.4rem',
            marginTop: '0.85rem',
            overflowX: 'auto',
            paddingBottom: '0.35rem',
            scrollbarWidth: 'thin',
          }}
        >
          {navItems.map((item) => {
            const isActive = activeModule === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveModule(item.id)}
                className={`nav-tab ${isActive ? 'active' : ''}`}
                style={{
                  whiteSpace: 'nowrap',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  padding: '0.45rem 0.85rem',
                  borderRadius: '8px',
                  fontSize: '0.82rem',
                  fontWeight: isActive ? 800 : 600,
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                  position: 'relative',
                  border: isActive ? '1px solid var(--accent-teal)' : '1px solid transparent',
                  background: isActive ? 'var(--bg-elevated)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontFamily: 'var(--font-mono)',
                    color: isActive ? 'var(--accent-teal)' : 'var(--text-muted)',
                    fontWeight: 700,
                  }}
                >
                  [{item.code}]
                </span>
                <span style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Centered System & Architecture Settings Modal */}
      <SystemSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </header>
  );
}
