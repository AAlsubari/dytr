// ==============================================================================
// DYTR Studio Academic Sidebar Navigation Component (القائمة الجانبية التفاعلية - )
// Inspired by Google Gemini App & Workspace Navigation Drawer
// Modern Glassmorphism Architecture - Zero Emojis Policy - Responsive RTL/LTR
// Position: Fixed to Viewport (Right side for Arabic RTL / Left side for English LTR)
// Top Spark Logo click toggles Drawer (Collapsed 68px Dock vs Expanded 280px Drawer)
// Integrated Gemini-style 4-Pointed Spark Logo, 10 Stages, Unified Settings & Researcher Avatar
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { academicDictionary } from '../../utils/academicDictionary';
import SystemSettingsModal from './SystemSettingsModal';
import DytrLogo from '../common/DytrLogo';

// Colorful 4-Pointed AI Spark Logo 
function GeminiSparkIcon({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" style={{ display: 'block', flexShrink: 0 }}>
      <defs>
        <linearGradient id="dytrGeminiGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#4E82EE" />
          <stop offset="30%" stopColor="#2BB8D6" />
          <stop offset="65%" stopColor="#8E54E9" />
          <stop offset="85%" stopColor="#EB5757" />
          <stop offset="100%" stopColor="#F2994A" />
        </linearGradient>
        <filter id="sparkGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      <path
        d="M12 0C12 6.627 6.627 12 0 12C6.627 12 12 17.373 12 24C12 17.373 17.373 12 24 12C17.373 12 12 6.627 12 0Z"
        fill="url(#dytrGeminiGrad)"
        filter="url(#sparkGlow)"
      />
    </svg>
  );
}

// Dedicated High-Contrast Outline Vector SVG Icons for each Stage
function NavStageIcon({ icon, size = 20, color = 'currentColor' }) {
  switch (icon) {
    case 'showcase': // HQ
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z" />
        </svg>
      );
    case 'academy': // 00
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
          <path d="M6 12v5c0 2 3 3 6 3s6-1 6-3v-5" />
        </svg>
      );
    case 'problem': // 01
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
          <path d="M11 8v6M8 11h6" />
        </svg>
      );
    case 'dataprep': // 02
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          <line x1="12" y1="11" x2="12" y2="17" />
          <line x1="9" y1="14" x2="15" y2="14" />
        </svg>
      );
    case 'builder': // 03
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="16" height="16" rx="3" />
          <circle cx="9" cy="9" r="1.5" />
          <circle cx="15" cy="9" r="1.5" />
          <circle cx="9" cy="15" r="1.5" />
          <circle cx="15" cy="15" r="1.5" />
          <line x1="9" y1="10.5" x2="9" y2="13.5" />
          <line x1="15" y1="10.5" x2="15" y2="13.5" />
          <line x1="10.5" y1="9" x2="13.5" y2="9" />
          <line x1="10.5" y1="15" x2="13.5" y2="15" />
        </svg>
      );
    case 'arena': // 04
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
      );
    case 'observatory': // 05
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
          <line x1="2" y1="20" x2="22" y2="20" />
        </svg>
      );
    case 'evaluation': // 06
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <polyline points="9 12 11 14 15 10" />
        </svg>
      );
    case 'playground': // 07
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none" />
        </svg>
      );
    case 'export': // 08
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="16 18 22 12 16 6" />
          <polyline points="8 6 2 12 8 18" />
        </svg>
      );
    case 'continual': // 09
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
        </svg>
      );
    default:
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
        </svg>
      );
  }
}

// Theme Toggle Icon (Sun rays with Crescent Moon center - Gemini style)
function ThemeGeminiIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      {/* Circle center with crescent fill */}
      <circle cx="12" cy="12" r="5" />
      <path d="M12 7a5 5 0 0 1 5 5 5 5 0 0 1-5 5" fill="currentColor" fillOpacity="0.25" />
      {/* 8 Sun Rays */}
      <line x1="12" y1="2" x2="12" y2="4.2" />
      <line x1="12" y1="19.8" x2="12" y2="22" />
      <line x1="4.93" y1="4.93" x2="6.45" y2="6.45" />
      <line x1="17.55" y1="17.55" x2="19.07" y2="19.07" />
      <line x1="2" y1="12" x2="4.2" y2="12" />
      <line x1="19.8" y1="12" x2="22" y2="12" />
      <line x1="4.93" y1="19.07" x2="6.45" y2="17.55" />
      <line x1="17.55" y1="6.45" x2="19.07" y2="4.93" />
    </svg>
  );
}

// Language Toggle Icon (Earth Globe / كرة أرضية)
function GlobeLanguageIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

export default function SidebarNav({ isCollapsed, setIsCollapsed }) {
  const {
    activeModule,
    setActiveModule,
    language,
    toggleLanguage,
    theme,
    toggleTheme,
    capabilities,
    isLibraryConnected,
  } = useStudioStore();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [hoveredTooltip, setHoveredTooltip] = useState(null);

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  const navItems = [
    { id: 'academy', code: '00', label_ar: 'الأكاديمية التعليمية', label_en: 'Academy Wizard', icon: 'academy', desc_ar: 'الدليل الأكاديمي الشامل وتأسيس المفاهيم', desc_en: 'Academic foundation & concept wizard' },
    { id: 'problem', code: '01', label_ar: 'تحديد وصياغة المشكلة', label_en: 'Problem Framing', icon: 'problem', desc_ar: 'صياغة المسألة والتحقق من ضرورة الذكاء الاصطناعي', desc_en: 'Framing AI necessity & task hypotheses' },
    { id: 'dataprep', code: '02', label_ar: 'تجهيز وتعدين البيانات', label_en: 'Data Mining & Prep', icon: 'dataprep', desc_ar: 'استيراد وتنظيف وفحص جودة البيانات', desc_en: 'CRISP-DM ingestion & dataset quality audit' },
    { id: 'builder', code: '03', label_ar: 'بناء واختيار المعمارية', label_en: 'Architecture Studio', icon: 'builder', desc_ar: 'تحديد أبعاد المحول وتحميل النماذج ونقاط الحفظ', desc_en: 'Transformer sizing & local checkpoint loader' },
    { id: 'arena', code: '04', label_ar: 'التدريب و بناء النماذج', label_en: 'Training Arena', icon: 'arena', desc_ar: 'ضبط المعلمات الفائقة ودورة التعلم الحية', desc_en: 'Hyperparameter setup & dynamic training loop' },
    { id: 'observatory', code: '05', label_ar: 'مرصد القياسات الحية', label_en: 'Telemetry Observatory', icon: 'observatory', desc_ar: 'منحنيات التقارب اللحظية وتصدير المشاريع', desc_en: 'Live loss telemetry & scientific bundle export' },
    { id: 'evaluation', code: '06', label_ar: 'التقييم والتشخيص', label_en: 'Evaluation Lab', icon: 'evaluation', desc_ar: 'مصفوفة الارتباك ودقة الفئات وتحليل الأخطاء', desc_en: 'Confusion matrix & error diagnostics' },
    { id: 'playground', code: '07', label_ar: 'مختبر الاستنتاج اللحظي', label_en: 'Inference Playground', icon: 'playground', desc_ar: 'تجربة تنبؤات النموذج وفحص أوزان الانتباه', desc_en: 'Interactive inference & softmax distribution' },
    { id: 'export', code: '08', label_ar: 'تصدير الأكواد والمشاريع', label_en: 'Code Export Studio', icon: 'export', desc_ar: 'توليد حزم بايثون ودفاتر Jupyter و ONNX', desc_en: 'Production Python, Jupyter & ONNX generators' },
    { id: 'continual', code: '09', label_ar: 'التعلم المستمر وحماية الذاكرة', label_en: 'Continual Learning', icon: 'continual', desc_ar: 'حماية المعرفة السابقة عبر EWC و Replay Buffer', desc_en: 'Sequential adaptation without catastrophic forgetting' },
  ];

  return (
    <>
      <aside
        style={{
          width: isCollapsed ? '68px' : '280px',
          minWidth: isCollapsed ? '68px' : '280px',
          maxWidth: isCollapsed ? '68px' : '280px',
          height: '100vh',
          position: 'fixed',
          top: 0,
          bottom: 0,
          right: isAr ? 0 : 'auto',
          left: isAr ? 'auto' : 0,
          zIndex: 1000,
          background: 'var(--bg-secondary)',
          borderInlineEnd: '1px solid var(--border-subtle)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          transition: 'width 0.28s cubic-bezier(0.16, 1, 0.3, 1), min-width 0.28s cubic-bezier(0.16, 1, 0.3, 1), max-width 0.28s cubic-bezier(0.16, 1, 0.3, 1)',
          boxShadow: isAr
            ? '-4px 0 28px rgba(0, 0, 0, 0.18)'
            : '4px 0 28px rgba(0, 0, 0, 0.18)',
          userSelect: 'none',
          overflow: 'hidden',
        }}
      >
        {/* ============================================================== */}
        {/* Top Header Section: Gemini Spark Logo (Toggle Button) & Brand */}
        {/* ============================================================== */}
        <div
          style={{
            padding: isCollapsed ? '1rem 0' : '1.15rem 1.15rem',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: isCollapsed ? 'center' : 'stretch',
            gap: '0.65rem',
          }}
        >
          {/* Logo Bar - Clicking Spark toggles drawer like Gemini */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: isCollapsed ? 'center' : 'space-between',
              gap: '0.65rem',
            }}
          >
            {/* Spark Logo Button */}
            <button
              type="button"
              onClick={() => setIsCollapsed(!isCollapsed)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                minWidth: 0,
                cursor: 'pointer',
                background: 'transparent',
                border: 'none',
                padding: isCollapsed ? '0.4rem' : '0.2rem',
                borderRadius: '10px',
                transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                outline: 'none',
              }}
              title={
                isCollapsed
                  ? (isAr ? 'فتح القائمة الجانبية ' : 'Expand Sidebar ')
                  : (isAr ? 'طي القائمة الجانبية' : 'Collapse Sidebar')
              }
            >
              <div
                style={{
                  width: '38px',
                  height: '38px',
                  borderRadius: '10px',
                  background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%)',
                  border: '1px solid var(--accent-teal)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 16px rgba(20, 184, 166, 0.25)',
                  flexShrink: 0,
                  transition: 'transform 0.2s ease',
                }}
              >
                <DytrLogo variant="icon" size={28} glow />
              </div>

              {!isCollapsed && (
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    setActiveModule('showcase');
                  }}
                  style={{ textAlign: isAr ? 'right' : 'left', overflow: 'hidden', whiteSpace: 'nowrap', cursor: 'pointer' }}
                  title={isAr ? 'الانتقال إلى واجهة الاستعراض والتعريف الرئيسية' : 'Go to Showcase & Overview'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontWeight: '900', fontSize: '1.08rem', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                      DYTR Studio
                    </span>
                    <span
                      style={{
                        fontSize: '0.62rem',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 800,
                        padding: '0.05rem 0.35rem',
                        borderRadius: '4px',
                        background: 'rgba(14, 165, 233, 0.12)',
                        color: 'var(--accent-teal)',
                        border: '1px solid rgba(14, 165, 233, 0.3)',
                      }}
                    >
                      v0.1.3
                    </span>
                  </div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '1px' }}>
                    {isAr ? 'منصة المحولات والتعلم المستمر' : 'Dynamic Transformers Platform'}
                  </div>
                </div>
              )}
            </button>

            {/* Collapse / Close Arrow Button (when expanded) */}
            {!isCollapsed && (
              <button
                type="button"
                onClick={() => setIsCollapsed(true)}
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '6px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-medium)',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  padding: 0,
                  transition: 'all 0.15s ease',
                  flexShrink: 0,
                }}
                title={isAr ? 'طي القائمة' : 'Collapse Sidebar'}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{
                    transform: isAr ? 'rotate(180deg)' : 'none',
                  }}
                >
                  <polyline points="11 17 6 12 11 7" />
                  <polyline points="18 17 13 12 18 7" />
                </svg>
              </button>
            )}
          </div>

          {/* Hardware & Live Engine Status Mini-Badge (Expanded Only) */}
          {!isCollapsed && (
            <div
              style={{
                marginTop: '0.4rem',
                padding: '0.45rem 0.65rem',
                borderRadius: '6px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '0.7rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span
                  style={{
                    width: '7px',
                    height: '7px',
                    borderRadius: '50%',
                    background: isLibraryConnected ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                    boxShadow: isLibraryConnected ? '0 0 8px var(--accent-emerald)' : 'none',
                  }}
                />
                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
                  {isLibraryConnected ? (isAr ? 'محرك متصل' : 'PyTorch Live') : (isAr ? 'وضع المحاكاة' : 'Simulation')}
                </span>
              </div>
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.65rem',
                  color: capabilities?.cuda_available ? 'var(--accent-emerald)' : 'var(--text-muted)',
                  fontWeight: 700,
                }}
              >
                {capabilities?.cuda_available ? 'CUDA GPU' : 'CPU'}
              </span>
            </div>
          )}
        </div>

        {/* ============================================================== */}
        {/* ============================================================== */}
        {/* Center Section: Scrollable 10-Stage Academic Navigation List */}
        {/* ============================================================== */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            padding: isCollapsed ? '0.75rem 0.35rem' : '0.75rem 0.65rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
            alignItems: isCollapsed ? 'center' : 'stretch',
            scrollbarWidth: 'thin',
            width: '100%',
            maxWidth: '100%',
          }}
        >
          {navItems.map((item) => {
            const isActive = activeModule === item.id;
            const isHovered = hoveredTooltip?.item?.id === item.id;

            return (
              <div
                key={item.id}
                style={{
                  width: isCollapsed ? '44px' : '100%',
                  display: 'flex',
                  justifyContent: 'center',
                }}
              >
                <button
                  type="button"
                  onClick={() => setActiveModule(item.id)}
                  onMouseEnter={(e) => {
                    if (isCollapsed) {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setHoveredTooltip({
                        top: rect.top + rect.height / 2,
                        item,
                      });
                    }
                  }}
                  onMouseLeave={() => setHoveredTooltip(null)}
                  style={{
                    width: isCollapsed ? '44px' : '100%',
                    height: isCollapsed ? '44px' : 'auto',
                    display: 'flex',
                    alignItems: 'center',
                    gap: isCollapsed ? 0 : '0.65rem',
                    justifyContent: isCollapsed ? 'center' : 'flex-start',
                    padding: isCollapsed ? 0 : '0.6rem 0.75rem',
                    borderRadius: isCollapsed ? '10px' : '8px',
                    border: 'none',
                    borderInlineStart: !isCollapsed && isActive
                      ? '4px solid var(--accent-teal)'
                      : '4px solid transparent',
                    background: isActive
                      ? isCollapsed
                        ? 'linear-gradient(135deg, rgba(78, 130, 238, 0.18) 0%, rgba(14, 165, 233, 0.12) 100%)'
                        : 'linear-gradient(90deg, rgba(14, 165, 233, 0.14) 0%, rgba(14, 165, 233, 0.03) 100%)'
                      : isHovered
                      ? 'var(--bg-elevated)'
                      : 'transparent',
                    boxShadow: isCollapsed && isActive
                      ? '0 0 12px rgba(78, 130, 238, 0.25), inset 0 0 0 1px rgba(78, 130, 238, 0.4)'
                      : 'none',
                    color: isActive
                      ? 'var(--accent-teal)'
                      : isHovered
                      ? 'var(--text-primary)'
                      : 'var(--text-secondary)',
                    cursor: 'pointer',
                    transition: 'all 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
                    textAlign: isAr ? 'right' : 'left',
                    outline: 'none',
                    flexShrink: 0,
                  }}
                  title={isCollapsed ? `[${item.code}] ${isAr ? item.label_ar : item.label_en}` : undefined}
                >
                  {/* Stage Icon */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isActive ? 'var(--accent-teal)' : 'inherit',
                      flexShrink: 0,
                    }}
                  >
                    <NavStageIcon icon={item.icon} size={20} color={isActive ? 'var(--accent-teal)' : 'currentColor'} />
                  </div>

                  {/* Stage Label & Code when Expanded */}
                  {!isCollapsed && (
                    <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.35rem' }}>
                        <span
                          style={{
                            fontSize: '0.82rem',
                            fontWeight: isActive ? 800 : 600,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {isAr ? item.label_ar : item.label_en}
                        </span>
                        <span
                          style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.68rem',
                            fontWeight: 700,
                            color: isActive ? 'var(--accent-teal)' : 'var(--text-muted)',
                            background: isActive ? 'rgba(14, 165, 233, 0.15)' : 'var(--bg-tertiary)',
                            padding: '0.05rem 0.35rem',
                            borderRadius: '4px',
                            flexShrink: 0,
                          }}
                        >
                          [{item.code}]
                        </span>
                      </div>
                    </div>
                  )}
                </button>
              </div>
            );
          })}
        </div>

        {/* ============================================================== */}
        {/* Bottom Section: Settings Gear, Theme (Sun/Moon) & Language (Globe) */}
        {/* ============================================================== */}
        <div
          style={{
            padding: isCollapsed ? '0.75rem 0' : '0.85rem 0.85rem',
            borderTop: '1px solid var(--border-subtle)',
            background: 'var(--bg-tertiary)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: isCollapsed ? 'center' : 'stretch',
            gap: '0.45rem',
          }}
        >
          {/* Settings Gear Button */}
          <button
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            style={{
              width: isCollapsed ? '44px' : '100%',
              height: isCollapsed ? '44px' : 'auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: isCollapsed ? 'center' : 'flex-start',
              gap: isCollapsed ? 0 : '0.65rem',
              padding: isCollapsed ? 0 : '0.55rem 0.75rem',
              borderRadius: isCollapsed ? '10px' : '8px',
              border: '1px solid var(--border-medium)',
              background: 'var(--bg-elevated)',
              color: 'var(--text-primary)',
              fontWeight: 700,
              fontSize: '0.8rem',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            title={isAr ? 'إعدادات المنصة والمحرك والمظهر' : 'Platform, Engine & Theme Settings'}
          >
            <svg
              width="19"
              height="19"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--accent-teal)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ flexShrink: 0 }}
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            {!isCollapsed && <span>{isAr ? 'الإعدادات والمحرك' : 'Settings & Engine'}</span>}
          </button>

          {/* Collapsed Mode: Direct Icon Buttons for Theme & Globe */}
          {isCollapsed && (
            <>
              {/* Theme Toggle Button (Sun with Moon Crescent) */}
              <button
                type="button"
                onClick={toggleTheme}
                style={{
                  width: '44px',
                  height: '44px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '10px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title={
                  theme === 'dark'
                    ? (isAr ? 'التحويل للوضع الفاتح (Theme)' : 'Switch to Light Theme')
                    : (isAr ? 'التحويل للوضع الداكن (Theme)' : 'Switch to Dark Theme')
                }
              >
                <ThemeGeminiIcon size={19} color={theme === 'dark' ? '#F59E0B' : 'var(--text-primary)'} />
              </button>

              {/* Language Toggle Button (Globe Earth Sphere) */}
              <button
                type="button"
                onClick={toggleLanguage}
                style={{
                  width: '44px',
                  height: '44px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '10px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--accent-teal)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                title={isAr ? 'Switch to English (اللغة)' : 'التحويل للغة العربية (Language)'}
              >
                <GlobeLanguageIcon size={19} color="var(--accent-teal)" />
              </button>
            </>
          )}

          {/* Expanded Mode: Full 2-Column Buttons with Icons & Text */}
          {!isCollapsed && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem', marginTop: '0.1rem' }}>
              {/* Language with Globe Icon */}
              <button
                type="button"
                onClick={toggleLanguage}
                style={{
                  padding: '0.45rem 0.5rem',
                  borderRadius: '7px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-secondary)',
                  fontSize: '0.74rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.4rem',
                  transition: 'all 0.15s ease',
                }}
                title={isAr ? 'Switch to English' : 'التحويل للغة العربية'}
              >
                <GlobeLanguageIcon size={15} color="var(--accent-teal)" />
                <span>{isAr ? 'English' : 'عربي'}</span>
              </button>

              {/* Theme with Sun/Moon Icon */}
              <button
                type="button"
                onClick={toggleTheme}
                style={{
                  padding: '0.45rem 0.5rem',
                  borderRadius: '7px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-secondary)',
                  fontSize: '0.74rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.4rem',
                  transition: 'all 0.15s ease',
                }}
                title={theme === 'dark' ? 'Switch to Light' : 'التحويل للوضع الداكن'}
              >
                <ThemeGeminiIcon size={15} color={theme === 'dark' ? '#F59E0B' : 'var(--text-primary)'} />
                <span>{theme === 'dark' ? (isAr ? 'فاتح' : 'Light') : (isAr ? 'داكن' : 'Dark')}</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Floating Tooltip Portal when Collapsed on Hover (Fixed Viewport Docking - Zero Layout Overflow) */}
      {isCollapsed && hoveredTooltip && (
        <div
          style={{
            position: 'fixed',
            top: `${hoveredTooltip.top}px`,
            [isAr ? 'right' : 'left']: '78px',
            transform: 'translateY(-50%)',
            background: 'var(--bg-modal)',
            border: '1px solid var(--border-medium)',
            borderRadius: '8px',
            padding: '0.55rem 0.85rem',
            boxShadow: '0 8px 28px rgba(0, 0, 0, 0.45)',
            zIndex: 9999,
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.15rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--accent-teal)', fontWeight: 800 }}>
              [{hoveredTooltip.item.code}]
            </span>
            <strong style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>
              {isAr ? hoveredTooltip.item.label_ar : hoveredTooltip.item.label_en}
            </strong>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            {isAr ? hoveredTooltip.item.desc_ar : hoveredTooltip.item.desc_en}
          </div>
        </div>
      )}

      {/* Global System Settings Modal */}
      <SystemSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </>
  );
}
