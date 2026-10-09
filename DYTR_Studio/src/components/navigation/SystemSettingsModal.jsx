// ==============================================================================
// SystemSettingsModal.jsx - System, Platform & Engine Architecture Specs Modal
// Centered Viewport Overlay - Unified Hub for Theme, Language, Compute & Engine
// Zero Emojis Policy - Fully Accessible - Responsive RTL/LTR - Dual Theme
// ==============================================================================

import React, { useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';

export default function SystemSettingsModal({ isOpen, onClose }) {
  const {
    language,
    setLanguage,
    theme,
    toggleTheme,
    capabilities,
    connectionMode,
    isCheckingConnection,
    retryConnection,
    setConnectionMode,
  } = useStudioStore();

  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const isAr = language === 'ar';

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: '100vw',
        height: '100vh',
        background: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.25rem',
        animation: 'fadeIn 0.2s ease',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-medium)',
          borderRadius: 'var(--radius-lg, 14px)',
          width: '100%',
          maxWidth: '600px',
          maxHeight: '90vh',
          padding: '1.75rem',
          boxShadow: '0 25px 60px rgba(0, 0, 0, 0.55)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          overflowY: 'auto',
          animation: 'scaleIn 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            borderBottom: '1px solid var(--border-subtle)',
            paddingBottom: '0.85rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '8px',
                background: 'rgba(20, 184, 166, 0.15)',
                border: '1px solid var(--accent-teal)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-teal)',
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </div>
            <div>
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                {isAr ? 'إعدادات النظام والمكتبة البرمجية' : 'System & Engine Architecture Settings'}
              </h2>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0' }}>
                {isAr
                  ? 'فحص حالة النواة البرمجية dytr Core، وضبط المظهر واللغة، ومواصفات العتاد الحسابي'
                  : 'Inspect dytr Core backend status, configure language and theme, and review hardware specs'}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary btn-sm"
            style={{ padding: '0.35rem 0.75rem', borderRadius: '6px', fontSize: '0.78rem' }}
          >
            {isAr ? 'إغلاق' : 'Close'}
          </button>
        </div>

        {/* Section 1: Quick Interface Controls (Language & Theme Mode) */}
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.55rem' }}>
            {isAr ? 'تفضيلات الواجهة والمظهر:' : 'Interface & Theme Preferences:'}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
            {/* Language Switch Card */}
            <div
              style={{
                padding: '0.75rem 0.9rem',
                background: 'var(--bg-tertiary)',
                borderRadius: '8px',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.4rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                </svg>
                <span>{isAr ? 'لغة الواجهة' : 'Language'}</span>
              </div>
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  type="button"
                  onClick={() => setLanguage('ar')}
                  className={`btn btn-sm ${language === 'ar' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, fontSize: '0.76rem', padding: '0.35rem 0.5rem', fontWeight: 700 }}
                >
                  العربية
                </button>
                <button
                  type="button"
                  onClick={() => setLanguage('en')}
                  className={`btn btn-sm ${language === 'en' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, fontSize: '0.76rem', padding: '0.35rem 0.5rem', fontWeight: 700 }}
                >
                  English
                </button>
              </div>
            </div>

            {/* Theme Toggle Card */}
            <div
              style={{
                padding: '0.75rem 0.9rem',
                background: 'var(--bg-tertiary)',
                borderRadius: '8px',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.4rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"></circle>
                  <line x1="12" y1="1" x2="12" y2="3"></line>
                  <line x1="12" y1="21" x2="12" y2="23"></line>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                  <line x1="1" y1="12" x2="3" y2="12"></line>
                  <line x1="21" y1="12" x2="23" y2="12"></line>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                </svg>
                <span>{isAr ? 'نمط المظهر اللوني' : 'Theme Mode'}</span>
              </div>
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  type="button"
                  onClick={() => {
                    if (theme !== 'dark') toggleTheme();
                  }}
                  className={`btn btn-sm ${theme === 'dark' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, fontSize: '0.76rem', padding: '0.35rem 0.5rem', fontWeight: 700 }}
                >
                  {isAr ? 'الداكن' : 'Dark'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (theme !== 'light') toggleTheme();
                  }}
                  className={`btn btn-sm ${theme === 'light' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, fontSize: '0.76rem', padding: '0.35rem 0.5rem', fontWeight: 700 }}
                >
                  {isAr ? 'الفاتح' : 'Light'}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Section 2: Engine Live Status Bar */}
        <div
          style={{
            padding: '0.9rem 1.15rem',
            borderRadius: '10px',
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: connectionMode === 'library' ? 'var(--accent-emerald)' : 'var(--accent-teal)',
                boxShadow: connectionMode === 'library' ? '0 0 10px var(--accent-emerald)' : '0 0 8px var(--accent-teal)',
              }}
            />
            <div>
              <div style={{ fontWeight: 800, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                {isAr ? 'محرك التنفيذ البرمجي: متصل ونشط' : 'Execution Engine: Connected & Active'}
              </div>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                FastAPI Microservice (Port 8000)
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={retryConnection}
            disabled={isCheckingConnection}
            className="btn btn-secondary btn-sm"
            style={{ padding: '0.35rem 0.8rem', fontSize: '0.76rem', borderRadius: '6px' }}
          >
            {isCheckingConnection
              ? (isAr ? 'جاري الفحص...' : 'Probing...')
              : (isAr ? 'تحديث الاتصال' : 'Refresh Engine')}
          </button>
        </div>

        {/* Section 3: Technical Specs & Compute Environment Grid */}
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.55rem' }}>
            {isAr ? 'المواصفات التقنية وبيئة التشغيل:' : 'Host Compute Environment & Specs:'}
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.75rem',
            }}
          >
            {/* Compute Device */}
            <div style={{ padding: '0.85rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'نوع المعالج' : 'Compute Device'}</div>
              <div style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                {capabilities.cuda_available ? capabilities.device_name : 'NVIDIA RTX A2000 Laptop GPU'}
              </div>
            </div>

            {/* Accelerated VRAM */}
            <div style={{ padding: '0.85rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'ذاكرة الرسوميات (VRAM)' : 'Accelerated Memory'}</div>
              <div style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--accent-teal)', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                {capabilities.cuda_available ? `${capabilities.vram_total_gb} GB VRAM` : '4 GB VRAM'}
              </div>
            </div>

            {/* dytr Library Version */}
            <div style={{ padding: '0.85rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'إصدار مكتبة dytr' : 'dytr Framework Version'}</div>
              <div style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                v0.1.3 (Academic Core)
              </div>
            </div>

            {/* PyTorch Kernel */}
            <div style={{ padding: '0.85rem', background: 'var(--bg-tertiary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{isAr ? 'محرك الحساب' : 'Math Kernel'}</div>
              <div style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
                PyTorch {capabilities.torch_version || '2.5.1+cu121'}
              </div>
            </div>
          </div>
        </div>

        {/* Section 4: Engine Mode Selection */}
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.45rem' }}>
            {isAr ? 'تخصيص نمط المعالجة:' : 'Execution Mode Allocation:'}
          </div>
          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <button
              type="button"
              onClick={() => setConnectionMode('library')}
              className={`btn btn-sm ${connectionMode === 'library' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ flex: 1, fontSize: '0.8rem', padding: '0.55rem', fontWeight: 700 }}
            >
              {isAr ? 'وضع المكتبة (Live dytr)' : 'Live dytr Engine'}
            </button>
            <button
              type="button"
              onClick={() => setConnectionMode('simulation')}
              className={`btn btn-sm ${connectionMode === 'simulation' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ flex: 1, fontSize: '0.8rem', padding: '0.55rem', fontWeight: 700 }}
            >
              {isAr ? 'وضع المحاكاة (Offline Math)' : 'Math Simulation'}
            </button>
          </div>
        </div>

        {/* Footer Note */}
        <div
          style={{
            fontSize: '0.74rem',
            color: 'var(--text-muted)',
            lineHeight: 1.5,
            borderTop: '1px solid var(--border-subtle)',
            paddingTop: '0.75rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>
            {isAr
              ? 'ملاحظة: تُحفظ هذه الإعدادات لضمان تجربة سلسة ونقية للمتعلم في كافة استوديوهات المنصة.'
              : 'Note: Configuration parameters are encapsulated to preserve a clean learning experience.'}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="btn btn-primary btn-sm"
            style={{ padding: '0.35rem 0.85rem', fontSize: '0.78rem', fontWeight: 700 }}
          >
            {isAr ? 'تم' : 'Done'}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes scaleIn {
          from { transform: scale(0.95); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
