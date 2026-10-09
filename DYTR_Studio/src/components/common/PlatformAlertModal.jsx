// ==============================================================================
// DyTrStudio Platform Alert Modal
// Custom Academic Glassmorphic Alert System - Zero Browser Popups Policy
// Aligned with Studio Design System (Dark/Light tokens, RTL/LTR, Lucide Icons)
// ==============================================================================

import React, { useEffect, useState } from 'react';
import { AlertCircle, AlertTriangle, Info, CheckCircle2, X } from 'lucide-react';
import { useStudioStore } from '../../store/studioStore';

/**
 * Global trigger function to display a studio-native alert from anywhere
 */
export function showStudioAlert(title, message, type = 'error') {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('studio-alert', {
        detail: { title, message, type, timestamp: Date.now() },
      })
    );
  }
}

export default function PlatformAlertModal({ alert: controlledAlert, onClose: controlledOnClose }) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const [activeAlert, setActiveAlert] = useState(null);

  // Sync with controlled props if passed
  useEffect(() => {
    if (controlledAlert !== undefined) {
      setActiveAlert(controlledAlert);
    }
  }, [controlledAlert]);

  // Global event listener + window.alert interception
  useEffect(() => {
    // 1. Listen for custom 'studio-alert' events
    const handleStudioAlertEvent = (e) => {
      if (e.detail) {
        setActiveAlert(e.detail);
      }
    };
    window.addEventListener('studio-alert', handleStudioAlertEvent);

    // 2. Intercept standard browser window.alert calls
    const originalAlert = window.alert;
    window.alert = (msg) => {
      setActiveAlert({
        title: isAr ? 'تنبيه المنصة الأكاديمية' : 'DYTR Studio Alert',
        message: typeof msg === 'object' ? JSON.stringify(msg, null, 2) : String(msg),
        type: 'error',
      });
    };

    return () => {
      window.removeEventListener('studio-alert', handleStudioAlertEvent);
      window.alert = originalAlert;
    };
  }, [isAr]);

  // Keyboard accessibility: Escape to dismiss
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!activeAlert) return;
      if (e.key === 'Escape' || e.key === 'Enter') {
        handleClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeAlert]);

  const handleClose = () => {
    setActiveAlert(null);
    if (controlledOnClose) controlledOnClose();
  };

  if (!activeAlert) return null;

  const type = activeAlert.type || 'error';

  // Visual configuration by alert severity
  const typeConfig = {
    error: {
      color: 'var(--accent-rose, #ef4444)',
      bgBadge: 'rgba(239, 68, 68, 0.15)',
      borderColor: 'rgba(239, 68, 68, 0.35)',
      icon: <AlertCircle size={24} strokeWidth={2.3} />,
      defaultTitle: isAr ? 'خطأ في عملية التحميل أو المعالجة' : 'Operation Diagnostic Alert',
    },
    warning: {
      color: 'var(--accent-amber, #f59e0b)',
      bgBadge: 'rgba(245, 158, 11, 0.15)',
      borderColor: 'rgba(245, 158, 11, 0.35)',
      icon: <AlertTriangle size={24} strokeWidth={2.3} />,
      defaultTitle: isAr ? 'تنبيه هام من المنصة' : 'System Notice',
    },
    info: {
      color: 'var(--accent-cyan, #06b6d4)',
      bgBadge: 'rgba(6, 182, 212, 0.15)',
      borderColor: 'rgba(6, 182, 212, 0.35)',
      icon: <Info size={24} strokeWidth={2.3} />,
      defaultTitle: isAr ? 'إشعار توضيحي' : 'Information',
    },
    success: {
      color: 'var(--accent-teal, #10b981)',
      bgBadge: 'rgba(16, 185, 129, 0.15)',
      borderColor: 'rgba(16, 185, 129, 0.35)',
      icon: <CheckCircle2 size={24} strokeWidth={2.3} />,
      defaultTitle: isAr ? 'اكتملت العملية بنجاح' : 'Success',
    },
  };

  const currentCfg = typeConfig[type] || typeConfig.error;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.72)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100000,
        padding: '1.25rem',
        animation: 'studioAlertFadeIn 0.18s ease-out forwards',
      }}
      onClick={handleClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: `1px solid ${currentCfg.borderColor}`,
          borderRadius: 'var(--radius-lg, 14px)',
          padding: '1.75rem',
          maxWidth: '520px',
          width: '100%',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.55), 0 0 0 1px var(--border-medium)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          direction: isAr ? 'rtl' : 'ltr',
          animation: 'studioAlertScaleUp 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background: currentCfg.bgBadge,
                color: currentCfg.color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                border: `1px solid ${currentCfg.borderColor}`,
              }}
            >
              {currentCfg.icon}
            </div>
            <div>
              <h4 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {activeAlert.title || currentCfg.defaultTitle}
              </h4>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)', display: 'block', marginTop: '2px' }}>
                {isAr ? 'نظام تشخيص وتنبيهات استوديو المحولات DYTR Studio' : 'DYTR Studio Diagnostic Alert'}
              </span>
            </div>
          </div>

          <button
            onClick={handleClose}
            aria-label="Close"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '0.4rem',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background 0.15s ease, color 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--bg-tertiary)';
              e.currentTarget.style.color = 'var(--text-primary)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = 'var(--text-muted)';
            }}
          >
            <X size={19} />
          </button>
        </div>

        {/* Modal Body / Message */}
        <div
          style={{
            background: 'var(--bg-tertiary)',
            padding: '1rem 1.15rem',
            borderRadius: 'var(--radius-md, 8px)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            fontSize: '0.88rem',
            lineHeight: 1.65,
            wordBreak: 'break-word',
          }}
        >
          {activeAlert.message}
        </div>

        {/* Modal Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.65rem' }}>
          <button
            onClick={handleClose}
            className="btn btn-primary btn-sm"
            style={{
              padding: '0.55rem 1.65rem',
              fontWeight: 800,
              fontSize: '0.86rem',
              borderRadius: 'var(--radius-sm, 6px)',
            }}
          >
            {isAr ? 'حسناً، فهمت' : 'Dismiss'}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes studioAlertFadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes studioAlertScaleUp {
          from { transform: scale(0.95); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
