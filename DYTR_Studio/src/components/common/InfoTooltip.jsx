// ==============================================================================
// InfoTooltip - 3-Layer Progressive Disclosure Academic Explanation Component
// Level 1: Instant Hover Card (Quick Definition)
// Level 2: Click Popover (Why use it? Practical Impact & Rules)
// Level 3: Deep-Dive Modal (Mathematical Formulation & Best Practices)
// Full Dark Mode / Light Mode Compatibility using CSS Custom Properties
// Zero Emojis Policy - Bilingual RTL/LTR - React Portals for Zero-Clipping
// ==============================================================================

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useStudioStore } from '../../store/studioStore';
import { X, ArrowRight } from 'lucide-react';

export default function InfoTooltip({
  term,
  explanation,
  title,
  termKey,
  what,
  why,
  how,
  formula,
  size = 'sm',
  children,
}) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  const [isHovered, setIsHovered] = useState(false);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const triggerRef = useRef(null);
  const hoverTimeoutRef = useRef(null);
  const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0, width: 340 });
  const [hoverPos, setHoverPos] = useState({ top: 0, left: 0, width: 290 });

  // Compute 3-Layer structured content
  const displayTitle = title || (typeof term === 'string' ? term : null) || (isAr ? 'المفهوم الأكاديمي' : 'Academic Concept');
  
  // Level 1: What is it? (Definition)
  const level1Text = what || (typeof explanation === 'string' ? explanation : (isAr ? 'توضيح مفهوم المعمارية والتعلم العصبي.' : 'Neural architectural parameter explanation.'));
  
  // Level 2: Why use it? (Significance)
  const level2Text = why || (isAr 
    ? 'يساعد ضبط هذا المعامل في تحقيق الاستقرار الرياضي وتفادي النسيان الكارثي أو فرط التخصيص أثناء المعالجة.'
    : 'Essential parameter tuning to ensure mathematical stability and prevent catastrophic drift or overfitting.');
  
  // Level 3: How to choose / Formula / Best practices
  const level3Text = how || formula || (isAr
    ? 'اختر القيمة الافتراضية الموصى بها، أو ارفعها تدريجياً مع مراقبة منحنيات الدقة والخسارة (Loss / Accuracy).'
    : 'Use recommended default scale, or tune progressively while inspecting validation loss and accuracy curves.');

  // Viewport-constrained positioning
  const updatePositions = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Hover Card (Layer 1)
    const hWidth = Math.min(300, vw - 20);
    let hTop = rect.top - 120;
    if (hTop < 10) hTop = rect.bottom + 8;
    let hLeft = isAr ? rect.right - hWidth : rect.left;
    if (hLeft < 10) hLeft = 10;
    if (hLeft + hWidth > vw - 10) hLeft = vw - hWidth - 10;
    setHoverPos({ top: Math.max(10, hTop), left: hLeft, width: hWidth });

    // Popover (Layer 2)
    const pWidth = Math.min(360, vw - 24);
    let pTop = rect.bottom + 8;
    if (pTop + 280 > vh - 10) {
      pTop = rect.top - 290;
      if (pTop < 10) pTop = Math.max(10, (vh - 280) / 2);
    }
    let pLeft = isAr ? rect.right - pWidth : rect.left;
    if (pLeft < 10) pLeft = 10;
    if (pLeft + pWidth > vw - 10) pLeft = vw - pWidth - 10;
    setPopoverPos({ top: pTop, left: pLeft, width: pWidth });
  }, [isAr]);

  useEffect(() => {
    if (isPopoverOpen || isHovered) {
      updatePositions();
      window.addEventListener('scroll', updatePositions, true);
      window.addEventListener('resize', updatePositions);
      return () => {
        window.removeEventListener('scroll', updatePositions, true);
        window.removeEventListener('resize', updatePositions);
      };
    }
  }, [isPopoverOpen, isHovered, updatePositions]);

  const handleMouseEnter = () => {
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovered(true);
    }, 150);
  };

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    setIsHovered(false);
  };

  const iconSize = size === 'md' ? 22 : 18;
  const fontSize = size === 'md' ? '0.78rem' : '0.68rem';

  return (
    <>
      <span
        ref={triggerRef}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.35rem',
          verticalAlign: 'middle',
          cursor: 'pointer',
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {children || (term && <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{term}</span>)}

        {/* Question Mark Trigger Circle (?) */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setIsHovered(false);
            setIsPopoverOpen(!isPopoverOpen);
          }}
          title={isAr ? 'انقر للشرح ثلاثي المستويات (ما هو؟ لماذا؟ وكيف تختار؟)' : 'Click for 3-Level Explanation'}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: `${iconSize}px`,
            height: `${iconSize}px`,
            borderRadius: '50%',
            background: isPopoverOpen ? 'var(--accent-teal)' : 'var(--bg-elevated)',
            border: `1.5px solid ${isPopoverOpen ? 'var(--accent-teal)' : 'var(--border-medium)'}`,
            color: isPopoverOpen ? '#ffffff' : 'var(--accent-teal)',
            fontSize: fontSize,
            fontWeight: 800,
            cursor: 'pointer',
            padding: 0,
            lineHeight: 1,
            boxShadow: 'var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.1))',
            transition: 'all 0.15s ease',
            flexShrink: 0,
          }}
        >
          ?
        </button>
      </span>

      {/* LAYER 1: Quick Hover Card (Portal) */}
      {isHovered && !isPopoverOpen && !isModalOpen && createPortal(
        <div
          dir={isAr ? 'rtl' : 'ltr'}
          style={{
            position: 'fixed',
            top: `${hoverPos.top}px`,
            left: `${hoverPos.left}px`,
            width: `${hoverPos.width}px`,
            zIndex: 99999,
            background: 'var(--bg-elevated)',
            color: 'var(--text-primary)',
            borderRadius: '8px',
            border: '1px solid var(--border-medium)',
            boxShadow: 'var(--shadow-md)',
            padding: '0.85rem 1rem',
            fontSize: '0.8rem',
            lineHeight: 1.5,
            pointerEvents: 'none',
            animation: 'fadeIn 0.15s ease-out',
            opacity: 1,
          }}
        >
          {displayTitle && (
            <div style={{ fontWeight: 800, color: 'var(--accent-teal)', marginBottom: '0.35rem', fontSize: '0.84rem' }}>
              {displayTitle}
            </div>
          )}
          <div style={{ color: 'var(--text-primary)', fontSize: '0.78rem' }}>
            {level1Text}
          </div>
          <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.35rem' }}>
            {isAr ? 'انقر على (?) لعرض الشرح ثلاثي المستويات والمعادلات' : 'Click (?) for full 3-level breakdown & rules'}
          </div>
        </div>,
        document.body
      )}

      {/* LAYER 2: Contextual 3-Level Popover (Portal) */}
      {isPopoverOpen && !isModalOpen && createPortal(
        <div
          dir={isAr ? 'rtl' : 'ltr'}
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'fixed',
            top: `${popoverPos.top}px`,
            left: `${popoverPos.left}px`,
            width: `${popoverPos.width}px`,
            zIndex: 99999,
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            borderRadius: '10px',
            border: '1.5px solid var(--accent-teal)',
            boxShadow: 'var(--shadow-lg)',
            padding: '1.15rem',
            animation: 'popIn 0.2s ease-out',
            opacity: 1,
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span className="badge badge-teal" style={{ fontSize: '0.65rem' }}>
                {isAr ? 'شرح 3 مستويات' : '3-Level Concept'}
              </span>
              <span style={{ fontWeight: 800, fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                {displayTitle}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setIsPopoverOpen(false)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: '1.1rem',
                cursor: 'pointer',
                padding: '0 0.25rem',
              }}
            >
              <X size={15} />
            </button>
          </div>

          {/* 3 Structured Tiers */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {/* Level 1: Definition */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem 0.8rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 700, fontSize: '0.74rem', color: 'var(--accent-teal)', marginBottom: '0.2rem' }}>
                {isAr ? '1. ما هو؟ (التعريف والمفهوم):' : '1. What is it? (Core Concept):'}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-primary)', lineHeight: 1.45 }}>
                {level1Text}
              </div>
            </div>

            {/* Level 2: Importance */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem 0.8rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 700, fontSize: '0.74rem', color: 'var(--accent-amber)', marginBottom: '0.2rem' }}>
                {isAr ? '2. لماذا نستخدمه؟ (الأهمية والتأثير):' : '2. Why use it? (Significance):'}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                {level2Text}
              </div>
            </div>

            {/* Level 3: How to Choose */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem 0.8rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 700, fontSize: '0.74rem', color: 'var(--accent-emerald)', marginBottom: '0.2rem' }}>
                {isAr ? '3. كيف تختار؟ (القيمة الموصى بها والمعادلة):' : '3. How to choose? (Best Practice & Formula):'}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-primary)', lineHeight: 1.45, fontFamily: 'var(--font-mono)' }}>
                {level3Text}
              </div>
            </div>
          </div>

          {/* Footer Actions */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.85rem', paddingTop: '0.6rem', borderTop: '1px solid var(--border-subtle)' }}>
            <button
              type="button"
              onClick={() => {
                setIsPopoverOpen(false);
                setIsModalOpen(true);
              }}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--accent-teal)',
                fontSize: '0.74rem',
                fontWeight: 700,
                cursor: 'pointer',
                padding: 0,
                textDecoration: 'underline',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
            >
              <span>{isAr ? 'فتح التحليل المعمق الكامل' : 'Open Deep-Dive Modal'}</span>
              <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
            </button>
            <button
              type="button"
              onClick={() => setIsPopoverOpen(false)}
              className="btn btn-secondary btn-sm"
              style={{ padding: '0.25rem 0.65rem', fontSize: '0.74rem' }}
            >
              {isAr ? 'إغلاق' : 'Close'}
            </button>
          </div>
        </div>,
        document.body
      )}

      {/* LAYER 3: Full Deep-Dive Modal (Portal) */}
      {isModalOpen && createPortal(
        <div
          dir={isAr ? 'rtl' : 'ltr'}
          onClick={() => setIsModalOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            background: 'rgba(0, 0, 0, 0.65)',
            backdropFilter: 'blur(4px)',
            zIndex: 999999,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '1rem',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: 'var(--bg-secondary)',
              color: 'var(--text-primary)',
              borderRadius: '12px',
              border: '2px solid var(--accent-teal)',
              width: '100%',
              maxWidth: '540px',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '1.5rem',
              boxShadow: 'var(--shadow-lg)',
              opacity: 1,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.75rem' }}>
              <div>
                <span className="badge badge-teal" style={{ fontSize: '0.68rem', marginBottom: '0.25rem' }}>
                  {isAr ? 'التحليل الأكاديمي المتقدم (المستوى 3)' : 'Academic Deep-Dive (Level 3)'}
                </span>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {displayTitle}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '1.35rem',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-teal)', marginBottom: '0.35rem' }}>
                  {isAr ? 'التعريف والأساس العلمي:' : 'Theoretical Foundation:'}
                </h4>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
                  {level1Text}
                </p>
              </div>

              <div>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-amber)', marginBottom: '0.35rem' }}>
                  {isAr ? 'التأثير المعماري وحل المشاكل:' : 'Architectural Impact & Problem Formulation:'}
                </h4>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
                  {level2Text}
                </p>
              </div>

              <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-emerald)', marginBottom: '0.35rem' }}>
                  {isAr ? 'الصيغة الرياضية والقيمة التشغيلية:' : 'Mathematical Specification & Setting:'}
                </h4>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                  {level3Text}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="btn btn-primary"
                style={{ padding: '0.5rem 1.25rem', fontSize: '0.85rem' }}
              >
                {isAr ? 'إغلاق' : 'Close'}
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
