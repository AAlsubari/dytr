// ==============================================================================
// DyTrStudio 3-Layer Academic Explanation Component
// Layer 1: Instant Hover Card (Title + Tooltip)
// Layer 2: Click Popover with Context & Practical Examples
// Layer 3: Modal Dialog with Mathematical Formulations & APA 7th Citations
// Uses React Portals + Viewport Bounding to Guarantee On-Screen Visibility
// Zero Emojis Policy - High Contrast Dual Theme (Dark / Light)
// ==============================================================================

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { academicGlossary } from '../../lib/data/glossary';
import { useStudioStore } from '../../store/studioStore';
import MathRenderer from './MathRenderer';
import { ArrowRight, HelpCircle, X } from 'lucide-react';

export default function TermTooltip({
  termKey,
  id,
  customLabel,
  explanation,
  title,
  size = 'md',
  hideLabel = false,
  children,
}) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';
  const key = termKey || id;
  let term = key ? academicGlossary[key] : null;

  // Dynamic 3-Layer term generator if not found in dictionary but explanation provided
  if (!term && (explanation || title || customLabel)) {
    const fallbackTitle = title || customLabel || (typeof children === 'string' ? children : key) || (isAr ? 'المفهوم الأكاديمي' : 'Academic Concept');
    const fallbackExpl = explanation || (isAr ? 'توضيح أكاديمي لمفهوم الذكاء الاصطناعي وبناء المحولات.' : 'Academic explanation for AI and Transformer architectures.');
    term = {
      id: key || 'dynamic_term',
      title_ar: fallbackTitle,
      title_en: fallbackTitle,
      term_ar: fallbackTitle,
      term_en: fallbackTitle,
      layer1_tooltip_ar: fallbackExpl,
      layer1_tooltip_en: fallbackExpl,
      layer1_ar: fallbackExpl,
      layer1_en: fallbackExpl,
      layer2_summary_ar: `${fallbackExpl} يُعد هذا المفهوم جزءاً أساسياً من بناء وتدريب وتطوير نماذج المحولات الديناميكية واستدامة المعرفة.`,
      layer2_summary_en: `${fallbackExpl} Fundamental concept in training dynamic transformers and continual representation learning.`,
      layer2_ar: fallbackExpl,
      layer2_en: fallbackExpl,
      example_ar: `تطبيق عملي: توظيف ${fallbackTitle} في معالجة واستنتاج البيانات بدقة عالية.`,
      example_en: `Practical usage: Employing ${fallbackTitle} in neural inference and processing.`,
      layer3_content_ar: `يستند هذا المفهوم إلى المعايير الأكاديمية الراسخة في علوم الحاسوب والتعلم العميق (Goodfellow et al., 2016; Vaswani et al., 2017).`,
      layer3_content_en: `Rooted in established machine learning literature (Goodfellow et al., 2016; Vaswani et al., 2017).`,
      layer3_ar: {
        formula: 'y = f(x; \\theta)',
        meaning_ar: 'الصيغة الرياضية العامة الدالة على المعالجة العصبية المدعومة بالمعاملات القابلة للتحسين.',
        symbols_ar: [
          { symbol: 'x', desc: 'المدخل اللغوي أو الرياضي (Input Vector)' },
          { symbol: 'f', desc: 'الدالة العصبية أو المحول (Neural Transformer Function)' },
          { symbol: '\\theta', desc: 'معاملات وأوزان النموذج (Model Parameters)' },
        ],
      },
      reference_apa: 'Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.',
      citation: 'Goodfellow et al. (2016). MIT Press.',
    };
  }

  const [isHovered, setIsHovered] = useState(false);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  const triggerRef = useRef(null);
  const hoverTimeoutRef = useRef(null);

  const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0 });
  const [hoverPos, setHoverPos] = useState({ top: 0, left: 0 });

  if (!term) {
    if (hideLabel) return null;
    return <span>{children || customLabel || key}</span>;
  }

  const label = customLabel || children || (isAr ? (term.title_ar || term.term_ar) : (term.title_en || term.term_en));
  const tooltipText = isAr ? (term.layer1_ar || term.layer1_tooltip_ar) : (term.layer1_en || term.layer1_tooltip_en);
  const summaryText = isAr ? (term.layer2_ar || term.layer2_summary_ar) : (term.layer2_en || term.layer2_summary_en);
  const exampleText = isAr ? term.example_ar : term.example_en;
  const deepContent = isAr ? term.layer3_content_ar : term.layer3_content_en;
  const formulaText = typeof term.layer3_ar?.formula === 'string' ? term.layer3_ar.formula : term.formula;
  const formulaMeaning = isAr ? (term.formula_meaning_ar || term.layer3_ar?.meaning_ar) : (term.formula_meaning_en || term.layer3_ar?.meaning_en);
  const symbolsList = isAr ? (term.symbols_ar || term.layer3_ar?.symbols_ar) : (term.symbols_en || term.layer3_ar?.symbols_en);
  const citationText = term.citation || term.reference_apa;

  // Calculate viewport-constrained coordinates to prevent popovers from rendering off-screen
  const updatePositions = useCallback(() => {
    if (!triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Hover Card (Layer 1) - Approx 280px wide
    let hWidth = 280;
    let hTop = rect.top - 110;
    if (hTop < 10) hTop = rect.bottom + 8;
    let hLeft = isAr ? rect.right - hWidth : rect.left;
    if (hLeft < 10) hLeft = 10;
    if (hLeft + hWidth > vw - 10) hLeft = vw - hWidth - 10;

    setHoverPos({ top: Math.max(10, hTop), left: hLeft });

    // Popover (Layer 2) - Approx 340px wide x 260px high
    let pWidth = Math.min(340, vw - 20);
    let pTop = rect.bottom + 8;
    if (pTop + 260 > vh - 10) {
      pTop = rect.top - 270;
      if (pTop < 10) pTop = Math.max(10, (vh - 260) / 2);
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

  const iconSize = size === 'sm' ? 18 : 20;

  return (
    <>
      <span
        ref={triggerRef}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.35rem',
          verticalAlign: 'middle',
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {!hideLabel && (
          <span
            style={{
              borderBottom: '1px dotted var(--accent-teal)',
              cursor: 'help',
              color: 'inherit',
              fontWeight: 600,
            }}
            onClick={() => setIsPopoverOpen(!isPopoverOpen)}
          >
            {label}
          </span>
        )}

        {/* Level 2/3 Trigger Button (?) */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setIsHovered(false);
            setIsPopoverOpen(!isPopoverOpen);
          }}
          title={isAr ? 'انقر للشرح الأكاديمي الموسع' : 'Click for academic context'}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: `${iconSize}px`,
            height: `${iconSize}px`,
            borderRadius: '50%',
            background: isPopoverOpen ? 'var(--accent-teal)' : 'var(--bg-elevated)',
            border: `1px solid ${isPopoverOpen ? 'var(--accent-teal)' : 'var(--border-medium)'}`,
            color: isPopoverOpen ? '#ffffff' : 'var(--accent-teal)',
            cursor: 'pointer',
            padding: 0,
            userSelect: 'none',
            boxShadow: 'var(--shadow-sm)',
            transition: 'all 0.15s ease',
          }}
        >
          <HelpCircle size={size === 'sm' ? 12 : 14} />
        </button>
      </span>

      {/* Layer 1: Instant Hover Card via Portal */}
      {isHovered && !isPopoverOpen && !isModalOpen && createPortal(
        <div
          style={{
            position: 'fixed',
            top: `${hoverPos.top}px`,
            left: `${hoverPos.left}px`,
            background: 'var(--bg-secondary)',
            color: 'var(--text-primary)',
            textAlign: isAr ? 'right' : 'left',
            padding: '0.65rem 0.85rem',
            borderRadius: 'var(--radius-md, 8px)',
            border: '1px solid var(--border-medium)',
            boxShadow: 'var(--shadow-lg)',
            width: '280px',
            zIndex: 99999,
            pointerEvents: 'none',
            fontSize: '0.78rem',
            lineHeight: 1.5,
            direction: isAr ? 'rtl' : 'ltr',
          }}
        >
          <span style={{ fontWeight: 800, color: 'var(--accent-teal)', display: 'block', marginBottom: '0.2rem' }}>
            {label}
          </span>
          <span style={{ color: 'var(--text-secondary)', display: 'block' }}>
            {tooltipText}
          </span>
        </div>,
        document.body
      )}

      {/* Layer 2: Popover Dropdown via Portal */}
      {isPopoverOpen && createPortal(
        <>
          <div
            onClick={() => setIsPopoverOpen(false)}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 99998,
              background: 'transparent',
            }}
          />
          <div
            dir={isAr ? 'rtl' : 'ltr'}
            style={{
              position: 'fixed',
              top: `${popoverPos.top}px`,
              left: `${popoverPos.left}px`,
              width: popoverPos.width ? `${popoverPos.width}px` : '340px',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-lg, 12px)',
              boxShadow: 'var(--shadow-lg)',
              padding: '1.15rem',
              zIndex: 99999,
              textAlign: isAr ? 'right' : 'left',
              color: 'var(--text-primary)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
                {label}
              </span>
              <button
                type="button"
                onClick={() => setIsPopoverOpen(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: '0.2rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <X size={14} />
              </button>
            </div>

            <p style={{ fontSize: '0.82rem', lineHeight: 1.55, color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              {summaryText}
            </p>

            {exampleText && (
              <div
                style={{
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius-sm, 6px)',
                  padding: '0.65rem 0.85rem',
                  borderLeft: isAr ? 'none' : '3px solid var(--accent-teal)',
                  borderRight: isAr ? '3px solid var(--accent-teal)' : 'none',
                  marginBottom: '0.85rem',
                }}
              >
                <span style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--accent-teal)', display: 'block', marginBottom: '0.15rem' }}>
                  {isAr ? 'مثال تطبيقي:' : 'Empirical Example:'}
                </span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-primary)', lineHeight: 1.45 }}>
                  {exampleText}
                </span>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.65rem' }}>
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
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  padding: 0,
                  textDecoration: 'underline',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                }}
              >
                <span>{isAr ? 'المقال الأكاديمي الموسع (المستوى 3)' : 'Read Full Academic Reference (Layer 3)'}</span>
                <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
              </button>
            </div>
          </div>
        </>,
        document.body
      )}

      {/* Layer 3: Modal Dialog (Deep Academic Explanation) via Portal */}
      {isModalOpen && createPortal(
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.65)',
            backdropFilter: 'blur(4px)',
            zIndex: 100000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
          }}
          onClick={() => setIsModalOpen(false)}
        >
          <div
            dir={isAr ? 'rtl' : 'ltr'}
            style={{
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-lg, 12px)',
              width: '100%',
              maxWidth: '680px',
              maxHeight: '85vh',
              overflowY: 'auto',
              padding: '1.75rem',
              boxShadow: 'var(--shadow-lg)',
              textAlign: isAr ? 'right' : 'left',
              color: 'var(--text-primary)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '1.25rem',
                borderBottom: '1px solid var(--border-subtle)',
                paddingBottom: '0.85rem',
              }}
            >
              <div>
                <span className="badge badge-teal" style={{ fontSize: '0.7rem', marginBottom: '0.35rem', display: 'inline-block' }}>
                  {isAr ? 'التفسير المعرفي الأكاديمي (المستوى 3)' : 'Epistemological Academic Reference (Layer 3)'}
                </span>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', margin: 0 }}>
                  {label}
                </h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="btn btn-secondary btn-sm"
                style={{
                  fontSize: '0.85rem',
                }}
              >
                {isAr ? 'إغلاق' : 'Close'}
              </button>
            </div>

            {/* Deep Conceptual Paragraphs */}
            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-teal)', marginBottom: '0.5rem' }}>
                {isAr ? 'التأصيل النظري والمفاهيمي:' : 'Theoretical Framework:'}
              </h4>
              <p style={{ fontSize: '0.88rem', lineHeight: 1.7, color: 'var(--text-secondary)', whiteSpace: 'pre-line' }}>
                {deepContent || summaryText}
              </p>
            </div>

            {/* Mathematical Formula & KaTeX Renderer with Symbol Breakdown ("حيث :") */}
            <MathRenderer
              formula={formulaText}
              meaning={formulaMeaning}
              symbols={symbolsList}
              isAr={isAr}
            />

            {/* APA Academic Citation */}
            {citationText && (
              <div
                style={{
                  background: 'var(--bg-tertiary)',
                  border: '1px dashed var(--border-medium)',
                  borderRadius: 'var(--radius-md, 8px)',
                  padding: '0.85rem 1rem',
                }}
              >
                <span style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                  {isAr ? 'المرجع الأكاديمي المعتمد (APA 7th):' : 'Academic Citation (APA 7th):'}
                </span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic', direction: 'ltr', textAlign: 'left', display: 'block' }}>
                  {citationText}
                </span>
              </div>
            )}
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

