// ==============================================================================
// DyTrStudio High-Rigor Mathematical Formula & Algorithm Renderer (KaTeX Engine)
// Renders LaTeX Equations, Formula Meanings, and Symbol Breakdown ("حيث :")
// Zero Emojis Policy - High Contrast Dual Theme (Dark / Light)
// ==============================================================================

import React, { useEffect, useRef } from 'react';

// Helper component to render inline math symbols via KaTeX
function KatexSymbol({ math }) {
  const spanRef = useRef(null);

  useEffect(() => {
    if (math && spanRef.current) {
      if (window.katex) {
        try {
          window.katex.render(math, spanRef.current, {
            displayMode: false,
            throwOnError: false,
          });
        } catch (err) {
          spanRef.current.innerText = math;
        }
      } else {
        spanRef.current.innerText = math;
      }
    }
  }, [math]);

  return <span ref={spanRef} style={{ display: 'inline-block', verticalAlign: 'middle' }} />;
}

export default function MathRenderer({ formula, meaning, symbols = [], isAr = true }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (formula && containerRef.current) {
      if (window.katex) {
        try {
          window.katex.render(formula, containerRef.current, {
            displayMode: true,
            throwOnError: false,
          });
        } catch (err) {
          containerRef.current.innerText = formula;
        }
      } else {
        containerRef.current.innerText = formula;
      }
    }
  }, [formula]);

  if (!formula) return null;

  return (
    <div
      style={{
        background: 'var(--bg-tertiary)',
        border: '1px solid var(--border-medium)',
        borderRadius: 'var(--radius-md, 8px)',
        padding: '1.15rem 1.25rem',
        marginBottom: '1.25rem',
        textAlign: isAr ? 'right' : 'left',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '0.65rem' }}>
        <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-teal)' }}>
          {isAr ? 'الصيغة الرياضية والخوارزمية:' : 'Mathematical Formulation & Algorithm:'}
        </span>
      </div>

      {/* KaTeX Main Equation Output Box */}
      <div
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm, 6px)',
          padding: '0.85rem 1rem',
          marginBottom: '0.85rem',
          overflowX: 'auto',
          direction: 'ltr',
          textAlign: 'center',
          color: 'var(--accent-teal)',
          fontSize: '1.15rem',
        }}
      >
        <div ref={containerRef}>{formula}</div>
      </div>

      {/* General Meaning of Formula / Algorithm */}
      {meaning && (
        <div style={{ marginBottom: '0.85rem', fontSize: '0.84rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
          <span style={{ fontWeight: 800, color: 'var(--accent-teal)', display: 'block', marginBottom: '0.2rem' }}>
            {isAr ? 'تفسير المعنى العام للصيغة والخوارزمية:' : 'General Formulation Meaning:'}
          </span>
          <p style={{ margin: 0, color: 'var(--text-secondary)' }}>{meaning}</p>
        </div>
      )}

      {/* Structured Symbol Breakdown ("حيث :") */}
      {symbols && symbols.length > 0 && (
        <div style={{ borderTop: '1px dashed var(--border-subtle)', paddingTop: '0.75rem' }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-teal)', display: 'block', marginBottom: '0.55rem' }}>
            {isAr ? 'حيث :' : 'Where :'}
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '0.55rem' }}>
            {symbols.map((item, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.65rem',
                  background: 'var(--bg-secondary)',
                  padding: '0.45rem 0.75rem',
                  borderRadius: 'var(--radius-sm, 6px)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.8rem',
                }}
              >
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minWidth: '32px',
                    minHeight: '28px',
                    background: 'var(--accent-teal-glow)',
                    color: 'var(--accent-teal)',
                    borderRadius: '4px',
                    padding: '0.15rem 0.4rem',
                    fontSize: '0.95rem',
                    direction: 'ltr',
                    textAlign: 'center',
                  }}
                >
                  <KatexSymbol math={item.symbol} />
                </span>
                <span style={{ color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                  {item.desc}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

