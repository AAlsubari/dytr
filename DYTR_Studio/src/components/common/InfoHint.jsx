// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// InfoHint Educational Tooltip Component - Industrial Swiss Design & Zero Emojis Policy

import React, { useState, useRef, useEffect } from 'react';

/**
 * InfoHint Component: Renders a sleek circular '?' question mark button.
 * Clicking or hovering opens an interactive educational popover with title, analogy, and technical breakdown.
 *
 * @param {Object} props
 * @param {string} props.title - Title of the concept being explained
 * @param {string} props.content - Deep educational explanation
 * @param {string} [props.analogy] - Simplified human analogy
 * @param {string} [props.formula] - Mathematical formula or technical note
 */
export default function InfoHint({ title, content, analogy, formula }) {
  const [isOpen, setIsOpen] = useState(false);
  const hintRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (hintRef.current && !hintRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <span
      ref={hintRef}
      style={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        marginInlineStart: '6px',
        verticalAlign: 'middle',
      }}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        aria-label={title || 'معلومات توضيحية'}
        style={{
          width: '18px',
          height: '18px',
          borderRadius: '50%',
          border: '1px solid var(--border-strong)',
          background: isOpen ? 'var(--accent-steel)' : 'var(--bg-tertiary)',
          color: isOpen ? '#FFFFFF' : 'var(--text-secondary)',
          fontSize: '11px',
          fontWeight: '700',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          transition: 'all 0.15s ease',
        }}
      >
        ?
      </button>

      {isOpen && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 8px)',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '320px',
            maxWidth: '90vw',
            background: 'var(--bg-surface-solid)',
            border: '1px solid var(--accent-steel)',
            borderRadius: '10px',
            padding: '1rem',
            boxShadow: 'var(--shadow-lg)',
            zIndex: 1000,
            fontSize: '0.82rem',
            lineHeight: '1.5',
            color: 'var(--text-primary)',
            textAlign: 'start',
            backdropFilter: 'blur(16px)',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '4px' }}>
            <strong style={{ color: 'var(--accent-steel)', fontSize: '0.88rem' }}>
              {title}
            </strong>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '700',
                padding: '0 4px',
              }}
            >
              ×
            </button>
          </div>

          {/* Body Content */}
          <p style={{ color: 'var(--text-secondary)', marginBottom: analogy || formula ? '8px' : '0' }}>
            {content}
          </p>

          {/* Plain Human Analogy */}
          {analogy && (
            <div
              style={{
                background: 'var(--accent-amber-bg)',
                borderInlineStart: '3px solid var(--accent-amber)',
                padding: '0.4rem 0.6rem',
                borderRadius: '4px',
                marginBottom: formula ? '6px' : '0',
                fontSize: '0.78rem',
                color: 'var(--text-primary)',
              }}
            >
              <strong style={{ color: 'var(--accent-amber)', display: 'block', marginBottom: '2px' }}>
                تشبيه مبسط:
              </strong>
              {analogy}
            </div>
          )}

          {/* Technical / Mathematical Formula */}
          {formula && (
            <div
              className="mono-font"
              style={{
                background: 'var(--bg-tertiary)',
                padding: '0.35rem 0.55rem',
                borderRadius: '4px',
                fontSize: '0.74rem',
                color: 'var(--accent-pine)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              {formula}
            </div>
          )}
        </div>
      )}
    </span>
  );
}
