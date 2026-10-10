// ==============================================================================
// SectionCard - Reusable Content Section Container
// Consistent card styling across all pages with title, subtitle, and content.
// Zero Emojis Policy - Bilingual RTL/LTR
// ==============================================================================

import React from 'react';
import TermTooltip from './TermTooltip';

/**
 * SectionCard renders a styled container for content sections.
 *
 * @param {string} title - Section heading
 * @param {string} [subtitle] - Optional descriptive text below the title
 * @param {string} [tooltipText] - If provided, shows a 3-layer [?] tooltip next to the title
 * @param {string} [termKey] - Optional glossary key for the 3-layer tooltip
 * @param {string} [sectionCode] - Optional numeric code like "01", "02"
 * @param {React.ReactNode} children - Section content
 * @param {object} [style] - Additional style overrides
 * @param {string} [variant] - 'default' | 'elevated' | 'outlined'
 */
export default function SectionCard({
  title,
  subtitle,
  tooltipText,
  termKey,
  sectionCode,
  children,
  style = {},
  variant = 'default',
}) {
  const variantStyles = {
    default: {
      background: 'var(--bg-secondary)',
      border: '1px solid var(--border-subtle)',
    },
    elevated: {
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-medium)',
      boxShadow: '0 2px 12px rgba(0,0,0,0.1)',
    },
    outlined: {
      background: 'transparent',
      border: '1px solid var(--border-medium)',
    },
  };

  const vStyle = variantStyles[variant] || variantStyles.default;

  return (
    <section
      style={{
        ...vStyle,
        borderRadius: 'var(--radius-lg, 12px)',
        padding: '1.75rem',
        marginBottom: '1.5rem',
        transition: 'border-color 0.2s ease',
        ...style,
      }}
    >
      {/* Header area */}
      {(title || sectionCode) && (
        <div
          style={{
            marginBottom: subtitle ? '0.35rem' : '1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
          }}
        >
          {sectionCode && (
            <span
              style={{
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent-teal)',
                background: 'var(--bg-surface)',
                padding: '0.2rem 0.5rem',
                borderRadius: 'var(--radius-sm, 4px)',
                border: '1px solid var(--border-subtle)',
                fontWeight: 600,
              }}
            >
              {sectionCode}
            </span>
          )}

          <h3
            style={{
              margin: 0,
              fontSize: '1.15rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              lineHeight: 1.3,
            }}
          >
            {title}
          </h3>

          {(termKey || tooltipText) && (
            <TermTooltip termKey={termKey} explanation={tooltipText} title={title} hideLabel={true} size="sm" />
          )}
        </div>
      )}

      {/* Subtitle */}
      {subtitle && (
        <p
          style={{
            margin: '0 0 1.25rem 0',
            fontSize: '0.88rem',
            color: 'var(--text-muted)',
            lineHeight: 1.6,
            paddingInlineStart: sectionCode ? '2.8rem' : '0',
          }}
        >
          {subtitle}
        </p>
      )}

      {/* Content */}
      <div>{children}</div>
    </section>
  );
}
