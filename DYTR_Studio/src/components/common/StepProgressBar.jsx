// ==============================================================================
// StepProgressBar - Linear Step Progress Indicator
// Shows numbered steps with active/complete/pending states.
// Zero Emojis Policy - Bilingual RTL/LTR
// ==============================================================================

import React from 'react';

/**
 * StepProgressBar renders a horizontal progress bar with numbered steps.
 *
 * @param {number} currentStep - The currently active step (1-indexed)
 * @param {number} totalSteps - Total number of steps
 * @param {string[]} [labels] - Optional labels for each step
 * @param {function} [onStepClick] - Optional callback when a step is clicked
 * @param {string} [language='ar'] - Language for RTL support
 */
export default function StepProgressBar({
  currentStep,
  totalSteps,
  labels = [],
  onStepClick,
  language = 'ar',
}) {
  const steps = Array.from({ length: totalSteps }, (_, i) => i + 1);
  const progressPercent = ((currentStep - 1) / Math.max(1, totalSteps - 1)) * 100;

  return (
    <div style={{ width: '100%', marginBottom: '2rem' }}>
      {/* Progress track with fill */}
      <div
        style={{
          position: 'relative',
          height: '4px',
          background: 'var(--bg-tertiary)',
          borderRadius: '2px',
          marginBottom: '0.75rem',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            [language === 'ar' ? 'right' : 'left']: 0,
            height: '100%',
            width: `${progressPercent}%`,
            background: 'var(--accent-teal)',
            borderRadius: '2px',
            transition: 'width 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
          }}
        />
      </div>

      {/* Step indicators */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
        }}
      >
        {steps.map((stepNum) => {
          const isComplete = stepNum < currentStep;
          const isActive = stepNum === currentStep;
          const isPending = stepNum > currentStep;
          const label = labels[stepNum - 1] || '';

          return (
            <div
              key={stepNum}
              onClick={() => onStepClick && onStepClick(stepNum)}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.35rem',
                cursor: onStepClick ? 'pointer' : 'default',
                flex: '0 1 auto',
                minWidth: '32px',
              }}
            >
              {/* Circle */}
              <div
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono)',
                  transition: 'all 0.3s ease',
                  background: isActive
                    ? 'var(--accent-teal)'
                    : isComplete
                    ? 'var(--accent-teal)'
                    : 'var(--bg-surface)',
                  color: isActive || isComplete ? '#fff' : 'var(--text-muted)',
                  border: isPending
                    ? '2px solid var(--border-medium)'
                    : '2px solid var(--accent-teal)',
                  opacity: isComplete ? 0.7 : 1,
                }}
              >
                {isComplete ? (
                  // Checkmark SVG for completed steps
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path
                      d="M2 6L5 9L10 3"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : (
                  stepNum
                )}
              </div>

              {/* Label */}
              {label && (
                <span
                  style={{
                    fontSize: '0.7rem',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: isActive ? 600 : 400,
                    textAlign: 'center',
                    maxWidth: '80px',
                    lineHeight: 1.3,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {label}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
