// ==============================================================================
// DYTR Studio Official Brand Logo & Icon Component
// Scalable Vector SVG / Transparent Asset Integration
// Dynamically Adapts to Night (Dark) & Day (Light) Modes
// Standards: Academic UI - Zero Emojis - Responsive Design
// ==============================================================================

import React, { useState } from 'react';
import { useStudioStore } from '../../store/studioStore';

import iconDarkPng from '../../assets/dytr-icon-dark.png';
import iconLightPng from '../../assets/dytr-icon-light.png';
import logoDarkPng from '../../assets/dytr-logo-dark.png';
import logoLightPng from '../../assets/dytr-logo-light.png';

/**
 * DytrLogo Component
 * Renders the authentic high-resolution DYTR Studio logo & icon without distortion.
 *
 * @param {('icon'|'full')} [variant='icon'] - Logo display variant
 * @param {number|string} [size=32] - Primary dimension in pixels
 * @param {('dark'|'light'|'auto')} [theme='auto'] - Theme selection
 * @param {boolean} [glow=false] - Subtle cyan/teal ambient glow effect
 * @param {string} [className=''] - Additional CSS classes
 * @param {object} [style={}] - Custom inline styles
 * @param {function} [onClick] - Optional click handler
 */
export default function DytrLogo({
  variant = 'icon',
  size = 32,
  theme = 'auto',
  glow = false,
  className = '',
  style = {},
  onClick,
}) {
  const storeTheme = useStudioStore((state) => state.theme);
  const [isHovered, setIsHovered] = useState(false);

  // Resolve active theme: explicit prop or current store theme (defaults to dark)
  const resolvedTheme = theme === 'auto' ? (storeTheme || 'dark') : theme;
  const isLight = resolvedTheme === 'light';

  // Determine authentic asset sources directly from high-resolution PNGs
  let src = '';
  let aspectRatio = '1 / 1';

  if (variant === 'full') {
    aspectRatio = '555 / 330';
    src = isLight ? logoLightPng : logoDarkPng;
  } else {
    aspectRatio = '235 / 225';
    src = isLight ? iconLightPng : iconDarkPng;
  }

  // Calculate width and height based on size prop and aspect ratio
  const numSize = typeof size === 'number' ? size : parseInt(size, 10) || 32;
  const width = variant === 'full' ? Math.round(numSize * (555 / 330)) : numSize;
  const height = numSize;

  const glowFilter = glow || isHovered
    ? isLight
      ? 'drop-shadow(0 2px 8px rgba(14, 165, 233, 0.25))'
      : 'drop-shadow(0 0 12px rgba(20, 241, 217, 0.35))'
    : 'none';

  return (
    <div
      className={`dytr-logo-container ${className}`}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        lineHeight: 0,
        position: 'relative',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'transform 0.2s ease, filter 0.2s ease',
        transform: isHovered && onClick ? 'scale(1.04)' : 'scale(1)',
        filter: glowFilter,
        ...style,
      }}
    >
      <img
        src={src}
        alt="DYTR Studio"
        width={width}
        height={height}
        style={{
          width: `${width}px`,
          height: `${height}px`,
          objectFit: 'contain',
          aspectRatio,
          display: 'block',
          userSelect: 'none',
          pointerEvents: 'none',
          imageRendering: 'auto',
        }}
      />
    </div>
  );
}

// Named exports for convenient modular usage
export { DytrLogo };
