// ==============================================================================
// DyTrStudio Animated Batch Tokenizer Component (المُجزئ البصري المتدفق لدفعات النصوص)
// Interactive token-stream animation with Vocab IDs, Positions, and Tensor Slicing
// Zero Emojis Policy - Zero Layout Shift (CLS = 0) - Bilingual RTL/LTR Support
// ==============================================================================

import React, { useState, useEffect, useRef } from 'react';
import TermTooltip from './TermTooltip';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export default function AnimatedBatchTokenizer({ sentence, batchSize = 16, embedDim = 768, isAr = true }) {
  const tokens = (sentence || (isAr ? 'أنصح الجميع بالتعامل معهم، احترافية وأمانة في العمل.' : 'The transformer model provides exceptional convergence stability.')).split(/\s+/).filter(Boolean);
  const [activeIdx, setActiveIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1); // 0.5x, 1x, 2x

  const intervalRef = useRef(null);
  const tapeContainerRef = useRef(null);
  const activeBtnRef = useRef(null);

  useEffect(() => {
    if (!isPlaying || tokens.length === 0) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    const intervalMs = Math.round(1300 / speed);
    intervalRef.current = setInterval(() => {
      setActiveIdx((prev) => (prev + 1) % tokens.length);
    }, intervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, tokens.length]);

  // Smoothly scroll the active token button into view if overflowed without affecting page scroll
  useEffect(() => {
    if (activeBtnRef.current && tapeContainerRef.current) {
      const container = tapeContainerRef.current;
      const btn = activeBtnRef.current;
      const btnLeft = btn.offsetLeft;
      const btnWidth = btn.offsetWidth;
      const containerWidth = container.offsetWidth;
      const targetScroll = btnLeft - (containerWidth / 2) + (btnWidth / 2);
      container.scrollTo({ left: targetScroll, behavior: 'smooth' });
    }
  }, [activeIdx]);

  const currentToken = tokens[activeIdx] || tokens[0] || '';
  
  // Deterministic token ID based on string hash for realistic vocabulary IDs
  const getVocabId = (str, idx) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return Math.abs(hash % 28000) + 100 + idx * 7;
  };

  const vocabId = getVocabId(currentToken, activeIdx);

  return (
    <div
      style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md, 8px)',
        padding: '1.15rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        width: '100%',
        boxSizing: 'border-box',
        minWidth: 0,
      }}
    >
      {/* Top Header & Interactive Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.65rem',
          minHeight: '34px',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', minWidth: 0 }}>
          <span style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <span>{isAr ? 'المُجزئ البصري المتدفق لدفعات النصوص (Animated Token Stream):' : 'Interactive Animated Token Stream:'}</span>
            <TermTooltip termKey="batch_size" size="sm" hideLabel={true} />
          </span>
          <span className="badge badge-teal" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', whiteSpace: 'nowrap' }}>
            [{batchSize}, L={tokens.length}, d={embedDim || 768}]
          </span>
        </div>

        {/* Playback Controls Toolbar */}
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.35rem',
            background: 'var(--bg-elevated)',
            padding: '0.2rem 0.4rem',
            borderRadius: '6px',
            border: '1px solid var(--border-subtle)',
            height: '32px',
            boxSizing: 'border-box',
            flexShrink: 0,
          }}
        >
          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              background: isPlaying ? 'var(--accent-teal)' : 'var(--bg-secondary)',
              color: isPlaying ? '#ffffff' : 'var(--text-primary)',
              border: 'none',
              borderRadius: '4px',
              padding: '0.25rem 0.55rem',
              fontSize: '0.72rem',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.25rem',
              height: '24px',
              boxSizing: 'border-box',
              whiteSpace: 'nowrap',
            }}
          >
            {isPlaying ? (isAr ? 'إيقاف مؤقت' : 'Pause') : (isAr ? 'تشغيل حي' : 'Play')}
          </button>

          <button
            type="button"
            onClick={() => {
              setIsPlaying(false);
              setActiveIdx((prev) => (prev - 1 + tokens.length) % tokens.length);
            }}
            style={{
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: 'none',
              borderRadius: '4px',
              padding: '0.25rem 0.45rem',
              fontSize: '0.72rem',
              fontWeight: 700,
              cursor: 'pointer',
              height: '24px',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title={isAr ? 'الكلمة السابقة' : 'Prev Token'}
          >
            {isAr ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>

          <button
            type="button"
            onClick={() => {
              setIsPlaying(false);
              setActiveIdx((prev) => (prev + 1) % tokens.length);
            }}
            style={{
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: 'none',
              borderRadius: '4px',
              padding: '0.25rem 0.45rem',
              fontSize: '0.72rem',
              fontWeight: 700,
              cursor: 'pointer',
              height: '24px',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            title={isAr ? 'الكلمة التالية' : 'Next Token'}
          >
            {isAr ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
          </button>

          <div style={{ height: '14px', width: '1px', background: 'var(--border-subtle)', margin: '0 0.15rem' }} />

          {[0.5, 1, 2].map((sp) => (
            <button
              key={sp}
              type="button"
              onClick={() => setSpeed(sp)}
              style={{
                background: speed === sp ? 'var(--accent-teal-glow)' : 'transparent',
                color: speed === sp ? 'var(--accent-teal)' : 'var(--text-muted)',
                border: speed === sp ? '1px solid var(--accent-teal)' : '1px solid transparent',
                borderRadius: '3px',
                padding: '0.15rem 0.35rem',
                fontSize: '0.68rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                cursor: 'pointer',
                height: '24px',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxSizing: 'border-box',
              }}
            >
              {sp}x
            </button>
          ))}
        </div>
      </div>

      {/* Active Token Spotlight Display Box - Fixed Height and Stable Layout (Zero CLS) */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.08) 0%, rgba(14, 165, 233, 0.03) 100%)',
          border: '1px solid var(--accent-teal)',
          borderRadius: '8px',
          padding: '0.85rem 1.15rem',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.85rem',
          alignItems: 'center',
          minHeight: '74px',
          boxSizing: 'border-box',
          width: '100%',
        }}
      >
        {/* Cell 1: Active Token */}
        <div style={{ minWidth: 0, overflow: 'hidden' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {isAr ? 'التوكن النشط (Active Token):' : 'Current Active Token:'}
          </span>
          <div style={{ height: '1.6rem', display: 'flex', alignItems: 'center', minWidth: 0 }}>
            <span
              style={{
                fontSize: '1.15rem',
                fontWeight: 800,
                color: 'var(--accent-teal)',
                fontFamily: 'var(--font-primary)',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                maxWidth: '100%',
                display: 'inline-block',
              }}
              title={currentToken}
            >
              «{currentToken}»
            </span>
          </div>
        </div>

        {/* Cell 2: Vocab ID */}
        <div style={{ minWidth: 0, overflow: 'hidden' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {isAr ? 'مُعرّف القاموس (Vocab ID):' : 'Vocabulary Index:'}
          </span>
          <div style={{ height: '1.6rem', display: 'flex', alignItems: 'center', minWidth: 0 }}>
            <span style={{ fontSize: '0.95rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>
              id: {vocabId}
            </span>
          </div>
        </div>

        {/* Cell 3: Position */}
        <div style={{ minWidth: 0, overflow: 'hidden' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {isAr ? 'الموقع التسلسلي (Position):' : 'Sequence Position:'}
          </span>
          <div style={{ height: '1.6rem', display: 'flex', alignItems: 'center', minWidth: 0 }}>
            <span style={{ fontSize: '0.95rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', whiteSpace: 'nowrap' }}>
              pos: {activeIdx + 1} / {tokens.length}
            </span>
          </div>
        </div>

        {/* Cell 4: Tensor Slice */}
        <div style={{ minWidth: 0, overflow: 'hidden' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {isAr ? 'متجه التضمين (Tensor):' : 'Embedding Projection:'}
          </span>
          <div style={{ height: '1.6rem', display: 'flex', alignItems: 'center', minWidth: 0 }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)', whiteSpace: 'nowrap' }}>
              x[b, {activeIdx}, :] -&gt; [1, {embedDim || 768}]
            </span>
          </div>
        </div>
      </div>

      {/* Horizontal Token Flow Tape with Laser Cursor - Fixed Container Dimensions */}
      <div style={{ width: '100%', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
          <span style={{ fontSize: '0.73rem', color: 'var(--text-secondary)' }}>
            {isAr ? 'شريط التدفق اللحظي للتوكنات (انقر على أي توكن للانتقال المباشر إليه):' : 'Interactive token tape (click any token to inspect):'}
          </span>
          <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
            {activeIdx + 1}/{tokens.length}
          </span>
        </div>

        <div
          ref={tapeContainerRef}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            overflowX: 'auto',
            overflowY: 'hidden',
            padding: '0.35rem 0.2rem',
            scrollbarWidth: 'thin',
            minHeight: '44px',
            height: '44px',
            boxSizing: 'border-box',
            width: '100%',
          }}
        >
          {tokens.map((tok, idx) => {
            const isCurrent = idx === activeIdx;
            const isPast = idx < activeIdx;
            return (
              <button
                key={idx}
                ref={isCurrent ? activeBtnRef : null}
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveIdx(idx);
                }}
                style={{
                  background: isCurrent
                    ? 'var(--accent-teal)'
                    : isPast
                    ? 'var(--bg-elevated)'
                    : 'var(--bg-tertiary)',
                  color: isCurrent ? '#ffffff' : isPast ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: isCurrent
                    ? '1.5px solid var(--accent-teal)'
                    : isPast
                    ? '1px solid var(--border-medium)'
                    : '1px solid var(--border-subtle)',
                  borderRadius: '6px',
                  padding: '0 0.65rem',
                  height: '32px',
                  fontSize: '0.78rem',
                  fontWeight: isCurrent ? 800 : 500,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  boxShadow: isCurrent ? '0 0 0 1px var(--accent-teal), 0 2px 10px rgba(20, 184, 166, 0.35)' : 'none',
                  transition: 'background 0.2s ease, border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease',
                  flexShrink: 0,
                  boxSizing: 'border-box',
                }}
              >
                <span>{tok}</span>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-mono)',
                    color: isCurrent ? 'rgba(255,255,255,0.9)' : 'var(--text-muted)',
                    fontWeight: isCurrent ? 800 : 500,
                  }}
                >
                  #{idx + 1}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
