// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// RoPE Angle Compass Interactive Simulator
// Matching Academy Design Language: Mathematical Dial, Bilingual, Dual Mode
// ==============================================================================

import React, { useState, useEffect, useRef } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { api } from '../../utils/api';
import TermTooltip from '../common/TermTooltip';

export default function RoPECompass() {
  const { language, connectionMode, theme } = useStudioStore();
  const isAr = language === 'ar';
  const isLight = theme === 'light';

  const [tokenIndex, setTokenIndex] = useState(3);
  const [selectedPair, setSelectedPair] = useState(0);
  const [simData, setSimData] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    let isMounted = true;
    api.getRoPESim({ token_index: tokenIndex, dim: 16, base: 10000 }).then((data) => {
      if (isMounted && data) setSimData(data);
    });
    return () => {
      isMounted = false;
    };
  }, [tokenIndex]);

  // Draw 2D Vector Compass on Canvas with Dual Mode Theme Sensitivity
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simData) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = width * 0.38;

    ctx.clearRect(0, 0, width, height);

    // Background Grid & Circular Orbit - High Contrast Dual Mode
    ctx.strokeStyle = isLight ? 'rgba(71, 85, 105, 0.22)' : 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();

    // Crosshairs
    ctx.strokeStyle = isLight ? 'rgba(71, 85, 105, 0.3)' : 'rgba(148, 163, 184, 0.2)';
    ctx.beginPath();
    ctx.moveTo(centerX - radius - 15, centerY);
    ctx.lineTo(centerX + radius + 15, centerY);
    ctx.moveTo(centerX, centerY - radius - 15);
    ctx.lineTo(centerX, centerY + radius + 15);
    ctx.stroke();

    const currentPair = simData.pairs?.[selectedPair] || simData.pairs?.[0];
    if (!currentPair) return;

    // Draw Original Vector (Position 0: Angle 0)
    ctx.strokeStyle = isLight ? 'rgba(100, 116, 139, 0.7)' : 'rgba(148, 163, 184, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + radius, centerY);
    ctx.stroke();

    // Original Vector point
    ctx.fillStyle = isLight ? '#64748B' : '#94A3B8';
    ctx.beginPath();
    ctx.arc(centerX + radius, centerY, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Rotated Vector (Position m)
    const rotX = centerX + currentPair.vector_rotated[0] * radius;
    const rotY = centerY - currentPair.vector_rotated[1] * radius; // Invert Y for Cartesian

    // Arc between original and rotated
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.45)';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.4, 0, -currentPair.angle_rad, true);
    ctx.stroke();

    // Vector Ray
    ctx.strokeStyle = '#0EA5E9';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(rotX, rotY);
    ctx.stroke();

    // Glowing tip
    ctx.fillStyle = '#0EA5E9';
    ctx.shadowColor = '#0EA5E9';
    ctx.shadowBlur = isLight ? 4 : 10;
    ctx.beginPath();
    ctx.arc(rotX, rotY, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0;
  }, [simData, selectedPair, isLight]);

  const pairInfo = simData?.pairs?.[selectedPair];

  return (
    <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg, 12px)', padding: '1.5rem', border: '1px solid var(--border-subtle)', boxShadow: 'var(--shadow-sm)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {isAr ? 'بوصلة التدوير الموضعي الدوراني (RoPE Compass)' : 'Rotary Position Embeddings (RoPE) Compass'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            {isAr
              ? 'تدوير المتجه بزاوية mθ حسب موضع الكلمة m مع الحفاظ التام على طول المتجه ||v|| = 1.0 دون تشويه دلالي.'
              : 'Vector orientation rotates with token position m, strictly preserving semantic length ||v|| = 1.0.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className="badge badge-teal">
            {isAr ? 'حفظ المسافات النسبية' : 'Relative Distance Preserved'}
          </span>
          <span className="badge badge-neutral">
            {connectionMode === 'library' ? 'dytr Core' : (isAr ? 'محاكاة فيزيائية' : 'Simulation Engine')}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', alignItems: 'center' }}>
        {/* Visual Dial Canvas */}
        <div style={{ textAlign: 'center', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', padding: '1.25rem', border: '1px solid var(--border-subtle)' }}>
          <canvas ref={canvasRef} width={260} height={260} style={{ maxWidth: '100%' }} />
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'var(--text-muted)' }}>
              {isAr ? '[المتجه الأصلي m=0]' : '[Base Vector m=0]'}
            </span>
            <span style={{ color: 'var(--accent-teal)', fontWeight: 700 }}>
              {isAr ? `[المتجه المدور m=${tokenIndex}]` : `[Rotated m=${tokenIndex}]`}
            </span>
          </div>
        </div>

        {/* Interactive Sliders and Math readout */}
        <div>
          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'موضع التوكن في الجملة (Position m):' : 'Token Position (m) in Sequence:'}</span>
                <TermTooltip
                  title={isAr ? 'موضع الكلمة في التسلسل (m)' : 'Token Position (m)'}
                  explanation={
                    isAr
                      ? 'الموقع الترتيبي للكلمة داخل الجملة؛ يُحدد زاوية دوران المتجه في الفضاء القطبي الثنائي m·θ لحفظ المسافات النسبية بدقة.'
                      : 'Positional index of the token in the sequence, dictating rotation angle m·theta to mathematically encode relative distance.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>m =</span>
                <input
                  type="number"
                  min="0"
                  max="64"
                  step="1"
                  value={tokenIndex}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setTokenIndex(Math.min(64, Math.max(0, val)));
                  }}
                  style={{ width: '40px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Token Position in Sequence"
              min="0"
              max="64"
              value={tokenIndex}
              onChange={(e) => setTokenIndex(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>

          <div className="form-group" style={{ marginBottom: '1.25rem' }}>
            <div className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem', fontSize: '0.88rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                <span>{isAr ? 'زوج الأبعاد الفرعي (Subspace Pair i):' : 'Subspace Feature Pair (i):'}</span>
                <TermTooltip
                  title={isAr ? 'زوج الأبعاد الفرعي (θ_i)' : 'Subspace Feature Pair (θ_i)'}
                  explanation={
                    isAr
                      ? 'تقسم خوارزمية RoPE فضاء التضمين إلى أزواج ثنائية (2D pairs)، يدور كل زوج بتردد أسي يتناقص مع رتبة البعد θ_i = 10000^(-2i/d).'
                      : 'RoPE splits embedding dimensions into orthogonal 2D subspaces, each rotating at distinct exponential base frequencies.'
                  }
                  size="sm"
                  hideLabel={true}
                />
              </span>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: '6px', padding: '0.15rem 0.45rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>i =</span>
                <input
                  type="number"
                  min="0"
                  max={(simData?.num_pairs || 8) - 1}
                  step="1"
                  value={selectedPair}
                  onChange={(e) => {
                    const maxP = (simData?.num_pairs || 8) - 1;
                    const val = parseInt(e.target.value);
                    if (!isNaN(val)) setSelectedPair(Math.min(maxP, Math.max(0, val)));
                  }}
                  style={{ width: '35px', background: 'transparent', border: 'none', outline: 'none', textAlign: 'center', fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', padding: 0 }}
                />
              </div>
            </div>
            <input
              type="range"
              aria-label="Subspace Feature Pair"
              min="0"
              max={(simData?.num_pairs || 8) - 1}
              value={selectedPair}
              onChange={(e) => setSelectedPair(parseInt(e.target.value))}
              className="slider-input"
            />
          </div>

          {pairInfo && (
            <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-sm)', fontSize: '0.84rem', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {isAr ? 'التردد الأساسي (Base Freq θ_i):' : 'Base Frequency (θ_i):'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{pairInfo.theta_i}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {isAr ? 'زاوية الدوران المحسوبة (m · θ_i):' : 'Rotation Angle (m · θ_i):'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700 }}>
                  {pairInfo.angle_deg}° ({pairInfo.angle_rad} rad)
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {isAr ? 'طول المتجه المحفوظ (Vector Norm):' : 'Preserved Norm (||v||):'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)', fontWeight: 700 }}>
                  {pairInfo.norm_preserved}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
