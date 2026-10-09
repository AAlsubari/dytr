// ==============================================================================
// DyTrStudio Interactive Multi-Layer Neural Training Loop Animator
// Visualizes Multi-Layer Neural Architecture:
// Input Layer (x) -> Hidden Layer (h) -> Output Layer (ŷ) -> Loss & Backpropagation
// Forward Signal Waves -> Loss Evaluation -> Backprop Gradient Waves -> Model Parameter Updates
// Zero Emojis Policy - High Performance SVG & CSS Transitions - Dual Theme
// ==============================================================================

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useStudioStore } from '../../store/studioStore';
import TermTooltip from '../common/TermTooltip';
import { ArrowRight } from 'lucide-react';

export default function NeuralTrainingLoopAnimator({ learningRate: defaultLr = 0.05 }) {
  const { language } = useStudioStore();
  const isAr = language === 'ar';

  // Animation phase: 1 = forward, 2 = loss, 3 = backward, 4 = update
  const [phase, setPhase] = useState(1);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1); // 0.5, 1, 2
  const [iteration, setIteration] = useState(1);
  const [activeTab, setActiveTab] = useState('inspector'); // 'inspector' | 'matrices' | 'math'

  // Adjustable Learning Rate (η) to make weight convergence visually striking
  const [lr, setLr] = useState(0.08);

  // Preset sample inputs
  const samplePresets = [
    { id: 'pos', name_ar: 'عينة إيجابية (جملة تصنيف 1)', name_en: 'Positive Sample (Class 1)', x: [0.85, 0.70, 0.65], target: [1.0, 0.0] },
    { id: 'neg', name_ar: 'عينة سلبية (جملة تصنيف 2)', name_en: 'Negative Sample (Class 2)', x: [0.20, 0.30, 0.15], target: [0.0, 1.0] },
    { id: 'neutral', name_ar: 'عينة مختلطة (سياق عام)', name_en: 'Mixed Sample (General)', x: [0.55, 0.50, 0.52], target: [1.0, 0.0] },
  ];

  // Selected Probe: { type: 'node' | 'synapse', ... }
  const [selectedProbe, setSelectedProbe] = useState({
    type: 'synapse',
    matrix: 'w1',
    i: 0,
    j: 0,
    source: 'x₁',
    target: 'h₁',
  });

  // Inputs: 3 nodes (Features / Token embeddings)
  const [inputs, setInputs] = useState([0.850, 0.700, 0.650]);

  // Target class output: [Class 1 (Positive), Class 2 (Negative)]
  const [targetClassIdx, setTargetClassIdx] = useState(0);
  const targets = useMemo(() => (targetClassIdx === 0 ? [1.0, 0.0] : [0.0, 1.0]), [targetClassIdx]);

  // Weight Matrices (Initial states with realistic values)
  // W1: 4 hidden nodes x 3 inputs (4x3)
  const [w1, setW1] = useState([
    [0.450, -0.280, 0.360],
    [-0.320, 0.540, 0.210],
    [0.390, 0.180, -0.420],
    [0.250, -0.310, 0.490],
  ]);
  const [b1, setB1] = useState([0.05, -0.02, 0.04, -0.01]);

  // W2: 2 output nodes x 4 hidden nodes (2x4)
  const [w2, setW2] = useState([
    [0.580, 0.620, -0.350, 0.410],
    [-0.420, -0.510, 0.380, -0.290],
  ]);
  const [b2, setB2] = useState([0.02, -0.02]);

  // Recorded last deltas for visual delta badges in Phase 4
  const [lastDeltas, setLastDeltas] = useState({
    w1: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    b1: [0, 0, 0, 0],
    w2: [[0, 0, 0, 0], [0, 0, 0, 0]],
    b2: [0, 0],
  });

  // Activation Function: Sigmoid for hidden, Softmax for output
  const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, z))));
  const sigmoidDeriv = (a) => a * (1 - a);

  // Forward Pass Calculations
  const forwardPass = useMemo(() => {
    // Hidden pre-activations (z_h) and activations (a_h)
    const z_h = [];
    const a_h = [];
    for (let i = 0; i < 4; i++) {
      let sum = b1[i];
      for (let j = 0; j < 3; j++) {
        sum += w1[i][j] * inputs[j];
      }
      z_h.push(sum);
      a_h.push(sigmoid(sum));
    }

    // Output pre-activations (z_o)
    const z_o = [];
    for (let i = 0; i < 2; i++) {
      let sum = b2[i];
      for (let j = 0; j < 4; j++) {
        sum += w2[i][j] * a_h[j];
      }
      z_o.push(sum);
    }

    // Softmax probabilities: y_hat
    const maxZ = Math.max(...z_o);
    const expZ = z_o.map((z) => Math.exp(z - maxZ));
    const sumExp = expZ.reduce((acc, v) => acc + v, 0);
    const y_hat = expZ.map((e) => e / Math.max(1e-9, sumExp));

    // Cross-Entropy Loss: - sum( y_i * log(y_hat_i) )
    const loss = -targets.reduce((acc, t, i) => acc + t * Math.log(Math.max(1e-7, y_hat[i])), 0);

    return { z_h, a_h, z_o, y_hat, loss };
  }, [inputs, w1, b1, w2, b2, targets]);

  // Backward Pass Calculations (Gradients)
  const backwardPass = useMemo(() => {
    const { a_h, y_hat } = forwardPass;

    // Output gradients: delta_o = y_hat - target
    const delta_o = y_hat.map((yh, i) => yh - targets[i]);

    // Gradients for W2: dL/dW2[i][j] = delta_o[i] * a_h[j]
    const grad_w2 = [];
    for (let i = 0; i < 2; i++) {
      grad_w2.push([]);
      for (let j = 0; j < 4; j++) {
        grad_w2[i].push(delta_o[i] * a_h[j]);
      }
    }
    const grad_b2 = [...delta_o];

    // Hidden gradients: delta_h[j] = (sum_i delta_o[i] * w2[i][j]) * sigmoid'(a_h[j])
    const delta_h = [];
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let i = 0; i < 2; i++) {
        sum += delta_o[i] * w2[i][j];
      }
      delta_h.push(sum * sigmoidDeriv(a_h[j]));
    }

    // Gradients for W1: dL/dW1[i][j] = delta_h[i] * inputs[j]
    const grad_w1 = [];
    for (let i = 0; i < 4; i++) {
      grad_w1.push([]);
      for (let j = 0; j < 3; j++) {
        grad_w1[i].push(delta_h[i] * inputs[j]);
      }
    }
    const grad_b1 = [...delta_h];

    return { delta_o, grad_w2, grad_b2, delta_h, grad_w1, grad_b1 };
  }, [forwardPass, targets, w2, inputs]);

  // Explicit, synchronous parameter update execution
  const executeWeightUpdate = useCallback(() => {
    const { grad_w1, grad_b1, grad_w2, grad_b2 } = backwardPass;

    // Compute exact deltas
    const delta_w1 = grad_w1.map((row) => row.map((g) => -lr * g));
    const delta_b1 = grad_b1.map((g) => -lr * g);
    const delta_w2 = grad_w2.map((row) => row.map((g) => -lr * g));
    const delta_b2 = grad_b2.map((g) => -lr * g);

    setLastDeltas({
      w1: delta_w1,
      b1: delta_b1,
      w2: delta_w2,
      b2: delta_b2,
    });

    // Apply updates directly to matrices
    setW1((prev) =>
      prev.map((row, i) =>
        row.map((val, j) => Number((val + delta_w1[i][j]).toFixed(5)))
      )
    );
    setB1((prev) =>
      prev.map((val, i) => Number((val + delta_b1[i]).toFixed(5)))
    );

    setW2((prev) =>
      prev.map((row, i) =>
        row.map((val, j) => Number((val + delta_w2[i][j]).toFixed(5)))
      )
    );
    setB2((prev) =>
      prev.map((val, i) => Number((val + delta_b2[i]).toFixed(5)))
    );

    setIteration((it) => it + 1);
  }, [backwardPass, lr]);

  // Animation Loop Scheduler
  useEffect(() => {
    if (!isPlaying) return;

    const stepDuration = 1100 / speed;
    const timer = setTimeout(() => {
      if (phase === 1) {
        setPhase(2); // Forward -> Loss
      } else if (phase === 2) {
        setPhase(3); // Loss -> Backward
      } else if (phase === 3) {
        // When transitioning to Phase 4, immediately execute weight update!
        executeWeightUpdate();
        setPhase(4);
      } else if (phase === 4) {
        setPhase(1); // Update -> Forward (New cycle)
      }
    }, stepDuration);

    return () => clearTimeout(timer);
  }, [isPlaying, phase, speed, executeWeightUpdate]);

  // Manual Step Controller
  const stepManual = (forward = true) => {
    setIsPlaying(false);
    if (forward) {
      if (phase === 3) {
        executeWeightUpdate();
        setPhase(4);
      } else if (phase === 4) {
        setPhase(1);
      } else {
        setPhase((p) => p + 1);
      }
    } else {
      setPhase((p) => (p > 1 ? p - 1 : 4));
    }
  };

  // Reset Model Parameters to Initial State
  const resetSimulation = () => {
    setIsPlaying(false);
    setPhase(1);
    setIteration(1);
    setInputs([0.850, 0.700, 0.650]);
    setTargetClassIdx(0);
    setW1([
      [0.450, -0.280, 0.360],
      [-0.320, 0.540, 0.210],
      [0.390, 0.180, -0.420],
      [0.250, -0.310, 0.490],
    ]);
    setB1([0.05, -0.02, 0.04, -0.01]);
    setW2([
      [0.580, 0.620, -0.350, 0.410],
      [-0.420, -0.510, 0.380, -0.290],
    ]);
    setB2([0.02, -0.02]);
    setLastDeltas({
      w1: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      b1: [0, 0, 0, 0],
      w2: [[0, 0, 0, 0], [0, 0, 0, 0]],
      b2: [0, 0],
    });
  };

  const applyPreset = (preset) => {
    setInputs([...preset.x]);
    setTargetClassIdx(preset.target[0] === 1.0 ? 0 : 1);
  };

  // Node Coordinates for SVG Network Canvas (viewBox: 0 0 920 330)
  const nodeCoords = useMemo(() => {
    return {
      inputs: [
        { id: 'x0', name: 'x₁', label: isAr ? 'المتجه السياقي x₁' : 'Token Vector x₁', x: 100, y: 65, val: inputs[0] },
        { id: 'x1', name: 'x₂', label: isAr ? 'المضمن اللغوي x₂' : 'Embedding x₂', x: 100, y: 165, val: inputs[1] },
        { id: 'x2', name: 'x₃', label: isAr ? 'الترميز الموضعي x₃' : 'Positional Code x₃', x: 100, y: 265, val: inputs[2] },
      ],
      hidden: [
        { id: 'h0', name: 'h₁', label: isAr ? 'خلية الاستعلام (Q) h₁' : 'Self-Attention (Q) h₁', x: 440, y: 50, z: forwardPass.z_h[0], a: forwardPass.a_h[0], grad: backwardPass.delta_h[0], biasDelta: lastDeltas.b1[0] },
        { id: 'h1', name: 'h₂', label: isAr ? 'خلية المفتاح (K) h₂' : 'Key Projection (K) h₂', x: 440, y: 125, z: forwardPass.z_h[1], a: forwardPass.a_h[1], grad: backwardPass.delta_h[1], biasDelta: lastDeltas.b1[1] },
        { id: 'h2', name: 'h₃', label: isAr ? 'خلية القيمة (V) h₃' : 'Value Projection (V) h₃', x: 440, y: 200, z: forwardPass.z_h[2], a: forwardPass.a_h[2], grad: backwardPass.delta_h[2], biasDelta: lastDeltas.b1[2] },
        { id: 'h3', name: 'h₄', label: isAr ? 'الخلية المغذية (FFN) h₄' : 'Feed-Forward MLP h₄', x: 440, y: 275, z: forwardPass.z_h[3], a: forwardPass.a_h[3], grad: backwardPass.delta_h[3], biasDelta: lastDeltas.b1[3] },
      ],
      outputs: [
        { id: 'o0', name: 'ŷ₁', label: isAr ? 'رأس التصنيف ŷ₁ (إيجابي)' : 'Class 1 (Positive) ŷ₁', x: 740, y: 105, z: forwardPass.z_o[0], prob: forwardPass.y_hat[0], target: targets[0], grad: backwardPass.delta_o[0], biasDelta: lastDeltas.b2[0] },
        { id: 'o1', name: 'ŷ₂', label: isAr ? 'رأس التصنيف ŷ₂ (سلبي)' : 'Class 2 (Negative) ŷ₂', x: 740, y: 225, z: forwardPass.z_o[1], prob: forwardPass.y_hat[1], target: targets[1], grad: backwardPass.delta_o[1], biasDelta: lastDeltas.b2[1] },
      ],
    };
  }, [inputs, forwardPass, backwardPass, targets, lastDeltas, isAr]);

  // Phase Definitions
  const phaseDetails = [
    {
      id: 1,
      title_ar: 'الانتشار الأمامي للشبكة',
      title_en: 'Forward Signal Flow',
      desc_ar: 'انسياب الإشارات وحساب المجموع الخطي z والتنشيط a والاحتمالات ŷ',
      desc_en: 'Signals propagate through synapses, computing linear sum z and Softmax ŷ',
      color: 'var(--accent-teal)',
    },
    {
      id: 2,
      title_ar: 'قياس دالة الخطأ (Loss)',
      title_en: 'Loss Evaluation',
      desc_ar: 'مقارنة الاحتمالات ŷ بالهدف y وحساب الخسارة Cross-Entropy ℒ',
      desc_en: 'Measuring divergence between target y and prediction ŷ via Cross-Entropy Loss',
      color: 'var(--accent-amber)',
    },
    {
      id: 3,
      title_ar: 'موجة التدرج العكسي (Backprop)',
      title_en: 'Backprop Gradient Wave',
      desc_ar: 'رجوع إشارات الخطأ عكسياً لتحديد مشتقات كافة الأوزان ∇W₂ و ∇W₁',
      desc_en: 'Reverse wave computing exact parameter gradients via chain rule',
      color: '#ec4899',
    },
    {
      id: 4,
      title_ar: 'تحديث أوزان الخلايا (Update)',
      title_en: 'Parameter Weight Update',
      desc_ar: 'تعديل مصفوفات الأوزان والانحيازات بمقدار ΔW = -η·∇ℒ لتقليل الخطأ',
      desc_en: 'Updating all weights and biases by delta rule W_new = W - lr * grad',
      color: 'var(--accent-emerald)',
    },
  ];

  // Active Probed Element Details
  const probeDetails = useMemo(() => {
    if (selectedProbe.type === 'node') {
      const { nodeId } = selectedProbe;
      if (nodeId.startsWith('x')) {
        const idx = parseInt(nodeId.replace('x', ''), 10);
        return {
          title: isAr ? `خلية المدخل x${idx + 1}` : `Input Node x${idx + 1}`,
          typeLabel: isAr ? 'طبقة المدخلات (Input Layer)' : 'Input Layer',
          primaryLabel: isAr ? 'القيمة المدخلة (Feature x):' : 'Feature Value (x):',
          primaryVal: inputs[idx].toFixed(4),
          secondaryLabel: isAr ? 'نوع المتجه:' : 'Vector Nature:',
          secondaryVal: isAr ? 'مضمن لغوي سياقي' : 'Token Embedding Feature',
        };
      } else if (nodeId.startsWith('h')) {
        const idx = parseInt(nodeId.replace('h', ''), 10);
        return {
          title: isAr ? `الخلية العصبية المخفية h${idx + 1}` : `Hidden Neuron h${idx + 1}`,
          typeLabel: isAr ? 'الطبقة المخفية (Transformer Hidden Layer)' : 'Hidden Layer',
          primaryLabel: isAr ? 'قيمة التنشيط a = σ(z):' : 'Activation a = σ(z):',
          primaryVal: forwardPass.a_h[idx].toFixed(4),
          secondaryLabel: isAr ? 'المجموع الخطي z:' : 'Linear Sum z:',
          secondaryVal: forwardPass.z_h[idx].toFixed(4),
          biasLabel: isAr ? 'الانحياز b₁ والتعديل Δb₁:' : 'Bias b₁ & Δb₁:',
          biasVal: `${b1[idx].toFixed(4)} (${lastDeltas.b1[idx] >= 0 ? '+' : ''}${lastDeltas.b1[idx].toFixed(5)})`,
          gradLabel: isAr ? 'تدرج الخطأ δ_h:' : 'Error Gradient δ_h:',
          gradVal: backwardPass.delta_h[idx].toFixed(5),
        };
      } else if (nodeId.startsWith('o')) {
        const idx = parseInt(nodeId.replace('o', ''), 10);
        return {
          title: isAr ? `خلية الإخراج والتوقع ŷ${idx + 1}` : `Output Prediction Node ŷ${idx + 1}`,
          typeLabel: isAr ? 'طبقة الإخراج (Classification Head)' : 'Output Layer',
          primaryLabel: isAr ? 'الاحتمال المتوقع ŷ:' : 'Predicted Probability ŷ:',
          primaryVal: `${(forwardPass.y_hat[idx] * 100).toFixed(2)}%`,
          secondaryLabel: isAr ? 'الهدف الحقيقي y:' : 'True Target y:',
          secondaryVal: targets[idx].toString(),
          biasLabel: isAr ? 'الانحياز b₂ والتعديل Δb₂:' : 'Bias b₂ & Δb₂:',
          biasVal: `${b2[idx].toFixed(4)} (${lastDeltas.b2[idx] >= 0 ? '+' : ''}${lastDeltas.b2[idx].toFixed(5)})`,
          gradLabel: isAr ? 'خطأ التوقع (ŷ - y):' : 'Output Error (ŷ - y):',
          gradVal: backwardPass.delta_o[idx].toFixed(5),
        };
      }
    } else if (selectedProbe.type === 'synapse') {
      const { matrix, i, j } = selectedProbe;
      if (matrix === 'w1') {
        const weightVal = w1[i][j];
        const gradVal = backwardPass.grad_w1[i][j];
        const deltaVal = lastDeltas.w1[i] ? lastDeltas.w1[i][j] : -lr * gradVal;
        return {
          title: isAr ? `الوصلة العصبية W₁(h${i + 1}, x${j + 1})` : `Synapse W₁(h${i + 1}, x${j + 1})`,
          typeLabel: isAr ? 'مصفوفة أوزان المدخلات إلى الطبقة المخفية' : 'Input-to-Hidden Weight Matrix',
          primaryLabel: isAr ? 'قيمة الوزن المحدث w:' : 'Current Weight w:',
          primaryVal: weightVal.toFixed(5),
          secondaryLabel: isAr ? 'الإشارة المنقولة x·w:' : 'Transmitted Signal x·w:',
          secondaryVal: (inputs[j] * weightVal).toFixed(4),
          gradLabel: isAr ? 'تدرج الوزن dL/dw:' : 'Weight Gradient dL/dw:',
          gradVal: gradVal.toFixed(5),
          deltaLabel: isAr ? 'التعديل المطبق Δw:' : 'Applied Delta Δw:',
          deltaVal: deltaVal >= 0 ? `+${deltaVal.toFixed(6)}` : deltaVal.toFixed(6),
        };
      } else if (matrix === 'w2') {
        const weightVal = w2[i][j];
        const gradVal = backwardPass.grad_w2[i][j];
        const deltaVal = lastDeltas.w2[i] ? lastDeltas.w2[i][j] : -lr * gradVal;
        return {
          title: isAr ? `الوصلة العصبية W₂(ŷ${i + 1}, h${j + 1})` : `Synapse W₂(ŷ${i + 1}, h${j + 1})`,
          typeLabel: isAr ? 'مصفوفة أوزان الطبقة المخفية إلى رأس الإخراج' : 'Hidden-to-Output Weight Matrix',
          primaryLabel: isAr ? 'قيمة الوزن المحدث w:' : 'Current Weight w:',
          primaryVal: weightVal.toFixed(5),
          secondaryLabel: isAr ? 'الإشارة المنقولة h·w:' : 'Transmitted Signal h·w:',
          secondaryVal: (forwardPass.a_h[j] * weightVal).toFixed(4),
          gradLabel: isAr ? 'تدرج الوزن dL/dw:' : 'Weight Gradient dL/dw:',
          gradVal: gradVal.toFixed(5),
          deltaLabel: isAr ? 'التعديل المطبق Δw:' : 'Applied Delta Δw:',
          deltaVal: deltaVal >= 0 ? `+${deltaVal.toFixed(6)}` : deltaVal.toFixed(6),
        };
      }
    }
    return null;
  }, [selectedProbe, inputs, forwardPass, backwardPass, targets, w1, b1, w2, b2, lastDeltas, lr, isAr]);

  return (
    <div
      style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg, 12px)',
        padding: '1.25rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem',
      }}
    >
      {/* Top Toolbar: Sample Presets, Learning Rate & Playback Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        {/* Sample Scenario Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-muted)' }}>
            {isAr ? 'السيناريو التدريبي:' : 'Sample:'}
          </span>
          {samplePresets.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => applyPreset(p)}
              className="btn btn-secondary btn-sm"
              style={{
                fontSize: '0.72rem',
                padding: '0.25rem 0.6rem',
                borderRadius: '5px',
                border: (p.target[0] === 1.0 && targetClassIdx === 0) || (p.target[0] === 0.0 && targetClassIdx === 1) ? '1px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                background: (p.target[0] === 1.0 && targetClassIdx === 0) || (p.target[0] === 0.0 && targetClassIdx === 1) ? 'rgba(20,184,166,0.1)' : 'transparent',
              }}
            >
              {isAr ? p.name_ar : p.name_en}
            </button>
          ))}
        </div>

        {/* Learning Rate (η) Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'var(--bg-elevated)', padding: '0.25rem 0.6rem', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-teal)' }}>
            η = {lr}
          </span>
          <div style={{ display: 'flex', gap: '0.2rem' }}>
            {[0.01, 0.05, 0.1, 0.2].map((val) => (
              <button
                key={val}
                type="button"
                onClick={() => setLr(val)}
                style={{
                  background: lr === val ? 'var(--accent-teal)' : 'transparent',
                  color: lr === val ? '#ffffff' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: '3px',
                  padding: '0.1rem 0.35rem',
                  fontSize: '0.68rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  fontWeight: lr === val ? 700 : 400,
                }}
              >
                {val}
              </button>
            ))}
          </div>
        </div>

        {/* Playback Controls & Speed Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            className={`btn btn-sm ${isPlaying ? 'btn-danger' : 'btn-primary'}`}
            style={{ minWidth: '80px', fontWeight: 700 }}
          >
            {isPlaying ? (isAr ? 'إيقاف مؤقت' : 'Pause') : (isAr ? 'تشغيل' : 'Play')}
          </button>

          <button
            type="button"
            onClick={() => stepManual(true)}
            className="btn btn-secondary btn-sm"
            title={isAr ? 'خطوة تالية للأمام وتحديث الأوزان' : 'Step Forward & Update'}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
          >
            <span>{isAr ? 'خطوة' : 'Step'}</span>
            <ArrowRight size={13} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          </button>

          <button
            type="button"
            onClick={resetSimulation}
            className="btn btn-secondary btn-sm"
            title={isAr ? 'إعادة ضبط المحاكاة' : 'Reset'}
          >
            {isAr ? 'إعادة' : 'Reset'}
          </button>

          {/* Speed Toggle */}
          <div style={{ display: 'flex', background: 'var(--bg-elevated)', borderRadius: '4px', border: '1px solid var(--border-subtle)', padding: '2px' }}>
            {[0.5, 1, 2].map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpeed(s)}
                style={{
                  background: speed === s ? 'var(--accent-teal)' : 'transparent',
                  color: speed === s ? '#ffffff' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: '3px',
                  padding: '0.15rem 0.45rem',
                  fontSize: '0.7rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer',
                  fontWeight: speed === s ? 700 : 400,
                }}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 4-Phase Flow Ribbon with Live Active Indicators */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.5rem' }}>
        {phaseDetails.map((st) => {
          const isActive = phase === st.id;
          return (
            <div
              key={st.id}
              onClick={() => {
                setIsPlaying(false);
                if (st.id === 4) executeWeightUpdate();
                setPhase(st.id);
              }}
              style={{
                background: isActive ? 'var(--bg-elevated)' : 'var(--bg-tertiary)',
                border: `1px solid ${isActive ? st.color : 'var(--border-subtle)'}`,
                borderBottom: `3px solid ${isActive ? st.color : 'transparent'}`,
                borderRadius: '6px',
                padding: '0.55rem 0.75rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
                <span style={{ fontSize: '0.76rem', fontWeight: 800, color: isActive ? st.color : 'var(--text-muted)' }}>
                  {isAr ? st.title_ar : st.title_en}
                </span>
                <span
                  style={{
                    width: '7px',
                    height: '7px',
                    borderRadius: '50%',
                    background: isActive ? st.color : 'var(--border-subtle)',
                    boxShadow: isActive ? `0 0 8px ${st.color}` : 'none',
                  }}
                />
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.35 }}>
                {isAr ? st.desc_ar : st.desc_en}
              </div>
            </div>
          );
        })}
      </div>

      {/* Interactive Multi-Layer Neural Network SVG Canvas */}
      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md, 8px)',
          padding: '1.25rem 0.75rem',
          position: 'relative',
          overflow: 'hidden',
          minHeight: '340px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        {/* Layer Header Column Labels */}
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 2rem 0.5rem 2rem', borderBottom: '1px dashed var(--border-subtle)', marginBottom: '0.5rem' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--accent-teal)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span>{isAr ? 'طبقة المدخلات (3 خلايا)' : 'Input Layer (3 Neurons)'}</span>
            <TermTooltip termKey="vector_embeddings" size="sm" hideLabel={true} />
          </div>
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span>{isAr ? 'الطبقة المخفية (4 خلايا عصبية)' : 'Hidden Layer (4 Neurons)'}</span>
            <TermTooltip termKey="neural_architecture" size="sm" hideLabel={true} />
          </div>
          <div style={{ fontSize: '0.78rem', fontWeight: 800, color: 'var(--accent-amber)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span>{isAr ? 'رأس التصنيف ودالة الخطأ' : 'Output Head & Loss Comparator'}</span>
            <TermTooltip termKey="softmax_layer" size="sm" hideLabel={true} />
          </div>
        </div>

        {/* SVG Network Schematic */}
        <svg viewBox="0 0 920 330" style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
          <defs>
            {/* Forward Synapse Gradient */}
            <linearGradient id="forwardGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--accent-teal)" stopOpacity="0.9" />
              <stop offset="100%" stopColor="var(--accent-cyan)" stopOpacity="1" />
            </linearGradient>

            {/* Backprop Synapse Gradient */}
            <linearGradient id="backGrad" x1="100%" y1="0%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="#ec4899" stopOpacity="1" />
              <stop offset="100%" stopColor="var(--accent-amber)" stopOpacity="0.9" />
            </linearGradient>

            {/* Weight Update Synapse Gradient */}
            <linearGradient id="updateGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="var(--accent-emerald)" stopOpacity="0.95" />
              <stop offset="100%" stopColor="var(--accent-teal)" stopOpacity="0.95" />
            </linearGradient>

            {/* High Performance Glow Filter */}
            <filter id="neonGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3.5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* =============================================================== */}
          {/* SYNAPSES LAYER 1: Inputs (3) -> Hidden (4) (12 Synapses)         */}
          {/* =============================================================== */}
          {nodeCoords.inputs.map((inp, j) =>
            nodeCoords.hidden.map((hid, i) => {
              const isSelected =
                selectedProbe.type === 'synapse' &&
                selectedProbe.matrix === 'w1' &&
                selectedProbe.i === i &&
                selectedProbe.j === j;

              const weight = w1[i][j];
              const strokeColor =
                phase === 1
                  ? 'url(#forwardGrad)'
                  : phase === 3
                  ? 'url(#backGrad)'
                  : phase === 4
                  ? 'url(#updateGrad)'
                  : 'var(--border-medium)';

              const strokeWidth = isSelected ? '3.5' : phase === 1 || phase === 3 ? '2.2' : '1.2';
              const opacity = isSelected ? 1 : Math.max(0.25, Math.abs(weight) * 0.85);

              return (
                <g key={`syn_w1_${i}_${j}`}>
                  <line
                    x1={inp.x + 22}
                    y1={inp.y}
                    x2={hid.x - 22}
                    y2={hid.y}
                    stroke={isSelected ? 'var(--accent-teal)' : strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={phase === 1 || phase === 3 ? '6,4' : 'none'}
                    opacity={opacity}
                    style={{
                      cursor: 'pointer',
                      transition: 'all 0.25s ease',
                      animation:
                        phase === 1
                          ? 'dashForward 0.8s linear infinite'
                          : phase === 3
                          ? 'dashBack 0.8s linear infinite'
                          : 'none',
                    }}
                    onClick={() =>
                      setSelectedProbe({
                        type: 'synapse',
                        matrix: 'w1',
                        i,
                        j,
                        source: inp.name,
                        target: hid.name,
                      })
                    }
                  />
                </g>
              );
            })
          )}

          {/* =============================================================== */}
          {/* SYNAPSES LAYER 2: Hidden (4) -> Output (2) (8 Synapses)          */}
          {/* =============================================================== */}
          {nodeCoords.hidden.map((hid, j) =>
            nodeCoords.outputs.map((out, i) => {
              const isSelected =
                selectedProbe.type === 'synapse' &&
                selectedProbe.matrix === 'w2' &&
                selectedProbe.i === i &&
                selectedProbe.j === j;

              const weight = w2[i][j];
              const strokeColor =
                phase === 1
                  ? 'url(#forwardGrad)'
                  : phase === 3
                  ? 'url(#backGrad)'
                  : phase === 4
                  ? 'url(#updateGrad)'
                  : 'var(--border-medium)';

              const strokeWidth = isSelected ? '3.5' : phase === 1 || phase === 3 ? '2.4' : '1.3';
              const opacity = isSelected ? 1 : Math.max(0.3, Math.abs(weight) * 0.9);

              return (
                <g key={`syn_w2_${i}_${j}`}>
                  <line
                    x1={hid.x + 22}
                    y1={hid.y}
                    x2={out.x - 34}
                    y2={out.y}
                    stroke={isSelected ? 'var(--accent-teal)' : strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={phase === 1 || phase === 3 ? '6,4' : 'none'}
                    opacity={opacity}
                    style={{
                      cursor: 'pointer',
                      transition: 'all 0.25s ease',
                      animation:
                        phase === 1
                          ? 'dashForward 0.8s linear infinite'
                          : phase === 3
                          ? 'dashBack 0.8s linear infinite'
                          : 'none',
                    }}
                    onClick={() =>
                      setSelectedProbe({
                        type: 'synapse',
                        matrix: 'w2',
                        i,
                        j,
                        source: hid.name,
                        target: out.name,
                      })
                    }
                  />
                </g>
              );
            })
          )}

          {/* =============================================================== */}
          {/* NODES: INPUT LAYER (3 Circles)                                  */}
          {/* =============================================================== */}
          {nodeCoords.inputs.map((inp) => {
            const isSelected = selectedProbe.type === 'node' && selectedProbe.nodeId === inp.id;
            return (
              <g
                key={inp.id}
                transform={`translate(${inp.x}, ${inp.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedProbe({ type: 'node', nodeId: inp.id })}
              >
                <circle
                  r="21"
                  fill="var(--bg-elevated)"
                  stroke={isSelected ? 'var(--accent-teal)' : phase === 1 ? 'var(--accent-teal)' : 'var(--border-medium)'}
                  strokeWidth={isSelected ? '3.5' : phase === 1 ? '2.5' : '1.5'}
                  filter={phase === 1 || isSelected ? 'url(#neonGlow)' : 'none'}
                />
                <text textAnchor="middle" dy="-3" fill="var(--text-muted)" fontSize="8.5" fontWeight="800">
                  {inp.name}
                </text>
                <text textAnchor="middle" dy="11" fill="var(--accent-teal)" fontSize="10.5" fontFamily="var(--font-mono)" fontWeight="700">
                  {inp.val.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* =============================================================== */}
          {/* NODES: HIDDEN LAYER (4 Circles)                                 */}
          {/* =============================================================== */}
          {nodeCoords.hidden.map((hid, idx) => {
            const isSelected = selectedProbe.type === 'node' && selectedProbe.nodeId === hid.id;
            return (
              <g
                key={hid.id}
                transform={`translate(${hid.x}, ${hid.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedProbe({ type: 'node', nodeId: hid.id })}
              >
                <circle
                  r="22"
                  fill="var(--bg-elevated)"
                  stroke={
                    isSelected
                      ? 'var(--accent-cyan)'
                      : phase === 1
                      ? 'var(--accent-cyan)'
                      : phase === 3
                      ? '#ec4899'
                      : phase === 4
                      ? 'var(--accent-emerald)'
                      : 'var(--border-medium)'
                  }
                  strokeWidth={isSelected ? '3.5' : phase === 1 || phase === 3 || phase === 4 ? '2.5' : '1.5'}
                  filter={phase === 1 || phase === 3 || phase === 4 || isSelected ? 'url(#neonGlow)' : 'none'}
                />
                <text textAnchor="middle" dy="-4" fill="var(--text-muted)" fontSize="8.5" fontWeight="800">
                  {hid.name}
                </text>
                <text textAnchor="middle" dy="10" fill={phase === 4 ? 'var(--accent-emerald)' : 'var(--accent-cyan)'} fontSize="10.5" fontFamily="var(--font-mono)" fontWeight="700">
                  {hid.a.toFixed(2)}
                </text>
                <text textAnchor="middle" dy="33" fill="var(--text-muted)" fontSize="7.5" fontFamily="var(--font-mono)">
                  b={b1[idx].toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* =============================================================== */}
          {/* NODES: OUTPUT LAYER (2 Rectangles)                              */}
          {/* =============================================================== */}
          {nodeCoords.outputs.map((out) => {
            const isSelected = selectedProbe.type === 'node' && selectedProbe.nodeId === out.id;
            return (
              <g
                key={out.id}
                transform={`translate(${out.x}, ${out.y})`}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedProbe({ type: 'node', nodeId: out.id })}
              >
                <rect
                  x="-34"
                  y="-22"
                  width="68"
                  height="44"
                  rx="7"
                  fill="var(--bg-elevated)"
                  stroke={
                    isSelected
                      ? 'var(--accent-amber)'
                      : phase === 2
                      ? 'var(--accent-amber)'
                      : phase === 3
                      ? '#ec4899'
                      : phase === 1
                      ? 'var(--accent-teal)'
                      : phase === 4
                      ? 'var(--accent-emerald)'
                      : 'var(--border-medium)'
                  }
                  strokeWidth={isSelected ? '3.5' : phase === 2 || phase === 3 || phase === 4 ? '2.8' : '1.5'}
                  filter={phase === 2 || phase === 3 || phase === 4 || isSelected ? 'url(#neonGlow)' : 'none'}
                />
                <text textAnchor="middle" dy="-5" fill="var(--text-muted)" fontSize="8.5" fontWeight="700">
                  {out.name} {isAr ? `(هدف: ${out.target})` : `(y=${out.target})`}
                </text>
                <text textAnchor="middle" dy="11" fill="var(--accent-amber)" fontSize="11" fontFamily="var(--font-mono)" fontWeight="800">
                  {(out.prob * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}

          {/* =============================================================== */}
          {/* LOSS COMPARATOR & TARGET GAUGE (Right Column)                   */}
          {/* =============================================================== */}
          <g transform="translate(860, 165)">
            <circle
              r="28"
              fill="var(--bg-elevated)"
              stroke={phase === 2 ? 'var(--accent-amber)' : phase === 3 ? '#ec4899' : 'var(--border-medium)'}
              strokeWidth={phase === 2 || phase === 3 ? '3' : '1.5'}
              filter={phase === 2 || phase === 3 ? 'url(#neonGlow)' : 'none'}
            />
            <text textAnchor="middle" dy="-7" fill="var(--text-muted)" fontSize="8.5" fontWeight="800">
              {isAr ? 'الخسارة ℒ' : 'Loss ℒ'}
            </text>
            <text textAnchor="middle" dy="9" fill="var(--accent-amber)" fontSize="11" fontFamily="var(--font-mono)" fontWeight="800">
              {forwardPass.loss.toFixed(4)}
            </text>
            <text textAnchor="middle" dy="23" fill="var(--accent-teal)" fontSize="7.5" fontFamily="var(--font-mono)">
              CE Loss
            </text>
          </g>

          {/* Comparator Connectors from Outputs to Loss */}
          <line
            x1={nodeCoords.outputs[0].x + 34}
            y1={nodeCoords.outputs[0].y}
            x2="835"
            y2="150"
            stroke={phase === 2 ? 'var(--accent-amber)' : phase === 3 ? '#ec4899' : 'var(--border-medium)'}
            strokeWidth={phase === 2 || phase === 3 ? '2.5' : '1.2'}
            strokeDasharray={phase === 2 || phase === 3 ? '4,3' : 'none'}
          />
          <line
            x1={nodeCoords.outputs[1].x + 34}
            y1={nodeCoords.outputs[1].y}
            x2="835"
            y2="180"
            stroke={phase === 2 ? 'var(--accent-amber)' : phase === 3 ? '#ec4899' : 'var(--border-medium)'}
            strokeWidth={phase === 2 || phase === 3 ? '2.5' : '1.2'}
            strokeDasharray={phase === 2 || phase === 3 ? '4,3' : 'none'}
          />

          {/* SVG Keyframes */}
          <style>{`
            @keyframes dashForward {
              from { stroke-dashoffset: 20; }
              to { stroke-dashoffset: 0; }
            }
            @keyframes dashBack {
              from { stroke-dashoffset: 0; }
              to { stroke-dashoffset: 20; }
            }
          `}</style>
        </svg>
      </div>

      {/* Tabs Switcher: Element Inspector | Weight Matrices | Mathematical Proof */}
      <div style={{ display: 'flex', gap: '0.4rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.4rem' }}>
        <button
          type="button"
          onClick={() => setActiveTab('inspector')}
          className={`btn btn-sm ${activeTab === 'inspector' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '0.76rem', padding: '0.35rem 0.75rem' }}
        >
          {isAr ? 'مجس الفحص اللحظي للعناصر' : 'Element Probing Inspector'}
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('matrices')}
          className={`btn btn-sm ${activeTab === 'matrices' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '0.76rem', padding: '0.35rem 0.75rem' }}
        >
          {isAr ? 'مصفوفات الأوزان الكاملة (W₁, W₂)' : 'Full Weight Matrices (W₁, W₂)'}
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('math')}
          className={`btn btn-sm ${activeTab === 'math' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ fontSize: '0.76rem', padding: '0.35rem 0.75rem' }}
        >
          {isAr ? 'الصياغة الرياضية والمشتقات' : 'Mathematical Formulation'}
        </button>
      </div>

      {/* TAB 1: Element Probing Inspector */}
      {activeTab === 'inspector' && probeDetails && (
        <div
          style={{
            background: 'var(--bg-tertiary)',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-md, 8px)',
            padding: '1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="badge badge-teal" style={{ fontSize: '0.72rem' }}>
                {probeDetails.typeLabel}
              </span>
              <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                {probeDetails.title}
              </strong>
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              {isAr ? 'انقر على أي خلية أو وصلة في الرسم البياني لتثبيت مجس الفحص عليها' : 'Click any node or synapse in diagram to attach probe'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem' }}>
            <div className="diagram-box" style={{ padding: '0.6rem' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{probeDetails.primaryLabel}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)', fontSize: '0.95rem' }}>
                {probeDetails.primaryVal}
              </span>
            </div>

            <div className="diagram-box" style={{ padding: '0.6rem' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{probeDetails.secondaryLabel}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                {probeDetails.secondaryVal}
              </span>
            </div>

            {probeDetails.biasVal !== undefined && (
              <div className="diagram-box" style={{ padding: '0.6rem', border: phase === 4 ? '1px solid var(--accent-emerald)' : undefined }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{probeDetails.biasLabel}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: phase === 4 ? 'var(--accent-emerald)' : 'var(--text-primary)', fontSize: '0.88rem' }}>
                  {probeDetails.biasVal}
                </span>
              </div>
            )}

            {probeDetails.gradVal !== undefined && (
              <div className="diagram-box" style={{ padding: '0.6rem', border: phase === 3 ? '1px solid #ec4899' : undefined }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                  <span>{probeDetails.gradLabel}</span>
                  <TermTooltip termKey="backpropagation" size="sm" hideLabel={true} />
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#ec4899', fontSize: '0.9rem' }}>
                  {probeDetails.gradVal}
                </span>
              </div>
            )}

            {probeDetails.deltaVal !== undefined && (
              <div className="diagram-box" style={{ padding: '0.6rem', border: phase === 4 ? '1px solid var(--accent-emerald)' : undefined }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                  <span>{probeDetails.deltaLabel}</span>
                  <TermTooltip termKey="weight_update" size="sm" hideLabel={true} />
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-emerald)', fontSize: '0.95rem' }}>
                  {probeDetails.deltaVal}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Weight Matrices Heatmap Grid (W1 & W2) */}
      {activeTab === 'matrices' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {/* Matrix W1 */}
          <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-teal)', marginBottom: '0.5rem' }}>
              {isAr ? 'مصفوفة أوزان الطبقة الأولى W₁ (4 خلايا × 3 مدخلات):' : 'Layer 1 Weights W₁ (4x3):'}
            </div>
            <table style={{ width: '100%', fontSize: '0.74rem', fontFamily: 'var(--font-mono)', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '4px', textAlign: 'start' }}>Neuron</th>
                  <th>x₁</th>
                  <th>x₂</th>
                  <th>x₃</th>
                  <th>b₁</th>
                </tr>
              </thead>
              <tbody>
                {w1.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px dashed var(--border-subtle)' }}>
                    <td style={{ padding: '4px', fontWeight: 700, color: 'var(--accent-cyan)' }}>h{i + 1}</td>
                    {row.map((val, j) => (
                      <td
                        key={j}
                        onClick={() =>
                          setSelectedProbe({
                            type: 'synapse',
                            matrix: 'w1',
                            i,
                            j,
                            source: `x${j + 1}`,
                            target: `h${i + 1}`,
                          })
                        }
                        style={{
                          padding: '4px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          color: val >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                          background: selectedProbe.matrix === 'w1' && selectedProbe.i === i && selectedProbe.j === j ? 'rgba(20,184,166,0.15)' : 'transparent',
                          borderRadius: '3px',
                          fontWeight: 700,
                        }}
                      >
                        {val.toFixed(3)}
                      </td>
                    ))}
                    <td style={{ padding: '4px', textAlign: 'center', color: 'var(--text-muted)' }}>{b1[i].toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Matrix W2 */}
          <div style={{ background: 'var(--bg-tertiary)', padding: '0.85rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 800, color: 'var(--accent-amber)', marginBottom: '0.5rem' }}>
              {isAr ? 'مصفوفة أوزان الطبقة الثانية W₂ (خليتان × 4 مخفية):' : 'Layer 2 Weights W₂ (2x4):'}
            </div>
            <table style={{ width: '100%', fontSize: '0.74rem', fontFamily: 'var(--font-mono)', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '4px', textAlign: 'start' }}>Head</th>
                  <th>h₁</th>
                  <th>h₂</th>
                  <th>h₃</th>
                  <th>h₄</th>
                  <th>b₂</th>
                </tr>
              </thead>
              <tbody>
                {w2.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px dashed var(--border-subtle)' }}>
                    <td style={{ padding: '4px', fontWeight: 700, color: 'var(--accent-amber)' }}>ŷ{i + 1}</td>
                    {row.map((val, j) => (
                      <td
                        key={j}
                        onClick={() =>
                          setSelectedProbe({
                            type: 'synapse',
                            matrix: 'w2',
                            i,
                            j,
                            source: `h${j + 1}`,
                            target: `ŷ${i + 1}`,
                          })
                        }
                        style={{
                          padding: '4px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          color: val >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                          background: selectedProbe.matrix === 'w2' && selectedProbe.i === i && selectedProbe.j === j ? 'rgba(245,158,11,0.15)' : 'transparent',
                          borderRadius: '3px',
                          fontWeight: 700,
                        }}
                      >
                        {val.toFixed(3)}
                      </td>
                    ))}
                    <td style={{ padding: '4px', textAlign: 'center', color: 'var(--text-muted)' }}>{b2[i].toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: Mathematical Formulations & Derivatives */}
      {activeTab === 'math' && (
        <div style={{ background: 'var(--bg-tertiary)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', fontSize: '0.82rem', lineHeight: 1.6 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.75rem' }}>
            <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: '6px' }}>
              <strong style={{ color: 'var(--accent-teal)' }}>{isAr ? '1. التمرير الأمامي (Forward):' : '1. Forward Pass:'}</strong>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginTop: '0.3rem', color: 'var(--text-secondary)' }}>
                z_h = W₁·x + b₁ <br />
                a_h = σ(z_h) = 1 / (1 + e^-z_h) <br />
                z_o = W₂·a_h + b₂ <br />
                ŷ = Softmax(z_o)
              </div>
            </div>

            <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: '6px' }}>
              <strong style={{ color: 'var(--accent-amber)' }}>{isAr ? '2. دالة الخسارة (Loss):' : '2. Cross-Entropy Loss:'}</strong>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginTop: '0.3rem', color: 'var(--text-secondary)' }}>
                ℒ = - ∑ y_i · log(ŷ_i) <br />
                δ_o = ∂ℒ/∂z_o = ŷ - y
              </div>
            </div>

            <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: '6px' }}>
              <strong style={{ color: '#ec4899' }}>{isAr ? '3. الانتشار العكسي (Backpropagation):' : '3. Backprop Gradients:'}</strong>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginTop: '0.3rem', color: 'var(--text-secondary)' }}>
                ∂ℒ/∂W₂ = δ_o · a_h^T <br />
                δ_h = (W₂^T · δ_o) ⊙ σ'(a_h) <br />
                ∂ℒ/∂W₁ = δ_h · x^T
              </div>
            </div>

            <div style={{ background: 'var(--bg-surface)', padding: '0.75rem', borderRadius: '6px' }}>
              <strong style={{ color: 'var(--accent-emerald)' }}>{isAr ? '4. تحديث الأوزان (Weight Update):' : '4. Parameter Update:'}</strong>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginTop: '0.3rem', color: 'var(--text-secondary)' }}>
                W_new = W_old - η · (∂ℒ/∂W) <br />
                b_new = b_old - η · (∂ℒ/∂b)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Global Network Performance Telemetry */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem' }}>
        <div className="diagram-box" style={{ padding: '0.65rem', border: phase === 1 ? '1px solid var(--accent-teal)' : undefined }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'خطوة التحسين (Step)' : 'Iteration Step'}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-teal)' }}>
            #{iteration}
          </span>
        </div>

        <div className="diagram-box" style={{ padding: '0.65rem', border: phase === 2 ? '1px solid var(--accent-amber)' : undefined }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
            <span>{isAr ? 'دالة الخسارة (Loss ℒ)' : 'Loss (ℒ)'}</span>
            <TermTooltip termKey="loss_function" size="sm" hideLabel={true} />
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-amber)' }}>
            {forwardPass.loss.toFixed(4)}
          </span>
        </div>

        <div className="diagram-box" style={{ padding: '0.65rem', border: phase === 3 ? '1px solid #ec4899' : undefined }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
            <span>{isAr ? 'أقصى تدرج |∇ℒ|' : 'Max Gradient |g|'}</span>
            <TermTooltip termKey="backpropagation" size="sm" hideLabel={true} />
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#ec4899' }}>
            {Math.max(...backwardPass.delta_o.map(Math.abs), ...backwardPass.delta_h.map(Math.abs)).toFixed(4)}
          </span>
        </div>

        <div className="diagram-box" style={{ padding: '0.65rem', border: phase === 4 ? '1px solid var(--accent-emerald)' : undefined }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
            <span>{isAr ? 'معدل التعلم (η)' : 'Learning Rate (η)'}</span>
            <TermTooltip termKey="learning_rate" size="sm" hideLabel={true} />
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, color: 'var(--accent-emerald)' }}>
            {lr}
          </span>
        </div>
      </div>
    </div>
  );
}
