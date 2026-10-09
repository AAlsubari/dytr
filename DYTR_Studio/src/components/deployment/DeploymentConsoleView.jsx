// ==============================================================================
// DyTrStudio Stage 09: Production Deployment & Serving Console
// Based on CRISP-DM Phase 6 (Deployment) & Production Serving Standards
// Apple macOS-Style Code Window with Traffic Lights & Gutter Line Numbers
// Full Syntax Highlighting - Dynamic Multi-Task Dataset Binding
// Zero Emojis Policy - Bilingual RTL/LTR - 3-Layer Academic TermTooltips
// ==============================================================================

import React, { useState, useMemo } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';
import { academicDictionary } from '../../utils/academicDictionary';
import TermTooltip from '../common/TermTooltip';
import SectionCard from '../common/SectionCard';
import { ArrowLeft, ArrowRight } from 'lucide-react';

// -----------------------------------------------------------------------------
// High-Precision Syntax Tokenizer & Colorizer (Apple macOS IDE Theme)
// -----------------------------------------------------------------------------
function HighlightedCodeLine({ line }) {
  const trimmed = line.trim();

  // Full Comment Line
  if (trimmed.startsWith('#') || trimmed.startsWith('//')) {
    return <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>{line}</span>;
  }

  // Tokenize using regex capturing keywords, strings, numbers, decorators, functions
  const tokenRegex = /(#.*$|\/\/.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(\b(?:import|from|def|class|return|if|else|elif|for|in|while|with|as|try|except|finally|raise|async|await|global|pass|break|continue|function|const|let|var|throw|new)\b)|(\b(?:True|False|None|self|true|false|null|undefined)\b)|(\b(?:torch|nn|optim|Dataset|DataLoader|ModelConfig|TaskConfig|TrainingStrategy|DynamicTransformer|FastAPI|BaseModel|Field|HTTPException|CORSMiddleware|np|pd|os|sys|time|json|math|re|uvicorn|onnxruntime|ort|websockets|asyncio|performance|fetch)\b)|(\b(?:print|len|range|sum|int|float|str|dict|list|set|tuple|abs|hash|enumerate|zip|min|max|round|super|type|isinstance|open|console|log|error|stringify|parse|now|toFixed)\b)|(@[a-zA-Z0-9_\.]+|\![a-zA-Z0-9_\-]+)|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([=+\-*/%&|^~<>!]+)/g;

  const elements = [];
  let lastIndex = 0;
  let match;

  while ((match = tokenRegex.exec(line)) !== null) {
    const matchIndex = match.index;
    if (matchIndex > lastIndex) {
      elements.push(
        <span key={lastIndex} style={{ color: '#e2e8f0' }}>
          {line.slice(lastIndex, matchIndex)}
        </span>
      );
    }

    const [
      fullMatch,
      commentMatch,
      stringMatch,
      keywordMatch,
      constMatch,
      classMatch,
      funcMatch,
      decoratorMatch,
      numberMatch,
      operatorMatch,
    ] = match;

    if (commentMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#94a3b8', fontStyle: 'italic' }}>
          {commentMatch}
        </span>
      );
    } else if (stringMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#34d399', fontWeight: 500 }}>
          {stringMatch}
        </span>
      );
    } else if (keywordMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#38bdf8', fontWeight: 700 }}>
          {keywordMatch}
        </span>
      );
    } else if (constMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#f472b6', fontWeight: 600 }}>
          {constMatch}
        </span>
      );
    } else if (classMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#2dd4bf', fontWeight: 700 }}>
          {classMatch}
        </span>
      );
    } else if (funcMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#818cf8', fontWeight: 600 }}>
          {funcMatch}
        </span>
      );
    } else if (decoratorMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#fb7185', fontWeight: 700 }}>
          {decoratorMatch}
        </span>
      );
    } else if (numberMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#fbbf24', fontWeight: 600 }}>
          {numberMatch}
        </span>
      );
    } else if (operatorMatch) {
      elements.push(
        <span key={matchIndex} style={{ color: '#93c5fd' }}>
          {operatorMatch}
        </span>
      );
    } else {
      elements.push(
        <span key={matchIndex} style={{ color: '#f1f5f9' }}>
          {fullMatch}
        </span>
      );
    }

    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < line.length) {
    elements.push(
      <span key={lastIndex} style={{ color: '#e2e8f0' }}>
        {line.slice(lastIndex)}
      </span>
    );
  }

  return elements.length > 0 ? <>{elements}</> : <span>{line || ' '}</span>;
}

export default function DeploymentConsoleView() {
  const {
    language,
    setActiveModule,
    modelConfig,
    deploymentState,
    updateDeploymentSettings,
    testDeploymentEndpoint,
  } = useStudioStore();

  const { currentDataset } = useDatasetStore();

  const d = academicDictionary[language] || academicDictionary.en;
  const isAr = language === 'ar';

  const [selectedMode, setSelectedMode] = useState(deploymentState.selectedMode || 'api');
  const [testText, setTestText] = useState(
    deploymentState.testInput || 'تتواصل الاستعدادات لافتتاح المجمع الرياضي الاولمبي الجديد المزود باحدث التقنيات الذكيه.'
  );
  const [activeCodeTab, setActiveCodeTab] = useState('python_api');
  const [isCopied, setIsCopied] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  // Active task and target labels
  const activeTask = modelConfig.tasks?.[0] || {
    task_name: 'task_topic',
    task_type: 'sentence_classification',
    text_column: 'content',
    label_column: 'category',
    classes: ['تقنية', 'اقتصاد', 'رياضة'],
  };

  const activeTaskName = activeTask.task_name;
  const activeClasses = activeTask.classes || ['تقنية', 'اقتصاد', 'رياضة'];

  const { settings, activeDeployments, testOutput } = deploymentState;
  const activeDep = activeDeployments[0] || {
    id: 'dep_prod_dytr_v1',
    name: 'DyTr Multi-Task Production Service',
    mode: 'api',
    endpointUrl: 'https://api.alsubari.dev/v1/predict/dytr_model_v1',
    apiKey: 'sk_live_dytr_99a8b7c6d5e4f3a2b1',
    status: 'active',
    uptime: '14h 42m',
    requestsLastHour: 1248,
    avgLatencyMs: 18.4,
    errorRate: '0.00%',
    driftScore: 0.042,
  };

  const deploymentModes = [
    {
      id: 'api',
      title: isAr ? 'خدمة ويب فورية (REST API / HTTP)' : 'HTTP REST Microservice',
      badge: isAr ? 'خدمة مباشرة' : 'Online Serving',
      desc: isAr
        ? 'نشر النموذج كخادم ويب متصل بالإنترنت يستقبل طلبات JSON ويجيب في أجزاء من الثانية (FastAPI / Triton).'
        : 'Deploy as an asynchronous FastAPI microservice with JSON I/O and low latency.',
    },
    {
      id: 'batch',
      title: isAr ? 'معالجة دفعية مجدولة (Batch Processing)' : 'Scheduled Batch Engine',
      badge: isAr ? 'معالجة ليلية' : 'Offline Batch',
      desc: isAr
        ? 'معالجة ملايين السجلات دفعة واحدة في أوقات محددة وفق جدول زمني محدد (Cron Scheduler / Pandas Pipeline).'
        : 'High-throughput offline batch processing for scheduled recurring pipelines.',
    },
    {
      id: 'edge',
      title: isAr ? 'تشغيل طرفي على الجهاز (Edge / Embedded)' : 'Edge & On-Device Serving',
      badge: isAr ? 'دون إنترنت' : 'Embedded WASM',
      desc: isAr
        ? 'تكميم وتصدير النموذج ليعمل محليا داخل المتصفح أو على الهواتف دون إرسال بيانات للخارج (ONNX Runtime / INT8).'
        : 'Quantized INT8/FP16 execution in browser (WebAssembly/WebGPU) or mobile clients.',
    },
    {
      id: 'streaming',
      title: isAr ? 'تدفق بيانات مستمر (Real-time Streaming)' : 'Event Stream Processor',
      badge: isAr ? 'تدفق لحظي' : 'Kafka / WebSocket',
      desc: isAr
        ? 'معالجة مستمرة للتدفقات اللحظية مثل شبكات التواصل الاجتماعي أو أجهزة إنترنت الأشياء (WebSocket / Kafka).'
        : 'Continuous inference on streaming event brokers (WebSocket, Kafka, MQTT).',
    },
  ];

  // Dynamic code generation
  const clientSnippets = useMemo(() => {
    const ep = activeDep.endpointUrl;
    const key = activeDep.apiKey;
    const samplePayload = testText.replace(/"/g, '\\"');
    const classListStr = JSON.stringify(activeClasses);

    return {
      python_api: {
        title: 'Python (Requests API Client)',
        filename: 'client_requests.py',
        code: `# ==============================================================================
# DyTR Production HTTP Client (Python Requests)
# Dynamic Multi-Task Inference Gateway
# Target Task: ${activeTaskName} | Classes: ${classListStr}
# ==============================================================================

import requests
import json
import time

ENDPOINT_URL = "${ep}"
API_KEY = "${key}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "text": "${samplePayload}",
    "task_name": "${activeTaskName}"
}

def invoke_model_serving():
    try:
        t_start = time.perf_counter()
        response = requests.post(ENDPOINT_URL, json=payload, headers=HEADERS, timeout=5)
        latency_ms = (time.perf_counter() - t_start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            print(f"Status: HTTP 200 OK | Latency: {latency_ms:.2f} ms")
            print(f"Predicted Class: {data.get('prediction')}")
            print(f"Confidence:      {data.get('confidence', 0.0) * 100:.2f}%")
            return data
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"Connection Failed: {exc}")

if __name__ == "__main__":
    invoke_model_serving()`,
      },

      javascript: {
        title: 'JavaScript (Fetch / Node.js)',
        filename: 'client_fetch.js',
        code: `// ==============================================================================
// DyTR Production Client (Modern ES6+ / Fetch API)
// Target Task: ${activeTaskName}
// ==============================================================================

async function queryDytrEndpoint() {
  const endpoint = "${ep}";
  const apiKey = "${key}";
  
  const payload = {
    text: "${samplePayload}",
    task_name: "${activeTaskName}"
  };

  const t0 = performance.now();
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Authorization": \`Bearer \${apiKey}\`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(\`HTTP Server Error: \${response.status}\`);
    }

    const { data } = await response.json();
    const latency = (performance.now() - t0).toFixed(2);

    console.log(\`Predicted: \${data.prediction} (\${(data.confidence * 100).toFixed(1)}%) in \${latency}ms\`);
    return data;
  } catch (err) {
    console.error("Inference Error:", err.message);
  }
}

queryDytrEndpoint();`,
      },

      curl: {
        title: 'cURL (Bash / Terminal)',
        filename: 'invoke_endpoint.sh',
        code: `#!/usr/bin/env bash
# ==============================================================================
# DyTR Production cURL CLI Invocation
# ==============================================================================

curl -X POST "${ep}" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "${samplePayload}",
    "task_name": "${activeTaskName}"
  }'`,
      },

      batch_python: {
        title: 'Python (Batch Processing Pipeline)',
        filename: 'batch_pipeline.py',
        code: `# ==============================================================================
# DyTR High-Throughput Batch Inference Pipeline (Pandas + Parallel Requests)
# Target Task: ${activeTaskName}
# ==============================================================================

import pandas as pd
import requests
import time

ENDPOINT_URL = "${ep}"
API_KEY = "${key}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
INPUT_CSV = "unlabeled_data.csv"
OUTPUT_CSV = "predictions_output.csv"
BATCH_SIZE = 64

def run_batch_inference():
    print(f"Ingesting input corpus: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    text_col = df.columns[0]
    
    predictions = []
    confidences = []
    total = len(df)
    
    print(f"Executing batch inference on {total} records (Batch Size: {BATCH_SIZE})...")
    for start_idx in range(0, total, BATCH_SIZE):
        chunk = df.iloc[start_idx : start_idx + BATCH_SIZE]
        for _, row in chunk.iterrows():
            payload = {"text": str(row[text_col]), "task_name": "${activeTaskName}"}
            try:
                res = requests.post(ENDPOINT_URL, json=payload, headers=HEADERS, timeout=10)
                if res.status_code == 200:
                    d = res.json().get("data", {})
                    predictions.append(d.get("prediction", "Unknown"))
                    confidences.append(d.get("confidence", 0.0))
                else:
                    predictions.append("Error")
                    confidences.append(0.0)
            except Exception:
                predictions.append("Failed")
                confidences.append(0.0)
                
        print(f"Progress: {min(start_idx + BATCH_SIZE, total)}/{total} processed...")

    df["predicted_class"] = predictions
    df["confidence_score"] = confidences
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Batch inference complete! Output stored in: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_batch_inference()`,
      },

      edge_onnx: {
        title: 'Python (Local Edge ONNX Runtime)',
        filename: 'edge_onnx_local.py',
        code: `# ==============================================================================
# DyTR Local Edge Runtime (ONNX Runtime / Zero-Network Execution)
# Target Task: ${activeTaskName} | Quantization: INT8
# ==============================================================================

import onnxruntime as ort
import numpy as np
import time

ONNX_MODEL_FILE = "dytr_quantized_int8.onnx"
MAX_SEQ_LEN = 128
VOCAB_SIZE = 30522
CLASSES = ${classListStr}

# 1. Initialize High-Performance ONNX Session
session = ort.InferenceSession(ONNX_MODEL_FILE, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"ONNX Session running on hardware provider: {session.get_providers()[0]}")

# 2. Tokenizer Embedding Mapping
def tokenize(text):
    tokens = str(text).strip().split()
    ids = [abs(hash(t)) % (VOCAB_SIZE - 100) + 100 for t in tokens]
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    else:
        ids += [0] * (MAX_SEQ_LEN - len(ids))
    return np.array([ids], dtype=np.int64)

# 3. Local High-Speed Inference
def predict_local(text):
    t0 = time.perf_counter()
    tensor_in = tokenize(text)
    outputs = session.run([output_name], {input_name: tensor_in})
    logits = outputs[0][0]
    
    # Softmax Normalization
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    latency_ms = (time.perf_counter() - t0) * 1000
    
    label = CLASSES[pred_idx] if pred_idx < len(CLASSES) else f"Class_{pred_idx}"
    return {"prediction": label, "confidence": round(conf, 4), "latency_ms": round(latency_ms, 2)}

if __name__ == "__main__":
    result = predict_local("${samplePayload}")
    print(f"Local Edge Prediction Result: {result}")`,
      },
    };
  }, [activeDep, activeTaskName, activeClasses, testText]);

  const activeSnippet = clientSnippets[activeCodeTab] || clientSnippets.python_api;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(activeSnippet.code);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleDownloadSnippet = () => {
    const ext = activeSnippet.filename.split('.').pop() || 'py';
    const mime = ext === 'js' ? 'text/javascript' : ext === 'sh' ? 'text/x-sh' : 'text/x-python';
    const blob = new Blob([activeSnippet.code], { type: `${mime};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeSnippet.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleRunEndpointTest = async () => {
    setIsTesting(true);
    await testDeploymentEndpoint(testText, activeTaskName);
    setIsTesting(false);
  };

  return (
    <div className="container" style={{ maxWidth: '1060px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Studio Header & Stage Indicator */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge badge-teal">
            {isAr ? 'المرحلة 09 من 10' : 'Stage 09 of 10'}
          </span>
          <span className="badge badge-neutral" style={{ fontFamily: 'var(--font-mono)' }}>
            CRISP-DM Phase 6: Deployment & Monitoring
          </span>
          <span className="badge badge-teal" style={{ fontFamily: 'var(--font-mono)' }}>
            Active Task: {activeTaskName} ({activeClasses.join(' · ')})
          </span>
        </div>

        <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
          {isAr ? 'منصة النشر والتشغيل والمراقبة الحية' : 'Production Deployment & Serving Console'}
        </h1>
        <p style={{ fontSize: '0.92rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          {isAr
            ? 'تحويل النموذج إلى خدمة برمجية حية: اختيار نمط النشر، ضبط قيود الاستخدام، تجربة الاستدعاء المباشر، ومراقبة انحراف البيانات (Data Drift).'
            : 'Operationalize trained transformers into production: REST APIs, scheduled batches, WASM edge runtimes, and drift observability.'}
        </p>
      </div>

      {/* Section 09.1: Deployment Topology Selection */}
      <SectionCard
        sectionCode="09.1"
        title={isAr ? 'اختر نمط وبيئة التشغيل (Deployment Mode)' : 'Select Serving Architecture'}
        subtitle={
          isAr
            ? 'تحديد كيفية استهلاك واستدعاء النموذج البرمجي وفق متطلبات السرعة والخصوصية والتكلفة.'
            : 'Select the optimal deployment paradigm for latency, cost, and infrastructure constraints.'
        }
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          {deploymentModes.map((mode) => {
            const isSelected = selectedMode === mode.id;
            return (
              <div
                key={mode.id}
                onClick={() => setSelectedMode(mode.id)}
                style={{
                  padding: '1.15rem',
                  borderRadius: '10px',
                  border: isSelected ? '2px solid var(--accent-teal)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'var(--bg-elevated)' : 'var(--bg-card)',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  boxShadow: isSelected ? '0 4px 12px rgba(45, 212, 191, 0.08)' : 'none',
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span className="badge badge-neutral" style={{ fontSize: '0.72rem' }}>{mode.badge}</span>
                    {isSelected && (
                      <span className="badge badge-teal" style={{ fontWeight: 700 }}>
                        {isAr ? 'مفعل' : 'Active'}
                      </span>
                    )}
                  </div>
                  <div style={{ fontWeight: 700, fontSize: '0.94rem', color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
                    {mode.title}
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                    {mode.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Dynamic Topology Settings */}
        <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--text-primary)', marginBottom: '0.85rem' }}>
            {isAr ? 'إعدادات النمط المختار:' : 'Mode Configuration Parameters:'}
          </div>

          {selectedMode === 'api' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  <TermTooltip id="rate_limit">
                    {isAr ? 'حد معدل الطلبات (Rate Limit / min):' : 'Rate Limit (req/min):'}
                  </TermTooltip>
                </label>
                <input
                  type="number"
                  value={settings.rateLimit}
                  onChange={(e) => updateDeploymentSettings({ rateLimit: Number(e.target.value) })}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  {isAr ? 'الطلبات المتزامنة القصوى (Concurrency):' : 'Max Concurrent Requests:'}
                </label>
                <input
                  type="number"
                  value={settings.maxConcurrent}
                  onChange={(e) => updateDeploymentSettings({ maxConcurrent: Number(e.target.value) })}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-primary)',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>
            </div>
          )}

          {selectedMode === 'batch' && (
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                {isAr ? 'جدول التنفيذ الدوري (Cron Schedule Expression):' : 'Execution Schedule (Cron):'}
              </label>
              <input
                type="text"
                value={settings.batchSchedule}
                onChange={(e) => updateDeploymentSettings({ batchSchedule: e.target.value })}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-mono)',
                  marginBottom: '0.35rem',
                }}
              />
              <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                {isAr
                  ? '0 2 * * * = تشغيل تلقائي يومياً عند الساعة 02:00 صباحاً لمعالجة سجلات قواعد البيانات المتراكمة.'
                  : '0 2 * * * = Runs daily at 02:00 AM UTC for offline batch inference.'}
              </span>
            </div>
          )}

          {selectedMode === 'edge' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  {isAr ? 'بيئة التشغيل الطرفية المستهدفة:' : 'Target Platform:'}
                </label>
                <select
                  value={settings.edgeTarget}
                  onChange={(e) => updateDeploymentSettings({ edgeTarget: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="browser">Web Browser (WASM / WebGPU)</option>
                  <option value="mobile_ios">Apple iOS CoreML</option>
                  <option value="mobile_android">Android TFLite / ONNX</option>
                  <option value="raspberry_pi">Embedded Linux / Edge IoT</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                  <TermTooltip id="edge_quantization">
                    {isAr ? 'تكميم النموذج (Quantization):' : 'Model Quantization:'}
                  </TermTooltip>
                </label>
                <select
                  value={settings.edgeQuantization}
                  onChange={(e) => updateDeploymentSettings({ edgeQuantization: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: '6px',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="int8">INT8 Quantization (تقليص الحجم 75% مع دقة فائقة)</option>
                  <option value="fp16">FP16 Half-Precision (توفير 50% من الذاكرة)</option>
                  <option value="none">FP32 Full Precision (الأصل الكامل)</option>
                </select>
              </div>
            </div>
          )}

          {selectedMode === 'streaming' && (
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
                {isAr ? 'قناة التدفق اللحظي (Event Stream Topic):' : 'Streaming Topic URL:'}
              </label>
              <input
                type="text"
                value={settings.streamingTopic}
                onChange={(e) => updateDeploymentSettings({ streamingTopic: e.target.value })}
                style={{
                  width: '100%',
                  padding: '0.5rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-mono)',
                }}
              />
            </div>
          )}
        </div>
      </SectionCard>

      {/* Section 09.2: Interactive Live Endpoint Sandbox */}
      <SectionCard
        sectionCode="09.2"
        title={isAr ? 'حقل الاختبار التفاعلي للنقطة النهائية (Endpoint Sandbox)' : 'Live Endpoint Testing Sandbox'}
        subtitle={
          isAr
            ? 'تجربة استدعاء نقطة النهاية المنشورة فوريا والتحقق من زمن الاستجابة ومخرجات JSON.'
            : 'Send real test payloads to the active deployment endpoint and verify latency.'
        }
      >
        <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', marginBottom: '1.25rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                {isAr ? 'عنوان نقطة النهاية السحابية (Endpoint URL):' : 'Production Endpoint URL:'}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', color: 'var(--accent-teal)', marginTop: '0.2rem', wordBreak: 'break-all' }}>
                {activeDep.endpointUrl}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                {isAr ? 'مفتاح المصادقة والترخيص (API Key):' : 'Secret Authorization Key:'}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {`${activeDep.apiKey.substring(0, 16)}...`}
              </div>
            </div>
          </div>

          {/* Test Input & Runner */}
          <div>
            <label style={{ display: 'block', fontWeight: 600, fontSize: '0.82rem', marginBottom: '0.35rem', color: 'var(--text-primary)' }}>
              {isAr ? `نص الفحص والاستنتاج المستهدف (رأس المهمة: ${activeTaskName}):` : `Payload Input for ${activeTaskName}:`}
            </label>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <input
                type="text"
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder={isAr ? 'أدخل نصا لتجربة النشر...' : 'Enter test payload...'}
                style={{
                  flex: 1,
                  minWidth: '260px',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border-subtle)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  fontSize: '0.88rem',
                }}
              />
              <button
                onClick={handleRunEndpointTest}
                disabled={isTesting}
                className="btn btn-primary"
                style={{
                  background: 'var(--accent-teal)',
                  color: '#ffffff',
                  whiteSpace: 'nowrap',
                  fontWeight: 700,
                  opacity: isTesting ? 0.7 : 1,
                }}
              >
                {isTesting ? (isAr ? 'جاري المعالجة...' : 'Processing...') : (isAr ? 'إرسال طلب فحص مباشر' : 'Send Test Request')}
              </button>
            </div>
          </div>

          {/* Test Response Inspector */}
          {testOutput && (
            <div
              style={{
                marginTop: '1.25rem',
                padding: '1rem',
                borderRadius: '8px',
                background: '#0a0f1d',
                border: '1px solid var(--border-active)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem', fontSize: '0.78rem' }}>
                <span className="badge badge-teal" style={{ fontWeight: 700 }}>
                  {isAr ? 'استجابة معيارية (HTTP 200 OK)' : 'HTTP 200 OK'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 600 }}>
                  {isAr ? `زمن المعالجة: ${testOutput.latencyMs} مللي ثانية` : `Latency: ${testOutput.latencyMs} ms`}
                </span>
              </div>
              <pre
                style={{
                  margin: 0,
                  fontSize: '0.82rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#34d399',
                  overflowX: 'auto',
                  lineHeight: 1.5,
                }}
              >
                {JSON.stringify(
                  {
                    status: 'success',
                    code: 200,
                    data: {
                      task_name: testOutput.taskName || activeTaskName,
                      prediction: testOutput.prediction,
                      confidence: +(testOutput.confidence || 0.94).toFixed(4),
                      probabilities: testOutput.probabilities || [
                        { class_name: testOutput.prediction, prob: testOutput.confidence || 0.94 },
                      ],
                      timestamp: testOutput.timestamp,
                    },
                    meta: {
                      latency_ms: testOutput.latencyMs,
                      serving_engine: 'FastAPI + DyTr Inference Engine v0.1.3',
                      rate_limit_remaining: 99,
                    },
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Section 09.3: Telemetry & Data Drift Monitoring */}
      <SectionCard
        sectionCode="09.3"
        title={isAr ? 'مراقبة صحة النشر وانحراف البيانات (Drift Observability)' : 'Deployment Health & Data Drift Observability'}
        subtitle={
          isAr
            ? 'مراقبة استقرار الخدمة واكتشاف تغير توزيع المدخلات في الإنتاج (Covariate Shift).'
            : 'Track service health and detect distribution shift in real-world traffic.'
        }
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ background: 'var(--bg-card)', padding: '1.15rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{isAr ? 'حالة الخدمة التشغيلية' : 'Service Status'}</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent-teal)', marginTop: '0.35rem' }}>
              ● {isAr ? 'نشط ومستقر' : 'Active'}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem', fontFamily: 'var(--font-mono)' }}>
              Uptime: {activeDep.uptime}
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1.15rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{isAr ? 'الطلبات المخدومة (1h)' : 'Requests (1h)'}</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', marginTop: '0.35rem' }}>
              {activeDep.requestsLastHour}
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {isAr ? 'معدل تدفق طبيعي' : 'Nominal throughput'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1.15rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{isAr ? 'متوسط زمن الاستجابة' : 'Avg Latency'}</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', marginTop: '0.35rem' }}>
              {activeDep.avgLatencyMs} ms
            </div>
            <div style={{ fontSize: '0.72rem', color: '#2dd4bf', marginTop: '0.25rem' }}>
              {isAr ? 'ضمن المعيار المعتمد (≤ 50ms)' : 'Target Met'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1.15rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              <TermTooltip id="data_drift">
                {isAr ? 'مؤشر انحراف البيانات (PSI / KS)' : 'Data Drift Score'}
              </TermTooltip>
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)', marginTop: '0.35rem' }}>
              {activeDep.driftScore}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#2dd4bf', marginTop: '0.25rem' }}>
              {isAr ? 'انحراف ضئيل (توزيع مستقر)' : 'Stable Distribution'}
            </div>
          </div>
        </div>

        {/* Client Integration Code Snippets - Apple macOS Window Style */}
        <div
          style={{
            background: '#090d16',
            borderRadius: '10px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            overflow: 'hidden',
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.45)',
          }}
        >
          {/* Apple macOS Window Chrome Header */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.65rem 1rem',
              background: '#0e1422',
              borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            {/* Traffic Lights & Title */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ display: 'flex', gap: '6px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ff5f56', display: 'inline-block' }} />
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ffbd2e', display: 'inline-block' }} />
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#27c93f', display: 'inline-block' }} />
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600 }}>
                {activeSnippet.filename}
              </span>
            </div>

            {/* Language Tabs */}
            <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
              {[
                { key: 'python_api', label: 'Python (API)' },
                { key: 'javascript', label: 'JavaScript (Fetch)' },
                { key: 'curl', label: 'cURL (Bash)' },
                { key: 'batch_python', label: 'Batch Pipeline' },
                { key: 'edge_onnx', label: 'Edge ONNX' },
              ].map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveCodeTab(tab.key)}
                  style={{
                    padding: '0.25rem 0.65rem',
                    borderRadius: '4px',
                    fontSize: '0.74rem',
                    fontWeight: activeCodeTab === tab.key ? 700 : 500,
                    background: activeCodeTab === tab.key ? 'var(--accent-teal)' : 'rgba(255, 255, 255, 0.05)',
                    color: activeCodeTab === tab.key ? '#000000' : '#94a3b8',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Copy & Download Actions */}
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button
                onClick={handleCopyCode}
                className="btn btn-secondary btn-sm"
                style={{ fontSize: '0.72rem', padding: '0.25rem 0.65rem' }}
              >
                {isCopied ? (isAr ? 'تم النسخ' : 'Copied') : (isAr ? 'نسخ الشيفرة' : 'Copy Code')}
              </button>
              <button
                onClick={handleDownloadSnippet}
                className="btn btn-primary btn-sm"
                style={{
                  fontSize: '0.72rem',
                  padding: '0.25rem 0.65rem',
                  background: 'var(--accent-teal)',
                  color: '#000000',
                  fontWeight: 700,
                }}
              >
                {isAr ? `تنزيل (${activeSnippet.filename.split('.').pop()})` : `Download (${activeSnippet.filename.split('.').pop()})`}
              </button>
            </div>
          </div>

          {/* Gutter Line Numbers and Highlighted Code Body */}
          <div
            style={{
              padding: '1rem',
              overflowX: 'auto',
              maxHeight: '440px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.82rem',
              lineHeight: 1.6,
              direction: 'ltr',
              textAlign: 'left',
            }}
          >
            {activeSnippet.code.split('\n').map((line, idx) => (
              <div key={idx} style={{ display: 'flex', minWidth: 'fit-content' }}>
                <span
                  style={{
                    display: 'inline-block',
                    width: '3.2rem',
                    userSelect: 'none',
                    color: '#475569',
                    textAlign: 'right',
                    paddingRight: '1rem',
                    flexShrink: 0,
                  }}
                >
                  {idx + 1}
                </span>
                <span style={{ flex: 1, whiteSpace: 'pre' }}>
                  <HighlightedCodeLine line={line} />
                </span>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Module Navigation Footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '2.5rem',
          paddingTop: '1.5rem',
          borderTop: '1px solid var(--border-subtle)',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <button
          onClick={() => setActiveModule('export')}
          className="btn btn-secondary"
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <ArrowLeft size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
          <span>{isAr ? 'العودة إلى تصدير الأكواد (08)' : 'Back to Export (08)'}</span>
        </button>

        <button
          onClick={() => setActiveModule('continual')}
          className="btn btn-primary"
          style={{ background: 'var(--accent-teal)', color: '#ffffff', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <span>{isAr ? 'المتابعة إلى التعلم المستمر واستدامة المعرفة (10)' : 'Proceed to Continual Learning (10)'}</span>
          <ArrowRight size={16} style={{ transform: isAr ? 'scaleX(-1)' : 'none' }} />
        </button>
      </div>
    </div>
  );
}
