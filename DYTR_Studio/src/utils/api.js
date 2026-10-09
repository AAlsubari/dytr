// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// ==============================================================================
// DYTR Studio REST API Client Helper - Dual Mode Engine
// Priority: dytr Core Library via FastAPI (Primary)
// Fallback: Real-time Academic Simulation Engine (Secondary / Offline)
// ==============================================================================

const BASE_URL = 'http://localhost:8000/api/v1';

// Connection state listeners
let onConnectionChangeCallback = null;

export function registerConnectionListener(cb) {
  onConnectionChangeCallback = cb;
}

export async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1800);
    const res = await fetch(`${BASE_URL}/simulations/capabilities`, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (res.ok) {
      if (onConnectionChangeCallback) onConnectionChangeCallback(true);
      return true;
    }
  } catch (e) {
    // Backend offline or unreachable
  }
  if (onConnectionChangeCallback) onConnectionChangeCallback(false);
  return false;
}

export async function fetchApi(endpoint, options = {}) {
  try {
    const controller = new AbortController();
    const timeoutMs = options.timeout || 60000;
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const { timeout, ...fetchOptions } = options;
    const isFormData = typeof FormData !== 'undefined' && fetchOptions.body instanceof FormData;

    const res = await fetch(`${BASE_URL}${endpoint}`, {
      headers: {
        ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
        ...fetchOptions.headers,
      },
      signal: controller.signal,
      ...fetchOptions,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let errBody = '';
      try {
        const errJson = await res.json();
        errBody = errJson.message || errJson.detail || JSON.stringify(errJson);
      } catch {
        errBody = res.statusText;
      }
      return {
        status: 'error',
        error: `HTTP ${res.status}: ${errBody}`,
        message: `HTTP ${res.status}: ${errBody}`,
      };
    }
    const data = await res.json();
    if (onConnectionChangeCallback) onConnectionChangeCallback(true);
    return data;
  } catch (err) {
    if (onConnectionChangeCallback) onConnectionChangeCallback(false);
    return {
      status: 'error',
      error: err.name === 'AbortError'
        ? 'انتهت مهلة الاتصال بالخادم (Timeout). يرجى تقليل عدد الصفحات أو تسريع الاتصال.'
        : (err.message || 'تعذر الاتصال بالخادم'),
      message: err.name === 'AbortError'
        ? 'انتهت مهلة الاتصال بالخادم (Timeout). يرجى تقليل عدد الصفحات أو تسريع الاتصال.'
        : (err.message || 'تعذر الاتصال بالخادم'),
    };
  }
}

// ==============================================================================
// High-Fidelity Mathematical Simulation Fallback Generators
// Used seamlessly whenever the library backend is offline or unreachable
// ==============================================================================

function simulateRoPE(tokenIndex = 1, dim = 16, base = 10000.0) {
  const numPairs = Math.floor(dim / 2);
  const pairs = [];
  for (let i = 0; i < numPairs; i++) {
    const theta_i = 1.0 / Math.pow(base, (2 * i) / dim);
    const angle_rad = tokenIndex * theta_i;
    const angle_deg = (angle_rad * (180 / Math.PI)) % 360;

    const base_x = 1.0;
    const base_y = 0.0;
    const rot_x = base_x * Math.cos(angle_rad) - base_y * Math.sin(angle_rad);
    const rot_y = base_x * Math.sin(angle_rad) + base_y * Math.cos(angle_rad);

    pairs.push({
      pair_index: i,
      dim_indices: [2 * i, 2 * i + 1],
      theta_i: +(theta_i.toFixed(6)),
      angle_rad: +(angle_rad.toFixed(4)),
      angle_deg: +(angle_deg.toFixed(2)),
      vector_original: [base_x, base_y],
      vector_rotated: [+(rot_x.toFixed(4)), +(rot_y.toFixed(4))],
      norm_preserved: +(Math.sqrt(rot_x * rot_x + rot_y * rot_y).toFixed(4)),
    });
  }
  return {
    token_index: tokenIndex,
    total_dim: dim,
    num_pairs: numPairs,
    base_theta: base,
    pairs,
  };
}

function simulateMultiHeadAttention(tokens = ['Dynamic', 'transformers', 'adapt', 'continually'], numHeads = 4, embedDim = 256) {
  const headDim = Math.floor(embedDim / (numHeads || 4));
  const scale = (1 / Math.sqrt(headDim)).toFixed(4);
  return {
    num_heads: numHeads,
    embed_dim: embedDim,
    head_dim: headDim,
    scale_factor: scale,
    tokens: tokens || [],
  };
}

function simulateFlashAttention(seqLen = 1024, hiddenDim = 512, numHeads = 8, blockSize = 64) {
  const headDim = Math.floor(hiddenDim / numHeads);
  const bytesPerElem = 2; // FP16
  const standardMatrixBytes = seqLen * seqLen * bytesPerElem * numHeads;
  const standardHbmBytes =
    (seqLen * hiddenDim * 3 + seqLen * seqLen * numHeads + seqLen * hiddenDim) * bytesPerElem;
  const numBlocks = Math.ceil(seqLen / blockSize);
  const flashSramBytes = blockSize * headDim * 3 * bytesPerElem * numHeads;
  const flashHbmBytes = (seqLen * hiddenDim * 3 + seqLen * hiddenDim) * bytesPerElem;
  const hbmSavingsPercent = +(
    Math.max(0, (1.0 - flashHbmBytes / Math.max(1, standardHbmBytes)) * 100.0)
  ).toFixed(2);

  return {
    seq_len: seqLen,
    hidden_dim: hiddenDim,
    num_heads: numHeads,
    head_dim: headDim,
    block_size: blockSize,
    num_blocks_per_seq: numBlocks,
    standard_attention: {
      matrix_footprint_mb: +(standardMatrixBytes / (1024 * 1024)).toFixed(3),
      hbm_transfers_mb: +(standardHbmBytes / (1024 * 1024)).toFixed(3),
      complexity: `O(N^2) = ${(seqLen * seqLen).toLocaleString()} elements`,
    },
    flash_attention: {
      sram_footprint_kb: +(flashSramBytes / 1024).toFixed(2),
      hbm_transfers_mb: +(flashHbmBytes / (1024 * 1024)).toFixed(3),
      complexity: `O(N) = ${(seqLen * blockSize).toLocaleString()} block computations`,
    },
    hbm_memory_traffic_reduction_percent: hbmSavingsPercent,
    estimated_speedup_factor: +(Math.max(1.8, 4.2 * (seqLen / 512))).toFixed(1),
  };
}

function simulateContinualLearning(useEwc = true, lambdaVal = 1000.0, numSteps = 30) {
  const steps = Array.from({ length: numSteps }, (_, i) => i + 1);
  const half = Math.floor(numSteps / 2);
  const task1Acc = [];
  const task2Acc = [];
  const fisherDiagonals = [];

  for (let t = 1; t <= numSteps; t++) {
    if (t <= half) {
      const progress = t / half;
      const t1Val = 0.5 + 0.45 * (1.0 - Math.exp(-3 * progress));
      const t2Val = 0.25 + 0.05 * Math.sin(progress);
      const fisher = useEwc ? 0.8 * progress : 0.0;
      task1Acc.push(+(t1Val * 100.0).toFixed(2));
      task2Acc.push(+(t2Val * 100.0).toFixed(2));
      fisherDiagonals.push(+fisher.toFixed(3));
    } else {
      const progress = (t - half) / half;
      const t2Val = 0.25 + 0.65 * (1.0 - Math.exp(-3 * progress));
      let t1Val;
      let fisher;
      if (useEwc) {
        const decayResistance = Math.min(1.0, 0.95 - 0.05 / (lambdaVal / 500 + 0.1));
        t1Val = 0.95 * (1.0 - (1.0 - decayResistance) * progress);
        fisher = 0.8 + 0.7 * progress;
      } else {
        t1Val = 0.95 * Math.exp(-4.5 * progress);
        fisher = 0.0;
      }
      task1Acc.push(+(t1Val * 100.0).toFixed(2));
      task2Acc.push(+(t2Val * 100.0).toFixed(2));
      fisherDiagonals.push(+fisher.toFixed(3));
    }
  }

  const bwt = +(task1Acc[task1Acc.length - 1] - task1Acc[half - 1]).toFixed(2);
  const avgAcc = +((task1Acc[task1Acc.length - 1] + task2Acc[task2Acc.length - 1]) / 2.0).toFixed(2);

  return {
    use_ewc: useEwc,
    ewc_lambda: lambdaVal,
    steps,
    task1_accuracy: task1Acc,
    task2_accuracy: task2Acc,
    fisher_importance: fisherDiagonals,
    final_metrics: {
      task1_final: task1Acc[task1Acc.length - 1],
      task2_final: task2Acc[task2Acc.length - 1],
      bwt,
      average_acc: avgAcc,
      forgetting_prevented: bwt > -15.0,
    },
  };
}

function simulateAdaLoRA(initRank = 16, targetRank = 4, dIn = 128, dOut = 128) {
  const singularValues = [];
  const sensitivities = [];
  for (let i = 0; i < initRank; i++) {
    const sv = +(1.0 / (1.0 + 0.4 * i)).toFixed(4);
    singularValues.push(sv);
    sensitivities.push(+(sv * (0.8 + 0.4 * Math.sin(i))).toFixed(4));
  }

  const sortedIndices = Array.from({ length: initRank }, (_, i) => i).sort(
    (a, b) => sensitivities[b] - sensitivities[a]
  );
  const retainedIndices = new Set(sortedIndices.slice(0, targetRank));

  const denseParams = dIn * dOut;
  const initLoraParams = initRank * (dIn + dOut) + initRank;
  const prunedLoraParams = targetRank * (dIn + dOut) + targetRank;

  const paramReductionVsDense = +((1.0 - prunedLoraParams / denseParams) * 100.0).toFixed(2);
  const paramReductionVsInitLora = +((1.0 - prunedLoraParams / initLoraParams) * 100.0).toFixed(2);

  const ranksInfo = [];
  for (let i = 0; i < initRank; i++) {
    ranksInfo.push({
      rank_index: i + 1,
      singular_value: singularValues[i],
      importance_score: sensitivities[i],
      status: retainedIndices.has(i) ? 'retained' : 'pruned',
    });
  }

  return {
    d_in: dIn,
    d_out: dOut,
    init_rank: initRank,
    target_rank: targetRank,
    dense_params: denseParams,
    init_lora_params: initLoraParams,
    pruned_lora_params: prunedLoraParams,
    param_reduction_vs_dense_percent: paramReductionVsDense,
    param_reduction_vs_init_lora_percent: paramReductionVsInitLora,
    ranks: ranksInfo,
  };
}

function simulateTokenAlignment(text = 'Barack Obama visited the United Nations in New York', taskType = 'token_classification') {
  const words = text.trim().split(/\s+/);
  const tokens = [];
  let tokenIdx = 0;

  words.forEach((w) => {
    if (w.length > 5 && Math.random() > 0.4) {
      const p1 = w.slice(0, Math.floor(w.length / 2));
      const p2 = '##' + w.slice(Math.floor(w.length / 2));
      tokens.push({
        token_id: tokenIdx++,
        token_piece: p1,
        parent_word: w,
        is_subword_continuation: false,
        assigned_label: 'B-ENT',
      });
      tokens.push({
        token_id: tokenIdx++,
        token_piece: p2,
        parent_word: w,
        is_subword_continuation: true,
        assigned_label: '-100',
      });
    } else {
      tokens.push({
        token_id: tokenIdx++,
        token_piece: w,
        parent_word: w,
        is_subword_continuation: false,
        assigned_label: 'B-ENT',
      });
    }
  });

  return {
    text,
    task_type: taskType,
    tokens,
    total_tokens: tokens.length,
    word_count: words.length,
    loss_masked_tokens_count: tokens.filter((t) => t.assigned_label === '-100').length,
  };
}

// ==============================================================================
// Unified API Surface with Dual-Mode (Library Priority + Simulation Fallback)
// ==============================================================================

function simulateDatasetSearch(query = '', numResults = 10) {
  // Arabic keyword synonym map for expanding search queries
  const arabicKeywordMap = {
    'رسائل': ['sms', 'messages', 'text'],
    'مزعجة': ['spam', 'junk', 'unwanted'],
    'مزعجه': ['spam', 'junk', 'unwanted'],
    'سبام': ['spam'],
    'تصنيف': ['classification', 'categorization'],
    'احتيال': ['fraud', 'spam', 'scam'],
    'نصوص': ['text', 'nlp', 'corpus'],
    'مشاعر': ['sentiment', 'emotion', 'opinion'],
    'آراء': ['sentiment', 'opinion', 'reviews'],
    'سكري': ['diabetes', 'medical', 'health'],
    'سرطان': ['cancer', 'medical', 'health'],
    'طبي': ['medical', 'health', 'clinical'],
    'اخبار': ['news', 'articles', 'topics'],
    'أخبار': ['news', 'articles', 'topics'],
    'عربي': ['arabic', 'arab'],
    'عربية': ['arabic', 'arab'],
    'اقتصاد': ['economy', 'gdp', 'finance'],
    'مالي': ['finance', 'financial', 'banking'],
    'بنك': ['bank', 'banking', 'worldbank'],
    'بيانات': ['data', 'dataset'],
    'صور': ['images', 'vision'],
    'زهور': ['flowers', 'iris', 'botanical'],
    'قلب': ['heart', 'cardiac'],
    'مرض': ['disease', 'medical'],
    'عقارات': ['housing', 'real_estate', 'prices'],
  };

  function normalizeAr(text) {
    if (!text) return '';
    text = text.toLowerCase();
    text = text.replace(/[\u064B-\u0652\u0640]/g, '');
    text = text.replace(/[إأآٱ]/g, 'ا');
    text = text.replace(/ى/g, 'ي').replace(/ة/g, 'ه');
    return text.replace(/\s+/g, ' ').trim();
  }

  function expandKeywords(q) {
    const norm = normalizeAr(q);
    const rawTerms = norm.split(/\s+/).filter((t) => t.length >= 2);
    const expanded = new Set([...rawTerms, ...q.toLowerCase().split(/\s+/).filter((t) => t.length >= 2)]);
    for (const term of rawTerms) {
      for (const [arKey, enValues] of Object.entries(arabicKeywordMap)) {
        const arNorm = normalizeAr(arKey);
        if (term === arNorm || arNorm.includes(term) || term.includes(arNorm)) {
          enValues.forEach((v) => expanded.add(v));
        }
      }
    }
    return [...expanded];
  }

  const catalog = [
    {
      result_id: 'sim_sms_spam',
      title: 'SMS Spam Collection Dataset (UCI ML & Kaggle)',
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'مستودع أبحاث الاتصالات لجامعة كاليفورنيا - بيانات معيارية مصنفة لتصنيف الرسائل المزعجة SMS',
      platform_desc_en: 'Classic UCI & Kaggle public set of 5,574 labeled SMS messages (ham vs spam)',
      description: 'مجموعة بيانات معيارية تضم 5,574 رسالة نصية قصيرة SMS مصنفة بين رسائل شرعية (ham بنسبة 86.6%) ورسائل ضارة أو مزعجة (spam بنسبة 13.4%).',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv',
      external_url: 'https://archive.ics.uci.edu/ml/datasets/sms+spam+collection',
      is_direct_downloadable: true,
      estimated_rows: 5574,
      estimated_cols: 2,
      tags: ['sms', 'spam', 'رسائل', 'مزعجة', 'مزعجه', 'احتيال', 'تصنيف', 'نصوص', 'text', 'nlp', 'uci', 'kaggle', 'ham', 'classification', 'messages', 'junk', 'binary'],
    },
    {
      result_id: 'sim_arabic_sentiment',
      title: 'Arabic Sentiment Analysis Corpus (مجموعة بيانات المشاعر العربية)',
      platform: 'github',
      platform_name: 'Awesome Public Datasets (GitHub)',
      platform_desc_ar: 'قائمة ضخمة ومنظمة على منصة GitHub تضم روابط لبيانات عامة مصنفة حسب التخصصات',
      platform_desc_en: 'Curated GitHub Arabic NLP benchmarks for classification & sentiment',
      description: 'بيانات عربية مصنفة تعبيرياً (إيجابي، سلبي، محايد) لتدريب نماذج معالجة اللغة الطبيعية والتعلم العميق.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/komari6/Arabic-twitter-corpus-ASTD/master/ASTD-Balanced.csv',
      external_url: 'https://github.com/awesomedata/awesome-public-datasets',
      is_direct_downloadable: true,
      estimated_rows: 2000,
      estimated_cols: 4,
      tags: ['arabic', 'sentiment', 'nlp', 'text', 'tweets', 'مشاعر', 'عربي', 'عربية', 'emotion', 'opinion'],
    },
    {
      result_id: 'sim_iris',
      title: "Fisher's Iris Flower Botanical Dataset",
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'مستودع أكاديمي شهير وبسيط لجامعة كاليفورنيا، يوفر بيانات مجانية ومباشرة بدون الحاجة لتسجيل الدخول',
      platform_desc_en: 'Classic UCI botanical benchmarks for multi-class classification',
      description: 'بيانات سوسن أيريس المعيارية الشهيرة لتصنيف الفئات الثلاث (Setosa, Versicolor, Virginica) مع أبعاد السبلات والبتلات.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
      external_url: 'https://archive.ics.uci.edu/ml/datasets/iris',
      is_direct_downloadable: true,
      estimated_rows: 150,
      estimated_cols: 5,
      tags: ['iris', 'flowers', 'classification', 'botanical', 'زهور', 'multi-class', 'sepal', 'petal'],
    },
    {
      result_id: 'sim_titanic',
      title: 'Titanic Survival Passenger Demographics',
      platform: 'github',
      platform_name: 'Awesome Public Datasets (GitHub)',
      platform_desc_ar: 'مستودع مفتوح لمجموعات البيانات المعيارية التعليمية والأكاديمية',
      platform_desc_en: 'Classic passenger demographic dataset for binary survival prediction',
      description: 'بيانات ركاب سفينة تيتانيك مع الخصائص الديموغرافية (السن، الجنس، الفئة، التذكرة، النجاة).',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
      external_url: 'https://github.com/awesomedata/awesome-public-datasets',
      is_direct_downloadable: true,
      estimated_rows: 891,
      estimated_cols: 12,
      tags: ['titanic', 'survival', 'demographics', 'binary', 'passengers', 'تايتانك'],
    },
    {
      result_id: 'sim_breast_cancer',
      title: 'Breast Cancer Wisconsin Diagnostic Dataset',
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'بيانات الفحص الطبي لسرطان الثدي لمعايرة نماذج التصنيف التشخيصي',
      platform_desc_en: 'UCI Wisconsin Diagnostic Breast Cancer (WDBC) benchmark dataset',
      description: 'بيانات التشخيص الطبي لسرطان الثدي (569 عينة مع 30 خاصية نسيجية وإشعاعية).',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv',
      external_url: 'https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)',
      is_direct_downloadable: true,
      estimated_rows: 569,
      estimated_cols: 32,
      tags: ['cancer', 'breast', 'medical', 'uci', 'health', 'classification', 'سرطان', 'طبي', 'diagnostic'],
    },
    {
      result_id: 'sim_wine_quality',
      title: 'UCI Wine Quality Benchmark Dataset',
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'بيانات الجودة والخواص الفيزيوكيميائية للتحليل والانحدار',
      platform_desc_en: 'UCI Wine Quality physicochemical benchmark dataset',
      description: 'بيانات جودة الخمور مع الخواص الكيميائية والفيزيائية (الحمضية، السكر، الكحول، التقييم).',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv',
      external_url: 'https://archive.ics.uci.edu/ml/datasets/Wine+Quality',
      is_direct_downloadable: true,
      estimated_rows: 4898,
      estimated_cols: 12,
      tags: ['wine', 'quality', 'regression', 'uci', 'tabular', 'chemistry'],
    },
    {
      result_id: 'sim_pima_diabetes',
      title: 'PIMA Indians Diabetes Prediction Dataset',
      platform: 'kaggle',
      platform_name: 'Kaggle Datasets',
      platform_desc_ar: 'بيانات السكري من المعهد الوطني لأمراض الهضم والسكري',
      platform_desc_en: 'Kaggle & UCI PIMA Diabetes Diagnostic Dataset',
      description: 'سجلات التنبؤ بمرض السكري بناءً على المؤشرات الطبية (الجلوكوز، ضغط الدم، الأنسولين، العمر).',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv',
      external_url: 'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database',
      is_direct_downloadable: true,
      estimated_rows: 768,
      estimated_cols: 9,
      tags: ['diabetes', 'pima', 'medical', 'kaggle', 'health', 'binary', 'سكري', 'طبي', 'glucose'],
    },
    {
      result_id: 'sim_customer_churn',
      title: 'Telco Customer Churn Benchmark Dataset',
      platform: 'kaggle',
      platform_name: 'Kaggle Datasets',
      platform_desc_ar: 'بيانات التنبؤ بتسرب العملاء وانتقالهم في قطاع الاتصالات',
      platform_desc_en: 'Kaggle Telco Customer Churn Prediction Dataset',
      description: 'بيانات التنبؤ بتسرب العملاء وانتقالهم في قطاع الاتصالات بناءً على العقود والفواتير.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv',
      external_url: 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn',
      is_direct_downloadable: true,
      estimated_rows: 7043,
      estimated_cols: 21,
      tags: ['churn', 'telecom', 'customer', 'business', 'kaggle', 'retention'],
    },
    {
      result_id: 'sim_credit_fraud',
      title: 'Credit Card Fraud Detection Dataset',
      platform: 'kaggle',
      platform_name: 'Kaggle Datasets',
      platform_desc_ar: 'بيانات كشف الاحتيال في البطاقات الائتمانية مع متجهات الخصائص الموجهة',
      platform_desc_en: 'Kaggle Anonymized Credit Card Fraud Transactions Dataset',
      description: 'بيانات كشف الاحتيال في البطاقات الائتمانية مع متجهات الخصائص الموجهة (PCA V1..V28).',
      format: 'CSV',
      download_url: null,
      external_url: 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
      is_direct_downloadable: false,
      estimated_rows: 284807,
      estimated_cols: 31,
      tags: ['fraud', 'creditcard', 'banking', 'kaggle', 'finance', 'احتيال', 'مالي', 'anomaly'],
    },
    {
      result_id: 'sim_arabic_news',
      title: 'Arabic News Multi-Class Corpus (مجموعة بيانات الأخبار العربية المصنفة)',
      platform: 'github',
      platform_name: 'Awesome Public Datasets (GitHub)',
      platform_desc_ar: 'بيانات مقالات أخبار عربية مقسمة ومصنفة على 5 مجالات معرفية',
      platform_desc_en: 'Curated Arabic News Articles Multi-Class Benchmark Dataset',
      description: 'مقالات وأخبار عربية مقسمة ومصنفة على 5 مجالات معرفية (السياسة، الاقتصاد، التقنية، الرياضة، الصحة).',
      format: 'CSV',
      download_url: null,
      external_url: 'https://github.com/topics/arabic-dataset',
      is_direct_downloadable: false,
      estimated_rows: 5000,
      estimated_cols: 5,
      tags: ['arabic', 'news', 'nlp', 'text', 'classification', 'اخبار', 'أخبار', 'عربي', 'topics'],
    },
    {
      result_id: 'sim_worldbank_gdp',
      title: 'World Bank Global Economic GDP Indicators',
      platform: 'worldbank',
      platform_name: 'World Bank Data (البنك الدولي)',
      platform_desc_ar: 'يوفر مؤشرات اقتصادية وتنموية واجتماعية شاملة ومباشرة لدول العالم',
      platform_desc_en: 'World Bank Official Indicator Data for global economic trends',
      description: 'مؤشرات الناتج المحلي الإجمالي والنمو الاقتصادي ومعدلات التضخم لجميع دول العالم عبر السنوات.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv',
      external_url: 'https://data.worldbank.org/',
      is_direct_downloadable: true,
      estimated_rows: 11500,
      estimated_cols: 4,
      tags: ['economy', 'gdp', 'worldbank', 'finance', 'اقتصاد', 'بنك', 'مالي', 'indicators'],
    },
    {
      result_id: 'sim_housing',
      title: 'Boston / California Housing Prices Dataset',
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'بيانات عقارية مرجعية لنماذج التنبؤ بأسعار المنازل',
      platform_desc_en: 'Classic housing price prediction dataset for regression tasks',
      description: 'بيانات أسعار المنازل مع متغيرات اقتصادية واجتماعية وجغرافية للتنبؤ بأسعار العقارات.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
      external_url: 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/',
      is_direct_downloadable: true,
      estimated_rows: 506,
      estimated_cols: 14,
      tags: ['housing', 'prices', 'regression', 'real_estate', 'boston', 'عقارات', 'أسعار'],
    },
    {
      result_id: 'sim_heart_disease',
      title: 'Heart Disease Clinical Records Dataset',
      platform: 'uci',
      platform_name: 'UCI Machine Learning Repository',
      platform_desc_ar: 'بيانات سريرية للتنبؤ بأمراض القلب والأوعية الدموية',
      platform_desc_en: 'Clinical records for heart failure prediction',
      description: 'بيانات مرضى القلب والأوعية الدموية مع المؤشرات السريرية (الضغط، الكوليسترول، السكر، العمر) للتنبؤ بالوفاة.',
      format: 'CSV',
      download_url: 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart-disease.csv',
      external_url: 'https://archive.ics.uci.edu/ml/datasets/Heart+Disease',
      is_direct_downloadable: true,
      estimated_rows: 303,
      estimated_cols: 14,
      tags: ['heart', 'disease', 'medical', 'uci', 'health', 'clinical', 'قلب', 'مرض', 'طبي'],
    },
    {
      result_id: 'sim_kaggle_search',
      title: 'Kaggle Global NLP & Sentiment Multi-Language Corpus',
      platform: 'kaggle',
      platform_name: 'Kaggle Datasets',
      platform_desc_ar: 'أشهر منصة عالمية تضم آلاف مجموعات البيانات المجانية والمجتمعية',
      platform_desc_en: "World's largest community platform for machine learning & data science",
      description: 'مجموعة بيانات ضخمة على منصة Kaggle لتحليل الآراء والنصوص متعددة اللغات مع مشاريع جاهزة.',
      format: 'Kaggle Page',
      download_url: null,
      external_url: `https://www.kaggle.com/datasets?search=${encodeURIComponent(query)}`,
      is_direct_downloadable: false,
      estimated_rows: 50000,
      estimated_cols: 6,
      tags: ['kaggle', 'nlp', 'sentiment', 'text', 'community'],
    },
  ];

  const keywords = expandKeywords(query);
  if (keywords.length === 0) return { status: 'success', query, results_count: catalog.length, results: catalog.slice(0, numResults) };

  // Score each item
  const scored = catalog.map((item) => {
    const titleNorm = normalizeAr(item.title);
    const descNorm = normalizeAr(item.description);
    const tagsNorm = item.tags.map((t) => normalizeAr(t));
    let score = 0;
    for (const kw of keywords) {
      const kwNorm = normalizeAr(kw);
      if (!kwNorm) continue;
      if (tagsNorm.includes(kwNorm)) score += 15;
      else if (tagsNorm.some((t) => t.includes(kwNorm) || kwNorm.includes(t))) score += 10;
      if (titleNorm.includes(kwNorm)) score += 12;
      if (descNorm.includes(kwNorm)) score += 5;
    }
    return { score, item };
  });

  scored.sort((a, b) => b.score - a.score);
  const matched = scored.filter((s) => s.score > 0).map((s) => s.item);
  const finalResults = matched.length > 0 ? matched : catalog;

  return {
    status: 'success',
    query,
    results_count: finalResults.length,
    results: finalResults.slice(0, numResults),
  };
}

export const api = {
  checkHealth: checkBackendHealth,

  getCapabilities: async () => {
    const res = await fetchApi('/simulations/capabilities');
    if (res) return res;
    return {
      dytr_version: '0.1.3',
      dytr_available: false,
      cuda_available: false,
      device_name: 'Academic Simulation Host (CPU)',
      vram_total_gb: 0.0,
      gated_gelu_ffn_supported: true,
      rope_supported: true,
      amp_supported: true,
      bf16_supported: false,
      torch_version: '2.4.0 (Simulated)',
    };
  },

  getRoPESim: async (params = {}) => {
    const p = new URLSearchParams(params);
    const res = await fetchApi(`/simulations/rope?${p}`);
    if (res) return res;
    return simulateRoPE(
      parseInt(params.token_index ?? 1),
      parseInt(params.dim ?? 16),
      parseFloat(params.base ?? 10000.0)
    );
  },

  getMultiHeadAttentionSim: async (params = {}) => {
    return simulateMultiHeadAttention(params.tokens, params.num_heads, params.embed_dim);
  },

  getContinualSim: async (params = {}) => {
    const p = new URLSearchParams(params);
    const res = await fetchApi(`/simulations/continual-learning?${p}`);
    if (res) return res;
    return simulateContinualLearning(
      params.use_ewc !== false && params.use_ewc !== 'false',
      parseFloat(params.ewc_lambda ?? 1000.0),
      parseInt(params.num_steps ?? 30)
    );
  },

  getAdaLoRASim: async (params = {}) => {
    const p = new URLSearchParams(params);
    const res = await fetchApi(`/simulations/adalora-svd?${p}`);
    if (res) return res;
    return simulateAdaLoRA(
      parseInt(params.init_rank ?? 16),
      parseInt(params.target_rank ?? 4),
      parseInt(params.d_in ?? 128),
      parseInt(params.d_out ?? 128)
    );
  },

  postTokenizeAlign: async (data = {}) => {
    const res = await fetchApi('/simulations/tokenize-align', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    if (res) return res;
    return simulateTokenAlignment(data.text, data.task_type);
  },

  getModelPresets: async () => {
    return fetchApi('/models/presets');
  },

  estimateModel: async (data) => {
    return fetchApi('/models/estimate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  getPrebuiltDatasets: async () => {
    return fetchApi('/datasets/prebuilt');
  },

  getTrainingStatus: async () => {
    return fetchApi('/training/status');
  },

  startTraining: async (data) => {
    return fetchApi('/training/start', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  stopTraining: async () => {
    return fetchApi('/training/stop', { method: 'POST' });
  },

  predict: async (data) => {
    return fetchApi('/inference/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  runEvaluation: async (data) => {
    return fetchApi('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  simulateTokenization: async (text) => {
    const res = await fetchApi('/simulations/tokenize-align', {
      method: 'POST',
      body: JSON.stringify({ text, task_type: 'token_classification' }),
    });
    if (res) return res;
    return simulateTokenAlignment(text, 'token_classification');
  },

  simulateContinualLearning: async (params = {}) => {
    let query = '';
    if (typeof params === 'object') {
      const q = new URLSearchParams();
      if (params.useEWC !== undefined) q.append('use_ewc', params.useEWC);
      if (params.ewcLambda !== undefined) q.append('ewc_lambda', params.ewcLambda);
      if (params.useReplay !== undefined) q.append('use_replay', params.useReplay);
      if (params.replayBufferSize !== undefined) q.append('replay_buffer_size', params.replayBufferSize);
      if (params.numSteps !== undefined) q.append('num_steps', params.numSteps);
      query = `?${q.toString()}`;
    } else {
      query = `?use_ewc=${Boolean(params)}`;
    }
    const res = await fetchApi(`/simulations/continual-learning${query}`);
    if (res && res.steps) return res;
    return simulateContinualLearning(params.useEWC ?? true, (params.ewcLambda || 5000) / 1000.0, params.numSteps || 30);
  },

  startRealContinualTraining: async (payload = {}) => {
    return fetchApi('/training/continual/start', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 120000,
    });
  },

  getCheckpoints: async () => {
    return fetchApi('/training/checkpoint/list');
  },

  loadCheckpoint: async (checkpointType = 'best', path = null) => {
    return fetchApi('/training/checkpoint/load', {
      method: 'POST',
      body: JSON.stringify({ checkpoint_type: checkpointType, path }),
    });
  },

  uploadCheckpoint: async (formData) => {
    return fetchApi('/training/checkpoint/upload', {
      method: 'POST',
      body: formData,
      timeout: 120000,
    });
  },

  getConcept: (name, lang = 'ar') =>
    fetchApi(`/learn/concept/${name}?lang=${lang}`),
  getAllConcepts: (lang = 'ar', category = null) =>
    fetchApi(`/learn/concepts?lang=${lang}${category ? `&category=${category}` : ''}`),
  translateMetric: (metricName, value, lang = 'ar') =>
    fetchApi('/learn/metric', {
      method: 'POST',
      body: JSON.stringify({ metric_name: metricName, value, lang }),
    }),
  getLearningPath: (track, lang = 'ar') =>
    fetchApi(`/learn/path/${track}?lang=${lang}`),
  suggestNextStep: (state) =>
    fetchApi('/learn/suggest', {
      method: 'POST',
      body: JSON.stringify(state),
    }),
  scrapeWebPage: async (urlOrPayload, extractMode = 'auto', options = {}) => {
    let body = {};
    if (typeof urlOrPayload === 'object' && urlOrPayload !== null) {
      body = urlOrPayload;
    } else {
      body = { url: urlOrPayload, extract_mode: extractMode, ...options };
    }
    return fetchApi('/datasets/scrape', {
      method: 'POST',
      body: JSON.stringify(body),
      timeout: 60000,
    });
  },
  inferUrlPattern: async (payload) => {
    return fetchApi('/datasets/infer_url_pattern', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 30000,
    });
  },
  scrapeMultiPage: async (payload) => {
    return fetchApi('/datasets/scrape_multipage', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 180000,
    });
  },
  searchWeb: async (query, numResults = 10, platforms = null) => {
    const body = { query, num_results: numResults };
    if (platforms && platforms.length > 0 && !platforms.includes('all')) {
      body.platforms = platforms;
    }
    const res = await fetchApi('/datasets/search', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    if (res && res.status === 'success' && res.results) return res;
    if (res && res.error) return res;
    return {
      status: 'error',
      error: 'تعذر الاتصال بخدمة البحث المباشر في السيرفر (FastAPI / Port 8000). يرجى التأكد من تشغيل الخادم.',
      results: [],
    };
  },
  fetchRemoteDataset: async (url, format = 'auto') => {
    return fetchApi('/datasets/fetch_remote', {
      method: 'POST',
      body: JSON.stringify({ url, format }),
    });
  },
};
