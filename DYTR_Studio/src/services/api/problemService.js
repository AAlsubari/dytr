// ==============================================================================
// DYTR Studio Problem Framing API Service
// Bridges React UI with Mock / FastAPI backend endpoints
// ==============================================================================

import { apiClient } from './apiClient';

// Offline static fallback data to ensure resilient continuous operation
const FALLBACK_SCENARIOS = [
  {
    id: 'sentiment',
    taskType: 'SENTENCE_CLASSIFICATION',
    title_ar: 'تصنيف المشاعر (Sentiment)',
    title_en: 'Sentiment Analysis',
    text_ar: 'تصنيف مشاعر تعليقات العملاء إلى إيجابية وسلبية',
    text_en: 'Binary sentiment classification of customer reviews',
  },
  {
    id: 'ner',
    taskType: 'TOKEN_CLASSIFICATION',
    title_ar: 'استخراج الكيانات (NER)',
    title_en: 'Named Entity Extraction',
    text_ar: 'استخراج أسماء الأشخاص والأماكن والشركات من النصوص الإخبارية',
    text_en: 'Named Entity Recognition (NER) in news articles',
  },
  {
    id: 'causal_lm',
    taskType: 'CAUSAL_LM',
    title_ar: 'مساعد محادثة (Causal LM)',
    title_en: 'Conversational AI',
    text_ar: 'بناء مساعد ذكي يجيب على أسئلة الطلاب في المنهاج الأكاديمي',
    text_en: 'Conversational QA bot for student queries',
  },
  {
    id: 'seq2seq',
    taskType: 'SEQ2SEQ',
    title_ar: 'ترجمة تسلسلية (Seq2Seq)',
    title_en: 'Seq2Seq Translation',
    text_ar: 'ترجمة المقالات التقنية من الإنجليزية إلى العربية',
    text_en: 'Neural Machine Translation from English to Arabic',
  },
];

const FALLBACK_QUESTIONS = [
  {
    id: 'fixedRules',
    order: 1,
    question_ar: 'هل قواعد اتخاذ القرار في مسألتك ثابتة وواضحة تماما؟',
    question_en: 'Are your decision logic and rules completely fixed and explicit?',
    options: [
      {
        value: 'yes',
        label_ar: 'نعم (أعرف القواعد المنطقية بدقة، مثل: إذا المبلغ > 1000)',
        label_en: 'Yes (Explicit deterministic logic)',
      },
      {
        value: 'no',
        label_ar: 'لا (القواعد غير واضحة وتعتمد على السياق واللغة)',
        label_en: 'No (Contextual, implicit or fuzzy logic)',
      },
    ],
  },
  {
    id: 'sufficientData',
    order: 2,
    question_ar: 'هل تتوفر لديك بيانات كافية تحتوي على أمثلة وحلولها؟',
    question_en: 'Do you have labeled historical training datasets?',
    options: [
      {
        value: 'yes',
        label_ar: 'نعم (مئات أو آلاف الأمثلة المصنفة)',
        label_en: 'Yes (>100 labeled samples)',
      },
      {
        value: 'unlabeled',
        label_ar: 'لدي نصوص غير مصنفة أو سأجمعها',
        label_en: 'Unlabeled texts / Planning to scrape',
      },
      {
        value: 'no',
        label_ar: 'لا تتوفر بيانات حاليا',
        label_en: 'No data available',
      },
    ],
  },
  {
    id: 'complexPattern',
    order: 3,
    question_ar: 'ما مدى تعقيد الأنماط اللغوية أو الإحصائية المطلوبة؟',
    question_en: 'How complex are the underlying linguistic patterns?',
    options: [
      {
        value: 'simple',
        label_ar: 'بسيطة (مطابقة كلمات محددة)',
        label_en: 'Simple keyword lookups',
      },
      {
        value: 'complex',
        label_ar: 'معقدة (فهم السخرية، السياق، المعنى الضمني)',
        label_en: 'Complex semantics & syntax',
      },
    ],
  },
];

const FALLBACK_TASK_TYPES = [
  {
    id: 'SENTENCE_CLASSIFICATION',
    title_ar: 'تصنيف النصوص (Sentence Classification)',
    title_en: 'Sentence Classification',
    badge_ar: 'تصنيف عام',
    badge_en: 'Sequence Level',
    desc_ar: 'إسناد تصنيف محدد للنص ككل. مثال: تحليل المشاعر (إيجابي/سلبي)، وتصنيف مواضيع الأخبار.',
    desc_en: 'Predict a categorical label for the entire text sequence (e.g. Sentiment, Topic, Intent).',
    example_ar: 'المدخل: "الخدمة ممتازة" -> المخرج: إيجابي',
    example_en: 'Input: "Great service" -> Output: Positive',
  },
  {
    id: 'TOKEN_CLASSIFICATION',
    title_ar: 'تصنيف الرموز والكيانات (Token Classification / NER)',
    title_en: 'Token Classification (NER)',
    badge_ar: 'تصنيف تفصيلي',
    badge_en: 'Token Level',
    desc_ar: 'إسناد وسم لكل كلمة أو رمز في النص. مثال: استخراج أسماء الأشخاص والأماكن والشركات (NER).',
    desc_en: 'Assign semantic tags to individual tokens (e.g., Named Entity Recognition BIO tags).',
    example_ar: 'المدخل: "زار أحمد الرياض" -> أحمد: شخص، الرياض: مكان',
    example_en: 'Input: "Turing visited London" -> Turing: Person',
  },
  {
    id: 'CAUSAL_LM',
    title_ar: 'توليد النصوص والمحادثة (Causal LM / Generation)',
    title_en: 'Causal Language Modeling',
    badge_ar: 'توليد تسلسلي',
    badge_en: 'Autoregressive',
    desc_ar: 'توقع الكلمة التالية لإنشاء محادثة أو إكمال نصوص ذكية وفق سياق موجه.',
    desc_en: 'Autoregressively predict subsequent tokens for chatbots and creative text generation.',
    example_ar: 'المدخل: "عاصمة المملكة العربية السعودية هي" -> المخرج: "الرياض"',
    example_en: 'Input: "The capital of France is" -> Output: "Paris"',
  },
  {
    id: 'SEQ2SEQ',
    title_ar: 'تحويل التسلسلات (Seq2Seq / Transformation)',
    title_en: 'Sequence-to-Sequence',
    badge_ar: 'تحويل نصي',
    badge_en: 'Encoder-Decoder',
    desc_ar: 'تحويل نص مدخل من شكل إلى نص مخرج آخر. مثال: الترجمة الآلية وتلخيص المقالات.',
    desc_en: 'Transform input sequence into target sequence (Machine Translation, Abstractive Summarization).',
    example_ar: 'المدخل: "Hello World" -> المخرج: "مرحبا بالعالم"',
    example_en: 'Input: "Hello World" -> Output: "مرحبا بالعالم"',
  },
];

const FALLBACK_MODELS = [
  { id: 'custom_dytr', name_ar: 'DynamicTransformer (من الصفر)', name_en: 'DynamicTransformer (From Scratch)', recommended: true },
  { id: 'bert', name_ar: 'BERT Pretrained (مشفر مسبق التدريب)', name_en: 'BERT Pretrained Encoder', recommended: false },
  { id: 'roberta', name_ar: 'RoBERTa Backbone', name_en: 'RoBERTa Backbone', recommended: false },
  { id: 'gpt2', name_ar: 'GPT-2 Tiny Causal Backbone', name_en: 'GPT-2 Tiny Causal', recommended: false },
  { id: 't5', name_ar: 'T5 Seq2Seq Encoder-Decoder', name_en: 'T5 Seq2Seq', recommended: false },
];

export const problemService = {
  /**
   * Fetches problem scenario templates from the API with resilience fallback.
   */
  async getScenarios() {
    try {
      return await apiClient.get('/problem/scenarios');
    } catch (e) {
      console.warn('Using offline problem scenarios fallback:', e.message);
      return FALLBACK_SCENARIOS;
    }
  },

  /**
   * Fetches AI feasibility diagnostic questions.
   */
  async getDiagnosticQuestions() {
    try {
      return await apiClient.get('/problem/diagnostic-questions');
    } catch (e) {
      console.warn('Using offline diagnostic questions fallback:', e.message);
      return FALLBACK_QUESTIONS;
    }
  },

  /**
   * Fetches supported neural task architectures.
   */
  async getTaskTypes() {
    try {
      return await apiClient.get('/problem/task-types');
    } catch (e) {
      console.warn('Using offline task types fallback:', e.message);
      return FALLBACK_TASK_TYPES;
    }
  },

  /**
   * Fetches supported foundation model backbones.
   */
  async getFoundationModels() {
    try {
      return await apiClient.get('/problem/foundation-models');
    } catch (e) {
      console.warn('Using offline foundation models fallback:', e.message);
      return FALLBACK_MODELS;
    }
  },

  /**
   * Suggests a task type based on keyword analysis of the user's problem description.
   */
  suggestTaskType(statementText = '') {
    const t = statementText.toLowerCase();
    if (t.includes('تصنيف') || t.includes('مشاعر') || t.includes('sentiment') || t.includes('classify')) {
      return 'SENTENCE_CLASSIFICATION';
    }
    if (t.includes('استخراج') || t.includes('أسماء') || t.includes('ner') || t.includes('token') || t.includes('كيانات')) {
      return 'TOKEN_CLASSIFICATION';
    }
    if (t.includes('توليد') || t.includes('شات') || t.includes('بوت') || t.includes('chat') || t.includes('generate')) {
      return 'CAUSAL_LM';
    }
    if (t.includes('ترجمة') || t.includes('تلخيص') || t.includes('seq2seq') || t.includes('translate') || t.includes('summarize')) {
      return 'SEQ2SEQ';
    }
    return 'SENTENCE_CLASSIFICATION';
  },

  /**
   * Evaluates AI necessity diagnostic answers.
   */
  evaluateNecessity(answers = {}) {
    const { fixedRules, sufficientData, complexPattern } = answers;
    if (fixedRules === 'yes' && complexPattern === 'simple') {
      return 'not_needed';
    }
    if (sufficientData === 'no' || sufficientData === 'unlabeled') {
      return 'collect_data';
    }
    if (fixedRules === 'unsure') {
      return 'start_academy';
    }
    return 'ai_needed';
  },
};
