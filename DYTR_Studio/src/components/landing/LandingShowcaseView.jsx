// ==============================================================================
// DYTR Studio Showcase & Marketing Landing Experience
// Pure Product Marketing & High-Conversion Copy (Zero Technical Jargon)
// Dynamic Rotating Headlines + Illustrated AI Mockups + Featured Projects + Quotes + FAQ
// Bilingual RTL/LTR - Design Token Compliant - Zero Emojis Policy
// ==============================================================================

import React, { useState, useEffect } from 'react';
import { useStudioStore } from '../../store/studioStore';
import { useDatasetStore } from '../../store/useDatasetStore';
import { getSampleDatasetById } from '../../lib/data/synthetic';
import DytrLogo from '../common/DytrLogo';

// 4-Pointed Spark / Brand Icon 
function BrandSparkLogo({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" style={{ display: 'block', flexShrink: 0 }}>
      <defs>
        <linearGradient id="brandGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0EA5E9" />
          <stop offset="50%" stopColor="#38BDF8" />
          <stop offset="100%" stopColor="#0284C7" />
        </linearGradient>
      </defs>
      <path
        d="M12 0C12 6.627 6.627 12 0 12C6.627 12 12 17.373 12 24C12 17.373 17.373 12 24 12C17.373 12 12 6.627 12 0Z"
        fill="url(#brandGrad)"
      />
    </svg>
  );
}

// Minimalist Vector Icons (No Emojis)
function DatasetVectorIcon({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#0EA5E9" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" fill="rgba(14, 165, 233, 0.12)" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}

function ModelVectorIcon({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" fill="rgba(16, 185, 129, 0.12)" />
      <rect x="9" y="9" width="6" height="6" fill="rgba(16, 185, 129, 0.25)" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  );
}

function EvaluationVectorIcon({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#F59E0B" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" fill="rgba(245, 158, 11, 0.12)" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function MemoryVectorIcon({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#0D9488" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" fill="rgba(13, 148, 136, 0.12)" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}

function ArrowUpRightIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="7" y1="17" x2="17" y2="7" />
      <polyline points="7 7 17 7 17 17" />
    </svg>
  );
}

function SunVectorIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonVectorIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

function GlobeVectorIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

export default function LandingShowcaseView() {
  const { language, setLanguage, theme, toggleTheme, setActiveModule } = useStudioStore();
  const isAr = language === 'ar';

  // 1. Rotating Hero Slides (Pure AI Product Marketing - Dynamic & Clear)
  const slides = isAr
    ? [
        {
          id: 'data_to_ai',
          prefix: 'حوّل بياناتك وجداولك إلى',
          highlight: 'نماذج ذكاء اصطناعي ذكية',
          color: '#0EA5E9',
          illustrationType: 'pipeline',
        },
        {
          id: 'effortless_learning',
          prefix: 'تعلّم ودرّب الذكاء الاصطناعي',
          highlight: 'بكل سهولة وبدون تعقيد',
          color: '#10B981',
          illustrationType: 'training',
        },
        {
          id: 'fair_evaluation',
          prefix: 'اختبر نموذجك بأمان ومصداقية على',
          highlight: 'بيانات التقييم المستقلة المقسمة',
          color: '#F59E0B',
          illustrationType: 'evaluation',
        },
        {
          id: 'continual_memory',
          prefix: 'علّم نموذجك مهارات جديدة',
          highlight: 'بذاكرة مستمرة وخصوصية تامة',
          color: '#0D9488',
          illustrationType: 'shield',
        },
      ]
    : [
        {
          id: 'data_to_ai',
          prefix: 'Turn your files & data into',
          highlight: 'Production-Ready AI Models',
          color: '#0EA5E9',
          illustrationType: 'pipeline',
        },
        {
          id: 'effortless_learning',
          prefix: 'Learn and train your AI models',
          highlight: 'Effortlessly without barriers',
          color: '#10B981',
          illustrationType: 'training',
        },
        {
          id: 'fair_evaluation',
          prefix: 'Validate your model fairly on',
          highlight: 'Real Split Evaluation Data',
          color: '#F59E0B',
          illustrationType: 'evaluation',
        },
        {
          id: 'continual_memory',
          prefix: 'Teach your model new tasks with',
          highlight: 'Permanent Memory & Full Privacy',
          color: '#0D9488',
          illustrationType: 'shield',
        },
      ];

  // 2. Featured Public Projects (Real Built-in Working Benchmarks)
  const featuredProjects = isAr
    ? [
        {
          id: 'sms_spam_detection',
          datasetKey: 'sms_spam_detection',
          title: 'كشف الرسائل المزعجة وتصنيف النصوص',
          sources: '5,160 رسالة حقيقية (sms.tsv) · تصنيف فوري',
          icon: DatasetVectorIcon,
          targetStage: 'dataprep',
        },
        {
          id: 'arabic_sentiment',
          datasetKey: 'arabic_sentiment',
          title: 'تحليل مشاعر وتقييمات العملاء',
          sources: 'مراجعات عربية حقيقية · جاهز للتدريب',
          icon: ModelVectorIcon,
          targetStage: 'builder',
        },
        {
          id: 'arabic_news_classification',
          datasetKey: 'arabic_news_classification',
          title: 'تصنيف وتلخيص الأخبار والمقالات',
          sources: '100 مقالة حقيقية · 4 مجالات معرفية',
          icon: EvaluationVectorIcon,
          targetStage: 'arena',
        },
        {
          id: 'continual_learning_model',
          datasetKey: 'arabic_sentiment',
          title: 'التعلم التتابعي مع درع الذاكرة الذكي',
          sources: 'إضافة مهام جديدة دون نسيان القديم',
          icon: MemoryVectorIcon,
          targetStage: 'continual',
        },
      ]
    : [
        {
          id: 'sms_spam_detection',
          datasetKey: 'sms_spam_detection',
          title: 'SMS Spam & Text Classification',
          sources: '5,160 real samples (sms.tsv) · Ready to train',
          icon: DatasetVectorIcon,
          targetStage: 'dataprep',
        },
        {
          id: 'arabic_sentiment',
          datasetKey: 'arabic_sentiment',
          title: 'Customer Sentiment & Reviews Analysis',
          sources: 'Real-world customer feedback · Multi-class',
          icon: ModelVectorIcon,
          targetStage: 'builder',
        },
        {
          id: 'arabic_news_classification',
          datasetKey: 'arabic_news_classification',
          title: 'News & Topic Article Classifier',
          sources: '100 labeled articles · 4 categories',
          icon: EvaluationVectorIcon,
          targetStage: 'arena',
        },
        {
          id: 'continual_learning_model',
          datasetKey: 'arabic_sentiment',
          title: 'Continual Learning & Memory Shield',
          sources: 'Sequential tasks · Lifelong memory retention',
          icon: MemoryVectorIcon,
          targetStage: 'continual',
        },
      ];

  // 3. Testimonials / Platform Highlights
  const quotes = isAr
    ? [
        {
          quote: '“أبسط وأقوى منصة لبناء وتدريب نماذج الذكاء الاصطناعي على البيانات الحقيقية بثقة تامة.”',
          source: 'مجتمع مطوري الذكاء الاصطناعي',
        },
        {
          quote: '“ميزة درع الذاكرة والتقييم العادل على بيانات الاختبار المستقلة نقلة نوعية في موثوقية النماذج.”',
          source: 'مختبر أبحاث النماذج اللغوية',
        },
        {
          quote: '“تجربة محلية آمنة وسلسة تحمي بياناتنا الحساسة وتمنحنا تحكماً كاملاً بالنماذج والكود المصدر.”',
          source: 'رواد حلول البيانات المؤسسية',
        },
      ]
    : [
        {
          quote: '“The cleanest and most reliable studio for building and training custom AI models on real data.”',
          source: 'AI Developers Network',
        },
        {
          quote: '“Fair evaluation on dedicated split data combined with memory shields sets a new standard for trust.”',
          source: 'NLP Research Lab',
        },
        {
          quote: '“A secure, offline-first experience that protects sensitive corporate data with full code ownership.”',
          source: 'Enterprise Intelligence Review',
        },
      ];

  // 4. Grounded in Your Sources Q&A (Non-Technical, Intuitive)
  const qaPresets = isAr
    ? [
        {
          q: 'كيف أبدأ ببناء وتدريب نموذج ذكاء اصطناعي على ملفاتي؟',
          a: 'ببساطة اسحب ملف بياناتك (CSV, TSV, Excel, JSON) داخل المنصة. تقوم المنصة تلقائياً بقراءة الأعمدة وتحديد النصوص والفئات واقتراح أفضل بنية عصبية، لتتمكن من بدء التدريب والمراقبة الحية بنقرة زر واحدة.',
          source: 'سحب وإفلات فوري · تعرف آلي على النصوص والفئات',
        },
        {
          q: 'هل بياناتي وملفاتي الحساسة آمنة ومحمية من التسريب؟',
          a: 'نعم، تعمل المنصة بشكل محلي كامل وآمن 100% داخل جهازك. بياناتك ونصوصك الحساسة وقواعد بياناتك لا تغادر بيئتك إطلاقاً، وتتيح لك بناء وتجربة النماذج دون الحاجة للاتصال بخوادم سحابية خارجية.',
          source: 'بيئة محلية آمنة · حماية تامة للبيانات الحساسة',
        },
        {
          q: 'كيف تضمن المنصة مصداقية التقييم وعدم وجود نتائج وهمية؟',
          a: 'تقسم المنصة بياناتك آلياً إلى بيانات تدريب وبيانات تقييم مستقلة (Train / Test Split). ويتم قياس كفاءة النموذج حصرياً على بيانات الاختبار التي لم يشاهدها مسبقاً، لحساب الدقة الواقعية ومصفوفة الالتباس التجريبية مع استعراض الأخطاء الحقيقية.',
          source: 'تقسيم علمي للبيانات · تقييم واقعي مع مصفوفة الالتباس',
        },
        {
          q: 'ما هو درع الذاكرة والتعلم المستمر (Continual Learning)؟',
          a: 'هي ميزة فريدة تحمي المعرفة والمهام التي أتقنها النموذج سابقاً عند تدريبه على مهمة جديدة تتابعياً، مما يسمح للنموذج باكتساب مهارات إضافية متراكمة دون التعرض للنسيان الكارثي.',
          source: 'درع الذاكرة EWC · حفظ المهام السابقة مع التعلم الجديد',
        },
      ]
    : [
        {
          q: 'How do I start building and training an AI model on my files?',
          a: 'Simply drag and drop your dataset (CSV, TSV, Excel, JSON) into the studio. The platform automatically detects columns, labels, and classes, designs a matching Transformer architecture, and starts training with a single click.',
          source: 'Instant Drag & Drop · Automatic Text & Label Detection',
        },
        {
          q: 'Is my private and sensitive data secure from leakage?',
          a: 'Yes, DYTR Studio operates locally and securely on your own hardware. Your private files, datasets, and proprietary models never leave your system, giving you complete privacy and full governance.',
          source: 'Local & Offline Privacy · Zero Data Leakage',
        },
        {
          q: 'How does the platform ensure authentic evaluation metrics?',
          a: 'The studio automatically partitions your dataset into dedicated training and test splits. The model is strictly evaluated on unseen test data, producing genuine confusion matrices, empirical accuracy, and authentic error inspection.',
          source: 'Clean Train/Test Split · Genuine Confusion Matrix',
        },
        {
          q: 'What is Continual Learning & the Memory Shield?',
          a: 'It is a state-of-the-art capability that shields the weights of previously learned tasks when training on new sequential datasets, allowing your model to expand its capabilities without catastrophic forgetting.',
          source: 'EWC Memory Shield · Cumulative Multi-Tasking',
        },
      ];

  // 5. Frequently Asked Questions (FAQ - Zero Technical Clutter)
  const faqs = isAr
    ? [
        {
          q: 'هل أحتاج إلى خبرة برمجية عميقة لبناء وتدريب النماذج؟',
          a: 'لا على الإطلاق! المنصة مصممة لتكون سهلة وبديهية عبر واجهات بصرية تفاعلية وإرشادات خطوة بخطوة، مع توفير خيار تصدير الكود البرمجي بالكامل (PyTorch) للمطورين والباحثين.',
        },
        {
          q: 'هل يمكنني استيراد نماذج محلية أو نماذج جاهزة من HuggingFace؟',
          a: 'نعم، تدعم المنصة تحميل النماذج المحلية والمدربة مسبقاً بسهولة، وتخصيص رؤوس المهام اللغوية عليها واختبارها محلياً بأمان وسرعة.',
        },
        {
          q: 'كيف أتأكد أن النموذج تم اختباره على بيانات مستقلة لم يتدرب عليها؟',
          a: 'تقوم المنصة بعزل نسبة من بياناتك كبيانات تقييم مستقلة (Test Split)، وتُظهر لك عينات الأخطاء الحقيقية وتوزيع التنبؤات المباشر لضمان دقة قياس النموذج قبل نشره.',
        },
        {
          q: 'هل يمكنني تصدير النموذج واستخدامه في تطبيقي أو موقعي؟',
          a: 'نعم، بنقرة واحدة يمكنك تصدير ملفات النموذج وكود بايثون الجاهز للإنتاج، أو استخدام واجهة الاستدلال الفوري ونقاط نهاية API المحلية المدمجة.',
        },
      ]
    : [
        {
          q: 'Do I need deep coding experience to build and train models?',
          a: 'Not at all! DYTR Studio is crafted with intuitive visual studios and step-by-step guides for everyone, while providing complete Python/PyTorch code exports for engineers.',
        },
        {
          q: 'Can I load local models or pretrained HuggingFace weights?',
          a: 'Yes, the platform seamlessly supports loading local and pretrained foundation weights, customizing task heads, and testing them completely offline.',
        },
        {
          q: 'How can I verify that testing runs on independent split data?',
          a: 'The platform partitions your data into clean training and evaluation splits, providing real sample misclassifications and unbiased validation curves.',
        },
        {
          q: 'Can I export the model to use in my production applications?',
          a: 'Yes, with one click you can export production-ready PyTorch scripts, serialized model weights, or interact with local REST API endpoints.',
        },
      ];

  const [activeIdx, setActiveIdx] = useState(0);
  const [fadeState, setFadeState] = useState('in');
  const [activeQuoteIdx, setActiveQuoteIdx] = useState(0);
  const [activeQaIdx, setActiveQaIdx] = useState(0);
  const [openFaqIdx, setOpenFaqIdx] = useState(null);

  // Rotating Hero Slide
  useEffect(() => {
    const interval = setInterval(() => {
      setFadeState('out');
      setTimeout(() => {
        setActiveIdx((prev) => (prev + 1) % slides.length);
        setFadeState('in');
      }, 320);
    }, 4200);

    return () => clearInterval(interval);
  }, [slides.length]);

  // Rotating Quotes
  useEffect(() => {
    const qInterval = setInterval(() => {
      setActiveQuoteIdx((prev) => (prev + 1) % quotes.length);
    }, 5000);

    return () => clearInterval(qInterval);
  }, [quotes.length]);

  const currentSlide = slides[activeIdx];
  const currentQuote = quotes[activeQuoteIdx];

  const handleLaunchStudio = (stage = 'academy') => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('dytr_has_seen_landing', 'true');
    }
    setActiveModule(stage);
  };

  const handleLaunchProjectTemplate = (item) => {
    if (item && item.datasetKey) {
      const data = getSampleDatasetById(item.datasetKey);
      if (data && Array.isArray(data) && data.length > 0) {
        useDatasetStore.getState().setDataset(data, `${item.datasetKey}.csv`, 'sample');
      }
    }
    handleLaunchStudio(item.targetStage || 'dataprep');
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        background: theme === 'dark' ? '#0B0F14' : '#FFFFFF',
        color: 'var(--text-primary)',
        fontFamily: 'var(--font-sans)',
        display: 'flex',
        flexDirection: 'column',
        userSelect: 'none',
      }}
    >
      {/* ==================================================================== */}
      {/* 1. TOP HEADER NAVIGATION (Clean, Zero Jargon, Zero Social Clutter) */}
      {/* ==================================================================== */}
      <nav
        style={{
          padding: '1.25rem 2.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          maxWidth: '1360px',
          width: '100%',
          margin: '0 auto',
        }}
      >
        {/* Brand Logo */}
        <div
          onClick={() => handleLaunchStudio('academy')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
            cursor: 'pointer',
          }}
        >
          <DytrLogo variant="icon" size={28} glow />
          <span style={{ fontSize: '1.2rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            DYTR Studio
          </span>
        </div>

        {/* Right Controls: Theme Icon + Language Globe Icon + Get Started Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {/* Theme Switcher Icon Button */}
          <button
            type="button"
            onClick={toggleTheme}
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: theme === 'dark' ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.04)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              color: 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
            }}
            title={theme === 'dark' ? (isAr ? 'التبديل إلى المظهر الفاتح' : 'Switch to Light Mode') : (isAr ? 'التبديل إلى المظهر الداكن' : 'Switch to Dark Mode')}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.08)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
          >
            {theme === 'dark' ? <SunVectorIcon size={18} /> : <MoonVectorIcon size={18} />}
          </button>

          {/* Language Switcher Icon Button (Globe) */}
          <button
            type="button"
            onClick={() => setLanguage(isAr ? 'en' : 'ar')}
            style={{
              height: '40px',
              padding: '0 0.85rem',
              borderRadius: '999px',
              background: theme === 'dark' ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.04)',
              border: '1px solid var(--border-subtle)',
              cursor: 'pointer',
              color: 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              transition: 'all 0.2s ease',
            }}
            title={isAr ? 'Switch language to English' : 'تغيير اللغة إلى العربية'}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.04)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
          >
            <GlobeVectorIcon size={17} />
            <span>{isAr ? 'EN' : 'عربي'}</span>
          </button>

          {/* Get Started Button */}
          <button
            type="button"
            onClick={() => handleLaunchStudio('academy')}
            style={{
              background: 'var(--text-primary)',
              color: 'var(--bg-primary)',
              border: 'none',
              borderRadius: '999px',
              padding: '0.6rem 1.4rem',
              fontSize: '0.9rem',
              fontWeight: 800,
              cursor: 'pointer',
              marginInlineStart: '0.35rem',
              transition: 'opacity 0.2s ease, transform 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.9'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'translateY(0)'; }}
          >
            {isAr ? 'الدخول للمنصة' : 'Enter Studio'}
          </button>
        </div>
      </nav>

      {/* ==================================================================== */}
      {/* 2. MAIN CENTER HERO CANVAS (Rotating Headline + Dynamic Visuals) */}
      {/* ==================================================================== */}
      <section style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem 2rem 3rem' }}>
        <div
          style={{
            maxWidth: '1280px',
            width: '100%',
            minHeight: '600px',
            background: theme === 'dark' ? '#14181F' : '#F1F5F9',
            borderRadius: '28px',
            padding: '3.5rem 2.5rem 2.5rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'relative',
            overflow: 'hidden',
            boxShadow: theme === 'dark' ? '0 24px 60px rgba(0, 0, 0, 0.45)' : '0 20px 50px rgba(0, 0, 0, 0.05)',
            transition: 'background 0.3s ease',
          }}
        >
          {/* Ambient Glow */}
          <div
            style={{
              position: 'absolute',
              top: '-150px',
              left: '50%',
              transform: 'translateX(-50%)',
              width: '500px',
              height: '350px',
              borderRadius: '50%',
              background: `radial-gradient(circle, ${currentSlide.color}22 0%, transparent 70%)`,
              pointerEvents: 'none',
              transition: 'background 0.5s ease',
            }}
          />

          {/* Rotating Headline */}
          <div style={{ textAlign: 'center', zIndex: 10, maxWidth: '880px' }}>
            <h1
              style={{
                fontSize: 'clamp(2.4rem, 5.2vw, 4.2rem)',
                fontWeight: 800,
                lineHeight: 1.18,
                letterSpacing: '-0.03em',
                marginBottom: '1.5rem',
                color: 'var(--text-primary)',
                opacity: fadeState === 'in' ? 1 : 0,
                transform: fadeState === 'in' ? 'translateY(0)' : 'translateY(8px)',
                transition: 'opacity 0.35s ease, transform 0.35s ease',
              }}
            >
              <span>{currentSlide.prefix}</span>
              <br />
              <span style={{ color: currentSlide.color, display: 'inline-block' }}>
                {currentSlide.highlight}
              </span>
            </h1>

            {/* Pill CTA Button */}
            <button
              type="button"
              onClick={() => handleLaunchStudio('dataprep')}
              style={{
                background: theme === 'dark' ? '#0F1216' : '#1E232B',
                color: '#FFFFFF',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '999px',
                padding: '0.85rem 2.4rem',
                fontSize: '1rem',
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 4px 18px rgba(0, 0, 0, 0.3)',
                transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.04)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
            >
              {isAr ? 'ابدأ بسحب بياناتك وبناء نموذجك الآن' : 'Start with Your Data & Build AI'}
            </button>
          </div>

          {/* Dynamic Illustrated AI Mockups Canvas */}
          <div
            style={{
              width: '100%',
              maxWidth: '980px',
              height: '310px',
              marginTop: '2.5rem',
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: fadeState === 'in' ? 1 : 0,
              transform: fadeState === 'in' ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.98)',
              transition: 'opacity 0.4s ease, transform 0.4s ease',
            }}
          >
            {/* Slide 1: AI Pipeline Mockup */}
            {currentSlide.illustrationType === 'pipeline' && (
              <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="100%" height="100%" viewBox="0 0 800 280" style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
                  <path d="M 230 140 C 310 140, 330 140, 400 140" fill="none" stroke="#0EA5E9" strokeWidth="3" opacity="0.8" />
                  <path d="M 540 140 C 600 140, 620 90, 680 90" fill="none" stroke="#38BDF8" strokeWidth="2.5" strokeDasharray="5,5" opacity="0.7" />
                  <path d="M 540 140 C 600 140, 620 190, 680 190" fill="none" stroke="#38BDF8" strokeWidth="2.5" strokeDasharray="5,5" opacity="0.7" />
                </svg>

                {/* Left: Data Ingestion Card */}
                <div
                  style={{
                    position: 'absolute',
                    left: '5%',
                    width: '190px',
                    background: theme === 'dark' ? '#1B212B' : '#FFFFFF',
                    borderRadius: '16px',
                    padding: '1.1rem',
                    boxShadow: '0 12px 30px rgba(0, 0, 0, 0.15)',
                    border: '1px solid rgba(14, 165, 233, 0.25)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                  }}
                >
                  <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: 'rgba(14, 165, 233, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <DatasetVectorIcon size={22} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.86rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                      {isAr ? 'سحب البيانات' : 'Dataset Ingestion'}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#0EA5E9', fontWeight: 700 }}>
                      sms.tsv (5,160 صف)
                    </div>
                  </div>
                </div>

                {/* Center: Dynamic Neural Core */}
                <div
                  style={{
                    position: 'absolute',
                    left: '40%',
                    transform: 'translateX(-50%)',
                    width: '200px',
                    background: theme === 'dark' ? '#151C28' : '#F0F9FF',
                    borderRadius: '18px',
                    padding: '1.25rem',
                    boxShadow: '0 16px 36px rgba(14, 165, 233, 0.2)',
                    border: '1.5px solid #0EA5E9',
                    textAlign: 'center',
                    zIndex: 5,
                  }}
                >
                  <div style={{ fontSize: '0.72rem', color: '#0EA5E9', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    DyTR Architecture
                  </div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', margin: '0.35rem 0' }}>
                    BERT Transformer
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center', gap: '0.35rem' }}>
                    <span className="badge" style={{ background: 'rgba(14, 165, 233, 0.2)', color: '#0EA5E9', fontSize: '0.68rem', padding: '0.2rem 0.5rem', borderRadius: '4px', fontWeight: 700 }}>
                      4 Layers
                    </span>
                    <span className="badge" style={{ background: 'rgba(14, 165, 233, 0.2)', color: '#0EA5E9', fontSize: '0.68rem', padding: '0.2rem 0.5rem', borderRadius: '4px', fontWeight: 700 }}>
                      4 Heads
                    </span>
                  </div>
                </div>

                {/* Right: Dedicated Task Heads & Real Predictions */}
                <div style={{ position: 'absolute', right: '4%', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  <div style={{ width: '160px', background: '#10B981', color: '#FFFFFF', borderRadius: '12px', padding: '0.65rem 0.9rem', fontWeight: 800, fontSize: '0.8rem', boxShadow: '0 6px 20px rgba(16, 185, 129, 0.35)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{isAr ? 'رسالة عادية (Ham)' : 'Ham Class'}</span>
                    <span style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.25)', padding: '0.1rem 0.4rem', borderRadius: '4px' }}>96.8%</span>
                  </div>
                  <div style={{ width: '160px', background: '#EF4444', color: '#FFFFFF', borderRadius: '12px', padding: '0.65rem 0.9rem', fontWeight: 800, fontSize: '0.8rem', boxShadow: '0 6px 20px rgba(239, 68, 68, 0.35)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{isAr ? 'رسالة مزعجة (Spam)' : 'Spam Class'}</span>
                    <span style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.25)', padding: '0.1rem 0.4rem', borderRadius: '4px' }}>3.2%</span>
                  </div>
                </div>
              </div>
            )}

            {/* Slide 2: Live Training Arena */}
            {currentSlide.illustrationType === 'training' && (
              <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div
                  style={{
                    width: '480px',
                    background: theme === 'dark' ? '#1B212B' : '#FFFFFF',
                    borderRadius: '20px',
                    padding: '1.4rem',
                    boxShadow: '0 20px 45px rgba(16, 185, 129, 0.18)',
                    border: '1.5px solid rgba(16, 185, 129, 0.35)',
                    zIndex: 5,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10B981' }} />
                      <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                        {isAr ? 'تدريب فوري ومراقبة حية' : 'Live Training Arena'}
                      </strong>
                    </div>
                    <span style={{ fontSize: '0.74rem', fontWeight: 800, color: '#10B981', background: 'rgba(16, 185, 129, 0.15)', padding: '0.2rem 0.6rem', borderRadius: '999px' }}>
                      Epoch 3/3 (100%)
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '0.85rem' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.1)', borderRadius: '10px', padding: '0.65rem', textAlign: 'center', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'الدقة التدريبية' : 'Train Acc'}</div>
                      <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#10B981', fontFamily: 'var(--font-mono)' }}>96.2%</div>
                    </div>
                    <div style={{ background: 'rgba(14, 165, 233, 0.1)', borderRadius: '10px', padding: '0.65rem', textAlign: 'center', border: '1px solid rgba(14, 165, 233, 0.2)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'الخسارة (Loss)' : 'Loss'}</div>
                      <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0EA5E9', fontFamily: 'var(--font-mono)' }}>0.084</div>
                    </div>
                    <div style={{ background: 'rgba(245, 158, 11, 0.1)', borderRadius: '10px', padding: '0.65rem', textAlign: 'center', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{isAr ? 'سرعة المعالجة' : 'Throughput'}</div>
                      <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#F59E0B', fontFamily: 'var(--font-mono)' }}>128 s/e</div>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{isAr ? 'تم التدريب على جهازك بدون رفع بيانات للخارج' : 'Trained on your device with complete privacy'}</span>
                    <span style={{ color: '#10B981', fontWeight: 800 }}>{isAr ? 'مكتمل بنجاح' : 'Success'}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Slide 3: Real Split Evaluation Mockup */}
            {currentSlide.illustrationType === 'evaluation' && (
              <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div
                  style={{
                    width: '460px',
                    background: theme === 'dark' ? '#1B212B' : '#FFFFFF',
                    borderRadius: '20px',
                    padding: '1.4rem',
                    boxShadow: '0 20px 45px rgba(245, 158, 11, 0.18)',
                    border: '1.5px solid rgba(245, 158, 11, 0.35)',
                    zIndex: 5,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <EvaluationVectorIcon size={20} />
                      <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                        {isAr ? 'مختبر التقييم التجريبي (بيانات الاختبار المستقلة)' : 'Empirical Evaluation Lab (Test Split)'}
                      </strong>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: '#F59E0B', fontWeight: 800 }}>
                      20% Test Split
                    </span>
                  </div>

                  {/* Confusion Matrix Simulation */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.65rem', marginBottom: '0.85rem' }}>
                    <div style={{ background: 'rgba(16, 185, 129, 0.12)', border: '1px solid #10B981', borderRadius: '10px', padding: '0.65rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>True Positive (Ham)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#10B981' }}>884 عينة</div>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '0.65rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>False Positive (خطأ)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#EF4444' }}>14 عينة</div>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '10px', padding: '0.65rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>False Negative (خطأ)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#EF4444' }}>18 عينة</div>
                    </div>
                    <div style={{ background: 'rgba(16, 185, 129, 0.12)', border: '1px solid #10B981', borderRadius: '10px', padding: '0.65rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>True Negative (Spam)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#10B981' }}>116 عينة</div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    <span>{isAr ? 'الدقة الفعلية: 96.8% · F1: 94.2%' : 'Empirical Accuracy: 96.8% · F1: 94.2%'}</span>
                    <span style={{ color: '#F59E0B', fontWeight: 800 }}>{isAr ? 'تقييم غير متحيز' : 'Unbiased'}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Slide 4: Continual Learning & Memory Shield */}
            {currentSlide.illustrationType === 'shield' && (
              <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div
                  style={{
                    width: '440px',
                    background: theme === 'dark' ? '#1B212B' : '#FFFFFF',
                    borderRadius: '24px',
                    padding: '1.5rem',
                    boxShadow: '0 20px 50px rgba(13, 148, 136, 0.18)',
                    border: '1.5px solid rgba(13, 148, 136, 0.4)',
                    textAlign: 'center',
                    zIndex: 5,
                  }}
                >
                  <div style={{ width: '50px', height: '50px', borderRadius: '50%', background: 'rgba(13, 148, 136, 0.15)', margin: '0 auto 0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <MemoryVectorIcon size={26} />
                  </div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                    {isAr ? 'درع الذاكرة والتعلم التتابعي' : 'Continuous Memory Shield'}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                    {isAr ? 'حماية المهمة السابقة (96.8%) مع اكتساب المهمة الجديدة (94.1%)' : 'Retains prior tasks (96.8%) while learning new tasks (94.1%)'}
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
                    <div style={{ background: 'rgba(13, 148, 136, 0.12)', padding: '0.4rem 0.8rem', borderRadius: '8px', border: '1px solid rgba(13, 148, 136, 0.3)', fontSize: '0.74rem', color: '#0D9488', fontWeight: 700 }}>
                      EWC Fisher Matrix Active
                    </div>
                    <div style={{ background: 'rgba(14, 165, 233, 0.12)', padding: '0.4rem 0.8rem', borderRadius: '8px', border: '1px solid rgba(14, 165, 233, 0.3)', fontSize: '0.74rem', color: '#0EA5E9', fontWeight: 700 }}>
                      LoRA Adapters Injected
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Dots Indicator */}
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1.5rem', zIndex: 10 }}>
            {slides.map((_, dotIdx) => (
              <span
                key={dotIdx}
                onClick={() => setActiveIdx(dotIdx)}
                style={{
                  width: activeIdx === dotIdx ? '24px' : '8px',
                  height: '8px',
                  borderRadius: '999px',
                  background: activeIdx === dotIdx ? currentSlide.color : 'rgba(255, 255, 255, 0.2)',
                  cursor: 'pointer',
                  transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 3. EXPLORE FEATURED PUBLIC PROJECTS (Ready-to-Use Benchmarks) */}
      {/* ==================================================================== */}
      <section
        style={{
          maxWidth: '1280px',
          width: '100%',
          margin: '2rem auto 4.5rem',
          padding: '0 2.5rem',
          display: 'grid',
          gridTemplateColumns: 'minmax(280px, 1fr) 2fr',
          gap: '3.5rem',
          alignItems: 'center',
        }}
      >
        <div>
          <h2
            style={{
              fontSize: 'clamp(2rem, 3.5vw, 2.8rem)',
              fontWeight: 800,
              lineHeight: 1.2,
              letterSpacing: '-0.025em',
              color: 'var(--text-primary)',
              marginBottom: '1.25rem',
            }}
          >
            {isAr ? 'استكشف مشاريع وبيانات جاهزة للتدريب فوراً' : 'Explore Featured Ready-to-Train Projects'}
          </h2>
          <p
            style={{
              fontSize: '1rem',
              color: 'var(--text-secondary)',
              lineHeight: 1.65,
            }}
          >
            {isAr
              ? 'اختر أحد النماذج والمجموعات التجريبية الحقيقية الجاهزة، واختبر الاستوديو وتعرف على مراحل معالجة النصوص والتدريب والتقييم بضغطة زر واحدة.'
              : 'Select one of our real-world datasets, inspect data columns, and experience the full training and evaluation lifecycle in seconds.'}
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
          {featuredProjects.map((item) => {
            const IconComp = item.icon;
            return (
              <div
                key={item.id}
                onClick={() => handleLaunchProjectTemplate(item)}
                style={{
                  background: theme === 'dark' ? '#14181F' : '#FFFFFF',
                  borderRadius: '24px',
                  padding: '2rem 1.75rem',
                  border: '1px solid rgba(14, 165, 233, 0.15)',
                  boxShadow: theme === 'dark' ? '0 12px 35px rgba(0, 0, 0, 0.35)' : '0 10px 30px rgba(0, 0, 0, 0.04)',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  minHeight: '210px',
                  cursor: 'pointer',
                  transition: 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.4)';
                  e.currentTarget.style.boxShadow = '0 18px 45px rgba(14, 165, 233, 0.15)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = 'rgba(14, 165, 233, 0.15)';
                  e.currentTarget.style.boxShadow = theme === 'dark' ? '0 12px 35px rgba(0, 0, 0, 0.35)' : '0 10px 30px rgba(0, 0, 0, 0.04)';
                }}
              >
                <div>
                  <div style={{ marginBottom: '1.25rem' }}>
                    <IconComp size={36} />
                  </div>
                  <h3
                    style={{
                      fontSize: '1.15rem',
                      fontWeight: 800,
                      color: 'var(--text-primary)',
                      lineHeight: 1.4,
                      marginBottom: '0.5rem',
                    }}
                  >
                    {item.title}
                  </h3>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {item.sources}
                  </span>
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      background: 'rgba(14, 165, 233, 0.1)',
                      color: '#0EA5E9',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <ArrowUpRightIcon size={16} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 4. GROUNDED IN YOUR SOURCES Q&A (Non-Technical, Intuitive) */}
      {/* ==================================================================== */}
      <section
        style={{
          maxWidth: '1280px',
          width: '100%',
          margin: '0 auto 4.5rem',
          padding: '0 2.5rem',
        }}
      >
        <div
          style={{
            background: theme === 'dark' ? '#14181F' : '#F8FAFC',
            borderRadius: '24px',
            padding: '2.5rem',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ textAlign: 'center', maxWidth: '640px', margin: '0 auto 2rem' }}>
            <span style={{ fontSize: '0.76rem', fontWeight: 800, color: '#0EA5E9', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {isAr ? 'الميزات الأساسية والخصوصية والأمان' : 'Core Capabilities & Security'}
            </span>
            <h2 style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.35rem' }}>
              {isAr ? 'كل ما تود معرفته عن منصة DYTR Studio' : 'Everything You Need to Know About DYTR Studio'}
            </h2>
          </div>

          {/* Quick Query Pills */}
          <div style={{ display: 'flex', gap: '0.65rem', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '1.75rem' }}>
            {qaPresets.map((item, qIdx) => (
              <button
                key={qIdx}
                type="button"
                onClick={() => setActiveQaIdx(qIdx)}
                style={{
                  background: activeQaIdx === qIdx ? '#0EA5E9' : (theme === 'dark' ? '#1B212B' : '#FFFFFF'),
                  color: activeQaIdx === qIdx ? '#FFFFFF' : 'var(--text-primary)',
                  border: '1px solid rgba(14, 165, 233, 0.25)',
                  borderRadius: '999px',
                  padding: '0.55rem 1.15rem',
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: activeQaIdx === qIdx ? '0 4px 14px rgba(14, 165, 233, 0.3)' : 'none',
                }}
              >
                {item.q}
              </button>
            ))}
          </div>

          {/* Dynamic Grounded Answer Card */}
          <div
            style={{
              background: theme === 'dark' ? '#0B0F14' : '#FFFFFF',
              borderRadius: '16px',
              padding: '1.75rem',
              border: '1.5px solid rgba(14, 165, 233, 0.3)',
              boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
            }}
          >
            <div style={{ fontSize: '1rem', color: 'var(--text-primary)', lineHeight: 1.7, marginBottom: '1.25rem' }}>
              {qaPresets[activeQaIdx].a}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
              <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 700 }}>
                {isAr ? 'الميزة التأسيسية:' : 'Core Capability:'}
              </span>
              <span style={{ fontSize: '0.74rem', color: '#0EA5E9', fontWeight: 800, background: 'rgba(14, 165, 233, 0.1)', padding: '0.2rem 0.6rem', borderRadius: '6px' }}>
                {qaPresets[activeQaIdx].source}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 5. TESTIMONIAL QUOTES WITH FLOWING WAVE RIBBONS */}
      {/* ==================================================================== */}
      <section
        style={{
          position: 'relative',
          padding: '5rem 2rem',
          margin: '1rem 0 3rem',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
        }}
      >
        <svg
          width="100%"
          height="100%"
          viewBox="0 0 1200 300"
          preserveAspectRatio="none"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 1,
          }}
        >
          <path
            d="M 0 150 C 150 50, 300 250, 450 150 C 600 50, 750 250, 900 150 C 1050 50, 1150 200, 1200 150"
            fill="none"
            stroke="rgba(14, 165, 233, 0.45)"
            strokeWidth="3.5"
          />
          <path
            d="M 0 180 C 200 280, 400 80, 600 180 C 800 280, 1000 80, 1200 180"
            fill="none"
            stroke="rgba(13, 148, 136, 0.35)"
            strokeWidth="2.5"
          />
        </svg>

        <div style={{ position: 'relative', zIndex: 5, maxWidth: '820px', padding: '0 1rem' }}>
          <blockquote
            style={{
              fontSize: 'clamp(2rem, 4.2vw, 3.2rem)',
              fontWeight: 800,
              lineHeight: 1.28,
              letterSpacing: '-0.025em',
              color: 'var(--text-primary)',
              marginBottom: '1.25rem',
            }}
          >
            {currentQuote.quote}
          </blockquote>

          <cite
            style={{
              display: 'block',
              fontSize: '0.96rem',
              fontWeight: 700,
              color: 'var(--text-muted)',
              fontStyle: 'normal',
              textDecoration: 'underline',
              textUnderlineOffset: '4px',
              marginBottom: '2rem',
            }}
          >
            {currentQuote.source}
          </cite>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem' }}>
            {quotes.map((_, qIdx) => (
              <span
                key={qIdx}
                onClick={() => setActiveQuoteIdx(qIdx)}
                style={{
                  width: activeQuoteIdx === qIdx ? '24px' : '8px',
                  height: '8px',
                  borderRadius: '999px',
                  background: activeQuoteIdx === qIdx ? 'var(--text-primary)' : 'rgba(120, 120, 120, 0.35)',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 6. FREQUENTLY ASKED QUESTIONS (FAQ - Zero Technical Clutter) */}
      {/* ==================================================================== */}
      <section
        style={{
          maxWidth: '860px',
          width: '100%',
          margin: '0 auto 5rem',
          padding: '0 2rem',
        }}
      >
        <h2 style={{ fontSize: '1.9rem', fontWeight: 800, textAlign: 'center', color: 'var(--text-primary)', marginBottom: '2rem' }}>
          {isAr ? 'الأسئلة الشائعة' : 'Frequently Asked Questions'}
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {faqs.map((faq, fIdx) => {
            const isOpen = openFaqIdx === fIdx;
            return (
              <div
                key={fIdx}
                style={{
                  background: theme === 'dark' ? '#14181F' : '#F8FAFC',
                  borderRadius: '16px',
                  border: '1px solid var(--border-subtle)',
                  overflow: 'hidden',
                  transition: 'all 0.2s ease',
                }}
              >
                <button
                  type="button"
                  onClick={() => setOpenFaqIdx(isOpen ? null : fIdx)}
                  style={{
                    width: '100%',
                    padding: '1.25rem 1.5rem',
                    background: 'transparent',
                    border: 'none',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    textAlign: isAr ? 'right' : 'left',
                    cursor: 'pointer',
                    fontSize: '1rem',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                  }}
                >
                  <span>{faq.q}</span>
                  <span style={{ fontSize: '1.2rem', color: '#0EA5E9', transform: isOpen ? 'rotate(45deg)' : 'rotate(0)', transition: 'transform 0.2s ease' }}>
                    +
                  </span>
                </button>

                {isOpen && (
                  <div style={{ padding: '0 1.5rem 1.25rem', fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6, borderTop: '1px solid var(--border-subtle)', paddingTop: '0.85rem' }}>
                    {faq.a}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 7. MINIMAL CLEAN FOOTER */}
      {/* ==================================================================== */}
      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '1.75rem 2.5rem',
          maxWidth: '1360px',
          width: '100%',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.82rem',
          color: 'var(--text-muted)',
        }}
      >
        <div>DYTR Studio · AI Research & Dynamic Transformer Studio</div>
        <div style={{ display: 'flex', gap: '1.75rem', fontWeight: 600 }}>
          <span onClick={() => handleLaunchStudio('academy')} style={{ cursor: 'pointer' }}>{isAr ? 'الأكاديمية' : 'Academy'}</span>
          <span onClick={() => handleLaunchStudio('dataprep')} style={{ cursor: 'pointer' }}>{isAr ? 'البيانات' : 'Data'}</span>
          <span onClick={() => handleLaunchStudio('builder')} style={{ cursor: 'pointer' }}>{isAr ? 'المعمارية' : 'Architecture'}</span>
          <span onClick={() => handleLaunchStudio('arena')} style={{ cursor: 'pointer' }}>{isAr ? 'التدريب' : 'Training'}</span>
          <span onClick={() => handleLaunchStudio('evaluation')} style={{ cursor: 'pointer' }}>{isAr ? 'التقييم' : 'Evaluation'}</span>
          <span onClick={() => handleLaunchStudio('playground')} style={{ cursor: 'pointer' }}>{isAr ? 'الاستدلال' : 'Playground'}</span>
        </div>
      </footer>
    </div>
  );
}
