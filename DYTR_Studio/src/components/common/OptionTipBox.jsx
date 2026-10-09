// ==============================================================================
// DyTrStudio Academic Option Tip Box Component
// Provides intuitive academic explanations and tips for non-obvious dropdown options
// Zero Emojis Policy - High Contrast Dual Theme (Dark / Light)
// ==============================================================================

import React from 'react';

// Comprehensive dictionary of intuitive tips for dropdown choices across the platform
export const optionTipDictionary = {
  // --- Acquisition & Search Platforms ---
  platform_all: {
    title_ar: 'البحث الشامل والموحد',
    desc_ar: 'يقوم بالبحث التجميعي اللحظي في كافة المستودعات العالمية الخمسة بنفس الوقت لجلب متنوع وموثوق للبيانات.',
    recommendation_ar: 'يوصى به للمبتدئين وللبحث السريع عن أفضل الملفات.',
  },
  platform_uci: {
    title_ar: 'مستودع UCI ML الأكاديمي',
    desc_ar: 'أعرق مستودع أكاديمي قياسي في جامعة كاليفورنيا. يحتوي على قواعد بيانات ناهزت 600+ مسألة مفهرسة.',
    recommendation_ar: 'مثالي للتجارب الأكاديمية المقارنة ولأغراض الأبحاث النظرية.',
  },
  platform_datagov: {
    title_ar: 'منصة Data.gov الحكومية',
    desc_ar: 'البوابة الرسمية للبيانات المفتوحة الحكومية. تضم مؤشرات القطاعات الخدمية، البيئية، والاجتماعية.',
    recommendation_ar: 'مناسب لدراسات التخطيط العمراني، الإحصاء السكاني، والسياسات العامة.',
  },
  platform_worldbank: {
    title_ar: 'مجموعات البنك الدولي (World Bank Data)',
    desc_ar: 'بيانات اقتصادية وتنموية عالمية تغطي التطور الاقتصادي، التضخم، ومؤشرات التعليم والتنمية عبر الدول.',
    recommendation_ar: 'ممتاز للتحليل المالي الكلي والتنبؤ بالمؤشرات الاقتصادية.',
  },
  platform_github: {
    title_ar: 'مستودعات GitHub البرمجية المفتوحة',
    desc_ar: 'استخراج الملفات وقواعد البيانات الخام المرفوعة مباشرة في المشاريع البرمجية الحية على جيثب.',
    recommendation_ar: 'يفضل عندما تبحث عن بيانات حقيقية حديثة أو عينات نصوص عربية متخصصة.',
  },
  platform_kaggle: {
    title_ar: 'مستودع Kaggle التنافسي',
    desc_ar: 'أكبر منصة عالمية لبيانات ومسابقات علوم البيانات، تتميز بنظافة الهيكلة وتوافر التقييمات.',
    recommendation_ar: 'خيار ممتاز لتجارب التنافسية وبناء نماذج الذكاء الاصطناعي الاحترافية.',
  },
  platform_google: {
    title_ar: 'محرك البحث المفتوح والمستودعات الأكاديمية',
    desc_ar: 'فهرس متخصص للوصول إلى ملفات CSV وجداول البيانات المفتوحة مباشرة من الخوادم الجامعية ومراكز الأبحاث.',
    recommendation_ar: 'مثالي للوصول إلى مجموعات بيانات علمية موثوقة وروابط تحميل مباشرة.',
  },
  platform_huggingface: {
    title_ar: 'مستودع Hugging Face Hub العالمي',
    desc_ar: 'المنصة العالمية الأولى لنماذج ومجموعات بيانات الذكاء الاصطناعي ومعالجة اللغات الطبيعية (NLP).',
    recommendation_ar: 'الخيار الأفضل للمدونات اللغوية العربية الضخمة ولمهام تدريب المحولات العصبية.',
  },

  // --- Synthetic Data Generation Types ---
  synth_sentiment: {
    title_ar: 'بيانات المشاعر العربية (Arabic Sentiment)',
    desc_ar: 'توليد عينات نصوص عربية منوعة تغطي المشاعر الإيجابية، السلبية، والحيادية مع وسومها.',
    recommendation_ar: 'مثالي لتدريب نماذج تصنيف النصوص وتجربة خوارزميات dytr.',
  },
  synth_news: {
    title_ar: 'الأخبار العربية المتعددة (Arabic News Categories)',
    desc_ar: 'توليد نصوص إخبارية موزعة على تصنيفات: سياسة، اقتصاد، تكنولوجيا، ورياضة.',
    recommendation_ar: 'مفيد لمهمات تصنيف المواضيع وتحديد الكيانات اللغوية.',
  },
  synth_iris: {
    title_ar: 'قاعدة بيانات أزهار الإيريس القياسية (Iris Dataset)',
    desc_ar: 'مجموعة أرقام قياسية شهيرة في أدبيات الإحصاء (4 خصائص هندسية لـ 3 أنواع من الأزهار).',
    recommendation_ar: 'ممتازة لاختبار خوارزميات التقييس، الضغط المعرفي، والتقسيم الهيكلي.',
  },
  synth_titanic: {
    title_ar: 'بيانات ركاب التايتانك (Titanic Demographics)',
    desc_ar: 'بيانات تحتوي على متغيرات مختلطة (أرقام، أسماء، فئات) تغطي الديموغرافيا ونسب النجاح.',
    recommendation_ar: 'خيار مثالي لتجربة معالجة القيم المفقودة والتحويل الأحادي (One-Hot Encoding).',
  },

  // --- Preprocessing & Outliers Handling ---
  outlier_iqr: {
    title_ar: 'طريقة المدى الربيعي (IQR Method)',
    desc_ar: 'تحدد القيم الشاذة بالاعتماد على الفجوة بين الربيع الأول والثالث (1.5 × IQR). مقاومة للانحرافات.',
    recommendation_ar: 'الخيار الآمن والأفضل للبيانات التي لا تتبع التوزيع الطبيعي.',
  },
  outlier_zscore: {
    title_ar: 'طريقة الدرجة المعيارية (Z-Score Method)',
    desc_ar: 'تعتبر القيمة شاذة إذا ابتعدت عن المتوسط بأكثر من 3 انحرافات معيارية (|Z| > 3).',
    recommendation_ar: 'تستخدم فقط عندما تكون بياناتك ذات توزيع طبيعي (Bell Curve).',
  },
  outlier_action_drop: {
    title_ar: 'حذف الصفوف الشاذة (Drop Outlier Rows)',
    desc_ar: 'إزالة الأسطر التي تحتوي على قيم شاذة بشكل كامل من جدول البيانات.',
    recommendation_ar: 'ينصح به عندما تكون العينات الشاذة قليلة العدد وناتجة عن خطأ في الإدخال.',
  },
  outlier_action_clip: {
    title_ar: 'تقليم القيم الشاذة (Winsorize / Clip Boundaries)',
    desc_ar: 'تعديل القيمة الشاذة المرتفعة لتساوي الحد الأقصى المسموح، والشاذة المنخفضة للحد الأدنى.',
    recommendation_ar: 'الأفضل للحفاظ على حجم بياناتك بالكامل دون خسارة أي صفوف.',
  },
  outlier_action_median: {
    title_ar: 'الاستبدال بالوسيط الإحصائي (Impute with Median)',
    desc_ar: 'استبدال القيمة الشاذة بالوسيط الإحصائي للعمود الحسابي.',
    recommendation_ar: 'يحمي النموذج العصبي من انحراف المتوسط الحسابي الناتج عن الشذوذ.',
  },

  // --- Transformation & Scaling Strategies ---
  scale_minmax: {
    title_ar: 'التحجيم الأدنى والأقصى (Min-Max Normalization)',
    desc_ar: 'تحويل كافة أرقام العمود لتصبح محصورة في النطاق المحدد [0, 1].',
    recommendation_ar: 'ضروري وموصى به للشبكات العصبية ذات دوال التفعيل Sigmoid / Softmax.',
  },
  scale_standard: {
    title_ar: 'التقييس المعياري (Standardization Z-Score)',
    desc_ar: 'إعادة تجميع أرقام العمود لتصبح ذات متوسط يساوي 0 وانحراف معياري يساوي 1.',
    recommendation_ar: 'الخيار القياسي لمعظم خوارزميات التعلم الآلي والتجميع الهيكلي.',
  },
  scale_robust: {
    title_ar: 'التقييس المقاوم للشذوذ (Robust Scaling)',
    desc_ar: 'يستخدم الوسيط والمدى الربيعي بدلاً من المتوسط لتقييس الأرقام.',
    recommendation_ar: 'ممتاز للبيانات التي تحتوي على ضوضاء أو قيم شاذة لا ترغب في حذفها.',
  },
  encode_onehot: {
    title_ar: 'الترميز الأحادي (One-Hot Encoding)',
    desc_ar: 'تحويل العمود الفئوي (مثل: الرياض، جدة، الدمام) إلى أعمدة ثنائية 0 أو 1 مستقلة.',
    recommendation_ar: 'يمنع النموذج العصبي من افتراض ترتيب وهمي بين الفئات غير الترتيبية.',
  },
  encode_label: {
    title_ar: 'الترميز الترتيبي (Label Encoding)',
    desc_ar: 'إعطاء رقم متسلسل لكل فئة (مثال: ضعيف=0، متوسط=1، ممتاز=2).',
    recommendation_ar: 'مناسب حصرياً للفئات التي تحمل ترتيباً منطقياً وأبعاداً متدرجة.',
  },

  // --- Transformation & Join Options ---
  join_inner: {
    title_ar: 'الدمج الداخلي التقاطعي (Inner Join)',
    desc_ar: 'يجلب الصفوف التي تتطابق قيم مفتاح الربط فيها في كلا الجدولين فقط، ويستبعد أي صف غير موجود في أحدهما.',
    recommendation_ar: 'الخيار الأفضل للبيانات النظيفة المترابطة 100%.',
  },
  join_left: {
    title_ar: 'الدمج اليساري المباشر (Left Join)',
    desc_ar: 'يحافظ على كافة صفوف جدولك الرئيسي الحالي دون حذف أي منها، ويضيف بيانات الجدول الثاني للصفوف المطابقة فقط.',
    recommendation_ar: 'الخيار الموصى به والأكثر أماناً للحفاظ على بياناتك الحالية دون خسارة صفوف.',
  },
  join_right: {
    title_ar: 'الدمج اليميني (Right Join)',
    desc_ar: 'يحافظ على جميع صفوف الجدول الثاني المرفق، ويستورد من جدولك الحالي الصفوف التي تطابقه فقط.',
    recommendation_ar: 'يستخدم فقط إذا كان الجدول المرفق الثاني هو الجدول الأساسي المعتمد.',
  },
  join_outer: {
    title_ar: 'الدمج الخارجي الشامل (Full Outer Join)',
    desc_ar: 'يحافظ على جميع الأسطر والبيانات من الجدولين معاً دون استبعاد، ويضع فراغات (NULL) في القيم غير المطابقة.',
    recommendation_ar: 'مناسب لدمج شامل واستكشاف النقص الإحصائي بين الجدولين.',
  },

  // --- Architecture Complexity Options ---
  comp_tiny: {
    title_ar: 'نموذج خفيف جداً (Tiny Architecture)',
    desc_ar: 'طبقة واحدة وبُعد تضمين 64. يستهلك أدنى قدر من الذاكرة والعمليات.',
    recommendation_ar: 'مناسب للمسائل البسيطة والأجهزة ذات الموارد المحدودة.',
  },
  comp_base: {
    title_ar: 'النموذج القياسي المتوازن (Base Architecture)',
    desc_ar: '4 طبقات عصبية وبُعد 256. التوازن الأكاديمي المثالي بين السرعة والدقة.',
    recommendation_ar: 'الخيار الموصى به لمعظم الأبحاث والتطبيقات الحقيقية.',
  },
  comp_large: {
    title_ar: 'النموذج العميق (Large Architecture)',
    desc_ar: '8 طبقات وبُعد 512. قدرة استيعابية عالية للأنماط المعقدة جداً.',
    recommendation_ar: 'يتطلب بيانات ضخمة وقوة معالجة حاسوبية أعلى.',
  },
};

export default function OptionTipBox({ optionKey, customTip, isAr = true }) {
  const tip = customTip || optionTipDictionary[optionKey];

  if (!tip) return null;

  return (
    <div
      style={{
        marginTop: '0.55rem',
        padding: '0.65rem 0.85rem',
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border-medium)',
        borderRight: isAr ? '3px solid var(--accent-teal)' : '1px solid var(--border-medium)',
        borderLeft: isAr ? '1px solid var(--border-medium)' : '3px solid var(--accent-teal)',
        borderRadius: 'var(--radius-sm, 6px)',
        fontSize: '0.78rem',
        lineHeight: 1.5,
        color: 'var(--text-primary)',
        textAlign: isAr ? 'right' : 'left',
        transition: 'all 0.2s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.2rem' }}>
        <span style={{ fontWeight: 800, color: 'var(--accent-teal)' }}>
          {tip.title_ar || tip.title}
        </span>
        <span className="badge badge-teal" style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>
          {isAr ? 'توضيح الاختيار' : 'Choice Explanation'}
        </span>
      </div>
      <p style={{ margin: 0, color: 'var(--text-secondary)', marginBottom: tip.recommendation_ar ? '0.25rem' : 0 }}>
        {tip.desc_ar || tip.desc}
      </p>
      {tip.recommendation_ar && (
        <span style={{ display: 'block', fontSize: '0.74rem', color: 'var(--accent-amber)', fontWeight: 600 }}>
          {isAr ? 'توصية النظام: ' : 'System Tip: '}{tip.recommendation_ar}
        </span>
      )}
    </div>
  );
}
