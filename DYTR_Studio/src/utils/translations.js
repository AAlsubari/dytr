// Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
// Bilingual Translations Dictionary (Arabic & English) - Zero Emojis Policy

export const translations = {
  ar: {
    // Navigation & App Header
    app_title: "DYTR Studio",
    app_subtitle: "استوديو الذكاء الاصطناعي التفاعلي والتعليمي والتعلم المستمر",
    mode_wizard: "المصمم الإرشادي",
    mode_datasets: "استوديو البيانات",
    mode_builder: "استوديو المعمارية",
    mode_arena: "التدريب وبناء وتحديث الأوزان",
    mode_observatory: "مرصد مصفوفة R",
    mode_glassroom: "المنطقة التعليمية",
    mode_playground: "غرفة الاستنتاج",
    lang_name: "English",
    theme_dark: "الوضع الليلي",
    theme_light: "الوضع النهاري",
    help_button: "دليل المبتدئين",
    system_status: "النظام جاهز",
    engine_live: "محرك DyTR المباشر",
    engine_simulation: "محرك المحاكاة التفاعلي",
    engine_reconnect: "إعادة الاتصال بالخادم",
    engine_connecting: "جاري الاتصال...",

    // Step-by-Step Wizard
    wizard_title: "ابنِ ودرب نموذج ذكاء اصطناعي بنفسك في 4 خطوات سهلة",
    wizard_subtitle: "لا تحتاج إلى أي خبرة برمجية أو رياضية مسبقة. اتبع الخطوات وسيقوم النظام بتنفيذ التدريب والمعالجة تلقائياً.",
    
    step1_title: "الخطوة 1: ماذا تريد أن يفعل نموذجك؟",
    step1_desc: "اختر المهارة التي تود تعليمها للذكاء الاصطناعي اليوم:",
    task_sentiment_title: "فهم المشاعر (إيجابي أو سلبي)",
    task_sentiment_desc: "تحديد ما إذا كان النص أو التقييم سعيداً وراضياً أم حزيناً وغير راضٍ (مثل مراجعات المنتجات والأفلام).",
    task_sentiment_badge: "تصنيف مشاعر",
    
    task_topic_title: "تصنيف الأخبار والمواضيع",
    task_topic_desc: "قراءة النصوص وتحديد مجالها وموضوعها فوراً (رياضة، تكنولوجيا، سياسة، اقتصاد).",
    task_topic_badge: "تصنيف نصوص",

    task_ner_title: "صياد الأسماء والكيانات (NER)",
    task_ner_desc: "قراءة النصوص واستخراج أسماء الأشخاص والشركات والمدن والدول بدقة مجهرية.",
    task_ner_badge: "استخراج كيانات",

    task_story_title: "المؤلف والكاتب الذكي",
    task_story_desc: "البدء بجملة افتتاحية وإكمال صياغة نص متناسق بصورة توليدية ذاتية.",
    task_story_badge: "توليد نصوص",

    step2_title: "الخطوة 2: اختر حجم عقل النموذج",
    step2_desc: "كلما زاد حجم النموذج زادت قدرته الاستيعابية، بينما النماذج المدمجة تتدرب بسرعة فائقة:",
    brain_small_title: "العقل المدمج السريع (Pocket Brain)",
    brain_small_desc: "خفيف وسريع جداً، يتدرب في ثوانٍ ومناسب للحواسيب القياسية.",
    brain_medium_title: "العقل المتوازن (Smart Brain)",
    brain_medium_desc: "قوي ودقيق، يجمع بين كفاءة السرعة والفهم اللغوي المتكامل.",
    brain_large_title: "العقل المتقدم (Super Brain)",
    brain_large_desc: "أعلى سعة استيعابية، يمتلك ذاكرة واسعة لمعالجة المقالات الطويلة.",

    step3_title: "الخطوة 3: قوة الذاكرة ومكافحة النسيان",
    step3_desc: "تفعيل تقنيات التسريع والذاكرة المتقدمة المطورة في DyTR:",
    shield_si_title: "درع الذاكرة المستمرة (EWC & Replay)",
    shield_si_desc: "حماية المعارف السابقة؛ عندما يتعلم النموذج مهمة جديدة لا ينسى المهارات التي تدرب عليها سابقاً.",
    flash_title: "محرك التسريع الفائق (FlashAttention)",
    flash_desc: "تسريع حسابات الانتباه وخفض استهلاك الذاكرة عبر التقطيع المباشر داخل SRAM.",

    step4_title: "الخطوة 4: بدء التدريب والمتابعة المباشرة",
    step4_desc: "شاهد الذكاء الاصطناعي وهو يقرأ البيانات خطوة بخطوة ويطور أداءه لحظياً:",
    btn_start_training: "ابدأ تدريب النموذج الآن",
    btn_stop_training: "إيقاف التدريب",
    training_in_progress: "جاري التدريب... الذكاء الاصطناعي يستوعب البيانات",
    training_complete: "اكتمل تدريب النموذج بنجاح وهو جاهز للاختبار",
    training_step_reading: "قراءة البيانات وتحليل التراكيب اللغوية...",
    training_step_connecting: "تحديث الأوزان والروابط العصبية في المعمارية...",
    training_step_testing: "فحص مصفوفة الذاكرة والتأكد من ثبات المهام السابقة...",

    // Playground
    playground_title: "غرفة تجربة واختبار النموذج المباشرة",
    playground_subtitle: "أدخل أي نص في الحقل أدناه واضغط الزر لاختبار استجابة النموذج وتصنيفه فوراً.",
    input_placeholder: "اكتب النص هنا للتجربة... مثال: 'هذا العمل التقني متميز ويقدم كفاءة استثنائية'",
    btn_test_now: "اختبار استجابة النموذج",
    result_title: "نتيجة تحليل النموذج:",
    confidence_label: "مستوى ثقة التنبؤ:",

    // Educational Glassroom
    glassroom_title: "المنطقة  التعليمية: كيف تعمل النماذج العصبية؟",
    glassroom_subtitle: "محاكيات بصرية تفاعلية تبسط المفاهيم الرياضية والهندسية العميقة.",
    
    sim_rope_title: "1. بوصلة التضمينات الدوارة (RoPE)",
    sim_rope_simple: "تدوير متجهات الكلمات زاوياً في الفضاء المركب بناءً على موقعها الزمني، مع الحفاظ الكامل على طول المتجه الدلالي الأصلي.",
    
    sim_flash_title: "2. تقطيع الذاكرة والسرعة الفائقة (FlashAttention)",
    sim_flash_simple: "تقسيم حسابات مصفوفات الانتباه إلى كتل محلية تتم معالجتها داخل ذاكرة SRAM السريعة بدلاً من النقل المتكرر من وإلى HBM.",

    sim_cl_title: "3. مختبر التعلم المستمر وحماية المعرفة (EWC & Replay)",
    sim_cl_simple: "وضع قيود تربيعية على المعلمات الحيوية لحمايتها من التدهور أو النسيان الكارثي عند تدريب النموذج على مهام متعاقبة.",

    sim_adalora_title: "4. أغلفة التكيف متعددة المهام (Multi-Task LoRA)",
    sim_adalora_simple: "تفكيك تحديثات الأوزان إلى مصفوفات منخفضة الرتبة لكل مهمة وتوفير الذاكرة مع إمكانية دمج الأوزان.",

    sim_token_title: "5. محاذاة المقاطع الفرعية وعزل الخسارة برمز -100",
    sim_token_simple: "تجزئة الكلمات الطويلة إلى مقاطع فرعية وإسناد القيمة -100 للمقاطع اللاحقة لتتجاهلها دالة الخسارة وتمنع تضخيم التدرجات.",

    // Guide Modal
    guide_title: "دليل المبتدئين: كيف يتعلم الذكاء الاصطناعي؟",
    guide_p1: "الشبكة العصبية هي نموذج رياضي مستوحى من البنية العصبية، يتعلم من خلال معالجة الأمثلة وتعديل الأوزان تدريجياً.",
    guide_p2: "1. تغذية البيانات: تزويد النموذج بنصوص وبيانات مصنفة ومجهزة.",
    guide_p3: "2. التعديل التدريجي: حساب فارق الخطأ بين التوقع والإجابة الصحيحة وتعديل الأوزان بالتدرج العكسي.",
    guide_p4: "3. الاستنتاج: يصبح النموذج قادراً على تعميم القواعد والإجابة على مدخلات جديدة لم يشهدها مسبقاً.",
    btn_close: "إغلاق الدليل وبدء التجربة",
    btn_copy_code: "نسخ الكود البرمجي",
    code_copied: "تم نسخ الكود بنجاح",
  },

  en: {
    // Navigation & App Header
    app_title: "DYTR Studio",
    app_subtitle: "Interactive & Continual Learning Neural Architecture Studio",
    mode_wizard: "Guided Wizard",
    mode_datasets: "Data Engine",
    mode_builder: "Architecture Studio",
    mode_arena: "Training Arena",
    mode_observatory: "R-Matrix Observatory",
    mode_glassroom: "Learning Glassroom",
    mode_playground: "Inference Playground",
    lang_name: "العربية",
    theme_dark: "Dark Mode",
    theme_light: "Light Mode",
    help_button: "Beginner Guide",
    system_status: "System Ready",
    engine_live: "DyTR Live Engine",
    engine_simulation: "Interactive Simulation",
    engine_reconnect: "Reconnect Server",
    engine_connecting: "Connecting...",

    // Step-by-Step Wizard
    wizard_title: "Build and Train Your AI Model in 4 Easy Steps",
    wizard_subtitle: "No prior programming or mathematics required. Follow the guided steps and let the studio execute the underlying training pipeline.",

    step1_title: "Step 1: What should your model do?",
    step1_desc: "Select the primary capability you want to teach your AI model today:",
    task_sentiment_title: "Understand Sentiment (Positive / Negative)",
    task_sentiment_desc: "Detects emotional polarity in customer reviews, feedback, and sentiment commentary.",
    task_sentiment_badge: "Sentiment Classification",

    task_topic_title: "Categorize News & Topics",
    task_topic_desc: "Reads articles and instantly assigns topical categories (Sports, Technology, Politics, Business).",
    task_topic_badge: "Topic Classification",

    task_ner_title: "Entity & Name Extraction (NER)",
    task_ner_desc: "Scans text and extracts names of people, organizations, locations, and geopolitical entities.",
    task_ner_badge: "Token Extraction",

    task_story_title: "Text Generation & Authoring",
    task_story_desc: "Takes an initial prompt and generates coherent, continuous text passages autoregressively.",
    task_story_badge: "Generative LM",

    step2_title: "Step 2: Choose Model Capacity",
    step2_desc: "Larger capacities capture more nuanced patterns, while compact models train rapidly on standard hardware:",
    brain_small_title: "Compact Engine (Pocket Brain)",
    brain_small_desc: "Lightweight and fast, trains within seconds on standard CPUs and mobile hardware.",
    brain_medium_title: "Balanced Engine (Smart Brain)",
    brain_medium_desc: "Optimized balance between training throughput and deep textual representation.",
    brain_large_title: "Expanded Engine (Super Brain)",
    brain_large_desc: "Maximum capacity for processing lengthy articles and intricate linguistic structures.",

    step3_title: "Step 3: Memory & Hardware Acceleration",
    step3_desc: "Enable continual learning protection and hardware kernels engineered in DyTR:",
    shield_si_title: "Continual Memory Shield (EWC & Replay)",
    shield_si_desc: "Shields prior knowledge so that learning new tasks does not overwrite previously learned skills.",
    flash_title: "Hardware Acceleration (FlashAttention)",
    flash_desc: "Accelerates attention computation and cuts memory footprint via SRAM block-tiling.",

    step4_title: "Step 4: Execute Training & Live Telemetry",
    step4_desc: "Monitor the neural network as it ingests data and refines its parameters in real time:",
    btn_start_training: "Start Model Training",
    btn_stop_training: "Pause Training",
    training_in_progress: "Training in progress... Optimizing neural weights",
    training_complete: "Model training completed successfully. Ready for inference.",
    training_step_reading: "Ingesting datasets and mapping linguistic representations...",
    training_step_connecting: "Updating neural parameters and layer transformations...",
    training_step_testing: "Evaluating retention matrix and verifying memory stability...",

    // Playground
    playground_title: "Live Inference Playground",
    playground_subtitle: "Input text into the prompt field to evaluate real-time predictions across active task heads.",
    input_placeholder: "Type text here to evaluate... e.g. 'This architectural design is exceptionally efficient'",
    btn_test_now: "Evaluate Model Response",
    result_title: "Model Prediction Response:",
    confidence_label: "Prediction Confidence Score:",

    // Educational Glassroom
    glassroom_title: "The Learning Glassroom: Neural Mechanics",
    glassroom_subtitle: "Interactive visual simulators breaking down core mathematical and architectural concepts.",

    sim_rope_title: "1. Rotary Position Embeddings (RoPE)",
    sim_rope_simple: "Rotates vector dimensions in complex space by token position while strictly preserving vector norm.",

    sim_flash_title: "2. IO-Aware Memory Tiling (FlashAttention)",
    sim_flash_simple: "Partitions attention matrices into localized blocks computed entirely within high-speed SRAM.",

    sim_cl_title: "3. Continual Learning & Retention (EWC & Replay)",
    sim_cl_simple: "Imposes quadratic regularizers on critical parameters to eliminate catastrophic forgetting across sequential tasks.",

    sim_adalora_title: "4. Multi-Task LoRA Adaptation",
    sim_adalora_simple: "Decomposes weight updates into task-isolated low-rank matrices with parameter-efficient fine-tuning and merging.",

    sim_token_title: "5. Subword Token Alignment & -100 Loss Masking",
    sim_token_simple: "Fragments words into subwords and assigns -100 to subsequent sub-tokens to suppress redundant loss gradients.",

    // Guide Modal
    guide_title: "Beginner Guide: How Neural Networks Learn",
    guide_p1: "A neural network is a mathematical architecture that learns patterns through iterative gradient adjustments on sample data.",
    guide_p2: "1. Data Ingestion: The model receives tokenized text and structured labels.",
    guide_p3: "2. Optimization: Loss is computed and gradients adjust parameter weights via backpropagation.",
    guide_p4: "3. Inference: The trained model generalizes learned representations to unseen test prompts.",
    btn_close: "Close Guide and Start",
    btn_copy_code: "Copy Python Code",
    code_copied: "Code copied successfully",
  },
};
