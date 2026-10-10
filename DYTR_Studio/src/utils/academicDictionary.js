// ==============================================================================
// DYTR Studio Academic Dictionary & Conceptual Reference
// Zero Emojis Policy - High Rigor Arabic & English Educational Guidance
// Covers all 8 modules + 10 Academy steps + tooltips + page content
// ==============================================================================

export const academicDictionary = {
  ar: {
    // Navigation Modules (12 Modules)
    module_problem: "تحديد المشكلة",
    module_academy: "الأكاديمية التعليمية",
    module_dataprep: "تجهيز وتنقيب البيانات",
    module_builder: "بناء النماذج الذكية",
    module_arena: "التدريب وبناء الأوزان",
    module_observatory: "مرصد القياسات الحية",
    module_glassroom: "تشريح آليات النموذج",
    module_playground: "مختبر الاستنتاج والاختبار",
    module_export: "تصدير الأكواد والنماذج",
    module_evaluation: "مختبر التقييم والتشخيص",
    module_deployment: "منصة النشر والتشغيل",
    module_continual: "التعلم المستمر واستدامة الذاكرة",

    // System Status
    system_status: "حالة النظام",
    engine_live: "محرك dytr الحي متصل",
    engine_sim: "وضع المحاكاة الاكاديمية",
    cuda_detected: "معالج الرسومات CUDA مفعل",
    cpu_mode: "معالجة حوسبية CPU",
    vram_label: "ذاكرة الفيديو",

    // General Navigation
    btn_start_wizard: "ابدا المسار الارشادي",
    btn_next_step: "الخطوة التالية",
    btn_prev_step: "الخطوة السابقة",
    btn_apply_preset: "تطبيق الاعداد الموصى به",
    btn_train_now: "بدء جلسة التدريب",
    btn_stop_train: "ايقاف التدريب",
    btn_test_model: "تجربة النموذج الان",
    btn_export_script: "تصدير كود بايثون",
    btn_export_colab: "تصدير دفتر Google Colab",
    btn_reset: "اعادة الضبط الافتراضي",
    btn_copy: "نسخ الكود",
    copied_success: "تم النسخ بنجاح",
    btn_go_to_next_module: "انتقل الى الخطوة التالية",
    btn_go_back: "العودة",
    step_of: "من",

    // =========================================================================
    // Academy Steps (10 Steps)
    // =========================================================================
    academy_title: "الاكاديمية التعليمية",
    academy_subtitle: "رحلة تعليمية متكاملة من الصفر -- من مفهوم الالة الى بناء نماذج الذكاء الاصطناعي",
    academy_why: "لماذا هذه الخطوة مهمة؟",

    // Step 1: What is a Machine?
    step1_title: "ما هي الالة؟",
    step1_why: "فهم مفهوم الالة هو الاساس الذي يبنى عليه كل شيء في علوم الحاسوب والذكاء الاصطناعي.",
    step1_content_1: "الالة هي نظام يحوّل مدخلات الى مخرجات وفق قواعد محددة. هذا التعريف البسيط هو اساس كل شيء.",
    step1_content_2: "الالة التقليدية: قواعد يكتبها الانسان صراحة. مثال: الة حاسبة. القاعدة: اذا ضغطت 2+2، اظهر 4. المشكلة: لا يمكن كتابة قاعدة لكل احتمال.",
    step1_content_3: "الالة الذكية (نظام AI): قواعد تُستخرج من البيانات، لا تُكتب يدويا. مثال: نموذج ترجمة. لا احد كتب كيف تُترجم كلمة 'book' -- النموذج استنتجها من ملايين الامثلة.",
    step1_content_4: "النقطة الجوهرية: الالة الذكية لا تفهم بالمعنى البشري. هي تكتشف انماطا احصائية في البيانات.",
    step1_diagram_input: "مدخل",
    step1_diagram_machine: "الة",
    step1_diagram_output: "مخرج",
    step1_diagram_rules: "قواعد",
    step1_traditional: "الة تقليدية",
    step1_smart: "الة ذكية",
    step1_traditional_desc: "الانسان يكتب القواعد",
    step1_smart_desc: "(من البيانات تستنتج الانماط) البيانات تستخرج القواعد",

    // Step 2: What is AI?
    step2_title: "ما هو الذكاء الاصطناعي؟",
    step2_why: "معرفة فروع الذكاء الاصطناعي تساعدك على تحديد اي فرع يناسب مشكلتك.",
    step2_content_1: "الذكاء الاصطناعي هو مجال علمي يهدف لبناء انظمة تؤدي مهام كانت تتطلب ذكاء بشريا.",
    step2_branch_ml: "تعلم الالة (Machine Learning)",
    step2_branch_ml_desc: "انظمة تتعلم من البيانات دون برمجة صريحة. مثال: تصنيف البريد المزعج.",
    step2_branch_ml_tooltip: "تعلم الالة هو فرع من الذكاء الاصطناعي يمكّن الحاسوب من التعلم من الامثلة بدلا من البرمجة المباشرة.",
    step2_branch_dl: "التعلم العميق (Deep Learning)",
    step2_branch_dl_desc: "تعلم الة باستخدام شبكات عصبية عميقة. مثال: التعرف على الوجوه.",
    step2_branch_dl_tooltip: "التعلم العميق يستخدم شبكات عصبية ذات طبقات متعددة لاكتشاف انماط معقدة في البيانات.",
    step2_branch_nlp: "معالجة اللغة الطبيعية (NLP)",
    step2_branch_nlp_desc: "فهم وانتاج اللغة البشرية. مثال: ChatGPT، الترجمة.",
    step2_branch_nlp_tooltip: "معالجة اللغة الطبيعية تمكّن الحاسوب من قراءة وفهم وانتاج النصوص البشرية.",
    step2_branch_cv: "الرؤية الحاسوبية (Computer Vision)",
    step2_branch_cv_desc: "فهم الصور والفيديو. مثال: السيارات ذاتية القيادة.",
    step2_branch_cv_tooltip: "الرؤية الحاسوبية تمكّن الحاسوب من تحليل وفهم محتوى الصور والفيديو.",
    step2_branch_rl: "التعلم المعزز (Reinforcement Learning)",
    step2_branch_rl_desc: "التعلم عبر التجربة والمكافاة. مثال: AlphaGo.",
    step2_branch_rl_tooltip: "التعلم المعزز يتعلم فيه النظام عبر التجربة والخطا والحصول على مكافاة عند الاداء الصحيح.",

    // Step 3: How does a machine learn?
    step3_title: "كيف تتعلم الالة؟",
    step3_why: "فهم عملية التعلم هو مفتاح فهم كيف تعمل كل نماذج الذكاء الاصطناعي.",
    step3_content_1: "المبدا الاساسي: التعلم = تعديل الاوزان لتقليل الخطا.",
    step3_content_2: "تخيل انك تعلّم طفلا التمييز بين القطط والكلاب: تُريه صورة، تساله ما هذا؟ يجيب خطا، تصحح له. يُعدّل فهمه قليلا. تكرر العملية الاف المرات حتى يتقن.",
    step3_content_3: "النموذج يفعل الشيء نفسه، لكن بارقام.",
    step3_cycle_forward: "الانتشار الامامي (Forward Pass)",
    step3_cycle_forward_desc: "النموذج يُنتج توقعا بناء على ما تعلمه حتى الان.",
    step3_cycle_forward_tooltip: "الانتشار الامامي هو المرحلة التي يقوم فيها النموذج بحساب المخرجات من المدخلات عبر كل طبقاته.",
    step3_cycle_loss: "حساب الخطا (Loss)",
    step3_cycle_loss_desc: "قياس الفرق بين التوقع والحقيقة.",
    step3_cycle_loss_tooltip: "دالة الخسارة (Loss Function) تقيس مقدار الفرق بين ما توقعه النموذج والاجابة الصحيحة. كلما قل الرقم، كان النموذج افضل.",
    step3_cycle_backprop: "الانتشار العكسي (Backpropagation)",
    step3_cycle_backprop_desc: "حساب من المسؤول عن الخطا.",
    step3_cycle_backprop_tooltip: "الانتشار العكسي يحسب مساهمة كل وزن في الخطا الكلي، ليعرف النموذج اي اوزان يجب تعديلها واي اتجاه.",
    step3_cycle_update: "تحديث الاوزان (Optimization)",
    step3_cycle_update_desc: "تعديل الاوزان لتقليل الخطا.",
    step3_cycle_update_tooltip: "المحسّن (Optimizer) مثل Adam او SGD يعدّل الاوزان بخطوات صغيرة محسوبة لتقليل الخطا تدريجيا.",
    step3_training_label: "التدريب (Training)",
    step3_training_desc: "هذه الدورة تُكرر ملايين المرات حتى يتعلم النموذج.",

    // Step 4: What is a Model (Function)?
    step4_title: "ما هو النموذج (الدالة)؟",
    step4_why: "النموذج هو الجوهر الذي ستبنيه وتدربه. فهم ماهيته ضروري قبل البدء.",
    step4_content_1: "النموذج هو بنية رياضية تحوّل المدخلات الى مخرجات.",
    step4_content_2: "فكّر في النموذج كمصنع ذكي: تدخل بيانات (المواد الخام)، يمر عبر طبقات من العمليات الحسابية (خطوط الانتاج)، ويخرج اجابة (المنتج).",
    step4_diagram_input: "بيانات x",
    step4_diagram_function: "المصنع f",
    step4_diagram_output: "اجابة y",
    step4_params_title: "المعاملات (Parameters)",
    step4_params_desc: "ارقام يتعلمها النموذج من البيانات. قد تكون ملايين او مليارات.",
    step4_params_tooltip: "المعاملات هي الاوزان الرقمية التي يعدّلها النموذج اثناء التدريب. كلما زاد عددها، زادت قدرة النموذج على تمثيل المعرفة.",
    step4_arch_title: "المعمارية (Architecture)",
    step4_arch_desc: "كيفية ترتيب المعاملات: طبقات، اتصالات، وعمليات.",
    step4_arch_tooltip: "المعمارية تحدد شكل النموذج: كم طبقة فيه، كيف تتصل ببعضها، وما نوع العمليات الحسابية في كل طبقة.",
    step4_foundation: "النموذج الاساسي (Foundation Model)",
    step4_foundation_desc: "نموذج ضخم دُرّب على بيانات عامة، يُعاد استخدامه لمهام محددة. مثال: BERT، GPT، LLaMA.",
    step4_foundation_tooltip: "النماذج الاساسية تُدرّب على كميات هائلة من البيانات (مثل كل الانترنت) ثم تُكيّف لمهام محددة عبر الضبط الدقيق.",
    step4_finetuning: "الضبط الدقيق (Fine-tuning)",
    step4_finetuning_desc: "تدريب النموذج المُسبق على مهمتك الخاصة بكمية صغيرة من البيانات.",
    step4_finetuning_tooltip: "الضبط الدقيق ياخذ نموذجا مدربا مسبقا ويعيد تدريبه على بياناتك المحددة. هذا اسرع واكثر كفاءة من التدريب من الصفر.",

    // Step 5: What is a Token? (Interactive)
    step5_title: "ما هو التوكن؟",
    step5_why: "الحاسوب لا يقرا الحروف كما نفعل نحن. فهم الترميز يوضح كيف يعالج النموذج النصوص.",
    step5_content_1: "الحواسيب لا تقرا الحروف كما نفعل نحن. لذلك نكسر الجملة الى قطع صغيرة تسمى توكنات.",
    step5_content_2: "التوكن قد يكون كلمة كاملة، او جزءا من كلمة، او حتى حرفا واحدا. كل توكن يحصل على رقم فريد.",
    step5_content_3: "تخيل التوكن وكانه قطعة ليجو. نجمع القطع لتكوين الكلمات والجمل، ونعطي لكل قطعة رقما ليفهمه الحاسوب.",
    step5_try_it: "جرّب بنفسك: اكتب نصا وشاهد كيف يتحول الى توكنات",
    step5_input_placeholder: "اكتب جملة هنا لتجزئتها...",
    step5_token_id: "الرقم",
    step5_token_piece: "القطعة",
    step5_parent_word: "الكلمة الام",
    step5_is_subword: "جزء من كلمة",

    // Step 6: Embeddings
    step6_title: "المتجهات المضمنة (Embeddings)",
    step6_why: "التضمين يحوّل الكلمات الى ارقام يستطيع النموذج فهمها ومقارنتها.",
    step6_content_1: "بعد تحويل الكلمات الى ارقام (توكنات)، نحتاج وضعها في خريطة رقمية متعددة الابعاد.",
    step6_content_2: "في هذه الخريطة، الكلمات المتشابهة في المعنى تكون قريبة من بعضها. مثلا: 'ملك' و'ملكة' قريبتان، و'قطة' و'كلب' قريبتان.",
    step6_content_3: "كل توكن يتحول الى متجه (قائمة من الارقام) بطول محدد، مثلا 256 رقما. هذه الارقام تمثل معنى الكلمة في فضاء رياضي.",
    step6_tooltip: "المتجه المضمن (Embedding Vector) هو تمثيل رقمي للكلمة في فضاء متعدد الابعاد، حيث تكون الكلمات المتقاربة في المعنى قريبة رقميا.",

    // Step 7: Self-Attention (Interactive)
    step7_title: "الاهتمام الذاتي (Self-Attention)",
    step7_why: "الية الاهتمام هي الابتكار الذي جعل المحولات (Transformers) تتفوق على كل المعماريات السابقة.",
    step7_content_1: "الاهتمام الذاتي هو طريقة تجعل كل كلمة في الجملة تنظر لكل الكلمات الاخرى وتقرر ايها اكثر اهمية لفهم معناها.",
    step7_content_2: "مثال: في جملة 'الطالب ذهب الى المدرسة لانه يحب العلم'، كلمة 'لانه' تحتاج ان تعرف ان 'هو' تعود على 'الطالب' لا على 'المدرسة'.",
    step7_content_3: "الاهتمام يحسب درجة ارتباط كل كلمتين، ثم يعطي وزنا اكبر للكلمات الاكثر صلة.",
    step7_tooltip: "الية الاهتمام الذاتي تحسب ثلاث مصفوفات: الاستعلام (Query)، المفتاح (Key)، والقيمة (Value). حاصل ضرب Q و K يعطي درجات الاهتمام، التي تُضرب في V للحصول على التمثيل النهائي.",

    // Step 8: Multi-Task Learning
    step8_title: "التعلم متعدد المهام (Multi-Task Learning)",
    step8_why: "مكتبة dytr مصممة خصيصا لهذا النوع من التعلم. فهمه يساعدك على استخدام المنصة بكفاءة.",
    step8_content_1: "بدلا من بناء نموذج منفصل لكل مهمة، نبني نموذجا واحدا يتعلم عدة مهام معا.",
    step8_content_2: "تخيل شجرة: الجذع (العمود الفقري المشترك) واحد، لكن الفروع (رؤوس المهام) متعددة. كل فرع متخصص في مهمة مختلفة.",
    step8_content_3: "الفائدة: الطبقات المشتركة تتعلم تمثيلات عامة افضل، واستهلاك الذاكرة اقل من بناء نماذج منفصلة.",
    step8_backbone: "العمود الفقري المشترك (Shared Backbone)",
    step8_backbone_tooltip: "العمود الفقري المشترك هو الجزء من النموذج الذي تشترك فيه كل المهام. يتعلم تمثيلات لغوية عامة تفيد الجميع.",
    step8_task_head: "راس المهمة (Task Head)",
    step8_task_head_tooltip: "راس المهمة هو طبقة متخصصة في الاعلى تُضاف لكل مهمة. مثلا: راس تصنيف المشاعر، راس تصنيف الموضوعات.",

    // Step 9: Catastrophic Forgetting & Continual Learning (Interactive)
    step9_title: "النسيان الكارثي والتعلم المستمر",
    step9_why: "حل مشكلة النسيان الكارثي هو القيمة الجوهرية لمكتبة dytr وما يميزها عن غيرها.",
    step9_content_1: "المشكلة: عندما يتعلم النموذج مهمة جديدة، ينسى المهمة القديمة. هذا يسمى النسيان الكارثي (Catastrophic Forgetting).",
    step9_content_2: "الحل في dytr: خوارزمية EWC (Elastic Weight Consolidation) تحدد الاوزان المهمة للمهام القديمة وتحميها من التغيير الكبير.",
    step9_content_3: "حل اضافي: ذاكرة الاسترجاع (Experience Replay) تحتفظ بعينات من المهام القديمة وتعيد تدريبها مع المهمة الجديدة.",
    step9_ewc_tooltip: "EWC (Elastic Weight Consolidation) خوارزمية تقيس اهمية كل وزن بالنسبة للمهام السابقة عبر مصفوفة فيشر (Fisher Information Matrix)، ثم تضيف عقوبة على تغيير الاوزان المهمة.",
    step9_replay_tooltip: "ذاكرة الاسترجاع (Experience Replay) تخزّن عينة صغيرة من بيانات المهام السابقة، ثم تدرّبها مع بيانات المهمة الجديدة لمنع النسيان.",
    step9_with_protection: "مع الحماية (EWC + Replay)",
    step9_without_protection: "بدون حماية",
    step9_toggle_label: "شغّل/اوقف الحماية لترى الفرق",

    // Step 10: Multi-Task LoRA
    step10_title: "التكيف منخفض الرتبة (Multi-Task LoRA)",
    step10_why: "LoRA يمكّنك من تدريب نماذج ضخمة على اجهزة محدودة الموارد وتوجيه كل مهمة بغلاف مخصص.",
    step10_content_1: "المشكلة: النماذج الكبيرة تحتوي على ملايين او مليارات المعاملات. تدريبها كلها يحتاج ذاكرة ووقتا هائلين.",
    step10_content_2: "الحل: LoRA يُجمّد كل الاوزان الاصلية ويضيف مصفوفتين صغيرتين جدا قابلتين للتدريب (A و B). هذا يقلل المعاملات المدربة بأكثر من 90%.",
    step10_content_3: "تتميز مكتبة dytr بدعم دمج مصفوفات LoRA مع الأوزان الأساسية مباشرة (LoRALayer.merge) بدون أي تأخير استنتاج.",
    step10_frozen: "مصفوفة مجمدة (اوزان اصلية)",
    step10_trainable: "مصفوفة صغيرة (قابلة للتدريب)",
    step10_frozen_tooltip: "الاوزان الاصلية للنموذج المدرب مسبقا. لا يتم تعديلها اثناء الضبط الدقيق باستخدام LoRA.",
    step10_trainable_tooltip: "مصفوفتان صغيرتان (A و B) حجمهما اقل بكثير من المصفوفة الاصلية. فقط هاتان المصفوفتان يتم تدريبهما.",

    // =========================================================================
    // Page 02: Data Preparation
    // =========================================================================
    dataprep_title: "تجهيز وتنقيب البيانات",
    dataprep_subtitle: "البيانات هي الوقود الذي يتعلم منه النموذج. جودة البيانات تحدد جودة النتائج.",
    dataprep_why: "بدون بيانات جيدة، لن يتعلم النموذج بشكل صحيح مهما كانت المعمارية متقدمة.",

    data_source_title: "اختر مصدر البيانات",
    data_source_prebuilt: "مجموعات بيانات جاهزة",
    data_source_prebuilt_desc: "مجموعات بيانات اكاديمية مُعدّة مسبقا للتجربة والتعلم.",
    data_source_upload: "رفع ملف بيانات",
    data_source_upload_desc: "ارفع ملف CSV او JSON يحتوي على بياناتك الخاصة.",

    data_quality_title: "فحص جودة البيانات",
    data_quality_rows: "عدد الصفوف",
    data_quality_balance: "توزيع الفئات",
    data_quality_missing: "بيانات ناقصة",
    data_quality_warning_imbalance: "تحذير: البيانات غير متوازنة. الفئة الاكبر تمثل اكثر من 80% من البيانات. هذا قد يجعل النموذج متحيزا.",
    data_quality_tooltip: "جودة البيانات تعني: هل البيانات نظيفة (بلا اخطاء)؟ هل الفئات متوازنة (ليس فئة واحدة تطغى على الباقي)؟ هل هناك بيانات ناقصة؟",

    data_split_title: "تقسيم البيانات",
    data_split_train: "التدريب (Train)",
    data_split_train_desc: "البيانات التي يتعلم منها النموذج.",
    data_split_val: "التحقق (Validation)",
    data_split_val_desc: "بيانات لضبط الاعدادات اثناء التدريب.",
    data_split_test: "الاختبار (Test)",
    data_split_test_desc: "بيانات للحكم النهائي. لا تُلمس اثناء التدريب.",
    data_split_tooltip: "تقسيم البيانات ضروري لمنع تسريب البيانات (Data Leakage). اذا استخدمت بيانات الاختبار في التدريب، ستحصل على نتائج مضللة.",

    // =========================================================================
    // Page 03: Architecture Builder
    // =========================================================================
    builder_title: "مختبر بناء النماذج الذكية",
    builder_subtitle: "بناء وتخصيص معمارية النموذج الذكي ليتعلم من بياناتك، بلغة بسيطة ومخططات بصرية تفاعلية.",

    complexity_auto: "تلقائي",
    complexity_auto_desc: "المنصة تختار لك الاعدادات المناسبة بناء على بياناتك.",
    complexity_medium: "متوسط",
    complexity_medium_desc: "تختار الاعدادات الاساسية: حجم النموذج ونوع المهمة.",
    complexity_advanced: "متقدم",
    complexity_advanced_desc: "تحكم كامل في كل اعداد: الطبقات، الرؤوس، الابعاد.",

    config_vocab_size: "حجم المفردات",
    config_vocab_size_tooltip: "عدد الكلمات الفريدة التي يعرفها النموذج. عادة 30000 الى 50000 كلمة.",
    config_embed_dim: "ابعاد التضمين",
    config_embed_dim_tooltip: "طول المتجه الذي يمثل كل كلمة. كلما زاد، زادت قدرة النموذج على تمثيل المعاني الدقيقة.",
    config_num_layers: "عدد الطبقات",
    config_num_layers_tooltip: "كل طبقة تضيف مستوى من الفهم. المزيد من الطبقات = فهم اعمق، لكن تدريب ابطا.",
    config_num_heads: "عدد رؤوس الاهتمام",
    config_num_heads_tooltip: "كل راس اهتمام ينظر للنص من زاوية مختلفة. مثل فريق محللين كل واحد متخصص في جانب.",

    task_head_manager_title: "مدير المهام",
    task_head_add: "اضف مهمة جديدة",
    task_head_name: "اسم المهمة",
    task_head_type: "نوع المهمة",
    task_head_classes: "عدد الفئات",
    task_type_classification: "تصنيف الجمل",
    task_type_ner: "تصنيف الكلمات (NER)",
    task_type_generation: "توليد النصوص",

    arch_summary_title: "ملخص المعمارية",
    arch_summary_params: "اجمالي المعاملات",
    arch_summary_memory: "حجم الذاكرة المطلوب",
    arch_summary_estimated_time: "الوقت المتوقع للتدريب",

    // =========================================================================
    // Page 04: Visual Glassroom
    // =========================================================================
    glassroom_title: "الغرفة الزجاجية للمحاكاة",
    glassroom_subtitle: "شاهد كيف يعمل النموذج من الداخل عبر محاكيات تفاعلية.",

    glassroom_token_path: "مسار التوكن",
    glassroom_token_path_desc: "اكتب نصا وشاهد كيف يتحول الى ارقام يفهمها النموذج.",
    glassroom_rope: "تضمين الموضع الدوراني (RoPE)",
    glassroom_rope_desc: "شاهد كيف يتم تدوير المتجهات حسب موقع الكلمة في الجملة.",
    glassroom_flash: "الاهتمام السريع (FlashAttention)",
    glassroom_adalora: "أغلفة التكيف متعددة المهام (Multi-Task LoRA)",
    glassroom_adalora_desc: "شاهد كيف يتم تدريب مصفوفات LoRA لكل مهمة مع تجميد أوزان النموذج الأساسي.",

    // =========================================================================
    // Page 05: Training Arena
    // =========================================================================
    arena_title: "التدريب",
    arena_subtitle: "ابدا تدريب نموذجك وتابع تقدمه لحظيا.",

    arena_pre_summary: "ملخص ما قبل التدريب",
    arena_dataset: "البيانات المختارة",
    arena_architecture: "المعمارية",
    arena_tasks: "المهام",

    arena_config_title: "اعدادات التدريب",
    arena_epochs: "عدد الحقب (Epochs)",
    arena_epochs_tooltip: "الحقبة الواحدة = مرور كامل على كل بيانات التدريب. عادة 3 الى 10 حقب.",
    arena_batch_size: "حجم الدفعة (Batch Size)",
    arena_batch_size_tooltip: "عدد الامثلة التي يعالجها النموذج في كل خطوة. اكبر = اسرع لكن يحتاج ذاكرة اكثر.",
    arena_lr: "معدل التعلم (Learning Rate)",
    arena_lr_tooltip: "حجم الخطوة في تحديث الاوزان. صغير جدا = بطيء. كبير جدا = غير مستقر. القيمة المعتادة: 0.0001 الى 0.001.",
    arena_slider_label: "اسرع / ادق",

    arena_training_status: "حالة التدريب",
    arena_training_progress: "جاري التعلم...",
    arena_training_complete: "اكتمل التدريب",
    arena_time_remaining: "الوقت المتبقي المتوقع",

    // =========================================================================
    // Page 06: Observatory
    // =========================================================================
    observatory_title: "مرصد القياسات الحية",
    observatory_subtitle: "تابع مقاييس التدريب لحظيا بلغة مفهومة.",

    obs_loss_label: "الخطا (Loss)",
    obs_loss_desc_improving: "النموذج يتحسن بشكل جيد",
    obs_loss_desc_stable: "النموذج وصل لمستوى مستقر",
    obs_loss_desc_stuck: "النموذج متوقف عن التحسن -- جرّب تغيير الاعدادات",

    obs_acc_label: "الدقة (Accuracy)",
    obs_acc_desc: "من كل 100 تنبؤ، {n} صحيحة",

    obs_bwt_label: "مقياس النسيان العكسي (BWT)",
    obs_bwt_desc_good: "النموذج حافظ على {n}% من معرفته السابقة",
    obs_bwt_desc_bad: "النموذج فقد {n}% من معرفته السابقة -- تحقق من اعدادات EWC",

    obs_rmatrix_title: "مصفوفة الاداء عبر المهام (R-Matrix)",
    obs_rmatrix_tooltip: "كل خلية تعرض دقة مهمة معينة بعد تدريب مرحلة معينة. القطر يعرض الاداء المباشر، والقيم خارج القطر تعرض تاثير التعلم المتقاطع.",

    // =========================================================================
    // Page 07: Inference Playground
    // =========================================================================
    playground_title: "مختبر الاستنتاج والاختبار",
    playground_subtitle: "جرّب النموذج المدرب مباشرة. ادخل نصا واحصل على تنبؤ.",

    playground_input_label: "ادخل النص",
    playground_input_placeholder: "اكتب جملة هنا لاختبار النموذج...",
    playground_select_task: "اختر المهمة",
    playground_run: "تشغيل التنبؤ",

    playground_result_title: "النتيجة",
    playground_prediction: "التنبؤ",
    playground_confidence: "نسبة الثقة",
    playground_probabilities: "توزيع الاحتمالات",

    playground_history_title: "سجل التنبؤات السابقة",
    playground_history_empty: "لم يتم اجراء اي تنبؤات بعد.",

    // =========================================================================
    // Page 08: Export
    // =========================================================================
    export_title: "تصدير النماذج والاكواد",
    export_subtitle: "صدّر نموذجك بالصيغة المناسبة واحصل على كود جاهز للاستخدام.",

    export_format_pt: "PyTorch (.pt)",
    export_format_pt_desc: "الصيغة الاصلية. مناسبة للتدريب والضبط الدقيق.",
    export_format_onnx: "ONNX (.onnx)",
    export_format_onnx_desc: "صيغة مفتوحة تعمل في كل البيئات والمنصات.",
    export_format_safetensors: "Safetensors",
    export_format_safetensors_desc: "صيغة امنة وسريعة التحميل من Hugging Face.",

    export_code_title: "كود الاستخدام",
    export_code_python: "Python",
    export_code_javascript: "JavaScript",
    export_code_curl: "cURL",

    export_download: "تحميل النموذج",
    export_size: "الحجم المتوقع",
  },

  en: {
    // Navigation Modules (12 Modules)
    module_problem: "Problem Framing",
    module_academy: "AI Academy",
    module_dataprep: "Data Prep & Mining",
    module_builder: "AI Model Builder",
    module_glassroom: "Neural Mechanics Lab",
    module_arena: "Training Arena",
    module_observatory: "Live Observatory",
    module_playground: "Inference Playground",
    module_export: "Export Models & Code",
    module_evaluation: "Evaluation & Diagnostics",
    module_deployment: "Deployment Console",
    module_continual: "Continual Learning",

    // System Status
    system_status: "System Status",
    engine_live: "dytr Live Engine Connected",
    engine_sim: "Academic Simulation Mode",
    cuda_detected: "CUDA GPU Accelerator Active",
    cpu_mode: "CPU Compute Mode",
    vram_label: "Video Memory",

    // General Navigation
    btn_start_wizard: "Start Guided Academy",
    btn_next_step: "Next Step",
    btn_prev_step: "Previous Step",
    btn_apply_preset: "Apply Recommended Preset",
    btn_train_now: "Start Training Session",
    btn_stop_train: "Halt Training",
    btn_test_model: "Test Model Live",
    btn_export_script: "Export Python Script",
    btn_export_colab: "Export Google Colab Notebook",
    btn_reset: "Reset Defaults",
    btn_copy: "Copy Code",
    copied_success: "Copied to Clipboard",
    btn_go_to_next_module: "Go to Next Step",
    btn_go_back: "Go Back",
    step_of: "of",

    // =========================================================================
    // Academy Steps (10 Steps)
    // =========================================================================
    academy_title: "AI Academy",
    academy_subtitle: "A complete educational journey from zero -- from the concept of a machine to building AI models",
    academy_why: "Why is this step important?",

    // Step 1: What is a Machine?
    step1_title: "What is a Machine?",
    step1_why: "Understanding the concept of a machine is the foundation upon which everything in computer science and AI is built.",
    step1_content_1: "A machine is a system that converts inputs into outputs according to defined rules. This simple definition is the basis of everything.",
    step1_content_2: "Traditional Machine: rules written explicitly by humans. Example: a calculator. Rule: if you press 2+2, display 4. Problem: you cannot write a rule for every possibility.",
    step1_content_3: "Smart Machine (AI System): rules extracted from data, not written manually. Example: a translation model. Nobody wrote how to translate the word 'book' -- the model figured it out from millions of examples.",
    step1_content_4: "The key point: a smart machine does not 'understand' in the human sense. It discovers statistical patterns in data.",
    step1_diagram_input: "Input",
    step1_diagram_machine: "Machine",
    step1_diagram_output: "Output",
    step1_diagram_rules: "Rules",
    step1_traditional: "Traditional Machine",
    step1_smart: "Smart Machine",
    step1_traditional_desc: "Humans write the rules",
    step1_smart_desc: "Data extracts the rules",

    // Step 2: What is AI?
    step2_title: "What is Artificial Intelligence?",
    step2_why: "Knowing the branches of AI helps you identify which branch fits your problem.",
    step2_content_1: "Artificial Intelligence is a scientific field aimed at building systems that perform tasks which previously required human intelligence.",
    step2_branch_ml: "Machine Learning (ML)",
    step2_branch_ml_desc: "Systems that learn from data without explicit programming. Example: spam classification.",
    step2_branch_ml_tooltip: "Machine Learning is a branch of AI that enables computers to learn from examples instead of direct programming.",
    step2_branch_dl: "Deep Learning (DL)",
    step2_branch_dl_desc: "ML using deep neural networks. Example: face recognition.",
    step2_branch_dl_tooltip: "Deep Learning uses neural networks with multiple layers to discover complex patterns in data.",
    step2_branch_nlp: "Natural Language Processing (NLP)",
    step2_branch_nlp_desc: "Understanding and generating human language. Example: ChatGPT, translation.",
    step2_branch_nlp_tooltip: "NLP enables computers to read, understand, and generate human text.",
    step2_branch_cv: "Computer Vision (CV)",
    step2_branch_cv_desc: "Understanding images and video. Example: self-driving cars.",
    step2_branch_cv_tooltip: "Computer Vision enables computers to analyze and understand the content of images and video.",
    step2_branch_rl: "Reinforcement Learning (RL)",
    step2_branch_rl_desc: "Learning through trial and reward. Example: AlphaGo.",
    step2_branch_rl_tooltip: "In Reinforcement Learning, the system learns by trial and error, receiving rewards for correct actions.",

    // Step 3: How does a machine learn?
    step3_title: "How Does a Machine Learn?",
    step3_why: "Understanding the learning process is the key to understanding how all AI models work.",
    step3_content_1: "The fundamental principle: Learning = adjusting weights to reduce error.",
    step3_content_2: "Imagine teaching a child to distinguish cats from dogs: you show them an image, ask 'what is this?'. They answer wrong, you correct them. They adjust their understanding slightly. You repeat thousands of times until they master it.",
    step3_content_3: "The model does the same thing, but with numbers.",
    step3_cycle_forward: "Forward Pass",
    step3_cycle_forward_desc: "The model produces a prediction based on what it has learned so far.",
    step3_cycle_forward_tooltip: "The Forward Pass is the stage where the model calculates outputs from inputs through all its layers.",
    step3_cycle_loss: "Loss Calculation",
    step3_cycle_loss_desc: "Measuring the difference between prediction and truth.",
    step3_cycle_loss_tooltip: "The Loss Function measures how far the model's prediction is from the correct answer. Lower loss = better model.",
    step3_cycle_backprop: "Backpropagation",
    step3_cycle_backprop_desc: "Calculating which weights are responsible for the error.",
    step3_cycle_backprop_tooltip: "Backpropagation calculates each weight's contribution to the total error, so the model knows which weights to adjust and in which direction.",
    step3_cycle_update: "Weight Update (Optimization)",
    step3_cycle_update_desc: "Adjusting the weights to reduce the error.",
    step3_cycle_update_tooltip: "The Optimizer (like Adam or SGD) adjusts weights in small calculated steps to gradually reduce the error.",
    step3_training_label: "Training",
    step3_training_desc: "This cycle repeats millions of times until the model learns.",

    // Step 4: What is a Model (Function)?
    step4_title: "What is a Model (Function)?",
    step4_why: "The model is the core thing you will build and train. Understanding what it is is essential before starting.",
    step4_content_1: "A model is a mathematical structure that converts inputs into outputs.",
    step4_content_2: "Think of a model as a smart factory: data goes in (raw materials), passes through layers of computation (production lines), and an answer comes out (the product).",
    step4_diagram_input: "Data x",
    step4_diagram_function: "Factory f",
    step4_diagram_output: "Answer y",
    step4_params_title: "Parameters",
    step4_params_desc: "Numbers the model learns from data. Can be millions or billions.",
    step4_params_tooltip: "Parameters are the numerical weights that the model adjusts during training. More parameters = greater capacity to represent knowledge.",
    step4_arch_title: "Architecture",
    step4_arch_desc: "How parameters are arranged: layers, connections, and operations.",
    step4_arch_tooltip: "Architecture defines the model's shape: how many layers, how they connect, and what type of computation each layer performs.",
    step4_foundation: "Foundation Model",
    step4_foundation_desc: "A large model pre-trained on general data, reused for specific tasks. Examples: BERT, GPT, LLaMA.",
    step4_foundation_tooltip: "Foundation models are trained on massive amounts of data (like the entire internet) then adapted for specific tasks through fine-tuning.",
    step4_finetuning: "Fine-tuning",
    step4_finetuning_desc: "Training a pre-trained model on your specific task with a small amount of data.",
    step4_finetuning_tooltip: "Fine-tuning takes a pre-trained model and retrains it on your specific data. This is faster and more efficient than training from scratch.",

    // Step 5
    step5_title: "What is a Token?",
    step5_why: "Computers do not read letters the way we do. Understanding tokenization clarifies how models process text.",
    step5_content_1: "Computers do not read letters like we do. So we break sentences into small pieces called tokens.",
    step5_content_2: "A token may be a full word, part of a word, or even a single character. Each token gets a unique number.",
    step5_content_3: "Think of a token as a LEGO brick. We assemble bricks to form words and sentences, and give each brick a number for the computer.",
    step5_try_it: "Try it yourself: type text and watch it turn into tokens",
    step5_input_placeholder: "Type a sentence here to tokenize...",
    step5_token_id: "ID",
    step5_token_piece: "Piece",
    step5_parent_word: "Parent Word",
    step5_is_subword: "Subword",

    // Step 6
    step6_title: "Vector Embeddings",
    step6_why: "Embeddings convert words into numbers that the model can understand and compare.",
    step6_content_1: "After converting words to numbers (tokens), we need to place them on a multi-dimensional numeric map.",
    step6_content_2: "On this map, words with similar meanings are close to each other. For example: 'king' and 'queen' are close, and 'cat' and 'dog' are close.",
    step6_content_3: "Each token becomes a vector (a list of numbers) of a fixed length, like 256 numbers. These numbers represent the word's meaning in mathematical space.",
    step6_tooltip: "An Embedding Vector is a numerical representation of a word in multi-dimensional space, where semantically similar words are numerically close.",

    // Step 7
    step7_title: "Self-Attention",
    step7_why: "The attention mechanism is the innovation that made Transformers outperform all previous architectures.",
    step7_content_1: "Self-Attention is a method where every word in a sentence looks at all other words and decides which ones are most important for understanding its meaning.",
    step7_content_2: "Example: in 'The student went to school because he loves learning', the word 'he' needs to know it refers to 'student' not 'school'.",
    step7_content_3: "Attention computes the relevance between every pair of words, then gives more weight to the most relevant ones.",
    step7_tooltip: "Self-Attention computes three matrices: Query (Q), Key (K), and Value (V). The dot product of Q and K gives attention scores, which are multiplied by V to get the final representation.",

    // Step 8
    step8_title: "Multi-Task Learning (MTL)",
    step8_why: "The dytr library is specifically designed for this type of learning. Understanding it helps you use the platform efficiently.",
    step8_content_1: "Instead of building a separate model for each task, we build one model that learns multiple tasks together.",
    step8_content_2: "Imagine a tree: the trunk (shared backbone) is one, but the branches (task heads) are many. Each branch specializes in a different task.",
    step8_content_3: "Benefit: shared layers learn better general representations, and memory usage is lower than building separate models.",
    step8_backbone: "Shared Backbone",
    step8_backbone_tooltip: "The Shared Backbone is the part of the model that all tasks share. It learns general language representations that benefit everyone.",
    step8_task_head: "Task Head",
    step8_task_head_tooltip: "A Task Head is a specialized layer added on top for each task. Example: sentiment classification head, topic classification head.",

    // Step 9
    step9_title: "Catastrophic Forgetting & Continual Learning",
    step9_why: "Solving catastrophic forgetting is the core value of the dytr library and what sets it apart from others.",
    step9_content_1: "The Problem: when a model learns a new task, it forgets the old task. This is called Catastrophic Forgetting.",
    step9_content_2: "The Solution in dytr: EWC (Elastic Weight Consolidation) identifies the important weights for old tasks and protects them from large changes.",
    step9_content_3: "Additional solution: Experience Replay keeps samples from old tasks and retrains them with the new task.",
    step9_ewc_tooltip: "EWC (Elastic Weight Consolidation) measures the importance of each weight for previous tasks using the Fisher Information Matrix, then penalizes changes to important weights.",
    step9_replay_tooltip: "Experience Replay stores a small sample of previous task data, then trains it alongside new task data to prevent forgetting.",
    step9_with_protection: "With Protection (EWC + Replay)",
    step9_without_protection: "Without Protection",
    step9_toggle_label: "Toggle protection to see the difference",

    // Step 10
    step10_title: "Low-Rank Adaptation (Multi-Task LoRA)",
    step10_why: "LoRA enables you to train massive models on resource-limited hardware with task-isolated adapters.",
    step10_content_1: "The Problem: large models contain millions or billions of parameters. Training all of them requires enormous memory and time.",
    step10_content_2: "The Solution: LoRA freezes all original weights and adds two very small trainable matrices (A and B). This reduces trainable parameters by over 90%.",
    step10_content_3: "DyTR library supports direct weight merging via LoRALayer.merge() to eliminate any runtime inference latency overhead.",
    step10_frozen: "Frozen Matrix (original weights)",
    step10_trainable: "Small Matrix (trainable)",
    step10_frozen_tooltip: "The original pre-trained model weights. They are not modified during LoRA fine-tuning.",
    step10_trainable_tooltip: "Two small matrices (A and B) much smaller than the original. Only these two matrices are trained.",

    // =========================================================================
    // Page 02: Data Preparation
    // =========================================================================
    dataprep_title: "Data Preparation & Mining",
    dataprep_subtitle: "Data is the fuel the model learns from. Data quality determines result quality.",
    dataprep_why: "Without good data, the model will not learn correctly no matter how advanced the architecture.",

    data_source_title: "Choose Data Source",
    data_source_prebuilt: "Pre-built Datasets",
    data_source_prebuilt_desc: "Academic datasets pre-prepared for experimentation and learning.",
    data_source_upload: "Upload Data File",
    data_source_upload_desc: "Upload a CSV or JSON file containing your own data.",

    data_quality_title: "Data Quality Inspection",
    data_quality_rows: "Number of Rows",
    data_quality_balance: "Class Distribution",
    data_quality_missing: "Missing Data",
    data_quality_warning_imbalance: "Warning: Data is imbalanced. The largest class represents more than 80% of the data. This may cause a biased model.",
    data_quality_tooltip: "Data quality means: Is the data clean (error-free)? Are classes balanced? Is there missing data?",

    data_split_title: "Data Splitting",
    data_split_train: "Training",
    data_split_train_desc: "Data the model learns from.",
    data_split_val: "Validation",
    data_split_val_desc: "Data for tuning settings during training.",
    data_split_test: "Test",
    data_split_test_desc: "Data for final judgment. Not touched during training.",
    data_split_tooltip: "Splitting data is necessary to prevent Data Leakage. Using test data in training gives misleading results.",

    // =========================================================================
    // Page 03: Architecture Builder
    // =========================================================================
    builder_title: "AI Model Builder Lab",
    builder_subtitle: "Construct and customize the smart neural model with interactive visual schematics and simple human explanations.",

    complexity_auto: "Automatic",
    complexity_auto_desc: "The platform chooses appropriate settings based on your data.",
    complexity_medium: "Intermediate",
    complexity_medium_desc: "You choose basic settings: model size and task type.",
    complexity_advanced: "Advanced",
    complexity_advanced_desc: "Full control over every setting: layers, heads, dimensions.",

    config_vocab_size: "Vocabulary Size",
    config_vocab_size_tooltip: "The number of unique words the model knows. Usually 30,000 to 50,000 words.",
    config_embed_dim: "Embedding Dimensions",
    config_embed_dim_tooltip: "The length of the vector representing each word. Higher = more capacity to represent precise meanings.",
    config_num_layers: "Number of Layers",
    config_num_layers_tooltip: "Each layer adds a level of understanding. More layers = deeper understanding, but slower training.",
    config_num_heads: "Number of Attention Heads",
    config_num_heads_tooltip: "Each attention head looks at text from a different angle. Like a team of analysts, each specialized in an aspect.",

    task_head_manager_title: "Task Manager",
    task_head_add: "Add New Task",
    task_head_name: "Task Name",
    task_head_type: "Task Type",
    task_head_classes: "Number of Classes",
    task_type_classification: "Sentence Classification",
    task_type_ner: "Token Classification (NER)",
    task_type_generation: "Text Generation",

    arch_summary_title: "Architecture Summary",
    arch_summary_params: "Total Parameters",
    arch_summary_memory: "Required Memory",
    arch_summary_estimated_time: "Estimated Training Time",

    // =========================================================================
    // Page 04: Visual Glassroom
    // =========================================================================
    glassroom_title: "Simulation Glassroom",
    glassroom_subtitle: "See how the model works from the inside through interactive simulations.",

    glassroom_token_path: "Token Path",
    glassroom_token_path_desc: "Type text and watch how it converts to numbers the model understands.",
    glassroom_rope: "Rotary Position Embedding (RoPE)",
    glassroom_rope_desc: "Watch how vectors are rotated based on word position in the sentence.",
    glassroom_flash: "FlashAttention",
    glassroom_flash_desc: "Compare traditional vs. fast attention in terms of memory usage.",
    glassroom_adalora: "Multi-Task LoRA Adaptation",
    glassroom_adalora_desc: "Watch how LoRA matrices are trained for task isolation with frozen base weights.",

    // =========================================================================
    // Page 05: Training Arena
    // =========================================================================
    arena_title: "Training Arena",
    arena_subtitle: "Start training your model and follow its progress in real-time.",

    arena_pre_summary: "Pre-Training Summary",
    arena_dataset: "Selected Dataset",
    arena_architecture: "Architecture",
    arena_tasks: "Tasks",

    arena_config_title: "Training Settings",
    arena_epochs: "Number of Epochs",
    arena_epochs_tooltip: "One epoch = one complete pass through all training data. Usually 3 to 10 epochs.",
    arena_batch_size: "Batch Size",
    arena_batch_size_tooltip: "Number of examples the model processes in each step. Larger = faster but needs more memory.",
    arena_lr: "Learning Rate",
    arena_lr_tooltip: "The step size in weight updates. Too small = slow. Too large = unstable. Typical range: 0.0001 to 0.001.",
    arena_slider_label: "Faster / More Accurate",

    arena_training_status: "Training Status",
    arena_training_progress: "Learning in progress...",
    arena_training_complete: "Training Complete",
    arena_time_remaining: "Estimated Time Remaining",

    // =========================================================================
    // Page 06: Observatory
    // =========================================================================
    observatory_title: "Live Metrics Observatory",
    observatory_subtitle: "Follow training metrics in real-time in understandable language.",

    obs_loss_label: "Loss",
    obs_loss_desc_improving: "The model is improving well",
    obs_loss_desc_stable: "The model reached a stable level",
    obs_loss_desc_stuck: "The model stopped improving -- try changing settings",

    obs_acc_label: "Accuracy",
    obs_acc_desc: "Out of every 100 predictions, {n} are correct",

    obs_bwt_label: "Backward Transfer (BWT)",
    obs_bwt_desc_good: "The model retained {n}% of its previous knowledge",
    obs_bwt_desc_bad: "The model lost {n}% of its previous knowledge -- check EWC settings",

    obs_rmatrix_title: "Cross-Task Performance Matrix (R-Matrix)",
    obs_rmatrix_tooltip: "Each cell shows a specific task's accuracy after training a specific stage. Diagonal shows direct performance; off-diagonal shows cross-learning effects.",

    // =========================================================================
    // Page 07: Inference Playground
    // =========================================================================
    playground_title: "Inference & Testing Lab",
    playground_subtitle: "Try the trained model directly. Enter text and get a prediction.",

    playground_input_label: "Enter Text",
    playground_input_placeholder: "Type a sentence here to test the model...",
    playground_select_task: "Select Task",
    playground_run: "Run Prediction",

    playground_result_title: "Result",
    playground_prediction: "Prediction",
    playground_confidence: "Confidence",
    playground_probabilities: "Probability Distribution",

    playground_history_title: "Previous Predictions Log",
    playground_history_empty: "No predictions made yet.",

    // =========================================================================
    // Page 08: Export
    // =========================================================================
    export_title: "Export Models & Code",
    export_subtitle: "Export your model in the right format and get ready-to-use code.",

    export_format_pt: "PyTorch (.pt)",
    export_format_pt_desc: "The original format. Suitable for training and fine-tuning.",
    export_format_onnx: "ONNX (.onnx)",
    export_format_onnx_desc: "An open format that works across all environments and platforms.",
    export_format_safetensors: "Safetensors",
    export_format_safetensors_desc: "A safe and fast-loading format from Hugging Face.",

    export_code_title: "Usage Code",
    export_code_python: "Python",
    export_code_javascript: "JavaScript",
    export_code_curl: "cURL",

    export_download: "Download Model",
    export_size: "Expected Size",
  },
};
