# Copyright 2026 Jamal Al-Mansoub (Jamalalmansoub@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Educational Content API for DyTrStudio.

Provides bilingual (Arabic/English) concept explanations, metric translations
to human-readable language, and context-aware learning path suggestions.

Author: Jamal Al-Mansoub
Email: Jamalalmansoub@gmail.com
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

router = APIRouter(prefix="/learn", tags=["education"])


# Bilingual concept dictionary:
# Every academic term used in the platform has a plain-language explanation
# in both Arabic and English, along with a concrete analogy.
CONCEPT_LIBRARY: Dict[str, Dict[str, str]] = {
    "machine": {
        "en_title": "Machine",
        "ar_title": "الالة",
        "en_explanation": "A system that converts inputs into outputs according to rules. "
                          "A traditional machine follows rules written by humans. "
                          "An intelligent machine discovers its own rules from data.",
        "ar_explanation": "نظام يحوّل مدخلات الى مخرجات وفق قواعد محددة. "
                          "الالة التقليدية تتبع قواعد كتبها الانسان. "
                          "الالة الذكية تستخرج قواعدها من البيانات بنفسها.",
        "en_analogy": "Like a calculator: you give it 2+2, it gives you 4. "
                      "But an AI system learns the patterns from examples instead.",
        "ar_analogy": "مثل الالة الحاسبة: تعطيها 2+2 تعطيك 4. "
                      "لكن نظام الذكاء الاصطناعي يتعلم الانماط من الامثلة.",
        "category": "foundations",
    },
    "artificial_intelligence": {
        "en_title": "Artificial Intelligence",
        "ar_title": "الذكاء الاصطناعي",
        "en_explanation": "A scientific field that aims to build systems capable of performing "
                          "tasks that normally require human intelligence. "
                          "Branches: Machine Learning, Deep Learning, NLP, Computer Vision, Reinforcement Learning.",
        "ar_explanation": "مجال علمي يهدف لبناء انظمة قادرة على اداء مهام كانت تتطلب "
                          "ذكاء بشريا. "
                          "الفروع: تعلم الالة، التعلم العميق، معالجة اللغات، الرؤية الحاسوبية، التعلم المعزز.",
        "en_analogy": "Like teaching a child to recognize animals from pictures, "
                      "except the student is a computer program and the pictures are data.",
        "ar_analogy": "مثل تعليم طفل التعرف على الحيوانات من الصور، "
                      "الا ان الطالب هنا برنامج حاسوبي والصور هي بيانات.",
        "category": "foundations",
    },
    "model": {
        "en_title": "Model (Mathematical Function)",
        "ar_title": "النموذج (الدالة الرياضية)",
        "en_explanation": "A mathematical structure that converts inputs into outputs. "
                          "It consists of millions of numbers (parameters) that are adjusted "
                          "during training to produce accurate results.",
        "ar_explanation": "بنية رياضية تحوّل المدخلات الى مخرجات. "
                          "يتكون من ملايين الارقام (المعاملات) التي تُعدّل "
                          "اثناء التدريب لانتاج نتائج دقيقة.",
        "en_analogy": "Think of it as a smart factory: raw materials (data) go in, "
                      "finished products (predictions) come out.",
        "ar_analogy": "فكر فيه كمصنع ذكي: المواد الخام (البيانات) تدخل، "
                      "والمنتجات الجاهزة (التنبؤات) تخرج.",
        "category": "foundations",
    },
    "token": {
        "en_title": "Token",
        "ar_title": "التوكن (القطعة اللغوية)",
        "en_explanation": "Computers cannot read letters directly. Text is broken into small pieces "
                          "called tokens. Each piece gets a unique number. "
                          "A word might become one token or several sub-tokens.",
        "ar_explanation": "الحاسوب لا يستطيع قراءة الحروف مباشرة. يُكسر النص الى قطع صغيرة "
                          "تسمى توكنات. كل قطعة تحصل على رقم فريد. "
                          "الكلمة الواحدة قد تصبح توكن واحد او عدة توكنات فرعية.",
        "en_analogy": "Like building blocks (LEGO): a sentence is a building, "
                      "and tokens are the individual bricks.",
        "ar_analogy": "مثل مكعبات البناء (ليجو): الجملة هي المبنى، "
                      "والتوكنات هي الطوب الفردي.",
        "category": "nlp",
    },
    "embedding": {
        "en_title": "Embedding (Vector Representation)",
        "ar_title": "التضمين (التمثيل المتجهي)",
        "en_explanation": "A method of placing words on a numerical map where "
                          "similar words are close to each other. "
                          "Each word becomes a list of numbers (a vector).",
        "ar_explanation": "طريقة لوضع الكلمات على خريطة رقمية بحيث "
                          "تكون الكلمات المتشابهة قريبة من بعضها. "
                          "كل كلمة تصبح قائمة ارقام (متجه).",
        "en_analogy": "Imagine a city map where restaurants are near each other "
                      "and hospitals are near each other. Embeddings do the same with words.",
        "ar_analogy": "تخيل خريطة مدينة حيث المطاعم قريبة من بعضها "
                      "والمستشفيات قريبة من بعضها. التضمينات تفعل الشيء نفسه مع الكلمات.",
        "category": "nlp",
    },
    "attention": {
        "en_title": "Self-Attention",
        "ar_title": "الاهتمام الذاتي",
        "en_explanation": "A mechanism that allows the model to focus on the most relevant "
                          "words in a sentence when processing each word. "
                          "Every word looks at every other word and decides what matters.",
        "ar_explanation": "الية تسمح للنموذج بالتركيز على الكلمات الاكثر صلة "
                          "في الجملة عند معالجة كل كلمة. "
                          "كل كلمة تنظر لكل الكلمات الاخرى وتقرر ما يهمها.",
        "en_analogy": "When reading 'The cat sat on the mat because it was tired', "
                      "attention helps the model know that 'it' refers to 'the cat'.",
        "ar_analogy": "عند قراءة 'جلس القط على السجادة لانه كان متعبا'، "
                      "الاهتمام يساعد النموذج على معرفة ان 'هو' تشير الى 'القط'.",
        "category": "transformer",
    },
    "transformer": {
        "en_title": "Transformer",
        "ar_title": "المحوّل (Transformer)",
        "en_explanation": "A neural network architecture based on the Self-Attention mechanism. "
                          "It processes all words in a sentence simultaneously (not one by one). "
                          "This makes it faster and more accurate than older architectures.",
        "ar_explanation": "معمارية شبكة عصبية مبنية على الية الاهتمام الذاتي. "
                          "تعالج كل الكلمات في الجملة بشكل متزامن (ليس واحدة تلو الاخرى). "
                          "هذا يجعلها اسرع وادق من المعماريات القديمة.",
        "en_analogy": "Older models read a book word by word. "
                      "Transformer reads the entire page at once and understands context.",
        "ar_analogy": "النماذج القديمة تقرأ الكتاب كلمة كلمة. "
                      "المحوّل يقرأ الصفحة كاملة ويفهم السياق.",
        "category": "transformer",
    },
    "multi_task_learning": {
        "en_title": "Multi-Task Learning (MTL)",
        "ar_title": "التعلم متعدد المهام",
        "en_explanation": "Teaching a single model multiple skills at the same time. "
                          "A shared backbone learns general patterns, "
                          "while specialized heads learn task-specific patterns.",
        "ar_explanation": "تعليم نموذج واحد عدة مهارات في نفس الوقت. "
                          "عمود فقري مشترك يتعلم الانماط العامة، "
                          "ورؤوس متخصصة تتعلم الانماط الخاصة بكل مهمة.",
        "en_analogy": "Like a medical student who learns anatomy, physiology, "
                      "and pharmacology together -- they reinforce each other.",
        "ar_analogy": "مثل طالب طب يتعلم التشريح وعلم وظائف الاعضاء "
                      "والصيدلة معا -- كل مادة تعزز الاخرى.",
        "category": "continual_learning",
    },
    "catastrophic_forgetting": {
        "en_title": "Catastrophic Forgetting",
        "ar_title": "النسيان الكارثي",
        "en_explanation": "When a model learns a new task, it forgets what it learned before. "
                          "This happens because new training overwrites old knowledge. "
                          "Solutions include EWC and Experience Replay.",
        "ar_explanation": "عندما يتعلم النموذج مهمة جديدة، ينسى ما تعلمه سابقا. "
                          "يحدث هذا لان التدريب الجديد يكتب فوق المعرفة القديمة. "
                          "الحلول تشمل EWC واعادة تشغيل التجارب.",
        "en_analogy": "Like learning French and then studying Spanish so intensively "
                      "that you forget your French. EWC is like revision notes that prevent this.",
        "ar_analogy": "مثل تعلم الفرنسية ثم دراسة الاسبانية بشكل مكثف "
                      "حتى تنسى الفرنسية. EWC مثل ملاحظات المراجعة التي تمنع ذلك.",
        "category": "continual_learning",
    },
    "ewc": {
        "en_title": "Elastic Weight Consolidation (EWC)",
        "ar_title": "تثبيت الاوزان المرن (EWC)",
        "en_explanation": "A technique that identifies which parameters (weights) were important "
                          "for previous tasks and penalizes changes to them during new training. "
                          "It uses Fisher Information to measure importance.",
        "ar_explanation": "تقنية تحدد اي المعاملات (الاوزان) كانت مهمة "
                          "للمهام السابقة وتعاقب التغييرات عليها اثناء التدريب الجديد. "
                          "تستخدم معلومات فيشر لقياس الاهمية.",
        "en_analogy": "Like putting locks on important drawers: "
                      "you can still organize the room (learn new tasks), "
                      "but the locked drawers (critical knowledge) stay safe.",
        "ar_analogy": "مثل وضع اقفال على الادراج المهمة: "
                      "يمكنك ترتيب الغرفة (تعلم مهام جديدة)، "
                      "لكن الادراج المقفلة (المعرفة الحرجة) تبقى آمنة.",
        "category": "continual_learning",
    },
    "lora": {
        "en_title": "LoRA (Low-Rank Adaptation)",
        "ar_title": "LoRA (التكيف منخفض الرتبة)",
        "en_explanation": "A technique that freezes most of the model and only trains "
                          "two small matrices that are added to the original weights. "
                          "This drastically reduces training time and memory.",
        "ar_explanation": "تقنية تجمّد معظم النموذج وتدرّب فقط "
                          "مصفوفتين صغيرتين تُضافان الى الاوزان الاصلية. "
                          "هذا يقلل وقت التدريب والذاكرة بشكل كبير.",
        "en_analogy": "Instead of repainting an entire building (full training), "
                      "you only repaint a few accent walls (LoRA layers).",
        "ar_analogy": "بدلا من اعادة طلاء مبنى كامل (تدريب كامل)، "
                      "تطلي فقط بعض الجدران المميزة (طبقات LoRA).",
        "category": "peft",
    },
    "lora_multitask": {
        "en_title": "Multi-Task LoRA (dytr LoRAWrapper)",
        "ar_title": "LoRA متعدد المهام (أغلفة dytr LoRA)",
        "en_explanation": "Parameter-Efficient Fine-Tuning using lightweight low-rank decomposition matrices "
                          "(A and B) per task while keeping the core transformer backbone frozen. "
                          "Allows rapid task switching and cuts trainable parameters by over 90%.",
        "ar_explanation": "ضبط دقيق عالي الكفاءة للمعاملات باستخدام مصفوفات منخفضة الرتبة "
                          "(A و B) لكل مهمة مع تجميد العمود الفقري للنموذج. "
                          "يتيح التبديل السريع بين المهام ويخفض المعاملات القابلة للتدريب بأكثر من 90%.",
        "en_analogy": "Instead of building a separate computer for each task, "
                      "you plug in small specialized USB cartridges into the same machine.",
        "ar_analogy": "بدلاً من شراء حاسوب مستقل لكل مهمة، "
                      "تستخدم نفس الحاسوب وتوصل به شرائح USB خفيفة متخصصة لكل مهمة.",
        "category": "peft",
    },
    "rope": {
        "en_title": "Rotary Position Embedding (RoPE)",
        "ar_title": "التضمين الموضعي الدوراني (RoPE)",
        "en_explanation": "A method for encoding the position of each token in a sequence "
                          "by rotating its embedding vector. Nearby tokens get similar rotations, "
                          "so the model learns that position 5 and position 6 are close.",
        "ar_explanation": "طريقة لتشفير موقع كل توكن في التسلسل "
                          "عن طريق تدوير متجه التضمين الخاص به. التوكنات المتجاورة "
                          "تحصل على دورانات متشابهة، فيتعلم النموذج ان الموقع 5 و6 قريبان.",
        "en_analogy": "Like the hours on a clock: "
                      "3 o'clock and 4 o'clock are close, 3 and 9 are far apart.",
        "ar_analogy": "مثل الساعات على الساعة: "
                      "الساعة 3 والساعة 4 قريبتان، الساعة 3 و9 بعيدتان.",
        "category": "transformer",
    },
    "replay_buffer": {
        "en_title": "Experience Replay Buffer (dytr Memory)",
        "ar_title": "ذاكرة استرجاع التجارب (ReplayBuffer)",
        "en_explanation": "A continual learning memory mechanism that stores representative exemplars "
                          "from previous tasks. During new task training, rehearsal samples are mixed into "
                          "batches to prevent catastrophic forgetting and maintain past accuracy.",
        "ar_explanation": "آلية ذاكرة للتعلم المستمر تخزن عينات تمثيلية نموذجية "
                          "من المهام السابقة. أثناء تدريب المهام الجديدة، يتم دمج هذه العينات في الدفعات "
                          "لمنع النسيان الكارثي والحفاظ على دقة المهام السابقة.",
        "en_analogy": "Like an experienced doctor reviewing classic patient cases "
                      "from previous years while studying a new medical specialization.",
        "ar_analogy": "مثل طبيب يراجع دورياً ملفات مرضى سابقة "
                      "أثناء دراسته لتخصص طبي جديد لإنعاش ذاكرته الطبية.",
        "category": "continual_learning",
    },
    "loss": {
        "en_title": "Loss Function",
        "ar_title": "دالة الخسارة (الخطأ)",
        "en_explanation": "A mathematical formula that measures how far the model's predictions "
                          "are from the correct answers. The goal of training is to minimize this number. "
                          "Lower loss = better model.",
        "ar_explanation": "صيغة رياضية تقيس مدى بُعد تنبؤات النموذج "
                          "عن الاجابات الصحيحة. هدف التدريب هو تقليل هذا الرقم. "
                          "خسارة اقل = نموذج افضل.",
        "en_analogy": "Like a student's error rate on exams: "
                      "at first many wrong answers, gradually fewer.",
        "ar_analogy": "مثل معدل اخطاء الطالب في الامتحانات: "
                      "في البداية اخطاء كثيرة، تتناقص تدريجيا.",
        "category": "training",
    },
    "accuracy": {
        "en_title": "Accuracy",
        "ar_title": "الدقة",
        "en_explanation": "The percentage of correct predictions out of all predictions. "
                          "If the model makes 100 predictions and 87 are correct, accuracy is 87%.",
        "ar_explanation": "نسبة التنبؤات الصحيحة من اجمالي التنبؤات. "
                          "اذا تنبأ النموذج 100 مرة و87 كانت صحيحة، الدقة 87%.",
        "en_analogy": "Like a basketball player's shooting percentage: "
                      "87% means 87 successful shots out of 100 attempts.",
        "ar_analogy": "مثل نسبة تسديد لاعب كرة السلة: "
                      "87% تعني 87 تسديدة ناجحة من 100 محاولة.",
        "category": "evaluation",
    },
    "bwt": {
        "en_title": "Backward Transfer (BWT)",
        "ar_title": "النقل العكسي (BWT)",
        "en_explanation": "Measures how much the model forgot about previous tasks after learning new ones. "
                          "BWT = -2.3% means the model retained 97.7% of its previous knowledge.",
        "ar_explanation": "يقيس مقدار ما نسيه النموذج من المهام السابقة بعد تعلم مهام جديدة. "
                          "BWT = -2.3% يعني ان النموذج احتفظ بـ 97.7% من معرفته السابقة.",
        "en_analogy": "If a student scored 95 in math last month and 93 now "
                      "after learning physics, BWT is -2.",
        "ar_analogy": "اذا حصل الطالب على 95 في الرياضيات الشهر الماضي و93 الان "
                      "بعد تعلم الفيزياء، فإن BWT يساوي -2.",
        "category": "evaluation",
    },
    "epoch": {
        "en_title": "Epoch",
        "ar_title": "الحقبة",
        "en_explanation": "One complete pass through the entire training dataset. "
                          "If you have 5000 examples and train for 3 epochs, "
                          "the model sees each example 3 times.",
        "ar_explanation": "مرور كامل واحد على كل بيانات التدريب. "
                          "اذا كان لديك 5000 مثال وتدربت لـ 3 حقب، "
                          "فالنموذج يرى كل مثال 3 مرات.",
        "en_analogy": "Like re-reading a textbook: "
                      "each read-through is one epoch.",
        "ar_analogy": "مثل اعادة قراءة كتاب دراسي: "
                      "كل قراءة كاملة هي حقبة واحدة.",
        "category": "training",
    },
    "batch": {
        "en_title": "Batch",
        "ar_title": "الدفعة",
        "en_explanation": "A small group of training examples processed together at once. "
                          "Instead of learning from one example at a time, "
                          "the model learns from 16 or 32 examples simultaneously.",
        "ar_explanation": "مجموعة صغيرة من امثلة التدريب تُعالج معا دفعة واحدة. "
                          "بدلا من التعلم من مثال واحد في كل مرة، "
                          "يتعلم النموذج من 16 او 32 مثال بشكل متزامن.",
        "en_analogy": "Like a teacher grading papers: "
                      "instead of one paper at a time, they grade a stack of 16.",
        "ar_analogy": "مثل معلم يصحح اوراق: "
                      "بدلا من ورقة واحدة كل مرة، يصحح حزمة من 16.",
        "category": "training",
    },
    "learning_rate": {
        "en_title": "Learning Rate",
        "ar_title": "معدل التعلم",
        "en_explanation": "Controls how much the model adjusts its weights after each batch. "
                          "Too high: the model overshoots and never converges. "
                          "Too low: the model learns too slowly.",
        "ar_explanation": "يتحكم في مقدار تعديل النموذج لاوزانه بعد كل دفعة. "
                          "اذا كان عاليا جدا: النموذج يتجاوز الهدف ولا يستقر. "
                          "اذا كان منخفضا جدا: النموذج يتعلم ببطء شديد.",
        "en_analogy": "Like the step size when walking to a destination: "
                      "big steps get you there faster but you might overshoot, "
                      "small steps are precise but slow.",
        "ar_analogy": "مثل حجم الخطوة عند المشي نحو هدف: "
                      "خطوات كبيرة توصلك اسرع لكن قد تتجاوز الهدف، "
                      "خطوات صغيرة دقيقة لكن بطيئة.",
        "category": "training",
    },
    "overfitting": {
        "en_title": "Overfitting",
        "ar_title": "فرط التخصيص",
        "en_explanation": "When the model memorizes training data instead of learning general patterns. "
                          "It performs well on training data but poorly on new, unseen data.",
        "ar_explanation": "عندما يحفظ النموذج بيانات التدريب بدلا من تعلم الانماط العامة. "
                          "يؤدي بشكل جيد على بيانات التدريب لكن بشكل سيئ على بيانات جديدة.",
        "en_analogy": "Like a student who memorizes exam answers without understanding: "
                      "they score 100% on practice tests but fail the real exam.",
        "ar_analogy": "مثل طالب يحفظ اجابات الامتحان بدون فهم: "
                      "يحصل على 100% في الامتحانات التدريبية لكن يرسب في الحقيقي.",
        "category": "training",
    },
    "backpropagation": {
        "en_title": "Backpropagation",
        "ar_title": "الانتشار العكسي",
        "en_explanation": "The algorithm used to calculate how much each weight contributed to the error. "
                          "It works backwards from the output to the input, "
                          "distributing blame for the error across all weights.",
        "ar_explanation": "الخوارزمية المستخدمة لحساب مقدار مساهمة كل وزن في الخطأ. "
                          "تعمل بشكل عكسي من المخرج الى المدخل، "
                          "وتوزع مسؤولية الخطأ على كل الاوزان.",
        "en_analogy": "Like tracing a mistake in a relay race: "
                      "you go back through each runner to find who dropped the baton.",
        "ar_analogy": "مثل تتبع خطأ في سباق تتابع: "
                      "ترجع لكل عدّاء لتجد من اسقط العصا.",
        "category": "training",
    },
}


@router.get("/concept/{name}")
async def get_concept_explanation(
    name: str,
    lang: str = Query("ar", description="Language: ar (Arabic) or en (English)"),
):
    """Return a plain-language explanation of an academic concept.

    Each concept includes: title, explanation, analogy, and category.
    If the concept is not found, returns available concepts.
    """
    concept = CONCEPT_LIBRARY.get(name)
    if not concept:
        return {
            "status": "not_found",
            "message_ar": f"المفهوم '{name}' غير موجود في القاموس.",
            "message_en": f"Concept '{name}' not found in the dictionary.",
            "available_concepts": list(CONCEPT_LIBRARY.keys()),
        }

    prefix = "ar" if lang == "ar" else "en"
    return {
        "status": "found",
        "name": name,
        "title": concept[f"{prefix}_title"],
        "explanation": concept[f"{prefix}_explanation"],
        "analogy": concept[f"{prefix}_analogy"],
        "category": concept["category"],
    }


@router.get("/concepts")
async def list_all_concepts(
    lang: str = Query("ar", description="Language: ar or en"),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """Return all available concepts, optionally filtered by category."""
    prefix = "ar" if lang == "ar" else "en"
    results = []
    for key, concept in CONCEPT_LIBRARY.items():
        if category and concept["category"] != category:
            continue
        results.append({
            "name": key,
            "title": concept[f"{prefix}_title"],
            "explanation": concept[f"{prefix}_explanation"],
            "category": concept["category"],
        })
    return {
        "total": len(results),
        "concepts": results,
        "categories": sorted(set(c["category"] for c in CONCEPT_LIBRARY.values())),
    }


# Metric translation: converts raw numbers to human-readable sentences
METRIC_TEMPLATES = {
    "loss": {
        "en": lambda v: (
            f"Error: {v:.3f} -- "
            + ("The model is learning well, error is decreasing." if v < 0.5
               else "The model is still adjusting, error is moderate." if v < 1.5
               else "The model has just started learning, error is high.")
        ),
        "ar": lambda v: (
            f"الخطأ: {v:.3f} -- "
            + ("النموذج يتعلم بشكل جيد، الخطأ يتناقص." if v < 0.5
               else "النموذج لا يزال يتعلم، الخطأ متوسط." if v < 1.5
               else "النموذج بدأ التعلم للتو، الخطأ مرتفع.")
        ),
    },
    "accuracy": {
        "en": lambda v: (
            f"Accuracy: {v:.1f}% -- "
            f"Out of every 100 predictions, {int(v)} are correct."
        ),
        "ar": lambda v: (
            f"الدقة: {v:.1f}% -- "
            f"من كل 100 تنبؤ، {int(v)} صحيحة."
        ),
    },
    "bwt": {
        "en": lambda v: (
            f"Backward Transfer: {v:.1f}% -- "
            + (f"The model retained {100 + v:.1f}% of previous knowledge." if v < 0
               else f"The model improved on previous tasks by {v:.1f}%.")
        ),
        "ar": lambda v: (
            f"النقل العكسي: {v:.1f}% -- "
            + (f"النموذج احتفظ بـ {100 + v:.1f}% من معرفته السابقة." if v < 0
               else f"النموذج تحسن في المهام السابقة بنسبة {v:.1f}%.")
        ),
    },
    "vram_mb": {
        "en": lambda v: f"GPU Memory: {v} MB used.",
        "ar": lambda v: f"ذاكرة المعالج الرسومي: {v} ميغابايت مستخدمة.",
    },
    "tokens_per_sec": {
        "en": lambda v: f"Processing Speed: {v:,} tokens per second.",
        "ar": lambda v: f"سرعة المعالجة: {v:,} توكن في الثانية.",
    },
}


class MetricTranslateRequest(BaseModel):
    metric_name: str = Field(..., example="loss")
    value: float = Field(..., example=0.34)
    lang: str = Field("ar", example="ar")


@router.post("/metric")
async def translate_metric(req: MetricTranslateRequest):
    """Translate a raw metric value into a human-readable sentence."""
    template = METRIC_TEMPLATES.get(req.metric_name)
    if not template:
        return {
            "status": "unknown_metric",
            "raw_value": req.value,
            "available_metrics": list(METRIC_TEMPLATES.keys()),
        }

    lang = "ar" if req.lang == "ar" else "en"
    return {
        "status": "translated",
        "metric_name": req.metric_name,
        "raw_value": req.value,
        "human_text": template[lang](req.value),
    }


class SuggestRequest(BaseModel):
    current_page: str = Field("academy", example="academy")
    model_configured: bool = Field(False)
    dataset_selected: bool = Field(False)
    training_complete: bool = Field(False)
    lang: str = Field("ar")


@router.post("/suggest")
async def suggest_next_step(req: SuggestRequest):
    """Suggest the next logical step based on the user's current progress."""
    if req.lang == "ar":
        if not req.dataset_selected and req.current_page == "academy":
            return {
                "next_page": "dataprep",
                "message": "الخطوة التالية: اختر او ارفع مجموعة بيانات لتبدأ التدريب.",
                "readiness": "partial",
            }
        if req.dataset_selected and not req.model_configured:
            return {
                "next_page": "builder",
                "message": "الخطوة التالية: اعدّ معمارية النموذج الذي تريد تدريبه.",
                "readiness": "partial",
            }
        if req.model_configured and not req.training_complete:
            return {
                "next_page": "arena",
                "message": "كل شيء جاهز. يمكنك بدء التدريب الان.",
                "readiness": "ready",
            }
        if req.training_complete:
            return {
                "next_page": "playground",
                "message": "النموذج مدرّب. جرّبه الان في ملعب التنبؤ.",
                "readiness": "complete",
            }
    else:
        if not req.dataset_selected and req.current_page == "academy":
            return {
                "next_page": "dataprep",
                "message": "Next step: Select or upload a dataset to start training.",
                "readiness": "partial",
            }
        if req.dataset_selected and not req.model_configured:
            return {
                "next_page": "builder",
                "message": "Next step: Configure the model architecture you want to train.",
                "readiness": "partial",
            }
        if req.model_configured and not req.training_complete:
            return {
                "next_page": "arena",
                "message": "Everything is ready. You can start training now.",
                "readiness": "ready",
            }
        if req.training_complete:
            return {
                "next_page": "playground",
                "message": "Model is trained. Try it now in the Prediction Playground.",
                "readiness": "complete",
            }

    return {
        "next_page": "academy",
        "message": "ابدأ من الاكاديمية لتتعلم الاساسيات." if req.lang == "ar"
                   else "Start from the Academy to learn the basics.",
        "readiness": "initial",
    }


# Learning path: ordered sequence of concepts for systematic learning
LEARNING_PATHS = {
    "beginner": [
        "machine", "artificial_intelligence", "model", "token",
        "embedding", "attention", "transformer",
        "loss", "accuracy", "epoch", "batch", "learning_rate",
        "overfitting", "backpropagation",
    ],
    "continual_learning": [
        "multi_task_learning", "catastrophic_forgetting", "ewc", "bwt",
    ],
    "optimization": [
        "lora", "adalora", "rope", "flash_attention",
    ],
}


@router.get("/path/{track}")
async def get_learning_path(
    track: str,
    lang: str = Query("ar", description="Language: ar or en"),
):
    """Return an ordered learning path for a specific track."""
    concept_names = LEARNING_PATHS.get(track)
    if not concept_names:
        return {
            "status": "not_found",
            "available_tracks": list(LEARNING_PATHS.keys()),
        }

    prefix = "ar" if lang == "ar" else "en"
    ordered = []
    for i, name in enumerate(concept_names, 1):
        concept = CONCEPT_LIBRARY.get(name, {})
        ordered.append({
            "order": i,
            "name": name,
            "title": concept.get(f"{prefix}_title", name),
            "category": concept.get("category", ""),
        })

    return {
        "track": track,
        "total_concepts": len(ordered),
        "concepts": ordered,
    }
