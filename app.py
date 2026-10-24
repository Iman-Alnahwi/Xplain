"""
Xplain — Privacy-First Multi-File & Action Agent
=================================================

One input box. The agent decides what to do from what the user sends:

1. Code Tutor   — pasted code (or a code file) is explained line by line,
                  with an interactive, expandable execution tree (variable
                  values + output at every step), a flowchart and a quiz.
                  Images are generated ONLY when the student asks for them.
2. File Actions — a screenshot / PDF is simplified and turned into one-click
                  actions (a bill becomes "amount + due date" with a button to
                  the payment page, a calendar reminder and CSV/Excel export).
3. Follow-ups   — any other text is answered as a question, about the last
                  code / file when there is one.

Models (open-weight):
  * Gemma 4 via the Gemini API (default)  — text + images
  * Any open-weight model on Groq         — OpenAI-compatible API
  * FLUX.1-schnell (Apache-2.0) via Hugging Face — on-demand images
  * Demo mode — works offline with sample answers, so the app never crashes.

Run:  streamlit run app.py
"""

import base64
import datetime as dt
import html
import io
import json
import os
import re
import urllib.parse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    from pypdf import PdfReader
except ImportError:  # PDF support is optional
    PdfReader = None


# ---------------------------------------------------------------------------
# Constants & palette (Pantone)
# ---------------------------------------------------------------------------

NAVY = "#00263E"         # Pantone 2965 C — primary: main actions, headings
POWDER = "#94B2C4"       # Pantone 550 C Powder Blue — secondary: highlights
COCOA = "#594536"        # Pantone 19-1119 Cocoa — accent: outputs, amounts
MILK = "#F0EDE5"         # Pantone 11-0608 Coconut Milk — background
NAVY_SOFT = "#DCE6EC"
POWDER_SOFT = "#E6EEF2"
COCOA_SOFT = "#EAE2D9"
CARD = "#FBFAF7"
MUTED = "#5E6B75"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FLUX_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_TEXT = "gemma4:e4b"     # Google Gemma 4 (open weights) — real Arabic/Turkish, explains code
DEFAULT_OLLAMA_VISION = "gemma4:e4b"   # the same model also reads photos of bills and documents
# "llama3.2" also works for English-only text (type it in Settings)
DEFAULT_GEMMA_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

APP_NAME = "Xplain"  # a neutral, letters-only name that reads the same in every language

CODE_EXT = {"py", "js", "ts", "c", "h", "cpp", "hpp", "cc", "java", "cs", "go", "rs", "php", "rb", "kt",
            "swift", "sql", "html", "css", "sh", "json", "dart", "r", "m", "lua", "scala"}
FILE_TYPES = ["png", "jpg", "jpeg", "pdf", "txt"] + sorted(CODE_EXT)
IMAGE_EXT = {"png", "jpg", "jpeg"}
FINANCE_WORDS = [
    "invoice", "bill", "amount due", "total due", "due date", "payment", "receipt",
    "فاتورة", "المبلغ", "المستحق", "تاريخ الاستحقاق", "دفع",
    "fatura", "tutar", "son ödeme", "ödeme",
]

LANG_LABELS = {"ar": "العربية", "en": "English", "tr": "Türkçe"}
LANG_NAMES = {
    "ar": "Modern Standard Arabic (الفصحى) only — never use any colloquial dialect",
    "en": "English",
    "tr": "Turkish",
}

SAMPLE_CODE = """for i in range(3):
    for j in range(2):
        print(i, j)
print("done")"""


# ---------------------------------------------------------------------------
# UI translations — Arabic (Modern Standard Arabic) first, then English, Turkish
# ---------------------------------------------------------------------------

T = {
    "ar": {
        "title": APP_NAME,
        "logo": "X",
        "subtitle": "وكيل ذكاء اصطناعي مفتوح المصدر يشرح الشيفرة البرمجية سطراً سطراً، ويحوّل ملفاتك إلى إجراءات.",
        "settings": "⚙️ الإعدادات",
        "provider": "النموذج",
        "provider_gemma": "Gemma 4 (Gemini API)",
        "provider_groq": "نموذج مفتوح على Groq",
        "provider_demo": "الوضع التجريبي (دون مفتاح)",
        "provider_ollama": "Gemma 4 على جهازك (Ollama، دون مفتاح)",
        "ollama_down": "⚠️ برنامج Ollama لا يعمل على هذا الجهاز، لذلك تُعرض أمثلة تجريبية. شغّله ثم أعد المحاولة.",
        "vision_model": "نموذج قراءة الصور",
        "local_notice": "💻 يعمل نموذج Gemma 4 على جهازك: لا حاجة إلى مفتاح، ولا تغادر شيفرتك وملفاتك هذا الحاسوب.",
        "api_key": "مفتاح API",
        "model": "اسم النموذج",
        "hf_token": "مفتاح Hugging Face (لإنشاء الصور)",
        "level": "مستوى الشرح",
        "level_beginner": "مبتدئ",
        "level_uni": "طالب جامعي",
        "privacy": "🔒 تُعالَج الملفات في الذاكرة فقط ولا تُحفَظ، ولا تُرسَل إلا إلى النموذج الذي تختاره.",
        "demo_notice": "🧪 الوضع التجريبي: النتائج المعروضة أمثلة توضيحية. اختر Gemma 4 على جهازك من الإعدادات للحصول على شرح حقيقي.",
        "input_placeholder": "الصق الشيفرة البرمجية أو اكتب سؤالك، أو أرفق صورة أو ملفاً…",
        "welcome_title": "ماذا تريد أن تفهم اليوم؟",
        "welcome_1": "🧑‍🏫 الصق شيفرة بأي لغة برمجة، فأخبرك بلغتها وأشرحها لك سطراً سطراً، وأبيّن كيف نحصل على المُخرَج.",
        "welcome_2": "🧾 أرفق صورة فاتورة، فأستخرج لك المبلغ المطلوب وآخر موعد للدفع.",
        "welcome_3": "📄 أرفق مستنداً، فألخّصه لك وأقترح الإجراء المناسب.",
        "example": "جرّب مثالاً: حلقة متداخلة",
        "you_sent": "أرسلت",
        "role_tutor": "🧑‍🏫 المعلّم: شرح الشيفرة",
        "role_file": "📂 تبسيط الملف وتنفيذ الإجراءات",
        "role_answer": "💬 إجابة عن سؤالك",
        "thinking": "يقرأ المعلّم الشيفرة…",
        "analyzing": "تجري قراءة الملف…",
        "answering": "تجري كتابة الإجابة…",
        "summary": "الفكرة العامة",
        "line_by_line": "الشرح سطراً سطراً",
        "step_of": "السطر {i} من {n}",
        "prev": "→ السابق",
        "next": "التالي ←",
        "all_lines": "عرض الأسطر كلها",
        "concept": "المفهوم",
        "trace": "مخطط التنفيذ التفاعلي",
        "trace_hint": "اضغط على أي قسم لتوسيعه، أو تتبّع تنفيذ البرنامج خطوة خطوة.",
        "t_step": "الخطوة التالية",
        "t_back": "السابقة",
        "t_expand": "توسيع الكل",
        "t_collapse": "طيّ الكل",
        "t_line": "السطر",
        "t_output": "المُخرَج",
        "final_output": "المُخرَج النهائي",
        "flowchart": "المخطط الانسيابي",
        "quiz": "اختبر نفسك",
        "correct": "✅ إجابة صحيحة!",
        "wrong": "❌ إجابة غير صحيحة.",
        "learn_more": "للاستزادة",
        "docs": "التوثيق الرسمي",
        "so": "ابحث في StackOverflow",
        "yes": "نعم",
        "no": "لا",
        "img_btn": "🎨 أنشئ صورة توضيحية",
        "img_what": "أيّ جزء تريد أن توضّحه الصورة؟",
        "img_custom": "✍️ وصف من عندي",
        "img_custom_label": "صِف الصورة التي تريدها بدقة",
        "img_go": "أنشئ الصورة",
        "img_wait": "يجري إنشاء الصورة…",
        "img_need_token": "أضف مفتاح Hugging Face في الإعدادات لإنشاء الصور (النموذج FLUX.1-schnell مجاني ومفتوح المصدر).",
        "img_failed": "تعذّر إنشاء الصورة",
        "preview": "معاينة",
        "detected": "النوع المكتشَف",
        "ctx_code": "شيفرة برمجية",
        "ctx_invoice": "فاتورة",
        "ctx_document": "مستند",
        "ctx_image": "صورة",
        "amount_due": "المبلغ المطلوب",
        "due_date": "آخر موعد للدفع",
        "biller": "الجهة المُصدِرة",
        "reference": "رقم الحساب أو المرجع",
        "pay": "💳 انتقل إلى صفحة الدفع",
        "pay_hint": "سيُفتح الموقع: {domain}. تحقّق من الرابط قبل إدخال أي بيانات دفع.",
        "pay_search_hint": "لا يتضمّن الملف رابطاً للدفع، لذلك سنبحث عن صفحة الدفع الرسمية للجهة المُصدِرة.",
        "reminder": "⏰ أضف تذكيراً بموعد الدفع",
        "csv": "⬇️ CSV",
        "xlsx": "⬇️ Excel",
        "items": "التفاصيل",
        "doc_summary": "الملخّص في ثلاث نقاط",
        "gmail": "✉️ أنشئ مسودة رسالة في Gmail",
        "actions": "الإجراءات",
        "followup_hint": "يمكنك أن تسأل عن هذه النتيجة في مربع الإدخال، أو أن ترسل شيفرة أو ملفاً جديداً.",
        "error_api": "تعذّر الاتصال بالنموذج، لذلك عُرض مثال تجريبي بدلاً من ذلك.",
        "pdf_missing": "مكتبة pypdf غير مثبّتة، لذلك تتعذّر قراءة نص ملف PDF.",
        "pdf_pages": "عدد الصفحات: {n}",
        "writing": "✍️ يكتب النموذج الإجابة على جهازك… ({n} كلمة حتى الآن)",
        "choose_action": "ماذا تريد أن تفعل؟",
        "opt_explain": "📖 اشرح لي الملف",
        "opt_pay": "💳 ادفع الفاتورة",
        "opt_export": "⬇️ صدّر البيانات",
        "pick_option": "اختر خياراً: أشرح لك الملف، أو أنفّذ الإجراء مباشرة.",
        "pay_opened": "فُتحت منصة الدفع في نافذة جديدة. إن منعها المتصفح، فاضغط الزر.",
        "ctx_task": "طلب إجراء",
        "action_title": "⚡ الإجراء المطلوب",
        "act_email": "إرسال بريد إلكتروني إلى {to}",
        "act_pay": "دفع الفاتورة",
        "act_link": "فتح الرابط: {url}",
        "act_call": "الاتصال بالرقم {phone}",
        "act_calendar": "إضافة الموعد {date} إلى التقويم",
        "open_gmail": "✉️ افتح Gmail",
        "open_link": "🔗 افتح الرابط",
        "call_btn": "📞 اتصل الآن",
        "auto_opened": "فُتحت الصفحة تلقائياً في نافذة جديدة. إن منعها المتصفح، فاضغط الزر.",
        "language": "اللغة",
        "status": "الحالة",
        "lang_badge": "لغة البرمجة",
        "written_in": "هذه الشيفرة مكتوبة بلغة {lang}.",
        "how_output": "كيف نحصل على هذا المُخرَج؟",
        "glossary": "مصطلحات للمبتدئين",
        "history": "🕘 السجل",
        "new": "＋ طلب جديد",
        "history_empty": "لا توجد طلبات سابقة بعد.",
        "diagram_added": "أُضيف المخطط كما طلبت.",
        "image_added": "أُنشئت الصورة كما طلبت.",
    },
    "en": {
        "title": APP_NAME,
        "logo": "X",
        "subtitle": "Open-source AI agent that explains code line by line and turns your files into actions.",
        "settings": "⚙️ Settings",
        "provider": "Model",
        "provider_gemma": "Gemma 4 (Gemini API)",
        "provider_groq": "Open-weight model on Groq",
        "provider_demo": "Demo mode (no key)",
        "provider_ollama": "Gemma 4 on this computer (Ollama, no key)",
        "ollama_down": "⚠️ Ollama is not running on this computer, so demo answers are shown. Start it and try again.",
        "vision_model": "Image-reading model",
        "local_notice": "💻 Gemma 4 is running on your computer: no key needed, and your code and files never leave this machine.",
        "api_key": "API key",
        "model": "Model name",
        "hf_token": "Hugging Face token (for images)",
        "level": "Explanation level",
        "level_beginner": "Beginner",
        "level_uni": "University student",
        "privacy": "🔒 Files are processed in memory only, never stored, and sent only to the model you choose.",
        "demo_notice": "🧪 Demo mode: results are samples. Choose Gemma 4 on this computer in Settings for real explanations.",
        "input_placeholder": "Paste code or ask a question, or attach an image or file…",
        "welcome_title": "What do you want to understand today?",
        "welcome_1": "🧑‍🏫 Paste code in any programming language: I'll name the language, explain it line by line and show how the output is produced.",
        "welcome_2": "🧾 Attach a photo of a bill and I'll pull out the amount and the due date.",
        "welcome_3": "📄 Attach a document and I'll summarize it and suggest the next action.",
        "example": "Try an example: nested loop",
        "you_sent": "You sent",
        "role_tutor": "🧑‍🏫 Tutor: explaining code",
        "role_file": "📂 Simplifying the file and suggesting actions",
        "role_answer": "💬 Answering your question",
        "thinking": "The tutor is reading your code…",
        "analyzing": "Reading your file…",
        "answering": "Writing the answer…",
        "summary": "Big picture",
        "line_by_line": "Line by line",
        "step_of": "Line {i} of {n}",
        "prev": "← Previous",
        "next": "Next →",
        "all_lines": "Show all lines",
        "concept": "Concept",
        "trace": "Interactive execution tree",
        "trace_hint": "Click any part to expand it, or walk through the program step by step.",
        "t_step": "Next step",
        "t_back": "Back",
        "t_expand": "Expand all",
        "t_collapse": "Collapse all",
        "t_line": "line",
        "t_output": "Output",
        "final_output": "Final output",
        "flowchart": "Flowchart",
        "quiz": "Test yourself",
        "correct": "✅ Correct!",
        "wrong": "❌ Not quite.",
        "learn_more": "Learn more",
        "docs": "Official docs",
        "so": "Search StackOverflow",
        "yes": "yes",
        "no": "no",
        "img_btn": "🎨 Generate an image",
        "img_what": "Which part should the image explain?",
        "img_custom": "✍️ My own description",
        "img_custom_label": "Describe exactly the image you want",
        "img_go": "Generate image",
        "img_wait": "Drawing your image…",
        "img_need_token": "Add a Hugging Face token in Settings to generate images (FLUX.1-schnell is free and open).",
        "img_failed": "Could not generate the image",
        "preview": "Preview",
        "detected": "Detected type",
        "ctx_code": "Code",
        "ctx_invoice": "Invoice / bill",
        "ctx_document": "Document",
        "ctx_image": "Image",
        "amount_due": "Amount to pay",
        "due_date": "Pay before",
        "biller": "Issued by",
        "reference": "Account / reference",
        "pay": "💳 Go to payment page",
        "pay_hint": "Opens: {domain}. Check the link before entering any payment details.",
        "pay_search_hint": "No payment link was printed on the file, so we'll search for the biller's official payment page.",
        "reminder": "⏰ Add a due-date reminder",
        "csv": "⬇️ CSV",
        "xlsx": "⬇️ Excel",
        "items": "Details",
        "doc_summary": "3-point summary",
        "gmail": "✉️ Create Gmail draft",
        "actions": "Actions",
        "followup_hint": "Ask about this result in the input box, or send new code or a new file.",
        "error_api": "The model call failed, so a demo answer is shown instead.",
        "pdf_missing": "pypdf is not installed, so PDF text cannot be read.",
        "pdf_pages": "Pages: {n}",
        "writing": "✍️ The model is writing on your computer… ({n} words so far)",
        "choose_action": "What would you like to do?",
        "opt_explain": "📖 Explain it",
        "opt_pay": "💳 Pay the bill",
        "opt_export": "⬇️ Export data",
        "pick_option": "Pick an option: I can explain the file or take the action for you.",
        "pay_opened": "The payment platform was opened in a new tab. If your browser blocked it, press the button.",
        "ctx_task": "Action request",
        "action_title": "⚡ Requested action",
        "act_email": "Send an email to {to}",
        "act_pay": "Pay the bill",
        "act_link": "Open the link: {url}",
        "act_call": "Call {phone}",
        "act_calendar": "Add {date} to your calendar",
        "open_gmail": "✉️ Open Gmail",
        "open_link": "🔗 Open link",
        "call_btn": "📞 Call now",
        "auto_opened": "It was opened automatically in a new tab. If your browser blocked it, press the button.",
        "language": "Language",
        "status": "Status",
        "lang_badge": "Language",
        "written_in": "This code is written in {lang}.",
        "how_output": "How do we get this output?",
        "glossary": "Beginner glossary",
        "history": "🕘 History",
        "new": "＋ New request",
        "history_empty": "No previous requests yet.",
        "diagram_added": "The diagram was added as you asked.",
        "image_added": "The image was created as you asked.",
    },
    "tr": {
        "title": APP_NAME,
        "logo": "X",
        "subtitle": "Kodu satır satır açıklayan ve dosyalarını eylemlere dönüştüren açık kaynak yapay zekâ ajanı.",
        "settings": "⚙️ Ayarlar",
        "provider": "Model",
        "provider_gemma": "Gemma 4 (Gemini API)",
        "provider_groq": "Groq üzerinde açık model",
        "provider_demo": "Demo modu (anahtarsız)",
        "provider_ollama": "Bu bilgisayarda Gemma 4 (Ollama, anahtarsız)",
        "ollama_down": "⚠️ Ollama bu bilgisayarda çalışmıyor, bu yüzden demo yanıtları gösteriliyor. Başlatıp tekrar dene.",
        "vision_model": "Görsel okuma modeli",
        "local_notice": "💻 Gemma 4 bilgisayarında çalışıyor: anahtar gerekmez, kodun ve dosyaların bu makineden çıkmaz.",
        "api_key": "API anahtarı",
        "model": "Model adı",
        "hf_token": "Hugging Face anahtarı (görseller için)",
        "level": "Açıklama seviyesi",
        "level_beginner": "Yeni başlayan",
        "level_uni": "Üniversite öğrencisi",
        "privacy": "🔒 Dosyalar yalnızca bellekte işlenir, saklanmaz ve sadece seçtiğin modele gönderilir.",
        "demo_notice": "🧪 Demo modu: sonuçlar örnektir. Gerçek açıklamalar için Ayarlar'dan bu bilgisayardaki Gemma 4'ü seç.",
        "input_placeholder": "Kod yapıştır veya soru sor, ya da görsel veya dosya ekle…",
        "welcome_title": "Bugün neyi anlamak istiyorsun?",
        "welcome_1": "🧑‍🏫 Herhangi bir dilde kod yapıştır: dilini söyleyip satır satır açıklayayım ve çıktının nasıl oluştuğunu göstereyim.",
        "welcome_2": "🧾 Bir fatura fotoğrafı ekle, ödenecek tutarı ve son tarihi çıkarayım.",
        "welcome_3": "📄 Bir belge ekle, özetleyip uygun eylemi önereyim.",
        "example": "Örnek dene: iç içe döngü",
        "you_sent": "Gönderdiğin",
        "role_tutor": "🧑‍🏫 Öğretmen: kod açıklaması",
        "role_file": "📂 Dosyayı sadeleştirme ve eylemler",
        "role_answer": "💬 Sorunun yanıtı",
        "thinking": "Öğretmen kodunu okuyor…",
        "analyzing": "Dosya okunuyor…",
        "answering": "Yanıt yazılıyor…",
        "summary": "Genel fikir",
        "line_by_line": "Satır satır",
        "step_of": "Satır {i} / {n}",
        "prev": "← Önceki",
        "next": "Sonraki →",
        "all_lines": "Tüm satırları göster",
        "concept": "Kavram",
        "trace": "Etkileşimli çalışma ağacı",
        "trace_hint": "Genişletmek için bir bölüme tıkla veya programı adım adım izle.",
        "t_step": "Sonraki adım",
        "t_back": "Geri",
        "t_expand": "Tümünü aç",
        "t_collapse": "Tümünü kapat",
        "t_line": "satır",
        "t_output": "Çıktı",
        "final_output": "Son çıktı",
        "flowchart": "Akış şeması",
        "quiz": "Kendini test et",
        "correct": "✅ Doğru!",
        "wrong": "❌ Tam değil.",
        "learn_more": "Daha fazla öğren",
        "docs": "Resmi dokümanlar",
        "so": "StackOverflow'da ara",
        "yes": "evet",
        "no": "hayır",
        "img_btn": "🎨 Görsel oluştur",
        "img_what": "Görsel hangi kısmı açıklasın?",
        "img_custom": "✍️ Kendi açıklamam",
        "img_custom_label": "İstediğin görseli tam olarak tarif et",
        "img_go": "Görseli oluştur",
        "img_wait": "Görsel çiziliyor…",
        "img_need_token": "Görsel oluşturmak için Ayarlar'a Hugging Face anahtarı ekle (FLUX.1-schnell ücretsiz ve açık).",
        "img_failed": "Görsel oluşturulamadı",
        "preview": "Önizleme",
        "detected": "Algılanan tür",
        "ctx_code": "Kod",
        "ctx_invoice": "Fatura",
        "ctx_document": "Belge",
        "ctx_image": "Görsel",
        "amount_due": "Ödenecek tutar",
        "due_date": "Son ödeme tarihi",
        "biller": "Düzenleyen",
        "reference": "Hesap / referans",
        "pay": "💳 Ödeme sayfasına git",
        "pay_hint": "Açılacak: {domain}. Ödeme bilgisi girmeden önce bağlantıyı kontrol et.",
        "pay_search_hint": "Dosyada ödeme bağlantısı yok; kurumun resmi ödeme sayfasını arayacağız.",
        "reminder": "⏰ Son ödeme hatırlatıcısı ekle",
        "csv": "⬇️ CSV",
        "xlsx": "⬇️ Excel",
        "items": "Detaylar",
        "doc_summary": "3 maddelik özet",
        "gmail": "✉️ Gmail taslağı oluştur",
        "actions": "Eylemler",
        "followup_hint": "Bu sonuç hakkında giriş kutusundan soru sorabilir veya yeni kod ya da dosya gönderebilirsin.",
        "error_api": "Model çağrısı başarısız oldu, bunun yerine demo yanıtı gösteriliyor.",
        "pdf_missing": "pypdf kurulu değil, PDF metni okunamıyor.",
        "pdf_pages": "Sayfa sayısı: {n}",
        "writing": "✍️ Model bilgisayarında yazıyor… (şu ana kadar {n} kelime)",
        "choose_action": "Ne yapmak istersin?",
        "opt_explain": "📖 Açıkla",
        "opt_pay": "💳 Faturayı öde",
        "opt_export": "⬇️ Verileri dışa aktar",
        "pick_option": "Bir seçenek seç: dosyayı açıklayayım ya da işlemi senin için yapayım.",
        "pay_opened": "Ödeme platformu yeni sekmede açıldı. Tarayıcın engellediyse düğmeye bas.",
        "ctx_task": "Eylem isteği",
        "action_title": "⚡ İstenen eylem",
        "act_email": "{to} adresine e-posta gönder",
        "act_pay": "Faturayı öde",
        "act_link": "Bağlantıyı aç: {url}",
        "act_call": "{phone} numarasını ara",
        "act_calendar": "{date} tarihini takvime ekle",
        "open_gmail": "✉️ Gmail'i aç",
        "open_link": "🔗 Bağlantıyı aç",
        "call_btn": "📞 Şimdi ara",
        "auto_opened": "Yeni sekmede otomatik açıldı. Tarayıcın engellediyse düğmeye bas.",
        "language": "Dil",
        "status": "Durum",
        "lang_badge": "Dil",
        "written_in": "Bu kod {lang} dilinde yazılmış.",
        "how_output": "Bu çıktıyı nasıl elde ediyoruz?",
        "glossary": "Yeni başlayanlar için sözlük",
        "history": "🕘 Geçmiş",
        "new": "＋ Yeni istek",
        "history_empty": "Henüz önceki istek yok.",
        "diagram_added": "İstediğin diyagram eklendi.",
        "image_added": "İstediğin görsel oluşturuldu.",
    },
}


def lang():
    return st.session_state.get("lang", "ar")


def t(key, **kw):
    """Translate a UI string into the current language."""
    text = T[lang()].get(key) or T["en"].get(key, key)
    return text.format(**kw) if kw else text


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css(rtl):
    direction = """
        [data-testid="stMainBlockContainer"], [data-testid="stSidebarUserContent"] { direction: rtl; }
        [data-testid="stChatInput"] textarea::placeholder { direction: rtl; text-align: right; }
        .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3 { text-align: right; }
    """ if rtl else ""
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, .stApp, button, input, textarea, select { font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif; }
.block-container { padding-top: 3.5rem; padding-bottom: 4rem; max-width: 1120px; }
__DIRECTION__
pre, code, .codeview, .ltr { direction: ltr !important; text-align: left !important; }
/* the single smart input: each line picks its own direction (Arabic text or code) */
[data-testid="stChatInput"] { direction: ltr; border: 2px solid __NAVY__ !important; border-radius: 18px !important; background: #fff !important;
  box-shadow: 0 8px 24px rgba(0,38,62,.10); }
[data-testid="stChatInput"] textarea { unicode-bidi: plaintext; text-align: start; font-size: 15px !important; }
[data-testid="stBottomBlockContainer"] { background: __MILK__; }
.hero { display:flex; align-items:center; gap:14px; }
.hero .logo { width:48px; height:48px; border-radius:14px; background: __NAVY__; color: __MILK__;
  display:flex; align-items:center; justify-content:center; font-size:26px; font-weight:700; }
.hero h1 { margin:0; padding:0; font-size: 2rem; color: __NAVY__; }
.hero-sub { color: __MUTED__; margin: .3rem 0 1rem 0; }
.card { border:1px solid #DDD6C8; border-radius:16px; padding:16px 18px; background: __CARD__; margin-bottom:12px; }
.card.primary { border-color: __POWDER__; background: __POWDER_SOFT__; }
.card.accent { border-color: __COCOA__; background: __COCOA_SOFT__; }
.card h4 { margin: 0 0 6px 0; font-size: 0.95rem; color: __MUTED__; font-weight:600; }
.greeting { text-align: center !important; font-size: 1.9rem; font-weight: 700; color: __NAVY__; margin: 14vh 0 18px 0; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; }
.welcome div { padding: 6px 0; font-size: 1.02rem; }
.sent { font-size: .9rem; color: __MUTED__; }
.sent code { direction: ltr; unicode-bidi: isolate; background: __NAVY_SOFT__; color: __NAVY__; padding: 1px 6px; border-radius: 6px; }
.big-number { font-size: 2.2rem; font-weight: 700; color: __NAVY__; line-height:1.2; }
.card.accent .big-number { color: __COCOA__; }
.pill { display:inline-block; padding:3px 12px; border-radius:999px; font-size:0.82rem; font-weight:600;
  background: __NAVY__; color: __MILK__; margin: 2px; }
.pill.soft { background: __POWDER_SOFT__; color: __NAVY__; border: 1px solid __POWDER__; }
.pill.cocoa { background: __COCOA_SOFT__; color: __COCOA__; }
.codeview { font-family:'JetBrains Mono', monospace; font-size:14px; border-radius:14px; overflow:hidden;
  background: __NAVY__; color: __MILK__; margin-bottom: 10px; }
.codeview .cl { display:flex; padding:3px 12px; white-space:pre; opacity:.5; }
.codeview .cl.on { opacity:1; background: rgba(148,178,196,.35); border-left: 4px solid __POWDER__; }
.codeview .ln { width:2.2em; color: __POWDER__; user-select:none; flex-shrink:0; }
.explain-box { font-size: 1.05rem; line-height: 1.8; }
.stButton button[kind="primary"], .stLinkButton a[kind="primary"] { background: __NAVY__; border-color: __NAVY__; color: __MILK__; border-radius: 12px; font-weight:600; }
.stButton button[kind="secondary"], .stDownloadButton button, .stLinkButton a[kind="secondary"] { border-radius: 12px; border-color: __POWDER__; background: __CARD__; }
.stButton button[kind="tertiary"] { color: __NAVY__; text-decoration: underline; text-underline-offset: 4px; }
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stBaseButton-segmented_controlActive"] { background: __NAVY__ !important; color: __MILK__ !important; border-color: __NAVY__ !important; }
</style>
"""
    for k, v in {"__DIRECTION__": direction, "__NAVY_SOFT__": NAVY_SOFT, "__NAVY__": NAVY,
                 "__POWDER_SOFT__": POWDER_SOFT, "__POWDER__": POWDER, "__COCOA_SOFT__": COCOA_SOFT,
                 "__COCOA__": COCOA, "__MILK__": MILK, "__CARD__": CARD, "__MUTED__": MUTED}.items():
        css = css.replace(k, v)
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LLM layer (Gemma 4 / Groq / demo)
# ---------------------------------------------------------------------------

class LLMError(Exception):
    pass


def settings():
    return st.session_state["settings"]


@st.cache_data(ttl=20, show_spinner=False)
def ollama_models():
    """Models installed in the local Ollama server, or None when it is not running."""
    try:
        names = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()["models"]]
        return names + [n[:-len(":latest")] for n in names if n.endswith(":latest")]
    except Exception:
        return None


def is_demo():
    s = settings()
    if s["provider"] == "ollama":
        return not ollama_models()
    return s["provider"] == "demo" or not s["api_key"]


def call_llm(prompt, images=None, json_mode=False):
    """Send a prompt (+ optional images as (mime, bytes)) to the selected model. Returns text."""
    s = settings()
    images = images or []

    if s["provider"] == "ollama":
        model = s["vision_model"] if images else s["model"]
        installed = ollama_models() or []
        if model not in installed:
            # e.g. Gemma 4 is still downloading: use a model that is already there for text
            fallback = next((m for m in installed if not images and "vision" not in m), None)
            if not fallback:
                raise LLMError(f"Ollama model '{model}' is not installed yet. Run: ollama pull {model}")
            model = fallback
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(data).decode() for _, data in images]}],
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        if model.startswith("gemma4"):
            body["think"] = False  # Gemma 4 can "think" first; off = much faster answers for students
        body["stream"] = True
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600, stream=True)
        if resp.status_code != 200:
            raise LLMError(f"Ollama {resp.status_code}: {resp.text[:300]}")
        progress, parts = st.empty(), []
        for i, raw in enumerate(resp.iter_lines()):
            if not raw:
                continue
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise LLMError(f"Ollama: {chunk['error']}")
            parts.append(chunk.get("message", {}).get("content", ""))
            if i % 12 == 0:  # show the student that something is happening
                progress.caption(t("writing", n=len("".join(parts).split())))
        progress.empty()
        return "".join(parts)

    if s["provider"] == "gemma":
        parts = [{"text": prompt}]
        for mime, data in images:
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
        resp = requests.post(
            GEMINI_URL.format(model=s["model"]),
            headers={"x-goog-api-key": s["api_key"], "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(f"Gemini API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            out_parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise LLMError(f"Unexpected response: {str(data)[:300]}")
        # Skip "thought" parts if thinking is enabled on the model
        return "".join(p.get("text", "") for p in out_parts if not p.get("thought"))

    if s["provider"] == "groq":
        content = [{"type": "text", "text": prompt}]
        for mime, data in images:
            url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {s['api_key']}"},
            json={
                "model": s["model"],
                "messages": [{"role": "user", "content": content if images else prompt}],
                "temperature": 0.3,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(f"Groq API {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    raise LLMError("No provider configured")


def parse_json(text):
    """Extract the first JSON object from a model answer (tolerates ``` fences and trailing commas)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("The model did not return JSON")
    raw = text[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))


def show_error(e):
    st.warning(t("error_api"))
    with st.expander("details"):
        st.code(str(e))


def run_json(prompt, images, fallback):
    """Call the model and parse JSON; fall back to a demo answer on any failure."""
    if is_demo():
        return fallback()
    try:
        return parse_json(call_llm(prompt, images, json_mode=True))
    except Exception as e:  # never crash the UI
        show_error(e)
        return fallback()


def run_text(prompt, images, fallback):
    if is_demo():
        return fallback
    try:
        return call_llm(prompt, images)
    except Exception as e:
        show_error(e)
        return fallback


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TUTOR_PROMPT = """You are an expert and very patient programming teacher for complete beginners.
The student pasted code that can be in ANY programming language (Python, C, C++, Java, JavaScript, TypeScript,
C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Dart, SQL, HTML, CSS, Bash, ...). First identify the language exactly.
Write ALL natural-language text in <<LANG>> — even if the code or its comments use another language.
Keep code, identifiers and printed values exactly as they are. Student level: <<LEVEL>>.

Teaching style:
- Very simple words and short sentences. Explain every technical term the first time it appears.
- Use a small real-life analogy when it helps.
- For each line say WHAT it does and WHY it is there.
- Explain step by step HOW the program produces its output.

Return ONLY valid JSON (no markdown, no comments) with exactly this shape:
{
  "language": "exact language name, e.g. C, Python, Java",
  "purpose": "1-2 sentences: what this program is for",
  "summary": "2-4 simple sentences: how it works overall",
  "lines": [{"line": 1, "code": "exact source line", "explanation": "1-2 short sentences: what it does and why", "concept": "short concept name or empty"}],
  "how_output": ["simple step 1", "simple step 2"],
  "final_output": "the exact full output, or empty if nothing is printed",
  "glossary": [{"term": "beginner term used in this code", "meaning": "simple meaning"}],
  "concepts": ["key concepts used"],
  "needs_trace": false,
  "trace": null,
  "needs_flowchart": false,
  "flowchart": "",
  "image_ideas": [],
  "quiz": [{"question": "...", "options": ["...", "...", "..."], "answer_index": 0, "why": "short reason"}]
}

Decide visuals like a good teacher — only when they truly help, never by default:
- "needs_trace": true ONLY if the code has loops, recursion or several function calls whose order is not obvious.
  Then "trace" = {"label": "start", "children": [NODE]}. Otherwise false and null.
- "needs_flowchart": true ONLY if the code has branching (if/else, switch) or loops AND more than about 5 meaningful lines.
  Then "flowchart" = Mermaid "flowchart TD" code with EVERY label in double quotes. Otherwise false and "".
- "image_ideas": 0-2 short ideas for a real-life analogy picture, only for an idea a picture really clarifies. Usually [].
NODE = {"label": "short step name such as i = 0", "line": 1, "vars": {"name": "value"}, "output": "printed at this step, or empty", "detail": "one simple sentence", "children": [NODE]}
Trace rules: follow the REAL execution order; each loop iteration is its own node whose children are the steps inside it
(so nested loops become nested nodes); put printed output on the node that prints it; at most 4 iterations per loop,
then one node labelled "..." that summarizes the rest.
Other rules: skip blank and comment-only lines in "lines"; "glossary" has 2-4 items; "quiz" has exactly 2 questions.
Be concise: short sentences, no repetition.

Student code:
```
<<CODE>>
```"""

FLOW_PROMPT = """Return ONLY Mermaid "flowchart TD" code (no markdown fences) that shows how this code runs.
Write every label in <<LANG>> and wrap every label in double quotes.
```
<<CODE>>
```"""

FILE_PROMPT = """You are an assistant that SIMPLIFIES files and turns them into actions.
Look at the attached file content. Write ALL natural-language text in <<LANG>>.
Return ONLY valid JSON with exactly this shape:
{
  "doc_type": "invoice" | "task" | "document" | "other",
  "title": "short title",
  "simple_summary": ["point 1", "point 2", "point 3"],
  "invoice": {
    "biller": "company that issued the bill, or empty",
    "amount_due": "number only, e.g. 45000",
    "currency": "currency code or symbol",
    "due_date": "YYYY-MM-DD or empty",
    "reference": "account / customer / invoice number, or empty",
    "payment_url": "payment URL printed on the bill (must start with https://), or empty — NEVER invent one",
    "line_items": [{"item": "...", "amount": "..."}]
  },
  "email_subject": "subject for an email about this file",
  "email_body": "short polite email body about this file",
  "action": {
    "type": "email" | "pay" | "link" | "call" | "calendar" | "none",
    "to": "email address written in the file, or empty",
    "subject": "email subject if type is email",
    "body": "short polite email body (in <<LANG>>) if type is email",
    "url": "https URL written in the file, or empty",
    "phone": "phone number written in the file, or empty",
    "date": "YYYY-MM-DD or empty"
  }
}
Fill "invoice" only when doc_type is "invoice"; otherwise leave its fields empty.
"action": if the file asks the reader to DO something — send an email to someone, pay, visit a link, call a number,
attend on a date — set "type" and copy addresses, links and numbers EXACTLY as written. NEVER invent them.
Use doc_type "task" when the main point of the file is such a request. Otherwise "type" is "none".
<<HINT>>
<<TEXT>>"""

ANSWER_PROMPT = """You are a patient teacher. Answer in <<LANG>> using short, clear Markdown.
If the question is about programming, explain it simply for a <<LEVEL>> and add a tiny example.
Question: <<REQUEST>>
<<CONTEXT>>"""


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", v)
    return template


def level_name():
    return t("level_beginner") if settings()["level"] == "beginner" else t("level_uni")


# ---------------------------------------------------------------------------
# Demo answers (used when no API key is set or a call fails)
# ---------------------------------------------------------------------------

DEMO_TXT = {
    "ar": {
        "purpose": "يطبع البرنامج كل الأزواج الممكنة من قيم المتغيرين i و j، ثم يطبع الكلمة done ليعلن انتهاءه.",
        "how_output": [
            "تبدأ الحلقة الخارجية بالقيمة i = 0.",
            "تعمل الحلقة الداخلية كاملةً: j = 0 ثم j = 1، فيُطبَع السطران 0 0 و 0 1.",
            "تنتقل الحلقة الخارجية إلى i = 1، فتعمل الحلقة الداخلية من جديد وتطبع 1 0 و 1 1.",
            "يتكرر الأمر مع i = 2، فيُطبَع 2 0 و 2 1.",
            "تنتهي الحلقتان، فيُطبَع السطر الأخير done.",
        ],
        "glossary": [
            ("الحلقة (loop)", "أمر يكرّر مجموعة من الأسطر عدة مرات، مثل عدّاد يعدّ خطوة خطوة."),
            ("المتغير (variable)", "اسم نحفظ فيه قيمة يمكن أن تتغير، مثل صندوق عليه اسم."),
            ("الدالة range", "تُنشئ سلسلة أرقام؛ فالتعبير range(3) يعني الأرقام 0 و 1 و 2."),
            ("الحلقات المتداخلة", "حلقة داخل حلقة؛ تعمل الداخلية كاملةً في كل دورة من دورات الخارجية."),
        ],
        "image_ideas": ["الحلقات المتداخلة مثل عقربَي الساعة: يدور عقرب الدقائق دورة كاملة مع كل خطوة لعقرب الساعات"],
        "generic_how": "أضف مفتاح Gemma 4 ليشرح المعلّم كيف يُنتج البرنامج مُخرَجه خطوة خطوة.",
        "summary": "يحتوي البرنامج على حلقتين متداخلتين: تعدّ الحلقة الخارجية المتغير i من 0 إلى 2، وفي كل دورة منها تُنفَّذ الحلقة الداخلية كاملةً فتعدّ المتغير j من 0 إلى 1، ثم يطبع البرنامج الكلمة done.",
        "lines": [
            ("الحلقة الخارجية: يأخذ المتغير i القيم 0 ثم 1 ثم 2.", "حلقة for"),
            ("الحلقة الداخلية: مع كل قيمة للمتغير i، يأخذ المتغير j القيمتين 0 ثم 1.", "الحلقات المتداخلة"),
            ("يطبع القيمتين الحاليتين للمتغيرين i و j في سطر واحد.", "دالة الطباعة print"),
            ("بعد انتهاء الحلقتين، يطبع الكلمة done مرة واحدة فقط.", "نهاية البرنامج"),
        ],
        "outer": "الحلقة الخارجية، الدورة رقم {n}: قيمة i الآن {i}.",
        "inner": "الحلقة الداخلية: j = {j}، وما زالت قيمة i هي {i}.",
        "print": "يطبع {i} {j}.",
        "done": "انتهت الحلقتان، فيطبع البرنامج done.",
        "start": "بداية البرنامج",
        "concepts": ["حلقة for", "الحلقات المتداخلة", "الدالة range", "دالة الطباعة print"],
        "flow": ['"البداية"', '"i من 0 إلى 2"', '"j من 0 إلى 1"', '"اطبع i و j"', '"اطبع done"', '"النهاية"'],
        "quiz": [
            ("كم مرة يُنفَّذ السطر print(i, j)؟", ["3", "5", "6"], 2, "ثلاث قيم للمتغير i × قيمتان للمتغير j = ست مرات."),
            ("ما أول سطر يُطبَع؟", ["0 0", "0 1", "done"], 0, "أول قيمة للمتغير i هي 0، وأول قيمة للمتغير j هي 0."),
            ("متى تُطبَع الكلمة done؟", ["بعد كل دورة", "مرة واحدة في النهاية", "لا تُطبَع أبداً"], 1, "لأنها خارج الحلقتين."),
        ],
        "generic": "الوضع التجريبي: أضف مفتاح Gemma 4 ليشرح المعلّم هذا السطر بالتفصيل.",
        "generic_summary": "هذا شرح تجريبي. أضف مفتاح API من الإعدادات لتحصل على شرح حقيقي لشيفرتك.",
        "invoice_biller": "الشركة الوطنية للكهرباء",
        "invoice_summary": ["فاتورة كهرباء لشهر أكتوبر.", "المبلغ المطلوب 45,000 دينار.", "ادفع قبل الموعد لتتجنّب الغرامة."],
        "doc_summary": ["هذا ملخّص تجريبي للمستند.", "أضف مفتاح API لتحصل على ملخّص حقيقي.", "يمكنك أيضاً إنشاء مسودة رسالة عنه."],
        "answer": "🧪 الوضع التجريبي: أضف مفتاح API في الإعدادات لأجيب عن سؤالك إجابة حقيقية.",
    },
    "en": {
        "purpose": "The program prints every possible pair of values of i and j, then prints done to show it has finished.",
        "how_output": [
            "The outer loop starts with i = 0.",
            "The inner loop runs completely: j = 0 then j = 1, so 0 0 and 0 1 are printed.",
            "The outer loop moves to i = 1, the inner loop runs again and prints 1 0 and 1 1.",
            "The same happens for i = 2, printing 2 0 and 2 1.",
            "Both loops are finished, so the last line, done, is printed.",
        ],
        "glossary": [
            ("loop", "An instruction that repeats some lines several times, like a counter counting step by step."),
            ("variable", "A name that stores a value which can change, like a labelled box."),
            ("range", "Creates a sequence of numbers; range(3) means 0, 1 and 2."),
            ("nested loops", "A loop inside a loop; the inner one runs completely for every round of the outer one."),
        ],
        "image_ideas": ["Nested loops as the two hands of a clock: the minute hand makes a full turn for every step of the hour hand"],
        "generic_how": "Add a Gemma 4 key so the tutor can explain step by step how the program produces its output.",
        "summary": "The program has two nested loops: the outer loop counts i from 0 to 2, and for each i the inner loop runs completely, counting j from 0 to 1. Finally it prints done.",
        "lines": [
            ("Outer loop: i takes the values 0, then 1, then 2.", "for loop"),
            ("Inner loop: for every value of i, j takes the values 0 then 1.", "nested loops"),
            ("Prints the current values of i and j on one line.", "print"),
            ("After both loops finish, prints done exactly once.", "end of program"),
        ],
        "outer": "Outer loop, round {n}: i is now {i}.",
        "inner": "Inner loop: j = {j} while i is still {i}.",
        "print": "Prints {i} {j}.",
        "done": "Both loops are finished, so it prints done.",
        "start": "Program start",
        "concepts": ["for loop", "nested loops", "range", "print"],
        "flow": ['"Start"', '"i from 0 to 2"', '"j from 0 to 1"', '"print i, j"', '"print done"', '"End"'],
        "quiz": [
            ("How many times does print(i, j) run?", ["3", "5", "6"], 2, "3 values of i × 2 values of j = 6 times."),
            ("What is the first line printed?", ["0 0", "0 1", "done"], 0, "The first i is 0 and the first j is 0."),
            ("When is done printed?", ["After every round", "Once at the end", "Never"], 1, "It is outside both loops."),
        ],
        "generic": "Demo mode: add a Gemma 4 key so the tutor can explain this line in detail.",
        "generic_summary": "This is a demo explanation. Add an API key in Settings for a real explanation of your code.",
        "invoice_biller": "National Electricity Co.",
        "invoice_summary": ["Electricity bill for October.", "You need to pay 45,000 IQD.", "Pay before the due date to avoid a late fee."],
        "doc_summary": ["This is a demo summary of the document.", "Add an API key for a real summary.", "You can also create an email draft about it."],
        "answer": "🧪 Demo mode: add an API key in Settings to get a real answer.",
    },
    "tr": {
        "purpose": "Program i ve j değerlerinin tüm olası çiftlerini yazdırır, sonra bittiğini göstermek için done yazdırır.",
        "how_output": [
            "Dış döngü i = 0 ile başlar.",
            "İç döngü tamamen çalışır: j = 0 sonra j = 1; 0 0 ve 0 1 yazdırılır.",
            "Dış döngü i = 1 olur, iç döngü yeniden çalışır ve 1 0 ile 1 1 yazdırılır.",
            "Aynısı i = 2 için olur; 2 0 ve 2 1 yazdırılır.",
            "İki döngü de biter ve son satır done yazdırılır.",
        ],
        "glossary": [
            ("döngü (loop)", "Bazı satırları birkaç kez tekrarlayan komut; adım adım sayan bir sayaç gibi."),
            ("değişken (variable)", "Değişebilen bir değeri saklayan isim; üzerinde etiket olan bir kutu gibi."),
            ("range", "Bir sayı dizisi oluşturur; range(3) 0, 1 ve 2 demektir."),
            ("iç içe döngüler", "Döngü içinde döngü; içteki, dıştakinin her turunda tamamen çalışır."),
        ],
        "image_ideas": ["İç içe döngüler saatin iki kolu gibi: akrep bir adım atarken yelkovan tam tur döner"],
        "generic_how": "Programın çıktısını adım adım nasıl ürettiğini öğretmenin açıklaması için Gemma 4 anahtarı ekle.",
        "summary": "Programda iç içe iki döngü var: dış döngü i'yi 0'dan 2'ye sayar, her i için iç döngü tamamen çalışır ve j'yi 0'dan 1'e sayar. En sonda done yazdırır.",
        "lines": [
            ("Dış döngü: i sırasıyla 0, 1 ve 2 değerlerini alır.", "for döngüsü"),
            ("İç döngü: her i değeri için j önce 0 sonra 1 olur.", "iç içe döngüler"),
            ("i ve j'nin o anki değerlerini tek satırda yazdırır.", "print"),
            ("İki döngü bitince done kelimesini bir kez yazdırır.", "programın sonu"),
        ],
        "outer": "Dış döngü, {n}. tur: i şimdi {i}.",
        "inner": "İç döngü: i hâlâ {i} iken j = {j}.",
        "print": "{i} {j} yazdırır.",
        "done": "İki döngü de bitti, done yazdırılır.",
        "start": "Program başlangıcı",
        "concepts": ["for döngüsü", "iç içe döngüler", "range", "print"],
        "flow": ['"Başla"', '"i: 0..2"', '"j: 0..1"', '"yazdır i, j"', '"yazdır done"', '"Bitiş"'],
        "quiz": [
            ("print(i, j) kaç kez çalışır?", ["3", "5", "6"], 2, "3 i değeri × 2 j değeri = 6 kez."),
            ("İlk yazdırılan satır nedir?", ["0 0", "0 1", "done"], 0, "İlk i 0 ve ilk j 0."),
            ("done ne zaman yazdırılır?", ["Her turdan sonra", "En sonda bir kez", "Hiç"], 1, "Çünkü iki döngünün de dışında."),
        ],
        "generic": "Demo modu: öğretmenin bu satırı detaylı açıklaması için Gemma 4 anahtarı ekle.",
        "generic_summary": "Bu bir demo açıklamasıdır. Gerçek açıklama için Ayarlar'dan API anahtarı ekle.",
        "invoice_biller": "Ulusal Elektrik Şirketi",
        "invoice_summary": ["Ekim ayı elektrik faturası.", "Ödenecek tutar 45.000 IQD.", "Gecikme cezası olmaması için son tarihten önce öde."],
        "doc_summary": ["Bu, belgenin demo özetidir.", "Gerçek özet için API anahtarı ekle.", "Bu belge hakkında e-posta taslağı da oluşturabilirsin."],
        "answer": "🧪 Demo modu: gerçek bir yanıt için Ayarlar'dan API anahtarı ekle.",
    },
}


def detect_language(code):
    """Best-effort guess of the programming language (used in demo mode and for docs links)."""
    c = code
    rules = [
        (r"#include\s*<(iostream|vector|string)>|std::|cout\s*<<", "C++"),
        (r"#include", "C"),
        (r"public\s+static\s+void\s+main|System\.out\.", "Java"),
        (r"Console\.Write|using\s+System", "C#"),
        (r"package\s+main|fmt\.Print", "Go"),
        (r"\bfn\s+main|println!", "Rust"),
        (r"<\?php", "PHP"),
        (r"(?i)\bselect\b.+\bfrom\b", "SQL"),
        (r"(?i)<(html|div|body|p)\b", "HTML"),
        (r"console\.log|\bfunction\b|=>|\b(let|const)\s+\w+", "JavaScript"),
        (r"\bfun\s+main|\bval\s+\w+", "Kotlin"),
        (r"\bputs\b|\bend\s*$", "Ruby"),
        (r"\bdef\s+\w+\(|\bprint\(|\bin\s+range\(|:\s*$", "Python"),
    ]
    for pattern, name in rules:
        if re.search(pattern, c, re.M):
            return name
    return ""


def code_traits(code):
    """Cheap signals used when the model does not say whether visuals are needed."""
    meaningful = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith(("#", "//"))]
    loops = bool(re.search(r"\b(for|while|do|foreach|forEach|loop)\b|\.map\(", code))
    branches = bool(re.search(r"\b(if|else|elif|switch|case|match)\b", code))
    return {"trace": loops, "flowchart": (loops or branches) and len(meaningful) > 5, "lines": meaningful}


def demo_tutor(code):
    d = DEMO_TXT[lang()]
    if code.strip() == SAMPLE_CODE.strip():
        src = SAMPLE_CODE.splitlines()
        children, output = [], []
        for i in range(3):
            inner = []
            for j in range(2):
                output.append(f"{i} {j}")
                inner.append({
                    "label": f"j = {j}", "line": 2, "vars": {"i": i, "j": j},
                    "detail": d["inner"].format(i=i, j=j),
                    "children": [{"label": "print(i, j)", "line": 3, "vars": {"i": i, "j": j},
                                  "output": f"{i} {j}", "detail": d["print"].format(i=i, j=j)}],
                })
            children.append({"label": f"i = {i}", "line": 1, "vars": {"i": i},
                             "detail": d["outer"].format(n=i + 1, i=i), "children": inner})
        output.append("done")
        children.append({"label": 'print("done")', "line": 4, "output": "done", "detail": d["done"]})
        f, yes, no = d["flow"], t("yes"), t("no")
        return {
            "language": "Python",
            "purpose": d["purpose"],
            "summary": d["summary"],
            "lines": [{"line": n + 1, "code": src[n], "explanation": e, "concept": c}
                      for n, (e, c) in enumerate(d["lines"])],
            "how_output": d["how_output"],
            "final_output": "\n".join(output),
            "glossary": [{"term": a, "meaning": b} for a, b in d["glossary"]],
            "concepts": d["concepts"],
            "needs_trace": True,
            "trace": {"label": d["start"], "children": children},
            "needs_flowchart": True,
            "flowchart": (
                f"flowchart TD\n A({f[0]}) --> B{{{f[1]}}}\n B -- {yes} --> C{{{f[2]}}}\n"
                f" C -- {yes} --> D[{f[3]}] --> C\n C -- {no} --> B\n B -- {no} --> E[{f[4]}] --> F({f[5]})"
            ),
            "image_ideas": d["image_ideas"],
            "quiz": [{"question": q, "options": o, "answer_index": a, "why": w} for q, o, a, w in d["quiz"]],
        }

    # Generic demo for any other code: real structure, placeholder explanations
    traits = code_traits(code)
    src = [ln for ln in code.splitlines() if ln.strip()]
    return {
        "language": detect_language(code),
        "purpose": "",
        "summary": d["generic_summary"],
        "lines": [{"line": n + 1, "code": ln, "explanation": d["generic"], "concept": ""} for n, ln in enumerate(src)],
        "how_output": [d["generic_how"]],
        "final_output": "",
        "glossary": [],
        "concepts": [],
        "needs_trace": traits["trace"],
        "trace": simple_trace(src) if traits["trace"] else None,
        "needs_flowchart": traits["flowchart"],
        "flowchart": simple_flowchart(src) if traits["flowchart"] else "",
        "image_ideas": [],
        "quiz": [],
    }


def simple_trace(src):
    return {"label": DEMO_TXT[lang()]["start"],
            "children": [{"label": ln.strip()[:40], "line": n + 1} for n, ln in enumerate(src[:15])]}


def simple_flowchart(src):
    nodes = [f'N{n}["{ln.strip()[:40].replace(chr(34), chr(39))}"]' for n, ln in enumerate(src[:10])]
    return "flowchart TD\n " + " --> ".join(nodes)


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*")
URL_RE = re.compile(r"https://[^\s<>\"')]+")
ACTION_WORDS = ["send", "email", "mail", "contact", "write to", "أرسل", "ارسل", "راسل", "بريد", "إيميل", "ايميل",
                "gönder", "e-posta", "yaz"]


def demo_file(ctx, text=""):
    d = DEMO_TXT[lang()]
    emails = EMAIL_RE.findall(text)
    if ctx == "task" and emails:
        return {"doc_type": "task", "title": emails[0], "simple_summary": [text.strip()[:160]], "invoice": {},
                "email_subject": "", "email_body": "",
                "action": {"type": "email", "to": emails[0], "subject": "", "body": text.strip()[:400]}}
    if ctx in ("invoice", "image"):
        due = (dt.date.today() + dt.timedelta(days=7)).isoformat()
        return {
            "doc_type": "invoice",
            "title": d["invoice_biller"],
            "simple_summary": d["invoice_summary"],
            "invoice": {
                "biller": d["invoice_biller"], "amount_due": "45000", "currency": "IQD",
                "due_date": due, "reference": "EL-2026-10-7781", "payment_url": "",
                "line_items": [{"item": "kWh 520", "amount": "39000"}, {"item": "Service fee", "amount": "6000"}],
            },
            "email_subject": d["invoice_biller"],
            "email_body": " ".join(d["invoice_summary"]),
        }
    return {
        "doc_type": "document", "title": "Demo", "simple_summary": d["doc_summary"], "invoice": {},
        "email_subject": "Summary", "email_body": "\n".join(f"- {s}" for s in d["doc_summary"]),
    }


# ---------------------------------------------------------------------------
# Visual components
# ---------------------------------------------------------------------------

TREE_HTML = """
<div id="wrap" dir="__DIR__">
  <div class="bar">
    <button id="back" class="ghost">__BACK__</button>
    <button id="step" class="primary">▶ __STEP__</button>
    <span class="spacer"></span>
    <button id="exp" class="ghost">__EXPAND__</button>
    <button id="col" class="ghost">__COLLAPSE__</button>
  </div>
  <ul id="tree" class="root"></ul>
</div>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap');
  * { box-sizing: border-box; }
  body { margin:0; font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif; color:__NAVY__; background:__MILK__; }
  .bar { display:flex; gap:8px; align-items:center; margin-bottom:10px; position:sticky; top:0; background:__MILK__; padding:4px 0; z-index:2; }
  .spacer { flex:1; }
  button { border-radius:10px; padding:6px 14px; font-weight:600; cursor:pointer; font-family:inherit; font-size:14px; }
  button.primary { background:__NAVY__; color:__MILK__; border:1px solid __NAVY__; }
  button.ghost { background:__CARD__; color:__NAVY__; border:1px solid __POWDER__; }
  ul { list-style:none; margin:0; padding-inline-start:22px; border-inline-start:2px dashed __POWDER__; }
  ul.root { padding-inline-start:0; border:none; }
  li { margin:6px 0; }
  .node { display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:8px 12px; border-radius:12px;
          border:1px solid #DDD6C8; background:__CARD__; cursor:pointer; transition: all .15s; }
  .node:hover { border-color:__NAVY__; }
  .node.has { background:__POWDER_SOFT__; border-color:__POWDER__; }
  .node.active { outline:3px solid __COCOA__; background:__COCOA_SOFT__; }
  .caret { width:18px; color:__NAVY__; font-weight:700; }
  .label { font-family:'JetBrains Mono', 'IBM Plex Sans Arabic', monospace; font-weight:600; direction:ltr; unicode-bidi:isolate; }
  .line { font-size:12px; color:__MUTED__; background:#E9E4D9; border-radius:6px; padding:1px 6px; }
  .var { font-family:'JetBrains Mono', monospace; font-size:12px; background:__NAVY_SOFT__; color:__NAVY__; border-radius:999px; padding:1px 8px; direction:ltr; }
  .out { font-family:'JetBrains Mono', monospace; font-size:13px; background:__COCOA__; color:__MILK__; border-radius:999px; padding:1px 10px; direction:ltr; }
  .detail { width:100%; font-size:13px; color:#3D4B55; margin-top:2px; }
  .hidden { display:none; }
</style>
<script>
const DATA = __DATA__;
const T = {line: "__LINE__", out: "__OUT__"};
const CLOSED = document.getElementById('wrap').dir === 'rtl' ? '◂' : '▸';  // caret points along the reading direction
const order = [];  // DFS order of nodes for step-by-step walking
let pos = -1;

function make(node, depth, parentLi) {
  const li = document.createElement('li');
  const kids = node.children || [];
  const box = document.createElement('div');
  box.className = 'node' + (kids.length ? ' has' : '');
  const caret = document.createElement('span');
  caret.className = 'caret';
  caret.textContent = kids.length ? (depth < 1 ? '▾' : CLOSED) : '•';
  box.appendChild(caret);
  const lab = document.createElement('span'); lab.className = 'label'; lab.textContent = node.label || '';
  box.appendChild(lab);
  if (node.line) { const s = document.createElement('span'); s.className='line'; s.textContent = T.line + ' ' + node.line; box.appendChild(s); }
  Object.entries(node.vars || {}).forEach(([k, v]) => {
    const s = document.createElement('span'); s.className='var'; s.textContent = k + ' = ' + v; box.appendChild(s);
  });
  if (node.output) { const s = document.createElement('span'); s.className='out'; s.textContent = '📤 ' + node.output; s.title = T.out; box.appendChild(s); }
  if (node.detail) { const s = document.createElement('div'); s.className='detail'; s.textContent = node.detail; box.appendChild(s); }
  li.appendChild(box);
  li._parent = parentLi;
  li._box = box;
  order.push(li);
  if (kids.length) {
    const ul = document.createElement('ul');
    if (depth >= 1) ul.classList.add('hidden');
    kids.forEach(k => ul.appendChild(make(k, depth + 1, li)));
    li.appendChild(ul);
    li._ul = ul; li._caret = caret;
    box.addEventListener('click', () => toggle(li));
  }
  return li;
}
function setOpen(li, open) {
  if (!li._ul) return;
  li._ul.classList.toggle('hidden', !open);
  li._caret.textContent = open ? '▾' : CLOSED;
}
function toggle(li) { setOpen(li, li._ul.classList.contains('hidden')); }
function all(open) { order.forEach(li => setOpen(li, open)); }
function focusAt(i) {
  order.forEach(li => li._box.classList.remove('active'));
  if (i < 0 || i >= order.length) return;
  const li = order[i];
  let p = li._parent; while (p) { setOpen(p, true); p = p._parent; }
  li._box.classList.add('active');
  // scroll inside this frame only, never the parent page
  const top = li._box.getBoundingClientRect().top + window.scrollY - 90;
  window.scrollTo({top: Math.max(0, top), behavior: 'smooth'});
}
document.getElementById('tree').appendChild(make(DATA, 0, null));
document.getElementById('step').onclick = () => { pos = Math.min(pos + 1, order.length - 1); focusAt(pos); };
document.getElementById('back').onclick = () => { pos = Math.max(pos - 1, 0); focusAt(pos); };
document.getElementById('exp').onclick = () => all(true);
document.getElementById('col').onclick = () => { all(false); setOpen(order[0], true); };
</script>
"""


def palette(page):
    for k, v in {"__NAVY_SOFT__": NAVY_SOFT, "__NAVY__": NAVY, "__POWDER_SOFT__": POWDER_SOFT,
                 "__POWDER__": POWDER, "__COCOA_SOFT__": COCOA_SOFT, "__COCOA__": COCOA,
                 "__MILK__": MILK, "__CARD__": CARD, "__MUTED__": MUTED}.items():
        page = page.replace(k, v)
    return page


def render_tree(trace):
    page = (TREE_HTML
            .replace("__DATA__", json.dumps(trace, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__DIR__", "rtl" if lang() == "ar" else "ltr")
            .replace("__STEP__", t("t_step")).replace("__BACK__", t("t_back"))
            .replace("__EXPAND__", t("t_expand")).replace("__COLLAPSE__", t("t_collapse"))
            .replace("__LINE__", t("t_line")).replace("__OUT__", t("t_output")))
    visible = 1 + len(trace.get("children") or [])  # first level is open on load
    components.html(palette(page), height=max(260, min(560, 90 + 70 * visible)), scrolling=True)


def render_mermaid(code):
    page = """
<div class="mermaid">__CODE__</div>
<pre id="err" style="display:none;white-space:pre-wrap;font-size:12px;color:__MUTED__"></pre>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
mermaid.initialize({ startOnLoad:false, theme:'base', securityLevel:'strict',
  themeVariables:{ primaryColor:'__POWDER_SOFT__', primaryBorderColor:'__NAVY__', primaryTextColor:'__NAVY__',
                   lineColor:'__COCOA__', secondaryColor:'__COCOA_SOFT__', tertiaryColor:'__MILK__',
                   fontFamily:'IBM Plex Sans Arabic, sans-serif' } });
try { await mermaid.run({ querySelector: '.mermaid' }); }
catch (e) { const el = document.getElementById('err'); el.style.display='block';
            el.textContent = document.querySelector('.mermaid').textContent; document.querySelector('.mermaid').remove(); }
</script>
<style>body{margin:0;display:flex;justify-content:center;font-family:sans-serif;background:__MILK__}</style>
""".replace("__CODE__", html.escape(code))
    components.html(palette(page), height=420, scrolling=True)


def render_codeview(lines, active):
    rows = []
    for ln in lines:
        cls = "cl on" if ln.get("line") == active else "cl"
        rows.append(f'<div class="{cls}"><span class="ln">{ln.get("line", "")}</span>'
                    f'<span>{html.escape(str(ln.get("code", "")))}</span></div>')
    st.markdown(f'<div class="codeview">{"".join(rows)}</div>', unsafe_allow_html=True)


def focus_input():
    """Put the cursor in the smart input on load so the main action is obvious."""
    components.html(
        "<script>setTimeout(()=>{const a=window.parent.document.querySelector('[data-testid=\"stChatInputTextArea\"]');"
        "if(a&&!a.value)a.focus();},400);</script>",
        height=0,
    )


# ---------------------------------------------------------------------------
# Image generation (on demand only)
# ---------------------------------------------------------------------------

def generate_image(prompt):
    token = settings()["hf_token"]
    if not token:
        raise LLMError(t("img_need_token"))
    resp = requests.post(FLUX_URL, headers={"Authorization": f"Bearer {token}"},
                         json={"inputs": prompt}, timeout=180)
    if resp.status_code != 200 or not resp.headers.get("content-type", "").startswith("image"):
        raise LLMError(f"{resp.status_code}: {resp.text[:300]}")
    return resp.content


def image_prompt(subject, custom):
    style = ("Friendly flat educational illustration for programming beginners, clean shapes, "
             "palette of navy #00263E, powder blue #94B2C4, cocoa brown #594536 on a coconut-milk #F0EDE5 background, "
             "no text, no letters, no code.")
    if custom:
        return f"{custom}. {style}"
    return f"An illustration that explains the programming idea '{subject}' with a simple real-life analogy. {style}"


def image_panel(result, prefix):
    with st.popover(t("img_btn"), width="stretch"):
        if not settings()["hf_token"]:
            st.info(t("img_need_token"))
        options = list(result.get("image_ideas") or result.get("concepts") or [])
        options.append(t("img_custom"))
        choice = st.selectbox(t("img_what"), options, key=f"{prefix}_imgsel")
        custom = ""
        if choice == t("img_custom"):
            custom = st.text_area(t("img_custom_label"), key=f"{prefix}_imgtxt", height=80)
        if st.button(t("img_go"), type="primary", key=f"{prefix}_imggo", width="stretch"):
            make_image(prefix, choice, custom)
    if st.session_state.get(f"{prefix}_imgerr"):
        st.error(st.session_state[f"{prefix}_imgerr"])
    if st.session_state.get(f"{prefix}_img"):
        img, caption = st.session_state[f"{prefix}_img"]
        st.image(img, caption=caption, width="stretch")


def make_image(prefix, subject, custom=""):
    st.session_state.pop(f"{prefix}_imgerr", None)
    with st.spinner(t("img_wait")):
        try:
            st.session_state[f"{prefix}_img"] = (generate_image(image_prompt(subject, custom)), custom or subject)
        except Exception as e:
            # shown outside the popover, which closes on rerun
            st.session_state[f"{prefix}_imgerr"] = f'{t("img_failed")}: {e}'


# ---------------------------------------------------------------------------
# Role 1 — Code tutor
# ---------------------------------------------------------------------------

def explain_code(code):
    prompt = fill(TUTOR_PROMPT, LANG=LANG_NAMES[lang()], LEVEL=level_name(), CODE=code)
    result = run_json(prompt, None, lambda: demo_tutor(code))
    result.setdefault("lines", [])
    result.setdefault("trace", {"label": "start", "children": []})
    return result


DOCS = {
    "python": "https://docs.python.org/3/search.html?q={q}",
    "javascript": "https://developer.mozilla.org/en-US/search?q={q}",
    "typescript": "https://developer.mozilla.org/en-US/search?q={q}",
    "html": "https://developer.mozilla.org/en-US/search?q={q}",
    "css": "https://developer.mozilla.org/en-US/search?q={q}",
    "c++": "https://en.cppreference.com/mwiki/index.php?search={q}",
    "c": "https://en.cppreference.com/mwiki/index.php?search={q}",
    "java": "https://docs.oracle.com/search/?q={q}",
    "c#": "https://learn.microsoft.com/en-us/search/?terms={q}",
    "go": "https://pkg.go.dev/search?q={q}",
    "rust": "https://doc.rust-lang.org/std/?search={q}",
    "php": "https://www.php.net/manual-lookup.php?pattern={q}",
}


def docs_links(language, concepts):
    q = urllib.parse.quote(" ".join(concepts[:2]) if concepts else language)
    docs = DOCS.get((language or "").strip().lower(), "https://devdocs.io/#q={q}").format(q=q)
    so = f"https://stackoverflow.com/search?q={urllib.parse.quote(f'[{language.lower()}] ' if language else '')}{q}"
    return docs, so


def as_bool(v):
    return v is True or str(v).strip().lower() == "true"


def visual_flags(view):
    """Which visuals to show: what the model decided, or what the student explicitly asked for."""
    r, traits = view["result"], code_traits(view["code"])
    trace_ok = bool((r.get("trace") or {}).get("children"))
    forced = view.get("force_diagram")
    return {
        "trace": trace_ok and (forced or as_bool(r.get("needs_trace", traits["trace"]))),
        "flowchart": bool(r.get("flowchart")) and (forced or as_bool(r.get("needs_flowchart", traits["flowchart"]))),
        "image": bool(r.get("image_ideas")) or view.get("show_image", False),
    }


def render_tutor(view, prefix):
    result = view["result"]
    lines = result.get("lines", [])
    flags = visual_flags(view)
    language = (result.get("language") or detect_language(view["code"])).strip()

    # --- Language + what the program is for + how it works ---
    if language:
        st.markdown(f'<span class="pill soft">🔤 {t("lang_badge")}: <b>{html.escape(language)}</b></span>',
                    unsafe_allow_html=True)
    intro = t("written_in", lang=language) + " " if language else ""
    paragraphs = [p for p in (intro + result.get("purpose", ""), result.get("summary", "")) if p.strip()]
    st.markdown(f'<div class="card primary"><h4>{t("summary")}</h4>'
                + "".join(f'<div class="explain-box">{html.escape(p)}</div>' for p in paragraphs) + "</div>",
                unsafe_allow_html=True)
    if result.get("concepts"):
        st.markdown("".join(f'<span class="pill soft">{html.escape(c)}</span>' for c in result["concepts"]),
                    unsafe_allow_html=True)

    # --- Line by line stepper ---
    st.subheader(t("line_by_line"), anchor=False)
    if lines:
        key = f"{prefix}_step"
        st.session_state.setdefault(key, 0)
        idx = max(0, min(st.session_state[key], len(lines) - 1))
        cur = lines[idx]
        left, right = st.columns([1.1, 1])
        with left:
            render_codeview(lines, cur.get("line"))
        with right:
            st.caption(t("step_of", i=idx + 1, n=len(lines)))
            st.markdown(f'<div class="card"><div class="ltr" style="font-family:JetBrains Mono,monospace;'
                        f'font-weight:600;margin-bottom:6px">{html.escape(cur.get("code", ""))}</div>'
                        f'<div class="explain-box">{html.escape(cur.get("explanation", ""))}</div></div>',
                        unsafe_allow_html=True)
            if cur.get("concept"):
                st.markdown(f'<span class="pill cocoa">{t("concept")}: {html.escape(cur["concept"])}</span>',
                            unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            if b1.button(t("prev"), key=f"{prefix}_prev", width="stretch", disabled=idx == 0):
                st.session_state[key] = idx - 1
                st.rerun()
            if b2.button(t("next"), key=f"{prefix}_next", type="primary", width="stretch",
                         disabled=idx >= len(lines) - 1):
                st.session_state[key] = idx + 1
                st.rerun()
        with st.expander(t("all_lines")):
            for ln in lines:
                st.markdown(f'**{ln.get("line")}.** `{ln.get("code", "").strip()}` — {ln.get("explanation", "")}')

    # --- How the output is produced ---
    steps = [s for s in result.get("how_output") or [] if str(s).strip()]
    if steps or result.get("final_output"):
        st.subheader(t("how_output"), anchor=False)
        if steps:
            st.markdown('<div class="card"><ol style="margin:0;padding-inline-start:1.4em">'
                        + "".join(f'<li class="explain-box">{html.escape(str(s))}</li>' for s in steps)
                        + "</ol></div>", unsafe_allow_html=True)
        if result.get("final_output"):
            st.markdown(f'<span class="pill cocoa">📤 {t("final_output")}</span>', unsafe_allow_html=True)
            st.code(result["final_output"], language="text")

    # --- Visuals: only when they help or when asked ---
    if flags["trace"]:
        st.subheader(t("trace"), anchor=False)
        st.caption(t("trace_hint"))
        render_tree(result["trace"])
    if flags["flowchart"]:
        st.subheader(t("flowchart"), anchor=False)
        render_mermaid(result["flowchart"])
    if flags["image"]:
        image_panel(result, prefix)

    # --- Quiz / glossary / links ---
    tabs = st.tabs([t("quiz"), t("glossary"), t("learn_more")])
    with tabs[0]:
        for qi, q in enumerate(result.get("quiz", [])):
            st.markdown(f'**{qi + 1}. {q.get("question", "")}**')
            pick = st.radio(" ", q.get("options", []), key=f"{prefix}_q{qi}", index=None,
                            label_visibility="collapsed")
            if pick is not None:
                ok = q.get("options", []).index(pick) == q.get("answer_index")
                (st.success if ok else st.error)(f'{t("correct") if ok else t("wrong")} {q.get("why", "")}')
        if not result.get("quiz"):
            st.caption("—")
    with tabs[1]:
        for g in result.get("glossary", []):
            st.markdown(f'**{g.get("term", "")}** — {g.get("meaning", "")}')
        if not result.get("glossary"):
            st.caption("—")
    with tabs[2]:
        docs, so = docs_links(language, result.get("concepts", []))
        c1, c2 = st.columns(2)
        c1.link_button(f'📘 {t("docs")}', docs, width="stretch")
        c2.link_button(f'🔎 {t("so")}', so, width="stretch")


# ---------------------------------------------------------------------------
# Role 2 — Files & actions
# ---------------------------------------------------------------------------

def read_file(f):
    """Return (ext, raw bytes, extracted text, page count)."""
    ext = f.name.rsplit(".", 1)[-1].lower()
    data = f.getvalue()
    text, pages = "", 0
    if ext == "pdf" and PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            text = "\n".join((p.extract_text() or "") for p in reader.pages[:6])[:8000]
        except Exception:
            text = ""
    elif ext in CODE_EXT or ext == "txt":
        text = data.decode("utf-8", errors="replace")[:12000]
    return ext, data, text, pages


CODE_PATTERNS = re.compile(
    r"(\bdef\s+\w+\s*\(|\bfunction\b|#include|\bclass\s+\w+|=>|\breturn\b|;\s*$|\bimport\s+\w+|"
    r"\bprint\s*\(|console\.log|\bfor\s*\(|\bfor\s+\w+\s+in\b|\bwhile\b.*:|\bif\s*\(|\bif\b.*:\s*$|"
    r"\b(int|let|const|var|public|static|void)\s+\w+|^\s*[A-Za-z_]\w*\s*=\s*\S|[{}]\s*$|\b\w+\.\w+\(.*\))",
    re.M,
)


def looks_like_code(text):
    """Heuristic router: is this text source code (vs. a question in natural language)?"""
    if "```" in text:
        return True
    arabic = len(re.findall(r"[؀-ۿ]", text))
    if arabic > 0.3 * max(1, len(text.replace(" ", ""))):
        return False
    hits = len(CODE_PATTERNS.findall(text))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return hits >= 2 or (hits >= 1 and len(lines) <= 2 and len(text) < 120)


def strip_fences(text):
    m = re.search(r"```[\w+-]*\n(.*?)```", text, re.S)
    return m.group(1).rstrip() if m else text


def detect_context(name, ext, text):
    if ext in CODE_EXT or (ext == "txt" and looks_like_code(text)):
        return "code"
    blob = f"{name} {text}".lower()
    if any(w in blob for w in FINANCE_WORDS):
        return "invoice"
    if EMAIL_RE.search(text):
        return "task"
    if ext in IMAGE_EXT:
        return "image"  # the model decides what the picture is
    return "document"


def image_parts(ext, data):
    return [(f"image/{'jpeg' if ext == 'jpg' else ext}", data)] if ext in IMAGE_EXT else None


def analyze_file(name, ext, data, text, ctx):
    hint = f"Hint: the file name is '{name}' and it looks like: {ctx}."
    body = f"File text:\n{text}" if text else "The file is the attached image."
    prompt = fill(FILE_PROMPT, LANG=LANG_NAMES[lang()], HINT=hint, TEXT=body)
    return run_json(prompt, image_parts(ext, data), lambda: demo_file(ctx, text))


def safe_https(url):
    try:
        p = urllib.parse.urlparse((url or "").strip())
        return url.strip() if p.scheme == "https" and p.netloc else ""
    except Exception:
        return ""


def calendar_link(title, due):
    try:
        day = dt.date.fromisoformat(due)
    except (TypeError, ValueError):
        return ""
    dates = f"{day:%Y%m%d}/{day + dt.timedelta(days=1):%Y%m%d}"
    return "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode(
        {"action": "TEMPLATE", "text": title, "dates": dates})


def resolve_action(res, text):
    """Turn the model's action into (kind, label, url). Addresses must really appear in the file or the answer."""
    act = dict(res.get("action") or {})
    kind = (act.get("type") or "none").lower()
    if kind == "none" and res.get("doc_type") == "task":
        kind = "email" if EMAIL_RE.search(text + json.dumps(act)) else "none"
    if kind == "email":
        to = act.get("to") or ""
        if not EMAIL_RE.fullmatch(to.strip()):
            found = EMAIL_RE.findall(text) or EMAIL_RE.findall(json.dumps(res, ensure_ascii=False))
            to = found[0] if found else ""
        if not to:
            return None
        url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(
            {"view": "cm", "fs": "1", "to": to.strip(), "su": act.get("subject", ""), "body": act.get("body", "")})
        return "email", t("act_email", to=to.strip()), url
    if kind == "link":
        url = safe_https(act.get("url")) or next(iter(URL_RE.findall(text)), "")
        return ("link", t("act_link", url=urllib.parse.urlparse(url).netloc), url) if url else None
    if kind == "call" and re.sub(r"[^\d+]", "", act.get("phone", "")):
        phone = re.sub(r"[^\d+]", "", act["phone"])
        return "call", t("act_call", phone=phone), f"tel:{phone}"
    if kind == "calendar" and calendar_link("x", act.get("date")):
        return "calendar", t("act_calendar", date=act["date"]), calendar_link(res.get("title", ""), act["date"])
    return None


def payment_search_url(inv):
    words = {"ar": "دفع الفاتورة إلكترونياً", "en": "pay bill online", "tr": "online fatura ödeme"}[lang()]
    return "https://www.google.com/search?q=" + urllib.parse.quote(f'{inv.get("biller", "")} {words}')


def payment_url(inv):
    """The payment link printed on the bill, or a search for that biller's official payment platform."""
    url = safe_https(inv.get("payment_url"))
    if url:
        return url, t("pay_hint", domain=urllib.parse.urlparse(url).netloc)
    return payment_search_url(inv), t("pay_search_hint")


def open_once(view, key, url):
    """Open a page in a new tab right after the user picks the option (only once per option)."""
    opened = view.setdefault("opened", [])
    if key not in opened:
        opened.append(key)
        components.html(f"<script>window.open({json.dumps(url)}, '_blank');</script>", height=0)


def file_options(view):
    """The possible operations for this file: explain it, or take one of the actions it allows."""
    res = view["result"]
    inv = res.get("invoice") or {}
    action = resolve_action(res, view["text"])
    options = ["explain"]
    if res.get("doc_type") == "invoice":
        options.append("pay")
        if calendar_link("x", inv.get("due_date")):
            options.append("reminder")
        options.append("export")
    if action and action[0] not in options:
        options.insert(1, action[0])
    if not action and res.get("doc_type") != "invoice":
        options.append("email_draft")
    return options, action


def fmt_amount(value):
    try:
        return f"{float(str(value).replace(',', '')):,.0f}"
    except ValueError:
        return str(value)


def invoice_explanation(res):
    inv = res.get("invoice") or {}
    amount = fmt_amount(inv.get("amount_due", ""))
    a, b = st.columns(2)
    a.markdown(f'<div class="card primary"><h4>{t("amount_due")}</h4>'
               f'<div class="big-number">{html.escape(str(amount))} {html.escape(inv.get("currency", ""))}</div></div>',
               unsafe_allow_html=True)
    b.markdown(f'<div class="card accent"><h4>{t("due_date")}</h4>'
               f'<div class="big-number">{html.escape(inv.get("due_date") or "—")}</div></div>',
               unsafe_allow_html=True)
    st.markdown(f'**{t("biller")}:** {inv.get("biller") or "—"}  \n**{t("reference")}:** {inv.get("reference") or "—"}')
    if inv.get("line_items"):
        with st.expander(t("items")):
            st.dataframe(pd.DataFrame(inv["line_items"]), width="stretch", hide_index=True)
    return amount


def export_buttons(res):
    inv = res.get("invoice") or {}
    rows = [{"field": k, "value": v} for k, v in inv.items() if k != "line_items"]
    rows += [{"field": i.get("item", ""), "value": i.get("amount", "")} for i in inv.get("line_items") or []]
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    c1.download_button(t("csv"), df.to_csv(index=False).encode("utf-8-sig"), "invoice.csv", "text/csv",
                       width="stretch", key="file_csv")
    buf = io.BytesIO()
    try:
        df.to_excel(buf, index=False)
        c2.download_button(t("xlsx"), buf.getvalue(), "invoice.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch", key="file_xlsx")
    except ImportError:
        pass


def render_file_options(view):
    res = view["result"]
    inv = res.get("invoice") or {}
    options, action = file_options(view)
    labels = {"explain": t("opt_explain"), "pay": t("opt_pay"), "reminder": t("reminder"), "export": t("opt_export"),
              "email": t("open_gmail"), "link": t("open_link"), "call": t("call_btn"), "email_draft": t("gmail")}
    # an explicit request in the file (e.g. "send an email to x@y") is done right away, as asked
    default = "email" if action and action[0] == "email" else None
    choice = st.pills(t("choose_action"), options, format_func=labels.get, default=default, key="file_choice")
    if choice is None:
        st.caption(t("pick_option"))
        return

    if choice == "explain":
        if res.get("doc_type") == "invoice":
            invoice_explanation(res)
        st.markdown(f'<div class="card"><h4>{t("doc_summary")}</h4>'
                    + "".join(f"<div>• {html.escape(x)}</div>" for x in res.get("simple_summary", [])[:3])
                    + "</div>", unsafe_allow_html=True)
    elif choice == "pay":
        url, hint = payment_url(inv)
        st.markdown(f'<div class="card accent"><h4>{t("amount_due")}</h4><div class="big-number">'
                    f'{html.escape(fmt_amount(inv.get("amount_due", "")))} {html.escape(inv.get("currency", ""))}</div></div>',
                    unsafe_allow_html=True)
        st.link_button(t("pay"), url, type="primary", width="stretch")
        st.caption(hint)
        open_once(view, "pay", url)
        st.caption(t("pay_opened"))
    elif choice == "reminder":
        url = calendar_link(f'{t("due_date")}: {inv.get("biller", "")} {inv.get("amount_due", "")} {inv.get("currency", "")}',
                            inv.get("due_date"))
        st.link_button(t("reminder"), url, type="primary", width="stretch")
        open_once(view, "reminder", url)
        st.caption(t("auto_opened"))
    elif choice == "export":
        export_buttons(res)
    elif choice == "email_draft":
        url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(
            {"view": "cm", "fs": "1", "su": res.get("email_subject", ""), "body": res.get("email_body", "")})
        st.link_button(t("gmail"), url, type="primary", width="stretch")
        open_once(view, "email_draft", url)
        st.caption(t("auto_opened"))
    elif action and choice == action[0]:
        kind, label, url = action
        st.markdown(f'<div class="card accent"><h4>{t("action_title")}</h4>'
                    f'<div class="explain-box"><b>{html.escape(label)}</b></div></div>', unsafe_allow_html=True)
        st.link_button(labels[kind], url, type="primary", width="stretch")
        if kind != "call":
            open_once(view, kind, url)
            st.caption(t("auto_opened"))


def render_file(view):
    ext, data, text = view["ext"], view["data"], view["text"]
    col1, col2 = st.columns([1, 1.25], gap="large")
    with col1:
        st.markdown(f"<h4>{t('preview')}</h4>", unsafe_allow_html=True)
        if ext in IMAGE_EXT:
            st.image(Image.open(io.BytesIO(data)), width="stretch")
        elif ext == "pdf":
            if not PdfReader:
                st.warning(t("pdf_missing"))
            st.caption(t("pdf_pages", n=view["pages"]))
            st.text(text[:1500] or "—")
        else:
            st.code(text[:4000], language="text")
    with col2:
        res = view["result"]
        st.markdown(f'<span class="pill soft">{t("detected")}: {t("ctx_" + view["ctx"])}</span>', unsafe_allow_html=True)
        st.markdown(f"<h3>{html.escape(res.get('title', ''))}</h3>", unsafe_allow_html=True)
        render_file_options(view)


# ---------------------------------------------------------------------------
# The agent: one input, the role is chosen from what the user sends
# ---------------------------------------------------------------------------

def reset_view_state():
    for k in list(st.session_state):
        if k.startswith(("tutor_", "file_")):
            del st.session_state[k]


def answer(question, view=None):
    """Answer free text, using the current code / file as context when there is one."""
    images, context = None, ""
    if view and view["role"] == "tutor":
        context = f"Context — the student's code:\n```\n{view['code']}\n```"
    elif view and view["role"] == "file":
        images = image_parts(view["ext"], view["data"])
        context = f"Context — file '{view['name']}':\n{view['text'][:8000]}" if view["text"] else \
            f"Context — file '{view['name']}' (image attached)."
    prompt = fill(ANSWER_PROMPT, LANG=LANG_NAMES[lang()], LEVEL=level_name(), REQUEST=question, CONTEXT=context)
    return run_text(prompt, images, DEMO_TXT[lang()]["answer"])


class StoredFile:
    """An uploaded file kept in memory so the result can be re-generated (e.g. after a language change)."""

    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


ROUTER_PROMPT = """Decide how an assistant app should handle a user message.
What is on screen now: <<SCREEN>>
User message: <<MESSAGE>>
Reply with exactly one word:
followup — the message asks about, or asks to change/translate/summarize, what is on screen now
general  — the message is a new, unrelated question"""


IMAGE_WORDS = ["صورة", "ارسم", "image", "picture", "draw", "illustration", "resim", "görsel", "çiz"]
DIAGRAM_WORDS = ["مخطط", "رسم بياني", "شجرة", "diagram", "flowchart", "chart", "tree", "şema", "diyagram", "akış"]


def route_text(text, current):
    """Pick the role for a text-only message."""
    if looks_like_code(text):
        return "tutor"
    low = text.lower()
    if current and current["role"] == "tutor":
        if any(w in low for w in DIAGRAM_WORDS):  # first: "draw a diagram" is a diagram, not a picture
            return "diagram"
        if any(w in low for w in IMAGE_WORDS):
            return "image"
    if any(w in text.lower() for w in FINANCE_WORDS) and len(text) > 25:
        return "file"  # pasted bill / document text -> simplify it and offer actions
    if EMAIL_RE.search(text) and any(w in text.lower() for w in ACTION_WORDS):
        return "task"
    if not current or current["role"] == "answer":
        return "general"
    if is_demo():
        return "followup"
    screen = "the explanation of this code:\n" + current["code"][:1500] if current["role"] == "tutor" else \
        f"the summary of the file '{current['name']}'"
    try:
        word = call_llm(fill(ROUTER_PROMPT, SCREEN=screen, MESSAGE=text)).strip().lower()
        return "general" if word.startswith("general") else "followup"
    except Exception:
        return "followup"


def build_view(text, files, role=None):
    """Run the chosen role and return the new board (view)."""
    if files:
        f = files[0]
        ext, data, ftext, pages = read_file(f)
        ctx = detect_context(f.name, ext, ftext)
        sent = {"file": f.name, "text": text}
        if ctx == "code":
            with st.spinner(t("thinking")):
                view = {"role": "tutor", "code": ftext, "result": explain_code(ftext), "sent": sent}
        else:
            with st.spinner(t("analyzing")):
                view = {"role": "file", "name": f.name, "ext": ext, "data": data, "text": ftext,
                        "pages": pages, "ctx": ctx, "result": analyze_file(f.name, ext, data, ftext, ctx),
                        "sent": sent}
    elif role == "tutor":
        code = strip_fences(text)
        with st.spinner(t("thinking")):
            view = {"role": "tutor", "code": code, "result": explain_code(code), "sent": {"text": code}}
    elif role in ("file", "task"):
        ctx = "invoice" if role == "file" else "task"
        with st.spinner(t("analyzing")):
            view = {"role": "file", "name": "text", "ext": "txt", "data": text.encode(), "text": text,
                    "pages": 0, "ctx": ctx, "result": analyze_file("text", "txt", b"", text, ctx),
                    "sent": {"text": text}}
    else:
        with st.spinner(t("answering")):
            view = {"role": "answer", "sent": {"text": text}, "answer": answer(text)}
    view.update({"followups": [], "lang": lang(),
                 "inputs": {"text": text, "files": [StoredFile(f.name, f.getvalue()) for f in files], "role": role}})
    if files and text:  # a note sent with a file is a request about that file
        with st.spinner(t("answering")):
            view["followups"].append((text, answer(text, view)))
    return view


def add_history(view):
    seq = st.session_state.get("hist_seq", 0) + 1
    st.session_state["hist_seq"] = seq
    view["id"] = seq
    sent = view["sent"]
    label = sent.get("file") or (sent.get("text", "").strip().splitlines() or [""])[0]
    history = st.session_state.setdefault("history", [])
    history.insert(0, {"id": seq, "label": label[:38], "time": dt.datetime.now().strftime("%H:%M"), "view": view})
    del history[20:]


def restore_history(entry_id):
    for entry in st.session_state.get("history", []):
        if entry["id"] == entry_id:
            reset_view_state()
            st.session_state["view"] = entry["view"]


def new_request():
    reset_view_state()
    st.session_state.pop("view", None)


def render_history(box):
    icons = {"tutor": "🧑‍🏫", "file": "📂", "answer": "💬"}
    current = (st.session_state.get("view") or {}).get("id")
    with box:
        st.subheader(t("history"), anchor=False)
        st.button(t("new"), on_click=new_request, type="primary", width="stretch")
        history = st.session_state.get("history", [])
        if not history:
            st.caption(t("history_empty"))
        for entry in history:
            mark = "● " if entry["id"] == current else ""
            # FSI/PDI marks keep code labels (left-to-right) readable inside the Arabic sidebar
            st.button(f'{mark}{icons.get(entry["view"]["role"], "•")} \u2068{entry["label"]}\u2069', key=f'hist_{entry["id"]}',
                      help=entry["time"], width="stretch", on_click=restore_history, args=(entry["id"],))
        st.divider()


def handle(text, files):
    """Route the message to the right role. A new role replaces the whole board."""
    text = (text or "").strip()
    current = st.session_state.get("view")
    role = None if files else route_text(text, current)
    if role == "diagram":
        add_diagram(current)
        current["followups"].append((text, t("diagram_added")))
        return
    if role == "image":
        current["show_image"] = True
        make_image("tutor", text, text)
        if st.session_state.get("tutor_img"):
            current["followups"].append((text, t("image_added")))
        return
    if role == "followup":
        with st.spinner(t("answering")):
            current["followups"].append((text, answer(text, current)))
        return
    reset_view_state()
    st.session_state["view"] = build_view(text, files, role)
    add_history(st.session_state["view"])


def add_diagram(view):
    """The student asked for a diagram: show the tree and flowchart even for simple code."""
    view["force_diagram"] = True
    r, src = view["result"], [ln for ln in view["code"].splitlines() if ln.strip()]
    if not (r.get("trace") or {}).get("children"):
        r["trace"] = simple_trace(src)
    if not r.get("flowchart"):
        flow = ""
        if not is_demo():
            try:
                flow = call_llm(fill(FLOW_PROMPT, LANG=LANG_NAMES[lang()], CODE=view["code"]))
                flow = re.sub(r"```(?:mermaid)?", "", flow).strip()
            except Exception:
                flow = ""
        r["flowchart"] = flow or simple_flowchart(src)


def refresh_language():
    """Re-generate the board in the newly selected language so no old-language text stays on screen."""
    view = st.session_state.get("view")
    if not view or view.get("lang") == lang():
        return
    inputs = view["inputs"]
    questions = [q for q, _ in view["followups"]][1 if inputs["files"] and inputs["text"] else 0:]
    reset_view_state()
    new = build_view(inputs["text"], inputs["files"], inputs["role"])
    for q in questions:
        with st.spinner(t("answering")):
            new["followups"].append((q, answer(q, new)))
    new["id"] = view.get("id")
    for entry in st.session_state.get("history", []):
        if entry["id"] == new["id"]:
            entry["view"] = new
    st.session_state["view"] = new


def render_sent(sent):
    if sent.get("file"):
        body = f"📎 <code>{html.escape(sent['file'])}</code>"
        if sent.get("text"):
            body += f" — {html.escape(sent['text'][:120])}"
    else:
        first = sent.get("text", "").strip().splitlines()[0] if sent.get("text", "").strip() else ""
        body = f"<code>{html.escape(first[:80])}</code>" + (" …" if "\n" in sent.get("text", "").strip() else "")
    st.markdown(f'<div class="sent">{t("you_sent")}: {body}</div>', unsafe_allow_html=True)


def render_view(view):
    render_sent(view["sent"])
    st.markdown(f'<span class="pill">{t("role_" + view["role"])}</span>', unsafe_allow_html=True)
    st.write("")
    if view["role"] == "tutor":
        render_tutor(view, "tutor")
    elif view["role"] == "file":
        render_file(view)
    else:
        st.markdown(f'<div class="card primary">{html.escape(view["sent"]["text"])}</div>', unsafe_allow_html=True)
        st.markdown(view["answer"])

    for q, a in view.get("followups", []):
        st.divider()
        st.markdown(f'<div class="card"><b>💬</b> {html.escape(q)}</div>', unsafe_allow_html=True)
        st.markdown(a)
    if view["role"] != "answer":
        st.caption(t("followup_hint"))


def render_welcome():
    st.markdown(f'<div class="greeting">{t("welcome_title")}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header, sidebar & main
# ---------------------------------------------------------------------------

def header():
    """Top bar like other AI apps: the name on one side, a language dropdown on the other."""
    def apply_language():
        st.session_state["lang"] = st.session_state["lang_pick"]

    st.session_state.setdefault("lang", "ar")
    st.session_state.setdefault("lang_pick", st.session_state["lang"])
    title_col, lang_col = st.columns([4, 1.2], vertical_alignment="center")
    with title_col:
        st.markdown(f'<div class="hero"><div class="logo">{t("logo")}</div><h1>{t("title")}</h1></div>',
                    unsafe_allow_html=True)
    with lang_col:
        st.selectbox(t("language"), list(LANG_LABELS), format_func=lambda c: f"🌐 {LANG_LABELS[c]}",
                     key="lang_pick", on_change=apply_language, label_visibility="collapsed")


def sidebar():
    with st.sidebar.expander(t("settings"), expanded=False):
        providers = {"ollama": t("provider_ollama"), "groq": t("provider_groq"), "gemma": t("provider_gemma"),
                     "demo": t("provider_demo")}
        default_provider = ("ollama" if ollama_models() else "groq" if os.getenv("GROQ_API_KEY")
                            else "gemma" if os.getenv("GEMINI_API_KEY") else "demo")
        provider = st.radio(t("provider"), list(providers), format_func=providers.get,
                            index=list(providers).index(default_provider))

        api_key, model, vision_model = "", "", ""
        if provider == "ollama":
            model = st.text_input(t("model"), value=DEFAULT_OLLAMA_TEXT)
            vision_model = st.text_input(t("vision_model"), value=DEFAULT_OLLAMA_VISION)
        elif provider == "gemma":
            api_key = st.text_input(t("api_key"), value=os.getenv("GEMINI_API_KEY", ""), type="password",
                                    help="https://aistudio.google.com/apikey")
            model = st.text_input(t("model"), value=DEFAULT_GEMMA_MODEL)
        elif provider == "groq":
            api_key = st.text_input(t("api_key"), value=os.getenv("GROQ_API_KEY", ""), type="password",
                                    help="https://console.groq.com/keys")
            model = st.text_input(t("model"), value=DEFAULT_GROQ_MODEL)

        hf_token = st.text_input(t("hf_token"), value=os.getenv("HF_TOKEN", ""), type="password",
                                 help="https://huggingface.co/settings/tokens")
        level = st.radio(t("level"), ["beginner", "uni"],
                         format_func=lambda v: t("level_beginner") if v == "beginner" else t("level_uni"))
    with st.sidebar:
        st.caption(t("privacy"))
        st.caption("Gemma 4 · Ollama · FLUX.1-schnell · MIT License")

    st.session_state["settings"] = {"provider": provider, "api_key": api_key.strip(), "model": model.strip(),
                                    "vision_model": vision_model.strip(), "hf_token": hf_token.strip(), "level": level}


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧑‍🏫", layout="wide", initial_sidebar_state="expanded")
    header()
    inject_css(lang() == "ar")
    history_box = st.sidebar.container()  # filled after the message is handled, shown above the settings
    sidebar()
    # status lives in the sidebar so the page itself stays just the input
    with st.sidebar:
        if settings()["provider"] == "ollama":
            (st.warning if is_demo() else st.caption)(t("ollama_down") if is_demo() else t("local_notice"))
        elif is_demo():
            st.caption(t("demo_notice"))

    greeting = st.empty()  # filled after handling, so it disappears as soon as there is a result
    # The single smart input — inside a container so it sits inline under the header
    with st.container():
        msg = st.chat_input(t("input_placeholder"), accept_file=True, file_type=FILE_TYPES)
    focus_input()
    if msg is not None:
        handle(msg.text, list(msg.files or []))
    else:
        refresh_language()
    render_history(history_box)
    if not st.session_state.get("view"):
        with greeting:
            render_welcome()

    if st.session_state.get("view"):
        render_view(st.session_state["view"])


if __name__ == "__main__":
    main()
