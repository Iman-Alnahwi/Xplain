# 🧑‍🏫 Xplain

**Privacy-First Multi-File & Action Agent** — an open-source AI agent built on open-weight models (Gemma 4, FLUX.1-schnell).

وكيل ذكاء اصطناعي مفتوح المصدر يشرح الشيفرة البرمجية سطراً سطراً للمبتدئين، ويحوّل ملفاتك إلى إجراءات. العربية الفصحى أولاً، مع الإنجليزية والتركية.

## One input, the agent picks the role
There is a single input box (text + file attachment). The agent routes the message:
- **code** (pasted or a `.py/.js/.cpp/.json` file) → Code tutor
- **image / PDF / text file** → File actions
- **any other text** → an answer, about the code or file already on screen when there is one

Language (العربية / English / Türkçe) is switched from the top of the page.
Palette: Pantone 2965 C navy `#00263E`, 550 C powder blue `#94B2C4`, 19-1119 cocoa `#594536`, 11-0608 coconut milk `#F0EDE5`.

### 1. Code tutor (اشرح الكود)
- Paste code in **any language** → the tutor names the language, explains what the program is for, then goes line by line (current line highlighted) and shows step by step **how the output is produced**.
- **Visuals only when they help**: the model decides whether an interactive execution tree (every loop iteration is its own node with variable values and output) or a flowchart is useful. Simple code gets none — unless the student asks ("draw a diagram").
- **🎨 Image only when useful or asked** (FLUX.1-schnell): from the model's analogy ideas or the student's own description.
- Quiz, beginner glossary, links to official docs and StackOverflow.
- **History** in the sidebar: reopen any earlier request in this session, or start a new one.
- Ask a follow-up question about the code in the same box.

### 2. Files & actions (ملفات وإجراءات)
- Upload PNG / JPG / PDF / code files.
- Auto-detects **code**, **invoice / bill**, or **document**.
- **Bill** → just the amount to pay and the due date, plus actions: go to the payment page, add a calendar reminder, export CSV / Excel.
- **Document** → 3-point summary + pre-filled Gmail draft.
- **Code file** → goes straight to the tutor.
- Type a follow-up in the same box (translate, summarize in detail, …).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**No key needed:** the default model is **Gemma 4 (`gemma4:e4b`) running locally with [Ollama](https://ollama.com)** — code and files never leave the computer.

```bash
ollama serve            # or run tools/ollama/ollama serve
ollama pull gemma4:e4b  # 6.6 GB, text + images, real Arabic and Turkish
```

Groq or the Gemini API can be chosen in Settings instead (keys in `.env.example`). If no model is reachable the app runs in **demo mode** with sample answers, so it never crashes.

| Part | Open model / library |
|---|---|
| Explanations, bill reading (vision) | Gemma 4 `gemma4:e4b` locally via Ollama (default) · Gemini API · any open-weight model on Groq |
| On-demand images | FLUX.1-schnell (Apache-2.0) via Hugging Face |
| Flowcharts | Mermaid |
| UI | Streamlit |

## Deploy on DigitalOcean (live demo)
1. App Platform → **Create App** → GitHub → this repo. The `Dockerfile` is detected (port 8080).
2. Add an encrypted environment variable **`DO_MODEL_ACCESS_KEY`** (Control Panel → AI → Model access keys).
3. The app then uses **Gemma 4 31B (`gemma-4-31B-it`) on DigitalOcean Serverless Inference** — no GPU server to manage.

## Privacy
Files are processed in memory and never stored. They are sent only to the model provider you select. Payment links are taken from the bill itself — the app never invents one; if none is printed it opens a web search for the biller's official page.

## License
MIT
