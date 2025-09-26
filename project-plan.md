# Ultimate Japanese Quiz Solver - Project Plan

## High-level approach (one-sentence)
Use a robust OCR pipeline → deterministic rule-checks (dates, readings, katakana) → morphological analysis → retrieval + LLM (structured JSON) → ensemble decision + GUI → continuous human-in-the-loop labeling and (optionally) targeted fine-tuning or RAG.

## Phase 0 — Project organization (COMPLETED)
Keep your existing repo. Create branches:

- main — stable GUI + core features ✅
- feature/ocr-improve — OCR & preprocessing 
- feature/rules-engine — rule modules (dates, katakana)
- feature/morphology — MeCab/Fugashi integration
- feature/rag-llm — retrieval + LLM / fine-tuning experiments

Project structure created:
- ocr/, rules/, morph/, models/, tests/, data/, logs/, errors/, notebooks/ ✅

## Phase 1 — Quick wins (2–6 hours) — immediate impact
**Objective**: eliminate the common OCR/reading errors (dates, small katakana issues), and make LLM deterministic.

### Tasks (exact):

1. **OCR preprocessing** — add `ocr/ocr_preprocess.py` (use OpenCV + PIL). Upscale, denoise (bilateral), adaptive threshold, morphological open. (This reduces Tesseract misreads.)

2. **Multi-PSM OCR** — add `ocr/ocr_multi_psm.py` that tries PSMs [6,3,4,11,12] and picks the result with highest Japanese-character ratio.

3. **Date/Reading rule engine** — add `rules/rules_date.py` (the むいか → 六日 mapping). Run this BEFORE calling the LLM. If rule finds exact match in options, use it as final answer.

4. **Katakana fuzzy match** — add `rules/fuzzy_kata.py` (difflib or Levenshtein). If OCR gives noisy katakana, match with highest similarity to options; if score > 0.75, prefer it.

5. **Strict LLM JSON response** — update LLM prompt to force JSON (keys: choice_index, choice_text, confidence, reason). Use temperature=0.0. Parse the JSON. If parse fails, retry once.

6. **GUI change** — display columns: OCR_text | Rule_choice | LLM_choice (conf) | Final_choice | tiny OCR image. Add an Accept button so you can label mistakes quickly.

**Expected outcome**: many prior errors (dates, katakana, simple misreads) will be fixed without ML.

## Phase 2 — Medium improvements (1–3 days)
**Objective**: add linguistic analysis and ensemble logic so the system understands sentence structure and reduces semantic errors (e.g., 母 vs 毌).

### Tasks:

1. **Morphological parsing** — add `morph/morph.py` using fugashi[unidic-lite] or MeCab+unidic. Extract readings for nouns/particles and token POS.

2. **Vocabulary & Kanji DB** — build `data/kanji_vocab.json` and a mapping of common JLPT words → readings → kanji.

3. **Ensemble decision logic** — add `decision/final_decision.py`:
   - Priority: Rule (exact date/reading) > Morph-match > Katakana fuzzy > LLM (unless LLM_confidence >> 0.9 and no conflicts).

4. **Confusion handling** — if two candidates tie, display both and require manual accept.

5. **Logging + error capture** — save all processing steps to sqlite for analysis.

**Expected outcome**: semantic errors drop significantly.

## Phase 3 — RAG + Fine-tuning experiments (1–3 weeks, optional but recommended)
**Objective**: when rules/heuristics are insufficient, add retrieval and targeted fine-tuning to push accuracy >95%.

### Two complementary approaches:

**A — RAG (Retrieval-Augmented Generation) / Vector DB**
- Build knowledge base of JLPT Q&A pairs, grammar notes, common readings
- Use embeddings (OpenAI embeddings or sentence-transformers) to index into FAISS
- Inject top-5 relevant passages into LLM prompt before asking

**B — Fine-tune a Japanese QA model**
- Collect labeled data from logged quiz_history.db + JLPT sample papers
- Fine-tune using LoRA on Japanese-capable base model
- Use validation split and early stopping

## Phase 4 — Testing, metrics, QA (3–5 days)
**Objective**: prove >95% accuracy on representative test sets.

### Tasks:
1. **Create evaluation dataset** — 2,000 labeled samples covering all question types
2. **Write tests/accuracy_test.py** — run whole pipeline and compute per-type accuracy
3. **Unit tests** — Date mapping, katakana fuzzy, morphological mapping, LLM JSON parse
4. **Acceptance criteria**: test-set accuracy >= 95% and per-type accuracy >= 92%
5. **CI setup** — GitHub Actions for automated testing

## Phase 5 — Production hardening & UX (1–3 days)
**Objective**: reliable, fast, and debuggable GUI.

### Tasks:
1. **Manual override & labeling UI** — Accept/Correct/Edit buttons for human feedback
2. **Rate & throttle** — limit scanning frequency, stable detection
3. **Config panel** — thresholds for OCR_confidence, LLM_conf, fuzzy_threshold
4. **Persistence & export** — export to CSV/Anki format
5. **Privacy options** — local-only vs remote API modes

## Exact file names to add/modify:
- ocr/ocr_preprocess.py
- ocr/ocr_multi_psm.py  
- rules/rules_date.py
- rules/fuzzy_kata.py
- morph/morph.py
- decision/final_decision.py
- models/rag_index.py (RAG retriever)
- models/train_classifier.py (optional fine-tune)
- ui/result_panel.py (show OCR snippet + columns)
- tests/accuracy_test.py
- notebooks/analysis.ipynb (error analysis)

## LLM prompt (copy/paste; enforce JSON)

**System prompt:**
```
You are a deterministic JLPT multiple-choice assistant. ALWAYS output valid JSON only with keys:
{ "choice_index": <1..4>, "choice_text": "<kanji/kana/katakana>", "confidence": <0.0-1.0>, "reason": "<one-line reason max 30 words>" }

Rules:
- Use only the provided options.
- Temperature: 0.0. Short answers only.
```

**User input:**
```
Question: "<full Japanese sentence>"
Options: ["opt1", "opt2", "opt3", "opt4"]  
Context: "<retrieved passages if any>"
```

Parse and validate JSON. If parse fails, retry once with "Respond only with valid JSON."

## Metrics & monitoring (what to log)
- Overall accuracy, per-question-type accuracy (date/katakana/kanji/reading/listening)
- OCR_confidence distribution
- LLM_confidence distribution  
- Daily new labeled corrections count
- Confusion matrices & top-20 error examples

## Hardware & cost guidance
- **Development**: 16–32GB GPU (RTX 3080/4090) for local fine-tuning. LoRA reduces needs.
- **RAG/embeddings**: CPU is ok; faster with small GPU
- **Production**: small ongoing API cost for cloud LLM, or compute & disk for local models

## Target Performance Metrics
- **Overall Accuracy**: >95%
- **OCR Accuracy**: >92%  
- **Rule Engine Coverage**: ~30% of questions
- **LLM Response Time**: <3 seconds
- **End-to-End Processing**: <5 seconds

## Current Status
- ✅ Phase 0: Project organization complete
- ⏳ Phase 1: Starting OCR improvements and rule engines
- 🔄 Phase 2: Pending - morphological analysis
- 🔄 Phase 3: Pending - RAG/fine-tuning experiments  
- 🔄 Phase 4: Pending - testing and evaluation
- 🔄 Phase 5: Pending - production hardening

## Next Steps
1. Implement OCR preprocessing pipeline
2. Add multi-PSM testing
3. Create date/reading rule mappings
4. Implement katakana fuzzy matching
5. Update LLM prompts for structured JSON
6. Enhance GUI with new result display panels
