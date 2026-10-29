# Dialect & Noise-Robust Telugu Civic Assistant

A Telugu civic assistant for government welfare information across regional dialects and noisy wording. The Streamlit dashboard shows the **full NLP pipeline** (normalization → intent → hybrid retrieval → extractive answer), not only a chat box.

## What is implemented

| Stage | Status | Code |
| --- | --- | --- |
| Telugu dialect / ASR-spelling normalization | Operational | `src/dialect_normalizer.py` |
| Intent classification (4 classes) plus RAG question-category detection | Operational if `models/intent_model.pkl` is present; retrieval separately recognizes purpose, deadlines, status, renewal, grievances, and helplines | `src/intent_api.py`, `src/query_analysis.py` |
| Hybrid retrieval (BM25 + char TF-IDF, optional multilingual dense/Chroma, RRF) | Operational over the legacy corpus and optional ingested corpus | `src/retriever.py` |
| Neural dense retrieval | Optional; off unless `TELUGU_CIVIC_DENSE=1` | `src/retriever.py` |
| Grounded answers | Safe extractive fallback; optional Ollama/OpenAI-compatible generation | `src/answer_generator.py`, `src/llm_generator.py` |
| Live ASR / Whisper | **Not integrated** (placeholder) | `src/asr.py` |
| Simulated ASR spelling noise (experiments) | Operational | `src/asr_noise.py` |
| Streamlit dashboard | Operational | `app.py`, `ui/` |

The checked-in knowledge base holds **six schemes across two states** (42 passages): Telangana Rythu Bharosa (12), Andhra Pradesh Dr. NTR Vaidya Seva / Aarogyasri (6), Andhra Pradesh Talliki Vandanam / Ammavodi (4), and Telangana Aasara Pension (4). All are preserved when ingestion runs.

Each scheme's `data/schemes_data.json` entries carry a `state` field (Telangana or Andhra Pradesh) and an opaque `scheme_identifier` (`RYTHU_BHAROSA_TG`, `NTR_VAIDYA_SEVA_AP`, `TALLIKI_VANDANAM_AP`, `RYTHU_BANDHU_TG`). The RAG layer keys scheme filters on `scheme_identifier` when present, so a query for one state never mixes in that state's passages. Telugu questions on Andhra Pradesh schemes (Aarogyasri, Talliki Vandanam) and Telangana schemes (Rythu Bharosa) now return evidence from the correct state, not both.

## Knowledge ingestion

`data/knowledge_sources.json` contains configurable official URLs for Central Government, Andhra Pradesh, and Telangana portals. Fetching is deliberately not automatic at app startup.
Run this from the repository root:

```powershell
python src/ingest.py
```

For bounded discovery, ingestion, indexing, and coverage evaluation in one
repeatable command:

```powershell
python src\research.py --iterations 1 --max-depth 1 --pages-per-domain 10 --timeout 20
```

The crawler follows only same-domain links from explicitly trusted configured
sources, checks `robots.txt`, skips login/account paths, caps response size and
requests, uses a local fetch cache, and never disables TLS verification. It
writes machine-readable `outputs/official_research_report.json` and a concise
human-readable `outputs/official_research_report.md`. Use `--refresh` only when
you intentionally want to refetch cached documents.

The command keeps the legacy JSON records, fetches HTML and (when `pypdf` is installed)
PDF text, cleans and chunks fetched pages, deduplicates by document ID, and writes
`data/knowledge_corpus.jsonl`. It prints fetched sources, documents, chunks, failures,
and indexed-record counts. A failed or unreachable official page is reported and does not
delete existing knowledge. Verify a source and its current content before using it for
high-stakes decisions.

The corpus is audited by question category, not by record count:

```powershell
python src/evaluate_coverage.py
```

This writes `outputs/knowledge_coverage.json`, including evidence categories,
official-source counts, missing categories, and checklist questions with the
retrieved evidence document IDs. Discovery portals such as myScheme are retained
in the source registry but are not indexed as evidence for a particular scheme.
Missing or outdated categories remain unanswered and are linked only as sources
to check; they are not inferred from nearby passages.

Knowledge reports distinguish `legacy_unverified`,
`official_accessed_currentness_unverified`, and `verified_current`. The current
corpus does not mark any passage `verified_current`; a successful request to an
official domain is not treated as proof that every claim is current. The coverage
report also classifies missing-knowledge and retrieval-failure cases separately
and writes aggregate evidence coverage, verified coverage, retrieval recall,
grounded-answer rate, abstention rate, unsupported-answer rate, and isolation
metrics to `outputs/knowledge_coverage.json`.

At query time, the RAG layer resolves Telugu and English scheme aliases and applies
scheme and question-category filters before answer generation. Each scheme's
`state` and `scheme_identifier` let the filters keep Andhra Pradesh and Telangana
passages separate: querying an Andhra Pradesh scheme never returns Telangana
passages of the same scheme, and vice versa. The teammate-owned four-class intent
prediction remains visible for diagnostics, but it cannot turn a purpose question
into an eligibility answer. Follow-up questions reuse the last explicit scheme for
the current Streamlit session only. If no passage in the requested category exists,
the extractive fallback abstains and shows scheme-specific official source hints
instead of combining passages from another scheme.

The Ask page includes a persistent `తెలుగు` / `English` answer-language selector.
The selected language is passed into retrieval, fallback rendering, follow-ups, and
no-evidence messages. English official passages are rendered in English; the
provider-free Telugu fallback uses the supported bilingual phrase adapter where
available. For unrestricted translation of newly discovered English passages,
configure and test an LLM provider; the application does not invent translations
when a provider is unavailable.

The research report separates newly changed documents, unchanged cached
documents, duplicate documents skipped, fetch failures, and extracted chunks
before deduplication. Retrieval coverage is an automated relevance check, not a
human verification of the factual correctness of every source claim.

## Optional grounded LLM

The default app needs no paid API and uses the extractive grounded fallback. For local
Ollama:

```powershell
$env:TELUGU_CIVIC_LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.2"
```

For an OpenAI-compatible provider, set `TELUGU_CIVIC_LLM_PROVIDER=openai`,
`OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL`/`OPENAI_MODEL`. Credentials are
read only from the environment. A provider is not called unless evidence is retrieved;
failed or uncited responses fall back to the extractive answer and are labelled as such.
The model is instructed to use only retrieved passages, but users should still check the
linked official source.

## Project structure

- `app.py` — Streamlit entry point (sidebar: Dashboard, Ask, Schemes, History, Evaluation, About)
- `ui/` — presentation only
- `src/` — NLP pipeline (`pipeline.py` is what the UI calls)
- `data/` — schemes (6 schemes / 42 passages), intent dataset, dialect lexicon, civic vocabulary, held-out queries
- `models/` — `intent_model.pkl` + `model_card.json`
- `outputs/plots/` — Experiment 2 and 3 figures (after `evaluate_nlp.py`)
- `tests/` — `python tests/test_nlp.py`

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

If `models/intent_model.pkl` is missing, or sklearn versions differ from `model_card.json` (unpickling will fail or warn):

```bash
python src/generate_intent_data.py
python src/train_intent_model.py
```

Windows consoles: prefix Python commands with `$env:PYTHONIOENCODING="utf-8";` if Telugu prints as `?`.

## Run the dashboard

From the repository root:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Use **Ask the Assistant** for the end-to-end text flow.

Query history is **session-only** (browser session). It is not written to disk.

## NLP pipeline (for the UI / Member 3)

```python
from pipeline import CivicAssistant
bot = CivicAssistant()
bot.run("రైతు బంధు పైసలు ఎప్పుడొస్తయి?")
```

Lower-level intent API:

```python
from intent_api import IntentService
svc = IntentService()
svc.predict("రైతు బంధు పైసలు ఎప్పుడొస్తయి?")
# normalized, intent, confidence, in_scope, edits
```

`edits` lists every rewrite with the rule that fired — shown on the Ask page.

## Optional components

- **Neural dense retrieval:** `TELUGU_CIVIC_DENSE=1` (downloads a multilingual MiniLM the first time).
- **Live ASR:** set `TELUGU_CIVIC_WHISPER_MODEL` to a local openai-whisper model name/path **and** install `openai-whisper`. Without this, audio upload reports that ASR is unavailable. `asr_noise.py` is **not** speech recognition.

## Evaluation

```bash
python src/evaluate_nlp.py
python tests/test_nlp.py
```

The Evaluation page shows:

- Metrics stored in `models/model_card.json`
- Live scores of the loaded intent model on `data/heldout_intent_queries.csv` (standard vs dialect)
- Experiment plots if they exist under `outputs/plots/`
- **Not evaluated yet** for ASR WER/CER, retrieval Recall@3/MRR, and answer-quality annotation (those files are not in the repo)

No dashboard number is invented to look complete.

## Knowledge base rule

Answers are copied from retrieved passages. The UI does not add eligibility rules,
benefit amounts, or deadlines that are not in the preserved legacy records or the
verified-source `data/knowledge_corpus.jsonl`. Discovery pages are not automatically
treated as useful evidence: they must pass the crawler's relevance and deduplication
filters. Run the coverage audit before claiming that a scheme is comprehensively
supported.

## Repeatable commands

```powershell
# normal incremental bounded discovery + ingestion + coverage report
python src\research.py --iterations 1 --max-depth 1 --pages-per-domain 10

# repeat with a full refetch of cached pages (still bounded)
python src\research.py --iterations 1 --max-depth 1 --pages-per-domain 10 --refresh

# coverage only
python src\evaluate_coverage.py

# tests
python -m pytest tests\test_nlp.py tests\test_rag.py -q

# dashboard
streamlit run app.py
```

## Fixed ASR experiment (added workflow)

This experiment uses the OpenSLR SLR66 Telugu speech corpus as a **general Telugu ASR baseline**. It is not used to train the civic intent classifier and it is not a regional-dialect benchmark. SLR66 contains audio recordings paired with reference transcripts. Dataset page and license: https://www.openslr.org/66/ (CC BY-SA 4.0; follow the license and cite the corpus in reports).

### 1. Download one subset only

To limit time and disk usage, download the **female audio archive** `te_in_female.zip` and its matching index `line_index_female.tsv` from https://www.openslr.org/66/. Extract the ZIP into a folder such as `data/slr66/` and put the matching TSV in that folder too. Do not mix the female TSV with the male archive. The script searches recursively, so the WAV files can remain in their extracted subfolder.

### 2. Install and verify local ASR dependencies

Use the same Python environment as Streamlit:

```powershell
python -m pip install -r requirements.txt
python -c "import whisper; print('Whisper import successful')"
ffmpeg -version
```

If `ffmpeg -version` fails on Windows, install FFmpeg and make sure its `bin` folder is on PATH, then reopen the terminal. The first Whisper run downloads model weights. Start with `base`; use `small` only if your computer handles it comfortably. The model runs locally after the weights are downloaded.

### 3. Create a fixed, reproducible 20-clip manifest

From the repository root:

```powershell
python scripts/prepare_slr66_manifest.py data/slr66 --limit 20 --seed 42
```

This writes `data/asr_eval_manifest.csv`. Keep it unchanged for every comparison. If it reports missing audio/index matches, check that the extracted WAV archive and TSV are the matching pair.

### 4. Run the one predefined ASR/noise experiment

```powershell
python scripts/evaluate_asr.py --model base --limit 20 --snr-db 10
```

It evaluates the same clips under three conditions: clean audio, synthetic additive white noise at 10 dB SNR, and that noisy audio after `noisereduce` processing. It writes per-clip transcripts and WER/CER to `outputs/asr_results.csv`, and average metrics to `outputs/asr_summary.json`. Lower WER/CER is better. Do not assume noise reduction will help; report what the measurements show. This is a controlled synthetic-noise experiment, not real-world background-noise evaluation.

### 5. Connect live voice input in the app

In PowerShell, from the project root:

```powershell
$env:TELUGU_CIVIC_WHISPER_MODEL="base"
streamlit run app.py
```

Then use **Ask the Assistant → Voice input** and upload/record a short Telugu question. The first model load may take time. Keep the model setting consistent with the evaluation when reporting results. The ASR wrapper uses `fp16=False` for compatibility with CPU inference.

### 6. Test the actual civic use case separately

`data/civic_questions_to_record.csv` gives eight fixed welfare-scheme prompts spanning the four supported intent classes. Ask willing speakers to record each prompt as WAV files. Preserve the exact words spoken in the `reference_text` column; do not silently correct the transcript to what the speaker intended. Fill `data/civic_audio_eval_template.csv` with each audio path, exact reference transcript, expected intent, and expected scheme, then run:

```powershell
python scripts/evaluate_civic_audio.py --model base
```

Results are saved to `outputs/civic_audio_results.csv` and `outputs/civic_audio_summary.json`. This measures the end-to-end behaviour on your small recorded civic set: ASR WER/CER, intent accuracy, whether a passage for the expected scheme appears in the top five, and whether an answer is grounded. It is a small project evaluation set, not a statistically conclusive dialect study.

### What each experiment establishes

- **SLR66 clean audio:** a repeatable baseline for a pretrained Telugu recognizer using audio with known reference transcripts.
- **Synthetic 10 dB noise:** whether controlled added noise increases transcription errors.
- **Noise reduction:** whether the selected denoising method changes WER/CER on the same noisy clips.
- **Recorded civic questions:** whether voice transcripts continue through normalization, intent prediction, retrieval, and grounded answering in the actual application.

Do not describe this as Whisper fine-tuning: the project uses a pretrained model for inference. Do not claim SLR66 proves Telangana/Rayalaseema/other regional dialect robustness; the project needs genuinely spoken regional examples to make that claim.
