# Member 3 Implementation & Integration Audit
**Project:** Dialect and Noise-Robust Telugu Civic Assistant  
**Role:** Member 3 — Search, RAG & UI Integration Lead (`S20240030382`)  
**Audit Date:** October 2026  
**Scope:** `src/hybrid_retriever.py`, `src/simplifier.py`, `src/evaluate_rag.py`, `app.py`, `data/demo_cache.json`, and integration interfaces with Members 1 & 2.

---

## 1. Executive Summary & Team Boundaries

The "Dialect and Noise-Robust Telugu Civic Assistant" is structured as a collaborative 3-member NLP system:
* **Member 1 (`S20240010065`):** Audio acquisition, noise augmentation, spectral denoising, and Telugu Whisper ASR (owns WER/CER evaluation).
* **Member 2 (`S20240010125`):** Telugu dialect normalization (`dialect_normalizer.py`), civic lexicons, knowledge-base curation, and intent classification (`intent_api.py`, `intent_model.pkl`).
* **Member 3 (`S20240030382` — My Role):** Hybrid retrieval (BM25 + Dense ChromaDB), Reciprocal Rank Fusion (RRF), grounded Telugu answer generation, evidence validation, conversational follow-up handling, Streamlit UI integration, and RAG evaluation (Recall@K, MRR, factuality).

### Boundary Verification:
- **Member 1 files:** `src/asr_noise.py`, acoustic components. (Untouched and preserved).
- **Member 2 files:** `src/dialect_normalizer.py`, `src/intent_api.py`, `src/train_intent_model.py`, `models/intent_model.pkl`, `models/model_card.json`, `data/intent_dataset.csv`, `data/heldout_intent_queries.csv`. (Untouched and preserved).
- **Member 3 files:** `src/hybrid_retriever.py`, `src/simplifier.py`, `src/evaluate_rag.py`, `app.py`, `data/demo_cache.json`, `data/schemes_data.json` (as shared KB consumer).

---

## 2. Audit of Existing Member 3 Components

### A. `src/hybrid_retriever.py` (Hybrid Search Engine)
* **What works:**
  - Implements `BM25Okapi` sparse retrieval and `ChromaDB` dense vector retrieval using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  - Implements Reciprocal Rank Fusion ($k=60$) combining ranks from BM25 and Dense search.
  - Custom suffix stripper (`stem_telugu_token`) strips nominal case markers (`-లతో, -లకు, -తో, -కు, -లో, -ల`).
* **Deficiencies & Critical Issues Identified:**
  1. **Empty Query Crash:** Passing an empty query or whitespace (`""`) raises indexing/tokenization exceptions rather than returning a clean empty result `[]`.
  2. **Inflexible Suffix Normalization:** Suffix stripping was hardcoded inside the tokenizer without an ablation toggle (`use_suffix_stripping=True/False`). This made it impossible to formally prove or disprove the empirical benefit of suffix stripping on retrieval metrics.
  3. **Synthetic Corpus Augmentation:** In BM25 indexing, `CATEGORY_TE_MAP` was prepended to document text with extensive English and Telugu synonyms, artificially boosting lexical match on synthetic terms rather than matching authentic official document text.
  4. **Score Interpretation:** Raw BM25 scores and cosine distances were not explicitly surfaced alongside ranks in the metadata dictionary.
  5. **Resource Overhead:** Without a caching layer, re-instantiating `TeluguHybridRetriever` repeatedly reloads the 470MB transformer model and recomputes embeddings in memory.

### B. `src/simplifier.py` (Answer Generation & Guardrails)
* **What works:**
  - Formats retrieved text into clear, readable Telugu bullet points.
  - Implements an anti-hallucination fallback phrase (`క్షమించండి, అందించిన అధికారిక సమాచారంలో దీనికి సంబంధించిన వివరాలు లేవు.`) when context is empty or out-of-scope.
  - Guided fallback routing provides official state helplines (`1907`, `104`, `1902`, `1800-200-1001`) and portals for unrecognized scheme queries.
* **Deficiencies & Critical Issues Identified:**
  1. **Cache Bypass Anti-Pattern:** The offline demo cache (`data/demo_cache.json`) was checked **before** retrieval. If a query matched a cache key, the system returned the hardcoded string immediately, completely bypassing retrieval, RRF ranking, and grounding validation!
  2. **Coarse Sentence Extraction (No Sub-Question Focusing):** `simplify_context` simply splits the entire retrieved passage into sentences and prepends bullet points. If a user asks a specific question ("Do tenant farmers get this benefit?"), the system dumps all 3-4 sentences of the eligibility document without highlighting the exact answer ("Tenant farmers are not eligible").
  3. **Lack of Evidence Validation Module:** There was no dedicated validation layer (`answer_validator.py`) verifying whether the retrieved passage actually entails the query or whether the system should abstain.
  4. **Zero Conversational Context:** The module was strictly single-turn and stateless. Follow-up queries (e.g., "Who is eligible for it?", "What documents are required?") fail because the scheme name is absent from the follow-up text.

### C. `src/evaluate_rag.py` (Evaluation & Benchmarks)
* **What works:**
  - Evaluates Recall@1, Recall@2, Recall@3, and MRR.
  - Evaluates Factuality & Hallucination rejection rate across 35 queries.
* **Deficiencies & Critical Issues Identified:**
  1. **Heuristic Ground-Truth Mapping:** The benchmark previously reused Member 2's `heldout_intent_queries.csv` and synthesized ground truth doc IDs using a string-matching heuristic (`get_ground_truth_doc_id`), rather than evaluating against a dedicated, curated civic evaluation benchmark with human-verified relevant passage IDs.
  2. **Missing Ablation Suite:** Did not evaluate retrieval performance *with vs. without* suffix stripping.
  3. **No Conversational Evaluation:** Did not test conversational follow-up resolution, anaphora resolution, or scheme-switching dialogue turns.
  4. **Lack of `data/evaluation_queries.json`:** The repository lacked a standardized, self-contained evaluation benchmark file containing test IDs, queries, target schemes, passage IDs, intent labels, and abstention flags.

### D. `app.py` (Streamlit UI & Integration)
* **What works:**
  - Dual-mode input (Text typing + Audio .wav upload).
  - Pipeline inspection trace: Normalization -> Intent Classification -> Hybrid RRF -> Grounded Answer.
  - Audio waveform visualization (Noisy vs Clean Spectral Denoising).
  - Telugu TTS voice output via `gTTS`.
* **Deficiencies & Critical Issues Identified:**
  1. **Stateless Single-Turn UI:** Only supported a single question input. There was no interactive chat interface (`st.chat_message`) with session history.
  2. **Follow-Up Failure:** When users attempted to ask follow-up questions ("What about documents?"), the model lost track of the previous scheme.
  3. **Rerun Inefficiencies:** Retrying or clicking tabs caused redundant model invocations when not cached properly via `st.cache_resource`.

### E. `data/demo_cache.json` (Offline Demo Fixture)
* **Deficiencies:**
  - Served as an opaque shortcut that simulated pipeline execution rather than a legitimate caching mechanism.
  - Must be restructured so that caching is strictly reserved for expensive resources (embedding models, ChromaDB index, KB data), and any answer cache must validate against the real knowledge base and evidence grounding.

---

## 3. Remediation Roadmap for Member 3

To achieve the Definition of Done while respecting strict team boundaries:

| Phase | Milestone | Target File(s) | Action |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Hybrid Retrieval & Metadata** | `src/hybrid_retriever.py` | Add ablation toggle for suffix stripping, graceful empty query handling, strict passage metadata retention, and resource caching. |
| **Phase 2** | **Answer Generation & Validation** | `src/simplifier.py`, `src/answer_validator.py` | Implement direct sentence-level salience targeting, evidence entailment verification, and strict abstention guardrails. |
| **Phase 3** | **Conversation Manager** | `src/conversation_manager.py` | Implement multi-turn session tracking, anaphora/ellipsis resolution ("దానికి", "ఆ పథకం"), ambiguous reference detection, and context clearing. |
| **Phase 4** | **Benchmark Dataset & Evaluation** | `data/evaluation_queries.json`, `src/evaluate_rag.py` | Create 50+ curated evaluation items (standalone, follow-ups, out-of-scope, trick queries). Run full ablations (BM25 vs Dense vs RRF, with/without suffix stripping). |
| **Phase 5** | **Streamlit Conversational UI** | `app.py` | Add interactive conversational chat tab (`st.chat_message`) with follow-up chips alongside the technical pipeline inspector. |
| **Phase 6** | **Documentation Suite** | `docs/MEMBER3_*.md` | Produce `MEMBER3_ARCHITECTURE.md`, `MEMBER3_EVALUATION.md`, and `MEMBER3_VIVA.md`. |
