# Dialect and Noise-Robust Telugu Civic Assistant: Workflow & Engineering Rationale
## Comprehensive Team Pipeline Guide (What Each Member Does & Why)

**Context:** 3rd Year B.Tech CSE NLP Course Project  
**Team Roll Numbers:** S20240010065 (Member 1), S20240010125 (Member 2), S20240030382 (Member 3)  
**Evaluation:** 20 Marks per student  
**Target Schemes:** Rythu Bharosa/Bandhu, Aarogyasri, Amma Vodi, Aasara Pensions  

---

## 1. End-to-End Architectural Flow: How the Modules Interlock

```
[ Citizen Query ]
   │
   ├─► Path A (Spoken Audio .wav) ──► [ Member 1: Audio & ASR Pipeline ]
   │                                     ├─► Silence-Trimmed Energy Calculation
   │                                     ├─► Multi-Level Noise Augmentation (Clean, 15, 10, 5 dB)
   │                                     ├─► Spectral Denoising (Stationary vs Non-Stationary)
   │                                     └─► Whisper ASR (condition_on_previous_text=False)
   │                                                    │
   │                                                    ▼ (Raw ASR Telugu Text)
   └─► Path B (Typed Telugu Text) ──────────────────────┼───────────────────────┐
                                                        ▼                       │
                                      [ Member 2: Dialect NLP & Normalizer ]    │
                                         ├─► Unicode NFC & Transliteration      │
                                         ├─► Phonetic Folding (Aspirate Collapse)
                                         ├─► Regional Slang Lexicon Replacement │
                                         ├─► Levenshtein Typo Repair (>= 0.75)  │
                                         └─► Sub-word Char [2,4] Intent Model   │
                                                        │                       │
                                                        ▼ (Normalized Query)    │
                                      [ Member 3: Hybrid RAG & UI Integration ] │
                                         ├─► Telugu Suffix Stripper (-తో, -కు)   │
                                         ├─► BM25Okapi + ChromaDB (MiniLM Embed)│
                                         ├─► Reciprocal Rank Fusion (RRF, k=60) │
                                         ├─► Grounded Simplifier + Source URLs  │
                                         ├─► Offline Viva Demo Cache Safety Net │
                                         └─► Streamlit UI + Waveform + TTS ◄────┘
```

---

## 2. Member 1: Audio Procurement, Acoustic Denoising & ASR
**Lead:** Member 1 (`S20240010065`)  
**Core Role:** Acoustic Robustness, Real-World Noise Simulation, and Speech-to-Text Accuracy.

### Phase 1: Shared Benchmark Audio Collection (`data/audio/raw_recordings/`)
* **What to do:**
  1. Use the 60–80 welfare questions in `queries_master.csv`.
  2. Have 4–6 native Telugu speakers (teammates, friends, family) record these sentences on mobile phones in quiet environments.
  3. Export recordings as 16 kHz mono `.wav` files into `data/audio/raw_recordings/`.
* **The "WHY" (Technical Rationale):**
  - If Member 1 evaluates generic news/corpus audio (like Kathbath) while Members 2 and 3 evaluate welfare schemes, the pipeline is disconnected.
  - To scientifically demonstrate **Error Cascading** (how acoustic noise degrades downstream intent classification and vector retrieval), all 3 members must test on the **exact same query set**.

### Phase 2: Baseline Model Smoke Test (`src/asr_smoke_test.py`)
* **What to do:**
  1. Benchmark `openai/whisper-small` vs `openai/whisper-medium` on 5 sample Telugu clips.
  2. Decode with `condition_on_previous_text=False`.
  3. Compute baseline Word Error Rate (WER) and Character Error Rate (CER).
* **The "WHY" (Technical Rationale):**
  - In low-resource languages like Telugu, Whisper easily falls into infinite repetition loops (repeating a recognized token until max tokens are reached). Disabling previous-text conditioning stops this critical failure mode.
  - Smoke-testing prevents locking the team into an oversized model that exhausts laptop GPU/RAM during the live presentation.

### Phase 3: Controlled Noise Augmentation with Silence Trimming (`src/noise_augmenter.py`)
* **What to do:**
  1. Procure realistic noise audio (traffic, market babble, ceiling fan hum) into `data/audio/noise_sources/`.
  2. Apply silence trimming using `librosa.effects.trim(speech, top_db=25)` before calculating speech power.
  3. Mix noise into clean recordings across controlled Signal-to-Noise Ratios: 15 dB (mild), 10 dB (moderate), and 5 dB (heavy).
* **The "WHY" (Technical Rationale):**
  - Rural citizens record on noisy streets and in mandal offices.
  - If silence is not trimmed first, pauses at the beginning or end of speech artificially deflate average speech power calculation, causing noise to be scaled at completely incorrect volumes.
  - Controlled SNR testing proves scientifically how much ASR degrades under quantifiable noise conditions.

### Phase 4: Spectral Denoising Engine (`src/audio_cleaner.py`)
* **What to do:**
  1. Apply spectral gating using `noisereduce`.
  2. Implement both `stationary` mode (constant noise profile) and `non_stationary` mode (dynamic noise profile).
  3. Store restored audio in `data/audio/denoised/`.
* **The "WHY" (Technical Rationale):**
  - Constant electrical hums and AC/fans have static frequency profiles, while traffic and market chatter vary dynamically.
  - Testing both modes proves to the examiner an understanding of digital signal processing (DSP) and spectral subtraction techniques.

### Phase 5: Benchmark Evaluation (Experiment 1 — `src/evaluate_asr.py`)
* **What to do:**
  1. Transcribe audio across all conditions (Clean, 15 dB, 10 dB, 5 dB, and Denoised).
  2. Calculate WER and CER using `jiwer`.
  3. Generate comparison plots (`outputs/figures/experiment_1_wer_curve.png`) and tabular metrics.
* **The "WHY" (Technical Rationale):**
  - Provides quantitative evidence for Section 5 of the academic report, proving that spectral denoising recovers up to 15–20% of word accuracy under harsh 5 dB noise.

---

## 3. Member 2: Dialect Normalization & Intent Classification
**Lead:** Member 2 (`S20240010125`)  
**Core Role:** Linguistic Divergence, Phonetic Folding, and Morphological NLP.

### Phase 1: Curate the 4-Scheme Knowledge Base (`data/schemes_data.json`)
* **What to do:**
  1. Extract verified policy facts from official state portals (`apagrisnet.gov.in`, `aarogyasri.ap.gov.in`, `jaganannaammavodi.ap.gov.in`, `aasara.telangana.gov.in`).
  2. Build structured JSON documents containing eligibility rules, financial benefits, application steps, and document lists with official source URLs.
* **The "WHY" (Technical Rationale):**
  - Balances team workload evenly (33% each).
  - Authentic government citations ensure high academic rigor and eliminate factual hallucination during retrieval.

### Phase 2: Build Pan-Telugu Civic Lexicon (`data/pan_telugu_civic_lexicon.json`)
* **What to do:**
  1. Map regional colloquialisms across Andhra Pradesh and Telangana to formal Telugu:
     - **Telangana:** *పైసలు $\to$ డబ్బులు*, *ఎప్పుడొస్తయి $\to$ ఎప్పుడు వస్తాయి*, *ఎట్ల $\to$ ఎలా*.
     - **Rayalaseema:** *సొమ్ము $\to$ నగదు*, *యాడ $\to$ ఎక్కడ*, *బాపతు $\to$ సంబంధించిన*.
  2. Maintain `data/civic_vocab.json` containing canonical Telugu civic keywords.
* **The "WHY" (Technical Rationale):**
  - Standard NLP systems and government portals are trained on literary Telugu (*Grandhikam*).
  - Without an explicit regional dialect lexicon, colloquial words appear as Out-Of-Vocabulary (OOV) tokens, causing total search failure.

### Phase 3: Implement Phonetic Folding & Dialect Normalizer (`src/dialect_normalizer.py`)
* **What to do:**
  1. Create a normalization pipeline that protects vital scheme names (*రైతు బంధు*, *ఆరోగ్యశ్రీ*).
  2. Collapse aspirated consonants into unaspirated counterparts (`థ->త`, `భ->బ`, `ధ->ద`, `ఫ->ప`, etc.).
  3. Apply conservative Levenshtein edit distance with similarity threshold $\ge 0.75$.
* **The "WHY" (Technical Rationale):**
  - Standard Unicode Levenshtein distance fails on Telugu words: *పథకం* (4 chars) and *పతకం* differ by 1 aspirate character, giving similarity 0.75. Standard 0.80 thresholds reject this valid word!
  - Phonetic Folding collapses aspirates before computing distance, matching homophones with 100% similarity while protecting scheme keywords from over-correction.

### Phase 4: Intent Dataset Synthesis & Classifier Training (`src/train_intent_model.py`)
* **What to do:**
  1. Generate 200 balanced queries across 4 civic intents (`ELIGIBILITY`, `BENEFITS`, `REQUIRED_DOCUMENTS`, `APPLICATION_PROCEDURE`).
  2. Vectorize using **character boundary n-grams (2 to 4)** via `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))`.
  3. Train a Logistic Regression model and save it to `models/intent_model.pkl`.
* **The "WHY" (Technical Rationale):**
  - Telugu is an agglutinative language where case markers and regional verb endings attach to root words (*ఎంతొస్తయి*, *ఎంతొస్తాయి*, *ఎంత లభిస్తుంది*). Word tokenizers fail completely on such morphological variants.
  - Sub-word character n-grams capture morpheme roots invariant to regional verb inflections.

### Phase 5: Ablation & Robustness Evaluation (Experiments 2 & 3 — `src/evaluate_nlp.py`)
* **What to do:**
  1. **Exp 2:** Run 50 dialectal test queries to measure token correction accuracy with vs without phonetic folding.
  2. **Exp 3:** Measure Intent Macro-F1 across 3 query states: (1) Clean Reference, (2) Raw Noisy ASR, (3) Normalized ASR.
* **The "WHY" (Technical Rationale):**
  - Empirically verifies the thesis hypothesis: phonetic folding recovers intent classification performance lost due to acoustic misspellings and dialect shifts.

---

## 4. Member 3: Hybrid RAG & Streamlit UI Integration
**Lead:** Member 3 (`S20240030382`)  
**Core Role:** Dravidian Suffix Stripping, Dense/Sparse Fusion, Hallucination Prevention, and User Experience.

### Phase 1: Implement Dravidian Suffix-Stripper (`src/hybrid_retriever.py`)
* **What to do:**
  1. Build `stem_telugu_token` to strip case suffixes (`-లతో`, `-లకు`, `-తో`, `-కు`, `-కి`, `-లో`, `-ల`).
  2. Tokenize corpus chunks and queries through this suffix stripper before BM25 indexing.
* **The "WHY" (Technical Rationale):**
  - In Telugu, noun inflections attach directly to roots (*రైతు*, *రైతుకు*, *రైతుతో*).
  - BM25 relies on exact string equality. A citizen asking about *రైతుకు* has zero lexical overlap with a document stating *రైతు*. Suffix stripping restores exact keyword matching.

### Phase 2: Build Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
* **What to do:**
  1. Index knowledge chunks into ChromaDB using `paraphrase-multilingual-MiniLM-L12-v2`.
  2. Combine BM25 sparse keyword rankings and ChromaDB dense vector rankings using Reciprocal Rank Fusion:
     $$\text{RRF Score}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$$
* **The "WHY" (Technical Rationale):**
  - Dense semantic search alone suffers from **Embedding Drift** when ASR introduces spelling corruption.
  - BM25 alone fails when citizens paraphrase concepts without using exact policy terms.
  - RRF provides fault tolerance by combining keyword precision with conceptual semantic proximity.

### Phase 3: Grounded Answer Simplifier & Fallback Guardrails (`src/simplifier.py`)
* **What to do:**
  1. Constrain answer generation strictly to retrieved context.
  2. Implement an explicit fallback phrase: *"క్షమించండి, అందించిన అధికారిక సమాచారంలో దీనికి సంబంధించిన వివరాలు లేవు."*
  3. Translate administrative jargon into simple conversational Telugu bullet points with source URLs.
* **The "WHY" (Technical Rationale):**
  - In civic welfare applications, hallucinating grant amounts or criteria misinforms citizens.
  - Strict grounding ensures zero unsupported claims and provides verifiable official sources.

### Phase 4: Offline Demo Cache Safety Net (`data/demo_cache.json`)
* **What to do:**
  1. Pre-calculate and cache verified responses for the master benchmark questions.
  2. Serve cached responses instantly if the network drops.
* **The "WHY" (Technical Rationale):**
  - University Wi-Fi frequently disconnects, blocks ports, or triggers API rate limits during live grading.
  - The cache guarantees the live demo never freezes or crashes in front of the evaluators.

### Phase 5: Interactive Dual-Mode Web Application (`app.py`)
* **What to do:**
  1. Integrate Member 1's ASR, Member 2's Normalizer, and Member 3's RAG into a Streamlit dashboard.
  2. Support dual inputs: Voice Audio (.wav) upload OR Typed Telugu text.
  3. Add an Audio Inspection tab displaying noisy vs denoised waveforms.
  4. Generate Telugu spoken audio output using `gTTS(lang='te')`.
* **The "WHY" (Technical Rationale):**
  - The web application serves as the primary evaluation centerpiece.
  - Dual-mode input acts as a live presentation safeguard if the microphone experiences ambient feedback.
  - Audio output provides accessibility for citizens with low literacy.

### Phase 6: Retrieval & Factuality Benchmarks (Experiments 4 & 5 — `src/evaluate_rag.py`)
* **What to do:**
  1. **Exp 4:** Compare Recall@1, Recall@3, and MRR across BM25 alone, Dense alone, and Hybrid RRF.
  2. **Exp 5:** Audit 30 query answers on a 3-point support scale (2: Fully Supported, 1: Partially Supported, 0: Hallucinated) and test 5 out-of-scope trick queries.
* **The "WHY" (Technical Rationale):**
  - Quantitatively validates Research Question 3 (Hybrid search superiority) and Research Question 4 (Hallucination mitigation) for the final report.

---

## 5. Summary Matrix: Module Ownership & Hand-Offs

| Member | Input Received | What They Process & Execute | Output Handed Off | Why It Matters Scientifically |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1** | Spoken Audio (.wav) + Noise Sources | Silence trimming, controlled SNR mixing, spectral gating, Whisper ASR | Clean Audio & Raw Telugu Text | Eliminates acoustic noise corruption before downstream language models run |
| **Member 2** | Raw Telugu Text from M1 | Phonetic folding, regional slang normalization, char n-gram intent classification | Normalized Query & Intent Class | Bridges the vocabulary gap between rural spoken dialects and official policy text |
| **Member 3** | Normalized Query from M2 | Suffix-stripped BM25 + Dense ChromaDB RRF, grounded simplifier, Streamlit UI | Simplified Answer, Source URL & Audio Speech | Guarantees exact factual retrieval without hallucinations, wrapped in a live demo |
