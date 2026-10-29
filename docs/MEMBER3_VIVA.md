# 🎓 Member 3 Viva Defense Guide & Professor Q&A Sheet

**Student Name / Roll Number:** Member 3 (`S20240030382`)  
**Project:** Dialect and Noise-Robust Telugu Civic Assistant  
**Role:** Search, RAG & UI Integration Lead  
**Evaluation Target:** 20 Marks (NLP Course Project)  

---

## ⏱️ 3-Minute Live Demonstration Flow

When the professor asks you to demonstrate your system, follow this sequence:

| Timestamp | What You Show in Streamlit UI (`http://localhost:8501`) | What You Say (Your Viva Script) |
| :--- | :--- | :--- |
| **0:00 – 0:45** | Open **"💬 బహుళ-టర్న్ సంభాషణా సహాయకుడు"**.<br>Type Turn 1: `రైతు బంధు పథకానికి ఎవరు అర్హులు?` | *"Here we initiate a citizen session on Rythu Bandhu. Notice our hybrid retriever matches the official eligibility document (`rb_elig_01`) with RRF score 0.045, displaying the grounded rules, source citation (`apagrisnet.gov.in`), and audio speech synthesis."* |
| **0:45 – 1:30** | Type Turn 2 (Elliptical): `ఏమేం కాగితాలు కావాలి?`<br>*(No scheme mentioned)* | *"In conversational Telugu, citizens rarely repeat the scheme name. Our ConversationManager detects this ellipsis, resolves the active context to Rythu Bandhu, and extracts the required document checklist (`rb_docs_03`)."* |
| **1:30 – 2:15** | Type Turn 3 (Polar Verification): `కౌలు రైతులకు వర్తిస్తుందా?` | *"Citizens often ask Yes/No questions. Notice our system synthesizes: `❌ లేదు, అధికారిక నిబంధనల ప్రకారం కౌలు రైతులకు (Tenant Farmers) ఈ పథకం వర్తించదు`, followed by the exact related conditions without hallucinating."* |
| **2:15 – 3:00** | Click **"🔍 ఈ సమాధానం ఎలా ప్రాసెస్ చేయబడింది? (Pipeline Diagnostics)"** expander. | *"Behind the UI, we show full diagnostic transparency: normalized query text, Member 2's intent classification, BM25 rank, dense vector rank, and the fused Reciprocal Rank Fusion score."* |

---

## 🎯 Top 10 Professor Questions & Bulletproof Answers

### Q1: Why did you use Hybrid Search (BM25 + Dense) instead of pure ChromaDB vector search?
> **Answer:**  
> *"Pure dense embeddings (like multilingual MiniLM) suffer from **Embedding Drift** on low-resource regional dialects. When rural citizens use dialectal synonyms like `పైసలు` (money) or misspelled terms, dense cosine similarity drops drastically (producing only 17.86% Recall@1). Pure BM25, on the other hand, fails when queries are phrased conceptually without exact official keywords. Combining BM25 and ChromaDB via Reciprocal Rank Fusion (RRF with $k=60$) balances sparse keyword precision with dense semantic context, achieving **92.86% Recall@3** on our civic benchmark."*

---

### Q2: Why was Telugu Suffix-Stripping necessary before BM25 indexing?
> **Answer:**  
> *"Telugu is an agglutinative Dravidian language where nominal postpositions and case markers attach directly to nouns:
> - $\text{రైతు}$ (farmer) $\longleftrightarrow$ $\text{రైతుకు}$ (to the farmer) $\longleftrightarrow$ $\text{రైతుతో}$ (with the farmer).
> 
> Standard BM25 performs exact token matching. If a citizen types `రైతులకు` (dative plural), an unstemmed BM25 engine treats it as a distinct token from `రైతు` in the official gazette. Our suffix stripper removes 10 nominal case suffixes (`-లతో, -లకు, -తో, -కు, -లో, -ల`), which our ablation study proved **boosts Mean Reciprocal Rank (MRR) from 0.7997 to 0.8069**."*

---

### Q3: What is Reciprocal Rank Fusion (RRF) and why not simple linear score addition?
> **Answer:**  
> *"BM25 produces unbounded positive relevance scores (e.g. 1.5 to 18.0), whereas dense retrieval produces cosine similarities bounded between $-1.0$ and $+1.0$. Because their score distributions are on fundamentally incompatible scales, simple weighted addition $(\alpha S_{\text{bm25}} + \beta S_{\text{dense}})$ is brittle and requires continuous recalibration.
> 
> Reciprocal Rank Fusion is scale-invariant because it operates purely on **ranks**, not raw scores:
> $$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$
> With smoothing constant $k=60$, RRF heavily rewards passages ranked near the top by both models while preventing lower-ranked outliers from dominating."*

---

### Q4: How does your system guarantee Zero Hallucinations?
> **Answer:**  
> *"We enforce deterministic in-context grounding through a four-stage verification layer (`EvidenceValidator`):
> 1. **Retrieval Thresholding:** If the top passage has an RRF score below the minimum threshold ($0.016$), the system strictly abstains.
> 2. **Lexical Evidence Guard:** If a query contains zero civic terms and zero lexical overlap with our knowledge base (e.g. asking for gold prices or cricket scores), it is rejected.
> 3. **Closed-Book Grounding:** Answer generation extracts text strictly from the verified government passage and cites the official state URL (`source_url`).
> 4. **Safe Abstention Phrase:** When information is missing, the system outputs the standardized fallback: `క్షమించండి, అందించిన అధికారిక సమాచారంలో దీనికి సంబంధించిన వివరాలు లేవు.`
> In our Experiment 5 audit across 34 test queries, our system achieved **0.0% hallucinations**."*

---

### Q5: What happens when a citizen asks a question about an official scheme that is not covered in your 20 policy chunks?
> **Answer:**  
> *"Instead of leaving the citizen stranded with a generic error, we built a **Guided Helpline Fallback System**. When the query targets an official scheme (e.g. asking for the physical address of the Aarogyasri office), the system detects the scheme key and immediately provides:
> - The 24x7 state toll-free helpline (`104` or `1800-599-1111`),
> - The official state portal URL (`https://aarogyasri.ap.gov.in`),
> - The designated contact office (`నెట్‌వర్క్ ఆసుపత్రిలోని ఆరోగ్యమిత్ర డెస్క్`)."*

---

### Q6: How do you handle multi-turn conversational follow-up questions?
> **Answer:**  
> *"Our `ConversationManager` maintains session state across up to 5 dialogue turns:
> 1. **Anaphora Resolution:** Detects Telugu deictic markers (`దానికి`, `దీనికి`, `ఆ పథకం`, `ఈ పథకానికి`) and binds them to the active scheme context.
> 2. **Ellipsis Resolution:** When a query lacks an overt subject (e.g. `ఏమేం కాగితాలు కావాలి?`), the active scheme is injected into the query before search.
> 3. **Scheme Context Switching:** If a citizen asks about *Amma Vodi* after discussing *Rythu Bandhu*, the system detects the new scheme keywords and updates the state.
> 4. **Ambiguity Guard:** If a follow-up is asked on an empty session without context, it conservatively flags the query as ambiguous instead of guessing."*

---

### Q7: Why does your answer generator output conversational phrasing instead of raw chunk text?
> **Answer:**  
> *"Raw chunk dumps force rural citizens to read bureaucratic legal text. Our `TeluguSimplifier` classifies the citizen's speech act:
> - **Polar (Yes/No):** Direct affirmation or denial lead-in (`❌ లేదు` or `✅ అవును`),
> - **Financial amounts:** Prefaced with `💰 ఆర్థిక సహాయం & ప్రయోజనాలు`,
> - **Documents:** Prefaced with `📑 దరఖాస్తుకు అవసరమైన అధికారిక పత్రాలు`,
> - **Helplines:** Prefaced with `📞 అధికారిక టోల్-ఫ్రీ హెల్ప్‌లైన్`.
> Crucially, all figures, limits, and rules are retained verbatim from the official passage—ensuring the answer is natural without altering any factual substance."*

---

### Q8: What if the college Wi-Fi disconnects during the live viva?
> **Answer:**  
> *"We built an **Offline Demo Cache** in `data/demo_cache.json`. Verified Q&A pairs for our primary demo questions are pre-cached. If the server loses internet access or external API calls fail, the system serves the verified answers from cache, ensuring zero crashes during live evaluation."*

---

### Q9: What are the key mathematical formulas behind your evaluation metrics?
> **Answer:**  
> 1. **Recall@K:**
>    $$\text{Recall@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le K)$$
> 2. **Mean Reciprocal Rank (MRR):**
>    $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
> 3. **Reciprocal Rank Fusion (RRF):**
>    $$\text{RRF}(d) = \sum_{m} \frac{w_m}{k + r_m(d)} \quad (k=60)$$

---

### Q10: How do you respect team member boundaries in your codebase?
> **Answer:**  
> *"Our project enforces clean micro-pipeline encapsulation:
> - **Member 1 (`S20240010065`)** owns acoustic processing and Whisper ASR in `src/asr_noise.py`.
> - **Member 2 (`S20240010125`)** owns Telugu dialect normalization and intent classification in `src/dialect_normalizer.py` and `src/intent_api.py`.
> - **Member 3 (`S20240030382` — My Role)** consumes their clean outputs via standardized Python APIs (`normalize()`, `predict_intent()`) to drive Hybrid RAG, Evidence Validation, and the Streamlit UI, without modifying their core model training files."*

---

### Q11: Why didn't you ship open-vocabulary character-by-character Tanglish transliteration?
> **Answer:**  
> *"We rigorously evaluated character-level Tanglish conversion and identified two severe real-world bottlenecks:
> 1. **Model Deployment Barriers:** Modern neural transliterators like `ai4bharat/IndicXlit` depend on `fairseq`, which requires C++ compilers and fails to build on Windows under Python 3.12.
> 2. **Phonetic Distortion in Rule-Based Tools:** Rule-based tools (like `indic_transliteration` / Sanscript / ITRANS) severely distort conversational colloquial typing: for example, `arogya sri eppudu vastundi` is transliterated into `అరోగ్య స్రి ఏప్పుదు వస్తుంది`, corrupting vowel lengths and sibilants. This causes BM25 lexical match to fail completely, dropping recall to 0%.
> 3. **Civic Welfare Safety:** When providing life-critical government subsidy and healthcare details (e.g. ₹5 Lakh medical coverage, ₹13,500 agrarian aid), guessing on corrupted words risks misinforming citizens.
> 
> **Our Solution:**
> We established **Native Telugu Unicode Script** and **Telugu Voice Audio (ASR)** as the core input channels. For Latin/English typing, we built a **Verified Scheme-Alias Router** (`src/transliterate.py`) that deterministically maps verified scheme names and civic intents (`rythu bandhu eligibility`, `aarogya sri documents`) to canonical Telugu queries. For unverified arbitrary Latin queries, the system engages the **Civic Language Guardrail (భాషా మార్గదర్శకం)**, guiding citizens to use Telugu script or the microphone, backed by 1-click verified prompts."*

