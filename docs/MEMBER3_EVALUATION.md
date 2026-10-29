# Member 3 Evaluation & Ablation Report
**Project:** Dialect and Noise-Robust Telugu Civic Assistant  
**Role:** Member 3 — Search, RAG & UI Integration Lead (`S20240030382`)  
**Evaluation Script:** `src/evaluate_rag.py`  
**Evaluation Benchmark:** `data/evaluation_queries.json`  

---

## 1. Evaluation Benchmark Construction

The evaluation suite uses a curated test set of **34 items** in `data/evaluation_queries.json` categorized into:
1. **Standard In-Scope Civic Queries (20 items):** Covering all 4 schemes across all 5 official categories (Eligibility, Benefits, Required Documents, Application Procedure, Helplines/FAQs).
2. **Dialectal & Colloquial Variants (4 items):** Testing Telangana/Rayalaseema colloquial expressions (`పైసలు ఎప్పుడొస్తయి?`, `ఏమేం కాగితాలు కావాలె?`, `బడి హాజరు ఎంత?`).
3. **Conversational Follow-Up Queries (4 items):** Testing anaphora (`దానికి ఎవరు అర్హులు?`) and ellipsis (`ఏమేం కాగితాలు కావాలి?`) under active session context.
4. **Ambiguous Context Query (1 item):** Testing ambiguity detection on a fresh session (`దానికి ఎంత ఇస్తారు?`).
5. **Out-of-Scope Trick Queries (5 items):** Testing safe abstention on weather, gold rates, movie tickets, cricket scores, and bus timings.

Relevance ground truth is labeled strictly against actual document IDs (`rb_elig_01`, `as_ben_06`, etc.) in `data/schemes_data.json`.

---

## 2. Metric Definitions

1. **Recall@1 (%):** Percentage of queries where a ground-truth relevant document is retrieved at Rank 1.
   $$\text{Recall@1} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q = 1) \times 100$$
2. **Recall@3 (%):** Percentage of queries where a ground-truth relevant document is present within the Top 3 candidates.
   $$\text{Recall@3} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le 3) \times 100$$
3. **Mean Reciprocal Rank (MRR):** Evaluates how high the correct document is placed in the ranked list.
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
4. **Answer Grounding Support Score (0–2):**
   - 2 = Fully Grounded in retrieved passage or Verified Safe Abstention.
   - 1 = Partially supported claim.
   - 0 = Hallucination or fabricated claim.
5. **Hallucination Rate (%):** Percentage of answers containing unverified claims or false rejections.

---

## 3. Actual Measured Results

### Experiment 4A: Retrieval Performance & Ablation Study
Evaluated on 28 in-scope queries (`outputs/tables/experiment_4_retrieval_ablation.csv`):

| Retrieval Configuration | Recall@1 (%) | Recall@3 (%) | Mean Reciprocal Rank (MRR) |
| :--- | :---: | :---: | :---: |
| **BM25 (Raw Tokens, No Suffix Stripping)** | 64.29% | 92.86% | 0.7730 |
| **BM25 (With Telugu Suffix Stripping)** | **64.29%** | **92.86%** | **0.7789** |
| **Dense ChromaDB (MiniLM-L12-v2)** | 17.86% | 39.29% | 0.3352 |
| **Hybrid RRF ($k=60$)** | 50.00% | 78.57% | 0.6738 |

#### Key Ablation Finding:
* **Effect of Suffix Stripping:** Suffix stripping improved MRR from **0.7730 to 0.7789**. When users ask inflected queries (e.g. `రైతులకు` with dative plural `-లకు`), raw BM25 penalizes the mismatch with the document root `రైతు`. Stripping suffixes aligns inflected queries to base document roots, placing target documents higher in rank.
* **Dense Retrieval Bottleneck in Indic Dialects:** Multilingual dense embeddings (`MiniLM`) perform poorly alone on low-resource dialectal Telugu (Recall@1 of only 17.86%). They suffer from *Semantic Vector Drift* when colloquial slang (`పైసలు`, `కాగితాలు`) or phonetic variations are present. Lexical BM25 provides the necessary keyword anchors.

---

### Experiment 4B: Conversational Follow-Up Resolution
Evaluated on conversational queries (`outputs/tables/experiment_4_conversational.csv`):

| Test ID | User Query | Active Scheme Context | Resolved Query | Status | Resolution Correct |
| :---: | :--- | :--- | :--- | :---: | :---: |
| `eval_25` | దానికి ఎవరు అర్హులు? | అమ్మ ఒడి / తల్లికి వందనం | అమ్మ ఒడి పథకానికి ఎవరు అర్హులు? | `RESOLVED` | **TRUE** |
| `eval_26` | ఏమేం కాగితాలు కావాలి? | రైతు బంధు / రైతు భరోసా | రైతు బంధు ఏమేం కాగితాలు కావాలి? | `RESOLVED` | **TRUE** |
| `eval_27` | ఎక్కడ దరఖాస్తు చేసుకోవాలి? | వైఎస్ఆర్ ఆరోగ్యశ్రీ | వైఎస్ఆర్ ఆరోగ్యశ్రీ ఎక్కడ దరఖాస్తు చేసుకోవాలి? | `RESOLVED` | **TRUE** |
| `eval_28` | టోల్ ఫ్రీ నంబర్ ఎంత? | ఆసరా పెన్షన్ / పెన్షన్ కానుక | ఆసరా పెన్షన్ టోల్ ఫ్రీ నంబర్ ఎంత? | `RESOLVED` | **TRUE** |
| `eval_29` | దానికి ఎంత ఇస్తారు? | None (Fresh Session) | దానికి ఎంత ఇస్తారు? | `AMBIGUOUS` | **TRUE** |

**Success Rate:** **5 / 5 (100.0%)** correct conversational resolution.

---

### Experiment 5: Factuality & Abstention Audit
Evaluated on all 34 benchmark test cases (`outputs/tables/experiment_5_grounding_audit.csv`):
- **Total Test Cases:** 34
- **Fully Supported Grounded Answers / Safe Rejections:** **34 / 34 (100.0%)**
- **Average Grounding Support Score:** **2.00 / 2.00**
- **Hallucinations Detected:** **0**
- **Measured Hallucination Rate:** **0.0%**

---

## 4. Failure Cases & Limitations

1. **Short Query Ambiguity:** If a user enters a single non-specific word like `డబ్బులు` without any active context, lexical search matches multiple schemes equally, resulting in split ranking.
2. **Compound Suffixes:** Deeply nested morphological agglutinations (e.g. `రైతులందరితోటి`) are only partially stripped by single-pass suffix matching.
3. **Cross-Lingual Code-Mixing:** Unseen English slang mixed into Telugu script without prior lexicon mapping relies solely on dense retrieval.

---

## 5. Experiments Not Run & Justification
- **Fine-Tuning MiniLM:** Fine-tuning dense multilingual sentence transformers requires thousands of labeled positive/negative Telugu passage pairs, which exceeds the scope of this undergraduate course project.
- **Full Generative LLM Fine-Tuning:** Not executed to maintain an offline, local undergraduate deployment that does not require expensive GPU servers or proprietary API keys.
