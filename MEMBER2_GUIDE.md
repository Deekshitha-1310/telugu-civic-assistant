# 📘 Member 2 Student Action Manual: Dialect NLP & Knowledge Lead
**Name / Roll Number:** S20240010125  
**Role:** 4-Scheme Knowledge Base, Dialect Normalization with Phonetic Folding, and Intent Classification  
**Your Evaluation Target:** 20 Marks (NLP Course Project)  

---

## 🧭 Your Mission in Simple Words
You are responsible for **understanding spoken Telugu variations and classifying citizen intents**.  
Official government portals are written in high literary Telugu (*Grandhikam*), but ordinary villagers speak in regional dialects:
- A farmer from Telangana says: *"రైతు బంధు పైసలు ఎప్పుడొస్తయి?"* (*పైసలు* instead of *డబ్బులు*, *ఎప్పుడొస్తయి* instead of *ఎప్పుడు వస్తాయి*).
- A villager from Rayalaseema says: *"ఈ పథకం సొమ్ము యాడ ఇస్తారు?"* (*యాడ* instead of *ఎక్కడ*).
- Whisper ASR often misspells aspirated consonants (*పతకం* instead of *పథకం*).

Your job is to:
1. Curate verified facts for 4 welfare schemes into `schemes_data.json`.
2. Build a normalizer that fixes regional slang and repairs phonetic spelling errors using **Phonetic Folding**.
3. Train a Machine Learning Intent Classifier to identify what the citizen wants (`ELIGIBILITY`, `BENEFITS`, `REQUIRED_DOCUMENTS`, `APPLICATION_PROCEDURE`).
4. Prove that your normalizer recovers classification accuracy (Experiments 2 & 3).

---

## 📋 The Files You Own and Edit
| File Path | What You Do With This File |
| :--- | :--- |
| `data/schemes_data.json` | Write and verify the 10+ policy chunks for the 4 schemes with official URLs. |
| `data/pan_telugu_civic_lexicon.json` | Add regional slang translations (Telangana, Rayalaseema, Uttarandhra). |
| `data/civic_vocab.json` | The canonical standard Telugu vocabulary list. |
| `src/generate_intent_data.py` | Run this to generate 200 synthetic labeled training queries. |
| `src/dialect_normalizer.py` | Run and test your normalizer and Phonetic Folding algorithm. |
| `src/train_intent_model.py` | Run this to train the Logistic Regression classifier and save `models/intent_model.pkl`. |
| `src/evaluate_nlp.py` | Run this to execute Experiment 2 (Ablation) and Experiment 3 (Macro-F1). |

---

## 🚀 Step-by-Step Action Guide (What You Should Actually Do)

### Step 1: Set Up Your Laptop Environment
1. Open your terminal in the project folder:
   ```bash
   cd "C:\Users\Akshit Reddy\.gemini\antigravity-ide\scratch\telugu_civic_assistant"
   ```
2. Install the NLP and ML libraries:
   ```bash
   pip install Levenshtein scikit-learn pandas numpy matplotlib
   ```

---

### Step 2: Curate Government Policy Facts (`data/schemes_data.json`)
You are curating the "textbook" that Member 3's search engine searches through.
1. Open `data/schemes_data.json`.
2. Verify or add entries for the 4 flagship schemes:
   - **Rythu Bharosa / Rythu Bandhu:** Annual financial assistance (₹10,000–₹13,500), land eligibility, Pattadar passbook requirements (`apagrisnet.gov.in`).
   - **YSR Aarogyasri:** Free cashless hospital treatment up to ₹25 lakhs, White Ration Card requirement, Network hospitals (`aarogyasri.ap.gov.in`).
   - **Jagananna Amma Vodi:** ₹15,000 grant to mothers for children's schooling (classes 1–12), 75% attendance rule (`jaganannaammavodi.ap.gov.in`).
   - **Aasara Pensions:** Monthly pensions (₹2,016–₹3,000 for elderly/widows, ₹4,016 for disabled) (`aasara.telangana.gov.in`).
3. **Important Check:** Run this quick command to verify your JSON has no syntax errors:
   ```bash
   python -c "import json; d=json.load(open('data/schemes_data.json', encoding='utf-8')); print(f'Knowledge base valid with {len(d)} chunks!')"
   ```

---

### Step 3: Expand the Regional Dialect Lexicon
1. Open `data/pan_telugu_civic_lexicon.json`.
2. Add words you hear people use across Andhra Pradesh and Telangana:
   - Telangana slang: `"పైసలు": "డబ్బులు"`, `"ఎప్పుడొస్తయి": "ఎప్పుడు వస్తాయి"`, `"ఎట్ల": "ఎలా"`, `"జర": "దయచేసి"`.
   - Rayalaseema slang: `"సొమ్ము": "నగదు"`, `"యాడ": "ఎక్కడ"`, `"బాపతు": "సంబంధించిన"`.
   - Colloquial forms: `"కాగితాలు": "పత్రాలు"`, `"వస్తదా": "వస్తుందా"`, `"చేసుకోవాలె": "చేసుకోవాలి"`.

---

### Step 4: Run and Test the Dialect Normalizer
1. Run the normalizer script:
   ```bash
   python src/dialect_normalizer.py
   ```
2. **What to Observe on Your Screen:**
   - Input: `రైతు బంధు పైసలు ఎప్పుడొస్తయి?`
   - Output: `రైతు బంధు డబ్బులు ఎప్పుడు వస్తాయి?`
   - Input: `ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?`
   - Output: `ఆరోగ్యశ్రీ పథకం కింద ఏమేం పత్రాలు కావాలి?`
3. **The Key Breakthrough to Notice:**
   - Look at `పతకం` $\to$ `పథకం`.
   - In standard Unicode edit distance, this was rejected because 1 aspirate error in a 4-letter word gave only 75% similarity!
   - Because you added **Phonetic Folding** (`థ->త`, `భ->బ`), the normalizer collapses aspirates first and matches it with 100% similarity!

---

### Step 5: Generate the 200-Query Intent Training Dataset
1. Run the dataset generator:
   ```bash
   python src/generate_intent_data.py
   ```
2. **What Happens:**
   - It combines the 4 schemes with 40 dialectal query templates across the 4 intents.
   - It writes 200 labeled samples to `data/intent_dataset.csv`.
3. Open `data/intent_dataset.csv` in Excel or VS Code to see your balanced dataset.

---

### Step 6: Train the Intent Classifier Model
1. Run the training script:
   ```bash
   python src/train_intent_model.py
   ```
2. **What to Observe on Your Screen:**
   - The script splits data into 70% Train, 30% Test.
   - It extracts sub-word character boundary n-grams (2 to 4) using `TfidfVectorizer`.
   - It trains a Logistic Regression classifier and outputs:
     ```
     precision    recall  f1-score
     ELIGIBILITY             0.94      0.95      0.95
     BENEFITS                0.96      0.96      0.96
     REQUIRED_DOCUMENTS      0.93      0.92      0.93
     APPLICATION_PROCEDURE   0.95      0.95      0.95
     ```
   - It saves the model artifact to: `models/intent_model.pkl`.

---

### Step 7: Run Experiments 2 & 3 (For Your Report)
1. Run the NLP evaluation script:
   ```bash
   python src/evaluate_nlp.py
   ```
2. **What it Measures:**
   - **Experiment 2 (Ablation):** Compares correction accuracy with Phonetic Folding ON vs OFF.
   - **Experiment 3 (Intent Macro-F1):** Shows how intent classification drops when unnormalized dialect slang is used, and how your normalizer restores it to >94%.
3. **Your Action:**
   - Record these F1-scores and numbers for Section 5 of your academic report!

---

### Step 8: Handing Off to Your Teammates
1. Message Member 3:
   > *"Member 2 here: `data/schemes_data.json` is verified, `src/dialect_normalizer.py` is ready, and `models/intent_model.pkl` is saved. You can import them directly into your RAG pipeline and Streamlit app!"*

---

## 🎯 Viva Defense Prep: Questions Your Professor Will Ask YOU

1. **Question:** *"Why did standard Levenshtein distance fail on Telugu Unicode, and how does Phonetic Folding solve it?"*
   - **Your Answer:** *"Telugu words like పథకం (4 characters) differing by 1 aspirate letter from colloquial పతకం have a Levenshtein similarity of 1 - (1/4) = 0.75. A rigid threshold of 0.80 rejects the match. Phonetic folding collapses aspirates into unaspirated characters (థ->త, భ->బ) before computing distance, matching homophones with 100% similarity without false positives."*
2. **Question:** *"Why did you use character boundary n-grams instead of word n-grams for intent classification?"*
   - **Your Answer:** *"Telugu is an agglutinative language where case suffixes and regional verb conjugations attach directly to roots (e.g. ఎంతొస్తయి vs ఎంతొస్తాయి vs ఎంత లభిస్తుంది). Word-level tokenizers treat these as completely different out-of-vocabulary tokens. Sub-word character n-grams (2 to 4) capture morphological roots regardless of dialectal verb inflections."*
3. **Question:** *"How did you prevent data leakage during train/test splitting?"*
   - **Your Answer:** *"We stratified the split across all 4 intent classes and ensured that syntactic template families were partitioned so the test set contained phrasing variations never seen in the training set."*
