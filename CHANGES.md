# Audit fixes (v1 -> v2) — Member 2 components

All numbers below are reproducible with `python src/train_intent_model.py && python src/evaluate_nlp.py`.

## 1. Data leakage in the train/test split  (train_intent_model.py, generate_intent_data.py)
* v1: plain `StratifiedShuffleSplit`; 80/80 test rows had a same-template sibling in training -> "1.00 F1".
* v2: `template_id` column + `GroupShuffleSplit`; grouped 5-fold CV; baselines.
  Grouped CV macro-F1: majority 0.08 | word-LR 0.78±0.13 | char-NB 0.76±0.16 | **char-LR 0.80±0.09**
  (char n-grams > word n-grams is real but small relative to the spread: only 55 templates).
* Final model is refit on all data (v1 shipped the 70 % model). `models/model_card.json` records versions.
* Generator no longer pairs nonsense template x scheme (tenant-farmer + Aarogyasri, passbook + pension ...).

## 2. Normalizer rewrote valid Telugu  (dialect_normalizer.py)
v1 rule 3 (suffix-strip + Levenshtein >= 0.75 on EVERY unknown token) changed 14 % of clean tokens:
ఆఫ్‌లైన్->ఆన్‌లైన్ (offline->online), తీసుకోవాలి->తెలుసుకోవాలి (take->know), ఇస్తారు->చేస్తారు,
అవుతుంది->అందుతుంది, అర్హులా->అర్హులు (question lost), పథకానికి->పథకం (suffix dropped).
v2: unique phonetic-key lookup (aspirate+sibilant+retroflex+vowel-length folding), suffix KEPT, known-word guard,
optional guarded Levenshtein<=1 (off by default). Also: phrase lexicon now works (4 v1 entries were dead),
punctuation/danda/quotes handled, ZWNJ-safe, whole-word transliteration, productive dialect verb endings
(ాలె, స్తయి, తది ...), `explain()` returns every edit + rule for the UI.
Lexicon: removed కౌలు (inserted garbage), ఇస్తారా, ఆడ ("women" vs "there"), వెళ్ళాలా; question markers preserved.
Tried and REJECTED: repairing dropped final anusvara (rewrites ప్రభుత్వ, వైద్య, కుటుంబ ...). FP test caught it.

## 3. Experiment 2 redesigned
v1: 49 hand-written probes built around the lexicon (circular), substring check, fold decisive in 2 probes.
v2: 755 simulated ASR corruptions of the civic vocabulary + 296 clean tokens for false positives.

| config | recovery | false-pos |
|---|---|---|
| 1 lexicon only | 0 % | 0 |
| 2 plain Levenshtein, no fold | 64.5 % | 30 (10.1 %) |
| 3 v1 as shipped | 63.7 % | 42 (14.2 %) |
| 4 v2 aspirate-only fold | 30.6 % | 0 |
| **5 v2 full fold (shipped)** | **71.9 %** | **0** |
| 6 v2 + Levenshtein<=1 (opt-in) | 90.5 % | 4 (1.35 %) |

Caveat for the report: corruptions are SIMULATED (asr_noise.py); swap in Member 1's real Whisper output.
Not handled by default: dropped anusvara, deleted consonants (config 6 recovers them at a small FP cost).

## 4. Experiment 3 redesigned (and an honest result)
v1 trained on RAW text and tested on NORMALIZED text (train/test mismatch). v2 trains and tests through the same
normalizer. Held-out set moved to `data/heldout_intent_queries.csv`; bootstrap CIs added (n = 40).
* Protocol A (dialect seen in training): normalization does NOT help (0.924 -> 0.901, CI of difference [-0.12,+0.06]).
* Protocol B (dialect unseen): 0.750 -> 0.804 (+0.054, CI [-0.07,+0.18]); with simulated ASR noise +2.7 to +3.0.
* Conclusion to write: direction is positive under dialect shift, NOT statistically significant at n = 40.
  The v1 normalizer scored similarly on intent F1 because its aggressive stemming acts as feature canonicalisation —
  but it changes meaning, which matters for retrieval and display.
* The ">94 %" target in the manual is not supported by these data. Do not claim it.

## 5. Housekeeping
README (control characters), .gitignore (BOM; model was ignored yet committed), requirements, path-safe `_validate.py`,
`tests/test_nlp.py` (10 regression tests), `intent_api.py` (confidence threshold for out-of-scope queries).

## 6. Still open (needs you / teammates)
* Knowledge base facts — see KB_AUDIT.md.
* Native-speaker review of new lexicon entries (ఇట్ల, అట్ల, ఏం, ఎవురు, ఎవురికి) and MORPH_RULES.
* More templates: 55 families is why CV std is ±0.09-0.16. Write 20+ more, especially dialect.
* Replace simulated noise with real ASR transcripts of queries_master.csv; enlarge the held-out set to 60-80.
* `out_of_scope_threshold = 0.45` was set from a handful of probes; calibrate on real out-of-domain queries.
