# Dialect & Noise-Robust Telugu Civic Assistant

A voice-enabled Telugu civic assistant for government welfare information across regional dialects and noisy environments.

## Project structure
- `data/` – curated schemes (`schemes_data.json`), intent dataset, dialect lexicon, civic vocabulary, held-out test queries, audio folders.
- `src/` – `dialect_normalizer.py`, `asr_noise.py`, `generate_intent_data.py`, `train_intent_model.py`, `intent_api.py`, `evaluate_nlp.py`.
- `models/` – `intent_model.pkl` + `model_card.json` (library versions, metrics, out-of-scope threshold).
- `outputs/plots/` – Experiment 2 and 3 figures.
- `tests/` – regression tests (`python tests/test_nlp.py`).
- `app.py` – Streamlit dashboard (Member 3).

## Setup
```bash
pip install -r requirements.txt
```
## Run order (Windows: prefix with `$env:PYTHONIOENCODING="utf-8";`)
```bash
python src/generate_intent_data.py
python src/train_intent_model.py
python src/evaluate_nlp.py
python tests/test_nlp.py
```
## For Member 3 (RAG / UI)
```python
from intent_api import IntentService
svc = IntentService()
svc.predict("రైతు బంధు పైసలు ఎప్పుడొస్తయి?")
# {'normalized': '...', 'intent': 'BENEFITS', 'confidence': 0.7, 'in_scope': True, 'edits': [...]}
```
`edits` lists every rewrite with the rule that fired - use it for the "Normalization" inspection tab.
