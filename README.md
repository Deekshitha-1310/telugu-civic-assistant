# Dialect & Noise-Robust Telugu Civic Assistant

A voice-enabled Telugu civic assistant for government welfare information access across regional dialects and noisy environments.

## Project Structure
- data/: Curated government schemes, intent dataset, dialect lexicon, and audio folders.
- src/: Core Python modules for audio processing, dialect normalization, and RAG search.
- models/: Saved intent classification models.
- outputs/: Audio responses and experiment figures.
- pp.py: Streamlit web dashboard.

## Setup
`ash
pip install -r requirements.txt
streamlit run app.py
`
