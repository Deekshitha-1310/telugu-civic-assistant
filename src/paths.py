"""Project paths resolved from this file so the app works from any cwd."""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

SCHEMES_PATH = os.path.join(DATA_DIR, "schemes_data.json")
KNOWLEDGE_CORPUS_PATH = os.path.join(DATA_DIR, "knowledge_corpus.jsonl")
KNOWLEDGE_SOURCES_PATH = os.path.join(DATA_DIR, "knowledge_sources.json")
KNOWLEDGE_CACHE_DIR = os.path.join(DATA_DIR, ".knowledge_cache")
DISCOVERED_SOURCES_PATH = os.path.join(DATA_DIR, "discovered_sources.json")
CHROMA_DIR = os.path.join(ROOT, ".chroma")
INTENT_DATASET_PATH = os.path.join(DATA_DIR, "intent_dataset.csv")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout_intent_queries.csv")
ABLATION_PROBES_PATH = os.path.join(DATA_DIR, "ablation_probes.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "intent_model.pkl")
MODEL_CARD_PATH = os.path.join(MODELS_DIR, "model_card.json")
