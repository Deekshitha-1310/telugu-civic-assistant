"""Load real evaluation artefacts. Missing files become 'Not evaluated yet'."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import INTENT_ORDER
from paths import HELDOUT_PATH, MODEL_CARD_PATH, PLOTS_DIR


def load_model_card():
    if not os.path.exists(MODEL_CARD_PATH):
        return None
    with open(MODEL_CARD_PATH, encoding="utf-8") as f:
        return json.load(f)


def heldout_intent_metrics(intent_service):
    """Evaluate the deployed IntentService on the 40 held-out queries. Not fabricated."""
    if intent_service is None or not os.path.exists(HELDOUT_PATH):
        return None
    ho = pd.read_csv(HELDOUT_PATH, encoding="utf-8-sig")
    labels = INTENT_ORDER
    out = {"n": int(len(ho)), "conditions": {}}
    for col, name in [("query_standard", "standard (held-out)"), ("query_dialect", "dialect (held-out)")]:
        y_true, y_pred = [], []
        for row in ho.itertuples():
            pred = intent_service.predict(getattr(row, col))
            y_true.append(row.intent_label)
            y_pred.append(pred["intent"])
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["conditions"][name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "per_class": {
                lab: {
                    "precision": report[lab]["precision"],
                    "recall": report[lab]["recall"],
                    "f1": report[lab]["f1-score"],
                    "support": report[lab]["support"],
                }
                for lab in labels
            },
            "confusion_matrix": cm.astype(int).tolist(),
            "labels": labels,
        }
    return out


def plot_paths():
    found = {}
    mapping = {
        "exp2_phonetic_ablation.png": "Experiment 2 — normalizer ablation",
        "exp3_intent_macro_f1.png": "Experiment 3 — intent macro-F1 under dialect/noise",
    }
    if not os.path.isdir(PLOTS_DIR):
        return found
    for fname, title in mapping.items():
        path = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(path):
            found[title] = path
    return found


def evaluation_catalogue():
    """What exists vs what has never been run. Used by the Evaluation page."""
    plots = plot_paths()
    return {
        "intent_model_card": os.path.exists(MODEL_CARD_PATH),
        "heldout_file": os.path.exists(HELDOUT_PATH),
        "normalization_plots": plots,
        "asr": {
            "status": "not_evaluated",
            "note": (
                "No WER/CER on real audio is stored in this repository. "
                "`asr_noise.py` is a phonetic simulation used in Experiment 3, not live ASR."
            ),
        },
        "retrieval": {
            "status": "not_evaluated",
            "note": (
                "No labelled query–document set for Recall@3 or MRR is present, "
                "so BM25 / dense / hybrid retrieval have not been scored."
            ),
        },
        "normalization_metrics_file": {
            "status": "plots_only" if plots else "not_evaluated",
            "note": (
                "Token recovery and harmful-correction rates are produced by "
                "`python src/evaluate_nlp.py` (Experiment 2). Figures appear here when "
                f"`{PLOTS_DIR}` contains the saved plots."
            ),
        },
        "answer_quality": {
            "status": "not_evaluated",
            "note": "No factual-grounding or source-support annotation file is in the repo.",
        },
        "dialect_vs_noise": (
            "Experiment 3 (when run) reports held-out dialect queries and simulated "
            "ASR spelling noise separately. There is no recorded evaluation on real "
            "noisy recordings in this tree."
        ),
    }
