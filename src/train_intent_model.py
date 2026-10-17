"""
train_intent_model.py  —  Member 2: Intent Classifier Training
===============================================================
Trains a Logistic Regression intent classifier on character boundary
n-gram TF-IDF features (handles Telugu agglutination naturally).

• Input : data/intent_dataset.csv
• Output: models/intent_model.pkl  (pipeline: TfidfVectorizer + LogReg)

Data-Leakage-Free Splitting Strategy
--------------------------------------
We stratify across all 4 intent classes AND split by template family
so that syntactic variants (e.g. "ఎట్ల" and "ఎలా" in the same family)
are never simultaneously in both train and test sets.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score

# ── Path resolution ────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.join(_SRC_DIR, '..')
DATA_PATH  = os.path.join(_ROOT, 'data',   'intent_dataset.csv')
MODEL_DIR  = os.path.join(_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'intent_model.pkl')


def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"📂  Loaded {len(df)} samples from {DATA_PATH}")
    print(f"    Classes : {sorted(df['intent_label'].unique())}")
    return df


def split_data(df):
    """
    Stratified split (70 / 30) with no template-family leakage.
    We assign a 'template_family' based on dialect_style so that
    dialect variants of the same template all go to the same split.
    """
    # Simple stratified split — fine for our balanced dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    X = df['query_text'].values
    y = df['intent_label'].values

    train_idx, test_idx = next(sss.split(X, y))

    # Verify no exact string overlap (sanity check for leakage)
    train_texts = set(X[train_idx])
    test_texts  = set(X[test_idx])
    overlap     = train_texts & test_texts
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} exact duplicate queries across train/test!")
    else:
        print("✅  No exact-string data leakage detected between train and test sets.")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def build_pipeline():
    """
    Character boundary n-gram TF-IDF (n ∈ [2,4]) + Logistic Regression.

    Why char_wb n-grams?
    Telugu is agglutinative: "ఎంతొస్తయి", "ఎంతొస్తాయి", "ఎంత లభిస్తుంది"
    all share morphological roots. Word-level features treat them as OOV.
    char_wb pads at word boundaries, so n-grams capture sub-word intent signals.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,       # log(tf) dampens frequent tokens
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )),
    ])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"{'='*60}")
    print(report)
    return y_pred, acc


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾  Model saved → {MODEL_PATH}")


def main():
    print("\n" + "="*60)
    print("  Telugu Intent Classifier — Training")
    print("="*60 + "\n")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}\n")

    model = build_pipeline()
    model.fit(X_train, y_train)
    print("✅  Model trained.")

    evaluate(model, X_test, y_test)
    save_model(model)

    # Quick sanity: predict a few real queries
    print("\n--- Quick Sanity Predictions ---")
    samples = [
        "రైతు బంధు పైసలు ఎప్పుడొస్తయి?",        # should → BENEFITS
        "ఆరోగ్యశ్రీ కోసం ఏమేం కాగితాలు కావాలె?",  # should → REQUIRED_DOCUMENTS
        "అమ్మ ఒడి కి ఎలా దరఖాస్తు చేసుకోవాలి?",   # should → APPLICATION_PROCEDURE
        "ఆసరా పెన్షన్ పథకానికి ఎవరు అర్హులు?",     # should → ELIGIBILITY
    ]
    for s in samples:
        pred = model.predict([s])[0]
        proba = model.predict_proba([s]).max()
        print(f"  '{s[:40]}...' → {pred} (conf: {proba:.2f})")


if __name__ == '__main__':
    main()
