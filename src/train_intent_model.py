"""
train_intent_model.py (v2) — Member 2: Intent classifier
=========================================================
Char-boundary n-gram TF-IDF + Logistic Regression.

What changed from v1 (see CHANGES.md)
  * The split is by TEMPLATE FAMILY (GroupShuffleSplit on `template_id`).  v1 used a
    plain StratifiedShuffleSplit, so every test query had a same-template sibling in
    training (only the scheme name differed) and the 100 % score was leakage.
  * Grouped 5-fold CV + baselines (majority, word n-gram LR, char NB) are reported, so
    "char n-grams > word n-grams" is measured instead of asserted.
  * The classifier is trained on NORMALIZED text, because the deployed pipeline always
    normalizes before predicting (v1 trained on raw text and tested on normalized text;
    that train/test mismatch alone cost 2-5 F1 points - see CHANGES.md).
  * The saved model is refit on ALL data after evaluation (v1 shipped the 70 % model).
  * A model card (models/model_card.json) records versions + metrics + the confidence
    threshold used for the out-of-scope fallback.

Outputs: models/intent_model.pkl, models/model_card.json
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialect_normalizer import TeluguDialectNormalizer

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC_DIR, '..')
DATA_PATH = os.path.join(_ROOT, 'data', 'intent_dataset.csv')
MODEL_DIR = os.path.join(_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'intent_model.pkl')
CARD_PATH = os.path.join(MODEL_DIR, 'model_card.json')
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"Loaded {len(df)} samples, {df['template_id'].nunique()} template families")
    return df


def build_pipeline(kind='char_lr'):
    """kind: char_lr (shipped) | word_lr | char_nb | majority (baselines)."""
    if kind == 'majority':
        return Pipeline([('clf', DummyClassifier(strategy='most_frequent'))])
    if kind == 'word_lr':
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                              token_pattern=r'\S+', sublinear_tf=True)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    elif kind == 'char_nb':
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        clf = MultinomialNB(alpha=0.1)
    else:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                              max_features=50_000, sublinear_tf=True)
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                                 random_state=SEED)
    return Pipeline([('tfidf', vec), ('clf', clf)])


def grouped_cv(df, kind, k=5):
    X, y, g = df['query_text'].values, df['intent_label'].values, df['template_id'].values
    s = []
    for tr, te in GroupKFold(k).split(X, y, g):
        m = build_pipeline(kind).fit(X[tr], y[tr])
        s.append(f1_score(y[te], m.predict(X[te]), average='macro'))
    return float(np.mean(s)), float(np.std(s))


def main():
    df = load_data()
    norm = TeluguDialectNormalizer()
    df['query_text'] = [norm.normalize(q) for q in df['query_text']]   # same preprocessing as inference
    X, y, g = df['query_text'].values, df['intent_label'].values, df['template_id'].values

    # 1) Template-grouped hold-out split (no template appears on both sides)
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(X, y, g))
    assert not set(g[tr]) & set(g[te]), 'template leakage!'
    print(f"Train {len(tr)} / Test {len(te)}  |  shared templates: 0")
    m = build_pipeline().fit(X[tr], y[tr])
    pred = m.predict(X[te])
    print(f"\nGrouped hold-out  acc={accuracy_score(y[te], pred):.3f}  "
          f"macro-F1={f1_score(y[te], pred, average='macro'):.3f}")
    print(classification_report(y[te], pred, digits=3, zero_division=0))

    # 2) Grouped 5-fold CV + baselines (honest model comparison)
    print("Grouped 5-fold CV macro-F1 (mean ± std):")
    cv = {}
    for kind in ['majority', 'word_lr', 'char_nb', 'char_lr']:
        mu, sd = grouped_cv(df, kind)
        cv[kind] = {'mean': round(mu, 4), 'std': round(sd, 4)}
        print(f"  {kind:9s} {mu:.3f} ± {sd:.3f}")

    # 3) Final model: refit on ALL data, save with a model card
    final = build_pipeline().fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final, f)
    card = {'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'n_train': int(len(df)), 'classes': list(final.classes_),
            'features': "TfidfVectorizer(char_wb, ngram_range=(2,4), sublinear_tf)",
            'classifier': 'LogisticRegression(C=1.0)',
            'grouped_cv_macro_f1': cv,
            'grouped_holdout_macro_f1': round(float(f1_score(y[te], pred, average='macro')), 4),
            'trained_on_normalized_text': True, 'out_of_scope_threshold': 0.45}
    with open(CARD_PATH, 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {MODEL_PATH}  (+ model_card.json)")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
