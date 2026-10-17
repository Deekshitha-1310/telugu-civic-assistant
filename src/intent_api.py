"""
intent_api.py — the ONE function Member 3's pipeline should call.

    from intent_api import IntentService
    svc = IntentService()
    svc.predict("రైతు బంధు పైసలు ఎప్పుడొస్తయి?")
    -> {'normalized': 'రైతు బంధు డబ్బులు ఎప్పుడు అందుతాయి?', 'intent': 'BENEFITS',
        'confidence': 0.71, 'in_scope': True, 'edits': [...]}

If confidence < threshold the query is flagged `in_scope=False` so the RAG layer can
return the "information not available" fallback instead of guessing an intent.
"""
import json
import os
import pickle

from dialect_normalizer import TeluguDialectNormalizer

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


class IntentService:
    def __init__(self, model_path=None, card_path=None, normalizer=None):
        model_path = model_path or os.path.join(_ROOT, 'models', 'intent_model.pkl')
        card_path = card_path or os.path.join(_ROOT, 'models', 'model_card.json')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.threshold = 0.45
        if os.path.exists(card_path):
            with open(card_path, encoding='utf-8') as f:
                self.threshold = json.load(f).get('out_of_scope_threshold', 0.45)
        self.normalizer = normalizer or TeluguDialectNormalizer()

    def predict(self, text: str, normalize: bool = True):
        norm, edits = self.normalizer.explain(text) if normalize else (text, [])
        proba = self.model.predict_proba([norm])[0]
        i = int(proba.argmax())
        return {'normalized': norm, 'intent': str(self.model.classes_[i]),
                'confidence': float(proba[i]),
                'in_scope': bool(proba[i] >= self.threshold), 'edits': edits}
