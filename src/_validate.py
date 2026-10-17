"""Quick integrity check - works from any directory:  python src/_validate.py"""
import json, os, pickle, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
P = lambda *a: os.path.join(ROOT, *a)
sys.path.insert(0, os.path.join(ROOT, 'src'))

kb = json.load(open(P('data', 'schemes_data.json'), encoding='utf-8'))
need = {'doc_id', 'scheme', 'category', 'source_url', 'content'}
assert all(need <= set(c) for c in kb), 'KB chunk missing a field'
assert len({c['doc_id'] for c in kb}) == len(kb), 'duplicate doc_id'
lex = json.load(open(P('data', 'pan_telugu_civic_lexicon.json'), encoding='utf-8'))
voc = json.load(open(P('data', 'civic_vocab.json'), encoding='utf-8'))
assert len(voc) == len(set(voc)), 'duplicate vocab entries'
m = pickle.load(open(P('models', 'intent_model.pkl'), 'rb'))
card = json.load(open(P('models', 'model_card.json'), encoding='utf-8'))
import sklearn
if card['sklearn_version'] != sklearn.__version__:
    print(f"WARNING: model trained with sklearn {card['sklearn_version']}, running {sklearn.__version__} -> retrain")
from intent_api import IntentService
r = IntentService().predict('రైతు బంధు పైసలు ఎప్పుడొస్తయి?')
print(f"KB chunks {len(kb)} | lexicon {len(lex)} | vocab {len(voc)} | sample -> {r['intent']} ({r['confidence']:.2f}) | {r['normalized']}")
print('ALL FILES VALID')
