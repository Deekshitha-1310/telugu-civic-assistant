import sys, urllib.request, urllib.parse, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

def google_transliterate(text):
    try:
        url = 'https://inputtools.google.com/request?' + urllib.parse.urlencode({
            'text': text,
            'itc': 'te-t-i0-und',
            'num': '1'
        })
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            if data[0] == 'SUCCESS' and data[1]:
                return ' '.join([item[1][0] for item in data[1]])
    except Exception as e:
        print('Transliterate err:', e)
    return text

raw_q = 'nenu kowlu raithu ni naaku dabulu vasthaya'
tel_q = google_transliterate(raw_q)
print('1. Raw Tanglish:        ', raw_q)
print('2. Neural Transliterated:', tel_q)

from dialect_normalizer import TeluguDialectNormalizer
from intent_api import IntentService
from hybrid_retriever import TeluguHybridRetriever
from simplifier import TeluguSimplifier

normer = TeluguDialectNormalizer()
norm_q, edits = normer.explain(tel_q)
print('3. Dialect Normalized:  ', norm_q)

intent_svc = IntentService()
intent_res = intent_svc.predict(norm_q, normalize=False)
print('4. Predicted Intent:    ', intent_res['intent'], f"confidence: {intent_res['confidence']:.2f}")

retriever = TeluguHybridRetriever()
top_docs = retriever.retrieve_hybrid(norm_q, top_k=2)
print('5. Top Retrieved Doc:   ', top_docs[0]['doc_id'] if top_docs else 'None')
print('   Top Doc Scheme:      ', top_docs[0]['scheme'] if top_docs else 'None')
print('   RRF Score:           ', top_docs[0]['rrf_score'] if top_docs else 0)

simplifier = TeluguSimplifier()
ans = simplifier.generate_response(norm_q, top_docs, intent_info=intent_res)
print('6. Grounded Answer:\n', ans['answer'])
