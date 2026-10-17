import sys, json, os, pickle
sys.stdout.reconfigure(encoding='utf-8')

d   = json.load(open('data/schemes_data.json', encoding='utf-8'))
lex = json.load(open('data/pan_telugu_civic_lexicon.json', encoding='utf-8'))
voc = json.load(open('data/civic_vocab.json', encoding='utf-8'))
m   = pickle.load(open('models/intent_model.pkl', 'rb'))

print(f"schemes_data.json  : {len(d)} chunks  | categories: {sorted(set(x['category'] for x in d))}")
print(f"lexicon            : {len(lex)} dialect mappings")
print(f"civic_vocab        : {len(voc)} canonical words")
pred = m.predict(['రైతు బంధు పైసలు ఎంత వస్తయి?'])[0]
print(f"intent_model.pkl   : loaded OK | sample prediction = {pred}")
print("\n✅  ALL FILES VALID")
