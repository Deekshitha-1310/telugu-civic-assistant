import unicodedata
import json
import Levenshtein

class TeluguDialectNormalizer:
    def __init__(self, lexicon_path='telugu_civic_assistant/data/pan_telugu_civic_lexicon.json', vocab_path='telugu_civic_assistant/data/civic_vocab.json'):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            self.dialect_map = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.civic_vocab = set(json.load(f))
        self.protected = {'రైతు', 'బంధు', 'భరోసా', 'ఆరోగ్యశ్రీ', 'అమ్మ', 'ఒడి', 'ఆసరా', 'పెన్షన్', 'పథకం'}
        self.fold = str.maketrans({'థ': 'త', 'ధ': 'ద', 'భ': 'బ', 'ఫ': 'ప', 'ఖ': 'క', 'ఘ': 'గ', 'ఛ': 'చ', 'ఝ': 'జ', 'ఠ': 'ట', 'ఢ': 'డ'})

    def clean_text(self, text):
        text = unicodedata.normalize('NFC', text)
        translit = {'rythu': 'రైతు', 'bandhu': 'బంధు', 'aarogyasri': 'ఆరోగ్యశ్రీ', 'amma vodi': 'అమ్మ ఒడి'}
        for k, v in translit.items():
            text = text.lower().replace(k, v)
        text = text.replace('రైతుబంధు', 'రైతు బంధు').replace('అమ్మఒడి', 'అమ్మ ఒడి')
        return ' '.join(text.strip().split())

    def normalize(self, query):
        tokens = self.clean_text(query).split()
        out = []
        for t in tokens:
            clean_t = t.rstrip('?,.!')
            punct = t[len(clean_t):]
            if clean_t in self.protected:
                out.append(clean_t + punct)
                continue
            if clean_t in self.dialect_map:
                out.append(self.dialect_map[clean_t] + punct)
                continue
            folded = clean_t.translate(self.fold)
            best, best_sim = None, 0.0
            for c in self.civic_vocab:
                folded_c = c.translate(self.fold)
                dist = Levenshtein.distance(folded, folded_c)
                sim = 1.0 - (dist / max(len(folded), len(folded_c)))
                if sim > best_sim:
                    best_sim, best = sim, c
            if best_sim >= 0.75 and best and abs(len(clean_t) - len(best)) <= 2:
                out.append(best + punct)
            else:
                out.append(t)
        return ' '.join(out)

if __name__ == '__main__':
    norm = TeluguDialectNormalizer()
    tests = [
        'రైతు బంధు పైసలు ఎప్పుడొస్తయి?',
        'ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?',
        'rythu bandhu దరఖాస్తు ఎట్ల చేసుకోవాలె?',
        'ఈ పథకం సొమ్ము యాడ ఇస్తారు?'
    ]
    for q in tests:
        print(f'{q} -> {norm.normalize(q)}')
