import unicodedata
import json
import os
import Levenshtein

# Resolve data paths relative to this file's location (works from any cwd)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, '..', 'data')

DEFAULT_LEXICON = os.path.join(_DATA_DIR, 'pan_telugu_civic_lexicon.json')
DEFAULT_VOCAB   = os.path.join(_DATA_DIR, 'civic_vocab.json')


class TeluguDialectNormalizer:
    def __init__(self, lexicon_path=DEFAULT_LEXICON, vocab_path=DEFAULT_VOCAB):
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            self.dialect_map = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.civic_vocab = set(json.load(f))

        # Protected scheme entity tokens — never alter these
        self.protected = {
            'రైతు', 'బంధు', 'భరోసా', 'ఆరోగ్యశ్రీ', 'అమ్మ', 'ఒడి',
            'ఆసరా', 'పెన్షన్', 'పథకం', 'చేయూత', 'కార్డు', 'జగనన్న'
        }

        # Phonetic Folding: collapses aspirates → unaspirates before edit-distance
        # This ensures పతకం == పథకం after folding (0 distance)
        self.fold = str.maketrans({
            'థ': 'త', 'ధ': 'ద', 'భ': 'బ', 'ఫ': 'ప',
            'ఖ': 'క', 'ఘ': 'గ', 'ఛ': 'చ', 'ఝ': 'జ',
            'ఠ': 'ట', 'ఢ': 'డ'
        })

        # Telugu case suffixes — stripped before fuzzy matching so
        # పతకానికి (suffix: ానికి) → పత → fuzzy matches పథకం
        # (same list used by Member 3's BM25 stemmer for consistency)
        self.suffixes = sorted([
            "లతో", "లకు", "లోని", "ానికి", "ాన్ని", "ంలో", "ంకు", "ంగా",
            "ాని", "తో", "కు", "కి", "లో", "ను", "ని", "ల",
        ], key=len, reverse=True)  # longest first so greedy match

    def strip_suffix(self, token: str) -> str:
        """Strip one Telugu case suffix to expose root for fuzzy matching."""
        for s in self.suffixes:
            if token.endswith(s) and len(token) - len(s) >= 3:
                return token[:-len(s)]
        return token

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        # Fix English transliterations emitted by Whisper ASR
        translit = {
            'rythu': 'రైతు', 'bandhu': 'బంధు',
            'aarogyasri': 'ఆరోగ్యశ్రీ', 'amma vodi': 'అమ్మ ఒడి'
        }
        for eng, tel in translit.items():
            text = text.lower().replace(eng, tel)
        # Fix Sandhi compound splits
        text = text.replace('రైతుబంధు', 'రైతు బంధు').replace('అమ్మఒడి', 'అమ్మ ఒడి')
        return ' '.join(text.strip().split())

    def normalize(self, query: str) -> str:
        tokens = self.clean_text(query).split()
        out = []
        for t in tokens:
            clean_t = t.rstrip('?,.!')
            punct   = t[len(clean_t):]

            # Rule 1: Protected scheme entity — never alter
            if clean_t in self.protected:
                out.append(clean_t + punct)
                continue

            # Rule 2: Exact regional dialect lexicon match
            if clean_t in self.dialect_map:
                out.append(self.dialect_map[clean_t] + punct)
                continue

            # Rule 3: Phonetic Folding + suffix-stripping + Levenshtein
            # Step 3a: strip case suffix to get candidate root
            root = self.strip_suffix(clean_t)
            # Step 3b: fold aspirates in root
            folded_root = root.translate(self.fold)

            best, best_sim = None, 0.0
            for cand in self.civic_vocab:
                folded_cand = cand.translate(self.fold)
                dist = Levenshtein.distance(folded_root, folded_cand)
                sim  = 1.0 - (dist / max(len(folded_root), len(folded_cand), 1))
                if sim > best_sim:
                    best_sim, best = sim, cand

            # Conservative threshold — high enough to avoid false positives
            if best_sim >= 0.75 and best and abs(len(root) - len(best)) <= 2:
                out.append(best + punct)
            else:
                out.append(t)

        return ' '.join(out)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    norm = TeluguDialectNormalizer()
    tests = [
        ('రైతు బంధు పైసలు ఎప్పుడొస్తయి?',
         'Expected: పైసలు→డబ్బులు, ఎప్పుడొస్తయి→ఎప్పుడు అందుతాయి'),
        ('ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?',
         'Expected: పతకం→పథకం (Phonetic Fold), కాగితాలు→పత్రాలు, కావాలె→కావాలి'),
        ('అమ్మ ఒడి పతకానికి ఎవరు అర్హులు?',
         'Expected: పతకానికి→పథకం (suffix strip + fold)'),
        ('ఆసరా పెన్షన్ పతకంలో ఎంత వస్తుంది?',
         'Expected: పతకంలో→పథకం (suffix strip + fold)'),
        ('rythu bandhu దరఖాస్తు ఎట్ల చేసుకోవాలె?',
         'Expected: rythu→రైతు, bandhu→బంధు, ఎట్ల→ఎలా'),
        ('ఈ పథకం సొమ్ము యాడ ఇస్తారు?',
         'Expected: సొమ్ము→ఆర్థిక సహాయం, యాడ→ఎక్కడ'),
        ('సర్కారు ఆసరా పెన్షన్ పైసలు ఎప్పుడొస్తయి?',
         'Expected: సర్కారు→ప్రభుత్వం, పైసలు→డబ్బులు'),
        ('అమ్మ ఒడి పతకం కింద ఏమేం డాక్యుమెంట్లు కావాలె?',
         'Expected: పతకం→పథకం, డాక్యుమెంట్లు→పత్రాలు'),
    ]
    print("=" * 65)
    print("  TeluguDialectNormalizer — Verification Tests")
    print("=" * 65)
    for q, note in tests:
        out = norm.normalize(q)
        print(f'  IN : {q}')
        print(f'  OUT: {out}')
        print(f'  NOTE: {note}')
        print()
