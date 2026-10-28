import os
import sys
import json
import unicodedata

# Ensure UTF-8 output in Windows consoles without linter warnings
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

try:
    import Levenshtein
    def calc_lev_dist(s1: str, s2: str) -> int:
        return Levenshtein.distance(s1, s2)
except ImportError:
    # Pure Python Levenshtein fallback
    def calc_lev_dist(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return calc_lev_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr
        return prev[-1]


class TeluguDialectNormalizer:
    def __init__(self, lexicon_path=None, vocab_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if lexicon_path is None:
            lexicon_path = os.path.join(base_dir, "data", "pan_telugu_civic_lexicon.json")
        if vocab_path is None:
            vocab_path = os.path.join(base_dir, "data", "civic_vocab.json")

        self.dialect_map = {}
        if os.path.exists(lexicon_path):
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.dialect_map = json.load(f)

        self.civic_vocab = set()
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.civic_vocab = set(json.load(f))

        # Protected scheme entities that should not be mutated
        self.protected_entities = {
            "రైతు", "బంధు", "భరోసా", "ఆరోగ్యశ్రీ", "అమ్మ", "ఒడి", 
            "ఆసరా", "పెన్షన్", "చేయూత", "పథకం", "కార్డు"
        }

        # Phonetic Folding Table: Collapses aspirates to unaspirates
        self.phonetic_fold = str.maketrans({
            "థ": "త", "ధ": "ద", "భ": "బ", "ఫ": "ప", "ఖ": "క", 
            "ఘ": "గ", "ఛ": "చ", "ఝ": "జ", "ఠ": "ట", "ఢ": "డ"
        })

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        # Fix English transliterations if Whisper emits Latin script
        translit_map = {
            "rythu": "రైతు", "bandhu": "బంధు", 
            "aarogyasri": "ఆరోగ్యశ్రీ", "amma vodi": "అమ్మ ఒడి",
            "aasara": "ఆసరా", "pension": "పెన్షన్"
        }
        for eng, tel in translit_map.items():
            text = text.lower().replace(eng, tel)
        # Phrase-level replacement for compound words (Sandhi)
        text = text.replace("రైతుబంధు", "రైతు బంధు").replace("అమ్మఒడి", "అమ్మ ఒడి")
        return " ".join(text.strip().split())

    def normalize(self, query: str) -> str:
        cleaned = self.clean_text(query)
        # Strip trailing punctuation for token matching
        tokens = cleaned.replace("?", " ?").replace(".", "").replace(",", "").split()
        normalized_tokens = []

        for token in tokens:
            if token == "?":
                normalized_tokens.append(token)
                continue

            # Rule 1: Protected Scheme Entity -> Never alter
            if token in self.protected_entities:
                normalized_tokens.append(token)
                continue

            # Rule 2: Exact Regional Dialect Match
            if token in self.dialect_map:
                normalized_tokens.append(self.dialect_map[token])
                continue

            # Rule 3: Phonetic Folding + Levenshtein Matching
            folded_token = token.translate(self.phonetic_fold)
            best_cand, best_sim = None, 0.0

            for cand in self.civic_vocab:
                folded_cand = cand.translate(self.phonetic_fold)
                dist = calc_lev_dist(folded_token, folded_cand)
                max_len = max(len(folded_token), len(folded_cand))
                sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
                if sim > best_sim:
                    best_sim, best_cand = sim, cand

            # Conservative threshold: matches pathakam/patakam with 1.0 folded similarity
            if best_sim >= 0.75 and best_cand and abs(len(token) - len(best_cand)) <= 2:
                normalized_tokens.append(best_cand)
            else:
                normalized_tokens.append(token)

        result = " ".join(normalized_tokens)
        result = result.replace(" ?", "?")
        return result


if __name__ == "__main__":
    normalizer = TeluguDialectNormalizer()
    test_queries = [
        "రైతు బంధు పైసలు ఎప్పుడొస్తయి?",
        "ఆరోగ్యశ్రీ పతకం ఎలా దరఖాస్తు చేయాలె",
        "rythu bandhu డబ్బులు ఎప్పుడొస్తది"
    ]
    for q in test_queries:
        norm = normalizer.normalize(q)
        print(f"Original  : {q}")
        print(f"Normalized: {norm}\n")
