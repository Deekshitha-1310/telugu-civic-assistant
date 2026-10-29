"""
dialect_normalizer.py  (v2)  —  Member 2
=========================================
Pan-Telugu dialect + ASR-spelling normalizer.

Design principle (the main change from v1)
------------------------------------------
The normalizer must REPAIR ASR / dialect variants of known civic words, but must
NEVER rewrite valid Telugu into a different word.  v1 used suffix-stripping + a
0.75 Levenshtein threshold on *every* unknown token and therefore turned
ఆఫ్‌లైన్->ఆన్‌లైన్, తీసుకోవాలి->తెలుసుకోవాలి, ఇస్తారు->చేస్తారు, అర్హులా->అర్హులు ...

Pipeline per token (first rule that fires wins)
  0. Phrase lexicon  (multi-word dialect phrases, longest match first)
  1. Known word      (protected scheme words / civic vocab / KB words) -> keep
  2. Token lexicon   (Telangana / Rayalaseema / colloquial slang)
  3. Phonetic key    (aspirate / sibilant / vowel-length folding; must be UNIQUE)
  4. Suffix + key    (case suffix is kept, only the root is repaired)
  5. Edit-distance<=1 on phonetic keys (optional, long tokens only, unique, guarded)
"""
import json
import os
import re
import unicodedata

import Levenshtein

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, '..', 'data')
DEFAULT_LEXICON = os.path.join(_DATA_DIR, 'pan_telugu_civic_lexicon.json')
DEFAULT_VOCAB = os.path.join(_DATA_DIR, 'civic_vocab.json')
DEFAULT_KB = os.path.join(_DATA_DIR, 'schemes_data.json')

INVISIBLE = re.compile('[\u200b\u200c\u200d\ufeff]')           # ZWSP / ZWNJ / ZWJ / BOM
TELUGU = re.compile('[\u0c00-\u0c7f]')
LEAD_PUNCT = '"\'“”‘’([{'
TRAIL_PUNCT = '?.,!;:।)]}"\'“”‘’'

# ── Phonetic folding tables ──────────────────────────────────────────────────
ASPIRATES = {'థ': 'త', 'ధ': 'ద', 'భ': 'బ', 'ఫ': 'ప', 'ఖ': 'క',
             'ఘ': 'గ', 'ఛ': 'చ', 'ఝ': 'జ', 'ఠ': 'ట', 'ఢ': 'డ'}
SIBILANTS = {'శ': 'స', 'ష': 'స'}
RETROFLEX = {'ణ': 'న', 'ళ': 'ల', 'ఱ': 'ర'}
VOWEL_LEN = {'ీ': 'ి', 'ూ': 'ు', 'ే': 'ె', 'ో': 'ొ',
             'ఈ': 'ఇ', 'ఊ': 'ఉ', 'ఏ': 'ఎ', 'ఓ': 'ఒ', 'ఆ': 'అ'}

FOLD_TABLES = {
    'none': {},
    'aspirate': ASPIRATES,                                       # what v1 did
    'full': {**ASPIRATES, **SIBILANTS, **RETROFLEX, **VOWEL_LEN},
}

# Case suffixes: the suffix is KEPT, only the root is repaired.
SUFFIXES = sorted(['ానికి', 'ాన్ని', 'లకు', 'లతో', 'లోని', 'తో', 'కు', 'కి',
                   'లో', 'ను', 'ని', 'ల'], key=len, reverse=True)
# Extra dialect particles that the morph rules don't cover
DIALECT_PARTICLES = [
    ('కంటె', 'కంటే'),   # colloquial comparative
    ('కంటా', 'కంటే'),   # colloquial comparative
    ('తడి', 'తాడి'),     # speech-level adverb
    ('తది', 'తాడి'),     # dialect root
    ('పైకి', 'పైకి'),    # intact (already standard)
    ('కి', 'కు'),         # generic dative (variable usage)
    ('నందు', 'లో'),       # locative in some dialects
    ('పైకి', 'పైకి'),    # intact (already standard)
]
# Extra dialect particles that the morph rules don't cover
DIALECT_PARTICLES = [
    ('కంటె', 'కంటే'),   # colloquial comparative
    ('కంటా', 'కంటే'),   # colloquial comparative
    ('తడి', 'తాడి'),     # speech-level adverb
    ('తది', 'తాడి'),     # dialect root
    ('పైకి', 'పైకి'),    # intact (already standard)
    ('కి', 'కు'),         # generic dative (variable usage)
    ('నందు', 'లో'),       # locative in some dialects
    ('పైకి', 'పైకి'),    # intact (already standard)
]


# Productive dialect verb endings -> standard endings.  These generalise the enumerated
# lexicon (చేయించాలె, చేసుకోవాలె, కావాలె ... all end in ాలె).  Patterns are attested in
# Telangana / Rayalaseema speech but SHOULD BE REVIEWED BY A NATIVE SPEAKER before the viva.
MORPH_RULES = [
    (re.compile(r'^(.{2,})ాలె$'), r'\1ాలి'),          # కావాలె  -> కావాలి
    (re.compile(r'^(.{2,})స్తయి$'), r'\1స్తాయి'),       # ఇస్తయి  -> ఇస్తాయి
    (re.compile(r'^(.{2,})తయి$'), r'\1తాయి'),          # అందుతయి -> అందుతాయి
    (re.compile(r'^(.{2,})తదా$'), r'\1తుందా'),         # వస్తదా  -> వస్తుందా
    (re.compile(r'^(.{2,})([తట])ది$'), r'\1\2ుంది'),    # వస్తది / ఉంటది -> వస్తుంది / ఉంటుంది
    (re.compile(r'^(.{2,})([తట])ాది$'), r'\1\2ుంది'),   # ఉంటాది / వస్తాది (Rayalaseema)
]


def strip_invisible(s: str) -> str:
    return INVISIBLE.sub('', unicodedata.normalize('NFC', s))


def split_punct(tok: str):
    """'"పైసలు"?' -> ('"', 'పైసలు', '"?')"""
    i, j = 0, len(tok)
    while i < j and tok[i] in LEAD_PUNCT:
        i += 1
    while j > i and tok[j - 1] in TRAIL_PUNCT:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


class TeluguDialectNormalizer:
    def __init__(self, lexicon_path=DEFAULT_LEXICON, vocab_path=DEFAULT_VOCAB,
                 kb_path=DEFAULT_KB, use_lexicon=True, fold_level='full',
                 fuzzy=False, fuzzy_max_dist=1, fuzzy_min_len=6):
        self.use_lexicon = use_lexicon
        self.fold_level = fold_level
        self.fold = str.maketrans(FOLD_TABLES[fold_level])
        self.fuzzy = fuzzy
        self.fuzzy_max_dist = fuzzy_max_dist
        self.fuzzy_min_len = fuzzy_min_len

        with open(lexicon_path, encoding='utf-8') as f:
            raw_lex = json.load(f)
        with open(vocab_path, encoding='utf-8') as f:
            vocab = json.load(f)

        # Scheme words that must never be altered
        self.protected = {strip_invisible(w) for w in [
            'రైతు', 'బంధు', 'భరోసా', 'ఆరోగ్యశ్రీ', 'అమ్మ', 'ఒడి', 'ఆసరా',
            'పెన్షన్', 'పథకం', 'చేయూత', 'కార్డు', 'జగనన్న']}

        # Lexicon: single-token keys and multi-token phrase keys
        self.lex1, self.lexN = {}, {}
        for k, v in raw_lex.items():
            key = tuple(strip_invisible(k).split())
            (self.lex1.__setitem__(key[0], v) if len(key) == 1
             else self.lexN.__setitem__(key, v))
        self.max_phrase = max([len(k) for k in self.lexN] + [1])

        self.vocab = {strip_invisible(w) for w in vocab}
        self.targets = self.vocab | self.protected      # words we can repair TO

        # Words known to be valid (never fuzzy-rewritten): vocab, lexicon, KB text
        self.known = set(self.targets) | set(self.lex1) | {
            strip_invisible(w) for v in raw_lex.values() for w in v.split()}
        if kb_path and os.path.exists(kb_path):
            with open(kb_path, encoding='utf-8') as f:
                for c in json.load(f):
                    for w in re.split(r'\s+', c.get('content', '')):
                        self.known.add(split_punct(strip_invisible(w))[1])

        # Phonetic-key index (only unambiguous keys are usable)
        idx = {}
        for w in self.targets:
            idx.setdefault(self.key(w), set()).add(w)
        self.ambiguous = {k: v for k, v in idx.items() if len(v) > 1}
        self.key_index = {k: next(iter(v)) for k, v in idx.items() if len(v) == 1}
        # stem index for words ending in anusvara: పథకం + ానికి -> పథకానికి
        self.stem_index = {}
        for w in self.targets:
            if w.endswith('ం') and len(w) > 3:
                self.stem_index.setdefault(self.key(w[:-1]), set()).add(w[:-1])
        self.stem_index = {k: next(iter(v)) for k, v in self.stem_index.items()
                           if len(v) == 1 and k not in self.key_index}

    # ── helpers ──────────────────────────────────────────────────────────────
    def key(self, w: str) -> str:
        return strip_invisible(w).translate(self.fold)

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        translit = {'rythu': 'రైతు', 'raithu': 'రైతు', 'bandhu': 'బంధు',
                    'bharosa': 'భరోసా', 'aarogyasri': 'ఆరోగ్యశ్రీ',
                    'arogyasri': 'ఆరోగ్యశ్రీ', 'amma': 'అమ్మ', 'vodi': 'ఒడి',
                    'aasara': 'ఆసరా', 'pension': 'పెన్షన్', 'card': 'కార్డు'}
        # whole Latin words only (v1 used str.replace and could hit substrings)
        text = re.sub(r'[A-Za-z]+',
                      lambda m: translit.get(m.group(0).lower(), m.group(0)), text)
        for a, b in [('రైతుబంధు', 'రైతు బంధు'), ('రైతుభరోసా', 'రైతు భరోసా'),
                     ('అమ్మఒడి', 'అమ్మ ఒడి')]:
            text = text.replace(a, b)
        return ' '.join(text.split())

    # ── single-token resolution ──────────────────────────────────────────────
    def resolve(self, core: str):
        """Returns (new_token, rule_name). rule_name is None if unchanged."""
        c = strip_invisible(core)
        if c in self.protected or c in self.targets:
            return core, None
        if self.use_lexicon and c in self.lex1:
            return self.lex1[c], 'lexicon'
        if len(c) < 3 or not TELUGU.search(c):
            return core, None

        # 1. Explicit dialect particle / comparison particle
        for dialect, standard in DIALECT_PARTICLES:
            if c == dialect:
                return standard, 'dialect-particle'

        k = self.key(c)
        w = self.key_index.get(k)
        if w:
            return w, 'phonetic'
        # NOTE: repairing a dropped final anusvara (పథక -> పథకం) was tried and REJECTED: it
        # also rewrites valid bound forms (ప్రభుత్వ పాఠశాల, వైద్య సేవలు, కుటుంబ సభ్యులు).
        # productive dialect verb endings (ాలె, స్తయి, తది ...)
        if self.use_lexicon and c not in self.known:
            for pat, rep in MORPH_RULES:
                if pat.match(c):
                    return pat.sub(rep, c), 'dialect-morph'
        # suffix route: keep suffix, repair root
        for s in SUFFIXES:
            if c.endswith(s) and len(c) - len(s) >= 3:
                root = c[:-len(s)]
                if root in self.targets:
                    return core, None
                kr = self.key(root)
                fixed = (self.key_index.get(kr) or self.stem_index.get(kr) or '')
                if fixed and fixed + s != c:                 # only report real changes
                    return fixed + s, 'phonetic+suffix'
                break
        # guarded edit-distance tier
        if self.fuzzy and c not in self.known and len(k) >= self.fuzzy_min_len:
            cands = [w for kk, w in self.key_index.items()
                     if abs(len(kk) - len(k)) <= self.fuzzy_max_dist
                     and Levenshtein.distance(k, kk) <= self.fuzzy_max_dist]
            if len(cands) == 1:
                return cands[0], 'fuzzy'
        return core, None

    # ── public API ───────────────────────────────────────────────────────────
    def explain(self, query: str):
        """Returns (normalized_text, [(original, replacement, rule), ...])."""
        parts = [split_punct(t) for t in self.clean_text(query).split()]
        out, edits, i = [], [], 0
        while i < len(parts):
            hit = False
            if self.use_lexicon:
                for n in range(min(self.max_phrase, len(parts) - i), 1, -1):
                    seg = parts[i:i + n]
                    if any(p[2] for p in seg[:-1]) or any(p[0] for p in seg[1:]):
                        continue            # punctuation inside the phrase
                    key = tuple(strip_invisible(p[1]) for p in seg)
                    if key in self.lexN:
                        new = self.lexN[key]
                        out.append(seg[0][0] + new + seg[-1][2])
                        edits.append((' '.join(key), new, 'lexicon-phrase'))
                        i += n
                        hit = True
                        break
            if hit:
                continue
            lead, core, trail = parts[i]
            new, rule = self.resolve(core)
            out.append(lead + new + trail)
            if rule:
                edits.append((core, new, rule))
            i += 1
        return ' '.join(out), edits

    def normalize(self, query: str) -> str:
        return self.explain(query)[0]


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    norm = TeluguDialectNormalizer()
    for q in ['రైతు బంధు పైసలు ఎప్పుడొస్తయి?',
              'ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?',
              'అమ్మ ఒడి పతకానికి ఎవరు అర్హులు?',
              'rythu bandhu దరఖాస్తు ఎట్ల చేసుకోవాలె?',
              'ఆసరా పెన్షన్ ఆఫ్‌లైన్ దరఖాస్తు ఎలా?',
              'ఆరోగ్యశ్రీ కార్డు స్టేటస్ ఎట్ల చెక్ చేయాలె?']:
        out, edits = norm.explain(q)
        print(f'IN : {q}\nOUT: {out}\n     {edits}\n')
