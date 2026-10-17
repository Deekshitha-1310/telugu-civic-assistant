"""
asr_noise.py — simulated ASR spelling noise (Member 2)
=======================================================
Whisper-style phonetic confusions for Telugu, used to (a) enumerate repair probes for
Experiment 2 and (b) stress-test the intent classifier in Experiment 3.

IMPORTANT (state this in the report): this is a SIMULATION.  It is not real ASR output.
Replace/augment it with Member 1's real Whisper transcripts of queries_master.csv as soon
as they exist.
"""
import random
import re

_PAIRS = {
    'aspirate':  [('త', 'థ'), ('ద', 'ధ'), ('బ', 'భ'), ('క', 'ఖ'), ('ప', 'ఫ'),
                  ('గ', 'ఘ'), ('చ', 'ఛ'), ('జ', 'ఝ'), ('ట', 'ఠ'), ('డ', 'ఢ')],
    'sibilant':  [('స', 'శ'), ('స', 'ష'), ('శ', 'ష')],
    'retroflex': [('న', 'ణ'), ('ల', 'ళ')],
    'vowel_len': [('ి', 'ీ'), ('ు', 'ూ'), ('ె', 'ే'), ('ొ', 'ో')],
}
SWAP = {}
for cat, pairs in _PAIRS.items():
    for a, b in pairs:
        SWAP.setdefault(a, []).append((cat, b))
        SWAP.setdefault(b, []).append((cat, a))

# Categories the normalizer's fold table was NOT designed for (honest stress test)
OTHER = ['anusvara_drop', 'char_deletion']


def variants(word):
    """All single-edit corruptions of `word` -> list of (category, corrupted)."""
    out = []
    for i, ch in enumerate(word):
        for cat, rep in SWAP.get(ch, []):
            out.append((cat, word[:i] + rep + word[i + 1:]))
        if ch == 'ం':
            out.append(('anusvara_drop', word[:i] + word[i + 1:]))
        if i > 0 and re.match('[\u0c15-\u0c39]', ch) and i < len(word) - 1:
            out.append(('char_deletion', word[:i] + word[i + 1:]))
    return [(c, v) for c, v in dict.fromkeys(out) if v != word]


def corrupt_text(text, p, rng: random.Random, categories=None):
    """Corrupt each Telugu token (len>=4) with probability p using one random variant."""
    toks = []
    for t in text.split():
        core = t.rstrip('?.,!')
        tail = t[len(core):]
        if len(core) >= 4 and re.search('[\u0c00-\u0c7f]', core) and rng.random() < p:
            vs = [v for c, v in variants(core) if categories is None or c in categories]
            if vs:
                core = rng.choice(vs)
        toks.append(core + tail)
    return ' '.join(toks)
