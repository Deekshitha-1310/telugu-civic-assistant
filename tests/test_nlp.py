"""Run from anywhere:  python tests/test_nlp.py   (or: pytest tests/)
Each test pins down a bug found in the v1 audit."""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from dialect_normalizer import TeluguDialectNormalizer          # noqa: E402

N = TeluguDialectNormalizer()
DF = pd.read_csv(os.path.join(ROOT, 'data', 'intent_dataset.csv'), encoding='utf-8-sig')
HO = pd.read_csv(os.path.join(ROOT, 'data', 'heldout_intent_queries.csv'), encoding='utf-8-sig')


def test_flagship_examples():
    assert N.normalize('రైతు బంధు పైసలు ఎప్పుడొస్తయి?') == 'రైతు బంధు డబ్బులు ఎప్పుడు అందుతాయి?'
    assert N.normalize('ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?') == 'ఆరోగ్యశ్రీ పథకం కింద ఏమేం పత్రాలు కావాలి?'
    assert N.normalize('rythu bandhu దరఖాస్తు ఎట్ల చేసుకోవాలె?') == 'రైతు బంధు దరఖాస్తు ఎలా చేసుకోవాలి?'


def test_suffix_is_kept_when_repairing_root():
    assert N.normalize('పతకానికి') == 'పథకానికి'            # v1 returned పథకం (dropped ానికి)
    assert N.normalize('పతకంలో') == 'పథకంలో'


def test_valid_words_are_never_rewritten():
    # v1 turned these into different words
    for w in ['ఆఫ్‌లైన్', 'తీసుకోవాలి', 'ఇస్తారు', 'అవుతుంది', 'అర్హులా', 'వస్తుందా', 'ఇస్తుంది']:
        assert N.normalize(w) == w, w


def test_ambiguous_dialect_words_not_in_lexicon():
    assert N.normalize('ఆడ రైతులు') == 'ఆడ రైతులు'          # ఆడ = "women" here, not "there"
    assert 'కౌలు రైతు' in N.normalize('కౌలు రైతు')           # v1 inserted "నిబంధన"
    assert 'నిబంధన' not in N.normalize('కౌలు రైతు')


def test_phrase_lexicon_and_punctuation():
    assert N.normalize('పైసలు పడ్డాయా లేదా?') == 'డబ్బులు జమ అయినాయా లేదా?'
    assert N.normalize('"పైసలు" ఎప్పుడొస్తయి।').startswith('"డబ్బులు"')


def test_idempotent():
    for q in list(DF.query_text) + list(HO.query_dialect):
        once = N.normalize(q)
        assert N.normalize(once) == once, q


def test_no_spelling_repair_on_clean_standard_text():
    for q in DF[DF.dialect_style == 'standard'].query_text:
        _, edits = N.explain(q)
        assert not [e for e in edits if e[2] in ('phonetic', 'phonetic+suffix', 'fuzzy', 'dialect-morph')], (q, edits)


def test_phonetic_keys_unambiguous():
    assert not N.ambiguous, N.ambiguous


def test_template_split_has_no_leakage():
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(1, test_size=.3, random_state=42).split(DF, groups=DF.template_id))
    assert not set(DF.template_id.iloc[tr]) & set(DF.template_id.iloc[te])


def test_heldout_disjoint_from_training():
    assert not set(HO.query_standard) & set(DF.query_text)
    assert not set(HO.query_dialect) & set(DF.query_text)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn(); print('PASS', name)
            except AssertionError as e:
                fails += 1; print('FAIL', name, e)
    sys.exit(1 if fails else 0)
