"""
evaluate_nlp.py (v2) — Member 2: Experiments 2 & 3
===================================================
EXPERIMENT 2  Normalizer ablation
  Token-level repair of simulated ASR spelling errors (asr_noise.py) + false-positive
  rate on CLEAN text.  Configurations:
     1 lexicon only                           (no spelling repair)
     2 v1-style Levenshtein, no folding       (plain Unicode edit distance, sim>=0.75)
     3 v1 as originally shipped               (aspirate fold + suffix-strip + sim>=0.75)
     4 v2 phonetic key, aspirate-only fold
     5 v2 phonetic key, full fold   <- shipped default
     6 v2 full fold + guarded Levenshtein<=1
  plus a sentence-level dialect-lexicon regression suite (data/ablation_probes.csv).

EXPERIMENT 3  Intent macro-F1 under dialect shift and ASR noise
  Two training protocols:  A) all styles  B) standard-only (dialect never seen)
  Conditions: clean | raw dialect | v1-normalized | v2-normalized | noisy (+/- v2).
  95 % bootstrap CIs, and fixed/broken counts, because n = 40 queries.
"""
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Levenshtein
from sklearn.metrics import f1_score

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, '..')
sys.path.insert(0, _SRC)
from dialect_normalizer import (TeluguDialectNormalizer, FOLD_TABLES, strip_invisible,
                                split_punct, TELUGU)
from asr_noise import variants, corrupt_text
from train_intent_model import build_pipeline

DATA = os.path.join(_ROOT, 'data')
PLOTS = os.path.join(_ROOT, 'outputs', 'plots')
os.makedirs(PLOTS, exist_ok=True)
RULES_SPELLING = {'phonetic', 'phonetic+suffix', 'fuzzy'}


# ── v1 re-implementation (baseline only; this is the code that shipped before) ──
LEGACY_SUFFIXES = sorted(["లతో", "లకు", "లోని", "ానికి", "ాన్ని", "ంలో", "ంకు", "ంగా",
                          "ాని", "తో", "కు", "కి", "లో", "ను", "ని", "ల"], key=len, reverse=True)


class LegacyNormalizer:
    def __init__(self, base: TeluguDialectNormalizer, fold_level, suffix_strip):
        self.b, self.suffix_strip = base, suffix_strip
        self.fold = str.maketrans(FOLD_TABLES[fold_level])

    def resolve(self, core):
        b = self.b
        if core in b.protected:
            return core, None
        if core in b.lex1:
            return b.lex1[core], 'lexicon'
        root = core
        if self.suffix_strip:
            for s in LEGACY_SUFFIXES:
                if core.endswith(s) and len(core) - len(s) >= 3:
                    root = core[:-len(s)]
                    break
        fr = root.translate(self.fold)
        best, bs = None, 0.0
        for cand in b.vocab:
            fc = cand.translate(self.fold)
            sim = 1 - Levenshtein.distance(fr, fc) / max(len(fr), len(fc), 1)
            if sim > bs:
                bs, best = sim, cand
        if bs >= 0.75 and best and abs(len(root) - len(best)) <= 2:
            return best, 'legacy-fuzzy'
        return core, None

    def explain(self, q):
        out, edits = [], []
        for t in self.b.clean_text(q).split():
            lead, core, trail = split_punct(t)
            new, rule = self.resolve(core)
            out.append(lead + new + trail)
            if rule:
                edits.append((core, new, rule))
        return ' '.join(out), edits

    def normalize(self, q):
        return self.explain(q)[0]


def make_configs():
    mk = TeluguDialectNormalizer
    base = mk(fold_level='full')
    return {
        '1 lexicon only':               mk(fold_level='none', fuzzy=False),
        '2 plain Levenshtein (no fold)': LegacyNormalizer(base, 'none', False),
        '3 v1 as shipped':              LegacyNormalizer(base, 'aspirate', True),
        '4 v2 key: aspirate fold':      mk(fold_level='aspirate'),
        '5 v2 key: full fold (SHIPPED)': mk(fold_level='full'),
        '6 v2 full fold + Lev<=1':      mk(fold_level='full', fuzzy=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2
# ══════════════════════════════════════════════════════════════════════════════
def clean_tokens(base):
    """Valid Telugu tokens the normalizer must leave alone (not lexicon keys)."""
    df = pd.read_csv(os.path.join(DATA, 'intent_dataset.csv'), encoding='utf-8-sig')
    ho = pd.read_csv(os.path.join(DATA, 'heldout_intent_queries.csv'), encoding='utf-8-sig')
    texts = list(df[df.dialect_style == 'standard'].query_text) + list(ho.query_standard)
    kb = json.load(open(os.path.join(DATA, 'schemes_data.json'), encoding='utf-8'))
    texts += [c['content'] for c in kb]
    toks = set()
    for t in texts:
        for w in t.split():
            core = split_punct(strip_invisible(w))[1]
            if len(core) >= 3 and TELUGU.search(core) and core not in base.lex1:
                toks.add(core)
    return sorted(toks)


def run_experiment2():
    print('\n' + '=' * 72 + '\n  EXPERIMENT 2: Normalizer ablation (simulated ASR spelling errors)\n' + '=' * 72)
    cfgs = make_configs()
    base = cfgs['5 v2 key: full fold (SHIPPED)']
    words = sorted(base.targets - {'చేయూత'})
    probes = [(cat, orig, bad) for orig in words for cat, bad in variants(orig)
              if bad not in base.known]                       # skip corruptions that are real words
    # suffix-attached probes: aspirate/sibilant/vowel corruption of root + 'లో'
    suf = [(f'{cat}+suffix', orig + 'లో', bad + 'లో') for orig in words
           for cat, bad in variants(orig)
           if cat in ('aspirate', 'sibilant', 'vowel_len') and not orig.endswith('ా')
           and (bad + 'లో') not in base.known]
    probes += suf
    cats = sorted({c for c, _, _ in probes})
    print(f'  {len(probes)} corrupted tokens from {len(words)} civic words; categories: {cats}')
    clean = clean_tokens(base)
    print(f'  {len(clean)} distinct clean Telugu tokens for the false-positive check')

    rows = []
    for name, norm in cfgs.items():
        rec = {c: [0, 0] for c in cats}
        for cat, orig, bad in probes:
            out = norm.normalize(bad)
            rec[cat][0] += (out == orig)
            rec[cat][1] += 1
        tot_ok = sum(v[0] for v in rec.values())
        fp = []
        for t in clean:
            out, edits = norm.explain(t)
            if out != t and any(r in RULES_SPELLING or r in ('legacy-fuzzy', 'dialect-morph') for _, _, r in edits):
                fp.append((t, out))
        row = {'config': name,
               'recovery_all': tot_ok / len(probes),
               **{f'rec_{c}': rec[c][0] / rec[c][1] for c in cats},
               'false_pos': len(fp), 'fp_rate': len(fp) / len(clean), '_fp': fp}
        rows.append(row)
    df = pd.DataFrame(rows)

    print(f"\n  {'config':32s} {'recovery':>9s} {'FP tokens':>10s} {'FP rate':>8s}")
    for r in rows:
        print(f"  {r['config']:32s} {r['recovery_all']*100:8.1f}% {r['false_pos']:10d} {r['fp_rate']*100:7.2f}%")
    print('\n  Recovery by corruption category (%):')
    print('  ' + 'config'.ljust(32) + ''.join(c[:11].rjust(12) for c in cats))
    for r in rows:
        print('  ' + r['config'].ljust(32) + ''.join(f"{r['rec_'+c]*100:11.0f}%" for c in cats))
    for r in rows:
        if r['config'].startswith(('3 ', '6 ')) and r['_fp']:
            print(f"\n  False positives in [{r['config']}] (first 8):")
            for a, b in r['_fp'][:8]:
                print(f'      {a}  ->  {b}')

    # sentence-level lexicon regression suite (in-sample: the lexicon was written for these)
    ab = pd.read_csv(os.path.join(DATA, 'ablation_probes.csv'), encoding='utf-8-sig')
    shipped, nofold = cfgs['5 v2 key: full fold (SHIPPED)'], cfgs['1 lexicon only']
    def ok(norm, r):
        o = norm.normalize(r.query)
        return (r.expected_token in o) and (r.dialect_token not in o.split())
    a_ship = np.mean([ok(shipped, r) for r in ab.itertuples()])
    a_lex = np.mean([ok(nofold, r) for r in ab.itertuples()])
    print(f'\n  Sentence-level lexicon regression ({len(ab)} probes, IN-SAMPLE, not evidence of '
          f'generalisation):  lexicon-only {a_lex*100:.1f}%  |  shipped {a_ship*100:.1f}%')

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [r['config'][:2] + r['config'][1:].strip()[:22] for r in rows]
    short = [r['config'].split(' ', 1)[0] for r in rows]
    cols = ['#95a5a6', '#e67e22', '#e74c3c', '#f1c40f', '#2ecc71', '#3498db']
    ax[0].bar(short, [r['recovery_all'] * 100 for r in rows], color=cols, edgecolor='black')
    for i, r in enumerate(rows):
        ax[0].text(i, r['recovery_all'] * 100 + 1, f"{r['recovery_all']*100:.0f}%", ha='center', fontweight='bold')
    ax[0].set_ylim(0, 110); ax[0].set_ylabel('Token recovery (%)'); ax[0].set_xlabel('configuration')
    ax[0].set_title('Recovery of simulated ASR spelling errors')
    ax[1].bar(short, [r['fp_rate'] * 100 for r in rows], color=cols, edgecolor='black')
    for i, r in enumerate(rows):
        ax[1].text(i, r['fp_rate'] * 100 + .3, f"{r['false_pos']}", ha='center', fontweight='bold')
    ax[1].set_ylabel('False-positive rate on clean tokens (%)'); ax[1].set_xlabel('configuration')
    ax[1].set_title('Valid words wrongly rewritten (count on bars)')
    fig.text(0.5, -0.04, '   '.join(r['config'] for r in rows[:3]) + '\n' + '   '.join(r['config'] for r in rows[3:]),
             ha='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS, 'exp2_phonetic_ablation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'\n  plot -> {path}')
    return df.drop(columns=['_fp'])


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3
# ══════════════════════════════════════════════════════════════════════════════
def boot_ci(y, p, n=2000, seed=0):
    rng, y, p = np.random.default_rng(seed), np.array(y), np.array(p)
    s = [f1_score(y[i], p[i], average='macro') for i in
         (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


def paired_boot(y, p_a, p_b, n=2000, seed=0):
    """CI for  F1(b) - F1(a)  (paired bootstrap over queries)."""
    rng, y, p_a, p_b = np.random.default_rng(seed), np.array(y), np.array(p_a), np.array(p_b)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], p_b[i], average='macro') - f1_score(y[i], p_a[i], average='macro'))
    return np.percentile(d, [2.5, 97.5])


def run_experiment3():
    print('\n' + '=' * 72 + '\n  EXPERIMENT 3: Intent macro-F1  (n = 40 hand-written held-out queries)\n' + '=' * 72)
    df = pd.read_csv(os.path.join(DATA, 'intent_dataset.csv'), encoding='utf-8-sig')
    ho = pd.read_csv(os.path.join(DATA, 'heldout_intent_queries.csv'), encoding='utf-8-sig')
    assert not set(ho.query_standard) & set(df.query_text)
    assert not set(ho.query_dialect) & set(df.query_text)
    cfg = make_configs()
    ident = type('Identity', (), {'normalize': staticmethod(lambda q: q)})()
    pipelines = {'no normalizer': ident,
                 'v1 normalizer': cfg['3 v1 as shipped'],
                 'v2 normalizer': cfg['5 v2 key: full fold (SHIPPED)']}
    v2 = pipelines['v2 normalizer']
    y = ho.intent_label.tolist()
    X_clean, X_dial = ho.query_standard.tolist(), ho.query_dialect.tolist()
    NOISE = (0.15, 0.30)
    protocols = {'A: train on ALL styles': df,
                 'B: train on STANDARD only (dialect unseen in training)': df[df.dialect_style == 'standard']}
    summary = {}
    for pname, tr in protocols.items():
        print(f'\n  Protocol {pname}   (train n = {len(tr)})')
        print(f"    {'pipeline (train & test both normalized)':40s} {'clean':>7s} {'dialect':>8s}  95% CI dialect   "
              + ''.join(f'noise{p:.2f}'.rjust(12) for p in NOISE))
        res, preds = {}, {}
        for pn, norm in pipelines.items():
            m = build_pipeline().fit([norm.normalize(q) for q in tr.query_text], tr.intent_label)
            pc = m.predict([norm.normalize(q) for q in X_clean])
            pd_ = m.predict([norm.normalize(q) for q in X_dial]); preds[pn] = pd_
            f_c, f_d = f1_score(y, pc, average='macro'), f1_score(y, pd_, average='macro')
            lo, hi = boot_ci(y, pd_)
            nz = {}
            for p_noise in NOISE:
                sc = []
                for seed in range(30):
                    rng = random.Random(seed)
                    noisy = [norm.normalize(corrupt_text(q, p_noise, rng)) for q in X_dial]
                    sc.append(f1_score(y, m.predict(noisy), average='macro'))
                nz[p_noise] = (np.mean(sc), np.std(sc))
            res[pn] = {'clean': f_c, 'dialect': (f_d, lo, hi), 'noise': nz}
            print(f'    {pn:40s} {f_c:7.3f} {f_d:8.3f}  [{lo:.3f}, {hi:.3f}]  '
                  + ''.join(f'{nz[p][0]:.3f}±{nz[p][1]:.3f}'.rjust(12) for p in NOISE))
        # the mistake v1 made: train raw, normalize only at test time
        m_raw = build_pipeline().fit(tr.query_text, tr.intent_label)
        pm = m_raw.predict([v2.normalize(q) for q in X_dial])
        print(f"    {'(mismatch) train RAW, test v2-normalized':40s} {'':7s} {f1_score(y, pm, average='macro'):8.3f}")
        raw_ok = np.array(preds['no normalizer']) == np.array(y)
        ok = np.array(preds['v2 normalizer']) == np.array(y)
        d_lo, d_hi = paired_boot(y, preds['no normalizer'], preds['v2 normalizer'])
        d = res['v2 normalizer']['dialect'][0] - res['no normalizer']['dialect'][0]
        print(f"    v2 vs none on dialect: fixed {int((~raw_ok & ok).sum())}, broken {int((raw_ok & ~ok).sum())}, "
              f"dF1 = {d:+.3f}  95% CI [{d_lo:+.3f}, {d_hi:+.3f}]")
        summary[pname] = res

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    groups = ['clean', 'dialect', 'noise 0.15', 'noise 0.30']
    cols = {'no normalizer': '#e74c3c', 'v1 normalizer': '#e67e22', 'v2 normalizer': '#2ecc71'}
    for ax, (pname, res) in zip(axs, summary.items()):
        for j, (pn, r) in enumerate(res.items()):
            vals = [r['clean'], r['dialect'][0], r['noise'][0.15][0], r['noise'][0.30][0]]
            err = [0, 0, r['noise'][0.15][1], r['noise'][0.30][1]]
            err[1] = None
            xs = np.arange(4) + (j - 1) * 0.27
            ax.bar(xs, [v * 100 for v in vals], 0.26, label=pn, color=cols[pn], edgecolor='black')
            ax.errorbar(xs[1], vals[1] * 100, yerr=[[ (vals[1]-r['dialect'][1])*100], [(r['dialect'][2]-vals[1])*100]],
                        color='black', capsize=3)
            ax.errorbar(xs[2:], [v * 100 for v in vals[2:]], yerr=[e * 100 for e in err[2:]], fmt='none', color='black', capsize=3)
        ax.set_xticks(range(4)); ax.set_xticklabels(groups); ax.set_ylim(40, 105)
        ax.set_title(pname, fontsize=10)
    axs[0].set_ylabel('Intent macro-F1 (%)'); axs[0].legend(loc='lower left', fontsize=8)
    plt.suptitle('Experiment 3: intent macro-F1, 40 held-out queries (bars: CI for dialect, std over 30 seeds for noise)', y=1.02, fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS, 'exp3_intent_macro_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'\n  plot -> {path}')
    return summary


def main():
    run_experiment2()
    run_experiment3()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
