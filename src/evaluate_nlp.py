"""
evaluate_nlp.py  —  Member 2: Experiments 2 & 3
================================================
Runs TWO controlled experiments and produces console tables + saved plots.

Experiment 2: Dialect Normalization Ablation Study
----------------------------------------------------
Tests the normalizer on 50 dialectal queries.
Measures token-level correction accuracy WITH and WITHOUT Phonetic Folding.
Key verification: পথকం → పథకం fixed by Phonetic Folding alone.

Experiment 3: Intent Classification Macro-F1 Comparison
---------------------------------------------------------
Compares Intent Macro-F1 across three input conditions on held-out queries:
  1. Ground Truth (clean standard Telugu)
  2. Raw ASR-like text (dialectal slang, no normalization)
  3. Normalized text (after our normalizer)
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving figs
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# ── Path resolution ────────────────────────────────────────────────────────────
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.join(_SRC_DIR, '..')
MODEL_PATH = os.path.join(_ROOT, 'models',  'intent_model.pkl')
DATA_PATH  = os.path.join(_ROOT, 'data',    'intent_dataset.csv')
PLOTS_DIR  = os.path.join(_ROOT, 'outputs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Add src to import path so we can import the normalizer
sys.path.insert(0, _SRC_DIR)
from dialect_normalizer import TeluguDialectNormalizer


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 – Dialect Normalization Ablation
# ══════════════════════════════════════════════════════════════════════════════

# 50 dialectal probe queries with their expected normalized forms
# Each tuple: (dialect_input, expected_output_token_hint, scheme_class)
# We test specific token replacements that prove the normalizer works
ABLATION_PROBES = [
    # (dialect_query, expected_corrected_token, dialect_token_to_check)
    # Phonetic Folding probes  (aspirated ↔ unaspirated)
    ("ఆరోగ్యశ్రీ పతకం కింద ఏమేం కాగితాలు కావాలె?",  "పథకం", "పతకం"),
    ("రైతు బంధు పతకం అర్హులు ఎవరు?",                "పథకం", "పతకం"),
    ("అమ్మ ఒడి పతకానికి ఎవరు అర్హులు?",              "పథకం", "పతకం"),
    ("ఆసరా పెన్షన్ పతకంలో ఎంత వస్తుంది?",            "పథకం", "పతకం"),
    ("సర్కారు పతకం గురించి చెప్పండి",                  "పథకం", "పతకం"),

    # Lexicon mapping probes (Telangana dialect)
    ("రైతు బంధు పైసలు ఎప్పుడొస్తయి?",                "డబ్బులు", "పైసలు"),
    ("ఆరోగ్యశ్రీ పైసలు ఎప్పుడొస్తయి?",                "డబ్బులు", "పైసలు"),
    ("అమ్మ ఒడి పైసలు ఎంత వస్తయి?",                   "డబ్బులు", "పైసలు"),
    ("ఆసరా పెన్షన్ పైసలు పడ్డాయా?",                   "డబ్బులు", "పైసలు"),
    ("రైతు భరోసా పైసలు ఏ నెలలో వస్తయి?",              "డబ్బులు", "పైసలు"),

    # ఎప్పుడొస్తయి → ఎప్పుడు అందుతాయి
    ("రైతు బంధు సాయం ఎప్పుడొస్తయి?",                 "ఎప్పుడు", "ఎప్పుడొస్తయి"),
    ("ఆసరా పెన్షన్ ఎప్పుడొస్తయి?",                   "ఎప్పుడు", "ఎప్పుడొస్తయి"),
    ("అమ్మ ఒడి డబ్బులు ఎప్పుడొస్తయి?",               "ఎప్పుడు", "ఎప్పుడొస్తయి"),

    # ఎట్ల → ఎలా
    ("రైతు బంధు దరఖాస్తు ఎట్ల చేయాలె?",              "ఎలా", "ఎట్ల"),
    ("ఆరోగ్యశ్రీ రిజిస్ట్రేషన్ ఎట్ల చేయించాలె?",      "ఎలా", "ఎట్ల"),
    ("అమ్మ ఒడి కి ఎట్ల అప్లై చేయాలె?",               "ఎలా", "ఎట్ల"),
    ("ఆసరా పెన్షన్ ఎట్ల పొందాలె?",                    "ఎలా", "ఎట్ల"),

    # యాడ → ఎక్కడ (Rayalaseema)
    ("రైతు బంధు యాడ దరఖాస్తు చేయాలి?",               "ఎక్కడ", "యాడ"),
    ("ఆరోగ్యశ్రీ కి యాడ అర్జీ ఇవ్వాలి?",              "ఎక్కడ", "యాడ"),
    ("అమ్మ ఒడి ఆఫీసు యాడ ఉంటది?",                     "ఎక్కడ", "యాడ"),
    ("ఆసరా పెన్షన్ యాడ తీసుకుంటారు?",                 "ఎక్కడ", "యాడ"),

    # సొమ్ము → ఆర్థిక సహాయం (Rayalaseema)
    ("రైతు బంధు సొమ్ము ఎంత వస్తది?",                  "ఆర్థిక", "సొమ్ము"),
    ("ఆరోగ్యశ్రీ సొమ్ము ఎక్కడ తీసుకోవాలి?",           "ఆర్థిక", "సొమ్ము"),
    ("అమ్మ ఒడి సొమ్ము ఎప్పుడు వస్తది?",               "ఆర్థిక", "సొమ్ము"),

    # కాగితాలు → ధ్రువీకరణ పత్రాలు
    ("రైతు బంధు కోసం ఏమేం కాగితాలు కావాలె?",          "పత్రాలు", "కాగితాలు"),
    ("ఆరోగ్యశ్రీ కాగితాలు ఏమి కావాలె?",               "పత్రాలు", "కాగితాలు"),
    ("అమ్మ ఒడి కాగితాలు ఎక్కడ తీసుకోవాలె?",           "పత్రాలు", "కాగితాలు"),
    ("ఆసరా పెన్షన్ కాగితాలు ఏమి ఉండాలె?",             "పత్రాలు", "కాగితాలు"),

    # అప్లై → దరఖాస్తు
    ("రైతు బంధు కు ఎలా అప్లై చేయాలి?",               "దరఖాస్తు", "అప్లై"),
    ("ఆరోగ్యశ్రీ కి అప్లై విధానం చెప్పండి",            "దరఖాస్తు", "అప్లై"),
    ("అమ్మ ఒడి అప్లై చేసే స్థలం ఎక్కడ?",              "దరఖాస్తు", "అప్లై"),

    # సర్కారు → ప్రభుత్వం
    ("సర్కారు పథకం రైతు బంధు ఎంత ఇస్తున్నది?",         "ప్రభుత్వం", "సర్కారు"),
    ("సర్కారు ఆరోగ్యశ్రీ పథకం ఏమిటి?",                "ప్రభుత్వం", "సర్కారు"),
    ("సర్కారు అమ్మ ఒడి ఎంత ఇస్తుంది?",                "ప్రభుత్వం", "సర్కారు"),
    ("సర్కారు ఆసరా పెన్షన్ ఎంత?",                     "ప్రభుత్వం", "సర్కారు"),

    # బాపతు → సంబంధించిన
    ("రైతు బంధు బాపతు అర్హత ఉందా?",                   "సంబంధించిన", "బాపతు"),
    ("ఆరోగ్యశ్రీ బాపతు నిబంధనలు ఏమిటి?",              "సంబంధించిన", "బాపతు"),

    # ఇస్తారా → మంజూరు చేస్తారా
    ("రైతు బంధు మా కుటుంబానికి ఇస్తారా?",              "మంజూరు", "ఇస్తారా"),
    ("ఆసరా పెన్షన్ మాకు ఇస్తారా?",                    "మంజూరు", "ఇస్తారా"),

    # డాక్యుమెంట్లు → ధ్రువీకరణ పత్రాలు
    ("రైతు బంధు డాక్యుమెంట్లు ఏమి కావాలి?",            "పత్రాలు", "డాక్యుమెంట్లు"),
    ("ఆరోగ్యశ్రీ డాక్యుమెంట్లు సమర్పించాలా?",          "పత్రాలు", "డాక్యుమెంట్లు"),
    ("అమ్మ ఒడి డాక్యుమెంట్లు ఎక్కడ?",                 "పత్రాలు", "డాక్యుమెంట్లు"),

    # ఏటి → ఏమిటి
    ("రైతు బంధు ఏటి?",                                "ఏమిటి", "ఏటి"),
    ("ఆరోగ్యశ్రీ పథకం ఏటి?",                           "ఏమిటి", "ఏటి"),

    # Mixed dialect + phonetic folding compound probes
    ("ఆరోగ్యశ్రీ పతకం కోసం కాగితాలు ఏమేం కావాలె?",     "పథకం", "పతకం"),
    ("రైతు బంధు పతకం సొమ్ము ఎంతొస్తయి?",              "పథకం", "పతకం"),
    ("సర్కారు పతకం కింద పైసలు ఎప్పుడొస్తయి?",          "పథకం", "పతకం"),
    ("అమ్మ ఒడి పతకం ఎట్ల అప్లై చేయాలె?",              "పథకం", "పతకం"),
    ("ఆసరా పెన్షన్ పతకం యాడ దరఖాస్తు ఇవ్వాలె?",       "పథకం", "పతకం"),
]


def run_experiment2():
    """
    Experiment 2: Ablation – Phonetic Folding ON vs OFF.
    Tests 50 dialectal probes and reports:
    - Correction accuracy WITH Phonetic Folding
    - Correction accuracy WITHOUT Phonetic Folding (remove fold step)
    """
    print("\n" + "="*65)
    print("  EXPERIMENT 2: Dialect Normalization Ablation Study")
    print("="*65)

    norm_with_fold    = TeluguDialectNormalizer()
    norm_without_fold = TeluguDialectNormalizer()
    # Disable folding by replacing with identity table
    norm_without_fold.fold = str.maketrans({})  # empty → no folding

    results = []
    for dialect_query, expected_token, dialect_token in ABLATION_PROBES:
        out_with    = norm_with_fold.normalize(dialect_query)
        out_without = norm_without_fold.normalize(dialect_query)

        corrected_with    = expected_token in out_with
        corrected_without = expected_token in out_without

        results.append({
            "query":            dialect_query[:55],
            "dialect_token":    dialect_token,
            "expected":         expected_token,
            "with_fold":        corrected_with,
            "without_fold":     corrected_without,
            "fold_helped":      corrected_with and not corrected_without,
        })

    df = pd.DataFrame(results)
    n  = len(df)

    acc_with    = df['with_fold'].sum() / n
    acc_without = df['without_fold'].sum() / n
    fold_boosts = df['fold_helped'].sum()

    print(f"\n  Total probes                    : {n}")
    print(f"  Corrections WITH Phonetic Fold  : {df['with_fold'].sum():>3d}  ({acc_with*100:.1f}%)")
    print(f"  Corrections WITHOUT Fold        : {df['without_fold'].sum():>3d}  ({acc_without*100:.1f}%)")
    print(f"  Queries where Fold was decisive : {fold_boosts}")

    # Show failures (if any) with fold ON
    failures = df[~df['with_fold']]
    if not failures.empty:
        print(f"\n  ⚠  Missed corrections ({len(failures)}):")
        for _, row in failures.iterrows():
            print(f"     '{row['query']}' — expected '{row['expected']}'")

    # Show sample phonetic-folding wins
    fold_wins = df[df['fold_helped']]
    print(f"\n  Sample Phonetic-Folding wins:")
    for _, row in fold_wins.head(5).iterrows():
        print(f"     '{row['dialect_token']}' → '{row['expected']}'  (folding was decisive)")

    # ── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ['Without\nPhonetic Folding', 'With\nPhonetic Folding'],
        [acc_without * 100, acc_with * 100],
        color=['#e74c3c', '#2ecc71'],
        width=0.4,
        edgecolor='black'
    )
    for bar, val in zip(bars, [acc_without * 100, acc_with * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Token Correction Accuracy (%)', fontsize=11)
    ax.set_title('Experiment 2: Phonetic Folding Ablation\n(50 Dialectal Telugu Probes)', fontsize=12)
    ax.axhline(y=acc_without * 100, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'exp2_phonetic_ablation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved → {plot_path}")

    return acc_with, acc_without


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 – Intent Classification Macro-F1
# ══════════════════════════════════════════════════════════════════════════════

# Held-out test set: 40 unseen queries (10 per intent)
# These use phrasing templates NOT present in the training set
HELD_OUT = [
    # ELIGIBILITY (10)
    ("రైతు భరోసా కి మా పొలం అర్హమవుతుందా?",                 "ELIGIBILITY"),
    ("ఆరోగ్యశ్రీ పొందాలంటే ఏమి చేయాలి?",                   "ELIGIBILITY"),
    ("అమ్మ ఒడి కి విద్యార్థి హాజరు ఎంత కావాలి?",            "ELIGIBILITY"),
    ("ఆసరా పెన్షన్ కోసం వయస్సు ఎంత ఉండాలి?",               "ELIGIBILITY"),
    ("రైతు బంధు కి మహిళా రైతులు అర్హులా?",                  "ELIGIBILITY"),
    ("ఆరోగ్యశ్రీ కార్డు ఎవరికి దొరుకుతుంది?",               "ELIGIBILITY"),
    ("అమ్మ ఒడి కి పిల్లలు ప్రైవేట్ స్కూల్లో చదివినా వస్తుందా?", "ELIGIBILITY"),
    ("ఆసరా పెన్షన్ కి వితంతువులు అర్హులా?",                  "ELIGIBILITY"),
    ("రైతు భరోసా కి కౌలు రైతు అర్హతలు ఏమిటి?",              "ELIGIBILITY"),
    ("అమ్మ ఒడి కి 75% హాజరు తప్పనిసరా?",                   "ELIGIBILITY"),

    # BENEFITS (10)
    ("రైతు బంధు ద్వారా ఎకరాకు ఎంత వస్తుంది?",               "BENEFITS"),
    ("ఆరోగ్యశ్రీ కింద ఏయే వ్యాధులకు చికిత్స ఉచితం?",        "BENEFITS"),
    ("అమ్మ ఒడి ఇచ్చే గ్రాంట్ మొత్తం ఎంత?",                  "BENEFITS"),
    ("ఆసరా పెన్షన్ నెలకు ఎంత ఇస్తారు?",                     "BENEFITS"),
    ("రైతు భరోసా ఖరీఫ్ సీజన్ లో ఎంత వస్తుంది?",             "BENEFITS"),
    ("ఆరోగ్యశ్రీ లో ఆపరేషన్ కూడా ఉచితమా?",                 "BENEFITS"),
    ("అమ్మ ఒడి గ్రాంట్ ఏ నెలలో వస్తుంది?",                  "BENEFITS"),
    ("ఆసరా పెన్షన్ దివ్యాంగులకు ఎంత?",                      "BENEFITS"),
    ("రైతు బంధు ఏడాదికి ఎన్ని సార్లు ఇస్తారు?",              "BENEFITS"),
    ("ఆరోగ్యశ్రీ నెట్‌వర్క్ ఆసుపత్రుల్లో ఏమి అందుతుంది?",   "BENEFITS"),

    # REQUIRED_DOCUMENTS (10)
    ("రైతు బంధు కి పాస్ పుస్తకం కావాలా?",                   "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కోసం ఆసుపత్రిలో ఏ పత్రాలు చూపించాలి?",    "REQUIRED_DOCUMENTS"),
    ("అమ్మ ఒడి కి స్కూల్ సర్టిఫికెట్ కావాలా?",               "REQUIRED_DOCUMENTS"),
    ("ఆసరా పెన్షన్ కి ఆయుష్ కార్డు కావాలా?",                "REQUIRED_DOCUMENTS"),
    ("రైతు భరోసా కి జమాబందీ కావాలా?",                        "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కోసం రేషన్ కార్డు తప్పనిసరిగా ఉండాలా?",    "REQUIRED_DOCUMENTS"),
    ("అమ్మ ఒడి కి హాజరు పత్రం సమర్పించాలా?",                "REQUIRED_DOCUMENTS"),
    ("ఆసరా పెన్షన్ కోసం పేదరిక ధ్రువీకరణ కావాలా?",           "REQUIRED_DOCUMENTS"),
    ("రైతు బంధు కి భూ యాజమాన్య పత్రం కావాలా?",              "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కార్డు పొందడానికి ఏమి ఇవ్వాలి?",            "REQUIRED_DOCUMENTS"),

    # APPLICATION_PROCEDURE (10)
    ("రైతు బంధు నమోదు ఎలా చేయించుకోవాలి?",                   "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ కోసం సచివాలయానికి వెళ్ళాలా?",              "APPLICATION_PROCEDURE"),
    ("అమ్మ ఒడి నమోదు చేసే స్థలం ఎక్కడ?",                    "APPLICATION_PROCEDURE"),
    ("ఆసరా పెన్షన్ ఫారం ఎక్కడ తీసుకోవాలి?",                  "APPLICATION_PROCEDURE"),
    ("రైతు భరోసా ఆన్‌లైన్‌లో నమోదు చేయవచ్చా?",              "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ మీసేవాలో అప్లై చేయవచ్చా?",                 "APPLICATION_PROCEDURE"),
    ("అమ్మ ఒడి దరఖాస్తు ప్రక్రియ ఏమిటి?",                   "APPLICATION_PROCEDURE"),
    ("ఆసరా పెన్షన్ ఆఫ్‌లైన్ దరఖాస్తు ఎలా?",                 "APPLICATION_PROCEDURE"),
    ("రైతు బంధు పోర్టల్ లో లాగిన్ అవడం ఎలా?",               "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ కార్డు దరఖాస్తు స్టేటస్ చెక్ చేయడం ఎలా?",  "APPLICATION_PROCEDURE"),
]

# Dialect versions of the HELD_OUT queries (same intent, different surface form)
HELD_OUT_DIALECT = [
    ("రైతు భరోసా కి మా పొలం వస్తదా?",                         "ELIGIBILITY"),
    ("ఆరోగ్యశ్రీ పొందాలంటే ఏం చేయాలె?",                       "ELIGIBILITY"),
    ("అమ్మ ఒడి కి పిల్లోడు స్కూలుకు ఎంత పోవాలె?",             "ELIGIBILITY"),
    ("ఆసరా పెన్షన్ కి వయసు ఎంత ఉండాలె?",                     "ELIGIBILITY"),
    ("రైతు బంధు కి ఆడ రైతులు అర్హులా?",                        "ELIGIBILITY"),
    ("ఆరోగ్యశ్రీ కార్డు ఎవరికి దొరుకుతది?",                    "ELIGIBILITY"),
    ("అమ్మ ఒడి కి ప్రైవేట్ స్కూలు పిల్లలకు వస్తదా?",           "ELIGIBILITY"),
    ("ఆసరా పెన్షన్ కి వితంతు మహిళలు అర్హులా?",                 "ELIGIBILITY"),
    ("రైతు భరోసా కి కౌలు రైతు అర్హతలు ఏమిటె?",                "ELIGIBILITY"),
    ("అమ్మ ఒడి కి 75% హాజరు ఉండాలా లేదా?",                    "ELIGIBILITY"),

    ("రైతు బంధు పైసలు ఎకరాకు ఎంత వస్తయి?",                    "BENEFITS"),
    ("ఆరోగ్యశ్రీ కింద ఏయే జబ్బులకు ఉచిత వైద్యం?",              "BENEFITS"),
    ("అమ్మ ఒడి పైసలు ఎంత వస్తయి?",                            "BENEFITS"),
    ("ఆసరా పెన్షన్ నెలకు ఎంత పైసలు ఇస్తారు?",                  "BENEFITS"),
    ("రైతు భరోసా ఖరీఫ్ లో ఎన్ని పైసలు వస్తయి?",                "BENEFITS"),
    ("ఆరోగ్యశ్రీ లో ఆపరేషన్ కూడా ఫ్రీయా?",                   "BENEFITS"),
    ("అమ్మ ఒడి పైసలు ఏ నెలలో వస్తయి?",                        "BENEFITS"),
    ("ఆసరా పెన్షన్ దివ్యాంగులకు ఎంత పైసలు?",                   "BENEFITS"),
    ("రైతు బంధు ఏడాదికి ఎన్ని సార్లు పైసలు ఇస్తయి?",           "BENEFITS"),
    ("ఆరోగ్యశ్రీ నెట్‌వర్క్ ఆసుపత్రిలో ఏం ఇస్తయి?",            "BENEFITS"),

    ("రైతు బంధు కి పాస్ పుస్తకం కావాలా లేదా?",                  "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కోసం ఆసుపత్రిలో ఏ కాగితాలు చూపించాలె?",      "REQUIRED_DOCUMENTS"),
    ("అమ్మ ఒడి కి స్కూలు సర్టిఫికెట్ కావాలా?",                  "REQUIRED_DOCUMENTS"),
    ("ఆసరా పెన్షన్ కి ఆయుష్ కార్డు కావాలా లేదా?",               "REQUIRED_DOCUMENTS"),
    ("రైతు భరోసా కి జమాబందీ కాగితం కావాలా?",                   "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కోసం రేషన్ కార్డు తప్పక ఉండాలా?",             "REQUIRED_DOCUMENTS"),
    ("అమ్మ ఒడి కి హాజరు కాగితం ఇవ్వాలా?",                     "REQUIRED_DOCUMENTS"),
    ("ఆసరా పెన్షన్ కోసం పేదరిక సర్టిఫికెట్ కావాలా?",            "REQUIRED_DOCUMENTS"),
    ("రైతు బంధు కి భూమి కాగితం కావాలా?",                       "REQUIRED_DOCUMENTS"),
    ("ఆరోగ్యశ్రీ కార్డు కు ఏమేం కాగితాలు ఇవ్వాలె?",             "REQUIRED_DOCUMENTS"),

    ("రైతు బంధు నమోదు ఎట్ల చేయించుకోవాలె?",                    "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ కోసం సచివాలయానికి పోవాలా?",                    "APPLICATION_PROCEDURE"),
    ("అమ్మ ఒడి నమోదు యాడ చేయించాలె?",                         "APPLICATION_PROCEDURE"),
    ("ఆసరా పెన్షన్ ఫారం యాడ తీసుకోవాలె?",                      "APPLICATION_PROCEDURE"),
    ("రైతు భరోసా ఆన్‌లైన్లో నమోదు ఎట్ల చేయాలె?",               "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ మీసేవాలో అప్లై ఎట్ల చేయాలె?",                  "APPLICATION_PROCEDURE"),
    ("అమ్మ ఒడి దరఖాస్తు ఎట్ల చేయాలె?",                        "APPLICATION_PROCEDURE"),
    ("ఆసరా పెన్షన్ అప్లై చేసే విధానం ఏమిటె?",                   "APPLICATION_PROCEDURE"),
    ("రైతు బంధు పోర్టల్ లో లాగిన్ ఎట్ల?",                     "APPLICATION_PROCEDURE"),
    ("ఆరోగ్యశ్రీ కార్డు స్టేటస్ ఎట్ల చెక్ చేయాలె?",             "APPLICATION_PROCEDURE"),
]


def run_experiment3(model):
    """
    Experiment 3: Intent Macro-F1 across three conditions.
    """
    print("\n" + "="*65)
    print("  EXPERIMENT 3: Intent Classification Macro-F1 Comparison")
    print("="*65)

    norm = TeluguDialectNormalizer()

    # Condition 1: Ground truth (clean standard Telugu)
    X_gt,   y_gt   = zip(*HELD_OUT)
    # Condition 2: Raw dialect (no normalization)
    X_dial, y_dial = zip(*HELD_OUT_DIALECT)
    # Condition 3: Normalized dialect
    X_norm  = [norm.normalize(q) for q in X_dial]

    def eval_condition(name, X, y):
        preds  = model.predict(list(X))
        macro_f1 = f1_score(list(y), preds, average='macro')
        report   = classification_report(list(y), preds, digits=4)
        print(f"\n  [{name}]  Macro-F1: {macro_f1:.4f}")
        print(report)
        return macro_f1, preds

    f1_gt,   preds_gt   = eval_condition("Condition 1 – Ground Truth",   X_gt,   y_gt)
    f1_dial, preds_dial = eval_condition("Condition 2 – Raw Dialect",    X_dial, y_dial)
    f1_norm, preds_norm = eval_condition("Condition 3 – Normalized",     X_norm, y_dial)

    # Print normalization gain
    gain = f1_norm - f1_dial
    print(f"\n  Normalization gain (Cond 3 vs Cond 2): +{gain:.4f}")
    print(f"  Recovery gap from Ground Truth       :  {f1_gt - f1_norm:.4f}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['Ground Truth\n(Standard Telugu)', 'Raw Dialect\n(No Normalization)', 'After Normalization\n(Our System)']
    f1s    = [f1_gt, f1_dial, f1_norm]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars   = ax.bar(labels, [v * 100 for v in f1s], color=colors, width=0.4, edgecolor='black')

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(50, 105)
    ax.set_ylabel('Intent Classification Macro-F1 (%)', fontsize=11)
    ax.set_title('Experiment 3: Dialect Normalization → Intent Recovery\n(40 Held-Out Queries)', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'exp3_intent_macro_f1.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Plot saved → {plot_path}")

    return f1_gt, f1_dial, f1_norm


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model not found at {MODEL_PATH}")
        print("    Run:  python src/train_intent_model.py  first!")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅  Loaded model from {MODEL_PATH}")

    acc_with, acc_without = run_experiment2()
    f1_gt, f1_dial, f1_norm = run_experiment3(model)

    # Summary table
    print("\n" + "="*65)
    print("  SUMMARY TABLE (for your Academic Report)")
    print("="*65)
    print(f"  Exp 2 — Token Correction Accuracy")
    print(f"          Without Phonetic Folding : {acc_without*100:6.1f}%")
    print(f"          With    Phonetic Folding : {acc_with*100:6.1f}%")
    print(f"          Improvement             : +{(acc_with-acc_without)*100:.1f}%")
    print(f"\n  Exp 3 — Intent Macro-F1")
    print(f"          Condition 1 (Ground Truth) : {f1_gt*100:.2f}%")
    print(f"          Condition 2 (Raw Dialect)  : {f1_dial*100:.2f}%")
    print(f"          Condition 3 (Normalized)   : {f1_norm*100:.2f}%")
    print(f"          Normalization Gain         : +{(f1_norm-f1_dial)*100:.2f}%")
    print(f"\n  Plots saved in: {PLOTS_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
