"""
generate_intent_data.py  —  Member 2: Intent Dataset Generator
================================================================
Generates 200 labeled Telugu civic queries from the 4 welfare schemes × 4 intents.
Uses template families with standard + dialect variants to ensure every intent class
is balanced and every dialect style is represented.

Output: data/intent_dataset.csv
"""

import os
import csv

# Resolve paths relative to this file
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, '..', 'data')
OUTPUT_PATH = os.path.join(_DATA_DIR, 'intent_dataset.csv')

# ── Scheme names ──────────────────────────────────────────────────────────────
SCHEMES = [
    "రైతు భరోసా",
    "రైతు బంధు",
    "ఆరోగ్యశ్రీ",
    "అమ్మ ఒడి",
    "ఆసరా పెన్షన్",
]

# ── Template banks per intent × dialect style ─────────────────────────────────
TEMPLATES = {
    "ELIGIBILITY": {
        "standard": [
            "{scheme} పథకానికి ఎవరు అర్హులు?",
            "{scheme} కి కౌలు రైతులు అర్హులా?",
            "{scheme} వర్తించాలంటే ఎంత భూమి ఉండాలి?",
            "{scheme} పథకం ఎవరికి వర్తిస్తుంది?",
            "{scheme} అర్హుల జాబితా ఎక్కడ చూడాలి?",
            "{scheme} పథకం ఎవరికి అందుతుంది?",
            "{scheme} కు అర్హత ఉందా అని ఎలా తెలుసుకోవాలి?",
            "{scheme} కు వయస్సు పరిమితి ఏమిటి?",
        ],
        "dialect_telangana": [
            "{scheme} ఎవరికి వస్తది ఎవరికి రాదు?",
            "{scheme} కి వయస్సు పరిమితి ఎంత ఉండాలె?",
            "{scheme} మాకు వస్తదా అర్హత ఉందా?",
            "{scheme} రూల్స్ మరియు నిబంధనలు ఏమిటి?",
            "{scheme} ఎవరికి వర్తిస్తదో చెప్పండి?",
        ],
        "dialect_rayalaseema": [
            "{scheme} బాపతు అర్హత ఎవరికి ఉంటాది?",
            "{scheme} ఏ వాళ్ళకు వస్తాది?",
        ],
    },
    "BENEFITS": {
        "standard": [
            "{scheme} ద్వారా ఎంత మొత్తం లభిస్తుంది?",
            "{scheme} పథకం ఆర్థిక సహాయం ఎంత?",
            "{scheme} లో ఏడాదికి ఎన్ని విడతల్లో ఇస్తారు?",
            "{scheme} కింద బ్యాంకు ఖాతాలో ఎంత జమ అవుతుంది?",
            "{scheme} అందించే సేవలు మరియు సహాయం ఏమిటి?",
            "{scheme} ద్వారా ఏయే వైద్య సేవలు అందుతాయి?",
        ],
        "dialect_telangana": [
            "{scheme} కింద ఎన్ని పైసలు ఇస్తారు?",
            "{scheme} పైసలు ఎప్పుడొస్తయి సర్కారు నుంచి?",
            "{scheme} సాయం ఎంత అందుతుంది మాకు?",
            "{scheme} పైసలు పడ్డాయా లేదా ఎలా తెలుసుకోవాలి?",
            "{scheme} ఎంత డబ్బు ఇస్తారు?",
        ],
        "dialect_rayalaseema": [
            "{scheme} సొమ్ము ఖాతాలో ఎప్పుడు పడుతుంది?",
            "{scheme} పైసలు ఎంత వస్తాయి?",
        ],
        "colloquial": [
            "{scheme} డబ్బులు ఎప్పుడు వస్తాయి?",
        ],
    },
    "REQUIRED_DOCUMENTS": {
        "standard": [
            "{scheme} దరఖాస్తుకు కావలసిన ధ్రువీకరణ పత్రాలు ఏవి?",
            "{scheme} కి ఆధార్ కార్డు రేషన్ కార్డు తప్పనిసరిగా ఉండాలా?",
            "{scheme} కి పట్టాదారు పాస్ పుస్తకం అవసరమా?",
            "{scheme} రిజిస్ట్రేషన్ కోసం ఏ పత్రాలు సమర్పించాలి?",
            "{scheme} డాక్యుమెంట్లు ఎక్కడ వెరిఫై చేయించాలి?",
            "{scheme} కు ఏ సర్టిఫికెట్లు అవసరం?",
        ],
        "dialect_telangana": [
            "{scheme} కోసం ఏమేం కాగితాలు కావాలె?",
            "{scheme} కి ఏమేమి కాగితాలు జతచేయాలి?",
            "{scheme} కి ఏమేం పత్రాలు కావాలి?",
        ],
        "dialect_rayalaseema": [
            "{scheme} బాపతు డాక్యుమెంట్లు ఏమి కావాలి?",
            "{scheme} కు ఏ కాగితాలు కావాలె?",
        ],
        "colloquial": [
            "{scheme} కోసం బ్యాంకు పాస్ బుక్ జిరాక్స్ కావాలా?",
        ],
    },
    "APPLICATION_PROCEDURE": {
        "standard": [
            "{scheme} కి ఎలా దరఖాస్తు చేసుకోవాలి?",
            "{scheme} కి మీసేవా లేదా సచివాలయంలో దరఖాస్తు చేయవచ్చా?",
            "{scheme} ఆన్‌లైన్ లో అప్లై చేసే విధానం ఏమిటి?",
            "{scheme} దరఖాస్తు గడువు తేదీ ఎప్పుడు?",
            "{scheme} రిజిస్ట్రేషన్ ఎలా పూర్తి చేయాలి?",
            "{scheme} స్టేటస్ ఎలా చెక్ చేసుకోవాలి?",
        ],
        "dialect_telangana": [
            "{scheme} దరఖాస్తు ఎట్ల చేసుకోవాలె?",
            "{scheme} లో రిజిస్ట్రేషన్ ఎలా చేయించుకోవాలె?",
        ],
        "dialect_rayalaseema": [
            "{scheme} కి యాడ దరఖాస్తు ఇవ్వాల?",
            "{scheme} కు ఎక్కడ పెట్టాలో చెప్పండి?",
        ],
        "colloquial": [
            "{scheme} కోసం ఎక్కడ అప్లై చేయాలి?",
            "{scheme} ఫారం ఎక్కడ దొరుకుతుంది?",
        ],
    },
}


def generate_dataset():
    rows = []
    qid = 1
    for intent, style_map in TEMPLATES.items():
        for style, template_list in style_map.items():
            for tmpl in template_list:
                for scheme in SCHEMES:
                    query_text = tmpl.format(scheme=scheme)
                    rows.append({
                        "query_id":     f"q_{qid:03d}",
                        "query_text":   query_text,
                        "intent_label": intent,
                        "target_scheme": scheme,
                        "dialect_style": style,
                    })
                    qid += 1

    # Write CSV
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "query_text", "intent_label", "target_scheme", "dialect_style"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅  Generated {len(rows)} labeled queries → {OUTPUT_PATH}")

    # Print class distribution
    from collections import Counter
    dist = Counter(r['intent_label'] for r in rows)
    print("\nClass distribution:")
    for intent, count in sorted(dist.items()):
        print(f"  {intent:<28s}: {count}")

    return rows


if __name__ == '__main__':
    generate_dataset()
