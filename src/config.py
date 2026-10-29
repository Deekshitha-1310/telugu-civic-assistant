"""Display labels and pipeline constants. No invented metrics live here."""

INTENT_EN = {
    "PURPOSE": "Purpose",
    "ELIGIBILITY": "Eligibility",
    "BENEFITS": "Benefits",
    "REQUIRED_DOCUMENTS": "Documents",
    "APPLICATION_PROCEDURE": "Procedure",
}

INTENT_TE = {
    "PURPOSE": "ఉద్దేశం",
    "ELIGIBILITY": "అర్హత",
    "BENEFITS": "ప్రయోజనాలు",
    "REQUIRED_DOCUMENTS": "పత్రాలు",
    "APPLICATION_PROCEDURE": "దరఖాస్తు విధానం",
}

# Labels for every question category the answer layer can report.
CATEGORY_EN = {
    "PURPOSE": "Purpose",
    "ELIGIBILITY": "Eligibility",
    "BENEFITS": "Benefits",
    "REQUIRED_DOCUMENTS": "Required documents",
    "APPLICATION_PROCEDURE": "How to apply",
    "DEADLINES": "Deadlines",
    "STATUS_CHECK": "Status check",
    "RENEWAL": "Renewal",
    "GRIEVANCE": "Grievances",
    "HELPLINE": "Helpline",
    "FAQ": "FAQ",
    "EXCEPTIONS": "Exclusions",
}
CATEGORY_TE = {
    "PURPOSE": "ఉద్దేశం",
    "ELIGIBILITY": "అర్హత",
    "BENEFITS": "ప్రయోజనాలు",
    "REQUIRED_DOCUMENTS": "అవసరమైన పత్రాలు",
    "APPLICATION_PROCEDURE": "దరఖాస్తు విధానం",
    "DEADLINES": "గడువు తేదీలు",
    "STATUS_CHECK": "స్థితి తనిఖీ",
    "RENEWAL": "పునరుద్ధరణ",
    "GRIEVANCE": "ఫిర్యాదులు",
    "HELPLINE": "హెల్ప్‌లైన్",
    "FAQ": "తరచుగా అడిగే ప్రశ్నలు",
    "EXCEPTIONS": "మినహాయింపులు",
}

INTENT_ORDER = [
    "ELIGIBILITY",
    "BENEFITS",
    "REQUIRED_DOCUMENTS",
    "APPLICATION_PROCEDURE",
]

NO_EVIDENCE_TE = (
    "క్షమించండి, ఈ ప్రశ్నకు సంబంధించిన సమాచారం మా జ్ఞాన నిధిలో దొరకలేదు. "
    "దయచేసి పథకం పేరును స్పష్టంగా పేర్కొని మళ్లీ అడగండి."
)
NO_EVIDENCE_EN = (
    "I could not find reliable evidence for this question in the knowledge base. "
    "Please check the linked official source or ask about another supported topic."
)

OOS_TE = (
    "ఈ ప్రశ్న మద్దతు ఉన్న నాలుగు అంశాల్లో (అర్హత, ప్రయోజనాలు, పత్రాలు, దరఖాస్తు) "
    "స్పష్టంగా లేదు, మరియు జ్ఞాన నిధిలో సరిపోలే ఆధారాలు దొరకలేదు."
)
OOS_EN = "I could not match this question to a supported topic with reliable evidence."

EXAMPLE_QUERIES = [
    "రైతు బంధు పథకానికి నేను అర్హుడినా?",
    "ఆరోగ్యశ్రీ దరఖాస్తు చేయడానికి ఏ పత్రాలు కావాలి?",
    "అమ్మ ఒడి పథకం వల్ల ఏ ప్రయోజనాలు ఉంటాయి?",
    "ఆసరా పెన్షన్ దరఖాస్తు ఎలా చేసుకోవాలి?",
    "ఈ పథకానికి నేను అర్హుడినా?",
    "దరఖాస్తు చేయడానికి ఏ పత్రాలు కావాలి?",
    "ఈ పథకం వల్ల ఏ ప్రయోజనాలు ఉంటాయి?",
    "దరఖాస్తు ఎలా చేసుకోవాలి?",
]

# Demonstratives that make a follow-up depend on the previous scheme.
FOLLOWUP_MARKERS = ("ఈ పథకం", "ఈ పథకానికి", "ఈ స్కీం", "అది", "దానికి")

RETRIEVAL_DEFAULTS = {
    "top_k": 5,
    "bm25_weight": 1.8,
    "dense_weight": 1.0,
    "rrf_k": 60,
    "suffix_stripping": True,
}
