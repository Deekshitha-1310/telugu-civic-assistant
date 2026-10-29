"""Question and scheme analysis for retrieval, separate from the intent model.

Category rules are ordered: the first category whose marker matches wins.
ASCII markers are matched on word boundaries (so "aim" does not fire inside
"claim", and "apply" does not fire inside "exceptions apply" because
EXCEPTIONS is checked first). Telugu markers are matched as substrings/stems
because Telugu attaches case suffixes to the word (అర్హ + ులను, అర్హ + త ...).
"""
import re


_CATEGORY_MARKERS = (
    ("PURPOSE", (
        "purpose", "objective", "aim", "ఉద్దేశం", "ఉద్దేశ్యం", "లక్ష్యం",
        "ఎందుకు", "దేనికోసం",
    )),
    ("REQUIRED_DOCUMENTS", (
        "document", "documents", "certificates", "పత్రం", "పత్రాలు", "కాగితం",
        "కాగితాలు", "డాక్యుమెంట్",
    )),
    ("DEADLINES", ("deadline", "last date", "when is", "గడువు", "చివరి తేదీ")),
    ("STATUS_CHECK", ("status", "payment status", "స్థితి", "చెల్లింపు ఎప్పుడు")),
    ("RENEWAL", ("renew", "renewal", "పునరుద్ధ")),
    ("FAQ", ("faq", "faqs", "frequently asked")),
    # Exclusions are a kind of eligibility question but are checked before
    # APPLICATION_PROCEDURE so that "what exceptions apply" is not "apply".
    ("EXCEPTIONS", (
        "exception", "exceptions", "exclusion", "exclusions", "excluded",
        "not eligible", "ineligible", "వర్తించని", "అనర్హ",
    )),
    ("APPLICATION_PROCEDURE", (
        "apply", "application", "procedure", "దరఖాస్తు", "నమోదు", "ఎలా చేస",
        "ఎలా దరఖాస్తు",
    )),
    ("GRIEVANCE", ("grievance", "complain", "complaint", "appeal", "ఫిర్యాదు", "అప్పీల్")),
    ("HELPLINE", ("helpline", "phone", "contact", "హెల్ప్‌లైన్", "ఫోన్", "సంప్రదించ")),
    ("ELIGIBILITY", (
        "eligible", "eligibility", "qualify", "qualifies", "qualification",
        "criteria", "requirements", "who can get", "who can apply",
        # Telugu stem covers అర్హత, అర్హులు, అర్హులను, అర్హుడిని, అర్హురాలు ...
        "అర్హ", "వర్తిస్తుందా", "ఎవరికి వర్తిస్తుంది", "ఎవరికి ఇస్తారు",
    )),
    ("BENEFITS", (
        "benefit", "benefits", "amount", "how much", "ప్రయోజనం", "ప్రయోజనాలు",
        "ఎంత", "డబ్బు", "సాయం", "లబ్ధి", "కవరేజ్", "చికిత్స",
        "ఏమి వస్తాది", "ఏమి సహాయం వస్తుంది", "ఏమి మంచిది",
        "వస్తాది", "వస్తుంది", "వచ్చేది", "వచ్చింది", "ఇస్తారు", "ఇస్తుంది", "ఇచ్చింది",
    )),
)

# Categories for which the corpus checklist expects evidence of the same
# category. EXCEPTIONS questions are answered from ELIGIBILITY evidence (the
# checklist already maps exception-te -> evidence_category ELIGIBILITY).
EVIDENCE_CATEGORY = {"EXCEPTIONS": "ELIGIBILITY"}

_SCHEME_ALIASES = (
    ((
        "రైతు భరోసా", "రైతు బంధు", "రైతుబంధు", "రైతుభరోసా",
        "rythu bharosa", "rythu bandhu", "raithu bandhu", "rythubandhu",
    ), "రైతు బంధు / రైతు భరోసా"),
    ((
        "అమ్మ ఒడి", "అమ్మఒడి", "ammavodi", "amma vodi", "jagananna amma vodi",
    ), "జగనన్న అమ్మ ఒడి"),
    ((
        "ఆరోగ్యశ్రీ", "ఆరోగ్య శ్రీ", "aarogyasri", "arogyasri",
    ), "వైఎస్ఆర్ ఆరోగ్యశ్రీ"),
    (("ఆసరా", "aasara", "asara pension", "pension kanuka"), "ఆసరా పెన్షన్ / పెన్షన్ కానుక"),
)


def _is_ascii(marker):
    return all(ord(ch) < 128 for ch in marker)


def _marker_hits(marker, value):
    marker = marker.casefold()
    if _is_ascii(marker):
        return re.search(r"(?<![a-z0-9])" + re.escape(marker) + r"(?![a-z0-9])", value) is not None
    return marker in value


def category_matches(text):
    """All categories whose markers appear, in rule order (for diagnostics)."""
    value = (text or "").casefold()
    return [
        category for category, markers in _CATEGORY_MARKERS
        if any(_marker_hits(m, value) for m in markers)
    ]


def detect_category(text):
    """Return a retrieval category when the wording gives reliable evidence."""
    matches = category_matches(text)
    return matches[0] if matches else None


def evidence_category(category):
    """The corpus category that holds evidence for a question category."""
    return EVIDENCE_CATEGORY.get(category, category)


def identify_scheme(text, known_schemes=()):
    """Resolve an explicit scheme alias to the corpus' canonical scheme name."""
    value = (text or "").casefold()
    for aliases, canonical in _SCHEME_ALIASES:
        if any(alias.casefold() in value for alias in aliases):
            return canonical
    for scheme in known_schemes:
        for part in re.split(r"\s*/\s*", scheme):
            if part and part.casefold() in value:
                return scheme
    return None
