"""Small deterministic language adapter for the no-provider fallback."""
import re


_ENGLISH_TO_TELUGU = (
    (r"the government of telangana launched the rythu bharosa scheme", "తెలంగాణ ప్రభుత్వం రైతు భరోసా పథకాన్ని ప్రారంభించింది"),
    (
        r"This initiative focuses on offering financial aid and access to agricultural resources to empower farmers and improve productivity\.",
        "ఈ కార్యక్రమం రైతులకు ఆర్థిక సహాయం మరియు వ్యవసాయ వనరులు అందించి, వారిని సాధికారికం చేయడం మరియు ఉత్పాదకతను మెరుగుపరచడం పై దృష్టి పెడుతుంది.",
    ),
    (
        r"The scheme complements other farmer-friendly policies like Crop Loan Waiver and aims to provide working capital to the farmers at the beginning of each crop cycle\.",
        "ఈ పథకం పంట రుణమాఫీ వంటి రైతు సంక్షేమ విధానాలకు తోడ్పడుతూ, ప్రతి పంట చక్రం ప్రారంభంలో రైతులకు పని మూలధనం అందించడాన్ని లక్ష్యంగా పెట్టుకుంది.",
    ),
    (
        r"By integrating various welfare measures, Rythu Bharosa seeks to enhance the socio-economic conditions of farmers and boost the overall agricultural sector in Telangana, reflecting the government's commitment to rural development and farmer welfare\.",
        "వివిధ సంక్షేమ చర్యలను ఏకీకృతం చేసి, రైతుల సామాజిక-ఆర్థిక పరిస్థితులను మెరుగుపరచడం మరియు తెలంగాణలో వ్యవసాయ రంగాన్ని అభివృద్ధి చేయడం రైతు భరోసా లక్ష్యం.",
    ),
)


def _contains_telugu(text):
    return bool(re.search(r"[\u0c00-\u0c7f]", text or ""))


def render_scheme_name(scheme, state=None, language="te"):
    if language == "en":
        names = {
            "రైతు బంధు / రైతు భరోసా": "Rythu Bharosa",
            "రైతు భరోసా": "Rythu Bharosa",
            "రైతు బంధు": "Rythu Bharosa",
            "జగనన్న అమ్మ ఒడి": "Jagananna Amma Vodi",
            "తలికీ వండనం": "Talliki Vandanam",
            "వైఎస్ఆర్ ఆరోగ్యశ్రీ": "Dr. NTR Vaidya Seva",
            "తెలంగాణ ఆసరా": "Telangana Aasara",
            "ఆసరా పెన్షన్ / పెన్షన్ కానుక": "Aasara Pension / Pension Kanuka",
        }
        base = names.get(scheme, scheme)
        if state and state.lower() != base.lower():
            return f"{base} ({state})"
        return base
    # Telugu: append state in parentheses if different from the main scheme name
    if state and state != scheme:
        return f"{scheme} ({state})"
    return scheme


def summarize_evidence(content: str, max_chars: int = 200) -> str:
    """Truncate *content* to at most *max_chars* characters, respecting Unicode boundaries.
    If the content is longer, an ellipsis is appended.
    """
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped
    # Ensure we don't break a surrogate pair.
    truncated = stripped[:max_chars]
    # If the last character is a high surrogate, drop it.
    if ord(truncated[-1]) >= 0xD800 and ord(truncated[-1]) <= 0xDBFF:
        truncated = truncated[:-1]
    return truncated.rstrip() + "…"

# Export the new helper
__all__ = [
    "render_scheme_name",
    "render_evidence",
    "summarize_evidence",
]

def render_evidence(content, language="te"):
    if language == "te" and _contains_telugu(content):
        return content.strip()
    if language == "en" and _contains_telugu(content):
        # For English UI, try to translate Telugu evidence to English using
        # the pattern dictionary. If no pattern matches, suppress the
        # Telugu content (there is no English rendering) so English answers
        # never contain Telugu script.
        value = content.strip()
        for pattern, replacement in _ENGLISH_TO_TELUGU:
            translated = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
            if translated != value:
                # Translate sentence-by-sentence to keep as much as possible
                sentences = re.split(r"(?<=[.!?।])\s+", value)
                translated_sentences = []
                for sentence in sentences:
                    for pat, repl in _ENGLISH_TO_TELUGU:
                        if re.search(pat, sentence, flags=re.IGNORECASE):
                            sentence = re.sub(pat, repl, sentence, flags=re.IGNORECASE)
                            break
                    translated_sentences.append(sentence)
                return " ".join(translated_sentences).strip()
        return ""  # suppress untranslatable Telugu for English UI
    if language == "en":
        return content.strip()
    value = content.strip()
    for pattern, replacement in _ENGLISH_TO_TELUGU:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    if value == content.strip():
        return "ఆధారం ఆంగ్లంలో మాత్రమే ఉంది: " + value
    sentences = re.split(r"(?<=[.!?।])\s+", value)
    return " ".join(sentences[:3]).strip()
