"""Extractive, source-grounded answers. Never paraphrases facts that are not in retrieved chunks.

How an answer is built (in plain words, for the viva):

1. Keep only passages of the asked scheme whose category matches the asked
   question category (eligibility questions never get benefit-only passages).
2. Choose ONE evidence tier, best first:
      official records of that category  >  legacy (team-written, unverified)
      records of that category  >  official documents that list the category
      only as a secondary topic.
   Legacy records are not mixed with official ones, so an undated legacy
   "₹10,000-13,500" claim cannot sit next to a dated government order.
3. From each chosen passage pick the 1-2 sentences most relevant to the
   question category (government-order headers, reference lists and
   "copy to" blocks are skipped). The full passage is never pasted into the
   answer; it stays available in ``used_passages`` / ``evidence``.
4. Historical and currentness-unverified sources are framed as such inside
   the answer text, and the UI shows structured ``notices``.
5. If nothing qualifies, abstain with a category-specific message.
"""
import re
from urllib.parse import urlsplit, urlunsplit

from answer_language import render_evidence, render_scheme_name, summarize_evidence
from config import (
    CATEGORY_EN, CATEGORY_TE, NO_EVIDENCE_EN, NO_EVIDENCE_TE, OOS_EN, OOS_TE,
)
from query_analysis import detect_category, evidence_category

# Hard ceiling on the answer text. Real answers are far shorter; this guards
# against any future regression that would paste a whole document.
MAX_ANSWER_CHARS = 680
EXCERPT_CHARS = 240

_MONEY = re.compile(r"(?:₹|రూ\.?|\bRs\.?)\s*\d", re.IGNORECASE)

_CATEGORY_KEYWORDS = {
    "ELIGIBILITY": (
        "eligib", "pattadar", "rofr", "forest right", "title holder", "patta holder",
        "beneficiar", "owned", "owning", "land record", "ccla", "tenant",
        "exclu", "sold", "mutat", "income", "resident", "family", "card holder",
        "agency area", "fresh pattadar", "years of age",
        "అర్హ", "పట్టాదారు", "కౌలు", "ఆదాయ", "వయస్సు", "కుటుంబ", "భూమి",
        "వర్తిస్తుంది", "వర్తించదు", "కార్డు",
    ),
    "EXCEPTIONS": (
        "exclu", "not eligible", "ineligible", "sold", "mutat", "not extended",
        "no provision", "వర్తించదు", "వర్తించని", "అనర్హ",
    ),
    "BENEFITS": (
        "rs", "₹", "per acre", "grant", "assistance", "amount", "season", "coverage",
        "lakh", "investment support", "ఎకరాకు", "రూ", "లక్ష", "ప్రయోజన", "సహాయం",
        "జమ", "కవరేజ్", "చికిత్స",
    ),
    "REQUIRED_DOCUMENTS": (
        "document", "certificate", "aadhaar", "aadhar", "passbook", "proof", "card",
        "పత్ర", "ఆధార్", "సర్టిఫికేట్", "పాస్ పుస్తకం", "జిరాక్స్",
    ),
    "APPLICATION_PROCEDURE": (
        "apply", "application", "register", "portal", "submit", "form", "mee seva",
        "office", "దరఖాస్తు", "నమోదు", "సమర్పించ", "పోర్టల్", "మీ-సేవ", "సంప్రదించ",
    ),
    "PURPOSE": (
        "objective", "aim", "purpose", "initiative", "focus", "to provide",
        "welfare", "empower", "debt", "లక్ష్య", "ఉద్దేశ", "సాధికార",
    ),
}

# Fragments typical of government-order front matter or web-page chrome.
_BOILERPLATE = (
    "read the following", "copy to", "by order and in the name", "lr.no", "lr. no",
    "rc.no", "login", "captcha", "visitor", "copyright", "all rights reserv",
    "abstract", "orders - issued", "orders – issued", "the state information officer",
    "hon’ble minister", "hon'ble minister", "user name", "password",
)

# Telugu labels for concepts found in English evidence. Used only when the
# answer is Telugu and the official evidence is English: a label is shown
# only if its pattern actually occurs in the selected evidence sentences, so
# nothing is added that the source does not mention.
_ELIG = ("ELIGIBILITY",)
_BEN = ("BENEFITS",)
_PUR = ("PURPOSE", "BENEFITS")
_CONCEPTS_TE = (
    (r"\bpattadar", "పట్టాదారులు (భూ రికార్డుల్లో పేరు ఉన్న భూ యజమానులు)", _ELIG),
    (r"\brofr\b|forest rights", "అటవీ హక్కుల గుర్తింపు (RoFR) పట్టాదారులు", _ELIG),
    (r"\bccla\b|land records|digitally signed data", "ప్రభుత్వ భూ రికార్డుల (CCLA) డేటా", _ELIG),
    (r"agency area", "ఏజెన్సీ ప్రాంతాలు", _ELIG),
    (r"sold out|mutat", "అమ్మిన / మ్యుటేషన్ అయిన భూమికి సహాయం నిలిపివేత", _ELIG),
    (r"fresh pattadar", "కొత్త పట్టాదారులను తదుపరి ఆర్థిక సంవత్సరం నుంచి పరిగణించడం", _ELIG),
    (r"aadha?ar", "ఆధార్ ఆధారిత నకిలీ తొలగింపు", _ELIG),
    (r"\btenant", "కౌలు రైతులు", _ELIG),
    (r"bank account|e-kuber", "బ్యాంకు ఖాతాకు నేరుగా జమ", _BEN),
    (r"horticulture", "వ్యవసాయ మరియు ఉద్యాన పంటలు", _BEN),
    (r"seeds|fertili[sz]ers|pesticides", "విత్తనాలు, ఎరువులు, పురుగుమందుల కొనుగోలుకు పెట్టుబడి సాయం", _PUR),
    (r"\bdebt", "రైతులను అప్పుల నుంచి బయటపడేయడం", _PUR),
)

_TELUGU_ABBREV_STEMS = ("రూ", "ము", "సి", "జి", "వి", "సం", "కి", "వరకు", "తో", "పై", "కింద", "మీద", "లో", "గా")


def _looks_tex_abbreviation(block, end):
    """True when the period at ``end`` closes a Telugu abbreviation
    (``రూ.``, ``ము.``, ``సి.``) instead of a sentence."""
    look = block[max(0, end - 8):end]
    return any(look.endswith(s) for s in _TELUGU_ABBREV_STEMS)


_PER_ACRE = re.compile(
    r"Rs\.?\s*([\d,]+)\s*/?-?\s*(?:/\s*|per\s+)acre\s*(?:/\s*|per\s+)?"
    r"(?:farmer\s+)?(?:each\s+)?season",
    re.IGNORECASE,
)

_ABBREVIATIONS = {
    "rs", "no", "dt", "lr", "sri", "smt", "dept", "govt", "viz", "etc", "rc",
    "agri", "dr", "mr", "mrs", "ms", "st", "vs", "e.g", "i.e", "nos", "rt", "g.o",
}


def _canonical_url(url):
    value = (url or "").strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _requested_category(question):
    """Kept for backwards compatibility; delegates to the shared detector."""
    return detect_category(question)


def _contains_telugu(text):
    return bool(re.search(r"[\u0c00-\u0c7f]", text or ""))


def _is_official(p):
    return (
        p.get("source_role") != "legacy_record"
        and p.get("source_classification") != "legacy_unverified"
        and p.get("verification_status") != "legacy_unverified"
    )


def _is_historical(p):
    return str(p.get("temporal_scope") or "").startswith("historical_") or (
        p.get("source_classification") == "official_historical"
    )


def _supports(p, category):
    return category in (p.get("supported_categories") or [])


def split_sentences(text):
    """Split prose and numbered order clauses into sentences.

    Abbreviations such as "Rs." and "G.O.Rt.No." do not end a sentence.
    """
    value = re.sub(r"\s+", " ", text or "").strip()
    if not value:
        return []
    # Numbered clauses "1) ...", "12) ..." in government orders start new units.
    value = re.sub(r"\s(?=\d{1,2}\)\s)", "\n", value)
    pieces = []
    for block in value.split("\n"):
        start = 0
        for match in re.finditer(r"[.!?।]\s+", block):
            end = match.start()
            prev = re.search(r"([A-Za-z.]+)$", block[start:end])
            token = (prev.group(1) if prev else "").casefold().strip(".")
            if token in _ABBREVIATIONS:
                continue
            # Telugu abbreviations ("రూ.", "ము.", "సి.") are not sentence ends.
            if end >= 1:
                look = block[max(0, end - 8):end]
                if any(look.endswith(s) for s in _TELUGU_ABBREV_STEMS):
                    continue
            nxt = block[match.end():match.end() + 1]
            if nxt and nxt.islower():
                continue
            pieces.append(block[start:match.end()].strip())
            start = match.end()
        tail = block[start:].strip()
        if tail:
            pieces.append(tail)
    return [p for p in pieces if p]


def _keyword_in(keyword, low_text):
    """ASCII keywords match at a word start ("rs" must not match "holders");
    Telugu keywords are stems and match anywhere."""
    if all(ord(ch) < 128 for ch in keyword):
        return re.search(r"(?<![a-z0-9])" + re.escape(keyword), low_text) is not None
    return keyword in low_text


def _sentence_score(sentence, category, query_tokens):
    low = sentence.casefold()
    if len(sentence) < 25:
        return -3.0
    score = 0.0
    if any(b in low for b in _BOILERPLATE):
        score -= 4.0
    if re.search(r"g\.?o\.?\s*rt\.?\s*no", low) and re.search(r"\bdt[:.]", low):
        score -= 3.0  # reference lists of earlier orders
    hits = sum(1 for kw in _CATEGORY_KEYWORDS.get(category, ()) if _keyword_in(kw, low))
    if category in _CATEGORY_KEYWORDS and hits == 0:
        return -1.0  # query-word overlap alone never makes a sentence on-topic
    score += hits
    if category == "BENEFITS" and _MONEY.search(sentence):
        score += 1.5
        if re.search(r"\b20\d{2}-\d{2}\b", sentence):
            score += 0.5  # amount tied to a stated period
    if category != "BENEFITS" and _MONEY.search(sentence):
        # An amount inside an eligibility/procedure sentence is usually a
        # benefit figure leaking in; keep it only if the sentence is strongly
        # on-topic (e.g. an income ceiling for eligibility).
        score -= 1.5
    score += 0.3 * sum(1 for t in query_tokens if len(t) > 3 and t in low)
    if len(sentence) > 420:
        score -= 1.0
    return score


def select_sentences(content, category, question="", limit=2, whole_if_short=False):
    """Return up to ``limit`` on-topic sentences, best first.

    ``whole_if_short``: a short record already labelled with the asked
    category is itself the evidence, even if no keyword fires.
    """
    # Lower the floor: a short, correctly-labelled record is evidence even
    # when no keyword fires, so Telugu questions on documents/benefits never
    # fall through to the empty ``all_english`` path.
    if whole_if_short and content and len(content) <= EXCERPT_CHARS:
        return [content]
    tokens = re.findall(r"[\u0c00-\u0c7f]+|[A-Za-z0-9]+", (question or "").casefold())
    sentences = split_sentences(content)
    scored = [(_sentence_score(s, category, tokens), i, s) for i, s in enumerate(sentences)]
    good = [item for item in scored if item[0] > 0]
    good.sort(key=lambda item: (-item[0], item[1]))
    picked = good[:limit]  # best first
    return [_window(_clean_excerpt(s), category) for _, _, s in picked]


def _window(sentence, category, size=EXCERPT_CHARS):
    """Shorten a long sentence around its most relevant part instead of
    always keeping its (often irrelevant) beginning."""
    if len(sentence) <= size:
        return sentence
    low = sentence.casefold()
    anchor = None
    if category == "BENEFITS":
        match = _MONEY.search(sentence)
        anchor = match.start() if match else None
    if anchor is None:
        hits = [low.find(k) for k in _CATEGORY_KEYWORDS.get(category, ()) if low.find(k) >= 0]
        anchor = min(hits) if hits else 0
    start = max(0, anchor - 90)
    # begin at a word boundary
    if start:
        space = sentence.find(" ", start)
        start = space + 1 if 0 <= space < anchor else start
    end = min(len(sentence), start + size)
    piece = sentence[start:end].strip()
    if end < len(sentence):
        cut = piece.rfind(" ")
        piece = (piece[:cut] if cut > size // 2 else piece) + "…"
    return ("…" if start else "") + piece


def _clean_excerpt(sentence):
    value = re.sub(r"^\(?\d{1,2}\)\s*", "", sentence.strip())
    value = re.sub(r"\s+(?:and|or|&)$", "", value)
    value = re.sub(r"\s+\d{1,2}$", "", value)  # page numbers glued to the end
    return value.strip()


def _period_label(p):
    scope = str(p.get("temporal_scope") or "")
    match = re.search(r"(\d{4})(?:_(\d{2}))?", scope)
    if match:
        return match.group(1) + (f"-{match.group(2)}" if match.group(2) else "")
    date = p.get("published_date") or ""
    return date[:4] if date else ""


def _short_title(p):
    title = p.get("source_title") or p.get("scheme") or ""
    return title.split(" - ")[0].strip() if " - " in title else title


def _format_date(value):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value or "")
    return f"{match.group(3)}-{match.group(2)}-{match.group(1)}" if match else (value or "")


def _source_label(p):
    title = _short_title(p)
    date = _format_date(p.get("published_date") or p.get("updated_date") or "")
    return f"{title} ({date})" if date else title


def _unique_sources(passages):
    seen = {}
    for p in passages:
        key = _canonical_url(p.get("source_url")) or (p.get("scheme"), p.get("doc_id"))
        item = {
            "scheme": p.get("scheme"), "source_url": p.get("source_url") or "",
            "doc_id": p.get("doc_id"), "source_title": p.get("source_title", ""),
            "published_date": p.get("published_date"),
            "updated_date": p.get("updated_date"),
            "source_classification": p.get("source_classification", ""),
            "verification_status": p.get("verification_status", ""),
            "temporal_scope": p.get("temporal_scope", ""),
            "authority": p.get("issuing_authority") or p.get("authority") or "",
        }
        if key not in seen:
            seen[key] = item
        else:
            existing = seen[key]
            for field in ("source_title", "published_date", "updated_date",
                          "source_classification", "doc_id", "temporal_scope", "authority"):
                if not existing.get(field) and item.get(field):
                    existing[field] = item[field]
    return list(seen.values())


def _source_quality(chosen):
    statuses = {p.get("verification_status") for p in chosen if p.get("verification_status")}
    classifications = {p.get("source_classification") for p in chosen if p.get("source_classification")}
    if any(_is_historical(p) for p in chosen):
        return "historical_official"
    if "legacy_unverified" in classifications or "legacy_unverified" in statuses:
        return "legacy_unverified"
    if "official_accessed_currentness_unverified" in statuses:
        return "official_accessed_currentness_unverified"
    if statuses == {"verified_current"}:
        return "verified_current"
    return "verified_or_mixed"


def _tiers(candidates, category):
    """Evidence tiers for a category, best first (see module docstring)."""
    primary_official = [p for p in candidates if p.get("category") == category and _is_official(p)]
    primary_legacy = [p for p in candidates if p.get("category") == category and not _is_official(p)]

    return [(name, tier) for name, tier in (
        ("official", primary_official), ("legacy", primary_legacy)) if tier]


def _abstain(category, scheme, in_scope, language, passages, source_hints):
    # State is carried in the chunk; look it up if the scheme is shared by
    # multiple states (e.g. Rythu Bharosa AP vs Telangana).
    state = None
    state_map = {p.get("scheme"): p.get("state", "") for p in passages}
    if state_map:
        state = state_map.get(scheme, "")
    if category and scheme:
        if language == "te":
            text = (
                f"{render_scheme_name(scheme, state=state, language='te')} పథకానికి "
                f"«{CATEGORY_TE.get(category, category)}» గురించి నమ్మదగిన ఆధారం మా జ్ఞాన నిధిలో లేదు. "
                "అందుకే ఊహించి సమాధానం ఇవ్వడం లేదు. కింద ఉన్న అధికారిక మూలాలను చూడండి."
            )
        else:
            text = (
                f"The knowledge base has no reliable evidence about "
                f"{CATEGORY_EN.get(category, category).lower()} for "
                f"{render_scheme_name(scheme, state=state, language='en')}, so no answer was guessed. "
                "Please check the official sources listed below."
            )
    else:
        text = (NO_EVIDENCE_TE if in_scope else OOS_TE) if language == "te" else (
            NO_EVIDENCE_EN if in_scope else OOS_EN
        )
    return {
        "text": text,
        "grounded": False,
        "status": "no_evidence",
        "used_passages": [],
        "sources": [],
        "source_hints": _unique_sources(source_hints or passages),
        "notices": [],
        "evidence": [],
        "category": category,
    }


def _caveat(quality, language, periods):
    period_txt = ", ".join(p for p in periods if p)
    if quality == "historical_official":
        if language == "te":
            return (
                "ప్రస్తుత నిబంధనలను నిర్ధారించే తాజా ధృవీకరించిన ఆధారం మా వద్ద లేదు. "
                f"అందుబాటులో ఉన్నవి పాత ప్రభుత్వ ఉత్తర్వులు ({period_txt}); "
                "వాటి ఆధారంగా ఇప్పటి నియమాలను ఖచ్చితంగా చెప్పలేము."
            )
        return (
            f"No verified current evidence. These are older government orders ({period_txt}); "
            "today's rules cannot be confirmed from them."
        )
    if quality == "legacy_unverified":
        return (
            "ఈ సమాచారం ధృవీకరించని పాత రికార్డు నుండి తీసుకున్నది; ఇది ప్రస్తుత నియమమని నిర్ధారించలేదు."
            if language == "te" else
            "This comes from an older, unverified project record; it has not been confirmed as current."
        )
    if quality == "official_accessed_currentness_unverified":
        return ""


def _closing(language, quality):
    if quality == "verified_current":
        return ""
    return (
        "ప్రస్తుత నిబంధనలను సంబంధిత అధికారి వద్ద లేదా కింద ఇచ్చిన అధికారిక మూలంలో ధృవీకరించండి."
        if language == "te" else
        "Confirm the current rules with the relevant office or the official source below."
    )


def _telugu_points_from_english(chosen, excerpts_by_doc, category):
    """Grounded Telugu bullet points for English official evidence.

    Amounts are read only from the selected sentences and always tied to the
    order (title + date + period) they come from.
    """
    points = []
    if category == "BENEFITS":
        for p in chosen:
            amounts = []
            for sentence in excerpts_by_doc.get(p["doc_id"], []):
                for match in _PER_ACRE.finditer(sentence):
                    amount = "రూ. " + match.group(1).replace(",", "")
                    year = re.search(r"\b(20\d{2}-\d{2})\b", sentence[match.end():match.end() + 60])
                    if year:
                        amount += f" ({year.group(1)})"
                    base = amount.split(" (")[0]
                    same = [a for a in amounts if a.split(" (")[0] == base]
                    if not same:
                        amounts.append(amount)
                    elif year and same[0] == base:
                        amounts[amounts.index(base)] = amount
            if amounts:
                period = _period_label(p)
                points.append(
                    f"{_source_label(p)}"
                    + (f" [{period} ఉత్తర్వు]" if period else "")
                    + ": ఎకరాకు ఒక్కో సీజన్‌కు " + ", ".join(amounts)
                )
    joined = " ".join(" ".join(excerpts_by_doc.get(p["doc_id"], [])) for p in chosen).casefold()
    for pattern, label, categories in _CONCEPTS_TE:
        if category in categories and re.search(pattern, joined) and label not in points:
            points.append(label)
    return points[:5]


def generate_answer(passages, in_scope=True, min_bm25=0.0, intent=None, question=None,
                    scheme=None, category=None, source_hints=None, language="te",
                    category_uncertain=False):
    """Build a concise answer from retrieved passages, or abstain.

    ``category`` is the question category (ELIGIBILITY, BENEFITS, ...). When it
    is not given it is detected from the question; failing that the
    classifier ``intent`` is used. A passage is considered only if it shares
    tokens with the query (BM25 > ``min_bm25``) and belongs to ``scheme``.
    """
    language = "en" if language == "en" else "te"
    question_category = category or detect_category(question) or intent
    wanted = evidence_category(question_category) if question_category else None

    lexical = [
        p for p in passages
        if (p.get("score_bm25") if p.get("score_bm25") is not None else 1.0) > min_bm25
        and (not scheme or p.get("scheme") == scheme)
    ]
    if wanted:
        tiers = _tiers(lexical, wanted)
    else:
        # No category signal at all: answer only from the best-ranked
        # passage's category, never a mix of topics.
        top_cat = lexical[0].get("category") if lexical else None
        tiers = [("uncategorised", [p for p in lexical if p.get("category") == top_cat])]
        wanted = top_cat
        category_uncertain = True

    # Use the best tier that yields usable sentences. An official page that
    # is only navigation chrome yields nothing, so the next tier is tried.
    excerpts_by_doc, chosen, tier = {}, [], None
    for tier_name, candidates in tiers:
        for p in candidates:
            if len(chosen) >= 3:
                break
            if question_category == "EXCEPTIONS":
                picks = select_sentences(p.get("content", ""), "EXCEPTIONS", question or "", 2)
            else:
                picks = select_sentences(p.get("content", ""), wanted, question or "", 3,
                                         whole_if_short=True)
            if picks:
                chosen.append(p)
                excerpts_by_doc[p["doc_id"]] = picks
        if chosen:
            tier = tier_name
            break

    if not chosen:
        return _abstain(question_category, scheme, in_scope, language, passages, source_hints)

    quality = _source_quality(chosen)
    notices = []
    if quality == "historical_official":
        notices.append("historical")
    if any(p.get("verification_status") == "official_accessed_currentness_unverified" for p in chosen):
        notices.append("currentness_unverified")
    if quality == "legacy_unverified":
        notices.append("legacy_unverified")
    if category_uncertain:
        notices.append("category_uncertain")

    # Most recent historical order first; otherwise keep retrieval order.
    if quality == "historical_official":
        chosen.sort(key=lambda p: p.get("published_date") or "", reverse=True)

    schemes = []
    scheme_states = {}
    for p in chosen:
        if p["scheme"] not in schemes:
            schemes.append(p["scheme"])
            scheme_states[p["scheme"]] = p.get("state", "")
    shown_category = question_category if question_category in CATEGORY_EN else wanted
    cat_label = (CATEGORY_TE if language == "te" else CATEGORY_EN).get(shown_category, "")
    header = " · ".join(
        render_scheme_name(s, state=scheme_states.get(s, ""), language=language)
        for s in schemes
    )
    if cat_label:
        header = f"{header} — {cat_label}"
    lines = [header]
    periods = sorted({_period_label(p) for p in chosen if _is_historical(p)})
    caveat = _caveat(quality, language, periods)
    if caveat:
        lines.append(caveat)

    # Dedup: skip passages from the same scheme if the scheme is already covered by an earlier one.
    schemes_seen = set()
    deduped = []
    for p in chosen:
        scheme = p.get("scheme")
        if scheme in schemes_seen:
            continue
        schemes_seen.add(scheme)
        deduped.append(p)
    chosen = deduped
    chosen = deduped

    # Compute the English-only check **before** the if/else so the else
    # branch always has a defined value (fixes a NameError in the else branch).
    # Compute the English-only check before the if/else so the else branch
    # always has a defined value (fixes a NameError in the else branch).
    all_english = all(
        not _contains_telugu(" ".join(excerpts_by_doc[p["doc_id"]])) for p in chosen
    )

    if language == "te" and all_english:
        points = _telugu_points_from_english(chosen, excerpts_by_doc, wanted)
        if points:
            lines.append(
                "పాత ఉత్తర్వుల్లో ప్రస్తావించిన అంశాలు:" if quality == "historical_official"
                else "అధికారిక ఆధారంలో (ఆంగ్లంలో) ప్రస్తావించిన అంశాలు:"
            )
            lines.extend("• " + pt for pt in points)
        else:
            lines.append(render_evidence(excerpts_by_doc[chosen[0]["doc_id"]][0], "te"))
    else:
        # Show at most 2 sentences from the best source and 1 from each other
        # source, so several documents fit in a short answer.
        many = len(chosen) > 1
        shown = {
            p["doc_id"]: [_window(x, wanted, 170 if many else 230)
                          for x in excerpts_by_doc[p["doc_id"]][:1 if many else 2]]
            for p in chosen[:3]
        }
        bullets = sum(len(v) for v in shown.values()) > 1
        for p in chosen[:3]:
            if _is_historical(p):
                lines.append(f"{_source_label(p)} ప్రకారం:" if language == "te"
                             else f"According to {_source_label(p)}:")
            for sentence in shown[p["doc_id"]]:
                # Show the raw Telugu sentence only when the answer language is
                # Telugu. For English, render_evidence() returns "" (suppressed)
                # when no English translation pattern matches, so English answers
                # never contain Telugu script.
                if language == "te" and _contains_telugu(sentence):
                    rendered = sentence
                else:
                    rendered = render_evidence(sentence, language)
                lines.append(("• " + rendered) if bullets else rendered)

    evidence = [
        {
            "doc_id": p["doc_id"],
            "title": p.get("source_title") or p.get("scheme"),
            "authority": p.get("issuing_authority") or p.get("authority") or "",
            "date": p.get("published_date") or p.get("updated_date"),
            "verification_status": p.get("verification_status", ""),
            "source_classification": p.get("source_classification", ""),
            "temporal_scope": p.get("temporal_scope", ""),
            "url": p.get("source_url") or "",
            "excerpts": excerpts_by_doc[p["doc_id"]],
        }
        for p in chosen
    ]
    return {
        "text": _fit(lines),
        "grounded": True,
        "status": "grounded",
        "used_passages": chosen,
        "sources": _unique_sources(chosen),
        "source_quality": quality,
        "notices": notices,
        "evidence": evidence,
        "evidence_tier": tier,
        "category": wanted,
        "excluded_doc_ids": [p["doc_id"] for p in lexical if p not in chosen],
    }


def _fit(lines):
    """Join answer lines within MAX_ANSWER_CHARS.

    The header, caveat and closing advice are always kept; evidence bullets
    are shortened, then dropped (longest first) if still too long.
    """
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    if len(text) > MAX_ANSWER_CHARS:
        lines = [_window(l, "", 170) if l.startswith("• ") else l for l in lines]
        text = "\n".join(lines)
    while len(text) > MAX_ANSWER_CHARS:
        bullets = [i for i, line in enumerate(lines) if line.startswith("• ")]
        if len(bullets) <= 1:
            break
        lines.pop(max(bullets, key=lambda i: len(lines[i])))
        text = "\n".join(lines)
    # drop "According to X:" headings whose bullets were all removed
    cleaned = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.endswith(":") and not line.startswith("• ") and not nxt.startswith("• "):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    if len(text) > MAX_ANSWER_CHARS:
        text = summarize_evidence(text, MAX_ANSWER_CHARS)
    return text
