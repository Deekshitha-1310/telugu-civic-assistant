"""Repeatable official-source ingestion.

Usage: python src/ingest.py [--sources data/knowledge_sources.json]
The command keeps legacy scheme records, writes a JSONL corpus, and reports
fetch/index counts. Network failures are recorded rather than fatal.
"""
import argparse
import difflib
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, build_opener, urlopen
from urllib.robotparser import RobotFileParser

from kb import load_chunks
from paths import (
    DISCOVERED_SOURCES_PATH,
    KNOWLEDGE_CACHE_DIR,
    KNOWLEDGE_CORPUS_PATH,
    KNOWLEDGE_SOURCES_PATH,
    SCHEMES_PATH,
)


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
        self._skip_elements = []
        self.skip_tags = {"script", "style", "noscript", "nav", "header", "footer",
                          "form", "aside", "button", "svg"}

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs = dict(attrs)
        marker = f"{attrs.get('id', '')} {attrs.get('class', '')}".lower()
        boilerplate_markers = (
            "nav", "menu", "header", "footer", "sidebar", "breadcrumb", "login",
            "captcha", "banner", "popup", "modal", "social", "visitor", "copyright",
        )
        should_skip = tag in self.skip_tags or any(
            word in marker for word in boilerplate_markers
        )
        if should_skip:
            self.skip += 1
            self._skip_elements.append(tag)

    def handle_endtag(self, tag):
        if self._skip_elements and self._skip_elements[-1] == tag and self.skip:
            self._skip_elements.pop()
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.title = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "a" and attrs.get("href"):
            self.links.append((attrs["href"], ""))
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)


def _clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_cached_text(text):
    value = _clean(text)
    patterns = (
        r"Join our WhatsApp Channel\s*[×x]?",
        r"Login(?:\s+Login)?\s+User Name:\s*Password:\s*Captcha:?",
        r"Visitors Count\s*:\s*[\d,]+",
        r"Copyright\s*:?.*?(?:National Informatics Centre|All rights reserved)",
        r"©\s*Design\s*&\s*Developed\s*By.*$",
        r"\*?Due to the upgradation of the application.*?restored shortly\.?",
        r"Attention Doctors and Nurses:.*?(?:click here|https?://\S+)",
        r"Due to the Model Code of Conduct.*?SEC\.\)?\s*",
        r"(?:Bill Discounting Guidance video|Guidance Video for Updation of Aadhaar and Other in the EHS Application)",
        r"Your browser does not support the video tag\.?",
        r"\bClick here\b",
    )
    for pattern in patterns:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    return _clean(value)


def _substantive_sentences(text, max_chars=5000):
    value = _clean_cached_text(text)
    sentences = re.split(r"(?<=[.!?।])\s+", value)
    kept = [
        sentence.strip(" -:;")
        for sentence in sentences
        if len(sentence.strip()) >= 35
        and not re.search(
            r"\b(login|captcha|visitor|copyright|whatsapp|menu|home|contact us|forms?)\b",
            sentence,
            flags=re.IGNORECASE,
        )
    ]
    return _clean(" ".join(kept))[:max_chars]


def _scope_source_text(source, text):
    """Keep evidence narrowly scoped when a source entry has an explicit scope."""
    scope = source.get("evidence_scope")
    if scope == "purpose_only":
        sentences = re.split(r"(?<=[.!?।])\s+", _clean_cached_text(text))
        selected = [
            sentence.strip()
            for sentence in sentences
            if any(term.casefold() in sentence.casefold() for term in (
                "Dr. NTR Vaidya Seva", "quality healthcare", "poor",
                "flagship scheme", "health initiatives",
            ))
        ]
        return _clean(" ".join(selected))[:1200]
    return text


def _document_date(text):
    match = re.search(
        r"(?:last updated|updated on|published on|తాజా నవీకరణ)\s*[:\-]?\s*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        text or "", flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _cache_key(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _fetch_document(url, timeout=20, max_bytes=8_000_000, cache_dir=None, refresh=False):
    cache_dir = Path(cache_dir or KNOWLEDGE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(url)}.json"
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["text"] = _substantive_sentences(cached.get("text", ""))
        return cached
    req = Request(url, headers={"User-Agent": "Telugu-Civic-Assistant/1.0"})
    with urlopen(req, timeout=timeout) as response:
        content_length = int(response.headers.get("Content-Length", "0") or 0)
        if content_length and content_length > max_bytes:
            raise RuntimeError(f"response exceeds max_bytes ({max_bytes})")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise RuntimeError(f"response exceeds max_bytes ({max_bytes})")
        content_type = response.headers.get("Content-Type", "").lower()
        final_url = response.geturl()
    if ".pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            text = _clean(" ".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages))
            result = {
                "original_url": url, "final_url": final_url, "title": Path(urlparse(final_url).path).name,
                "content_type": content_type, "fetched_at": _now(), "document_date": _document_date(text),
                "text": text, "links": [], "content_hash": hashlib.sha256(body).hexdigest(),
            }
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return result
        except ImportError as exc:
            raise RuntimeError("PDF source requires optional pypdf dependency") from exc
    decoded = body.decode("utf-8", errors="ignore")
    parser = _TextParser()
    parser.feed(decoded)
    link_parser = _LinkParser()
    link_parser.feed(decoded)
    result = {
        "original_url": url, "final_url": final_url, "title": _clean(" ".join(link_parser.title)) or url,
        "content_type": content_type, "fetched_at": _now(), "document_date": _document_date(_clean(" ".join(parser.parts))),
        "text": _substantive_sentences(" ".join(parser.parts)), "links": link_parser.links,
        "content_hash": hashlib.sha256(body).hexdigest(),
    }
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _fetch(url, **kwargs):
    return _fetch_document(url, **kwargs)["text"]


def _domain(url):
    return urlparse(url).netloc.lower().split(":")[0]


def _is_trusted(url, trusted_domains):
    domain = _domain(url)
    return any(domain == item or domain.endswith("." + item) for item in trusted_domains)


def _is_crawlable(url):
    path = urlparse(url).path.lower()
    return not any(token in path for token in ("/login", "/signin", "/account", "/admin")) and (
        not path.endswith("/home-v1.html")
    )


def _is_relevant(url, title, text):
    haystack = f"{url} {title} {text[:4000]}".lower()
    terms = (
        "scheme", "yojana", "bharosa", "pension", "aarogy", "health", "benefit",
        "eligib", "guideline", "government order", "circular", "faq", "application",
        "form", "document", "grievance", "helpline", "status", "పథకం", "అర్హత",
        "దరఖాస్తు", "పెన్షన్", "రైతు", "ఆరోగ్యశ్రీ",
    )
    score = sum(term in haystack for term in terms)
    path_signal = any(
        term in haystack for term in (
            "form", "guideline", "circular", "order", "faq", "contact",
            "status", "grievance", "helpline", "document",
        )
    )
    return url.lower().endswith(".pdf") or (score >= 2 and path_signal)


def _robots_allowed(url, timeout):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        with urlopen(Request(robots_url, headers={"User-Agent": "Telugu-Civic-Assistant/1.0"}), timeout=timeout) as response:
            parser.parse(response.read(200_000).decode("utf-8", errors="ignore").splitlines())
        return parser.can_fetch("Telugu-Civic-Assistant/1.0", url), None
    except Exception as exc:
        return True, f"robots.txt unavailable: {type(exc).__name__}: {exc}"


def discover_sources(sources, max_depth=1, pages_per_domain=10, timeout=20,
                     max_bytes=8_000_000, delay=0.25, refresh=False,
                     focus_terms=None):
    trusted = set()
    queue = []
    focus_terms = tuple(term.lower() for term in (focus_terms or ()))
    for source in sources:
        parsed = urlparse(source["url"])
        trusted.add(parsed.netloc.lower().split(":")[0])
        queue.append((source["url"], source, 0))
    seen = set()
    discovered = []
    failures = []
    skipped = []
    fetched_by_domain = {}
    source_attempts = {}
    successful_pages = []
    while queue:
        url, source, depth = queue.pop(0)
        url = urldefrag(url)[0]
        if url in seen:
            continue
        seen.add(url)
        domain = _domain(url)
        if depth > max_depth or fetched_by_domain.get(domain, 0) >= pages_per_domain:
            skipped.append({"url": url, "reason": "crawl_budget"})
            continue
        if not _is_trusted(url, trusted) or not _is_crawlable(url):
            skipped.append({"url": url, "reason": "untrusted_or_protected"})
            continue
        allowed, robots_warning = _robots_allowed(url, timeout)
        if not allowed:
            skipped.append({"url": url, "reason": "robots_denied"})
            continue
        try:
            document = _fetch_document(url, timeout, max_bytes, refresh=refresh)
            fetched_by_domain[domain] = fetched_by_domain.get(domain, 0) + 1
            source_attempts.setdefault(source["url"], {"attempted": True, "success": True})
            successful_pages.append(url)
            time.sleep(max(0.0, delay))
        except Exception as exc:
            source_attempts.setdefault(source["url"], {"attempted": True, "success": False})
            failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        useful = len(document["text"]) >= 160 and _is_relevant(url, document["title"], document["text"])
        discovered.append({
            **source,
            "url": url,
            "original_url": document["original_url"],
            "final_url": document["final_url"],
            "title": document["title"],
            "source_domain": _domain(document["final_url"]),
            "fetched_at": document["fetched_at"],
            "document_date": document["document_date"],
            "discovery_depth": depth,
            "useful": useful,
            "robots_warning": robots_warning,
        })
        if depth < max_depth and "html" in document["content_type"]:
            children = []
            for href, _anchor in document["links"]:
                child = urldefrag(urljoin(document["final_url"], href))[0]
                if _is_trusted(child, trusted) and _is_crawlable(child):
                    priority = 0 if (
                        child.lower().endswith(".pdf")
                        or any(term in child.lower() for term in (
                            "form", "guideline", "circular", "order", "faq",
                            "contact", "status", "grievance", "helpline",
                        ))
                        or any(term in child.lower() for term in focus_terms)
                    ) else 1
                    children.append((priority, child))
            for _priority, child in sorted(children):
                queue.append((child, source, depth + 1))
    useful_sources = [item for item in discovered if item["useful"]]
    return {
        "discovered": useful_sources,
        "all_visited": discovered,
        "failures": failures,
        "skipped": skipped,
        "domains": sorted(fetched_by_domain),
        "source_attempts": source_attempts,
        "successful_pages": successful_pages,
    }


def _chunks(text, size=900, overlap=120):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def _doc_id(url, ordinal):
    return "web_" + hashlib.sha1(f"{url}#{ordinal}".encode()).hexdigest()[:16]


def _infer_category(title, text, default="GENERAL"):
    haystack = f"{title} {text[:5000]}".lower()
    markers = [
        ("REQUIRED_DOCUMENTS", ("document", "required", "పత్ర", "కాగిత", "డాక్యుమెంట్")),
        ("APPLICATION_PROCEDURE", ("apply", "application", "form", "దరఖాస్తు", "నమోదు")),
        ("DEADLINES", ("deadline", "last date", "పట్టిక", "గడువు", "చివరి తేదీ")),
        ("STATUS_CHECK", ("status", "payment status", "స్థితి", "చెల్లింపు స్థితి")),
        ("GRIEVANCE", ("grievance", "complaint", "appeal", "ఫిర్యాదు", "అప్పీల్")),
        ("HELPLINE", ("helpline", "contact", "phone", "హెల్ప్‌లైన్", "ఫోన్")),
        ("ELIGIBILITY", ("eligib", "eligible", "అర్హత", "అర్హులు")),
        ("BENEFITS", ("benefit", "amount", "payment", "ప్రయోజన", "చెల్లింపు")),
        ("FAQ", ("faq", "frequently asked", "తరచుగా అడిగే")),
        ("PURPOSE", ("purpose", "objective", "aim", "లక్ష్యం", "ఉద్దేశ్యం")),
    ]
    for category, terms in markers:
        if any(term in haystack for term in terms):
            return category
    return default


def _merge_sources(configured, discovered):
    merged = {}
    for source in configured + discovered:
        key = source.get("final_url") or source.get("url")
        if key and source.get("evidence", True) is not False:
            merged[key] = source
    return list(merged.values())


def ingest(sources_path=KNOWLEDGE_SOURCES_PATH, corpus_path=KNOWLEDGE_CORPUS_PATH,
           discover=False, max_depth=1, pages_per_domain=10, timeout=20,
           max_bytes=8_000_000, delay=0.25, refresh=False, focus_terms=None):
    legacy = load_chunks(SCHEMES_PATH)
    by_id = {c["doc_id"]: c for c in legacy}
    previous = {}
    existing_corpus = Path(corpus_path)
    if existing_corpus.exists():
        for line in existing_corpus.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                item.setdefault(
                    "source_classification",
                    "legacy_unverified" if item.get("source_role") == "legacy_record"
                    else "official_primary",
                )
                item.setdefault(
                    "verification_status",
                    "legacy_unverified" if item.get("source_role") == "legacy_record"
                    else "official_accessed_currentness_unverified",
                )
                by_id[item.get("doc_id")] = item
                previous[item.get("final_url") or item.get("source_url", "")] = item.get("document_hash")
    configured = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    discovery = {
        "discovered": [], "all_visited": [], "failures": [], "skipped": [],
        "domains": [], "source_attempts": {}, "successful_pages": [],
    }
    if discover:
        discovery = discover_sources(
            configured, max_depth=max_depth, pages_per_domain=pages_per_domain,
            timeout=timeout, max_bytes=max_bytes, delay=delay, refresh=refresh,
            focus_terms=focus_terms,
        )
        Path(DISCOVERED_SOURCES_PATH).write_text(
            json.dumps(discovery, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    sources = _merge_sources(configured, discovery["discovered"])
    summary = {"sources": len(sources), "configured_sources": len(configured),
               "source_attempts": discovery["source_attempts"],
               "successful_pages": discovery["successful_pages"],
               "evidence_sources": 0, "fetched": 0, "documents": 0,
               "chunks": 0, "failures": discovery["failures"], "skipped": discovery["skipped"],
               "discovered": len(discovery["discovered"]), "domains": discovery["domains"],
               "indexed_records": 0, "changed_documents": 0, "unchanged_documents": 0,
               "duplicate_documents": 0, "useful_documents": 0, "extracted_chunks": 0,
               "useful_evidence_pages": sum(
                   1 for item in discovery["discovered"] if item.get("evidence", True)
               ),
               "useful_discovery_leads": sum(
                   1 for item in discovery["discovered"] if not item.get("evidence", True)
               )}
    known_hashes = {
        item.get("document_hash") for item in by_id.values()
        if item.get("document_hash")
    }
    for source in sources:
        if source.get("evidence", True) is False:
            continue
        summary["evidence_sources"] += 1
        url = source["url"]
        try:
            document = _fetch_document(
                source.get("final_url") or url, timeout=timeout, max_bytes=max_bytes,
                refresh=refresh,
            )
            text = document["text"]
            if len(text) < 80:
                raise RuntimeError("source returned too little text")
            summary["fetched"] += 1
            text = _scope_source_text(source, _substantive_sentences(text))
            parts = _chunks(text)
            summary["documents"] += 1
            summary["extracted_chunks"] += len(parts)
            category = source.get("category", "GENERAL")
            if source.get("discovery_depth", 0) > 0 or category == "GENERAL":
                category = _infer_category(document["title"], text, category)
            source_url = document["final_url"]
            if previous.get(source_url) == document.get("content_hash"):
                summary["unchanged_documents"] += 1
            else:
                summary["changed_documents"] += 1
            same_source = any(
                item.get("final_url") == source_url for item in by_id.values()
            )
            if document.get("content_hash") in known_hashes and not same_source:
                summary["duplicate_documents"] += 1
                continue
            known_hashes.add(document.get("content_hash"))
            summary["useful_documents"] += 1
            for doc_id in [
                doc_id for doc_id, item in by_id.items()
                if item.get("final_url") == source_url
            ]:
                del by_id[doc_id]
            for ordinal, content in enumerate(parts):
                doc_id = _doc_id(source_url, ordinal)
                by_id[doc_id] = {
                    "doc_id": doc_id, "document_id": doc_id, "scheme": source.get("scheme", source.get("title", source_url)),
                    "category": category, "retrieval_category": category,
                    "state": source.get("state", "Central Government"), "authority": source.get("authority", ""),
                    "issuing_authority": source.get("issuing_authority", source.get("authority", "")),
                    "topic": source.get("topic", ""), "source_title": source.get("title", url),
                    "source_url": source_url, "original_url": source.get("original_url", url),
                    "final_url": source_url, "content": content,
                    "scheme_identifier": source.get("scheme_identifier", ""),
                    "scheme_name_as_published": source.get("scheme_name_as_published", ""),
                    "scheme_relationship": source.get("scheme_relationship", ""),
                    "related_schemes": source.get("related_schemes", []),
                    "supported_categories": source.get("supported_categories", [category]),
                    "temporal_scope": source.get("temporal_scope", ""),
                    "published_date": source.get("published_date") or document.get("document_date"),
                    "updated_date": source.get("updated_date"),
                    "source_role": source.get("role", "scheme_evidence"),
                    "source_classification": source.get("source_classification", "official_primary"),
                    "verification_status": source.get(
                        "verification_status",
                        "official_accessed_currentness_unverified",
                    ),
                    "fetched_at": document.get("fetched_at"),
                    "document_hash": document.get("content_hash"),
                }
                summary["chunks"] += 1
        except Exception as exc:
            summary["failures"].append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    records = []
    seen_texts = []
    for record in by_id.values():
        if record.get("source_role") == "legacy_record":
            records.append(record)
            continue
        content = record.get("content", "")
        if any(difflib.SequenceMatcher(None, content, previous).ratio() >= 0.97 for previous in seen_texts):
            summary["duplicate_documents"] += 1
            continue
        seen_texts.append(content)
        records.append(record)
    Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(corpus_path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary["indexed_records"] = len(records)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=KNOWLEDGE_SOURCES_PATH)
    parser.add_argument("--corpus", default=KNOWLEDGE_CORPUS_PATH)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--pages-per-domain", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(ingest(
        args.sources, args.corpus, discover=args.discover, max_depth=args.max_depth,
        pages_per_domain=args.pages_per_domain, timeout=args.timeout,
        max_bytes=args.max_bytes, delay=args.delay, refresh=args.refresh,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
