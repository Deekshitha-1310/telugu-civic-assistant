"""Load and group the scheme knowledge base. Does not invent scheme facts."""
import json
import os
from collections import OrderedDict

from paths import KNOWLEDGE_CORPUS_PATH, SCHEMES_PATH


def load_chunks(path=None):
    with open(path or SCHEMES_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    if path is None and os.path.exists(KNOWLEDGE_CORPUS_PATH):
        with open(KNOWLEDGE_CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    unique = {}
    for chunk in chunks:
        unique[chunk.get("doc_id")] = chunk
    chunks = list(unique.values())
    required = {"doc_id", "scheme", "category", "source_url", "content"}
    for chunk in chunks:
        missing = required - set(chunk)
        if missing:
            raise ValueError(f"{chunk.get('doc_id')} missing fields: {missing}")
        chunk.setdefault("state", _state_from_url(chunk.get("source_url", "")))
        chunk.setdefault("authority", _authority_from_url(chunk.get("source_url", "")))
        chunk.setdefault("topic", chunk.get("category", ""))
        chunk.setdefault("source_title", chunk.get("scheme", ""))
        chunk.setdefault("document_id", chunk["doc_id"])
        chunk.setdefault("retrieval_category", chunk.get("category", ""))
        chunk.setdefault("scheme_identifier", "")
        chunk.setdefault("published_date", None)
        chunk.setdefault("updated_date", None)
        chunk.setdefault("source_role", "legacy_record")
        chunk.setdefault(
            "source_classification",
            "legacy_unverified" if chunk.get("source_role") == "legacy_record"
            else "official_primary",
        )
        chunk.setdefault(
            "verification_status",
            "legacy_unverified" if chunk.get("source_role") == "legacy_record"
            else "official_accessed_currentness_unverified",
        )
    return chunks


def _state_from_url(url):
    host = (url or "").lower()
    if ".gov.in" not in host:
        return "Unknown"
    if "telangana" in host or host.endswith(".telangana.gov.in") or "aasara" in host:
        return "Telangana"
    if (
        host.endswith(".ap.gov.in") or ".ap.gov.in/" in host or "ap.gov.in" in host
        or any(part in host for part in ("apagrisnet", "aarogyasri", "jaganannaammavodi"))
    ):
        return "Andhra Pradesh"
    return "Central Government"


def _authority_from_url(url):
    host = (url or "").split("/")[2] if "://" in (url or "") else ""
    return host or "Unknown official source"


def group_schemes(chunks):
    grouped = OrderedDict()
    for chunk in chunks:
        grouped.setdefault(chunk["scheme"], []).append(chunk)
    return grouped


def scheme_summaries(chunks):
    summaries = []
    for name, docs in group_schemes(chunks).items():
        categories = sorted({d["category"] for d in docs})
        urls = []
        for d in docs:
            if d.get("source_url") and d["source_url"] not in urls:
                urls.append(d["source_url"])
        first = docs[0]["content"]
        summaries.append({
            "scheme": name,
            "n_chunks": len(docs),
            "categories": categories,
            "source_urls": urls,
            "short_description": first,
            "chunks": docs,
            "verified_on": max(
                (d.get("updated_date") or d.get("published_date") or "")
                for d in docs
            ) or None,
        })
    return summaries
