import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ingest import _TextParser, ingest
from evaluate_coverage import audit
from answer_generator import generate_answer
from kb import load_chunks
from llm_generator import generate_grounded_answer
from retriever import HybridRetriever
from pipeline import CivicAssistant


def test_retriever_defaults_and_empty_query():
    retriever = HybridRetriever(enable_neural=False)
    assert retriever.bm25_weight == 1.8
    assert retriever.dense_weight == 1.0
    assert retriever.rrf_k == 60
    assert retriever.search("") == []


def test_legacy_metadata_and_telugu_retrieval():
    chunks = load_chunks()
    assert len(chunks) >= 16
    assert all(chunk["document_id"] for chunk in chunks)
    assert next(c for c in chunks if c["doc_id"] == "rb_elig_01")["state"] == "Andhra Pradesh"
    result = HybridRetriever(chunks, enable_neural=False).search("రైతు బంధు పత్రాలు", top_k=2)
    assert result and result[0]["source_url"].startswith("https://")


def test_ingestion_is_repeatable_and_keeps_legacy(tmp_path, monkeypatch):
    source_path = tmp_path / "sources.json"
    corpus_path = tmp_path / "corpus.jsonl"
    source_path.write_text(json.dumps([{
        "title": "Fixture", "url": "https://example.gov.in/page", "scheme": "Fixture",
        "state": "Central Government", "authority": "example.gov.in", "category": "GENERAL"
    }]), encoding="utf-8")
    monkeypatch.setattr(
        "ingest._fetch_document",
        lambda url, **_kwargs: {
            "original_url": url, "final_url": url, "title": "Fixture",
            "content_type": "text/html", "fetched_at": "now",
            "document_date": None, "text": "ఇది అధికారిక పథకం సమాచారం " * 40,
            "links": [], "content_hash": "fixture",
        },
    )
    first = ingest(str(source_path), str(corpus_path))
    second = ingest(str(source_path), str(corpus_path))
    assert first["fetched"] == second["fetched"] == 1
    assert first["indexed_records"] == second["indexed_records"]
    assert first["indexed_records"] >= 17


def test_html_extractor_unwinds_marked_boilerplate():
    parser = _TextParser()
    parser.feed(
        '<div class="header"><span>Menu</span></div>'
        '<main><p>Official scheme purpose and eligibility information.</p></main>'
    )
    assert "Official scheme purpose" in " ".join(parser.parts)


def test_ntr_vaidya_record_is_purpose_only_and_not_current_policy_evidence():
    record = next(
        chunk for chunk in load_chunks()
        if chunk.get("source_url") == "https://drntrvaidyaseva.ap.gov.in/"
    )
    assert record["category"] == "PURPOSE"
    assert record["scheme"] == "Dr. NTR Vaidya Seva"
    assert record["scheme_name_as_published"] == "Dr. NTR Vaidya Seva"
    assert "వైఎస్ఆర్ ఆరోగ్యశ్రీ" in record["related_schemes"]
    assert record["verification_status"] == "official_accessed_currentness_unverified"
    assert "formal rename" in record["scheme_relationship"]
    assert "quality healthcare to the poor" in record["content"]
    assert not any(
        marker in record["content"].casefold()
        for marker in ("eligib", "required document", "deadline", "rs.", "apply")
    )

    for category, question in (
        ("ELIGIBILITY", "Who is eligible?"),
        ("REQUIRED_DOCUMENTS", "What documents are required?"),
        ("BENEFITS", "What benefit limit is provided?"),
        ("DEADLINES", "What is the deadline?"),
        ("APPLICATION_PROCEDURE", "How do I apply?"),
    ):
        answer = generate_answer(
            [record], question=question, scheme=record["scheme"], category=category
        )
        assert answer["status"] == "no_evidence"


def test_ntr_about_page_adds_only_purpose_evidence():
    records = [
        chunk for chunk in load_chunks()
        if chunk.get("source_url") == "https://drntrvaidyaseva.ap.gov.in/aboutus"
    ]
    assert len(records) == 1
    record = records[0]
    assert record["scheme"] == "Dr. NTR Vaidya Seva"
    assert record["category"] == "PURPOSE"
    assert record["verification_status"] == "official_accessed_currentness_unverified"
    assert (
        "health care" in record["content"].casefold()
        or "health camps" in record["content"].casefold()
        or "universal health coverage" in record["content"].casefold()
    )
    for category in ("ELIGIBILITY", "REQUIRED_DOCUMENTS", "BENEFITS", "DEADLINES", "APPLICATION_PROCEDURE"):
        answer = generate_answer(
            [record],
            question=f"What is the {category.lower()}?",
            scheme=record["scheme"],
            category=category,
        )
        assert answer["status"] == "no_evidence"


def test_historical_rythu_orders_are_time_scoped_and_warned():
    chunks = [
        chunk for chunk in load_chunks()
        if chunk.get("source_url", "").endswith(("GO280.pdf", "GO202.pdf", "GO265.pdf"))
    ]
    assert len(chunks) == 3
    assert all(
        chunk["source_classification"] == "official_historical"
        and chunk["verification_status"] == "official_accessed_currentness_unverified"
        and chunk["temporal_scope"].startswith("historical_")
        and chunk["published_date"]
        for chunk in chunks
    )
    answer = generate_answer(
        chunks,
        question="What benefit amount is provided?",
        scheme="రైతు బంధు / రైతు భరోసా",
        category="BENEFITS",
        min_bm25=-1.0,
    )
    assert answer["grounded"]
    assert answer["source_quality"] == "historical_official"
    assert all(
        source["temporal_scope"].startswith("historical_")
        for source in answer["sources"]
    )


def test_llm_defaults_to_safe_fallback():
    chunk = {
        "doc_id": "x", "scheme": "పథకం", "category": "BENEFITS",
        "source_url": "https://example.gov.in", "content": "అధికారిక సమాచారం"
    }
    answer = generate_grounded_answer("ప్రయోజనాలు?", [dict(chunk, score_bm25=1.0)])
    assert answer["generation"] == "extractive_fallback"
    assert answer["grounded"] is True


def test_coverage_audit_reports_missing_categories_and_exception_evidence():
    report = audit()
    scheme = report["schemes"]["రైతు బంధు / రైతు భరోసా"]
    assert scheme["official_source_count"] >= 2
    assert scheme["accessed_unverified_source_count"] >= 1
    assert scheme["verified_source_count"] == 0
    assert scheme["questions_with_verified_evidence"] == 0
    assert "DEADLINES" in scheme["missing_categories"]
    exception = next(q for q in scheme["questions"] if q["id"] == "exception-te")
    assert exception["evidence_found"] is True


def test_pipeline_isolates_rythu_bharosa_purpose_evidence():
    result = CivicAssistant().run("రైతు భరోసా పథకం ఉద్దేశం ఏమిటి?")
    assert result["question_category"] == "PURPOSE"
    assert result["scheme"] == "రైతు బంధు / రైతు భరోసా"
    assert result["answer"]["status"] == "grounded"
    assert result["answer"]["used_passages"]
    assert all(
        p["scheme"] == result["scheme"] and p["category"] == "PURPOSE"
        for p in result["answer"]["used_passages"]
    )
    assert all(s["scheme"] == result["scheme"] for s in result["answer"]["sources"])


def test_pipeline_eligibility_and_unsupported_purpose_do_not_mix_schemes():
    bot = CivicAssistant()
    eligibility = bot.run("రైతు భరోసాకు ఎవరు అర్హులు?")
    assert eligibility["question_category"] == "ELIGIBILITY"
    assert all(
        p["scheme"] == "రైతు బంధు / రైతు భరోసా"
        for p in eligibility["answer"]["used_passages"]
    )

    amma_purpose = bot.run("జగనన్న అమ్మ ఒడి పథకం ఉద్దేశం ఏమిటి?")
    assert amma_purpose["question_category"] == "PURPOSE"
    assert amma_purpose["answer"]["status"] == "no_evidence"
    assert not amma_purpose["answer"]["used_passages"]


def test_followup_documents_keeps_active_scheme():
    bot = CivicAssistant()
    first = bot.run("రైతు భరోసాకు ఎవరు అర్హులు?")
    followup = bot.run(
        "దరఖాస్తుకు ఏ పత్రాలు కావాలి?",
        previous_schemes=first["schemes_mentioned"],
    )
    assert followup["scheme"] == "రైతు బంధు / రైతు భరోసా"
    assert followup["question_category"] == "REQUIRED_DOCUMENTS"
    assert all(
        p["scheme"] == followup["scheme"]
        and p["category"] == "REQUIRED_DOCUMENTS"
        for p in followup["answer"]["used_passages"]
    )


def test_amma_vodi_followup_preserves_scheme_and_source_quality():
    bot = CivicAssistant()
    first = bot.run("జగనన్న అమ్మ ఒడి పథకం ఉద్దేశం ఏమిటి?")
    followup = bot.run(
        "దరఖాస్తుకు ఏ పత్రాలు కావాలి?",
        previous_schemes=first["schemes_mentioned"],
    )
    assert first["intent"] == "ELIGIBILITY"
    assert first["question_category"] == "PURPOSE"
    assert first["answer"]["status"] == "no_evidence"
    assert followup["scheme"] == "జగనన్న అమ్మ ఒడి"
    assert followup["question_category"] == "REQUIRED_DOCUMENTS"
    assert followup["answer"]["status"] == "grounded"
    assert followup["answer"]["source_quality"] == "legacy_unverified"
    assert all(
        p["scheme"] == "జగనన్న అమ్మ ఒడి"
        and p["category"] == "REQUIRED_DOCUMENTS"
        for p in followup["answer"]["used_passages"]
    )
    assert all(
        source["scheme"] == "జగనన్న అమ్మ ఒడి"
        and source["source_classification"] == "legacy_unverified"
        for source in followup["answer"]["sources"]
    )


def test_explicit_scheme_switch_overrides_followup_context():
    bot = CivicAssistant()
    amma = bot.run("జగనన్న అమ్మ ఒడి పథకం ఉద్దేశం ఏమిటి?")
    switched = bot.run(
        "రైతు భరోసా పథకం ఉద్దేశం ఏమిటి?",
        previous_schemes=amma["schemes_mentioned"],
    )
    assert switched["scheme"] == "రైతు బంధు / రైతు భరోసా"
    assert all(
        passage["scheme"] == "రైతు బంధు / రైతు భరోసా"
        for passage in switched["answer"]["used_passages"]
    )


def test_language_switch_rerenders_same_query_without_retrieval_changes():
    bot = CivicAssistant()
    telugu = bot.run("రైతు భరోసా పథకం చివరి తేదీ ఏమిటి?", answer_language="te")
    english = bot.run("రైతు భరోసా పథకం చివరి తేదీ ఏమిటి?", answer_language="en")
    assert telugu["answer"]["status"] == english["answer"]["status"] == "no_evidence"
    assert any("\u0c00" <= char <= "\u0c7f" for char in telugu["answer"]["text"])
    assert not any("\u0c00" <= char <= "\u0c7f" for char in english["answer"]["text"])
    assert telugu["scheme"] == english["scheme"]
    assert telugu["question_category"] == english["question_category"] == "DEADLINES"


def test_no_evidence_deduplicates_canonical_source_urls():
    answer = generate_grounded_answer(
        "deadline?",
        [],
        category="DEADLINES",
        source_hints=[
            {
                "scheme": "Scheme", "doc_id": "a", "source_url": "https://example.gov.in/",
                "source_title": "First", "source_classification": "legacy_unverified",
            },
            {
                "scheme": "Scheme", "doc_id": "b", "source_url": "https://EXAMPLE.gov.in",
                "source_title": "Second", "source_classification": "legacy_unverified",
            },
        ],
    )
    assert len(answer["source_hints"]) == 1
    assert answer["source_hints"][0]["source_url"] == "https://example.gov.in/"


def test_no_evidence_preserves_distinct_source_urls_with_same_title():
    answer = generate_grounded_answer(
        "deadline?",
        [],
        category="DEADLINES",
        source_hints=[
            {
                "scheme": "Scheme", "doc_id": "a", "source_url": "https://a.gov.in/",
                "source_title": "Official scheme", "source_classification": "official_primary",
            },
            {
                "scheme": "Scheme", "doc_id": "b", "source_url": "https://b.gov.in/",
                "source_title": "Official scheme", "source_classification": "official_primary",
            },
        ],
    )
    assert [item["source_url"] for item in answer["source_hints"]] == [
        "https://a.gov.in/",
        "https://b.gov.in/",
    ]


def test_aasara_alias_resolves_to_corpus_scheme():
    result = CivicAssistant().run("Aasara pension eligibility")
    assert result["scheme"] == "ఆసరా పెన్షన్ / పెన్షన్ కానుక"
    assert result["question_category"] == "ELIGIBILITY"


def test_coverage_report_exposes_reliability_and_failure_metrics():
    report = audit()
    metrics = report["metrics"]
    assert metrics["question_count"] >= 40
    assert 0.0 <= metrics["evidence_coverage"] <= 1.0
    assert 0.0 <= metrics["verified_evidence_coverage"] <= 1.0
    assert 0.0 <= metrics["retrieval_recall"] <= 1.0
    assert metrics["scheme_category_isolation"] is True
    assert all(
        "relevance_status" in question
        and "failure_reason" in question
        for scheme in report["schemes"].values()
        for question in scheme["questions"]
    )


def test_retrieval_and_answer_filter_mixed_scheme_results():
    chunks = load_chunks()
    results = HybridRetriever(chunks, enable_neural=False).search("పథకం ఉద్దేశం", top_k=10)
    mixed = [
        dict(results[0], scheme="రైతు బంధు / రైతు భరోసా", category="PURPOSE"),
        dict(results[0], scheme="జగనన్న అమ్మ ఒడి", category="ELIGIBILITY"),
    ]
    answer = generate_grounded_answer(
        "రైతు భరోసా పథకం ఉద్దేశం ఏమిటి?",
        mixed,
        scheme="రైతు బంధు / రైతు భరోసా",
        category="PURPOSE",
    )
    assert answer["grounded"]
    assert all(
        p["scheme"] == "రైతు బంధు / రైతు భరోసా"
        and p["category"] == "PURPOSE"
        for p in answer["used_passages"]
    )


def test_deadline_without_evidence_abstains():
    result = CivicAssistant().run("రైతు భరోసా పథకం చివరి తేదీ ఏమిటి?")
    assert result["question_category"] == "DEADLINES"
    assert result["answer"]["status"] == "no_evidence"
    assert not result["answer"]["used_passages"]


def test_english_and_telugu_answers_are_concise_and_cited():
    bot = CivicAssistant()
    telugu = bot.run("రైతు భరోసా పథకం ఉద్దేశం ఏమిటి?", answer_language="te")
    english = bot.run("What is the purpose of Rythu Bharosa?", answer_language="en")

    for result in (telugu, english):
        answer = result["answer"]
        assert answer["status"] == "grounded"
        assert len(answer["text"]) < 700
        assert answer["sources"]
        assert all(
            source["source_url"].startswith("https://")
            and source["scheme"] == result["scheme"]
            for source in answer["sources"]
        )
        assert not any(
            marker in answer["text"].lower()
            for marker in ("login", "captcha", "visitor", "copyright", "whatsapp")
        )
    assert any("\u0c00" <= char <= "\u0c7f" for char in telugu["answer"]["text"])
    assert not any("\u0c00" <= char <= "\u0c7f" for char in english["answer"]["text"])


def test_question_category_variants_and_missing_topics():
    bot = CivicAssistant()
    cases = (
        ("What are the benefits of Rythu Bharosa?", "BENEFITS"),
        ("రైతు భరోసాకు కావలసిన డాక్యుమెంట్లు ఏమిటి?", "REQUIRED_DOCUMENTS"),
        ("How do I apply for Rythu Bharosa?", "APPLICATION_PROCEDURE"),
        ("రైతు భరోసా చెల్లింపు స్థితి ఎలా చూడాలి?", "STATUS_CHECK"),
        ("Rythu Bharosa helpline phone number?", "HELPLINE"),
    )
    for question, category in cases:
        result = bot.run(question)
        assert result["question_category"] == category
        assert all(
            passage["scheme"] == result["scheme"]
            and passage["category"] == category
            for passage in result["answer"]["used_passages"]
        )

    missing = bot.run("What is the renewal procedure for Rythu Bharosa?")
    assert missing["question_category"] == "RENEWAL"
    assert missing["answer"]["status"] == "no_evidence"
