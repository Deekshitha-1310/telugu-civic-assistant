"""Question-level coverage audit for the maintainable scheme corpus.

Usage: python src/evaluate_coverage.py
This reports evidence categories and unanswered checklist categories; it does
not convert missing information into a guessed answer.
"""
import json
import sys
from pathlib import Path

from kb import load_chunks
from paths import DATA_DIR, OUTPUTS_DIR
from retriever import HybridRetriever
from answer_generator import generate_answer


def _default_questions(scheme):
    return [
        {"id": f"{scheme}-purpose", "category": "PURPOSE", "text": f"What is the purpose of {scheme}?", "variants": [f"{scheme} లక్ష్యం ఏమిటి?"]},
        {"id": f"{scheme}-eligibility-te", "category": "ELIGIBILITY", "text": "ఈ పథకానికి ఎవరు అర్హులు?", "variants": [f"Who is eligible for {scheme}?"]},
        {"id": f"{scheme}-benefits", "category": "BENEFITS", "text": "What benefits and amounts are provided?", "variants": [f"{scheme}లో ఎంత డబ్బు వస్తుంది?"]},
        {"id": f"{scheme}-documents", "category": "REQUIRED_DOCUMENTS", "text": "What documents are needed?", "variants": ["ఏ పత్రాలు కావాలి?"]},
        {"id": f"{scheme}-apply-te", "category": "APPLICATION_PROCEDURE", "text": "ఎలా దరఖాస్తు చేసుకోవాలి?", "variants": [f"How do I apply for {scheme}?"]},
        {"id": f"{scheme}-deadline", "category": "DEADLINES", "text": "What is the deadline or application period?", "variants": ["దరఖాస్తు చివరి తేదీ ఎప్పుడు?"]},
        {"id": f"{scheme}-status", "category": "STATUS_CHECK", "text": "How can I check my application or payment status?", "variants": ["నా చెల్లింపు స్థితి ఎలా చూడాలి?"]},
        {"id": f"{scheme}-renewal", "category": "RENEWAL", "text": "How do I renew the benefit?", "variants": ["Do I need to renew it?"]},
        {"id": f"{scheme}-grievance", "category": "GRIEVANCE", "text": "Where can I complain or appeal?", "variants": ["ఫిర్యాదు ఎక్కడ చేయాలి?"]},
        {"id": f"{scheme}-helpline", "category": "HELPLINE", "text": "What is the official helpline?", "variants": ["హెల్ప్‌లైన్ నంబర్ ఏమిటి?"]},
        {"id": f"{scheme}-faq", "category": "FAQ", "text": "What frequently asked questions are answered officially?", "variants": ["Are there official FAQs?"]},
        {"id": f"{scheme}-exception", "category": "EXCEPTIONS", "text": "What important exclusions or exceptions apply?", "variants": ["ఈ పథకానికి వర్తించని వారు ఎవరు?"]},
    ]


def audit(checklist_path=None):
    checklist_path = checklist_path or Path(DATA_DIR) / "scheme_question_checklist.json"
    checklist = json.loads(Path(checklist_path).read_text(encoding="utf-8"))
    chunks = load_chunks()
    retriever = HybridRetriever(chunks, enable_neural=False)
    report = {"schemes": {}, "corpus_records": len(chunks)}
    schemes = sorted({c["scheme"] for c in chunks})
    for scheme in schemes:
        spec = checklist["schemes"].get(scheme, {})
        docs = [c for c in chunks if c["scheme"] == scheme]
        evidence_categories = sorted({c["category"] for c in docs})
        source_urls = sorted({c["source_url"] for c in docs if c.get("source_url")})
        questions = []
        questions_to_run = spec.get("questions") or _default_questions(scheme)
        for question in questions_to_run:
            evidence_category = question.get("evidence_category", question["category"])
            variants = [question["text"]] + question.get("variants", [])
            variant_results = []
            for variant in variants:
                results = retriever.search(
                    f"{scheme} {variant}", top_k=5, scheme=scheme,
                    category=evidence_category,
                )
                variant_results.append({
                    "text": variant,
                    "results": [
                        p for p in results
                        if p["scheme"] == scheme and p["category"] == evidence_category
                    ],
                })
            relevant = [p for item in variant_results for p in item["results"]]
            deduped = {p["doc_id"]: p for p in relevant}
            relevant = list(deduped.values())
            corpus_evidence = [
                c for c in docs if c["category"] == evidence_category
            ]
            verified = [
                p for p in corpus_evidence
                if p.get("verification_status") == "verified_current"
            ]
            valid = [
                p for p in corpus_evidence
                if str(p.get("source_url", "")).startswith(("http://", "https://"))
            ]
            answer = generate_answer(
                relevant,
                question=question["text"],
                scheme=scheme,
                category=evidence_category,
            )
            questions.append({
                **question,
                "evidence_found": bool(corpus_evidence),
                "retrieval_found": bool(relevant),
                "grounded_answer": answer["status"] == "grounded",
                "unsupported_answer": bool(
                    answer["status"] == "grounded" and any(
                        p.get("scheme") != scheme or p.get("category") != evidence_category
                        for p in answer.get("used_passages", [])
                    )
                ),
                "verified_evidence_found": bool(verified),
                "valid_source_evidence_found": bool(valid),
                "relevance_status": (
                    "verified_current" if verified else
                    "official_accessed_currentness_unverified" if corpus_evidence and any(
                        p.get("source_role") != "legacy_record" for p in corpus_evidence
                    ) else
                    "legacy_unverified" if corpus_evidence else "missing_knowledge"
                ),
                "failure_reason": (
                    None if relevant else
                    "A_missing_knowledge" if not corpus_evidence
                    else "B_retrieval_failure"
                ),
                "evidence_doc_ids": [p["doc_id"] for p in relevant],
                "source_urls": sorted({p["source_url"] for p in relevant if p.get("source_url")}),
                "source_classifications": sorted({
                    p.get("source_classification", "unknown") for p in relevant
                }),
                "variant_results": [
                    {
                        "text": item["text"],
                        "evidence_doc_ids": [p["doc_id"] for p in item["results"]],
                    }
                    for item in variant_results
                ],
            })
        report["schemes"][scheme] = {
            "records": len(docs),
            "evidence_categories": evidence_categories,
            "source_urls": source_urls,
            "official_source_count": len(source_urls),
            "official_domains": sorted({
                p.get("authority") for p in docs if p.get("authority")
            }),
            "verified_source_count": len({
                c["source_url"] for c in docs
                if c.get("verification_status") == "verified_current"
            }),
            "accessed_unverified_source_count": len({
                c["source_url"] for c in docs
                if c.get("verification_status") == "official_accessed_currentness_unverified"
            }),
            "legacy_source_count": len({
                c["source_url"] for c in docs if c.get("source_role") == "legacy_record"
            }),
            "missing_categories": [
                category for category in checklist["categories"]
                if category not in evidence_categories
            ],
            "category_status": {
                category: (
                    "verified_current" if any(
                        c["category"] == category and c.get("verification_status") == "verified_current"
                        for c in docs
                    ) else "official_accessed_unverified" if any(
                        c["category"] == category and c.get("source_role") != "legacy_record"
                        for c in docs
                    ) else "legacy_only" if category in evidence_categories else "not_found"
                )
                for category in checklist["categories"]
            },
            "question_count": len(questions),
            "questions_with_evidence": sum(q["evidence_found"] for q in questions),
            "questions_with_verified_evidence": sum(
                q["verified_evidence_found"] for q in questions
            ),
            "questions_with_valid_source_evidence": sum(
                q["valid_source_evidence_found"] for q in questions
            ),
            "questions": questions,
        }
    all_questions = [
        question
        for item in report["schemes"].values()
        for question in item["questions"]
    ]
    answerable = [q for q in all_questions if q["evidence_found"]]
    verified = [q for q in all_questions if q["verified_evidence_found"]]
    retrieved = [q for q in all_questions if q["retrieval_found"]]
    report["metrics"] = {
        "question_count": len(all_questions),
        "evidence_coverage": len(answerable) / len(all_questions) if all_questions else 0.0,
        "verified_evidence_coverage": len(verified) / len(all_questions) if all_questions else 0.0,
        "retrieval_recall": len(retrieved) / len(answerable) if answerable else 0.0,
        "grounded_answer_rate": (
            sum(q["grounded_answer"] for q in all_questions if q["evidence_found"])
            / len(answerable) if answerable else 0.0
        ),
        "unnecessary_abstention_rate": (
            sum(q["evidence_found"] and not q["retrieval_found"] for q in all_questions)
            / len(answerable) if answerable else 0.0
        ),
        "correct_abstention_rate": (
            sum(not q["evidence_found"] for q in all_questions) / len(all_questions)
            if all_questions else 0.0
        ),
        "unsupported_answer_rate": (
            sum(q["unsupported_answer"] for q in all_questions) / len(all_questions)
            if all_questions else 0.0
        ),
        "scheme_category_isolation": True,
        "metric_note": (
            "Evidence and retrieval metrics are corpus/retriever checks. "
            "Verified coverage requires verification_status=verified_current; "
            "no source is marked that way by the current ingestion run."
        ),
    }
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = audit()
    Path(OUTPUTS_DIR).mkdir(exist_ok=True)
    out = Path(OUTPUTS_DIR) / "knowledge_coverage.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
