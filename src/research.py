"""Bounded official-source discovery, ingestion, and coverage evaluation.

Usage:
    python src/research.py --iterations 2 --max-depth 1 --pages-per-domain 8

Each iteration uses the actual configured domains and crawler budget, preserves
the legacy corpus, updates the persistent corpus, and records coverage gaps.
The crawler does not infer facts or use an LLM as a correctness oracle.
"""
import argparse
import json
from pathlib import Path

from evaluate_coverage import audit
from ingest import ingest
from paths import KNOWLEDGE_SOURCES_PATH, OUTPUTS_DIR


def run(iterations=1, max_depth=1, pages_per_domain=10, timeout=20,
        max_bytes=8_000_000, delay=0.25, refresh=False):
    runs = []
    focus_terms = []
    baseline = audit()
    for iteration in range(1, max(1, iterations) + 1):
        ingestion = ingest(
            discover=True, max_depth=max_depth, pages_per_domain=pages_per_domain,
            timeout=timeout, max_bytes=max_bytes, delay=delay, refresh=refresh,
            focus_terms=focus_terms,
        )
        coverage = audit()
        runs.append({"iteration": iteration, "ingestion": ingestion, "coverage": coverage})
        focus_terms = sorted({
            category.lower().replace("_", " ")
            for item in coverage["schemes"].values()
            for category in item["missing_categories"]
        })
        if not ingestion["discovered"] and not ingestion["fetched"]:
            break
    report = {
        "configured_sources": json.loads(Path(KNOWLEDGE_SOURCES_PATH).read_text(encoding="utf-8")),
        "baseline_coverage": baseline,
        "iterations": runs,
        "last_coverage": runs[-1]["coverage"] if runs else None,
    }
    Path(OUTPUTS_DIR).mkdir(exist_ok=True)
    (Path(OUTPUTS_DIR) / "official_research_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["last_coverage"] is not None:
        (Path(OUTPUTS_DIR) / "knowledge_coverage.json").write_text(
            json.dumps(report["last_coverage"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    last_ingestion = runs[-1]["ingestion"] if runs else {}
    coverage = report["last_coverage"] or {}
    baseline_schemes = report["baseline_coverage"].get("schemes", {})
    lines = [
        "# Official research report", "",
        f"- Iterations: {len(runs)}",
        f"- Discovered useful pages: {last_ingestion.get('discovered', 0)}",
        f"- Useful evidence pages: {last_ingestion.get('useful_evidence_pages', 0)}",
        f"- Useful discovery leads (not indexed as evidence): {last_ingestion.get('useful_discovery_leads', 0)}",
        f"- Configured source entries: {last_ingestion.get('configured_sources', 0)}",
        f"- Source domains attempted: {len(last_ingestion.get('source_attempts', {}))}",
        f"- Fetched documents: {last_ingestion.get('fetched', 0)}",
        f"- Extracted chunks before deduplication: {last_ingestion.get('extracted_chunks', 0)}",
        f"- Useful documents indexed: {last_ingestion.get('useful_documents', 0)}",
        f"- Duplicate documents skipped: {last_ingestion.get('duplicate_documents', 0)}",
        f"- Changed documents: {last_ingestion.get('changed_documents', 0)}",
        f"- Unchanged documents: {last_ingestion.get('unchanged_documents', 0)}",
        f"- Failed fetches: {len(last_ingestion.get('failures', []))}",
        "",
        "## Coverage by scheme", "",
    ]
    for scheme, item in coverage.get("schemes", {}).items():
        before = baseline_schemes.get(scheme, {})
        lines.extend([
            f"### {scheme}",
            f"- Records: {item['records']}",
            f"- Official source URLs: {item['official_source_count']}",
            f"- Official domains: {', '.join(item.get('official_domains', [])) or 'none'}",
            f"- Categories: {', '.join(item['evidence_categories']) or 'none'}",
            f"- Questions with evidence: {item['questions_with_evidence']}/{item['question_count']}",
            f"- Questions with verified evidence: {item['questions_with_verified_evidence']}/{item['question_count']}",
            f"- Verified evidence change: {item['questions_with_verified_evidence'] - before.get('questions_with_verified_evidence', 0):+d}",
            f"- Missing categories: {', '.join(item['missing_categories']) or 'none'}",
            "",
        ])
    (Path(OUTPUTS_DIR) / "official_research_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--pages-per-domain", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=8_000_000)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    report = run(
        args.iterations, args.max_depth, args.pages_per_domain, args.timeout,
        args.max_bytes, args.delay, args.refresh,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
