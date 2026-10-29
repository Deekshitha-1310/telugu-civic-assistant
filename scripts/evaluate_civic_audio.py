"""End-to-end evaluation on a small set of recorded Telugu civic questions.

CSV columns: audio_path,reference_text,expected_intent,expected_scheme
expected_scheme should be a substring found in the project's scheme names, e.g. ఆరోగ్యశ్రీ.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import CivicAssistant  # noqa: E402
from evaluate_asr import word_error_rate, char_error_rate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "civic_audio_eval.csv"))
    parser.add_argument("--model", default="base")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    if not manifest.exists():
        raise SystemExit(f"Create the manifest first: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("audio_path") or "").strip()]
    if not rows:
        raise SystemExit("No rows with audio_path found in civic evaluation CSV.")

    import whisper
    model = whisper.load_model(args.model)
    bot = CivicAssistant()
    results = []
    for i, row in enumerate(rows, 1):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (ROOT / audio_path).resolve()
        if not audio_path.exists():
            raise SystemExit(f"Missing audio file: {audio_path}")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, sr)
            hyp = (model.transcribe(tmp_path, language="te", fp16=False, verbose=False).get("text") or "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        result = bot.run(hyp, input_mode="audio", transcript=hyp)
        passages = result.get("passages") or []
        scheme_needle = (row.get("expected_scheme") or "").strip()
        retrieval_hit = bool(scheme_needle) and any(scheme_needle in (p.get("scheme") or "") for p in passages)
        expected_intent = (row.get("expected_intent") or "").strip().upper()
        predicted_intent = (result.get("intent") or "").strip().upper()
        results.append({
            "audio_file": audio_path.name,
            "reference_text": row.get("reference_text", ""),
            "asr_transcript": hyp,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": bool(expected_intent) and expected_intent == predicted_intent,
            "expected_scheme": scheme_needle,
            "retrieval_hit_at_5": retrieval_hit,
            "answer_grounded": bool((result.get("answer") or {}).get("grounded")),
            "WER": word_error_rate(row.get("reference_text", ""), hyp),
            "CER": char_error_rate(row.get("reference_text", ""), hyp),
            "answer_text": (result.get("answer") or {}).get("text", ""),
        })
        print(f"[{i}/{len(rows)}] {audio_path.name}: intent={predicted_intent}, retrieval_hit@5={retrieval_hit}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "civic_audio_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    def mean(key):
        vals = [r[key] for r in results if isinstance(r[key], (int, float)) and not isinstance(r[key], bool)]
        return round(float(np.mean(vals)), 4) if vals else None
    summary = {
        "n_recordings": len(results),
        "mean_WER": mean("WER"),
        "mean_CER": mean("CER"),
        "intent_accuracy_on_recorded_examples": round(sum(r["intent_correct"] for r in results) / len(results), 4),
        "retrieval_hit_at_5_rate": round(sum(r["retrieval_hit_at_5"] for r in results) / len(results), 4),
        "grounded_answer_rate": round(sum(r["answer_grounded"] for r in results) / len(results), 4),
        "warning": "Small in-project test set only; not evidence of general regional-dialect robustness.",
    }
    json_path = out_dir / "civic_audio_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
