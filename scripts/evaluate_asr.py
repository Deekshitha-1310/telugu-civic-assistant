"""Evaluate Telugu Whisper ASR on a fixed SLR66 manifest.

Runs the same clips in three conditions: clean, synthetic 10 dB SNR noise,
and noise-reduced. Results are written to outputs/asr_results.csv and
outputs/asr_summary.json. This is a controlled baseline, not a dialect benchmark.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def clean_text(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s\u0c00-\u0c7f]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_error_rate(reference, hypothesis):
    r, h = clean_text(reference).split(), clean_text(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def char_error_rate(reference, hypothesis):
    r = clean_text(reference).replace(" ", "")
    h = clean_text(hypothesis).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def add_white_noise(audio, snr_db, rng):
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 1e-12:
        return audio.copy()
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    noisy = audio + noise
    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    if peak > 0.999:
        noisy = noisy * (0.999 / peak)
    return noisy


def transcribe(model, audio, sr, noise_reduce=False, rng=None):
    work = np.asarray(audio, dtype=np.float32)
    if noise_reduce:
        import noisereduce as nr
        work = nr.reduce_noise(y=work, sr=sr, stationary=False).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
    try:
        sf.write(temp_path, work, sr)
        result = model.transcribe(temp_path, language="te", fp16=False, verbose=False)
        return (result.get("text") or "").strip()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "asr_eval_manifest.csv"))
    parser.add_argument("--model", default="base", help="Whisper model: tiny, base, small, medium, large or local model path")
    parser.add_argument("--snr-db", type=float, default=10.0, help="Fixed synthetic noise SNR in dB")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}\nRun scripts/prepare_slr66_manifest.py first.")
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[:args.limit]
    if not rows:
        raise SystemExit("Manifest contains no evaluation rows.")

    try:
        import whisper
    except ImportError:
        raise SystemExit("Whisper missing. Run: python -m pip install openai-whisper")
    try:
        import noisereduce  # noqa: F401
    except ImportError:
        raise SystemExit("noisereduce missing. Run: python -m pip install noisereduce")

    print(f"Loading Whisper '{args.model}' (first use may download model weights)...")
    model = whisper.load_model(args.model)
    rng = np.random.default_rng(2026)
    output_rows = []

    for i, row in enumerate(rows, 1):
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (ROOT / audio_path).resolve()
        if not audio_path.exists():
            raise SystemExit(f"Audio file missing for row {i}: {audio_path}")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        ref = row["reference_text"]
        clean_hyp = transcribe(model, audio, sr)
        noisy_audio = add_white_noise(audio, args.snr_db, rng)
        noisy_hyp = transcribe(model, noisy_audio, sr)
        denoised_hyp = transcribe(model, noisy_audio, sr, noise_reduce=True)
        record = {
            "file_id": row.get("file_id", audio_path.stem),
            "reference_text": ref,
            "clean_transcript": clean_hyp,
            "noisy_transcript": noisy_hyp,
            "denoised_transcript": denoised_hyp,
            "clean_WER": word_error_rate(ref, clean_hyp),
            "noisy_WER": word_error_rate(ref, noisy_hyp),
            "denoised_WER": word_error_rate(ref, denoised_hyp),
            "clean_CER": char_error_rate(ref, clean_hyp),
            "noisy_CER": char_error_rate(ref, noisy_hyp),
            "denoised_CER": char_error_rate(ref, denoised_hyp),
        }
        output_rows.append(record)
        print(f"[{i}/{len(rows)}] {record['file_id']} | WER clean/noisy/denoised: "
              f"{record['clean_WER']:.3f}/{record['noisy_WER']:.3f}/{record['denoised_WER']:.3f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "asr_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    metric_names = ["clean_WER", "noisy_WER", "denoised_WER", "clean_CER", "noisy_CER", "denoised_CER"]
    summary = {
        "dataset": "OpenSLR SLR66 Telugu speech subset",
        "n_clips": len(output_rows),
        "whisper_model": args.model,
        "synthetic_noise_snr_db": args.snr_db,
        "noise_method": "additive white Gaussian noise; fixed random seed 2026",
        "denoising_method": "noisereduce.reduce_noise (non-stationary)",
        "metrics_are_errors_lower_is_better": True,
        "mean_metrics": {m: round(float(np.mean([r[m] for r in output_rows])), 4) for m in metric_names},
        "limitation": "General Telugu speech benchmark only; not a regional dialect or civic-domain benchmark.",
    }
    json_path = out_dir / "asr_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMean metrics (lower is better):")
    for key, value in summary["mean_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved detailed results: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
