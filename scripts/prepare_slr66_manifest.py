"""Build a deterministic ASR evaluation manifest from extracted OpenSLR SLR66 files.

Expected inputs: extracted WAV files and line_index_female.tsv and/or
line_index_male.tsv anywhere under the supplied dataset directory.
"""
import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Folder containing extracted SLR66 WAVs and line_index_*.tsv files")
    parser.add_argument("--limit", type=int, default=20, help="Number of clips to evaluate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for repeatable selection")
    parser.add_argument("--output", default=str(ROOT / "data" / "asr_eval_manifest.csv"))
    args = parser.parse_args()

    base = Path(args.dataset_dir).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"Dataset folder not found: {base}")

    tsvs = sorted(base.rglob("line_index*.tsv"))
    if not tsvs:
        raise SystemExit("No line_index*.tsv found. Download the SLR66 index TSV and place it under the dataset folder.")

    wav_by_stem = {}
    for wav in base.rglob("*.wav"):
        wav_by_stem.setdefault(wav.stem, wav)
    if not wav_by_stem:
        raise SystemExit("No .wav files found. Extract one SLR66 audio ZIP before running this script.")

    rows, seen = [], set()
    for tsv in tsvs:
        with tsv.open("r", encoding="utf-8-sig", newline="") as f:
            for parts in csv.reader(f, delimiter="\t"):
                if len(parts) < 2:
                    continue
                file_id, transcript = parts[0].strip(), "\t".join(parts[1:]).strip()
                if not file_id or not transcript:
                    continue
                if file_id.lower() in {"fileid", "file_id", "id"}:
                    continue
                stem = Path(file_id).stem
                wav = wav_by_stem.get(stem)
                if wav is None or stem in seen:
                    continue
                seen.add(stem)
                try:
                    rel_path = wav.relative_to(ROOT).as_posix()
                except ValueError:
                    rel_path = str(wav)
                rows.append({"audio_path": rel_path, "reference_text": transcript, "file_id": stem})

    if len(rows) < args.limit:
        raise SystemExit(
            f"Only {len(rows)} audio/index matches found, but limit={args.limit}. "
            "Check that the TSV file(s) correspond to the extracted archive(s)."
        )
    random.Random(args.seed).shuffle(rows)
    selected = rows[:args.limit]
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference_text", "file_id"])
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {len(selected)} fixed evaluation clips to: {out}")
    print("Keep this manifest unchanged for repeatable comparisons.")


if __name__ == "__main__":
    main()
