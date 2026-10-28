import os
import sys
import json
import urllib.request
import tarfile
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass


def resample_and_save(raw_bytes: bytes, out_path: str, target_sr: int = 16000):
    """Decodes audio bytes from memory and saves as 16kHz mono 16-bit PCM WAV."""
    import io
    import soundfile as sf
    import scipy.signal as sps
    import math

    buf = io.BytesIO(raw_bytes)
    audio, sr = sf.read(buf, dtype='float32')
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        gcd = math.gcd(target_sr, sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = sps.resample_poly(audio, up, down).astype(np.float32)

    # Normalize amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    sf.write(out_path, audio.astype(np.float32), target_sr, subtype='PCM_16')


def fetch_real_human_telugu_speech(num_samples: int = 10):
    """
    Downloads real native human Telugu speech recordings from the Google FLEURS
    (Few-shot Learning Evaluation of Universal Representations of Speech) dataset.
    Source: https://huggingface.co/datasets/google/fleurs (Config: te_in)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "audio", "raw_recordings")
    os.makedirs(raw_dir, exist_ok=True)

    print("=" * 75)
    print(" 🎙️ FETCHING REAL HUMAN TELUGU SPEECH RECORDINGS FROM INTERNET DATASET")
    print("=" * 75)
    print("Source Dataset : Google FLEURS (Hugging Face: google/fleurs, te_in)")
    print("Type           : Authentic Human Spoken Telugu Recordings (Native Speakers)")
    print(f"Target Directory: {raw_dir}\n")

    # Step 1: Download the dev.tsv metadata mapping
    tsv_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/te_in/dev.tsv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    print("  • Fetching human transcription index from Google FLEURS...")
    req = urllib.request.Request(tsv_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        tsv_content = resp.read().decode("utf-8")

    meta_map = {}
    for line in tsv_content.splitlines():
        parts = line.split("\t")
        if len(parts) >= 7:
            # parts[1] is filename, parts[2] is raw text, parts[6] is gender
            wav_name = parts[1].strip()
            raw_text = parts[2].strip()
            gender = parts[6].strip()
            meta_map[wav_name] = {"text": raw_text, "gender": gender}

    print(f"    ✓ Loaded metadata for {len(meta_map)} native human Telugu utterances.")

    # Step 2: Stream audio archive from Hugging Face and extract first N audio files
    archive_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/te_in/audio/dev.tar.gz"
    print(f"\n  • Streaming real human voice archives from Hugging Face (extracting {num_samples} clips)...")

    req_tar = urllib.request.Request(archive_url, headers=headers)
    extracted_records = []

    with urllib.request.urlopen(req_tar, timeout=30.0) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            count = 0
            for member in tar:
                if member.name.endswith(".wav"):
                    base_wav = os.path.basename(member.name)
                    if base_wav in meta_map:
                        f = tar.extractfile(member)
                        if f is not None:
                            raw_bytes = f.read()
                            sample_id = f"sample_{count+1:02d}.wav"
                            out_path = os.path.join(raw_dir, sample_id)

                            resample_and_save(raw_bytes, out_path, target_sr=16000)
                            file_size_kb = os.path.getsize(out_path) / 1024

                            info = meta_map[base_wav]
                            extracted_records.append({
                                "id": f"sample_{count+1:02d}",
                                "file": sample_id,
                                "original_id": base_wav,
                                "dialect_query": info["text"],
                                "standard_query": info["text"],
                                "dialect_zone": f"Real Native Human ({info['gender'].capitalize()})",
                                "intent": "NATIVE_HUMAN_CORPUS",
                                "scheme": "GOOGLE_FLEURS_TELUGU",
                                "dataset_source": "Google FLEURS (te_in, Hugging Face)",
                                "speaker_gender": info["gender"]
                            })

                            print(f"    ✓ Saved {sample_id} ({file_size_kb:5.1f} KB) [{info['gender']}] -> '{info['text'][:55]}...'")
                            count += 1
                            if count >= num_samples:
                                break

    # Step 3: Update dataset manifest with authentic native human recordings
    manifest_path = os.path.join(raw_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Updated Dataset Manifest -> {manifest_path} ({len(extracted_records)} real human recordings)")
    print("\n" + "=" * 75)
    print("✅ Successfully imported REAL HUMAN TELUGU SPEECH from Google FLEURS dataset!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    fetch_real_human_telugu_speech(num_samples=10)
