import os
import sys
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


import subprocess
import soundfile as sf
import scipy.signal as sps
import math


def convert_phone_audio_to_16k_wav(input_path: str, output_path: str, target_sr: int = 16000) -> bool:
    """
    Converts mobile phone audio/video recordings (.mp4, .m4a, .mp3, .aac, .ogg, .3gp, .wav)
    into standard 16kHz mono 16-bit PCM WAV required for Indic ASR models.
    """
    # 1. Use imageio_ffmpeg if available for universal codec decoding
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    temp_wav = output_path + ".temp.wav"

    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", input_path,
                "-vn",                  # Ignore video stream
                "-ac", "1",             # Mono
                "-ar", str(target_sr),  # 16000 Hz
                "-f", "wav",
                temp_wav
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(temp_wav):
                # Post-process: Highpass filter (<90Hz fan rumble) & safe volume normalize
                data, sr = sf.read(temp_wav, dtype='float32')
                sos = sps.butter(4, 90, btype='highpass', fs=target_sr, output='sos')
                data = sps.sosfilt(sos, data).astype(np.float32)

                max_v = np.max(np.abs(data))
                if max_v > 0.05:
                    data = (data / max_v) * 0.90

                sf.write(output_path, data, target_sr, subtype='PCM_16')
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
                return True
        except Exception:
            pass

    # 2. Fallback using soundfile directly
    try:
        data, sr = sf.read(input_path, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        if sr != target_sr:
            gcd = math.gcd(target_sr, sr)
            up = target_sr // gcd
            down = sr // gcd
            data = sps.resample_poly(data, up, down).astype(np.float32)

        max_v = np.max(np.abs(data))
        if max_v > 0:
            data = (data / max_v) * 0.90

        sf.write(output_path, data, target_sr, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"  ✗ Error converting {os.path.basename(input_path)}: {e}")
        return False


def process_dropped_recordings():
    """
    Scans data/audio/raw_recordings/ for phone audio files (.mp4, .m4a, .mp3, etc.)
    and formats them into standard 16kHz WAV files.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "audio", "raw_recordings")

    print("=" * 70)
    print(" 📱 IMPORTING REAL HUMAN PHONE RECORDINGS")
    print("=" * 70)
    print(f"Scanning directory: {raw_dir}\n")

    files = sorted(os.listdir(raw_dir))
    phone_extensions = ('.mp4', '.m4a', '.mp3', '.aac', '.ogg', '.flac', '.3gp', '.webm')
    phone_files = [f for f in files if f.lower().endswith(phone_extensions)]

    if not phone_files:
        print("ℹ️ No mobile phone audio files (.mp4, .m4a, .mp3) found in raw_recordings/.")
        print("=" * 70)
        return

    print(f"Found {len(phone_files)} mobile recordings to convert:\n")
    converted_count = 0
    for pf in phone_files:
        in_p = os.path.join(raw_dir, pf)
        base_name = os.path.splitext(pf)[0]
        out_p = os.path.join(raw_dir, f"{base_name}.wav")
        success = convert_phone_audio_to_16k_wav(in_p, out_p)
        if success:
            converted_count += 1
            size_kb = os.path.getsize(out_p) / 1024
            dur = len(sf.read(out_p)[0]) / 16000
            print(f"  ✓ Converted {pf:<16} -> {base_name}.wav ({size_kb:5.1f} KB, {dur:4.2f}s, 16kHz mono)")

    print(f"\n✅ {converted_count}/{len(phone_files)} phone recordings successfully converted to 16kHz WAV!")
    print("=" * 70)


if __name__ == "__main__":
    process_dropped_recordings()
