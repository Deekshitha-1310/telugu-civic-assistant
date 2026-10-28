import os
import sys
import wave
import numpy as np

# Ensure UTF-8 output on Windows without triggering linter attribute warnings
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass


def read_wav(filename: str):
    """
    Reads a WAV audio file supporting 8-bit, 16-bit, 24-bit, and 32-bit audio.
    Prioritizes soundfile for universal compatibility, falling back to wave.
    Returns (audio_numpy_float32, sample_rate).
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(filename, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except Exception:
        pass

    with wave.open(filename, 'r') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)

    if sampwidth == 2:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        n_samples = len(raw_bytes) // 3
        raw_reshaped = raw_bytes[:n_samples * 3].reshape(-1, 3)
        padded = np.pad(raw_reshaped, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        int32_samples = padded.view(dtype='<i4').flatten()
        audio = (int32_samples / 2147483648.0).astype(np.float32)
    elif sampwidth == 4:
        try:
            audio = np.frombuffer(data, dtype=np.float32)
        except Exception:
            audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1 and len(audio) >= n_channels:
        usable_len = (len(audio) // n_channels) * n_channels
        audio = audio[:usable_len].reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.float32), sr


def write_wav(filename: str, audio: np.ndarray, sr: int = 16000):
    """Writes a 16-bit PCM mono .wav file."""
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    try:
        import soundfile as sf
        sf.write(filename, audio, sr, subtype='PCM_16')
        return
    except Exception:
        pass

    max_val = np.max(np.abs(audio))
    norm_audio = audio / max_val if max_val > 0 else audio
    int16_samples = np.int16(np.clip(norm_audio, -1.0, 1.0) * 32767)

    with wave.open(filename, 'w') as wf:
        wf.setparams((1, 2, sr, len(int16_samples), 'NONE', 'noncompressed'))
        wf.writeframes(int16_samples.tobytes())


def trim_silence(signal: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Trims leading and trailing silence frames.
    Silence lowers the average root-mean-square (RMS) speech power, which causes
    the noise to be scaled at completely the wrong loudness. Trimming ensures SNR
    is computed solely on active speech frames.
    """
    active = np.where(np.abs(signal) > threshold)[0]
    if len(active) == 0:
        return signal
    return signal[active[0]:active[-1] + 1]


def mix_noise_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Mixes background noise into active speech at an exact Signal-to-Noise Ratio (SNR).
    Formula:
        SNR_dB = 10 * log10(P_speech / P_noise)
        P_noise_target = P_speech / (10 ^ (SNR_dB / 10))
    """
    speech_trimmed = trim_silence(speech)
    p_speech = float(np.mean(speech_trimmed ** 2))
    if p_speech == 0.0:
        return speech

    # Match noise length to speech
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise_matched = np.tile(noise, repeats)[:len(speech)]
    else:
        noise_matched = noise[:len(speech)]

    p_noise = float(np.mean(noise_matched ** 2))
    if p_noise == 0.0:
        return speech

    # Calculate scale factor for target SNR
    target_p_noise = p_speech / (10 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(target_p_noise / p_noise)
    scaled_noise = noise_matched * scale_factor

    noisy_speech = speech + scaled_noise
    max_val = np.max(np.abs(noisy_speech))
    if max_val > 1.0:
        noisy_speech = noisy_speech / max_val

    return noisy_speech.astype(np.float32)


def run_noise_augmentation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "audio", "raw_recordings")
    noise_dir = os.path.join(base_dir, "data", "audio", "noise_sources")
    traffic_file = os.path.join(noise_dir, "traffic_noise.wav")
    augmented_dir = os.path.join(base_dir, "data", "audio", "augmented_noisy")

    print("=" * 65)
    print(" 🔊 MEMBER 1: CONTROLLED NOISE AUGMENTATION PIPELINE")
    print("=" * 65)

    # Check if raw files exist; if not, set them up
    if not os.path.exists(traffic_file) or not os.path.exists(raw_dir):
        print("⚠️ Audio sources not found. Running setup_member1_audio...")
        try:
            from setup_member1_audio import setup_member1_audio_files
            setup_member1_audio_files()
        except Exception as e:
            print(f"Error setting up audio: {e}")
            return

    market_file = os.path.join(noise_dir, "market_babble.wav")
    traffic_audio, sr_noise = read_wav(traffic_file)
    market_audio = read_wav(market_file)[0] if os.path.exists(market_file) else traffic_audio

    # Composite environmental noise: street traffic + crowded market babble
    min_len = min(len(traffic_audio), len(market_audio))
    composite_noise = 0.55 * traffic_audio[:min_len] + 0.45 * market_audio[:min_len]
    print(f"Loaded noise profiles: Traffic ({len(traffic_audio)/sr_noise:.1f}s) & Market Babble")

    snr_levels = [
        (15.0,  "snr_15db",       "Mild Noise (Quiet Office)"),
        (10.0,  "snr_10db",       "Moderate Noise (Street Traffic)"),
        (5.0,   "snr_5db",        "Heavy Noise (Market / Panchayat)"),
        (0.0,   "snr_0db",        "Severe Noise (Noise Power = Speech Power)"),
        (-3.0,  "snr_minus3db",   "Extreme Noise (Noise is 2x Louder than Telugu)"),
        (-5.0,  "snr_minus5db",   "Overwhelming Noise (Noise is 3.16x Louder than Telugu)")
    ]

    audio_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".wav")])
    if not audio_files:
        print(f"No .wav files found in '{raw_dir}'.")
        return

    print(f"\nProcessing {len(audio_files)} clean audio clips from raw_recordings/:\n")

    for snr, snr_folder, label in snr_levels:
        target_folder = os.path.join(augmented_dir, snr_folder)
        os.makedirs(target_folder, exist_ok=True)
        print(f"▶ Generating {label} [{snr:4.1f} dB SNR] -> data/audio/augmented_noisy/{snr_folder}/")

        for fname in audio_files:
            clean_path = os.path.join(raw_dir, fname)
            speech, sr_speech = read_wav(clean_path)

            trimmed = trim_silence(speech)
            p_speech = float(np.mean(trimmed ** 2))

            noisy_audio = mix_noise_at_snr(speech, composite_noise, snr)

            out_path = os.path.join(target_folder, fname)
            write_wav(out_path, noisy_audio, sr_speech)

            noise_part = noisy_audio - speech
            p_noise = float(np.mean((trim_silence(noise_part)) ** 2))
            actual_snr = 10 * np.log10(p_speech / (p_noise + 1e-9))
            print(f"    • {fname:<14} -> Target: {snr:4.1f} dB | Measured: {actual_snr:4.1f} dB")

    print("\n✅ All noisy audio sets (15 dB down to -5 dB extreme noise) generated successfully!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_noise_augmentation()
