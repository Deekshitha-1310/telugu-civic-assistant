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

    # 16-bit PCM (standard)
    if sampwidth == 2:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    # 8-bit PCM
    elif sampwidth == 1:
        audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    # 24-bit PCM (common on phones)
    elif sampwidth == 3:
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        n_samples = len(raw_bytes) // 3
        # Reshape into 3 bytes per sample and pack to 32-bit int
        raw_reshaped = raw_bytes[:n_samples * 3].reshape(-1, 3)
        padded = np.pad(raw_reshaped, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        int32_samples = padded.view(dtype='<i4').flatten()
        audio = (int32_samples / 2147483648.0).astype(np.float32)
    # 32-bit int / float
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


class SpectralAudioCleaner:
    def __init__(self, mode: str = "non_stationary"):
        """
        Spectral Subtraction Audio Cleaner.
        - 'non_stationary': Dynamic time-frequency spectral gating (handles fluctuating traffic/babble).
        - 'stationary': Constant spectral threshold (handles stationary fan/line hum).
        """
        self.mode = mode

    def clean_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        try:
            import noisereduce as nr
            is_stationary = (self.mode == "stationary")
            cleaned = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=is_stationary,
                prop_decrease=0.85
            )
            return cleaned.astype(np.float32)
        except Exception:
            # High-performance overlap-add STFT spectral subtraction fallback
            frame_len = 512
            hop_len = 256
            if len(audio) < frame_len:
                return audio

            num_frames = (len(audio) - frame_len) // hop_len + 1
            if num_frames <= 0:
                return audio

            window = np.hanning(frame_len).astype(np.float32)
            stft = []
            for i in range(num_frames):
                frame = audio[i * hop_len: i * hop_len + frame_len] * window
                stft.append(np.fft.rfft(frame))
            stft = np.array(stft)

            # Ambient noise magnitude estimate from leading frames
            noise_frames = max(2, min(8, num_frames // 10))
            noise_mag_profile = np.mean(np.abs(stft[:noise_frames, :]), axis=0)

            # Spectral subtraction with oversubtraction factor alpha
            alpha = 1.35 if self.mode == "non_stationary" else 1.0
            beta = 0.08
            
            clean_stft = np.zeros_like(stft)
            for i in range(num_frames):
                frame_mag = np.abs(stft[i, :])
                frame_phase = np.angle(stft[i, :])
                sub_mag = frame_mag - alpha * noise_mag_profile
                floor_mag = beta * frame_mag
                clean_mag = np.maximum(sub_mag, floor_mag)
                clean_stft[i, :] = clean_mag * np.exp(1j * frame_phase)

            # Overlap-add synthesis with exact boundary protection
            cleaned = np.zeros(len(audio), dtype=np.float32)
            window_sum = np.zeros(len(audio), dtype=np.float32)

            for i in range(num_frames):
                frame_time = np.fft.irfft(clean_stft[i, :]) * window
                start_idx = i * hop_len
                end_idx = min(start_idx + frame_len, len(audio))
                slice_len = end_idx - start_idx
                cleaned[start_idx: end_idx] += frame_time[:slice_len].astype(np.float32)
                window_sum[start_idx: end_idx] += (window[:slice_len] ** 2)

            nonzero = window_sum > 1e-4
            cleaned[nonzero] /= window_sum[nonzero]
            return cleaned.astype(np.float32)


def run_audio_cleaner_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    noisy_dir = os.path.join(base_dir, "data", "audio", "augmented_noisy", "snr_10db")
    denoised_dir = os.path.join(base_dir, "data", "audio", "denoised")
    os.makedirs(denoised_dir, exist_ok=True)

    print("=" * 65)
    print(" 🧹 MEMBER 1: SPECTRAL SUBTRACTION DENOISING PIPELINE")
    print("=" * 65)
    print("Cleaner Algorithm   : Non-Stationary Spectral Gating")
    print("Target Noise Profile: 10 dB SNR Traffic Audio -> Cleaned Output")
    print(f"Reading Input From  : data/audio/augmented_noisy/snr_10db/")
    print(f"Saving Output To    : data/audio/denoised/\n")

    # If noisy files have not been generated yet, auto-generate them
    if not os.path.exists(noisy_dir) or len([f for f in os.listdir(noisy_dir) if f.endswith(".wav")]) == 0:
        print("⚠️ Noisy audio directory empty. Auto-running noise augmentation...")
        try:
            from noise_augmenter import run_noise_augmentation
            run_noise_augmentation()
        except Exception as e:
            print(f"Error auto-running noise augmenter: {e}")
            return

    cleaner = SpectralAudioCleaner(mode="non_stationary")
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".wav")])

    for fname in noisy_files:
        in_path = os.path.join(noisy_dir, fname)
        noisy_audio, sr = read_wav(in_path)

        # Apply spectral denoising
        cleaned_audio = cleaner.clean_audio(noisy_audio, sr=sr)

        # Save to denoised directory as sample_01_clean.wav etc.
        out_name = fname.replace(".wav", "_clean.wav")
        out_path = os.path.join(denoised_dir, out_name)
        write_wav(out_path, cleaned_audio, sr)

        noisy_energy = float(np.mean(noisy_audio ** 2))
        clean_energy = float(np.mean(cleaned_audio ** 2))
        reduction_db = 10 * np.log10(noisy_energy / (clean_energy + 1e-9))

        print(f"  • {fname:<14} -> {out_name:<20} | Noise Suppressed: -{reduction_db:4.1f} dB")

    print("\n✅ All audio files successfully cleaned and saved in data/audio/denoised/!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_audio_cleaner_pipeline()
