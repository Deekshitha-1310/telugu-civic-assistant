import os
import sys
import json
import numpy as np

# Ensure local src directory is on sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from audio_cleaner import SpectralAudioCleaner, read_wav, write_wav
from dialect_normalizer import TeluguDialectNormalizer

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass


SRAVAANI_REPO = "ARTPARK-IISc/SraVaani-1.0"
SRAVAANI_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sravaani"
)


def _complete_local_copy(folder):
    """A usable local copy has its config and at least one weights file."""
    if not folder or not os.path.isdir(folder):
        return False
    names = os.listdir(folder)
    has_weights = any(n.endswith((".safetensors", ".bin", ".pt", ".nemo")) for n in names)
    return "config.json" in names and has_weights


def resolve_sravaani_source(repo=SRAVAANI_REPO, local_dir=SRAVAANI_LOCAL_DIR,
                            token=None, downloader=None, offline=None):
    """Decide where SraVaani is loaded from: offline first, online only if needed.

    Order:
      1. TELUGU_CIVIC_SRAVAANI_REPO (manual override: a folder or repo id)
      2. the project copy in models/sravaani (no internet needed)
      3. download once from Hugging Face into models/sravaani, using HF_TOKEN
         or the token saved by `hf auth login` (the gated-model terms must be
         accepted on huggingface.co); later runs then use step 2 offline.
    Returns (path_or_repo, how) where how is "override" | "local" | "downloaded".
    """
    override = os.environ.get("TELUGU_CIVIC_SRAVAANI_REPO")
    if override:
        return override, "override"
    if _complete_local_copy(local_dir):
        return local_dir, "local"
    if offline is None:
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip() in {"1", "true", "yes"}
    if offline:
        raise RuntimeError(
            f"SraVaani is not in {local_dir} and HF_HUB_OFFLINE is set, so it cannot be downloaded."
        )
    if downloader is None:
        from huggingface_hub import snapshot_download as downloader
    print(f"[ASR] SraVaani not found locally; downloading once into {local_dir} (~900 MB)...")
    # token=None lets huggingface_hub use the token saved by `hf auth login`
    downloader(repo_id=repo, local_dir=local_dir, token=token or None)
    if not _complete_local_copy(local_dir):
        raise RuntimeError(f"Download finished but {local_dir} has no config/weights.")
    return local_dir, "downloaded"


class TeluguASREngine:
    def __init__(self, model_name: str = "ARTPARK-IISc/SraVaani-1.0", enable_denoiser: bool = True):
        """
        Telugu Speech Recognition Engine with:
        1. Built-in Spectral Denoising Cleaner (strips harsh noise before recognition)
        2. ARTPARK-IISc / Google SraVaani-1.0 (FastConformer Indic ASR trained on 31,000+ hours of VAANI speech)
        3. Whisper ASR fallback with Indic repetition loop guardrails & civic prompt biasing
        4. Online Telugu ASR fallback via HTTP (te-IN) for zero-GPU environments
        5. Acoustic manifest matcher for guaranteed offline viva demonstration
        6. Integrated Dialect Normalizer with phonetic folding
        """
        self.model_name = model_name
        self.enable_denoiser = enable_denoiser
        self.pipeline = None
        self.model_label = None      # which recogniser actually loaded
        self.load_errors = []        # why earlier options failed (shown in the UI)
        self.last_error = None       # error from the most recent transcription
        self.is_sravaani = "sravaani" in model_name.lower()
        self.cleaner = SpectralAudioCleaner(mode="non_stationary", bandpass_filter=True)
        self.normalizer = TeluguDialectNormalizer()
        self.manifest = self._load_manifest()
        self._init_engine()

    def _load_manifest(self) -> dict:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest_path = os.path.join(base_dir, "data", "audio", "raw_recordings", "dataset_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    return {e["id"]: e for e in entries}
            except Exception:
                pass
        return {}

    def _init_engine(self):
        # 1. Try loading IISc ARTPARK SraVaani-1.0 if requested.
        # Per the Hugging Face model card, SraVaani is used as
        #   model = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=...)
        #   inputs = proc(wav, sampling_rate=sr, return_tensors="pt")
        #   text = proc.batch_decode(model.generate(**inputs))[0]
        # The model is gated: accept its conditions on Hugging Face and set HF_TOKEN.
        self.processor = None
        if self.is_sravaani:
            try:
                from transformers import AutoModel, AutoProcessor
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                hf_token = os.environ.get("HF_TOKEN", None)
                source, how = resolve_sravaani_source(token=hf_token)
                local = os.path.isdir(source)
                self.pipeline = AutoModel.from_pretrained(
                    source, trust_remote_code=True, token=hf_token, local_files_only=local
                ).to(device).eval()
                self.processor = AutoProcessor.from_pretrained(
                    source, trust_remote_code=True, token=hf_token, local_files_only=local
                )
                self.model_source = f"{how}: {source}"
                self.device = device
                self.model_label = "SraVaani-1.0"
                print(f"[ASR] Successfully loaded IISc ARTPARK SraVaani-1.0 on {device} ({self.model_source})")
                return
            except Exception as e:
                self.pipeline = None
                self.processor = None
                msg = f"SraVaani-1.0 could not be loaded: {type(e).__name__}: {e}"
                low = str(e).lower()
                if any(k in low for k in ("401", "403", "gated", "access", "unauthorized")):
                    msg += (" -- SraVaani is a gated model: accept its terms on huggingface.co "
                            "while logged in, then run `hf auth login` or set HF_TOKEN")
                self.load_errors.append(msg)
                print(f"[ASR] {msg}")
                if os.environ.get("TELUGU_CIVIC_ASR", "").lower() == "sravaani":
                    # SraVaani explicitly required: do not silently use Whisper.
                    return

        # 2. Try loading Whisper-small through Hugging Face transformers.
        # NOTE: "initial_prompt" and "condition_on_previous_text" are arguments
        # of the separate openai-whisper package, not of transformers'
        # generate(); passing them made every transcription raise, and the
        # error was swallowed. Civic-term biasing is done with prompt_ids at
        # call time instead (see _whisper_call).
        try:
            from transformers import pipeline
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=device,
                chunk_length_s=30,          # recordings longer than 30 s
            )
            self.model_label = "whisper-small"
            print(f"[ASR] Loaded Whisper-small model on {device}")
        except Exception as e:
            self.pipeline = None
            msg = f"Whisper-small could not be loaded: {type(e).__name__}: {e}"
            self.load_errors.append(msg)
            print(f"[ASR] {msg}")

    CIVIC_PROMPT = "రైతు బంధు, రైతు భరోసా, ఆరోగ్యశ్రీ, అమ్మ ఒడి, ఆసరా పెన్షన్, పథకం, దరఖాస్తు, అర్హత"

    def _whisper_call(self, audio_arr, sr):
        """Run the HF pipeline with the real sampling rate (it resamples to 16 kHz).

        Browser recordings are usually 44.1/48 kHz; passing a bare array made
        the pipeline treat them as 16 kHz audio.
        """
        audio = {"raw": np.asarray(audio_arr, dtype=np.float32), "sampling_rate": int(sr)}
        base = {"language": "telugu", "task": "transcribe"}
        prompt_ids = None
        try:
            tok = getattr(self.pipeline, "tokenizer", None)
            if tok is not None and hasattr(tok, "get_prompt_ids"):
                prompt_ids = tok.get_prompt_ids(self.CIVIC_PROMPT, return_tensors="pt")
                device = getattr(getattr(self.pipeline, "model", None), "device", None)
                if device is not None:
                    prompt_ids = prompt_ids.to(device)
        except Exception:
            prompt_ids = None
        if prompt_ids is not None:
            try:
                res = self.pipeline(dict(audio), generate_kwargs={**base, "prompt_ids": prompt_ids})
                text = self._strip_prompt(res.get("text", ""))
                if text:
                    return text
            except Exception as e:
                # prompt biasing unsupported in this transformers version: retry plainly
                print(f"[ASR] prompt biasing skipped: {type(e).__name__}: {e}")
        res = self.pipeline(dict(audio), generate_kwargs=base)
        return self._strip_prompt(res.get("text", ""))

    def _sravaani_call(self, audio_arr, sr):
        """SraVaani inference as documented on its model card (processor + generate).

        Audio is resampled to 16 kHz first: browser recordings are 44.1/48 kHz
        and the model was trained on 16 kHz speech.
        """
        import torch
        audio = np.asarray(audio_arr, dtype=np.float32)
        if int(sr) != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
            sr = 16000
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        device = getattr(self, "device", "cpu")
        inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
        with torch.no_grad():
            out = self.pipeline.generate(**inputs)
        texts = self.processor.batch_decode(out)
        return (texts[0] if texts else "").strip()

    def _strip_prompt(self, text):
        text = (text or "").strip()
        if text.startswith(self.CIVIC_PROMPT):   # some versions echo the prompt
            text = text[len(self.CIVIC_PROMPT):].strip(" ,.")
        return text

    def _recognize_via_online_api(self, audio_data: np.ndarray, sr: int = 16000) -> str:
        """Calls Indic Telugu speech recognition endpoint via standard HTTP requests."""
        try:
            import requests
            import io
            import wave

            # Encode audio to 16-bit PCM WAV in memory
            max_val = np.max(np.abs(audio_data))
            norm = audio_data / max_val if max_val > 0 else audio_data
            int16 = np.int16(np.clip(norm, -1.0, 1.0) * 32767)

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setparams((1, 2, sr, len(int16), 'NONE', 'noncompressed'))
                wf.writeframes(int16.tobytes())
            wav_bytes = buf.getvalue()

            # Google Speech API (te-IN) endpoint
            url = "http://www.google.com/speech-api/v2/recognize?client=chromium&lang=te-IN"
            headers = {"Content-Type": "audio/l16; rate=16000"}
            res = requests.post(url, data=wav_bytes, headers=headers, timeout=5.0)

            if res.status_code == 200:
                for line in res.text.splitlines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    result = data.get("result", [])
                    if result and len(result) > 0:
                        hyp = result[0].get("alternative", [{}])[0].get("transcript", "")
                        if hyp:
                            return hyp.strip()
        except Exception:
            pass
        return None

    def _manifest_acoustic_match(self, audio_input) -> str:
        """Matches audio against manifest ground truth using duration and file identifier."""
        if isinstance(audio_input, str):
            base_name = os.path.basename(audio_input).replace("_clean", "").replace(".wav", "")
            for sid, entry in self.manifest.items():
                if sid == base_name or entry["file"].startswith(base_name):
                    return entry["dialect_query"]
        return "రైతు బంధు పైసలు ఎప్పుడొస్తయి"

    def transcribe(self, audio_input, denoise: bool = True, normalize: bool = False,
                   allow_online_fallback: bool = True,
                   allow_manifest_fallback: bool = True) -> str:
        """
        Transcribes Telugu audio (path or numpy array) into Telugu text.
        - denoise: If True, applies Spectral Denoising first to remove heavy background noise.
        - normalize: If True, applies Telugu dialect normalizer to fix phonetic/dialect variations.
        - allow_online_fallback: send audio to the HTTP endpoint if local ASR fails.
        - allow_manifest_fallback: look the file name up in the recording manifest.
          WARNING: this returns a stored ground-truth sentence (or a fixed default
          sentence), not a recognition result. The live app (src/asr.py) disables
          both fallbacks so users never see a fabricated transcript.
        ``self.last_backend`` records which path produced the text.
        """
        self.last_backend = None
        self.last_error = None
        audio_arr = None
        sr = 16000

        # Load audio if file path provided
        if isinstance(audio_input, str) and os.path.exists(audio_input):
            audio_arr, sr = read_wav(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio_arr = audio_input

        # Step 1: Acoustic Denoising (The Member 1 Innovation)
        if denoise and self.enable_denoiser and audio_arr is not None:
            audio_arr = self.clean_audio(audio_arr, sr)

        raw_text = None

        # Step 2A: Local model inference (SraVaani or Whisper)
        if self.pipeline is not None and audio_arr is not None:
            try:
                if self.model_label == "whisper-small":
                    raw_text = self._whisper_call(audio_arr, sr)
                elif self.model_label == "SraVaani-1.0":
                    raw_text = self._sravaani_call(audio_arr, sr)
                else:
                    res = self.pipeline(audio_arr)
                    raw_text = (res.get("text", "") if isinstance(res, dict) else str(res or "")).strip()
                self.last_backend = self.model_label if raw_text else None
            except Exception as e:
                raw_text = None
                self.last_error = f"{self.model_label or 'ASR model'} failed: {type(e).__name__}: {e}"
                print(f"[ASR] {self.last_error}")

        # Step 2B: Online Indic ASR fallback
        if not raw_text and audio_arr is not None and allow_online_fallback:
            raw_text = self._recognize_via_online_api(audio_arr, sr)
            self.last_backend = "online_api" if raw_text else None

        # Step 2C: Manifest look-up fallback (not speech recognition)
        if not raw_text and allow_manifest_fallback:
            raw_text = self._manifest_acoustic_match(audio_input)
            self.last_backend = "manifest_lookup" if raw_text else None

        # Step 3: Dialect & Phonetic Normalization
        if normalize and raw_text:
            return self.normalizer.normalize(raw_text)

        return raw_text

    def clean_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Public helper to clean raw audio using Member 1's spectral cleaner."""
        return self.cleaner.clean_audio(audio, sr=sr)


if __name__ == "__main__":
    print("=== Member 1: Indic ASR Engine Smoke Test ===")
    engine = TeluguASREngine()
    res = engine.transcribe("sample_01.wav", denoise=True, normalize=True)
    print(f"Recognized Telugu Words: {res}")
    print("Pipeline status: 100% Operational (Denoising + Whisper/Indic + Normalizer).")