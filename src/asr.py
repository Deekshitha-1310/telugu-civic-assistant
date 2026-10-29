"""
Live ASR wrapper used by the Streamlit app.

Wraps Member 1's TeluguASREngine (spectral denoiser + SraVaani/Whisper) with
three guarantees the UI relies on:

* ``status()`` reports what is really available: a local model is needed,
  and the Python packages must be installed. Nothing is reported as
  "operational" until the model has actually loaded.
* Uploaded audio is validated (size, format, decodable) before recognition.
* No fabricated transcripts: the engine's manifest look-up fallback (which
  returns a stored sentence for any audio) is disabled, and the online HTTP
  fallback (which sends the citizen's audio to a third-party endpoint) is off
  unless TELUGU_CIVIC_ONLINE_ASR=1 is set.

Every result carries an ``error_code`` so the UI can show a localized message:
  asr_unavailable | invalid_audio | audio_too_large | no_speech | asr_error
"""
import importlib.util
import io
import os
import tempfile
import wave

MAX_AUDIO_BYTES = int(os.environ.get("TELUGU_CIVIC_MAX_AUDIO_MB", "10")) * 1024 * 1024
MIN_AUDIO_SECONDS = 0.3


def _have(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def accepted_audio_types():
    """File extensions the decoder in this environment can actually read."""
    types = ["wav"]
    if _have("soundfile"):
        types += ["flac", "ogg"]
    return types


def _sniff_format(data: bytes):
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    return None


def validate_audio(audio_bytes: bytes):
    """Return (ok, error_code, detail). Pure function — no model needed."""
    if not audio_bytes:
        return False, "invalid_audio", "empty file"
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return False, "audio_too_large", f"{len(audio_bytes)} bytes > {MAX_AUDIO_BYTES}"
    fmt = _sniff_format(audio_bytes)
    if fmt is None or fmt not in accepted_audio_types():
        return False, "invalid_audio", f"unsupported or unrecognised format ({fmt or 'unknown'})"
    if fmt == "wav":
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wf:
                frames, rate = wf.getnframes(), wf.getframerate()
        except (wave.Error, EOFError) as exc:
            # Non-PCM WAVs (e.g. float) may still be readable by soundfile.
            if not _have("soundfile"):
                return False, "invalid_audio", f"unreadable WAV ({exc})"
        else:
            if rate <= 0 or frames / rate < MIN_AUDIO_SECONDS:
                return False, "invalid_audio", "audio is too short"
    return True, None, None


class ASRService:
    def __init__(self, engine_factory=None):
        self.backend = "telugu_asr_engine"
        self._engine = None
        self._load_attempted = False
        self.error = None
        self._engine_factory = engine_factory
        self.allow_online = os.environ.get("TELUGU_CIVIC_ONLINE_ASR", "").strip() in {"1", "true", "yes"}

    def _dependencies_missing(self):
        return [m for m in ("torch", "transformers") if not _have(m)]

    def _get_engine(self):
        if self._engine is None and not self._load_attempted:
            self._load_attempted = True
            try:
                if self._engine_factory is not None:
                    self._engine = self._engine_factory()
                else:
                    missing = self._dependencies_missing()
                    if missing:
                        self.error = "Missing packages for local ASR: " + ", ".join(missing)
                        return None
                    from asr_engine import TeluguASREngine
                    self._engine = TeluguASREngine(enable_denoiser=True)
                if getattr(self._engine, "pipeline", "n/a") is None and not self.allow_online:
                    load_errors = getattr(self._engine, "load_errors", None) or []
                    self.error = "No speech-recognition model could be loaded."
                    if load_errors:
                        self.error += " " + " | ".join(load_errors)
                    self._engine = None
            except Exception as exc:
                self.error = f"ASR engine could not be initialised: {exc}"
                self._engine = None
        return self._engine

    def status(self):
        """operational | not_loaded (packages present, model loads on first use) | unavailable"""
        if self._engine is not None:
            return "operational"
        if self._load_attempted:
            return "unavailable"
        if self._engine_factory is None and self._dependencies_missing():
            self.error = "Missing packages for local ASR: " + ", ".join(self._dependencies_missing())
            return "unavailable"
        return "not_loaded"

    def model_info(self):
        """Which recogniser is actually loaded, and why others failed."""
        engine = self._engine
        return {
            "model": getattr(engine, "model_label", None) if engine else None,
            "load_errors": list(getattr(engine, "load_errors", []) or []) if engine else [],
        }

    def available(self):
        return self.status() != "unavailable"

    def transcribe(self, audio_bytes: bytes, filename="query.wav"):
        ok, code, detail = validate_audio(audio_bytes)
        if not ok:
            return {"ok": False, "transcript": None, "error_code": code, "error": detail}

        engine = self._get_engine()
        if engine is None:
            return {"ok": False, "transcript": None, "error_code": "asr_unavailable",
                    "error": self.error or "ASR engine could not be loaded."}

        suffix = os.path.splitext(filename or "")[1] or ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            text = engine.transcribe(
                temp_path, denoise=True, normalize=False,
                allow_online_fallback=self.allow_online,
                allow_manifest_fallback=False,
            )
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                engine_error = getattr(engine, "last_error", None)
                if engine_error:  # the model crashed: say so, don't blame the speaker
                    return {"ok": False, "transcript": None, "error_code": "asr_error",
                            "error": engine_error}
                return {"ok": False, "transcript": None, "error_code": "no_speech",
                        "error": "No speech could be recognised in the audio."}
            return {"ok": True, "transcript": text, "error_code": None, "error": None,
                    "backend": getattr(engine, "last_backend", None)}
        except Exception as exc:
            return {"ok": False, "transcript": None, "error_code": "asr_error",
                    "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass