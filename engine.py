"""
TTS engine: loads all 11 MMS-TTS (VITS) models at startup and keeps them
resident in memory for the life of the process (never reload per-request).
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

logger = logging.getLogger("indic_tts.engine")

LANGUAGES = {
    "tam": "Tamil", "hin": "Hindi", "tel": "Telugu", "mal": "Malayalam",
    "kan": "Kannada", "mar": "Marathi", "guj": "Gujarati", "ben": "Bengali",
    "pan": "Punjabi", "ory": "Odia", "asm": "Assamese",
}

_CACHE_MAX_ENTRIES = 256

# VITS's duration predictor is stochastic (noise_scale/noise_scale_duration
# sample randomly per call) - rarely, that sampling produces a pathological
# runaway duration for one token, heard as the output getting "stuck"
# repeating a syllable for many seconds (a known VITS failure mode, not
# specific to any particular input text - reproduced directly: the exact
# same text that triggered it once synthesized completely normally when
# re-run in isolation). Since it's a bad random draw rather than a bad
# input, the fix is detect-and-resample: real Indic speech never drops
# below ~4 chars/sec of synthesized audio, so anything slower than that
# gets retried (a fresh forward pass draws fresh noise).
_MIN_CHARS_PER_SEC = 4.0
_MAX_SYNTH_RETRIES = 3

# VITS's own defaults (config.noise_scale=0.667, config.noise_scale_duration
# =0.8) trade clarity for natural-sounding variation. Lowering
# noise_scale_duration specifically also has a reliability side-benefit: it's
# the exact source of the runaway-duration failure mode above, so reducing
# it below the model's default lowers the odds of hitting that failure mode
# in the first place (the retry logic above remains the backstop for
# whatever residual probability remains).
_DEFAULT_NOISE_SCALE = 0.6
_DEFAULT_NOISE_SCALE_DURATION = 0.6


class _AudioCache:
    """Small process-local LRU cache keyed on (lang, normalized_text).
    Avoids re-running inference for repeated identical requests (common for
    canned prompts/greetings)."""

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES):
        self._data: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    @staticmethod
    def _key(lang: str, text: str) -> str:
        return hashlib.sha1(f"{lang}:{text}".encode("utf-8")).hexdigest()

    def get(self, lang: str, text: str) -> tuple[np.ndarray, int] | None:
        key = self._key(lang, text)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, lang: str, text: str, audio: np.ndarray, sr: int) -> None:
        key = self._key(lang, text)
        with self._lock:
            self._data[key] = (audio, sr)
            self._data.move_to_end(key)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)


class TTSEngine:
    def __init__(self, num_threads: int = 4):
        torch.set_num_threads(num_threads)
        self._models: dict[str, VitsModel] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}
        self._sample_rates: dict[str, int] = {}
        # One lock per language model - VITS forward passes are not
        # guaranteed thread-safe to call concurrently on the *same* model
        # instance, but different languages' models are fully independent
        # and can run truly concurrently.
        self._model_locks: dict[str, threading.Lock] = {}
        self.cache = _AudioCache()

    def load_all(self) -> None:
        for code, name in LANGUAGES.items():
            t0 = time.time()
            model_id = f"facebook/mms-tts-{code}"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id)
            model.eval()
            self._models[code] = model
            self._tokenizers[code] = tokenizer
            self._sample_rates[code] = model.config.sampling_rate
            self._model_locks[code] = threading.Lock()
            logger.info("loaded %s (%s) in %.2fs", code, name, time.time() - t0)

    def languages(self) -> dict[str, str]:
        return dict(LANGUAGES)

    def sample_rate(self, lang: str) -> int:
        return self._sample_rates[lang]

    def synthesize(
        self,
        text: str,
        lang: str,
        speed: float = 1.0,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
    ) -> tuple[np.ndarray, int]:
        if lang not in self._models:
            raise ValueError(f"Unsupported language: {lang}")

        noise_scale = _DEFAULT_NOISE_SCALE if noise_scale is None else noise_scale
        noise_scale_duration = _DEFAULT_NOISE_SCALE_DURATION if noise_scale_duration is None else noise_scale_duration

        cache_key = f"{lang}|speed={speed:.2f}|ns={noise_scale:.2f}|nsd={noise_scale_duration:.2f}"
        cached = self.cache.get(cache_key, text)
        if cached is not None:
            audio, sr = cached
            return audio, sr

        tokenizer = self._tokenizers[lang]
        model = self._models[lang]
        sr = self._sample_rates[lang]

        inputs = tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            # Every character in `text` was out of this language's grapheme
            # vocabulary (tokenizer drops unknown chars silently rather than
            # erroring) - the VITS model can't handle an empty sequence, so
            # fail clearly here instead of letting it crash inside forward().
            raise ValueError(
                f"Text contains no characters recognizable in language '{lang}' "
                f"after normalization: {text!r}"
            )
        with self._model_locks[lang]:
            # speaking_rate is VITS's native duration-scale knob (scales the
            # predicted phoneme durations before synthesis) - it changes
            # tempo without the pitch distortion a naive resample would
            # introduce, and costs nothing extra since it's applied inside
            # the model's own forward pass. noise_scale/noise_scale_duration
            # are set the same way (instance attributes read directly by
            # forward(), not forward() kwargs - transformers' VITS doesn't
            # expose them as call parameters).
            model.speaking_rate = speed if speed > 0 else 1.0
            model.noise_scale = noise_scale
            model.noise_scale_duration = noise_scale_duration

            audio = None
            for attempt in range(_MAX_SYNTH_RETRIES):
                with torch.inference_mode():
                    output = model(**inputs).waveform
                candidate = output.squeeze().numpy().astype(np.float32)
                duration_s = len(candidate) / sr
                chars_per_sec = len(text) / duration_s if duration_s > 0 else 0

                if chars_per_sec >= _MIN_CHARS_PER_SEC or attempt == _MAX_SYNTH_RETRIES - 1:
                    audio = candidate
                    if attempt > 0:
                        logger.warning(
                            "lang=%s: resampled after %d anomalous-duration attempt(s) "
                            "(%.1f chars/sec on the accepted attempt)",
                            lang, attempt, chars_per_sec,
                        )
                    break
                logger.warning(
                    "lang=%s: anomalous synthesis duration (%.1f chars/sec, "
                    "expected >=%.1f) on attempt %d/%d, retrying",
                    lang, chars_per_sec, _MIN_CHARS_PER_SEC, attempt + 1, _MAX_SYNTH_RETRIES,
                )

        self.cache.put(cache_key, text, audio, sr)
        return audio, sr
