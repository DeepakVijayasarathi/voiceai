"""
Voice cloning: proxies to the existing IndicF5 service rather than
reimplementing cloning here. IndicF5 (F5-TTS, reference-audio-conditioned)
is the only model on this box that actually supports it - this service's
own engine (VITS) is a fixed single-speaker-per-language model by design,
chosen specifically for speed (see README). Cloning stays slow (IndicF5's
30-70s/request on this CPU) because that's inherent to the architecture
doing the cloning, not something this proxy can speed up.
"""
import base64
import binascii
import os

import httpx

INDICF5_BASE_URL = os.environ.get("INDICF5_BASE_URL", "http://127.0.0.1:8500")
INDICF5_API_KEY = os.environ.get("INDICF5_API_KEY")

_client = httpx.Client(timeout=120.0)


class CloneError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def clone_speech(text: str, ref_audio_b64: str, ref_text: str) -> bytes:
    """Returns WAV bytes synthesized in the cloned voice, via IndicF5."""
    try:
        base64.b64decode(ref_audio_b64, validate=True)
    except binascii.Error:
        raise CloneError(400, "ref_audio_b64 is not valid base64")

    try:
        resp = _client.post(
            f"{INDICF5_BASE_URL}/tts",
            headers={"x-api-key": INDICF5_API_KEY} if INDICF5_API_KEY else {},
            json={"text": text, "ref_audio_b64": ref_audio_b64, "ref_text": ref_text},
        )
    except httpx.HTTPError as e:
        raise CloneError(502, f"IndicF5 service unreachable: {e}")

    if resp.status_code != 200:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise CloneError(resp.status_code, f"IndicF5 cloning failed: {detail}")

    return resp.content
