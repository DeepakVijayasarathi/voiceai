import io
import wave

import numpy as np
import pytest
from PIL import Image

from video_gen import VideoGenError, compose_video


def _tiny_png() -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 60, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tiny_wav(duration_s: float = 0.5, sr: int = 24000) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def test_compose_video_produces_valid_mp4():
    video_bytes = compose_video(_tiny_png(), _tiny_wav())
    assert len(video_bytes) > 0
    # MP4 files carry an 'ftyp' box a few bytes into the file - a cheap,
    # real signature check that this is actually a valid MP4 container,
    # not just "ffmpeg exited 0 and wrote something."
    assert b"ftyp" in video_bytes[:32]


def test_compose_video_raises_on_garbage_input():
    with pytest.raises(VideoGenError):
        compose_video(b"not an image", b"not audio either")
