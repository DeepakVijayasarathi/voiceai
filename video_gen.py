"""
Static-image + narration audio -> a real, shareable MP4, via ffmpeg.

Deliberately NOT an AI-generated moving video (that needs a fundamentally
different, much slower/costlier generative-video model, e.g. Sora-class -
out of scope here, and not something this service has access to verified
pricing/limits for). What's here is the honest, immediately-useful
version: mux the cover image this service already generates (image_gen.py)
with the narration audio it already generates into a standard video file
that plays anywhere (WhatsApp, YouTube, Instagram, etc), using ffmpeg -
already installed on this host, no new heavy dependency.
"""
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger("indic_tts.video_gen")


class VideoGenError(Exception):
    pass


def compose_video(image_bytes: bytes, audio_bytes: bytes) -> bytes:
    """Returns MP4 bytes: the still image held for the full duration of
    the audio track. Raises VideoGenError if ffmpeg isn't available or
    fails - this is opt-in (the caller explicitly asked for a video), so
    failure should be visible, not silently dropped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "cover.png")
        audio_path = os.path.join(tmpdir, "narration.wav")
        video_path = os.path.join(tmpdir, "story.mp4")

        with open(image_path, "wb") as f:
            f.write(image_bytes)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
        except FileNotFoundError:
            raise VideoGenError("ffmpeg is not installed on this host")
        except subprocess.TimeoutExpired:
            raise VideoGenError("ffmpeg timed out composing the video")

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
            logger.error("ffmpeg composition failed: %s", stderr)
            raise VideoGenError(f"ffmpeg failed: {stderr}")

        with open(video_path, "rb") as f:
            return f.read()
