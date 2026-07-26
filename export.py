"""
Multi-format export, built entirely from data this service already has -
no new heavy ML models, matching the CPU-only design goal.

- Game: the existing branch feature (see app.py's POST /stories/{id}/branch)
  exposed as a navigable choice-tree - "game" here means interactive
  fiction, not a game engine.
- Comic: the scene images from images.py, paired with the native-script
  story excerpt each came from, returned as JSON panels - no server-side
  text/speech-bubble compositing, which would need a bundled Unicode font
  covering all 11 Indic scripts; the client renders the panel layout.
- Video: an ffmpeg slideshow of those same scene images, each held for its
  time range, muxed with the story's saved narration audio. Requires the
  story to have been generated with images=true (see app.py's
  /dream-to-story, which only persists narration audio to disk in that
  case), and requires the `ffmpeg` binary on the host - the one genuinely
  new external (non-Python) dependency in this service, checked at call
  time rather than assumed.
"""
import logging
import os
import shutil
import subprocess
import tempfile

import db
from images import STATIC_DIR

logger = logging.getLogger("indic_tts.export")

_MAX_TREE_DEPTH = 20  # defensive bound, not expected to ever bite in practice


class ExportError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _build_tree_node(story_id: int, depth: int) -> dict:
    story = db.get_story(story_id)
    if not story:
        return {"id": story_id, "title": None, "text": None, "branch_note": None, "children": []}
    node = {
        "id": story["id"],
        "title": story["title"],
        "text": story["text"],
        "branch_note": story.get("branch_note"),
        "children": [],
    }
    if depth < _MAX_TREE_DEPTH:
        for child in db.get_child_branches(story_id):
            node["children"].append(_build_tree_node(child["id"], depth + 1))
    return node


def build_game_tree(story_id: int) -> dict:
    """Recursively walks this story's branches (db.get_child_branches) into
    a navigable choice-tree: {id, title, text, branch_note, children: [...]}.
    A leaf (no children yet) is just a story with an empty children list -
    a client walks the tree by presenting each node's children as the
    available choices from that point."""
    story = db.get_story(story_id)
    if not story:
        raise ExportError(404, f"Story {story_id} not found")
    return _build_tree_node(story_id, depth=0)


def build_comic_panels(story_id: int) -> dict:
    """Returns {"panels": [{"image_url", "caption", "start_time_s"}]} from
    this story's saved scene images (see images.py / db.story_images).
    `caption` is the native-script story excerpt each image came from, not
    the English image-generation prompt - a reader should see their
    story's own text, not the model's internal instructions. Empty panels
    (not an error) for a story that wasn't generated with images=true."""
    story = db.get_story(story_id)
    if not story:
        raise ExportError(404, f"Story {story_id} not found")
    images = db.get_images_for_story(story_id)
    panels = [
        {"image_url": img["url"], "caption": img["excerpt"], "start_time_s": img["start_time_s"]}
        for img in images
    ]
    return {"panels": panels}


def render_video(story_id: int) -> bytes:
    """Builds an MP4 slideshow: each scene image held for its time range,
    muxed with the story's saved narration audio (see app.py's
    _save_story_audio - only written when the story was generated with
    images=true). Raises ExportError (404/409/501/500) rather than
    crashing on any of: unknown story, no saved audio/images, or a missing
    `ffmpeg` binary - checked here rather than assumed, since it's the one
    dependency in this whole feature set that isn't a Python package."""
    story = db.get_story(story_id)
    if not story:
        raise ExportError(404, f"Story {story_id} not found")

    if shutil.which("ffmpeg") is None:
        raise ExportError(501, "Video export requires the 'ffmpeg' binary, which isn't installed on this host.")

    audio_path = os.path.join(STATIC_DIR, "audio", f"{story_id}.wav")
    if not os.path.isfile(audio_path):
        raise ExportError(409, f"Story {story_id} has no saved narration audio - generate it with images=true on /dream-to-story first.")

    images = db.get_images_for_story(story_id)
    if not images:
        raise ExportError(409, f"Story {story_id} has no scene images - generate it with images=true on /dream-to-story first.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        concat_list_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for i, img in enumerate(images):
                image_path = os.path.join(STATIC_DIR, img["url"].removeprefix("/static/").replace("/", os.sep))
                duration = max(0.1, img["end_time_s"] - img["start_time_s"])
                f.write(f"file '{image_path}'\nduration {duration}\n")
            # ffmpeg's concat demuxer needs the last image repeated without
            # a trailing duration, or it drops the final segment's hold time.
            if images:
                last_path = os.path.join(STATIC_DIR, images[-1]["url"].removeprefix("/static/").replace("/", os.sep))
                f.write(f"file '{last_path}'\n")

        output_path = os.path.join(tmp_dir, "output.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-i", audio_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1024:1024",
            "-c:a", "aac", "-shortest",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed for story_id=%s: %s", story_id, e.stderr.decode(errors="replace"))
            raise ExportError(500, "Video rendering failed (ffmpeg error)")
        except subprocess.TimeoutExpired:
            raise ExportError(500, "Video rendering timed out")

        with open(output_path, "rb") as f:
            return f.read()
