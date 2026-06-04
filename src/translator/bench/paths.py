"""Small helpers for locating bench translation outputs.

Keeps the ``{part_id}-{suffix}`` convention used by ``translate_part`` in
one place so run/check/report agree on where translations live.
"""

from __future__ import annotations

from pathlib import Path

from translator.config import PATHS


def label_suffix(label: str) -> str:
    """The output suffix produced by ``bench run --label <label>``."""
    return f"bench-{label}"


def output_dir(part_id: str, suffix: str | None) -> Path:
    if suffix:
        return PATHS.output / f"{part_id}-{suffix}"
    return PATHS.output / part_id


def translation_path(part_id: str, suffix: str | None) -> Path:
    return output_dir(part_id, suffix) / "translation.en.txt"
