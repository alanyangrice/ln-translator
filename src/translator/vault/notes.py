"""Generic frontmatter-aware Markdown note I/O.

Every artifact in the vault is a Markdown file with a YAML frontmatter
block at the top and free-form Markdown body below. This module is the
only place that knows how to parse and serialize that format; everything
else (rules, deviations, summaries) wraps these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter


@dataclass
class Note:
    """A single Markdown note in the vault.

    ``meta`` is the YAML frontmatter as a dict. ``body`` is the Markdown
    body without the leading ``---`` block. ``path`` is set by readers
    and ignored by writers.
    """

    meta: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    path: Path | None = None


def read_note(path: Path) -> Note:
    """Parse a frontmatter Markdown file into a :class:`Note`.

    Raises ``FileNotFoundError`` if the path doesn't exist; raises
    ``ValueError`` if the frontmatter block is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # frontmatter raises various YAML errors
        raise ValueError(f"Malformed frontmatter in {path}: {exc}") from exc
    return Note(meta=dict(post.metadata), body=post.content, path=path)


def write_note(path: Path, note: Note, *, mkdir: bool = True) -> None:
    """Serialize a :class:`Note` to disk as ``frontmatter + body``.

    If ``mkdir`` is True (default), parent directories are created as needed.
    Existing files are overwritten — callers that want append-only semantics
    must check beforehand.
    """
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(content=note.body, **note.meta)
    path.write_text(frontmatter.dumps(post), encoding="utf-8")


def list_notes(directory: Path, *, suffix: str = ".md") -> list[Path]:
    """Return all notes under ``directory`` with the given suffix.

    Returns an empty list if the directory doesn't exist (callers
    don't need to special-case the uninitialized-vault case).
    """
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob(f"*{suffix}") if p.is_file())
