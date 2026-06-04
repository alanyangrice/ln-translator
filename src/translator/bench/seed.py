"""Seed the issue ledger from past chat critiques.

The user's chapter-by-chapter chats contain dozens of concrete,
line-anchored critiques of the form::

    @data/output/part_233-v2-rag-deepseek-auto/translation.en.txt:40 weird
    wording, specifically with "just force my order through"

This module mines those comments for parts 230-236, resolves each
``@file:line`` reference to the EN excerpt on disk, and captures the
assistant's follow-up turn(s) as ``resolution_guidance`` (per the user's
request: "look at how the LLM response reacts afterwards and use that as
a guide for what to check for"). An optional LLM ``--distill`` pass
compresses that into crisp guidance + a sharper category/severity.

Seeding is deliberately conservative: it only emits draft issues from
clearly-anchored ``@file:line`` comments, then the user reviews/trims the
generated ``issues.jsonl``. Prose-only critiques (no file anchor) can be
added later with ``bench issues add``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from translator.bench.ledger import Issue, next_issue_id
from translator.config import MODELS, PATHS, REASONING

# Matches an @-reference to a translation output, capturing the path, the
# part id, the line spec, and the trailing comment up to the next @-ref of
# any kind (output, vault, src, absolute) or end-of-text.
_REF_RE = re.compile(
    r"@(data/output/(part_\d+)[^/\s]*/translation\.en\.txt):(\d+(?:-\d+)?)\s*"
    r"(.*?)(?=@data/output|@knowledge-vault|@src/|@/|\Z)",
    re.DOTALL,
)


def _clean_comment(text: str) -> str:
    """Strip the chat wrappers and collapse whitespace.

    Comments captured as the *last* @-ref in a message tend to trail into
    the message's closing ``</user_query>`` and any global instruction the
    user appended; cut those off so the comment is just the critique.
    """
    text = text.split("</user_query>")[0]
    text = text.split("</attached_files>")[0]
    # Drop a trailing global instruction the user often appends after the
    # per-line notes (kept conservative to avoid eating real critique).
    for marker in ("\nPlease analyze", "\nPlease look", "\nPlease suggest", "\nPlease begin"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return " ".join(text.split()).strip()


def default_transcripts_dir() -> Path:
    """Best-effort guess of the Cursor agent-transcripts dir for this repo."""
    slug = str(PATHS.repo_root).strip("/").replace("/", "-")
    return Path.home() / ".cursor" / "projects" / slug / "agent-transcripts"


@dataclass
class RawComment:
    part_id: str
    suffix: str
    rel_path: str
    line_spec: str
    comment: str
    en_excerpt: str
    assistant_followup: str
    chat_id: str
    turn_index: int


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def _parse_transcript(path: Path) -> list[dict]:
    """Return an ordered list of ``{role, text}`` turns (text concatenated)."""
    turns: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("role")
        content = obj.get("message", {}).get("content", [])
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        turns.append({"role": role, "text": text})
    return turns


def _parent_transcripts(transcripts_dir: Path) -> list[Path]:
    if not transcripts_dir.exists():
        return []
    out: list[Path] = []
    for p in transcripts_dir.glob("*/*.jsonl"):
        if "subagents" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def _resolve_excerpt(rel_path: str, line_spec: str) -> str:
    path = PATHS.repo_root / rel_path
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if "-" in line_spec:
        a, b = line_spec.split("-", 1)
        try:
            lo, hi = int(a), int(b)
        except ValueError:
            return ""
    else:
        try:
            lo = hi = int(line_spec)
        except ValueError:
            return ""
    chunk = [lines[i - 1] for i in range(lo, hi + 1) if 1 <= i <= len(lines)]
    return " ".join(s.strip() for s in chunk if s.strip())


def _followup(turns: list[dict], user_index: int, *, max_chars: int = 2500) -> str:
    parts: list[str] = []
    for t in turns[user_index + 1:]:
        if t["role"] == "user":
            break
        if t["role"] == "assistant" and t["text"].strip():
            parts.append(t["text"].strip())
        if sum(len(p) for p in parts) >= max_chars:
            break
    text = "\n\n".join(parts).strip()
    return text[:max_chars]


def mine_comments(
    transcripts_dir: Path | None = None,
    *,
    parts: range = range(230, 237),
) -> list[RawComment]:
    """Extract anchored critiques for the target parts across all chats."""
    tdir = transcripts_dir or default_transcripts_dir()
    raw: list[RawComment] = []
    for tpath in _parent_transcripts(tdir):
        chat_id = tpath.stem
        turns = _parse_transcript(tpath)
        for idx, turn in enumerate(turns):
            if turn["role"] != "user":
                continue
            text = turn["text"]
            if "@data/output" not in text:
                continue
            followup = _followup(turns, idx)
            for m in _REF_RE.finditer(text):
                rel_path, part_id, line_spec, comment = m.groups()
                try:
                    num = int(part_id.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if num not in parts:
                    continue
                dirname = Path(rel_path).parts[-2]  # part_233-v2-rag-deepseek-auto
                suffix = dirname[len(part_id) + 1:] if dirname.startswith(part_id + "-") else ""
                comment = _clean_comment(comment)
                if len(comment) < 4:
                    continue
                raw.append(
                    RawComment(
                        part_id=part_id,
                        suffix=suffix,
                        rel_path=rel_path,
                        line_spec=line_spec,
                        comment=comment,
                        en_excerpt=_resolve_excerpt(rel_path, line_spec),
                        assistant_followup=followup,
                        chat_id=chat_id,
                        turn_index=idx,
                    )
                )
    return raw


# ---------------------------------------------------------------------------
# Heuristic categorization (a starting point; user refines on review)
# ---------------------------------------------------------------------------

_CATEGORY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("formatting", ("quote", "stars", "asterisk", "line break", "spacing", "no line")),
    ("tense", ("tense", "past tense", "should be was", "should be \"was\"", "present tense")),
    ("untranslated_term", ("glossary", "what is", "what's", "isn't this", "is this ramen", "nabe", "mizuna", "edamame", "napa")),
    ("terminology", ("order", "request", "name", "right name", "naming")),
    ("coordination_and", ("and structure", "using and", "and all the time", "becomes two sentences", "two sentences")),
    ("preferred_variant", ("liked", "previous version", "before in the previous")),
    ("metaphor", ("metaphor", "analogy", "caterpillar", "simile")),
)


def _guess_category(comment: str) -> str:
    low = comment.lower()
    for cat, hints in _CATEGORY_HINTS:
        if any(h in low for h in hints):
            return cat
    return "translationese"


def _guess_severity(comment: str) -> str:
    low = comment.lower()
    if any(h in low for h in ("really bad", "fundamental", "error", "wrong", "must", "violat")):
        return "major"
    return "minor"


# ---------------------------------------------------------------------------
# Optional LLM distillation
# ---------------------------------------------------------------------------

_DISTILL_SCHEMA: dict = {
    "name": "distilled_issue",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "severity": {"type": "string", "enum": ["major", "minor"]},
            "resolution_guidance": {"type": "string"},
            "preferred_fix": {"type": "string"},
        },
        "required": ["category", "severity", "resolution_guidance", "preferred_fix"],
        "additionalProperties": False,
    },
}

def _distill(rc: RawComment, model: str | None) -> dict:
    from string import Template

    from translator.bench.categories import CATEGORY_KEYS
    from translator.bench.prompts import load_template
    from translator.inference.provider import complete

    prompt = Template(load_template("distill_issue.md")).safe_substitute(
        categories=", ".join(CATEGORY_KEYS),
        comment=rc.comment,
        excerpt=rc.en_excerpt or "(not recorded)",
        followup=rc.assistant_followup or "(none)",
    )
    raw = complete(
        model=model or MODELS.clustering,
        prompt=prompt,
        temperature=0.0,
        max_tokens=8192,
        reasoning_effort=REASONING.clustering,  # type: ignore[arg-type]
        json_schema=_DISTILL_SCHEMA,
    )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Build draft issues
# ---------------------------------------------------------------------------

def build_seed_issues(
    transcripts_dir: Path | None = None,
    *,
    parts: range = range(230, 237),
    distill: bool = False,
    model: str | None = None,
    progress=None,
) -> list[Issue]:
    """Mine + dedup + convert to draft :class:`Issue` objects (not saved)."""
    raw = mine_comments(transcripts_dir, parts=parts)

    # Dedup on (part_id, line_spec, normalized comment).
    seen: set[tuple[str, str, str]] = set()
    deduped: list[RawComment] = []
    for rc in raw:
        key = (rc.part_id, rc.line_spec, " ".join(rc.comment.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rc)

    issues: list[Issue] = []
    allocated: list[Issue] = []  # to keep id allocation unique within this batch
    for rc in deduped:
        category = _guess_category(rc.comment)
        severity = _guess_severity(rc.comment)
        guidance = rc.assistant_followup or None
        preferred = None
        if distill:
            try:
                d = _distill(rc, model)
                category = d.get("category", category) or category
                severity = d.get("severity", severity) or severity
                guidance = d.get("resolution_guidance") or guidance
                preferred = d.get("preferred_fix") or None
                if progress:
                    progress(f"distilled {rc.part_id}:{rc.line_spec}")
            except Exception as exc:  # keep heuristic on failure
                if progress:
                    progress(f"distill failed {rc.part_id}:{rc.line_spec}: {exc}")

        issue = Issue(
            id=next_issue_id(rc.part_id, allocated),
            part_id=rc.part_id,
            category=category,
            severity=severity,
            user_comment=rc.comment,
            first_seen_suffix=rc.suffix,
            en_excerpt_original=rc.en_excerpt,
            resolution_guidance=guidance,
            preferred_fix=preferred,
            evidence_refs={
                "chat_id": rc.chat_id,
                "turn_index": rc.turn_index,
                "ref": f"{rc.rel_path}:{rc.line_spec}",
            },
            source="auto-seed",
        )
        issues.append(issue)
        allocated.append(issue)
    return issues
