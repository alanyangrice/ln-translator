"""Write a PartContent to disk in two forms.

For each entry we write two files under ``data/parallel/``:

  * ``{id}.{side}.json`` — structured paragraph list (kind-classified).
    Used by validators that need paragraph-level granularity.
  * ``{id}.{side}.txt``  — plain text the translation prompt feeds to the
    model. Blank paragraphs render as empty lines so scene breaks survive
    the round-trip into the prompt.

The raw HTML is already cached under ``data/raw/{jp,en}/`` by the http client.
"""

from __future__ import annotations

from translator.config import PATHS
from translator.scraper.models import PartContent


def write_part_content(content: PartContent) -> None:
    PATHS.parallel.mkdir(parents=True, exist_ok=True)
    json_path = PATHS.parallel / f"{content.id}.{content.side}.json"
    txt_path = PATHS.parallel / f"{content.id}.{content.side}.txt"

    json_path.write_text(
        content.model_dump_json(indent=2),
        encoding="utf-8",
    )

    text_lines: list[str] = []
    for p in content.paragraphs:
        if p.kind == "blank":
            text_lines.append("")
        else:
            text_lines.append(p.text)
    txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
