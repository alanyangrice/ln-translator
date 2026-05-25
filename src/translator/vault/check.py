"""Offline pre-flight checks for the knowledge vault and glossary.

Runs entirely without API keys. Intended as a gate before the first
paid translation or eval round so misconfigured templates / empty
glossary surface early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

from translator.config import PATHS, VAULT
from translator.glossary import load_glossary
from translator.prep.holdout import load_holdout
from translator.vault.init import is_initialized, vault_status
from translator.vault.rules import load_active_rules


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class VaultCheckReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))


_REQUIRED_TEMPLATE_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    VAULT.prompt_template: (
        "$rules",
        "$glossary",
        "$style_profile",
        "$reference_parts",
        "$new_part_id",
        "$new_jp_chapter",
    ),
    VAULT.comparison_template: (
        "$active_rules",
        "$glossary",
        "$style_profile",
        "$part_id",
        "$pov",
        "$jp_source",
        "$llm_translation",
        "$reference_translation",
    ),
    VAULT.clustering_template: ("$active_rules", "$pruned_rules", "$deviations"),
    VAULT.critique_template: (
        "$active_rules",
        "$glossary",
        "$style_profile",
        "$part_id",
        "$pov",
        "$jp_source",
        "$draft",
    ),
    VAULT.revise_template: (
        "$rules",
        "$glossary",
        "$style_profile",
        "$reference_parts",
        "$new_part_id",
        "$new_jp_chapter",
        "$critic_flags",
        "$previous_draft",
    ),
}


def _check_template(path_key: str) -> CheckResult:
    path = PATHS.knowledge_vault / path_key
    if not path.exists():
        return CheckResult(path_key, False, "missing — run `translator vault init`")
    text = path.read_text(encoding="utf-8")
    missing = [p for p in _REQUIRED_TEMPLATE_PLACEHOLDERS[path_key] if p not in text]
    if missing:
        return CheckResult(path_key, False, f"missing placeholders: {', '.join(missing)}")
    # Template() will raise on invalid $ syntax; safe_substitute is enough to smoke-test.
    try:
        Template(text).safe_substitute()
    except ValueError as exc:
        return CheckResult(path_key, False, f"invalid Template syntax: {exc}")
    return CheckResult(path_key, True, "ok")


def run_vault_check(*, sample_part: str | None = None) -> VaultCheckReport:
    """Run all offline vault / glossary / prompt checks."""
    report = VaultCheckReport()

    report.add("vault initialized", is_initialized(), str(PATHS.knowledge_vault))

    status = vault_status()
    report.add(
        "glossary file present",
        status.glossary_present,
        VAULT.glossary_file,
    )

    entries = load_glossary()
    report.add(
        "glossary loads",
        len(entries) > 0,
        f"{len(entries)} entries",
    )

    from translator.style import DIMENSIONS, load_style_profile

    profile = load_style_profile()
    style_dir = PATHS.knowledge_vault / VAULT.style
    expected_dimensions = len(DIMENSIONS)
    if not style_dir.exists():
        report.add("style directory present", False, "missing — run `translator vault init`")
    elif profile.has_content and profile.n_dimensions == expected_dimensions:
        report.add(
            "style profile extracted",
            True,
            (
                f"{profile.n_dimensions}/{expected_dimensions} dimensions, "
                f"through {profile.extracted_through or '?'}, "
                f"{profile.n_chapters} chapters, "
                f"model {profile.model or '?'}"
            ),
        )
    elif profile.has_content:
        report.add(
            "style profile extracted",
            False,
            f"incomplete: only {profile.n_dimensions}/{expected_dimensions} dimensions on disk",
        )
    else:
        report.add(
            "style profile extracted",
            False,
            "no dimensions yet — run `translator style extract` to bootstrap (translation continues without it)",
        )

    for template_key in _REQUIRED_TEMPLATE_PLACEHOLDERS:
        result = _check_template(template_key)
        report.add(f"template: {template_key}", result.ok, result.detail)

    active = load_active_rules()
    report.add(
        "active rules readable",
        True,
        f"{len(active)} active (empty is fine before round 1)",
    )

    holdout = load_holdout()
    part_id = sample_part
    if part_id is None and holdout and holdout.part_ids:
        part_id = holdout.part_ids[0]
    if part_id is None:
        part_id = "part_004"

    try:
        from translator.inference.prompt import assemble_prompt
        from translator.inference.window import build_window

        window = build_window(part_id, holdout=holdout)
        prompt = assemble_prompt(part_id, window)
        ok = (
            prompt.template_source == "vault"
            and len(prompt.text) > 500
            and "マフラー" in prompt.text  # glossary injected
            and part_id in prompt.text
        )
        detail = (
            f"part={part_id}, template={prompt.template_source}, "
            f"window={len(prompt.window_part_ids)} parts, "
            f"chars={len(prompt.text):,}, rules={len(prompt.active_rule_ids)}"
        )
        report.add("prompt dry assembly", ok, detail)
    except Exception as exc:
        report.add("prompt dry assembly", False, str(exc))

    return report


__all__ = ["VaultCheckReport", "run_vault_check"]
