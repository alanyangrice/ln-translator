"""Typer CLI entrypoint.

Subcommand groups follow the v3 workflow:

    scrape     fetch JP/EN content
    prep       POV lookup + stratified holdout selection
    vault      knowledge-vault init / status
    glossary   show / dump glossary
    style      extract + inspect the writing-style profile
    translate  assemble prompt + translate one part (supports --dry-run)
    validate   run dialogue/name/length checks on an existing translation
    evaluate   self-evaluation loop (deviations, cluster, judge, score, report)
    info       resolved config

The ``scrape`` group is wired end-to-end. Most of ``evaluate`` requires
API keys; ``translate --dry-run`` and ``validate`` work without them.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from translator.config import MODELS, PATHS, SCRAPER, THRESHOLDS

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="ln-translator: literary translation pipeline (sliding window + rule vault).",
)

scrape_app = typer.Typer(no_args_is_help=True, help="Fetch JP and EN sources.")
prep_app = typer.Typer(no_args_is_help=True, help="POV lookup + stratified holdout.")
vault_app = typer.Typer(no_args_is_help=True, help="knowledge-vault init/status.")
glossary_app = typer.Typer(no_args_is_help=True, help="View glossary entries.")
style_app = typer.Typer(no_args_is_help=True, help="Extract / inspect the writing-style profile.")
evaluate_app = typer.Typer(no_args_is_help=True, help="Self-evaluation loop.")
precedents_app = typer.Typer(
    no_args_is_help=True,
    help="Build / inspect the precedent RAG index (JP↔EN parallel corpus).",
)

app.add_typer(scrape_app, name="scrape")
app.add_typer(prep_app, name="prep")
app.add_typer(vault_app, name="vault")
app.add_typer(glossary_app, name="glossary")
app.add_typer(style_app, name="style")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(precedents_app, name="precedents")

console = Console()


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------

@scrape_app.command("toc")
def scrape_toc(
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch the ToC pages even if cached."),
) -> None:
    """Build data/metadata/toc.json from the EN ToC and Kakuyomu work index."""
    from translator.scraper.toc import build_toc

    PATHS.ensure()
    entries = build_toc(refresh=refresh)
    PATHS.toc_json.write_text(
        json.dumps([e.model_dump() for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(
        f"[green]Wrote[/green] {len(entries)} entries to {PATHS.toc_json.relative_to(PATHS.repo_root)}"
    )


@scrape_app.command("jp")
def scrape_jp(
    only: str | None = typer.Option(None, "--only", help="Comma-separated ids to scrape."),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N entries."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch even if cached."),
) -> None:
    """Download Kakuyomu episodes for every ToC entry that has a kakuyomu_episode_id."""
    from translator.scraper.jp_source import scrape_all_jp

    PATHS.ensure()
    only_ids = set(only.split(",")) if only else None
    stats = scrape_all_jp(only_ids=only_ids, limit=limit, refresh=refresh)
    _print_kv_table(stats, "JP scrape results")


@scrape_app.command("en")
def scrape_en(
    only: str | None = typer.Option(None, "--only", help="Comma-separated ids."),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N entries."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch even if cached."),
) -> None:
    """Download avelilium.com EN posts for every ToC entry with a url_en."""
    from translator.scraper.en_source import scrape_all_en

    PATHS.ensure()
    only_ids = set(only.split(",")) if only else None
    stats = scrape_all_en(only_ids=only_ids, limit=limit, refresh=refresh)
    _print_kv_table(stats, "EN scrape results")


@scrape_app.command("review")
def scrape_review() -> None:
    """Print the JP-mapping needs_review entries with nearby Kakuyomu chapter
    candidates, to help build data/metadata/kakuyomu_overrides.json."""
    from translator.scraper.jp_source import load_toc
    from translator.scraper.kakuyomu import fetch_work_chapters

    PATHS.ensure()
    entries = load_toc()
    chapters = fetch_work_chapters()

    table = Table(title="Entries needing JP mapping review")
    table.add_column("entry_id")
    table.add_column("EN ch")
    table.add_column("EN parts")
    table.add_column("current KU chapter")
    table.add_column("candidates (±3)")
    by_ch: dict[int, list] = {}
    for e in entries:
        if e.kind == "part" and e.mapping_confidence == "needs_review" and e.chapter_number:
            by_ch.setdefault(e.chapter_number, []).append(e)
    for en_ch in sorted(by_ch.keys()):
        ents = by_ch[en_ch]
        n_parts = len(ents)
        current_title = ents[0].chapter_title_jp or "?"
        current_pos = next(
            (i for i, c in enumerate(chapters) if c.id == ents[0].kakuyomu_chapter_id),
            None,
        )
        if current_pos is None:
            candidates = "n/a"
        else:
            lo, hi = max(0, current_pos - 3), min(len(chapters), current_pos + 4)
            candidates = " | ".join(
                f"[{i}] n={len(chapters[i].episode_ids)} {chapters[i].title}" for i in range(lo, hi)
            )
        ids = ", ".join(e.id for e in ents)
        table.add_row(ids, str(en_ch), str(n_parts), current_title, candidates)
    console.print(table)


@scrape_app.command("verify")
def scrape_verify() -> None:
    """Check that every ToC entry has both JP and EN clean text on disk."""
    from translator.scraper.align import verify_pairs

    PATHS.ensure()
    report = verify_pairs()

    table = Table(title="JP / EN parallel pair verification (numbered parts only)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("In-scope entries (numbered parts)", str(report.in_scope_entries))
    table.add_row("Out-of-scope (interludes / extras / SS / etc.)", str(report.out_of_scope_entries))
    table.add_row("JP files present", str(report.jp_present))
    table.add_row("EN files present", str(report.en_present))
    table.add_row("Complete pairs", str(report.complete_pairs))
    table.add_row("Missing JP (scrape failed)", str(len(report.missing_jp)))
    table.add_row("Missing JP (LN-exclusive part, expected)", str(len(report.missing_jp_ln_only)))
    table.add_row("Missing EN (scrape failed)", str(len(report.missing_en)))
    table.add_row("Missing EN (JP-only tail, expected)", str(len(report.missing_en_jp_only)))
    table.add_row("Paragraph-count skew >25%", str(len(report.paragraph_skew)))
    console.print(table)


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------

@prep_app.command("pov")
def prep_pov(
    pov: str | None = typer.Option(None, "--pov", help="Filter to a single POV."),
) -> None:
    """Print the POV breakdown of the corpus from data/metadata/toc.json."""
    from translator.prep import load_pov_lookup

    lookup = load_pov_lookup()
    parts = lookup.parts_only()
    counts: dict[str, int] = {}
    for entry in parts:
        counts[entry.pov] = counts.get(entry.pov, 0) + 1

    table = Table(title=f"POV distribution ({len(parts)} numbered parts)")
    table.add_column("POV")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    for k in sorted(counts):
        if pov and k != pov:
            continue
        share = counts[k] / max(len(parts), 1)
        table.add_row(k, str(counts[k]), f"{share:.0%}")
    console.print(table)


@prep_app.command("calibrate")
def prep_calibrate(
    margin: float = typer.Option(
        0.10, "--margin", help="Safety margin to widen the length-ratio band by."
    ),
) -> None:
    """Fit validator thresholds to the observed corpus distribution.

    Walks every translated pair, runs the three validators, and prints
    a distribution summary plus recommended values for the loose
    thresholds in `config.Thresholds`. Re-run after the corpus changes
    materially.
    """
    from translator.prep import calibrate

    report = calibrate()
    if report.pair_count == 0:
        console.print("[red]No translated pairs on disk — run `translator scrape jp/en` first.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Calibrated against {report.pair_count} JP-EN pairs.[/green]\n")

    lr = report.length_ratio
    table = Table(title="EN/JP visible-character ratio")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    for label, value in [
        ("n", lr.n),
        ("mean", f"{lr.mean:.2f}"),
        ("stdev", f"{lr.stdev:.2f}"),
        ("min", f"{lr.minimum:.2f}"),
        ("p05", f"{lr.p05:.2f}"),
        ("p25", f"{lr.p25:.2f}"),
        ("p50", f"{lr.p50:.2f}"),
        ("p75", f"{lr.p75:.2f}"),
        ("p95", f"{lr.p95:.2f}"),
        ("max", f"{lr.maximum:.2f}"),
    ]:
        table.add_row(label, str(value))
    console.print(table)

    lo, hi = report.length_ratio_recommendation(margin=margin)
    console.print(
        f"[cyan]Recommended[/cyan]  length_ratio_min = {lo:.2f}  "
        f"length_ratio_max = {hi:.2f}  "
        f"(p05/p95 ± {margin:.0%} margin)\n"
    )

    ds = report.dialogue_skew
    table = Table(title="Dialogue parity skew (|jp - en| / max)")
    table.add_column("statistic")
    table.add_column("value", justify="right")
    for label, value in [
        ("n", ds.n),
        ("mean", f"{ds.mean:.3f}"),
        ("p50", f"{ds.p50:.3f}"),
        ("p75", f"{ds.p75:.3f}"),
        ("p95", f"{ds.p95:.3f}"),
        ("max", f"{ds.maximum:.3f}"),
    ]:
        table.add_row(label, str(value))
    console.print(table)
    console.print(
        f"[cyan]Recommended[/cyan]  dialogue_parity_max_skew = "
        f"{report.dialogue_skew_recommendation():.3f}  (p95)\n"
    )

    if report.name_skew_per_character:
        table = Table(title="Name-frequency skew per character")
        for col in ("character", "n", "mean", "p50", "p95", "max"):
            table.add_column(col)
        for name, dist in report.name_skew_per_character.items():
            table.add_row(
                name, str(dist.n),
                f"{dist.mean:.2f}", f"{dist.p50:.2f}",
                f"{dist.p95:.2f}", f"{dist.maximum:.2f}",
            )
        console.print(table)
        console.print(
            "[dim]name_frequency thresholds in checks.py are intentionally loose "
            "(EN expands subject-elided JP). Tighten only if the p95 skew is "
            "well below the current 25% warn / 50% fail bands.[/dim]"
        )


@prep_app.command("detect-pov")
def prep_detect_pov(
    write: bool = typer.Option(
        True, "--write/--no-write", help="Persist resolved POVs to data/metadata/toc.json."
    ),
    only_needs_review: bool = typer.Option(
        True,
        "--only-needs-review/--all-parts",
        help="Default: only re-detect parts marked needs_review (i.e. JP-only tail). "
        "Use --all-parts to also re-detect EN-tagged parts.",
    ),
) -> None:
    """Detect POV from the JP narrator for parts whose POV is uncertain.

    Iterates ``data/parallel/*.jp.json`` for the targeted parts, counts
    character mentions in narration only, and rewrites ``toc.json``
    in place. The ``--all-parts`` flag is mainly useful as a sanity
    check against EN-derived POV tags.
    """
    import json as _json

    from translator.prep import detect_pov_from_disk
    from translator.prep import load_pov_lookup
    from translator.scraper.models import TocEntry

    lookup = load_pov_lookup()
    parts = lookup.parts_only(supported_pov_only=False)
    targets = [
        p for p in parts
        if (not only_needs_review) or p.mapping_confidence == "needs_review"
    ]
    if not targets:
        console.print("[yellow]No parts to detect (everything already settled).[/yellow]")
        return

    table = Table(title=f"POV detection ({len(targets)} parts)")
    table.add_column("part")
    table.add_column("仙台", justify="right")
    table.add_column("宮城", justify="right")
    table.add_column("from")
    table.add_column("to")
    table.add_column("confident")

    updated: dict[str, TocEntry] = {}
    skipped = 0
    for entry in targets:
        det = detect_pov_from_disk(entry.id)
        if det is None or det.pov is None:
            skipped += 1
            table.add_row(
                entry.id,
                str(det.sendai_count) if det else "-",
                str(det.miyagi_count) if det else "-",
                entry.pov, entry.pov, "[yellow]no signal[/yellow]"
            )
            continue
        new_entry = entry.model_copy(update={
            "pov": det.pov,
            "mapping_confidence": (
                "auto" if det.confident else entry.mapping_confidence
            ),
        })
        updated[entry.id] = new_entry
        confidence_marker = "[green]yes[/green]" if det.confident else "[yellow]weak[/yellow]"
        table.add_row(
            entry.id,
            str(det.sendai_count),
            str(det.miyagi_count),
            entry.pov, det.pov, confidence_marker,
        )
    console.print(table)
    console.print(
        f"[cyan]Resolved[/cyan]: {len(updated)}  [yellow]Skipped (no signal)[/yellow]: {skipped}"
    )

    if write and updated:
        raw = _json.loads(PATHS.toc_json.read_text(encoding="utf-8"))
        rows: list[dict] = []
        for row in raw:
            entry = TocEntry.model_validate(row)
            new = updated.get(entry.id, entry)
            rows.append(new.model_dump(mode="json"))
        PATHS.toc_json.write_text(
            _json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {PATHS.toc_json.relative_to(PATHS.repo_root)}")


@prep_app.command("holdout")
def prep_holdout(
    target: int = typer.Option(THRESHOLDS.test_holdout_target_count, "--target", help="Target holdout size."),
    seed: int = typer.Option(THRESHOLDS.random_seed, "--seed", help="RNG seed (deterministic across runs)."),
    write: bool = typer.Option(True, "--write/--no-write", help="Persist to data/metadata/holdout.json."),
) -> None:
    """Build a stratified ~30-part test holdout, deterministic across runs."""
    from translator.prep import build_holdout
    from translator.prep.holdout import write_holdout

    plan = build_holdout(target_count=target, seed=seed)
    table = Table(title=f"Stratified holdout (n={len(plan.part_ids)}, seed={plan.seed})")
    table.add_column("Bucket")
    table.add_column("Miyagi", justify="right")
    table.add_column("Sendai", justify="right")
    for bucket in ("early", "mid", "late"):
        cell = plan.per_bucket_counts.get(bucket, {})
        table.add_row(bucket, str(cell.get("miyagi", 0)), str(cell.get("sendai", 0)))
    console.print(table)
    console.print("[dim]Members:[/dim] " + ", ".join(plan.part_ids))

    if write:
        write_holdout(plan)
        console.print(
            f"[green]Wrote[/green] {PATHS.holdout_json.relative_to(PATHS.repo_root)}"
        )


# ---------------------------------------------------------------------------
# vault
# ---------------------------------------------------------------------------

@vault_app.command("init")
def vault_init(
    overwrite_templates: bool = typer.Option(
        False, "--overwrite-templates", help="Forcibly rewrite prompt + comparison templates."
    ),
) -> None:
    """Create knowledge-vault/ with the v3 directory layout and seed templates."""
    from translator.vault import init_vault

    root = init_vault(overwrite_templates=overwrite_templates)
    console.print(f"[green]Vault ready at[/green] {root}")


@vault_app.command("check")
def vault_check_cmd(
    part: str | None = typer.Option(None, "--part", help="Part id to dry-assemble a prompt for (defaults to first holdout member)."),
) -> None:
    """Offline pre-flight: verify vault, glossary, templates, and prompt assembly."""
    from translator.vault.check import run_vault_check

    sample = _normalize_part(part) if part else None
    report = run_vault_check(sample_part=sample)
    table = Table(title="knowledge-vault pre-flight (no API)")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for c in report.checks:
        color = "green" if c.ok else "red"
        table.add_row(c.name, f"[{color}]{'pass' if c.ok else 'FAIL'}[/{color}]", c.detail)
    console.print(table)
    if not report.passed:
        raise typer.Exit(1)


@vault_app.command("status")
def vault_status_cmd() -> None:
    """Summarize the current vault contents (rules, deviations, evaluations)."""
    from translator.vault import vault_status

    s = vault_status()
    table = Table(title="knowledge-vault status")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("root", str(s.root))
    table.add_row("initialized", str(s.initialized))
    table.add_row("active rules", str(s.active_rule_count))
    table.add_row("candidate rules", str(s.candidate_rule_count))
    table.add_row("pruned rules", str(s.pruned_rule_count))
    table.add_row("inactive rules", str(s.inactive_rule_count))
    table.add_row("deviation rounds", str(s.deviation_round_count))
    table.add_row("deviation notes", str(s.deviation_note_count))
    table.add_row("evaluation summaries", str(s.evaluation_round_count))
    table.add_row("glossary present", str(s.glossary_present))
    console.print(table)
    if not s.initialized:
        console.print("[yellow]Vault not initialized — run `translator vault init`.[/yellow]")


# ---------------------------------------------------------------------------
# glossary
# ---------------------------------------------------------------------------

@glossary_app.command("show")
def glossary_show() -> None:
    """Print the active glossary (vault if initialized, else seed)."""
    from translator.glossary import format_for_prompt, load_glossary

    entries = load_glossary()
    if not entries:
        console.print("[yellow]No glossary entries found.[/yellow]")
        raise typer.Exit(0)
    console.print(format_for_prompt(entries))
    console.print(f"\n[dim]{len(entries)} entries.[/dim]")


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

@style_app.command("extract")
def style_extract(
    through: str = typer.Option(
        "part_050",
        "--through",
        help="Highest part_id to include in the corpus (e.g. part_050 or just 50).",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override MODELS.style_extraction."
    ),
    include_holdout: bool = typer.Option(
        False,
        "--include-holdout",
        help="Include holdout members in the extraction corpus (off by default to keep test set clean).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print corpus stats and prompt size; do not call the model.",
    ),
    write: bool = typer.Option(
        True,
        "--write/--no-write",
        help="Write per-dimension files under knowledge-vault/style/.",
    ),
) -> None:
    """Extract a 16-dimension writing-style profile from the EN reference corpus.

    One-shot bootstrap: reads parts 1..N (minus holdout by default), asks
    the extraction model to characterize the prose along 16 dimensions,
    and writes one file per dimension under knowledge-vault/style/.
    Subsequent translation, deviation, and judge calls concatenate the
    bodies and inject them via $style_profile.
    """
    from string import Template

    from translator.config import MODELS, REASONING
    from translator.style import write_style_dimensions
    from translator.style.extract import (
        EXTRACTION_PROMPT,
        _format_corpus,
        _gather_corpus,
        extract_style_profile,
    )

    target_part_id = _normalize_part(through)
    corpus, skipped_holdout = _gather_corpus(
        through_part_id=target_part_id,
        exclude_holdout=not include_holdout,
    )
    if not corpus:
        console.print(
            f"[red]No EN-translated parts found through {target_part_id}.[/red]"
        )
        raise typer.Exit(1)

    table = Table(title="Style extraction plan")
    table.add_column("field")
    table.add_column("value")
    table.add_row("through", target_part_id)
    table.add_row("chapters in corpus", str(len(corpus)))
    table.add_row("holdout members skipped", str(skipped_holdout))
    pov_counts: dict[str, int] = {}
    for _, pov, _ in corpus:
        pov_counts[pov] = pov_counts.get(pov, 0) + 1
    table.add_row("by POV", ", ".join(f"{p}={n}" for p, n in sorted(pov_counts.items())))
    chosen_model = model or MODELS.style_extraction
    table.add_row("model", chosen_model)
    table.add_row("reasoning effort", REASONING.style_extraction)
    rendered_prompt = Template(EXTRACTION_PROMPT).safe_substitute(
        corpus=_format_corpus(corpus)
    )
    table.add_row("prompt size", f"{len(rendered_prompt):,} chars")
    console.print(table)

    if dry_run:
        console.print("[dim]dry-run: skipping model call[/dim]")
        return

    console.print("[dim]calling model — this can take several minutes for high-effort reasoning…[/dim]")
    result = extract_style_profile(
        through_part_id=target_part_id,
        exclude_holdout=not include_holdout,
        model=model,
    )

    if not write:
        console.print(
            f"[dim]--no-write set; parsed {len(result.dimensions)} dimensions[/dim]\n"
        )
        console.print(result.body)
        return

    paths = write_style_dimensions(
        result.dimensions,
        extracted_through=result.extracted_through,
        n_chapters=result.n_chapters,
        model=result.model,
        extracted_at=result.extracted_at,
    )
    console.print(
        f"[green]Wrote {len(paths)} dimension files[/green] "
        f"({result.n_chapters} chapters, {len(result.body):,} chars total)"
    )
    for p in paths:
        console.print(f"  • {p.relative_to(PATHS.repo_root)}")


@style_app.command("show")
def style_show(
    full: bool = typer.Option(
        False, "--full", help="Print the concatenated profile body instead of a summary."
    ),
    dimension: int | None = typer.Option(
        None,
        "--dimension",
        "-d",
        help="Print only the body of the given dimension number (1–16).",
    ),
) -> None:
    """Display the current style profile and its provenance."""
    from translator.style import DIMENSIONS, load_style_profile

    profile = load_style_profile()
    table = Table(title="Style profile")
    table.add_column("field")
    table.add_column("value")
    table.add_row(
        "directory",
        str(profile.style_dir) if profile.style_dir else "(none)",
    )
    table.add_row("has content", str(profile.has_content))
    table.add_row("dimensions on disk", f"{profile.n_dimensions}/{len(DIMENSIONS)}")
    table.add_row("extracted_through", profile.extracted_through or "—")
    table.add_row("n_chapters", str(profile.n_chapters))
    table.add_row("extracted_at", profile.extracted_at or "—")
    table.add_row("model", profile.model or "—")
    table.add_row("version", str(profile.version))
    console.print(table)

    if profile.dimensions:
        per_dim = Table(title="Dimensions on disk")
        per_dim.add_column("#", justify="right")
        per_dim.add_column("name")
        per_dim.add_column("file")
        per_dim.add_column("body chars", justify="right")
        for d in sorted(profile.dimensions, key=lambda x: x.number):
            per_dim.add_row(
                str(d.number),
                d.name,
                d.filename,
                f"{len(d.body):,}",
            )
        console.print(per_dim)

    if dimension is not None:
        match = next(
            (d for d in profile.dimensions if d.number == dimension), None
        )
        if match is None:
            console.print(
                f"[yellow]Dimension {dimension} not on disk.[/yellow]"
            )
            return
        console.print("\n---\n")
        console.print(match.body)
        return

    if full and profile.has_content:
        console.print("\n---\n")
        console.print(profile.render_body())
        return

    if not profile.has_content:
        console.print(
            "[yellow]No dimensions extracted yet — run `translator style extract`.[/yellow]"
        )


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

@app.command("translate")
def translate(
    part: str = typer.Option(..., "--part", help="Part id (e.g. part_004) or part number."),
    model: str | None = typer.Option(None, "--model", help="Override MODELS.translation."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip the network call; only assemble the prompt."),
    no_holdout_skip: bool = typer.Option(False, "--no-holdout-skip", help="Don't skip holdout members in window."),
    revise: bool = typer.Option(
        False,
        "--revise/--no-revise",
        help=(
            "Run the inline critic + revision loop after the first-pass "
            "translation. Default is off: with precedents at k=150 the "
            "first-pass draft already adopts the vast majority of the "
            "critic's suggested fixes, and the revision round costs an "
            "extra ~$4 + ~6 min per chapter for marginal polish."
        ),
    ),
    critic_model: str | None = typer.Option(
        None, "--critic-model", help="Override MODELS.critic."
    ),
    max_revisions: int | None = typer.Option(
        None,
        "--max-revisions",
        help="Cap revision passes (default THRESHOLDS.critique_max_revisions). 0 = audit only.",
    ),
    revise_severity: str | None = typer.Option(
        None,
        "--revise-severity",
        help="Severity gate: 'major' (default) or 'minor' (revise on any flag).",
    ),
    revise_minor_threshold: int | None = typer.Option(
        None,
        "--revise-minor-threshold",
        help="Minor flag count that triggers revision when no major flags fire.",
    ),
    use_precedents: bool = typer.Option(
        True,
        "--precedents/--no-precedents",
        help=(
            "Inject paragraph-level reference precedents from the RAG index "
            "into the translate / revise / critique prompts. Disable for "
            "ablations or when the index hasn't been built yet."
        ),
    ),
    precedent_k: int | None = typer.Option(
        None,
        "--precedent-k",
        help=(
            "Override THRESHOLDS.precedents_per_chapter for this run "
            "(e.g. --precedent-k 50 to test a richer slate). Default "
            "uses the configured value (25)."
        ),
    ),
    output_suffix: str | None = typer.Option(
        None,
        "--output-suffix",
        help=(
            "Append a suffix to the output directory name (e.g. 'k50') "
            "so ablation runs don't clobber each other."
        ),
    ),
) -> None:
    """Translate one part end-to-end with the inline critic + revise loop.

    The critic always runs (cheap audit); revision is gated by severity
    so chapters that come out clean don't pay the extra translator
    call. Use --no-revise to skip the loop entirely (cheap baseline)
    or --max-revisions 0 for audit-only mode (run the critic but never
    revise).
    """
    from translator.inference import UnsupportedTargetError, translate_part

    # Apply --precedent-k as a runtime override on the shared
    # THRESHOLDS dataclass for the duration of this call. The
    # frozen=False dataclass allows mutation; we restore on exit so
    # subsequent CLI invocations in the same process see the default.
    original_k = THRESHOLDS.precedents_per_chapter
    if precedent_k is not None:
        object.__setattr__(THRESHOLDS, "precedents_per_chapter", precedent_k)
        console.print(
            f"[dim]ablation:[/dim] precedents_per_chapter={precedent_k} "
            f"(default {original_k})"
        )

    part_id = _normalize_part(part)
    holdout_arg: object = ... if not no_holdout_skip else None
    try:
        result = translate_part(
            part_id,
            model=model,
            dry_run=dry_run,
            holdout=holdout_arg,
            revise=revise,
            critic_model=critic_model,
            max_revisions=max_revisions,
            revise_severity=revise_severity,  # type: ignore[arg-type]
            revise_minor_threshold=revise_minor_threshold,
            use_precedents=use_precedents,
            output_suffix=output_suffix,
        )
    except UnsupportedTargetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    finally:
        if precedent_k is not None:
            object.__setattr__(
                THRESHOLDS, "precedents_per_chapter", original_k
            )

    table = Table(title=f"Translation: {part_id}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("model", result.model)
    table.add_row("dry_run", str(result.dry_run))
    table.add_row("template_source", result.prompt.template_source)
    table.add_row("window parts", ", ".join(result.prompt.window_part_ids) or "(none)")
    table.add_row("active rules", ", ".join(result.prompt.active_rule_ids) or "(none)")
    if result.prompt.precedents is not None:
        table.add_row(
            "precedents",
            f"{len(result.prompt.precedents.paragraphs)} paragraph pair(s)",
        )
    elif use_precedents:
        table.add_row("precedents", "(no index built — skipped)")
    else:
        table.add_row("precedents", "(disabled via --no-precedents)")
    if result.critiques:
        table.add_row(
            "revision passes",
            f"{result.revision_count} (max {max_revisions if max_revisions is not None else '—'})",
        )
        for i, c in enumerate(result.critiques):
            label = "draft_v1" if i == 0 else f"after revision #{i}"
            table.add_row(
                f"critic [{label}]",
                f"{len(c.major_flags)} major + {len(c.minor_flags)} minor",
            )
    if result.output_path:
        table.add_row("output", str(result.output_path.relative_to(PATHS.repo_root)))
    console.print(table)
    for note in result.prompt.notes + result.notes:
        console.print(f"[dim]note:[/dim] {note}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@app.command("validate")
def validate(
    part: str = typer.Option(..., "--part", help="Part id or number."),
    en_path: str | None = typer.Option(None, "--en", help="Override EN file path (defaults to data/output/{part}/translation.en.txt)."),
) -> None:
    """Run dialogue/name/length validators on a translation."""
    from pathlib import Path

    from translator.prep.corpus import load_part_jp
    from translator.validation import validate_translation

    part_id = _normalize_part(part)
    jp = load_part_jp(part_id)
    if en_path:
        en = Path(en_path).read_text(encoding="utf-8")
    else:
        candidate = PATHS.output / part_id / "translation.en.txt"
        if not candidate.exists():
            console.print(f"[red]No translation found at {candidate}[/red]")
            raise typer.Exit(1)
        en = candidate.read_text(encoding="utf-8")

    report = validate_translation(part_id, jp, en)
    table = Table(title=f"Validation: {part_id} ({report.status})")
    table.add_column("check")
    table.add_column("status")
    table.add_column("message")
    for c in report.checks:
        color = {"pass": "green", "warn": "yellow", "fail": "red"}[c.status]
        table.add_row(c.name, f"[{color}]{c.status}[/{color}]", c.message)
    console.print(table)
    raise typer.Exit(0 if report.passed else 1)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@evaluate_app.command("deviations")
def evaluate_deviations(
    round_number: int = typer.Option(..., "--round", help="Round number (used in note frontmatter and folder name)."),
    parts: str = typer.Option(..., "--parts", help="Comma-separated part ids (e.g. part_010,part_011) or 'holdout'."),
    model: str | None = typer.Option(None, "--model", help="Override MODELS.comparison."),
) -> None:
    """Extract deviations for a batch and write per-chapter notes to the vault."""
    from translator.eval import extract_deviations
    from translator.prep.corpus import load_part_en, load_part_jp
    from translator.prep.holdout import load_holdout

    if parts.strip() == "holdout":
        plan = load_holdout()
        if plan is None:
            console.print("[red]No holdout on disk; run `translator prep holdout` first.[/red]")
            raise typer.Exit(1)
        part_ids = list(plan.part_ids)
    else:
        part_ids = [p.strip() for p in parts.split(",") if p.strip()]

    for pid in part_ids:
        out_path = PATHS.output / pid / "translation.en.txt"
        if not out_path.exists():
            console.print(f"[yellow]skip {pid}: no translation at {out_path}[/yellow]")
            continue
        reference = load_part_en(pid)
        if reference is None:
            console.print(f"[yellow]skip {pid}: no reference translation[/yellow]")
            continue
        note = extract_deviations(
            part_id=pid,
            round_number=round_number,
            jp=load_part_jp(pid),
            llm_translation=out_path.read_text(encoding="utf-8"),
            reference=reference,
            model=model,
        )
        console.print(f"[green]{pid}[/green]: {len(note.deviations)} deviations recorded")


@evaluate_app.command("critique")
def evaluate_critique(
    parts: str = typer.Option(..., "--parts", help="Comma-separated part ids (or 'holdout')."),
    model: str | None = typer.Option(None, "--model", help="Override MODELS.critic."),
    show_flags: bool = typer.Option(False, "--show-flags", help="Print each flag's span + suggested rewrite."),
    use_precedents: bool = typer.Option(
        True,
        "--precedents/--no-precedents",
        help="Inject paragraph-level reference precedents into the critique prompt.",
    ),
) -> None:
    """Run the inline critic on existing drafts without re-translating.

    Reads each part's ``data/output/<part>/translation.en.txt`` and
    persists ``critique.json`` next to it. Useful for inspecting the
    critic's calibration before turning on the full revision loop.
    """
    from translator.eval import critique_translation
    from translator.prep.corpus import load_part_jp
    from translator.prep.holdout import load_holdout

    if parts.strip() == "holdout":
        plan = load_holdout()
        if plan is None:
            console.print("[red]No holdout on disk; run `translator prep holdout` first.[/red]")
            raise typer.Exit(1)
        part_ids = list(plan.part_ids)
    else:
        part_ids = [p.strip() for p in parts.split(",") if p.strip()]

    for pid in part_ids:
        draft_path = PATHS.output / pid / "translation.en.txt"
        if not draft_path.exists():
            console.print(f"[yellow]skip {pid}: no draft at {draft_path}[/yellow]")
            continue
        result = critique_translation(
            part_id=pid,
            jp=load_part_jp(pid),
            draft=draft_path.read_text(encoding="utf-8"),
            model=model,
            use_precedents=use_precedents,
        )
        result.write()
        console.print(
            f"[green]{pid}[/green]: {len(result.major_flags)} major + "
            f"{len(result.minor_flags)} minor flag(s) "
            f"[dim](critique.json written)[/dim]"
        )
        if show_flags:
            for i, f in enumerate(result.flags, 1):
                console.print(
                    f"  [bold]{i}.[/bold] [{f.severity}/{f.category}] "
                    f"{f.span!r} → {f.suggested_rewrite!r}"
                )
                if f.notes:
                    console.print(f"     [dim]{f.notes}[/dim]")


@evaluate_app.command("cluster")
def evaluate_cluster(
    rounds: str = typer.Option(..., "--rounds", help="Comma-separated round numbers to read deviations from."),
    promote_round: int = typer.Option(..., "--promote-round", help="Round number stamped on the new candidate rules."),
    model: str | None = typer.Option(None, "--model", help="Override MODELS.clustering."),
) -> None:
    """Cluster deviation notes into candidate rules and write them to the vault."""
    from translator.eval import cluster_into_candidate_rules

    round_numbers = [int(r) for r in rounds.split(",") if r.strip()]
    rules = cluster_into_candidate_rules(
        round_numbers=round_numbers,
        promote_round=promote_round,
        model=model,
    )
    console.print(f"[green]Wrote {len(rules)} candidate rules.[/green]")


@evaluate_app.command("score")
def evaluate_score(
    parts: str = typer.Option(..., "--parts", help="Comma-separated part ids or 'holdout'."),
    skip_comet: bool = typer.Option(False, "--skip-comet", help="Skip COMET (still slow even on CPU)."),
    skip_bertscore: bool = typer.Option(False, "--skip-bertscore", help="Skip BERTScore."),
) -> None:
    """COMET + BERTScore on (LLM, reference) pairs. Requires --extra ml."""
    from translator.eval import bertscore, comet_score
    from translator.prep.corpus import load_part_en, load_part_jp
    from translator.prep.holdout import load_holdout

    if parts.strip() == "holdout":
        plan = load_holdout()
        if plan is None:
            console.print("[red]No holdout on disk; run `translator prep holdout` first.[/red]")
            raise typer.Exit(1)
        part_ids = list(plan.part_ids)
    else:
        part_ids = [p.strip() for p in parts.split(",") if p.strip()]

    sources, hyps, refs = [], [], []
    for pid in part_ids:
        out_path = PATHS.output / pid / "translation.en.txt"
        if not out_path.exists():
            console.print(f"[yellow]skip {pid}: no translation[/yellow]")
            continue
        ref = load_part_en(pid)
        if ref is None:
            console.print(f"[yellow]skip {pid}: no reference[/yellow]")
            continue
        sources.append(load_part_jp(pid))
        hyps.append(out_path.read_text(encoding="utf-8"))
        refs.append(ref)
    if not hyps:
        console.print("[red]Nothing to score.[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Scores ({len(hyps)} chapters)")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    if not skip_comet:
        c = comet_score(sources=sources, hypotheses=hyps, references=refs)
        table.add_row("COMET", f"{c.mean:.4f}")
    if not skip_bertscore:
        b = bertscore(hypotheses=hyps, references=refs)
        table.add_row("BERTScore F1", f"{b.mean:.4f}")
    console.print(table)


@evaluate_app.command("judge")
def evaluate_judge(
    parts: str = typer.Option(..., "--parts", help="Comma-separated part ids or 'holdout'."),
    model: str | None = typer.Option(None, "--model", help="Override MODELS.judge."),
    show_rationales: bool = typer.Option(
        True,
        "--rationales/--no-rationales",
        help="Print per-axis rationales under the score table.",
    ),
) -> None:
    """LLM-as-judge rubric scoring per chapter.

    Writes ``data/output/<part>/judge.json`` for each chapter so the
    rationales and raw judge payload are inspectable after the run; the
    in-memory ``JudgeResult`` is otherwise discarded once the table prints.
    """
    from datetime import UTC, datetime

    from translator.config import MODELS
    from translator.eval import judge_translation
    from translator.prep.corpus import load_part_en, load_part_jp
    from translator.prep.holdout import load_holdout
    from translator.prep.pov import load_pov_lookup

    if parts.strip() == "holdout":
        plan = load_holdout()
        if plan is None:
            console.print("[red]No holdout on disk; run `translator prep holdout` first.[/red]")
            raise typer.Exit(1)
        part_ids = list(plan.part_ids)
    else:
        part_ids = [p.strip() for p in parts.split(",") if p.strip()]

    lookup = load_pov_lookup()
    table = Table(title="Judge ratings")
    for col in ("part", "POV", "sem", "voice", "natural", "style", "mean"):
        table.add_column(col)

    judged: list[tuple[str, object]] = []
    for pid in part_ids:
        out_path = PATHS.output / pid / "translation.en.txt"
        if not out_path.exists():
            continue
        ref = load_part_en(pid)
        if ref is None:
            continue
        result = judge_translation(
            pov=lookup.pov(pid),
            jp=load_part_jp(pid),
            candidate=out_path.read_text(encoding="utf-8"),
            reference=ref,
            model=model,
        )
        table.add_row(
            pid, lookup.pov(pid),
            str(result.semantic_accuracy), str(result.voice_fidelity),
            str(result.naturalness), str(result.style_match), f"{result.mean:.2f}",
        )

        judge_path = PATHS.output / pid / "judge.json"
        payload = {
            "part_id": pid,
            "pov": lookup.pov(pid),
            "model": model or MODELS.judge,
            "judged_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "scores": {
                "semantic_accuracy": result.semantic_accuracy,
                "voice_fidelity": result.voice_fidelity,
                "naturalness": result.naturalness,
                "style_match": result.style_match,
                "mean": result.mean,
            },
            "rationales": result.rationales,
            "raw": result.raw,
        }
        judge_path.parent.mkdir(parents=True, exist_ok=True)
        judge_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        judged.append((pid, result))

    console.print(table)

    if judged:
        console.print(f"[green]Wrote[/green] judge.json for {len(judged)} part(s)")
    if show_rationales:
        for pid, result in judged:
            console.print(f"\n[bold cyan]{pid}[/bold cyan] rationales")
            for axis in ("semantic_accuracy", "voice_fidelity", "naturalness", "style_match"):
                score = getattr(result, axis)
                rationale = result.rationales.get(axis, "")
                console.print(f"  [bold]{axis}[/bold] [{score}]: {rationale}")


@evaluate_app.command("report")
def evaluate_report(
    round_number: int = typer.Option(..., "--round", help="Round to summarize."),
    notes: str = typer.Option("", "--notes", help="Free-form notes appended to the summary body."),
) -> None:
    """Write a knowledge-vault evaluation summary for the given round."""
    from translator.eval import RoundSummary, write_round_summary
    from translator.vault import vault_status

    s = vault_status()
    summary = RoundSummary(
        round_number=round_number,
        chapters_evaluated=0,
        rules_active_total=s.active_rule_count,
        notes=notes,
    )
    path = write_round_summary(summary)
    console.print(f"[green]Wrote[/green] {path}")
    console.print(
        "[dim]Fill in metrics by editing the note in Obsidian, "
        "or extend `evaluate report` to populate them automatically.[/dim]"
    )


# ---------------------------------------------------------------------------
# precedents (RAG index)
# ---------------------------------------------------------------------------

@precedents_app.command("build")
def precedents_build(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Discard the existing index and rebuild from scratch (else incremental).",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated part ids to index (default: every translated part).",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override MODELS.embedding (used only for the JP side)."
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Print one line per chapter as the build proceeds.",
    ),
    workers: int = typer.Option(
        16,
        "--workers",
        help="Concurrent chapter workers (default 16).",
    ),
) -> None:
    """Build (or extend) the precedent RAG index from the parallel corpus.

    Reads every translated, supported-POV part by default; runs
    length-only DP paragraph alignment (no EN embedding); embeds the
    JP side with text-embedding-3-small; writes the result to
    ``data/precedent_index/`` for retrieval at translation time.

    The build is incremental: re-running adds any newly-translated
    chapters. Pass ``--rebuild`` to wipe the index and start over (use
    after changing the embedding model or the alignment constants).
    """
    from translator.precedents import build_index

    only_ids = (
        [_normalize_part(p) for p in only.split(",") if p.strip()] if only else None
    )

    stats = build_index(
        parts=only_ids,
        rebuild=rebuild,
        model=model,
        progress=progress,
        workers=workers,
    )

    table = Table(title="Precedent index build")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("model", stats.model)
    table.add_row("dim", str(stats.dim))
    table.add_row("parts indexed (total)", str(len(stats.parts_indexed)))
    table.add_row("parts skipped this run", str(len(stats.parts_skipped)))
    for shape, count in sorted(stats.shape_counts.items()):
        table.add_row(f"shape {shape}", str(count))
    table.add_row("total entries", str(stats.total_count))
    console.print(table)
    if stats.parts_skipped:
        console.print(
            f"[dim]skipped:[/dim] {', '.join(stats.parts_skipped[:10])}"
            + (" …" if len(stats.parts_skipped) > 10 else "")
        )


@precedents_app.command("query")
def precedents_query(
    part: str = typer.Option(..., "--part", help="Target part id or number."),
    paragraph_k: int | None = typer.Option(
        None,
        "--paragraph-k",
        help="Per-chapter cap on paragraph-level precedents (default config value).",
    ),
    exclude: str | None = typer.Option(
        None,
        "--exclude",
        help="Comma-separated additional part ids to exclude (target is always excluded).",
    ),
) -> None:
    """Debug retrieval: print what precedents would be injected for ``--part``.

    Useful before/after a corpus change to confirm the new chapter pulls
    sensible precedents.
    """
    from translator.precedents import format_precedents_for_prompt, retrieve_for_part

    part_id = _normalize_part(part)
    excluded = (
        [_normalize_part(p) for p in exclude.split(",") if p.strip()] if exclude else None
    )
    result = retrieve_for_part(
        part_id,
        paragraph_k=paragraph_k,
        exclude=excluded,
    )

    table = Table(title=f"Precedents for {part_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("paragraph precedents", str(len(result.paragraphs)))
    table.add_row("total", str(result.total))
    console.print(table)
    for note in result.notes:
        console.print(f"[dim]note:[/dim] {note}")
    body = format_precedents_for_prompt(result)
    console.print("\n" + body)


@precedents_app.command("stats")
def precedents_stats() -> None:
    """Print the current precedent index size, model, and provenance."""
    from translator.precedents import index_exists, load_meta

    if not index_exists():
        console.print(
            "[yellow]No precedent index found. Run "
            "`translator precedents build` to populate it.[/yellow]"
        )
        raise typer.Exit(1)
    meta = load_meta()
    table = Table(title="Precedent index")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("model", str(meta.get("model", "?")))
    table.add_row("dim", str(meta.get("dim", "?")))
    for shape, count in sorted(meta.get("shape_counts", {}).items()):
        table.add_row(f"shape {shape}", str(count))
    table.add_row("parts indexed", str(len(meta.get("parts_indexed", []))))
    table.add_row("length ratio", f"{meta.get('length_ratio', 0):.3f}")
    table.add_row("length variance", f"{meta.get('length_var', 0):.2f}")
    table.add_row("last updated", str(meta.get("last_updated", "?")))
    if meta.get("validated_at"):
        table.add_row("validated at", str(meta["validated_at"]))
    console.print(table)


@precedents_app.command("validate")
def precedents_validate(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Recompute semantic_score for every pair (else incremental).",
    ),
    workers: int = typer.Option(
        8,
        "--workers",
        help="Concurrent embedding workers (default 8).",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Print one line per batch as the pass proceeds.",
    ),
    show_suspects: bool = typer.Option(
        True,
        "--suspects/--no-suspects",
        help=(
            "Print the 5 worst suspect pairs (high length_score but low "
            "semantic_score — typical length-DP false positives)."
        ),
    ),
) -> None:
    """Compute cross-lingual semantic_score for every indexed pair.

    Length-only DP alignment can lock onto an adjacent paragraph with
    a near-identical character count but different content. This
    pass embeds the EN side of each pair, computes cosine similarity
    against the stored JP embedding, and writes the result back into
    ``pairs.jsonl`` as a ``semantic_score`` field. EN embeddings are
    discarded after the pass — the retrieval hot path stays JP-only.

    Calibration uses chapters where JP and EN paragraph counts match
    exactly (alignment-correct by construction). The 5th percentile
    of their semantic-score distribution is the recommended filter
    threshold for the rest of the corpus.
    """
    from translator.precedents import validate_index

    stats = validate_index(rebuild=rebuild, workers=workers, progress=progress)
    table = Table(title="Precedent index validation")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("pairs scored this run", str(stats.pairs_validated))
    if stats.pairs_skipped:
        table.add_row("pairs already validated", str(stats.pairs_skipped))
    table.add_row("semantic_score median (corpus)", f"{stats.semantic_score_median:.3f}")
    table.add_row("semantic_score mean (corpus)", f"{stats.semantic_score_mean:.3f}")
    table.add_row("p25 / p10 / p05 (corpus)",
        f"{stats.semantic_score_p25:.3f} / {stats.semantic_score_p10:.3f} / {stats.semantic_score_p05:.3f}")
    table.add_row("calibration pairs (1:1 from exact-match chapters)", str(stats.calibration_pairs))
    if stats.calibration_pairs:
        table.add_row("calibration median", f"{stats.calibration_median:.3f}")
        table.add_row("calibration mean", f"{stats.calibration_mean:.3f}")
        table.add_row("calibration p05 (suggested threshold)", f"{stats.suggested_threshold:.3f}")
    table.add_row("suspect pairs (length≥0.3 but semantic<p05)", str(stats.suspect_count))
    console.print(table)
    if show_suspects and stats.suspect_examples:
        console.print("\n[bold]5 worst suspects:[/bold]")
        for ex in stats.suspect_examples:
            console.print(
                f"  [{ex['part_id']}, {ex['shape']}] "
                f"length={ex['length_score']:.2f} semantic={ex['semantic_score']:.2f}"
            )
            console.print(f"    JP: {ex['jp_text']}")
            console.print(f"    EN: {ex['en_text']}")


# ---------------------------------------------------------------------------
# info / helpers
# ---------------------------------------------------------------------------

@app.command("info")
def info() -> None:
    """Print resolved paths, model IDs, and source URLs."""
    table = Table(title="ln-translator config")
    table.add_column("key")
    table.add_column("value")
    table.add_row("repo_root", str(PATHS.repo_root))
    table.add_row("data", str(PATHS.data))
    table.add_row("knowledge_vault", str(PATHS.knowledge_vault))
    table.add_row("kakuyomu_work", f"{SCRAPER.kakuyomu_base}/works/{SCRAPER.kakuyomu_work_id}")
    table.add_row("avelilium_toc", SCRAPER.avelilium_toc_url)
    table.add_row("MODELS.translation", MODELS.translation)
    table.add_row("MODELS.comparison", MODELS.comparison)
    table.add_row("MODELS.clustering", MODELS.clustering)
    table.add_row("MODELS.judge", MODELS.judge)
    table.add_row("THRESHOLDS.window_size", str(THRESHOLDS.window_size))
    console.print(table)


def _normalize_part(value: str) -> str:
    """Accept ``part_004`` / ``4`` shorthand, and pass other ids through.

    Numeric shorthand is convenient for the common case (numbered parts);
    any other string is forwarded as-is so the downstream guard can
    produce a clear "kind/POV not supported" error rather than a generic
    parse failure.
    """
    value = value.strip()
    try:
        n = int(value)
    except ValueError:
        return value
    return f"part_{n:03d}"


def _print_kv_table(stats: dict, title: str) -> None:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
