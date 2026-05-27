"""Central configuration: paths, model IDs, tunable thresholds.

All paths are derived from the repo root so the CLI works from any cwd.
Anything tunable lives here so we don't hunt through modules to change it.

The v3 architecture (sliding window + self-improving rules vault) treats the
on-disk knowledge vault as a peer of ``data/``: ``data/`` holds raw and
intermediate artifacts (scrapes, parallel text, output, indices); the vault
holds the deviation/rule/glossary notes that feed the translation prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
VAULT_DIR = REPO_ROOT / "knowledge-vault"


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. ``ensure()`` creates only the directories that
    the scraper and downstream code unconditionally need; the knowledge-vault
    tree is created lazily by ``vault init``."""

    repo_root: Path = REPO_ROOT
    data: Path = DATA_DIR
    raw_jp: Path = DATA_DIR / "raw" / "jp"
    raw_en: Path = DATA_DIR / "raw" / "en"
    raw_toc: Path = DATA_DIR / "raw" / "toc"
    parallel: Path = DATA_DIR / "parallel"
    metadata: Path = DATA_DIR / "metadata"
    toc_json: Path = DATA_DIR / "metadata" / "toc.json"
    pov_overrides: Path = DATA_DIR / "metadata" / "pov_overrides.json"
    glossary_seed: Path = DATA_DIR / "metadata" / "glossary_seed.json"
    holdout_json: Path = DATA_DIR / "metadata" / "holdout.json"
    output: Path = DATA_DIR / "output"
    # Precedent RAG index: JP↔EN paragraph pairs aligned by length-only
    # DP plus the JP-side embeddings. Built once via ``translator
    # precedents build`` and validated via ``translator precedents
    # validate``; reused by every translation/revise/critique call.
    # Lives inside the knowledge-vault because it's derived knowledge
    # over the corpus (alongside rules, glossary, style profiles)
    # rather than raw data.
    precedent_index: Path = VAULT_DIR / "precedent-index"

    knowledge_vault: Path = VAULT_DIR

    def ensure(self) -> None:
        for p in (
            self.data,
            self.raw_jp,
            self.raw_en,
            self.raw_toc,
            self.parallel,
            self.metadata,
            self.output,
        ):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass(frozen=True)
class VaultLayout:
    """Subdirectory layout inside the knowledge-vault. v3 treats this as the
    canonical knowledge store; every path here is relative to ``PATHS.knowledge_vault``.

    Mirrors the structure documented in the v3 handoff under "Obsidian vault structure".
    """

    deviations: str = "deviations"
    rules: str = "rules"
    rules_active: str = "rules/active"
    rules_candidate: str = "rules/candidate"
    rules_pruned: str = "rules/pruned"
    rules_inactive: str = "rules/inactive"
    evaluations: str = "evaluations"
    glossary: str = "glossary"
    glossary_file: str = "glossary/glossary.md"
    # Style profile is split across one Markdown file per dimension
    # under ``style/`` (01-tone.md, 02-voice.md, …). The
    # ``style_readme`` is a hand-edited overview in the same dir.
    style: str = "style"
    style_readme: str = "style/README.md"
    config: str = "config"
    prompt_template: str = "config/prompt-template.md"
    comparison_template: str = "config/comparison-prompt-template.md"
    clustering_template: str = "config/clustering-prompt-template.md"
    # Inference-time critic looks at (jp, draft) without a reference and
    # surfaces translationese / voice / style-profile drift. The output
    # feeds the revision pass below.
    critique_template: str = "config/critique-prompt-template.md"
    # Revision pass is a sister of the translation prompt — same
    # rules/glossary/style/window — plus the previous draft and the
    # critic's flags. Lives next to ``prompt_template`` so user edits
    # to either evolve together.
    revise_template: str = "config/revise-prompt-template.md"


VAULT = VaultLayout()


@dataclass(frozen=True)
class Models:
    """Model IDs. v3 splits the work across multiple models so self-evaluation
    isn't biased by the translator's own blind spots:

    * ``translation`` writes the new chapter — Anthropic's flagship for the
      best literary voice (1M context fits the full sliding window cleanly).
    * ``comparison`` extracts deviations between the LLM output and the human
      reference. **Must be a different family from ``translation``** to avoid
      the self-graded blind-spot problem.
    * ``clustering`` groups deviations into candidate rules. Either family
      is fine; this step operates on extracted deviation notes, not raw text.
    * ``judge`` produces the LLM-as-judge ratings during eval rounds.

    Defaults track the latest generally-available frontier models:
    ``claude-opus-4-7`` (Anthropic) and ``gpt-5.5`` (OpenAI). The OpenAI
    side runs through the Responses API so per-call ``reasoning.effort``
    is settable; per-role defaults live in ``ReasoningEffort`` below.
    Override any of these via ``.env``.
    """

    translation: str = os.getenv("MODEL_TRANSLATION", "claude-opus-4-7")
    comparison: str = os.getenv("MODEL_COMPARISON", "gpt-5.5")
    clustering: str = os.getenv("MODEL_CLUSTERING", "gpt-5.5")
    judge: str = os.getenv("MODEL_JUDGE", "gpt-5.5")
    # Style extraction is a one-shot characterization over a large EN
    # corpus. gpt-5.5 with high reasoning is overkill for most chapters
    # but justified here: the output is consumed by every subsequent
    # translation, so we want the deepest analysis possible.
    style_extraction: str = os.getenv("MODEL_STYLE_EXTRACTION", "gpt-5.5")
    # Inline critic that audits the draft for translationese / voice /
    # style-profile drift at inference time. Defaults to the same model
    # as the offline deviation auditor so calibration carries over and
    # the translator/critic blind spots stay diverse (Anthropic vs.
    # OpenAI).
    critic: str = os.getenv("MODEL_CRITIC", os.getenv("MODEL_COMPARISON", "gpt-5.5"))
    # Cross-lingual embedding model used to align JP↔EN segments and
    # to retrieve precedents at translation time. ``text-embedding-3-small``
    # is multilingual, 1536-dim, and cheap enough that re-indexing the
    # whole corpus costs ~$0.07. Bump to -large only if Phase 3
    # validation shows the small model's recall is the bottleneck.
    embedding: str = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")


MODELS = Models()


@dataclass(frozen=True)
class ReasoningEffort:
    """Per-role default ``reasoning.effort`` for OpenAI Responses-API calls.

    Allowed values per OpenAI: ``low`` / ``medium`` / ``high`` / ``xhigh``
    (and ``none`` to disable reasoning entirely on a model that supports
    it). Ignored for Anthropic models. Override via ``.env``.

    * ``comparison`` and ``judge`` are graded for accuracy, not latency,
      so they default to ``high``.
    * ``clustering`` only operates on extracted deviation notes (small
      input, structural output) so ``medium`` is plenty.
    """

    comparison: str = os.getenv("REASONING_EFFORT_COMPARISON", "high")
    clustering: str = os.getenv("REASONING_EFFORT_CLUSTERING", "medium")
    judge: str = os.getenv("REASONING_EFFORT_JUDGE", "high")
    style_extraction: str = os.getenv("REASONING_EFFORT_STYLE_EXTRACTION", "high")
    # Critic runs on every translation; ``high`` is the same effort the
    # offline deviation auditor uses since it produces the same kind of
    # output. Drop to ``medium`` for cost-sensitive batches.
    critic: str = os.getenv("REASONING_EFFORT_CRITIC", "high")


REASONING = ReasoningEffort()


@dataclass(frozen=True)
class Thresholds:
    """Tunable knobs. Keep them here so eval reports can log the exact values used."""

    # Sliding-window size: number of consecutive recent translated parts shown
    # as JP-EN reference pairs in the prompt.
    window_size: int = 10

    # Per-chapter pre-window validators. Calibrated against the 229
    # JP-EN pairs of the (correctly-aligned) parallel corpus via
    # ``translator prep calibrate``. Re-run that command and update
    # these whenever the corpus changes materially (e.g. new EN parts
    # land or scrape parsers change).
    #
    # length_ratio = EN visible chars / JP visible chars
    #   corpus p05 = 2.05, p95 = 2.67, mean = 2.33, stdev = 0.19.
    #   ±10% safety margin around p05/p95 → 1.85 / 2.94.
    length_ratio_min: float = 1.85
    length_ratio_max: float = 2.94
    # dialogue_parity_max_skew = |jp「」 - en quoted| / max(both)
    #   corpus p95 = 0.107, max = 0.375.
    #   Set to p95 so the check flags only the most-skewed 5% of pairs.
    dialogue_parity_max_skew: float = 0.107
    # name_frequency_*_skew = max over tracked characters of
    #   |jp_count - en_count| / max(jp_count, en_count)
    #   Corpus per-character mean ≈ 0.30 (EN expands subject-elided JP),
    #   p95 ≈ 0.50–0.64 depending on character. Bands kept loose so
    #   the check primarily catches catastrophic name-elision rather
    #   than normal subject expansion.
    name_frequency_warn_skew: float = 0.65
    name_frequency_fail_skew: float = 0.90

    # Test set construction
    test_holdout_target_count: int = 30
    random_seed: int = 7

    # Eval gates
    comet_score_floor: float = 0.80
    comet_stretch_goal: float = 0.85

    # Rule lifecycle
    active_rule_soft_cap: int = 25  # warn when exceeded; consider consolidating

    # Self-evaluation cadence
    deviation_batch_size: int = 30

    # Translation output cap (passed as ``max_tokens`` to the provider).
    # The longest EN chapter in the corpus is ~17.6K chars (~4.4K
    # tokens). 16K gives generous headroom and stays well under Claude
    # Opus 4.7's 32K output limit. Bump if you start translating
    # longer arcs or a chapter near the cap.
    translation_max_tokens: int = 16384

    # Critic + revision loop knobs. The critic always runs (cheap);
    # whether a *revision pass* is triggered depends on the gates
    # below.
    #
    # * A revision is requested when the critic emits any flag with
    #   severity == ``critique_revise_severity`` OR when the total
    #   minor flag count meets ``critique_revise_minor_threshold``.
    # * ``critique_max_revisions`` caps how many revision rounds run
    #   per chapter. 1 is usually enough; 2 is the practical max
    #   before diminishing returns.
    critique_revise_severity: str = "major"
    critique_revise_minor_threshold: int = 3
    critique_max_revisions: int = 1

    # Precedent RAG knobs. Precedents are paragraph-level JP↔EN
    # pairs aligned by length-only DP (no EN embedding). The
    # retriever returns up to ``precedents_per_chapter`` paragraph
    # pairs per target chapter (after dedup across queries). The
    # per-query top-K caps how many neighbors each JP paragraph asks
    # the index for before deduplicating.
    #
    # Defaults sized to ~5K tokens of injected context (~6% of an
    # 80K-token translation prompt). Lower the per-chapter cap if
    # prompt bloat becomes a problem; raise it if recall is the
    # bottleneck.
    #
    # Ablation on part_230 (k = 25 / 50 / 75) showed that the critic
    # flag count keeps trending down as k rises, while the precedent
    # block stays well under 5% of the total prompt. The cost delta
    # between k=25 and k=150 is ~$0.08 per chapter on Opus, which is
    # not enough to justify a thinner reference shelf.
    precedents_per_chapter: int = 150
    precedents_top_k_per_query: int = 3
    # Drop stored precedents whose JP↔EN length-DP score is below
    # this floor. Length-only alignment can lock onto an adjacent
    # paragraph with a near-identical length but unrelated content;
    # those entries embed correctly on the JP side but their EN
    # field would mislead the LLM. ~14% of corpus entries fall below
    # 0.3 — the corresponding JP queries still match good entries
    # for the same content via the dedup-and-cap pipeline.
    precedents_min_length_score: float = 0.3
    # Cross-lingual JP↔EN cosine threshold (set by ``translator
    # precedents validate``). Catches length-DP false positives the
    # length filter alone can't see (e.g., adjacent paragraphs with
    # similar character counts but different content). Calibrated
    # against chapters with exact paragraph-count alignment: the 5th
    # percentile of their cross-lingual cosine distribution is
    # ~0.30, so anything below keeps 95% of true alignments while
    # filtering the 5% of pairs that look semantically unrelated.
    # Set to 0.0 to bypass this filter (e.g., before validate runs).
    precedents_min_semantic_score: float = 0.3


THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Scraper:
    """HTTP scraping config. Conservative defaults; override via .env."""

    user_agent: str = os.getenv(
        "SCRAPER_USER_AGENT",
        "ln-translator/0.2 (research; +https://github.com/)",
    )
    kakuyomu_work_id: str = "1177354054894027232"
    kakuyomu_base: str = "https://kakuyomu.jp"
    avelilium_toc_url: str = (
        "https://avelilium.com/story-about-buying-my-classmate-once-a-week/"
    )
    delay_kakuyomu_s: float = float(os.getenv("SCRAPER_DELAY_KAKUYOMU", "1.5"))
    delay_avelilium_s: float = float(os.getenv("SCRAPER_DELAY_AVELILIUM", "1.0"))
    max_retries: int = 4
    request_timeout_s: float = 30.0


SCRAPER = Scraper()


@dataclass(frozen=True)
class Secrets:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))


SECRETS = Secrets()


def require_openai_key() -> str:
    if not SECRETS.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return SECRETS.openai_api_key


def require_anthropic_key() -> str:
    if not SECRETS.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return SECRETS.anthropic_api_key
