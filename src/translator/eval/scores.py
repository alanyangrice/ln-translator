"""COMET + BERTScore wrappers.

Both depend on the ``ml`` optional extra. Imports happen inside the
functions so the rest of the pipeline (and CI) doesn't pay the cost of
loading these heavy libraries unless we're actually scoring.

Both functions take *aligned* lists ``(sources, hypotheses, references)``
and return a list of per-segment scores plus an aggregate mean. The
caller decides what segment granularity makes sense (whole chapter vs
paragraph vs sentence).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    per_segment: list[float]
    mean: float


def comet_score(
    *,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> ScoreResult:  # pragma: no cover - heavy ml import
    """Compute COMET (Unbabel) per-segment scores.

    Requires ``uv sync --extra ml``. Raises ``ImportError`` with a
    helpful message if the ``comet`` package isn't available.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise ImportError(
            "COMET scoring requires the 'ml' extra. Run "
            "`uv sync --extra ml` to install unbabel-comet."
        ) from exc
    if not (len(sources) == len(hypotheses) == len(references)):
        raise ValueError("sources, hypotheses, references must be the same length")
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references, strict=True)]
    output = model.predict(data, batch_size=8, gpus=0)
    return ScoreResult(per_segment=list(output["scores"]), mean=float(output["system_score"]))


def bertscore(
    *,
    hypotheses: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str | None = None,
) -> ScoreResult:  # pragma: no cover - heavy ml import
    """Compute BERTScore (F1) per segment over the (hyp, ref) pairs.

    Requires ``uv sync --extra ml``.
    """
    try:
        from bert_score import score
    except ImportError as exc:
        raise ImportError(
            "BERTScore requires the 'ml' extra. Run `uv sync --extra ml` "
            "to install bert-score."
        ) from exc
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    _, _, f1 = score(
        cands=hypotheses,
        refs=references,
        lang=lang,
        model_type=model_type,
        verbose=False,
    )
    f1_list = [float(x) for x in f1.tolist()]
    return ScoreResult(per_segment=f1_list, mean=sum(f1_list) / max(len(f1_list), 1))
