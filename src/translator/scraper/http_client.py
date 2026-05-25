"""Shared HTTP client: retry, rate limiting, on-disk URL cache.

Caching policy: every successful GET is written to data/raw/<bucket>/<safe>.html.
`refresh=True` bypasses the cache. We use a deterministic filename derived from the
URL path (not a hash) so cached files are inspectable by hand.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from translator.config import PATHS, SCRAPER

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str
    text: str
    from_cache: bool
    cache_path: Path


class Throttle:
    """Per-host minimum interval between requests."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def wait(self, host: str, delay_s: float) -> None:
        now = time.monotonic()
        prev = self._last.get(host)
        if prev is not None:
            elapsed = now - prev
            if elapsed < delay_s:
                time.sleep(delay_s - elapsed)
        self._last[host] = time.monotonic()


_THROTTLE = Throttle()


def _delay_for(host: str) -> float:
    if "kakuyomu" in host:
        return SCRAPER.delay_kakuyomu_s
    if "avelilium" in host or "amawashigroup" in host or "wordpress.com" in host:
        return SCRAPER.delay_avelilium_s
    return 1.0


def _safe_name(url: str) -> str:
    """Stable, inspectable filename for a URL."""
    no_scheme = re.sub(r"^https?://", "", url)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", no_scheme)
    if len(cleaned) > 180:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        cleaned = cleaned[:170] + "_" + digest
    return cleaned + ".html"


def _cache_path(url: str, bucket: str) -> Path:
    bucket_dir = {
        "jp": PATHS.raw_jp,
        "en": PATHS.raw_en,
        "toc": PATHS.raw_toc,
    }.get(bucket, PATHS.data / "raw" / bucket)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    return bucket_dir / _safe_name(url)


@retry(
    reraise=True,
    stop=stop_after_attempt(SCRAPER.max_retries),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def _do_get(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(url, timeout=SCRAPER.request_timeout_s, follow_redirects=True)
    if resp.status_code in (429, 500, 502, 503, 504):
        raise httpx.HTTPStatusError(
            f"{resp.status_code} for {url}", request=resp.request, response=resp
        )
    resp.raise_for_status()
    return resp


_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            headers={
                "User-Agent": SCRAPER.user_agent,
                "Accept-Language": "ja,en;q=0.8",
            },
            http2=False,
        )
    return _CLIENT


def fetch(url: str, *, bucket: str, refresh: bool = False) -> FetchResult:
    """Fetch ``url`` with throttling, retry, and on-disk caching.

    ``bucket`` controls where the cached HTML is stored: 'jp', 'en', or 'toc'.
    """
    path = _cache_path(url, bucket)
    if path.exists() and not refresh:
        return FetchResult(url=url, text=path.read_text(encoding="utf-8"), from_cache=True, cache_path=path)

    host = httpx.URL(url).host or ""
    _THROTTLE.wait(host, _delay_for(host))
    logger.info("GET %s", url)
    resp = _do_get(_client(), url)
    text = resp.text
    path.write_text(text, encoding="utf-8")
    return FetchResult(url=url, text=text, from_cache=False, cache_path=path)
