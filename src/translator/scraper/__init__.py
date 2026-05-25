"""Scrape JP (Kakuyomu) and EN (avelilium.com / amawashigroup) sources.

Builds the unified ToC (``data/metadata/toc.json``) and downloads each
part's JP and EN bodies into ``data/parallel/`` as both raw ``.txt`` and
structured ``.json``. The structured form preserves paragraph kind
(narration / dialogue / blank) so the ``.txt`` rendering keeps scene-break
blank lines intact for the translation prompt.
"""
