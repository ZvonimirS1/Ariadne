"""
Client for NCBI's E-utilities API (PubMed search + abstract fetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No API key is required, but requests are rate-limited to 3/sec without one
and 10/sec with one - always send yours once you have it.

Three things here exist for reasons that aren't obvious from the API docs:

  - A real rate limiter. NCBI blocks by IP, and getting blocked mid-semester
    would be genuinely disruptive.
  - A disk cache. Evaluation reruns the same queries constantly; without a
    cache every run costs time, hammers a free public service, and - because
    PubMed results drift - isn't reproducible.
  - POST for efetch. Fetching 200 PMIDs by GET produces a URL long enough to
    be rejected, which shows up as a confusing 414 rather than a clear error.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from src.config import CACHE_DIR, NCBI_API_KEY, NCBI_EMAIL

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI's published ceilings. Staying just under them.
_RATE_WITH_KEY = 9.0
_RATE_WITHOUT_KEY = 2.5

MAX_IDS_PER_FETCH = 200
_MAX_RETRIES = 4
_TIMEOUT = 30


class PubMedError(RuntimeError):
    """A PubMed request failed after exhausting retries."""


class _RateLimiter:
    """
    Minimum-interval limiter, shared across threads.

    The previous implementation slept 0.1s *after* each response, which limits
    nothing - the sleep happens once the request is already in flight, and
    concurrent callers each sleep independently.
    """

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval


_limiter = _RateLimiter(_RATE_WITH_KEY if NCBI_API_KEY else _RATE_WITHOUT_KEY)


def _common_params() -> dict:
    params = {"tool": "ariadne", "email": NCBI_EMAIL or ""}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _cache_path(endpoint: str, params: dict) -> Path:
    """
    Key on the semantic part of the request only.

    Credentials and the tool name are excluded deliberately: they vary between
    machines but don't change the response, and a cache that misses whenever a
    collaborator uses their own API key is not much of a cache.
    """
    semantic = {k: v for k, v in sorted(params.items())
                if k not in ("api_key", "email", "tool")}
    blob = json.dumps([endpoint, semantic], sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{endpoint}-{digest}.cache"


def _request(endpoint: str, params: dict, *, use_cache: bool = True, post: bool = False) -> str:
    cache_file = _cache_path(endpoint, params)
    if use_cache and cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except OSError:
            pass  # unreadable cache entry is not a failure; just refetch

    url = f"{EUTILS_BASE}/{endpoint}.fcgi"
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        _limiter.acquire()
        try:
            if post:
                resp = requests.post(url, data=params, timeout=_TIMEOUT)
            else:
                resp = requests.get(url, params=params, timeout=_TIMEOUT)

            # 429 and 5xx are transient; back off and try again.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = PubMedError(f"HTTP {resp.status_code} from {endpoint}")
                time.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            text = resp.text
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
            return text

        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise PubMedError(f"{endpoint} failed after {_MAX_RETRIES} attempts: {last_error}")


def search_pubmed(
    query: str,
    max_results: int = 10,
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    sort: str | None = None,
    retstart: int = 0,
    use_cache: bool = True,
) -> list[str]:
    """
    Return a list of PMIDs matching the query.

    `min_year` supports the "anything published in the last 2 years?" question
    Module 2 needs when cross-checking a repurposing candidate against recent
    literature. `retstart` enables paging, which the eval harness needs to
    measure recall beyond the first page.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retstart": retstart,
        "retmode": "json",
        **_common_params(),
    }
    if sort:
        params["sort"] = sort
    if min_year or max_year:
        params["datetype"] = "pdat"
        params["mindate"] = str(min_year or 1900)
        params["maxdate"] = str(max_year or 3000)

    raw = _request("esearch", params, use_cache=use_cache)
    try:
        return json.loads(raw).get("esearchresult", {}).get("idlist", [])
    except json.JSONDecodeError as exc:
        raise PubMedError(f"esearch returned non-JSON: {raw[:200]}") from exc


def _parse_articles(xml_text: str) -> dict[str, dict]:
    root = ET.fromstring(xml_text)
    results: dict[str, dict] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstracts can be split into multiple labeled sections
        # (e.g. BACKGROUND, METHODS, RESULTS) - join them in order.
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        year_el = article.find(".//JournalIssue/PubDate/Year")
        pub_date = year_el.text if year_el is not None else None

        # Publication types are the most reliable signal for study design;
        # inferring design from abstract prose alone is markedly noisier.
        pub_types = [
            (pt.text or "").strip()
            for pt in article.findall(".//PublicationTypeList/PublicationType")
            if pt.text
        ]

        results[pmid] = {
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date,
            "publication_types": pub_types,
        }

    return results


def fetch_abstracts(pmids: list[str], *, use_cache: bool = True) -> dict[str, dict]:
    """
    Given a list of PMIDs, return {pmid: {"title": ..., "abstract": ...,
    "journal": ..., "pub_date": ..., "publication_types": [...]}}.

    Chunks automatically, so callers can pass an arbitrarily long list.
    """
    if not pmids:
        return {}

    results: dict[str, dict] = {}
    for start in range(0, len(pmids), MAX_IDS_PER_FETCH):
        chunk = pmids[start:start + MAX_IDS_PER_FETCH]
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "rettype": "abstract",
            "retmode": "xml",
            **_common_params(),
        }
        xml_text = _request("efetch", params, use_cache=use_cache, post=True)
        try:
            results.update(_parse_articles(xml_text))
        except ET.ParseError as exc:
            raise PubMedError(f"efetch returned unparseable XML: {exc}") from exc

    return results


def search_and_fetch(
    query: str,
    max_results: int = 10,
    **kwargs,
) -> dict[str, dict]:
    """Convenience path used by Module 1: search, then fetch, in one call."""
    use_cache = kwargs.pop("use_cache", True)
    pmids = search_pubmed(query, max_results, use_cache=use_cache, **kwargs)
    return fetch_abstracts(pmids, use_cache=use_cache)
