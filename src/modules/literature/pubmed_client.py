"""
Minimal client for NCBI's E-utilities API (PubMed search + abstract fetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
No API key is required, but requests are rate-limited to 3/sec without one
and 10/sec with one - always send yours once you have it.
"""
from __future__ import annotations
import time
import xml.etree.ElementTree as ET

import requests

from src.config import NCBI_API_KEY, NCBI_EMAIL

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _common_params() -> dict:
    params = {"tool": "bioagent", "email": NCBI_EMAIL or ""}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Return a list of PMIDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        **_common_params(),
    }
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> dict[str, dict]:
    """
    Given a list of PMIDs, return {pmid: {"title": ..., "abstract": ...,
    "journal": ..., "pub_date": ...}}.

    A small courtesy delay is added since this hits NCBI's servers directly -
    be a good citizen of a free public API.
    """
    if not pmids:
        return {}

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        **_common_params(),
    }
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=20)
    resp.raise_for_status()
    time.sleep(0.1)  # stay well under rate limits on repeated calls

    root = ET.fromstring(resp.text)
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

        results[pmid] = {
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date,
        }

    return results
