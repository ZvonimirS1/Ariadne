"""
Run this after setup to confirm your PubMed client actually works:

    python scripts/smoke_test.py

If it prints real titles and abstracts below, your environment, API key,
and network path to NCBI are all correctly wired up. This is intentionally
the very first thing you should get running - everything else builds on it.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.literature.pubmed_client import search_pubmed, fetch_abstracts


def main():
    query = "metformin cardiovascular outcomes randomized trial"
    print(f"Searching PubMed for: {query!r}\n")

    pmids = search_pubmed(query, max_results=5)
    print(f"Found {len(pmids)} PMIDs: {pmids}\n")

    if not pmids:
        print("No results - check your query or network connection.")
        return

    articles = fetch_abstracts(pmids)
    for pmid, data in articles.items():
        print(f"[{pmid}] {data['title']}  ({data['journal']}, {data['pub_date']})")
        abstract_preview = (data["abstract"] or "")[:250]
        print(f"  {abstract_preview}...\n")


if __name__ == "__main__":
    main()
