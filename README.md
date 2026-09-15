# Ariadne

A biomedical knowledge agent: literature synthesis, drug-repurposing over a
knowledge graph, and rare-disease phenotype matching, unified under one
orchestrator. Every claim the system makes carries a source back to where
it came from - no unsourced assertions, ever.

## Status

🚧 Early scaffold. Module 1 (literature search) has a working PubMed client.
Modules 2 and 3, and the orchestrator itself, are next.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in NCBI_API_KEY and NCBI_EMAIL
```

## Verify it works

```bash
python scripts/smoke_test.py
```

You should see real PubMed titles and abstracts printed to the console.

## Project structure

```
src/
  config.py                          # loads .env
  schemas.py                         # Source / Claim / SourcedAnswer - the
                                      # shared "no claim without a source" contract
  modules/
    literature/
      pubmed_client.py               # PubMed search + abstract fetch
scripts/
  smoke_test.py                      # first end-to-end sanity check
```

## Design principle

Every module returns a `SourcedAnswer` (see `src/schemas.py`). A `Claim`
without at least one `Source` attached should never reach a final answer -
`SourcedAnswer.ungrounded_claims()` exists specifically so the eval harness
can catch it if one slips through.

## Data sources used (each under its own license - see individual sites)

- PubMed / NCBI E-utilities - literature
- Hetionet - drug/gene/disease knowledge graph _(coming next)_
- HPO + Orphanet - rare disease phenotypes _(coming next)_
- MIMIC-IV - credentialed clinical data, stretch module _(pending PhysioNet approval)_

## License

MIT - see `LICENSE`.
