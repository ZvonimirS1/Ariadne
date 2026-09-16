# Ariadne

A biomedical knowledge agent: literature synthesis, drug-repurposing over a
knowledge graph, and rare-disease phenotype matching, unified under one
orchestrator. Every claim the system makes carries a source back to where
it came from - no unsourced assertions, ever.

## Status

**Module 1 (Literature Synthesis) is built and runs end to end, but has not been
evaluated yet** - there are no measured accuracy numbers so far. The shared
infrastructure - provenance checking, the tool-calling orchestrator, telemetry,
and the eval harness - is built, and the offline test suite passes.

| Component                       | State                                   |
| ------------------------------- | --------------------------------------- |
| Shared infrastructure           | Built; offline tests pass               |
| Module 1 - Literature Synthesis | Built and working; eval not yet run     |
| Module 2 - Drug Repurposing     | Stub that refuses; weeks 6-8  |
| Module 3 - Phenotype Matching   | Stub that refuses; weeks 9-11 |

The stubs are not placeholders returning empty results - they return an
explicit "not implemented", and both the orchestrator prompt and the tool
output instruct the model not to answer from background knowledge instead.
`tests/test_routing.py` asserts this.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in the keys below
```

You need:

- **`ANTHROPIC_API_KEY`** (required) - <https://console.anthropic.com/settings/keys>
- `NCBI_API_KEY` and `NCBI_EMAIL` (recommended) - raises the PubMed rate limit
  from 3 to 10 requests/second. <https://www.ncbi.nlm.nih.gov/account/>

## Verify it works

```bash
pytest                          # offline, free - no API calls at all
python scripts/smoke_test.py    # checks the PubMed path end to end
```

Then ask it something:

```bash
python cli.py "Does metformin improve glycemic control in type 2 diabetes?"
```

Expect this to take a couple of minutes (about 2.5 minutes in the first
measured run). The orchestrator may run several PubMed searches, and every
retrieved abstract gets its own extraction call. Progress is printed as it
goes. The answer is followed by the first 10 verified claims; use
`--show-claims 0` to print all of them, or `--json` for the full result.

And measure it:

```bash
python eval/run_eval.py --module literature --limit 2   # 2 questions, roughly $0.75
python eval/run_eval.py --module literature             # all 8 questions, roughly $3
```

> Cost estimates are based on a measured ~$0.04 per extraction call on
> `claude-opus-5`, at up to 10 abstracts per question - treat them as rough.
> PubMed responses are cached to `data/cache/`, so reruns re-extract but never
> re-download.

## The design principle, and how it is enforced

Every module returns a `SourcedAnswer` (see [src/schemas.py](src/schemas.py)),
and a `Claim` without a `Source` never reaches a final answer. But "attach a
PMID" is a weak guarantee on its own - a model can invent a PMID, or attach a
real one to a claim that paper never made.

So Ariadne requires something harder to fake: **every extracted finding must
carry a verbatim quote from the abstract it cites**, and
[src/provenance.py](src/provenance.py) checks that the quote genuinely occurs
in that abstract. This catches both failure modes:

1. **Fabricated citation** - the PMID was never in the retrieved set.
2. **Fabricated content** - the PMID is real, but the quote is not in the paper.

The useful consequence: **the hallucination rate is mechanically measurable
with no human labeling.** The extraction layer returns _candidates_; only
verified findings become claims. Rejected candidates are counted and reported
rather than silently dropped, because their rate is the headline result.

Contradiction detection follows the same philosophy - it compares typed enum
fields across findings that measure the same intervention and outcome, rather
than asking a model whether the evidence "seems mixed". A flagged contradiction
always points at two specific papers you can go read.

## Project structure

```
src/
  config.py                          # loads .env; require() fails loudly on missing keys
  schemas.py                         # the shared contract - Source / Claim / SourcedAnswer,
                                     #   and CandidateFinding vs VerifiedFinding
  provenance.py                      # THE CHOKEPOINT - quote verification
  llm.py                             # Anthropic client, structured output, prompt caching
  telemetry.py                       # JSONL run logs
  orchestrator/
    router.py                        # routing policy / system prompt
    tools.py                         # the tool surface the model sees
    loop.py                          # tool-calling loop
  modules/
    literature/                      # Module 1: pubmed_client, extract, synthesize
    repurposing/                     # Module 2: stub
    phenotype/                       # Module 3: stub
eval/
  datasets/literature.yaml           # ground truth, hand-curated
  metrics.py                         # hallucination rate, direction accuracy, contradiction F1
  harness.py, run_eval.py
tests/                               # all offline, all free
cli.py                               # ask a question
```

## Data sources

Each under its own license - see the individual sites.

- **PubMed / NCBI E-utilities** - literature
- **Hetionet** (CC0) - drug/gene/disease knowledge graph _(week 6)_
- **HPO + Orphanet** - rare disease phenotypes _(week 9)_

Note for later: Hetionet's data files are stored with Git LFS.
`raw.githubusercontent.com` returns a pointer stub rather than the data;
`media.githubusercontent.com` serves the real bytes.

## Ethical framing

Module 3, when built, is a **research triage aid** - it ranks candidate
diagnoses for a researcher to investigate further. It is **not a diagnostic
tool** and must not be presented or used as one. This is stated in the module
source as well as here.

More broadly: this system is built to surface and cite published evidence, not
to give medical advice.

## License

MIT - see `LICENSE`.
