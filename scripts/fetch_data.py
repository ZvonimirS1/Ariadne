#!/usr/bin/env python3
"""
Download the datasets Modules 2 and 3 need.

    python scripts/fetch_data.py            # everything
    python scripts/fetch_data.py --only hpo

Not needed for Module 1 - run it when you start week 6. It exists now so the
two non-obvious facts below are recorded in code rather than rediscovered:

  1. Hetionet's data files are stored with Git LFS. raw.githubusercontent.com
     serves a ~130-byte text pointer, not the data. The bytes live on
     media.githubusercontent.com. Fetching the pointer and trying to bunzip2
     it produces a confusing failure a long way from the cause.

  2. The HPO release is pinned to a specific version. HPO ships monthly, and
     an eval number computed against "whatever was current that week" is not
     reproducible. Bump the pin deliberately, not by accident.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402

HPO_RELEASE = "v2026-09-01"
_HPO_BASE = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download"
    f"/{HPO_RELEASE}"
)
_HETIONET_BASE = "https://media.githubusercontent.com/media/hetio/hetionet/main"

# name -> (url, destination, expected leading magic bytes or None)
DATASETS: dict[str, list[tuple[str, str, bytes | None]]] = {
    "hetionet": [
        (
            f"{_HETIONET_BASE}/hetnet/json/hetionet-v1.0.json.bz2",
            "hetionet/hetionet-v1.0.json.bz2",
            b"BZh",
        ),
        (
            "https://raw.githubusercontent.com/hetio/hetionet/main"
            "/hetnet/json/hetionet-v1.0-metagraph.json",
            "hetionet/hetionet-v1.0-metagraph.json",
            None,  # small plain JSON, not stored in LFS
        ),
    ],
    "hpo": [
        (f"{_HPO_BASE}/hp.obo", f"hpo/{HPO_RELEASE}/hp.obo", None),
        (f"{_HPO_BASE}/phenotype.hpoa", f"hpo/{HPO_RELEASE}/phenotype.hpoa", None),
        (
            f"{_HPO_BASE}/genes_to_phenotype.txt",
            f"hpo/{HPO_RELEASE}/genes_to_phenotype.txt",
            None,
        ),
    ],
}


def _download(url: str, dest: Path, magic: bytes | None) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.relative_to(DATA_DIR)} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get   {dest.relative_to(DATA_DIR)} ...", end="", flush=True)

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
    except requests.RequestException as exc:
        print(f" FAILED ({exc})")
        tmp.unlink(missing_ok=True)
        return False

    if magic:
        with tmp.open("rb") as fh:
            head = fh.read(len(magic))
        if head != magic:
            # Almost always means an LFS pointer came back instead of the data.
            print(f" FAILED (expected {magic!r}, got {head!r} - LFS pointer?)")
            tmp.unlink(missing_ok=True)
            return False

    tmp.rename(dest)
    print(f" done ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Module 2/3 datasets.")
    parser.add_argument("--only", choices=sorted(DATASETS), help="fetch just one dataset")
    args = parser.parse_args()

    targets = [args.only] if args.only else sorted(DATASETS)
    ok = True
    for name in targets:
        print(f"\n{name}:")
        for url, relative, magic in DATASETS[name]:
            ok &= _download(url, DATA_DIR / relative, magic)

    print("\nDone." if ok else "\nSome downloads failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
