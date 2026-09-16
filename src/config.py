"""
Central place for loading configuration from environment variables.
Every module should pull its keys from here rather than calling
os.environ directly - makes it obvious where secrets come from,
and makes testing easier later (just monkeypatch this module).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinned deliberately rather than tracking "latest": the eval numbers in the
# writeup are only meaningful if the model that produced them is named.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or "claude-opus-5"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw downloads (Hetionet, HPO) and the HTTP response cache. Both are
# regenerable, so `/data/` stays gitignored - see .gitignore.
DATA_DIR = Path(os.getenv("ARIADNE_DATA_DIR") or PROJECT_ROOT / "data")
CACHE_DIR = DATA_DIR / "cache"

# Where telemetry JSONL run logs land.
RUN_LOG_DIR = DATA_DIR / "runs"


class MissingConfig(RuntimeError):
    """Raised when a required key is absent, with instructions for fixing it."""


_SETUP_HINT = {
    "ANTHROPIC_API_KEY": (
        "Get a key at https://console.anthropic.com/settings/keys and add it "
        "to your .env file."
    ),
    "NCBI_API_KEY": (
        "Optional but recommended. Get one at https://www.ncbi.nlm.nih.gov/account/ "
        "(Settings -> API Key Management)."
    ),
    "NCBI_EMAIL": (
        "NCBI's usage policy asks you to identify yourself. Any address you "
        "actually read is fine."
    ),
}


def require(name: str) -> str:
    """
    Return the named setting, or fail immediately with an actionable message.

    Without this, a missing ANTHROPIC_API_KEY surfaces much later as an opaque
    401 from deep inside an extraction call, which is a genuinely confusing
    thing to debug. Fail at the boundary instead.
    """
    value = globals().get(name) or os.getenv(name)
    if not value:
        hint = _SETUP_HINT.get(name, "Add it to your .env file.")
        raise MissingConfig(f"{name} is not set. {hint}")
    return value


def ensure_dirs() -> None:
    """Create the data directories if they don't exist yet. Safe to call repeatedly."""
    for directory in (DATA_DIR, CACHE_DIR, RUN_LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
