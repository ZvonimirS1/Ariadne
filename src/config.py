"""
Central place for loading configuration from environment variables.
Every module should pull its keys from here rather than calling
os.environ directly - makes it obvious where secrets come from,
and makes testing easier later (just monkeypatch this module).
"""
import os
from dotenv import load_dotenv

load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
