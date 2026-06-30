"""
src/utils/config.py
===================
Shared configuration and Gemini client factory for the AI Resume Intelligence Engine.

This is the SINGLE SOURCE OF TRUTH for:
    - API key loading
    - Gemini model initialisation
    - Default model identifiers
    - Project-wide path constants

All scoring modules, agents, and future pipeline stages MUST import from here.
Nothing should call ``genai.configure()`` or ``os.getenv("GEMINI_API_KEY")`` directly.

Author  : Resume Intelligence Engine — Config Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("config")

# ---------------------------------------------------------------------------
# Project-wide Path Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data directories
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Well-known files
CANDIDATES_JSONL: Path = RAW_DATA_DIR / "candidates.jsonl"
JD_TXT: Path = RAW_DATA_DIR / "job_description.txt"
PARSED_CANDIDATES_JSON: Path = PROCESSED_DATA_DIR / "parsed_candidates.json"
PARSED_JD_JSON: Path = PROCESSED_DATA_DIR / "parsed_job_description.json"
SCORING_OUTPUT_JSON: Path = OUTPUTS_DIR / "scoring_results.json"

# ---------------------------------------------------------------------------
# Default Model Identifiers
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "gemini-2.0-flash"
REASONING_MODEL: str = "gemini-2.0-flash"   # upgrade to gemini-1.5-pro if needed
EMBEDDING_MODEL: str = "models/text-embedding-004"

# ---------------------------------------------------------------------------
# Gemini Client Factory
# ---------------------------------------------------------------------------

# Module-level cache so ``genai.configure()`` is called at most once per process
_CONFIGURED: bool = False


def _ensure_configured() -> None:
    """
    Idempotently configure the ``google.generativeai`` library with the API key.

    Raises
    ------
    EnvironmentError
        If ``GEMINI_API_KEY`` is not found in the environment.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    genai.configure(api_key=api_key)
    _CONFIGURED = True
    logger.debug("Gemini API configured successfully.")


def get_gemini_model(
    model_name: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
) -> genai.GenerativeModel:
    """
    Create and return a configured ``GenerativeModel`` instance.

    The API key is read from the environment exactly once (cached).
    Subsequent calls with the same ``model_name`` create a new model object
    but do not re-configure the library.

    Parameters
    ----------
    model_name : str
        Gemini model identifier.  Defaults to ``DEFAULT_MODEL``.
    system_instruction : str, optional
        System-level prompt to bake into the model.

    Returns
    -------
    genai.GenerativeModel
        Ready-to-use generative model instance.

    Example
    -------
    >>> from src.utils.config import get_gemini_model
    >>> model = get_gemini_model(system_instruction="You are an expert scorer.")
    >>> response = model.generate_content("Score this resume.")
    """
    _ensure_configured()
    kwargs: dict = {"model_name": model_name}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    return genai.GenerativeModel(**kwargs)


def get_generation_config(
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
) -> genai.GenerationConfig:
    """
    Return a ``GenerationConfig`` with sensible scoring defaults.

    Parameters
    ----------
    temperature : float
        Sampling temperature.  Use low values (≤ 0.2) for deterministic scoring.
    max_output_tokens : int
        Maximum tokens in the response.

    Returns
    -------
    genai.GenerationConfig
    """
    return genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def ensure_output_dirs() -> None:
    """
    Create all required output directories if they do not exist.
    Safe to call multiple times.
    """
    for path in (PROCESSED_DATA_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
