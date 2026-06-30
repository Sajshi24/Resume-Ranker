"""
src/scoring/__init__.py
=======================
Scoring package initialisation.

Exports the shared ``ScoreResult`` base model and the ``BaseScorer``
abstract class so all scoring modules can import them from one place.

    from src.scoring import ScoreResult, BaseScorer

Author  : Resume Intelligence Engine — Scoring Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("scoring")

# ---------------------------------------------------------------------------
# Shared Output Model — every scorer returns a subclass of this
# ---------------------------------------------------------------------------


class ScoreResult(BaseModel):
    """
    Universal explainable score envelope returned by every scoring module.

    Downstream modules (Recruiter Agent, Dashboard, Ranking) consume
    this model directly.  Add scorer-specific fields in subclasses.
    """

    # Identity
    scorer_name: str = Field(description="Name of the scoring module")
    candidate_id: Optional[str] = Field(None, description="Candidate identifier")
    scored_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO-8601 UTC timestamp",
    )

    # Core numerics
    score: float = Field(ge=0.0, le=100.0, description="Overall score for this dimension (0-100)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this score (0-1)")

    # Explainability
    reasoning: str = Field(default="", description="Natural-language explanation of the score")
    evidence: list[str] = Field(default_factory=list, description="Facts that support the score")
    positive_signals: list[str] = Field(default_factory=list, description="Positive indicators found")
    negative_signals: list[str] = Field(default_factory=list, description="Gaps or red flags found")
    improvement_suggestions: list[str] = Field(default_factory=list, description="Actionable suggestions")

    # Sub-dimension breakdown
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores for individual sub-dimensions (0-100 each)",
    )

    # Diagnostics
    parsing_warnings: list[str] = Field(default_factory=list)
    parsing_errors: list[str] = Field(default_factory=list)
    is_valid: bool = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Abstract Base Scorer
# ---------------------------------------------------------------------------


class BaseScorer(ABC):
    """
    Abstract base class that every scoring module must implement.

    Contract
    --------
    Each scorer must:
    1. Accept a ``Candidate`` and a ``JobProfile`` in ``score()``.
    2. Return a ``ScoreResult`` subclass (never raise to the caller).
    3. Log at INFO level when scoring starts and finishes.
    4. Capture all exceptions internally and embed them in the result.

    Usage
    -----
    >>> scorer = MyScorer()
    >>> result = scorer.score(candidate, job_profile)
    >>> print(result.score, result.reasoning)
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def scorer_name(self) -> str:
        """Human-readable name of this scorer, e.g. 'ProjectScorer'."""
        ...

    @abstractmethod
    def score(self, candidate: Any, job_profile: Any) -> ScoreResult:
        """
        Score the candidate against the job profile.

        Parameters
        ----------
        candidate : Candidate
            Parsed candidate object from ``resume_parser``.
        job_profile : JobProfile
            Parsed job profile object from ``jd_parser``.

        Returns
        -------
        ScoreResult
            Validated result object.  Never raises.
        """
        ...

    def _safe_score(self, candidate: Any, job_profile: Any) -> ScoreResult:
        """
        Wrapper that calls ``score()`` and catches all unhandled exceptions,
        returning a minimal failed result instead of crashing.
        """
        try:
            return self.score(candidate, job_profile)
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Unhandled exception in %s: %s", self.scorer_name, exc)
            return ScoreResult(
                scorer_name=self.scorer_name,
                candidate_id=getattr(candidate, "candidate_id", None),
                score=0.0,
                confidence=0.0,
                reasoning=f"Scoring failed due to an internal error: {exc}",
                parsing_errors=[str(exc)],
                is_valid=False,
            )


__all__ = ["ScoreResult", "BaseScorer"]
