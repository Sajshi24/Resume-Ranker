"""
src/scoring/domain_score.py
============================
Domain Fit Scorer

Evaluates how well a candidate's background fits the target domain.
Uses projects, experience, education, research, skills, achievements,
business context, and career progression — never keyword matching.

Sub-dimensions:
    industry_fit, domain_expertise, product_domain_fit, technology_alignment,
    career_trajectory, research_alignment, business_understanding,
    cross_domain_experience

Output: DomainScoreResult (extends ScoreResult)

Author  : Resume Intelligence Engine — Scoring Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from src.scoring import BaseScorer, ScoreResult
from src.utils.config import get_gemini_model, get_generation_config
from src.utils.helpers import (
    candidate_to_text,
    clamp_confidence,
    clamp_score,
    job_profile_to_text,
    score_via_gemini,
    weighted_average,
)

logger = logging.getLogger("domain_score")

DOMAIN_DIMENSION_WEIGHTS: dict[str, float] = {
    "industry_fit": 1.5,
    "domain_expertise": 2.0,
    "product_domain_fit": 1.3,
    "technology_alignment": 1.4,
    "career_trajectory": 1.2,
    "research_alignment": 0.8,
    "business_understanding": 1.0,
    "cross_domain_experience": 0.8,
}

DOMAIN_SYSTEM_PROMPT = """
You are a senior talent strategist evaluating domain fit for an AI-powered hiring engine.

Your task: assess how well the candidate's background aligns with the target role's domain.
Reason holistically — do NOT just match keywords.

Score these dimensions (0-100):
- industry_fit: Has the candidate worked in the same or adjacent industry?
- domain_expertise: Depth of expertise in the specific technical/business domain required?
- product_domain_fit: Alignment between candidate's product experience and the target product?
- technology_alignment: Do the technologies the candidate knows map to what's needed?
- career_trajectory: Is the candidate's career direction converging towards this role?
- research_alignment: Does the candidate have relevant research or intellectual contribution?
- business_understanding: Does the candidate show evidence of business context awareness?
- cross_domain_experience: Valuable cross-domain experience that brings unique perspective?

Provide:
- overall_score (0-100)
- confidence (0.0-1.0)
- reasoning (2-4 sentences)
- domain_verdict: one of 'strong_fit' | 'moderate_fit' | 'adjacent_fit' | 'weak_fit'
- domain_gap: key gaps in domain experience
- transferable_domain_skills: domain skills from adjacent fields that apply here
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class DomainScoreResult(ScoreResult):
    """Scoring result for the Domain Fit dimension."""

    scorer_name: str = "DomainScorer"
    domain_verdict: str = Field(
        "unknown",
        description="'strong_fit' | 'moderate_fit' | 'adjacent_fit' | 'weak_fit'",
    )
    domain_gap: list[str] = Field(default_factory=list)
    transferable_domain_skills: list[str] = Field(default_factory=list)


class DomainScorer(BaseScorer):
    """
    Evaluates domain fit between candidate background and job requirements.

    Uses Gemini to reason about industry experience, domain expertise,
    career trajectory, and technology alignment.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=DOMAIN_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.1)
        logger.info("DomainScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "DomainScorer"

    def score(self, candidate: Any, job_profile: Any) -> DomainScoreResult:
        """
        Score domain fit for a candidate against the job profile.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        DomainScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "DomainScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        prompt = (
            f"JOB PROFILE:\n{jd_text}\n\n"
            f"CANDIDATE PROFILE:\n{cand_text}\n\n"
            "Evaluate domain fit across all dimensions. Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return DomainScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        # Extract dimension scores
        dimension_scores: dict[str, float] = {}
        for dim in DOMAIN_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 50)))

        dim_values = list(dimension_scores.values())
        dim_weights = list(DOMAIN_DIMENSION_WEIGHTS.values())
        weighted = weighted_average(dim_values, dim_weights)
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        final_score = 0.65 * weighted + 0.35 * llm_overall

        return DomainScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.7)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            domain_verdict=str(raw.get("domain_verdict", "unknown")),
            domain_gap=list(raw.get("domain_gap", [])),
            transferable_domain_skills=list(raw.get("transferable_domain_skills", [])),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> DomainScoreResult:
    """Module-level convenience function."""
    scorer = DomainScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
