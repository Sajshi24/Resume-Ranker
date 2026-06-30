"""
src/scoring/growth_score.py
============================
Growth Potential Scorer

Predicts a candidate's future growth trajectory. This is the most forward-looking
scorer — it assesses potential, not just current state.

Sub-dimensions:
    learning_agility, innovation_potential, research_orientation,
    career_acceleration, leadership_trajectory, technical_curiosity,
    adaptability_index, cross_domain_synthesis, impact_potential

Output: GrowthScoreResult (extends ScoreResult)

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

logger = logging.getLogger("growth_score")

GROWTH_DIMENSION_WEIGHTS: dict[str, float] = {
    "learning_agility": 2.0,
    "innovation_potential": 1.8,
    "research_orientation": 1.2,
    "career_acceleration": 1.5,
    "leadership_trajectory": 1.4,
    "technical_curiosity": 1.6,
    "adaptability_index": 1.3,
    "cross_domain_synthesis": 1.1,
    "impact_potential": 1.8,
}

GROWTH_SYSTEM_PROMPT = """
You are a futurist talent analyst specialising in growth potential prediction.
Your task: predict how much this candidate will grow in the next 2-3 years in this role.

Focus on TRAJECTORY, not current state. A junior who is accelerating fast
can beat a senior who has plateaued.

Score these dimensions (0-100):
- learning_agility: Speed and quality of adopting new skills/domains
- innovation_potential: Likelihood of producing novel, creative technical solutions
- research_orientation: Intellectual depth; likely to contribute beyond implementation?
- career_acceleration: Rate of seniority/impact growth relative to time in field
- leadership_trajectory: Current leadership signals and likelihood of growth into leadership
- technical_curiosity: Breadth of exploration beyond job requirements; side projects, experiments
- adaptability_index: Can the candidate thrive in ambiguous, fast-changing environments?
- cross_domain_synthesis: Can the candidate connect ideas from different domains creatively?
- impact_potential: Likelihood of driving significant business/technical impact in 2-3 years

Provide:
- overall_score (0-100) — the GROWTH POTENTIAL score
- confidence (0.0-1.0)
- reasoning (3-5 sentences on the growth trajectory)
- growth_prediction: 2-3 sentence prediction of where this candidate will be in 2-3 years
- potential_ceiling: estimated seniority ceiling ('senior' | 'staff' | 'principal' | 'director' | 'vp' | 'unknown')
- risk_factors (list of things that could limit growth)
- growth_catalysts (list of factors that will accelerate growth)
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class GrowthScoreResult(ScoreResult):
    """Scoring result for the Growth Potential dimension."""

    scorer_name: str = "GrowthScorer"
    growth_prediction: str = Field("", description="Narrative prediction of candidate's 2-3 year trajectory")
    potential_ceiling: str = Field(
        "unknown",
        description="Estimated seniority ceiling: 'senior' | 'staff' | 'principal' | 'director' | 'vp'",
    )
    risk_factors: list[str] = Field(default_factory=list, description="Factors that could limit growth")
    growth_catalysts: list[str] = Field(default_factory=list, description="Factors that will accelerate growth")


class GrowthScorer(BaseScorer):
    """
    Predicts future growth potential using Gemini reasoning.

    Rewards trajectory, curiosity, adaptability, and early leadership signals.
    Does not penalise for current seniority level — a junior with high velocity
    can score higher than a senior who has plateaued.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=GROWTH_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.2)
        logger.info("GrowthScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "GrowthScorer"

    def score(self, candidate: Any, job_profile: Any) -> GrowthScoreResult:
        """
        Predict the candidate's growth potential.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        GrowthScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "GrowthScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Build trajectory-focused context
        trajectory_lines: list[str] = []

        timeline = getattr(candidate, 'timeline', [])
        if timeline:
            for evt in timeline:
                trajectory_lines.append(
                    f"Timeline: [{evt.event_type}] {evt.title} | {evt.start_date} -> {evt.end_date or 'present'}"
                )

        for cert in getattr(candidate, 'certifications', [])[:5]:
            trajectory_lines.append(f"Cert: {cert.name} ({cert.completion_date}) — {cert.platform}")

        for res in getattr(candidate, 'research', []):
            trajectory_lines.append(f"Research: {res.title} | {res.status}")

        for ach in getattr(candidate, 'achievements', []):
            trajectory_lines.append(
                f"Achievement [{ach.category or 'other'}]: {ach.title} — {ach.description or ''}"
            )

        trajectory_context = "\n".join(trajectory_lines) or "Limited trajectory data available."

        prompt = (
            f"TARGET ROLE (for seniority and growth context):\n{jd_text}\n\n"
            f"CANDIDATE OVERVIEW:\n{cand_text}\n\n"
            f"CAREER TRAJECTORY:\n{trajectory_context}\n\n"
            "Predict growth potential and 2-3 year trajectory. Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return GrowthScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        dimension_scores: dict[str, float] = {}
        for dim in GROWTH_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 50)))

        weighted = weighted_average(list(dimension_scores.values()), list(GROWTH_DIMENSION_WEIGHTS.values()))
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        final_score = 0.6 * weighted + 0.4 * llm_overall

        return GrowthScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.6)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            growth_prediction=str(raw.get("growth_prediction", "")),
            potential_ceiling=str(raw.get("potential_ceiling", "unknown")),
            risk_factors=list(raw.get("risk_factors", [])),
            growth_catalysts=list(raw.get("growth_catalysts", [])),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> GrowthScoreResult:
    """Module-level convenience function."""
    scorer = GrowthScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
