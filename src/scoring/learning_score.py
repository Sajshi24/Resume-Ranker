"""
src/scoring/learning_score.py
==============================
Learning Velocity Scorer

Infers how fast a candidate learns and adapts. Rewards intellectual curiosity,
self-directed learning, and technology adoption velocity.

Sub-dimensions:
    learning_velocity, technology_adoption, research_interest, self_learning,
    certifications_quality, hackathon_activity, open_source_contribution,
    career_growth_rate, knowledge_expansion

Output: LearningScoreResult (extends ScoreResult)

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

logger = logging.getLogger("learning_score")

LEARNING_DIMENSION_WEIGHTS: dict[str, float] = {
    "learning_velocity": 2.0,
    "technology_adoption": 1.8,
    "research_interest": 1.2,
    "self_learning": 1.5,
    "certifications_quality": 1.0,
    "hackathon_activity": 1.0,
    "open_source_contribution": 1.2,
    "career_growth_rate": 1.5,
    "knowledge_expansion": 1.3,
}

LEARNING_SYSTEM_PROMPT = """
You are a talent intelligence expert specialising in learning potential assessment.
Your task: evaluate how fast and effectively this candidate learns and evolves.

Reason from evidence — do NOT just count certifications. Infer velocity from patterns.

Score these dimensions (0-100):
- learning_velocity: How fast does the candidate adopt new skills? Evidence of rapid upskilling?
- technology_adoption: Has the candidate moved from older to modern technologies over time?
- research_interest: Publications, conference talks, research papers, preprints, reading groups?
- self_learning: MOOCs, personal projects outside work, online courses, self-taught skills?
- certifications_quality: Quality and relevance of certifications (not just quantity)?
- hackathon_activity: Participation in hackathons, competitions, coding challenges?
- open_source_contribution: GitHub contributions, PR reviews, package maintenance?
- career_growth_rate: How quickly has the candidate progressed in seniority and scope?
- knowledge_expansion: Has the candidate branched into adjacent domains beyond their core?

Provide:
- overall_score (0-100)
- confidence (0.0-1.0)
- reasoning (2-4 sentences)
- learning_archetype: one of 'fast_learner' | 'steady_learner' | 'specialist' | 'research_minded' | 'unknown'
- key_learning_signals (list of strongest evidence of learning behaviour)
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class LearningScoreResult(ScoreResult):
    """Scoring result for the Learning Velocity dimension."""

    scorer_name: str = "LearningScorer"
    learning_archetype: str = Field(
        "unknown",
        description="'fast_learner' | 'steady_learner' | 'specialist' | 'research_minded' | 'unknown'",
    )
    key_learning_signals: list[str] = Field(default_factory=list)


class LearningScorer(BaseScorer):
    """
    Infers learning velocity and intellectual growth from the candidate profile.

    Rewards curiosity, self-directed learning, open source, and career acceleration.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=LEARNING_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.15)
        logger.info("LearningScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "LearningScorer"

    def score(self, candidate: Any, job_profile: Any) -> LearningScoreResult:
        """
        Score learning velocity and intellectual growth.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        LearningScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "LearningScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Build enriched learning context
        learning_context_lines: list[str] = []

        certs = getattr(candidate, 'certifications', [])
        if certs:
            for c in certs:
                learning_context_lines.append(
                    f"Certification: {c.name} | {c.platform} | {c.completion_date}"
                )

        research = getattr(candidate, 'research', [])
        if research:
            for r in research:
                learning_context_lines.append(
                    f"Research: {r.title} | {r.status} | {r.institution}"
                )

        achievements = getattr(candidate, 'achievements', [])
        for ach in achievements:
            if ach.category in ('hackathon', 'competition', 'open_source', 'research', 'publication'):
                learning_context_lines.append(
                    f"Achievement [{ach.category}]: {ach.title} — {ach.description}"
                )

        timeline = getattr(candidate, 'timeline', [])
        if timeline:
            for evt in timeline[:10]:
                learning_context_lines.append(
                    f"Timeline [{evt.event_type}]: {evt.title} | {evt.start_date} -> {evt.end_date}"
                )

        learning_context = "\n".join(learning_context_lines) or "No specific learning activities found."

        prompt = (
            f"JOB PROFILE (for context on what learning matters here):\n{jd_text}\n\n"
            f"CANDIDATE OVERVIEW:\n{cand_text}\n\n"
            f"LEARNING ACTIVITIES (certifications, research, achievements, timeline):\n{learning_context}\n\n"
            "Evaluate learning velocity and intellectual growth. Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return LearningScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        dimension_scores: dict[str, float] = {}
        for dim in LEARNING_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 40)))

        weighted = weighted_average(list(dimension_scores.values()), list(LEARNING_DIMENSION_WEIGHTS.values()))
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        final_score = 0.65 * weighted + 0.35 * llm_overall

        return LearningScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.65)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            learning_archetype=str(raw.get("learning_archetype", "unknown")),
            key_learning_signals=list(raw.get("key_learning_signals", [])),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> LearningScoreResult:
    """Module-level convenience function."""
    scorer = LearningScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
