"""
src/scoring/skill_score.py
===========================
Skill Intelligence Scorer

Evaluates a candidate's skill profile against job requirements.
Goes beyond matching — assesses depth, breadth, modernity, relevance, and growth.

Sub-dimensions:
    core_skill_match, supporting_skill_match, modern_tech_adoption,
    skill_breadth, skill_depth, skill_relevance, industry_demand,
    learning_progression, skill_maturity

Output: SkillScoreResult (extends ScoreResult)

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
    deduplicate,
    get_all_candidate_skills,
    get_all_jd_skills,
    job_profile_to_text,
    score_via_gemini,
    skill_overlap_ratio,
    weighted_average,
)

logger = logging.getLogger("skill_score")

SKILL_DIMENSION_WEIGHTS: dict[str, float] = {
    "core_skill_match": 2.5,
    "supporting_skill_match": 1.5,
    "modern_tech_adoption": 1.3,
    "skill_breadth": 1.0,
    "skill_depth": 2.0,
    "skill_relevance": 1.8,
    "industry_demand": 1.2,
    "learning_progression": 1.1,
    "skill_maturity": 1.4,
}

SKILL_SYSTEM_PROMPT = """
You are a senior engineering recruiter with deep technical expertise.
Your task: evaluate a candidate's skill profile against job requirements.

NEVER just match keywords. Reason about skill depth, progression, and context.

Score these dimensions (0-100):
- core_skill_match: Coverage and depth of skills explicitly required by the job
- supporting_skill_match: Coverage of preferred/bonus skills
- modern_tech_adoption: Is the candidate using current, industry-relevant technologies?
- skill_breadth: Width of technical knowledge across domains
- skill_depth: Depth within specific technical areas (not just "knows Python" but "builds production ML pipelines in Python")
- skill_relevance: How relevant are these skills to this specific role and domain?
- industry_demand: Are these skills in high demand in the broader industry right now?
- learning_progression: Has the candidate adopted newer technologies over time?
- skill_maturity: Evidence of using skills in real production systems, not just tutorials?

Provide:
- overall_score (0-100)
- confidence (0.0-1.0)
- reasoning (2-4 sentences)
- matched_required_skills (list of required skills the candidate has)
- missing_required_skills (list of required skills the candidate lacks)
- matched_preferred_skills (list of preferred skills the candidate has)
- skill_gap_severity: one of 'critical' | 'moderate' | 'minor' | 'none'
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class SkillScoreResult(ScoreResult):
    """Scoring result for the Skill Intelligence dimension."""

    scorer_name: str = "SkillScorer"
    matched_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    skill_gap_severity: str = Field("unknown", description="'critical' | 'moderate' | 'minor' | 'none'")
    raw_overlap_ratio: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Simple set-overlap ratio (used as a sanity anchor, not the primary score)",
    )


class SkillScorer(BaseScorer):
    """
    Evaluates candidate skill profile vs job profile using Gemini reasoning.

    Uses both a deterministic overlap ratio (as a sanity anchor) and
    LLM-based reasoning (for depth, maturity, and modernity assessment).
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=SKILL_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.1)
        logger.info("SkillScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "SkillScorer"

    def score(self, candidate: Any, job_profile: Any) -> SkillScoreResult:
        """
        Score the candidate's skill profile against the job profile.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        SkillScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "SkillScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        # Deterministic overlap (sanity anchor)
        cand_skills = get_all_candidate_skills(candidate)
        jd_skills = get_all_jd_skills(job_profile)
        overlap = skill_overlap_ratio(cand_skills, jd_skills)

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Build skill-focused section
        s = getattr(candidate, 'skills', None)
        skill_detail = ""
        if s:
            buckets = {
                "Programming Languages": s.programming_languages,
                "Frameworks": s.frameworks,
                "Libraries": s.libraries,
                "Databases": s.databases,
                "Cloud": s.cloud,
                "DevOps": s.devops,
                "AI/ML": s.ai_ml,
                "Data Science": s.data_science,
                "Tools": s.tools,
                "Soft Skills": s.soft_skills,
            }
            skill_detail = "\n".join(
                f"  {k}: {', '.join(v)}"
                for k, v in buckets.items() if v
            )

        prompt = (
            f"JOB PROFILE:\n{jd_text}\n\n"
            f"CANDIDATE SKILLS (detailed):\n{skill_detail}\n\n"
            f"CANDIDATE OVERVIEW:\n{cand_text}\n\n"
            f"(Note: raw keyword overlap ratio is {overlap:.2f} — use this as context only, not the primary score)\n\n"
            "Evaluate skill depth, breadth, relevance, and maturity. Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return SkillScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        dimension_scores: dict[str, float] = {}
        for dim in SKILL_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 50)))

        # Blend: weighted dimensions 60% + LLM overall 30% + overlap anchor 10%
        weighted = weighted_average(list(dimension_scores.values()), list(SKILL_DIMENSION_WEIGHTS.values()))
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        overlap_anchor = clamp_score(overlap * 100)
        final_score = 0.60 * weighted + 0.30 * llm_overall + 0.10 * overlap_anchor

        return SkillScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.75)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            matched_required_skills=deduplicate(list(raw.get("matched_required_skills", []))),
            missing_required_skills=deduplicate(list(raw.get("missing_required_skills", []))),
            matched_preferred_skills=deduplicate(list(raw.get("matched_preferred_skills", []))),
            skill_gap_severity=str(raw.get("skill_gap_severity", "unknown")),
            raw_overlap_ratio=round(overlap, 4),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> SkillScoreResult:
    """Module-level convenience function."""
    scorer = SkillScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
