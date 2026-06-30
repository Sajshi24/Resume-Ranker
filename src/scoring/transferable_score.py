"""
src/scoring/transferable_score.py
==================================
Transferable Skills Scorer

Infers soft skills and non-technical competencies from observable evidence.
Never self-reports ("I am a good communicator") — infers from BEHAVIOUR.

Sub-dimensions:
    communication, leadership, ownership, critical_thinking,
    problem_solving, collaboration, mentoring, decision_making,
    adaptability, product_thinking

Output: TransferableScoreResult (extends ScoreResult)

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

logger = logging.getLogger("transferable_score")

TRANSFERABLE_DIMENSION_WEIGHTS: dict[str, float] = {
    "communication": 1.5,
    "leadership": 1.8,
    "ownership": 2.0,
    "critical_thinking": 1.6,
    "problem_solving": 1.7,
    "collaboration": 1.4,
    "mentoring": 1.2,
    "decision_making": 1.5,
    "adaptability": 1.3,
    "product_thinking": 1.0,
}

TRANSFERABLE_SYSTEM_PROMPT = """
You are an expert organisational psychologist and talent strategist.
Your task: infer the candidate's transferable and soft skills from OBSERVABLE BEHAVIOUR only.

Do NOT score based on self-claimed traits. Score based on:
- Leadership: Did they lead a team, project, club, or initiative?
- Ownership: Did they own an outcome end-to-end, not just execute tasks?
- Communication: Writing papers, talks, documentation, cross-team collaboration?
- Problem Solving: Novel solutions to hard problems in projects or experience?
- Collaboration: Cross-functional work, team projects, open source?
- Mentoring: Did they mentor juniors, write tutorials, do code reviews?
- Adaptability: Career pivots, learning new domains, joining early-stage teams?
- Product Thinking: Did they think about users, business impact, not just code?
- Decision Making: Did they make technical architecture decisions with real stakes?
- Critical Thinking: Research, analysis, nuanced problem framing?

Score these dimensions (0-100):
- communication, leadership, ownership, critical_thinking, problem_solving,
  collaboration, mentoring, decision_making, adaptability, product_thinking

Provide:
- overall_score (0-100)
- confidence (0.0-1.0)
- reasoning (2-4 sentences)
- strongest_trait: the single most evident transferable skill
- weakest_trait: the dimension with least evidence
- behavioural_evidence (list of specific behaviours observed from the profile)
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class TransferableScoreResult(ScoreResult):
    """Scoring result for the Transferable Skills dimension."""

    scorer_name: str = "TransferableScorer"
    strongest_trait: str = Field("", description="Most clearly evidenced transferable skill")
    weakest_trait: str = Field("", description="Transferable skill with least evidence")
    behavioural_evidence: list[str] = Field(
        default_factory=list,
        description="Specific behaviours observed that support the scores",
    )


class TransferableScorer(BaseScorer):
    """
    Infers soft skills and transferable competencies from behavioural evidence.

    Never accepts self-reported claims. All scores are grounded in observable
    actions: leading projects, writing documentation, mentoring, pivoting.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=TRANSFERABLE_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.2)
        logger.info("TransferableScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "TransferableScorer"

    def score(self, candidate: Any, job_profile: Any) -> TransferableScoreResult:
        """
        Score transferable and soft skills from behavioural evidence.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        TransferableScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "TransferableScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Build a behavioural signal-rich context
        behaviour_lines: list[str] = []

        for lead in getattr(candidate, 'leadership', []):
            behaviour_lines.append(
                f"Leadership: {lead.position} at {lead.club or lead.organisation} — {lead.impact or ''}"
            )
        for vol in getattr(candidate, 'volunteer', []):
            behaviour_lines.append(
                f"Volunteer: {vol.role} at {vol.organisation} — {vol.impact or ''}"
            )
        for ach in getattr(candidate, 'achievements', []):
            behaviour_lines.append(
                f"Achievement [{ach.category}]: {ach.title} — {ach.description or ''}"
            )
        for exp in getattr(candidate, 'experience', []):
            for resp in (exp.responsibilities or [])[:3]:
                behaviour_lines.append(f"Responsibility @ {exp.company}: {resp}")
            for ach in (exp.achievements or [])[:2]:
                behaviour_lines.append(f"Achievement @ {exp.company}: {ach}")
        for proj in getattr(candidate, 'projects', []):
            if proj.contribution:
                behaviour_lines.append(f"Project Contribution ({proj.name}): {proj.contribution}")
            if proj.role and 'lead' in (proj.role or '').lower():
                behaviour_lines.append(f"Led project: {proj.name}")

        behaviour_context = "\n".join(behaviour_lines) or "No specific behavioural signals found."

        prompt = (
            f"JOB PROFILE (hiring signals and soft skill expectations):\n{jd_text}\n\n"
            f"CANDIDATE OVERVIEW:\n{cand_text}\n\n"
            f"BEHAVIOURAL EVIDENCE:\n{behaviour_context}\n\n"
            "Infer transferable skills from behaviour ONLY. Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return TransferableScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        dimension_scores: dict[str, float] = {}
        for dim in TRANSFERABLE_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 40)))

        weighted = weighted_average(list(dimension_scores.values()), list(TRANSFERABLE_DIMENSION_WEIGHTS.values()))
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        final_score = 0.65 * weighted + 0.35 * llm_overall

        return TransferableScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.6)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            strongest_trait=str(raw.get("strongest_trait", "")),
            weakest_trait=str(raw.get("weakest_trait", "")),
            behavioural_evidence=list(raw.get("behavioural_evidence", [])),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> TransferableScoreResult:
    """Module-level convenience function."""
    scorer = TransferableScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
