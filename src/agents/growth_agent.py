"""
src/agents/growth_agent.py
===========================
Growth Intelligence Agent

Interprets ``GrowthScoreResult`` into recruiter-facing growth potential intelligence.

No LLM calls — pure interpreter of existing scores.

Responsibilities:
    - Interpret growth score into plain-language trajectory prediction
    - Surface career acceleration signals
    - Explain learning velocity evidence
    - Predict future potential ceiling
    - Highlight risk factors for long-term growth

Author  : Resume Intelligence Engine — Agent Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import clamp_score

logger = logging.getLogger("growth_agent")


class GrowthAgentReport(BaseModel):
    """Recruiter-facing growth potential report."""

    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None

    # Verdict
    growth_verdict: str = Field(
        description="'high_flyer' | 'strong_growth' | 'steady' | 'plateau_risk' | 'stagnant'"
    )
    growth_headline: str = Field(description="One-sentence growth assessment for recruiter")
    growth_narrative: str = Field(description="2-3 sentence growth narrative")

    # Trajectory
    trajectory_direction: str = Field(
        description="'accelerating' | 'steady' | 'decelerating' | 'unknown'"
    )
    predicted_ceiling: str = Field(
        "", description="Estimated seniority ceiling from GrowthScorer"
    )
    two_year_prediction: str = Field("", description="Where this candidate could be in 2 years")

    # Evidence
    growth_catalysts: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    learning_velocity_signals: list[str] = Field(default_factory=list)
    career_acceleration_signals: list[str] = Field(default_factory=list)

    # Score (echoed)
    growth_score: float = Field(0.0, ge=0.0, le=100.0)


class GrowthAgent:
    """
    Interprets ``GrowthScoreResult`` into recruiter-facing growth intelligence.

    Pure rule-based interpreter — no LLM calls.

    Usage
    -----
    >>> agent = GrowthAgent()
    >>> report = agent.interpret(candidate, growth_result, learning_result)
    """

    def __init__(self) -> None:
        logger.info("GrowthAgent initialised (no-LLM interpreter).")

    def interpret(
        self,
        candidate: Any,
        growth_result: Any,   # GrowthScoreResult
        learning_result: Any, # LearningScoreResult
    ) -> GrowthAgentReport:
        """
        Interpret growth and learning results into a recruiter growth report.

        Parameters
        ----------
        candidate : Candidate
        growth_result : GrowthScoreResult
        learning_result : LearningScoreResult

        Returns
        -------
        GrowthAgentReport
        """
        try:
            return self._build_report(candidate, growth_result, learning_result)
        except Exception as exc:  # noqa: BLE001
            logger.error("GrowthAgent.interpret failed: %s", exc)
            return GrowthAgentReport(
                candidate_id=getattr(candidate, "candidate_id", None),
                candidate_name=getattr(candidate, "name", None),
                growth_verdict="unknown",
                growth_headline="Growth analysis unavailable.",
                growth_narrative=f"Analysis failed: {exc}",
                trajectory_direction="unknown",
            )

    def _build_report(
        self,
        candidate: Any,
        gr: Any,
        lr: Any,
    ) -> GrowthAgentReport:
        growth_score = getattr(gr, "score", 0.0)
        learning_score = getattr(lr, "score", 0.0)
        ceiling = getattr(gr, "potential_ceiling", "unknown") or "unknown"
        prediction = getattr(gr, "growth_prediction", "") or ""
        catalysts = getattr(gr, "growth_catalysts", []) or []
        risks = getattr(gr, "risk_factors", []) or []
        lr_signals = getattr(lr, "key_learning_signals", []) or []
        archetype = getattr(lr, "learning_archetype", "unknown") or "unknown"
        positive = getattr(gr, "positive_signals", []) or []
        dim_scores = getattr(gr, "dimension_scores", {}) or {}

        verdict = self._score_to_verdict(growth_score, learning_score)
        trajectory = self._infer_trajectory(dim_scores)
        headline = self._build_headline(candidate, verdict, trajectory, ceiling)
        narrative = self._build_narrative(
            candidate, growth_score, learning_score, archetype,
            catalysts, risks, prediction
        )
        accel_signals = self._career_acceleration_signals(candidate)

        return GrowthAgentReport(
            candidate_id=getattr(candidate, "candidate_id", None),
            candidate_name=getattr(candidate, "name", None),
            growth_verdict=verdict,
            growth_headline=headline,
            growth_narrative=narrative,
            trajectory_direction=trajectory,
            predicted_ceiling=ceiling,
            two_year_prediction=prediction[:300] if prediction else "",
            growth_catalysts=list(catalysts[:5]),
            risk_factors=list(risks[:4]),
            learning_velocity_signals=list(lr_signals[:5]),
            career_acceleration_signals=accel_signals,
            growth_score=round(clamp_score(growth_score), 1),
        )

    @staticmethod
    def _score_to_verdict(growth_score: float, learning_score: float) -> str:
        avg = (growth_score + learning_score) / 2
        if avg >= 78:
            return "high_flyer"
        if avg >= 62:
            return "strong_growth"
        if avg >= 45:
            return "steady"
        if avg >= 30:
            return "plateau_risk"
        return "stagnant"

    @staticmethod
    def _infer_trajectory(dim_scores: dict) -> str:
        agility = dim_scores.get("learning_agility", 50)
        accel = dim_scores.get("career_acceleration", 50)
        if agility >= 70 and accel >= 65:
            return "accelerating"
        if agility >= 50 and accel >= 45:
            return "steady"
        if agility < 40 or accel < 35:
            return "decelerating"
        return "steady"

    @staticmethod
    def _build_headline(
        candidate: Any,
        verdict: str,
        trajectory: str,
        ceiling: str,
    ) -> str:
        name = getattr(candidate, "name", "Candidate") or "Candidate"
        ceiling_str = f"; estimated ceiling: {ceiling}" if ceiling and ceiling != "unknown" else ""
        if verdict == "high_flyer":
            return f"{name} is a high-growth candidate with an {trajectory} trajectory{ceiling_str}."
        if verdict == "strong_growth":
            return f"{name} shows strong growth potential with a {trajectory} career direction{ceiling_str}."
        if verdict == "steady":
            return f"{name} demonstrates steady growth, reliable but not accelerating quickly{ceiling_str}."
        if verdict == "plateau_risk":
            return f"{name} shows plateau risk — growth signals are limited; probe for learning culture fit."
        return f"{name} shows limited growth signals; this may be a specialist hire without trajectory upside."

    @staticmethod
    def _build_narrative(
        candidate: Any,
        growth_score: float,
        learning_score: float,
        archetype: str,
        catalysts: list,
        risks: list,
        prediction: str,
    ) -> str:
        name = getattr(candidate, "name", "The candidate") or "The candidate"
        parts: list[str] = [
            f"{name} scores {growth_score:.0f}/100 on growth potential "
            f"and {learning_score:.0f}/100 on learning velocity."
        ]
        if archetype and archetype != "unknown":
            parts.append(f"Learning archetype: {archetype.replace('_', ' ')}.")
        if catalysts:
            parts.append(f"Key growth catalysts: {'; '.join(catalysts[:2])}.")
        if risks:
            parts.append(f"Risk factor: {risks[0]}.")
        return " ".join(parts[:4])

    @staticmethod
    def _career_acceleration_signals(candidate: Any) -> list[str]:
        signals: list[str] = []
        exp_list = getattr(candidate, "experience", [])
        if len(exp_list) >= 2:
            roles = [e.role for e in exp_list if e.role]
            if roles:
                signals.append(f"Progressed through {len(exp_list)} role(s): {' -> '.join(roles[-3:])}")
        certs = getattr(candidate, "certifications", [])
        if certs:
            signals.append(f"{len(certs)} professional certification(s) completed.")
        research = getattr(candidate, "research", [])
        if research:
            signals.append(f"{len(research)} research contribution(s) evidenced.")
        achievements = getattr(candidate, "achievements", [])
        high_value = [a for a in achievements if a.category in ("hackathon", "publication", "patent", "research")]
        if high_value:
            signals.append(f"{len(high_value)} high-signal achievement(s): {high_value[0].title or 'see profile'}.")
        return signals[:5]
