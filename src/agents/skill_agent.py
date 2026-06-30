"""
src/agents/skill_agent.py
==========================
Skill Intelligence Agent

Interprets ``SkillScoreResult`` and generates actionable recruiter-facing insights
about a candidate's technical skill profile.

No LLM calls — pure interpreter of existing scores.

Responsibilities:
    - Summarise technical strengths in recruiter language
    - Identify rare or highly valuable skills
    - Surface skill gaps vs the role requirements
    - Highlight learning progression
    - Generate targeted interview questions per skill gap

Author  : Resume Intelligence Engine — Agent Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import clamp_score

logger = logging.getLogger("skill_agent")


class SkillAgentReport(BaseModel):
    """Recruiter-facing skill intelligence report."""

    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None

    # Summary
    skill_verdict: str = Field(description="'exceptional' | 'strong' | 'adequate' | 'weak'")
    skill_headline: str = Field(description="One-sentence skill pitch")
    skill_narrative: str = Field(description="2-3 sentence narrative for recruiter")

    # Signals
    rare_skills: list[str] = Field(default_factory=list, description="Uncommon, high-value skills")
    core_skill_strengths: list[str] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list, description="Required skills the candidate lacks")
    nice_to_have_gaps: list[str] = Field(default_factory=list)
    gap_severity: str = Field("unknown", description="'critical' | 'moderate' | 'minor' | 'none'")

    # Interview
    skill_interview_questions: list[str] = Field(default_factory=list)

    # Scores (echoed)
    skill_score: float = Field(0.0, ge=0.0, le=100.0)
    raw_overlap_ratio: float = Field(0.0, ge=0.0, le=1.0)


# Rare/high-value skills that signal advanced expertise
_RARE_SKILLS: set[str] = {
    "rust", "cuda", "triton", "jax", "tpu", "rlhf", "dpo", "ray",
    "kubeflow", "mlflow", "onnx", "tensorrt", "vllm", "lora", "qlora",
    "grpc", "ebpf", "webassembly", "formal verification", "theorem proving",
    "compiler design", "llvm", "cuda kernels", "distributed training",
    "reinforcement learning", "causal inference", "differential privacy",
    "federated learning", "quantization", "model pruning", "system design",
    "kafka", "flink", "spark", "iceberg", "delta lake", "clickhouse",
    "kubernetes", "istio", "envoy", "pulumi", "terraform",
}


class SkillAgent:
    """
    Interprets ``SkillScoreResult`` into recruiter-facing skill intelligence.

    Pure rule-based interpreter — no LLM calls.

    Usage
    -----
    >>> agent = SkillAgent()
    >>> report = agent.interpret(candidate, skill_result, job_profile)
    >>> print(report.skill_headline)
    """

    def __init__(self) -> None:
        logger.info("SkillAgent initialised (no-LLM interpreter).")

    def interpret(
        self,
        candidate: Any,
        skill_result: Any,  # SkillScoreResult
        job_profile: Any,
    ) -> SkillAgentReport:
        """
        Interpret a ``SkillScoreResult`` into recruiter skill intelligence.

        Parameters
        ----------
        candidate : Candidate
        skill_result : SkillScoreResult
        job_profile : JobProfile

        Returns
        -------
        SkillAgentReport
        """
        try:
            return self._build_report(candidate, skill_result, job_profile)
        except Exception as exc:  # noqa: BLE001
            logger.error("SkillAgent.interpret failed: %s", exc)
            return SkillAgentReport(
                candidate_id=getattr(candidate, "candidate_id", None),
                candidate_name=getattr(candidate, "name", None),
                skill_verdict="unknown",
                skill_headline="Skill analysis unavailable.",
                skill_narrative=f"Analysis failed: {exc}",
            )

    def _build_report(
        self,
        candidate: Any,
        sr: Any,
        job_profile: Any,
    ) -> SkillAgentReport:
        score = getattr(sr, "score", 0.0)
        overlap = getattr(sr, "raw_overlap_ratio", 0.0)
        matched_required = getattr(sr, "matched_required_skills", []) or []
        missing_required = getattr(sr, "missing_required_skills", []) or []
        matched_preferred = getattr(sr, "matched_preferred_skills", []) or []
        gap_severity = getattr(sr, "skill_gap_severity", "unknown") or "unknown"
        positive = getattr(sr, "positive_signals", []) or []
        negative = getattr(sr, "negative_signals", []) or []
        dim_scores = getattr(sr, "dimension_scores", {}) or {}

        # Detect rare skills
        all_cand_skills = self._flatten_skills(candidate)
        rare = [
            s for s in all_cand_skills
            if s.lower() in _RARE_SKILLS
        ]

        verdict = self._score_to_verdict(score)
        headline = self._build_headline(candidate, score, verdict, missing_required)
        narrative = self._build_narrative(
            candidate, score, overlap, matched_required,
            missing_required, gap_severity, positive, negative
        )
        questions = self._build_questions(missing_required, dim_scores, all_cand_skills)

        return SkillAgentReport(
            candidate_id=getattr(candidate, "candidate_id", None),
            candidate_name=getattr(candidate, "name", None),
            skill_verdict=verdict,
            skill_headline=headline,
            skill_narrative=narrative,
            rare_skills=rare[:8],
            core_skill_strengths=list(matched_required[:8]),
            critical_gaps=list(missing_required[:6]),
            nice_to_have_gaps=[
                s.name for s in getattr(job_profile, "preferred_skills", [])
                if s.name.lower() not in {x.lower() for x in all_cand_skills}
            ][:5],
            gap_severity=gap_severity,
            skill_interview_questions=questions,
            skill_score=round(clamp_score(score), 1),
            raw_overlap_ratio=round(clamp_score(overlap, 0.0, 1.0), 4),
        )

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score >= 80:
            return "exceptional"
        if score >= 65:
            return "strong"
        if score >= 45:
            return "adequate"
        return "weak"

    @staticmethod
    def _build_headline(
        candidate: Any,
        score: float,
        verdict: str,
        missing: list,
    ) -> str:
        name = getattr(candidate, "name", "Candidate") or "Candidate"
        if score >= 75:
            return f"{name} demonstrates a {verdict} technical skill profile well-aligned to role requirements."
        if score >= 50:
            return f"{name} has an {verdict} skill profile with {len(missing)} key gap(s) to address."
        return f"{name} has significant skill gaps ({len(missing)} required skill(s) missing) for this role."

    @staticmethod
    def _build_narrative(
        candidate: Any,
        score: float,
        overlap: float,
        matched: list,
        missing: list,
        severity: str,
        positive: list,
        negative: list,
    ) -> str:
        name = getattr(candidate, "name", "The candidate") or "The candidate"
        parts: list[str] = [
            f"{name} scores {score:.0f}/100 on skills with a {overlap*100:.0f}% skill overlap with requirements."
        ]
        if matched:
            parts.append(f"Confirmed strengths include: {', '.join(matched[:4])}{'...' if len(matched) > 4 else ''}.")
        if missing:
            parts.append(
                f"{len(missing)} required skill(s) not evidenced: "
                f"{', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}. "
                f"Gap severity: {severity}."
            )
        elif score >= 70:
            parts.append("All required skills are evidenced in the profile.")
        if positive:
            parts.append(positive[0])
        return " ".join(parts[:4])

    @staticmethod
    def _build_questions(missing: list, dim_scores: dict, all_skills: list) -> list[str]:
        questions: list[str] = []
        for gap in missing[:3]:
            questions.append(f"You appear to lack '{gap}' — do you have any experience with it?")
        if dim_scores.get("skill_depth", 100) < 55:
            questions.append(
                "Tell me about a time you used your core skills to solve a genuinely hard problem."
            )
        if dim_scores.get("modern_tech_adoption", 100) < 55:
            questions.append("Which new technologies have you adopted in the last 12 months?")
        if not questions:
            questions.append("Walk me through how you stay current with emerging technologies.")
        return questions[:6]

    @staticmethod
    def _flatten_skills(candidate: Any) -> list[str]:
        s = getattr(candidate, "skills", None)
        if s is None:
            return []
        all_s = (
            getattr(s, "programming_languages", [])
            + getattr(s, "frameworks", [])
            + getattr(s, "libraries", [])
            + getattr(s, "databases", [])
            + getattr(s, "cloud", [])
            + getattr(s, "devops", [])
            + getattr(s, "ai_ml", [])
            + getattr(s, "data_science", [])
            + getattr(s, "tools", [])
        )
        return all_s
