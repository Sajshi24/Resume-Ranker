"""
src/agents/recruiter_agent.py
==============================
Recruiter Intelligence Agent — Final Decision Maker

The RecruiterAgent is the top-level decision-making agent in the pipeline.
It consumes ALL scoring results AND agent reports to produce a complete,
evidence-grounded hiring recommendation.

Architecture
------------
Input:
    Candidate          — from resume_parser
    JobProfile         — from jd_parser
    FinalScoreResult   — from final_score
    ProjectAgentReport — from project_agent
    SkillAgentReport   — from skill_agent
    GrowthAgentReport  — from growth_agent

Output:
    RecruiterReport    — fully explainable hiring decision

LLM Usage
---------
One Gemini call to generate the executive summary narrative.
ALL other fields are computed deterministically from existing results.

Author  : Resume Intelligence Engine — Agent Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.config import get_gemini_model, get_generation_config
from src.utils.helpers import (
    call_gemini,
    candidate_to_text,
    clamp_confidence,
    clamp_score,
    job_profile_to_text,
    parse_json_from_llm,
)

logger = logging.getLogger("recruiter_agent")

# ---------------------------------------------------------------------------
# Hiring recommendation colour / label map
# ---------------------------------------------------------------------------

REC_LABELS: dict[str, str] = {
    "strong_hire": "STRONG HIRE",
    "hire": "HIRE",
    "borderline": "BORDERLINE",
    "no_hire": "NO HIRE",
    "strong_no_hire": "STRONG NO HIRE",
    "unknown": "UNKNOWN",
}

# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    """A targeted interview question with rationale."""

    question: str
    rationale: str = Field(description="Why this question is important for this candidate")
    area: str = Field(description="'technical' | 'behavioural' | 'culture' | 'clarification'")


class RecruiterReport(BaseModel):
    """
    The complete, evidence-grounded hiring decision produced by the Recruiter Agent.

    This is the terminal output of the entire pipeline. Consumed by the Dashboard
    and saved to outputs/final_results.json.
    """

    # Identity
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    role_title: Optional[str] = None
    company_name: Optional[str] = None
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    # ---- Core Decision ----
    rank: int = Field(0, description="Rank among all candidates (1 = best)")
    overall_score: float = Field(0.0, ge=0.0, le=100.0)
    potential_score: float = Field(0.0, ge=0.0, le=100.0, description="Growth + Learning composite")
    hiring_recommendation: str = Field("unknown")
    hiring_recommendation_label: str = Field("")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # ---- Executive Summary ----
    executive_summary: str = Field("", description="2-3 sentence executive summary for recruiter")
    one_liner: str = Field("", description="Single sentence decision statement")

    # ---- Explainability ----
    strengths: list[str] = Field(default_factory=list, description="Top 3-5 hiring arguments")
    weaknesses: list[str] = Field(default_factory=list, description="Top 3-5 gaps or concerns")
    risk_factors: list[str] = Field(default_factory=list)
    growth_opportunity: str = Field("", description="Long-term potential narrative")
    fit_verdict: str = Field("", description="Overall fit in one phrase")

    # ---- Score Breakdown ----
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    # ---- Interview Plan ----
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)

    # ---- Agent Narratives ----
    project_narrative: str = Field("")
    skill_narrative: str = Field("")
    growth_narrative: str = Field("")

    # ---- Full reasoning chain ----
    overall_reasoning: str = Field("", description="Full LLM reasoning for the recommendation")

    # ---- Diagnostics ----
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System Prompt (one LLM call per candidate)
# ---------------------------------------------------------------------------

RECRUITER_SYSTEM_PROMPT = """
You are a world-class senior technical recruiter with 15+ years of experience
at top technology companies. You synthesise detailed AI-generated analysis
into concise, actionable hiring intelligence.

Your task: generate an executive summary and one-liner for a hiring recommendation.

You will receive:
- The job profile
- The candidate profile
- Scores from 7 AI analysis modules
- Insights from project, skill, and growth intelligence agents

Provide:
- executive_summary: 2-3 sentences that a busy hiring manager would read. Be specific, evidence-based, and direct.
- one_liner: One sentence that captures the entire recommendation.
- fit_verdict: 3-5 words describing overall fit (e.g. "Strong technical, moderate domain fit")

Rules:
- Reference specific evidence, not generic platitudes.
- Be honest about gaps; do not over-sell.
- Return ONLY valid JSON. No markdown, no explanation.
""".strip()


# ---------------------------------------------------------------------------
# Recruiter Agent
# ---------------------------------------------------------------------------


class RecruiterAgent:
    """
    Decision-making agent that synthesises all pipeline outputs into a
    final ``RecruiterReport``.

    Uses one Gemini call per candidate to generate the executive summary.
    All other fields are computed deterministically.

    Usage
    -----
    >>> agent = RecruiterAgent()
    >>> report = agent.decide(
    ...     candidate, job_profile,
    ...     final_result, project_report, skill_report, growth_report
    ... )
    >>> print(report.hiring_recommendation, report.executive_summary)
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=RECRUITER_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.2, max_output_tokens=1024)
        logger.info("RecruiterAgent initialised | model=%s", model_name)

    def decide(
        self,
        candidate: Any,
        job_profile: Any,
        final_result: Any,           # FinalScoreResult
        project_report: Any,         # ProjectAgentReport
        skill_report: Any,           # SkillAgentReport
        growth_report: Any,          # GrowthAgentReport
        rank: int = 0,
    ) -> RecruiterReport:
        """
        Produce a complete ``RecruiterReport`` for a single candidate.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile
        final_result : FinalScoreResult
        project_report : ProjectAgentReport
        skill_report : SkillAgentReport
        growth_report : GrowthAgentReport
        rank : int
            Final rank position (set by pipeline after batch scoring).

        Returns
        -------
        RecruiterReport
        """
        errors: list[str] = []

        try:
            # ---- Core scores ----
            overall = getattr(final_result, "score", 0.0)
            growth_s = getattr(final_result, "growth_score", 0.0)
            learning_s = getattr(final_result, "learning_score", 0.0)
            potential = round((growth_s + learning_s) / 2, 2)
            confidence = getattr(final_result, "confidence", 0.5)
            hiring_rec = getattr(final_result, "hiring_recommendation", "unknown")

            score_breakdown = {
                "Overall": round(overall, 1),
                "Projects": round(getattr(final_result, "project_score", 0.0), 1),
                "Domain Fit": round(getattr(final_result, "domain_score", 0.0), 1),
                "Skills": round(getattr(final_result, "skill_score", 0.0), 1),
                "Learning": round(learning_s, 1),
                "Soft Skills": round(getattr(final_result, "transferable_score", 0.0), 1),
                "Growth": round(growth_s, 1),
                "Semantic Fit": round(getattr(final_result, "semantic_score", 0.0), 1),
            }

            # ---- Combine signals from all agents ----
            strengths = list(getattr(final_result, "strengths", []))
            if getattr(project_report, "strongest_signals", []):
                strengths += project_report.strongest_signals[:2]
            if getattr(skill_report, "core_skill_strengths", []):
                strengths += skill_report.core_skill_strengths[:2]
            strengths = list(dict.fromkeys(strengths))[:6]  # dedup, cap

            weaknesses = list(getattr(final_result, "weaknesses", []))
            if getattr(skill_report, "critical_gaps", []):
                weaknesses += [f"Missing skill: {g}" for g in skill_report.critical_gaps[:2]]
            if getattr(growth_report, "risk_factors", []):
                weaknesses += growth_report.risk_factors[:1]
            weaknesses = list(dict.fromkeys(weaknesses))[:6]

            risk_factors = list(getattr(final_result, "risk_factors", []))
            if getattr(growth_report, "risk_factors", []):
                risk_factors += growth_report.risk_factors
            risk_factors = list(dict.fromkeys(risk_factors))[:5]

            # ---- Interview questions ----
            interview_qs = self._build_interview_questions(
                project_report, skill_report, final_result
            )

            # ---- LLM: executive summary ----
            llm_data = self._get_executive_summary(
                candidate, job_profile, final_result,
                project_report, skill_report, growth_report,
                overall, score_breakdown,
            )
            if llm_data.get("error"):
                errors.append(llm_data["error"])

            executive_summary = str(llm_data.get("executive_summary", ""))
            if not executive_summary:
                executive_summary = self._fallback_summary(
                    candidate, overall, hiring_rec, strengths, weaknesses
                )

            one_liner = str(llm_data.get("one_liner", ""))
            if not one_liner:
                one_liner = self._fallback_one_liner(candidate, hiring_rec, overall)

            fit_verdict = str(llm_data.get("fit_verdict", ""))
            if not fit_verdict:
                fit_verdict = self._fallback_fit_verdict(score_breakdown)

            return RecruiterReport(
                candidate_id=getattr(candidate, "candidate_id", None),
                candidate_name=getattr(candidate, "name", None),
                role_title=getattr(job_profile, "role_title", None),
                company_name=getattr(job_profile, "company_name", None),
                rank=rank,
                overall_score=round(clamp_score(overall), 2),
                potential_score=round(clamp_score(potential), 2),
                hiring_recommendation=hiring_rec,
                hiring_recommendation_label=REC_LABELS.get(hiring_rec, hiring_rec),
                confidence=round(clamp_confidence(confidence), 4),
                executive_summary=executive_summary,
                one_liner=one_liner,
                strengths=strengths,
                weaknesses=weaknesses,
                risk_factors=risk_factors,
                growth_opportunity=str(getattr(final_result, "growth_opportunity", "")),
                fit_verdict=fit_verdict,
                score_breakdown=score_breakdown,
                interview_questions=interview_qs,
                focus_areas=list(getattr(final_result, "interview_focus_areas", []))[:6],
                project_narrative=getattr(project_report, "portfolio_narrative", ""),
                skill_narrative=getattr(skill_report, "skill_narrative", ""),
                growth_narrative=getattr(growth_report, "growth_narrative", ""),
                overall_reasoning=str(getattr(final_result, "overall_reasoning", "")),
                is_valid=len(errors) == 0,
                errors=errors,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("RecruiterAgent.decide failed: %s", exc)
            return RecruiterReport(
                candidate_id=getattr(candidate, "candidate_id", None),
                candidate_name=getattr(candidate, "name", None),
                rank=rank,
                overall_score=0.0,
                hiring_recommendation="unknown",
                executive_summary=f"Report generation failed: {exc}",
                one_liner="Unable to generate recommendation.",
                is_valid=False,
                errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_executive_summary(
        self,
        candidate: Any,
        job_profile: Any,
        final_result: Any,
        project_report: Any,
        skill_report: Any,
        growth_report: Any,
        overall: float,
        score_breakdown: dict,
    ) -> dict:
        """Call Gemini for the executive summary (1 call per candidate)."""
        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        scores_text = "\n".join(f"  {k}: {v}/100" for k, v in score_breakdown.items())
        project_h = getattr(project_report, "portfolio_headline", "")
        skill_h = getattr(skill_report, "skill_headline", "")
        growth_h = getattr(growth_report, "growth_headline", "")
        rec = getattr(final_result, "hiring_recommendation", "unknown")

        prompt = (
            f"JOB PROFILE:\n{jd_text}\n\n"
            f"CANDIDATE PROFILE:\n{cand_text}\n\n"
            f"AI SCORING SUMMARY (overall: {overall:.1f}/100, recommendation: {rec}):\n"
            f"{scores_text}\n\n"
            f"AGENT INSIGHTS:\n"
            f"  Projects: {project_h}\n"
            f"  Skills:   {skill_h}\n"
            f"  Growth:   {growth_h}\n\n"
            "Generate executive_summary, one_liner, and fit_verdict. "
            "Return a single valid JSON object."
        )

        try:
            raw = call_gemini(self._model, prompt, self._gen_config)
            if raw.strip():
                return parse_json_from_llm(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RecruiterAgent LLM call failed: %s", exc)

        return {"error": "LLM unavailable"}

    @staticmethod
    def _build_interview_questions(
        project_report: Any,
        skill_report: Any,
        final_result: Any,
    ) -> list[InterviewQuestion]:
        questions: list[InterviewQuestion] = []

        # From project agent
        for q in getattr(project_report, "interview_questions", [])[:2]:
            questions.append(InterviewQuestion(
                question=q,
                rationale="Project depth validation",
                area="technical",
            ))

        # From skill agent
        for q in getattr(skill_report, "skill_interview_questions", [])[:2]:
            questions.append(InterviewQuestion(
                question=q,
                rationale="Skill gap clarification",
                area="technical",
            ))

        # From final score focus areas
        for area in getattr(final_result, "interview_focus_areas", [])[:2]:
            questions.append(InterviewQuestion(
                question=f"Tell me about your experience with: {area}",
                rationale=f"Identified focus area from holistic analysis",
                area="clarification",
            ))

        # Standard behavioural
        questions.append(InterviewQuestion(
            question="Describe the most technically challenging problem you've solved. What was your approach?",
            rationale="Evaluates problem-solving depth and communication",
            area="behavioural",
        ))

        return questions[:8]

    @staticmethod
    def _fallback_summary(
        candidate: Any,
        score: float,
        rec: str,
        strengths: list,
        weaknesses: list,
    ) -> str:
        name = getattr(candidate, "name", "The candidate") or "The candidate"
        s = strengths[0] if strengths else "technical background"
        w = weaknesses[0] if weaknesses else "some gaps"
        return (
            f"{name} scores {score:.0f}/100 overall (recommendation: {rec.replace('_', ' ')}). "
            f"Key strength: {s}. "
            f"Primary concern: {w}."
        )

    @staticmethod
    def _fallback_one_liner(candidate: Any, rec: str, score: float) -> str:
        name = getattr(candidate, "name", "Candidate") or "Candidate"
        label = REC_LABELS.get(rec, rec)
        return f"{name}: {label} — Overall score {score:.0f}/100."

    @staticmethod
    def _fallback_fit_verdict(score_breakdown: dict) -> str:
        skill = score_breakdown.get("Skills", 0)
        domain = score_breakdown.get("Domain Fit", 0)
        growth = score_breakdown.get("Growth", 0)
        parts = []
        if skill >= 70:
            parts.append("strong technical")
        elif skill >= 50:
            parts.append("moderate technical")
        else:
            parts.append("weak technical")
        if domain >= 65:
            parts.append("strong domain")
        elif domain >= 45:
            parts.append("adjacent domain")
        else:
            parts.append("domain mismatch")
        if growth >= 65:
            parts.append("high potential")
        return ", ".join(parts)
