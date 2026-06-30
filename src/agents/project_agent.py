"""
src/agents/project_agent.py
============================
Project Intelligence Agent

Interprets ``ProjectScoreResult`` and generates recruiter-friendly insights
about a candidate's project portfolio.

This agent does NOT call Gemini. It is a pure interpreter of the already-
computed ``ProjectScoreResult``. It surfaces the most signal-rich information
in a format that the Recruiter Agent and Dashboard can consume directly.

Responsibilities:
    - Interpret project scores into recruiter language
    - Highlight the strongest projects with evidence
    - Surface innovation, technical depth, deployment maturity
    - Generate interview questions about specific projects
    - Produce a project portfolio narrative

Author  : Resume Intelligence Engine — Agent Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import clamp_score

logger = logging.getLogger("project_agent")


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------


class ProjectInsight(BaseModel):
    """Recruiter-friendly insight about a single project."""

    project_name: str
    headline: str = Field(description="One-sentence recruiter pitch for this project")
    technical_depth_verdict: str = Field(description="'exceptional' | 'strong' | 'moderate' | 'basic'")
    innovation_signal: str = Field(description="What is innovative or unique about this project")
    deployment_signal: str = Field(description="Evidence of production/deployment maturity")
    interview_angle: str = Field(description="What to ask the candidate about this project")


class ProjectAgentReport(BaseModel):
    """
    Full project intelligence report produced by the ProjectAgent.

    Consumed by RecruiterAgent and Dashboard.
    """

    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None

    # Portfolio-level summary
    portfolio_verdict: str = Field(
        description="'exceptional' | 'strong' | 'solid' | 'average' | 'weak'"
    )
    portfolio_headline: str = Field(description="One-sentence portfolio pitch for recruiter")
    portfolio_narrative: str = Field(description="2-3 sentence narrative on portfolio quality")

    # Top project
    top_project_name: Optional[str] = None
    top_project_insight: Optional[ProjectInsight] = None

    # Per-project insights
    project_insights: list[ProjectInsight] = Field(default_factory=list)

    # Signal lists
    strongest_signals: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(default_factory=list)

    # Numeric summary (echoed from ProjectScoreResult for convenience)
    portfolio_score: float = Field(0.0, ge=0.0, le=100.0)
    portfolio_depth: float = Field(0.0, ge=0.0, le=100.0)
    portfolio_breadth: float = Field(0.0, ge=0.0, le=100.0)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ProjectAgent:
    """
    Interprets ``ProjectScoreResult`` into recruiter-facing intelligence.

    No LLM calls — pure rule-based interpretation of existing scores.
    Produces a ``ProjectAgentReport`` for consumption by RecruiterAgent.

    Usage
    -----
    >>> agent = ProjectAgent()
    >>> report = agent.interpret(candidate, project_score_result)
    >>> print(report.portfolio_headline)
    """

    def __init__(self) -> None:
        logger.info("ProjectAgent initialised (no-LLM interpreter).")

    def interpret(
        self,
        candidate: Any,
        project_result: Any,  # ProjectScoreResult
    ) -> ProjectAgentReport:
        """
        Interpret a ``ProjectScoreResult`` into recruiter-friendly intelligence.

        Parameters
        ----------
        candidate : Candidate
        project_result : ProjectScoreResult

        Returns
        -------
        ProjectAgentReport
        """
        try:
            return self._build_report(candidate, project_result)
        except Exception as exc:  # noqa: BLE001
            logger.error("ProjectAgent.interpret failed: %s", exc)
            return ProjectAgentReport(
                candidate_id=getattr(candidate, "candidate_id", None),
                candidate_name=getattr(candidate, "name", None),
                portfolio_verdict="unknown",
                portfolio_headline="Project analysis unavailable.",
                portfolio_narrative=f"Analysis failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_report(self, candidate: Any, pr: Any) -> ProjectAgentReport:
        score = getattr(pr, "score", 0.0)
        depth = getattr(pr, "portfolio_depth", 0.0)
        breadth = getattr(pr, "portfolio_breadth", 0.0)
        top_project = getattr(pr, "top_project", None)
        projects = getattr(candidate, "projects", [])
        positive = getattr(pr, "positive_signals", [])
        negative = getattr(pr, "negative_signals", [])
        dim_scores = getattr(pr, "dimension_scores", {})

        verdict = self._score_to_verdict(score)
        headline = self._build_headline(candidate, score, top_project, verdict)
        narrative = self._build_narrative(candidate, score, depth, breadth, dim_scores, positive, negative)
        interview_questions = self._build_interview_questions(candidate, dim_scores, negative)
        project_insights = self._build_project_insights(projects, dim_scores)

        top_insight = next((p for p in project_insights if p.project_name == top_project), None)
        if not top_insight and project_insights:
            top_insight = project_insights[0]

        return ProjectAgentReport(
            candidate_id=getattr(candidate, "candidate_id", None),
            candidate_name=getattr(candidate, "name", None),
            portfolio_verdict=verdict,
            portfolio_headline=headline,
            portfolio_narrative=narrative,
            top_project_name=top_project,
            top_project_insight=top_insight,
            project_insights=project_insights[:5],
            strongest_signals=list(positive[:5]),
            concerns=list(negative[:4]),
            interview_questions=interview_questions,
            portfolio_score=round(clamp_score(score), 1),
            portfolio_depth=round(clamp_score(depth), 1),
            portfolio_breadth=round(clamp_score(breadth), 1),
        )

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        if score >= 80:
            return "exceptional"
        if score >= 65:
            return "strong"
        if score >= 50:
            return "solid"
        if score >= 35:
            return "average"
        return "weak"

    @staticmethod
    def _build_headline(candidate: Any, score: float, top_project: Optional[str], verdict: str) -> str:
        name = getattr(candidate, "name", "Candidate") or "Candidate"
        count = len(getattr(candidate, "projects", []))
        if count == 0:
            return f"{name} has no listed projects."
        if score >= 75:
            tp = f" — standout project: {top_project}." if top_project else "."
            return f"{name} brings a {verdict} portfolio of {count} project(s){tp}"
        if score >= 50:
            return f"{name} has a {verdict} portfolio of {count} project(s) with room for growth."
        return f"{name} has {count} project(s) but the portfolio lacks depth for this role."

    @staticmethod
    def _build_narrative(
        candidate: Any,
        score: float,
        depth: float,
        breadth: float,
        dim_scores: dict,
        positive: list,
        negative: list,
    ) -> str:
        name = getattr(candidate, "name", "The candidate") or "The candidate"
        parts: list[str] = []

        arch = dim_scores.get("architecture", 50)
        deploy = dim_scores.get("deployment", 50)
        innov = dim_scores.get("innovation", 50)

        parts.append(
            f"{name}'s project portfolio scores {score:.0f}/100 overall, "
            f"with depth {depth:.0f}/100 and breadth {breadth:.0f}/100."
        )
        if arch >= 70:
            parts.append("Architecture quality is notably strong.")
        if deploy >= 65:
            parts.append("Evidence of deployment maturity is present.")
        if innov >= 70:
            parts.append("Several projects show genuine technical innovation.")
        if positive:
            parts.append(positive[0])
        if negative:
            parts.append(f"Key gap: {negative[0]}")

        return " ".join(parts[:4])

    @staticmethod
    def _build_interview_questions(candidate: Any, dim_scores: dict, negative: list) -> list[str]:
        questions: list[str] = []
        projects = getattr(candidate, "projects", [])

        if projects:
            p = projects[0]
            if p.name:
                questions.append(f"Walk me through the architecture decisions in your '{p.name}' project.")
            if p.problem_solved:
                questions.append(f"How did you define the problem statement for '{p.name or 'your top project'}'?")

        if dim_scores.get("scalability", 100) < 55:
            questions.append("How would you scale your most complex project to 10x the current load?")
        if dim_scores.get("testing", 100) < 55:
            questions.append("What is your testing strategy for production systems?")
        if dim_scores.get("deployment", 100) < 55:
            questions.append("Describe your experience deploying applications to production environments.")
        if dim_scores.get("collaboration", 100) < 55:
            questions.append("Tell me about a project you built with a team — what was your specific role?")
        if negative:
            questions.append(f"Your profile shows '{negative[-1]}' — can you elaborate on this?")

        return questions[:6]

    @staticmethod
    def _build_project_insights(projects: list, dim_scores: dict) -> list[ProjectInsight]:
        insights: list[ProjectInsight] = []
        deploy_score = dim_scores.get("deployment", 50)
        innov_score = dim_scores.get("innovation", 50)
        depth_score = dim_scores.get("technical_depth", 50)

        depth_verdict = (
            "exceptional" if depth_score >= 80
            else "strong" if depth_score >= 65
            else "moderate" if depth_score >= 45
            else "basic"
        )

        for proj in projects[:5]:
            name = proj.name or "Unnamed Project"
            tech_str = ", ".join(
                (proj.technologies or [])[:4]
                + (proj.frameworks or [])[:2]
            )
            headline = f"{name} — {tech_str}." if tech_str else f"{name}."
            if proj.results:
                headline = f"{name}: {proj.results[:80]}"

            innovation_signal = (
                proj.description[:100] if innov_score >= 65 and proj.description
                else "Standard implementation, limited innovation signals."
            )
            deploy_signal = (
                proj.deployment or "Production deployment evidence present."
                if deploy_score >= 60
                else "No clear deployment evidence found."
            )

            interview_angle = (
                f"Deep-dive into '{name}': what was the hardest technical challenge?"
                if proj.challenges
                else f"Ask about architecture choices in '{name}'."
            )

            insights.append(ProjectInsight(
                project_name=name,
                headline=headline[:120],
                technical_depth_verdict=depth_verdict,
                innovation_signal=innovation_signal[:120],
                deployment_signal=deploy_signal[:100],
                interview_angle=interview_angle[:120],
            ))

        return insights
