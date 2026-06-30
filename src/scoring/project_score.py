"""
src/scoring/project_score.py
=============================
Project Intelligence Scorer

Evaluates every project in a candidate's portfolio using Gemini reasoning.
Never uses keyword matching. Infers quality from contextual understanding.

Sub-dimensions scored (each 0-100):
    problem_complexity, technical_complexity, innovation, architecture,
    scalability, deployment, cloud_usage, testing, documentation,
    technical_depth, ai_usage, real_world_usage, business_impact,
    candidate_contribution, leadership, collaboration, engineering_quality

Output: ProjectScoreResult (extends ScoreResult)

Imports from:
    src.utils.config  — get_gemini_model, get_generation_config
    src.utils.helpers — candidate_to_text, score_via_gemini, clamp_score,
                        clamp_confidence, weighted_average
    src.scoring       — ScoreResult, BaseScorer

Author  : Resume Intelligence Engine — Scoring Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any, Optional

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

logger = logging.getLogger("project_score")

# ---------------------------------------------------------------------------
# Sub-dimension weights
# ---------------------------------------------------------------------------

PROJECT_DIMENSION_WEIGHTS: dict[str, float] = {
    "problem_complexity": 1.2,
    "technical_complexity": 1.5,
    "innovation": 1.3,
    "architecture": 1.2,
    "scalability": 1.0,
    "deployment": 0.8,
    "cloud_usage": 0.7,
    "testing": 0.8,
    "documentation": 0.6,
    "technical_depth": 1.5,
    "ai_usage": 1.0,
    "real_world_usage": 1.2,
    "business_impact": 1.3,
    "candidate_contribution": 1.4,
    "leadership": 0.9,
    "collaboration": 0.8,
    "engineering_quality": 1.2,
}

# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------


class ProjectScoreResult(ScoreResult):
    """
    Scoring result for the Project Intelligence dimension.

    In addition to the universal ``ScoreResult`` fields, exposes per-project
    breakdowns so the Recruiter Agent can reference specific projects.
    """

    scorer_name: str = "ProjectScorer"
    project_count: int = Field(0, description="Number of projects evaluated")
    top_project: Optional[str] = Field(None, description="Name of the highest-scoring project")
    per_project_scores: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Score breakdown per individual project",
    )
    portfolio_breadth: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Score for variety of project domains and technologies",
    )
    portfolio_depth: float = Field(
        0.0, ge=0.0, le=100.0,
        description="Score for technical depth across projects",
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

PROJECT_SYSTEM_PROMPT = """
You are an expert senior engineering manager evaluating candidate projects
for an AI-powered hiring intelligence engine.

Your task is to evaluate a candidate's technical projects against the job requirements.
You must REASON about quality, not match keywords.

For EACH project, score these dimensions (0-100):
- problem_complexity: How hard is the underlying problem being solved?
- technical_complexity: How sophisticated is the technical implementation?
- innovation: Is the approach novel, creative, or non-obvious?
- architecture: How well-designed and maintainable is the system?
- scalability: Does the design scale? Evidence of scale-aware thinking?
- deployment: Is there evidence of real deployment (prod/cloud/docker)?
- cloud_usage: Cloud services, managed infra, serverless, Kubernetes?
- testing: Unit tests, integration tests, CI/CD evidence?
- documentation: README, API docs, inline comments, architecture diagrams?
- technical_depth: Depth of understanding — not just using libraries but understanding them?
- ai_usage: Use of AI/ML/LLM components appropriate to the role?
- real_world_usage: Is this solving a genuine real-world problem with real users/data?
- business_impact: Measurable business or societal impact?
- candidate_contribution: What specifically did THIS candidate build?
- leadership: Did the candidate lead the project or team?
- collaboration: Evidence of team collaboration, open source contributions?
- engineering_quality: Code quality, patterns, best practices?

Also provide:
- overall_score (weighted summary, 0-100)
- confidence (0.0-1.0)
- reasoning (2-4 sentences on overall project quality vs role requirements)
- positive_signals (list of strings)
- negative_signals (list of strings)
- improvement_suggestions (list of strings)
- top_project (name of the best project)
- portfolio_breadth_score (variety across domains/tech, 0-100)
- portfolio_depth_score (depth of expertise, 0-100)

RULES:
- Return ONLY valid JSON. No markdown fences, no explanation.
- Do NOT penalise for missing sections — use available information.
- Infer deployment from context (e.g., mentions of AWS, Docker, heroku, vercel).
- A strong project doesn't need all dimensions — excellence in a few outweighs mediocrity in all.
- Compare projects to what a top 10% engineer at the target seniority level would build.
""".strip()


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ProjectScorer(BaseScorer):
    """
    Evaluates a candidate's project portfolio using Gemini reasoning.

    Scores each project individually, then aggregates to a portfolio score.
    Never uses keyword matching — quality is inferred from context.

    Usage
    -----
    >>> scorer = ProjectScorer()
    >>> result = scorer.score(candidate, job_profile)
    >>> print(result.score, result.top_project)
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=PROJECT_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.1, max_output_tokens=8192)
        logger.info("ProjectScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "ProjectScorer"

    def score(
        self,
        candidate: Any,
        job_profile: Any,
    ) -> ProjectScoreResult:
        """
        Score the candidate's project portfolio against the job profile.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        ProjectScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "ProjectScorer | candidate=%s | projects=%d",
            getattr(candidate, 'candidate_id', 'unknown'),
            len(getattr(candidate, 'projects', [])),
        )

        projects = getattr(candidate, 'projects', [])

        if not projects:
            warnings.append("No projects found in candidate profile.")
            return ProjectScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.3,
                reasoning="Candidate has no projects listed. Cannot evaluate project quality.",
                negative_signals=["No projects found"],
                improvement_suggestions=["Add projects with descriptions, tech stack, and results."],
                parsing_warnings=warnings,
                is_valid=True,
            )

        # Build prompt
        prompt = self._build_prompt(candidate, job_profile)

        # Call LLM
        raw_result = score_via_gemini(self._model, prompt, self._gen_config)

        if raw_result.get("error"):
            errors.append(raw_result["error"])
            return ProjectScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        return self._build_result(candidate, raw_result, errors, warnings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, candidate: Any, job_profile: Any) -> str:
        """Construct the full scoring prompt."""
        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Detailed project text
        project_lines = []
        for i, proj in enumerate(getattr(candidate, 'projects', []), start=1):
            parts = [f"Project {i}: {proj.name or 'Unnamed'}"]
            if proj.description:
                parts.append(f"  Description: {proj.description}")
            all_tech = (proj.technologies or []) + (proj.frameworks or []) + (proj.programming_languages or [])
            if all_tech:
                parts.append(f"  Tech: {', '.join(all_tech)}")
            if proj.role:
                parts.append(f"  Role: {proj.role}")
            if proj.contribution:
                parts.append(f"  Contribution: {proj.contribution}")
            if proj.problem_solved:
                parts.append(f"  Problem Solved: {proj.problem_solved}")
            if proj.results:
                parts.append(f"  Results: {proj.results}")
            if proj.real_world_use:
                parts.append(f"  Real-world Use: {proj.real_world_use}")
            if proj.challenges:
                parts.append(f"  Challenges: {proj.challenges}")
            if proj.deployment:
                parts.append(f"  Deployment: {proj.deployment}")
            if proj.github_link:
                parts.append(f"  GitHub: {proj.github_link}")
            if proj.live_demo:
                parts.append(f"  Live Demo: {proj.live_demo}")
            if proj.team_size:
                parts.append(f"  Team Size: {proj.team_size}")
            project_lines.append("\n".join(parts))

        projects_text = "\n\n".join(project_lines)

        return (
            f"JOB REQUIREMENTS:\n{jd_text}\n\n"
            f"CANDIDATE OVERVIEW:\n{cand_text}\n\n"
            f"CANDIDATE PROJECTS (full detail):\n{projects_text}\n\n"
            "Evaluate ALL projects above against the job requirements. "
            "Return a single valid JSON object with all fields from the schema."
        )

    def _build_result(
        self,
        candidate: Any,
        raw: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> ProjectScoreResult:
        """Map the raw LLM dict to a ProjectScoreResult."""
        # Extract per-dimension scores
        dimension_scores: dict[str, float] = {}
        for dim in PROJECT_DIMENSION_WEIGHTS:
            raw_val = raw.get(dim) or raw.get(f"{dim}_score", 50)
            dimension_scores[dim] = clamp_score(raw_val)

        # Weighted aggregate
        dim_values = list(dimension_scores.values())
        dim_weights = list(PROJECT_DIMENSION_WEIGHTS.values())
        weighted = weighted_average(dim_values, dim_weights)

        # Allow LLM's overall_score to anchor (blend 70% weighted, 30% LLM)
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        final_score = 0.7 * weighted + 0.3 * llm_overall

        return ProjectScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.7)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])) or [],
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            project_count=len(getattr(candidate, 'projects', [])),
            top_project=str(raw.get("top_project", "")) or None,
            per_project_scores=list(raw.get("per_project_scores", [])),
            portfolio_breadth=clamp_score(raw.get("portfolio_breadth_score", 50)),
            portfolio_depth=clamp_score(raw.get("portfolio_depth_score", 50)),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def score(
    candidate: Any,
    job_profile: Any,
    model_name: str = "gemini-2.0-flash",
) -> ProjectScoreResult:
    """
    Module-level convenience function.

    Parameters
    ----------
    candidate : Candidate
    job_profile : JobProfile
    model_name : str

    Returns
    -------
    ProjectScoreResult
    """
    scorer = ProjectScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
