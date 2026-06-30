"""
src/scoring/final_score.py
===========================
Final Score Orchestrator

Runs all 7 scoring modules in sequence, aggregates results into a weighted
overall score, and produces a fully explainable ``FinalScoreResult``.

Scoring pipeline:
    ProjectScorer     → project dimension
    DomainScorer      → domain dimension
    SkillScorer       → skill dimension
    LearningScorer    → learning dimension
    TransferableScorer→ transferable dimension
    GrowthScorer      → growth dimension
    SemanticScorer    → semantic dimension
         ↓
    FinalScorer       → weighted aggregate + hiring recommendation

Output: FinalScoreResult (extends ScoreResult)

Saves results to: outputs/scoring_results.json

Integration points for Recruiter Agent:
    - FinalScoreResult.hiring_recommendation
    - FinalScoreResult.strengths
    - FinalScoreResult.weaknesses
    - FinalScoreResult.risk_factors
    - FinalScoreResult.dimension_scores  (all 7 sub-scores)
    - FinalScoreResult.per_dimension_results (full detail per scorer)

Performance notes:
    - Each scorer calls Gemini once (7 calls per candidate total).
    - Results are cached in-memory per FinalScorer instance.
    - Use score_batch() for multiple candidates to share model instances.

Author  : Resume Intelligence Engine — Scoring Layer
Python  : 3.11+
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from src.scoring import BaseScorer, ScoreResult
from src.scoring.domain_score import DomainScoreResult, DomainScorer
from src.scoring.growth_score import GrowthScoreResult, GrowthScorer
from src.scoring.learning_score import LearningScoreResult, LearningScorer
from src.scoring.project_score import ProjectScoreResult, ProjectScorer
from src.scoring.semantic_score import SemanticScoreResult, SemanticScorer
from src.scoring.skill_score import SkillScoreResult, SkillScorer
from src.scoring.transferable_score import TransferableScoreResult, TransferableScorer
from src.utils.config import (
    SCORING_OUTPUT_JSON,
    ensure_output_dirs,
    get_gemini_model,
    get_generation_config,
)
from src.utils.helpers import (
    candidate_to_text,
    clamp_confidence,
    clamp_score,
    job_profile_to_text,
    score_via_gemini,
    weighted_average,
)

logger = logging.getLogger("final_score")

# ---------------------------------------------------------------------------
# Final aggregation weights
# These weights are the policy levers. Adjust to shift emphasis.
# ---------------------------------------------------------------------------

FINAL_DIMENSION_WEIGHTS: dict[str, float] = {
    "project_score": 2.0,     # Demonstrates real engineering capability
    "domain_score": 1.8,      # Domain fit is critical for fast ramp-up
    "skill_score": 2.2,       # Core technical match
    "learning_score": 1.5,    # Future adaptability
    "transferable_score": 1.2, # Soft skills matter, but less than tech
    "growth_score": 1.8,      # Future potential (this engine rewards potential)
    "semantic_score": 2.0,    # Holistic fit
}

# ---------------------------------------------------------------------------
# Hiring Recommendation Thresholds
# ---------------------------------------------------------------------------

HIRING_THRESHOLDS = {
    "strong_hire": 78.0,
    "hire": 62.0,
    "borderline": 48.0,
    "no_hire": 35.0,
    # Below no_hire → "strong_no_hire"
}

FINAL_EXPLAINER_SYSTEM_PROMPT = """
You are a senior recruiter and talent strategy expert.
You have received detailed scoring results from 7 AI analysis modules
for a candidate being evaluated for a specific role.

Your task: synthesise these scores into a final hiring recommendation with full explanation.

Provide:
- hiring_recommendation: one of 'strong_hire' | 'hire' | 'borderline' | 'no_hire' | 'strong_no_hire'
- overall_reasoning: 4-6 sentences explaining the recommendation holistically
- top_3_strengths: the 3 strongest reasons to hire this candidate
- top_3_weaknesses: the 3 most significant gaps or concerns
- risk_factors: factors that could make this hire problematic
- growth_opportunity: how the candidate could grow into an ideal fit
- interview_focus_areas: what the interviewer should probe deeper
- confidence: your confidence in this recommendation (0.0-1.0)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------


class FinalScoreResult(ScoreResult):
    """
    The complete, fully explainable scoring result for a single candidate.

    Consumed by: Recruiter Agent, Dashboard, Ranking Engine.
    """

    scorer_name: str = "FinalScorer"

    # ---- Sub-scores (0-100 each) ----
    project_score: float = Field(0.0, ge=0.0, le=100.0)
    domain_score: float = Field(0.0, ge=0.0, le=100.0)
    skill_score: float = Field(0.0, ge=0.0, le=100.0)
    learning_score: float = Field(0.0, ge=0.0, le=100.0)
    transferable_score: float = Field(0.0, ge=0.0, le=100.0)
    growth_score: float = Field(0.0, ge=0.0, le=100.0)
    semantic_score: float = Field(0.0, ge=0.0, le=100.0)

    # ---- Hiring Decision ----
    hiring_recommendation: str = Field(
        "unknown",
        description=(
            "'strong_hire' | 'hire' | 'borderline' | 'no_hire' | 'strong_no_hire'"
        ),
    )
    overall_reasoning: str = Field("", description="Full narrative reasoning for the recommendation")

    # ---- Explainability ----
    strengths: list[str] = Field(default_factory=list, description="Top candidate strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Top candidate gaps")
    risk_factors: list[str] = Field(default_factory=list)
    growth_opportunity: str = Field("", description="How candidate could grow into ideal fit")
    interview_focus_areas: list[str] = Field(
        default_factory=list,
        description="Areas the interviewer should probe",
    )

    # ---- Full Sub-Scorer Results ----
    per_dimension_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Full ScoreResult dict from each individual scorer",
    )

    # ---- Metadata ----
    candidate_name: Optional[str] = None
    role_title: Optional[str] = None
    company_name: Optional[str] = None
    scoring_duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Final Scorer
# ---------------------------------------------------------------------------


class FinalScorer:
    """
    Orchestrates all 7 scoring modules and produces a ``FinalScoreResult``.

    Each scorer is initialised once and reused across multiple candidates
    via ``score_batch()`` for efficiency.

    Usage
    -----
    >>> scorer = FinalScorer()
    >>> result = scorer.score(candidate, job_profile)
    >>> print(result.score, result.hiring_recommendation)

    Batch usage (recommended for multiple candidates):
    >>> results = scorer.score_batch(candidates, job_profile)
    >>> scorer.save_results(results)
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        """
        Initialise all 7 sub-scorers and the final explainer model.

        Parameters
        ----------
        model_name : str
            Gemini model to use for all sub-scorers and the final explainer.
        """
        self.logger = logging.getLogger("FinalScorer")
        self.model_name = model_name

        # Initialise all sub-scorers once
        self._project_scorer = ProjectScorer(model_name)
        self._domain_scorer = DomainScorer(model_name)
        self._skill_scorer = SkillScorer(model_name)
        self._learning_scorer = LearningScorer(model_name)
        self._transferable_scorer = TransferableScorer(model_name)
        self._growth_scorer = GrowthScorer(model_name)
        self._semantic_scorer = SemanticScorer(model_name)

        # Final explainer model
        self._explainer_model = get_gemini_model(
            model_name=model_name,
            system_instruction=FINAL_EXPLAINER_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.15)

        logger.info(
            "FinalScorer initialised | model=%s | sub-scorers=7",
            model_name,
        )

    def score(self, candidate: Any, job_profile: Any) -> FinalScoreResult:
        """
        Run the full 7-module scoring pipeline for a single candidate.

        Parameters
        ----------
        candidate : Candidate
            Parsed candidate object.
        job_profile : JobProfile
            Parsed job profile object.

        Returns
        -------
        FinalScoreResult
            Fully explainable aggregate result.  Never raises.
        """
        start = datetime.utcnow()
        candidate_id = getattr(candidate, 'candidate_id', 'unknown')

        logger.info(
            "=== FinalScorer | START | candidate=%s ===",
            candidate_id,
        )

        errors: list[str] = []
        warnings: list[str] = []

        # ---- Run all 7 sub-scorers ----
        project_result = self._run_scorer(
            "project", self._project_scorer, candidate, job_profile, errors
        )
        domain_result = self._run_scorer(
            "domain", self._domain_scorer, candidate, job_profile, errors
        )
        skill_result = self._run_scorer(
            "skill", self._skill_scorer, candidate, job_profile, errors
        )
        learning_result = self._run_scorer(
            "learning", self._learning_scorer, candidate, job_profile, errors
        )
        transferable_result = self._run_scorer(
            "transferable", self._transferable_scorer, candidate, job_profile, errors
        )
        growth_result = self._run_scorer(
            "growth", self._growth_scorer, candidate, job_profile, errors
        )
        semantic_result = self._run_scorer(
            "semantic", self._semantic_scorer, candidate, job_profile, errors
        )

        # ---- Aggregate scores ----
        sub_scores = {
            "project_score": project_result.score,
            "domain_score": domain_result.score,
            "skill_score": skill_result.score,
            "learning_score": learning_result.score,
            "transferable_score": transferable_result.score,
            "growth_score": growth_result.score,
            "semantic_score": semantic_result.score,
        }

        score_values = list(sub_scores.values())
        weight_values = [FINAL_DIMENSION_WEIGHTS[k] for k in sub_scores]
        weighted_total = weighted_average(score_values, weight_values)

        # ---- Confidence: average of all sub-scorer confidences ----
        all_confidences = [
            project_result.confidence, domain_result.confidence,
            skill_result.confidence, learning_result.confidence,
            transferable_result.confidence, growth_result.confidence,
            semantic_result.confidence,
        ]
        avg_confidence = sum(all_confidences) / len(all_confidences)

        # ---- Get hiring recommendation from LLM explainer ----
        recommendation_data = self._get_hiring_recommendation(
            candidate, job_profile, sub_scores, weighted_total,
            project_result, domain_result, skill_result,
            learning_result, transferable_result, growth_result, semantic_result,
        )

        # ---- Rule-based fallback recommendation ----
        rule_rec = self._rule_based_recommendation(weighted_total)
        hiring_rec = recommendation_data.get("hiring_recommendation", rule_rec)

        # ---- Duration ----
        duration = (datetime.utcnow() - start).total_seconds()

        logger.info(
            "=== FinalScorer | DONE | candidate=%s | score=%.1f | rec=%s | duration=%.1fs ===",
            candidate_id,
            weighted_total,
            hiring_rec,
            duration,
        )

        return FinalScoreResult(
            candidate_id=candidate_id,
            candidate_name=getattr(candidate, 'name', None),
            role_title=getattr(job_profile, 'role_title', None),
            company_name=getattr(job_profile, 'company_name', None),

            # Aggregate
            score=round(clamp_score(weighted_total), 2),
            confidence=round(clamp_confidence(avg_confidence), 4),
            reasoning=str(recommendation_data.get("overall_reasoning", "")),
            evidence=[],
            positive_signals=list(recommendation_data.get("top_3_strengths", [])),
            negative_signals=list(recommendation_data.get("top_3_weaknesses", [])),
            improvement_suggestions=list(recommendation_data.get("interview_focus_areas", [])),
            dimension_scores={k: round(v, 2) for k, v in sub_scores.items()},

            # Sub-scores
            **sub_scores,  # type: ignore[arg-type]

            # Hiring decision
            hiring_recommendation=str(hiring_rec),
            overall_reasoning=str(recommendation_data.get("overall_reasoning", "")),

            # Explainability
            strengths=list(recommendation_data.get("top_3_strengths", [])),
            weaknesses=list(recommendation_data.get("top_3_weaknesses", [])),
            risk_factors=list(recommendation_data.get("risk_factors", [])),
            growth_opportunity=str(recommendation_data.get("growth_opportunity", "")),
            interview_focus_areas=list(recommendation_data.get("interview_focus_areas", [])),

            # Full sub-scorer results
            per_dimension_results={
                "project": project_result.model_dump(mode="json"),
                "domain": domain_result.model_dump(mode="json"),
                "skill": skill_result.model_dump(mode="json"),
                "learning": learning_result.model_dump(mode="json"),
                "transferable": transferable_result.model_dump(mode="json"),
                "growth": growth_result.model_dump(mode="json"),
                "semantic": semantic_result.model_dump(mode="json"),
            },

            # Metadata
            scoring_duration_seconds=round(duration, 2),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )

    def score_batch(
        self,
        candidates: list[Any],
        job_profile: Any,
    ) -> list[FinalScoreResult]:
        """
        Score a list of candidates against the same job profile.

        Sub-scorer instances are shared across all candidates (no re-init cost).

        Parameters
        ----------
        candidates : list[Candidate]
        job_profile : JobProfile

        Returns
        -------
        list[FinalScoreResult]
            Results in the same order as input candidates.
        """
        results: list[FinalScoreResult] = []
        total = len(candidates)

        logger.info("FinalScorer | Batch scoring %d candidate(s)", total)

        for idx, candidate in enumerate(candidates, start=1):
            logger.info(
                "FinalScorer | Batch %d/%d | candidate=%s",
                idx, total,
                getattr(candidate, 'candidate_id', 'unknown'),
            )
            try:
                result = self.score(candidate, job_profile)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "FinalScorer | Batch item %d failed: %s", idx, exc
                )
                result = FinalScoreResult(
                    candidate_id=getattr(candidate, 'candidate_id', None),
                    candidate_name=getattr(candidate, 'name', None),
                    score=0.0,
                    confidence=0.0,
                    reasoning=f"Scoring failed: {exc}",
                    hiring_recommendation="unknown",
                    parsing_errors=[str(exc)],
                    is_valid=False,
                )
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        logger.info(
            "FinalScorer | Batch complete | %d results | top_score=%.1f",
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    def save_results(
        self,
        results: list[FinalScoreResult],
        output_path: Path = SCORING_OUTPUT_JSON,
    ) -> Path:
        """
        Serialise and persist scoring results to JSON.

        Parameters
        ----------
        results : list[FinalScoreResult]
        output_path : Path

        Returns
        -------
        Path
            Resolved output path.
        """
        ensure_output_dirs()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_candidates": len(results),
                "scorer_version": "1.0.0",
                "scoring_weights": FINAL_DIMENSION_WEIGHTS,
                "hiring_thresholds": HIRING_THRESHOLDS,
            },
            "ranked_candidates": [r.model_dump(mode="json") for r in results],
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info(
            "FinalScorer | Saved %d results -> %s",
            len(results),
            output_path,
        )
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_scorer(
        self,
        name: str,
        scorer: BaseScorer,
        candidate: Any,
        job_profile: Any,
        errors: list[str],
    ) -> ScoreResult:
        """
        Run a single sub-scorer with full exception isolation.
        Appends errors to the mutable ``errors`` list.
        """
        logger.info("Running sub-scorer: %s", name)
        try:
            result = scorer._safe_score(candidate, job_profile)
            if result.parsing_errors:
                errors.extend([f"[{name}] {e}" for e in result.parsing_errors])
            return result
        except Exception as exc:  # noqa: BLE001
            msg = f"[{name}] unhandled exception: {exc}"
            logger.error(msg)
            errors.append(msg)
            return ScoreResult(
                scorer_name=name,
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning=f"Sub-scorer {name} failed: {exc}",
                parsing_errors=[msg],
                is_valid=False,
            )

    def _get_hiring_recommendation(
        self,
        candidate: Any,
        job_profile: Any,
        sub_scores: dict[str, float],
        weighted_total: float,
        project_result: ScoreResult,
        domain_result: ScoreResult,
        skill_result: ScoreResult,
        learning_result: ScoreResult,
        transferable_result: ScoreResult,
        growth_result: ScoreResult,
        semantic_result: ScoreResult,
    ) -> dict[str, Any]:
        """
        Call the Gemini explainer to synthesise all 7 sub-scores into a
        hiring recommendation with full narrative reasoning.
        """
        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        score_summary = "\n".join(
            f"  {k}: {v:.1f}/100" for k, v in sub_scores.items()
        )
        signal_summary_parts: list[str] = []
        for result in (project_result, domain_result, skill_result,
                       learning_result, transferable_result, growth_result, semantic_result):
            if result.positive_signals:
                signal_summary_parts.append(
                    f"[{result.scorer_name}] STRENGTHS: {'; '.join(result.positive_signals[:2])}"
                )
            if result.negative_signals:
                signal_summary_parts.append(
                    f"[{result.scorer_name}] GAPS: {'; '.join(result.negative_signals[:2])}"
                )

        signal_context = "\n".join(signal_summary_parts)

        prompt = (
            f"ROLE BEING FILLED:\\n{jd_text}\\n\\n"
            f"CANDIDATE:\\n{cand_text}\\n\\n"
            f"SCORING RESULTS (weighted aggregate: {weighted_total:.1f}/100):\\n{score_summary}\\n\\n"
            f"KEY SIGNALS FROM ANALYSIS:\\n{signal_context}\\n\\n"
            "Synthesise a final hiring recommendation. Return a single valid JSON object."
        )

        result = score_via_gemini(self._explainer_model, prompt, self._gen_config)

        if result.get("error"):
            logger.warning("Explainer LLM failed; using rule-based recommendation.")
            return {
                "hiring_recommendation": self._rule_based_recommendation(weighted_total),
                "overall_reasoning": (
                    f"Rule-based recommendation: score {weighted_total:.1f}/100. "
                    "LLM explainer unavailable."
                ),
            }

        return result

    @staticmethod
    def _rule_based_recommendation(score: float) -> str:
        """
        Derive a hiring recommendation from the aggregate score alone.
        Used as fallback when the LLM explainer fails.
        """
        if score >= HIRING_THRESHOLDS["strong_hire"]:
            return "strong_hire"
        if score >= HIRING_THRESHOLDS["hire"]:
            return "hire"
        if score >= HIRING_THRESHOLDS["borderline"]:
            return "borderline"
        if score >= HIRING_THRESHOLDS["no_hire"]:
            return "no_hire"
        return "strong_no_hire"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def score(
    candidate: Any,
    job_profile: Any,
    model_name: str = "gemini-2.0-flash",
) -> FinalScoreResult:
    """
    Score a single candidate against a job profile.

    Parameters
    ----------
    candidate : Candidate
    job_profile : JobProfile
    model_name : str

    Returns
    -------
    FinalScoreResult
    """
    scorer = FinalScorer(model_name=model_name)
    return scorer.score(candidate, job_profile)


def score_and_rank(
    candidates: list[Any],
    job_profile: Any,
    model_name: str = "gemini-2.0-flash",
    save: bool = True,
    output_path: Path = SCORING_OUTPUT_JSON,
) -> list[FinalScoreResult]:
    """
    Score and rank a list of candidates, optionally saving results to disk.

    Parameters
    ----------
    candidates : list[Candidate]
    job_profile : JobProfile
    model_name : str
    save : bool
        If True, persist results to ``output_path``.
    output_path : Path
        Output file path (defaults to outputs/scoring_results.json).

    Returns
    -------
    list[FinalScoreResult]
        Ranked candidates (highest score first).

    Example
    -------
    >>> from src.parsers.resume_parser import run_parser
    >>> from src.parsers.jd_parser import run_jd_parser
    >>> from src.scoring.final_score import score_and_rank
    >>>
    >>> candidates = run_parser()
    >>> job_profile = run_jd_parser()
    >>> results = score_and_rank(candidates, job_profile)
    >>> for r in results:
    ...     print(f"{r.candidate_name}: {r.score:.1f} | {r.hiring_recommendation}")
    """
    scorer = FinalScorer(model_name=model_name)
    results = scorer.score_batch(candidates, job_profile)
    if save:
        scorer.save_results(results, output_path)
    return results
