"""
src/scoring/semantic_score.py
==============================
Semantic Matching Scorer

Compares a candidate holistically against the job profile using Gemini reasoning.
NEVER uses keyword overlap as the primary metric.

Compares across 8 semantic dimensions:
    technology_match, domain_match, experience_match, project_match,
    growth_match, responsibility_match, leadership_match, research_match

Uses Google's text-embedding-004 model when available for vector similarity,
with Gemini reasoning as the primary (and robust fallback) approach.

Output: SemanticScoreResult (extends ScoreResult)

Author  : Resume Intelligence Engine — Scoring Layer
Python  : 3.11+
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import Field

from src.scoring import BaseScorer, ScoreResult
from src.utils.config import (
    EMBEDDING_MODEL,
    get_gemini_model,
    get_generation_config,
)
from src.utils.helpers import (
    candidate_to_text,
    clamp_confidence,
    clamp_score,
    get_all_candidate_skills,
    get_all_jd_skills,
    job_profile_to_text,
    score_via_gemini,
    skill_overlap_ratio,
    weighted_average,
)

logger = logging.getLogger("semantic_score")

SEMANTIC_DIMENSION_WEIGHTS: dict[str, float] = {
    "technology_match": 2.0,
    "domain_match": 1.8,
    "experience_match": 1.6,
    "project_match": 1.5,
    "growth_match": 1.2,
    "responsibility_match": 1.7,
    "leadership_match": 1.0,
    "research_match": 0.8,
}

SEMANTIC_SYSTEM_PROMPT = """
You are an expert talent intelligence analyst performing deep semantic matching
between a candidate and a job profile for an AI-powered hiring engine.

Do NOT perform keyword matching. Reason about semantic similarity:
- A candidate who built "recommendation systems using collaborative filtering"
  semantically matches a role needing "personalisation engine experience"
  even if the exact words differ.

Score these dimensions (0-100):
- technology_match: How well does the candidate's tech stack align with the JD?
  Consider compatibility (e.g. PyTorch vs TensorFlow both valid for ML roles).
- domain_match: How semantically close is the candidate's domain to the target domain?
- experience_match: Does the candidate's seniority, scope, and type of work match?
- project_match: Do the candidate's projects semantically solve problems similar to the role?
- growth_match: Does the candidate's growth trajectory point toward this role?
- responsibility_match: Can the candidate handle the listed responsibilities?
- leadership_match: Leadership experience alignment with role expectations?
- research_match: Intellectual depth alignment with role's research/innovation needs?

Provide:
- overall_score (0-100) — holistic semantic match
- confidence (0.0-1.0)
- reasoning (3-5 sentences explaining the semantic match)
- match_summary: One sentence verdict ("Strong match", "Adjacent match", etc.)
- semantic_gaps (list: areas where the semantic distance is large)
- semantic_strengths (list: areas with strong semantic overlap)
- positive_signals (list)
- negative_signals (list)
- improvement_suggestions (list)

Return ONLY valid JSON. No markdown, no explanation.
""".strip()


class SemanticScoreResult(ScoreResult):
    """Scoring result for the Semantic Matching dimension."""

    scorer_name: str = "SemanticScorer"
    match_summary: str = Field("", description="One-sentence semantic match verdict")
    semantic_gaps: list[str] = Field(default_factory=list)
    semantic_strengths: list[str] = Field(default_factory=list)
    embedding_similarity: Optional[float] = Field(
        None,
        ge=0.0, le=1.0,
        description="Cosine similarity from text embeddings (None if unavailable)",
    )
    raw_overlap_ratio: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Simple skill overlap ratio (sanity anchor, not primary score)",
    )


class SemanticScorer(BaseScorer):
    """
    Semantic matching scorer using Gemini reasoning + optional embedding similarity.

    Primary scoring path: Gemini reasoning over full candidate + JD text.
    Secondary anchor: cosine similarity via text-embedding-004 (when available).
    Tertiary anchor: raw skill overlap ratio (used as 10% anchor).

    The embedding path is attempted but never required — Gemini reasoning is
    always the primary signal.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        super().__init__()
        self._model = get_gemini_model(
            model_name=model_name,
            system_instruction=SEMANTIC_SYSTEM_PROMPT,
        )
        self._gen_config = get_generation_config(temperature=0.1)
        logger.info("SemanticScorer initialised with model: %s", model_name)

    @property
    def scorer_name(self) -> str:
        return "SemanticScorer"

    def score(self, candidate: Any, job_profile: Any) -> SemanticScoreResult:
        """
        Semantically compare the candidate with the job profile.

        Parameters
        ----------
        candidate : Candidate
        job_profile : JobProfile

        Returns
        -------
        SemanticScoreResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "SemanticScorer | candidate=%s",
            getattr(candidate, 'candidate_id', 'unknown'),
        )

        cand_text = candidate_to_text(candidate)
        jd_text = job_profile_to_text(job_profile)

        # Deterministic anchor: raw skill overlap
        cand_skills = get_all_candidate_skills(candidate)
        jd_skills = get_all_jd_skills(job_profile)
        overlap = skill_overlap_ratio(cand_skills, jd_skills)

        # Optional: embedding similarity
        embedding_sim: Optional[float] = self._try_embedding_similarity(cand_text, jd_text)
        if embedding_sim is not None:
            logger.info("Embedding similarity: %.4f", embedding_sim)

        prompt = (
            f"JOB PROFILE:\\n{jd_text}\\n\\n"
            f"CANDIDATE PROFILE:\\n{cand_text}\\n\\n"
            f"(Context: raw skill keyword overlap = {overlap:.2f}; "
            + (f"embedding similarity = {embedding_sim:.4f}" if embedding_sim is not None else "embedding unavailable")
            + ")\\n\\n"
            "Perform deep semantic matching across all 8 dimensions. "
            "Return a single valid JSON object."
        )

        raw = score_via_gemini(self._model, prompt, self._gen_config)

        if raw.get("error"):
            errors.append(raw["error"])
            return SemanticScoreResult(
                candidate_id=getattr(candidate, 'candidate_id', None),
                score=0.0,
                confidence=0.0,
                reasoning="LLM scoring failed.",
                parsing_errors=errors,
                is_valid=False,
            )

        dimension_scores: dict[str, float] = {}
        for dim in SEMANTIC_DIMENSION_WEIGHTS:
            dimension_scores[dim] = clamp_score(raw.get(dim, raw.get(f"{dim}_score", 50)))

        weighted = weighted_average(
            list(dimension_scores.values()),
            list(SEMANTIC_DIMENSION_WEIGHTS.values()),
        )
        llm_overall = clamp_score(raw.get("overall_score", weighted))
        overlap_anchor = clamp_score(overlap * 100)

        # Blend: 55% LLM reasoning + 30% weighted dimensions + 10% overlap + 5% embedding
        if embedding_sim is not None:
            embed_anchor = clamp_score(embedding_sim * 100)
            final_score = (
                0.55 * llm_overall
                + 0.30 * weighted
                + 0.10 * overlap_anchor
                + 0.05 * embed_anchor
            )
        else:
            final_score = (
                0.55 * llm_overall
                + 0.35 * weighted
                + 0.10 * overlap_anchor
            )

        return SemanticScoreResult(
            candidate_id=getattr(candidate, 'candidate_id', None),
            score=round(clamp_score(final_score), 2),
            confidence=clamp_confidence(raw.get("confidence", 0.7)),
            reasoning=str(raw.get("reasoning", "")),
            evidence=list(raw.get("evidence", [])),
            positive_signals=list(raw.get("positive_signals", [])),
            negative_signals=list(raw.get("negative_signals", [])),
            improvement_suggestions=list(raw.get("improvement_suggestions", [])),
            dimension_scores=dimension_scores,
            match_summary=str(raw.get("match_summary", "")),
            semantic_gaps=list(raw.get("semantic_gaps", [])),
            semantic_strengths=list(raw.get("semantic_strengths", [])),
            embedding_similarity=round(embedding_sim, 6) if embedding_sim is not None else None,
            raw_overlap_ratio=round(overlap, 4),
            parsing_errors=errors,
            parsing_warnings=warnings,
            is_valid=len(errors) == 0,
        )

    # ------------------------------------------------------------------
    # Embedding similarity (best-effort, never required)
    # ------------------------------------------------------------------

    @staticmethod
    def _try_embedding_similarity(text_a: str, text_b: str) -> Optional[float]:
        """
        Attempt to compute cosine similarity between two text strings using
        Google's text-embedding-004 model.

        Returns None on any failure (model unavailable, API error, etc.).
        Never raises.

        Parameters
        ----------
        text_a : str
        text_b : str

        Returns
        -------
        float | None
            Cosine similarity in [0.0, 1.0], or None.
        """
        try:
            import google.generativeai as genai

            # Truncate to avoid token limit issues
            a = text_a[:2000]
            b = text_b[:2000]

            result_a = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=a,
                task_type="SEMANTIC_SIMILARITY",
            )
            result_b = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=b,
                task_type="SEMANTIC_SIMILARITY",
            )

            vec_a = result_a["embedding"]
            vec_b = result_b["embedding"]

            return SemanticScorer._cosine_similarity(vec_a, vec_b)

        except Exception as exc:  # noqa: BLE001
            logger.debug("Embedding similarity unavailable: %s", exc)
            return None

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two equal-length float vectors."""
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return clamp_score(dot / (norm_a * norm_b), 0.0, 1.0)


def score(candidate: Any, job_profile: Any, model_name: str = "gemini-2.0-flash") -> SemanticScoreResult:
    """Module-level convenience function."""
    scorer = SemanticScorer(model_name=model_name)
    return scorer._safe_score(candidate, job_profile)  # type: ignore[return-value]
