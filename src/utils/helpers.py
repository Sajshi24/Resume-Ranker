"""
src/utils/helpers.py
====================
Shared utility functions for the AI Resume Intelligence Engine.

This module is the SINGLE SOURCE OF TRUTH for:
    - JSON parsing from LLM responses
    - String normalisation
    - List deduplication
    - Score clamping and confidence coercion
    - Candidate/JobProfile serialisation for LLM prompts
    - Pydantic model helpers

All scoring modules, agents, and parsers MUST import from here.
No module may rewrite these helpers locally.

Author  : Resume Intelligence Engine — Helpers Layer
Python  : 3.11+
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports at runtime; these are only used in type hints.
    from src.parsers.resume_parser import Candidate
    from src.parsers.jd_parser import JobProfile

logger = logging.getLogger("helpers")

# ---------------------------------------------------------------------------
# String Normalisation
# ---------------------------------------------------------------------------


def clean_str(value: Any) -> Optional[str]:
    """Strip whitespace; return None for empty or non-string values."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def safe_list(value: Any) -> list:
    """Coerce value to list; wrap scalar in list; return [] for None."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deduplicate(items: list[str]) -> list[str]:
    """Remove duplicate strings (case-insensitive) while preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def normalise_skill_list(items: list[Any]) -> list[str]:
    """Clean and deduplicate a list of skill/technology strings."""
    cleaned = [clean_str(i) for i in items if clean_str(i)]
    return deduplicate([s for s in cleaned if s])


# ---------------------------------------------------------------------------
# Numeric Helpers
# ---------------------------------------------------------------------------


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def clamp_score(value: Any, lo: float = 0.0, hi: float = 100.0) -> float:
    """
    Coerce ``value`` to a float and clamp to [lo, hi].
    Returns ``lo`` on any conversion error.
    """
    try:
        return clamp(float(value), lo, hi)
    except (TypeError, ValueError):
        return lo


def clamp_confidence(value: Any) -> float:
    """Coerce ``value`` to a float in [0.0, 1.0]. Defaults to 0.5 on error."""
    try:
        return clamp(float(value), 0.0, 1.0)
    except (TypeError, ValueError):
        return 0.5


def weighted_average(scores: list[float], weights: list[float]) -> float:
    """
    Compute a weighted average of ``scores`` using ``weights``.

    Parameters
    ----------
    scores : list[float]
        Raw scores (any range; typically 0-100).
    weights : list[float]
        Corresponding non-negative weights.  Need not sum to 1.

    Returns
    -------
    float
        Weighted average, clamped to [0, 100].  Returns 0.0 if total weight is zero.
    """
    if not scores or not weights or len(scores) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    raw = sum(s * w for s, w in zip(scores, weights)) / total_weight
    return clamp(raw, 0.0, 100.0)


# ---------------------------------------------------------------------------
# LLM Response Parsing
# ---------------------------------------------------------------------------


def parse_json_from_llm(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object from an LLM response string.

    Strips markdown code fences if present, attempts full JSON parse,
    then falls back to regex extraction of the outermost ``{...}`` block.

    Parameters
    ----------
    raw_text : str
        Raw string returned by the generative model.

    Returns
    -------
    dict
        Parsed JSON dictionary.

    Raises
    ------
    ValueError
        If no valid JSON object can be extracted.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # LLM returned a JSON array — wrap it
        return {"items": result}
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract a valid JSON object from LLM response "
        f"(first 200 chars): {raw_text[:200]!r}"
    )


def call_gemini(
    model: Any,  # genai.GenerativeModel
    prompt: str,
    generation_config: Any = None,  # genai.GenerationConfig
) -> str:
    """
    Call a Gemini model and return the raw text response.

    Extracts text from all content parts and concatenates them.
    Returns an empty string if the model produces no content.

    Parameters
    ----------
    model : genai.GenerativeModel
        Pre-configured model instance (from ``config.get_gemini_model``).
    prompt : str
        Full user prompt.
    generation_config : genai.GenerationConfig, optional
        Sampling parameters.

    Returns
    -------
    str
        Raw text from the model's first candidate.

    Raises
    ------
    RuntimeError
        If the Gemini API call raises any exception.
    """
    try:
        kwargs: dict = {"contents": prompt}
        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        response = model.generate_content(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    raw_text = ""
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            raw_text += part.text
    return raw_text


def score_via_gemini(
    model: Any,
    prompt: str,
    generation_config: Any = None,
) -> dict[str, Any]:
    """
    Call Gemini and parse the response as a JSON scoring object.

    This is a convenience wrapper around ``call_gemini`` + ``parse_json_from_llm``
    with proper error handling so callers always receive a dict.

    Parameters
    ----------
    model : genai.GenerativeModel
        Pre-configured model.
    prompt : str
        Full scoring prompt.
    generation_config : genai.GenerationConfig, optional
        Sampling config.

    Returns
    -------
    dict
        Parsed scoring result.  On failure returns
        ``{"error": "<message>", "score": 0, "confidence": 0.0}``.
    """
    try:
        raw = call_gemini(model, prompt, generation_config)
        if not raw.strip():
            return {"error": "Gemini returned empty response", "score": 0, "confidence": 0.0}
        return parse_json_from_llm(raw)
    except (RuntimeError, ValueError) as exc:
        logger.error("score_via_gemini failed: %s", exc)
        return {"error": str(exc), "score": 0, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Candidate / JobProfile Serialisation Helpers
# ---------------------------------------------------------------------------


def candidate_to_text(candidate: "Candidate") -> str:
    """
    Serialise a ``Candidate`` object to a compact, LLM-readable text block.

    Uses only the most signal-rich fields to stay within token limits.

    Parameters
    ----------
    candidate : Candidate
        Parsed candidate Pydantic object.

    Returns
    -------
    str
        Plain-text summary of the candidate.
    """
    lines: list[str] = []

    def _add(label: str, value: Any) -> None:
        v = clean_str(str(value)) if value is not None else None
        if v:
            lines.append(f"{label}: {v}")

    _add("Name", candidate.name)
    _add("Location", candidate.location)
    _add("Summary", candidate.summary)

    # Education
    for edu in candidate.education:
        parts = filter(None, [edu.degree, edu.branch, edu.college, edu.university])
        edu_str = " | ".join(parts)
        if edu.cgpa:
            edu_str += f" | CGPA {edu.cgpa}"
        if edu.graduation_year:
            edu_str += f" | Grad {edu.graduation_year}"
        if edu_str.strip():
            lines.append(f"Education: {edu_str}")

    # Experience
    for exp in candidate.experience:
        parts = filter(None, [exp.role, exp.company, exp.employment_type, exp.duration])
        exp_str = " | ".join(parts)
        if exp.technologies_used:
            exp_str += f" | Tech: {', '.join(exp.technologies_used[:8])}"
        if exp_str.strip():
            lines.append(f"Experience: {exp_str}")
        for resp in exp.responsibilities[:3]:
            lines.append(f"  - {resp}")

    # Projects
    for proj in candidate.projects:
        proj_str = proj.name or "Unnamed Project"
        if proj.description:
            proj_str += f": {proj.description[:120]}"
        all_tech = proj.technologies + proj.frameworks + proj.programming_languages
        if all_tech:
            proj_str += f" | Tech: {', '.join(all_tech[:8])}"
        if proj.results:
            proj_str += f" | Results: {proj.results[:100]}"
        lines.append(f"Project: {proj_str}")

    # Skills
    s = candidate.skills
    all_langs = s.programming_languages[:6]
    all_fw = s.frameworks[:6]
    all_ai = s.ai_ml[:6]
    all_tools = s.tools[:4]
    skill_parts = []
    if all_langs:
        skill_parts.append(f"Languages: {', '.join(all_langs)}")
    if all_fw:
        skill_parts.append(f"Frameworks: {', '.join(all_fw)}")
    if all_ai:
        skill_parts.append(f"AI/ML: {', '.join(all_ai)}")
    if all_tools:
        skill_parts.append(f"Tools: {', '.join(all_tools)}")
    if skill_parts:
        lines.append("Skills: " + " | ".join(skill_parts))

    # Achievements
    for ach in candidate.achievements[:5]:
        lines.append(f"Achievement: {ach.title or ''} [{ach.category or ''}] {ach.description or ''}")

    # Certifications
    for cert in candidate.certifications[:4]:
        lines.append(f"Certification: {cert.name} | {cert.platform}")

    # Research
    for res in candidate.research[:3]:
        lines.append(f"Research: {res.title} | {res.status or ''}")

    # Leadership
    for lead in candidate.leadership[:3]:
        lines.append(f"Leadership: {lead.position} @ {lead.club or lead.organisation}")

    return "\n".join(lines)


def job_profile_to_text(job_profile: "JobProfile") -> str:
    """
    Serialise a ``JobProfile`` object to a compact, LLM-readable text block.

    Parameters
    ----------
    job_profile : JobProfile
        Parsed job profile Pydantic object.

    Returns
    -------
    str
        Plain-text summary of the job profile.
    """
    lines: list[str] = []

    def _add(label: str, value: Any) -> None:
        v = clean_str(str(value)) if value is not None else None
        if v:
            lines.append(f"{label}: {v}")

    _add("Role", job_profile.role_title)
    _add("Company", job_profile.company_name)
    _add("Department", job_profile.department)
    _add("Work Mode", job_profile.work_mode)
    _add("Location", job_profile.location)
    _add("Summary", job_profile.role_summary)

    # Experience requirements
    exp = job_profile.experience_requirements
    if exp.minimum_years is not None:
        _add("Min Experience", f"{exp.minimum_years}+ years")
    if exp.seniority_level:
        _add("Seniority", exp.seniority_level)

    # Required skills
    req_names = [s.name for s in job_profile.required_skills[:12]]
    if req_names:
        lines.append(f"Required Skills: {', '.join(req_names)}")

    # Preferred skills
    pref_names = [s.name for s in job_profile.preferred_skills[:8]]
    if pref_names:
        lines.append(f"Preferred Skills: {', '.join(pref_names)}")

    # Responsibilities
    for resp in job_profile.responsibilities[:6]:
        lines.append(f"Responsibility: {resp.description}")

    # Business context
    biz = job_profile.business_context
    _add("Industry", biz.industry)
    _add("Business Domain", biz.business_domain)
    _add("Company Stage", biz.company_stage)

    # Hiring signals
    for sig in job_profile.hiring_signals[:5]:
        lines.append(f"Signal [{sig.signal_type}]: {sig.description}")

    # Hidden expectations
    for exp_item in job_profile.hidden_expectations[:5]:
        lines.append(f"Hidden Expectation [{exp_item.category}]: {exp_item.expectation}")

    return "\n".join(lines)


def get_all_candidate_skills(candidate: "Candidate") -> list[str]:
    """
    Return a flat, deduplicated list of ALL skills from a candidate.

    Parameters
    ----------
    candidate : Candidate

    Returns
    -------
    list[str]
        All skills across all categories.
    """
    s = candidate.skills
    all_skills = (
        s.programming_languages
        + s.frameworks
        + s.libraries
        + s.databases
        + s.cloud
        + s.devops
        + s.ai_ml
        + s.data_science
        + s.tools
        + s.other
    )
    # Also pull from projects and experience
    for proj in candidate.projects:
        all_skills += proj.technologies + proj.frameworks + proj.programming_languages
    for exp in candidate.experience:
        all_skills += exp.technologies_used
    return deduplicate(all_skills)


def get_all_jd_skills(job_profile: "JobProfile") -> list[str]:
    """
    Return a flat, deduplicated list of ALL skills from a job profile.

    Parameters
    ----------
    job_profile : JobProfile

    Returns
    -------
    list[str]
        All required + preferred skills.
    """
    required = [s.name for s in job_profile.required_skills]
    preferred = [s.name for s in job_profile.preferred_skills]
    tsm = job_profile.technical_skill_map
    map_skills = (
        tsm.programming_languages + tsm.frameworks + tsm.libraries
        + tsm.databases + tsm.cloud + tsm.devops + tsm.ai_ml
        + tsm.llms + tsm.nlp + tsm.computer_vision + tsm.data_engineering
        + tsm.backend + tsm.frontend + tsm.mobile + tsm.security
        + tsm.testing + tsm.tools
    )
    return deduplicate(required + preferred + map_skills)


def skill_overlap_ratio(candidate_skills: list[str], jd_skills: list[str]) -> float:
    """
    Compute the ratio of JD skills covered by the candidate's skills.

    Case-insensitive comparison.

    Parameters
    ----------
    candidate_skills : list[str]
    jd_skills : list[str]

    Returns
    -------
    float
        Coverage ratio in [0.0, 1.0].  Returns 0.0 if jd_skills is empty.
    """
    if not jd_skills:
        return 0.0
    cand_lower = {s.lower() for s in candidate_skills}
    hits = sum(1 for s in jd_skills if s.lower() in cand_lower)
    return hits / len(jd_skills)
