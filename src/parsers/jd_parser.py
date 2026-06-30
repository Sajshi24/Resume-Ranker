"""
src/parsers/jd_parser.py
========================
Enterprise-grade AI Resume Intelligence Engine — Job Description Parser Module

Responsibilities:
    - Load a raw job description from data/raw/job_description.txt
    - Use Google Gemini to intelligently extract AND infer structured information
    - Go beyond keyword extraction: understand recruiter intent and hidden expectations
    - Normalize and validate data via Pydantic models
    - Persist processed output to data/processed/parsed_job_description.json

This module performs ONLY parsing/extraction.
No scoring, ranking, embeddings, semantic matching, or domain fit logic is
implemented here. All Pydantic models are designed so that downstream modules
(Project Intelligence, Domain Fit, Semantic Matching, Recruiter Agent, Growth
Prediction) can consume the ``JobProfile`` object directly.

Design consistency:
    - Same import conventions as resume_parser.py
    - Same helper function signatures (_clean_str, _normalise_skill_list, _safe_list, etc.)
    - Same Gemini wiring (GenerativeModel + system_instruction + GenerationConfig)
    - Same logging format (module-level logger, structured messages)
    - Same Pydantic v2 patterns (BaseModel, Field, field_validator, model_config)
    - Same save/load lifecycle (load → extract → normalise → validate → save)

Author  : Resume Intelligence Engine — JD Parser Layer
Python  : 3.11+
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Safe stdout reconfiguration for Windows (fixes UnicodeEncodeError)
# ---------------------------------------------------------------------------
if sys.stdout.encoding.lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("jd_parser")

# ---------------------------------------------------------------------------
# Path Constants  (mirrors resume_parser.py convention)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JD_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "job_description.txt"
JD_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_job_description.json"

# ---------------------------------------------------------------------------
# Shared normalisation helpers
# (Intentionally re-implemented here so jd_parser has zero dependency on
#  resume_parser at import time, keeping both modules independently runnable.)
# ---------------------------------------------------------------------------


def _clean_str(value: Any) -> Optional[str]:
    """Strip whitespace; return None for empty or non-string values."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def _safe_list(value: Any) -> list:
    """Coerce value to list; wrap scalar in list; return [] for None."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicate strings (case-insensitive) while preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def _normalise_skill_list(items: list[Any]) -> list[str]:
    """Clean and deduplicate a list of skill/technology strings."""
    cleaned = [_clean_str(i) for i in items if _clean_str(i)]
    return _deduplicate([s for s in cleaned if s])


def _clamp_confidence(value: Any) -> float:
    """Coerce a confidence score to a float in [0.0, 1.0]."""
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return 0.5  # safe default when LLM omits the value


def _parse_json_from_llm_response(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object from an LLM response string.

    Strips markdown code fences if present, then attempts full parse.
    Falls back to regex extraction of the first ``{...}`` block.

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
        If no valid JSON object can be extracted from the response.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the outermost { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError("Could not extract a valid JSON object from LLM response.")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RequiredSkill(BaseModel):
    """
    A skill or technology that is explicitly required by the job description.

    Consumed by:  Semantic Matching, Domain Fit, Recruiter Agent
    """

    name: str = Field(description="Canonical skill name, e.g. 'Python', 'React', 'Kubernetes'")
    category: Optional[str] = Field(
        None,
        description=(
            "'programming_language' | 'framework' | 'library' | 'database' | "
            "'cloud' | 'devops' | 'ai_ml' | 'llm' | 'nlp' | 'computer_vision' | "
            "'data_engineering' | 'backend' | 'frontend' | 'mobile' | "
            "'security' | 'testing' | 'tool' | 'soft_skill' | 'other'"
        ),
    )
    proficiency_level: Optional[str] = Field(
        None,
        description="'beginner' | 'intermediate' | 'advanced' | 'expert' | null",
    )
    years_required: Optional[float] = Field(
        None, ge=0.0, description="Minimum years of experience with this skill"
    )
    context: Optional[str] = Field(
        None, description="Sentence from the JD that mentions this skill"
    )
    is_explicit: bool = Field(
        True, description="True if the skill is explicitly named; False if inferred"
    )

    @field_validator("years_required", mode="before")
    @classmethod
    def coerce_years(cls, v: Any) -> Optional[float]:
        """Convert string years to float; return None on failure."""
        if v is None or v == "":
            return None
        try:
            return float(str(v).replace("+", "").strip())
        except (ValueError, TypeError):
            return None


class PreferredSkill(BaseModel):
    """
    A skill or technology that is preferred or nice-to-have.

    Consumed by:  Semantic Matching, Growth Prediction
    """

    name: str
    category: Optional[str] = None
    context: Optional[str] = Field(
        None, description="Sentence from the JD that mentions this skill"
    )
    advantage_description: Optional[str] = Field(
        None,
        description="Why having this skill gives an edge, inferred from context",
    )


class Responsibility(BaseModel):
    """
    A single role responsibility extracted from the job description.

    Consumed by:  Project Intelligence, Recruiter Agent
    """

    description: str = Field(description="Full text of the responsibility")
    is_mandatory: bool = Field(
        True,
        description="True if phrased as a requirement; False if optional/nice-to-have",
    )
    domain_area: Optional[str] = Field(
        None,
        description="Broad domain this responsibility belongs to, e.g. 'backend', 'ml', 'leadership'",
    )
    requires_leadership: bool = False
    requires_collaboration: bool = False
    requires_research: bool = False
    technical_complexity: Optional[str] = Field(
        None, description="'low' | 'medium' | 'high'"
    )


class Qualification(BaseModel):
    """
    Education or academic qualification entry from the job description.

    Consumed by:  Domain Fit
    """

    degree: Optional[str] = Field(
        None,
        description="Required or preferred degree, e.g. 'B.Tech', 'M.S.', 'PhD'",
    )
    field_of_study: Optional[str] = Field(
        None,
        description="Domain, e.g. 'Computer Science', 'AI/ML', 'Statistics'",
    )
    is_required: bool = Field(
        True, description="True = required; False = preferred"
    )
    research_preferred: bool = Field(
        False, description="Whether a research background is explicitly or implicitly preferred"
    )
    notes: Optional[str] = Field(
        None, description="Any additional context around this qualification"
    )


class HiringSignal(BaseModel):
    """
    A structured signal extracted from the JD that reveals what the recruiter
    truly values — beyond the written words.

    Consumed by:  Recruiter Agent, Explainable Ranking
    """

    signal_type: str = Field(
        description=(
            "'must_have' | 'nice_to_have' | 'deal_breaker' | "
            "'growth_indicator' | 'leadership_indicator' | "
            "'research_indicator' | 'innovation_indicator'"
        )
    )
    description: str = Field(description="Human-readable description of the signal")
    evidence: Optional[str] = Field(
        None, description="Exact quote or paraphrase from the JD that triggered this signal"
    )
    importance: Optional[str] = Field(
        None, description="'critical' | 'high' | 'medium' | 'low'"
    )
    affects_domains: list[str] = Field(
        default_factory=list,
        description="Which downstream modules care about this signal",
    )


class HiddenExpectation(BaseModel):
    """
    A recruiter expectation that is NOT explicitly written in the JD but can
    be inferred from the overall context, company culture signals, or role level.

    This is the intelligence layer that separates this engine from ATS tools.

    Consumed by:  Recruiter Agent, Growth Prediction, Explainable Ranking
    """

    expectation: str = Field(
        description="Short label for the expectation, e.g. 'Startup Mindset'"
    )
    evidence: str = Field(
        description="Excerpt or paraphrase from the JD that suggests this expectation"
    )
    reasoning: str = Field(
        description="Explanation of why this expectation was inferred"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM confidence that this expectation is real (0.0 – 1.0)",
    )
    category: Optional[str] = Field(
        None,
        description=(
            "'mindset' | 'technical_depth' | 'communication' | 'leadership' | "
            "'ownership' | 'curiosity' | 'product_thinking' | 'research_orientation' | "
            "'scalability_thinking' | 'customer_obsession' | 'innovation' | "
            "'system_thinking' | 'analytical_reasoning' | 'decision_making' | 'other'"
        ),
    )
    downstream_impact: Optional[str] = Field(
        None,
        description="Which pipeline stage benefits most from this signal",
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        """Ensure confidence is always a valid float in [0, 1]."""
        return _clamp_confidence(v)


class TechnicalSkillMap(BaseModel):
    """
    Structured technical skill taxonomy for the job profile.
    Mirrors ``Skills`` in resume_parser.py for clean cross-matching.

    Consumed by:  Semantic Matching, Domain Fit
    """

    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    libraries: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    devops: list[str] = Field(default_factory=list)
    ai_ml: list[str] = Field(default_factory=list)
    llms: list[str] = Field(default_factory=list)
    nlp: list[str] = Field(default_factory=list)
    computer_vision: list[str] = Field(default_factory=list)
    data_engineering: list[str] = Field(default_factory=list)
    backend: list[str] = Field(default_factory=list)
    frontend: list[str] = Field(default_factory=list)
    mobile: list[str] = Field(default_factory=list)
    security: list[str] = Field(default_factory=list)
    testing: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ExperienceRequirements(BaseModel):
    """
    Experience requirements extracted from the job description.

    Consumed by:  Domain Fit, Recruiter Agent
    """

    minimum_years: Optional[float] = Field(
        None, ge=0.0, description="Minimum years of total professional experience"
    )
    preferred_years: Optional[float] = Field(
        None, ge=0.0, description="Preferred years of total professional experience"
    )
    requires_leadership: bool = False
    requires_mentoring: bool = False
    requires_management: bool = False
    requires_research: bool = False
    leadership_scope: Optional[str] = Field(
        None,
        description="Scale of leadership expected, e.g. 'team of 5', 'department', 'organisation'",
    )
    seniority_level: Optional[str] = Field(
        None,
        description="'intern' | 'junior' | 'mid' | 'senior' | 'staff' | 'principal' | 'director' | 'vp'",
    )
    domain_experience_required: list[str] = Field(
        default_factory=list,
        description="Specific domains where prior experience is required, e.g. ['fintech', 'healthcare']",
    )

    @field_validator("minimum_years", "preferred_years", mode="before")
    @classmethod
    def coerce_years(cls, v: Any) -> Optional[float]:
        """Handle '5+' or '3-5 years' style strings."""
        if v is None or v == "":
            return None
        try:
            return float(str(v).replace("+", "").split("-")[0].strip())
        except (ValueError, TypeError):
            return None


class BusinessContext(BaseModel):
    """
    Business and product domain context inferred from the JD.

    Consumed by:  Domain Fit, Semantic Matching
    """

    industry: Optional[str] = Field(
        None, description="Industry vertical, e.g. 'fintech', 'healthcare', 'edtech', 'e-commerce'"
    )
    business_domain: Optional[str] = Field(
        None,
        description="Functional business domain, e.g. 'payments', 'recommendation systems', 'fraud detection'",
    )
    product_domain: Optional[str] = Field(
        None,
        description="Product area, e.g. 'consumer mobile app', 'enterprise SaaS', 'developer tools'",
    )
    target_customers: Optional[str] = Field(
        None, description="Who the company's customers are, e.g. 'B2B SMEs', 'direct consumers', 'enterprises'"
    )
    business_problems: list[str] = Field(
        default_factory=list,
        description="Core business problems the company/role is solving",
    )
    company_stage: Optional[str] = Field(
        None,
        description="'early_stage_startup' | 'growth_startup' | 'scaleup' | 'enterprise' | 'public_company' | null",
    )
    team_size_hint: Optional[str] = Field(
        None, description="Any hints about team or company size"
    )


class JobProfile(BaseModel):
    """
    Top-level structured job profile produced by the JD parser.

    This is the single source of truth consumed by all downstream modules:
    - Project Intelligence: matches project scope against responsibilities
    - Domain Fit:           matches candidate domain vs job domain
    - Semantic Matching:    aligns candidate skills vs required/preferred skills
    - Recruiter Agent:      uses hiring signals + hidden expectations for evaluation
    - Growth Prediction:    uses growth indicators and hidden expectations

    All fields are optional or default to safe empty values so the object is
    always constructible even from an incomplete or empty job description.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    job_profile_id: Optional[str] = Field(
        None, description="Unique identifier, typically auto-assigned"
    )
    parsed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO-8601 UTC timestamp when this profile was parsed",
    )
    parser_version: str = "1.0.0"
    source_format: Optional[str] = Field(
        None, description="'plain_text' | 'markdown' | 'html' | 'unknown'"
    )
    raw_character_count: int = Field(
        0, description="Character length of the original JD text"
    )

    # ------------------------------------------------------------------
    # Basic Information
    # ------------------------------------------------------------------
    role_title: Optional[str] = Field(
        None, description="Job title, e.g. 'Senior ML Engineer'"
    )
    department: Optional[str] = Field(
        None, description="Department or team, e.g. 'AI Research', 'Platform Engineering'"
    )
    employment_type: Optional[str] = Field(
        None,
        description="'full_time' | 'part_time' | 'contract' | 'internship' | 'freelance'",
    )
    work_mode: Optional[str] = Field(
        None, description="'remote' | 'hybrid' | 'onsite' | 'flexible'"
    )
    location: Optional[str] = Field(
        None, description="Office location, city, country, or 'Anywhere'"
    )
    company_name: Optional[str] = None
    company_description: Optional[str] = Field(
        None, description="Brief description of the company as stated in the JD"
    )
    role_summary: Optional[str] = Field(
        None, description="1-3 sentence summary of what this role is about"
    )

    # ------------------------------------------------------------------
    # Structured Sections
    # ------------------------------------------------------------------
    required_skills: list[RequiredSkill] = Field(default_factory=list)
    preferred_skills: list[PreferredSkill] = Field(default_factory=list)
    technical_skill_map: TechnicalSkillMap = Field(default_factory=TechnicalSkillMap)
    soft_skills: list[str] = Field(default_factory=list)
    responsibilities: list[Responsibility] = Field(default_factory=list)
    qualifications: list[Qualification] = Field(default_factory=list)
    experience_requirements: ExperienceRequirements = Field(
        default_factory=ExperienceRequirements
    )
    business_context: BusinessContext = Field(default_factory=BusinessContext)

    # ------------------------------------------------------------------
    # Intelligence Layer
    # ------------------------------------------------------------------
    hiring_signals: list[HiringSignal] = Field(default_factory=list)
    hidden_expectations: list[HiddenExpectation] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Parsed Diagnostics
    # ------------------------------------------------------------------
    parsing_warnings: list[str] = Field(default_factory=list)
    parsing_errors: list[str] = Field(default_factory=list)
    is_valid: bool = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# LLM Extraction System Prompt
# ---------------------------------------------------------------------------

JD_EXTRACTION_SYSTEM_PROMPT = """
You are an expert recruiter intelligence analyst for an AI Resume Ranking Engine.

Your task is to analyse a raw job description and return a single valid JSON object
that captures BOTH explicit information AND inferred recruiter intent.

CRITICAL RULES:
- Return ONLY a valid JSON object. No markdown fences, no explanation.
- Use null for missing fields. Never omit top-level keys.
- Lists must always be JSON arrays, never comma-separated strings.
- Be thorough: extract every skill, responsibility, and signal mentioned.
- For hidden_expectations: infer at least 3 and up to 10 expectations even if subtle.
- confidence_score must be a float between 0.0 and 1.0.
- employment_type: one of "full_time", "part_time", "contract", "internship", "freelance".
- work_mode: one of "remote", "hybrid", "onsite", "flexible".
- seniority_level: one of "intern", "junior", "mid", "senior", "staff", "principal", "director", "vp".
- company_stage: one of "early_stage_startup", "growth_startup", "scaleup", "enterprise", "public_company" or null.
- skill category: one of "programming_language", "framework", "library", "database", "cloud",
  "devops", "ai_ml", "llm", "nlp", "computer_vision", "data_engineering", "backend", "frontend",
  "mobile", "security", "testing", "tool", "soft_skill", "other".
- signal_type: one of "must_have", "nice_to_have", "deal_breaker", "growth_indicator",
  "leadership_indicator", "research_indicator", "innovation_indicator".
- technical_complexity: one of "low", "medium", "high" or null.
- HiddenExpectation category: one of "mindset", "technical_depth", "communication",
  "leadership", "ownership", "curiosity", "product_thinking", "research_orientation",
  "scalability_thinking", "customer_obsession", "innovation", "system_thinking",
  "analytical_reasoning", "decision_making", "other".

OUTPUT SCHEMA (return exactly this structure):
{
  "role_title": string | null,
  "department": string | null,
  "employment_type": string | null,
  "work_mode": string | null,
  "location": string | null,
  "company_name": string | null,
  "company_description": string | null,
  "role_summary": string | null,

  "required_skills": [
    {
      "name": string,
      "category": string | null,
      "proficiency_level": string | null,
      "years_required": number | null,
      "context": string | null,
      "is_explicit": boolean
    }
  ],

  "preferred_skills": [
    {
      "name": string,
      "category": string | null,
      "context": string | null,
      "advantage_description": string | null
    }
  ],

  "technical_skill_map": {
    "programming_languages": [string],
    "frameworks": [string],
    "libraries": [string],
    "databases": [string],
    "cloud": [string],
    "devops": [string],
    "ai_ml": [string],
    "llms": [string],
    "nlp": [string],
    "computer_vision": [string],
    "data_engineering": [string],
    "backend": [string],
    "frontend": [string],
    "mobile": [string],
    "security": [string],
    "testing": [string],
    "tools": [string]
  },

  "soft_skills": [string],

  "responsibilities": [
    {
      "description": string,
      "is_mandatory": boolean,
      "domain_area": string | null,
      "requires_leadership": boolean,
      "requires_collaboration": boolean,
      "requires_research": boolean,
      "technical_complexity": string | null
    }
  ],

  "qualifications": [
    {
      "degree": string | null,
      "field_of_study": string | null,
      "is_required": boolean,
      "research_preferred": boolean,
      "notes": string | null
    }
  ],

  "experience_requirements": {
    "minimum_years": number | null,
    "preferred_years": number | null,
    "requires_leadership": boolean,
    "requires_mentoring": boolean,
    "requires_management": boolean,
    "requires_research": boolean,
    "leadership_scope": string | null,
    "seniority_level": string | null,
    "domain_experience_required": [string]
  },

  "business_context": {
    "industry": string | null,
    "business_domain": string | null,
    "product_domain": string | null,
    "target_customers": string | null,
    "business_problems": [string],
    "company_stage": string | null,
    "team_size_hint": string | null
  },

  "hiring_signals": [
    {
      "signal_type": string,
      "description": string,
      "evidence": string | null,
      "importance": string | null,
      "affects_domains": [string]
    }
  ],

  "hidden_expectations": [
    {
      "expectation": string,
      "evidence": string,
      "reasoning": string,
      "confidence_score": number,
      "category": string | null,
      "downstream_impact": string | null
    }
  ]
}
""".strip()


# ---------------------------------------------------------------------------
# JDParser
# ---------------------------------------------------------------------------


class JDParser:
    """
    AI-powered job description parser that converts an unstructured JD text
    into a fully-validated ``JobProfile`` Pydantic object.

    Pipeline
    --------
    raw_text (str)
        → detect_format()
        → extract_with_llm()         # Gemini structured extraction
        → normalise_job_profile()    # clean / deduplicate / coerce
        → build_job_profile()        # Pydantic model construction
        → validate_job_profile()     # semantic checks
        → JobProfile

    The parser never raises exceptions to the caller.  All errors are
    captured in ``JobProfile.parsing_errors`` and logged appropriately.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        """
        Initialise the JDParser and configure the Gemini generative client.

        Parameters
        ----------
        model_name : str
            Gemini model identifier.  Defaults to ``gemini-2.0-flash``.

        Raises
        ------
        EnvironmentError
            If ``GEMINI_API_KEY`` is not found in the environment.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=JD_EXTRACTION_SYSTEM_PROMPT,
        )
        logger.info("JDParser initialised with model: %s", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_jd(self, path: Path = JD_RAW_PATH) -> str:
        """
        Load the raw job description text from a file.

        Supports plain text, markdown, and copy-pasted JD content.
        Returns an empty string (and logs a warning) if the file does
        not exist or is empty — the parser never crashes on missing input.

        Parameters
        ----------
        path : Path
            Absolute path to the job description file.

        Returns
        -------
        str
            Raw JD text. Empty string if file is missing or empty.
        """
        if not path.exists():
            logger.warning("Job description file not found: %s", path)
            return ""

        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Failed to read JD file %s: %s", path, exc)
            return ""

        if not raw:
            logger.warning("Job description file is empty: %s", path)
            return ""

        logger.info(
            "Loaded JD from %s | %d characters", path, len(raw)
        )
        return raw

    def parse_jd(self, jd_text: str) -> JobProfile:
        """
        Full parse pipeline for a single raw job description string.

        This is the primary entry point for parsing an already-loaded JD.
        Call ``load_jd()`` first if you need to load from disk.

        Parameters
        ----------
        jd_text : str
            Raw job description text (any format).

        Returns
        -------
        JobProfile
            Fully validated JobProfile object. Errors are embedded in the
            object rather than raised, so callers always receive a usable object.
        """
        errors: list[str] = []
        warnings: list[str] = []
        source_format = self._detect_format(jd_text)

        logger.info(
            "Parsing JD | format=%s | length=%d chars",
            source_format,
            len(jd_text),
        )

        # Step 1 — LLM extraction
        llm_data: dict[str, Any] = {}
        if jd_text.strip():
            try:
                llm_data = self.extract_with_llm(jd_text)
            except Exception as exc:  # noqa: BLE001
                msg = f"LLM extraction failed: {exc}"
                logger.error(msg)
                errors.append(msg)
        else:
            msg = "JD text is empty — skipping LLM extraction."
            logger.warning(msg)
            warnings.append(msg)

        # Step 2 — Normalise
        normalised = self.normalise_job_profile(llm_data, warnings)

        # Step 3 — Build Pydantic model
        job_profile = self._build_job_profile(
            data=normalised,
            source_format=source_format,
            raw_character_count=len(jd_text),
            errors=errors,
            warnings=warnings,
        )

        # Step 4 — Validate
        job_profile = self.validate_job_profile(job_profile)

        if job_profile.is_valid:
            logger.info(
                "JD parsing successful | role=%s | company=%s",
                job_profile.role_title or "unknown",
                job_profile.company_name or "unknown",
            )
        else:
            logger.warning(
                "JD parsing completed with %d error(s) | role=%s",
                len(job_profile.parsing_errors),
                job_profile.role_title or "unknown",
            )

        return job_profile

    def extract_with_llm(self, jd_text: str) -> dict[str, Any]:
        """
        Send the job description to Google Gemini and parse the structured
        JSON response into a Python dictionary.

        This method intentionally uses a low temperature (0.1) to keep
        extraction deterministic. The system prompt instructs the model to
        return ONLY valid JSON with no markdown fences or explanations.

        Parameters
        ----------
        jd_text : str
            Raw job description text.

        Returns
        -------
        dict
            Structured extraction dictionary matching the JD schema.

        Raises
        ------
        ValueError
            If the JD text is empty or the model returns no usable content.
        RuntimeError
            If the Gemini API call fails for any reason.
        """
        if not jd_text.strip():
            raise ValueError("Empty JD text — nothing to extract.")

        prompt = (
            "Analyse the following job description thoroughly.\n"
            "Extract all explicit information AND infer all recruiter expectations.\n"
            "Return a single valid JSON object matching the provided schema.\n\n"
            f"JOB DESCRIPTION:\n{jd_text}"
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,        # deterministic extraction
                    max_output_tokens=8192,
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

        raw_text: str = ""
        if response and response.candidates:
            for part in response.candidates[0].content.parts:
                raw_text += part.text

        if not raw_text.strip():
            raise ValueError("Gemini returned an empty response.")

        logger.debug("Raw LLM response length: %d chars", len(raw_text))

        return _parse_json_from_llm_response(raw_text)

    def normalise_job_profile(
        self,
        data: dict[str, Any],
        warnings: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Clean, deduplicate, and coerce all fields in the raw extracted dict.

        Mutates and returns the dict for convenience (same pattern as
        ``ResumeParser.normalise_candidate``).

        Parameters
        ----------
        data : dict
            Raw extracted JD data from the LLM.
        warnings : list[str], optional
            Mutable list to append normalisation warnings to.

        Returns
        -------
        dict
            Normalised JD data dictionary.
        """
        if warnings is None:
            warnings = []

        if not data:
            warnings.append("LLM returned no data — all fields will be empty.")
            return {}

        # Basic scalar fields
        data["role_title"] = _clean_str(data.get("role_title"))
        data["department"] = _clean_str(data.get("department"))
        data["company_name"] = _clean_str(data.get("company_name"))
        data["company_description"] = _clean_str(data.get("company_description"))
        data["role_summary"] = _clean_str(data.get("role_summary"))
        data["location"] = _clean_str(data.get("location"))
        data["employment_type"] = _clean_str(data.get("employment_type"))
        data["work_mode"] = _clean_str(data.get("work_mode"))

        # Required skills — ensure name is always present
        required: list[dict] = []
        for skill in _safe_list(data.get("required_skills")):
            if not isinstance(skill, dict):
                continue
            name = _clean_str(skill.get("name"))
            if not name:
                warnings.append("Skipped a required_skill entry with no name.")
                continue
            skill["name"] = name
            skill["context"] = _clean_str(skill.get("context"))
            required.append(skill)
        data["required_skills"] = required

        # Preferred skills
        preferred: list[dict] = []
        for skill in _safe_list(data.get("preferred_skills")):
            if not isinstance(skill, dict):
                continue
            name = _clean_str(skill.get("name"))
            if not name:
                warnings.append("Skipped a preferred_skill entry with no name.")
                continue
            skill["name"] = name
            skill["context"] = _clean_str(skill.get("context"))
            skill["advantage_description"] = _clean_str(skill.get("advantage_description"))
            preferred.append(skill)
        data["preferred_skills"] = preferred

        # Technical skill map — normalise every category
        tsm = data.get("technical_skill_map") or {}
        if isinstance(tsm, dict):
            for key in (
                "programming_languages", "frameworks", "libraries", "databases",
                "cloud", "devops", "ai_ml", "llms", "nlp", "computer_vision",
                "data_engineering", "backend", "frontend", "mobile",
                "security", "testing", "tools",
            ):
                tsm[key] = _normalise_skill_list(_safe_list(tsm.get(key)))
            data["technical_skill_map"] = tsm

        # Soft skills
        data["soft_skills"] = _normalise_skill_list(
            _safe_list(data.get("soft_skills"))
        )

        # Responsibilities — clean description text
        responsibilities: list[dict] = []
        for resp in _safe_list(data.get("responsibilities")):
            if not isinstance(resp, dict):
                continue
            desc = _clean_str(resp.get("description"))
            if not desc:
                continue
            resp["description"] = desc
            resp["domain_area"] = _clean_str(resp.get("domain_area"))
            responsibilities.append(resp)
        data["responsibilities"] = responsibilities

        # Qualifications
        qualifications: list[dict] = []
        for qual in _safe_list(data.get("qualifications")):
            if not isinstance(qual, dict):
                continue
            qual["degree"] = _clean_str(qual.get("degree"))
            qual["field_of_study"] = _clean_str(qual.get("field_of_study"))
            qual["notes"] = _clean_str(qual.get("notes"))
            qualifications.append(qual)
        data["qualifications"] = qualifications

        # Experience requirements — sub-object
        exp_req = data.get("experience_requirements") or {}
        if isinstance(exp_req, dict):
            exp_req["leadership_scope"] = _clean_str(exp_req.get("leadership_scope"))
            exp_req["seniority_level"] = _clean_str(exp_req.get("seniority_level"))
            exp_req["domain_experience_required"] = _normalise_skill_list(
                _safe_list(exp_req.get("domain_experience_required"))
            )
            data["experience_requirements"] = exp_req

        # Business context — sub-object
        biz = data.get("business_context") or {}
        if isinstance(biz, dict):
            biz["industry"] = _clean_str(biz.get("industry"))
            biz["business_domain"] = _clean_str(biz.get("business_domain"))
            biz["product_domain"] = _clean_str(biz.get("product_domain"))
            biz["target_customers"] = _clean_str(biz.get("target_customers"))
            biz["company_stage"] = _clean_str(biz.get("company_stage"))
            biz["team_size_hint"] = _clean_str(biz.get("team_size_hint"))
            biz["business_problems"] = [
                _clean_str(p)
                for p in _safe_list(biz.get("business_problems"))
                if _clean_str(p)
            ]
            data["business_context"] = biz

        # Hiring signals — clean description and evidence
        signals: list[dict] = []
        for signal in _safe_list(data.get("hiring_signals")):
            if not isinstance(signal, dict):
                continue
            desc = _clean_str(signal.get("description"))
            if not desc:
                continue
            signal["description"] = desc
            signal["evidence"] = _clean_str(signal.get("evidence"))
            signal["affects_domains"] = _normalise_skill_list(
                _safe_list(signal.get("affects_domains"))
            )
            signals.append(signal)
        data["hiring_signals"] = signals

        # Hidden expectations — core intelligence layer
        expectations: list[dict] = []
        for exp in _safe_list(data.get("hidden_expectations")):
            if not isinstance(exp, dict):
                continue
            expectation_label = _clean_str(exp.get("expectation"))
            evidence = _clean_str(exp.get("evidence"))
            reasoning = _clean_str(exp.get("reasoning"))
            if not expectation_label or not reasoning:
                warnings.append(
                    "Skipped a hidden_expectation entry missing 'expectation' or 'reasoning'."
                )
                continue
            exp["expectation"] = expectation_label
            exp["evidence"] = evidence or ""
            exp["reasoning"] = reasoning
            exp["confidence_score"] = _clamp_confidence(exp.get("confidence_score", 0.5))
            exp["category"] = _clean_str(exp.get("category"))
            exp["downstream_impact"] = _clean_str(exp.get("downstream_impact"))
            expectations.append(exp)
        data["hidden_expectations"] = expectations

        return data

    def validate_job_profile(self, job_profile: JobProfile) -> JobProfile:
        """
        Run semantic validation checks on a parsed ``JobProfile`` object.

        Validation is non-fatal: issues are recorded in
        ``parsing_warnings``; ``is_valid`` is set to ``False`` only when a
        critical structural problem is detected.

        Parameters
        ----------
        job_profile : JobProfile
            The profile object to validate.

        Returns
        -------
        JobProfile
            The same object with validation metadata updated in place.
        """
        # Critical: a profile with no role title is nearly unusable
        if not job_profile.role_title:
            job_profile.parsing_warnings.append(
                "role_title is missing — JD may be incomplete or unparseable."
            )
            job_profile.is_valid = False

        # A profile with zero required skills is suspicious
        if not job_profile.required_skills:
            job_profile.parsing_warnings.append(
                "No required_skills extracted — check JD content."
            )

        # A profile with zero responsibilities is suspicious
        if not job_profile.responsibilities:
            job_profile.parsing_warnings.append(
                "No responsibilities extracted — check JD content."
            )

        # Hiring signals completeness
        signal_types = {s.signal_type for s in job_profile.hiring_signals}
        if "must_have" not in signal_types:
            job_profile.parsing_warnings.append(
                "No 'must_have' hiring signal found — recruiter intent may be underspecified."
            )

        # Hidden expectations should always be present for a useful profile
        if len(job_profile.hidden_expectations) < 2:
            job_profile.parsing_warnings.append(
                f"Only {len(job_profile.hidden_expectations)} hidden expectation(s) inferred — "
                "JD may be very sparse."
            )

        # Experience years sanity
        exp = job_profile.experience_requirements
        if (
            exp.minimum_years is not None
            and exp.preferred_years is not None
            and exp.preferred_years < exp.minimum_years
        ):
            job_profile.parsing_warnings.append(
                f"preferred_years ({exp.preferred_years}) < minimum_years "
                f"({exp.minimum_years}) — possible data error."
            )

        # Cross-validate: skills mentioned in required_skills should appear
        # somewhere in technical_skill_map (advisory only)
        tsm_all_skills: set[str] = set()
        tsm = job_profile.technical_skill_map
        for bucket in (
            tsm.programming_languages, tsm.frameworks, tsm.libraries,
            tsm.databases, tsm.cloud, tsm.devops, tsm.ai_ml, tsm.llms,
            tsm.nlp, tsm.computer_vision, tsm.data_engineering,
            tsm.backend, tsm.frontend, tsm.mobile, tsm.security,
            tsm.testing, tsm.tools,
        ):
            tsm_all_skills.update(s.lower() for s in bucket)

        unmapped = [
            r.name
            for r in job_profile.required_skills
            if r.name.lower() not in tsm_all_skills
        ]
        if unmapped:
            job_profile.parsing_warnings.append(
                f"{len(unmapped)} required skill(s) not found in technical_skill_map: "
                f"{', '.join(unmapped[:5])}{'...' if len(unmapped) > 5 else ''}"
            )

        return job_profile

    def parse_from_file(self, path: Path = JD_RAW_PATH) -> JobProfile:
        """
        Convenience method: load a JD file and run the full parse pipeline.

        Parameters
        ----------
        path : Path
            Path to the job description text file.

        Returns
        -------
        JobProfile
            Parsed and validated job profile.

        Example
        -------
        >>> parser = JDParser()
        >>> profile = parser.parse_from_file()
        >>> print(profile.role_title)
        """
        jd_text = self.load_jd(path)
        return self.parse_jd(jd_text)

    def save_processed(
        self,
        job_profile: JobProfile,
        output_path: Path = JD_PROCESSED_PATH,
    ) -> Path:
        """
        Serialise and persist a ``JobProfile`` object to JSON.

        Parent directories are created automatically if they do not exist.

        Parameters
        ----------
        job_profile : JobProfile
            The parsed job profile to save.
        output_path : Path
            Destination file path.

        Returns
        -------
        Path
            The resolved path where the output was written.
        """
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "parser_version": job_profile.parser_version,
                "source_format": job_profile.source_format,
                "raw_character_count": job_profile.raw_character_count,
                "is_valid": job_profile.is_valid,
                "warning_count": len(job_profile.parsing_warnings),
                "error_count": len(job_profile.parsing_errors),
            },
            "job_profile": job_profile.model_dump(mode="json"),
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Saved job profile → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(jd_text: str) -> str:
        """
        Heuristically detect the format of the input job description.

        Returns one of: ``'markdown'``, ``'plain_text'``, ``'unknown'``.
        """
        if not jd_text.strip():
            return "unknown"

        markdown_signals = (
            re.search(r"^#{1,6}\s", jd_text, re.MULTILINE),   # headings
            re.search(r"^\s*[\*\-]\s", jd_text, re.MULTILINE), # bullet lists
            "**" in jd_text or "__" in jd_text,                 # bold
            "[" in jd_text and "](" in jd_text,                 # links
        )
        if sum(bool(s) for s in markdown_signals) >= 2:
            return "markdown"

        return "plain_text"

    def _build_job_profile(
        self,
        data: dict[str, Any],
        source_format: str,
        raw_character_count: int,
        errors: list[str],
        warnings: list[str],
    ) -> JobProfile:
        """
        Construct a ``JobProfile`` Pydantic model from a normalised dictionary.

        Any Pydantic validation error is captured in ``errors`` rather than
        raised, guaranteeing the parser never crashes.

        Parameters
        ----------
        data : dict
            Normalised JD extraction dictionary.
        source_format : str
            Detected format of the original JD text.
        raw_character_count : int
            Character length of the original JD text.
        errors : list[str]
            Mutable list of errors accumulated so far.
        warnings : list[str]
            Mutable list of warnings accumulated so far.

        Returns
        -------
        JobProfile
            Constructed model, or a minimal fallback on construction failure.
        """
        try:
            job_profile = JobProfile(
                source_format=source_format,
                raw_character_count=raw_character_count,

                # Basic information
                role_title=data.get("role_title"),
                department=data.get("department"),
                employment_type=data.get("employment_type"),
                work_mode=data.get("work_mode"),
                location=data.get("location"),
                company_name=data.get("company_name"),
                company_description=data.get("company_description"),
                role_summary=data.get("role_summary"),

                # Skills
                required_skills=[
                    RequiredSkill(**s)
                    for s in _safe_list(data.get("required_skills"))
                    if isinstance(s, dict) and s.get("name")
                ],
                preferred_skills=[
                    PreferredSkill(**s)
                    for s in _safe_list(data.get("preferred_skills"))
                    if isinstance(s, dict) and s.get("name")
                ],
                technical_skill_map=TechnicalSkillMap(
                    **(data.get("technical_skill_map") or {})
                ),
                soft_skills=_safe_list(data.get("soft_skills")),

                # Responsibilities and qualifications
                responsibilities=[
                    Responsibility(**r)
                    for r in _safe_list(data.get("responsibilities"))
                    if isinstance(r, dict) and r.get("description")
                ],
                qualifications=[
                    Qualification(**q)
                    for q in _safe_list(data.get("qualifications"))
                    if isinstance(q, dict)
                ],

                # Experience requirements
                experience_requirements=ExperienceRequirements(
                    **(data.get("experience_requirements") or {})
                ),

                # Business context
                business_context=BusinessContext(
                    **(data.get("business_context") or {})
                ),

                # Intelligence layer
                hiring_signals=[
                    HiringSignal(**s)
                    for s in _safe_list(data.get("hiring_signals"))
                    if isinstance(s, dict) and s.get("description") and s.get("signal_type")
                ],
                hidden_expectations=[
                    HiddenExpectation(**e)
                    for e in _safe_list(data.get("hidden_expectations"))
                    if isinstance(e, dict) and e.get("expectation") and e.get("reasoning")
                ],

                # Diagnostics
                parsing_errors=errors,
                parsing_warnings=warnings,
                is_valid=len(errors) == 0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("JobProfile Pydantic construction failed: %s", exc)
            errors.append(f"Model construction error: {exc}")
            job_profile = JobProfile(
                source_format=source_format,
                raw_character_count=raw_character_count,
                parsing_errors=errors,
                parsing_warnings=warnings,
                is_valid=False,
            )

        return job_profile


# ---------------------------------------------------------------------------
# Convenience entry-point  (mirrors run_parser() in resume_parser.py)
# ---------------------------------------------------------------------------


def run_jd_parser(
    input_path: Path = JD_RAW_PATH,
    output_path: Path = JD_PROCESSED_PATH,
    model_name: str = "gemini-2.0-flash",
) -> JobProfile:
    """
    Convenience function that wires together the full JD parser pipeline.

    Loads the JD from ``input_path``, parses it, saves the result to
    ``output_path``, and returns the ``JobProfile`` object.

    Parameters
    ----------
    input_path : Path
        Path to the raw job description text file.
    output_path : Path
        Path where ``parsed_job_description.json`` will be written.
    model_name : str
        Gemini model to use for extraction.

    Returns
    -------
    JobProfile
        The fully parsed and validated job profile.

    Example
    -------
    >>> from src.parsers.jd_parser import run_jd_parser
    >>> profile = run_jd_parser()
    >>> print(profile.role_title)
    >>> print(len(profile.hidden_expectations))
    """
    parser = JDParser(model_name=model_name)
    job_profile = parser.parse_from_file(input_path)
    parser.save_processed(job_profile, output_path)
    return job_profile


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile = run_jd_parser()
    print(f"\n[OK] JD Parsed | Role: {profile.role_title} | Company: {profile.company_name}")
    print(f"  Required skills : {len(profile.required_skills)}")
    print(f"  Preferred skills: {len(profile.preferred_skills)}")
    print(f"  Responsibilities: {len(profile.responsibilities)}")
    print(f"  Hiring signals  : {len(profile.hiring_signals)}")
    print(f"  Hidden expects  : {len(profile.hidden_expectations)}")
    print(f"  Valid           : {profile.is_valid}")
    print(f"  Output          -> {JD_PROCESSED_PATH}")
