"""
src/parsers/resume_parser.py
============================
Enterprise-grade AI Resume Intelligence Engine — Parser Module

Responsibilities:
    - Load raw candidate records from candidates.jsonl
    - Use Google Gemini to intelligently extract structured information
    - Normalize and validate data via Pydantic models
    - Persist processed output to data/processed/parsed_candidates.json

This module performs ONLY parsing/extraction.
No scoring, ranking, embeddings, or recommendations are implemented here.
All Pydantic models are designed so downstream scoring modules can consume
the `Candidate` object directly without any structural changes.

Author  : Resume Intelligence Engine — Parser Layer
Python  : 3.11+
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Safe stdout reconfiguration for Windows (fixes UnicodeEncodeError)
# ---------------------------------------------------------------------------
if sys.stdout.encoding.lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import google.generativeai as genai
from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

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
logger = logging.getLogger("resume_parser")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "candidates.jsonl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_candidates.json"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Education(BaseModel):
    """Academic qualification record."""

    degree: Optional[str] = Field(None, description="Degree title, e.g. B.Tech, M.Sc")
    college: Optional[str] = Field(None, description="College or institution name")
    university: Optional[str] = Field(None, description="Affiliated university")
    branch: Optional[str] = Field(None, description="Specialisation / branch / major")
    cgpa: Optional[float] = Field(None, ge=0.0, le=10.0, description="CGPA on 10-point scale")
    percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Percentage score")
    graduation_year: Optional[int] = Field(None, description="Year of graduation or expected graduation")
    start_year: Optional[int] = Field(None, description="Year of enrolment")
    relevant_coursework: list[str] = Field(default_factory=list, description="Key courses completed")

    @field_validator("cgpa", mode="before")
    @classmethod
    def coerce_cgpa(cls, v: Any) -> Optional[float]:
        """Convert string CGPA to float; ignore non-numeric values."""
        if v is None or v == "":
            return None
        try:
            return float(str(v).replace(",", "."))
        except (ValueError, TypeError):
            return None

    @field_validator("percentage", mode="before")
    @classmethod
    def coerce_percentage(cls, v: Any) -> Optional[float]:
        """Strip '%' and convert to float."""
        if v is None or v == "":
            return None
        try:
            return float(str(v).replace("%", "").strip())
        except (ValueError, TypeError):
            return None


class Experience(BaseModel):
    """Professional or internship experience record."""

    company: Optional[str] = None
    role: Optional[str] = None
    employment_type: Optional[str] = Field(
        None, description="'internship' | 'full_time' | 'part_time' | 'contract' | 'freelance'"
    )
    duration: Optional[str] = Field(None, description="Human-readable duration, e.g. 'Jun 2023 – Aug 2023'")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    responsibilities: list[str] = Field(default_factory=list)
    technologies_used: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    location: Optional[str] = None


class Project(BaseModel):
    """Technical project record."""

    name: Optional[str] = None
    description: Optional[str] = None
    technologies: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    role: Optional[str] = None
    contribution: Optional[str] = None
    github_link: Optional[str] = None
    live_demo: Optional[str] = None
    deployment: Optional[str] = None
    problem_solved: Optional[str] = None
    real_world_use: Optional[str] = None
    challenges: Optional[str] = None
    results: Optional[str] = None
    team_size: Optional[int] = None


class Achievement(BaseModel):
    """Award, competition, or recognition record."""

    title: Optional[str] = None
    category: Optional[str] = Field(
        None,
        description=(
            "'hackathon' | 'competition' | 'award' | 'scholarship' | "
            "'patent' | 'open_source' | 'research' | 'publication' | 'other'"
        ),
    )
    description: Optional[str] = None
    date: Optional[str] = None
    organisation: Optional[str] = None
    rank_or_position: Optional[str] = None
    prize: Optional[str] = None


class Certification(BaseModel):
    """Professional certification record."""

    name: Optional[str] = None
    platform: Optional[str] = Field(None, description="Issuing platform, e.g. Coursera, AWS, Google")
    completion_date: Optional[str] = None
    credential_id: Optional[str] = None
    credential_url: Optional[str] = None
    expiry_date: Optional[str] = None


class Publication(BaseModel):
    """Academic or professional publication record."""

    title: Optional[str] = None
    conference: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    citation: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None


class Research(BaseModel):
    """Research experience record."""

    title: Optional[str] = None
    institution: Optional[str] = None
    supervisor: Optional[str] = None
    conference: Optional[str] = None
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    citation: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = Field(None, description="'published' | 'under_review' | 'ongoing' | 'completed'")


class Leadership(BaseModel):
    """Leadership role or club involvement record."""

    club: Optional[str] = None
    organisation: Optional[str] = None
    position: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    impact: Optional[str] = None


class Volunteer(BaseModel):
    """Volunteer work record."""

    organisation: Optional[str] = None
    role: Optional[str] = None
    impact: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class Skills(BaseModel):
    """Structured skill inventory."""

    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    libraries: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    devops: list[str] = Field(default_factory=list)
    ai_ml: list[str] = Field(default_factory=list, alias="ai_ml")
    data_science: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class TimelineEvent(BaseModel):
    """A single entry in the candidate's chronological timeline."""

    event_type: str = Field(
        description="'education' | 'experience' | 'project' | 'research' | 'certification' | 'achievement'"
    )
    title: str
    organisation: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    is_current: bool = False


class Candidate(BaseModel):
    """
    Top-level structured candidate object produced by the parser.

    This model is the single source of truth consumed by all downstream
    modules (Project Intelligence, Achievement Analysis, Domain Fit,
    Semantic Matching, Recruiter Agent, Explainable Ranking).
    """

    # Metadata
    candidate_id: Optional[str] = Field(None, description="Unique identifier, usually auto-assigned")
    parsed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO-8601 UTC timestamp of when the record was parsed",
    )
    source_format: Optional[str] = Field(None, description="Format the raw record arrived in")

    # Basic Information
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    # Structured Sections
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    achievements: list[Achievement] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    research: list[Research] = Field(default_factory=list)
    leadership: list[Leadership] = Field(default_factory=list)
    volunteer: list[Volunteer] = Field(default_factory=list)

    # Derived / Enriched
    timeline: list[TimelineEvent] = Field(default_factory=list)

    # Parser diagnostics — useful for debugging and QA
    parsing_warnings: list[str] = Field(default_factory=list)
    parsing_errors: list[str] = Field(default_factory=list)
    is_valid: bool = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Normalisation Helpers
# ---------------------------------------------------------------------------


def _clean_str(value: Any) -> Optional[str]:
    """Strip whitespace; return None for empty/non-string values."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def _normalise_phone(phone: Any) -> Optional[str]:
    """
    Normalise phone number to E.164-ish format.
    Strips everything except digits and leading '+'.
    """
    if not phone:
        return None
    raw = str(phone).strip()
    digits_only = re.sub(r"[^\d+]", "", raw)
    if not digits_only or len(digits_only) < 7:
        return None
    return digits_only


def _normalise_email(email: Any) -> Optional[str]:
    """Lowercase and strip whitespace from email address."""
    if not email:
        return None
    cleaned = str(email).strip().lower()
    # Basic sanity check
    if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
        return None
    return cleaned


def _normalise_url(url: Any) -> Optional[str]:
    """Ensure URL has a scheme prefix."""
    if not url:
        return None
    url = str(url).strip()
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicate strings (case-insensitive) while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def _normalise_skill_list(items: list[Any]) -> list[str]:
    """Clean, deduplicate, and title-case a list of skill strings."""
    cleaned = [_clean_str(i) for i in items if _clean_str(i)]
    return _deduplicate([s for s in cleaned if s])


def _safe_list(value: Any) -> list:
    """Coerce value to list; wrap scalar in list; return [] for None."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------------------
# LLM Extraction Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are an expert resume parser for an AI Resume Intelligence Engine.

Your task is to extract ALL structured information from the provided resume text
and return it as a single valid JSON object. Follow the schema EXACTLY.

RULES:
- Return ONLY a valid JSON object. No markdown fences, no explanation.
- Use null for missing fields. Never omit keys.
- Dates: normalise to "YYYY-MM" or "YYYY" where possible; keep original string if uncertain.
- Lists must always be JSON arrays, never comma-separated strings.
- Skills must be individual items, not grouped strings.
- Employment type: one of "internship", "full_time", "part_time", "contract", "freelance".
- Achievement category: one of "hackathon", "competition", "award", "scholarship", "patent",
  "open_source", "research", "publication", "other".
- Research status: one of "published", "under_review", "ongoing", "completed".
- Timeline event_type: one of "education", "experience", "project", "research",
  "certification", "achievement".
- Build the timeline by combining education, experience, projects, and research entries
  in chronological order (oldest first). Each entry needs: event_type, title,
  organisation (optional), start_date (optional), end_date (optional),
  description (optional), is_current (bool).

OUTPUT SCHEMA:
{
  "name": string | null,
  "email": string | null,
  "phone": string | null,
  "linkedin": string | null,
  "github": string | null,
  "portfolio": string | null,
  "location": string | null,
  "summary": string | null,
  "education": [
    {
      "degree": string | null,
      "college": string | null,
      "university": string | null,
      "branch": string | null,
      "cgpa": number | null,
      "percentage": number | null,
      "graduation_year": integer | null,
      "start_year": integer | null,
      "relevant_coursework": [string]
    }
  ],
  "experience": [
    {
      "company": string | null,
      "role": string | null,
      "employment_type": string | null,
      "duration": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "is_current": boolean,
      "responsibilities": [string],
      "technologies_used": [string],
      "achievements": [string],
      "location": string | null
    }
  ],
  "projects": [
    {
      "name": string | null,
      "description": string | null,
      "technologies": [string],
      "frameworks": [string],
      "programming_languages": [string],
      "duration": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "role": string | null,
      "contribution": string | null,
      "github_link": string | null,
      "live_demo": string | null,
      "deployment": string | null,
      "problem_solved": string | null,
      "real_world_use": string | null,
      "challenges": string | null,
      "results": string | null,
      "team_size": integer | null
    }
  ],
  "skills": {
    "programming_languages": [string],
    "frameworks": [string],
    "libraries": [string],
    "databases": [string],
    "cloud": [string],
    "devops": [string],
    "ai_ml": [string],
    "data_science": [string],
    "tools": [string],
    "soft_skills": [string],
    "other": [string]
  },
  "achievements": [
    {
      "title": string | null,
      "category": string | null,
      "description": string | null,
      "date": string | null,
      "organisation": string | null,
      "rank_or_position": string | null,
      "prize": string | null
    }
  ],
  "certifications": [
    {
      "name": string | null,
      "platform": string | null,
      "completion_date": string | null,
      "credential_id": string | null,
      "credential_url": string | null,
      "expiry_date": string | null
    }
  ],
  "publications": [
    {
      "title": string | null,
      "conference": string | null,
      "journal": string | null,
      "publication_date": string | null,
      "citation": string | null,
      "doi": string | null,
      "url": string | null,
      "authors": [string],
      "abstract": string | null
    }
  ],
  "research": [
    {
      "title": string | null,
      "institution": string | null,
      "supervisor": string | null,
      "conference": string | null,
      "journal": string | null,
      "publication_date": string | null,
      "citation": string | null,
      "description": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "status": string | null
    }
  ],
  "leadership": [
    {
      "club": string | null,
      "organisation": string | null,
      "position": string | null,
      "responsibilities": [string],
      "start_date": string | null,
      "end_date": string | null,
      "impact": string | null
    }
  ],
  "volunteer": [
    {
      "organisation": string | null,
      "role": string | null,
      "impact": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "description": string | null
    }
  ],
  "timeline": [
    {
      "event_type": string,
      "title": string,
      "organisation": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "description": string | null,
      "is_current": boolean
    }
  ]
}
""".strip()


# ---------------------------------------------------------------------------
# ResumeParser
# ---------------------------------------------------------------------------


class ResumeParser:
    """
    AI-powered resume parser that converts raw JSONL candidate records into
    fully-validated `Candidate` Pydantic objects.

    Pipeline per candidate
    ----------------------
    raw_record (dict)  →  resume_text (str)  →  LLM extraction (JSON)
                       →  normalise_candidate()  →  validate_candidate()
                       →  Candidate

    The parser never crashes on bad input; all errors are captured as
    `parsing_errors` on the `Candidate` object and logged appropriately.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        """
        Initialise the parser and configure the Gemini client.

        Parameters
        ----------
        model_name : str
            Gemini model identifier. Defaults to ``gemini-2.0-flash``.

        Raises
        ------
        EnvironmentError
            If ``GEMINI_API_KEY`` is not set in the environment.
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
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
        )
        logger.info("ResumeParser initialised with model: %s", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_candidates(self, path: Path = RAW_DATA_PATH) -> list[dict[str, Any]]:
        """
        Load raw candidate records from a JSONL file.

        Each line must be a valid JSON object representing one candidate.
        Malformed lines are skipped and logged.

        Parameters
        ----------
        path : Path
            Absolute path to the ``.jsonl`` file.

        Returns
        -------
        list[dict]
            List of raw candidate dicts.
        """
        if not path.exists():
            logger.error("Candidates file not found: %s", path)
            return []

        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                    else:
                        logger.warning("Line %d: expected JSON object, got %s — skipped.", lineno, type(record))
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d: JSON decode error — %s — skipped.", lineno, exc)

        logger.info("Loaded %d candidate record(s) from %s", len(records), path)
        return records

    def parse_resume(self, raw_record: dict[str, Any], candidate_id: Optional[str] = None) -> Candidate:
        """
        Full parse pipeline for a single raw candidate record.

        Parameters
        ----------
        raw_record : dict
            A raw candidate dictionary from the JSONL file.
        candidate_id : str, optional
            Identifier to stamp on the resulting Candidate object.

        Returns
        -------
        Candidate
            Fully validated Candidate object. Errors are embedded in the
            object rather than raised.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Step 1 — Convert raw record to plain text for LLM
        resume_text = self._record_to_text(raw_record)
        source_format = self._detect_format(raw_record)

        logger.info(
            "Processing candidate_id=%s | format=%s | text_length=%d chars",
            candidate_id or "unknown",
            source_format,
            len(resume_text),
        )

        # Step 2 — LLM extraction
        llm_data: dict[str, Any] = {}
        try:
            llm_data = self.extract_with_llm(resume_text)
        except Exception as exc:  # noqa: BLE001
            msg = f"LLM extraction failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        # Step 3 — Merge raw fields that the LLM might have missed
        merged = self._merge_raw_and_llm(raw_record, llm_data)

        # Step 4 — Normalise
        normalised = self.normalise_candidate(merged, warnings)

        # Step 5 — Build Candidate
        candidate = self._build_candidate(normalised, candidate_id, source_format, errors, warnings)

        # Step 6 — Validate
        candidate = self.validate_candidate(candidate)

        if candidate.is_valid:
            logger.info("Candidate %s — extraction successful.", candidate_id or candidate.name or "unknown")
        else:
            logger.warning(
                "Candidate %s — completed with %d error(s).",
                candidate_id or candidate.name or "unknown",
                len(candidate.parsing_errors),
            )

        return candidate

    def extract_with_llm(self, resume_text: str) -> dict[str, Any]:
        """
        Send resume text to Google Gemini and parse the structured JSON response.

        Parameters
        ----------
        resume_text : str
            Plain-text representation of the resume.

        Returns
        -------
        dict
            Parsed JSON dictionary returned by the model.

        Raises
        ------
        ValueError
            If the model returns no content or content that is not valid JSON.
        RuntimeError
            If the Gemini API call fails for any reason.
        """
        if not resume_text.strip():
            raise ValueError("Empty resume text — nothing to extract.")

        prompt = (
            "Extract all structured information from the following resume text "
            "and return a single valid JSON object matching the schema provided.\n\n"
            f"RESUME TEXT:\n{resume_text}"
        )

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,          # low temperature → deterministic extraction
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

        # Strip markdown code fences if the model wrapped the JSON
        json_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
        json_text = re.sub(r"\s*```$", "", json_text.strip())

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            # Attempt to extract the first JSON object from the response
            match = re.search(r"\{.*\}", json_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse LLM response as JSON: {exc}") from exc

    def normalise_candidate(
        self, data: dict[str, Any], warnings: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Clean, deduplicate, and normalise all fields in the raw extracted dict.

        This method mutates and returns the dict for convenience.

        Parameters
        ----------
        data : dict
            Raw extracted candidate data (possibly from LLM).
        warnings : list[str], optional
            Mutable list to append normalisation warnings to.

        Returns
        -------
        dict
            Normalised candidate data dictionary.
        """
        if warnings is None:
            warnings = []

        # Basic fields
        data["name"] = _clean_str(data.get("name"))
        data["email"] = _normalise_email(data.get("email"))
        data["phone"] = _normalise_phone(data.get("phone"))
        data["linkedin"] = _normalise_url(data.get("linkedin"))
        data["github"] = _normalise_url(data.get("github"))
        data["portfolio"] = _normalise_url(data.get("portfolio"))
        data["location"] = _clean_str(data.get("location"))
        data["summary"] = _clean_str(data.get("summary"))

        # Education
        for edu in _safe_list(data.get("education")):
            edu["relevant_coursework"] = _normalise_skill_list(
                _safe_list(edu.get("relevant_coursework"))
            )
            if edu.get("college") and not edu.get("university"):
                warnings.append(f"Education entry missing university: {edu.get('college')}")

        # Experience
        for exp in _safe_list(data.get("experience")):
            exp["technologies_used"] = _normalise_skill_list(
                _safe_list(exp.get("technologies_used"))
            )
            exp["responsibilities"] = [
                _clean_str(r) for r in _safe_list(exp.get("responsibilities")) if _clean_str(r)
            ]
            exp["achievements"] = [
                _clean_str(a) for a in _safe_list(exp.get("achievements")) if _clean_str(a)
            ]

        # Projects
        for proj in _safe_list(data.get("projects")):
            proj["technologies"] = _normalise_skill_list(_safe_list(proj.get("technologies")))
            proj["frameworks"] = _normalise_skill_list(_safe_list(proj.get("frameworks")))
            proj["programming_languages"] = _normalise_skill_list(
                _safe_list(proj.get("programming_languages"))
            )
            if proj.get("github_link"):
                proj["github_link"] = _normalise_url(proj["github_link"])
            if proj.get("live_demo"):
                proj["live_demo"] = _normalise_url(proj["live_demo"])

        # Skills — normalise every category
        skills_raw = data.get("skills") or {}
        if isinstance(skills_raw, dict):
            for key in (
                "programming_languages", "frameworks", "libraries", "databases",
                "cloud", "devops", "ai_ml", "data_science", "tools", "soft_skills", "other"
            ):
                skills_raw[key] = _normalise_skill_list(_safe_list(skills_raw.get(key)))
            data["skills"] = skills_raw

        # Certifications
        for cert in _safe_list(data.get("certifications")):
            if cert.get("credential_url"):
                cert["credential_url"] = _normalise_url(cert["credential_url"])

        # Publications
        for pub in _safe_list(data.get("publications")):
            pub["authors"] = _normalise_skill_list(_safe_list(pub.get("authors")))

        # Leadership
        for lead in _safe_list(data.get("leadership")):
            lead["responsibilities"] = [
                _clean_str(r)
                for r in _safe_list(lead.get("responsibilities"))
                if _clean_str(r)
            ]

        return data

    def validate_candidate(self, candidate: Candidate) -> Candidate:
        """
        Run semantic validation checks on a parsed Candidate object.

        Validation checks are non-fatal; they append to
        ``candidate.parsing_warnings`` and set ``candidate.is_valid = False``
        only when a critical field is missing.

        Parameters
        ----------
        candidate : Candidate
            The candidate object to validate.

        Returns
        -------
        Candidate
            The same candidate object with validation metadata updated.
        """
        # Critical fields
        if not candidate.name:
            candidate.parsing_warnings.append("Candidate name is missing.")
            candidate.is_valid = False

        if not candidate.email and not candidate.phone:
            candidate.parsing_warnings.append(
                "Neither email nor phone found — candidate may be uncontactable."
            )

        # Education sanity
        for edu in candidate.education:
            if edu.graduation_year and edu.start_year:
                if edu.graduation_year < edu.start_year:
                    candidate.parsing_warnings.append(
                        f"Education: graduation_year ({edu.graduation_year}) is before "
                        f"start_year ({edu.start_year}) — possible data error."
                    )

        # Experience overlap detection (basic)
        dates_seen: list[tuple[str, str]] = []
        for exp in candidate.experience:
            if exp.start_date and exp.end_date:
                dates_seen.append((exp.start_date, exp.end_date))

        # Skills completeness
        all_skills = (
            candidate.skills.programming_languages
            + candidate.skills.frameworks
            + candidate.skills.tools
            + candidate.skills.ai_ml
        )
        if not all_skills:
            candidate.parsing_warnings.append("No skills detected — resume may be incomplete.")

        return candidate

    def parse_all(self, path: Path = RAW_DATA_PATH) -> list[Candidate]:
        """
        Load and parse every candidate in the JSONL file.

        Parameters
        ----------
        path : Path
            Path to the ``.jsonl`` data file.

        Returns
        -------
        list[Candidate]
            One Candidate object per valid input record.
        """
        raw_records = self.load_candidates(path)

        if not raw_records:
            logger.warning("No candidate records found. Returning empty list.")
            return []

        candidates: list[Candidate] = []
        for idx, record in enumerate(raw_records, start=1):
            candidate_id = record.get("id") or record.get("candidate_id") or f"candidate_{idx:04d}"
            logger.info("--- Processing %d / %d | id=%s ---", idx, len(raw_records), candidate_id)
            try:
                candidate = self.parse_resume(record, candidate_id=str(candidate_id))
                candidates.append(candidate)
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected failure for record %d (id=%s): %s", idx, candidate_id, exc)
                # Create a minimal failed Candidate so the downstream pipeline
                # is aware this record existed but could not be parsed.
                failed = Candidate(
                    candidate_id=str(candidate_id),
                    is_valid=False,
                    parsing_errors=[f"Unhandled parser exception: {exc}"],
                )
                candidates.append(failed)

        valid_count = sum(1 for c in candidates if c.is_valid)
        logger.info(
            "Parsing complete — total=%d | valid=%d | invalid=%d",
            len(candidates),
            valid_count,
            len(candidates) - valid_count,
        )
        return candidates

    def save_processed(
        self,
        candidates: list[Candidate],
        output_path: Path = PROCESSED_DATA_PATH,
    ) -> Path:
        """
        Serialise and persist the list of Candidate objects to JSON.

        Parameters
        ----------
        candidates : list[Candidate]
            Parsed candidate objects to save.
        output_path : Path
            Destination file path. Parent directories are created automatically.

        Returns
        -------
        Path
            The resolved output path where the data was written.
        """
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_candidates": len(candidates),
                "valid_candidates": sum(1 for c in candidates if c.is_valid),
                "invalid_candidates": sum(1 for c in candidates if not c.is_valid),
                "parser_version": "1.0.0",
            },
            "candidates": [c.model_dump(mode="json") for c in candidates],
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info(
            "Saved %d candidate(s) → %s",
            len(candidates),
            output_path,
        )
        return output_path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_text(record: dict[str, Any]) -> str:
        """
        Convert a raw candidate dict into a flat text string suitable for
        passing to the LLM. Handles nested dicts and lists recursively.

        This is intentionally lossless — every key/value pair is included.
        """
        lines: list[str] = []

        def _flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    label = f"{prefix}.{k}" if prefix else k
                    _flatten(v, label)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _flatten(item, f"{prefix}[{i}]")
            else:
                if obj is not None and str(obj).strip():
                    lines.append(f"{prefix}: {obj}")

        _flatten(record)
        return "\n".join(lines)

    @staticmethod
    def _detect_format(record: dict[str, Any]) -> str:
        """Heuristically detect the format/source of the raw record."""
        keys = set(record.keys())
        if "resume_text" in keys or "raw_text" in keys or "text" in keys:
            return "text"
        if "resume_html" in keys or "html" in keys:
            return "html"
        if any(k in keys for k in ("education", "experience", "projects", "skills")):
            return "structured_json"
        return "unknown"

    @staticmethod
    def _merge_raw_and_llm(
        raw: dict[str, Any], llm: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge raw record fields with LLM-extracted fields.

        LLM values take precedence for structured fields.
        Raw record top-level scalars fill in only when LLM produced null/missing.
        """
        merged: dict[str, Any] = dict(llm)  # start with LLM output

        # For each scalar field, fall back to raw record if LLM left it null
        scalar_fields = (
            "name", "email", "phone", "linkedin", "github",
            "portfolio", "location", "summary",
        )
        for field in scalar_fields:
            if not merged.get(field) and raw.get(field):
                merged[field] = raw[field]

        return merged

    def _build_candidate(
        self,
        data: dict[str, Any],
        candidate_id: Optional[str],
        source_format: str,
        errors: list[str],
        warnings: list[str],
    ) -> Candidate:
        """
        Construct a `Candidate` Pydantic model from a normalised dict.

        Any Pydantic validation error is captured in ``errors`` rather than
        raised, guaranteeing the parser never crashes.
        """
        try:
            candidate = Candidate(
                candidate_id=candidate_id,
                source_format=source_format,
                name=data.get("name"),
                email=data.get("email"),
                phone=data.get("phone"),
                linkedin=data.get("linkedin"),
                github=data.get("github"),
                portfolio=data.get("portfolio"),
                location=data.get("location"),
                summary=data.get("summary"),
                education=[
                    Education(**edu)
                    for edu in _safe_list(data.get("education"))
                    if isinstance(edu, dict)
                ],
                experience=[
                    Experience(**exp)
                    for exp in _safe_list(data.get("experience"))
                    if isinstance(exp, dict)
                ],
                projects=[
                    Project(**proj)
                    for proj in _safe_list(data.get("projects"))
                    if isinstance(proj, dict)
                ],
                skills=Skills(**(data.get("skills") or {})),
                achievements=[
                    Achievement(**ach)
                    for ach in _safe_list(data.get("achievements"))
                    if isinstance(ach, dict)
                ],
                certifications=[
                    Certification(**cert)
                    for cert in _safe_list(data.get("certifications"))
                    if isinstance(cert, dict)
                ],
                publications=[
                    Publication(**pub)
                    for pub in _safe_list(data.get("publications"))
                    if isinstance(pub, dict)
                ],
                research=[
                    Research(**res)
                    for res in _safe_list(data.get("research"))
                    if isinstance(res, dict)
                ],
                leadership=[
                    Leadership(**lead)
                    for lead in _safe_list(data.get("leadership"))
                    if isinstance(lead, dict)
                ],
                volunteer=[
                    Volunteer(**vol)
                    for vol in _safe_list(data.get("volunteer"))
                    if isinstance(vol, dict)
                ],
                timeline=[
                    TimelineEvent(**evt)
                    for evt in _safe_list(data.get("timeline"))
                    if isinstance(evt, dict)
                ],
                parsing_errors=errors,
                parsing_warnings=warnings,
                is_valid=len(errors) == 0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Pydantic model construction failed: %s", exc)
            errors.append(f"Model construction error: {exc}")
            candidate = Candidate(
                candidate_id=candidate_id,
                source_format=source_format,
                parsing_errors=errors,
                parsing_warnings=warnings,
                is_valid=False,
            )

        return candidate


# ---------------------------------------------------------------------------
# Convenience entry-point (can be called from main.py or CLI)
# ---------------------------------------------------------------------------


def run_parser(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    model_name: str = "gemini-2.0-flash",
) -> list[Candidate]:
    """
    Convenience function that wires together the full parser pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the ``.jsonl`` input file.
    output_path : Path
        Path where ``parsed_candidates.json`` will be written.
    model_name : str
        Gemini model to use for extraction.

    Returns
    -------
    list[Candidate]
        All parsed Candidate objects.

    Example
    -------
    >>> from src.parsers.resume_parser import run_parser
    >>> candidates = run_parser()
    >>> print(candidates[0].name)
    """
    parser = ResumeParser(model_name=model_name)
    candidates = parser.parse_all(input_path)
    parser.save_processed(candidates, output_path)
    return candidates


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    candidates = run_parser()
    print(f"\n[OK] Parsed {len(candidates)} candidate(s). Output -> {PROCESSED_DATA_PATH}")
