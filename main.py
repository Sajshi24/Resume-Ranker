"""
main.py
=======
AI Resume Intelligence Engine — Complete Pipeline Orchestrator

Runs the full end-to-end pipeline:
    1. Parse Job Description
    2. Parse Resumes
    3. Score All Candidates (7 dimensions each)
    4. Run Intelligence Agents (Project, Skill, Growth, Recruiter)
    5. Generate Final Ranked Output
    6. Save to CSV + JSON

Outputs
-------
    outputs/ranked_candidates.csv   — sortable ranking table
    outputs/final_results.json      — complete explainable results
    outputs/debug_scores.json       — raw scoring data for debugging

Usage
-----
    python main.py
    python main.py --jd path/to/jd.txt --candidates path/to/candidates.jsonl
    python main.py --skip-parse     # use cached parsed data

Author  : Resume Intelligence Engine — Pipeline Layer
Python  : 3.11+
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
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

# ---------------------------------------------------------------------------
# Bootstrap logging before importing anything else
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Project imports (after logging bootstrap)
# ---------------------------------------------------------------------------

from src.parsers.resume_parser import run_parser, ResumeParser, PROCESSED_DATA_PATH, Candidate
from src.parsers.jd_parser import run_jd_parser, JDParser, JD_PROCESSED_PATH, JobProfile
from src.scoring.final_score import FinalScorer, FinalScoreResult, FINAL_DIMENSION_WEIGHTS
from src.agents.project_agent import ProjectAgent, ProjectAgentReport
from src.agents.skill_agent import SkillAgent, SkillAgentReport
from src.agents.growth_agent import GrowthAgent, GrowthAgentReport
from src.agents.recruiter_agent import RecruiterAgent, RecruiterReport
from src.utils.config import (
    CANDIDATES_JSONL,
    JD_TXT,
    OUTPUTS_DIR,
    PARSED_CANDIDATES_JSON,
    PARSED_JD_JSON,
    ensure_output_dirs,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

RANKED_CSV: Path = OUTPUTS_DIR / "ranked_candidates.csv"
FINAL_RESULTS_JSON: Path = OUTPUTS_DIR / "final_results.json"
DEBUG_SCORES_JSON: Path = OUTPUTS_DIR / "debug_scores.json"
DASHBOARD_CACHE_JSON: Path = OUTPUTS_DIR / "dashboard_cache.json"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Full AI Resume Intelligence Pipeline.

    Wraps every stage: parse → score → agents → rank → save.
    Each stage is independently skippable for iterative development.

    Usage
    -----
    >>> pipeline = Pipeline()
    >>> pipeline.run()
    """

    def __init__(
        self,
        jd_path: Path = JD_TXT,
        candidates_path: Path = CANDIDATES_JSONL,
        model_name: str = "gemini-2.0-flash",
        skip_parse: bool = False,
    ) -> None:
        self.jd_path = jd_path
        self.candidates_path = candidates_path
        self.model_name = model_name
        self.skip_parse = skip_parse

        # Stage results (populated sequentially)
        self.job_profile: Optional[JobProfile] = None
        self.candidates: list[Candidate] = []
        self.score_results: list[FinalScoreResult] = []
        self.recruiter_reports: list[RecruiterReport] = []

        # Agents (initialised once)
        self._project_agent = ProjectAgent()
        self._skill_agent = SkillAgent()
        self._growth_agent = GrowthAgent()
        self._recruiter_agent = RecruiterAgent(model_name=model_name)

        logger.info("Pipeline initialised | model=%s | skip_parse=%s", model_name, skip_parse)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> list[RecruiterReport]:
        """
        Execute the complete pipeline end-to-end.

        Returns
        -------
        list[RecruiterReport]
            Ranked recruiter reports (highest score first).
        """
        start = time.time()
        ensure_output_dirs()

        logger.info("=" * 60)
        logger.info("AI Resume Intelligence Engine — Pipeline Start")
        logger.info("=" * 60)

        # Stage 1: Parse JD
        self._stage_parse_jd()

        # Stage 2: Parse Resumes
        self._stage_parse_resumes()

        if not self.candidates:
            logger.error("No candidates to process. Exiting.")
            return []

        if self.job_profile is None:
            logger.error("No job profile to process. Exiting.")
            return []

        # Stage 3: Score
        self._stage_score()

        # Stage 4: Run Agents
        self._stage_run_agents()

        # Stage 5: Save Outputs
        self._stage_save_outputs()

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(
            "Pipeline Complete | %d candidates | %.1fs",
            len(self.recruiter_reports),
            elapsed,
        )
        self._print_summary()
        logger.info("=" * 60)

        return self.recruiter_reports

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_parse_jd(self) -> None:
        """Stage 1 — Parse the job description."""
        logger.info("[Stage 1/5] Parsing Job Description...")

        if self.skip_parse and PARSED_JD_JSON.exists():
            logger.info("  -> Loading cached parsed JD from %s", PARSED_JD_JSON)
            self.job_profile = self._load_cached_jd()
        else:
            self.job_profile = run_jd_parser(
                input_path=self.jd_path,
                output_path=PARSED_JD_JSON,
                model_name=self.model_name,
            )

        if self.job_profile:
            logger.info(
                "  [OK] JD parsed | role=%s | company=%s",
                self.job_profile.role_title,
                self.job_profile.company_name,
            )
        else:
            logger.error("  [FAIL] JD parsing failed.")

    def _stage_parse_resumes(self) -> None:
        """Stage 2 — Parse all candidate resumes."""
        logger.info("[Stage 2/5] Parsing Resumes...")

        if self.skip_parse and PARSED_CANDIDATES_JSON.exists():
            logger.info("  -> Loading cached parsed candidates from %s", PARSED_CANDIDATES_JSON)
            self.candidates = self._load_cached_candidates()
        else:
            self.candidates = run_parser(
                input_path=self.candidates_path,
                output_path=PARSED_CANDIDATES_JSON,
                model_name=self.model_name,
            )

        logger.info("  [OK] Parsed %d candidate(s).", len(self.candidates))

    def _stage_score(self) -> None:
        """Stage 3 — Score all candidates with the 7-module scoring engine."""
        logger.info(
            "[Stage 3/5] Scoring %d candidate(s)...",
            len(self.candidates),
        )
        scorer = FinalScorer(model_name=self.model_name)
        self.score_results = scorer.score_batch(self.candidates, self.job_profile)

        # Save debug scores immediately
        self._save_debug_scores(self.score_results)
        logger.info("  [OK] Scoring complete. Debug scores -> %s", DEBUG_SCORES_JSON)

    def _stage_run_agents(self) -> None:
        """Stage 4 — Run intelligence agents for all candidates."""
        logger.info("[Stage 4/5] Running Intelligence Agents...")

        total = len(self.score_results)
        reports: list[RecruiterReport] = []

        # Build a lookup from candidate_id → Candidate
        cand_lookup: dict[str, Any] = {
            str(getattr(c, "candidate_id", i)): c
            for i, c in enumerate(self.candidates)
        }

        for rank, final_result in enumerate(self.score_results, start=1):
            cid = str(getattr(final_result, "candidate_id", ""))
            candidate = cand_lookup.get(cid) or (
                self.candidates[rank - 1] if rank <= len(self.candidates) else None
            )
            if candidate is None:
                logger.warning("Could not find candidate for result %s. Skipping.", cid)
                continue

            logger.info(
                "  [%d/%d] Running agents for %s...",
                rank, total,
                getattr(candidate, "name", cid),
            )

            # Project Agent
            proj_dim = getattr(final_result, "per_dimension_results", {}).get("project", {})
            project_score_result = _dict_to_namespace(proj_dim)
            project_report = self._project_agent.interpret(candidate, project_score_result)

            # Skill Agent
            skill_dim = getattr(final_result, "per_dimension_results", {}).get("skill", {})
            skill_score_result = _dict_to_namespace(skill_dim)
            skill_report = self._skill_agent.interpret(candidate, skill_score_result, self.job_profile)

            # Growth Agent
            growth_dim = getattr(final_result, "per_dimension_results", {}).get("growth", {})
            learning_dim = getattr(final_result, "per_dimension_results", {}).get("learning", {})
            growth_score_result = _dict_to_namespace(growth_dim)
            learning_score_result = _dict_to_namespace(learning_dim)
            growth_report = self._growth_agent.interpret(
                candidate, growth_score_result, learning_score_result
            )

            # Recruiter Agent (decision maker)
            recruiter_report = self._recruiter_agent.decide(
                candidate=candidate,
                job_profile=self.job_profile,
                final_result=final_result,
                project_report=project_report,
                skill_report=skill_report,
                growth_report=growth_report,
                rank=rank,
            )
            reports.append(recruiter_report)

        self.recruiter_reports = reports
        logger.info("  [OK] Agents complete. %d reports generated.", len(reports))

    def _stage_save_outputs(self) -> None:
        """Stage 5 — Save all outputs."""
        logger.info("[Stage 5/5] Saving Outputs...")
        self._save_ranked_csv()
        self._save_final_results()
        self._save_dashboard_cache()
        logger.info("  [OK] All outputs saved.")

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_ranked_csv(self) -> None:
        """Save ranked_candidates.csv."""
        RANKED_CSV.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank", "name", "overall_score", "potential_score",
            "projects", "domain_fit", "skills", "learning",
            "soft_skills", "growth", "semantic_fit",
            "hiring_recommendation", "confidence",
        ]
        with RANKED_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.recruiter_reports:
                sb = r.score_breakdown
                writer.writerow({
                    "rank": r.rank,
                    "name": r.candidate_name or "",
                    "overall_score": r.overall_score,
                    "potential_score": r.potential_score,
                    "projects": sb.get("Projects", 0),
                    "domain_fit": sb.get("Domain Fit", 0),
                    "skills": sb.get("Skills", 0),
                    "learning": sb.get("Learning", 0),
                    "soft_skills": sb.get("Soft Skills", 0),
                    "growth": sb.get("Growth", 0),
                    "semantic_fit": sb.get("Semantic Fit", 0),
                    "hiring_recommendation": r.hiring_recommendation,
                    "confidence": r.confidence,
                })
        logger.info("  -> %s", RANKED_CSV)

    def _save_final_results(self) -> None:
        """Save final_results.json."""
        FINAL_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "total_candidates": len(self.recruiter_reports),
                "model": self.model_name,
                "pipeline_version": "1.0.0",
                "scoring_weights": FINAL_DIMENSION_WEIGHTS,
            },
            "job_profile": (
                self.job_profile.model_dump(mode="json")
                if self.job_profile else {}
            ),
            "ranked_candidates": [r.model_dump(mode="json") for r in self.recruiter_reports],
        }
        with FINAL_RESULTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        logger.info("  -> %s", FINAL_RESULTS_JSON)

    def _save_debug_scores(self, results: list[FinalScoreResult]) -> None:
        """Save debug_scores.json."""
        DEBUG_SCORES_JSON.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scores": [r.model_dump(mode="json") for r in results],
        }
        with DEBUG_SCORES_JSON.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def _save_dashboard_cache(self) -> None:
        """Save dashboard_cache.json — optimised for fast dashboard loading."""
        DASHBOARD_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)

        # Build a lightweight version for the ranking table
        ranking_table = []
        for r in self.recruiter_reports:
            ranking_table.append({
                "rank": r.rank,
                "candidate_id": r.candidate_id,
                "name": r.candidate_name,
                "overall_score": r.overall_score,
                "potential_score": r.potential_score,
                "hiring_recommendation": r.hiring_recommendation,
                "hiring_recommendation_label": r.hiring_recommendation_label,
                "confidence": r.confidence,
                "one_liner": r.one_liner,
                "score_breakdown": r.score_breakdown,
                "strengths": r.strengths[:3],
                "weaknesses": r.weaknesses[:3],
            })

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "job_profile": {
                "role_title": getattr(self.job_profile, "role_title", None),
                "company_name": getattr(self.job_profile, "company_name", None),
                "location": getattr(self.job_profile, "location", None),
                "seniority_level": getattr(
                    getattr(self.job_profile, "experience_requirements", None),
                    "seniority_level", None
                ),
            } if self.job_profile else {},
            "ranking_table": ranking_table,
            "full_reports_path": str(FINAL_RESULTS_JSON),
        }
        with DASHBOARD_CACHE_JSON.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        logger.info("  -> %s", DASHBOARD_CACHE_JSON)

    # ------------------------------------------------------------------
    # Cached data loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cached_jd() -> Optional[JobProfile]:
        """Load a previously saved JobProfile from disk."""
        try:
            with PARSED_JD_JSON.open(encoding="utf-8") as f:
                data = json.load(f)
            # The saved file has a "job_profile" wrapper
            jd_data = data.get("job_profile", data)
            return JobProfile.model_validate(jd_data)
        except Exception as exc:
            logger.warning("Could not load cached JD: %s. Re-parsing.", exc)
            return None

    @staticmethod
    def _load_cached_candidates() -> list[Candidate]:
        """Load previously parsed candidates from disk."""
        try:
            with PARSED_CANDIDATES_JSON.open(encoding="utf-8") as f:
                data = json.load(f)
            cands_data = data.get("candidates", data) if isinstance(data, dict) else data
            if not isinstance(cands_data, list):
                return []
            return [Candidate.model_validate(c) for c in cands_data]
        except Exception as exc:
            logger.warning("Could not load cached candidates: %s. Re-parsing.", exc)
            return []

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print a formatted ranking summary to stdout."""
        print("\n" + "=" * 70)
        print("  AI RESUME INTELLIGENCE ENGINE — FINAL RANKING")
        print("=" * 70)
        for r in self.recruiter_reports:
            rec = r.hiring_recommendation_label or r.hiring_recommendation
            print(
                f"  #{r.rank:2d} | {r.candidate_name or 'Unknown':<25} | "
                f"Score: {r.overall_score:5.1f} | Potential: {r.potential_score:5.1f} | "
                f"{rec}"
            )
        print("=" * 70)
        print(f"  Outputs saved to: {OUTPUTS_DIR}")
        print(f"    -> {RANKED_CSV.name}")
        print(f"    -> {FINAL_RESULTS_JSON.name}")
        print(f"    -> {DEBUG_SCORES_JSON.name}")
        print(f"    -> {DASHBOARD_CACHE_JSON.name}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Utility: dict → namespace (for accessing score sub-results as attributes)
# ---------------------------------------------------------------------------


class _Namespace:
    """Lightweight namespace that exposes dict keys as attributes."""

    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _Namespace(v))
            elif isinstance(v, list):
                setattr(self, k, v)
            else:
                setattr(self, k, v)

    def __getattr__(self, name: str) -> Any:
        return None  # safe fallback


def _dict_to_namespace(d: dict) -> _Namespace:
    """Convert a flat/nested dict into an attribute-accessible namespace."""
    return _Namespace(d or {})


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Resume Intelligence Engine — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jd",
        type=Path,
        default=JD_TXT,
        help=f"Path to job description .txt file (default: {JD_TXT})",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=CANDIDATES_JSONL,
        help=f"Path to candidates .jsonl file (default: {CANDIDATES_JSONL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip parsing and use cached parsed data (faster for re-scoring)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    pipeline = Pipeline(
        jd_path=args.jd,
        candidates_path=args.candidates,
        model_name=args.model,
        skip_parse=args.skip_parse,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
