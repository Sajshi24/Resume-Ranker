"""
dashboard.py
=============
AI Resume Intelligence Engine — Streamlit Dashboard

A modern SaaS-style hiring intelligence dashboard.

Pages:
    🏠 Home          — Upload & run pipeline
    📊 Rankings      — Ranked candidate table with search/sort
    👤 Candidate     — Full candidate detail with all scores + AI insights
    📋 Job Profile   — Structured JD analysis
    📈 Analytics     — Charts, distributions, comparison

Run:
    streamlit run dashboard.py

Author  : Resume Intelligence Engine — Dashboard Layer
Python  : 3.11+
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resume Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI Resume Intelligence Engine — Powered by Gemini",
    },
)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FINAL_RESULTS = OUTPUTS / "final_results.json"
DASHBOARD_CACHE = OUTPUTS / "dashboard_cache.json"
DEBUG_SCORES = OUTPUTS / "debug_scores.json"
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"

# Recommendation colour map
REC_COLORS = {
    "strong_hire":    "#10b981",  # emerald
    "hire":           "#3b82f6",  # blue
    "borderline":     "#f59e0b",  # amber
    "no_hire":        "#ef4444",  # red
    "strong_no_hire": "#7f1d1d",  # dark red
    "unknown":        "#6b7280",  # gray
}

REC_LABELS = {
    "strong_hire":    "⭐ Strong Hire",
    "hire":           "✅ Hire",
    "borderline":     "🟡 Borderline",
    "no_hire":        "❌ No Hire",
    "strong_no_hire": "🚫 Strong No Hire",
    "unknown":        "❓ Unknown",
}

SCORE_DIMS = ["Projects", "Domain Fit", "Skills", "Learning", "Soft Skills", "Growth", "Semantic Fit"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* ── Import fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── App background ── */
    .stApp {
        background: #0f1117;
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b27 !important;
        border-right: 1px solid #1e2535;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding: 1.5rem 1rem; }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2844 100%);
        border: 1px solid #2d3a50;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }

    /* ── Score badge ── */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ── Candidate row ── */
    .candidate-row {
        background: #1a2035;
        border: 1px solid #2d3a50;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .candidate-row:hover {
        border-color: #3b82f6;
        background: #1e2844;
        transform: translateX(4px);
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3a50;
    }

    /* ── Page title ── */
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }

    /* ── Pill tags ── */
    .pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem;
        background: #1e2844;
        border: 1px solid #3b82f6;
        color: #93c5fd;
    }
    .pill.green { border-color: #10b981; color: #6ee7b7; }
    .pill.red { border-color: #ef4444; color: #fca5a5; }
    .pill.amber { border-color: #f59e0b; color: #fcd34d; }

    /* ── Progress bar override ── */
    .stProgress > div > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

    /* ── Divider ── */
    hr { border-color: #2d3a50 !important; }

    /* ── Metric override ── */
    [data-testid="metric-container"] {
        background: #1a2035;
        border: 1px solid #2d3a50;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s ease;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #1a2035 !important;
        border: 1px solid #2d3a50 !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
    }

    /* ── Signal chips ── */
    .signal-positive { color: #10b981; }
    .signal-negative { color: #ef4444; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a2035;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: #1a2035;
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ── Info / Warning boxes ── */
    .stInfo { background: #1e3a5f; border-color: #3b82f6; }
    .stWarning { background: #3d2c00; border-color: #f59e0b; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #3b82f6; border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_final_results() -> Optional[dict]:
    """Load full results from final_results.json."""
    if not FINAL_RESULTS.exists():
        return None
    try:
        with FINAL_RESULTS.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_dashboard_cache() -> Optional[dict]:
    """Load lightweight dashboard cache."""
    if not DASHBOARD_CACHE.exists():
        return None
    try:
        with DASHBOARD_CACHE.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def has_results() -> bool:
    return FINAL_RESULTS.exists() and DASHBOARD_CACHE.exists()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def score_color(score: float) -> str:
    if score >= 75:
        return "#10b981"
    if score >= 60:
        return "#3b82f6"
    if score >= 45:
        return "#f59e0b"
    return "#ef4444"


def rec_badge(rec: str, label: Optional[str] = None) -> str:
    color = REC_COLORS.get(rec, "#6b7280")
    text = label or REC_LABELS.get(rec, rec)
    return (
        f'<span style="background:{color}22; color:{color}; '
        f'border:1px solid {color}; padding:4px 12px; border-radius:9999px; '
        f'font-size:0.82rem; font-weight:600;">{text}</span>'
    )


def score_gauge(score: float, label: str = "Score", key: str = "") -> go.Figure:
    """Create a gauge chart for a score 0-100."""
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"color": "#94a3b8", "size": 14}},
        number={"font": {"color": color, "size": 36}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a2035",
            "bordercolor": "#2d3a50",
            "steps": [
                {"range": [0, 35], "color": "#1a0000"},
                {"range": [35, 62], "color": "#1a1a00"},
                {"range": [62, 78], "color": "#001a10"},
                {"range": [78, 100], "color": "#001a20"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        margin=dict(l=20, r=20, t=40, b=10),
        font_color="#e2e8f0",
    )
    return fig


def radar_chart(scores: dict[str, float], title: str = "") -> go.Figure:
    """Create a filled radar chart for score dimensions."""
    dims = list(scores.keys())
    vals = list(scores.values())
    vals_closed = vals + [vals[0]]
    dims_closed = dims + [dims[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=dims_closed,
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2),
        name=title,
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#475569", size=10),
                gridcolor="#2d3a50",
                linecolor="#2d3a50",
            ),
            angularaxis=dict(
                tickfont=dict(color="#94a3b8", size=11),
                gridcolor="#2d3a50",
                linecolor="#2d3a50",
            ),
            bgcolor="#1a2035",
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        height=350,
        margin=dict(l=40, r=40, t=50, b=20),
        font_color="#e2e8f0",
        showlegend=False,
    )
    return fig


def bar_chart_scores(report: dict, title: str = "Score Breakdown") -> go.Figure:
    sb = report.get("score_breakdown", {})
    dims = [k for k in sb if k != "Overall"]
    vals = [sb[k] for k in dims]
    colors = [score_color(v) for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=dims,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#94a3b8", size=13)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a2035",
        xaxis=dict(range=[0, 110], gridcolor="#2d3a50", tickfont=dict(color="#475569")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
        height=280,
        margin=dict(l=10, r=60, t=40, b=20),
        font_color="#e2e8f0",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------


def sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="font-size:2.5rem;">🧠</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Resume Intelligence</div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">Powered by Gemini AI</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "📊 Rankings": "rankings",
            "👤 Candidate Detail": "candidate",
            "📋 Job Profile": "jd",
            "📈 Analytics": "analytics",
        }

        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")

        if has_results():
            cache = load_dashboard_cache()
            if cache:
                total = len(cache.get("ranking_table", []))
                jinfo = cache.get("job_profile", {})
                st.markdown(f"""
                <div style="background:#1a2035; border:1px solid #2d3a50; border-radius:10px; padding:1rem; font-size:0.82rem; color:#94a3b8;">
                    <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.5rem;">📌 Current Analysis</div>
                    <div>Role: <span style="color:#93c5fd">{jinfo.get('role_title','—')}</span></div>
                    <div>Company: <span style="color:#93c5fd">{jinfo.get('company_name','—')}</span></div>
                    <div>Candidates: <span style="color:#93c5fd">{total}</span></div>
                    <div>Generated: <span style="color:#64748b">{cache.get('generated_at','')[:10]}</span></div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No results yet. Run the pipeline from Home.")

        return pages[selected]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_home() -> None:
    st.markdown('<div class="page-title">AI Resume Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#64748b; margin-bottom:2rem;">Explainable AI-powered candidate ranking for modern recruiting teams.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Upload Data</div>', unsafe_allow_html=True)

        jd_file = st.file_uploader(
            "📋 Job Description (.txt)",
            type=["txt"],
            key="jd_upload",
            help="Upload the job description as a plain text file",
        )
        cands_file = st.file_uploader(
            "👥 Candidates (.jsonl)",
            type=["jsonl", "json"],
            key="cands_upload",
            help="Upload candidates as a JSONL file (one JSON object per line)",
        )

        st.markdown('<div class="section-header">Or Use Existing Files</div>', unsafe_allow_html=True)

        jd_exists = (RAW_DATA / "job_description.txt").exists()
        cands_exists = (RAW_DATA / "candidates.jsonl").exists()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div class="metric-card">📋 Job Description<br><span style="color:{"#10b981" if jd_exists else "#ef4444"}">'
                f'{"✓ Found" if jd_exists else "✗ Missing"}</span></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f'<div class="metric-card">👥 Candidates File<br><span style="color:{"#10b981" if cands_exists else "#ef4444"}">'
                f'{"✓ Found" if cands_exists else "✗ Missing"}</span></div>',
                unsafe_allow_html=True,
            )

        skip_parse = st.checkbox(
            "⚡ Skip parsing (use cached parsed data)",
            value=has_results() and PROCESSED_DATA.exists(),
            help="Faster re-runs when only scoring needs to change",
        )

        if st.button("🚀 Start Analysis", use_container_width=True):
            # Save uploaded files if provided
            if jd_file:
                (RAW_DATA / "job_description.txt").write_bytes(jd_file.read())
                st.success("Job description saved.")
            if cands_file:
                (RAW_DATA / "candidates.jsonl").write_bytes(cands_file.read())
                st.success("Candidates file saved.")

            # Run pipeline
            _run_pipeline_ui(skip_parse=skip_parse)

    with col2:
        st.markdown('<div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)
        pipeline_steps = [
            ("📄", "Parse Job Description", "Extracts role requirements, hidden expectations"),
            ("👤", "Parse Resumes", "Structures candidate profiles"),
            ("🎯", "Score (7 Dimensions)", "Project · Domain · Skills · Learning · Soft · Growth · Semantic"),
            ("🤖", "Run Agents", "Project · Skill · Growth · Recruiter Intelligence"),
            ("🏆", "Rank & Export", "CSV · JSON · Dashboard Cache"),
        ]
        for icon, step, desc in pipeline_steps:
            st.markdown(
                f'<div class="metric-card" style="padding:0.75rem 1rem;">'
                f'<span style="font-size:1.2rem">{icon}</span> '
                f'<span style="font-weight:600; color:#e2e8f0">{step}</span><br>'
                f'<span style="font-size:0.78rem; color:#64748b; padding-left:1.8rem">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if has_results():
            st.success("✅ Analysis complete! Navigate to Rankings to explore results.")


def _run_pipeline_ui(skip_parse: bool = False) -> None:
    """Run the full pipeline with a Streamlit progress UI."""
    import subprocess
    import sys

    with st.spinner("Running AI Analysis Pipeline..."):
        progress = st.progress(0)
        status = st.empty()

        stages = [
            "Parsing Job Description",
            "Parsing Resumes",
            "Scoring Candidates",
            "Running Intelligence Agents",
            "Saving Outputs",
        ]

        for i, stage in enumerate(stages):
            status.markdown(
                f'<div style="color:#94a3b8; font-size:0.9rem;">⏳ {stage}...</div>',
                unsafe_allow_html=True,
            )
            progress.progress((i + 1) / (len(stages) + 1))
            time.sleep(0.3)

        try:
            cmd = [sys.executable, str(ROOT / "main.py")]
            if skip_parse:
                cmd.append("--skip-parse")

            result = subprocess.run(
                cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )

            progress.progress(1.0)

            if result.returncode == 0:
                status.empty()
                st.success("🎉 Analysis complete! Navigate to Rankings to see results.")
                st.cache_data.clear()
            else:
                st.error(f"Pipeline failed:\n```\n{result.stderr[-1000:]}\n```")
        except subprocess.TimeoutExpired:
            st.error("Pipeline timed out after 10 minutes.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")


def page_rankings() -> None:
    data = load_dashboard_cache()
    full = load_final_results()

    if not data or not full:
        st.warning("No results found. Run the pipeline from the Home page first.")
        return

    jinfo = data.get("job_profile", {})
    table = data.get("ranking_table", [])

    # Header
    st.markdown('<div class="page-title">Candidate Rankings</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#64748b; margin-bottom:1.5rem;">Role: '
        f'<span style="color:#93c5fd; font-weight:600">{jinfo.get("role_title", "—")}</span>'
        f' at <span style="color:#93c5fd">{jinfo.get("company_name", "—")}</span></div>',
        unsafe_allow_html=True,
    )

    # KPI row
    total = len(table)
    strong_hire = sum(1 for r in table if r.get("hiring_recommendation") == "strong_hire")
    hire = sum(1 for r in table if r.get("hiring_recommendation") == "hire")
    borderline = sum(1 for r in table if r.get("hiring_recommendation") == "borderline")
    avg_score = sum(r.get("overall_score", 0) for r in table) / max(total, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Candidates", total)
    c2.metric("⭐ Strong Hire", strong_hire)
    c3.metric("✅ Hire", hire)
    c4.metric("🟡 Borderline", borderline)
    c5.metric("Average Score", f"{avg_score:.1f}")

    st.markdown("---")

    # Search + filter
    col_s, col_f = st.columns([3, 1])
    with col_s:
        search = st.text_input("🔍 Search candidates", placeholder="Name or keyword...", label_visibility="collapsed")
    with col_f:
        filter_rec = st.selectbox(
            "Filter",
            ["All"] + list(REC_LABELS.values()),
            label_visibility="collapsed",
        )

    # Filter table
    filtered = table
    if search:
        filtered = [r for r in filtered if search.lower() in (r.get("name") or "").lower()]
    if filter_rec != "All":
        rec_key = next((k for k, v in REC_LABELS.items() if v == filter_rec), None)
        if rec_key:
            filtered = [r for r in filtered if r.get("hiring_recommendation") == rec_key]

    # Ranking cards
    st.markdown(f'<div style="color:#64748b; font-size:0.85rem; margin-bottom:0.5rem;">{len(filtered)} result(s)</div>', unsafe_allow_html=True)

    for row in filtered:
        rec = row.get("hiring_recommendation", "unknown")
        color = REC_COLORS.get(rec, "#6b7280")
        score = row.get("overall_score", 0)
        potential = row.get("potential_score", 0)
        sb = row.get("score_breakdown", {})
        name = row.get("name", "Unknown")
        one_liner = row.get("one_liner", "")

        with st.expander(
            f"#{row.get('rank', '?')}  {name}   |   Score: {score:.0f}/100   |   {REC_LABELS.get(rec, rec)}",
            expanded=False,
        ):
            ec1, ec2 = st.columns([2, 1])
            with ec1:
                st.markdown(
                    f'<div style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.75rem; font-style:italic;">{one_liner}</div>',
                    unsafe_allow_html=True,
                )
                # Score pills
                pills_html = ""
                for dim, val in sb.items():
                    if dim == "Overall":
                        continue
                    pill_color = score_color(val)
                    pills_html += (
                        f'<span style="display:inline-block; background:{pill_color}22; '
                        f'color:{pill_color}; border:1px solid {pill_color}; '
                        f'padding:3px 10px; border-radius:9999px; font-size:0.78rem; '
                        f'font-weight:500; margin:2px;">{dim}: {val:.0f}</span> '
                    )
                st.markdown(pills_html, unsafe_allow_html=True)

                # Strengths / Weaknesses
                strengths = row.get("strengths", [])
                weaknesses = row.get("weaknesses", [])
                if strengths:
                    st.markdown(
                        "**Strengths:** " + " · ".join(f'<span class="signal-positive">✓ {s}</span>' for s in strengths[:3]),
                        unsafe_allow_html=True,
                    )
                if weaknesses:
                    st.markdown(
                        "**Gaps:** " + " · ".join(f'<span class="signal-negative">✗ {w}</span>' for w in weaknesses[:3]),
                        unsafe_allow_html=True,
                    )

                if st.button(f"View Full Profile →", key=f"view_{row.get('candidate_id', row.get('rank'))}"):
                    st.session_state["selected_candidate_id"] = row.get("candidate_id")
                    st.session_state["nav_page"] = "👤 Candidate Detail"
                    st.rerun()

            with ec2:
                st.markdown(
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:3rem; font-weight:800; color:{color};">{score:.0f}</div>'
                    f'<div style="color:#64748b; font-size:0.85rem;">Overall Score</div>'
                    f'<div style="margin-top:0.5rem;">{rec_badge(rec)}</div>'
                    f'<div style="margin-top:0.5rem; color:#94a3b8; font-size:0.82rem;">Potential: {potential:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def page_candidate_detail() -> None:
    full = load_final_results()
    if not full:
        st.warning("No results found. Run the pipeline first.")
        return

    ranked = full.get("ranked_candidates", [])
    if not ranked:
        st.warning("No candidates in results.")
        return

    # Candidate selector
    names = [f"#{r.get('rank', i+1)} {r.get('candidate_name','Unknown')}" for i, r in enumerate(ranked)]
    selected_id = st.session_state.get("selected_candidate_id")

    default_idx = 0
    if selected_id:
        for i, r in enumerate(ranked):
            if str(r.get("candidate_id")) == str(selected_id):
                default_idx = i
                break

    chosen = st.selectbox("Select Candidate", names, index=default_idx, label_visibility="collapsed")
    chosen_idx = names.index(chosen)
    report = ranked[chosen_idx]

    rec = report.get("hiring_recommendation", "unknown")
    color = REC_COLORS.get(rec, "#6b7280")
    overall = report.get("overall_score", 0)
    potential = report.get("potential_score", 0)
    sb = report.get("score_breakdown", {})
    name = report.get("candidate_name", "Unknown")

    # ---- Hero card ----
    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom:1.5rem;">
          <div style="display:flex; justify-content:space-between; align-items:start; flex-wrap:wrap; gap:1rem;">
            <div>
              <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">{name}</div>
              <div style="color:#64748b; margin:0.2rem 0;">{report.get('role_title','')} @ {report.get('company_name','')}</div>
              <div style="margin-top:0.5rem;">{rec_badge(rec, report.get('hiring_recommendation_label',''))}</div>
              <div style="margin-top:0.75rem; color:#94a3b8; font-style:italic; max-width:600px; font-size:0.92rem;">
                {report.get('one_liner','')}
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:3.5rem; font-weight:900; color:{color}; line-height:1;">{overall:.0f}</div>
              <div style="color:#64748b; font-size:0.85rem;">/ 100 Overall</div>
              <div style="color:#8b5cf6; font-size:1.2rem; font-weight:700; margin-top:0.25rem;">{potential:.0f} <span style="font-size:0.78rem; font-weight:400;">Potential</span></div>
              <div style="color:#94a3b8; font-size:0.82rem;">Confidence: {report.get('confidence',0)*100:.0f}%</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Scores", "💡 Intelligence", "🎯 Interview Plan", "📝 Full Report", "⚙️ Debug"
    ])

    with tab1:
        col_radar, col_bar = st.columns([1, 1])
        radar_scores = {k: v for k, v in sb.items() if k != "Overall"}
        with col_radar:
            st.plotly_chart(
                radar_chart(radar_scores, name),
                use_container_width=True,
            )
        with col_bar:
            st.plotly_chart(bar_chart_scores(report), use_container_width=True)

        # Dimension detail
        st.markdown('<div class="section-header">Dimension Breakdown</div>', unsafe_allow_html=True)
        dim_cols = st.columns(4)
        dim_items = [(k, v) for k, v in radar_scores.items()]
        for i, (dim, val) in enumerate(dim_items):
            with dim_cols[i % 4]:
                c = score_color(val)
                st.markdown(
                    f'<div class="metric-card" style="padding:0.75rem; text-align:center;">'
                    f'<div style="font-size:1.8rem; font-weight:800; color:{c};">{val:.0f}</div>'
                    f'<div style="color:#64748b; font-size:0.78rem;">{dim}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tab2:
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown('<div class="section-header">✅ Strengths</div>', unsafe_allow_html=True)
            for s in report.get("strengths", []):
                st.markdown(f'<div class="pill green">✓ {s}</div>', unsafe_allow_html=True)

        with col_w:
            st.markdown('<div class="section-header">⚠️ Gaps & Concerns</div>', unsafe_allow_html=True)
            for w in report.get("weaknesses", []):
                st.markdown(f'<div class="pill red">✗ {w}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🤖 AI Intelligence Narratives</div>', unsafe_allow_html=True)
        for label, key in [("Projects", "project_narrative"), ("Skills", "skill_narrative"), ("Growth", "growth_narrative")]:
            val = report.get(key, "")
            if val:
                st.markdown(f"**{label}:** {val}")

        st.markdown('<div class="section-header">🌱 Growth Opportunity</div>', unsafe_allow_html=True)
        go_text = report.get("growth_opportunity", "")
        st.markdown(go_text or "_No growth data available._")

        if report.get("risk_factors"):
            st.markdown('<div class="section-header">⚡ Risk Factors</div>', unsafe_allow_html=True)
            for rf in report.get("risk_factors", []):
                st.markdown(f'<div class="pill amber">⚡ {rf}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">🎯 Interview Focus Areas</div>', unsafe_allow_html=True)
        for area in report.get("focus_areas", []):
            st.markdown(f"• {area}")

        st.markdown('<div class="section-header">💬 Targeted Interview Questions</div>', unsafe_allow_html=True)
        for i, q in enumerate(report.get("interview_questions", []), start=1):
            if isinstance(q, dict):
                with st.expander(f"Q{i}: {q.get('question','')[:80]}..."):
                    st.markdown(f"**Question:** {q.get('question','')}")
                    st.markdown(f"**Area:** `{q.get('area','')}` | **Rationale:** {q.get('rationale','')}")
            else:
                st.markdown(f"**Q{i}:** {q}")

    with tab4:
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card" style="font-size:1rem; line-height:1.7; color:#e2e8f0;">'
            f'{report.get("executive_summary","")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-header">Full Reasoning</div>', unsafe_allow_html=True)
        st.markdown(report.get("overall_reasoning", "_No reasoning available._"))

        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        for k, v in report.get("score_breakdown", {}).items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(int(v))
            with col2:
                st.markdown(f"**{k}:** {v:.0f}")

    with tab5:
        st.markdown('<div class="section-header">Raw Scoring Data</div>', unsafe_allow_html=True)
        st.json(report, expanded=False)


def page_jd() -> None:
    full = load_final_results()
    if not full or not full.get("job_profile"):
        st.warning("No job profile found. Run the pipeline first.")
        return

    jd = full["job_profile"]
    meta = full.get("metadata", {})

    st.markdown('<div class="page-title">Job Profile Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div style="font-size:1.5rem; font-weight:700; color:#e2e8f0;">{jd.get('role_title','Unknown Role')}</div>
              <div style="color:#93c5fd; font-size:1rem; margin-top:0.2rem;">{jd.get('company_name','')}</div>
              <div style="color:#64748b; margin-top:0.5rem;">
                {jd.get('location','')}  ·  {jd.get('work_mode','').replace('_',' ')}  ·  {jd.get('employment_type','').replace('_',' ')}
              </div>
              <div style="margin-top:0.75rem; color:#94a3b8; font-size:0.9rem; line-height:1.6;">{jd.get('role_summary','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        exp = jd.get("experience_requirements", {})
        biz = jd.get("business_context", {})
        info_items = [
            ("Min Experience", f"{exp.get('minimum_years','?')}+ years"),
            ("Seniority", exp.get("seniority_level", "—")),
            ("Industry", biz.get("industry", "—")),
            ("Domain", biz.get("business_domain", "—")),
            ("Company Stage", biz.get("company_stage", "—")),
        ]
        for label, val in info_items:
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:0.4rem 0; '
                f'border-bottom:1px solid #2d3a50; color:#94a3b8; font-size:0.85rem;">'
                f'<span>{label}</span><span style="color:#e2e8f0; font-weight:500;">{val}</span></div>',
                unsafe_allow_html=True,
            )

    # Required skills
    req_skills = jd.get("required_skills", [])
    if req_skills:
        st.markdown('<div class="section-header">Required Skills</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="pill">{s.get("name","")}</span>'
            for s in req_skills
        )
        st.markdown(pills, unsafe_allow_html=True)

    # Preferred skills
    pref_skills = jd.get("preferred_skills", [])
    if pref_skills:
        st.markdown('<div class="section-header">Preferred Skills</div>', unsafe_allow_html=True)
        pills = "".join(
            f'<span class="pill" style="border-color:#8b5cf6; color:#c4b5fd;">{s.get("name","")}</span>'
            for s in pref_skills
        )
        st.markdown(pills, unsafe_allow_html=True)

    # Responsibilities
    resps = jd.get("responsibilities", [])
    if resps:
        st.markdown('<div class="section-header">Responsibilities</div>', unsafe_allow_html=True)
        for r in resps[:8]:
            icon = "🔴" if r.get("is_mandatory") else "🟡"
            st.markdown(f"{icon} {r.get('description','')}")

    # Hidden expectations
    hidden = jd.get("hidden_expectations", [])
    if hidden:
        st.markdown('<div class="section-header">🔍 Hidden Expectations (AI-Inferred)</div>', unsafe_allow_html=True)
        for h in hidden:
            conf = h.get("confidence_score", 0)
            conf_color = score_color(conf * 100)
            with st.expander(f"{h.get('expectation','')} (confidence: {conf*100:.0f}%)"):
                st.markdown(f"**Category:** `{h.get('category','')}`")
                st.markdown(f"**Evidence:** _{h.get('evidence','')}_")
                st.markdown(f"**Reasoning:** {h.get('reasoning','')}")
                st.markdown(f"**Downstream Impact:** {h.get('downstream_impact','')}")

    # Hiring signals
    signals = jd.get("hiring_signals", [])
    if signals:
        st.markdown('<div class="section-header">Hiring Signals</div>', unsafe_allow_html=True)
        for sig in signals:
            stype = sig.get("signal_type", "")
            icon_map = {
                "must_have": "🔴",
                "deal_breaker": "⛔",
                "growth_indicator": "📈",
                "leadership_indicator": "👑",
                "nice_to_have": "💡",
            }
            icon = icon_map.get(stype, "•")
            st.markdown(f"{icon} **[{stype}]** {sig.get('description','')}")


def page_analytics() -> None:
    full = load_final_results()
    if not full:
        st.warning("No results found. Run the pipeline first.")
        return

    ranked = full.get("ranked_candidates", [])
    if not ranked:
        return

    st.markdown('<div class="page-title">Analytics Dashboard</div>', unsafe_allow_html=True)

    # Score distribution
    scores = [r.get("overall_score", 0) for r in ranked]
    recs = [r.get("hiring_recommendation", "unknown") for r in ranked]
    names = [r.get("candidate_name", f"#{i+1}") for i, r in enumerate(ranked)]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            x=scores,
            nbins=10,
            labels={"x": "Overall Score", "y": "Count"},
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a2035",
            xaxis=dict(gridcolor="#2d3a50", tickfont=dict(color="#64748b")),
            yaxis=dict(gridcolor="#2d3a50", tickfont=dict(color="#64748b")),
            font_color="#e2e8f0",
            height=280,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Recommendation Breakdown</div>', unsafe_allow_html=True)
        from collections import Counter
        rec_counts = Counter(recs)
        labels = [REC_LABELS.get(k, k) for k in rec_counts]
        values = list(rec_counts.values())
        colors = [REC_COLORS.get(k, "#6b7280") for k in rec_counts]

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+value",
            textfont=dict(color="#e2e8f0"),
            hole=0.4,
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            font_color="#e2e8f0",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Score comparison table
    st.markdown('<div class="section-header">Score Comparison</div>', unsafe_allow_html=True)
    dims = ["Projects", "Domain Fit", "Skills", "Learning", "Soft Skills", "Growth", "Semantic Fit"]

    fig_heat = go.Figure()
    for dim in dims:
        dim_vals = [r.get("score_breakdown", {}).get(dim, 0) for r in ranked]
        fig_heat.add_trace(go.Bar(
            name=dim,
            x=names,
            y=dim_vals,
            hovertemplate=f"{dim}: %{{y:.0f}}<extra></extra>",
        ))

    fig_heat.update_layout(
        barmode="group",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a2035",
        xaxis=dict(gridcolor="#2d3a50", tickfont=dict(color="#94a3b8"), tickangle=-30),
        yaxis=dict(gridcolor="#2d3a50", tickfont=dict(color="#64748b"), range=[0, 110]),
        legend=dict(
            bgcolor="#1a2035",
            bordercolor="#2d3a50",
            font=dict(color="#94a3b8"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
        height=400,
        margin=dict(l=10, r=10, t=60, b=80),
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Potential vs Score scatter
    st.markdown('<div class="section-header">Potential vs Overall Score</div>', unsafe_allow_html=True)
    potentials = [r.get("potential_score", 0) for r in ranked]
    rec_colors_list = [REC_COLORS.get(r, "#6b7280") for r in recs]
    rec_labels_list = [REC_LABELS.get(r, r) for r in recs]

    fig_scatter = go.Figure()
    for rec_key in set(recs):
        mask = [r == rec_key for r in recs]
        fig_scatter.add_trace(go.Scatter(
            x=[s for s, m in zip(scores, mask) if m],
            y=[p for p, m in zip(potentials, mask) if m],
            text=[n for n, m in zip(names, mask) if m],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            marker=dict(
                size=14,
                color=REC_COLORS.get(rec_key, "#6b7280"),
                line=dict(width=1, color="#2d3a50"),
            ),
            name=REC_LABELS.get(rec_key, rec_key),
            hovertemplate="%{text}<br>Score: %{x:.0f}, Potential: %{y:.0f}<extra></extra>",
        ))

    fig_scatter.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a2035",
        xaxis=dict(title="Overall Score", gridcolor="#2d3a50", tickfont=dict(color="#64748b")),
        yaxis=dict(title="Potential Score", gridcolor="#2d3a50", tickfont=dict(color="#64748b")),
        legend=dict(bgcolor="#1a2035", bordercolor="#2d3a50", font=dict(color="#94a3b8")),
        height=420,
        margin=dict(l=10, r=10, t=20, b=40),
        font_color="#e2e8f0",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App Router
# ---------------------------------------------------------------------------


def main() -> None:
    # Init session state
    if "selected_candidate_id" not in st.session_state:
        st.session_state["selected_candidate_id"] = None

    page = sidebar()

    # Override page if navigated programmatically
    if st.session_state.get("nav_page"):
        page = st.session_state.pop("nav_page")

    if page == "home" or "Home" in page:
        page_home()
    elif "Rankings" in page:
        page_rankings()
    elif "Candidate" in page:
        page_candidate_detail()
    elif "Job Profile" in page or "JD" in page:
        page_jd()
    elif "Analytics" in page:
        page_analytics()


if __name__ == "__main__":
    main()
