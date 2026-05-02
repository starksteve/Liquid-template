import streamlit as st
import pandas as pd
import json
import os
import zipfile
from datetime import datetime
from io import StringIO, BytesIO

import streamlit.components.v1 as components

from utils.llm_config import get_llm_client
from utils.scala_code_generator import ScalaCodeGenerator
from utils.sample_transformer import SampleTransformer
from utils.pyspark_generator import PySparkGenerator
from utils.sql_generator import SparkSQLGenerator
from utils.test_generator import TestGenerator
from utils.data_profiler import DataProfiler
from utils.lineage_visualizer import build_lineage_html
from utils.schema_suggester import SchemaSuggester
from utils.quality_rules_generator import QualityRulesGenerator
from utils.readme_generator import ReadmeGenerator
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# MODERN BUSINESS-GRADE CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- Hero Section ---- */
.hero-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(30%, -30%);
}
.hero-section h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
}
.hero-section .subtitle {
    margin: 0.75rem 0 0 0;
    opacity: 0.85;
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Step Cards ---- */
.step-card-modern {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
}
.step-card-modern:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
}
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
}
.step-number {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1rem;
}
.step-number.green { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
.step-number.orange { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
.step-number.pink { background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); }
.step-number.cyan { background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); }
.step-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
}
.step-desc {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}

/* ---- Column Tags ---- */
.column-tag {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 6px 14px;
    border-radius: 8px;
    margin: 4px;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
}
.column-tag.target {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
}
.column-tag.crosswalk {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    box-shadow: 0 2px 4px rgba(6, 182, 212, 0.3);
}

/* ---- Transform Input Card ---- */
.transform-input {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.15s ease;
}
.transform-input:hover {
    border-color: #6366f1;
    background: #f1f5f9;
}
.transform-label {
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.transform-label .icon {
    color: #6366f1;
}

/* ---- Upload Zone ---- */
.upload-zone {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #f8fafc;
    transition: all 0.2s ease;
}
.upload-zone:hover {
    border-color: #6366f1;
    background: #f1f5f9;
}
.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.upload-text {
    color: #64748b;
    font-size: 0.95rem;
}

/* ---- Success Banner ---- */
.success-banner {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
}

/* ---- Metric Cards ---- */
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
}
.metric-label {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div { 
    background: linear-gradient(90deg, #6366f1, #8b5cf6); 
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
}

/* ---- Buttons ---- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    padding: 0.75rem 2rem;
    font-weight: 600;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
}

/* ---- Data Preview ---- */
.data-preview {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
}
.data-preview-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.75rem;
    font-weight: 600;
    color: #334155;
}

/* ---- Crosswalk Section ---- */
.crosswalk-item {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.crosswalk-item:hover {
    border-color: #06b6d4;
}
</style>
"""


def parse_comma_list(text):
    """Split a comma-separated string into a trimmed list, ignoring blanks."""
    if not text:
        return []
    return [c.strip() for c in text.split(",") if c.strip()]


def extract_columns_from_csv(uploaded_file):
    """Extract column headers from uploaded CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        return list(df.columns), df
    except Exception as e:
        return [], None


def records_from_text(text, columns):
    """Parse sample records entered by the user."""
    if not text or not text.strip():
        return []
    text = text.strip()

    # Try JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Fall back to CSV-style rows
    records = []
    for line in text.strip().split("\n"):
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(columns):
            records.append(dict(zip(columns, values)))
    return records


def render_column_tags(columns, css_class=""):
    """Render column names as styled tags."""
    html = ""
    for c in columns:
        html += '<span class="column-tag {}">{}</span>'.format(css_class, c)
    return html


def main():
    st.set_page_config(
        page_title="Liquid Template Generator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero Section ──
    st.markdown('''
        <div class="hero-section">
            <div class="hero-badge">✨ AI-Powered  |  Data Engineering</div>
            <h1>⚡ DataForge AI</h1>
            <p class="subtitle">
                End-to-end data engineering toolkit. Upload your schema, describe transformations in plain English,
                and instantly generate production-ready Scala/Spark, PySpark, and Spark SQL code — complete with
                unit tests, data lineage diagrams, and automated data profiling.
            </p>
        </div>
    ''', unsafe_allow_html=True)

    # Session state
    if "generated_result" not in st.session_state:
        st.session_state["generated_result"] = None
    if "pyspark_result" not in st.session_state:
        st.session_state["pyspark_result"] = None
    if "sql_result" not in st.session_state:
        st.session_state["sql_result"] = None
    if "test_result" not in st.session_state:
        st.session_state["test_result"] = None
    if "input_columns" not in st.session_state:
        st.session_state["input_columns"] = []
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None
    if "crosswalk_tables" not in st.session_state:
        st.session_state["crosswalk_tables"] = []
    if "generation_history" not in st.session_state:
        st.session_state["generation_history"] = []
    if "last_transformation_logic" not in st.session_state:
        st.session_state["last_transformation_logic"] = {}
    if "last_target_columns" not in st.session_state:
        st.session_state["last_target_columns"] = []
    if "last_input_columns" not in st.session_state:
        st.session_state["last_input_columns"] = []
    if "quality_result" not in st.session_state:
        st.session_state["quality_result"] = None
    if "pipeline_readme" not in st.session_state:
        st.session_state["pipeline_readme"] = None
    if "pipeline_stage" not in st.session_state:
        st.session_state["pipeline_stage"] = "idle"   # idle | validating | transforming | done
    if "ai_suggestions" not in st.session_state:
        st.session_state["ai_suggestions"] = {}

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Navigate",
            ["🏗️ Code Generator", "📊 Data Profiler", "📜 History", "ℹ️ About"],
            key="page_nav",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        max_iterations = st.slider(
            "🔄 AI Refinement Rounds",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of self-correction rounds the AI can perform",
        )
        st.caption(
            "The AI generates code, reviews it, and auto-corrects up to **{}** times if needed.".format(max_iterations)
        )

        st.markdown("---")
        show_debug = st.checkbox("🐛 Debug Mode", value=False)
        st.session_state["show_debug"] = show_debug

        st.markdown("---")
        st.markdown("""
        ### 📖 Quick Guide
        1. **Upload CSV** or enter columns manually
        2. **Define target columns** for output
        3. **Add crosswalk tables** for lookups *(optional)*
        4. **Describe transformations** in plain English
        5. **Add sample data** *(optional)*
        6. **Click Generate** 🚀
        """)
        
        st.markdown("---")
        st.markdown("##### 🔗 Crosswalk Tables")
        st.caption("Use crosswalk tables to map codes to values (e.g., 'M' → 'Male')")

    # ── Page routing ──
    if page == "📊 Data Profiler":
        show_data_profiler()
        return
    if page == "📜 History":
        show_history()
        return
    if page == "ℹ️ About":
        show_about()
        return
    # else: fall through to Code Generator
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1 — Input Columns (with CSV Upload)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('''
        <div class="step-card-modern">
            <div class="step-header">
                <div class="step-number">1</div>
                <div>
                    <p class="step-title">Input Columns</p>
                    <p class="step-desc">Upload a CSV file to auto-detect columns, or enter them manually</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    col_upload, col_manual = st.columns([1, 1])

    with col_upload:
        st.markdown("##### 📁 Upload CSV File")
        uploaded_csv = st.file_uploader(
            "Upload your source data CSV",
            type=["csv"],
            key="input_csv_upload",
            help="Upload a CSV file to automatically detect column headers",
        )
        
        if uploaded_csv is not None:
            columns, df = extract_columns_from_csv(uploaded_csv)
            if columns:
                st.session_state["input_columns"] = columns
                st.session_state["uploaded_df"] = df
                st.success("✅ Detected **{}** columns from CSV".format(len(columns)))
                
                with st.expander("👀 Preview uploaded data", expanded=False):
                    st.dataframe(df.head(5), use_container_width=True)

    with col_manual:
        st.markdown("##### ✏️ Or Enter Manually")
        input_cols_text = st.text_area(
            "Input column names (comma-separated):",
            height=100,
            placeholder="customer_id, first_name, last_name, email, gender_code, country_code",
            key="input_cols_text",
            value=", ".join(st.session_state.get("input_columns", [])) if st.session_state.get("input_columns") else "",
        )
        
        if input_cols_text.strip():
            manual_cols = parse_comma_list(input_cols_text)
            if manual_cols:
                st.session_state["input_columns"] = manual_cols

    input_columns = st.session_state.get("input_columns", [])
    
    if input_columns:
        st.markdown(
            "**📊 {} Input Columns:** {}".format(len(input_columns), render_column_tags(input_columns)),
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — Target Columns
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('''
        <div class="step-card-modern">
            <div class="step-header">
                <div class="step-number green">2</div>
                <div>
                    <p class="step-title">Target Columns</p>
                    <p class="step-desc">Define your desired output column names — they can differ from input columns</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    target_cols_text = st.text_area(
        "Target column names (comma-separated):",
        height=80,
        placeholder="customer_code, full_name, email_domain, gender, region, loyalty_score, is_premium",
        key="target_cols_text",
    )
    target_columns = parse_comma_list(target_cols_text)

    if target_columns:
        st.markdown(
            "**🎯 {} Target Columns:** {}".format(len(target_columns), render_column_tags(target_columns, "target")),
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3 — Crosswalk / Lookup Tables (NEW!)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('''
        <div class="step-card-modern">
            <div class="step-header">
                <div class="step-number cyan">3</div>
                <div>
                    <p class="step-title">Crosswalk / Lookup Tables <em style="font-weight:400; color:#64748b;">(Optional)</em></p>
                    <p class="step-desc">Upload reference tables for value lookups (e.g., 'M' → 'Male', 'US' → 'United States')</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    col_cw_upload, col_cw_preview = st.columns([1, 1])

    with col_cw_upload:
        st.markdown("##### 📎 Upload Crosswalk CSV")
        st.caption("CSV should have columns for key and value (e.g., `code, description`)")
        
        crosswalk_file = st.file_uploader(
            "Upload crosswalk table",
            type=["csv"],
            key="crosswalk_upload",
            help="Upload a CSV with mapping data (e.g., gender_code → gender_name)",
        )
        
        crosswalk_name = st.text_input(
            "Name this crosswalk table:",
            placeholder="e.g., gender_lookup, country_codes",
            key="crosswalk_name",
        )
        
        if st.button("➕ Add Crosswalk Table", key="add_crosswalk"):
            if crosswalk_file and crosswalk_name:
                try:
                    cw_df = pd.read_csv(crosswalk_file)
                    st.session_state["crosswalk_tables"].append({
                        "name": crosswalk_name,
                        "data": cw_df,
                        "columns": list(cw_df.columns),
                    })
                    st.success("✅ Added crosswalk: **{}**".format(crosswalk_name))
                except Exception as e:
                    st.error("Error reading crosswalk file: {}".format(e))
            else:
                st.warning("Please upload a file and provide a name")

    with col_cw_preview:
        st.markdown("##### 📋 Active Crosswalk Tables")
        
        crosswalk_tables = st.session_state.get("crosswalk_tables", [])
        
        if crosswalk_tables:
            for idx, cw in enumerate(crosswalk_tables):
                with st.expander("🔗 {} — {} columns".format(cw["name"], len(cw["columns"])), expanded=False):
                    st.markdown("**Columns:** {}".format(render_column_tags(cw["columns"], "crosswalk")), unsafe_allow_html=True)
                    st.dataframe(cw["data"].head(5), use_container_width=True)
                    if st.button("🗑️ Remove", key="remove_cw_{}".format(idx)):
                        st.session_state["crosswalk_tables"].pop(idx)
                        st.rerun()
        else:
            st.info("💡 No crosswalk tables added yet. Upload a CSV to map codes to values.")
        
        # Show example
        with st.expander("📖 Example crosswalk CSV format"):
            st.markdown("""
            ```csv
            code,description
            M,Male
            F,Female
            O,Other
            ```
            
            **Usage in transformation logic:**
            > *"Join with gender_lookup table on gender_code = code, get description as gender"*
            """)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — Transformation Logic  (with AI Schema Suggest)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('''
        <div class="step-card-modern">
            <div class="step-header">
                <div class="step-number orange">4</div>
                <div>
                    <p class="step-title">Transformation Logic</p>
                    <p class="step-desc">Describe in plain English how each target column should be derived — or let AI suggest it</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    transformation_logic = {}

    if not target_columns:
        st.info("👆 Enter target columns in Step 2 to see transformation fields")
    else:
        # ── AI Schema Suggest ────────────────────────────────────────
        with st.expander("🤖 AI Schema Suggest — auto-fill transformation logic", expanded=False):
            st.caption(
                "Upload a sample of your **target output CSV** (optional) and let AI suggest "
                "the transformation logic for each target column based on your source data."
            )
            col_suggest_upload, col_suggest_hint = st.columns([1, 1])
            with col_suggest_upload:
                target_sample_file = st.file_uploader(
                    "Target sample CSV (optional)",
                    type=["csv"],
                    key="target_sample_upload",
                    help="If you have a sample of what the output should look like, upload it here.",
                )
            with col_suggest_hint:
                extra_context = st.text_area(
                    "Any extra hints for the AI:",
                    height=80,
                    placeholder="e.g., member_id is always prefixed with 'MBR-', dates are in MM/DD/YYYY format",
                    key="suggest_extra_context",
                )

            if st.button("✨ Suggest Transformation Logic with AI", key="btn_suggest", type="secondary"):
                if not input_columns:
                    st.warning("Upload source data in Step 1 first.")
                elif st.session_state.get("uploaded_df") is None:
                    st.warning("Upload a source CSV in Step 1 so AI can read the data.")
                else:
                    target_df = None
                    if target_sample_file is not None:
                        try:
                            target_df = pd.read_csv(target_sample_file)
                        except Exception:
                            pass
                    with st.spinner("🤖 AI is analyzing your schema…"):
                        suggester = SchemaSuggester()
                        suggestions = suggester.suggest(
                            source_df=st.session_state["uploaded_df"],
                            target_columns=target_columns,
                            target_df=target_df,
                            extra_context=extra_context or "",
                        )
                    st.session_state["ai_suggestions"] = suggestions
                    st.success("✅ AI suggestions ready — review and edit below.")

            # Show AI notes if available
            if st.session_state.get("ai_suggestions"):
                notes = st.session_state["ai_suggestions"].get("notes", [])
                if notes:
                    for note in notes:
                        st.info(f"💡 {note}")

        # Show crosswalk hint if tables exist
        if crosswalk_tables:
            cw_names = [cw["name"] for cw in crosswalk_tables]
            st.success("🔗 **Available crosswalk tables:** {}. Reference them in your transformation logic!".format(", ".join(cw_names)))

        logic_mode = st.radio(
            "Entry mode:",
            ["✏️ Per-column fields", "📝 Bulk text"],
            horizontal=True,
            key="logic_mode",
        )

        ai_suggestions = st.session_state.get("ai_suggestions", {}).get("suggestions", {})
        ai_confidence = st.session_state.get("ai_suggestions", {}).get("confidence", {})

        if logic_mode == "✏️ Per-column fields":
            for col in target_columns:
                # Pre-fill with AI suggestion if available
                suggested_value = ai_suggestions.get(col, "")
                confidence = ai_confidence.get(col, "")
                conf_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "")

                label = "🔹 **{}** — transformation logic:".format(col)
                if conf_badge:
                    label += "  {} AI confidence: {}".format(conf_badge, confidence)

                st.markdown('<div class="transform-input">', unsafe_allow_html=True)
                transformation_logic[col] = st.text_area(
                    label,
                    height=70,
                    key="logic_{}".format(col),
                    value=suggested_value,
                    placeholder="e.g., Concatenate 'CUST-' with customer_id | Join with gender_lookup on code to get description",
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            placeholder_text = "\n".join(["{} = ".format(tc) for tc in target_columns])
            bulk_text = st.text_area(
                "Transformation rules (one per line: `column = logic`):",
                height=max(200, 50 * len(target_columns)),
                placeholder=placeholder_text,
                key="bulk_logic",
            )
            if bulk_text and bulk_text.strip():
                for line in bulk_text.strip().split("\n"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        col_name = parts[0].strip()
                        logic_text = parts[1].strip()
                        for tc in target_columns:
                            if tc.lower() == col_name.lower():
                                transformation_logic[tc] = logic_text
                                break

        # Summary
        filled = sum(1 for v in transformation_logic.values() if v and v.strip())

        if filled == len(target_columns):
            st.success("✅ All **{}** target columns have transformation logic!".format(len(target_columns)))
        elif filled > 0:
            st.warning("⚠️ Logic provided for **{}** of **{}** columns".format(filled, len(target_columns)))

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5 — Sample Input Records (Optional)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown('''
        <div class="step-card-modern">
            <div class="step-header">
                <div class="step-number pink">5</div>
                <div>
                    <p class="step-title">Sample Input Records <em style="font-weight:400; color:#64748b;">(Optional)</em></p>
                    <p class="step-desc">Provide sample data to see transformed output examples</p>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    col_sample, col_preview = st.columns([3, 2])

    # Auto-fill from uploaded CSV if available
    sample_placeholder = ""
    if st.session_state.get("uploaded_df") is not None:
        df = st.session_state["uploaded_df"]
        sample_placeholder = df.head(3).to_csv(index=False, header=False)

    with col_sample:
        sample_text = st.text_area(
            "Sample input records (CSV rows or JSON):",
            height=120,
            placeholder=sample_placeholder if sample_placeholder else "101,John,Doe,john@example.com,M,US\n102,Jane,Smith,jane@example.com,F,CA",
            key="sample_text",
        )

    sample_records = records_from_text(sample_text, input_columns)

    with col_preview:
        if sample_records:
            st.markdown("**✅ {} sample record(s) parsed:**".format(len(sample_records)))
            st.dataframe(pd.DataFrame(sample_records), use_container_width=True)
        else:
            st.info("💡 Optional — if provided, output will include transformed sample data")

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE STAGE VISUAL
    # ═══════════════════════════════════════════════════════════════════
    stage = st.session_state.get("pipeline_stage", "idle")
    stages = ["📁 Source", "✅ Quality Check", "⚙️ Transform", "📤 Output"]
    stage_map = {"idle": -1, "validating": 1, "transforming": 2, "done": 3}
    active_idx = stage_map.get(stage, -1)

    stage_html = '<div style="display:flex;align-items:center;gap:0;margin:1.5rem 0;">'
    for i, s in enumerate(stages):
        if i < active_idx:
            bg, color, border = "#22c55e", "white", "#22c55e"
        elif i == active_idx:
            bg, color, border = "#6366f1", "white", "#6366f1"
        else:
            bg, color, border = "#f1f5f9", "#94a3b8", "#e2e8f0"
        stage_html += (
            f'<div style="flex:1;text-align:center;padding:0.6rem 0.4rem;background:{bg};'
            f'color:{color};border:1.5px solid {border};font-size:0.85rem;font-weight:600;'
            f'{"border-radius:10px 0 0 10px;" if i==0 else ("border-radius:0 10px 10px 0;" if i==3 else "border-radius:0;")}'
            f'">{s}</div>'
        )
        if i < len(stages) - 1:
            stage_html += (
                f'<div style="width:0;height:0;border-top:20px solid transparent;'
                f'border-bottom:20px solid transparent;border-left:16px solid {bg};z-index:1;"></div>'
            )
    stage_html += "</div>"
    st.markdown(stage_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # GENERATE BUTTON
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, col_spacer, col_metric = st.columns([3, 1, 1])

    with col_btn:
        generate_clicked = st.button(
            "🚀 Generate All Code (Scala + PySpark + SQL + Tests)",
            type="primary",
            use_container_width=True,
        )

    with col_metric:
        st.metric("AI Rounds", max_iterations)

    # ═══════════════════════════════════════════════════════════════════
    # GENERATION FLOW
    # ═══════════════════════════════════════════════════════════════════
    if generate_clicked:
        if not input_columns:
            st.warning("⚠️ Please provide input columns in Step 1")
            return
        if not target_columns:
            st.warning("⚠️ Please provide target columns in Step 2")
            return

        if not any(v and v.strip() for v in transformation_logic.values()):
            st.warning("⚠️ Please provide transformation logic in Step 4")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message):
            pct = current / total if total > 0 else 0
            progress_bar.progress(min(pct, 1.0))
            status_text.text("🔄 {}".format(message))

        # Build crosswalk info for the prompt
        crosswalk_info = None
        if crosswalk_tables:
            crosswalk_info = []
            for cw in crosswalk_tables:
                crosswalk_info.append({
                    "name": cw["name"],
                    "columns": cw["columns"],
                    "sample_data": cw["data"].head(5).to_dict(orient="records"),
                })

        # ── Step A: Scala ────────────────────────────────────────────
        status_text.text("🔄 [1/3] Generating Scala/Spark code…")
        try:
            generator = ScalaCodeGenerator(max_iterations=max_iterations)
            result = generator.generate_scala_code(
                input_columns=input_columns,
                target_columns=target_columns,
                transformation_logic=transformation_logic,
                sample_records=sample_records if sample_records else None,
                progress_callback=None,
                crosswalk_tables=crosswalk_info,
            )
            st.session_state["generated_result"] = result
        except Exception as e:
            st.error("Scala generation failed: {}".format(e))
            if show_debug:
                st.exception(e)
            return

        progress_bar.progress(0.25)

        # ── Step B: PySpark ──────────────────────────────────────────
        status_text.text("🔄 [2/3] Generating PySpark code…")
        try:
            py_gen = PySparkGenerator(max_iterations=max_iterations)
            pyspark_result = py_gen.generate_pyspark_code(
                input_columns=input_columns,
                target_columns=target_columns,
                transformation_logic=transformation_logic,
                sample_records=sample_records if sample_records else None,
                crosswalk_tables=crosswalk_info,
            )
            st.session_state["pyspark_result"] = pyspark_result
        except Exception as e:
            st.session_state["pyspark_result"] = {"full_code": f"# PySpark generation failed: {e}", "per_column_code": {}, "sample_output": []}

        progress_bar.progress(0.50)

        # ── Step C: Spark SQL ────────────────────────────────────────
        status_text.text("🔄 [3/3] Generating Spark SQL…")
        try:
            sql_gen = SparkSQLGenerator()
            sql_result = sql_gen.generate_sql(
                input_columns=input_columns,
                target_columns=target_columns,
                transformation_logic=transformation_logic,
                crosswalk_tables=crosswalk_info,
                sample_records=sample_records if sample_records else None,
            )
            st.session_state["sql_result"] = sql_result
        except Exception as e:
            st.session_state["sql_result"] = {"ddl_sql": f"-- SQL generation failed: {e}", "view_sql": "", "select_sql": "", "column_exprs": {}}

        progress_bar.progress(1.0)
        status_text.text("✅ All code generated successfully!")

        # ── Step E: Data Quality Rules ───────────────────────────────
        st.session_state["pipeline_stage"] = "validating"
        if st.session_state.get("uploaded_df") is not None:
            try:
                dq_gen = QualityRulesGenerator()
                quality_result = dq_gen.generate(
                    source_df=st.session_state["uploaded_df"],
                    input_columns=input_columns,
                    target_columns=target_columns,
                    transformation_logic=transformation_logic,
                )
                st.session_state["quality_result"] = quality_result
            except Exception as e:
                st.session_state["quality_result"] = {
                    "pyspark_dq": f"# DQ generation failed: {e}",
                    "scala_dq": f"// DQ generation failed: {e}",
                    "rules_summary": [],
                }

        # ── Step F: Pipeline README ──────────────────────────────────
        st.session_state["pipeline_stage"] = "transforming"
        try:
            readme_gen = ReadmeGenerator()
            pipeline_name = "DataForge Pipeline — {} → {}".format(
                ", ".join(input_columns[:3]) + ("…" if len(input_columns) > 3 else ""),
                ", ".join(target_columns[:3]) + ("…" if len(target_columns) > 3 else ""),
            )
            readme_md = readme_gen.generate(
                input_columns=input_columns,
                target_columns=target_columns,
                transformation_logic=transformation_logic,
                crosswalk_tables=crosswalk_info,
                quality_rules=(st.session_state.get("quality_result") or {}).get("rules_summary"),
                pipeline_name=pipeline_name,
            )
            st.session_state["pipeline_readme"] = readme_md
        except Exception as e:
            st.session_state["pipeline_readme"] = f"# Pipeline README\n\n> README generation failed: {e}\n"

        st.session_state["pipeline_stage"] = "done"

        # ── Save transformation_logic to session state so results tabs always have it ──
        st.session_state["last_transformation_logic"] = transformation_logic
        st.session_state["last_target_columns"] = target_columns
        st.session_state["last_input_columns"] = input_columns
        st.session_state["last_sample_records"] = sample_records if sample_records else []

        # ── Save to history ──────────────────────────────────────────
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_columns": input_columns,
            "target_columns": target_columns,
            "transformation_logic": transformation_logic,
            "scala_code": st.session_state["generated_result"].get("full_code", ""),
            "pyspark_code": st.session_state["pyspark_result"].get("full_code", ""),
            "iterations_used": st.session_state["generated_result"].get("iterations_used", 0),
            "crosswalk_count": len(crosswalk_info) if crosswalk_info else 0,
        }
        st.session_state["generation_history"].insert(0, history_entry)
        # Keep only last 10
        st.session_state["generation_history"] = st.session_state["generation_history"][:10]

        progress_bar.empty()
        status_text.empty()
        # NOTE: no st.rerun() here — preserves widget state for lineage + history display


    # ═══════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═══════════════════════════════════════════════════════════════════
    result = st.session_state.get("generated_result")
    if result is None:
        return

    pyspark_result = st.session_state.get("pyspark_result") or {}
    sql_result = st.session_state.get("sql_result") or {}
    test_result = st.session_state.get("test_result") or {}

    # Use saved copies so lineage/history survive page rerenders
    saved_input_cols = st.session_state.get("last_input_columns") or input_columns
    saved_target_cols = st.session_state.get("last_target_columns") or target_columns
    saved_logic = st.session_state.get("last_transformation_logic") or transformation_logic

    st.markdown('<div class="success-banner">🎉 All Code Generated Successfully!</div>', unsafe_allow_html=True)

    # ── Metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎯 Target Columns", len(saved_target_cols) if saved_target_cols else 0)
    with c2:
        st.metric("🔄 AI Rounds", result.get("iterations_used", "?"))
    with c3:
        st.metric("📄 Scala Chars", "{:,}".format(len(result.get("full_code", ""))))
    with c4:
        st.metric("🐍 PySpark Chars", "{:,}".format(len(pyspark_result.get("full_code", ""))))
    with c5:
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        scala_code = result.get("full_code", "")
        py_code = pyspark_result.get("full_code", "")
        sql_code = sql_result.get("ddl_sql", "")
        scala_test = test_result.get("scala_tests", "")
        pytest_code = test_result.get("pytest_tests", "")

        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if scala_code:
                zf.writestr(f"scala/Transformation_{ts_now}.scala", scala_code)
            if py_code:
                zf.writestr(f"pyspark/transformation_{ts_now}.py", py_code)
            if sql_code:
                zf.writestr(f"sql/transformation_{ts_now}.sql", sql_code)
            if scala_test:
                zf.writestr(f"tests/TransformationSpec_{ts_now}.scala", scala_test)
            if pytest_code:
                zf.writestr(f"tests/test_transformation_{ts_now}.py", pytest_code)
            # Data quality
            qr = st.session_state.get("quality_result") or {}
            if qr.get("pyspark_dq"):
                zf.writestr(f"quality/data_quality_{ts_now}.py", qr["pyspark_dq"])
            if qr.get("scala_dq"):
                zf.writestr(f"quality/DataQuality_{ts_now}.scala", qr["scala_dq"])
            # Pipeline README
            readme = st.session_state.get("pipeline_readme") or ""
            if readme:
                zf.writestr("PIPELINE_README.md", readme)
            # Manifest
            manifest = {
                "generated_at": datetime.now().isoformat(),
                "input_columns": input_columns,
                "target_columns": target_columns,
                "transformation_logic": transformation_logic,
                "ai_rounds": result.get("iterations_used", 0),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zip_buf.seek(0)

        st.download_button(
            "📦 Export ZIP",
            data=zip_buf,
            file_name="dataforge_export_{}.zip".format(ts_now),
            mime="application/zip",
            use_container_width=True,
        )

    if result.get("error"):
        st.warning("⚠️ Scala generation note: {}".format(result["error"]))

    st.markdown("---")

    # ── Result Tabs ──────────────────────────────────────────────────
    tabs = st.tabs([
        "⚡ Scala/Spark",
        "🐍 PySpark",
        "🗄️ Spark SQL",
        "🧪 Unit Tests",
        "✅ Data Quality",
        "🔗 Data Lineage",
        "📋 Pipeline README",
        "🧩 Per-Column",
        "📊 Sample Output",
    ])

    # ── Tab 0: Scala ──────────────────────────────────────────────────
    with tabs[0]:
        full_code = result.get("full_code", "")
        st.code(full_code, language="scala", line_numbers=True)
        st.download_button(
            "⬇️ Download .scala",
            data=full_code,
            file_name="transformation_{}.scala".format(ts_now),
            mime="text/plain",
            key="dl_scala",
        )

    # ── Tab 1: PySpark ────────────────────────────────────────────────
    with tabs[1]:
        py_full = pyspark_result.get("full_code", "")
        if py_full:
            st.code(py_full, language="python", line_numbers=True)
            st.download_button(
                "⬇️ Download .py",
                data=py_full,
                file_name="transformation_{}.py".format(ts_now),
                mime="text/plain",
                key="dl_py",
            )
            per_col_py = pyspark_result.get("per_column_code", {})
            if per_col_py:
                st.markdown("---")
                st.markdown("#### 🧩 Per-Column Expressions")
                for col_name, snippet in per_col_py.items():
                    logic_desc = transformation_logic.get(col_name, "")
                    with st.expander("🔹 **{}**".format(col_name)):
                        if logic_desc:
                            st.caption("💬 *{}*".format(logic_desc))
                        st.code(snippet, language="python")
        else:
            st.info("PySpark code not yet generated.")

    # ── Tab 2: Spark SQL ──────────────────────────────────────────────
    with tabs[2]:
        if sql_result:
            sql_sub = st.tabs(["🔎 SELECT Query", "👁️ CREATE VIEW", "🏗️ CREATE TABLE (Delta)"])
            with sql_sub[0]:
                st.code(sql_result.get("select_sql", "-- Not available"), language="sql", line_numbers=True)
            with sql_sub[1]:
                st.code(sql_result.get("view_sql", "-- Not available"), language="sql", line_numbers=True)
            with sql_sub[2]:
                st.code(sql_result.get("ddl_sql", "-- Not available"), language="sql", line_numbers=True)
                st.download_button(
                    "⬇️ Download .sql",
                    data=sql_result.get("ddl_sql", ""),
                    file_name="transformation_{}.sql".format(ts_now),
                    mime="text/plain",
                    key="dl_sql",
                )
            if sql_result.get("column_exprs"):
                st.markdown("---")
                st.markdown("#### 🧩 Per-Column SQL Expressions")
                for col_name, expr in sql_result["column_exprs"].items():
                    logic_desc = transformation_logic.get(col_name, "")
                    with st.expander("🔹 **{}**".format(col_name)):
                        if logic_desc:
                            st.caption("💬 *{}*".format(logic_desc))
                        st.code(expr, language="sql")
        else:
            st.info("Spark SQL not yet generated.")

    # ── Tab 3: Unit Tests ─────────────────────────────────────────────
    with tabs[3]:
        st.info("⚠️ Unit test generation is **on-demand** to stay within free API token limits.")
        if st.button("🧪 Generate Unit Tests", key="btn_gen_tests"):
            with st.spinner("Generating unit tests…"):
                try:
                    _test_gen = TestGenerator()
                    _scala_code = st.session_state.get("generated_result", {}).get("full_code", "")
                    _pyspark_code = st.session_state.get("pyspark_result", {}).get("full_code", "")
                    _tr = _test_gen.generate_tests(
                        input_columns=saved_input_cols,
                        target_columns=saved_target_cols,
                        transformation_logic=saved_logic,
                        scala_code=_scala_code if _scala_code else None,
                        pyspark_code=_pyspark_code if _pyspark_code else None,
                    )
                    st.session_state["test_result"] = _tr
                    test_result = _tr
                    st.success("✅ Unit tests generated!")
                except Exception as _e:
                    st.error(f"Test generation failed: {_e}")
                    test_result = {}
        if test_result:
            test_sub = st.tabs(["🔬 ScalaTest", "🐍 pytest (PySpark)"])
            with test_sub[0]:
                scala_tests = test_result.get("scala_tests", "")
                if scala_tests:
                    st.code(scala_tests, language="scala", line_numbers=True)
                    st.download_button(
                        "⬇️ Download ScalaTest",
                        data=scala_tests,
                        file_name="TransformationSpec_{}.scala".format(ts_now),
                        mime="text/plain",
                        key="dl_scala_test",
                    )
                else:
                    st.info("ScalaTest not generated.")
            with test_sub[1]:
                pytest_tests = test_result.get("pytest_tests", "")
                if pytest_tests:
                    st.code(pytest_tests, language="python", line_numbers=True)
                    st.download_button(
                        "⬇️ Download pytest",
                        data=pytest_tests,
                        file_name="test_transformation_{}.py".format(ts_now),
                        mime="text/plain",
                        key="dl_pytest",
                    )
                else:
                    st.info("pytest not generated.")

    # ── Tab 4: Data Quality ───────────────────────────────────────────
    with tabs[4]:
        quality_result = st.session_state.get("quality_result")
        if quality_result:
            rules = quality_result.get("rules_summary", [])
            if rules:
                st.markdown("### 📋 Quality Rules Applied")
                rules_df = pd.DataFrame(rules)
                st.dataframe(rules_df, use_container_width=True, hide_index=True)
                st.markdown("---")
            else:
                st.info("No rules summary available.")

            dq_sub = st.tabs(["🐍 PySpark Validation", "⚡ Scala Validation"])
            with dq_sub[0]:
                st.code(quality_result.get("pyspark_dq", "# Not generated"), language="python", line_numbers=True)
                st.download_button(
                    "⬇️ Download PySpark DQ",
                    data=quality_result.get("pyspark_dq", ""),
                    file_name="data_quality_{}.py".format(ts_now),
                    mime="text/plain",
                    key="dl_dq_py",
                )
            with dq_sub[1]:
                st.code(quality_result.get("scala_dq", "// Not generated"), language="scala", line_numbers=True)
                st.download_button(
                    "⬇️ Download Scala DQ",
                    data=quality_result.get("scala_dq", ""),
                    file_name="DataQuality_{}.scala".format(ts_now),
                    mime="text/plain",
                    key="dl_dq_scala",
                )
        else:
            st.info("💡 Upload source CSV in Step 1 and generate to get data quality rules.")

    # ── Tab 5: Data Lineage ───────────────────────────────────────────
    with tabs[5]:
        if saved_input_cols and saved_target_cols:
            lineage_html = build_lineage_html(
                input_columns=saved_input_cols,
                target_columns=saved_target_cols,
                transformation_logic=saved_logic,
                crosswalk_tables=st.session_state.get("crosswalk_tables"),
            )
            diagram_height = max(400, max(len(saved_input_cols), len(saved_target_cols)) * 55 + 120)
            components.html(lineage_html, height=diagram_height, scrolling=True)
            st.caption(
                "🔵 Blue = input columns  |  🟢 Green = output columns  |  🟡 Yellow = crosswalk tables  |  "
                "Purple arrows = transformations"
            )
        else:
            st.info("Lineage diagram will appear here after generation.")

    # ── Tab 6: Pipeline README ────────────────────────────────────────
    with tabs[6]:
        readme_md = st.session_state.get("pipeline_readme")
        if readme_md:
            st.markdown(readme_md)
            st.markdown("---")
            st.download_button(
                "⬇️ Download README.md",
                data=readme_md,
                file_name="PIPELINE_README_{}.md".format(ts_now),
                mime="text/markdown",
                key="dl_readme",
            )
        else:
            st.info("Pipeline README will appear here after generation.")

    # ── Tab 7: Per-Column (Scala) ─────────────────────────────────────
    with tabs[7]:
        per_col = result.get("per_column_code", {})
        if per_col:
            for col_name, snippet in per_col.items():
                with st.expander("🔹 **{}**".format(col_name), expanded=True):
                    logic_desc = saved_logic.get(col_name, "")
                    if logic_desc:
                        st.caption("💬 *{}*".format(logic_desc))
                    st.code(snippet, language="scala")
        else:
            st.info("No per-column Scala code available.")

    # ── Tab 8: Sample Output ──────────────────────────────────────────
    with tabs[8]:
        sample_output = result.get("sample_output", [])
        if not sample_output:
            sample_output = pyspark_result.get("sample_output", [])
        if sample_output:
            st.markdown("**Transformed sample records:**")
            st.dataframe(pd.DataFrame(sample_output), use_container_width=True)
            saved_samples = st.session_state.get("last_sample_records", [])
            if saved_samples:
                st.markdown("---")
                st.markdown("#### 🔄 Side-by-Side Comparison")
                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("**📥 Input**")
                    st.dataframe(pd.DataFrame(saved_samples), use_container_width=True)
                with col_out:
                    st.markdown("**📤 Output**")
                    st.dataframe(pd.DataFrame(sample_output), use_container_width=True)
        else:
            st.info("💡 Add sample records in Step 5 to see transformed output.")


# ═══════════════════════════════════════════════════════════════════════════
# DATA PROFILER PAGE
# ═══════════════════════════════════════════════════════════════════════════

def show_data_profiler():
    st.markdown("## 📊 Data Profiler")
    st.caption("Upload any CSV to get a deep quality report before building your transformations.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="profiler_upload")
    if uploaded is None:
        st.info("👆 Upload a CSV to start profiling.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Could not read CSV: {}".format(e))
        return

    profiler = DataProfiler()
    with st.spinner("Profiling data…"):
        report = profiler.profile(df)

    summary = report["summary"]
    col_profiles = report["columns"]
    recs = report["recommendations"]

    # ── Quality Score ──────────────────────────────────────────────
    score = summary["quality_score"]
    score_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    score_label = "Excellent" if score >= 80 else "Fair" if score >= 60 else "Poor"

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);padding:2rem;border-radius:16px;
                    color:white;display:flex;align-items:center;gap:2rem;margin-bottom:1.5rem;">
          <div style="text-align:center;">
            <div style="font-size:3.5rem;font-weight:700;color:{score_color};">{score}</div>
            <div style="font-size:1rem;opacity:0.8;">Quality Score</div>
            <div style="font-size:0.85rem;color:{score_color};font-weight:600;">{score_label}</div>
          </div>
          <div style="flex:1;">
            <div style="font-size:1.3rem;font-weight:600;">{uploaded.name}</div>
            <div style="opacity:0.75;margin-top:0.4rem;">
              {summary['row_count']:,} rows &bull; {summary['col_count']} columns &bull;
              {summary['completeness_pct']}% complete &bull;
              {summary['duplicate_rows']} duplicate rows
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary metrics ────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Rows", f"{summary['row_count']:,}")
    m2.metric("🏷️ Columns", summary["col_count"])
    m3.metric("✅ Completeness", f"{summary['completeness_pct']}%")
    m4.metric("🔁 Duplicate Rows", summary["duplicate_rows"])
    m5.metric("⬛ Total Nulls", f"{summary['total_nulls']:,}")

    st.markdown("---")

    # ── Per-column detail ──────────────────────────────────────────
    st.markdown("### 🏷️ Column Analysis")
    col_search = st.text_input("🔍 Filter columns:", placeholder="Type column name…", key="col_search")

    col_items = [
        (name, p) for name, p in col_profiles.items()
        if col_search.lower() in name.lower()
    ]

    for col_name, p in col_items:
        null_pct = p["null_pct"]
        fill_bar = int(p["fill_rate_pct"] / 100 * 20)
        fill_visual = "█" * fill_bar + "░" * (20 - fill_bar)

        cat = p.get("col_category", "")
        cat_emoji = {"numeric": "🔢", "string": "🔤", "date": "📅"}.get(cat, "❓")

        with st.expander(
            f"{cat_emoji} **{col_name}**  —  {p['inferred_type']}  |  "
            f"{p['unique_count']} unique  |  {null_pct}% null",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Fill Rate", f"{p['fill_rate_pct']}%")
            c2.metric("Unique Values", p["unique_count"])
            c3.metric("Null Count", p["null_count"])

            st.code(fill_visual + f"  {p['fill_rate_pct']}% filled", language=None)

            if cat == "numeric":
                nc1, nc2, nc3, nc4 = st.columns(4)
                nc1.metric("Min", p.get("min", "—"))
                nc2.metric("Max", p.get("max", "—"))
                nc3.metric("Mean", p.get("mean", "—"))
                nc4.metric("Std Dev", p.get("std", "—"))
                if p.get("outlier_count", 0) > 0:
                    st.warning(f"⚠️ {p['outlier_count']} statistical outliers detected (IQR method).")

            elif cat == "string":
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Min Length", p.get("min_length", "—"))
                sc2.metric("Max Length", p.get("max_length", "—"))
                sc3.metric("Avg Length", p.get("avg_length", "—"))
                if p.get("top_values"):
                    st.markdown("**Top values:**")
                    top_df = pd.DataFrame(
                        list(p["top_values"].items()), columns=["Value", "Count"]
                    ).head(10)
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
                st.caption(f"Pattern: `{p.get('pattern_sample', '—')}`  |  Cardinality: {p.get('cardinality', '—')}")

            elif cat == "date":
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Min Date", p.get("min_date", "—"))
                dc2.metric("Max Date", p.get("max_date", "—"))
                dc3.metric("Range (days)", p.get("date_range_days", "—"))

            if p.get("sample_values"):
                st.caption("Sample values: " + ", ".join([f"`{v}`" for v in p["sample_values"]]))

    st.markdown("---")

    # ── Recommendations ────────────────────────────────────────────
    st.markdown("### 💡 Recommendations")
    for rec in recs:
        st.markdown(rec)

    # ── Export profile ─────────────────────────────────────────────
    st.markdown("---")
    profile_json = json.dumps(report, indent=2, default=str)
    st.download_button(
        "⬇️ Download Profile Report (JSON)",
        data=profile_json,
        file_name="data_profile_{}.json".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
        mime="application/json",
    )


# ═══════════════════════════════════════════════════════════════════════════
# HISTORY PAGE
# ═══════════════════════════════════════════════════════════════════════════

def show_history():
    st.markdown("## 📜 Generation History")
    st.caption("Last 10 code generation runs this session.")

    history = st.session_state.get("generation_history", [])
    if not history:
        st.info("No generations yet. Go to **🏗️ Code Generator** to get started.")
        return

    for i, entry in enumerate(history):
        with st.expander(
            f"🕐 {entry['timestamp']}  |  "
            f"{len(entry['input_columns'])} → {len(entry['target_columns'])} columns  |  "
            f"{entry.get('iterations_used', '?')} AI rounds",
            expanded=(i == 0),
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown("**📥 Input columns:**\n" + ", ".join(f"`{c}`" for c in entry["input_columns"]))
            c2.markdown("**📤 Target columns:**\n" + ", ".join(f"`{c}`" for c in entry["target_columns"]))
            c3.metric("Crosswalk Tables", entry.get("crosswalk_count", 0))

            st.markdown("**⚙️ Transformation Logic:**")
            for col, logic in entry.get("transformation_logic", {}).items():
                if logic:
                    st.markdown(f"- **{col}**: {logic}")

            code_tabs = st.tabs(["⚡ Scala", "🐍 PySpark"])
            with code_tabs[0]:
                st.code(entry.get("scala_code", "# Not available"), language="scala", line_numbers=True)
            with code_tabs[1]:
                st.code(entry.get("pyspark_code", "# Not available"), language="python", line_numbers=True)

    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state["generation_history"] = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═══════════════════════════════════════════════════════════════════════════

def show_about():
    st.markdown("## ℹ️ About DataForge AI")
    st.markdown("""
    ### 🚀 Project Overview
    **DataForge AI** is an end-to-end AI-powered data engineering platform built with **Python**, **Streamlit**,
    and **Azure OpenAI (GPT-4)**. It automates the most time-consuming parts of building Spark data pipelines.

    ---

    ### 🏗️ Architecture

    | Layer | Technology |
    |-------|-----------|
    | **Frontend** | Streamlit (Python web app) |
    | **AI Engine** | Azure OpenAI GPT-4 via LangChain |
    | **Code Generation** | Scala/Spark, PySpark, Spark SQL |
    | **Test Generation** | ScalaTest, pytest + PySpark |
    | **Data Profiling** | pandas-based statistical analysis |
    | **Visualization** | Custom SVG lineage diagrams |

    ---

    ### ✨ Key Features

    1. **Multi-language Code Generation** — Describe transformations in plain English, get production-ready
       Scala/Spark, PySpark, and Spark SQL code with a single click.

    2. **Agentic Self-Correction** — The AI generates code, tests it for syntax/logic issues, and
       auto-corrects up to N configurable rounds — no human debugging loop needed.

    3. **Crosswalk / Lookup Table Support** — Upload reference tables (e.g., gender codes, country codes)
       and the AI automatically generates broadcast joins with null-safe fallbacks.

    4. **Automated Unit Test Generation** — ScalaTest (AnyFlatSpec) and pytest suites are generated
       alongside the transformation code — covering happy path, nulls, schema validation, and boundary cases.

    5. **Data Lineage Visualization** — Interactive SVG diagram showing the complete data flow from
       input columns through transformation logic to output columns, including crosswalk joins.

    6. **Deep Data Profiling** — Upload any CSV to get a quality score, per-column statistics
       (nulls, cardinality, outliers, patterns), and actionable recommendations.

    7. **Export Package** — One-click ZIP export containing all artifacts:
       `.scala`, `.py`, `.sql`, test files, and a `manifest.json`.

    8. **Generation History** — Session-based history of all runs with full code artifacts.

    ---

    ### 🛠️ Tech Stack
    `Python 3.11` • `Streamlit` • `Azure OpenAI` • `LangChain` • `pandas` • `Apache Spark` (target runtime)

    ---

    ### 📁 Project Structure
    ```
    app.py                       # Main Streamlit application
    utils/
      llm_config.py              # Azure OpenAI client configuration
      scala_code_generator.py    # Scala/Spark code generation (agentic)
      pyspark_generator.py       # PySpark code generation
      sql_generator.py           # Spark SQL generation
      test_generator.py          # ScalaTest + pytest generation
      data_profiler.py           # CSV data quality profiling
      lineage_visualizer.py      # SVG data lineage diagram builder
      template_generator.py      # Liquid template generation (legacy)
      template_renderer.py       # Liquid template renderer
    data/                        # Sample crosswalk CSV files
    transformation/              # Real-world sample transformation files
    ```
    """)


if __name__ == "__main__":
    main()

