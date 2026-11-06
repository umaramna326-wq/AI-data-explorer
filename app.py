import json
import os
import re
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq


# ============================================================
# AI DATA EXPLORER & ANALYTICS ASSISTANT
# Files required: app.py and requirements.txt
# ============================================================

st.set_page_config(
    page_title="AI Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_FILE_SIZE_MB = 50
MAX_ROWS_FOR_AI = 100_000

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg,#f7f9fc 0%,#edf3f8 100%); }
    .block-container { max-width: 1400px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; color:#172b43; }
    .hero {
      padding:1.7rem 2rem; border-radius:22px; color:white; margin-bottom:1.3rem;
      background:linear-gradient(115deg,#14243a 0%,#15566d 55%,#16877e 100%);
      box-shadow:0 12px 35px rgba(20,36,58,.13);
    }
    .hero h1 { color:white; margin:0 0 .35rem 0; font-size:2.15rem; }
    .hero p { color:#dceaf1; margin:0; }
    div[data-testid="stMetric"] {
      background:white; border:1px solid #e5ebf2; padding:1rem;
      border-radius:15px; box-shadow:0 4px 14px rgba(20,36,58,.04);
    }
    .hint { color:#66788d; font-size:.9rem; }
    div.stButton > button[kind="primary"] {
      border-radius:11px; font-weight:700; min-height:2.8rem;
      background:linear-gradient(90deg,#176b78,#16877e); border:0;
    }
    [data-testid="stFileUploader"] { background:white; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Utility functions -----------------------------
def get_groq_key():
    """Read the Groq key from Streamlit Secrets or environment variables."""
    try:
        secret_value = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_value = ""
    return secret_value or os.getenv("GROQ_API_KEY", "")


def make_json_safe(value):
    """Convert NumPy/Pandas values into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def read_csv_safely(uploaded_file):
    """Try common encodings and separators while limiting upload size."""
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_FILE_SIZE_MB} MB.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The uploaded CSV file is empty.")

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(raw), encoding=encoding, low_memory=False)
            if frame.shape[1] == 1 and frame.shape[0] > 0:
                # Retry delimiter inference for semicolon/tab-separated exports.
                frame = pd.read_csv(
                    BytesIO(raw), encoding=encoding, sep=None,
                    engine="python", on_bad_lines="warn"
                )
            if frame.empty and len(frame.columns) == 0:
                raise ValueError("No rows or columns were found.")
            return frame
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("Could not read this CSV. Check its delimiter and formatting. " + errors[-1])


def normalize_column_names(df):
    """Make duplicate column labels unique and readable."""
    cols = []
    seen = {}
    for i, name in enumerate(df.columns):
        label = str(name).strip() or f"unnamed_{i + 1}"
        count = seen.get(label, 0)
        seen[label] = count + 1
        cols.append(label if count == 0 else f"{label}_{count + 1}")
    df.columns = cols
    return df


def infer_datetime_columns(df):
    """Identify likely date/time columns without forcing conversion of every field."""
    date_columns = []
    keywords = ("date", "time", "timestamp", "month", "year")
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        if any(word in str(col).lower() for word in keywords):
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() >= max(2, int(len(df) * 0.5)):
                df[col] = converted
                date_columns.append(col)
    return date_columns


def get_column_groups(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in datetime_cols
    ]
    return numeric_cols, categorical_cols, datetime_cols


def build_analysis(df):
    """Calculate core statistics locally; the LLM is not used for arithmetic."""
    rows, cols = df.shape
    numeric_cols, categorical_cols, datetime_cols = get_column_groups(df)

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing count": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Data type": [str(df[c].dtype) for c in df.columns],
    }).sort_values("Missing %", ascending=False)

    numeric_stats = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            numeric_stats[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "q1": float(series.quantile(.25)),
                "q3": float(series.quantile(.75)),
            }

    categorical_stats = {}
    for col in categorical_cols:
        values = df[col].dropna().astype(str)
        categorical_stats[col] = {
            "unique_count": int(values.nunique()),
            "top_values": values.value_counts().head(8).to_dict(),
        }

    correlations = df[numeric_cols].corr(numeric_only=True) if len(numeric_cols) >= 2 else pd.DataFrame()
    duplicate_rows = int(df.duplicated().sum())

    outlier_report = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) >= 4:
            q1, q3 = series.quantile(.25), series.quantile(.75)
            iqr = q3 - q1
            if iqr > 0:
                count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
                outlier_report[col] = count

    return {
        "rows": rows,
        "columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "missing": missing,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "correlations": correlations,
        "duplicate_rows": duplicate_rows,
        "outlier_report": outlier_report,
    }


def analysis_context(df, analysis):
    """Create a compact, privacy-conscious statistical context for the LLM."""
    compact = {
        "shape": {"rows": analysis["rows"], "columns": analysis["columns"]},
        "column_types": {
            "numeric": analysis["numeric_columns"],
            "categorical": analysis["categorical_columns"],
            "datetime": analysis["datetime_columns"],
        },
        "numeric_statistics": analysis["numeric_stats"],
        "categorical_top_values": analysis["categorical_stats"],
        "missing_values": analysis["missing"].to_dict(orient="records"),
        "duplicate_rows": analysis["duplicate_rows"],
        "potential_iqr_outlier_counts": analysis["outlier_report"],
    }
    if not analysis["correlations"].empty:
        compact["correlation_matrix"] = analysis["correlations"].round(3).to_dict()
    return json.dumps(make_json_safe(compact), ensure_ascii=False)


def ask_groq(question, context, api_key, mode="question"):
    """Ask Groq to explain computed facts without inventing statistics."""
    client = Groq(api_key=api_key, timeout=90.0, max_retries=2)
    system = """
You are a careful data analyst and a patient teacher.
You receive a compact JSON summary calculated from a user's CSV using Pandas.
Answer only from the supplied summary. Do not invent rows, values, trends,
correlations, causes, or calculations. If the summary does not contain enough
information to answer, say what additional calculation or columns are needed.
Distinguish correlation from causation. Explain technical terms in plain English.
When a user asks for an exact value not included in the summary, state that it
cannot be confirmed from the available summary rather than guessing.
Use concise headings and bullets when helpful.
"""
    prompt = f"DATASET SUMMARY JSON:\n{context}\n\nUSER REQUEST:\n{question}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1800 if mode == "question" else 2600,
    )
    return response.choices[0].message.content or "The AI returned an empty response."


def make_report(df, analysis, ai_insights=None):
    """Create a downloadable plain-text report."""
    lines = [
        "AI DATA EXPLORER - ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 55,
        f"Rows: {analysis['rows']}",
        f"Columns: {analysis['columns']}",
        f"Duplicate rows: {analysis['duplicate_rows']}",
        "",
        "COLUMN TYPES",
        f"Numeric: {', '.join(analysis['numeric_columns']) or 'None'}",
        f"Categorical: {', '.join(analysis['categorical_columns']) or 'None'}",
        f"Date/time: {', '.join(analysis['datetime_columns']) or 'None'}",
        "",
        "MISSING VALUES",
        analysis["missing"].to_string(index=False),
        "",
        "NUMERIC STATISTICS",
        pd.DataFrame(analysis["numeric_stats"]).T.to_string() if analysis["numeric_stats"] else "No numeric columns.",
        "",
        "POTENTIAL OUTLIERS (IQR METHOD)",
        json.dumps(analysis["outlier_report"], indent=2),
    ]
    if ai_insights:
        lines.extend(["", "AI-GENERATED INSIGHTS", ai_insights])
    return "\n".join(lines)


def display_value(value):
    if pd.isna(value):
        return "Missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}"
    return str(value)


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 📊 Data Explorer")
    st.caption("Understand your data. Discover the story.")
    st.markdown("---")
    st.markdown("**Your workflow**")
    st.markdown(
        "1. Upload a CSV file.\n"
        "2. Explore its structure and statistics.\n"
        "3. Build interactive charts.\n"
        "4. Ask AI to explain the findings.\n"
        "5. Download your cleaned data and report."
    )
    st.markdown("---")
    st.caption("Calculations are performed locally with Pandas/NumPy. Only compact statistical summaries are sent to Groq when you request AI help.")

# ----------------------------- Header -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>AI Data Explorer</h1>
      <p>Explore, visualize, and understand your CSV data — with an AI analyst by your side.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Upload -----------------------------
uploaded = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help=f"CSV files up to {MAX_FILE_SIZE_MB} MB are supported.",
)

if uploaded is None:
    st.info("👆 Upload a CSV file to begin exploring your data.")
    st.markdown("### What you can explore")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset overview", "Rows & columns")
    c2.metric("Data quality", "Missing & duplicates")
    c3.metric("Visual analytics", "Interactive charts")
    c4.metric("AI assistant", "Plain-English insights")
    st.stop()

try:
    raw_df = read_csv_safely(uploaded)
    raw_df = normalize_column_names(raw_df)
    if raw_df.empty:
        st.error("This CSV contains column headers but no data rows.")
        st.stop()
    if len(raw_df) > MAX_ROWS_FOR_AI:
        st.warning(
            f"This dataset has {len(raw_df):,} rows. Local exploration is supported, "
            "but the AI summary will be limited to compact statistics."
        )
    df = raw_df.copy()
    date_columns = infer_datetime_columns(df)
    analysis = build_analysis(df)
except Exception as exc:
    st.error(f"Unable to process this CSV: {exc}")
    st.stop()

# Persist current dataset name for report labels.
st.session_state["current_dataset_name"] = uploaded.name

# ----------------------------- Dashboard navigation -----------------------------
st.markdown("### Dataset workspace")
tabs = st.tabs([
    "🏠 Overview",
    "🧾 Data preview",
    "📈 Statistics",
    "🕳️ Missing values",
    "📊 Visualizations",
    "🤖 AI insights",
    "💬 Ask your data",
    "🧹 Data quality",
])

# ----------------------------- Overview -----------------------------
with tabs[0]:
    st.caption(f"Currently exploring: **{uploaded.name}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total rows", f"{analysis['rows']:,}")
    m2.metric("Total columns", f"{analysis['columns']:,}")
    m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    m4.metric("Duplicate rows", f"{analysis['duplicate_rows']:,}")

    st.markdown("#### Dataset structure")
    type_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Non-null values": [int(df[c].notna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(type_df, use_container_width=True, hide_index=True)

    st.markdown("#### Column groups")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Numerical columns**")
        st.write(", ".join(analysis["numeric_columns"]) or "None detected")
    with b:
        st.markdown("**Categorical columns**")
        st.write(", ".join(analysis["categorical_columns"]) or "None detected")
    with c:
        st.markdown("**Date/time columns**")
        st.write(", ".join(analysis["datetime_columns"]) or "None detected")

# ----------------------------- Preview -----------------------------
with tabs[1]:
    st.markdown("#### Interactive dataset table")
    st.dataframe(df, use_container_width=True, height=420)
    first_col, last_col = st.columns(2)
    with first_col:
        st.markdown("#### First 10 records")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    with last_col:
        st.markdown("#### Last 10 records")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

# ----------------------------- Statistics -----------------------------
with tabs[2]:
    st.markdown("#### Numerical statistics")
    if analysis["numeric_stats"]:
        stats_df = df[analysis["numeric_columns"]].describe().T.reset_index()
        stats_df = stats_df.rename(columns={"index": "Column"})
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        chosen_stat_col = st.selectbox(
            "Inspect a numerical column",
            analysis["numeric_columns"],
            key="stat_column",
        )
        stat = analysis["numeric_stats"].get(chosen_stat_col, {})
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Mean", display_value(stat.get("mean")))
        x2.metric("Median", display_value(stat.get("median")))
        x3.metric("Minimum", display_value(stat.get("min")))
        x4.metric("Maximum", display_value(stat.get("max")))
    else:
        st.info("No numerical columns were detected.")

    st.markdown("#### Categorical summary")
    if analysis["categorical_stats"]:
        cat_rows = []
        for col, details in analysis["categorical_stats"].items():
            cat_rows.append({
                "Column": col,
                "Unique values": details["unique_count"],
                "Most common values": ", ".join(
                    f"{k} ({v})" for k, v in list(details["top_values"].items())[:5]
                ),
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No categorical columns were detected.")

# ----------------------------- Missing values -----------------------------
with tabs[3]:
    st.markdown("#### Missing-value report")
    st.dataframe(analysis["missing"], use_container_width=True, hide_index=True)
    missing_total = int(df.isna().sum().sum())
    if missing_total == 0:
        st.success("Great! No missing values were found.")
    else:
        missing_chart = analysis["missing"].query("`Missing count` > 0")
        fig = px.bar(
            missing_chart.sort_values("Missing count", ascending=True),
            x="Missing count", y="Column", orientation="h",
            color="Missing %", color_continuous_scale="Tealgrn",
            title="Missing values by column",
        )
        fig.update_layout(height=max(320, 45 * len(missing_chart)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Suggested handling")
        for _, row in missing_chart.iterrows():
            col = row["Column"]
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                suggestion = "Consider median imputation for skewed data, or mean imputation when the distribution is reasonably symmetric. Check why values are missing first."
            else:
                suggestion = "Consider a meaningful 'Unknown' category or the most frequent value only when appropriate. Avoid filling values if missingness itself is informative."
            st.write(f"**{col} ({row['Missing %']}% missing):** {suggestion}")

# ----------------------------- Visualizations -----------------------------
with tabs[4]:
    st.markdown("#### Build a chart")
    numeric_cols = analysis["numeric_columns"]
    categorical_cols = analysis["categorical_columns"]
    datetime_cols = analysis["datetime_columns"]

    chart_type = st.selectbox(
        "Chart type",
        ["Histogram", "Bar chart", "Pie chart", "Box plot", "Scatter plot", "Correlation heatmap", "Line chart"],
    )

    if chart_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="hist_col")
            fig = px.histogram(df, x=col, nbins=30, marginal="box", title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A histogram requires at least one numerical column.")

    elif chart_type == "Bar chart":
        if categorical_cols:
            cat = st.selectbox("Category column", categorical_cols, key="bar_cat")
            count_df = df[cat].fillna("Missing").astype(str).value_counts().head(30).rename_axis(cat).reset_index(name="Count")
            fig = px.bar(count_df, x=cat, y="Count", color="Count", title=f"Top categories in {cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A bar chart requires at least one categorical column.")

    elif chart_type == "Pie chart":
        if categorical_cols:
            cat = st.selectbox("Category column", categorical_cols, key="pie_cat")
            count_df = df[cat].fillna("Missing").astype(str).value_counts().head(12).rename_axis(cat).reset_index(name="Count")
            fig = px.pie(count_df, names=cat, values="Count", hole=.35, title=f"Distribution of {cat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A pie chart requires at least one categorical column.")

    elif chart_type == "Box plot":
        if numeric_cols:
            col = st.selectbox("Numerical column", numeric_cols, key="box_col")
            fig = px.box(df, y=col, points="outliers", title=f"Potential outliers in {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A box plot requires at least one numerical column.")

    elif chart_type == "Scatter plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_options = [c for c in numeric_cols if c != x] or numeric_cols
            y = st.selectbox("Y-axis", y_options, key="scatter_y")
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_color")
            fig = px.scatter(
                df, x=x, y=y,
                color=None if color_by == "None" else color_by,
                hover_data=[c for c in df.columns if c not in [x, y]][:4],
                title=f"{y} vs {x}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A scatter plot requires at least two numerical columns.")

    elif chart_type == "Correlation heatmap":
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Numerical correlation heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A correlation heatmap requires at least two numerical columns.")

    elif chart_type == "Line chart":
        if datetime_cols and numeric_cols:
            date_col = st.selectbox("Date/time column", datetime_cols, key="line_date")
            value_col = st.selectbox("Numerical value", numeric_cols, key="line_value")
            line_df = df[[date_col, value_col]].dropna().sort_values(date_col)
            if len(line_df):
                # Aggregate repeated dates to a daily mean for a readable trend line.
                line_df = line_df.groupby(date_col, as_index=False)[value_col].mean()
                fig = px.line(line_df, x=date_col, y=value_col, markers=True, title=f"{value_col} over time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No usable date/value pairs were found.")
        else:
            st.info("A line chart requires a detected date/time column and a numerical column.")

# ----------------------------- AI Insights -----------------------------
with tabs[5]:
    st.markdown("#### Let AI explain your dataset")
    st.caption("The app calculates the statistics first. Groq receives a compact statistical summary, not the full CSV.")
    if st.button("✨ Generate AI insights", type="primary", key="generate_insights"):
        api_key = get_groq_key()
        if not api_key:
            st.error("Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets.")
        else:
            try:
                context = analysis_context(df, analysis)
                with st.spinner("The AI analyst is reviewing the computed statistics..."):
                    insights = ask_groq(
                        "Explain the most important patterns, data quality concerns, "
                        "relationships, potential outliers, and practical next steps. "
                        "Use plain English and clearly distinguish observations from possible explanations.",
                        context,
                        api_key,
                        mode="insights",
                    )
                st.session_state["ai_insights"] = insights
            except Exception as exc:
                message = str(exc)
                if "401" in message or "authentication" in message.lower():
                    st.error("Groq authentication failed. Check your API key in Streamlit Secrets.")
                elif "429" in message or "rate_limit" in message.lower():
                    st.error("Groq rate limit reached. Wait briefly and try again.")
                else:
                    st.error(f"AI analysis failed: {message}")

    if st.session_state.get("ai_insights"):
        st.markdown(st.session_state["ai_insights"])
    else:
        st.info("Select **Generate AI insights** to receive a plain-English explanation.")

# ----------------------------- Ask your data -----------------------------
with tabs[6]:
    st.markdown("#### Chat with your dataset")
    st.caption("Ask about the available summary, statistics, missing values, correlations, and data quality.")
    example_questions = [
        "Explain this dataset in simple English.",
        "Which columns have the most missing values?",
        "Which numerical columns have the strongest correlation?",
        "What should I investigate first in this dataset?",
    ]
    selected_question = st.selectbox("Try an example question", ["Choose an example..."] + example_questions)
    question = st.text_area(
        "Your question",
        value="" if selected_question == "Choose an example..." else selected_question,
        placeholder="For example: Which category has the most records?",
        height=100,
        key="data_question",
    )
    if st.button("Ask AI", type="primary", key="ask_data"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            api_key = get_groq_key()
            if not api_key:
                st.error("Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets.")
            else:
                try:
                    context = analysis_context(df, analysis)
                    with st.spinner("Finding an evidence-based answer..."):
                        answer = ask_groq(question.strip(), context, api_key)
                    st.session_state.setdefault("chat_history", []).append(
                        {"question": question.strip(), "answer": answer}
                    )
                except Exception as exc:
                    message = str(exc)
                    if "401" in message or "authentication" in message.lower():
                        st.error("Groq authentication failed. Check your API key.")
                    elif "429" in message or "rate_limit" in message.lower():
                        st.error("Rate limit reached. Please wait and try again.")
                    else:
                        st.error(f"Could not answer your question: {message}")

    for chat in reversed(st.session_state.get("chat_history", [])):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

# ----------------------------- Data quality -----------------------------
with tabs[7]:
    st.markdown("#### Data quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Duplicate records", f"{analysis['duplicate_rows']:,}")
    q2.metric("Columns with missing data", f"{int((df.isna().sum() > 0).sum()):,}")
    q3.metric("Potential outlier values", f"{sum(analysis['outlier_report'].values()):,}")

    st.markdown("**Potential outliers (IQR method)**")
    st.caption("The IQR method flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. These are review candidates, not automatically errors.")
    if analysis["outlier_report"]:
        out_df = pd.DataFrame([
            {"Column": col, "Potential outliers": count}
            for col, count in analysis["outlier_report"].items()
        ]).sort_values("Potential outliers", ascending=False)
        st.dataframe(out_df, use_container_width=True, hide_index=True)
    else:
        st.info("No numerical columns had enough data to run the IQR check.")

    st.markdown("**Possible data type issues**")
    type_issues = []
    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()
            if len(non_null):
                numeric_candidate = pd.to_numeric(non_null.astype(str).str.replace(",", ""), errors="coerce")
                if numeric_candidate.notna().mean() >= .85:
                    type_issues.append({
                        "Column": col,
                        "Current type": str(df[col].dtype),
                        "Recommendation": "Many values look numeric. Review formatting, then consider converting to a numeric type.",
                    })
    if type_issues:
        st.dataframe(pd.DataFrame(type_issues), use_container_width=True, hide_index=True)
    else:
        st.success("No obvious numeric-as-text columns were detected by this basic check.")

    st.markdown("**Cleaning options**")
    st.caption("Cleaning is applied to a copy for download. Your original uploaded data remains unchanged in this session.")
    drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
    fill_numeric = st.checkbox("Fill missing numerical values with column medians", value=False)
    fill_categorical = st.checkbox("Fill missing categorical values with the most frequent value", value=False)

    cleaned = df.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    if fill_numeric:
        for col in cleaned.select_dtypes(include=np.number).columns:
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    if fill_categorical:
        for col in cleaned.select_dtypes(include=["object", "category"]).columns:
            if cleaned[col].isna().any() and not cleaned[col].dropna().empty:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode(dropna=True).iloc[0])

    st.write(f"Original shape: **{df.shape[0]:,} × {df.shape[1]:,}**")
    st.write(f"Cleaned shape: **{cleaned.shape[0]:,} × {cleaned.shape[1]:,}**")
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ----------------------------- Download report -----------------------------
st.markdown("---")
st.markdown("### Download your analysis")
report_text = make_report(df, analysis, st.session_state.get("ai_insights"))
st.download_button(
    "⬇️ Download analysis report (.txt)",
    data=report_text.encode("utf-8"),
    file_name="data_analysis_report.txt",
    mime="text/plain",
    use_container_width=True,
)

st.caption(
    "Privacy: this app does not write uploaded data to a database or permanent storage. "
    "AI features send compact computed summaries and your question to Groq. "
    "Do not upload sensitive or confidential datasets unless you are authorized to do so."
)
