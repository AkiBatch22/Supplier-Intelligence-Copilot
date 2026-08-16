"""Streamlit interface for the Supplier Intelligence Copilot."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.analytics import (
    get_supplier_monthly_history,
    get_supplier_names,
    get_supplier_summary,
)
from src.copilot import (
    CopilotConfigurationError,
    CopilotError,
    SupplierNotFoundError,
    answer_question,
    build_diagnosis_table,
)
from src.database import get_database_health
from src.rag import VectorStoreNotBuiltError


st.set_page_config(page_title="Supplier Intelligence Copilot", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner=False)
def load_supplier_names() -> list[str]:
    return get_supplier_names()


@st.cache_data(show_spinner=False)
def load_supplier_snapshot(supplier_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return get_supplier_summary(supplier_name), get_supplier_monthly_history(supplier_name)


def format_percent(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.1f}%"


def format_currency(value: object) -> str:
    return "—" if pd.isna(value) else f"${float(value):,.0f}"


def render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        st.caption("This answer used structured SQL evidence only.")
        return

    for index, source in enumerate(sources, start=1):
        metadata = source["metadata"]
        st.markdown(
            f"**Source {index}: {metadata['document_type'].title()} · "
            f"{metadata['date']} · similarity {source['similarity']:.3f}**"
        )
        st.text(str(source["document"]))


st.title("Supplier Intelligence Copilot")
st.caption(
    "AI-powered supplier analytics combining SQL metrics with qualitative review "
    "and incident intelligence."
)

try:
    health = get_database_health()
except sqlite3.Error as exc:
    st.error(f"The supplier database could not be opened: {exc}")
    st.stop()
if not health["healthy"]:
    missing = ", ".join(health["missing_tables"])
    st.error(
        "The supplier database is not ready. Run `python scripts/build_database.py` "
        f"from the repository root. Missing tables: {missing}"
    )
    st.stop()

try:
    supplier_names = load_supplier_names()
except (OSError, ValueError, sqlite3.Error) as exc:
    st.error(f"The supplier database could not be read: {exc}")
    st.stop()

if not supplier_names:
    st.error("No suppliers were found in the database.")
    st.stop()

with st.sidebar:
    st.header("Analysis scope")
    supplier_name = st.selectbox("Supplier", supplier_names)
    show_route = st.toggle("Show routing details", value=True)
    st.divider()
    st.caption(
        "All suppliers, procurement records, incidents, reviews, and product telemetry "
        "in this project are synthetic."
    )

summary_df, history_df = load_supplier_snapshot(supplier_name)
if summary_df.empty:
    st.warning(f"Supplier '{supplier_name}' was not found.")
    st.stop()

summary = summary_df.iloc[0]
st.subheader(f"{supplier_name} overview")
metric_columns = st.columns(5)
metric_columns[0].metric("Average SLA", format_percent(summary["avg_sla"]))
metric_columns[1].metric(
    "On-time delivery", format_percent(summary["avg_on_time_delivery"])
)
metric_columns[2].metric("Defect rate", format_percent(summary["avg_defect_rate"]))
metric_columns[3].metric("Total spend", format_currency(summary["total_spend"]))
metric_columns[4].metric("Incidents", f"{int(summary['incident_count']):,}")

with st.expander("Recent KPI trend"):
    if history_df.empty:
        st.info("No monthly performance history is available for this supplier.")
    else:
        chart_data = history_df.set_index("month")[[
            "sla_compliance",
            "on_time_delivery_rate",
            "invoice_accuracy",
        ]]
        st.line_chart(chart_data)
        st.caption("Percent metrics for the latest six available months.")

st.subheader("Ask the copilot")
with st.form("supplier_question_form"):
    question = st.text_input(
        "Question",
        placeholder=f"Why is {supplier_name} performing poorly, and what should procurement do?",
    )
    submitted = st.form_submit_button("Ask Copilot", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question before asking the copilot.")
    else:
        try:
            with st.spinner("Analyzing structured and qualitative evidence..."):
                result = answer_question(question, supplier_name)
        except CopilotConfigurationError as exc:
            st.error(str(exc))
        except VectorStoreNotBuiltError as exc:
            st.error(str(exc))
        except SupplierNotFoundError as exc:
            st.warning(str(exc))
        except CopilotError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.warning(str(exc))
        else:
            if show_route:
                st.caption(f"Route used: **{result['route']}** · Model: `{result['model']}`")
            if result["route"] == "HYBRID":
                st.markdown("### Quantitative evidence")
                diagnosis_table = build_diagnosis_table(result["structured_evidence"])
                st.dataframe(diagnosis_table, hide_index=True, use_container_width=True)
                with st.expander("Full structured SQL evidence"):
                    st.code(result["sql_context"], language="text")

                st.markdown("### Qualitative evidence")
                if result["sources"]:
                    with st.expander(
                        f"Retrieved reviews and incidents ({len(result['sources'])})",
                        expanded=True,
                    ):
                        render_sources(result["sources"])
                else:
                    st.info("No supplier review or incident evidence was retrieved.")

                st.markdown("### Copilot synthesis and procurement actions")
                st.markdown(result["answer"])
            else:
                st.markdown(result["answer"])
                with st.expander("Evidence used"):
                    if result["sql_context"]:
                        st.markdown("**Structured SQL evidence**")
                        st.code(result["sql_context"], language="text")
                    if result["sources"]:
                        st.markdown("**Retrieved review and incident evidence**")
                    render_sources(result["sources"])
