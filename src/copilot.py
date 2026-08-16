"""Grounded SQL, RAG, and hybrid orchestration for supplier questions."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import pandas as pd

from src.analytics import (
    get_high_spend_underperformers,
    get_low_sla_suppliers,
    get_supplier_summary,
    get_supplier_trend,
    get_top_spend_suppliers,
)
from src.database import DatabasePath, PROJECT_ROOT
from src.rag import RetrievedDocument, build_context, retrieve_documents


Route: TypeAlias = Literal["SQL", "RAG", "HYBRID"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b:free"

_SQL_TERMS = {
    "spend",
    "sla",
    "service level",
    "defect",
    "delivery rate",
    "on-time delivery",
    "invoice accuracy",
    "invoice error",
    "payment delay",
    "ranking",
    "rank",
    "top supplier",
    "bottom supplier",
    "average",
    "kpi",
    "metric",
    "how many",
}
_RAG_TERMS = {
    "issue",
    "incident",
    "review feedback",
    "review note",
    "corrective action",
    "operational problem",
    "complaint",
    "resolution",
    "reported",
}
_HYBRID_TERMS = {
    "why",
    "diagnose",
    "explain",
    "root cause",
    "contributing factor",
    "performing poorly",
    "poor performance",
    "deteriorat",
    "what should procurement",
    "recommend",
    "next action",
}


class CopilotError(RuntimeError):
    """Base exception for user-facing copilot failures."""


class SupplierNotFoundError(CopilotError):
    """Raised when structured evidence cannot find the requested supplier."""


class CopilotConfigurationError(CopilotError):
    """Raised when required OpenRouter configuration is missing."""


def route_query(question: str) -> Route:
    """Route a question with deterministic, interview-friendly intent rules."""

    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if not normalized:
        raise ValueError("question must not be empty")

    has_sql_intent = any(term in normalized for term in _SQL_TERMS)
    has_rag_intent = any(term in normalized for term in _RAG_TERMS)
    has_hybrid_intent = any(term in normalized for term in _HYBRID_TERMS)

    if has_hybrid_intent or (has_sql_intent and has_rag_intent):
        return "HYBRID"
    if has_rag_intent:
        return "RAG"
    return "SQL"


def _format_dataframe(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No matching structured records were found."
    return dataframe.to_string(index=False, na_rep="Not available")


def get_supplier_structured_evidence(
    supplier_name: str,
    db_path: DatabasePath = None,
) -> dict[str, dict[str, Any]]:
    """Return separately labelled supplier summary and recent-trend evidence."""

    summary = get_supplier_summary(supplier_name, db_path)
    if summary.empty:
        raise SupplierNotFoundError(f"Supplier '{supplier_name}' was not found")

    trend = get_supplier_trend(supplier_name, db_path)
    return {
        "summary": summary.iloc[0].to_dict(),
        "trend": trend.iloc[0].to_dict() if not trend.empty else {},
    }


def _format_value(value: Any, *, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):,.{decimals}f}"


def _format_count(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{int(value):,}"


def _format_percentage_point_change(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):+.2f}"


def _format_supplier_evidence(structured_evidence: dict[str, dict[str, Any]]) -> str:
    summary = structured_evidence["summary"]
    trend = structured_evidence["trend"]

    summary_context = f"""SUPPLIER PROFILE
- Supplier: {summary['supplier_name']}
- Category: {summary['category']}
- Region: {summary['region']}
- Criticality: {summary['criticality']}

OVERALL HISTORICAL METRICS
These are descriptive averages across all available records, not targets or benchmarks.
- Average SLA: {_format_value(summary['avg_sla'])}%
- Average on-time delivery: {_format_value(summary['avg_on_time_delivery'])}%
- Average defect rate: {_format_value(summary['avg_defect_rate'])}%
- Average invoice accuracy: {_format_value(summary['avg_invoice_accuracy'])}%
- Average resolution time: {_format_value(summary['avg_resolution_days'])} days
- Total spend: ${_format_value(summary['total_spend'])}
- Invoice error rate: {_format_value(summary['invoice_error_rate'])}%

OPERATIONAL COUNTS
- Total escalations: {_format_count(summary['total_escalations'])}
- Total incidents: {_format_count(summary['incident_count'])}
- Severe incidents (High or Critical): {_format_count(summary['severe_incidents'])}"""

    if not trend:
        return summary_context + "\n\nRECENT TREND\nNot available."

    trend_context = f"""RECENT 3-MONTH VS PREVIOUS 3-MONTH TREND
All changes below are absolute percentage-point differences, not relative percent changes.

SLA:
- Recent 3M average: {_format_value(trend['recent_3m_sla'])}%
- Previous 3M average: {_format_value(trend['previous_3m_sla'])}%
- Change: {_format_percentage_point_change(trend['sla_change'])} percentage points

ON-TIME DELIVERY:
- Recent 3M average: {_format_value(trend['recent_3m_delivery'])}%
- Previous 3M average: {_format_value(trend['previous_3m_delivery'])}%
- Change: {_format_percentage_point_change(trend['delivery_change'])} percentage points

DEFECT RATE:
- Recent 3M average: {_format_value(trend['recent_3m_defect'])}%
- Previous 3M average: {_format_value(trend['previous_3m_defect'])}%
- Change: {_format_percentage_point_change(trend['defect_change'])} percentage points"""
    return summary_context + "\n\n" + trend_context


def build_diagnosis_table(
    structured_evidence: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a UI-ready diagnosis table from actual structured evidence."""

    summary = structured_evidence.get("summary", {})
    trend = structured_evidence.get("trend", {})

    def trend_period(recent_key: str, previous_key: str) -> str:
        recent = _format_value(trend.get(recent_key))
        previous = _format_value(trend.get(previous_key))
        if "Not available" in (recent, previous):
            return "Not available"
        return f"{recent}% recent 3M vs {previous}% previous 3M"

    def percentage_point_change(key: str) -> str:
        value = trend.get(key)
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):+.2f} pp"

    rows = [
        {
            "Metric": "SLA",
            "Recent / Current": trend_period("recent_3m_sla", "previous_3m_sla"),
            "Change": percentage_point_change("sla_change"),
        },
        {
            "Metric": "On-time delivery",
            "Recent / Current": trend_period(
                "recent_3m_delivery", "previous_3m_delivery"
            ),
            "Change": percentage_point_change("delivery_change"),
        },
        {
            "Metric": "Defect rate",
            "Recent / Current": trend_period(
                "recent_3m_defect", "previous_3m_defect"
            ),
            "Change": percentage_point_change("defect_change"),
        },
        {
            "Metric": "Escalations",
            "Recent / Current": f"{_format_count(summary.get('total_escalations'))} total",
            "Change": "—",
        },
        {
            "Metric": "Incidents",
            "Recent / Current": f"{_format_count(summary.get('incident_count'))} total",
            "Change": "—",
        },
        {
            "Metric": "Severe incidents",
            "Recent / Current": _format_count(summary.get("severe_incidents")),
            "Change": "—",
        },
    ]
    return pd.DataFrame(rows)


def _build_sql_context_from_evidence(
    question: str,
    structured_evidence: dict[str, dict[str, Any]],
    db_path: DatabasePath = None,
) -> str:
    sections = [_format_supplier_evidence(structured_evidence)]
    normalized = question.lower()

    if any(term in normalized for term in ("top", "rank", "highest")):
        if "spend" in normalized:
            ranking = get_top_spend_suppliers(db_path=db_path)
            sections.append("TOP SPEND SUPPLIERS\n" + _format_dataframe(ranking))
        elif "underperform" in normalized or "poor" in normalized:
            sla_filter = 92.0
            ranking = get_high_spend_underperformers(
                sla_threshold=sla_filter,
                db_path=db_path,
            )
            sections.append(
                "HIGH-SPEND SUPPLIERS FILTERED AT HISTORICAL AVERAGE "
                f"SLA < {sla_filter:.1f}%\n"
                "This is an explicit project query filter, not an external benchmark.\n"
                + _format_dataframe(ranking)
            )
        elif "sla" in normalized:
            sla_filter = 92.0
            ranking = get_low_sla_suppliers(
                sla_threshold=sla_filter,
                db_path=db_path,
            )
            sections.append(
                "SUPPLIERS FILTERED AT HISTORICAL AVERAGE "
                f"SLA < {sla_filter:.1f}%\n"
                "This is an explicit project query filter, not an external benchmark.\n"
                + _format_dataframe(ranking)
            )

    return "\n\n".join(sections)


def build_sql_context(
    question: str,
    supplier_name: str,
    db_path: DatabasePath = None,
) -> str:
    """Build quantitative evidence appropriate to the SQL intent."""

    structured_evidence = get_supplier_structured_evidence(supplier_name, db_path)
    return _build_sql_context_from_evidence(question, structured_evidence, db_path)


_COMMON_GROUNDING_RULES = """
Use only the evidence supplied in this prompt.
- Never invent metrics, incidents, supplier issues, or financial values.
- Distinguish quantitative observations from qualitative explanations.
- Do not claim causation unless the evidence explicitly supports it; otherwise use
  language such as "possible contributing factor," "likely contributing factor," or
  "associated with."
- Do not generalize beyond the supplier and retrieved evidence.
- If evidence is insufficient, say so directly.
- Keep the answer concise and useful to procurement professionals.
""".strip()

_HYBRID_GROUNDING_RULES = """
HYBRID grounding requirements:
- Every numeric claim must come directly from the structured SQL evidence.
- Every qualitative issue or operational explanation must come directly from the
  retrieved review or incident evidence.
- Never introduce industry averages, external benchmarks, targets, preferred
  thresholds, market norms, or industry standards unless a value is explicitly
  supplied in the structured SQL evidence. Never invent or infer benchmark values.
- Keep total escalations, total incidents, and severe incidents as three separate
  concepts. Severe incidents are not severe escalations.
- Describe differences between percentage KPIs in percentage points (or pp), not as
  percent change, unless the structured evidence explicitly provides relative change.
- Recommendations and procurement actions must be supported by the supplied SQL or
  retrieved evidence.
""".strip()


def build_messages(
    question: str,
    route: Route,
    *,
    sql_context: str = "",
    rag_context: str = "",
) -> list[dict[str, str]]:
    """Create a grounded prompt for a selected evidence route."""

    route_instructions = {
        "SQL": (
            "Answer the numerical supplier question from structured SQL evidence. "
            "Treat SQL as the source of truth for every number."
        ),
        "RAG": (
            "Answer from retrieved reviews and incidents. Identify whether evidence "
            "comes from reviews, incidents, or both. Do not introduce metrics that are "
            "not in the retrieved text."
        ),
        "HYBRID": (
            "Use SQL for quantitative observations and retrieved reviews/incidents for "
            "possible operational explanations. End with one or two evidence-supported "
            f"procurement actions when appropriate.\n\n{_HYBRID_GROUNDING_RULES}"
        ),
    }[route]

    system_prompt = (
        "You are a Supplier Intelligence Copilot for procurement and vendor-management "
        f"teams.\n\n{route_instructions}\n\n{_COMMON_GROUNDING_RULES}"
    )
    evidence_sections = []
    if sql_context:
        evidence_sections.append("STRUCTURED SQL EVIDENCE:\n" + sql_context)
    if rag_context:
        evidence_sections.append("RETRIEVED REVIEW / INCIDENT EVIDENCE:\n" + rag_context)
    evidence = "\n\n".join(evidence_sections) or "No evidence was available."

    user_prompt = f"QUESTION:\n{question.strip()}\n\n{evidence}\n\nProvide a grounded answer."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@lru_cache(maxsize=1)
def get_openrouter_client() -> Any:
    """Create the OpenRouter client only when an answer actually needs the LLM."""

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise CopilotConfigurationError(
            "Install python-dotenv and openai to use the copilot LLM"
        ) from exc

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise CopilotConfigurationError(
            "OPENROUTER_API_KEY is not configured. Add it to your local .env file."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def call_llm(
    messages: Sequence[dict[str, str]],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Call OpenRouter with an injectable client so tests never require network access."""

    active_client = client if client is not None else get_openrouter_client()
    model_name = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    try:
        response = active_client.chat.completions.create(
            model=model_name,
            messages=list(messages),
            temperature=0.2,
            max_tokens=1500,
            extra_body={
                "reasoning": {
                    "effort": "low",
                    "exclude": True
                }
            }
        )
        
    except Exception as exc:
        raise CopilotError(
            "The OpenRouter request failed. Check the model, API key, and network connection."
        ) from exc

    answer = response.choices[0].message.content

    if not answer:
        finish_reason = getattr(
            response.choices[0],
            "finish_reason",
            None
        )

        raise CopilotError(
            "The model returned an empty answer. "
            f"Finish reason: {finish_reason or 'unknown'}"
        )

    return answer.strip(), model_name


def answer_question(
    question: str,
    supplier_name: str,
    *,
    route: Route | None = None,
    top_k: int = 5,
    db_path: DatabasePath = None,
    vector_store_path: str | Path | None = None,
    llm_client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a supplier question through the selected SQL, RAG, or hybrid workflow."""

    question = question.strip()
    supplier_name = supplier_name.strip()
    if not question:
        raise ValueError("Enter a supplier question before asking the copilot")
    if not supplier_name:
        raise ValueError("Select a supplier before asking the copilot")

    selected_route = route or route_query(question)
    if selected_route not in {"SQL", "RAG", "HYBRID"}:
        raise ValueError("route must be SQL, RAG, or HYBRID")
    selected_route = cast(Route, selected_route)

    sql_context = ""
    structured_evidence: dict[str, dict[str, Any]] = {}
    sources: list[RetrievedDocument] = []
    if selected_route in {"SQL", "HYBRID"}:
        structured_evidence = get_supplier_structured_evidence(supplier_name, db_path)
        sql_context = _build_sql_context_from_evidence(
            question,
            structured_evidence,
            db_path,
        )

    if selected_route in {"RAG", "HYBRID"}:
        retrieval_arguments: dict[str, Any] = {
            "top_k": top_k,
            "supplier_name": supplier_name,
            "max_per_supplier": top_k,
        }
        if vector_store_path is not None:
            retrieval_arguments["vector_store_path"] = vector_store_path
        sources = retrieve_documents(question, **retrieval_arguments)

    rag_context = build_context(sources)
    messages = build_messages(
        question,
        selected_route,
        sql_context=sql_context,
        rag_context=rag_context,
    )
    answer, model_name = call_llm(messages, client=llm_client, model=model)

    return {
        "answer": answer,
        "route": selected_route,
        "supplier_name": supplier_name,
        "sql_context": sql_context,
        "structured_evidence": structured_evidence,
        "sources": sources,
        "model": model_name,
        "retrieval_top_k": top_k if sources else 0,
    }
