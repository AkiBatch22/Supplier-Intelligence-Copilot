from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.copilot as copilot_module
from src.analytics import get_supplier_summary, get_supplier_trend
from src.copilot import (
    answer_question,
    build_diagnosis_table,
    build_messages,
    build_sql_context,
    get_supplier_structured_evidence,
    route_query,
)
from src.database import REQUIRED_TABLES, get_connection, get_database_health
from src.rag import (
    format_incident_document,
    format_review_document,
    retrieve_documents,
)


def create_test_database(tmp_path):
    database_path = tmp_path / "database" / "test_supplier.db"
    suppliers = pd.DataFrame(
        [
            {
                "supplier_id": "SUP001",
                "supplier_name": "Alpha Supply",
                "category": "Hardware",
                "region": "North",
                "criticality": "High",
                "preferred_supplier": "No",
                "onboarding_date": "2024-01-01",
                "archetype": "generation metadata",
                "primary_issue_driver": "generation metadata",
            },
            {
                "supplier_id": "SUP002",
                "supplier_name": "Beta Supply",
                "category": "Logistics",
                "region": "West",
                "criticality": "Medium",
                "preferred_supplier": "Yes",
                "onboarding_date": "2023-01-01",
                "archetype": "generation metadata",
                "primary_issue_driver": "generation metadata",
            },
        ]
    )
    monthly_performance = pd.DataFrame(
        [
            {
                "supplier_id": "SUP001",
                "month": f"2026-{month:02d}-01",
                "sla_compliance": sla,
                "on_time_delivery_rate": delivery,
                "defect_rate": defect,
                "invoice_accuracy": 96.0,
                "avg_resolution_days": 3.0,
                "escalation_count": 1,
                "order_fulfillment_rate": delivery - 1,
            }
            for month, sla, delivery, defect in zip(
                range(1, 7),
                [90, 91, 92, 93, 94, 95],
                [89, 90, 91, 92, 93, 94],
                [5, 4, 3, 3, 2, 1],
            )
        ]
        + [
            {
                "supplier_id": "SUP002",
                "month": "2026-06-01",
                "sla_compliance": 98.0,
                "on_time_delivery_rate": 98.0,
                "defect_rate": 1.0,
                "invoice_accuracy": 99.0,
                "avg_resolution_days": 1.0,
                "escalation_count": 0,
                "order_fulfillment_rate": 98.0,
            }
        ]
    )
    invoices = pd.DataFrame(
        [
            {
                "invoice_id": "INV001",
                "supplier_id": "SUP001",
                "invoice_date": "2026-06-10",
                "invoice_amount": 1000.0,
                "approved_amount": 950.0,
                "payment_delay_days": 5,
                "invoice_error_flag": 1,
            },
            {
                "invoice_id": "INV002",
                "supplier_id": "SUP001",
                "invoice_date": "2026-06-20",
                "invoice_amount": 2000.0,
                "approved_amount": 2000.0,
                "payment_delay_days": 1,
                "invoice_error_flag": 0,
            },
            {
                "invoice_id": "INV003",
                "supplier_id": "SUP002",
                "invoice_date": "2026-06-15",
                "invoice_amount": 500.0,
                "approved_amount": 500.0,
                "payment_delay_days": 0,
                "invoice_error_flag": 0,
            },
        ]
    )
    incidents = pd.DataFrame(
        [
            {
                "incident_id": "INC001",
                "supplier_id": "SUP001",
                "incident_date": "2026-05-20",
                "severity": "High",
                "incident_type": "Delivery Delay",
                "description": "A shipment arrived late.",
                "resolution": "Weekly status updates were introduced.",
            }
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "review_id": "REV001",
                "supplier_id": "SUP001",
                "review_date": "2026-05-31",
                "review_type": "Monthly Performance Review",
                "performance_summary": "Delivery improved during the period.",
                "key_issues": "One late shipment remained open.",
                "corrective_actions": "Continue weekly status updates.",
                "reviewer_notes": "Monitor the next two cycles.",
            }
        ]
    )

    connection = get_connection(database_path)
    try:
        for table_name, dataframe in {
            "suppliers": suppliers,
            "monthly_performance": monthly_performance,
            "invoices": invoices,
            "incidents": incidents,
            "supplier_reviews": reviews,
        }.items():
            dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
    finally:
        connection.close()
    return database_path


def test_database_connection_and_required_tables(tmp_path):
    database_path = create_test_database(tmp_path)
    health = get_database_health(database_path)

    assert health["healthy"] is True
    assert set(health["tables"]) == REQUIRED_TABLES


def test_get_connection_creates_database_parent(tmp_path):
    database_path = tmp_path / "nested" / "supplier.db"
    connection = get_connection(database_path)
    connection.close()

    assert database_path.exists()


def test_supplier_summary_and_nonexistent_supplier(tmp_path):
    database_path = create_test_database(tmp_path)

    summary = get_supplier_summary("Alpha Supply", database_path)
    missing = get_supplier_summary("Missing Supply", database_path)

    assert summary.loc[0, "total_spend"] == 3000.0
    assert summary.loc[0, "invoice_error_rate"] == 50.0
    assert summary.loc[0, "incident_count"] == 1
    assert summary.loc[0, "total_escalations"] == 6
    assert summary.loc[0, "severe_incidents"] == 1
    assert "archetype" not in summary.columns
    assert "primary_issue_driver" not in summary.columns
    assert missing.empty


def test_supplier_trend_compares_latest_and_previous_three_months(tmp_path):
    database_path = create_test_database(tmp_path)

    trend = get_supplier_trend("Alpha Supply", database_path).iloc[0]

    assert trend["recent_3m_sla"] == 94.0
    assert trend["previous_3m_sla"] == 91.0
    assert trend["sla_change"] == 3.0
    assert trend["defect_change"] == -2.0
    assert get_supplier_trend("Missing Supply", database_path).empty


def test_rag_document_formatting_preserves_operational_metadata():
    review = format_review_document(
        {
            "review_id": "REV001",
            "supplier_id": "SUP001",
            "supplier_name": "Alpha Supply",
            "review_date": "2026-05-31",
            "performance_summary": "Performance was stable.",
            "key_issues": "Late delivery.",
            "corrective_actions": "Weekly checkpoints.",
            "reviewer_notes": "Continue monitoring.",
        }
    )
    incident = format_incident_document(
        {
            "incident_id": "INC001",
            "supplier_id": "SUP001",
            "supplier_name": "Alpha Supply",
            "incident_date": "2026-05-20",
            "incident_type": "Delivery Delay",
            "severity": "High",
            "description": "A shipment arrived late.",
            "resolution": "Weekly updates were introduced.",
        }
    )

    assert review["metadata"] == {
        "supplier_id": "SUP001",
        "supplier_name": "Alpha Supply",
        "document_type": "review",
        "date": "2026-05-31",
    }
    assert "Corrective Actions" in review["text"]
    assert incident["metadata"]["document_type"] == "incident"
    assert "Delivery Delay" in incident["text"]


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return np.array([[1.0, 0.0] for _ in texts])


class FakeCollection:
    def count(self):
        return 4

    def query(self, **kwargs):
        assert kwargs["where"] == {"supplier_name": "Alpha Supply"}
        return {
            "ids": [["review_1", "incident_1"]],
            "documents": [["Review evidence", "Incident evidence"]],
            "metadatas": [[
                {
                    "supplier_id": "SUP001",
                    "supplier_name": "Alpha Supply",
                    "document_type": "review",
                    "date": "2026-05-31",
                },
                {
                    "supplier_id": "SUP001",
                    "supplier_name": "Alpha Supply",
                    "document_type": "incident",
                    "date": "2026-05-20",
                },
            ]],
            "distances": [[0.1, 0.2]],
        }


def test_retrieval_applies_supplier_metadata_filter_without_real_model():
    results = retrieve_documents(
        "late delivery",
        top_k=2,
        supplier_name="Alpha Supply",
        collection=FakeCollection(),
        embedding_model=FakeEmbeddingModel(),
    )

    assert [result["metadata"]["document_type"] for result in results] == [
        "review",
        "incident",
    ]
    assert results[0]["similarity"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("question", "expected_route"),
    [
        ("What is Alpha Supply's average SLA?", "SQL"),
        ("What issues have been reported for Alpha Supply?", "RAG"),
        ("Why is Alpha Supply's delivery rate deteriorating?", "HYBRID"),
        ("What should procurement do about Alpha Supply?", "HYBRID"),
    ],
)
def test_query_routing(question, expected_route):
    assert route_query(question) == expected_route


def test_prompt_construction_separates_sql_and_rag_evidence():
    messages = build_messages(
        "Why is performance poor?",
        "HYBRID",
        sql_context="Average SLA: 90%",
        rag_context="SOURCE 1: late delivery review",
    )

    assert "Never invent" in messages[0]["content"]
    assert "Never introduce industry averages" in messages[0]["content"]
    assert "Severe incidents are not severe escalations" in messages[0]["content"]
    assert "percentage points (or pp)" in messages[0]["content"]
    assert "STRUCTURED SQL EVIDENCE" in messages[1]["content"]
    assert "RETRIEVED REVIEW / INCIDENT EVIDENCE" in messages[1]["content"]
    assert "industry average" not in messages[1]["content"].lower()


def test_structured_context_labels_percentage_point_trends_and_counts(tmp_path):
    database_path = create_test_database(tmp_path)

    context = build_sql_context(
        "Why is Alpha Supply performing poorly?",
        "Alpha Supply",
        database_path,
    )
    evidence = get_supplier_structured_evidence("Alpha Supply", database_path)
    diagnosis = build_diagnosis_table(evidence).set_index("Metric")

    assert "Recent 3M average: 94.00%" in context
    assert "Previous 3M average: 91.00%" in context
    assert "Change: +3.00 percentage points" in context
    assert "Total escalations: 6" in context
    assert "Total incidents: 1" in context
    assert "Severe incidents (High or Critical): 1" in context
    assert "industry average" not in context.lower()
    assert diagnosis.loc["SLA", "Change"] == "+3.00 pp"
    assert diagnosis.loc["Escalations", "Recent / Current"] == "6 total"
    assert diagnosis.loc["Incidents", "Recent / Current"] == "1 total"
    assert diagnosis.loc["Severe incidents", "Recent / Current"] == "1"


class FakeCompletions:
    def __init__(self):
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        assert kwargs["model"] == "test/model"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))]
        )


class FakeOpenRouterClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_sql_orchestration_uses_mocked_llm_without_network(tmp_path):
    database_path = create_test_database(tmp_path)

    result = answer_question(
        "What is Alpha Supply's average SLA?",
        "Alpha Supply",
        db_path=database_path,
        llm_client=FakeOpenRouterClient(),
        model="test/model",
    )

    assert result["route"] == "SQL"
    assert result["answer"] == "Grounded answer."
    assert "Alpha Supply" in result["sql_context"]
    assert result["sources"] == []


def test_hybrid_orchestration_uses_separate_grounded_evidence_without_network(
    tmp_path,
    monkeypatch,
):
    database_path = create_test_database(tmp_path)
    retrieved = [
        {
            "id": "incident_INC001",
            "document": "A shipment arrived late. Weekly updates were introduced.",
            "metadata": {
                "supplier_id": "SUP001",
                "supplier_name": "Alpha Supply",
                "document_type": "incident",
                "date": "2026-05-20",
            },
            "distance": 0.1,
            "similarity": 0.9,
        }
    ]
    monkeypatch.setattr(
        copilot_module,
        "retrieve_documents",
        lambda *args, **kwargs: retrieved,
    )
    client = FakeOpenRouterClient()

    result = answer_question(
        "Why is Alpha Supply performing poorly and what should procurement focus on?",
        "Alpha Supply",
        route="HYBRID",
        db_path=database_path,
        llm_client=client,
        model="test/model",
    )

    assert result["route"] == "HYBRID"
    assert result["answer"] == "Grounded answer."
    assert result["sources"] == retrieved
    assert set(result["structured_evidence"]) == {"summary", "trend"}
    assert result["structured_evidence"]["summary"]["total_escalations"] == 6
    assert result["structured_evidence"]["summary"]["incident_count"] == 1
    assert result["structured_evidence"]["summary"]["severe_incidents"] == 1
    system_prompt = client.completions.last_request["messages"][0]["content"]
    assert "Every numeric claim" in system_prompt
    assert "Every qualitative issue" in system_prompt
