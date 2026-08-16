"""Reusable SQL analytics for supplier performance and procurement intelligence."""

from __future__ import annotations

from contextlib import closing

import pandas as pd

from src.database import DatabasePath, get_connection


_INTELLIGENCE_CTES = """
WITH invoice_summary AS (
    SELECT
        supplier_id,
        COUNT(*) AS invoice_count,
        SUM(invoice_amount) AS total_spend,
        SUM(approved_amount) AS total_approved_amount,
        AVG(payment_delay_days) AS avg_payment_delay_days,
        100.0 * SUM(invoice_error_flag) / NULLIF(COUNT(*), 0) AS invoice_error_rate
    FROM invoices
    GROUP BY supplier_id
),
performance_summary AS (
    SELECT
        supplier_id,
        COUNT(*) AS performance_months,
        MAX(month) AS latest_performance_month,
        AVG(sla_compliance) AS avg_sla,
        AVG(on_time_delivery_rate) AS avg_on_time_delivery,
        AVG(defect_rate) AS avg_defect_rate,
        AVG(invoice_accuracy) AS avg_invoice_accuracy,
        AVG(avg_resolution_days) AS avg_resolution_days,
        SUM(escalation_count) AS total_escalations
    FROM monthly_performance
    GROUP BY supplier_id
),
incident_summary AS (
    SELECT
        supplier_id,
        COUNT(*) AS incident_count,
        SUM(CASE WHEN severity IN ('High', 'Critical') THEN 1 ELSE 0 END)
            AS severe_incident_count,
        MAX(incident_date) AS latest_incident_date
    FROM incidents
    GROUP BY supplier_id
)
"""

_INTELLIGENCE_COLUMNS = """
    s.supplier_id,
    s.supplier_name,
    s.category,
    s.region,
    s.criticality,
    s.preferred_supplier,
    COALESCE(i.invoice_count, 0) AS invoice_count,
    ROUND(COALESCE(i.total_spend, 0), 2) AS total_spend,
    ROUND(COALESCE(i.total_approved_amount, 0), 2) AS total_approved_amount,
    ROUND(i.avg_payment_delay_days, 2) AS avg_payment_delay_days,
    ROUND(i.invoice_error_rate, 2) AS invoice_error_rate,
    p.performance_months,
    p.latest_performance_month,
    ROUND(p.avg_sla, 2) AS avg_sla,
    ROUND(p.avg_on_time_delivery, 2) AS avg_on_time_delivery,
    ROUND(p.avg_defect_rate, 2) AS avg_defect_rate,
    ROUND(p.avg_invoice_accuracy, 2) AS avg_invoice_accuracy,
    ROUND(p.avg_resolution_days, 2) AS avg_resolution_days,
    COALESCE(p.total_escalations, 0) AS total_escalations,
    COALESCE(n.incident_count, 0) AS incident_count,
    COALESCE(n.severe_incident_count, 0) AS severe_incidents,
    n.latest_incident_date
"""

_INTELLIGENCE_JOINS = """
FROM suppliers AS s
LEFT JOIN invoice_summary AS i ON i.supplier_id = s.supplier_id
LEFT JOIN performance_summary AS p ON p.supplier_id = s.supplier_id
LEFT JOIN incident_summary AS n ON n.supplier_id = s.supplier_id
"""


def _read_sql(
    query: str,
    params: tuple[object, ...] = (),
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    with closing(get_connection(db_path)) as connection:
        return pd.read_sql_query(query, connection, params=params)


def _positive_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def get_supplier_names(db_path: DatabasePath = None) -> list[str]:
    """Return supplier names in a stable display order."""

    dataframe = _read_sql(
        "SELECT supplier_name FROM suppliers ORDER BY supplier_name",
        db_path=db_path,
    )
    return dataframe["supplier_name"].tolist()


def get_supplier_summary(
    supplier_name: str,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Return one supplier's aggregated operational, invoice, and incident metrics."""

    query = f"""
    {_INTELLIGENCE_CTES}
    SELECT {_INTELLIGENCE_COLUMNS}
    {_INTELLIGENCE_JOINS}
    WHERE s.supplier_name = ?
    """
    return _read_sql(query, (supplier_name,), db_path)


def get_supplier_trend(
    supplier_name: str,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Compare the latest three months with the preceding three for core KPIs."""

    query = """
    WITH ranked_months AS (
        SELECT
            s.supplier_id,
            s.supplier_name,
            p.month,
            p.sla_compliance,
            p.on_time_delivery_rate,
            p.defect_rate,
            ROW_NUMBER() OVER (
                PARTITION BY p.supplier_id
                ORDER BY p.month DESC
            ) AS month_rank
        FROM monthly_performance AS p
        JOIN suppliers AS s ON s.supplier_id = p.supplier_id
        WHERE s.supplier_name = ?
    ),
    period_averages AS (
        SELECT
            supplier_id,
            supplier_name,
            MAX(CASE WHEN month_rank = 1 THEN month END) AS latest_month,
            AVG(CASE WHEN month_rank BETWEEN 1 AND 3 THEN sla_compliance END)
                AS recent_3m_sla,
            AVG(CASE WHEN month_rank BETWEEN 4 AND 6 THEN sla_compliance END)
                AS previous_3m_sla,
            AVG(CASE WHEN month_rank BETWEEN 1 AND 3 THEN on_time_delivery_rate END)
                AS recent_3m_delivery,
            AVG(CASE WHEN month_rank BETWEEN 4 AND 6 THEN on_time_delivery_rate END)
                AS previous_3m_delivery,
            AVG(CASE WHEN month_rank BETWEEN 1 AND 3 THEN defect_rate END)
                AS recent_3m_defect,
            AVG(CASE WHEN month_rank BETWEEN 4 AND 6 THEN defect_rate END)
                AS previous_3m_defect
        FROM ranked_months
        GROUP BY supplier_id, supplier_name
    )
    SELECT
        supplier_id,
        supplier_name,
        latest_month,
        ROUND(recent_3m_sla, 2) AS recent_3m_sla,
        ROUND(previous_3m_sla, 2) AS previous_3m_sla,
        ROUND(recent_3m_sla - previous_3m_sla, 2) AS sla_change,
        ROUND(recent_3m_delivery, 2) AS recent_3m_delivery,
        ROUND(previous_3m_delivery, 2) AS previous_3m_delivery,
        ROUND(recent_3m_delivery - previous_3m_delivery, 2) AS delivery_change,
        ROUND(recent_3m_defect, 2) AS recent_3m_defect,
        ROUND(previous_3m_defect, 2) AS previous_3m_defect,
        ROUND(recent_3m_defect - previous_3m_defect, 2) AS defect_change
    FROM period_averages
    """
    return _read_sql(query, (supplier_name,), db_path)


def get_supplier_monthly_history(
    supplier_name: str,
    months: int = 6,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Return recent monthly KPI observations for display or further analysis."""

    query = """
    SELECT
        p.month,
        p.sla_compliance,
        p.on_time_delivery_rate,
        p.defect_rate,
        p.invoice_accuracy,
        p.escalation_count
    FROM monthly_performance AS p
    JOIN suppliers AS s ON s.supplier_id = p.supplier_id
    WHERE s.supplier_name = ?
    ORDER BY p.month DESC
    LIMIT ?
    """
    dataframe = _read_sql(query, (supplier_name, _positive_limit(months)), db_path)
    return dataframe.sort_values("month", ignore_index=True) if not dataframe.empty else dataframe


def get_top_spend_suppliers(
    limit: int = 10,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Rank suppliers by invoiced spend."""

    query = """
    SELECT
        s.supplier_id,
        s.supplier_name,
        s.category,
        ROUND(SUM(i.invoice_amount), 2) AS total_spend
    FROM suppliers AS s
    JOIN invoices AS i ON i.supplier_id = s.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.category
    ORDER BY total_spend DESC, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, (_positive_limit(limit),), db_path)


def get_low_sla_suppliers(
    sla_threshold: float = 92.0,
    limit: int = 10,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Return suppliers whose average SLA is below a business threshold."""

    query = """
    SELECT
        s.supplier_id,
        s.supplier_name,
        s.category,
        s.criticality,
        ROUND(AVG(p.sla_compliance), 2) AS avg_sla,
        ROUND(AVG(p.on_time_delivery_rate), 2) AS avg_on_time_delivery,
        ROUND(AVG(p.defect_rate), 2) AS avg_defect_rate
    FROM suppliers AS s
    JOIN monthly_performance AS p ON p.supplier_id = s.supplier_id
    GROUP BY s.supplier_id, s.supplier_name, s.category, s.criticality
    HAVING AVG(p.sla_compliance) < ?
    ORDER BY avg_sla, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, (float(sla_threshold), _positive_limit(limit)), db_path)


def get_high_spend_underperformers(
    sla_threshold: float = 92.0,
    limit: int = 10,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Rank below-threshold suppliers by spend using independent aggregate CTEs."""

    query = """
    WITH spend AS (
        SELECT supplier_id, SUM(invoice_amount) AS total_spend
        FROM invoices
        GROUP BY supplier_id
    ),
    performance AS (
        SELECT
            supplier_id,
            AVG(sla_compliance) AS avg_sla,
            AVG(on_time_delivery_rate) AS avg_on_time_delivery,
            AVG(defect_rate) AS avg_defect_rate
        FROM monthly_performance
        GROUP BY supplier_id
    )
    SELECT
        s.supplier_id,
        s.supplier_name,
        s.category,
        s.criticality,
        ROUND(sp.total_spend, 2) AS total_spend,
        ROUND(p.avg_sla, 2) AS avg_sla,
        ROUND(p.avg_on_time_delivery, 2) AS avg_on_time_delivery,
        ROUND(p.avg_defect_rate, 2) AS avg_defect_rate
    FROM suppliers AS s
    JOIN spend AS sp ON sp.supplier_id = s.supplier_id
    JOIN performance AS p ON p.supplier_id = s.supplier_id
    WHERE p.avg_sla < ?
    ORDER BY sp.total_spend DESC, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, (float(sla_threshold), _positive_limit(limit)), db_path)


def get_supplier_invoice_quality(
    supplier_name: str | None = None,
    limit: int = 10,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Return invoice error, payment-delay, and approval variance metrics."""

    where_clause = "WHERE s.supplier_name = ?" if supplier_name else ""
    params: tuple[object, ...] = (supplier_name,) if supplier_name else ()
    query = f"""
    SELECT
        s.supplier_id,
        s.supplier_name,
        COUNT(i.invoice_id) AS invoice_count,
        ROUND(100.0 * SUM(i.invoice_error_flag) / NULLIF(COUNT(i.invoice_id), 0), 2)
            AS invoice_error_rate,
        ROUND(AVG(i.payment_delay_days), 2) AS avg_payment_delay_days,
        ROUND(SUM(i.invoice_amount - i.approved_amount), 2) AS approval_variance
    FROM suppliers AS s
    JOIN invoices AS i ON i.supplier_id = s.supplier_id
    {where_clause}
    GROUP BY s.supplier_id, s.supplier_name
    ORDER BY invoice_error_rate DESC, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, params + (_positive_limit(limit),), db_path)


def get_supplier_incident_summary(
    supplier_name: str | None = None,
    limit: int = 10,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Summarize incident volume and severity from operational incident records."""

    where_clause = "WHERE s.supplier_name = ?" if supplier_name else ""
    params: tuple[object, ...] = (supplier_name,) if supplier_name else ()
    query = f"""
    SELECT
        s.supplier_id,
        s.supplier_name,
        COUNT(i.incident_id) AS incident_count,
        SUM(CASE WHEN i.severity = 'Critical' THEN 1 ELSE 0 END) AS critical_incidents,
        SUM(CASE WHEN i.severity = 'High' THEN 1 ELSE 0 END) AS high_incidents,
        MAX(i.incident_date) AS latest_incident_date
    FROM suppliers AS s
    JOIN incidents AS i ON i.supplier_id = s.supplier_id
    {where_clause}
    GROUP BY s.supplier_id, s.supplier_name
    ORDER BY incident_count DESC, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, params + (_positive_limit(limit),), db_path)


def get_supplier_intelligence_table(
    limit: int = 100,
    db_path: DatabasePath = None,
) -> pd.DataFrame:
    """Return a joined supplier intelligence view without generation-only metadata."""

    query = f"""
    {_INTELLIGENCE_CTES}
    SELECT {_INTELLIGENCE_COLUMNS}
    {_INTELLIGENCE_JOINS}
    ORDER BY p.avg_sla, s.supplier_name
    LIMIT ?
    """
    return _read_sql(query, (_positive_limit(limit),), db_path)
