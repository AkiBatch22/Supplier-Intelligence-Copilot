# Supplier Intelligence Copilot — Synthetic Data Dictionary

All organizations, events, metrics, and review notes in this package are synthetic and intended only for portfolio/demo use.

## Dataset design
The tables share `supplier_id` as the business key. Quantitative performance and qualitative review/incident text are generated from the same supplier behavior profiles so that RAG answers can be consistent with dashboard metrics.

### suppliers.csv
- `supplier_id`: synthetic supplier key
- `supplier_name`: synthetic company name
- `category`: procurement category
- `region`: operating region
- `criticality`: Low / Medium / High
- `preferred_supplier`: Yes / No
- `onboarding_date`: supplier onboarding date
- `archetype`: generation-only behavior profile
- `primary_issue_driver`: generation-only root-cause theme

### monthly_performance.csv
One row per supplier per month.
- `sla_compliance`: percentage
- `on_time_delivery_rate`: percentage
- `defect_rate`: percentage
- `invoice_accuracy`: percentage
- `avg_resolution_days`: average issue resolution time
- `escalation_count`: monthly escalations
- `order_fulfillment_rate`: percentage

### invoices.csv
- `invoice_id`
- `supplier_id`
- `invoice_date`
- `invoice_amount`
- `approved_amount`
- `payment_delay_days`
- `invoice_error_flag`: 0/1

### incidents.csv
Text-rich source suitable for RAG.
- `incident_id`
- `supplier_id`
- `incident_date`
- `severity`
- `incident_type`
- `description`
- `resolution`

### supplier_reviews.csv
Text-rich source suitable for RAG.
- `review_id`
- `supplier_id`
- `review_date`
- `review_type`
- `performance_summary`
- `key_issues`
- `corrective_actions`
- `reviewer_notes`

## Suggested architecture
Structured questions → SQL/Pandas over suppliers, performance, invoices.
Qualitative questions → RAG over incidents and supplier reviews.
Hybrid questions → combine structured KPI results with retrieved review/incident evidence.

## Important note
`archetype` and `primary_issue_driver` are included to make the synthetic generation transparent. For a realistic product demo, treat them as generator metadata and do not expose them directly to the copilot as ground-truth analytical features.
