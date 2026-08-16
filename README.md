# Supplier Intelligence Copilot

Supplier Intelligence Copilot is a portfolio-scale procurement analytics product built on fully synthetic data. It combines reproducible SQLite analytics with semantic retrieval over supplier reviews and incidents, then uses an OpenRouter-hosted language model to produce grounded, business-oriented answers.

The project demonstrates three related capabilities:

- Structured supplier analytics: spend, SLA compliance, delivery, defects, invoice quality, escalations, incidents, and recent trends.
- AI supplier copilot: deterministic SQL/RAG/HYBRID routing, persistent Chroma retrieval, and evidence-grounded LLM synthesis.
- AI product analytics: synthetic usage telemetry, route-level quality and latency analysis, and a Prompt V1/V2 experiment.

> Every supplier, transaction, operational record, and telemetry event is synthetic. This repository is a demonstration, not a production procurement system.

## Architecture

```text
                  User
                   |
               Streamlit
                   |
                Copilot
              /    |    \
            SQL   RAG   HYBRID
             |     |      |
          SQLite Chroma  SQL + RAG
              \     |     /
                   LLM
                    |
                  Answer
```

Structured numerical facts always come from parameterized SQLite queries. Qualitative evidence comes from supplier review and incident documents stored in a persistent cosine-similarity Chroma collection. The LLM synthesizes supplied evidence; it is not used as a database.

Generation-only fields in `suppliers.csv`—including `archetype` and `primary_issue_driver`—are deliberately excluded from production analytics and copilot evidence.

## Repository structure

```text
app.py                         Streamlit interface
src/database.py                Central SQLite connection and health checks
src/analytics.py               Reusable SQL-backed supplier analytics
src/rag.py                     Document formatting, indexing, and retrieval
src/copilot.py                 Routing, grounding prompts, and LLM orchestration
scripts/build_database.py      Validated CSV-to-SQLite build
data/                          Synthetic source data and dictionary
notebooks/01_*.ipynb           Structured supplier SQL analysis
notebooks/02_*.ipynb           RAG and hybrid prototype
notebooks/03_*.ipynb           Synthetic AI product analytics and experiment
tests/test_core.py             Isolated core behavior tests
```

Generated SQLite and Chroma artifacts are intentionally gitignored and can be rebuilt from tracked source data.

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own OpenRouter key. Never commit `.env`.

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

## Build the data stores

Build the deterministic SQLite database from the five tracked procurement CSVs:

```bash
python scripts/build_database.py
```

Build the persistent vector index. The embedding model is loaded only during indexing or retrieval, and normal application startup reuses the existing collection.

```bash
python -m src.rag
```

Use `python -m src.rag --rebuild` only when a deliberate full re-index is needed. Reviews and incidents are already natural semantic documents, so arbitrary character chunking is not used.

## Run

```bash
streamlit run app.py
```

Example questions:

- `What is Apex Global's average SLA and defect rate?` — SQL
- `What issues have been reported for Apex Global?` — RAG
- `Why is Apex Global performing poorly, and what should procurement do?` — HYBRID
- `Which suppliers have the highest spend?` — SQL ranking

Run the isolated test suite with:

```bash
pytest -q
```

## Notebooks

1. `01_supplier_analytics_sql.ipynb` develops the structured SQL queries and supplier intelligence view.
2. `02_rag_copilot.ipynb` validates document design, semantic retrieval, and SQL + RAG prompting. Its delete-and-recreate collection step is a notebook convenience only; production code uses a separate persistent cosine collection.
3. `03_ai_product_analytics.ipynb` analyzes synthetic copilot telemetry, defines adoption/acceptance/regeneration/feedback/latency KPIs, compares SQL/RAG/HYBRID routes, and evaluates Prompt V1 versus V2 with a two-proportion significance test.

## Key technologies

Python, SQLite/SQL, Pandas, SentenceTransformers (`all-MiniLM-L6-v2`), ChromaDB, OpenRouter via the OpenAI client, Streamlit, Matplotlib, and Pytest.

## Responsible limitations

- Results reflect synthetic patterns and do not validate real supplier risk.
- Semantic retrieval is only as complete as the indexed reviews and incidents.
- Similarity is relevance, not proof of causation; prompts explicitly use cautious language for possible contributing factors.
- The deterministic router is intentionally simple and explainable, not a general-purpose agent.
- LLM output can still be imperfect and should be reviewed before procurement action.
- The product telemetry notebook demonstrates an analysis workflow; the Streamlit demo does not write to or overwrite the synthetic telemetry dataset.
