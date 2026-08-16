"""Persistent semantic retrieval over supplier reviews and incidents."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from src.database import DatabasePath, PROJECT_ROOT, get_connection


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "supplier_intelligence_cosine_v1"

Document: TypeAlias = dict[str, Any]
RetrievedDocument: TypeAlias = dict[str, Any]


class VectorStoreNotBuiltError(RuntimeError):
    """Raised when retrieval is requested before the persistent index is built."""


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not recorded"
    return str(value).strip()


def format_review_document(row: Mapping[str, Any]) -> Document:
    """Format one supplier review as a natural semantic document with metadata."""

    supplier_name = _text(row["supplier_name"])
    date = _text(row["review_date"])
    document_text = f"""Supplier: {supplier_name}
Review Date: {date}

Performance Summary:
{_text(row['performance_summary'])}

Key Issues:
{_text(row['key_issues'])}

Corrective Actions:
{_text(row['corrective_actions'])}

Reviewer Notes:
{_text(row['reviewer_notes'])}"""

    return {
        "id": f"review_{_text(row['review_id'])}",
        "text": document_text,
        "metadata": {
            "supplier_id": _text(row["supplier_id"]),
            "supplier_name": supplier_name,
            "document_type": "review",
            "date": date,
        },
    }


def format_incident_document(row: Mapping[str, Any]) -> Document:
    """Format one supplier incident as a natural semantic document with metadata."""

    supplier_name = _text(row["supplier_name"])
    date = _text(row["incident_date"])
    document_text = f"""Supplier: {supplier_name}
Incident Date: {date}
Incident Type: {_text(row['incident_type'])}
Severity: {_text(row['severity'])}

Incident:
{_text(row['description'])}

Resolution:
{_text(row['resolution'])}"""

    return {
        "id": f"incident_{_text(row['incident_id'])}",
        "text": document_text,
        "metadata": {
            "supplier_id": _text(row["supplier_id"]),
            "supplier_name": supplier_name,
            "document_type": "incident",
            "date": date,
        },
    }


def load_documents(db_path: DatabasePath = None) -> list[Document]:
    """Load review and incident records without exposing generator-only supplier fields."""

    review_query = """
    SELECT
        r.review_id,
        r.supplier_id,
        s.supplier_name,
        r.review_date,
        r.performance_summary,
        r.key_issues,
        r.corrective_actions,
        r.reviewer_notes
    FROM supplier_reviews AS r
    JOIN suppliers AS s ON s.supplier_id = r.supplier_id
    ORDER BY r.review_date, r.review_id
    """
    incident_query = """
    SELECT
        i.incident_id,
        i.supplier_id,
        s.supplier_name,
        i.incident_date,
        i.incident_type,
        i.severity,
        i.description,
        i.resolution
    FROM incidents AS i
    JOIN suppliers AS s ON s.supplier_id = i.supplier_id
    ORDER BY i.incident_date, i.incident_id
    """

    with closing(get_connection(db_path)) as connection:
        reviews = pd.read_sql_query(review_query, connection)
        incidents = pd.read_sql_query(incident_query, connection)

    review_documents = [format_review_document(row) for row in reviews.to_dict("records")]
    incident_documents = [
        format_incident_document(row) for row in incidents.to_dict("records")
    ]
    return review_documents + incident_documents


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> Any:
    """Lazily load and cache the normalized sentence embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for indexing and retrieval"
        ) from exc
    return SentenceTransformer(model_name)


def _get_chroma_client(vector_store_path: str | Path = VECTOR_STORE_PATH) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required for persistent retrieval") from exc

    path = Path(vector_store_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_names(client: Any) -> set[str]:
    return {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    }


def get_collection(
    vector_store_path: str | Path = VECTOR_STORE_PATH,
    collection_name: str = COLLECTION_NAME,
) -> Any:
    """Open an existing production collection without creating it implicitly."""

    client = _get_chroma_client(vector_store_path)
    if collection_name not in _collection_names(client):
        raise VectorStoreNotBuiltError(
            "The supplier vector index has not been built. "
            "Run `python -m src.rag` from the repository root."
        )
    return client.get_collection(collection_name)


def build_vector_index(
    *,
    rebuild: bool = False,
    db_path: DatabasePath = None,
    vector_store_path: str | Path = VECTOR_STORE_PATH,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 128,
    embedding_model: Any | None = None,
) -> dict[str, object]:
    """Build or deliberately rebuild the persistent cosine-similarity collection."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    documents = load_documents(db_path)
    if not documents:
        raise ValueError("No supplier review or incident documents were found")

    # Load the model before changing a persistent collection. A missing model or
    # dependency must not destroy an otherwise usable index during a rebuild.
    model = embedding_model if embedding_model is not None else get_embedding_model()
    client = _get_chroma_client(vector_store_path)
    collection_exists = collection_name in _collection_names(client)
    if rebuild and collection_exists:
        client.delete_collection(collection_name)
        collection_exists = False

    if collection_exists:
        collection = client.get_collection(collection_name)
        metadata = collection.metadata or {}
        if metadata.get("hnsw:space") not in (None, "cosine"):
            raise ValueError(
                "The existing collection does not use cosine distance; rebuild it with --rebuild"
            )
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    expected_ids = {document["id"] for document in documents}

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        texts = [document["text"] for document in batch]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collection.upsert(
            ids=[document["id"] for document in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[document["metadata"] for document in batch],
        )

    existing = collection.get(include=[])
    stale_ids = list(set(existing.get("ids", [])) - expected_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "collection": collection_name,
        "document_count": collection.count(),
        "review_count": sum(
            document["metadata"]["document_type"] == "review"
            for document in documents
        ),
        "incident_count": sum(
            document["metadata"]["document_type"] == "incident"
            for document in documents
        ),
        "rebuilt": rebuild,
        "vector_store_path": str(Path(vector_store_path)),
    }


def get_index_status(
    vector_store_path: str | Path = VECTOR_STORE_PATH,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, object]:
    """Return lightweight persistent-index status without loading the embedding model."""

    client = _get_chroma_client(vector_store_path)
    if collection_name not in _collection_names(client):
        return {"built": False, "collection": collection_name, "document_count": 0}
    collection = client.get_collection(collection_name)
    return {
        "built": True,
        "collection": collection_name,
        "document_count": collection.count(),
    }


def retrieve_documents(
    query: str,
    *,
    top_k: int = 5,
    supplier_name: str | None = None,
    max_per_supplier: int = 2,
    vector_store_path: str | Path = VECTOR_STORE_PATH,
    collection: Any | None = None,
    embedding_model: Any | None = None,
) -> list[RetrievedDocument]:
    """Retrieve semantically similar evidence with optional supplier filtering."""

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0 or max_per_supplier <= 0:
        raise ValueError("top_k and max_per_supplier must be positive")

    active_collection = (
        collection if collection is not None else get_collection(vector_store_path)
    )
    count = active_collection.count()
    if count == 0:
        raise VectorStoreNotBuiltError("The supplier vector index is empty; rebuild it")

    model = embedding_model if embedding_model is not None else get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    candidate_count = min(count, max(top_k * 4, top_k))
    where = {"supplier_name": supplier_name} if supplier_name else None

    query_arguments: dict[str, Any] = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_arguments["where"] = where
    results = active_collection.query(**query_arguments)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    retrieved: list[RetrievedDocument] = []
    supplier_counts: dict[str, int] = {}
    supplier_cap = top_k if supplier_name else max_per_supplier

    for document_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        metadata = metadata or {}
        supplier = str(metadata.get("supplier_name", "Unknown supplier"))
        if supplier_counts.get(supplier, 0) >= supplier_cap:
            continue

        numeric_distance = float(distance)
        retrieved.append(
            {
                "id": document_id,
                "document": document,
                "metadata": metadata,
                "distance": numeric_distance,
                "similarity": max(-1.0, min(1.0, 1.0 - numeric_distance)),
            }
        )
        supplier_counts[supplier] = supplier_counts.get(supplier, 0) + 1
        if len(retrieved) == top_k:
            break

    return retrieved


def build_context(retrieved_documents: Sequence[RetrievedDocument]) -> str:
    """Format retrieved evidence into a source-labelled prompt context."""

    blocks = []
    for index, result in enumerate(retrieved_documents, start=1):
        metadata = result["metadata"]
        blocks.append(
            f"""SOURCE {index}
Supplier: {metadata.get('supplier_name', 'Unknown')}
Document Type: {metadata.get('document_type', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}

{result['document']}"""
        )
    return "\n\n".join(blocks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the supplier RAG vector index")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vector-store", type=Path, default=VECTOR_STORE_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = build_vector_index(
        rebuild=arguments.rebuild,
        batch_size=arguments.batch_size,
        vector_store_path=arguments.vector_store,
    )
    print(
        f"Indexed {result['document_count']:,} documents "
        f"({result['review_count']:,} reviews, {result['incident_count']:,} incidents) "
        f"in {result['collection']}."
    )


if __name__ == "__main__":
    main()
