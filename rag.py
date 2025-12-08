from __future__ import annotations

from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import torch


def _resolve_device(preferred: str | None = None) -> str:
    """Choose the embedding device, preferring CUDA when available."""
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int = 64,
    device: str | None = None,
) -> List[List[float]]:
    """Encode text chunks in batches to keep the GPU busy."""
    if not texts:
        return []
    embeddings = model.encode(
        texts,
        batch_size=max(1, batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
    )
    return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

def load_parquet_to_table(file_path: str) -> pa.Table:
    """
    Load a Parquet file into a PyArrow Table.
    
    Args:
        file_path (str): The path to the Parquet file.
        
    Returns:
        pa.Table: The loaded PyArrow Table.
    """
    table = pq.read_table(file_path)
    return table


def ingest_to_qdrant(
    client: QdrantClient,
    table: pa.Table,
    collection_name: str,
    *,
    batch_size: int = 256,
    flush_every: int = 512,
    device: str | None = None,
) -> None:
    """
    Ingest data from a PyArrow Table into a Qdrant collection.
    
    Args:
        client (qdrant_client.QdrantClient): The Qdrant client instance.
        table (pa.Table): The PyArrow Table containing the data.
        collection_name (str): The name of the Qdrant collection to ingest data into.
        batch_size (int): Number of chunks encoded at once on the device.
        flush_every (int): Number of pending chunks before sending vectors to Qdrant.
        device (str | None): Explicit device string (e.g., "cuda", "cuda:1", "cpu").
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,
                                              separators=["\n\n", "\n", "."])

    resolved_device = _resolve_device(device)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2', device=resolved_device)

    vector_params = VectorParams(
        size=embedding_model.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    )

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
        print(f"Deleted existing collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vector_params,
    )

    pending: List[Dict[str, Any]] = []
    point_id = 0

    def flush_pending() -> None:
        nonlocal pending, point_id
        if not pending:
            return
        texts = [item["text"] for item in pending]
        embeddings = _encode_texts(
            embedding_model,
            texts,
            batch_size=batch_size,
            device=resolved_device,
        )
        points = []
        for payload_wrapper, vector in zip(pending, embeddings):
            vector_list = vector.tolist() if hasattr(vector, "tolist") else vector
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector_list,
                    payload=payload_wrapper["payload"],
                )
            )
            point_id += 1
        client.upsert(collection_name=collection_name, points=points)
        pending.clear()

    rows = table.to_pylist()
    for row in tqdm(rows, desc="Indexing rows"):
        text = (row.get('abstract') or '').strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        if not chunks:
            continue

        base_payload = {
            "title": row.get('title', ''),
            "authors": row.get('authors', ''),
            "submission_date": row.get('submission_date', ''),
            "primary_subject": row.get('primary_subject', ''),
            "subjects": row.get('subjects', ''),
            "doi": row.get('doi', ''),
        }

        for chunk in chunks:
            payload = dict(base_payload)
            payload["text"] = chunk
            pending.append({"text": chunk, "payload": payload})
            if len(pending) >= flush_every:
                flush_pending()

    flush_pending()


def get_from_qdrant(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int,
    *,
    device: str | None = None,
) -> list[str]:
    """
    Retrieve the top K matches from a Qdrant collection based on a query.
    
    Args:
        client (qdrant_client.QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the Qdrant collection to query.
        query (str): The query string.
        top_k (int): The number of top matches to retrieve.
        
    Returns:
        list[str]: A list of the top K matching document texts.
    """
    resolved_device = _resolve_device(device)
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2', device=resolved_device)
    query_vector = _encode_texts(
        embedding_model,
        [query],
        batch_size=8,
        device=resolved_device,
    )[0]
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    
    return ["title:" + point.payload["title"] + 
            "; authors:" + ",".join(point.payload["authors"]) + 
            "; submission_date:" + point.payload["submission_date"] + 
            "; abstract: " + point.payload["text"]
            for point in results]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest data into Qdrant from Parquet file.")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into Qdrant.")
    args = parser.parse_args()
    
    qdrant_client = QdrantClient(url="http://localhost:13031")
    parquet_file_path = "data/arxiv_papers.parquet"
    table = load_parquet_to_table(parquet_file_path)
    if args.ingest:
        ingest_to_qdrant(qdrant_client, table, collection_name="documents")
    
    # result = get_from_qdrant(qdrant_client, collection_name="documents", query="fMRI", top_k=5)
    # for doc in result:
    #     print(doc)
    #     print("-----")