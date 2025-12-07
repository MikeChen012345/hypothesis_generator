### Memory management utilities
import logging
import uuid
from typing import Iterable, Sequence
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PointIdsList
from sentence_transformers import SentenceTransformer

import logging_config # apply logging configuration when importing

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = {
    "name": None,
    "model": None,
}


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Return a cached embedding model, reloading only when the name changes."""
    cache_name = _EMBEDDING_MODEL_CACHE["name"]
    cached_model = _EMBEDDING_MODEL_CACHE["model"]

    if cached_model is not None and cache_name == model_name:
        return cached_model

    logger.info("Loading embedding model '%s'", model_name)
    model = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_CACHE["name"] = model_name
    _EMBEDDING_MODEL_CACHE["model"] = model
    return model


def retrieve_user_memory(store: BaseStore, agent_memory_config: dict, config: dict, query: str) -> str:
    """Retrieve user memories from both vector and keyword stores using the provided store.

    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to search memories.
    Returns:
        A string containing the retrieved memories. Empty string if no store is provided 
        or no memories found.
    """
    info = ""
    # Retrieve using vector store
    if agent_memory_config.get("vector_store", {}).get("use", False):
        info += retrieve_user_memory_vector(
            agent_memory_config,
            config,
            query,
        )
    # Retrieve using keyword store
    if agent_memory_config.get("database", {}).get("long_term", {}).get("use", False):
        info += retrieve_user_memory_keyword(
            store,
            agent_memory_config,
            config,
            query,
        )
    return info


def store_user_memory(store: BaseStore, agent_memory_config: dict, config: dict, query: str) -> bool:
    """Store new user memories using both vector and keyword stores.

    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to store as memory.
    Returns:
        A boolean indicating whether the memory was successfully stored.
    """
    result = True
    # Store using keyword store
    if agent_memory_config.get("database", {}).get("long_term", {}).get("use", False):
        result = result and store_user_memory_keyword(
            store,
            agent_memory_config,
            config,
            query,
        )
    # Store using vector store
    if agent_memory_config.get("vector_store", {}).get("use", False):
        result = result and store_user_memory_vector(
            agent_memory_config,
            config,
            query,
        )
    return result


def forget_user_memory(store: BaseStore, agent_memory_config: dict, config: dict, *, query: str | None = None, memory_ids: Sequence[str] | None = None, top_k: int | None = None) -> int:
    """Delete user memories from both vector and keyword stores.

    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: An optional query string to find similar memories to delete.
        memory_ids: An optional list of explicit memory IDs to delete.
        top_k: The maximum number of similar memories to delete when using a query.
    Returns:
        The total number of deleted memories.
    """
    total_deleted = 0
    # Forget from keyword store
    if agent_memory_config.get("database", {}).get("long_term", {}).get("use", False):
        deleted = forget_user_memory_keyword(
            store,
            agent_memory_config,
            config,
            query=query,
            memory_ids=memory_ids,
            top_k=top_k,
        )
        total_deleted += deleted
    # Forget from vector store
    if agent_memory_config.get("vector_store", {}).get("use", False):
        deleted = forget_user_memory_vector(
            agent_memory_config,
            config,
            query=query,
            point_ids=memory_ids,
            top_k=top_k,
        )
        total_deleted += deleted
    return total_deleted


############################
### Keyword-based Memory ###
############################
def retrieve_user_memory_keyword(store: BaseStore, agent_memory_config: dict, config: dict, query: str) -> str:
    """Retrieve user memories using the provided store.

    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to search memories.
    Returns:
        A string containing the retrieved memories. Empty string if no store is provided 
        or no memories found.
    """
    if store is not None:
        user_id = config["configurable"]["user_id"]
        namespace = agent_memory_config.get("database", {}).get("long_term", {}).get("namespace", "user:{user_id}").format(user_id=user_id)
        top_k = agent_memory_config.get("database", {}).get("long_term", {}).get("top_k", 5)
        memories = store.search(namespace, query=str(query), limit=top_k)
        info = "\n".join([memory.value["data"] for memory in memories])
        info = f"Memories of user: {info}" if info else ""
        logging.info(f"Retrieved memories for user {user_id} with query {query}: {info}")
        return info
    return ""


def store_user_memory_keyword(store: BaseStore, agent_memory_config: dict, config: dict, query: str) -> bool:
    """Store new user memories using the provided store.
    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to store as memory.
    Returns:
        A boolean indicating whether the memory was successfully stored.
    """
    if store is not None:
        user_id = config["configurable"]["user_id"]
        namespace = agent_memory_config.get("database", {}).get("long_term", {}).get("namespace", "user:{user_id}").format(user_id=user_id)
        memory = query.lower() # use summary of query as memory instead?
        uid = str(uuid.uuid4())
        store.put(namespace, uid, {"data": memory})
        logging.info(f"User {user_id} stored memory {uid}: {memory}")
        return True
    return False


def forget_user_memory_keyword(store: BaseStore, agent_memory_config: dict, config: dict,
    *, query: str | None = None, memory_ids: Sequence[str] | None = None, top_k: int | None = None,
) -> int:
    """Delete keyword-based memories either by explicit IDs or by query similarity.

    Args:
        store: The memory store instance.
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: An optional query string to find similar memories to delete.
        memory_ids: An optional list of explicit memory IDs to delete.
        top_k: The maximum number of similar memories to delete when using a query.
    Returns:
        The number of deleted memories.

    """
    if store is None:
        logging.info("Keyword memory store is not configured; nothing to forget")
        return 0

    user_id = config["configurable"]["user_id"]
    namespace_template = agent_memory_config.get("database", {}).get("long_term", {}).get(
        "namespace", "user:{user_id}"
    )
    namespace = namespace_template.format(user_id=user_id)
    top_k = top_k or agent_memory_config.get("database", {}).get("long_term", {}).get("forget_k", 1)

    keys_to_delete: set[str] = set(memory_ids or [])
    if query:
        matches = store.search(namespace, query=str(query), limit=top_k)
        for match in matches:
            key = getattr(match, "key", None) or getattr(match, "id", None)
            if key:
                keys_to_delete.add(key)

    deleted = 0
    for key in keys_to_delete:
        try:
            store.delete(namespace, key)
            deleted += 1
        except Exception as exc:
            logging.error("Failed to delete keyword memory %s for user %s: %s", key, user_id, exc)

    logging.info("Deleted %s keyword memories for user %s", deleted, user_id)
    return deleted


###------------Dangerous operations------------###
def delete_checkpoint_for_thread(db_uri: str, thread_id: str) -> None:
    """Deletes all checkpoint data associated with a specific thread ID.

    Args:
        db_uri (str): The database connection string.
        thread_id (str): The thread ID whose checkpoint data should be deleted.
    """
    with PostgresSaver.from_conn_string(db_uri) as checkpoint:
        checkpoint.delete_thread(thread_id)


def delete_store_data_for_user(db_uri: str, user_id: str, key: str) -> None:
    """Deletes stored data associated with a specific user ID and key.
    (Not possible to delete all data for a user at once)

    Args:
        db_uri (str): The database connection string.
        user_id (str): The user ID whose stored data should be deleted.
        key (str): The key of the stored data to delete.
    """
    with PostgresStore.from_conn_string(db_uri) as store:
        namespace = ("memories", user_id)
        store.delete(namespace, key)


############################
### Vector-based Memory ####
############################
def retrieve_user_memory_vector(agent_memory_config: dict, config: dict, query: str) -> str:
    """Retrieve user memories using vector store.
    Args:
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to search memories.
    Returns:
        A string containing the retrieved memories. Empty string if no memories found.
    """
    try:
        user_id = config["configurable"]["user_id"]
        url = agent_memory_config.get("vector_store", {}).get("connection_url")
        client = QdrantClient(url=url)
        model_name = agent_memory_config.get("vector_store", {}).get("embedding_model", "sentence-transformers/paraphrase-albert-small-v2")
        embedding_model = _get_embedding_model(model_name)

        top_k = agent_memory_config.get("vector_store", {}).get("top_k", 5)

        query_vector = embedding_model.encode([query])[0].tolist()
        
        user_namespace = agent_memory_config.get("vector_store", {}).get(
            "namespace", "user:{user_id}").format(user_id=config["configurable"]["user_id"])
        if not client.collection_exists(user_namespace):
            logging.warning(f"Vector collection {user_namespace} does not exist.")
            return ""

        res = client.query_points(collection_name=user_namespace, 
                                query=query_vector, limit=top_k, 
                                with_payload=True).points
        res = [point.payload["text"] for point in res]
        info = "\n".join(res)
        info = f"Memories of user: {info}" if info else ""
        logging.info(f"Retrieved vector memories for user {user_id} with query {query}: {info}")
        return info
    except Exception as e:
        logging.error(f"Error retrieving vector memories for user {user_id} with query {query}: {e}")
        return ""


def store_user_memory_vector(agent_memory_config: dict, config: dict, query: str) -> bool:
    """Store new user memories using vector store.
    Args:
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: The query string to store as memory.
    Returns:
        A boolean indicating whether the memory was successfully stored.
    """
    try:
        user_id = config["configurable"]["user_id"]
        url = agent_memory_config.get("vector_store", {}).get("connection_url")
        client = QdrantClient(url=url)
        model_name = agent_memory_config.get("vector_store", {}).get("embedding_model", "sentence-transformers/paraphrase-albert-small-v2")
        embedding_model = _get_embedding_model(model_name)
        vector_size = embedding_model.get_sentence_embedding_dimension()

        distance_type = agent_memory_config.get("vector_store", {}).get("similarity_metric", "cosine").upper()
        if distance_type == "COSINE":
            distance = Distance.COSINE
        elif distance_type == "EUCLIDEAN":
            distance = Distance.EUCLID
        elif distance_type == "DOT":
            distance = Distance.DOT
        elif distance_type == "MANHATTAN":
            distance = Distance.MANHATTAN
        else:
            logging.warning(f"Unknown distance type {distance_type}, defaulting to COSINE")
            distance = Distance.COSINE

        vector_params = VectorParams(
            size=vector_size,
            distance=distance
        )

        user_namespace = agent_memory_config.get("vector_store", {}).get(
            "namespace", "user:{user_id}").format(user_id=user_id)
        print(user_namespace)
        if not client.collection_exists(user_namespace):
            client.create_collection(
                collection_name=user_namespace,
                vectors_config=vector_params,
            )
            logging.info(f"Vector collection {user_namespace} for user {user_id} does not exist. Created new collection.")

        vector = embedding_model.encode([query])[0].tolist()
        uid = str(uuid.uuid4())
        client.upsert(
            collection_name=user_namespace,
            points=[
                PointStruct(
                    id=uid,
                    vector=vector,
                    payload={"text": query}
                )
            ]
        )
        logging.info(f"User {user_id} stored vector memory {uid}: {query}")
        return True
    except Exception as e:
        logging.error(f"Error storing vector memory for user {user_id} with query {query}: {e}")
        return False


def forget_user_memory_vector(
    agent_memory_config: dict,
    config: dict,
    *,
    query: str | None = None,
    point_ids: Sequence[str] | None = None,
    top_k: int | None = None,
) -> int:
    """Delete vector memories via explicit IDs or similarity search.

    Args:
        agent_memory_config: Configuration for agent memory.
        config: The overall configuration including user ID.
        query: An optional query string to find similar memories to delete.
        point_ids: An optional list of explicit memory IDs to delete.
        top_k: The maximum number of similar memories to delete when using a query.
    Returns:
        The number of deleted memories.
    """
    try:
        user_id = config["configurable"]["user_id"]
        url = agent_memory_config.get("vector_store", {}).get("connection_url")
        client = QdrantClient(url=url)
        model_name = agent_memory_config.get("vector_store", {}).get(
            "embedding_model", "sentence-transformers/paraphrase-albert-small-v2"
        )
        user_namespace = agent_memory_config.get("vector_store", {}).get(
            "namespace", "user:{user_id}"
        ).format(user_id=user_id)

        if not client.collection_exists(user_namespace):
            logging.warning("Vector collection %s does not exist; nothing to forget", user_namespace)
            return 0

        ids_to_delete: set[str] = set(point_ids or [])

        if query:
            embedding_model = _get_embedding_model(model_name)
            limit = top_k or agent_memory_config.get("vector_store", {}).get("forget_k", 1)
            query_vector = embedding_model.encode([query])[0].tolist()
            points = client.query_points(
                collection_name=user_namespace,
                query=query_vector,
                limit=limit,
                with_payload=False,
            ).points
            ids_to_delete.update(str(point.id) for point in points if point.id is not None)

        if not ids_to_delete:
            logging.info("No vector memories matched the forget criteria for user %s", user_id)
            return 0

        selector = PointIdsList(points=list(ids_to_delete))
        client.delete(collection_name=user_namespace, points_selector=selector)
        logging.info("Deleted %s vector memories for user %s", len(ids_to_delete), user_id)
        return len(ids_to_delete)
    except Exception as exc:
        logging.error("Error forgetting vector memories for user %s: %s", config["configurable"]["user_id"], exc)
        return 0
    


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "test_thread",
            "user_id": "test_user",
        },
    }
    agent_memory_config = logging_config.get_agent_memory_config()
    store_user_memory_vector(agent_memory_config, config, "Sample query to store")
    print(retrieve_user_memory_vector(agent_memory_config, config, "Sample query"))