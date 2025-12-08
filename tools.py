### Storing the tools in a separate file for better organization
import logging
from typing import Any, Dict, Optional

from langchain.tools import tool
from langgraph.store.base import BaseStore
import feedparser
from qdrant_client import QdrantClient

import logging_config # apply logging configuration when importing
import memory_manager
import apis
import rag

logger = logging.getLogger(__name__)

app_config = logging_config.get_app_config()
api_sections = app_config.get("APIs", {})
google_search_config = api_sections.get("google_search", {})
semantic_scholar_config = api_sections.get("semantic_scholar", {})
arxiv_config = api_sections.get("arxiv", {})
web_search_config = app_config.get("WebSearch", {})
rag_config = app_config.get("RAG", {})


def get_all_tools() -> list:
    """Return a list of all available tools."""
    logger.debug("Retrieving all available tools")
    tools = [
        # retrieve_user_memory,
        # store_user_memory,
        # forget_user_memory,
    ]

    if web_search_config.get("use", True):
        tools.append(web_search)
    if semantic_scholar_config.get("use_api", True):
        tools.append(semantic_paper_search)
    if arxiv_config.get("use_api", True):
        tools.append(arxiv_paper_search)
    logger.info("Tools available: %d", len(tools))
    
    return tools


###------------User Memory Tools------------###
_DEFAULT_STORE: BaseStore | None = None
_DEFAULT_AGENT_MEMORY_CONFIG: Dict[str, Any] | None = None
_DEFAULT_CONFIG: Dict[str, Any] | None = None

def configure_memory_tool(*, store: BaseStore | None, agent_memory_config: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Configure defaults used by the memory retrieval tool.

    Args:
        store: Active memory store (set to ``None`` to disable retrieval).
        agent_memory_config: Agent memory configuration dictionary.
        config: Workflow/runtime configuration.
    """
    global _DEFAULT_STORE, _DEFAULT_AGENT_MEMORY_CONFIG, _DEFAULT_CONFIG
    _DEFAULT_STORE = store
    _DEFAULT_AGENT_MEMORY_CONFIG = agent_memory_config
    _DEFAULT_CONFIG = config
    logger.debug("Memory tool configured: store=%s", type(store).__name__ if store else "None")


@tool
def retrieve_user_memory(query: str) -> str:
    """Retrieve user memories using the configured defaults.

    Args:
        query: The query string to search memories.
    Returns:
        Retrieved memories as a single string, or an empty string when unavailable.
    """
    if _DEFAULT_STORE is None or _DEFAULT_AGENT_MEMORY_CONFIG is None or _DEFAULT_CONFIG is None:
        logger.warning("Memory tool called before configuration; cannot retrieve memory")
        return ""
    
    memories = memory_manager.retrieve_user_memory(
        _DEFAULT_STORE,
        _DEFAULT_AGENT_MEMORY_CONFIG,
        _DEFAULT_CONFIG,
        query,
    )
    return memories


@tool
def store_user_memory(query: str) -> bool:
    """Store new user memories using the configured defaults.

    Args:
        query: The query string to store as memory.
    Returns:
        A boolean indicating whether the memory was successfully stored.
    """
    if _DEFAULT_STORE is None or _DEFAULT_AGENT_MEMORY_CONFIG is None or _DEFAULT_CONFIG is None:
        logger.warning("Memory tool called before configuration; cannot store memory")
        return False

    success = memory_manager.store_user_memory(
        _DEFAULT_STORE,
        _DEFAULT_AGENT_MEMORY_CONFIG,
        _DEFAULT_CONFIG,
        query,
    )
    return success


@tool
def forget_user_memory(query: str, memory_ids: list[str] | None = None) -> int:
    """Forget user memories using the configured defaults.

    Args:
        query: A query string to find similar memories to delete.
        memory_ids: An optional list of explicit memory IDs to delete.
    Returns:
        The total number of deleted memories.
    """
    if _DEFAULT_STORE is None or _DEFAULT_AGENT_MEMORY_CONFIG is None or _DEFAULT_CONFIG is None:
        logger.warning("Memory tool called before configuration; cannot forget memory")
        return 0

    total_deleted = memory_manager.forget_user_memory(
        _DEFAULT_STORE,
        _DEFAULT_AGENT_MEMORY_CONFIG,
        _DEFAULT_CONFIG,
        query=query,
        memory_ids=memory_ids,
    )
    return total_deleted


###------------Web Search Tools------------###

@tool
def web_search(query: str) -> str:
    """Perform a web search for the given query and return a summary of results.
    Wrapper to choose between different web search implementations.

    Args:
        query: The search query string.
    Returns:
        A string summarizing the search results.
    """
    use_web_search = web_search_config.get("use", True)
    if not use_web_search:
        logger.error("Web search disabled in configuration")
        return "Web search is disabled. Please avoid using this tool."
    
    use_api = google_search_config.get("use_api", False)
    if not use_api:
        logger.warning("Web search API usage disabled in configuration. Redirecting to web scraping.")
        return web_search_scrap(query)
    
    # prefer API-based search if API keys are available.
    import os
    if os.getenv("GOOGLE_SEARCH_API_KEY") and os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
        result = apis.web_search_google_api(query)
        if result != "Web search configuration is missing." and result != "No results found.":
            return result
    
    return web_search_scrap(query)


def web_search_scrap(query: str) -> str:
    """Perform a web search for the given query with web scraping and return a summary of results.
    May be used when API-based search is not available.
    (Currently not working due to anti-scraping measures on search engines.)

    Args:
        query: The search query string.
    Returns:
        A string summarizing the search results.
    """
    import requests
    from bs4 import BeautifulSoup
    search_url = web_search_config.get("default_search_endpoint", "https://www.google.com/search")
    logger.info("Using web scraping for query: %s", query)
    search_url = f"{search_url}?q={query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        logger.error("Web search failed with status code %d", response.status_code)
        return "Web search failed."
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    print(soup.prettify())
    for g in soup.find_all('div', class_='LC20lb MBeuO DKV0Md'):
        print(g)
        title = g.find('h3')
        if title:
            title_text = title.get_text()
            link = g.find('a')['href']
            snippet = g.find('span', class_='aCOpRe').get_text() if g.find('span', class_='aCOpRe') else ''
            results.append(f"Title: {title_text}\nLink: {link}\nSnippet: {snippet}\n")
        if len(results) >= 5:
            break
    return "\n".join(results) if results else "No results found."


###------------Research Tools------------###
@tool
def semantic_paper_search(query: str, limit: int = 5) -> str:
    """Query Semantic Scholar for recent papers related to a hypothesis topic.

    Args:
        query: The research topic, variable, or hypothesis keyword to search.
        limit: Maximum number of papers to summarize (default 5).
    Returns:
        A formatted string summarizing the top matches.
    """
    if not semantic_scholar_config.get("use_api", False):
        return "Semantic Scholar search is disabled in the configuration."

    return apis.semantic_paper_search(query, limit)


@tool
def arxiv_paper_search(query: str, limit: int = 5) -> str:
    """Query arXiv for recent papers related to a hypothesis topic.
    This includes keyword search using arXiv's API and semantic search using RAG.

    Args:
        query: The research topic, variable, or hypothesis keyword to search.
        limit: Maximum number of papers to summarize (default 5).
    Returns:
        A formatted string summarizing the top matches.
    """
    results = ""
    if arxiv_config.get("use_api", True):
        api_results = apis.arxiv_paper_search(query, limit)
        results += "arXiv API Results:\n" + api_results
    if rag_config.get("use", False):
        rag_results = retrieve_relevant_arxiv_paper(query, limit)
        results += "\n\nRAG Results:\n" + rag_results

    return results



###------------RAG------------###
def retrieve_relevant_arxiv_paper(query: str, limit: int = 5) -> str:
    """Retrieve relevant arXiv papers for the given query using RAG approach.

    Args:
        query: The research topic, variable, or hypothesis keyword to search.
        limit: Maximum number of papers to summarize (default 5).
    Returns:
        A formatted string summarizing the top matches.
    """
    logger.info("Retrieving relevant arXiv papers from RAG for topic: %s", query)
    client = QdrantClient(url=rag_config.get("endpoint", "http://localhost:13031"))
    if client is None:
        return "RAG Qdrant client is not configured."
    results = rag.get_from_qdrant(client, collection_name="documents", query=query, top_k=limit)
    if not results:
        return "No relevant papers found for that topic."
    return "\n\n".join(results)



if __name__ == "__main__":
    # Simple test cases for the tools
    # print(web_search_scrap("Artificial Intelligence in Healthcare"))
    print(arxiv_paper_search("quantum computing", limit=3))