### Storing the tools in a separate file for better organization
import logging
from typing import Any, Dict, Optional

from langchain.tools import tool
from langgraph.store.base import BaseStore

import logging_config # apply logging configuration when importing
import memory_manager

logger = logging.getLogger(__name__)

app_config = logging_config.get_app_config()


def get_all_tools() -> list:
    """Return a list of all available tools."""
    logger.debug("Retrieving all available tools")
    return [
        retrieve_user_memory,
        store_user_memory,
        forget_user_memory,
        web_search,
        semantic_paper_search,
        hypothesis_groundedness_check,
    ]


###------------Memory Tools------------###
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
api_sections = app_config.get("APIs", {})
google_search_config = api_sections.get("google_search", {})
semantic_scholar_config = api_sections.get("semantic_scholar", {})
web_search_config = app_config.get("WebSearch", {})

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
        result = web_search_google_api(query)
        if result != "Web search configuration is missing." and result != "No results found.":
            return result
    
    return web_search_scrap(query)

def web_search_google_api(query: str) -> str:
    """Perform a web search for the given query using Google Custom Search API and 
    return a summary of results.

    Args:
        query: The search query string.
    Returns:
        A string summarizing the search results.
    """
    import os
    from googleapiclient.discovery import build
    logger.info("Using Google Custom Search API for query: %s", query)
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        logger.error("Google Search API key or Search Engine ID not set in environment variables")
        return "Web search configuration is missing."

    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=search_engine_id, num=5).execute()

    results = []
    for item in res.get("items", []):
        title = item.get("title")
        link = item.get("link")
        snippet = item.get("snippet")
        results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")

    return "\n".join(results) if results else "No results found."


def web_search_scrap(query: str) -> str:
    """Perform a web search for the given query with web scraping and return a summary of results.
    May be used when API-based search is not available.

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
    for g in soup.find_all('div', class_='g'):
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

    endpoint = semantic_scholar_config.get("endpoint", "https://api.semanticscholar.org/graph/v1").rstrip("/")
    url = f"{endpoint}/paper/search"
    params = {
        "query": query,
        "limit": max(1, min(int(limit or 5), 10)),
        "fields": "title,authors,year,url,abstract"
    }
    headers = {"Accept": "application/json"}
    api_key = semantic_scholar_config.get("api_key")
    if api_key and not api_key.startswith("${"):
        headers["x-api-key"] = api_key

    try:
        import requests
        response = requests.get(url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            logger.error("Semantic Scholar request failed: %s", response.text)
            return "Semantic Scholar search failed."
        data = response.json()
    except Exception as exc:
        logger.exception("Semantic Scholar search error")
        return f"Semantic Scholar search error: {exc}"

    papers = data.get("data", [])
    if not papers:
        return "No papers found for that topic."

    summaries: list[str] = []
    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "?")
        url = paper.get("url", "")
        abstract = paper.get("abstract", "No abstract provided.")
        authors = ", ".join(author.get("name", "") for author in paper.get("authors", [])[:3]) or "Unknown authors"
        summaries.append(
            f"[{idx}] {title} ({year}) by {authors}\nURL: {url}\nAbstract: {abstract[:400]}..."
        )

    return "\n\n".join(summaries)


@tool
def hypothesis_groundedness_check(statement: str, variables: Optional[str] = None, context: Optional[str] = None) -> str:
    """Provide a lightweight groundedness and clarity assessment for a hypothesis statement.

    Args:
        statement: The hypothesis to evaluate.
        variables: Optional description of independent/dependent variables.
        context: Optional study context or population notes.
    Returns:
        A textual report covering clarity, measurability, causal language, and potential gaps.
    """
    if not statement:
        return "No hypothesis provided."

    statement_lower = statement.lower()
    tokens = statement.split()
    length_score = 1 if len(tokens) >= 12 else 0
    causal_markers = ["cause", "lead", "impact", "affect", "increase", "decrease", "if"]
    has_causal_language = any(marker in statement_lower for marker in causal_markers)
    measurement_terms = ["percent", "rate", "score", "level", "concentration", "signal", "performance"]
    measurable = any(term in statement_lower for term in measurement_terms)
    comparison_terms = ["compared", "versus", "relative", "control", "baseline"]
    comparative = any(term in statement_lower for term in comparison_terms)

    score = length_score
    score += 1 if has_causal_language else 0
    score += 1 if measurable else 0
    score += 1 if comparative else 0
    if variables:
        score += 1

    if score >= 4:
        rating = "high"
    elif score >= 2:
        rating = "moderate"
    else:
        rating = "low"

    feedback = [
        f"Overall groundedness rating: {rating.upper()} (score={score}/5)",
        "✔ Contains measurable signals." if measurable else "✖ Add measurable signals (e.g., rates, levels).",
        "✔ Includes causal or directional language." if has_causal_language else "✖ Clarify causal direction (if applicable).",
        "✔ Contrasts conditions or baselines." if comparative else "✖ Specify what the hypothesis is compared against.",
    ]

    if not variables:
        feedback.append("✖ Provide explicit independent/dependent variables to tighten scope.")
    if context:
        feedback.append(f"Context noted: {context}")

    return "\n".join(feedback)
