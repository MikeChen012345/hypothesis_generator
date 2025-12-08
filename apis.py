import os
from googleapiclient.discovery import build
import feedparser
import logging
from typing import Dict, List
import logging_config

logger = logging.getLogger(__name__)

app_config = logging_config.get_app_config()
api_sections = app_config.get("APIs", {})
google_search_config = api_sections.get("google_search", {})
semantic_scholar_config = api_sections.get("semantic_scholar", {})
arxiv_config = api_sections.get("arxiv", {})
web_search_config = app_config.get("WebSearch", {})
rag_config = app_config.get("RAG", {})

def web_search_google_api(query: str) -> str:
    """Perform a web search for the given query using Google Custom Search API and 
    return a summary of results.

    Args:
        query: The search query string.
    Returns:
        A string summarizing the search results.
    """
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


def semantic_paper_search(query: str, limit: int = 5) -> str:
    """Query Semantic Scholar for recent papers related to a hypothesis topic.

    Args:
        query: The research topic, variable, or hypothesis keyword to search.
        limit: Maximum number of papers to summarize (default 5).
    Returns:
        A formatted string summarizing the top matches.
    """
    logger.info("Querying Semantic Scholar for topic: %s", query)
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
            return f"Semantic Scholar search failed: {response.text}."
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
        abstract = paper.get("abstract") or "No abstract provided."
        authors = ", ".join(author.get("name", "") for author in paper.get("authors", [])[:3]) or "Unknown authors"
        summaries.append(
            f"[{idx}] {title} ({year}) by {authors}\nURL: {url}\nAbstract: {abstract[:400]}..."
        )

    return "\n\n".join(summaries)


def arxiv_paper_search(query: str, limit: int = 5) -> str:
    """Query arXiv for recent papers related to a hypothesis topic.

    Args:
        query: The research topic, variable, or hypothesis keyword to search.
        limit: Maximum number of papers to summarize (default 5).
    Returns:
        A formatted string summarizing the top matches.
    """
    logger.info("Querying arXiv for topic: %s", query)
    endpoint = arxiv_config.get("endpoint", "http://export.arxiv.org/api/query").rstrip("/")
    params = {
        "search_query": query.replace(" ", "+"),
        "start": 0,
        "max_results": max(1, min(int(limit or 5), 10)),
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    query_string = "&".join(f"{key}={value}" for key, value in params.items())
    full_url = f"{endpoint}?{query_string}"

    try:
        feed = feedparser.parse(full_url)
    except Exception as exc:
        logger.exception("arXiv search error")
        return f"arXiv search error: {exc}"

    entries = feed.get("entries", [])
    if not entries:
        return "No papers found for that topic."

    summaries: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        title = entry.get("title", "Untitled").replace("\n", " ").strip()
        authors = ", ".join(author.name for author in entry.get("authors", [])[:3]) or "Unknown authors"
        published = entry.get("published", "?")[:10]
        link = entry.get("link", "")
        summary = entry.get("summary", "No abstract provided.").replace("\n", " ").strip()
        summaries.append(
            f"[{idx}] {title} ({published}) by {authors}\nURL: {link}\nAbstract: {summary[:400]}..."
        )

    return "\n\n".join(summaries)