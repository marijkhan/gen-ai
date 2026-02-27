from langchain_tavily import TavilySearch
from config import TAVILY_API_KEY

import os
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

_search_tool = TavilySearch(max_results=5)


def search_web(query: str) -> str:
    """Search the web for recent content on a topic. Returns formatted results."""
    try:
        raw = _search_tool.invoke(query)
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not results:
            return "No results found."

        formatted = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")
            formatted.append(f"**{title}**\nURL: {url}\n{content}")

        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"
