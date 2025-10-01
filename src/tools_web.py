# src/tools_web.py
from typing import List
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup
from strands import tool

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17 Safari/605.1.15"
}

def _ddg_search_html(query: str, k: int = 5) -> List[str]:
    """Scrape DuckDuckGo (HTML) for top results, no API key needed."""
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}&kl=us-en"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    out: List[str] = []
    for res in soup.select(".result")[:k]:
        a = res.select_one(".result__a")
        if not a: 
            continue
        title = a.get_text(" ", strip=True)
        link = a.get("href", "")
        snip = soup.new_tag("div")
        snippet_el = res.select_one(".result__snippet")
        snippet = (snippet_el.get_text(" ", strip=True) if snippet_el else "").strip()
        if snippet:
            out.append(f"{title} — {link} — {snippet}")
        else:
            out.append(f"{title} — {link}")
    return out

@tool
def web_search(query: str, k: int = 5) -> List[str]:
    """
    Search the web for up-to-date/local info. Returns up to k lines:
    'Title — URL — Snippet'. Use this for local places, news, hours, etc.
    """
    try:
        return _ddg_search_html(query, k=k)
    except Exception as e:
        # Keep it graceful for the agent
        return [f"(web_search error: {e})"]