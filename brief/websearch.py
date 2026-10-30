"""Live research tool: web search, allow-list filter, page fetch.

Search uses DuckDuckGo's HTML endpoint, which needs no key. Results whose host
fails the allow-list are dropped before any page is fetched. Fetched pages are
reduced to paragraph text. Scripts, navigation, and forms are skipped.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from brief.sources import live_host_allowed


SEARCH_URLS = (
    "https://html.duckduckgo.com/html/?q=",
    "https://lite.duckduckgo.com/lite/?q=",
)
SEARCH_RETRIES = 3
SEARCH_RETRY_DELAYS = (2, 5)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)
MAX_PAGE_BYTES = 2_000_000
MAX_PASSAGE_CHARS = 700
MIN_PARAGRAPH_CHARS = 60
FOOTNOTE_RE = re.compile(r"\[(?:\d+|[a-z]|citation needed|note \d+)\]", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
# Wikipedia reference-list items start with a back-link arrow or caret.
REFERENCE_MARKS = ("\u2191", "^")


class SearchError(RuntimeError):
    pass


def _get(url: str, timeout: float) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(MAX_PAGE_BYTES)
        charset = response.headers.get_content_charset()
    return _decode(raw, charset), content_type


def _decode(raw: bytes, charset: str | None) -> str:
    """Use the declared charset, else UTF-8, else Windows-1252 for older pages."""
    if charset:
        try:
            return raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


class _ResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._mode: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = attributes.get("class") or ""
        if "result__a" in classes:
            self.results.append({"href": attributes.get("href") or "", "title": "", "snippet": ""})
            self._mode = "title"
            self._buffer = []
        elif "result__snippet" in classes and self.results:
            self._mode = "snippet"
            self._buffer = []

    def handle_endtag(self, tag):
        if tag == "a" and self._mode:
            text = SPACE_RE.sub(" ", "".join(self._buffer)).strip()
            self.results[-1][self._mode] = text
            self._mode = None

    def handle_data(self, data):
        if self._mode:
            self._buffer.append(data)


def decode_result_url(href: str) -> str | None:
    """DuckDuckGo wraps each link as //duckduckgo.com/l/?uddg=<target>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or ""):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if not target:
            return None
        href = target[0]
    if not href.startswith(("http://", "https://")):
        return None
    return href


def parse_results(html: str) -> list[dict]:
    parser = _ResultParser()
    parser.feed(html)
    results, seen = [], set()
    for item in parser.results:
        url = decode_result_url(item["href"])
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({"url": url, "title": item["title"], "snippet": item["snippet"]})
    return results


def _get_retry(url: str, timeout: float, attempts: int = SEARCH_RETRIES) -> tuple[str, str]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _get(url, timeout + attempt * 5)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                time.sleep(SEARCH_RETRY_DELAYS[min(attempt, len(SEARCH_RETRY_DELAYS) - 1)])
    raise last  # type: ignore[misc]


def parse_plain_links(html: str) -> list[dict]:
    """Fallback when the result page is not the usual DuckDuckGo HTML layout."""
    results, seen = [], set()
    for match in re.finditer(r'href="(https?://[^"]+)"[^>]*>([^<]{3,160})', html, re.IGNORECASE):
        url = decode_result_url(match.group(1))
        if not url or url in seen or "duckduckgo.com" in url:
            continue
        seen.add(url)
        results.append({"url": url, "title": SPACE_RE.sub(" ", match.group(2)).strip(), "snippet": ""})
    return results


def search_web(query: str, timeout: float = 20.0) -> list[dict]:
    """Ask DuckDuckGo. Retry timeouts, then try the lite page if the HTML page fails."""
    encoded = urllib.parse.quote_plus(query)
    last: Exception | None = None
    for base in SEARCH_URLS:
        try:
            html, _type = _get_retry(base + encoded, timeout)
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        results = parse_results(html) or parse_plain_links(html)
        if results:
            return results
    if last:
        raise SearchError(f"search request failed: {last}") from last
    return []


class _TextParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "button"}
    BLOCKS = {"p", "li", "dd", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.blocks: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._block: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCKS and self._skip_depth == 0:
            self._flush()
            self._block = []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCKS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0 and self._block is not None:
            self._block.append(data)

    def _flush(self):
        if self._block is not None:
            text = clean_text("".join(self._block))
            if len(text) >= MIN_PARAGRAPH_CHARS and not text.startswith(REFERENCE_MARKS):
                self.blocks.append(text)
        self._block = None


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", FOOTNOTE_RE.sub("", text)).strip()


def extract_text(html: str) -> tuple[str, list[str]]:
    parser = _TextParser()
    parser.feed(html)
    parser.close()
    parser._flush()
    return clean_text(parser.title), parser.blocks


def _chunks(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_PASSAGE_CHARS:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > MAX_PASSAGE_CHARS:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current[:MAX_PASSAGE_CHARS])
    return out


def _topic_words(topic: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", topic.lower()) if len(word) > 2}


def fetch_passages(url: str, topic: str, per_page: int, timeout: float) -> tuple[str, list[str]]:
    html, content_type = _get(url, timeout)
    if "html" not in content_type.lower():
        raise SearchError(f"not an HTML page: {content_type or 'unknown type'}")
    title, blocks = extract_text(html)
    words = _topic_words(topic)
    chunks = [chunk for block in blocks for chunk in _chunks(block)]
    scored = []
    for index, chunk in enumerate(chunks):
        lowered = chunk.lower()
        score = sum(1 for word in words if word in lowered)
        scored.append((score, index, chunk))
    relevant = [item for item in scored if item[0] > 0] or scored
    top = sorted(relevant, key=lambda item: (-item[0], item[1]))[:per_page]
    return title, [chunk for _score, _index, chunk in sorted(top, key=lambda item: item[1])]


def live_search(topic: str, max_pages: int = 4, per_page: int = 6, timeout: float = 12.0,
                max_attempts: int = 8) -> dict:
    """Return allow-listed passages for a topic, plus what was dropped and why.

    Up to max_attempts allowed results are fetched in parallel. The first
    max_pages that return text are kept, in search-rank order.
    """
    results = search_web(topic, timeout=max(timeout, 20.0))
    allowed, dropped = [], []
    for item in results:
        (allowed if live_host_allowed(item["url"]) else dropped).append(item)
    targets = allowed[:max_attempts]

    def fetch(item: dict):
        try:
            title, chunks = fetch_passages(item["url"], topic, per_page, timeout)
            if not chunks:
                return item, "", [], "no paragraph text on the page"
            return item, title, chunks, None
        except Exception as exc:  # noqa: BLE001
            return item, "", [], str(exc)[:200]

    passages, failed, fetched = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        for item, title, chunks, error in pool.map(fetch, targets):
            if error:
                failed.append({"url": item["url"], "error": error})
                continue
            if len(fetched) >= max_pages:
                continue
            fetched.append(item["url"])
            for chunk in chunks:
                passages.append({
                    "url": item["url"],
                    "title": title or item["title"] or item["url"],
                    "snippet": chunk,
                    "section": "unlabelled",
                })
    return {
        "ok": True,
        "query": topic,
        "results_seen": len(results),
        "passages": passages,
        "fetched_urls": fetched,
        "dropped_urls": [item["url"] for item in dropped],
        "failed": failed,
    }
