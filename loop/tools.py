"""Narrow tools. The model may name one. The registry decides whether it runs."""

from __future__ import annotations

from baseline.score import host_allowed
from loop.corpus import CORPUS


class ToolRegistry:
    def __init__(self, corpus: tuple[dict, ...] = CORPUS) -> None:
        self.corpus = corpus
        self.executed: list[tuple[str, dict]] = []

    def names(self) -> set[str]:
        return {"search_corpus"}

    def execute(self, name: str, arguments: dict) -> dict:
        if name not in self.names():
            raise KeyError(name)
        self.executed.append((name, dict(arguments)))
        if name == "search_corpus":
            return search_corpus(str(arguments["query"]), self.corpus)
        raise KeyError(name)


def search_corpus(query: str, corpus: tuple[dict, ...] = CORPUS) -> dict:
    """Read-only. Off-allow-list rows are dropped before they become state."""
    needle = query.lower()
    passages = []
    for row in corpus:
        if not host_allowed(row["url"]):
            continue
        if any(topic in needle for topic in row["topics"]):
            passages.append({
                "url": row["url"],
                "title": row["title"],
                "snippet": row["snippet"],
            })
    return {"ok": True, "passages": passages}
