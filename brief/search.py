"""Allow-listed search over the offline corpus. Off-list hosts never return."""

from __future__ import annotations

from baseline.score import host_allowed
from brief.corpus import CORPUS


def search_corpus(query: str, corpus: tuple[dict, ...] = CORPUS) -> dict:
    needle = query.lower()
    passages = []
    dropped_urls = []
    for row in corpus:
        if not any(topic in needle for topic in row["topics"]):
            continue
        if not host_allowed(row["url"]):
            dropped_urls.append(row["url"])
            continue
        passages.append(dict(row))
    return {"ok": True, "passages": passages, "dropped_urls": dropped_urls}
