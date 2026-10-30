"""Live-mode allow-list. A search result outside it is dropped before it is fetched.

Add a domain here, or list extras in BRIEF_EXTRA_DOMAINS (comma separated).
A domain also admits its subdomains: "nvidia.com" admits "docs.nvidia.com".
"""

from __future__ import annotations

import os
from urllib.parse import urlparse


ALLOWED_DOMAINS: tuple[str, ...] = (
    # Reference and code hosting
    "wikipedia.org",
    "britannica.com",
    "github.com",
    "github.io",
    "readthedocs.io",
    "arxiv.org",
    "huggingface.co",
    "pypi.org",
    "python.org",
    # Vendor documentation and engineering blogs
    "nvidia.com",
    "microsoft.com",
    "google.com",
    "blog.google",
    "ibm.com",
    "amazon.com",
    "openai.com",
    "anthropic.com",
    "meta.com",
    "oracle.com",
    "intel.com",
    "redhat.com",
    "cloudflare.com",
    "cisco.com",
    "hpe.com",
    "dell.com",
    "vmware.com",
    "databricks.com",
    "snowflake.com",
    "mongodb.com",
    "salesforce.com",
    "sap.com",
    "apple.com",
    "mozilla.org",
    "docker.com",
    "kubernetes.io",
    # Tools from the course catalogue
    "aider.chat",
    "flowiseai.com",
    "browse.ai",
    "crewai.com",
    "langchain.com",
    # Technology press and standards bodies
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
    "zdnet.com",
    "venturebeat.com",
    "infoq.com",
    "technologyreview.com",
    "computerworld.com",
    "infoworld.com",
    "techtarget.com",
    "theregister.com",
    "forbes.com",
    "reuters.com",
    "bbc.co.uk",
    "bbc.com",
    "hbr.org",
    "mckinsey.com",
    "deloitte.com",
    "gartner.com",
    "ieee.org",
    "acm.org",
    "nature.com",
    "owasp.org",
    "w3.org",
    # Universities and government
    "edu",
    "gov",
    "ac.uk",
)


def allowed_domains() -> tuple[str, ...]:
    extra = os.getenv("BRIEF_EXTRA_DOMAINS", "")
    added = tuple(item.strip().lower() for item in extra.split(",") if item.strip())
    return ALLOWED_DOMAINS + added


def live_host_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains())
