"""Shared bounds. The orchestrator reads these. Prompts do not enforce them."""

MIN_EVIDENCE_ROWS = 3
MAX_REVISION_CYCLES = 1
MIN_WORDS = 320
MAX_WORDS = 480
PROMPT_VERSION = "crew-v1"

# Live mode: the proposal's cap of 2 revisions, so at most 3 writer passes.
LIVE_PROMPT_VERSION = "crew-live-v2"
LIVE_MAX_REVISION_CYCLES = 2
LIVE_MAX_EVIDENCE_ROWS = 12
LIVE_MAX_TRADEOFF_ROWS = 3
LOCAL_SEARCH_PAGES = 3
LOCAL_PASSAGES_PER_PAGE = 8
LIVE_MIN_EVIDENCE_WORDS = 200
LIVE_TOKEN_BUDGET = 150_000
LIVE_MAX_REVIEWER_ISSUES = 5
