"""
governance.py
=============
Part 4 Tasks 15 & 16 - the governance layer.

Task 15 (four-layer AI-governance model):
  Application layer  - LEAST AUTONOMY: only the Lookup Agent may call the ticket tool.
                       We prove no other agent is wired to it.
  (risk)             - classify this system Low/Medium/High with justification.
  Runtime layer      - a per-request token/cost budget cap; oversized requests are
                       rejected instead of silently running.

Task 16 (response caching):
  An in-memory cache keyed by normalized query text wraps the grounded-generation step,
  so a repeated identical query is a cache HIT that skips the real work. A call counter
  proves the underlying function was not called the second time.
"""

import time

import rag
from tools import _RETRIEVAL_COLLECTION
from crew import _build_crew, answer, CrewResponse


# ===========================================================================
# Task 15 (Application layer) - least autonomy
# ===========================================================================
LOOKUP_TOOL = "check_support_ticket_status"


def prove_least_autonomy() -> dict:
    """Inspect the real crew and confirm ONLY the Lookup Agent carries the ticket tool.

    Guard explanation: the tool is passed to exactly one Agent at construction time
    (see crew._build_crew - the Retrieval and Composer agents are built with no ticket
    tool). Because CrewAI can only call a tool an agent actually holds, wiring is the
    enforcement: there is no code path by which the Retrieval or Composer agent can
    invoke check_support_ticket_status. Below we verify that wiring rather than trust it.
    """
    crew = _build_crew()
    holders = {}
    for agent in crew.agents:
        tool_names = [getattr(t, "name", "") for t in getattr(agent, "tools", [])]
        holders[agent.role] = tool_names
    lookup_holders = [role for role, tools in holders.items() if LOOKUP_TOOL in tools]
    return {
        "tools_by_agent": holders,
        "agents_that_can_look_up": lookup_holders,
        "least_autonomy_ok": lookup_holders == ["Lookup Agent"],
    }


# ===========================================================================
# Task 15 (risk classification)
# ===========================================================================
def classify_risk() -> dict:
    """Low: summarization/transcription. Medium: code-gen / customer-support tickets.
    High: medical / hiring / financial data."""
    return {
        "risk_level": "Medium",
        "justification": (
            "This is a customer-support assistant: it answers support-policy questions "
            "and reads non-financial ticket metadata (status, age). It handles no "
            "medical data, makes no hiring or lending decisions, and moves no money, so "
            "it is not High risk; but it does face real customers and can act on masked "
            "PII, so it is above Low. Customer support sits squarely in the Medium band."
        ),
    }


# ===========================================================================
# Task 15 (Runtime layer) - per-request budget cap
# ===========================================================================
MAX_TOKENS_PER_REQUEST = 256  # small cap so an oversized request is easy to demonstrate


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token. Good enough for a budget guard."""
    return max(1, len(text) // 4)


def answer_within_budget(query: str, ticket_id: str = "TCK-0001",
                         max_tokens: int = MAX_TOKENS_PER_REQUEST) -> dict:
    """Reject a request whose estimated token cost exceeds the cap - before doing any
    expensive retrieval or crew work."""
    tokens = estimate_tokens(query)
    if tokens > max_tokens:
        return {"rejected": True, "reason": f"Request too large: ~{tokens} tokens "
                f"exceeds the {max_tokens}-token budget cap.", "estimated_tokens": tokens}
    result = answer(query, ticket_id)
    result["rejected"] = False
    result["estimated_tokens"] = tokens
    return result


# ===========================================================================
# Task 16 - response cache on the grounded-generation step
# ===========================================================================
class CachedGroundedGen:
    """Wrap rag.grounded_answer with an in-memory cache keyed by normalized query.
    `real_calls` counts how many times the underlying (expensive) function actually ran."""

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self.real_calls = 0

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.lower().split())  # lowercase + collapse whitespace

    def ask(self, query: str) -> dict:
        key = self._key(query)
        if key in self._cache:
            return self._cache[key]          # cache HIT - no real work
        self.real_calls += 1                  # cache MISS - do the work
        result = rag.grounded_answer(query, _RETRIEVAL_COLLECTION)
        self._cache[key] = result
        return result


def _demo():
    print("== Task 15 (Application): least autonomy ==")
    la = prove_least_autonomy()
    for role, tools in la["tools_by_agent"].items():
        print(f"  {role:<20} tools={tools}")
    print("  only Lookup Agent can look up tickets:", la["least_autonomy_ok"])

    print("\n== Task 15 (Risk classification) ==")
    risk = classify_risk()
    print("  risk_level:", risk["risk_level"])
    print(" ", risk["justification"])

    print("\n== Task 15 (Runtime): budget cap rejects an oversized request ==")
    ok = answer_within_budget("What are the business hours?")
    print(f"  normal request  -> rejected={ok['rejected']} (~{ok['estimated_tokens']} tokens)")
    huge = answer_within_budget("please explain in detail " * 120)
    print(f"  oversized request -> rejected={huge['rejected']}: {huge.get('reason')}")

    print("\n== Task 16: response cache hit ==")
    cache = CachedGroundedGen()
    q = "What are the support business hours?"
    t0 = time.perf_counter(); cache.ask(q);     miss_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); cache.ask(q + "  "); hit_ms = (time.perf_counter() - t0) * 1000
    print(f"  after 2 identical queries, real underlying calls = {cache.real_calls} (expected 1)")
    print(f"  miss took {miss_ms:.2f} ms, cache hit took {hit_ms:.3f} ms")


if __name__ == "__main__":
    import os
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    _demo()
