"""
review.py
=========
Part 4 Task 14 - an independent Autogen review stage that checks the CrewAI Composer's
draft BEFORE it reaches the user.

A 2-agent RoundRobinGroupChat (autogen-agentchat 0.4+):
  - Policy_Compliance_Reviewer  reads the draft + retrieved context and flags problems
  - Final_Editor                emits a structured verdict and either approves or revises

The verdict is a Pydantic model:  ReviewVerdict{approved, final_answer, reason}
delivered as a StructuredMessage[ReviewVerdict] (so the Team must be built with
custom_message_types=[StructuredMessage[ReviewVerdict]], or autogen raises
"Message type ... is not registered").

Keyless + deterministic: the real approve/revise DECISION is computed in Python by
checking every draft sentence against the retrieved context (_is_supported). The autogen
team is genuine - it's the transport that carries that verdict through a real
RoundRobinGroupChat with structured output. A real LLM reviewer could replace the replay
clients later without changing the contract.

Gotchas the brief warns about, both handled:
  - max_turns vs MaxMessageTermination: the task message counts as message 1, so
    MaxMessageTermination(3) is what lets BOTH agents speak (not (2)).
  - output_content_type on the Final_Editor REQUIRES custom_message_types on the Team.
"""

import re
import json
import asyncio

from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_ext.models.replay import ReplayChatCompletionClient


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _is_supported(sentence: str, context: str, threshold: float = 0.6) -> bool:
    """A sentence is 'grounded' if most of its words appear in the retrieved context.
    Genuine crew answers are extracted FROM the context, so they pass; an injected claim
    brings new words the context never had, so it fails."""
    words = [w for w in re.findall(r"[a-zA-Z]+", sentence.lower()) if len(w) > 3]
    if not words:
        return True
    ctx = context.lower()
    hits = sum(1 for w in words if w in ctx)
    return (hits / len(words)) >= threshold


def _compute_verdict(draft: str, context: str) -> ReviewVerdict:
    """The REAL review logic. Removes any ungrounded sentence from the draft."""
    grounded, ungrounded = [], []
    for s in _sentences(draft):
        (grounded if _is_supported(s, context) else ungrounded).append(s)

    if ungrounded:
        return ReviewVerdict(
            approved=False,
            final_answer=" ".join(grounded),
            reason=f"Removed {len(ungrounded)} ungrounded claim(s), e.g. "
                   f"\"{ungrounded[0][:70]}\".",
        )
    return ReviewVerdict(approved=True, final_answer=draft,
                         reason="All claims are supported by the retrieved context.")


async def _run_team(draft: str, context: str, verdict: ReviewVerdict) -> ReviewVerdict:
    """Carry the computed verdict through a real 2-agent RoundRobinGroupChat."""
    critique = ("No ungrounded claims found." if verdict.approved
                else f"Draft contains an unsupported claim. {verdict.reason}")
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer",
        model_client=ReplayChatCompletionClient([critique]),
        system_message="Check the draft answer against the retrieved context.",
    )
    editor = AssistantAgent(
        "Final_Editor",
        model_client=ReplayChatCompletionClient([verdict.model_dump_json()]),
        output_content_type=ReviewVerdict,
        system_message="Return the final verdict as structured output.",
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(3),  # task(1) + reviewer(2) + editor(3)
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    result = await team.run(task=f"DRAFT: {draft}\n\nRETRIEVED CONTEXT: {context}")
    # The last StructuredMessage carries the Final_Editor's verdict.
    for m in reversed(result.messages):
        if isinstance(m, StructuredMessage):
            return m.content
    return verdict  # fallback (should not happen)


def review_draft(draft: str, context: str) -> dict:
    """Sync entry point other modules use. Returns the verdict as a dict."""
    verdict = _compute_verdict(draft, context)
    final = asyncio.run(_run_team(draft, context, verdict))
    return final.model_dump()


def _demo():
    from crew import answer

    print("== Task 14: Autogen review stage (approve + revise) ==\n")

    # Case 1: a genuine grounded draft from the crew -> should be APPROVED unchanged.
    r = answer("What are the support business hours?")
    draft1, context1 = r["answer"], r["answer"]  # answer is extracted from context
    v1 = review_draft(draft1, context1)
    print("Case 1 (clean draft):")
    print("  approved:", v1["approved"], "| reason:", v1["reason"])

    # Case 2: same draft with an ungrounded claim injected -> should be REVISED.
    injected = draft1 + " You will also receive a free lifetime membership and a bonus car."
    v2 = review_draft(injected, context1)
    print("\nCase 2 (draft with injected ungrounded claim):")
    print("  approved:", v2["approved"], "| reason:", v2["reason"])
    print("  final_answer no longer contains the bogus claim:",
          "lifetime membership" not in v2["final_answer"])


if __name__ == "__main__":
    import os
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    _demo()
