"""
tools.py
========
The two tools the crew's specialist agents call. Both wrap logic that already exists
(Part 1's RAG core and Task 6's ticket lookup) and tag their output with TOOL_SENTINEL
so mock_llm can tell a real result apart from CrewAI's template example.

  knowledge_base_search(query)          -> grounded policy answer from the KB
  check_support_ticket_status(record_id)-> ticket status + designed escalation score
"""

import json

from crewai.tools import tool

import rag
from dataset import check_support_ticket_status as _lookup
from mock_llm import TOOL_SENTINEL

# Build both Chroma collections once at import. Task 5 found fixed-size chunking more
# precise, so the crew uses that collection as its retrieval source.
_FIXED_COLLECTION, _SENTENCE_COLLECTION = rag.build_collections()
_RETRIEVAL_COLLECTION = _FIXED_COLLECTION


@tool("knowledge_base_search")
def knowledge_base_search(query: str) -> str:
    """Search the Ola support knowledge base and return a grounded policy answer.

    Use this for any question about support POLICY (SLAs, refunds, escalation, hours,
    data retention, etc.). Returns JSON with the answer and its source documents.
    """
    result = rag.grounded_answer(query, _RETRIEVAL_COLLECTION)
    payload = json.dumps({
        "grounded": result["grounded"],
        "answer": result["answer"],
        "sources": result["sources"],
    })
    return TOOL_SENTINEL + payload  # single line - safe as a ReAct Observation


@tool("check_support_ticket_status")
def check_support_ticket_status(record_id: str) -> str:
    """Look up one support ticket by its record id (e.g. TCK-0007).

    Use this for questions about a SPECIFIC ticket. Returns JSON with the ticket's
    status, resolution time, and the designed escalation score.
    """
    result = _lookup(record_id)
    return TOOL_SENTINEL + json.dumps(result)
