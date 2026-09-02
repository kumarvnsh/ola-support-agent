"""
mock_llm.py
===========
The deterministic, keyless "brain" that lets the whole CrewAI crew run with ZERO API
keys and zero network - the mode every graded transcript must use.

CrewAI drives an agent with a ReAct text protocol. On each turn it calls my LLM and
parses the text I return:

    Thought: ...
    Action: <tool_name>
    Action Input: {"arg": "value"}          -> CrewAI runs the tool, appends an Observation

    Thought: ...
    Final Answer: <text>                     -> CrewAI stops and returns this

A real model would *decide* what to write. My mock decides with plain rules instead, so
the output is identical every run.

Two silent traps the brief warns about - both handled here:

  Trap 1 ("Observation:" template): CrewAI's own system prompt literally contains the
  example line `Observation: the result of the action`. If I decided "has a tool already
  run?" by searching for the string "Observation:", I would match that TEMPLATE on the
  very first turn - before any tool ran - and wrongly jump to a final answer. Fix: my
  tools tag their real output with a unique SENTINEL, and I look for the sentinel, never
  the bare word "Observation:". (Proven with _probe.py: turn 1 has "Observation:" but no
  sentinel; turn 2 has the sentinel.)

  Trap 2 (tool dispatch by name): I never guess a tool's argument by substring-matching
  the tool's NAME. I read the argument name straight from the tool's declared
  args_schema (see _first_arg), which is the tool's own contract.
"""

import re
import json

from crewai.llms.base_llm import BaseLLM

# A marker no human text would contain. Every tool return is prefixed with it (see
# tools.py), so a REAL tool observation is unmistakable and the template example is not.
TOOL_SENTINEL = "⟪TOOLRESULT⟫"  # ⟪TOOLRESULT⟫

# Argument names that mean "a record id" rather than "a free-text query".
_ID_ARGS = {"record_id", "ticket_id", "id", "ticketid"}
_TICKET_ID_RE = re.compile(r"TCK-\d{4}", re.IGNORECASE)
_TICKET_LINE_RE = re.compile(r"TICKET:\s*(TCK-\d{4})", re.IGNORECASE)
_QUESTION_RE = re.compile(r"QUESTION:\s*(.+)")


class MockLLM(BaseLLM):
    """A BaseLLM subclass (CrewAI's documented extension point for a non-litellm LLM)."""

    def __init__(self, model: str = "mock-llm"):
        super().__init__(model=model)

    # -- the one method CrewAI calls ---------------------------------------
    def call(self, messages, tools=None, callbacks=None,
             available_functions=None, from_task=None, from_agent=None):
        joined = self._join(messages)
        agent = from_agent or getattr(from_task, "agent", None)
        agent_tools = list(getattr(agent, "tools", []) or [])

        # (1) A real tool result is already in the conversation -> answer from it.
        observation = self._real_observation(joined)
        if observation is not None:
            return ("Thought: I now have the tool result and can give the final answer.\n"
                    f"Final Answer: {observation}")

        # (2) The agent owns a tool and hasn't used it yet -> call that tool.
        if agent_tools:
            tool = agent_tools[0]  # each specialist agent is given exactly one tool
            name = getattr(tool, "name", "unknown_tool")
            arg = self._first_arg(tool)
            value = self._arg_value(arg, joined)
            return (f"Thought: I should use {name} to get grounded facts.\n"
                    f"Action: {name}\n"
                    f"Action Input: {json.dumps({arg: value})}")

        # (3) No tool (the Composer) -> combine whatever context it was handed.
        return ("Thought: I can combine the retrieved policy answer and the ticket "
                "status into one reply.\n"
                f"Final Answer: {self._compose(joined)}")

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _join(messages) -> str:
        if isinstance(messages, str):
            return messages
        return "\n".join(m.get("content", "") for m in messages)

    @staticmethod
    def _real_observation(joined: str):
        """Return the text of a genuine tool observation, or None. Keyed on the SENTINEL
        so the template's 'Observation: the result of the action' never triggers it."""
        idx = joined.rfind(TOOL_SENTINEL)
        if idx == -1:
            return None
        after = joined[idx + len(TOOL_SENTINEL):]
        # The observation sits on one line; take up to the next newline.
        return after.split("\n", 1)[0].strip()

    @staticmethod
    def _first_arg(tool) -> str:
        """Read the tool's FIRST declared argument name from its args_schema - the tool's
        own contract - rather than guessing from its name."""
        schema = getattr(tool, "args_schema", None)
        fields = getattr(schema, "model_fields", None)
        if fields:
            return next(iter(fields))
        # Fallback: parse the "Tool Arguments: {'query': ...}" line in the description.
        m = re.search(r"Tool Arguments:\s*\{'([^']+)'", getattr(tool, "description", "") or "")
        return m.group(1) if m else "query"

    @staticmethod
    def _user_question(joined: str) -> str:
        m = _QUESTION_RE.search(joined)
        if m:
            return m.group(1).strip()
        return joined.strip()[:200]  # fallback: start of the prompt

    def _arg_value(self, arg: str, joined: str) -> str:
        """Fill the tool argument. If the argument is an id, take the id from the explicit
        `TICKET:` line the task provides - NOT the first TCK-#### anywhere in the text,
        which would wrongly grab the example id in the tool's own description."""
        question = self._user_question(joined)
        if arg.lower() in _ID_ARGS or arg.lower().endswith("id"):
            line = _TICKET_LINE_RE.search(joined)   # prefer the explicit "TICKET: TCK-####" line
            if line:
                return line.group(1).upper()
            anywhere = _TICKET_ID_RE.search(joined)  # fallback: first id in the text
            return anywhere.group(0).upper() if anywhere else question
        return question

    def _compose(self, joined: str):
        """Composer path: the specialist tasks' JSON outputs were injected into this
        agent's prompt as context. I locate each by its distinctive key and stitch them
        into one reply - that IS the combination the brief asks the Composer to make."""
        policy = self._find_json(joined, '"grounded"')   # from knowledge_base_search
        ticket = self._find_json(joined, '"found"')       # from check_support_ticket_status
        bits = [b for b in (policy, ticket) if b]
        if bits:
            return "  ".join(bits)
        return self._user_question(joined)

    @staticmethod
    def _find_json(joined: str, key: str):
        """Return the smallest {...} object in `joined` that contains `key`, by matching
        braces outward from the key. Returns None if not present."""
        pos = joined.find(key)
        if pos == -1:
            return None
        start = joined.rfind("{", 0, pos)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(joined)):
            if joined[i] == "{":
                depth += 1
            elif joined[i] == "}":
                depth -= 1
                if depth == 0:
                    return joined[start:i + 1]
        return None
