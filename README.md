# Ola Domain Support Agent — Final Capstone

**Track: Ola — Business Operations / Customer Support.** *(Question 1 domain = this track; it matches this GitHub submission.)*

A production-minded support agent that answers Ola support-policy questions from a
knowledge base, looks up a specific ticket's status, remembers a conversation, guards
against misuse, has its answers reviewed by a second independent agent team, runs under
an explicit governance policy, and is deployed behind a FastAPI backend — all evaluated
end to end.

**Everything runs with zero API keys and zero network access** via a deterministic
`MOCK_LLM`. Embeddings (SentenceTransformers) and the vector index (ChromaDB) are free
and local.

---

## 1. Setup (do this once)

You need **Python 3.12** (newer ML libraries don't yet build cleanly on 3.13/3.14).

```bash
# from the project folder
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

That installs SentenceTransformers, ChromaDB, CrewAI, autogen-agentchat, LangChain,
FastAPI, and Pydantic. The first run downloads the local embedding model
(`all-MiniLM-L6-v2`, ~90 MB) — that one download needs internet; nothing else does.

**Two environment flags** are set automatically inside the code, but for reference they
disable CrewAI's network telemetry (required for the zero-network rule):

```bash
export CREWAI_DISABLE_TELEMETRY=true
export OTEL_SDK_DISABLED=true
export CREWAI_TRACING_ENABLED=false
```

**No API keys, no accounts, no credit card.**

---

## 2. What each script does (plain language)

| File | What it is |
|------|-----------|
| `dataset.py` | Makes 45 fake support tickets (deterministic) + the ticket-lookup tool |
| `kb/` | 12 knowledge-base policy documents (SLA, refunds, escalation…) |
| `rag.py` | The search brain: chunk → embed → store → retrieve → grounded answer |
| `mock_llm.py` | The keyless "brain" that lets CrewAI run with no API key |
| `tools.py` | The two tools the agents call (KB search + ticket lookup) |
| `crew.py` | The 3-agent crew + memory + schema + guardrails |
| `app.py` | FastAPI server: 2 HTTP endpoints + 1 WebSocket + logging + chat UI |
| `evaluate.py` | Scores the agent on 15 questions × 4 quality dimensions |
| `review.py` | A 2-agent Autogen team that reviews every answer before it ships |
| `governance.py` | Least-autonomy proof, risk class, budget cap, response cache |
| `transcripts/` | Saved run output from each script |

### Function-by-function

**`dataset.py`**
- `generate_tickets()` — builds the `SUPPORT_TICKETS` list from a fixed seed, so the
  data is identical every run.
- `check_support_ticket_status(record_id)` — the Lookup Agent's tool. Returns the
  ticket's status, resolution time, and a **designed escalation score**.

**`rag.py`**
- `fixed_size_chunks()` / `sentence_chunks()` — two ways to cut docs into pieces.
- `build_collections()` — embeds both chunk sets into two separate ChromaDB collections.
- `retrieve()` — returns the top-k closest chunks + their cosine similarity.
- `grounded_answer()` — answers using only retrieved text, or refuses if nothing is
  similar enough (the "I don't know" fallback).
- `calibrate_threshold()` — measures the refuse/answer cut-off from real data.
- `precision_recall()` — scores each chunking strategy at the document level.

**`mock_llm.py`**
- `MockLLM.call()` — the one method CrewAI calls. It reads the conversation and returns
  ReAct text (`Action:` to use a tool, `Final Answer:` to finish). Deterministic.

**`crew.py`**
- `run_crew()` — runs the 3-agent crew once (both tools fire).
- `answer()` — the full pipeline: guardrails → crew → validate → grounded refusal.
- `chat()` — same, but remembers earlier turns in a session.
- `mask_pii()`, `detect_injection()`, `is_grounded()` — the three guardrails.
- `CrewResponse` — the Pydantic schema every answer must match.

**`app.py`** — `POST /ask`, `POST /add-document`, `WS /ws/chat`, plus `log_request()`
which writes one masked JSON line per request to `logs/requests.jsonl`.

**`review.py`** — `review_draft()` runs the Autogen review team; `ReviewVerdict` is the
structured `{approved, final_answer, reason}` output.

**`governance.py`** — `prove_least_autonomy()`, `classify_risk()`,
`answer_within_budget()`, `CachedGroundedGen`.

---

## 3. How to run each part

```bash
python dataset.py        # ticket counts, escalated %, sample lookup
python rag.py            # threshold calibration, grounded demo, precision/recall
python crew.py           # crew tools fire, memory, schema, guardrails
python evaluate.py       # 15 queries scored on 4 dimensions
python review.py         # Autogen review approves one draft, revises another
python governance.py     # least autonomy, risk, budget cap, cache hit
uvicorn app:app          # open http://127.0.0.1:8000 for the chat UI, or POST /ask
```

Every script prints evidence and has a matching saved file in `transcripts/`.

---

## 4. Dataset design choices (reproducible)

- **Seed:** `42`. **Total tickets:** 45 (≥40 required).
- **Category weights:** Billing 30, Technical Issue 30, Account Access 18,
  Product Defect 12, General Inquiry 10 (relative weights, normalised by `random.choices`).
- **Status weights:** Open 20, In Progress 20, Escalated 10, Resolved 30, Closed 20.
- **`resolution_time_hours` range:** 1–72 hours (a quick reset up to a 3-day specialist fix).
- **`escalated` probability:** 0.20 → measured **13.3%** True, inside the required 10–30% band.
- **Escalation score formula:** `0.6 × escalated_flag + 0.4 × (days_since_created / 30)`,
  always in [0, 1]. Recommend escalation when score ≥ **0.5**. This is a weighted blend,
  not a boolean OR — a stale un-escalated ticket can still surface.

## 5. Measured grounding threshold (not a guessed default)

Measured top-1 cosine similarity on real queries:

| Query type | Similarity range |
|---|---|
| In-scope | 0.44 – 0.78 |
| Out-of-scope (e.g. "capital of France") | 0.04 – 0.05 |

There is a clear gap between ~0.05 and ~0.44. **`GROUNDING_THRESHOLD = 0.30`** sits inside
that gap — above every out-of-scope query, below every in-scope one. Chunking comparison:
fixed-size chunking had higher mean precision (0.57 vs 0.47) at equal recall (1.00), so the
crew uses the **fixed-size** collection.

## 6. How the MOCK_LLM avoids the two known traps

1. **The "Observation:" template trap.** CrewAI's own system prompt contains the example
   line `Observation: the result of the action`. The mock never keys on the bare word
   "Observation:"; tools tag real output with a unique sentinel (`⟪TOOLRESULT⟫`) and the
   mock looks for that, so the template example can't trigger a premature answer.
2. **The tool-dispatch trap.** The mock reads a tool's argument name from its declared
   `args_schema`, never from the tool's name.

## 7. Governance summary

- **Application layer — least autonomy:** only the Lookup Agent is given
  `check_support_ticket_status`; the Retrieval and Composer agents hold no such tool, and
  `governance.prove_least_autonomy()` verifies this against the live crew.
- **Risk classification:** **Medium** — customer support with masked PII, no medical,
  hiring, or financial decisions.
- **Runtime layer — budget cap:** requests over 256 estimated tokens are rejected before
  any work runs.
- **Caching:** identical queries hit an in-memory cache; the underlying
  grounded-generation call runs once, not twice.

---

## 8. Submission

Single public GitHub repository containing `dataset.py`, the `kb/` documents, the RAG /
crew / review / governance / deployment code, `transcripts/` with saved run output, and
this README. Everything runs under `MOCK_LLM` with zero API keys.
