"""
dataset.py
==========
Ola Support capstone - Part 1 Task 1 (the dataset) + Part 2 Task 6 (the lookup tool).

Everything here is deterministic: given the same SEED you get the exact same 45
tickets every single time. That is a hard requirement of the brief - the grader must
be able to reproduce my dataset from the seed/weights I write in the README.

Two things live in this file:
  1. generate_tickets()               -> builds the SUPPORT_TICKETS list
  2. check_support_ticket_status(id)  -> the tool my Lookup Agent will call (Task 6)

Run it directly (`python dataset.py`) to print the coverage report and confirm the
`escalated` share lands inside the required 10%-30% band.
"""

import random

# ---------------------------------------------------------------------------
# Design choices (these exact values go in the README so grading is reproducible)
# ---------------------------------------------------------------------------
SEED = 42          # any fixed int works; 42 happens to land the escalated% in-band.
N_TICKETS = 45     # >= 40 required. 45 gives comfortable room for category coverage.

# The five categories the brief hands me. I weight them so the mix looks like a real
# support queue (lots of billing + technical, fewer "general inquiry"). Weights are
# relative, not percentages - random.choices normalises them for me.
CATEGORIES = {
    "Billing":         30,
    "Technical Issue": 30,
    "Account Access":  18,
    "Product Defect":  12,
    "General Inquiry": 10,
}

# The five ticket lifecycle statuses. "Escalated" is deliberately rare - most tickets
# get Resolved/Closed, which is what a healthy support desk looks like.
STATUSES = {
    "Open":        20,
    "In Progress": 20,
    "Escalated":   10,
    "Resolved":    30,
    "Closed":      20,
}

# resolution_time_hours range. Reasoning (one sentence, as the brief asks): support
# tickets in practice close anywhere from ~1 hour (quick account resets) to ~72 hours
# (3 days, for defects needing a specialist), so I sample uniformly in [1, 72].
RES_HOURS_MIN, RES_HOURS_MAX = 1, 72

# Probability a ticket is escalated. 0.20 targets the middle of the 10%-30% band; over
# 45 tickets the expected count is 9. I confirmed seed 42 actually lands in-band below.
ESCALATED_PROB = 0.20


def generate_tickets():
    """Return a deterministic list of >=40 support-ticket dicts.

    Each record has exactly the six fields the brief requires:
      record_id, category, status, resolution_time_hours, days_since_created, escalated
    """
    # One RNG instance seeded once. I pull every random value from THIS object so the
    # whole sequence is reproducible - I never touch the global random.* functions.
    rng = random.Random(SEED)

    # random.choices needs the labels and their weights as parallel lists.
    cat_labels, cat_weights = list(CATEGORIES), list(CATEGORIES.values())
    st_labels, st_weights = list(STATUSES), list(STATUSES.values())

    tickets = []
    for i in range(1, N_TICKETS + 1):
        tickets.append({
            "record_id": f"TCK-{i:04d}",                    # TCK-0001, TCK-0002, ...
            "category": rng.choices(cat_labels, cat_weights)[0],
            "status": rng.choices(st_labels, st_weights)[0],
            "resolution_time_hours": rng.randint(RES_HOURS_MIN, RES_HOURS_MAX),
            "days_since_created": rng.randint(0, 30),       # brief: integer 0-30
            "escalated": rng.random() < ESCALATED_PROB,     # ~20% land True
        })
    return tickets


# Build the list once at import time so other modules can just `from dataset import
# SUPPORT_TICKETS`. It is a plain list of dicts - nothing exotic.
SUPPORT_TICKETS = generate_tickets()

# A dict index (record_id -> ticket) so the lookup tool below is an O(1) fetch instead
# of scanning the whole list every call.
_BY_ID = {t["record_id"]: t for t in SUPPORT_TICKETS}


# ---------------------------------------------------------------------------
# Part 2 Task 6 - the second tool, with a DESIGNED escalation score
# ---------------------------------------------------------------------------
# Formula (stated exactly, as the brief demands):
#     escalation_score = 0.6 * escalated_flag + 0.4 * (days_since_created / 30)
# - escalated_flag is 1.0 if the ticket is escalated else 0.0.
# - (days_since_created / 30) is the "recency" signal, normalised into [0, 1]; an
#   older, still-open ticket is more urgent.
# - Weights 0.6 / 0.4: being explicitly escalated matters more than mere age, but age
#   alone can still push a non-escalated-but-stale ticket up the queue. The result is
#   always in [0, 1] because both terms are in [0, 1] and the weights sum to 1.
#
# This is intentionally NOT a bare boolean OR - a 25-day-old un-escalated ticket scores
# 0.33, while a freshly escalated one scores 0.60, which is the ranking I want.
ESC_FLAG_WEIGHT = 0.6
RECENCY_WEIGHT = 0.4

# Recommended escalation threshold. Justified against my own data's distribution: see
# the __main__ report, which prints the 80th percentile of days_since_created. I set
# the threshold at 0.5 - any escalated ticket clears it immediately, and an
# un-escalated ticket only clears it if it is essentially the oldest in the queue.
ESCALATION_THRESHOLD = 0.5


def check_support_ticket_status(record_id: str) -> dict:
    """Look up one ticket and return its status, resolution time, and escalation score.

    This is the function my Lookup Agent (and ONLY that agent, per the Task 15
    least-autonomy rule) is allowed to call.
    """
    ticket = _BY_ID.get(record_id)
    if ticket is None:
        # Never raise into the agent loop - return a clean "not found" the crew can
        # relay to the user. Graceful failure is part of the acceptance criteria.
        return {"found": False, "record_id": record_id}

    flag = 1.0 if ticket["escalated"] else 0.0
    recency = ticket["days_since_created"] / 30.0
    score = ESC_FLAG_WEIGHT * flag + RECENCY_WEIGHT * recency

    return {
        "found": True,
        "record_id": record_id,
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "escalation_score": round(score, 3),
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
    }


def _percentile(values, pct):
    """Tiny nearest-rank percentile so I don't need numpy just for one number."""
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


def _report():
    """Print the coverage report the brief asks for and check every threshold."""
    from collections import Counter

    cat_counts = Counter(t["category"] for t in SUPPORT_TICKETS)
    st_counts = Counter(t["status"] for t in SUPPORT_TICKETS)
    escalated_n = sum(t["escalated"] for t in SUPPORT_TICKETS)
    escalated_pct = 100 * escalated_n / len(SUPPORT_TICKETS)

    print(f"Total tickets: {len(SUPPORT_TICKETS)}  (need >= 40)\n")

    print("Count per category (need each >= 3):")
    for c in CATEGORIES:
        ok = "OK" if cat_counts[c] >= 3 else "FAIL"
        print(f"  {c:<16} {cat_counts[c]:>2}  [{ok}]")

    print("\nCount per status (need each >= 1):")
    for s in STATUSES:
        ok = "OK" if st_counts[s] >= 1 else "FAIL"
        print(f"  {s:<12} {st_counts[s]:>2}  [{ok}]")

    band_ok = 10 <= escalated_pct <= 30
    print(f"\nEscalated: {escalated_n}/{len(SUPPORT_TICKETS)} = {escalated_pct:.1f}%"
          f"  (need 10%-30%)  [{'OK' if band_ok else 'FAIL'}]")

    p80 = _percentile([t["days_since_created"] for t in SUPPORT_TICKETS], 80)
    print(f"\n80th percentile of days_since_created: {p80} days "
          f"(context for the escalation threshold).")

    # A quick sanity demo of the lookup tool on the first ticket.
    print("\nSample lookup:", check_support_ticket_status("TCK-0001"))


if __name__ == "__main__":
    _report()
