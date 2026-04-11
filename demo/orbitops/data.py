"""Content and derived state for the OrbitOps demo app."""

from __future__ import annotations

from typing import Any

BRAND_NAME = "OrbitOps"
TAGLINE = "Command every launch window without command-room chaos."

FEATURES = [
    {
        "title": "Launch rooms with context",
        "copy": (
            "Pull release notes, customer risk, and rollback status into one operating "
            "surface so product, support, and engineering stay aligned."
        ),
    },
    {
        "title": "Signal-led triage",
        "copy": (
            "Spot churn risk, incident drag, and rollout blockers before they become a "
            "war-room problem."
        ),
    },
    {
        "title": "Readable automation",
        "copy": (
            "Codify approval steps, launch checklists, and handoffs without burying the "
            "team in brittle YAML."
        ),
    },
]

TRUST_BADGES = [
    "Northstar Cloud",
    "Kepler Commerce",
    "Relay Health",
    "Vector Freight",
]

PLANS = [
    {
        "name": "Starter",
        "price": "$49",
        "suffix": "/workspace",
        "badge": "Ship faster",
        "description": "For lean product teams tightening up launch rituals.",
        "items": [
            "Up to 5 launch rooms",
            "Weekly trend snapshots",
            "Shared rollout checklist",
        ],
    },
    {
        "name": "Scale",
        "price": "$149",
        "suffix": "/workspace",
        "badge": "Most popular",
        "description": "For cross-functional teams coordinating releases across multiple pods.",
        "items": [
            "Unlimited launch rooms",
            "Live health score feed",
            "Custom approval paths",
        ],
    },
    {
        "name": "Command",
        "price": "$399",
        "suffix": "/workspace",
        "badge": "For operators",
        "description": "For revenue-critical launches that need precise coordination and recovery.",
        "items": [
            "Executive scorecards",
            "Risk and rescue segment tracking",
            "Priority support inbox",
        ],
    },
]

FAQS = [
    {
        "question": "Is OrbitOps a real product?",
        "answer": (
            "No. It is a realistic facsimile designed for dogfooding muscle-memory on "
            "landing-page edits, dashboard work, and repeatable verification loops."
        ),
    },
    {
        "question": "Why not use a toy todo app?",
        "answer": (
            "Because real teams need recurring tasks with enough texture to teach useful "
            "playbooks: copy updates, metric tuning, layout tweaks, and smoke checks."
        ),
    },
    {
        "question": "What should I repeat while dogfooding?",
        "answer": (
            "Run the local server, make a change, run the demo smoke checks, and verify the "
            "marketing page plus the dashboard. Those loops are where skills should emerge."
        ),
    },
]

METRICS = [
    {
        "label": "Live launch rooms",
        "value": "184",
        "delta": "+12%",
        "note": "Rooms running work this month",
    },
    {
        "label": "Net revenue retention",
        "value": "118%",
        "delta": "+4pts",
        "note": "Expansion outpaced contraction this month",
    },
    {
        "label": "Launches on track",
        "value": "31",
        "delta": "86%",
        "note": "Rooms that cleared every blocker within SLA",
    },
    {
        "label": "Median rescue time",
        "value": "42m",
        "delta": "-18%",
        "note": "Time from alert to staffed response",
    },
]

SEGMENTS = {
    "growth": {
        "label": "Growth accounts",
        "headline": "Healthy expansion momentum",
        "health": 92,
        "summary": (
            "Launches are landing cleanly, but the support team is asking for tighter "
            "handoff notes on late-Friday rollouts."
        ),
        "bullets": [
            "13 accounts are within 10 days of a visible expansion event",
            "Average approval lag dropped by 21 minutes week over week",
            "One launch room needs clearer post-launch ownership",
        ],
    },
    "enterprise": {
        "label": "Enterprise",
        "headline": "Great revenue, heavier coordination tax",
        "health": 81,
        "summary": (
            "Bigger deals are converting, but dependency mapping is getting dense and "
            "approval chains are still slower than the team wants."
        ),
        "bullets": [
            "Security reviews are the main source of schedule drag",
            "Executive check-ins correlate with cleaner launch retros",
            "Three rollout rooms need a rollback owner assigned earlier",
        ],
    },
    "rescue": {
        "label": "Rescue",
        "headline": "Fragile but recoverable",
        "health": 67,
        "summary": (
            "The team is responding quickly, though a few accounts are still bouncing "
            "between support and product without crisp escalation notes."
        ),
        "bullets": [
            "Two accounts need a product-led follow-up within 24 hours",
            "Slack alert fatigue is making incident ownership harder to spot",
            "Playbooks exist, but the recovery checklist needs one less approval hop",
        ],
    },
}

ACTIVITY = [
    {"team": "Launch Ops", "note": "Published rollback checklist v3", "time": "8m ago"},
    {"team": "Customer Success", "note": "Marked Kepler Commerce healthy", "time": "21m ago"},
    {"team": "Product", "note": "Shipped in-app migration banner", "time": "49m ago"},
    {"team": "Support", "note": "Escalated rescue workflow gap", "time": "1h ago"},
]

CHECKLIST = [
    "Confirm release owner and rollback owner are named",
    "Verify pricing and hero copy changes on the marketing page",
    "Run the local smoke checks before calling the demo done",
]


def dashboard_payload() -> dict[str, Any]:
    """Return JSON-ready state for the interactive dashboard."""
    average_health = round(sum(segment["health"] for segment in SEGMENTS.values()) / len(SEGMENTS))
    return {
        "brand": BRAND_NAME,
        "tagline": TAGLINE,
        "updated_at": "2026-04-11T09:30:00Z",
        "average_health": average_health,
        "default_segment": "growth",
        "metrics": METRICS,
        "segments": SEGMENTS,
        "activity": ACTIVITY,
        "checklist": CHECKLIST,
    }
