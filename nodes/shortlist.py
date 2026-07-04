"""
RecruitFlow AI - Human-in-the-Loop Shortlist Workflow (Member 4)

All nodes here are pure deterministic Python - no LLM call decides
whether to approve/reject/modify a shortlist, per the brief:
"Do not use an LLM to append 'Shortlisted Candidate A' to a list."

Flow:
    prepare_shortlist_node  -> builds shortlist_candidates from
                                ranked_candidates + pending_action
    (graph pauses here via interrupt_before=["human_confirmation"])
    human_confirmation_node -> no-op passthrough, exists purely as the
                                interrupt point
    route_human_decision    -> conditional-edge function reading
                                human_decision
        "yes"    -> finalize_shortlist_node
        "no"     -> cancel_action_node
        "modify" -> modify_shortlist_node -> loops back to
                     human_confirmation_node for re-review
"""
from __future__ import annotations

import re

from state import RecruitmentState


def _parse_top_n(pending_action: str) -> int | None:
    if not pending_action:
        return None
    m = re.match(r"shortlist_top_(\d+)", pending_action)
    return int(m.group(1)) if m else None


def prepare_shortlist_node(state: RecruitmentState) -> dict:
    """Reads ranked_candidates + pending_action, writes shortlist_candidates."""
    ranked_candidates = state.get("ranked_candidates") or []
    pending_action = state.get("pending_action") or ""

    if not ranked_candidates:
        return {
            "error_message": "prepare_shortlist_node: no ranked_candidates in state - run screen_candidates first.",
            "agent_logs": ["prepare_shortlist_node: missing ranked_candidates"],
        }

    top_n = _parse_top_n(pending_action)
    if top_n is not None:
        selected = ranked_candidates[:top_n]
    else:
        # "finalize_candidates" / "shortlist_generic" - default to top 3
        selected = ranked_candidates[:3]

    shortlist_candidates = [
        {
            "candidate_name": c.get("candidate_name"),
            "resume_source": c.get("resume_source"),
            "match_score": c.get("match_score"),
        }
        for c in selected
    ]

    return {
        "shortlist_candidates": shortlist_candidates,
        "human_decision": "",
        "human_feedback": "",
        "agent_logs": [f"Prepared shortlist of {len(shortlist_candidates)} candidate(s), awaiting human confirmation"],
    }


def human_confirmation_node(state: RecruitmentState) -> dict:
    """No-op passthrough. The graph is compiled with
    interrupt_before=["human_confirmation"], so execution pauses BEFORE
    this node runs. main.py collects the recruiter's decision and calls
    graph.update_state(...) to write human_decision/human_feedback, then
    resumes the graph, which runs this node (a no-op) and moves on to
    the conditional edge below."""
    return {}


def route_human_decision(state: RecruitmentState) -> str:
    decision = (state.get("human_decision") or "").strip().lower()
    if decision == "yes":
        return "finalize"
    if decision == "no":
        return "cancel"
    if decision == "modify":
        return "modify"
    return "cancel"  # safe default if resumed with an unrecognized decision


def finalize_shortlist_node(state: RecruitmentState) -> dict:
    shortlist_candidates = state.get("shortlist_candidates") or []
    names = ", ".join(c.get("candidate_name", "?") for c in shortlist_candidates)
    finalized = list(state.get("finalized_actions") or [])
    finalized.append(f"Shortlisted: {names}")

    return {
        "finalized_actions": finalized,
        "pending_action": "",
        "agent_logs": [f"Shortlist finalized: {names}"],
    }


def cancel_action_node(state: RecruitmentState) -> dict:
    return {
        "pending_action": "",
        "shortlist_candidates": [],
        "agent_logs": ["Shortlist action cancelled by recruiter"],
    }


def modify_shortlist_node(state: RecruitmentState) -> dict:
    """Deterministic parse of human_feedback like:
    'Shortlist only Candidate A and Candidate C' or
    'Remove Priya, keep the rest' - re-filters shortlist_candidates
    against the full ranked_candidates pool by name matching, then
    routes back to human_confirmation for re-review. Does not call an
    LLM - the rubric explicitly wants this deterministic."""
    feedback = (state.get("human_feedback") or "").lower()
    ranked_candidates = state.get("ranked_candidates") or []
    current_shortlist = state.get("shortlist_candidates") or []

    if not feedback:
        return {
            "human_decision": "",
            "agent_logs": ["modify_shortlist_node: no feedback given, returning to confirmation unchanged"],
        }

    all_names = {c.get("candidate_name", "").lower(): c for c in ranked_candidates}
    mentioned = [name for name in all_names if name and name in feedback]

    if "remove" in feedback or "except" in feedback or "not " in feedback:
        keep = [
            c for c in current_shortlist
            if (c.get("candidate_name") or "").lower() not in mentioned
        ]
    elif mentioned:
        keep = [all_names[name] for name in mentioned]
        keep = [
            {
                "candidate_name": c.get("candidate_name"),
                "resume_source": c.get("resume_source"),
                "match_score": c.get("match_score"),
            }
            for c in keep
        ]
    else:
        keep = current_shortlist

    return {
        "shortlist_candidates": keep,
        "human_decision": "",
        "human_feedback": "",
        "agent_logs": [f"Shortlist modified per feedback -> {len(keep)} candidate(s), awaiting re-confirmation"],
    }
