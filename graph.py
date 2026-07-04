"""
RecruitFlow AI - LangGraph Workflow (Member 4)
===============================================
Conversational routed agent, NOT a fixed pipeline. Every recruiter turn
enters at query_router; each query-type branch runs its own short
sub-flow and returns to the terminal loop, except shortlist_action,
which pauses for human confirmation via interrupt_before.

    START
      |
      v
  query_router --route_query-->
      +-- load_data          -> load_jd -> parse_jd -> load_resumes -> END
      +-- count_applicants   -> count_applicants -> END
      +-- screen_candidates  -> screen_candidates -> END
      +-- rewrite_jd         -> rewrite_jd -> END
      +-- interview_questions-> interview_questions -> END
      +-- salary_search      -> salary_search -> END
      +-- shortlist_action   -> prepare_shortlist -> [INTERRUPT]
      |                          human_confirmation --route_human_decision-->
      |                            +-- finalize -> finalize_shortlist -> END
      |                            +-- cancel   -> cancel_action -> END
      |                            +-- modify   -> modify_shortlist -> human_confirmation (loop)
      +-- help               -> help -> END
      +-- unknown            -> fallback -> END
"""
from langgraph.graph import StateGraph, START, END

from state import RecruitmentState

# Member 1
from nodes.data_loader import load_jd_node, load_resumes_node
from agents.jd_parser import parse_jd_node
from agents.screening_agent import screen_candidates_node

# Member 2
from router import query_router_node, route_query, count_applicants_node

# Member 3
from agents.rewrite_jd import rewrite_jd_node
from agents.interview_agent import interview_questions_node
from agents.salary_agent import salary_search_node

# Member 4
from nodes.shortlist import (
    prepare_shortlist_node,
    human_confirmation_node,
    route_human_decision,
    finalize_shortlist_node,
    cancel_action_node,
    modify_shortlist_node,
)
from nodes.help_node import help_node
from nodes.fallback import fallback_node


def build_graph(checkpointer):
    """Builds and compiles the full RecruitFlow graph against the given
    LangGraph checkpointer (e.g. an open SqliteSaver), so state persists
    across recruiter turns under the same thread_id."""
    builder = StateGraph(RecruitmentState)

    # --- nodes ---
    builder.add_node("query_router", query_router_node)

    builder.add_node("load_jd", load_jd_node)
    builder.add_node("parse_jd", parse_jd_node)
    builder.add_node("load_resumes", load_resumes_node)

    builder.add_node("count_applicants", count_applicants_node)
    builder.add_node("screen_candidates", screen_candidates_node)
    builder.add_node("rewrite_jd", rewrite_jd_node)
    builder.add_node("interview_questions", interview_questions_node)
    builder.add_node("salary_search", salary_search_node)

    builder.add_node("prepare_shortlist", prepare_shortlist_node)
    builder.add_node("human_confirmation", human_confirmation_node)
    builder.add_node("finalize_shortlist", finalize_shortlist_node)
    builder.add_node("cancel_action", cancel_action_node)
    builder.add_node("modify_shortlist", modify_shortlist_node)

    builder.add_node("help", help_node)
    builder.add_node("fallback", fallback_node)

    # --- entry ---
    builder.add_edge(START, "query_router")

    # --- router conditional edges ---
    builder.add_conditional_edges(
        "query_router",
        route_query,
        {
            "load_data": "load_jd",
            "count_applicants": "count_applicants",
            "screen_candidates": "screen_candidates",
            "rewrite_jd": "rewrite_jd",
            "interview_questions": "interview_questions",
            "salary_search": "salary_search",
            "shortlist_action": "prepare_shortlist",
            "help": "help",
            "unknown": "fallback",
        },
    )

    # --- load_data sub-flow ---
    builder.add_edge("load_jd", "parse_jd")
    builder.add_edge("parse_jd", "load_resumes")
    builder.add_edge("load_resumes", END)

    # --- single-node flows, return to turn end ---
    builder.add_edge("count_applicants", END)
    builder.add_edge("screen_candidates", END)
    builder.add_edge("rewrite_jd", END)
    builder.add_edge("interview_questions", END)
    builder.add_edge("salary_search", END)
    builder.add_edge("help", END)
    builder.add_edge("fallback", END)

    # --- human-in-the-loop shortlist flow ---
    builder.add_edge("prepare_shortlist", "human_confirmation")
    builder.add_conditional_edges(
        "human_confirmation",
        route_human_decision,
        {
            "finalize": "finalize_shortlist",
            "cancel": "cancel_action",
            "modify": "modify_shortlist",
        },
    )
    builder.add_edge("modify_shortlist", "human_confirmation")
    builder.add_edge("finalize_shortlist", END)
    builder.add_edge("cancel_action", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_confirmation"],
    )
