"""
RecruitFlow AI - Terminal Chatbot (Member 4)
=============================================
Multi-turn recruiter conversation over the LangGraph workflow in
graph.py. Same thread_id used for the whole session so state (parsed
JD, loaded resumes, ranked candidates, etc.) persists across turns, per
the brief's Workflow Design requirement.

Run:
    python main.py
    python main.py --thread-id demo-1
    python main.py --debug     # prints full state dict after each turn
"""
import argparse
import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph
from nodes.help_node import HELP_TEXT
from nodes.fallback import FALLBACK_TEXT

THREAD_ID = "recruiter-session-1"
DB_PATH = "recruitflow_session.db"

BANNER = """\
=====================================================
  RecruitFlow AI - conversational recruitment chatbot
  Type `help` to see commands, `quit` to exit.
=====================================================
"""


def _format_turn_result(state: dict) -> str:
    """Returns only what's relevant to the query just handled - never the
    full state dict (unless --debug is on, handled by the caller)."""
    query_type = state.get("query_type")
    error = state.get("error_message")

    if error:
        return str(error)

    if query_type == "load_data":
        return "\n".join(
            [
                "JD loaded and parsed.",
                f"Role: {state.get('jd_role')}",
                f"Skills: {state.get('jd_skills')}",
                f"Experience: {state.get('jd_experience')}",
                f"Resumes loaded: {state.get('resume_count')}",
            ]
        )

    elif query_type == "count_applicants":
        return f"{state.get('resume_count')} applicants loaded."

    elif query_type == "screen_candidates":
        ranked = state.get("ranked_candidates") or []
        lines = [f"Top candidates ({len(ranked)} screened):"]
        for i, c in enumerate(ranked[:5], start=1):
            lines.append(f"{i}. {c.get('candidate_name')} - score {c.get('match_score')}")
            lines.append(f"   matched: {c.get('matched_skills')}")
            lines.append(f"   missing: {c.get('missing_skills')}")
        return "\n".join(lines)

    elif query_type == "rewrite_jd":
        return f"Rewritten JD:\n\n{state.get('rewritten_jd')}"

    elif query_type == "interview_questions":
        candidate = state.get("selected_candidate")
        questions_by_candidate = state.get("interview_questions") or {}
        matched_key = next(
            (k for k in questions_by_candidate if candidate and candidate.lower() in k.lower()),
            None,
        )
        questions = questions_by_candidate.get(matched_key) or {}
        lines = [f"Interview questions for {matched_key or candidate}:"]
        if isinstance(questions, dict):
            for category, category_questions in questions.items():
                label = category.replace("_", " ").title()
                lines.append(f"{label}:")
                for q in category_questions:
                    lines.append(f"  - {q}")
        else:
            for q in questions:
                lines.append(f"- {q}")
        return "\n".join(lines)

    elif query_type == "salary_search":
        lines = ["Salary research:", str(state.get("salary_summary"))]
        return "\n".join(lines)

    elif query_type == "help":
        return HELP_TEXT

    elif query_type == "unknown":
        logs = state.get("agent_logs") or []
        reason = next((l for l in logs if l.startswith("LLM router reason:")), None)
        if reason:
            return f"{FALLBACK_TEXT}\n\n[{reason}]"
        return FALLBACK_TEXT

    else:
        return f"Done ({query_type})."


def _print_bot(message: str) -> None:
    print("RecruitFlow AI:")
    print(message)


def _handle_shortlist_confirmation(graph, config, debug: bool) -> None:
    """Loop while the graph is paused before human_confirmation. Prints
    the pending shortlist, collects a yes/no/modify decision, resumes
    the graph, and repeats if the recruiter chose 'modify'."""
    while graph.get_state(config).next == ("human_confirmation",):
        current = graph.get_state(config).values
        shortlist = current.get("shortlist_candidates") or []

        print("\nRecruitFlow AI:")
        print("Proposed shortlist:")
        for i, c in enumerate(shortlist, start=1):
            print(f"  {i}. {c.get('candidate_name')} (score {c.get('match_score')})")

        decision = input("confirm shortlist? [yes/no/modify] > ").strip().lower()
        if decision in {"y", "confirm", "approve"}:
            decision = "yes"
        elif decision in {"n", "cancel"}:
            decision = "no"
        feedback = ""
        if decision == "modify":
            feedback = input("What should change? > ").strip()
        elif decision not in ("yes", "no"):
            _print_bot("Please answer yes, no, or modify.")
            continue

        graph.update_state(config, {"human_decision": decision, "human_feedback": feedback})
        result = graph.invoke(None, config=config)

        if debug:
            print("\n[DEBUG] state after resume:", result)

        if graph.get_state(config).next != ("human_confirmation",):
            finalized = result.get("finalized_actions") or []
            if decision == "no":
                _print_bot("Shortlist action cancelled.")
            elif finalized:
                _print_bot(finalized[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="RecruitFlow AI chatbot")
    parser.add_argument(
        "--thread-id",
        default=THREAD_ID,
        help="Resume a saved recruiter session by thread id.",
    )
    parser.add_argument("--debug", action="store_true", help="Print full graph state.")
    args = parser.parse_args()
    debug = args.debug
    print(BANNER)
    print(f"[session thread_id={args.thread_id} - pass --thread-id {args.thread_id} to resume]\n")

    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        while True:
            try:
                user_query = input("recruiter> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nRecruitFlow AI:")
                print("Goodbye.")
                break

            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                _print_bot("Goodbye.")
                break

            result = graph.invoke(
                {"user_query": user_query, "error_message": ""}, config=config
            )

            if graph.get_state(config).next == ("human_confirmation",):
                _handle_shortlist_confirmation(graph, config, debug)
            else:
                _print_bot(_format_turn_result(result))

            if debug:
                print("\n[DEBUG] full state:", result)

            print()


if __name__ == "__main__":
    main()
