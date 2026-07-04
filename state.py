"""
RecruitFlow AI - Canonical Shared State Contract
=================================================
Owner: Member 4 (integration). Every node from every member reads/writes
ONLY these keys. This is the single source of truth merged from the
brief + Member 1's models/state.py + Member 2's state.py.

FIX (vs Member 1 / Member 2 copies): `agent_logs` is now
Annotated[list[str], operator.add]. Both member modules assumed logs
would ACCUMULATE across nodes in one turn (router log + downstream node
log both visible), but neither of their local TypedDicts declared a
reducer. Without it, LangGraph overwrites the key on every node write
and only the LAST node's log line survives. Declaring the reducer here
is what actually makes their existing `"agent_logs": [...]` return
values accumulate correctly - no node code had to change.

FIX (vs Member 1's models/state.py): `resumes_loaded` is typed
List[dict] here, not bool. Member 1's node actually writes
[{"filename", "candidate_name", "char_count"}, ...] (used by
screen_candidates_node as its candidate list), Member 2's copy typed it
`bool` but never reads or writes it. List[dict] matches the real
producer/consumer; keeping `bool` would just be a wrong type hint doing
nothing since TypedDict types aren't enforced at runtime.

`interview_questions` is a dict keyed by candidate name (per the
official brief's field list) so questions for multiple candidates can
be asked across turns without overwriting each other, e.g.:
    {"Priya Iyer": {"technical_questions": [...], "skill_gap_questions": [...]}}
"""
from typing import TypedDict, Annotated
from operator import add


class RecruitmentState(TypedDict, total=False):
    # --- routing / current turn (Member 2) ---
    user_query: str
    query_type: str

    # --- JD ingestion + structured parsing (Member 1) ---
    jd_path: str
    jd_text: str
    jd_role: str
    jd_skills: list[str]
    jd_experience: str

    # --- resume ingestion + RAG screening (Member 1) ---
    resume_directory: str
    resumes_loaded: list[dict]
    resume_count: int
    ranked_candidates: list[dict]
    selected_candidate: str

    # --- JD rewrite (Member 3) ---
    rewrite_tone: str
    rewritten_jd: str

    # --- interview questions (Member 3) ---
    interview_questions: dict

    # --- salary research (Member 3) ---
    salary_query: str
    salary_summary: str
    salary_sources: list[str]

    # --- human-in-the-loop shortlist (Member 4) ---
    pending_action: str
    shortlist_candidates: list[dict]
    human_decision: str
    human_feedback: str
    finalized_actions: list[str]

    # --- cross-cutting ---
    error_message: str
    agent_logs: Annotated[list[str], add]
