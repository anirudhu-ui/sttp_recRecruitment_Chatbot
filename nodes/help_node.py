"""RecruitFlow AI - Help node (Member 4). Deterministic, no LLM call."""
from state import RecruitmentState

HELP_TEXT = """Try one of these:
  - Here's the JD and resumes         (load JD + resumes)
  - How many applicants?              (plain Python count)
  - Get me top candidates             (RAG screening with match scores)
  - Rewrite this JD for a startup     (grounded JD rewrite)
  - Interview questions for <name>    (candidate-grounded questions)
  - Salary expectations for this role?(Tavily web search)
  - Shortlist the top 3               (human-confirmed action)
Type 'exit' to quit."""


def help_node(state: RecruitmentState) -> dict:
    return {"agent_logs": ["Help displayed (no LLM call)"]}
