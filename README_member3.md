# RecruitFlow AI — Member 3 Module
Role: Recruitment Generation and Live Market Research Engineer

Owns: JD Rewrite Agent, Candidate Interview Question Agent, Salary Research Agent,
Tavily salary search tool. Does NOT touch resume RAG, ranking, router, applicant
count, human confirmation, full graph wiring, or the terminal chatbot.

---

## 1. The three workflows

**JD Rewrite Agent** (`jd_rewrite_agent.py` → `rewrite_jd_node`)
Reads the structured JD (`jd_role`, `jd_skills`, `jd_experience`, `jd_text`) plus
`rewrite_tone` and optional `human_feedback`. Calls Groq LLM with
`with_structured_output(JDRewriteResult)` so the model can't return unstructured
text. System prompt hard-constrains it to preserve role purpose, critical skills,
and experience — it can only change tone/phrasing. Writes `rewritten_jd` +
`agent_logs`.

**Interview Question Agent** (`interview_question_agent.py` → `interview_questions_node`)
Looks up `selected_candidate` inside `ranked_candidates`, then calls the LLM with
`with_structured_output(InterviewQuestionSet)` to produce four grounded categories:
technical, candidate-specific, skill-gap, role-specific. Nothing generic — every
category is explicitly tied to JD skills or that candidate's resume record in the
prompt. Writes `interview_questions` + `agent_logs`.

**Salary Research Agent** (`salary_search_agent.py` → `salary_search_node` +
`salary_tool.py` → `salary_search_tool`)
Runs one or two live Tavily queries (role+location+experience, and a broader
fallback query without location if the narrow one is thin), filters out any result
without a real `http(s)` URL, then has the LLM summarize *only* what Tavily
returned — never inventing numbers. If Tavily is unavailable (no key, API error,
zero results), falls back to `data/salary_fallback.json`, and the summary is
prefixed `[CACHED FALLBACK - NOT LIVE DATA]` so it's never mistaken for live data.
Writes `salary_summary`, `salary_sources`, `agent_logs`.

## 2. Mapping to official MUST HAVEs

| Official requirement | My module |
|---|---|
| "Rewrite this JD" → LLM generation grounded in JD | `rewrite_jd_node` |
| "Interview questions for Candidate A" → grounded in JD skills + resume | `interview_questions_node` |
| "Salary expectations?" → Tavily web search, not RAG | `salary_search_node` + `salary_search_tool` |
| Modify works (rubric: Demo Quality) | Both `rewrite_jd_node` and `interview_questions_node` accept `human_feedback` and regenerate against it |
| Tools do work the LLM cannot (rubric: Tool Usage) | `salary_search_tool` hits live Tavily API; LLM never fabricates the figures |
| Cache 3-4 results as fallback JSON | `data/salary_fallback.json`, 4 entries, explicitly labeled cached |

## 3. Why salary uses Tavily, not RAG

RAG retrieves from a static, pre-loaded knowledge base (JD text, resumes) — it can
only ever be as current as when that data was indexed. Salary benchmarks change
continuously and live outside any file this project loads. The official brief is
explicit: `"Salary expectations?" -> web search via Tavily (not RAG — needs live
data)`. So this is the one place in Member 3's scope that reaches outside the
project's own documents onto the open web.

## 4. Tool permission isolation

Only `salary_search_agent.py` imports `salary_tool.py`. `jd_rewrite_agent.py` and
`interview_question_agent.py` import nothing but `state`, `schemas`, `llm_client`
— they have no code path that could reach Tavily even by accident. This isn't a
convention, it's structural: the tool object only exists in one module's import
graph.

## 5. Files created

```
member3/
├── state.py                        (copy of Member 2's shared state, for standalone use)
├── schemas.py                      (JDRewriteResult, InterviewQuestionSet)
├── llm_client.py                   (shared ChatGroq getter)
├── jd_rewrite_agent.py             (rewrite_jd_node)
├── interview_question_agent.py     (interview_questions_node)
├── salary_tool.py                  (salary_search_tool, TavilyNotConfigured)
├── salary_search_agent.py          (salary_search_node)
├── data/
│   └── salary_fallback.json
├── tests/
│   ├── test_jd_rewrite.py
│   ├── test_interview_questions.py
│   └── test_salary_search.py
├── .env.example
├── requirements.txt
└── README_member3.md
```

## 6. Code

See the files above — all complete, no pseudocode, no TODOs.

## 7. Fallback salary JSON

`data/salary_fallback.json` — 4 cached entries (AI Engineer, Software Engineer,
Data Scientist, ML Engineer), each tagged `cached_summary`/`cached_sources` and the
file carries a top-level `_source: "CACHED_FALLBACK - NOT LIVE DATA"` marker.

## 8. Pip install

```bash
pip install langgraph langchain-core langchain-groq tavily-python pydantic python-dotenv
```
(or `pip install -r requirements.txt` from inside `member3/`)

## 9. .env requirements

```
GROQ_KEY=your-groq-key-here
TAVILY_API_KEY=your-tavily-key-here
```
Both read via `os.getenv` + `python-dotenv`. No key is ever hardcoded.

## 10. Test commands

Run from inside the `member3/` folder:

```bash
python tests/test_jd_rewrite.py
python tests/test_interview_questions.py
python tests/test_salary_search.py
```

## 11. Expected output structures

**Test 1** — `rewrite_jd_node` returns:
```python
{
  "rewritten_jd": "<full startup-tone JD text, same role/skills/experience>",
  "agent_logs": ["rewrite_jd_node: rewrote JD (tone=startup, feedback_applied=False, preserved=[...])"]
}
```

**Test 2** — `interview_questions_node` returns:
```python
{
  "interview_questions": {
    "technical_questions": ["...", "..."],
    "candidate_specific_questions": ["...", "..."],
    "skill_gap_questions": ["...", "..."],
    "role_specific_questions": ["...", "..."]
  },
  "agent_logs": ["interview_questions_node: generated N questions for 'Candidate A' ..."]
}
```

**Test 3** — `salary_search_node` returns (live example):
```python
{
  "salary_summary": "<LLM summary grounded only in returned Tavily snippets, notes disagreement if any>",
  "salary_sources": ["https://...", "https://..."],
  "agent_logs": ["salary_search_node: used LIVE Tavily search (N results, role='AI Engineer')"]
}
```
or, if Tavily is unavailable:
```python
{
  "salary_summary": "[CACHED FALLBACK - NOT LIVE DATA] ...",
  "salary_sources": ["https://www.glassdoor.co.in/..."],
  "agent_logs": ["salary_search_node: TAVILY_API_KEY missing (...), trying fallback cache",
                 "salary_search_node: used FALLBACK CACHE (not live) for role='AI Engineer'"]
}
```

## 12. Files to send to Member 4 (LangGraph integration)

Send the whole `member3/` folder minus `tests/` and `.env.example` is optional
(Member 4 needs to know the two env vars exist, not the actual keys):

- `schemas.py`
- `llm_client.py`
- `jd_rewrite_agent.py`
- `interview_question_agent.py`
- `salary_tool.py`
- `salary_search_agent.py`
- `data/salary_fallback.json`
- the two required env var names: `GROQ_KEY`, `TAVILY_API_KEY`

Do NOT send my copy of `state.py` to overwrite Member 2's — it's included here
only so my module runs standalone. Member 4 should keep Member 2's `state.py` as
the single source of truth and just wire these three nodes into the graph as:

```python
graph.add_node("rewrite_jd", rewrite_jd_node)
graph.add_node("interview_questions", interview_questions_node)
graph.add_node("salary_search", salary_search_node)
```

## 13. Shared-state contract audit

Checked every key I read/write against the official contract list. All exact,
no renames:

| Key | I read | I write |
|---|---|---|
| `jd_text`, `jd_role`, `jd_skills`, `jd_experience` | ✅ (all 3 nodes) | — |
| `rewrite_tone` | ✅ | — |
| `human_feedback` | ✅ (2 nodes) | — |
| `rewritten_jd` | — | ✅ |
| `selected_candidate`, `ranked_candidates` | ✅ | — |
| `interview_questions` | — | ✅ |
| `salary_query` | ✅ | — |
| `salary_summary`, `salary_sources` | — | ✅ |
| `agent_logs` | — | ✅ (every node) |
| `error_message` | — | ✅ (on every handled failure) |

**One discrepancy found, flagged for Member 2/4:** `state.py` declares
`interview_questions: List[str]`, but the brief explicitly asks for a structured
Pydantic schema with multiple categories (technical/candidate-specific/skill-gap).
Python's `TypedDict` isn't enforced at runtime, so my node safely returns a `dict`
of 4 lists under that key — it works end-to-end — but the type hint in
`state.py` should be widened (e.g. `interview_questions: dict`) so it's accurate.
I did not edit `state.py` myself since I don't own it; flagging this for whoever
integrates the graph.

No other key mismatches. All other keys I touch match the contract exactly.
