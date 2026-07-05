# RecruitFlow AI

Conversational terminal HR assistant for the Agentic AI Bootcamp Hackathon
recruitment chatbot problem.

## Merged Modules

This folder is the final merged project .

| Member | Integrated files | Role |
|---|---|---|
| Member 1 | `nodes/data_loader.py`, `agents/jd_parser.py`, `agents/screening_agent.py`, `tools/resume_rag.py`, `models/schemas.py`, sample JD/resumes | JD loading/parsing, resume loading, RAG screening |
| Member 2 | `router.py`, `router_schema.py` | Query classification, routing, applicant counting |
| Member 3 | `agents/rewrite_jd.py`, `agents/interview_agent.py`, `agents/salary_agent.py`, `agents/member3_schemas.py`, `agents/llm_client.py`, `tools/salary_search.py`, `data/salary_fallback.json` | JD rewrite, interview questions, Tavily salary research |
| Member 4 | `graph.py`, `main.py`, `nodes/shortlist.py`, `nodes/help_node.py`, `nodes/fallback.py`, `state.py` | LangGraph integration, terminal UI, human confirmation |



## Setup

```bash
cd C:\Users\aniru\Downloads\sttp_hack\RecruitFlow-AI-final
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` for full AI behavior:

```env
GEMINI_KEY=your_gemini_api_key_here
GROQ_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Without keys, the app still runs using deterministic demo fallbacks.

## Add Resumes

Put resume files in:

```text
C:\Users\aniru\Downloads\sttp_hack\RecruitFlow-AI-final\data\resumes
```

Supported formats:

- `.txt`
- `.pdf`

After adding a PDF resume, restart the chatbot or reuse the same session and run:

```text
Here's the JD and resumes
How many applicants?
Get me top candidates
```

## Run

```bash
python main.py
python main.py --thread-id demo-1
python main.py --debug
```

Use the same `--thread-id` later to resume the same checkpointed recruiter
session.

Suggested chatbot demo:

```text
recruiter> Here's the JD and resumes
RecruitFlow AI:
JD loaded and parsed.

recruiter> How many applicants?
RecruitFlow AI:
15 applicants loaded.

recruiter> Get me top candidates
RecruitFlow AI:
Top candidates (15 screened):

recruiter> Rewrite this JD for a startup
RecruitFlow AI:
Rewritten JD:

recruiter> Interview questions for Karthik
RecruitFlow AI:
Interview questions for Karthik Menon:

recruiter> Salary expectations for this role in India
RecruitFlow AI:
Salary research:

recruiter> Shortlist the top 3
RecruitFlow AI:
Proposed shortlist:
confirm shortlist? [yes/no/modify] > yes
```

## Verify

```bash
python -m compileall -q .
$env:PYTHONPATH='.'; python tests\test_router.py
$env:PYTHONPATH='.'; python tests\test_member1.py
```
