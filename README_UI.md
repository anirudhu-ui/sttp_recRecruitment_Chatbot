# RecruitFlow UI

A minimal web UI on top of the existing RecruitFlow AI chatbot (`graph.py`,
`main.py`). It doesn't change any agent/router/node logic — `app.py` is a
thin Flask wrapper that drives the same LangGraph the terminal chatbot uses,
sending it the same kind of `user_query` strings a recruiter would type
("load resumes", "screen the candidates", "interview questions for X"),
scoped to a per-browser-session `thread_id`.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_KEY, GOOGLE_API_KEY (and TAVILY_API_KEY if used)
python app.py
```

Open **http://127.0.0.1:5000**.

## Flow

1. **Upload** — paste/edit a job description (prefilled from
   `data/job_descriptions/jd_sample.txt`) and drop in one resume (`.pdf`/`.txt`).
   `POST /api/upload` just stages both files under `uploads/<thread_id>/`.
2. **Scan resume** — `POST /api/scan` sends `user_query: "load resumes"`,
   the same route the chatbot uses for "load the resumes". Runs
   `load_jd_node -> parse_jd_node -> load_resumes_node`, returns the parsed
   JD fields + detected candidate name.
3. **Get score** — `POST /api/score` sends `user_query: "screen the
   candidates"`. Runs `screen_candidates_node` (ChromaDB-grounded RAG +
   structured-output LLM scoring), returns `match_score`, matched/missing
   skills, and the reasoning.
4. **Get questions** — `POST /api/questions` sends `user_query: "interview
   questions for <first name>"`. The router's own regex only captures a
   single word after "for", and `interview_agent`'s candidate lookup does
   case-insensitive substring matching either direction, so the first name
   alone resolves correctly even for multi-word candidate names.

Each step reuses the same LangGraph `thread_id`/checkpoint, so state
(parsed JD, loaded resume, ranked candidate) persists across the three
calls exactly like it would across chatbot turns.

## Notes

- "Get score" is disabled until "Scan resume" succeeds (score needs
  `jd_role`/`jd_skills` in state); "Get questions" is disabled until
  "Get score" succeeds (questions need `ranked_candidates`). This mirrors
  a real constraint in the underlying graph, not an artificial UI limit.
- Only one resume per session is scored — the underlying tool supports a
  whole `resume_directory` of candidates, but this UI keeps to "upload one
  resume, get its scan/score/questions" per the brief.
- Static frontend lives in `static/` (`index.html`, `style.css`, `app.js`)
  — vanilla JS, no build step.
