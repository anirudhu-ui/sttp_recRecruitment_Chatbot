# 🚀 RecruitFlow AI
### AI-Powered Multi-Agent Recruitment Assistant using LangGraph

RecruitFlow AI is an intelligent recruitment assistant that automates the hiring workflow using multiple AI agents orchestrated through **LangGraph**. It enables recruiters to parse job descriptions, screen resumes, rank candidates, generate interview questions, rewrite job descriptions, estimate salary ranges, and manage shortlisting—all through a conversational interface.

---

## 📌 Features

- 📄 Job Description Parsing
- 📂 Resume Upload & Processing (.PDF / .TXT)
- 🤖 AI Resume Screening
- 🏆 Candidate Ranking
- 💼 Salary Estimation
- ✍️ AI Job Description Rewriter
- 🎯 Personalized Interview Question Generator
- ✅ Human-in-the-Loop Shortlisting
- 💬 Conversational Recruiter Chatbot
- 🌐 Flask Web Interface
- 🔄 Persistent Sessions using LangGraph Checkpointing

---

# 🏗️ System Architecture

```
Recruiter
      │
      ▼
 Query Router
      │
 ┌────┴───────────────────────────────────────┐
 │                                            │
 ▼                                            ▼
Load Data                              Other Queries
 │                                            │
 ▼                                            ▼
Parse JD                           Rewrite JD
 │                                 Salary Search
 ▼                                 Interview Questions
Load Resumes                        Applicant Count
 │                                 Help/Fallback
 ▼
Screen Candidates
 │
 ▼
Prepare Shortlist
 │
 ▼
Human Confirmation
 │
 ├──────────► Finalize
 │
 ├──────────► Modify
 │
 └──────────► Cancel
```

---

# 📂 Project Structure

```
RecruitFlow/
│
├── agents/
│   ├── interview_agent.py
│   ├── jd_parser.py
│   ├── rewrite_jd.py
│   ├── salary_agent.py
│   ├── screening_agent.py
│   └── llm_client.py
│
├── nodes/
│   ├── data_loader.py
│   ├── shortlist.py
│   ├── help_node.py
│   └── fallback.py
│
├── models/
│   ├── schemas.py
│   └── state.py
│
├── tools/
│   ├── resume_rag.py
│   └── salary_search.py
│
├── router.py
├── graph.py
├── main.py
├── app.py
├── requirements.txt
├── data/
│   ├── resumes/
│   ├── job_descriptions/
│   └── salary_fallback.json
└── uploads/
```

---

# 🤖 AI Agents

### 📄 JD Parser Agent

- Extracts
  - Job Role
  - Skills
  - Experience
  - Qualifications

---

### 👨‍💼 Resume Screening Agent

- Matches resumes with JD
- Skill Matching
- Missing Skills
- Candidate Score
- Candidate Ranking

---

### ✍️ JD Rewrite Agent

Rewrites job descriptions for:

- Startup
- Corporate
- Remote
- Internship
- Senior Roles

---

### 💰 Salary Agent

Provides salary estimation using

- Tavily Search
- AI Reasoning
- Local fallback database

---

### 🎤 Interview Agent

Generates interview questions categorized by

- Technical
- Problem Solving
- Behavioural
- Experience Based

---

## 🛠 Tech Stack

### AI

- LangGraph
- LangChain
- Google Gemini
- Groq LLM
- Tavily Search

### Backend

- Python
- Flask

### Vector Search

- ChromaDB

### PDF Processing

- PyMuPDF

### Data Validation

- Pydantic

### Environment

- Python Dotenv

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>

cd RecruitFlow
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file

```env
GEMINI_KEY=YOUR_GEMINI_KEY
GROQ_KEY=YOUR_GROQ_KEY
TAVILY_API_KEY=YOUR_TAVILY_API_KEY
```

If API keys are not provided, RecruitFlow falls back to deterministic demo responses wherever supported.

---

# 📁 Supported Resume Formats

- PDF
- TXT

Place resumes inside

```
data/resumes/
```

Place Job Description inside

```
data/job_descriptions/
```

---

# ▶ Running the Terminal Chatbot

```bash
python main.py
```

or

```bash
python main.py --thread-id recruiter-1
```

Debug mode

```bash
python main.py --debug
```
- IMPORTANT :- LOAD THE JD AND RESUMES(USE HELP TO KNOW COMMAND)
---

# 🌐 Running the Web Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 💬 Example Conversation

```
Recruiter:
Here's the JD and resumes

RecruitFlow AI:
JD loaded successfully.
15 resumes loaded.

Recruiter:
How many applicants?

RecruitFlow AI:
15 applicants found.

Recruiter:
Screen candidates

RecruitFlow AI:
Top Candidates

1. Karthik Menon
Score: 94%

2. Ananya Rao
Score: 91%

3. Sneha Patil
Score: 88%

Recruiter:
Generate interview questions for Karthik

RecruitFlow AI:
Technical Questions
Behavioral Questions
Problem Solving Questions

Recruiter:
Rewrite this JD for a startup

RecruitFlow AI:
Rewritten Job Description...

Recruiter:
Salary expectations in India

RecruitFlow AI:
Estimated Salary Range...
```

---

# 🔄 Workflow

1. Load Job Description
2. Parse Requirements
3. Load Resumes
4. Screen Candidates
5. Rank Applicants
6. Rewrite JD (Optional)
7. Generate Interview Questions
8. Estimate Salary
9. Human Approval for Shortlisting
10. Finalize Candidate List

---

# 📦 Key Dependencies

```
langgraph
langchain
langchain-google-genai
langchain-groq
langchain-chroma
langchain-tavily
chromadb
flask
pymupdf
pydantic
python-dotenv
reportlab
```

---

# 📈 Highlights

- Multi-Agent AI Architecture
- LangGraph Workflow Orchestration
- Human-in-the-Loop Decision Making
- Resume RAG Pipeline
- Persistent Conversation Memory
- AI-Assisted Recruitment Automation
- Modular & Extensible Design

---

# 🚀 Future Enhancements

- Email Candidate Notifications
- ATS Integration
- LinkedIn Resume Import
- Calendar Interview Scheduling
- HR Analytics Dashboard
- Multi-language Support
- Authentication & Role Management

---

# 👥 Contributors

- AMATHUL LUBNA
- ANIRUDH UPADHYAY
- D. ADHYUMNA CHOWDHARY
- MAMIDIPLALLY HANSIKA
Developed as part of the **STTP Agentic AI Recruitment Chatbot Project**.

Built using:

- LangGraph
- LangChain
- Gemini
- Groq
- Tavily
- Flask
- ChromaDB

---

# 📄 License

This project is intended for educational and research purposes. Feel free to modify and extend it according to your requirements.
