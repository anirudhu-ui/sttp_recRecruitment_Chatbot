"""
app.py — thin Flask backend that exposes the existing RecruitFlow AI
LangGraph (graph.py) through simple HTTP endpoints for a batch resume UI:
upload (up to 20 resumes at once, .txt/.pdf, persisted permanently),
scan, score (ranks the whole batch), and questions (per selected candidate).

It does NOT change any node/agent/router logic — it drives the same
conversational graph the terminal chatbot (main.py) uses, by sending it
the same kind of user_query strings a recruiter would type.

Storage model:
  uploads/library/          - PERMANENT shared resume pool. Every resume
                               ever uploaded lands here and stays here
                               across restarts/sessions - nothing is
                               deleted after a scan. This is the "keep
                               storage once scanned" behavior: come back
                               later and the same resumes are still there,
                               no re-upload needed.
  uploads/sessions/<tid>/   - just the JD text for that browser session
                               (jd.txt). Kept separate from the resume
                               library so two visitors can score the same
                               shared resume pool against different JDs.

Run:
    pip install flask
    python app.py
    -> open http://127.0.0.1:5000
"""

from __future__ import annotations
import os
import io
import uuid
import glob
import hashlib
import sqlite3
import logging

from flask import Flask, request, jsonify, send_from_directory
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(PROJECT_ROOT, "uploads")
LIBRARY_DIR = os.path.join(UPLOAD_ROOT, "library")          # permanent, shared
SESSIONS_DIR = os.path.join(UPLOAD_ROOT, "sessions")         # per-thread JD only
DEFAULT_JD_PATH = os.path.join(PROJECT_ROOT, "data", "job_descriptions", "jd_sample.txt")
DB_PATH = os.path.join(PROJECT_ROOT, "recruitflow_ui_session.db")

MAX_LIBRARY_SIZE = 20          # cap on total resumes kept in the shared pool
ALLOWED_EXTENSIONS = (".pdf", ".txt")

os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")

# One shared checkpointer + compiled graph for the app's lifetime.
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_graph = build_graph(_checkpointer)


def _config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _session_dir(thread_id: str) -> str:
    path = os.path.join(SESSIONS_DIR, thread_id)
    os.makedirs(path, exist_ok=True)
    return path


def _library_files() -> list[str]:
    return sorted(
        glob.glob(os.path.join(LIBRARY_DIR, "*.txt"))
        + glob.glob(os.path.join(LIBRARY_DIR, "*.pdf"))
    )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _ingest_into_library(file_storage) -> tuple[str | None, str]:
    """
    Stores one uploaded file into the permanent library, deduplicated by
    content hash so re-uploading the exact same resume is a no-op rather
    than a growing pile of copies.

    Returns (stored_filename_or_None, status) where status is one of
    "added", "duplicate", "renamed".
    """
    data = file_storage.read()
    file_storage.seek(0)
    content_hash = _hash_bytes(data)

    for existing_path in _library_files():
        with open(existing_path, "rb") as f:
            if _hash_bytes(f.read()) == content_hash:
                return os.path.basename(existing_path), "duplicate"

    base_name = os.path.basename(file_storage.filename)
    name, ext = os.path.splitext(base_name)
    candidate_name = base_name
    suffix = 1
    status = "added"
    while os.path.exists(os.path.join(LIBRARY_DIR, candidate_name)):
        suffix += 1
        candidate_name = f"{name}_{suffix}{ext}"
        status = "renamed"

    dest_path = os.path.join(LIBRARY_DIR, candidate_name)
    with open(dest_path, "wb") as f:
        f.write(data)

    return candidate_name, status


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/default-jd")
def default_jd():
    try:
        with open(DEFAULT_JD_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = ""
    return jsonify({"jd_text": text})


@app.get("/api/library")
def library():
    """Lists everything already in the persistent resume pool, so a
    returning visitor sees what's stored without re-uploading anything."""
    files = _library_files()
    items = [
        {"filename": os.path.basename(p), "size_kb": round(os.path.getsize(p) / 1024, 1)}
        for p in files
    ]
    return jsonify({"resumes": items, "count": len(items), "max": MAX_LIBRARY_SIZE})


@app.post("/api/library/delete")
def library_delete():
    """Deletes a specific resume from the permanent library."""
    data = request.get_json(force=True)
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
        
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
        
    file_path = os.path.join(LIBRARY_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File not found"}), 404


@app.post("/api/upload")
def upload():
    """
    Accepts up to MAX_LIBRARY_SIZE resume files in one request (field
    name 'resumes', multiple) + optional JD text override. Resumes are
    added to the PERMANENT shared library (deduplicated by content hash);
    the JD text is staged per browser-session thread_id.
    """
    thread_id = request.form.get("thread_id") or str(uuid.uuid4())
    resume_files = request.files.getlist("resumes")
    jd_text = request.form.get("jd_text", "")

    if not resume_files:
        return jsonify({"error": "No resume files provided."}), 400

    existing_count = len(_library_files())
    if existing_count + len(resume_files) > MAX_LIBRARY_SIZE:
        return jsonify({
            "error": (
                f"Library already has {existing_count} resume(s); uploading "
                f"{len(resume_files)} more would exceed the {MAX_LIBRARY_SIZE}-resume cap."
            )
        }), 400

    added, duplicates, renamed, rejected = [], [], [], []
    for f in resume_files:
        if not f or f.filename == "":
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(f.filename)
            continue
        stored_name, status = _ingest_into_library(f)
        if status == "added":
            added.append(stored_name)
        elif status == "duplicate":
            duplicates.append(stored_name)
        elif status == "renamed":
            renamed.append(stored_name)

    session_dir = _session_dir(thread_id)
    jd_path = os.path.join(session_dir, "jd.txt")
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write(jd_text.strip() or open(DEFAULT_JD_PATH, encoding="utf-8").read())

    return jsonify({
        "thread_id": thread_id,
        "added": added,
        "duplicates": duplicates,
        "renamed": renamed,
        "rejected": rejected,
        "library_count": len(_library_files()),
    })


@app.post("/api/scan")
def scan():
    """Runs the chatbot's 'load_data' route over the WHOLE shared library
    (every resume ever uploaded), scoped to this session's JD."""
    data = request.get_json(force=True)
    thread_id = data.get("thread_id")
    if not thread_id:
        return jsonify({"error": "Missing thread_id — upload resumes first."}), 400

    if not _library_files():
        return jsonify({"error": "Resume library is empty — upload at least one resume."}), 400

    jd_path = os.path.join(_session_dir(thread_id), "jd.txt")

    result = _graph.invoke(
        {
            "user_query": "load resumes",
            "jd_path": jd_path,
            "resume_directory": LIBRARY_DIR,
            "error_message": "",
        },
        config=_config_for(thread_id),
    )

    if result.get("error_message"):
        return jsonify({"error": result["error_message"]}), 500

    resumes_loaded = result.get("resumes_loaded") or []

    return jsonify({
        "candidates": [
            {"filename": r["filename"], "candidate_name": r["candidate_name"]}
            for r in resumes_loaded
        ],
        "jd_role": result.get("jd_role"),
        "jd_skills": result.get("jd_skills"),
        "jd_experience": result.get("jd_experience"),
        "resume_count": result.get("resume_count"),
    })


@app.post("/api/score")
def score():
    """Runs the chatbot's 'screen_candidates' route and returns the FULL
    ranked list — every resume currently in the library, not just one."""
    data = request.get_json(force=True)
    thread_id = data.get("thread_id")
    if not thread_id:
        return jsonify({"error": "Missing thread_id — scan the library first."}), 400

    result = _graph.invoke(
        {"user_query": "screen the candidates", "error_message": ""},
        config=_config_for(thread_id),
    )

    if result.get("error_message"):
        return jsonify({"error": result["error_message"]}), 500

    ranked = result.get("ranked_candidates") or []
    if not ranked:
        return jsonify({"error": "No candidates were scored."}), 500

    return jsonify({
        "ranked_candidates": [
            {
                "candidate_name": c.get("candidate_name"),
                "match_score": c.get("match_score"),
                "matched_skills": c.get("matched_skills"),
                "missing_skills": c.get("missing_skills"),
                "match_reason": c.get("match_reason"),
            }
            for c in ranked
        ]
    })


@app.post("/api/questions")
def questions():
    """Runs the chatbot's 'interview_questions' route for ONE selected
    candidate out of the ranked batch."""
    data = request.get_json(force=True)
    thread_id = data.get("thread_id")
    candidate_name = data.get("candidate_name")
    if not thread_id or not candidate_name:
        return jsonify({"error": "Missing thread_id or candidate_name — score the batch first."}), 400

    # The router's regex only captures a single word after "for"; the
    # interview_agent's candidate lookup does case-insensitive substring
    # matching either direction, so the first name alone still resolves
    # to the right candidate.
    first_name = candidate_name.split()[0]
    query = f"interview questions for {first_name}"

    result = _graph.invoke(
        {"user_query": query, "error_message": ""},
        config=_config_for(thread_id),
    )

    if result.get("error_message"):
        return jsonify({"error": result["error_message"]}), 500

    questions_by_candidate = result.get("interview_questions") or {}
    matched_key = next(
        (k for k in questions_by_candidate if candidate_name.lower() in k.lower()
         or k.lower() in candidate_name.lower()),
        None,
    )
    question_set = questions_by_candidate.get(matched_key) or {}

    return jsonify({"candidate_name": matched_key or candidate_name, "questions": question_set})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
