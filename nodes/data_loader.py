"""
RecruitFlow AI - Member 1, Responsibility 1 & 3.

load_jd_node        -> reads a JD .txt file into state["jd_text"]
load_resumes_node    -> loads 10-20 resume .txt files and builds the
                        ChromaDB vector store used later by
                        screen_candidates_node
"""

from __future__ import annotations

import os
import sys

# Allow running this file directly (python nodes/data_loader.py) as well
# as importing it as part of the package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.state import RecruitmentState
from tools.resume_rag import (
    ResumeLoadError,
    VectorStoreError,
    build_resume_vectorstore,
    load_resume_files,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JD_PATH = os.path.join(PROJECT_ROOT, "data", "job_descriptions", "jd_sample.txt")
DEFAULT_RESUME_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "resumes")


def _resolve_jd_path(jd_path: str | None) -> str:
    if not jd_path:
        return DEFAULT_JD_PATH
    if os.path.isfile(jd_path):
        return jd_path

    # Member 1's standalone test used data/jd_sample.txt; the merged
    # project stores it under data/job_descriptions/jd_sample.txt.
    if os.path.basename(jd_path) == "jd_sample.txt" and os.path.isfile(DEFAULT_JD_PATH):
        return DEFAULT_JD_PATH
    return jd_path


def load_jd_node(state: RecruitmentState) -> dict:
    """Reads state['jd_path'], validates it, and writes state['jd_text'].

    The hackathon brief explicitly allows the JD to be a plain text file -
    no PDF parsing is required for this responsibility.
    """
    jd_path = _resolve_jd_path(state.get("jd_path"))

    if not os.path.isfile(jd_path):
        return {
            "error_message": f"JD file not found: '{jd_path}'.",
            "agent_logs": [f"load_jd_node: file not found ({jd_path})"],
        }

    try:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return {
            "error_message": f"Could not read JD file '{jd_path}': {e}",
            "agent_logs": ["load_jd_node: read failed"],
        }

    if not jd_text.strip():
        return {
            "error_message": f"JD file '{jd_path}' is empty.",
            "agent_logs": ["load_jd_node: empty JD file"],
        }

    return {
        "jd_path": jd_path,
        "jd_text": jd_text,
        "agent_logs": [f"Loaded JD from {os.path.basename(jd_path)}"],
    }


def load_resumes_node(state: RecruitmentState) -> dict:
    """Loads every resume .txt/.pdf file in state['resume_directory'], and
    builds/refreshes the persistent ChromaDB vector store used later for
    RAG screening.

    Writes:
      resume_count    - int, number of resumes loaded
      resumes_loaded  - list of lightweight dicts (filename, candidate_name,
                        char_count) - the full resume text stays inside the
                        vector store / on disk, not duplicated into shared
                        graph state
    """
    resume_directory = state.get("resume_directory") or DEFAULT_RESUME_DIRECTORY

    try:
        records = load_resume_files(resume_directory)
    except ResumeLoadError as e:
        return {
            "error_message": str(e),
            "resume_count": 0,
            "resumes_loaded": [],
            "agent_logs": [f"load_resumes_node: {e}"],
        }

    vectorstore_warning = ""
    try:
        build_resume_vectorstore(records, resume_directory=resume_directory)
    except VectorStoreError as e:
        vectorstore_warning = str(e)

    resumes_loaded = [
        {
            "filename": r.filename,
            "candidate_name": r.candidate_name,
            "char_count": len(r.text),
        }
        for r in records
    ]

    logs = [f"Loaded {len(records)} resumes from '{resume_directory}'"]
    if vectorstore_warning:
        logs.append(
            "load_resumes_node: vector store build failed; "
            f"continuing with deterministic screening fallback - {vectorstore_warning}"
        )
    else:
        logs[0] += " and indexed them into ChromaDB"

    return {
        "resume_directory": resume_directory,
        "resume_count": len(records),
        "resumes_loaded": resumes_loaded,
        "agent_logs": logs,
    }
