"""
RecruitFlow AI - Member 1 RAG layer.

Handles:
  - finding + loading resume .txt/.pdf files from a directory
  - splitting them into chunks (RecursiveCharacterTextSplitter, same as
    Day 4 - 06_vector_store_rag_chain.py)
  - embedding + storing them in a persistent ChromaDB collection
    (Chroma via langchain-chroma, embeddings via
    langchain-google-genai's GoogleGenerativeAIEmbeddings, same as the
    bootcamp notes)
  - a per-candidate retrieval helper used by screen_candidates_node

This module is intentionally free of LangGraph node code - `nodes/` and
`agents/` import from here so the RAG logic can be unit tested on its
own (see tests/test_member1.py).
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResumeLoadError(Exception):
    """Raised for problems reading the resume directory / files."""


class VectorStoreError(Exception):
    """Raised when embedding or ChromaDB operations fail."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResumeRecord:
    filename: str
    candidate_name: str
    text: str


@dataclass
class RetrievedChunk:
    text: str
    source: str
    candidate_name: str


# ---------------------------------------------------------------------------
# LLM / embedding model factories
# ---------------------------------------------------------------------------

_NAME_LINE_RE = re.compile(r"^\s*name\s*:\s*(.+)$", re.IGNORECASE)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Gemini embedding model, matching the bootcamp notes
    (models/gemini-embedding-001, GEMINI_KEY env var)."""
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise VectorStoreError(
            "GEMINI_KEY is not set in the environment. Add it to your .env file."
        )
    return GoogleGenerativeAIEmbeddings(
        api_key=api_key, model="models/gemini-embedding-001"
    )


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Groq LLM, matching the bootcamp notes (llama-3.3-70b-versatile,
    GROQ_KEY env var)."""
    api_key = os.getenv("GROQ_KEY")
    if not api_key:
        raise VectorStoreError(
            "GROQ_KEY is not set in the environment. Add it to your .env file."
        )
    return ChatGroq(
        api_key=api_key, model="openai/gpt-oss-120b", temperature=temperature
    )


# ---------------------------------------------------------------------------
# Resume loading
# ---------------------------------------------------------------------------


def _extract_candidate_name(text: str, fallback: str) -> str:
    """Pulls a 'Name: ...' line out of the resume text. Falls back to the
    filename (without extension) if no such line is found."""
    for line in text.splitlines()[:8]:
        m = _NAME_LINE_RE.match(line)
        if m:
            return m.group(1).strip()
    return fallback


def _read_pdf_text(path: str) -> str:
    try:
        import fitz
    except ModuleNotFoundError as e:
        raise ResumeLoadError(
            "PDF resume support requires PyMuPDF. Install it with `pip install pymupdf`."
        ) from e

    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        raise ResumeLoadError(f"Could not read PDF resume '{os.path.basename(path)}': {e}")


def _read_resume_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)
    if ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as e:
            raise ResumeLoadError(f"Could not read resume file '{filename}': {e}")
    if ext == ".pdf":
        return _read_pdf_text(path)
    raise ResumeLoadError(f"Unsupported resume file type '{ext}' for '{filename}'.")


def load_resume_files(resume_directory: str) -> List[ResumeRecord]:
    """Loads every .txt and .pdf resume in `resume_directory`.

    Raises ResumeLoadError with a clear message for:
      - missing directory
      - empty directory (no supported resume files)
      - a file that can't be read / decoded
      - a file that is empty after stripping whitespace
    """
    if not resume_directory or not os.path.isdir(resume_directory):
        raise ResumeLoadError(
            f"Resume directory not found: '{resume_directory}'."
        )

    file_paths = sorted(
        glob.glob(os.path.join(resume_directory, "*.txt"))
        + glob.glob(os.path.join(resume_directory, "*.pdf"))
    )
    if not file_paths:
        raise ResumeLoadError(
            f"No .txt or .pdf resume files found in '{resume_directory}'."
        )

    records: List[ResumeRecord] = []
    for path in file_paths:
        filename = os.path.basename(path)
        text = _read_resume_text(path)

        if not text.strip():
            raise ResumeLoadError(f"Resume file '{filename}' is empty.")

        candidate_name = _extract_candidate_name(
            text, fallback=os.path.splitext(filename)[0]
        )
        records.append(
            ResumeRecord(filename=filename, candidate_name=candidate_name, text=text)
        )

    return records


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

# Simple in-process cache so repeated graph invocations / node calls within
# the same run don't re-embed the same resume directory every time.
_VECTORSTORE_CACHE: dict[str, Chroma] = {}


def _collection_name_for(resume_directory: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.abspath(resume_directory))
    return f"resumes_{safe[-50:]}"


def build_resume_vectorstore(
    records: List[ResumeRecord],
    resume_directory: str,
    persist_directory: str = "./chroma_store",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    force_rebuild: bool = False,
) -> Chroma:
    """Splits resumes into chunks and builds/loads a persistent ChromaDB
    collection. Each chunk carries metadata (source filename + candidate
    name) so retrieved results can always be traced back to a candidate.
    """
    cache_key = os.path.abspath(persist_directory) + "::" + _collection_name_for(
        resume_directory
    )
    if not force_rebuild and cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    try:
        embedding_model = get_embedding_model()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        texts: List[str] = []
        metadatas: List[dict] = []
        for record in records:
            chunks = splitter.split_text(record.text)
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": record.filename,
                        "candidate_name": record.candidate_name,
                    }
                )

        if not texts:
            raise VectorStoreError("No resume text produced any chunks to embed.")

        os.makedirs(persist_directory, exist_ok=True)
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas,
            collection_name=_collection_name_for(resume_directory),
            persist_directory=persist_directory,
        )
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Failed to build the resume vector store: {e}")

    _VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore


def get_candidate_chunks(
    vectorstore: Chroma,
    query: str,
    source_filename: str,
    k: int = 3,
) -> List[RetrievedChunk]:
    """Retrieves the top-k chunks *for one specific candidate* that are
    most relevant to `query` (typically the JD's role + skills +
    experience). Filtering by source is what makes this real retrieval
    rather than decoration: every candidate gets scored against the JD's
    most relevant evidence from their own resume, not a random slice of it.
    """
    try:
        results = vectorstore.similarity_search(
            query, k=k, filter={"source": source_filename}
        )
    except Exception as e:
        raise VectorStoreError(
            f"Retrieval failed for '{source_filename}': {e}"
        )

    return [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", source_filename),
            candidate_name=doc.metadata.get("candidate_name", ""),
        )
        for doc in results
    ]
