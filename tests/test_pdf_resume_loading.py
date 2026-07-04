"""
PDF resume loading smoke test.

Run from project root:
    python tests/test_pdf_resume_loading.py
"""
from __future__ import annotations

import os
import sys
import tempfile

from reportlab.pdfgen import canvas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.resume_rag import load_resume_files


def _write_pdf(path: str, lines: list[str]) -> None:
    c = canvas.Canvas(path)
    y = 760
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.save()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "resume_16_pdf_candidate.pdf")
        _write_pdf(
            pdf_path,
            [
                "Name: PDF Candidate",
                "Email: pdf.candidate@example.com",
                "Skills: Python, Django, REST APIs, SQL, Docker",
                "Experience: 3 years building backend services.",
            ],
        )

        records = load_resume_files(tmp)

    assert len(records) == 1, records
    assert records[0].filename == "resume_16_pdf_candidate.pdf"
    assert records[0].candidate_name == "PDF Candidate"
    assert "Python" in records[0].text
    assert "backend services" in records[0].text
    print("PASS - PDF resume loaded and text extracted")


if __name__ == "__main__":
    main()
