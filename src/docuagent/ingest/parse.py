"""Document parsing: file -> page-tagged text blocks.

Phase 1 covers text PDFs only (PyMuPDF). Docling/unstructured for complex
layouts, tables, and scanned documents lands later — see Blueprint §5.1.
"""

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class TextBlock:
    text: str
    page: int
    section: str | None = None


def parse_pdf(file_path: str) -> list[TextBlock]:
    doc = fitz.open(file_path)
    blocks: list[TextBlock] = []
    try:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                blocks.append(TextBlock(text=text, page=page_index))
    finally:
        doc.close()
    return blocks


def parse_document(file_path: str, mime_type: str) -> list[TextBlock]:
    if mime_type == "application/pdf":
        return parse_pdf(file_path)
    raise NotImplementedError(
        f"Parsing for mime type '{mime_type}' isn't wired yet. Phase 1 covers "
        "text PDFs only — DOCX/HTML/Markdown and scanned-PDF layouts land in a "
        "later ingestion pass (Docling/unstructured)."
    )
