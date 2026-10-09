"""PDF text extraction using pdfplumber, with scanned-PDF and size handling.

Public entry points:
    extract_pdf_text(file)   -> str  (raises ValueError with a user-safe message)
    is_pdf(file)             -> bool
"""

from __future__ import annotations

from typing import IO

MAX_PDF_PAGES = 60
MAX_PDF_CHARS = 120_000

_PDF_MAGIC = b"%PDF-"


def is_pdf(file) -> bool:
    """Detect a PDF by magic bytes (works whether or not the stream is seekable)."""
    pos = file.tell()
    head = file.read(5)
    file.seek(pos)
    return head == _PDF_MAGIC


def extract_pdf_text(file) -> str:
    """Extract text from a file-like PDF object.

    Raises ValueError with a user-safe message for scanned PDFs, oversize
    documents, or unreadable files. Never leaks stack traces.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ValueError(
            "PDF support is not installed. Run: pip install pdfplumber"
        )

    pos = file.tell()
    try:
        with pdfplumber.open(file) as pdf:
            pages = pdf.pages
            if not pages:
                raise ValueError("The PDF appears to be empty.")

            if len(pages) > MAX_PDF_PAGES:
                raise ValueError(
                    f"The PDF is too large ({len(pages)} pages). "
                    f"Please upload a document of at most {MAX_PDF_PAGES} pages."
                )

            chunks: list[str] = []
            total = 0
            for i, page in enumerate(pages):
                text = page.extract_text() or ""
                chunks.append(f"[Page {i + 1}]\n{text}")
                total += len(text) + 12
                if total > MAX_PDF_CHARS:
                    raise ValueError(
                        f"The PDF text exceeds {MAX_PDF_CHARS // 1000}k characters. "
                        "Please upload a smaller document."
                    )

            full_text = "\n\n".join(chunks).strip()
            if not full_text or _mostly_blank(full_text):
                raise ValueError(
                    "No selectable text found in this PDF. It looks like a scanned "
                    "image — please paste the text or upload a text-based PDF."
                )
            return full_text

    except ValueError:
        raise  # user-safe messages pass through untouched
    except Exception as exc:  # broken/corrupt/encrypted file
        file.seek(pos)
        raise ValueError(
            "Could not read the PDF. The file may be corrupted or password-protected."
        ) from exc


def _mostly_blank(text: str) -> bool:
    """Detect scanned pages: almost nothing but page markers and whitespace."""
    stripped = text.replace("\x0c", "")
    visible = "".join(ch for ch in stripped if not ch.isspace())
    # Ignore our own [Page N] markers
    import re
    visible = re.sub(r"\[Page \d+\]", "", visible)
    return len(visible) < 40
