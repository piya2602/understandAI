"""Tests for PDF text extraction utilities (generates real PDFs on the fly)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io

import pytest

from app.pdf_utils import extract_pdf_text, is_pdf, MAX_PDF_PAGES


def make_pdf(pages_text):
    """Build an in-memory PDF with one page per string using pdfplumber's engine."""
    from reportlab.pdfgen import canvas  # optional dev dependency

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    for text in pages_text:
        y = 800
        for line in text.splitlines():
            pdf.drawString(72, y, line)
            y -= 14
        pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf


try:
    import reportlab  # noqa: F401
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def test_is_pdf_detects_magic_bytes():
    import io
    assert is_pdf(io.BytesIO(b"%PDF-1.7 rest"))
    assert not is_pdf(io.BytesIO(b"hello world"))


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
def test_extract_pdf_text_basic():
    pdf = make_pdf(["The university announced examinations on October 15.",
                    "Registration closes on October 5."])
    text = extract_pdf_text(pdf)
    assert "Page 1" in text
    assert "Page 2" in text
    assert "October 15" in text
    assert "October 5" in text


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
def test_extract_pdf_scanned_raises_friendly_error():
    # An image-only PDF: pdfplumber finds no selectable text.
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    pdf.showPage()  # blank page -> no text
    pdf.save()
    buf.seek(0)

    with pytest.raises(ValueError, match="scanned"):
        extract_pdf_text(buf)


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
def test_extract_pdf_too_many_pages():
    pdf = make_pdf(["page"] * (MAX_PDF_PAGES + 1))
    with pytest.raises(ValueError, match="too large"):
        extract_pdf_text(pdf)


def test_extract_pdf_garbage_file_raises_friendly_error():
    import io
    with pytest.raises(ValueError, match="Could not read"):
        extract_pdf_text(io.BytesIO(b"%PDF-1.4 this is not really a pdf"))
