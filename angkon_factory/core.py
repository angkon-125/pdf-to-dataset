"""PDF extraction, document profiling, and provenance-preserving chunking."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

MAX_PDF_BYTES = 60 * 1024 * 1024
MAX_PAGES = 1500


def clean_text(value: str) -> str:
    """Clean extraction artifacts without changing the substantive source text."""
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def detect_language(text: str) -> str:
    """Simple script-based heuristic; not a trained language detector."""
    sample = text[:100_000]
    bn = sum(1 for ch in sample if "\u0980" <= ch <= "\u09ff")
    en = sum(1 for ch in sample if "a" <= ch.lower() <= "z")
    if bn >= 10 and en >= 10 and min(bn, en) / max(bn, en) >= 0.04:
        return "Bangla-English mixed"
    if bn >= 10 and bn > en * 0.06:
        return "Bangla"
    if en >= 10:
        return "English"
    return "Unknown / other"


def detect_document_type(text: str) -> tuple[str, list[str]]:
    """Conservative keyword signals: useful guesses, not guaranteed labels."""
    sample = text[:40_000].lower()
    patterns = {
        "research_paper": [r"\babstract\b", r"\bmethodology\b", r"\bresults? and discussion\b", r"\breferences\b", r"\bdoi\s*:\s*"],
        "technical_manual": [r"\binstallation\b", r"\btroubleshooting\b", r"\buser guide\b", r"\boperating instructions\b", r"\bsafety instructions\b"],
        "business_report": [r"\bexecutive summary\b", r"\bkey findings\b", r"\bquarterly results\b", r"\bfinancial statements\b", r"\bmarket analysis\b"],
        "book_or_textbook": [r"\bchapter [0-9ivx]+\b", r"\btable of contents\b", r"\bcontents\b", r"\bchapter one\b", r"\bexercises\b"],
    }
    matches = {kind: [pat for pat in pats if re.search(pat, sample)] for kind, pats in patterns.items()}
    kind = max(matches, key=lambda k: len(matches[k]))
    if len(matches[kind]) < 2:
        if len(re.findall(r"\bchapter\s+[0-9ivx]+\b", sample)) >= 2:
            return "book_or_textbook", ["repeated chapter headings"]
        return "general_document", []
    return kind, matches[kind]


def extract_pdf(pdf_bytes: bytes, filename: str, *, use_ocr: bool = False,
                ocr_language: str = "eng", page_limit: int = MAX_PAGES) -> dict[str, Any]:
    """Return page text, source metadata, and explicit extraction warnings."""
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Upload a valid PDF file.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 60 MB local MVP limit.")
    if not 1 <= page_limit <= MAX_PAGES:
        raise ValueError(f"page_limit must be between 1 and {MAX_PAGES}.")
    source_id = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc
    warnings: list[str] = []
    pages: list[dict[str, Any]] = []
    try:
        if doc.needs_pass:
            raise ValueError("Password-protected PDFs must be unlocked by the user before upload.")
        total_pages = len(doc)
        limit = min(total_pages, page_limit)
        if total_pages > page_limit:
            warnings.append(f"Only first {limit} of {total_pages} pages processed; increase page limit for full coverage.")
        for page_index in range(limit):
            page = doc[page_index]
            extracted = clean_text(page.get_text("text", sort=True))
            extraction = "digital_text"
            image_count = len(page.get_images(full=True))
            if len(extracted) < 35:
                if use_ocr:
                    try:
                        # Requires system Tesseract and relevant language packs.
                        ocr_page = page.get_textpage_ocr(language=ocr_language, dpi=170, full=True)
                        ocr_text = clean_text(page.get_text("text", textpage=ocr_page, sort=True))
                        if len(ocr_text) > len(extracted):
                            extracted = ocr_text
                            extraction = "ocr"
                        else:
                            warnings.append(f"Page {page_index+1}: OCR found no additional text.")
                    except Exception as exc:
                        warnings.append(f"Page {page_index+1}: OCR unavailable or failed ({type(exc).__name__}: {exc}).")
                else:
                    warnings.append(f"Page {page_index+1}: little extractable text; enable OCR if scanned.")
            pages.append({
                "page": page_index + 1,
                "text": extracted,
                "characters": len(extracted),
                "extraction_method": extraction,
                "embedded_images": image_count,
            })
    finally:
        doc.close()
    all_text = "\n\n".join(x["text"] for x in pages if x["text"])
    doc_type, signals = detect_document_type(all_text)
    if not all_text:
        warnings.append("No text extracted. Check OCR availability and the PDF format.")
    if any(p["embedded_images"] for p in pages):
        warnings.append("Images/diagrams are not interpreted by this text-only MVP; verify content coverage.")
    if any(p["extraction_method"] == "digital_text" for p in pages):
        warnings.append("Multi-column reading order, tables, and complex layouts may require visual verification.")
    return {
        "source_id": source_id,
        "filename": Path(filename).name,
        "file_bytes": len(pdf_bytes),
        "page_count": total_pages,
        "processed_pages": len(pages),
        "language_heuristic": detect_language(all_text),
        "document_type_heuristic": doc_type,
        "type_signals": signals,
        "scanned_pages_ocr": sum(x["extraction_method"] == "ocr" for x in pages),
        "extracted_characters": len(all_text),
        "pages": pages,
        "warnings": warnings,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def split_text(text: str, target_chars: int = 2300) -> list[str]:
    """Split without overlaps, so excerpts remain easily attributable to a page."""
    if target_chars < 300:
        raise ValueError("target_chars must be >= 300")
    text = clean_text(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        # Long paragraphs split at spaces, retaining all text.
        while len(paragraph) > target_chars:
            cut = paragraph.rfind(" ", 0, target_chars + 1)
            if cut < max(100, target_chars // 3):
                cut = target_chars
            first, paragraph = paragraph[:cut].strip(), paragraph[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            if first:
                chunks.append(first)
        if paragraph:
            candidate = current + ("\n\n" if current else "") + paragraph
            if len(candidate) > target_chars and current:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def make_chunks(document: dict[str, Any], target_chars: int = 2300) -> list[dict[str, Any]]:
    result = []
    for p in document["pages"]:
        for item_index, body in enumerate(split_text(p["text"], target_chars), start=1):
            chunk_id = f"{document['source_id'][:12]}-p{p['page']:04d}-c{item_index:03d}"
            result.append({
                "chunk_id": chunk_id,
                "source_id": document["source_id"],
                "source_file": document["filename"],
                "source_page": p["page"],
                "language": detect_language(body),
                "document_type": (document.get("llm_profile") or {}).get("document_type") or document["document_type_heuristic"],
                "extraction_method": p["extraction_method"],
                "text": body,
                "characters": len(body),
            })
    return result
