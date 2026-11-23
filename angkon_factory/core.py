"""Source-preserving PDF extraction, optional OCR and tables, and page-local chunks.

Automatic text extraction is not equivalent to understanding a document's images,
layout, or conclusions. Pages/quotes remain attributable to the original PDF hash.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

MAX_PDF_BYTES = 60 * 1024 * 1024
MAX_PAGES = 1500
MAX_DOCUMENTS = 25
MAX_BATCH_BYTES = 120 * 1024 * 1024  # protect low-memory local machines

_EMAIL = re.compile(r"(?<![\w.])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.])")
_PHONE = re.compile(r"(?<!\d)(?:\+?880[ -]?1[3-9][ -]?\d{8}|01[3-9][ -]?\d{8})(?!\d)")
_CHAPTER = re.compile(r"^(?:chapter|unit|section|part)\s+(?:\d+|[ivxlcdm]+)\b.*$", re.I)


def clean_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def detect_language(text: str) -> str:
    sample = text[:100_000]
    bn = sum(1 for c in sample if "\u0980" <= c <= "\u09ff")
    en = sum(1 for c in sample if "a" <= c.lower() <= "z")
    if bn >= 10 and en >= 10 and min(bn, en) / max(bn, en) >= 0.04:
        return "Bangla-English mixed"
    if bn >= 10 and bn > en * 0.06:
        return "Bangla"
    if en >= 10:
        return "English"
    return "Unknown / other"


def detect_document_type(text: str) -> tuple[str, list[str]]:
    sample = text[:50_000].lower()
    patterns = {
        "research_paper": [r"\babstract\b", r"\bmethodology\b", r"\bresults? and discussion\b", r"\breferences\b", r"\bdoi\s*:"],
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


def pii_counts(text: str) -> dict[str, int]:
    """Limited regex signals only: NOT a comprehensive privacy scan."""
    return {"email_addresses": len(_EMAIL.findall(text)), "bd_phone_numbers": len(_PHONE.findall(text))}


def mask_detected_pii(text: str) -> str:
    """Optional *partial* redaction for visible text. Not a PDF redactor."""
    return _PHONE.sub("[REDACTED_PHONE]", _EMAIL.sub("[REDACTED_EMAIL]", text))


def _section_title(page_num: int, text: str, outline: list[dict[str, Any]]) -> str:
    from_toc = [s for s in outline if s["page"] <= page_num]
    if from_toc:
        return from_toc[-1]["title"]
    for line in text.splitlines()[:6]:
        line = line.strip()
        if 5 <= len(line) <= 140 and _CHAPTER.match(line):
            return line
    return ""


def extract_pdf(
    pdf_bytes: bytes, filename: str, *, use_ocr: bool = False,
    ocr_language: str = "eng", page_limit: int = MAX_PAGES,
    extract_tables: bool = False, redact_pii: bool = False,
) -> dict[str, Any]:
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Upload a valid PDF file.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 60 MB local processing limit.")
    if not 1 <= page_limit <= MAX_PAGES:
        raise ValueError(f"page_limit must be 1 to {MAX_PAGES}.")
    source_id = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc

    pages: list[dict[str, Any]] = []
    warnings: list[str] = []
    tables: list[dict[str, Any]] = []
    try:
        if doc.needs_pass:
            raise ValueError("Unlock password-protected PDFs before uploading.")
        total_pages = len(doc)
        limit = min(total_pages, page_limit)
        if total_pages > page_limit:
            warnings.append(f"Only first {limit} of {total_pages} pages processed; increase page limit for full coverage.")
        outline = [
            {"level": int(level), "title": str(title)[:300], "page": int(page)}
            for level, title, page in doc.get_toc()
            if 1 <= int(page) <= limit
        ]
        current_section = ""
        for idx in range(limit):
            page = doc[idx]
            text = clean_text(page.get_text("text", sort=True))
            extraction = "digital_text"
            if len(text) < 35:
                if use_ocr:
                    try:
                        text_page = page.get_textpage_ocr(language=ocr_language, dpi=160, full=True)
                        ocr_text = clean_text(page.get_text("text", textpage=text_page, sort=True))
                        if len(ocr_text) > len(text):
                            text, extraction = ocr_text, "ocr"
                        else:
                            warnings.append(f"Page {idx+1}: OCR returned no additional text.")
                    except Exception as exc:
                        warnings.append(f"Page {idx+1}: OCR unavailable/failed ({type(exc).__name__}: {exc}).")
                else:
                    warnings.append(f"Page {idx+1}: little extractable text; enable OCR for scanned pages.")
            if extract_tables:
                try:
                    detected = page.find_tables()
                    for number, table in enumerate(detected.tables, 1):
                        rows = table.extract()
                        if rows and redact_pii:
                            rows = [[mask_detected_pii(cell) if isinstance(cell, str) else cell for cell in row] for row in rows]
                        if rows:
                            tables.append({
                                "source_id": source_id, "source_file": Path(filename).name,
                                "source_page": idx + 1, "table_number": number,
                                "bbox": [round(float(v), 2) for v in table.bbox],
                                "rows": rows,
                                "extraction_method": "heuristic_pdf_table",
                            })
                except Exception as exc:
                    warnings.append(f"Page {idx+1}: table detection failed ({type(exc).__name__}: {exc}).")
            privacy = pii_counts(text)
            if redact_pii:
                text = mask_detected_pii(text)
            proposed = _section_title(idx + 1, text, outline)
            if proposed:
                current_section = proposed
            pages.append({
                "page": idx + 1, "text": text, "characters": len(text),
                "section_title": current_section, "extraction_method": extraction,
                "embedded_images": len(page.get_images(full=True)),
                "possible_pii": privacy,
            })
    finally:
        doc.close()
    combined = "\n\n".join(p["text"] for p in pages if p["text"])
    kind, signals = detect_document_type(combined)
    if not combined:
        warnings.append("No usable text extracted. Try OCR and inspect the source document.")
    if any(p["embedded_images"] for p in pages):
        warnings.append("Images, charts and diagrams are not semantically understood by the text extractor.")
    warnings.append("Complex columns, equations, tables and reading order require visual verification.")
    if any(sum(p["possible_pii"].values()) for p in pages):
        warnings.append("Possible personal data detected. Regex masking, if selected, is partial; manually check exports.")
    if tables:
        warnings.append("Extracted tables are heuristic and require visual validation.")
    return {
        "source_id": source_id, "filename": Path(filename).name,
        "file_bytes": len(pdf_bytes), "page_count": total_pages,
        "processed_pages": len(pages), "language_heuristic": detect_language(combined),
        "document_type_heuristic": kind, "type_signals": signals,
        "scanned_pages_ocr": sum(p["extraction_method"] == "ocr" for p in pages),
        "extracted_characters": len(combined), "pages": pages,
        "outline": outline, "tables": tables, "text_pii_masked": redact_pii,
        "warnings": warnings, "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def split_text(text: str, target_chars: int = 2300) -> list[str]:
    if target_chars < 300:
        raise ValueError("target_chars must be >= 300")
    text = clean_text(text)
    if not text:
        return []
    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = clean_text(paragraph)
        if not paragraph:
            continue
        while len(paragraph) > target_chars:
            cut = paragraph.rfind(" ", 0, target_chars + 1)
            if cut < max(100, target_chars // 3):
                cut = target_chars
            first, paragraph = paragraph[:cut].strip(), paragraph[cut:].strip()
            if current:
                pieces.append(current)
                current = ""
            if first:
                pieces.append(first)
        if paragraph:
            candidate = current + ("\n\n" if current else "") + paragraph
            if len(candidate) > target_chars and current:
                pieces.append(current)
                current = paragraph
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def make_chunks(document: dict[str, Any], target_chars: int = 2300) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for page in document["pages"]:
        for i, body in enumerate(split_text(page["text"], target_chars), start=1):
            result.append({
                "chunk_id": f"{document['source_id'][:12]}-p{page['page']:04d}-c{i:03d}",
                "source_id": document["source_id"], "source_file": document["filename"],
                "source_page": page["page"], "section_title": page.get("section_title", ""),
                "language": detect_language(body),
                "document_type": (document.get("llm_profile") or {}).get("document_type") or document["document_type_heuristic"],
                "extraction_method": page["extraction_method"], "text": body,
                "characters": len(body),
                "approx_tokens": math.ceil(len(body) / 4),  # coarse only; tokenizer dependent
            })
    return result
