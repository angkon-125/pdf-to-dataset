"""Revalidation, reviewer import, leakage-aware splits, and dataset ZIP bundles."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from .core import pii_counts
from .generation import norm, validate_candidate

REVIEW_COLUMNS = [
    "annotation_id", "source_id", "source_file", "source_page", "chunk_id",
    "section_title", "language", "document_type", "kind", "question", "answer",
    "evidence_quote", "validation_flags", "review_status", "reviewer", "review_note",
]


def _safe_csv_cell(value: Any) -> Any:
    """Avoid Excel/Sheets formula execution from an untrusted document or LLM."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def csvtext(items: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for item in items:
        row = {k: _safe_csv_cell(v) for k, v in item.items()}
        row["validation_flags"] = ";".join(item.get("validation_flags", []))
        writer.writerow(row)
    return buffer.getvalue()


def jsonlines(items: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in items)


def revalidate_annotations(annotations: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    seen: dict[str, str] = {}
    recent: list[str] = []
    result: list[dict[str, Any]] = []
    for item in annotations:
        entry = dict(item)
        chunk = lookup.get(entry.get("chunk_id"))
        flags = validate_candidate(entry, chunk) if chunk is not None else ["missing_source_chunk"]
        if chunk is not None:
            if entry.get("source_id") != chunk.get("source_id") or entry.get("source_page") != chunk.get("source_page"):
                flags.append("provenance_mismatch")
        question_key = norm(str(entry.get("question", "")))
        if question_key and question_key in seen:
            flags.append("duplicate_question")
        elif len(question_key) >= 30:
            if any(SequenceMatcher(None, question_key, prior).ratio() >= 0.97 for prior in recent[-300:]):
                flags.append("possible_near_duplicate_question")
        if question_key:
            seen[question_key] = str(entry.get("annotation_id", ""))
            recent.append(question_key)
        if sum(pii_counts(" ".join(str(entry.get(k, "")) for k in ("question", "answer", "evidence_quote"))).values()):
            flags.append("possible_pii")
        entry["validation_flags"] = list(dict.fromkeys(flags))
        if entry.get("review_status") not in ("approved", "rejected", "needs_review"):
            entry["review_status"] = "needs_review"
        if entry.get("review_status") == "approved" and not str(entry.get("reviewer", "")).strip():
            entry["validation_flags"].append("missing_reviewer")
        if entry.get("review_status") == "approved" and entry["validation_flags"]:
            entry["review_status"] = "needs_review"
            entry["review_note"] = (str(entry.get("review_note", "")) + " [Approval reset: failed validation]").strip()
        result.append(entry)
    return result


def apply_review_csv(
    current: list[dict[str, Any]], csv_bytes: bytes, *, confirmed: bool = False
) -> list[dict[str, Any]]:
    """Import human edits only; approving rows requires explicit user attestation."""
    try:
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Invalid UTF-8 review CSV") from exc
    original = {a["annotation_id"]: dict(a) for a in current}
    if not rows:
        raise ValueError("Review CSV is empty")
    seen: set[str] = set()
    permitted = {"question", "answer", "evidence_quote", "review_note", "reviewer", "review_status"}
    for row in rows:
        key = row.get("annotation_id", "")
        if key in seen:
            raise ValueError(f"Duplicate annotation_id in review CSV: {key}")
        seen.add(key)
        if key not in original:
            raise ValueError(f"Unknown annotation_id in review CSV: {key}")
        item = original[key]
        if row.get("source_id") and row["source_id"] != item.get("source_id"):
            raise ValueError(f"Source ID modified for annotation_id: {key}")
        for field in permitted:
            if field in row:
                raw = row[field] or ""
                # Reverse CSV spreadsheet-protection prefix for editable fields.
                item[field] = raw[1:] if raw.startswith("'") and raw[1:2] in ("=", "+", "-", "@") else raw
        if item.get("review_status") == "approved" and not confirmed:
            item["review_status"] = "needs_review"
            item["review_note"] += " [Approval ignored until review attestation]"
    return [original[a["annotation_id"]] for a in current]


def _approved_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
    }


def _split_sources(source_ids: list[str], seed: int = 42) -> dict[str, str]:
    uniq = sorted(set(source_ids), key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())
    if len(uniq) < 3:
        return {s: "train" for s in uniq}
    val_count = max(1, round(len(uniq) * 0.1))
    test_count = max(1, round(len(uniq) * 0.1))
    if len(uniq) - val_count - test_count < 1:
        val_count, test_count = 1, 1
    train_count = len(uniq) - val_count - test_count
    return {s: ("train" if i < train_count else "validation" if i < train_count + val_count else "test") for i, s in enumerate(uniq)}


def build_audit_events(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimal trace of review edits, without storing duplicate answer text."""
    previous = {r["annotation_id"]: r for r in before}
    events: list[dict[str, Any]] = []
    relevant = ("question", "answer", "evidence_quote", "review_status", "review_note", "reviewer")
    for item in after:
        old = previous.get(item["annotation_id"])
        if old is None:
            continue
        changes = [key for key in relevant if old.get(key, "") != item.get(key, "")]
        if changes:
            events.append({
                "annotation_id": item["annotation_id"], "source_id": item.get("source_id", ""),
                "reviewer": str(item.get("reviewer", "")),
                "old_status": old.get("review_status", "needs_review"),
                "new_status": item.get("review_status", "needs_review"),
                "fields_changed": changes,
                "changed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
    return events


def _dataset_card(manifest: dict[str, Any]) -> str:
    return f"""# Angkon Dataset Factory dataset card

Generated by Angkon Dataset Factory **v{manifest['version']}**.

## Scope
- Sources: {manifest['source_documents']} PDFs; {manifest['source_pages_processed']} pages processed
- Document languages: heuristic only, see `manifest.json`
- RAG chunks: {manifest['chunks_extracted']}
- Draft Q&A examples: {manifest['candidate_examples']}
- Human-approved SFT examples: {manifest['approved_examples']}
- Generation coverage: {manifest['chunks_generation_attempted']} / {manifest['chunks_extracted']} chunks

## Provenance and source rights
- Each chunk and annotation includes a source hash, file name, PDF page, and chunk ID.
- Original PDFs are **not** included; the extracted text *is* included.
- User rights attestation: {manifest['user_attested_rights']} (not independently verified).
- No permission to train or redistribute copyrighted content is implied.

## Quality checks and limitations
- A matching quote proves string occurrence, not that an answer is entailed or true.
- All generated Q&A begins as `needs_review`; approvals require named reviewers.
- OCR, tables, document classification, and privacy masking can be inaccurate.
- Exact/near-duplicate and simple PII checks are heuristic.
- No fabricated DPO rejected responses are included.
- Train/validation/test partitions group by **source PDF** to avoid same-book leakage.
- With fewer than three unique documents, all approved examples remain in train.

## Intended use
Source-linked retrieval and *human-reviewed* instruction-tuning examples.
Not a benchmark of verified factual truth and not a substitute for legal/privacy review.
"""


def build_batch_zip(
    documents: list[dict[str, Any]], chunks: list[dict[str, Any]],
    annotations: list[dict[str, Any]], *, backend: str = "none", model: str = "",
    generation_warnings: list[str] | None = None, chunks_attempted: int = 0,
    rights_attested: bool = False, audit_log: list[dict[str, Any]] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    clean = revalidate_annotations(annotations, chunks)
    source_map = _split_sources([d["source_id"] for d in documents])
    approved_rows = [x for x in clean if x["review_status"] == "approved" and not x["validation_flags"]] if rights_attested else []
    approved = [_approved_record(x) for x in approved_rows]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for x in approved_rows:
        splits[source_map[x["source_id"]]].append(_approved_record(x))
    all_warnings = [w for doc in documents for w in doc.get("warnings", [])] + list(generation_warnings or [])
    if backend == "none":
        all_warnings.append("Extraction-only mode: no LLM-generated Q&A examples included.")
    if backend != "none" and chunks_attempted < len(chunks):
        all_warnings.append(f"Generation attempted for {chunks_attempted} of {len(chunks)} chunks; unattempted chunks are still exported for retrieval.")
    if any(x["review_status"] == "needs_review" for x in clean):
        all_warnings.append("Unreviewed examples are draft candidates, NOT verified training data.")
    if not rights_attested and clean:
        all_warnings.append("User did not attest training rights; approved SFT exports withheld.")
    if len(documents) < 3:
        all_warnings.append("Fewer than three unique PDFs: no independent source-held-out validation/test split.")
    counts = Counter(flag for x in clean for flag in x["validation_flags"])
    pages_processed = sum(d["processed_pages"] for d in documents)
    pages_total = sum(d["page_count"] for d in documents)
    pages_with_text = sum(bool(p["text"].strip()) for d in documents for p in d["pages"])
    num_tables = sum(len(d.get("tables", [])) for d in documents)
    manifest: dict[str, Any] = {
        "project": "Angkon Dataset Factory", "version": "0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_documents": len(documents),
        "sources": [
            {
                "source_id": d["source_id"], "filename": d["filename"],
                "page_count": d["page_count"], "processed_pages": d["processed_pages"],
                "language_heuristic": d["language_heuristic"],
                "document_type_heuristic": d["document_type_heuristic"],
                "llm_document_profile": d.get("llm_profile"),
                "text_pii_masked": d.get("text_pii_masked", False),
                "split": source_map[d["source_id"]],
            } for d in documents
        ],
        "original_pdf_included": False,
        "source_pages_total": pages_total,
        "source_pages_processed": pages_processed,
        "pages_with_extracted_text": pages_with_text,
        "text_extraction_coverage_percent": round(100 * pages_with_text / pages_processed, 1) if pages_processed else 0,
        "chunks_extracted": len(chunks),
        "chunks_generation_attempted": chunks_attempted,
        "generation_chunk_coverage_percent": round(100 * chunks_attempted / len(chunks), 1) if chunks else 0,
        "tables_extracted": num_tables,
        "llm_backend": backend, "llm_model": model or None,
        "candidate_examples": len(clean), "approved_examples": len(approved),
        "rejected_examples": sum(x["review_status"] == "rejected" for x in clean),
        "unreviewed_examples": sum(x["review_status"] == "needs_review" for x in clean),
        "validation_flag_counts": dict(counts),
        "audit_events": len(audit_log or []),
        "approved_split_counts": {name: len(rows) for name, rows in splits.items()},
        "user_attested_rights": bool(rights_attested),
        "training_rights_verified": False,
        "source_quote_check": "string matching only; human semantic verification required",
        "warnings": list(dict.fromkeys(all_warnings)),
    }
    if len(documents) == 1:
        d = documents[0]
        manifest.update({
            "source_id": d["source_id"], "source_filename": d["filename"],
            "language_heuristic": d["language_heuristic"],
            "document_type_heuristic": d["document_type_heuristic"],
            "detection_signals": d["type_signals"],
            "llm_document_profile": d.get("llm_profile"),
        })
    source_metadata = [{k: v for k, v in d.items() if k not in ("pages", "tables")} for d in documents]
    review_records = [
        {
            "messages": _approved_record(x)["messages"],
            "annotation_id": x["annotation_id"], "source_id": x["source_id"],
            "source_page": x["source_page"], "chunk_id": x["chunk_id"],
            "evidence_quote": x["evidence_quote"],
            "validation_flags": x["validation_flags"], "review_status": x["review_status"],
        } for x in clean
    ]
    source_pages = [
        {"source_id": d["source_id"], "source_file": d["filename"], **p}
        for d in documents for p in d["pages"]
    ]
    all_tables = [t for d in documents for t in d.get("tables", [])]
    full_text = "\n\n".join(
        f"=== SOURCE {d['filename']} ({d['source_id'][:12]}) / PAGE {p['page']} ===\n{p['text']}"
        for d in documents for p in d["pages"]
    )
    quality = {
        "pages_with_text": pages_with_text, "pages_processed": pages_processed,
        "no_text_pages": pages_processed - pages_with_text,
        "generated_candidates": len(clean), "validation_flag_counts": dict(counts),
        "approved_candidates": len(approved),
        "unverified_semantic_accuracy": True,
        "document_types": dict(Counter(d["document_type_heuristic"] for d in documents)),
        "detected_languages": dict(Counter(d["language_heuristic"] for d in documents)),
    }
    readme = """ANGKON DATASET FACTORY v0.2 - OUTPUT GUIDE

rag_chunks.jsonl: all extracted, page-linked chunks (including non-generated chunks).
source_pages.jsonl: extracted page text and extraction warnings/metadata.
tables.jsonl: optional, heuristic table extraction; check visually.
qa_candidates.jsonl: unverified model-generated Q&A with source evidence and flags.
annotations_review.csv: editable review queue; do not auto-approve unreviewed rows.
sft_candidates.jsonl: draft instruction records, not training-ready.
sft_approved.jsonl: human-approved, mechanically valid, rights-attested SFT only.
sft_train.jsonl / sft_validation.jsonl / sft_test.jsonl: source-grouped splits.
manifest.json / quality_report.json / dataset_card.md: provenance and limitations.
extracted_text.txt: exported source text; original PDFs are not included.

Do NOT train on, publish, or distribute copyrighted/private material without rights.
String-matched evidence quotes do NOT establish factual accuracy or answer support.
PII masking is partial and does not remove images, embedded document metadata, etc.
DPO preference pairs are not fabricated.
"""
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("quality_report.json", json.dumps(quality, ensure_ascii=False, indent=2))
        z.writestr("dataset_card.md", _dataset_card(manifest))
        z.writestr("README_DATASET.txt", readme)
        z.writestr("source_documents.json", json.dumps(source_metadata, ensure_ascii=False, indent=2))
        z.writestr("audit_log.jsonl", jsonlines(audit_log or []))
        z.writestr("annotation_guidelines.md", """# Human annotation guidelines

## Required checks for approval

1. Read the original source page and the complete extracted chunk.
2. Verify the evidence quote actually supports the **answer** (string matching alone is insufficient).
3. Check accuracy, relevance, completeness, clarity, and safety of both question and answer.
4. Inspect OCR/table/section issues, including scanned diagrams and omitted context.
5. Check private/sensitive data and material rights before exporting or training.
6. Correct the answer or evidence, or mark it **rejected** if it cannot be repaired.
7. Set `review_status=approved` only after personal verification and enter `reviewer`.
8. If multiple reviewers disagree, record the reason in `review_note` and adjudicate.

Mechanical checks cannot establish semantic correctness. This file is guidance, not an employer-specific grading policy.
""")
        z.writestr("source_pages.jsonl", jsonlines(source_pages))
        z.writestr("rag_chunks.jsonl", jsonlines(chunks))
        z.writestr("tables.jsonl", jsonlines(all_tables))
        z.writestr("qa_candidates.jsonl", jsonlines(clean))
        z.writestr("annotations_review.csv", csvtext(clean, REVIEW_COLUMNS))
        z.writestr("sft_candidates.jsonl", jsonlines(review_records))
        z.writestr("sft_approved.jsonl", jsonlines(approved))
        for name, rows in splits.items():
            z.writestr(f"sft_{name}.jsonl", jsonlines(rows))
        z.writestr("extracted_text.txt", full_text)
    return b.getvalue(), manifest


def build_zip(
    document: dict[str, Any], chunks: list[dict[str, Any]],
    annotations: list[dict[str, Any]], *, backend: str = "none", model: str = "",
    generation_warnings: list[str] | None = None, chunks_attempted: int = 0,
    rights_attested: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Single-document API. Training rights must be explicitly attested."""
    return build_batch_zip(
        [document], chunks, annotations, backend=backend, model=model,
        generation_warnings=generation_warnings, chunks_attempted=chunks_attempted,
        rights_attested=rights_attested,
    )
