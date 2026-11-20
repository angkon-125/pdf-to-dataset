"""Export source datasets, review queues, and approved training examples."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any

from .generation import approved_examples, validate_candidate, norm


def revalidate_annotations(annotations: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {c["chunk_id"]: c for c in chunks}
    final = []
    seen_questions: set[str] = set()
    for record in annotations:
        entry = dict(record)
        chunk = lookup.get(entry.get("chunk_id"))
        flags = validate_candidate(entry, chunk) if chunk else ["missing_source_chunk"]
        question_key = norm(str(entry.get("question", "")))
        if question_key in seen_questions:
            flags.append("duplicate_question")
        seen_questions.add(question_key)
        entry["validation_flags"] = flags
        if entry.get("review_status") not in ("approved", "rejected", "needs_review"):
            entry["review_status"] = "needs_review"
        if entry["review_status"] == "approved" and flags:
            # An approval cannot override mechanical evidence failures.
            entry["review_status"] = "needs_review"
            entry["review_note"] = (str(entry.get("review_note", "")) + " [Failed validation; approval reset]").strip()
        final.append(entry)
    return final


def jsonlines(items: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def csvtext(items: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for item in items:
        row = dict(item)
        row["validation_flags"] = ";".join(item.get("validation_flags", []))
        writer.writerow(row)
    return buffer.getvalue()


def build_zip(document: dict[str, Any], chunks: list[dict[str, Any]],
              annotations: list[dict[str, Any]], *, backend: str = "none", model: str = "",
              generation_warnings: list[str] | None = None,
              chunks_attempted: int = 0) -> tuple[bytes, dict[str, Any]]:
    annotations = revalidate_annotations(annotations, chunks)
    approved = approved_examples(annotations)
    warnings = list(document.get("warnings", [])) + list(generation_warnings or [])
    if backend != "none" and chunks_attempted < len(chunks):
        warnings.append(f"Generation attempted for {chunks_attempted} of {len(chunks)} chunks. Extraction still includes all processed pages.")
    if backend == "none":
        warnings.append("Extraction-only mode: no LLM-generated question/answer examples included.")
    if any(a["review_status"] == "needs_review" for a in annotations):
        warnings.append("Generated training examples are draft until individually reviewed and approved.")
    manifest = {
        "project": "Angkon Dataset Factory",
        "version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": document["source_id"],
        "source_filename": document["filename"],
        "original_pdf_included": False,
        "source_pages_total": document["page_count"],
        "source_pages_processed": document["processed_pages"],
        "language_heuristic": document["language_heuristic"],
        "document_type_heuristic": document["document_type_heuristic"],
        "detection_signals": document["type_signals"],
        "llm_document_profile": document.get("llm_profile"),
        "chunks_extracted": len(chunks),
        "chunks_generation_attempted": chunks_attempted,
        "llm_backend": backend,
        "llm_model": model or None,
        "candidate_examples": len(annotations),
        "approved_examples": len(approved),
        "rejected_examples": sum(a["review_status"] == "rejected" for a in annotations),
        "unreviewed_examples": sum(a["review_status"] == "needs_review" for a in annotations),
        "source_quote_check": "exact matching only; semantic entailment requires human verification",
        "training_rights_verified": False,
        "warnings": warnings,
    }
    all_text = "\n\n".join(f"=== PAGE {p['page']} ===\n{p['text']}" for p in document["pages"])
    columns = [
        "annotation_id", "source_file", "source_page", "chunk_id", "language", "document_type",
        "kind", "question", "answer", "evidence_quote", "validation_flags",
        "review_status", "reviewer", "review_note", "source_id"
    ]
    # A lightweight Q&A training-candidate format, retaining provenance and review status.
    draft_sft = [{
        "messages": [{"role": "user", "content": a["question"]}, {"role": "assistant", "content": a["answer"]}],
        "annotation_id": a["annotation_id"],
        "source_page": a["source_page"],
        "evidence_quote": a["evidence_quote"],
        "validation_flags": a["validation_flags"],
        "review_status": a["review_status"],
    } for a in annotations]
    readme = "ANGKON DATASET FACTORY - DATASET CONTENTS\n\n"
    readme += "rag_chunks.jsonl: extracted page-referenced text for retrieval/indexing.\n"
    readme += "qa_candidates.jsonl: generated question/answer records, NOT automatically verified.\n"
    readme += "annotations_review.csv: editable candidate queue with references and validation flags.\n"
    readme += "sft_candidates.jsonl: draft instruction/answer records; do not train without review.\n"
    readme += "sft_approved.jsonl: approved and mechanically valid instruction/answer records ONLY.\n"
    readme += "manifest.json: processing provenance, counts, limitations, and warnings.\n"
    readme += "extracted_text.txt: human-inspectable extracted text with page boundaries.\n\n"
    readme += "IMPORTANT: A quote appearing in a PDF does not prove that a generated answer is true.\n"
    readme += "Human review must assess source support, fidelity, safety, rights, and privacy.\n"
    readme += "Generating datasets from copyrighted books does NOT grant permission to redistribute or train on them.\n"
    readme += "DPO preference pairs require two real responses and a genuine preference decision; this MVP does not fabricate them.\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr("rag_chunks.jsonl", jsonlines(chunks))
        z.writestr("qa_candidates.jsonl", jsonlines(annotations))
        z.writestr("annotations_review.csv", csvtext(annotations, columns))
        z.writestr("sft_candidates.jsonl", jsonlines(draft_sft))
        z.writestr("sft_approved.jsonl", jsonlines(approved))
        z.writestr("extracted_text.txt", all_text)
        z.writestr("README_DATASET.txt", readme)
    return buf.getvalue(), manifest
