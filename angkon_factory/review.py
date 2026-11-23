"""Reopen a generated archive and import an edited human-review CSV without LLM calls."""
from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from typing import Any

from .exports import apply_review_csv, build_audit_events, build_batch_zip


def _jsonl(z: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in z.read(name).decode("utf-8").splitlines() if line.strip()]


def review_existing_zip(
    previous_zip: bytes, review_csv: bytes, *, attest_reviewed: bool,
    rights_attested: bool,
) -> tuple[bytes, dict[str, Any]]:
    if not attest_reviewed:
        raise ValueError("Human review attestation is required to import approvals.")
    with zipfile.ZipFile(io.BytesIO(previous_zip), "r") as z:
        required = {"manifest.json", "source_documents.json", "source_pages.jsonl", "rag_chunks.jsonl", "qa_candidates.jsonl"}
        if not required.issubset(z.namelist()):
            raise ValueError("Use a v0.2+ archive with source document metadata.")
        manifest = json.loads(z.read("manifest.json"))
        metadata = json.loads(z.read("source_documents.json"))
        pages = _jsonl(z, "source_pages.jsonl")
        tables = _jsonl(z, "tables.jsonl") if "tables.jsonl" in z.namelist() else []
        chunks = _jsonl(z, "rag_chunks.jsonl")
        annotations = _jsonl(z, "qa_candidates.jsonl")
        old_audit = _jsonl(z, "audit_log.jsonl") if "audit_log.jsonl" in z.namelist() else []
    by_source_pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pages:
        by_source_pages[item["source_id"]].append({k: v for k, v in item.items() if k not in ("source_id", "source_file")})
    for item in tables:
        by_source_tables[item["source_id"]].append(item)
    documents = [
        {**d, "pages": by_source_pages[d["source_id"]], "tables": by_source_tables[d["source_id"]]}
        for d in metadata
    ]
    updated = apply_review_csv(annotations, review_csv, confirmed=attest_reviewed)
    audit = old_audit + build_audit_events(annotations, updated)
    return build_batch_zip(
        documents, chunks, updated,
        backend=str(manifest.get("llm_backend", "none")),
        model=str(manifest.get("llm_model") or ""),
        chunks_attempted=int(manifest.get("chunks_generation_attempted", 0)),
        generation_warnings=["Reviewed from a previous dataset ZIP. No LLM calls were made during import."],
        rights_attested=rights_attested, audit_log=audit,
    )
