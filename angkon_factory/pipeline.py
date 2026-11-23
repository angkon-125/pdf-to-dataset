"""Batch orchestration with local CPU-friendly generation and deterministic resumption."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .core import MAX_BATCH_BYTES, MAX_DOCUMENTS, MAX_PAGES, extract_pdf, make_chunks
from .generation import LLMClient, classify_document, generate_candidates, select_chunks


@dataclass(frozen=True)
class PipelineOptions:
    page_limit: int = MAX_PAGES
    chunk_chars: int = 1100
    use_ocr: bool = False
    ocr_language: str = "eng"
    extract_tables: bool = False
    redact_pii: bool = False
    examples_per_chunk: int = 2
    max_chunks_per_document: int = 12  # 0 means all; prefer a small pilot first
    sampling: str = "spread"
    max_source_chars: int = 950  # compatible with small local context windows
    llm_profile: bool = False  # optional extra model call per source
    resume_dir: str | Path | None = None
    rights_attested: bool = False


def run_batch(
    inputs: Iterable[tuple[str, bytes]], *, options: PipelineOptions | None = None,
    client: LLMClient | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    options = options or PipelineOptions()
    items = list(inputs)
    if not items:
        raise ValueError("Upload at least one PDF.")
    if len(items) > MAX_DOCUMENTS:
        raise ValueError(f"Maximum {MAX_DOCUMENTS} PDFs per batch.")
    if sum(len(data) for _, data in items) > MAX_BATCH_BYTES:
        raise ValueError("Batch exceeds 120 MB; process smaller groups on low-memory systems.")
    if client is not None and not options.rights_attested:
        raise ValueError("Confirm you have the necessary rights before generating training candidates.")
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    processed_hashes: set[str] = set()
    generated_chunks = 0
    for i, (name, data) in enumerate(items, start=1):
        document = extract_pdf(
            data, name, use_ocr=options.use_ocr,
            ocr_language=options.ocr_language, page_limit=options.page_limit,
            extract_tables=options.extract_tables, redact_pii=options.redact_pii,
        )
        if document["source_id"] in processed_hashes:
            warnings.append(f"Duplicate PDF skipped: {document['filename']} (same SHA-256 as another input).")
            if progress:
                progress(i, len(items), name)
            continue
        processed_hashes.add(document["source_id"])
        local_chunks = make_chunks(document, target_chars=options.chunk_chars)
        if client is not None and local_chunks and options.llm_profile:
            profile, extra = classify_document(local_chunks, client)
            document["llm_profile"] = profile
            warnings.extend(extra)
            local_chunks = make_chunks(document, target_chars=options.chunk_chars)
        if client is not None and local_chunks:
            planned = select_chunks(local_chunks, options.max_chunks_per_document, strategy=options.sampling)
            generated_chunks += len(planned)
            generated, extra = generate_candidates(
                local_chunks, client,
                examples_per_chunk=options.examples_per_chunk,
                max_chunks=options.max_chunks_per_document,
                sampling=options.sampling,
                max_source_chars=options.max_source_chars,
                cache_dir=options.resume_dir,
            )
            candidates.extend(generated)
            warnings.extend(extra)
        elif client is not None:
            warnings.append(f"{document['filename']}: no usable chunks for model generation.")
        documents.append(document)
        chunks.extend(local_chunks)
        if progress:
            progress(i, len(items), name)
    return {
        "documents": documents,
        "chunks": chunks,
        "annotations": candidates,
        "warnings": warnings,
        "chunks_attempted": generated_chunks,
        "llm_backend": getattr(client, "backend", "custom") if client is not None else "none",
        "llm_model": getattr(client, "model", "custom") if client is not None else "",
        "rights_attested": options.rights_attested,
        "options": options,
    }
