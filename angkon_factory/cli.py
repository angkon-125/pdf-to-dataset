"""CLI: extraction, multi-PDF candidate generation, and offline review import."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core import MAX_DOCUMENTS, MAX_PAGES
from .exports import build_batch_zip
from .generation import LLMClient
from .pipeline import PipelineOptions, run_batch
from .review import review_existing_zip


def _print_summary(output: Path, manifest: dict) -> None:
    print(json.dumps({
        "archive": str(output.resolve()),
        "documents": manifest["source_documents"],
        "pages_processed": manifest["source_pages_processed"],
        "rag_chunks": manifest["chunks_extracted"],
        "candidate_examples": manifest["candidate_examples"],
        "approved_sft": manifest["approved_examples"],
        "warnings": manifest["warnings"],
    }, indent=2, ensure_ascii=False))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        p = argparse.ArgumentParser(description="Import a human-reviewed CSV into an existing v0.2 dataset archive.")
        p.add_argument("review")
        p.add_argument("archive", type=Path)
        p.add_argument("edited_csv", type=Path)
        p.add_argument("--output", "-o", type=Path, default=Path("angkon_approved_dataset.zip"))
        p.add_argument("--attest-reviewed", action="store_true", help="Confirm you personally reviewed the approved rows")
        p.add_argument("--rights-confirmed", action="store_true", help="Confirm necessary dataset rights and permissions")
        args = p.parse_args()
        archive, manifest = review_existing_zip(
            args.archive.read_bytes(), args.edited_csv.read_bytes(),
            attest_reviewed=args.attest_reviewed,
            rights_attested=args.rights_confirmed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(archive)
        _print_summary(args.output, manifest)
        return

    p = argparse.ArgumentParser(description="Convert PDF books/papers/reports to source-linked datasets.")
    p.add_argument("pdf", type=Path, nargs="+", help=f"1 to {MAX_DOCUMENTS} PDF files")
    p.add_argument("--output", "-o", type=Path, default=Path("angkon_dataset.zip"))
    p.add_argument("--backend", choices=["none", "ollama", "openai_compatible"], default="none")
    p.add_argument("--model", default="qwen3:0.6b")
    p.add_argument("--endpoint", default="http://127.0.0.1:11434")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--max-output-tokens", type=int, default=550)
    p.add_argument("--rights-confirmed", action="store_true", help="Acknowledge the needed rights/permissions for training and processing")
    p.add_argument("--ocr", action="store_true", help="Use local Tesseract for pages with little text")
    p.add_argument("--ocr-language", default="eng", help="Installed Tesseract languages, e.g. eng+ben")
    p.add_argument("--page-limit", type=int, default=MAX_PAGES)
    p.add_argument("--chunk-chars", type=int, default=1100)
    p.add_argument("--tables", action="store_true", help="Extract detectable tables (heuristic, potentially slower)")
    p.add_argument("--redact-pii", action="store_true", help="Partially mask visible emails/BD phone numbers (not comprehensive)")
    p.add_argument("--max-chunks", type=int, default=12, help="Number of model calls per PDF; 0 = every chunk")
    p.add_argument("--examples-per-chunk", type=int, default=2)
    p.add_argument("--sampling", choices=["spread", "first"], default="spread")
    p.add_argument("--max-source-chars", type=int, default=950, help="Text sent to each model call; low-RAM-friendly default")
    p.add_argument("--llm-profile", action="store_true", help="Optional additional model call for document classification")
    p.add_argument("--resume-dir", type=Path, default=Path("runs/cache"), help="Local response cache for resumable generation")
    args = p.parse_args()

    if len(args.pdf) > MAX_DOCUMENTS:
        p.error(f"A batch cannot exceed {MAX_DOCUMENTS} PDFs")
    for name in args.pdf:
        if not name.is_file() or name.suffix.lower() != ".pdf":
            p.error(f"PDF not found or wrong extension: {name}")
    if args.backend != "none" and not args.rights_confirmed:
        p.error("--rights-confirmed is required before generating training candidates")
    client = None
    if args.backend != "none":
        endpoint = args.endpoint
        if args.backend == "openai_compatible" and endpoint == "http://127.0.0.1:11434":
            endpoint = "http://127.0.0.1:8080/v1"  # typical local llama.cpp OpenAI-compatible server
        client = LLMClient(
            args.backend, args.model, endpoint, api_key=os.getenv(args.api_key_env, ""),
            timeout=args.timeout, max_output_tokens=args.max_output_tokens,
        )
    options = PipelineOptions(
        page_limit=args.page_limit, chunk_chars=args.chunk_chars,
        use_ocr=args.ocr, ocr_language=args.ocr_language, extract_tables=args.tables,
        redact_pii=args.redact_pii, examples_per_chunk=args.examples_per_chunk,
        max_chunks_per_document=args.max_chunks, sampling=args.sampling,
        max_source_chars=args.max_source_chars, llm_profile=args.llm_profile,
        resume_dir=args.resume_dir if client is not None else None,
        rights_attested=args.rights_confirmed,
    )
    data = run_batch(
        [(item.name, item.read_bytes()) for item in args.pdf],
        options=options, client=client,
        progress=lambda current, total, label: print(f"[{current}/{total}] Finished: {label}", file=sys.stderr),
    )
    archive, manifest = build_batch_zip(
        data["documents"], data["chunks"], data["annotations"],
        backend=data["llm_backend"], model=data["llm_model"],
        generation_warnings=data["warnings"], chunks_attempted=data["chunks_attempted"],
        rights_attested=args.rights_confirmed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(archive)
    _print_summary(args.output, manifest)


if __name__ == "__main__":
    main()
