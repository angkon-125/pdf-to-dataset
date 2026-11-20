"""Command-line runner for notebooks, Kaggle, servers, and local machines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import MAX_PAGES, extract_pdf, make_chunks
from .exports import build_zip
from .generation import LLMClient, generate_candidates, classify_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF into traceable dataset candidates.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("angkon_dataset.zip"))
    parser.add_argument("--backend", choices=["none", "ollama", "openai_compatible"], default="none")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--ocr", action="store_true", help="Use Tesseract for pages without extractable text")
    parser.add_argument("--ocr-language", default="eng", help="Tesseract languages, e.g. eng+ben")
    parser.add_argument("--page-limit", type=int, default=MAX_PAGES)
    parser.add_argument("--chunk-chars", type=int, default=2300)
    parser.add_argument("--max-chunks", type=int, default=15, help="LLM-generation limit; 0 means all chunks")
    parser.add_argument("--examples-per-chunk", type=int, default=2)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF file does not exist")
    doc = extract_pdf(args.pdf.read_bytes(), args.pdf.name, use_ocr=args.ocr,
                      ocr_language=args.ocr_language, page_limit=args.page_limit)
    chunks = make_chunks(doc, target_chars=args.chunk_chars)
    generated = []
    warnings = []
    attempted = 0
    if args.backend != "none":
        import os
        endpoint = args.endpoint
        if args.backend == "openai_compatible" and endpoint == "http://localhost:11434":
            endpoint = "https://api.openai.com/v1"
        client = LLMClient(args.backend, args.model, endpoint, api_key=os.getenv(args.api_key_env, ""))
        profile, profile_warnings = classify_document(chunks, client)
        doc["llm_profile"] = profile
        warnings.extend(profile_warnings)
        chunks = make_chunks(doc, target_chars=args.chunk_chars)
        attempted = len(chunks) if args.max_chunks == 0 else min(args.max_chunks, len(chunks))
        generated, generation_warnings = generate_candidates(chunks, client, examples_per_chunk=args.examples_per_chunk,
                                                  max_chunks=args.max_chunks)
        warnings.extend(generation_warnings)
    bundle, manifest = build_zip(doc, chunks, generated, backend=args.backend, model=args.model if args.backend != "none" else "",
                                 generation_warnings=warnings, chunks_attempted=attempted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bundle)
    print(json.dumps({"zip": str(args.output.resolve()), "summary": manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
