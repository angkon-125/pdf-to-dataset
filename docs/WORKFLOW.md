# Angkon Dataset Factory v0.3: PDF ingestion foundation and guarantees

## Workflow

```
PDF files [1..25; 120 MB aggregate]
   |
   | sha256; reject encrypted or invalid PDF; skip duplicate binary files
   v
Per-page extraction (PyMuPDF text; optional local Tesseract OCR)
   |                 
   +-- optional heuristic tables -> tables.jsonl
   +-- optional regex partial masking -> page text
   +-- PDF outline -> section_title (fallback: chapter heading)
   v
Page-local segmentation -> source_id + source_page + chunk_id
   |           |
   |           +-- complete retrieval dataset: rag_chunks.jsonl
   v
Optional LLM model (Ollama or OpenAI-compatible/local llama.cpp)
   |  small prompt budget; first/spread sampling; optional local response cache
   v
Question/answer + copied evidence quote (review required)
   |
   +-- automatic checks: JSON shape, quote substring, IDs, duplicate questions,
   |                     possible PII, reviewer identifier
   v
Human review: approve / reject / correct; audit event log
   |   source-grounding and actual correctness cannot be automatically certified
   v
ZIP: draft Q&A, reviewer CSV, approved SFT, source-grouped splits, coverage,
     quality report, dataset card, source page text, tables, provenance hashes
```

## Responsibilities by module

- `angkon_factory/core.py`: PDF parsing, section/outline, OCR, tables, privacy signals, chunking.
- `angkon_factory/generation.py`: local/remote LLM backends, JSON candidate generation, response cache, sampling.
- `angkon_factory/pipeline.py`: multi-PDF orchestration, resource caps, duplicate PDF exclusion.
- `angkon_factory/exports.py`: evidence validation, CSV review protection, approval gate, train/test grouping, ZIP.
- `angkon_factory/review.py`: re-open a generated ZIP and import a human-edited CSV offline.
- `angkon_factory/cli.py`: batch CLI and review subcommand.
- `app.py`: local Streamlit UI with source inspection and annotation editor.

## Dataset formats

`rag_chunks.jsonl` includes `source_id`, `source_file`, `source_page`, `chunk_id`, `section_title`, `text`, `language`, extraction metadata, and approximate tokens.

`qa_candidates.jsonl` contains model-generated `question`, `answer`, `kind`, `evidence_quote`, `validation_flags`, `review_status`, `reviewer`, `review_note`, and provenance.

`sft_approved.jsonl` uses chat message arrays with user/assistant roles. Training examples enter it ONLY if all mechanical checks pass, a named reviewer explicitly approves them, and the processing user attests data-use rights.

`source_documents.json`, `source_pages.jsonl`, `audit_log.jsonl`, `quality_report.json`, `dataset_card.md`, and `manifest.json` enable traceability and re-review.

`train`, `validation`, and `test` splits are **grouped by PDF hash**. With fewer than three unique PDFs, only train is populated. This avoids contamination from different pages of the same source appearing in separate splits. It does not address near-duplicate books or shared content across different PDFs.

## Not guaranteed

1. OCR or extraction completeness on scanned, multi-column, mathematical, or diagram-rich PDFs.
2. Semantic entailment of a generated answer based solely on an exact source quote.
3. Perfect language/document type classification. The default classifier is heuristic.
4. Detection or removal of all private data. Regex masking only targets emails and two Bangladesh mobile number patterns in extracted text/tables.
5. Copyright/training permissions. User attestation is *not verification*.
6. Model availability, accuracy, or context fit. Local models may fail to produce valid JSON.
7. Production security (multi-tenant authentication, isolation, persistence, quotas, API allowlisting). This is a local prototype.
8. Accurate DPO data: preference pairs must be collected through real comparisons with human judgments, not synthesized as fake rejected answers.

## Conservative processing guidance for limited RAM

- Use one PDF and extraction-only mode first; test the pages/chunks.
- For a 2,048-token model context, start with `--chunk-chars 900`, `--max-source-chars 650`, `--examples-per-chunk 1`, `--max-output-tokens 350`, `--max-chunks 10`.
- Use `--max-chunks 0` only after verifying model prompt format and performance. This attempts *all* chunks and can be slow.
- Use `--resume-dir runs/cache` to reuse successful model calls. This folder stores model outputs in plaintext and is excluded from Git.
- Enable `--tables` and `--ocr` only when needed, as these have extra CPU cost.
- A document source with no reliable text should be visually reviewed, not automatically deemed complete.


## Professional gold release

The v0.3 release adds `expert_review.py`, `data_science.py`, `curation.py`, and `train_validate.py`. This foundation document mainly covers extraction and draft QA generation. See [ANNOTATOR_SCIENTIST.md](ANNOTATOR_SCIENTIST.md) for the new strict two-reviewer rubric, conflict handling, source-level splits, gold SFT/DPO JSONL, data-science diagnostics and pretraining validation.
