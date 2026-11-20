# Angkon Dataset Factory (MVP)

**PDF -> heuristic detection -> text/OCR -> chunking -> optional LLM classification -> annotation generation -> validation -> human approval -> dataset ZIP**

A runnable, local-first prototype for preparing traceable LLM annotation datasets from PDFs.
It produces **draft** annotation/SFT records for human review, not automatically certified training data.

## Quick start (local UI)

Python 3.10+ recommended.

```bash
python -m venv .venv
# Activate the environment for your operating system, then:
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit address shown in your terminal. Choose **Extraction only** to build a RAG/reading dataset without an LLM. To generate grounded Q&A candidates, choose a configured model backend.

## Optional local LLM with Ollama

Install Ollama separately and pull a model suitable for your hardware:

```bash
ollama pull qwen2.5:7b
ollama serve
```

If Ollama is already running, you do not need a second `ollama serve`. In the app, select **Ollama (local)** and confirm the model name and local URL.

For cloud or OpenAI-compatible endpoints, choose **OpenAI-compatible API** and enter its base URL, model ID, and API key. This **sends source content to that endpoint**. Only do so when authorized.

## Optional scanned-PDF OCR

Install the **Tesseract** executable separately, and make it accessible on your PATH. Install its `eng` English trained data, and `ben` if processing Bengali scans. Check available packs with `tesseract --list-langs`. Enable **Try OCR for scanned pages** in the app. Image-only pages otherwise lack extractable text.

The OCR feature uses PyMuPDF's Tesseract integration. Some deployments require configuring `TESSDATA_PREFIX`. OCR quality varies; review the resulting text before training.

## CLI (works in notebooks / Kaggle too)

Extraction-only workflow:

```bash
python -m angkon_factory.cli examples/demo.pdf --output dataset.zip
```

With local LLM generation:

```bash
python -m angkon_factory.cli examples/demo.pdf --output dataset.zip \
  --backend ollama --model qwen2.5:7b --endpoint http://localhost:11434 \
  --examples-per-chunk 2 --max-chunks 0
```

OpenAI-compatible endpoint (API key from environment):

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
python -m angkon_factory.cli book.pdf --backend openai_compatible \
  --endpoint https://api.openai.com/v1 --model gpt-4.1-mini --max-chunks 15
```

Kaggle does not automatically provide an Ollama server or a GPU inference service. If `localhost:11434` is unavailable, use an explicitly configured endpoint or extraction-only mode. Installation and network policies differ by environment.

Use `--ocr --ocr-language eng+ben` for scanned sources where both Tesseract language packs are installed. Increase `--page-limit` for longer books. `--max-chunks 0` attempts generation for **all extracted chunks** (may incur significant time/cost); extraction runs across all processed pages regardless.

## Included runnable examples

`examples/demo.pdf` is a small original test document. `examples/demo_output.zip` shows offline extraction; `examples/demo_candidates.zip` contains two **synthetic fixture** Q&A records to demonstrate the draft-review format (not produced by a real model).

## Archive outputs

| Filename | What it contains |
| --- | --- |
| `manifest.json` | PDF hash, type/language heuristics, processed pages, coverage, model, counts, warnings |
| `rag_chunks.jsonl` | Page-referenced, chunked extracted text for retrieval/analysis |
| `extracted_text.txt` | Extracted page text, for manual checking |
| `qa_candidates.jsonl` | Model-generated Q&A, evidence quotes, flags, and review status |
| `annotations_review.csv` | Review-friendly, spreadsheet-ready annotation records |
| `sft_candidates.jsonl` | Draft chat-style examples, **not approved by default** |
| `sft_approved.jsonl` | Only manually approved, structurally validated chat examples |
| `README_DATASET.txt` | Guide and limitations bundled with each output |

**Note:** The CLI produces only draft annotations. For manual approval, use the UI and export the updated ZIP. No DPO dataset is generated from a single PDF: DPO requires real response comparisons and preference judgments, not fabricated "rejected" answers.

## How the checks work

- Page number, chunk ID, and SHA-256 of the input PDF are attached to data.
- Generated `evidence_quote` must match an exact substring of its source chunk (after whitespace normalization).
- Missing/invalid evidence, malformed answers, invalid kinds, and duplicate questions are flagged.
- Human review is required to check actual meaning, truthfulness, safety, licensing, and privacy.
- Source instructions are explicitly treated as untrusted document text, not commands for the model.

**An exact evidence quote is not a semantic entailment guarantee.** A fabricated answer can quote a real sentence. The prototype does not claim to automatically verify factual correctness.

## Limitations and production work

This is an MVP, not a production multi-user service:

1. PDF input only; 60 MB / 1500-page hard ceilings. The UI defaults to processing at most 300 pages; configure for longer inputs.
2. Language detection is heuristic. Document type detection starts with **simple heuristics** and may be refined by an LLM on sampled snippets; neither is guaranteed correct. Complex multilingual layouts need better parsing/classification.
3. PDF text order may be incorrect for tables, two-column layouts, mathematical formulas, figures, and diagrams. These need layout-aware extraction, specialized table processing, or vision models.
4. OCR requires system dependencies and may introduce mistakes, especially with low-quality scans or Bengali material.
5. Generation may fail on long or poorly extracted chunks; model errors are recorded in the manifest.
6. The UI performs manual review but is **not** a full audit system. Production should add access control, audit trails, persistent review state, row-level assignments, inter-annotator agreement, deduplication at scale, and reviewer calibration.
7. For multi-user production, isolate PDF processing in a constrained container, restrict arbitrary API endpoints to a safe allowlist, rate-limit work, implement storage retention, and scan uploaded files.
8. Only ingest and use materials with appropriate rights and permissions. **Extracting a book does not confer permission to redistribute text or use it to train a commercial model.** Respect user privacy and the API provider's terms.

## Run tests

```bash
python -m pytest -q
```

## Suggested next build phases

- Better PDF structure recognition: pages, tables, chapters, headings, page-local image/figure descriptions.
- Schema/task router: SFT, factual Q&A, RAG, entity extraction, classification, and multilingual tasks.
- Second-pass grounding verifier and disagreement queues with human adjudication.
- Batch jobs, resumable processing, job database, optional GPU worker, storage permissions.
- True RLHF/DPO comparison labeling via multiple model outputs and reviewer preferences.
