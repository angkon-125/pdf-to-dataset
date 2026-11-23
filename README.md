# Angkon Dataset Factory v0.3

**PDF-to-dataset, local-first, with page-level provenance and human annotation review.**

Convert PDFs containing books, research papers, reports, manuals or notes into reviewable question/answer data, retrieval chunks, and human-audited training samples. The system works in extraction-only mode without an AI model or paid API. It can optionally use Ollama or a local llama.cpp/OpenAI-compatible endpoint to generate grounded **draft** Q&A candidates.

> **Not an automatic truth verifier.** A copied quote may appear in a document without supporting the generated answer. Always review candidates, training rights, and privacy before training or publishing datasets.

## NEW in v0.3: Human Annotator & Data Scientist Lab

**Do not treat an LLM-generated answer as a training label.** v0.3 adds a *second*, stricter release stage to the existing document-extraction workflow. You can continue using all v0.2 extraction/model-generation features.

1. **Generate draft Q&A** with `app.py` or the PDF CLI. All Q&A starts as `needs_review`.
2. **Edit inaccuracies first** (optional) using the existing CSV review workflow, then re-export a corrected source dataset ZIP. Edits invalidate old rubric signatures.
3. **Two independent human annotators** use `curator_app.py` or CSV templates. They check the original PDF, exact excerpt, answer and question, then each independently score: factual accuracy, grounding, instruction following, completeness, clarity and safety (0-3) and give a specific rationale.
4. **Data scientist curation** validates content hashes, rejects conflicting decisions, requires evidence verification, checks structural errors, de-duplicates prompts, quarantines detected cross-split near-duplicates, measures agreement, reviews source/language/task distributions, and creates split-aware exports.
5. **Gold release** contains only examples meeting all gates. A `gold_dpo.jsonl` preference dataset is possible **only** if a human supplies and judges a genuine alternate rejected answer; the system never invents negative answers.
6. **Pretraining validator** inspects the curated ZIP for SFT/DPO schema correctness, provenance, split consistency, duplicate prompts and basic PII signals.

### Visual workbench (recommended)

```powershell
cd 'E:\DOWNLOADS\angkon_dataset_factory'
.\.venv\Scripts\python.exe -m streamlit run curator_app.py
```

Upload an Angkon-generated ZIP in the workbench; review the source and answers, download reviewer-specific CSVs, and feed completed files into the Gold Release tab.

### Offline expert workflow (PowerShell)

The main CLI accepts `inspect`, `curate`, and `validate` subcommands in addition to the original PDF pipeline:

```powershell
cd 'E:\DOWNLOADS\angkon_dataset_factory'
$py = '.\.venv\Scripts\python.exe'

# First generate draft_qa.zip using a local Qwen/Ollama model in the main app.
& $py -m angkon_factory.cli inspect '.\draft_qa.zip' --output-dir '.\review_workbench'

# Two humans independently fill in a copy of rubric_template.csv:
Copy-Item '.\review_workbench\rubric_template.csv' '.\review_workbench\reviewer_a.csv'
Copy-Item '.\review_workbench\rubric_template.csv' '.\review_workbench\reviewer_b.csv'
# Open each CSV in Excel or a text editor, *actually evaluate the originals* and fill it out.

& $py -m angkon_factory.cli curate '.\draft_qa.zip' `
  --reviewer-csv '.\review_workbench\reviewer_a.csv' `
  --reviewer-csv '.\review_workbench\reviewer_b.csv' `
  --rights-confirmed --attest-human-reviewed `
  --output '.\angkon_gold_release.zip'

& $py -m angkon_factory.cli validate '.\angkon_gold_release.zip'
```

**Blank templates do not approve anything.** Each reviewer CSV needs `reviewer_id`, `decision=approve/reject/revise`, `evidence_verified=yes/no`, six 0-3 scores for approvals, and a meaningful `rationale`. Reviewers must be real people making independent judgments; the app does **not** authenticate their identities.

**Gold approval thresholds:** 2 independent positive reviews, actual evidence support confirmed by both, factual accuracy **3**, grounding **3**, safety **3**, and instruction adherence, completeness, clarity each **at least 2**. If an answer is incorrect or disputed, fix it and get new reviews; don't outvote a dissenting annotator automatically.

### Training file contract

| File in `angkon_gold_release.zip` | Recommended use |
| --- | --- |
| `gold_sft_train.jsonl` | **TRAIN** your instruction-following model (HF TRL conversation `messages`) |
| `gold_sft_validation.jsonl` | Holdout used during model selection, not optimization |
| `gold_sft_test.jsonl` | Final held-out evaluation (do not train on this) |
| `gold_prompt_completion.jsonl` | Alternate full-corpus prompt/completion dataset; includes heldout and must be split before training |
| `gold_sft.jsonl` | Full corpus for auditing only; **do not train on all data** when using held-out splits |
| `gold_dpo_train.jsonl` | DPO training **only if** human-labeled alternative responses exist |
| `gold_dpo_validation.jsonl` / `gold_dpo_test.jsonl` | Held-out preference evaluation |
| `gold_provenance.jsonl` | Reproducibility, PDF hash/page, quote, reviewers, record fingerprint |
| `data_science_quality.json` | Quality report, agreement, leakage checks, source balance |
| `review_decisions.jsonl` | Why records passed/failed gold gates |

HF TRL reference schemas: https://huggingface.co/docs/trl/sft_trainer and https://huggingface.co/docs/trl/dpo_trainer. No fine-tuning is performed by the dataset factory. Start model experiments with a small reviewed dataset and an independent evaluation benchmark; additional examples are useful only when genuinely accurate and diverse.

See [the full professional annotation guide](docs/ANNOTATOR_SCIENTIST.md).

---

## v0.2 extraction and generation foundation

- **Multiple PDFs** in one run, with SHA-256 deduplication and per-source/page references.
- **Chapter/section metadata** from PDF outlines where available, with heading fallback.
- **Optional table extraction** using PyMuPDF (heuristic; requires visual inspection).
- **Optional OCR** for pages with little extractable text (Tesseract installed separately).
- **Bangla/English script hints**, book/paper/report/manual heuristic classification; optional LLM document classification.
- **Low-RAM LLM generation** with configurable prompt input size, output limit, and `spread` chunk sampling across the whole document.
- **Local response cache/resume** for successful model calls; no repeated inference after the same prompt is cached.
- **Dataset quality flags** for evidence mismatch, invalid schemas, duplicates, possible PII, missing reviewers and provenance mismatch.
- **Reviewable CSV + Streamlit editor**, offline CSV re-import, optional audit events.
- **Approved SFT only** after manual review and a training-rights attestation; no fabricated preference/DPO pairs.
- **Source-grouped splits** (train/validation/test) to reduce leakage between chunks from the same PDF.
- **Dataset card, quality report, source metadata, and full extraction coverage information.**
- **Tests + GitHub Actions CI**, PowerShell setup/publishing helpers.

## Start on Windows (PowerShell)

1. Download and extract `Angkon_Dataset_Factory_v0.3.zip` into `E:\\DOWNLOADS`.
2. Open PowerShell inside `E:\\DOWNLOADS\\angkon_dataset_factory`.

```powershell
cd 'E:\DOWNLOADS\angkon_dataset_factory'
powershell -ExecutionPolicy Bypass -File '.\scripts\install_windows.ps1'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` if your browser does not launch automatically.

You can also install manually:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Python 3.10+ recommended. You do **not** need to buy new hardware for extraction-only mode; some heavy model workloads may be impractical on limited CPUs/RAM.

## CLI: offline extraction (no AI needed)

```powershell
.\.venv\Scripts\python.exe -m angkon_factory.cli '.\examples\demo.pdf' --output 'angkon_dataset.zip'
```

Several PDF files in one command:

```powershell
.\.venv\Scripts\python.exe -m angkon_factory.cli 'E:\DOWNLOADS\Book1.pdf' 'E:\DOWNLOADS\Book2.pdf' --output 'combined_dataset.zip' --tables
```

For scanned PDFs: install Tesseract and suitable language packs (`eng`, `ben`) separately, then add `--ocr --ocr-language eng+ben`.

## CLI: local model generation, low-memory configuration

### Ollama

Install and start Ollama separately, download a model that fits your PC, and specify the **exact model ID you installed**. This example uses a small model ID only if available on your Ollama setup:

```powershell
.\.venv\Scripts\python.exe -m angkon_factory.cli '.\examples\demo.pdf' `
  --backend ollama --endpoint 'http://127.0.0.1:11434' --model 'qwen3:0.6b' `
  --chunk-chars 900 --max-source-chars 650 --examples-per-chunk 1 `
  --max-chunks 10 --max-output-tokens 350 --rights-confirmed --output 'draft_qa.zip'
```

### llama.cpp OpenAI-compatible server (useful with existing locally hosted Qwen)

If you already run llama.cpp server, confirm its actual host/port and model ID. Example:

```powershell
.\.venv\Scripts\python.exe -m angkon_factory.cli '.\examples\demo.pdf' `
  --backend openai_compatible --endpoint 'http://127.0.0.1:8080/v1' --model 'local-model' `
  --chunk-chars 900 --max-source-chars 650 --examples-per-chunk 1 `
  --max-chunks 10 --max-output-tokens 350 --rights-confirmed --output 'draft_qa.zip'
```

These settings are a **starting point** for small context windows (such as 2,048 tokens); they do not guarantee every model accepts or correctly completes the prompt. Check warnings and model server logs. Using `--max-chunks 0` asks the model to process all extracted chunks, possibly involving hundreds of slow CPU calls. The extraction-only dataset covers all **processed pages** regardless of model sampling.

### OpenAI-compatible remote API

Use `--backend openai_compatible --endpoint <provider-api-base> --model <provider-model>`; provide an API key through an environment variable such as `OPENAI_API_KEY` if required. **Source text is transmitted to that provider.** Do not send sensitive or copyrighted material without appropriate rights/permissions and privacy review.

## Human review and approved SFT export

Model-generated records are always unreviewed drafts. To approve offline:

1. Extract the generated `draft_qa.zip` and open `annotations_review.csv` in a spreadsheet.
2. Check each answer, original PDF page and evidence quotation. Correct errors or mark the row `rejected`.
3. For truly verified records set `review_status` to `approved` and enter a nonempty `reviewer`.
4. Save the edited CSV as UTF-8.
5. Re-import without re-running the model:

```powershell
.\.venv\Scripts\python.exe -m angkon_factory.cli review `
  'draft_qa.zip' 'annotations_review.csv' --output 'approved_dataset.zip' `
  --attest-reviewed --rights-confirmed
```

Only approved rows passing mechanical validation appear in `sft_approved.jsonl`. **Human approval does not make source claims inherently true**: the reviewer must check evidence and context. Training-rights attestation is not independent legal verification.

## What is inside an output ZIP?

| Output | Purpose |
| --- | --- |
| `manifest.json` | SHA-256 source IDs, counts, pipeline flags, warning details |
| `quality_report.json` | Coverage, validation-flag distribution, types/languages |
| `dataset_card.md` | Intended uses, sources, model limitations, ethics |
| `annotation_guidelines.md` | Reviewer checklist and approval guidance |
| `source_documents.json` | Source metadata for reproducible re-review |
| `source_pages.jsonl` | Original extracted text (or regex-masked copy) by PDF page |
| `rag_chunks.jsonl` | Text chunks with source page/section/provenance |
| `tables.jsonl` | Optional heuristic table rows and page/region references |
| `qa_candidates.jsonl` | Draft QA candidates with evidence quotes and review status |
| `annotations_review.csv` | Spreadsheet annotation/review queue |
| `audit_log.jsonl` | Review-edit events, with reviewer IDs and timestamps |
| `sft_candidates.jsonl` | Unreviewed chat-style examples; **NOT** approved training data |
| `sft_approved.jsonl` | Only approved, validated, rights-attested chat messages |
| `sft_train.jsonl`, `sft_validation.jsonl`, `sft_test.jsonl` | Source-grouped split of approved SFT |
| `extracted_text.txt` | Human-readable extracted source, by document and page |

With fewer than three unique PDFs, only `sft_train.jsonl` is populated; splitting pages of one book across training and testing would leak source content.

## Privacy, copyright, and local storage

- The **original PDF is not included** in exports. Text extracted from it **is included** and may still be confidential.
- `--redact-pii` performs only **partial regex-based masking** of email addresses and common Bangladesh mobile formats in extracted text/tables. It does not scrub embedded images, other identifiers or all metadata. Do not rely on it for production privacy compliance.
- Generating training examples from a copyrighted book does not grant any right to reuse, redistribute, or commercially train on its content.
- The optional local generation cache is stored in `runs/cache/` as unencrypted JSON; it is ignored by Git. Delete it when no longer needed.
- `--rights-confirmed` and `--attest-reviewed` are **user acknowledgments**, not an external verification system.
- Do not deploy this Streamlit prototype as an unsecured public file-upload service. Production requires access controls, process isolation, input limits, retention policies, secret management, and endpoint allowlisting.

## Limits and common issues

- Hard limits: **60 MB per PDF, 1,500 pages per PDF, 25 PDFs per batch, 120 MB combined**. Streamlit starts with 300 pages per PDF by default; increase this for full-book extraction. The CLI defaults to 1,500.
- PDFs with scanned pages require Tesseract OCR; install `ben` for Bangla scans.
- Tables, formulas, images, and two-column text may extract incorrectly; check source pages visually.
- Local models can return malformed JSON or fail under a small context window; see warnings. Reduce `--max-source-chars`, `--examples-per-chunk`, or `--max-output-tokens` and retry.
- `--max-chunks` caps *generation calls per PDF*, **not** extraction coverage.
- Book chapter detection is based on PDF bookmarks/outlines when available, otherwise basic headings.
- It does not fabricate DPO labels or simulate human preference decisions.

## GitHub publishing (your repository)

Target: `https://github.com/angkon-125/pdf-to-dataset.git`.

1. Extract the v0.3 ZIP under `E:\DOWNLOADS` to `E:\DOWNLOADS\angkon_dataset_factory`.
2. Clone or update `E:\DOWNLOADS\pdf-to-dataset` and copy **code only** into it.
3. Commit and push. `scripts/publish_github.ps1` checks the remote and never force pushes. See the PowerShell commands in the accompanying ChatGPT answer.

**Do not** upload `.env` credentials, local model weights, private PDFs, generated copyrighted text, or `runs/cache` to GitHub. This repository contains a small original demonstration PDF for testing.

## Developers

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for design and technical guarantees. No software license has been selected for the project; choose one explicitly before distributing the source under an open-source license.

## One-script GitHub update (v0.3)

After extracting the ZIP into `E:\DOWNLOADS`, run:

```powershell
cd 'E:\DOWNLOADS\angkon_dataset_factory'
powershell -ExecutionPolicy Bypass -File '.\scripts\deploy_update_windows.ps1'
```

The script clones or fast-forward-updates `https://github.com/angkon-125/pdf-to-dataset.git`, copies code and the synthetic demo PDF, blocks common private dataset/file extensions from staging, and commits/pushes without a forced update. It stops if the local Git checkout is dirty. Always review files before publishing; scripts cannot guarantee that all private information is excluded.
