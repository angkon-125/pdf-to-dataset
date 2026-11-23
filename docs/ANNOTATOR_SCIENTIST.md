# Angkon v0.3 Professional Annotation & Data Science Playbook

## Purpose and limits

Turn a document-derived **draft** response into a source-grounded **human-reviewed training example**. This process helps reduce training-data errors but does not certify truth, permission, fairness, or model performance. Only human annotators can certify that they personally examined evidence. A local LLM may propose questions and answers; it **cannot** impersonate human reviewers.

The workbench is intended for a local single-user process where each annotator exports a signed-to-content review CSV. It does **not** authenticate user identity or implement enterprise-style access controls. Store private datasets securely and avoid uploading archives or reviewer files to public repositories.

## Roles and separation of duties

| Role | Responsibilities | Deliverable |
| --- | --- | --- |
| Document ingestion operator | Check rights and privacy, run extraction/OCR, inspect pages and tables | `source_pages.jsonl`, `rag_chunks.jsonl`, manifest |
| LLM draft generator | Propose source-grounded questions and answers, never auto-approve | `qa_candidates.jsonl` |
| Editorial annotator | Fix inaccurate/missing context or reject unrecoverable examples | Corrected `annotations_review.csv` / archive |
| Independent human annotator A | Read original PDF page, score rubric and evidence support without consulting B | Reviewer-A CSV |
| Independent human annotator B | Repeat independent verification | Reviewer-B CSV |
| Data scientist / quality lead | Diagnose distributions, duplicates, leakage, disagreement, risk and select heldout source groups | `data_science_quality.json`, verified split files |
| Model engineer | Validate formats, train only on train split, compare against untouched test baseline | Training logs and evaluation results |

A single person can practice multiple roles on non-production data, but **two independent human judgments** are required to call a release `gold` in this software. Reviewer ID strings are assertions, not proof of identity or independence.

## Annotator skill: task-by-task protocol

1. **Inspect the document, not just the generated snippet.** Check page number, headings, OCR quality and any relevant diagram/table; missing context can invalidate a seemingly plausible response.
2. **Understand the prompt.** Identify its actual question, implicit assumptions and required length/format.
3. **Decompose the answer into verifiable claims.** Mark omissions, unsupported facts, numbers and definitions; distinguish source-reported statements from facts established independently.
4. **Check quoted evidence semantically.** The quoted string must be present, relevant and sufficient to support the answer; mere substring matching is insufficient.
5. **Assess integrity and safety.** Avoid unnecessary personal identifiers, copyrighted reproductions, bias, harmful instructions and confidential source details.
6. **Score six dimensions independently.** Use the rubric below rather than a general impression of fluency.
7. **Write a rationale.** Explain why evidence does or does not support the answer, and name specific issues; do not just write “good answer.”
8. **Choose a decision:** approve / revise / reject. If answer/source needs repair, mark revise, perform editorial correction, then get both annotators to review the *new fingerprint*.

### Human scoring rubric (0–3)

| Dimension | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Factual accuracy | Critical falsehood | Major incorrect claims | Mostly correct, minor issue | No identified incorrect claims within scope |
| Grounding | No source support | Relevant but inadequate evidence | Mostly supported, minor gap | Direct and complete source support |
| Instruction following | Disregards core task | Major constraints missed | Minor constraint issues | Fully follows applicable request |
| Completeness | Unusable or missing | Major sections missing | Covers most requirements | Complete for scope |
| Clarity | Unintelligible | Confusing | Understandable, polish needed | Clear, appropriate tone and length |
| Safety | Serious violation | Significant concern | Minor risk requiring review | No identified violation under project policy |

**Gold gate:** each of at least **two distinct human reviewers** must choose approve, confirm source evidence, enter an explanatory rationale, and score factual accuracy / grounding / safety **3**, all other dimensions **at least 2**. Any reject/revise/disagreement prevents gold export. More reviewers do not simply outvote a dissent; correct the example, generate a fresh template and review again. The gate is an internal quality threshold, not an industry standard.

### Corrections and immutable fingerprints

Every rubric row includes `record_sha256`, which is a SHA-256 fingerprint of source id, PDF page, chunk, question, answer, evidence quote and task kind. Editing any of those fields after review invalidates the review. Edits must precede rubric annotation; rerun `inspect` on the corrected archive before asking independent people to review it.

A SHA-256 fingerprint detects changes, **not** who actually filled out a CSV. Treat the reviewer identity fields as self-declared.

## Data scientist skill: dataset diagnostics

Evaluate the following before fine-tuning:

- **Scope:** total PDFs/pages, OCR-vs-digital coverage, empty pages, model-generated chunk coverage. A document represented by 2 of 1000 pages is not comprehensive.
- **Provenance:** every record references an original PDF hash, page number and chunk identifier. Quotes remain attached to their source.
- **Quality:** structural flags, duplicate normalized prompts, possible PII, missing source evidence, low rubric scores, disagreement rates and reviewer notes.
- **Diversity:** distribution across sources, languages, question kinds, subject domains, and answer lengths. A single document or repetitive template can dominate training.
- **Privacy:** simple regex PII checks are not enough. Human/legal review of confidential text, non-BD identifiers and embedded images is needed.
- **Leakage:** group train/validation/test by source PDF. Check cross-source duplicate/near-duplicate prompts and shared documents; semantic near-duplicates may remain undetected.
- **Train-vs-test boundaries:** train only on `gold_sft_train.jsonl` or `gold_dpo_train.jsonl`. Avoid `gold_sft.jsonl` and `gold_prompt_completion.jsonl` if you are maintaining a separate test set; those contain all splits.
- **Agreement:** disagreement rate and exact score agreement help locate inconsistent labels but do not prove annotators are correct. Investigate examples with divergent rationales.
- **Task fit:** choose a representative holdout benchmark of problems the final LLM must solve; measure baseline performance, fine-tuned performance, and regressions.
- **Scaling:** the built-in approximate cross-split check is *bounded* and text-based. For large corpora, add MinHash/LSH or embedding-based clustering plus manual spot checks. No tool can guarantee zero contamination.

### Release decision

1. All human reviews are independent and fingerprints match current content.
2. Every exported gold example passes the strict rubric gate and existing source/quote validation.
3. Any label conflict, PII, invalid schema, or cross-source near duplicate flagged by the workflow is resolved or excluded.
4. Rights and processing permissions have been explicitly confirmed by the operator (no external validation).
5. Holdout separation is meaningful: with fewer than three source PDFs, no independent validation/test groups are generated.
6. Dataset report is inspected; sample sizes and distributions match the model's intended use.
7. A baseline evaluation exists before training, and gold/test data are not included in model training or prompt selection.

## Genuine human preference (DPO) workflow

DPO needs a **real** contrast for the **same prompt**. Generate or collect an alternate completion from a model/human process. Human reviewers compare it with the source-verified chosen answer, inspect its specific errors, and enter:

- `rejected_answer` = actual alternate completion, **not** an automatically invented negative;
- `preference_reason` = why the chosen answer is preferable;
- `review_status=approved`, `evidence_verified=yes`, named reviewer;
- exact original `record_sha256` and `chosen_answer` from the template.

The software includes a preference pair only when its chosen answer is independently gold-approved. It rejects identical alternatives and common PII signals, but does not verify the alternate answer's generation origin. DPO outputs use explicit conversational `prompt`, `chosen`, `rejected` keys as expected by modern [Hugging Face TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer). Without a human-supplied alternative, DPO files remain empty. Never re-label a random correct SFT example as a rejected answer.

## SFT training workflow (outside Angkon)

- Load the **training split** with Hugging Face Datasets, e.g. `load_dataset("json", data_files="gold_sft_train.jsonl")`.
- Use the model's tokenizer and chat template; confirm the trainer's expected conversational schema and loss masking. See [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer).
- Start with a small sanity run. Test chat template, token counts, truncation, multilingual strings and source-label distribution before a full fine-tune.
- Evaluate against *held-out* source PDF(s), including test cases manually reviewed for correctness and safety.
- Track dataset release hash, model ID, tokenizer, trainer version, hyperparameters and random seed. Compare before/after performance; do not assume high scores mean the model improved.
- On low-memory Windows PCs, use local models for **inference**/draft generation and move actual fine-tuning to a suitable GPU environment when needed. This dataset-factory release does not install PyTorch or fine-tune models.

## Command checklist (Windows)

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m angkon_factory.cli inspect '.\draft_qa.zip' --output-dir '.\review_workbench'
# Two humans independently complete separate copies of the rubric template
& $py -m angkon_factory.cli curate '.\draft_qa.zip' `
  --reviewer-csv '.\review_workbench\reviewer_a.csv' `
  --reviewer-csv '.\review_workbench\reviewer_b.csv' `
  --rights-confirmed --attest-human-reviewed -o '.\gold.zip'
& $py -m angkon_factory.cli validate '.\gold.zip'
```

## Threat model and non-goals

- PDF text can contain prompt-injection instructions; extraction/model-generation uses a restricted template but no isolation guarantee.
- OCR diagrams and equations can lose essential information; visually inspect the original.
- The workflow does not automatically prove legal training rights or remove all PII.
- Reviewers may be mistaken, collude, or use the same account. The local app provides **no authentication** and is not a regulated audit product.
- Reviewer CSVs and exported ZIPs are unencrypted. Store and transfer them with appropriate controls.
- This is a dataset preparation pipeline, **not** a replacement for domain experts or independent model evaluation.
