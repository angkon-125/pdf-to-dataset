"""Angkon Dataset Factory - local Streamlit user interface.

Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import uuid

import pandas as pd
import streamlit as st

from angkon_factory.core import MAX_PAGES, MAX_PDF_BYTES, extract_pdf, make_chunks
from angkon_factory.exports import build_zip, revalidate_annotations
from angkon_factory.generation import LLMClient, generate_candidates, classify_document

st.set_page_config(page_title="Angkon Dataset Factory", page_icon="📚", layout="wide")
st.title("Angkon Dataset Factory")
st.caption("PDF → document detection → extraction/OCR → chunking → LLM candidates → human review → dataset ZIP")

with st.sidebar:
    st.header("Pipeline configuration")
    backend_label = st.selectbox("Processing mode", ["Extraction only (offline)", "Ollama (local)", "OpenAI-compatible API"], index=0)
    backend = {"Extraction only (offline)": "none", "Ollama (local)": "ollama", "OpenAI-compatible API": "openai_compatible"}[backend_label]
    if backend == "ollama":
        model = st.text_input("Ollama model", value="qwen2.5:7b")
        endpoint = st.text_input("Ollama server URL", value="http://localhost:11434")
        api_key = ""
        st.caption("You must already have Ollama running and the selected model downloaded.")
    elif backend == "openai_compatible":
        model = st.text_input("Model ID", value="gpt-4.1-mini")
        endpoint = st.text_input("API base URL (include /v1)", value="https://api.openai.com/v1")
        api_key = st.text_input("API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        st.warning("Remote inference transmits document text to the selected API provider. Review privacy and provider terms.")
    else:
        model, endpoint, api_key = "", "", ""
    st.divider()
    use_ocr = st.checkbox("Try OCR for scanned pages", value=False)
    ocr_language = st.text_input("Tesseract OCR language(s)", value="eng+ben" if use_ocr else "eng", disabled=not use_ocr)
    st.caption("OCR requires the Tesseract executable and language packs; install ben for Bangla.")
    page_limit = st.number_input("Page limit", min_value=1, max_value=MAX_PAGES, value=300, step=50)
    chunk_chars = st.slider("Chunk size (characters)", 600, 5000, 2300, 100)
    if backend != "none":
        examples_per_chunk = st.slider("Examples per chunk", 1, 6, 2)
        max_chunks = st.number_input("LLM generation limit (0 = all extracted chunks)", min_value=0, value=15, step=5)
        st.caption("Start with 15 to control model calls. Use 0 for full coverage after testing quality/cost.")
    else:
        examples_per_chunk, max_chunks = 0, 0

st.subheader("1. Upload a source PDF")
uploaded = st.file_uploader("Upload a book, article, guide, or report as PDF", type=["pdf"])
st.caption(f"Local MVP limits: {MAX_PDF_BYTES // (1024*1024)} MB per PDF and {MAX_PAGES:,} pages. Original PDF is not included in exported ZIP.")
authorized = st.checkbox("I have the rights or permission needed to process this file and use any resulting dataset for my intended purpose.")
if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("source_digest") != digest:
        st.session_state.source_digest = digest
        st.session_state.pop("dataset_run", None)

can_process = uploaded is not None and authorized
if st.button("Run workflow", type="primary", disabled=not can_process):
    try:
        with st.spinner("Extracting PDF and detecting document properties..."):
            document = extract_pdf(pdf_bytes, uploaded.name, use_ocr=use_ocr,
                                   ocr_language=ocr_language, page_limit=int(page_limit))
            chunks = make_chunks(document, target_chars=chunk_chars)
        if not chunks:
            st.error("No usable content was extracted. Try enabling OCR and checking installed Tesseract languages.")
        else:
            candidates, model_warnings, attempted = [], [], 0
            if backend != "none":
                client = LLMClient(backend, model, endpoint, api_key=api_key)
                with st.spinner("Classifying document and selecting annotation focus..."):
                    profile, profile_warnings = classify_document(chunks, client)
                    document["llm_profile"] = profile
                    model_warnings.extend(profile_warnings)
                    chunks = make_chunks(document, target_chars=chunk_chars)
                attempted = len(chunks) if max_chunks == 0 else min(int(max_chunks), len(chunks))
                bar = st.progress(0, text="Generating source-grounded annotation candidates...")
                def update_progress(done: int, total: int) -> None:
                    bar.progress(done / max(total, 1), text=f"Generating: {done} / {total} chunks")
                candidates, generation_warnings = generate_candidates(
                    chunks, client, examples_per_chunk=examples_per_chunk,
                    max_chunks=int(max_chunks), progress=update_progress,
                )
                model_warnings.extend(generation_warnings)
                bar.empty()
            st.session_state.dataset_run = {
                "document": document, "chunks": chunks, "candidates": candidates,
                "backend": backend, "model": model, "warnings": model_warnings,
                "attempted": attempted,
                "run_id": uuid.uuid4().hex[:8],
            }
            st.success("Workflow finished. Inspect the results and download a dataset archive below.")
    except Exception as exc:
        st.error(f"Workflow could not finish: {type(exc).__name__}: {exc}")

run = st.session_state.get("dataset_run")
if run:
    doc, chunks, candidates = run["document"], run["chunks"], run["candidates"]
    st.divider()
    st.subheader("2. Detection and extraction report")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed pages", f"{doc['processed_pages']} / {doc['page_count']}")
    col2.metric("Extracted chunks", len(chunks))
    col3.metric("OCR pages", doc["scanned_pages_ocr"])
    col4.metric("Generated candidates", len(candidates))
    st.write(f"**Detected language (heuristic):** {doc['language_heuristic']}")
    st.write(f"**Detected document type (heuristic):** {doc['document_type_heuristic']}")
    if doc.get("llm_profile"):
        st.write(f"**LLM classification:** {doc['llm_profile']['document_type']} (subject: {doc['llm_profile']['subject']})")
    if doc["warnings"] or run["warnings"]:
        with st.expander("Extraction/generation limitations and warnings", expanded=False):
            for msg in doc["warnings"] + run["warnings"]:
                st.warning(msg)
    chunk_names = [f"Page {c['source_page']} · {c['chunk_id']}" for c in chunks]
    idx = st.selectbox("Inspect extracted source chunk", range(len(chunks)), format_func=lambda i: chunk_names[i])
    st.text_area("Source text preview", chunks[idx]["text"], height=200, disabled=True)

    reviewed = candidates
    if candidates:
        st.subheader("3. Review and correct candidate examples")
        st.write("All model-generated examples start as **needs_review**. Verify correctness, source support, and safety before approval.")
        reviewer_name = st.text_input("Reviewer label", value="local_reviewer")
        columns = ["annotation_id", "source_page", "kind", "question", "answer", "evidence_quote",
                   "validation_flags", "review_status", "review_note"]
        table = pd.DataFrame([{k: "; ".join(item.get(k, [])) if k == "validation_flags" else item.get(k, "")
                               for k in columns} for item in candidates])
        edited = st.data_editor(
            table, hide_index=True, use_container_width=True, num_rows="fixed", key=f"review_editor_{run['run_id']}",
            column_order=columns,
            disabled=["annotation_id", "source_page", "kind", "validation_flags"],
            column_config={
                "review_status": st.column_config.SelectboxColumn("Review status", options=["needs_review", "approved", "rejected"], required=True),
                "question": st.column_config.TextColumn("Question", width="large"),
                "answer": st.column_config.TextColumn("Answer", width="large"),
                "evidence_quote": st.column_config.TextColumn("Exact evidence quote", width="large"),
            },
        )
        manual_check = st.checkbox("I have manually checked every item I marked approved against its original source and licensing/privacy requirements.")
        reviewed = []
        for original, row in zip(candidates, edited.to_dict("records")):
            item = dict(original)
            for field in ("question", "answer", "evidence_quote", "review_status", "review_note"):
                value = row.get(field, "")
                item[field] = "" if pd.isna(value) else str(value)
            if item["review_status"] == "approved" and not manual_check:
                item["review_status"] = "needs_review"
            item["reviewer"] = reviewer_name if item["review_status"] != "needs_review" else ""
            reviewed.append(item)
        reviewed = revalidate_annotations(reviewed, chunks)
        st.caption("Review a candidate using its source page and chunk ID. A matching quote alone cannot establish factual correctness.")

    st.subheader("4. Export datasets")
    bundle, manifest = build_zip(
        doc, chunks, reviewed, backend=run["backend"], model=run["model"],
        generation_warnings=run["warnings"], chunks_attempted=run["attempted"]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Approved SFT examples", manifest["approved_examples"])
    c2.metric("Awaiting review", manifest["unreviewed_examples"])
    c3.metric("Rejected", manifest["rejected_examples"])
    st.download_button("Download Angkon dataset ZIP", data=bundle,
                       file_name=f"angkon_dataset_{doc['source_id'][:10]}.zip",
                       mime="application/zip", type="primary")
    st.caption("Archive contains rag_chunks.jsonl, qa_candidates.jsonl, review CSV, SFT candidates, approved-only SFT, extracted text, and processing manifest.")
else:
    st.info("Choose a processing mode, upload a PDF, confirm usage rights, and select **Run workflow**. Extraction-only mode works without a model.")

st.divider()
st.caption("Local prototype. Do not expose publicly without file isolation, access controls, endpoint allowlisting, resource quotas, and privacy/security review.")
