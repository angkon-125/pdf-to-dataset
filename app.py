"""Streamlit UI for Angkon Dataset Factory 0.2 (local-first, reviewed outputs)."""
from __future__ import annotations

import hashlib
import os
import uuid
from io import BytesIO

import pandas as pd
import streamlit as st

from angkon_factory.core import MAX_BATCH_BYTES, MAX_DOCUMENTS, MAX_PAGES, MAX_PDF_BYTES
from angkon_factory.exports import REVIEW_COLUMNS, apply_review_csv, build_audit_events, build_batch_zip, csvtext, revalidate_annotations
from angkon_factory.generation import LLMClient
from angkon_factory.pipeline import PipelineOptions, run_batch

st.set_page_config(page_title="Angkon | Dataset Factory", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1380px; padding-top: 2.3rem;}
.main .block-container h1 {letter-spacing: -.035em;}
[data-testid="stMetric"] {padding: .75rem; border: 1px solid rgba(125,125,125,.2); border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("Angkon Dataset Factory")
st.caption("v0.2 · PDF books, research papers and reports → auditable RAG chunks + draft QA + reviewer-approved SFT")

with st.sidebar:
    st.header("Processing settings")
    mode = st.selectbox("Generation provider", ["Extraction only", "Ollama (local)", "OpenAI-compatible (llama.cpp / API)"], index=0)
    backend = {"Extraction only": "none", "Ollama (local)": "ollama", "OpenAI-compatible (llama.cpp / API)": "openai_compatible"}[mode]
    if backend == "ollama":
        model = st.text_input("Model ID", "qwen3:0.6b")
        endpoint = st.text_input("Ollama endpoint", "http://127.0.0.1:11434")
        key = ""
    elif backend == "openai_compatible":
        model = st.text_input("Model ID", "local-model")
        endpoint = st.text_input("API base URL (ending in /v1)", "http://127.0.0.1:8080/v1")
        key = st.text_input("API key (blank for local server)", os.getenv("OPENAI_API_KEY", ""), type="password")
        st.warning("If this is a remote provider, source text will leave your computer.")
    else:
        model, endpoint, key = "", "", ""
    st.divider()
    page_limit = st.number_input("Maximum pages per PDF", min_value=1, max_value=MAX_PAGES, value=300, step=50)
    chunk_chars = st.slider("RAG chunk size (characters)", 400, 5000, 1100, 100)
    ocr = st.checkbox("OCR scanned / image-only pages", value=False)
    ocr_lang = st.text_input("Installed Tesseract languages", "eng+ben" if ocr else "eng", disabled=not ocr)
    tables = st.checkbox("Extract detectable tables (slower)", value=False)
    redact = st.checkbox("Mask emails and Bangladesh mobile numbers (partial)", value=False)
    if backend != "none":
        st.divider()
        st.subheader("Low-memory generation")
        max_chunks = st.number_input("Chunks to generate from, per PDF (0 = all)", min_value=0, max_value=50000, value=12, step=5)
        count = st.slider("Questions per chunk", 1, 5, 2)
        source_chars = st.slider("Characters per model request", 350, 3500, 950, 50)
        sampling = st.selectbox("Chunk sampling", ["spread", "first"], index=0,
                                help="Spread selects pages across the whole document when limiting model calls.")
        llm_profile = st.checkbox("Ask model to classify document (extra call)", value=False)
        keep_cache = st.checkbox("Cache local model responses to resume", value=True)
        st.caption("For a 2048-token context, start with 650–950 characters and one question per chunk. Increase only after testing.")
    else:
        max_chunks, count, source_chars, sampling, llm_profile, keep_cache = 0, 1, 950, "spread", False, False

st.subheader("1 · Add source documents")
uploads = st.file_uploader("Upload PDF books, papers, reports, or manuals", type=["pdf"], accept_multiple_files=True)
st.caption(f"Up to {MAX_DOCUMENTS} PDFs per batch, {MAX_PDF_BYTES // (1024 * 1024)} MB each, {MAX_BATCH_BYTES // (1024 * 1024)} MB total. Start with one book on low-memory PCs. Original PDFs are never included in the export ZIP.")
authorized = st.checkbox("I confirm I have the necessary rights/permissions for this processing and my intended dataset use.")
if uploads:
    for f in uploads:
        st.caption(f"{f.name} · {len(f.getvalue()) / (1024*1024):.2f} MB")
    hashes = ":".join(hashlib.sha256(f.getvalue()).hexdigest() for f in uploads)
    if st.session_state.get("uploaded_hashes") != hashes:
        st.session_state["uploaded_hashes"] = hashes
        st.session_state.pop("dataset_run", None)

ready = bool(uploads and authorized and len(uploads) <= MAX_DOCUMENTS and all(len(f.getvalue()) <= MAX_PDF_BYTES for f in uploads) and sum(len(f.getvalue()) for f in uploads) <= MAX_BATCH_BYTES)
if st.button("Run PDF-to-dataset workflow", type="primary", disabled=not ready):
    try:
        client = None
        if backend != "none":
            client = LLMClient(backend, model, endpoint, api_key=key, timeout=240, max_output_tokens=550)
        settings = PipelineOptions(
            page_limit=int(page_limit), chunk_chars=int(chunk_chars),
            use_ocr=ocr, ocr_language=ocr_lang, extract_tables=tables,
            redact_pii=redact, max_chunks_per_document=int(max_chunks),
            examples_per_chunk=int(count), sampling=sampling,
            max_source_chars=int(source_chars), llm_profile=llm_profile,
            rights_attested=authorized,
            resume_dir="runs/cache" if keep_cache and client is not None else None,
        )
        info = st.progress(0.0, text="Starting PDF extraction...")
        def on_progress(idx: int, total: int, filename: str) -> None:
            info.progress(idx / max(total, 1), text=f"Processed {idx}/{total}: {filename}")
        output = run_batch(
            [(f.name, f.getvalue()) for f in uploads],
            options=settings, client=client, progress=on_progress,
        )
        info.empty()
        output["run_id"] = uuid.uuid4().hex[:10]
        st.session_state.dataset_run = output
        st.success("Extraction and generation completed. Review candidates before exporting training data.")
    except Exception as exc:
        st.error(f"Processing failed: {type(exc).__name__}: {exc}")

run = st.session_state.get("dataset_run")
if run:
    docs, chunks, candidates = run["documents"], run["chunks"], run["annotations"]
    st.divider()
    st.subheader("2 · Inspect quality and coverage")
    summary = [
        ("Documents", len(docs)),
        ("Pages extracted", sum(d["processed_pages"] for d in docs)),
        ("RAG chunks", len(chunks)),
        ("Draft Q&A", len(candidates)),
    ]
    columns = st.columns(4)
    for c, (label, value) in zip(columns, summary):
        c.metric(label, value)
    doc_df = pd.DataFrame([
        {
            "File": d["filename"], "Type (heuristic)": d["document_type_heuristic"],
            "Language (heuristic)": d["language_heuristic"],
            "Pages": f"{d['processed_pages']}/{d['page_count']}",
            "Pages without text": sum(not x["text"].strip() for x in d["pages"]),
            "OCR pages": d["scanned_pages_ocr"], "Tables": len(d.get("tables", [])),
        } for d in docs
    ])
    st.dataframe(doc_df, hide_index=True, use_container_width=True)
    with st.expander("Source warnings and processing limitations"):
        for d in docs:
            for warning in d.get("warnings", []):
                st.write(f"**{d['filename']}:** {warning}")
        for warning in run["warnings"]:
            st.write(warning)
    if chunks:
        with st.expander("Explore extracted chunks", expanded=False):
            labels = [f"{c['source_file']} · page {c['source_page']} · {c['chunk_id']}" for c in chunks]
            idx = st.selectbox("Select source chunk", range(len(chunks)), format_func=lambda i: labels[i])
            st.caption(f"Section: {chunks[idx].get('section_title') or 'Unknown'} · Method: {chunks[idx]['extraction_method']}")
            st.text_area("Extracted source text", chunks[idx]["text"], height=240, disabled=True)

    st.subheader("3 · Human annotation and approval")
    if candidates:
        st.info("A source-matching quote is necessary but NOT proof of answer correctness. Check every approved answer against the original PDF.")
        reviewer = st.text_input("Reviewer identifier", value="", placeholder="e.g. angkon_reviewer")
        reviewed_attested = st.checkbox("I personally verified every row marked approved (meaning, source support, safety, and privacy).")
        visible = [
            {k: ";".join(row.get(k, [])) if k == "validation_flags" else row.get(k, "") for k in REVIEW_COLUMNS}
            for row in candidates
        ]
        edited = st.data_editor(
            pd.DataFrame(visible), hide_index=True, use_container_width=True, num_rows="fixed",
            key=f"review_{run['run_id']}",
            disabled=["annotation_id", "source_id", "source_file", "source_page", "chunk_id", "section_title", "language", "document_type", "kind", "validation_flags"],
            column_config={
                "review_status": st.column_config.SelectboxColumn("Review status", options=["needs_review", "approved", "rejected"], required=True),
                "question": st.column_config.TextColumn("Question", width="large"),
                "answer": st.column_config.TextColumn("Answer", width="large"),
                "evidence_quote": st.column_config.TextColumn("Evidence quote", width="large"),
            },
        )
        reviewed = []
        for original, row in zip(candidates, edited.to_dict("records")):
            item = dict(original)
            for key_name in ("question", "answer", "evidence_quote", "review_note", "review_status"):
                val = row.get(key_name, "")
                item[key_name] = "" if pd.isna(val) else str(val)
            if item["review_status"] == "approved" and not reviewed_attested:
                item["review_status"] = "needs_review"
            if item["review_status"] == "approved":
                item["reviewer"] = reviewer.strip() or str(row.get("reviewer", "")).strip()
            reviewed.append(item)
        st.download_button("Download editable review CSV", csvtext(revalidate_annotations(reviewed, chunks), REVIEW_COLUMNS),
                           file_name="angkon_review_queue.csv", mime="text/csv")
        csv_upload = st.file_uploader("Optional: import a previously edited review CSV", type=["csv"], key="upload_review_csv")
        if csv_upload:
            try:
                reviewed = apply_review_csv(candidates, csv_upload.getvalue(), confirmed=reviewed_attested)
            except ValueError as exc:
                st.error(f"Review CSV not accepted: {exc}")
        reviewed = revalidate_annotations(reviewed, chunks)
    else:
        reviewed = []
        st.write("No model-generated examples. Extraction-only RAG data is ready for export.")

    st.subheader("4 · Download source-linked datasets")
    bundle, manifest = build_batch_zip(
        docs, chunks, reviewed,
        backend=run["llm_backend"], model=run["llm_model"],
        generation_warnings=run["warnings"], chunks_attempted=run["chunks_attempted"],
        rights_attested=authorized, audit_log=build_audit_events(candidates, reviewed),
    )
    stats = st.columns(4)
    stats[0].metric("Approved examples", manifest["approved_examples"])
    stats[1].metric("Awaiting review", manifest["unreviewed_examples"])
    stats[2].metric("Flagged issues", sum(manifest["validation_flag_counts"].values()))
    stats[3].metric("Text page coverage", f"{manifest['text_extraction_coverage_percent']}%")
    st.download_button("Download Angkon dataset ZIP", bundle,
                       file_name=f"angkon_dataset_v2_{run['run_id']}.zip", mime="application/zip", type="primary")
    st.caption("Exports: manifest, quality report, dataset card, pages, RAG chunks, tables, review CSV, Q&A candidates, approved SFT, and source-grouped splits.")
else:
    st.info("Choose extraction-only or a model backend, upload a PDF, confirm document rights, and run the workflow.")

st.divider()
st.caption("Local prototype, not a secured public upload service. Never publish source documents, cache, private dataset archives or API keys to GitHub.")
