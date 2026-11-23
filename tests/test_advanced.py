"""Coverage for advanced batch, privacy, review, resume, and source-grouped exports."""
import csv
import io
import json
import zipfile
from pathlib import Path

import pymupdf
import pytest

from angkon_factory.core import detect_language, extract_pdf, make_chunks, pii_counts
from angkon_factory.exports import apply_review_csv, build_batch_zip, csvtext, revalidate_annotations, REVIEW_COLUMNS
from angkon_factory.generation import generate_candidates, select_chunks
from angkon_factory.pipeline import PipelineOptions, run_batch
from angkon_factory.review import review_existing_zip


def pdf_with_text(topic: str) -> bytes:
    doc = pymupdf.open()
    pg = doc.new_page()
    pg.insert_text((52, 55), f"CHAPTER 1 - {topic}", fontsize=14)
    pg.insert_text((52, 90), f"This document explains {topic} and practical examples.")
    pg.insert_text((52, 112), "Evidence is important for reliable training datasets.")
    doc.set_toc([[1, f"Chapter 1 - {topic}", 1]])
    result = doc.tobytes()
    doc.close()
    return result


class ModelStub:
    backend = "fake"
    model = "fake-model"
    endpoint = "local://testing"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, system_prompt=None):
        self.calls += 1
        # Only use source text actually present in the prompt.
        import re
        m = re.search(r"This document explains ([A-Za-z]+) and practical examples\.", prompt)
        assert m is not None
        topic = m.group(1)
        quote = f"This document explains {topic} and practical examples."
        return json.dumps({"items": [{
            "question": f"What subject is explained in the {topic} document?",
            "answer": f"The document explains {topic}.",
            "evidence_quote": quote,
            "kind": "factual_qa",
        }]})


def test_multidoc_batch_and_deduplicate():
    a, b = pdf_with_text("Physics"), pdf_with_text("Chemistry")
    result = run_batch(
        [("one.pdf", a), ("copy.pdf", a), ("two.pdf", b)],
        options=PipelineOptions(chunk_chars=800),
    )
    assert len(result["documents"]) == 2
    assert len(result["chunks"]) == 2
    assert any("Duplicate PDF skipped" in w for w in result["warnings"])
    assert all(c["section_title"].startswith("Chapter 1") for c in result["chunks"])
    assert all(len(c["source_id"]) == 64 for c in result["chunks"])


def test_spread_sampling_covers_first_and_last():
    chunks = [{"i": i} for i in range(22)]
    out = select_chunks(chunks, 4, "spread")
    assert len(out) == 4
    assert [x["i"] for x in out][0] == 0
    assert [x["i"] for x in out][-1] == 21
    assert select_chunks(chunks, 4, "first") == chunks[:4]
    assert select_chunks(chunks, 0) == chunks


def test_generation_resume_uses_cached_json(tmp_path: Path):
    doc = extract_pdf(pdf_with_text("Physics"), "test.pdf")
    chunks = make_chunks(doc, target_chars=750)
    client = ModelStub()
    records, warn = generate_candidates(chunks, client, max_chunks=0, max_source_chars=550, cache_dir=tmp_path)
    assert not warn and len(records) == 1 and client.calls == 1
    second, warn2 = generate_candidates(chunks, client, max_chunks=0, max_source_chars=550, cache_dir=tmp_path)
    assert not warn2 and client.calls == 1 and second == records
    assert list(tmp_path.glob("*.json"))


def test_rights_required_for_generating():
    with pytest.raises(ValueError, match="rights"):
        run_batch([("x.pdf", pdf_with_text("Physics"))], client=ModelStub())
    result = run_batch(
        [("x.pdf", pdf_with_text("Physics"))], client=ModelStub(),
        options=PipelineOptions(rights_attested=True, max_chunks_per_document=0),
    )
    assert len(result["annotations"]) == 1
    assert result["annotations"][0]["review_status"] == "needs_review"


def test_no_approved_export_without_attestation():
    result = run_batch([("x.pdf", pdf_with_text("Physics"))], client=ModelStub(),
                       options=PipelineOptions(rights_attested=True, max_chunks_per_document=0))
    records = result["annotations"]
    records[0]["review_status"], records[0]["reviewer"] = "approved", "tester"
    blob, meta = build_batch_zip(result["documents"], result["chunks"], records, rights_attested=False)
    assert meta["approved_examples"] == 0
    assert not zipfile.ZipFile(io.BytesIO(blob)).read("sft_approved.jsonl")
    blob2, meta2 = build_batch_zip(result["documents"], result["chunks"], records, rights_attested=True)
    assert meta2["approved_examples"] == 1
    assert meta2["training_rights_verified"] is False
    with zipfile.ZipFile(io.BytesIO(blob2)) as z:
        assert "dataset_card.md" in z.namelist()
        assert "quality_report.json" in z.namelist()
        assert "source_documents.json" in z.namelist()


def test_three_docs_use_source_grouped_splits():
    result = run_batch(
        [(f"{t}.pdf", pdf_with_text(t)) for t in ("Physics", "Chemistry", "Biology")],
        client=ModelStub(),
        options=PipelineOptions(rights_attested=True, max_chunks_per_document=0),
    )
    assert len(result["annotations"]) == 3
    for r in result["annotations"]:
        r["review_status"] = "approved"
        r["reviewer"] = "expert"
    blob, manifest = build_batch_zip(result["documents"], result["chunks"], result["annotations"], rights_attested=True)
    assert manifest["approved_examples"] == 3
    assert manifest["approved_split_counts"] == {"train": 1, "validation": 1, "test": 1}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        splits = [
            json.loads(z.read(f"sft_{group}.jsonl").decode().strip())
            for group in ("train", "validation", "test")
        ]
    questions = [x["messages"][0]["content"] for x in splits]
    assert len(set(questions)) == 3


def test_review_csv_import_without_model_calls():
    result = run_batch([("x.pdf", pdf_with_text("Physics"))], client=ModelStub(),
                       options=PipelineOptions(rights_attested=True, max_chunks_per_document=0))
    archive, manifest = build_batch_zip(result["documents"], result["chunks"], result["annotations"], rights_attested=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        raw = z.read("annotations_review.csv").decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows[0]["reviewer"] = "human_reviewer"
    rows[0]["review_status"] = "approved"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    revised, updated = review_existing_zip(archive, stream.getvalue().encode(),
                                           attest_reviewed=True, rights_attested=True)
    assert updated["approved_examples"] == 1
    with zipfile.ZipFile(io.BytesIO(revised)) as z:
        assert z.read("sft_approved.jsonl")
    with pytest.raises(ValueError, match="attestation"):
        review_existing_zip(archive, stream.getvalue().encode(), attest_reviewed=False, rights_attested=True)


def test_provenance_mismatch_blocks_approval():
    result = run_batch([("x.pdf", pdf_with_text("Physics"))], client=ModelStub(),
                       options=PipelineOptions(rights_attested=True))
    item = result["annotations"][0]
    item["review_status"] = "approved"
    item["reviewer"] = "tester"
    item["source_page"] = 99
    output = revalidate_annotations([item], result["chunks"])
    assert "provenance_mismatch" in output[0]["validation_flags"]
    assert output[0]["review_status"] == "needs_review"


def test_privacy_mask_and_pii_warnings():
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 90), "Contact us at person@example.com or +8801712345678 for more information.")
    data = doc.tobytes()
    doc.close()
    parsed = extract_pdf(data, "private.pdf", redact_pii=True)
    raw = parsed["pages"][0]["text"]
    assert "person@example.com" not in raw and "+8801712345678" not in raw
    assert "[REDACTED_EMAIL]" in raw and "[REDACTED_PHONE]" in raw
    assert any("Possible personal data" in w for w in parsed["warnings"])
    assert pii_counts("person@example.com +8801712345678") == {"email_addresses": 1, "bd_phone_numbers": 1}


def test_csv_formula_defense_and_review_id_check():
    a = {"annotation_id": "abc", "source_id": "sha", "question": "=2+2", "answer": "some answer", "review_status": "needs_review"}
    safe = csvtext([a], REVIEW_COLUMNS)
    assert "'=2+2" in safe
    fake = safe.replace("abc", "other")
    with pytest.raises(ValueError, match="Unknown annotation_id"):
        apply_review_csv([a], fake.encode(), confirmed=True)


def test_strong_language_heuristics():
    assert detect_language("This document is written in English and contains enough text") == "English"
