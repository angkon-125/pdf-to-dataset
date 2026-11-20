import io
import json
import zipfile

import pymupdf

from angkon_factory.core import extract_pdf, make_chunks, detect_language, detect_document_type, split_text
from angkon_factory.generation import generate_candidates, validate_candidate, extract_json_object
from angkon_factory.exports import build_zip


def synthetic_pdf() -> bytes:
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 90), "CHAPTER 1: Introduction to computing", fontsize=15)
    p.insert_text((60, 120), "A CPU executes instructions and manages general computing tasks.")
    p.insert_text((60, 140), "A GPU processes graphics and can run many calculations in parallel.")
    p2 = doc.new_page()
    p2.insert_text((60, 90), "CHAPTER 2: Data practices", fontsize=15)
    p2.insert_text((60, 120), "Training data should be evaluated for accuracy and legal reuse rights.")
    p2.insert_text((60, 140), "Source citations help reviewers inspect where a statement came from.")
    b = doc.tobytes()
    doc.close()
    return b


class FakeLLM:
    def generate(self, prompt):
        assert "SOURCE" in prompt
        if "CPU executes instructions" in prompt:
            return json.dumps({"items": [{
                "question": "What tasks does a CPU handle?",
                "answer": "It executes instructions and manages general computing tasks.",
                "evidence_quote": "A CPU executes instructions and manages general computing tasks.",
                "kind": "factual_qa"
            }]})
        return json.dumps({"items": [{
            "question": "Why should training data be evaluated?",
            "answer": "To help ensure accuracy and appropriate legal reuse rights.",
            "evidence_quote": "Training data should be evaluated for accuracy and legal reuse rights.",
            "kind": "factual_qa"
        }]})


def test_extraction_and_provenance():
    data = extract_pdf(synthetic_pdf(), "demo.pdf")
    assert data["page_count"] == 2
    assert data["processed_pages"] == 2
    assert data["document_type_heuristic"] == "book_or_textbook"
    assert data["language_heuristic"] == "English"
    chunks = make_chunks(data)
    assert len(chunks) == 2
    assert chunks[0]["source_page"] == 1
    assert chunks[1]["source_page"] == 2
    assert "A GPU" in chunks[0]["text"]


def test_gen_validate_review_and_export():
    data = extract_pdf(synthetic_pdf(), "demo.pdf")
    chunks = make_chunks(data)
    candidates, warns = generate_candidates(chunks, FakeLLM(), max_chunks=0)
    assert len(candidates) == 2 and not warns
    assert all(x["review_status"] == "needs_review" for x in candidates)
    assert all(not x["validation_flags"] for x in candidates)
    z, manifest = build_zip(data, chunks, candidates, backend="ollama", model="fake", chunks_attempted=2)
    assert manifest["approved_examples"] == 0
    with zipfile.ZipFile(io.BytesIO(z)) as f:
        assert len(f.read("rag_chunks.jsonl").decode().splitlines()) == 2
        assert f.read("sft_approved.jsonl") == b""
        assert "needs_review" in f.read("qa_candidates.jsonl").decode()
    candidates[0]["review_status"] = "approved"
    candidates[0]["reviewer"] = "reviewer1"
    z, manifest = build_zip(data, chunks, candidates, backend="ollama", model="fake", chunks_attempted=2)
    assert manifest["approved_examples"] == 1
    with zipfile.ZipFile(io.BytesIO(z)) as f:
        line = json.loads(f.read("sft_approved.jsonl").decode().splitlines()[0])
        assert line["messages"][1]["role"] == "assistant"


def test_invalid_evidence_cannot_enter_approved_sft():
    data = extract_pdf(synthetic_pdf(), "demo.pdf")
    chunks = make_chunks(data)
    rows, _ = generate_candidates(chunks, FakeLLM(), max_chunks=0)
    rows[0]["evidence_quote"] = "An unsupported claim from nowhere."
    rows[0]["review_status"] = "approved"
    _z, manifest = build_zip(data, chunks, rows, backend="ollama", chunks_attempted=2)
    assert manifest["approved_examples"] == 0
    assert manifest["unreviewed_examples"] == 2


def test_duplicate_questions_are_flagged_during_export():
    data = extract_pdf(synthetic_pdf(), "demo.pdf")
    chunks = make_chunks(data)
    rows, _ = generate_candidates(chunks, FakeLLM(), max_chunks=0)
    for x in rows:
        x["question"] = "What does the book teach?"
        x["review_status"] = "approved"
    _z, manifest = build_zip(data, chunks, rows, chunks_attempted=2)
    assert manifest["approved_examples"] == 1
    assert manifest["unreviewed_examples"] == 1


def test_language_detection_and_document_type():
    assert detect_language("বাংলা ভাষার একটি বই, with English words as well.") == "Bangla-English mixed"
    assert detect_document_type("Abstract\nMethodology\nReferences\n")[0] == "research_paper"
    assert split_text("short paragraph") == ["short paragraph"]


def test_json_fence_parsing():
    assert extract_json_object('```json\n{"items": []}\n```')["items"] == []


def test_llm_classifier_on_sampled_chunks():
    from angkon_factory.generation import classify_document
    class MockProfileClient:
        def generate(self, prompt, *, system_prompt=None):
            assert "Classify the source excerpts" in prompt
            assert "untrusted data" in system_prompt
            return '{"document_type":"educational_notes","subject":"AI basics","suggested_focus":"concept questions"}'
    data = extract_pdf(synthetic_pdf(), "demo.pdf")
    profile, warnings = classify_document(make_chunks(data), MockProfileClient())
    assert not warnings
    assert profile["document_type"] == "educational_notes"
    data["llm_profile"] = profile
    assert make_chunks(data)[0]["document_type"] == "educational_notes"


def test_pdf_page_limit_records_partial_coverage():
    data = extract_pdf(synthetic_pdf(), "demo.pdf", page_limit=1)
    assert data["processed_pages"] == 1
    assert any("first 1 of 2 pages" in warning for warning in data["warnings"])


def test_ollama_json_api_payload(monkeypatch):
    from angkon_factory.generation import LLMClient
    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return {"message": {"content": '{"items": []}'}}
    def mock_post(url, *, headers, json, timeout):
        assert url == "http://localhost:11434/api/chat"
        assert json["format"] == "json"
        assert json["stream"] is False
        assert len(json["messages"]) == 2
        assert timeout == 180
        return FakeResponse()
    monkeypatch.setattr("angkon_factory.generation.requests.post", mock_post)
    client = LLMClient("ollama", "demo-model", "http://localhost:11434")
    assert client.generate("TEST") == '{"items": []}'
