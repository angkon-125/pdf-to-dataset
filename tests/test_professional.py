"""Professional annotation and gold dataset pipeline tests: no LLM or network."""
import csv
import io
import json
import zipfile

import pymupdf
import pytest

from angkon_factory.core import extract_pdf, make_chunks
from angkon_factory.curation import curate_archive, load_source_archive, profile_archive
from angkon_factory.data_science import risk_score
from angkon_factory.expert_review import (
    DIMENSIONS, REVIEW_FIELDS, record_sha256, reviewer_template, preference_template,
)
from angkon_factory.exports import build_batch_zip


def _input_zip():
    docs, chunks, annotations = [], [], []
    for topic in ('Physics', 'Biology', 'Chemistry'):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((55, 70), f'This document explains {topic} and practical examples.')
        page.insert_text((55, 100), 'Careful annotation supports model training and evaluation.')
        pdf = doc.tobytes()
        doc.close()
        parsed = extract_pdf(pdf, f'{topic}.pdf')
        cs = make_chunks(parsed, target_chars=600)
        docs.append(parsed)
        chunks.extend(cs)
        annotations.append({
            'annotation_id': f'ex_{topic.lower()}', 'source_id': parsed['source_id'],
            'source_file': parsed['filename'], 'source_page': 1, 'chunk_id': cs[0]['chunk_id'],
            'section_title': '', 'language': 'English', 'document_type': 'book_or_textbook',
            'question': f'What subject does the {topic} document explain?',
            'answer': f'The document explains {topic}.',
            'evidence_quote': f'This document explains {topic} and practical examples.',
            'kind': 'factual_qa', 'review_status': 'needs_review',
            'reviewer': '', 'review_note': '', 'validation_flags': [],
        })
    archive, _ = build_batch_zip(docs, chunks, annotations, rights_attested=True)
    return archive


def _review_file(archive, reviewer='annotator_one', *, decision='approve',
                 score='3', verified='yes', subset=None):
    data = load_source_archive(archive)
    template = reviewer_template(data['annotations'])
    rows = list(csv.DictReader(io.StringIO(template.decode('utf-8-sig'))))
    for row in rows:
        if subset is not None and row['annotation_id'] not in subset:
            continue
        row['reviewer_id'] = reviewer
        row['decision'] = decision
        row['evidence_verified'] = verified
        row['rationale'] = 'This claim is directly supported by the quoted source evidence.'
        for d in DIMENSIONS:
            row[d] = score
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _dpo_file(archive, *, distinct=True):
    src = load_source_archive(archive)
    rows = list(csv.DictReader(io.StringIO(preference_template(src['annotations']).decode('utf-8-sig'))))
    row = rows[0]
    row['reviewer_id'] = 'preference_judge'
    row['rejected_answer'] = 'It explains a different subject.' if distinct else row['chosen_answer']
    row['preference_reason'] = 'The chosen answer follows the source while the alternative is unsupported.'
    row['review_status'] = 'approved'
    row['evidence_verified'] = 'yes'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _curate(archive, reviews, **kwargs):
    return curate_archive(archive, reviewer_csvs=reviews, rights_attested=True,
                          human_review_attested=True, **kwargs)


def test_inspect_generates_review_template_and_prioritized_queue():
    archive = _input_zip()
    report = profile_archive(archive)
    assert report['candidate_count'] == 3
    assert len(report['risk_queue']) == 3
    assert 'record_sha256' in report['rubric_template'].decode()
    assert 'rejected_answer' in report['preference_template'].decode()
    assert report['quality']['documents'] == 3


def test_two_independent_humans_release_gold_sft_and_heldout():
    archive = _input_zip()
    r1 = _review_file(archive, 'human_A')
    r2 = _review_file(archive, 'human_B')
    release, manifest = _curate(archive, [r1, r2])
    assert manifest['gold_sft_records'] == 3
    assert manifest['gold_split_counts'] == {'train': 1, 'validation': 1, 'test': 1}
    with zipfile.ZipFile(io.BytesIO(release)) as z:
        gold = [json.loads(v) for v in z.read('gold_sft.jsonl').decode().splitlines()]
        assert len(gold) == 3 and gold[0]['messages'][1]['role'] == 'assistant'
        assert len(z.read('evaluation_holdout.jsonl').decode().splitlines()) == 2
        provenance = json.loads(z.read('gold_provenance.jsonl').decode().splitlines()[0])
        assert len(provenance['reviewer_ids']) == 2
        assert 'source_pages.jsonl' not in z.namelist()
        quality = json.loads(z.read('data_science_quality.json'))
        assert quality['gold_count'] == 3
        assert quality['review_quality']['decision_disagreement_rate'] == 0.0
        assert quality['cross_split_overlap_audit']['cross_split_exact_prompts'] == []


def test_only_one_reviewer_yields_no_gold_but_reports_blocker():
    archive = _input_zip()
    release, manifest = _curate(archive, [_review_file(archive)])
    assert manifest['gold_sft_records'] == 0
    with zipfile.ZipFile(io.BytesIO(release)) as z:
        quality = json.loads(z.read('data_science_quality.json'))
        assert quality['dataset_status'] == 'not_ready_no_gold_records'
        assert quality['review_quality']['blockers']['insufficient_independent_reviews'] == 3


def test_same_reviewer_twice_is_rejected():
    archive = _input_zip()
    payload = _review_file(archive, 'one')
    with pytest.raises(ValueError, match='Duplicate review'):
        _curate(archive, [payload, payload])


def test_content_tampering_invalidates_prior_rubric():
    archive = _input_zip()
    reviewer = _review_file(archive, 'one')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        edited = z.read('annotations_review.csv').decode()
    edited = edited.replace('The document explains Physics.', 'The document explains Mysteries.')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        _curate(archive, [reviewer], edited_annotations_csv=edited.encode())


def test_disagreement_quarantines_not_majority_vote():
    archive = _input_zip()
    release, manifest = _curate(archive, [
        _review_file(archive, 'one', decision='approve'),
        _review_file(archive, 'two', decision='reject'),
    ])
    assert manifest['gold_sft_records'] == 0
    with zipfile.ZipFile(io.BytesIO(release)) as z:
        stats = json.loads(z.read('data_science_quality.json'))['review_quality']
        assert stats['decision_disagreement_rate'] == 1.0
        assert stats['blockers']['reviewer_disagreement'] == 3


def test_scores_below_threshold_and_unverified_are_blocked():
    archive = _input_zip()
    for score,verified in [('2','yes'),('3','no')]:
        release, manifest = _curate(archive, [
            _review_file(archive, 'one', score=score, verified=verified),
            _review_file(archive, 'two'),
        ])
        assert manifest['gold_sft_records'] == 0


def test_dpo_real_preference_only_and_reject_identical():
    archive = _input_zip()
    r1, r2 = _review_file(archive, 'reviewer_A'), _review_file(archive, 'reviewer_B')
    release, manifest = _curate(archive, [r1, r2], preferences_csv=_dpo_file(archive))
    assert manifest['human_approved_preference_pairs'] == 1
    with zipfile.ZipFile(io.BytesIO(release)) as z:
        dpo = json.loads(z.read('gold_dpo.jsonl').decode().splitlines()[0])
        assert list(dpo) == ['prompt', 'chosen', 'rejected']
        assert dpo['chosen'][0]['content'] != dpo['rejected'][0]['content']
    with pytest.raises(ValueError, match='distinct actual alternative'):
        _curate(archive, [r1,r2], preferences_csv=_dpo_file(archive, distinct=False))


def test_rights_attestation_is_required():
    archive = _input_zip()
    with pytest.raises(ValueError, match='attestations'):
        curate_archive(archive, reviewer_csvs=[], rights_attested=False,
                       human_review_attested=True)


def test_never_accept_fake_source_hash_or_invalid_id():
    archive = _input_zip()
    bad = _review_file(archive)
    parsed = list(csv.DictReader(io.StringIO(bad.decode())))
    parsed[0]['record_sha256'] = '0' * 64
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(parsed[0]))
    writer.writeheader();writer.writerows(parsed)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        _curate(archive, [stream.getvalue().encode()])


def test_risk_priority_tags_pii_and_bad_quote():
    risk, reasons = risk_score({'validation_flags':['quote_not_found_in_source'],
                                'answer':'x', 'question':'q',
                                'evidence_quote':'personal@example.org'})
    assert risk == 100
    assert 'potential_pii' in reasons


def test_validate_gold_release_checks_train_schema_and_leakage():
    from angkon_factory.train_validate import audit_gold_release
    archive = _input_zip()
    release, _ = _curate(archive, [
        _review_file(archive, 'reviewer_one'), _review_file(archive, 'reviewer_two')])
    lint = audit_gold_release(release)
    assert lint['ready_for_format_loading'] is True
    assert lint['training_only_sft_records'] == 1
    assert lint['validation_records'] == 1
    assert lint['test_records'] == 1
    assert lint['errors'] == []
    # Deliberately corrupt validation split to leak a training prompt.
    with zipfile.ZipFile(io.BytesIO(release)) as old:
        files = {name: old.read(name) for name in old.namelist()}
    files['gold_sft_validation.jsonl'] = files['gold_sft_train.jsonl']
    corrupted = io.BytesIO()
    with zipfile.ZipFile(corrupted, 'w', zipfile.ZIP_DEFLATED) as new:
        for name, data in files.items():
            new.writestr(name, data)
    lint = audit_gold_release(corrupted.getvalue())
    assert lint['ready_for_format_loading'] is False
    assert any('exact_question_leakage' in v for v in lint['errors'])


def test_preference_reviewer_cannot_change_shown_chosen_answer():
    archive = _input_zip()
    parsed = list(csv.DictReader(io.StringIO(_dpo_file(archive).decode())))
    parsed[0]['chosen_answer'] = 'THIS WAS NOT THE APPROVED ANSWER'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(parsed[0]))
    writer.writeheader(); writer.writerows(parsed)
    with pytest.raises(ValueError, match='chosen_answer was changed'):
        _curate(archive, [
            _review_file(archive, 'reviewer_one'), _review_file(archive, 'reviewer_two')],
            preferences_csv=stream.getvalue().encode())


def test_reviewer_cannot_change_displayed_question_without_rehashing():
    archive = _input_zip()
    parsed = list(csv.DictReader(io.StringIO(_review_file(archive, 'reviewer_one').decode())))
    parsed[0]['question'] = 'COMPLETELY DIFFERENT QUESTION'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(parsed[0]))
    writer.writeheader(); writer.writerows(parsed)
    with pytest.raises(ValueError, match='displayed question was altered'):
        _curate(archive, [stream.getvalue().encode()])


def test_no_gold_without_review_has_no_training_records():
    from angkon_factory.train_validate import audit_gold_release
    archive = _input_zip()
    release, _ = _curate(archive, [])
    report = audit_gold_release(release)
    assert report['ready_for_format_loading'] is False
    assert report['sft_record_count'] == 0
