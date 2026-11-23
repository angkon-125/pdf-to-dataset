"""Professional gold dataset release: independent reviews -> QA -> SFT/DPO JSONL.

Does not run model training or claim to certify factual truth.  The inputs are
reviewer-authored CSVs, and legal permission is an operator attestation.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .core import pii_counts
from .data_science import deduplicate_release, inspect_cross_split_overlap, prioritized_queue, quality_profile
from .expert_review import (
    parse_preferences, parse_review_files, preference_template, record_sha256,
    reviewer_template, judge_consensus,
)
from .exports import _split_sources, apply_review_csv, csvtext, jsonlines, revalidate_annotations

MAX_ARCHIVE_BYTES = 120 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024
MAX_ARCHIVE_FILES = 100


def _zip_read(z: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    try:
        raw = z.read(name).decode('utf-8-sig')
        rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Missing or invalid JSONL: {name} ({exc})') from exc
    if not all(isinstance(v, dict) for v in rows):
        raise ValueError(f'{name} must contain JSON objects')
    return rows


def load_source_archive(payload: bytes) -> dict[str, Any]:
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError('Source dataset ZIP exceeds the 120 MB file limit')
    try:
        z = zipfile.ZipFile(io.BytesIO(payload), 'r')
    except zipfile.BadZipFile as exc:
        raise ValueError('Dataset input must be an Angkon v0.2+ ZIP') from exc
    with z:
        info = z.infolist()
        if len(info) > MAX_ARCHIVE_FILES or sum(x.file_size for x in info) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError('Dataset ZIP exceeds decompressed content/file count limits')
        if len({x.filename for x in info}) != len(info):
            raise ValueError('Dataset ZIP contains duplicate archive entries')
        required = ('manifest.json', 'source_documents.json', 'source_pages.jsonl', 'rag_chunks.jsonl', 'qa_candidates.jsonl')
        if any(name not in z.namelist() for name in required):
            raise ValueError('Expected a v0.2+ Angkon archive with manifest and QA/source records')
        try:
            manifest = json.loads(z.read('manifest.json'))
            documents = json.loads(z.read('source_documents.json'))
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid manifest or source document metadata') from exc
        if not isinstance(manifest, dict) or not isinstance(documents, list):
            raise ValueError('Invalid manifest or source_documents data type')
        source_pages = _zip_read(z, 'source_pages.jsonl')
        chunks = _zip_read(z, 'rag_chunks.jsonl')
        annotations = _zip_read(z, 'qa_candidates.jsonl')
    for a in annotations:
        if not a.get('annotation_id') or not a.get('source_id') or not a.get('chunk_id'):
            raise ValueError('Candidate missing ID or source provenance')
    if len({a['annotation_id'] for a in annotations}) != len(annotations):
        raise ValueError('Duplicate annotation_id in source archive')
    by_id: dict[str, list[dict[str, Any]]] = {}
    for p in source_pages:
        by_id.setdefault(str(p.get('source_id')), []).append(p)
    documents = [
        {**d, 'pages': by_id.get(d['source_id'], [])}
        for d in documents
    ]
    return {'manifest': manifest, 'documents': documents, 'chunks': chunks, 'annotations': annotations}


def _sft(row: dict[str, Any]) -> dict[str, Any]:
    return {'messages': [
        {'role': 'user', 'content': row['question']},
        {'role': 'assistant', 'content': row['answer']},
    ]}


def _prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    return {'prompt': [{'role': 'user', 'content': row['question']}],
            'completion': [{'role': 'assistant', 'content': row['answer']}]}


def _dpo(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {
        'prompt': [{'role': 'user', 'content': item['question']}],
        'chosen': [{'role': 'assistant', 'content': row['chosen']}],
        'rejected': [{'role': 'assistant', 'content': row['rejected']}],
    }


def profile_archive(payload: bytes) -> dict[str, Any]:
    data = load_source_archive(payload)
    clean = revalidate_annotations(data['annotations'], data['chunks'])
    source_map = _split_sources([d['source_id'] for d in data['documents']])
    return {
        'quality': quality_profile(data['documents'], data['chunks'], clean, [], source_map),
        'risk_queue': prioritized_queue(clean, data['chunks']),
        'rubric_template': reviewer_template(clean),
        'preference_template': preference_template(clean),
        'candidate_count': len(clean),
    }


def curate_archive(
    source_zip: bytes, *, reviewer_csvs: list[bytes],
    rights_attested: bool, human_review_attested: bool,
    edited_annotations_csv: bytes | None = None,
    preferences_csv: bytes | None = None, min_reviewers: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    if not rights_attested or not human_review_attested:
        raise ValueError('Curated release requires explicit rights and independent human-review attestations')
    data = load_source_archive(source_zip)
    original_annotations = data['annotations']
    if edited_annotations_csv is not None:
        original_annotations = apply_review_csv(original_annotations, edited_annotations_csv, confirmed=True)
    clean = revalidate_annotations(original_annotations, data['chunks'])
    reviews, review_messages = parse_review_files(reviewer_csvs, clean)
    decisions, review_stats = judge_consensus(clean, reviews, min_reviewers=min_reviewers)
    eligible = [x for x in clean if decisions[x['annotation_id']]['ready']]
    dedup, dropped = deduplicate_release(eligible)
    splits_map = _split_sources([d['source_id'] for d in data['documents']])
    overlaps_before = inspect_cross_split_overlap(dedup, splits_map)
    # Conservative quarantine for detected cross-split lexical near-duplicates.
    near_ids = {
        item[key]
        for item in overlaps_before['cross_split_near_prompt_examples']
        for key in ('id_a', 'id_b')
    }
    gold = [x for x in dedup if x['annotation_id'] not in near_ids]
    for a_id in near_ids:
        dropped[a_id] = 'near_duplicate_cross_split_quarantined'
    lookup = {x['annotation_id']: x for x in gold}

    preferences = parse_preferences(preferences_csv, clean) if preferences_csv else []
    accepted_preferences: list[tuple[dict[str, Any], dict[str, Any]]] = []
    preference_blocked = Counter()
    for item in preferences:
        matched = lookup.get(item['annotation_id'])
        if matched is None:
            preference_blocked['not_gold_eligible'] += 1
        elif sum(pii_counts(item['rejected']).values()):
            preference_blocked['possible_pii_in_rejected'] += 1
        else:
            accepted_preferences.append((item, matched))

    source_splits = ('train', 'validation', 'test')
    gold_sft_by_split = {name: [_sft(x) for x in gold if splits_map.get(x['source_id']) == name] for name in source_splits}
    gold_pref_by_split = {
        name: [_dpo(row, matched) for row, matched in accepted_preferences if splits_map.get(matched['source_id']) == name]
        for name in source_splits
    }
    gold_provenance = [
        {
            'annotation_id': x['annotation_id'], 'source_id': x['source_id'],
            'source_file': x.get('source_file'), 'source_page': x.get('source_page'),
            'section_title': x.get('section_title'), 'chunk_id': x['chunk_id'],
            'kind': x.get('kind'), 'evidence_quote': x['evidence_quote'],
            'record_sha256': record_sha256(x),
            'reviewer_ids': decisions[x['annotation_id']]['reviewer_ids'],
            'split': splits_map.get(x['source_id'], 'train'),
        } for x in gold
    ]
    quality = quality_profile(data['documents'], data['chunks'], clean, gold, splits_map)
    quality['review_quality'] = review_stats
    quality['duplicates_dropped_from_gold'] = dropped
    quality['preference_blocked_counts'] = dict(preference_blocked)
    quality['gold_release_thresholds'] = {
        'min_independent_reviewers': min_reviewers,
        'required_factual_accuracy': 3,
        'required_grounding': 3,
        'required_safety': 3,
        'other_dimensions_minimum': 2,
        'all_reviewers_must_approve_and_verify_evidence': True,
    }
    quality['provisional_gold_count'] = len(eligible)
    quality['approved_preference_pairs'] = len(accepted_preferences)
    quality['preference_input_count'] = len(preferences)
    quality['sources_with_gold'] = len({x['source_id'] for x in gold})
    quality['source_group_split_caution'] = (
        'Fewer than 3 source PDFs; no independent held-out source split.' if len(data['documents']) < 3 else
        'Source-group split is not a guarantee against related or duplicated PDFs.'
    )
    quality['review_csv_messages'] = review_messages
    quality['dataset_status'] = ('has_curated_gold_records' if gold else 'not_ready_no_gold_records')
    release_fingerprint = hashlib.sha256(json.dumps(
        [record_sha256(x) for x in gold], sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    release_manifest = {
        'project': 'Angkon Dataset Factory', 'version': '0.3.0',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_archive_sha256': hashlib.sha256(source_zip).hexdigest(),
        'gold_content_sha256': release_fingerprint,
        'rights_attested': rights_attested,
        'rights_independently_verified': False,
        'human_review_attested': human_review_attested,
        'reviewer_identity_verified': False,
        'original_candidates': len(clean), 'gold_sft_records': len(gold),
        'reviewer_submissions': len(reviews),
        'human_approved_preference_pairs': len(accepted_preferences),
        'gold_split_counts': {name: len(rows) for name, rows in gold_sft_by_split.items()},
        'source_split_mapping': splits_map,
        'dataset_status': quality['dataset_status'],
        'warning': 'Training quality is not guaranteed. Review source truth, licensing, privacy and evaluation independently.',
    }
    docs = '''# Angkon professional curated dataset v0.3\n\n- `gold_sft_*.jsonl`: only independently reviewed, threshold-passing human labels, chat-style for HF TRL SFTTrainer.\n- `gold_prompt_completion.jsonl`: same gold examples, prompt/completion format.\n- `gold_dpo_*.jsonl`: preference pairs only when a human supplies and reviews an *actual* rejected alternative. Empty by default.\n- `gold_provenance.jsonl`: source PDF SHA-256, source page, exact evidence quote and reviewers.\n- `rubric_reviews.jsonl`: individual scores, rationale and decisions from reviewer CSVs.\n- `review_decisions.jsonl`: whether each candidate passed; rejected or uncertain candidates never appear in gold training JSONL.\n- `review_priority.csv`: risk-ordered review queue, **not** a ground-truth probability.\n- `evaluation_holdout.jsonl`: validated/test sources and gold answers; only populated for held-out source groups.\n- `data_science_quality.json`: coverage, duplicate drops, agreement and cross-split lexical audit.\n\nReviewer identity and human attestations are self-declared. Scores, quotes, and agreement do not establish real-world factual truth. Source permissions, privacy, bias, annotation policy and model performance still need human and legal review. This output intentionally excludes bulk extracted PDF text and original PDFs.\n\nSee `docs/ANNOTATOR_SCIENTIST.md` in the application repository.\n'''
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('release_manifest.json', json.dumps(release_manifest, indent=2, ensure_ascii=False))
        z.writestr('data_science_quality.json', json.dumps(quality, indent=2, ensure_ascii=False))
        z.writestr('README_CURATED.md', docs)
        z.writestr('gold_sft.jsonl', jsonlines([_sft(x) for x in gold]))
        z.writestr('gold_prompt_completion.jsonl', jsonlines([_prompt_completion(x) for x in gold]))
        for name in source_splits:
            z.writestr(f'gold_sft_{name}.jsonl', jsonlines(gold_sft_by_split[name]))
        z.writestr('gold_dpo.jsonl', jsonlines([_dpo(p, x) for p, x in accepted_preferences]))
        for name in source_splits:
            z.writestr(f'gold_dpo_{name}.jsonl', jsonlines(gold_pref_by_split[name]))
        z.writestr('gold_provenance.jsonl', jsonlines(gold_provenance))
        z.writestr('rubric_reviews.jsonl', jsonlines(reviews))
        z.writestr('review_decisions.jsonl', jsonlines(list(decisions.values())))
        z.writestr('review_priority.csv', csvtext(prioritized_queue(clean, data['chunks']),
            ['annotation_id','source_file','source_page','review_status','risk_priority','risk_reasons','question']))
        z.writestr('evaluation_holdout.jsonl', jsonlines([
            {'question': x['question'], 'reference_answer': x['answer'],
             'source_id': x['source_id'], 'source_page': x['source_page'],
             'evidence_quote': x['evidence_quote'], 'split': splits_map[x['source_id']]}
            for x in gold if splits_map[x['source_id']] in ('validation','test')
        ]))
    return out.getvalue(), release_manifest
