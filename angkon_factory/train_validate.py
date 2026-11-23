"""Offline pre-training release linter for Angkon curated training datasets."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from typing import Any

from .core import pii_counts
from .generation import norm


def _read_jsonl(z: zipfile.ZipFile, filename: str) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(s) for s in z.read(filename).decode('utf-8').splitlines() if s.strip()]
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Missing or invalid curated dataset file: {filename}') from exc
    if not all(isinstance(x, dict) for x in rows):
        raise ValueError(f'Non-object records in {filename}')
    return rows


def _valid_dialogue(record: dict[str, Any]) -> bool:
    entries = record.get('messages')
    return bool(
        isinstance(entries, list) and len(entries) == 2 and
        entries[0].get('role') == 'user' and entries[1].get('role') == 'assistant' and
        all(isinstance(x.get('content'), str) and x['content'].strip() for x in entries)
    ) if isinstance(entries, list) and all(isinstance(x, dict) for x in entries) else False


def audit_gold_release(payload: bytes) -> dict[str, Any]:
    if len(payload) > 120 * 1024 * 1024:
        raise ValueError('Curated ZIP exceeds maximum size')
    try:
        z = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ValueError('Expected a curated Angkon ZIP') from exc
    with z:
        if len(z.infolist()) > 100 or sum(x.file_size for x in z.infolist()) > 400 * 1024 * 1024:
            raise ValueError('Curated ZIP exceeds safe decompressed limits')
        if len(set(z.namelist())) != len(z.namelist()):
            raise ValueError('Duplicate archive entries')
        try:
            manifest = json.loads(z.read('release_manifest.json'))
            quality = json.loads(z.read('data_science_quality.json'))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError('Missing/invalid curated release manifest') from exc
        names = ('train', 'validation', 'test')
        splits = {name: _read_jsonl(z, f'gold_sft_{name}.jsonl') for name in names}
        all_gold = _read_jsonl(z, 'gold_sft.jsonl')
        provenance = _read_jsonl(z, 'gold_provenance.jsonl')
        all_pref = _read_jsonl(z, 'gold_dpo.jsonl')
        pref_splits = {name: _read_jsonl(z, f'gold_dpo_{name}.jsonl') for name in names}
        holdout = _read_jsonl(z, 'evaluation_holdout.jsonl')
    errors = []
    warnings = []
    if not manifest.get('rights_attested') or not manifest.get('human_review_attested'):
        errors.append('missing_rights_or_human_review_attestations')
    if not all(_valid_dialogue(x) for x in all_gold):
        errors.append('malformed_sft_dialogues')
    flat = [item for name in names for item in splits[name]]
    def canonical(x: dict[str, Any]) -> str:
        return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if Counter(map(canonical, flat)) != Counter(map(canonical, all_gold)):
        errors.append('split_rows_do_not_match_gold_all')
    if len(provenance) != len(all_gold):
        errors.append('provenance_count_mismatch')
    if any(len(set(item.get('reviewer_ids', []))) < 2 for item in provenance):
        errors.append('less_than_two_reviewers_in_provenance')
    by_split = {name: [norm(x.get('messages', [{}])[0].get('content','')) for x in rows if _valid_dialogue(x)] for name, rows in splits.items()}
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if set(by_split[name]) & set(by_split[other]):
                errors.append(f'exact_question_leakage_{name}_{other}')
    if any(sum(pii_counts(x['messages'][0]['content']+' '+x['messages'][1]['content']).values()) for x in all_gold if _valid_dialogue(x)):
        errors.append('possible_pii_in_gold_sft')
    if len(holdout) != len(splits['validation']) + len(splits['test']):
        errors.append('holdout_count_does_not_match_splits')
    if len(flat) < 30:
        warnings.append('Fewer than 30 gold records: probably insufficient for meaningful domain fine-tuning.')
    if not splits['validation'] or not splits['test']:
        warnings.append('Missing a distinct held-out source validation/test set; obtain more independent source PDFs.')
    if not all_pref:
        warnings.append('No human-scored preference pairs; DPO export is empty by design.')
    if quality.get('cross_split_overlap_audit', {}).get('near_overlap_audit_truncated'):
        warnings.append('Cross-split approximate overlap audit was limited; run deeper corpus deduplication.')
    if not all(isinstance(x,dict) and all(field in x for field in ('prompt', 'chosen', 'rejected')) and
               x['chosen'] != x['rejected'] for x in all_pref):
        errors.append('invalid_dpo_records')
    if Counter(map(canonical, all_pref)) != Counter(map(canonical, [r for name in names for r in pref_splits[name]])):
        errors.append('dpo_splits_do_not_match_all')
    return {
        'ready_for_format_loading': not errors and len(splits['train']) > 0,
        'errors': list(dict.fromkeys(errors)), 'warnings': warnings,
        'sft_record_count': len(all_gold), 'dpo_record_count': len(all_pref),
        'training_only_sft_records': len(splits['train']),
        'validation_records': len(splits['validation']),
        'test_records': len(splits['test']),
        'release_version': manifest.get('version'),
        'reminder': 'Format compliance is not proof of factuality, permission, safety, generalization, or model improvement.',
    }
