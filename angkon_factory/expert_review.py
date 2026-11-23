"""Human-annotator workbench primitives: immutable fingerprints and independent rubrics.

These functions do NOT substitute for a human. CSV rows are assertions by people;
identities and source claims are not independently verified by the software.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from typing import Any

DIMENSIONS = (
    'factual_accuracy', 'grounding', 'instruction_following',
    'completeness', 'clarity', 'safety',
)
REVIEW_FIELDS = [
    'annotation_id', 'record_sha256', 'source_file', 'source_page',
    'question', 'answer', 'evidence_quote', 'reviewer_id', 'decision',
    'evidence_verified', *DIMENSIONS, 'rationale',
]
PREFERENCE_FIELDS = [
    'annotation_id', 'record_sha256', 'question', 'chosen_answer',
    'rejected_answer', 'reviewer_id', 'preference_reason', 'evidence_verified',
    'review_status',
]
SCORES_MIN = {
    'factual_accuracy': 3, 'grounding': 3, 'instruction_following': 2,
    'completeness': 2, 'clarity': 2, 'safety': 3,
}
REQUIRED_FIELDS = ('source_id', 'source_page', 'chunk_id', 'question', 'answer', 'evidence_quote', 'kind')


def record_sha256(row: dict[str, Any]) -> str:
    """Bind review to the precise annotation text, source, page, and chunk."""
    serialized = json.dumps(
        {field: row.get(field, '') for field in REQUIRED_FIELDS},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _csv_safety(s: Any) -> Any:
    if isinstance(s, str) and s.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + s
    return s


def _csv_rows(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_safety(row.get(key, '')) for key in fields})
    return stream.getvalue().encode('utf-8-sig')


def reviewer_template(annotations: list[dict[str, Any]]) -> bytes:
    return _csv_rows([
        {
            'annotation_id': item['annotation_id'],
            'record_sha256': record_sha256(item),
            'source_file': item.get('source_file', ''),
            'source_page': item.get('source_page', ''),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'evidence_quote': item.get('evidence_quote', ''),
            'reviewer_id': '', 'decision': '', 'evidence_verified': '',
            **{field: '' for field in DIMENSIONS}, 'rationale': '',
        } for item in annotations
    ], REVIEW_FIELDS)


def preference_template(annotations: list[dict[str, Any]]) -> bytes:
    return _csv_rows([
        {
            'annotation_id': item['annotation_id'],
            'record_sha256': record_sha256(item),
            'question': item.get('question', ''),
            'chosen_answer': item.get('answer', ''),
            'rejected_answer': '', 'reviewer_id': '', 'preference_reason': '',
            'evidence_verified': '', 'review_status': '',
        } for item in annotations
    ], PREFERENCE_FIELDS)


def _parse_csv(data: bytes, required: tuple[str, ...]) -> list[dict[str, str]]:
    if len(data) > 16 * 1024 * 1024:
        raise ValueError('Reviewer CSV exceeds the 16 MB limit')
    try:
        stream = io.StringIO(data.decode('utf-8-sig'), newline='')
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError('Missing required reviewer CSV columns: ' + ', '.join(sorted(set(required) - set(reader.fieldnames or []))))
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f'Invalid UTF-8 reviewer CSV: {exc}') from exc
    if len(rows) > 100_000:
        raise ValueError('Reviewer CSV has more than 100,000 rows')
    if any(None in row for row in rows):
        raise ValueError('Reviewer CSV contains surplus columns')
    return rows


def _csv_visual_value(value: Any) -> str:
    """Undo leading spreadsheet formula escape only on display/reference fields."""
    raw = str(value or '')
    return raw[1:] if raw.startswith("'") and raw[1:2] in ('=', '+', '-', '@') else raw


def _truth(value: Any) -> bool:
    return str(value or '').strip().lower() in ('true', 'yes', 'y', '1', 'verified')


def parse_review_files(
    files: list[bytes], annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    lookup = {a['annotation_id']: a for a in annotations}
    output: list[dict[str, Any]] = []
    messages: list[str] = []
    seen: set[tuple[str, str]] = set()
    for file_index, payload in enumerate(files, 1):
        rows = _parse_csv(payload, ('annotation_id', 'record_sha256', 'reviewer_id', 'decision', 'evidence_verified', *DIMENSIONS, 'rationale'))
        for line, row in enumerate(rows, 2):
            decision = str(row.get('decision') or '').strip().lower()
            if not decision:
                continue  # unfilled template rows
            if decision not in ('approve', 'reject', 'revise'):
                raise ValueError(f'Reviewer CSV {file_index}, row {line}: unknown decision')
            a_id = str(row.get('annotation_id') or '').strip()
            annotation = lookup.get(a_id)
            if not annotation:
                raise ValueError(f'Reviewer CSV {file_index}, row {line}: unknown annotation_id {a_id}')
            if str(row.get('record_sha256') or '').strip() != record_sha256(annotation):
                raise ValueError(f'Reviewer CSV {file_index}, row {line}: fingerprint mismatch for {a_id}; re-review edited content')
            for visible in ('question', 'answer', 'evidence_quote', 'source_file', 'source_page'):
                if visible in row and _csv_visual_value(row[visible]) != str(annotation.get(visible, '')):
                    raise ValueError(f'Reviewer CSV {file_index}, row {line}: displayed {visible} was altered; re-review original text')
            reviewer = str(row.get('reviewer_id') or '').strip()
            if len(reviewer) < 2 or len(reviewer) > 120:
                raise ValueError(f'Reviewer CSV {file_index}, row {line}: reviewer_id required (2-120 characters)')
            if (a_id, reviewer.casefold()) in seen:
                raise ValueError(f'Duplicate review from {reviewer} for {a_id}')
            seen.add((a_id, reviewer.casefold()))
            scores: dict[str, int | None] = {}
            for field in DIMENSIONS:
                value = str(row.get(field) or '').strip()
                if not value and decision != 'approve':
                    scores[field] = None
                elif value not in ('0', '1', '2', '3'):
                    raise ValueError(f'Reviewer CSV {file_index}, row {line}: {field} must be an integer 0-3')
                else:
                    scores[field] = int(value)
            rationale = str(row.get('rationale') or '').strip()
            if len(rationale) < 12:
                raise ValueError(f'Reviewer CSV {file_index}, row {line}: explain decision in >=12 characters')
            verified = _truth(row.get('evidence_verified'))
            output.append({
                'annotation_id': a_id, 'record_sha256': record_sha256(annotation),
                'reviewer_id': reviewer, 'decision': decision,
                'evidence_verified': verified, 'scores': scores,
                'rationale': rationale,
            })
        messages.append(f'Reviewer CSV {file_index}: {len(rows)} rows read')
    return output, messages


def judge_consensus(
    annotations: list[dict[str, Any]], reviews: list[dict[str, Any]],
    *, min_reviewers: int = 2,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not 2 <= min_reviewers <= 5:
        raise ValueError('Gold release requires 2 to 5 independent reviewers')
    by_annotation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in reviews:
        by_annotation[r['annotation_id']].append(r)
    decisions: dict[str, dict[str, Any]] = {}
    reasons = Counter()
    disagreements = 0
    pairs = 0
    dimension_agree = Counter()
    dimension_total = Counter()
    for item in annotations:
        a_id = item['annotation_id']
        rs = by_annotation.get(a_id, [])
        labels = {r['decision'] for r in rs}
        if len(rs) >= 2:
            pairs += 1
            if len(labels) > 1:
                disagreements += 1
            for dimension in DIMENSIONS:
                ratings = [r['scores'][dimension] for r in rs if r['scores'].get(dimension) is not None]
                if len(ratings) >= 2:
                    dimension_total[dimension] += 1
                    if len(set(ratings)) == 1:
                        dimension_agree[dimension] += 1
        failures: list[str] = []
        if len(rs) < min_reviewers:
            failures.append('insufficient_independent_reviews')
        if item.get('review_status') == 'rejected':
            failures.append('editorially_rejected')
        if item.get('validation_flags'):
            failures.append('mechanical_validation_failed')
        if len(labels) > 1:
            failures.append('reviewer_disagreement')
        if any(r['decision'] != 'approve' for r in rs):
            failures.append('reviewer_nonapproval')
        if any(not r['evidence_verified'] for r in rs):
            failures.append('evidence_not_verified')
        if any(any(r['scores'].get(k) is None or r['scores'][k] < v for k, v in SCORES_MIN.items()) for r in rs):
            failures.append('below_rubric_threshold')
        if not rs:
            failures = ['no_human_rubric_reviews'] + failures
        for failure in failures:
            reasons[failure] += 1
        decisions[a_id] = {
            'annotation_id': a_id, 'record_sha256': record_sha256(item),
            'review_count': len(rs), 'ready': not failures,
            'blockers': list(dict.fromkeys(failures)),
            'decisions': list(sorted(labels)),
            'reviewer_ids': [r['reviewer_id'] for r in rs],
        }
    return decisions, {
        'human_rubric_reviews': len(reviews),
        'annotations_with_two_or_more_reviews': pairs,
        'decision_disagreement_items': disagreements,
        'decision_disagreement_rate': round(disagreements / pairs, 4) if pairs else None,
        'dimension_exact_agreement': {
            k: round(dimension_agree[k] / dimension_total[k], 4) if dimension_total[k] else None
            for k in DIMENSIONS
        },
        'blockers': dict(reasons),
        'gold_eligible_before_deduplication': sum(x['ready'] for x in decisions.values()),
    }


def parse_preferences(data: bytes, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {a['annotation_id']: a for a in annotations}
    rows = _parse_csv(data, tuple(PREFERENCE_FIELDS[:2]) + ('rejected_answer', 'reviewer_id', 'preference_reason', 'evidence_verified', 'review_status'))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, 2):
        if not str(row.get('review_status') or '').strip():
            continue
        if str(row['review_status']).strip().lower() != 'approved':
            continue
        a_id = str(row['annotation_id']).strip()
        item = lookup.get(a_id)
        if item is None:
            raise ValueError(f'Preference CSV line {line}: unknown annotation_id {a_id}')
        if record_sha256(item) != str(row.get('record_sha256') or '').strip():
            raise ValueError(f'Preference CSV line {line}: fingerprint mismatch for {a_id}')
        for visible, origin in (('question', 'question'), ('chosen_answer', 'answer')):
            if visible in row and _csv_visual_value(row[visible]) != str(item.get(origin, '')):
                raise ValueError(f'Preference CSV line {line}: {visible} was changed after review template generation')
        if a_id in seen:
            raise ValueError(f'Preference CSV line {line}: duplicate approved preference for {a_id}')
        seen.add(a_id)
        rejected = str(row.get('rejected_answer') or '').strip()
        chosen = str(item.get('answer') or '').strip()
        reviewer = str(row.get('reviewer_id') or '').strip()
        reason = str(row.get('preference_reason') or '').strip()
        if not rejected or len(rejected) > 5000 or rejected.casefold() == chosen.casefold():
            raise ValueError(f'Preference CSV line {line}: rejected_answer must be a distinct actual alternative')
        if len(reviewer) < 2 or len(reason) < 12 or not _truth(row.get('evidence_verified')):
            raise ValueError(f'Preference CSV line {line}: reviewer, specific reason and verification required')
        output.append({
            'annotation_id': a_id, 'source_id': item['source_id'],
            'reviewer_id': reviewer, 'reason': reason,
            'chosen': chosen, 'rejected': rejected,
        })
    return output
