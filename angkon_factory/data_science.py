"""Low-memory, transparent training-data QA and reviewer prioritization.

Checks are *diagnostic*; no metrics establish factual truth or copyright rights.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from .core import pii_counts
from .generation import norm


def _summary_numbers(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {'count': 0, 'min': None, 'median': None, 'mean': None, 'max': None, 'p95': None}
    ordered = sorted(values)
    return {
        'count': len(values), 'min': ordered[0], 'median': median(ordered),
        'mean': round(mean(ordered), 1), 'max': ordered[-1],
        'p95': ordered[math.ceil(0.95 * len(values)) - 1],
    }


def risk_score(item: dict[str, Any], chunk: dict[str, Any] | None = None) -> tuple[int, list[str]]:
    points = 0
    reasons = []
    flags = item.get('validation_flags', [])
    if flags:
        points += 70
        reasons.extend(f'validation:{flag}' for flag in flags)
    if str(item.get('language', '')).lower().startswith('unknown'):
        points += 10
        reasons.append('language_unknown')
    if len(str(item.get('answer', '')).strip()) < 40:
        points += 10
        reasons.append('very_short_answer')
    if len(str(item.get('question', '')).strip()) < 20:
        points += 8
        reasons.append('very_short_question')
    if len(str(item.get('answer', '')).strip()) > 1500:
        points += 8
        reasons.append('long_answer')
    if chunk is not None and chunk.get('extraction_method') == 'ocr':
        points += 20
        reasons.append('ocr_needs_visual_check')
    if chunk is not None and int(chunk.get('embedded_images', 0) or 0) > 0:
        points += 10
        reasons.append('figures_may_have_been_omitted')
    if sum(pii_counts(' '.join(str(item.get(k, '')) for k in ('question', 'answer', 'evidence_quote'))).values()):
        points += 50
        reasons.append('potential_pii')
    return min(points, 100), reasons


def prioritized_queue(annotations: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {c['chunk_id']: c for c in chunks}
    queue = []
    for x in annotations:
        priority, reasons = risk_score(x, lookup.get(x.get('chunk_id')))
        queue.append({
            'annotation_id': x['annotation_id'], 'source_file': x.get('source_file'),
            'source_page': x.get('source_page'), 'review_status': x.get('review_status'),
            'risk_priority': priority, 'risk_reasons': ';'.join(reasons),
            'question': x.get('question', ''),
        })
    return sorted(queue, key=lambda r: (-r['risk_priority'], str(r.get('source_file')), str(r['annotation_id'])))


def deduplicate_release(eligible: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Conservative: keep one example per exact normalized question, regardless of PDF.

    Also quarantine trivially similar questions within the same source. This is
    intentionally modest and is not semantic-deduplication.
    """
    seen: dict[str, str] = {}
    kept = []
    dropped: dict[str, str] = {}
    for item in sorted(eligible, key=lambda x: (x['source_id'], str(x.get('source_page')), x['annotation_id'])):
        question = norm(str(item['question']))
        if question in seen:
            dropped[item['annotation_id']] = f'duplicate_of:{seen[question]}'
        else:
            seen[question] = item['annotation_id']
            kept.append(item)
    return kept, dropped


def _tokens(text: str) -> set[str]:
    return set(re.findall(r'\w+', str(text).casefold(), flags=re.UNICODE))


def inspect_cross_split_overlap(
    items: list[dict[str, Any]], by_source: dict[str, str], *,
    max_pairs: int = 50000,
) -> dict[str, Any]:
    """Exact prompt reuse plus bounded approximate lexical overlap across splits."""
    exact: list[dict[str, str]] = []
    near: list[dict[str, Any]] = []
    compared = 0
    limited = False
    seen: dict[str, dict[str, str]] = {}
    for x in items:
        question = norm(str(x['question']))
        split = by_source.get(x['source_id'], 'train')
        if question in seen and seen[question]['split'] != split:
            exact.append({'annotation_id': x['annotation_id'], 'other': seen[question]['id']})
        else:
            seen[question] = {'id': x['annotation_id'], 'split': split}
    # Compare only similarly-sized questions to limit work; no semantic guarantees.
    token_sets = [(x, _tokens(x['question'])) for x in items]
    for index, (a, ta) in enumerate(token_sets):
        if len(near) >= 100:
            break
        for b, tb in token_sets[index+1:]:
            if by_source.get(a['source_id']) == by_source.get(b['source_id']):
                continue
            if compared >= max_pairs:
                limited = True
                break
            compared += 1
            if not ta or not tb or min(len(ta), len(tb)) / max(len(ta), len(tb)) < 0.85:
                continue
            similarity = len(ta & tb) / len(ta | tb)
            if similarity >= 0.90 and norm(a['question']) != norm(b['question']):
                near.append({'id_a': a['annotation_id'], 'id_b': b['annotation_id'], 'token_jaccard': round(similarity, 3)})
        if limited:
            break
    return {
        'cross_split_exact_prompts': exact,
        'cross_split_near_prompt_examples': near,
        'near_overlap_pairs_compared': compared,
        'near_overlap_audit_truncated': limited,
        'note': 'Lexical overlap checks cannot rule out paraphrase or semantic leakage.',
    }


def quality_profile(
    documents: list[dict[str, Any]], chunks: list[dict[str, Any]],
    annotations: list[dict[str, Any]], gold: list[dict[str, Any]],
    split_sources: dict[str, str],
) -> dict[str, Any]:
    pages = sum(int(d.get('processed_pages', 0)) for d in documents)
    with_text = sum(bool(p.get('text', '').strip()) for d in documents for p in d.get('pages', []))
    questions = Counter(norm(x.get('question', '')) for x in annotations if x.get('question'))
    document_grouped = Counter(x.get('source_id') for x in gold)
    gold_splits = Counter(split_sources.get(x['source_id'], 'train') for x in gold)
    overlaps = inspect_cross_split_overlap(gold, split_sources)
    return {
        'documents': len(documents),
        'pages_processed': pages,
        'pages_with_text': with_text,
        'page_text_coverage': round(with_text / pages, 4) if pages else None,
        'chunks': len(chunks),
        'candidate_count': len(annotations),
        'gold_count': len(gold),
        'duplicate_candidate_question_groups': sum(n > 1 for n in questions.values()),
        'candidate_kinds': dict(Counter(x.get('kind', 'unknown') for x in annotations)),
        'candidate_languages': dict(Counter(x.get('language', 'unknown') for x in annotations)),
        'candidate_document_types': dict(Counter(x.get('document_type', 'unknown') for x in annotations)),
        'candidate_question_length': _summary_numbers([len(x.get('question', '')) for x in annotations]),
        'candidate_answer_length': _summary_numbers([len(x.get('answer', '')) for x in annotations]),
        'gold_per_source': dict(document_grouped),
        'gold_per_split': {name: gold_splits[name] for name in ('train', 'validation', 'test')},
        'gold_answer_length': _summary_numbers([len(x.get('answer', '')) for x in gold]),
        'cross_split_overlap_audit': overlaps,
        'warnings': [
            'Gold labels are human judgments, not independently verified facts.',
            'Source-level splits do not exclude shared underlying text between different PDFs.',
            'Automatic language, copyright, and privacy checks are incomplete.',
        ],
    }
