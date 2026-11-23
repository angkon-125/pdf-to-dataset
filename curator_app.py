"""Angkon Human Annotator + Data Scientist local workbench.

Start: streamlit run curator_app.py
This standalone app is not a multi-user authentication system.
"""
from __future__ import annotations

import hashlib
import io
import json

import pandas as pd
import streamlit as st

from angkon_factory.curation import curate_archive, load_source_archive, profile_archive
from angkon_factory.expert_review import DIMENSIONS, REVIEW_FIELDS, record_sha256, _csv_rows
from angkon_factory.exports import csvtext

st.set_page_config(page_title='Angkon | Annotation & Data Science Lab', page_icon='🔬', layout='wide')
st.title('Angkon Annotation & Data Science Lab')
st.caption('v0.3 · Two independent human rubric reviews → quality gates → gold SFT and genuine human preferences')
st.warning('Local prototype only. Reviewer identities are self-declared; not suitable for unsecured multi-user deployment.')
source = st.file_uploader('Upload a dataset ZIP produced by Angkon PDF-to-Dataset v0.2+', type=['zip'])
if not source:
    st.info('First generate draft Q&A using the main app or CLI. Upload that dataset ZIP here to build independently reviewed gold datasets.')
    st.stop()

payload = source.getvalue()
archive_sha = hashlib.sha256(payload).hexdigest()
if st.session_state.get('active_archive_sha') != archive_sha:
    st.session_state.active_archive_sha = archive_sha
    st.session_state.pop('latest_gold_release', None)
    st.session_state.saved_review_rows = {}

try:
    data = load_source_archive(payload)
    profile = profile_archive(payload)
except ValueError as exc:
    st.error(f'Cannot use this archive: {exc}')
    st.stop()

annotations = data['annotations']
chunks = {item['chunk_id']: item for item in data['chunks']}
rows_by_id = {r['annotation_id']: r for r in annotations}
preflight = profile['quality']
metrics = st.columns(4)
metrics[0].metric('Draft candidates', len(annotations))
metrics[1].metric('Sources', preflight['documents'])
metrics[2].metric('Potential duplicate groups', preflight['duplicate_candidate_question_groups'])
metrics[3].metric('Source text pages', preflight['pages_with_text'])

tabs = st.tabs(['Human annotator', 'Data scientist QA', 'Gold release'])
with tabs[0]:
    st.subheader('Independent evidence-based labeling')
    st.write('Review the source chunk and original PDF page. Score the answer from 0 (major failure) to 3 (meets criteria). A reviewer must confirm evidence support, not merely that a quote exists.')
    st.download_button('Download blank rubric CSV (for each reviewer)', profile['rubric_template'],
                       file_name='angkon_rubric_template.csv', mime='text/csv')
    st.caption('Each annotator should make decisions independently and save a separate CSV. This app does not authenticate reviewer identities.')
    if not annotations:
        st.info('No Q&A candidates yet. Return to the main PDF generation app first.')
    else:
        reviewer = st.text_input('Reviewer ID', placeholder='example: annotator_01').strip()
        review_queue = profile['risk_queue']
        st.caption('Higher-priority items have more potential extraction, formatting or privacy risks. Priority is a heuristic, not correctness confidence.')
        ids = [x['annotation_id'] for x in review_queue]
        selected_id = st.selectbox('Select an example (highest review priority first)', ids,
                                   format_func=lambda x: f"{x} — {rows_by_id[x].get('question','')[:95]}")
        selected = rows_by_id[selected_id]
        chunk = chunks.get(selected['chunk_id'])
        left, right = st.columns([1, 1])
        with left:
            st.markdown('**Prompt/question**')
            st.write(selected.get('question', ''))
            st.markdown('**Model answer (unverified)**')
            st.write(selected.get('answer', ''))
            st.markdown('**Claimed evidence quotation**')
            st.code(selected.get('evidence_quote', ''), language=None, wrap_lines=True)
            st.caption(f"Source: {selected.get('source_file','')} • PDF page {selected.get('source_page','')} • {selected.get('section_title','')}")
        with right:
            st.markdown('**Original extracted chunk**')
            if chunk:
                st.text_area('Source text (read-only)', chunk.get('text',''), height=270, disabled=True,
                             key='src_' + selected_id)
            else:
                st.error('Missing source chunk; cannot approve.')
            st.markdown("**Reviewer's decision**")
            saved = st.session_state.saved_review_rows.get((reviewer.casefold(), selected_id), {})
            status = st.selectbox('Decision', ['unreviewed', 'approve', 'revise', 'reject'],
                                  index=['unreviewed','approve','revise','reject'].index(saved.get('decision','unreviewed')),
                                  key='decision_' + reviewer + '_' + selected_id)
            confirmed = st.checkbox('Evidence actually supports the answer and I checked the original document',
                                    value=saved.get('evidence_verified','') == 'yes',
                                    key='verified_' + reviewer + '_' + selected_id)
            st.caption('0 = fail · 1 = significant issues · 2 = minor issues · 3 = satisfies criterion')
            scores = {}
            for dimension in DIMENSIONS:
                scores[dimension] = st.select_slider(dimension.replace('_',' ').title(),
                    options=[0,1,2,3], value=int(saved.get(dimension, '1') or 1),
                    key='score_' + dimension + '_' + reviewer + '_' + selected_id)
            rationale = st.text_area('Reasoning: cite specific evidence and exact defects',
                                     value=saved.get('rationale',''), height=90,
                                     key='reason_' + reviewer + '_' + selected_id)
            if st.button('Save this human review', type='primary'):
                if len(reviewer) < 2:
                    st.error('Enter a reviewer ID (at least 2 characters).')
                elif status == 'unreviewed':
                    st.error('Choose approve, revise, or reject.')
                elif len(rationale.strip()) < 12:
                    st.error('Provide a specific rationale of at least 12 characters.')
                else:
                    st.session_state.saved_review_rows[(reviewer.casefold(), selected_id)] = {
                        'annotation_id': selected_id,
                        'record_sha256': record_sha256(selected),
                        'reviewer_id': reviewer, 'decision': status,
                        'evidence_verified': 'yes' if confirmed else 'no',
                        **{k: str(v) for k,v in scores.items()}, 'rationale': rationale.strip(),
                    }
                    st.success('Review saved in this browser session. Download your reviewer CSV to preserve it.')
        if reviewer:
            reviewed = [row for (r, _), row in st.session_state.saved_review_rows.items() if r == reviewer.casefold()]
            st.caption(f'{len(reviewed)} human decisions saved for {reviewer}')
            safe_reviewer_name = ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in reviewer)[:80]
            st.download_button('Download my reviewer CSV', _csv_rows(reviewed, REVIEW_FIELDS),
                               file_name=f'angkon_reviews_{safe_reviewer_name}.csv',
                               mime='text/csv', disabled=not reviewed)

with tabs[1]:
    st.subheader('Data scientist preflight')
    st.write('Inspect coverage and diversity, then use the prioritized queue to focus annotation work.')
    left, right = st.columns(2)
    with left:
        st.markdown('**Language and task distributions**')
        st.json({'languages': preflight['candidate_languages'], 'task_types': preflight['candidate_kinds'],
                 'document_types': preflight['candidate_document_types']})
    with right:
        st.markdown('**Source and length diagnostics**')
        st.json({'documents': preflight['documents'], 'source_page_coverage': preflight['page_text_coverage'],
                 'question_lengths': preflight['candidate_question_length'],
                 'answer_lengths': preflight['candidate_answer_length']})
    st.dataframe(pd.DataFrame(profile['risk_queue']), use_container_width=True, hide_index=True)
    st.download_button('Download risk-ranked reviewer queue',
                       csvtext(profile['risk_queue'], ['annotation_id','source_file','source_page','review_status','risk_priority','risk_reasons','question']).encode('utf-8-sig'),
                       file_name='review_priority.csv', mime='text/csv')
    st.download_button('Download preflight quality JSON', json.dumps(preflight, indent=2, ensure_ascii=False).encode(),
                       file_name='angkon_preflight_quality.json', mime='application/json')
    st.warning('These quality metrics are heuristics. Strong human labels and evaluation are necessary before claiming a model is improved.')

with tabs[2]:
    st.subheader('Gold training-data release')
    st.write('Upload rubric CSVs from two *distinct* human reviewers. The release process checks fingerprints, factuality/grounding/safety scores, evidence verification, disagreement, duplicates and source-level splitting.')
    rubric_uploads = st.file_uploader('Independent completed reviewer CSVs', type=['csv'], accept_multiple_files=True, key='human_csvs')
    editorial = st.file_uploader('Optional: corrected annotations_review.csv (edits must precede rubric review)', type=['csv'], key='editorial_csv')
    preference = st.file_uploader('Optional: real rejected response + human preference CSV', type=['csv'], key='dpo_csv')
    st.download_button('Download blank preference CSV', profile['preference_template'],
                       file_name='angkon_preference_template.csv', mime='text/csv')
    min_reviewers = st.number_input('Independent approving reviewers required', 2, 5, 2)
    st.caption('Strict gold rubric: factual accuracy, groundedness and safety = 3; instruction following, completeness and clarity ≥ 2. Both reviewers must explicitly approve.')
    rights = st.checkbox('I confirm the necessary rights and permissions for this dataset and intended model training.')
    attested = st.checkbox('I attest that these CSV ratings were completed by independent human reviewers, not auto-filled by a model.')
    active_inputs = [archive_sha.encode(), str(min_reviewers).encode(), str(rights).encode(), str(attested).encode()]
    active_inputs.extend(x.getvalue() for x in (rubric_uploads or []))
    if editorial:
        active_inputs.append(editorial.getvalue())
    if preference:
        active_inputs.append(preference.getvalue())
    current_release_key = hashlib.sha256(b'\x00'.join(active_inputs)).hexdigest()
    if st.session_state.get('current_release_key') != current_release_key:
        st.session_state.current_release_key = current_release_key
        st.session_state.pop('latest_gold_release', None)
    if st.button('Run data-science quality gates and build gold ZIP', type='primary',
                 disabled=not (rubric_uploads and rights and attested)):
        try:
            bundle, result = curate_archive(
                payload, reviewer_csvs=[x.getvalue() for x in rubric_uploads],
                edited_annotations_csv=editorial.getvalue() if editorial else None,
                preferences_csv=preference.getvalue() if preference else None,
                rights_attested=rights, human_review_attested=attested, min_reviewers=int(min_reviewers),
            )
            st.session_state.latest_gold_release = (bundle, result)
            st.success('Curated ZIP created. Inspect gold counts and blockers before training.')
        except ValueError as exc:
            st.error(f'Quality gates rejected the input: {exc}')
    release = st.session_state.get('latest_gold_release')
    if release:
        bundle, result = release
        a,b,c = st.columns(3)
        a.metric('Gold SFT records', result['gold_sft_records'])
        b.metric('Human preference pairs', result['human_approved_preference_pairs'])
        c.metric('Independent reviews', result['reviewer_submissions'])
        st.json({'split_counts':result['gold_split_counts'],'status':result['dataset_status']})
        st.download_button('Download strictly reviewed gold dataset ZIP', bundle,
                           file_name='Angkon_Gold_Training_Dataset_v0.3.zip', mime='application/zip', type='primary')
        with io.BytesIO(bundle) as buffer:
            import zipfile
            with zipfile.ZipFile(buffer) as z:
                st.download_button('Download detailed data-science report', z.read('data_science_quality.json'),
                                   file_name='angkon_data_science_quality.json', mime='application/json')

st.divider()
st.caption('Keep original PDFs/private source text off public GitHub. Gold SFT requires manual evidence verification; DPO requires independently supplied rejected answers. This workbench does not fine-tune a model.')
