#!/usr/bin/env python3
"""Join deduped prompts to classified rows by response text and export a cleaned CSV.

Output columns: original_prompt, response, classification, llm_classifier_raw_json, model
"""
from pathlib import Path
import pandas as pd
import logging
import json


ROOT = Path(__file__).resolve().parent.parent
DEDUPED_PATH = ROOT / 'data' / 'vehicle_prompt_results_deduped.csv'
CLASSIFIED_PATH = ROOT / 'data' / 'vehicle_prompt_results_classified.csv'
OUT_PATH = ROOT / 'data' / 'vehicle_prompt_results_joined_by_response.csv'


def _norm_text(x):
    if pd.isna(x):
        return ''
    return ' '.join(str(x).lower().split())


def find_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_deduped(path):
    df = pd.read_csv(path, dtype=str)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # expected: prompt_id, prompt, response maybe
    # keep prompt text as original_prompt
    if 'prompt' not in [c.lower() for c in df.columns]:
        raise ValueError('deduped CSV missing a prompt column')
    # ensure a response column exists? not necessary here
    return df


def load_classified(path):
    # Many exports were written without a header. Parse headerless using the
    # expected column order emitted by the classifier pipeline.
    # Expected columns (approx):
    expected = [
        "prompt_id","provider","model","tokens_in","tokens_out","approx_tokens",
        "response","unique_vehicles_llm","num_unique_vehicles_llm",
        "num_ev_llm","num_phev_llm","num_hev_llm","num_icv_llm","llm_classifier_raw"
    ]
    df = pd.read_csv(path, header=None, dtype=str)
    n = df.shape[1]
    if n >= len(expected):
        cols = expected + [f'extra_{i}' for i in range(n - len(expected))]
    else:
        cols = expected[:n]
    df.columns = cols
    # strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def choose_response_col(df):
    # Prefer columns with response-like names
    candidates = ['response', 'response_text', 'text', 'llm_response', 'llm_response_text']
    col = find_column(df, candidates)
    if col:
        return col
    # fallback: look for a long text-like column (max length)
    lengths = {c: df[c].astype(str).map(len).median() for c in df.columns}
    # pick column with median length > 20, else None
    sorted_by_len = sorted(lengths.items(), key=lambda x: x[1], reverse=True)
    if sorted_by_len and sorted_by_len[0][1] > 20:
        return sorted_by_len[0][0]
    return None


def choose_class_col(df):
    candidates = ['llm_classifier_raw', 'classification', 'classification_label', 'llm_classification', 'llm_classifier']
    return find_column(df, candidates)


def choose_model_col(df):
    candidates = ['model', 'llm_model', 'provider', 'model_name']
    return find_column(df, candidates)


def main():
    logging.basicConfig(level=logging.INFO)
    deduped = load_deduped(DEDUPED_PATH)
    classified = load_classified(CLASSIFIED_PATH)

    resp_col = choose_response_col(classified)
    class_col = choose_class_col(classified)
    model_col = choose_model_col(classified)

    logging.info(f'Chosen response column: %s', resp_col)
    logging.info(f'Chosen classification column: %s', class_col)
    logging.info(f'Chosen model column: %s', model_col)

    # Build normalized -> deduped prompt mapping by response if deduped has response
    deduped_cols = [c.lower() for c in deduped.columns]
    # Build normalized -> deduped prompt mapping by prompt text (deduped holds canonical prompts)
    if 'prompt' in [c.lower() for c in deduped.columns]:
        # deduped prompt column name (case-sensitive in DF)
        prompt_col = next(c for c in deduped.columns if c.lower() == 'prompt')
    else:
        raise RuntimeError('deduped CSV missing a prompt column')

    # create prompt_text -> set of prompts (usually one-to-one)
    resp_to_prompt = {}
    for _, r in deduped.iterrows():
        ptxt = r.get(prompt_col, '')
        k = _norm_text(ptxt)
        if k:
            resp_to_prompt.setdefault(k, []).append(ptxt)

    out_rows = []

    # If classified has a response column, join directly; otherwise attempt to use prompt_id mapping
    if resp_col:
        for _, row in classified.iterrows():
            resp = row.get(resp_col, '')
            key = _norm_text(resp)
            original_prompt = resp_to_prompt.get(key)
            # if not found, try to use prompt_id from classified and lookup in deduped
            if original_prompt is None and 'prompt_id' in classified.columns:
                pid = row.get('prompt_id')
                if pid and 'prompt_id' in deduped.columns:
                    match = deduped[deduped['prompt_id'].astype(str) == str(pid)]
                    if not match.empty:
                        original_prompt = match.iloc[0].get(next(c for c in match.columns if c.lower() == 'prompt'))
            # Prepare output fields
            classification = row.get(class_col) if class_col else ''
            model = row.get(model_col) if model_col else ''
            # try to extract raw json field if present
            raw_json = None
            for c in classified.columns:
                if 'json' in c.lower() or c.lower() == 'llm_classifier_raw':
                    raw_json = row.get(c)
                    break
            # if raw_json looks like a dict string, try to pretty print or keep as-is
            try:
                if raw_json and isinstance(raw_json, str):
                    # ensure valid json or keep original
                    parsed = json.loads(raw_json)
                    raw_json = json.dumps(parsed)
            except Exception:
                # leave raw_json as-is
                pass

            out_rows.append({
                'original_prompt': original_prompt or '',
                'response': resp or '',
                'classification': classification or '',
                'llm_classifier_raw_json': raw_json or '',
                'model': model or ''
            })
    else:
        # No response column detected; attempt to map using prompt_id -> prompt mapping
        if 'prompt_id' in classified.columns and 'prompt_id' in deduped.columns:
            id_to_prompt = dict(zip(deduped['prompt_id'].astype(str), deduped['prompt'].astype(str)))
            for _, row in classified.iterrows():
                pid = row.get('prompt_id')
                original_prompt = id_to_prompt.get(str(pid), '')
                # find a candidate response column by longest text heuristic
                resp = ''
                resp_cand = choose_response_col(classified)
                if resp_cand:
                    resp = row.get(resp_cand, '')
                classification = row.get(class_col) if class_col else ''
                model = row.get(model_col) if model_col else ''
                raw_json = ''
                for c in classified.columns:
                    if 'json' in c.lower() or 'llm_classifier_raw' == c.lower():
                        raw_json = row.get(c)
                        break
                out_rows.append({
                    'original_prompt': original_prompt or '',
                    'response': resp or '',
                    'classification': classification or '',
                    'llm_classifier_raw_json': raw_json or '',
                    'model': model or ''
                })
        else:
            raise RuntimeError('Unable to locate response or prompt_id columns to join on')

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False)
    logging.info('Wrote joined CSV to %s (rows=%d)', OUT_PATH, len(out_df))


if __name__ == '__main__':
    main()
