#!/usr/bin/env python3
"""Compute counts of rows in Excel sheet, classified CSV, and their overlap.

Prints three lines: <excel_count>, <classified_count>, <overlap_count>
"""
from pathlib import Path
import pandas as pd
import logging
import argparse


def _normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_excel_sheet(path, sheet_name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
        return _normalize_cols(df)
    except ImportError as e:
        # pandas failed to import the optional engine for xlsx (openpyxl)
        # Try to fall back to a CSV with the same base name if available, otherwise
        # raise a clear error with instructions.
        csv_path = path.with_suffix('.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return _normalize_cols(df)
        raise ImportError(
            "Missing Excel engine (openpyxl). Install it with: pip install openpyxl\n"
            "Or export the Excel sheet to CSV at: {csv_path} and re-run this script."
        ) from e
    except Exception:
        # Re-raise so callers can see the original failure
        raise


def load_classified_csv(path):
    path = Path(path)
    if not path.exists():
        logging.warning(f"Classified CSV not found at {path}. Proceeding with empty classified set.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_cols(df)


ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = ROOT / "data" / "Prompt library.xlsx"
SHEET_NAME = "Results Vehicle Status Quo"
CLASSIFIED_CSV = ROOT / "data" / "vehicle_prompt_results_classified.csv"


def _norm(s):
    return "" if pd.isna(s) else " ".join(str(s).split())


def main():
    parser = argparse.ArgumentParser(description="Count overlap between Excel prompts and classified results")
    parser.add_argument('--prompt-only', action='store_true', help='Ignore prompt_id and count/compare by prompt text only (unique prompts)')
    args = parser.parse_args()
    prompt_only = args.prompt_only

    excel_df = load_excel_sheet(EXCEL_PATH, SHEET_NAME)
    # also load the deduped CSV which contains canonical prompt and prompt_id
    deduped_path = ROOT / "data" / "vehicle_prompt_results_deduped.csv"
    deduped_df = None
    if deduped_path.exists():
        deduped_df = pd.read_csv(deduped_path)
        deduped_df = _normalize_cols(deduped_df)

    # Read classified CSV robustly: it may have been written without a header.
    # Try reading with header first; if key columns missing, re-read with header=None
    classified_df = None
    if not CLASSIFIED_CSV.exists():
        classified_df = pd.DataFrame()
    else:
        df_try = pd.read_csv(CLASSIFIED_CSV)
        df_try = _normalize_cols(df_try)
        # If expected columns are present, use it. Otherwise read without header.
        if any(c.lower() in ("prompt_id", "prompt") for c in df_try.columns):
            classified_df = df_try
        else:
            df_nohdr = pd.read_csv(CLASSIFIED_CSV, header=None, dtype=str)
            # expected columns from classify_row output (approx)
            expected = [
                "prompt_id","provider","model","tokens_in","tokens_out","approx_tokens",
                "response","unique_vehicles_llm","num_unique_vehicles_llm",
                "num_ev_llm","num_phev_llm","num_hev_llm","num_icv_llm","llm_classifier_raw"
            ]
            ncols = df_nohdr.shape[1]
            if ncols >= len(expected):
                # assign expected to first len(expected), others get generated names
                cols = expected + [f"extra_{i}" for i in range(ncols - len(expected))]
            else:
                cols = expected[:ncols]
            df_nohdr.columns = cols
            classified_df = _normalize_cols(df_nohdr)

    # Two modes: prompt-only (compare unique prompt text) or default id-first behavior.
    overlap = 0
    if prompt_only:
        # Build excel prompt set (use excel sheet's prompt column)
        excel_prompt_col = find_column(excel_df, ["prompt", "prompt_text", "prompt text", "prompt_text_raw"]) 
        if excel_prompt_col is None:
            logging.warning('No prompt column found in Excel; falling back to deduped prompt column if available')
            if deduped_df is not None and 'prompt' in deduped_df.columns:
                excel_prompts = set(_norm(x) for x in deduped_df['prompt'].fillna(""))
            else:
                excel_prompts = set()
        else:
            excel_prompts = set(_norm(x) for x in excel_df[excel_prompt_col].fillna(""))

        # Build classified prompt set
        class_prompt_col = find_column(classified_df, ["prompt", "prompt_text", "prompt text", "prompt_text_raw"]) if not classified_df.empty else None
        if class_prompt_col:
            classified_prompts = set(_norm(x) for x in classified_df[class_prompt_col].fillna(""))
        else:
            # Try to reconstruct from deduped mapping via prompt_id
            if deduped_df is not None and 'prompt_id' in deduped_df.columns and 'prompt' in deduped_df.columns:
                # build id->prompt map
                id_to_prompt = dict(zip(deduped_df['prompt_id'].astype(str), deduped_df['prompt'].astype(str)))
                # find prompt_id column in classified (or assume first col if headerless)
                class_id_col = find_column(classified_df, ["prompt_id", "prompt id", "id", "promptid"]) if not classified_df.empty else None
                if class_id_col:
                    ids = classified_df[class_id_col].dropna().astype(str).tolist()
                else:
                    # try headerless read: assume first column contains prompt_id values
                    try:
                        tmp = pd.read_csv(CLASSIFIED_CSV, header=None, dtype=str)
                        ids = tmp[0].dropna().astype(str).tolist()
                    except Exception:
                        ids = []
                classified_prompts = set(_norm(id_to_prompt.get(pid, "")) for pid in ids if pid in id_to_prompt)
            else:
                # as last resort, try reading first column of classified as raw prompt text
                try:
                    tmp = pd.read_csv(CLASSIFIED_CSV, header=None, dtype=str)
                    classified_prompts = set(_norm(x) for x in tmp[0].fillna(""))
                except Exception:
                    classified_prompts = set()

        excel_count = len(excel_prompts)
        classified_count = len(classified_prompts)
        overlap = len(excel_prompts & classified_prompts)
    else:
        # Try id-based matching first (prefer stable ids from deduped CSV)
        excel_id_col = find_column(excel_df, ["prompt_id", "prompt id", "id", "promptid", "prompt_id_raw"]) or None
        # If deduped_df is present, use its prompt_id mapping as the authoritative excel ids
        if deduped_df is not None and "prompt_id" in deduped_df.columns and "prompt" in deduped_df.columns:
            # Use deduped_df as the excel source for counting/ids
            canonical_df = deduped_df
            excel_count = len(canonical_df)
        else:
            canonical_df = excel_df
            excel_count = len(canonical_df)

        class_id_col = find_column(classified_df, ["prompt_id", "prompt id", "id", "promptid"]) if not classified_df.empty else None

        classified_count = len(classified_df)

        if class_id_col:
            # build excel id set from canonical_df (prefer deduped mapping)
            if "prompt_id" in canonical_df.columns:
                excel_ids = set(canonical_df['prompt_id'].dropna().astype(str))
            elif excel_id_col:
                excel_ids = set(canonical_df[excel_id_col].dropna().astype(str))
            else:
                excel_ids = set()
            class_ids = set(classified_df[class_id_col].dropna().astype(str))
            overlap = len(excel_ids & class_ids)
        else:
            # Fallback to prompt-text based matching (preferred) then response text
            excel_prompt_col = find_column(canonical_df, ["prompt", "prompt_text", "prompt text", "prompt_text_raw"]) 
            class_prompt_col = find_column(classified_df, ["prompt", "prompt_text", "prompt text", "prompt_text_raw"]) if not classified_df.empty else None
            if excel_prompt_col and class_prompt_col:
                excel_norm = set(_norm(x) for x in canonical_df[excel_prompt_col].fillna("") if excel_prompt_col in canonical_df.columns)
                class_norm = set(_norm(x) for x in classified_df[class_prompt_col].fillna(""))
                overlap = len(excel_norm & class_norm)
            else:
                # Fallback to normalized response match
                excel_resp_col = find_column(canonical_df, ["response", "response_text", "text"]) 
                class_resp_col = find_column(classified_df, ["response", "response_text", "text"]) if not classified_df.empty else None
                if excel_resp_col and class_resp_col:
                    excel_norm = set(_norm(x) for x in canonical_df[excel_resp_col].fillna(""))
                    class_norm = set(_norm(x) for x in classified_df[class_resp_col].fillna(""))
                    overlap = len(excel_norm & class_norm)
                else:
                    # Last resort: no reliable keys, set overlap to 0
                    overlap = 0

    print(excel_count)
    print(classified_count)
    print(overlap)


if __name__ == "__main__":
    main()
