import os
import sys
import pandas as pd

#!/usr/bin/env python3
# File: /Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/classifier/manual_check.py
# Purpose: build lists of prompts answered per model from joined CSV and list of prompts
#          from deduped CSV that are not present in the joined CSV. Print lengths.


JOINED_CSV = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_joined_by_response.csv"
DEDUPED_CSV = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_deduped.csv"


def read_csv_safe(path):
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def choose_column(df, candidates):
    cols = [c for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        for key, orig in lower.items():
            if cand in key:
                return orig
    # fallback to first string-like column
    for c in cols:
        if df[c].dtype == object or df[c].dtype == "string":
            return c
    return cols[0]


def normalize_prompts(s):
    # basic normalization: strip whitespace and collapse internal whitespace
    return " ".join(s.split()) if isinstance(s, str) else s


def main():
    joined = read_csv_safe(JOINED_CSV)
    deduped = read_csv_safe(DEDUPED_CSV)

    # heuristics to find the prompt column; prefer explicit known names
    prompt_col_joined = None
    if 'original_prompt' in joined.columns:
        prompt_col_joined = 'original_prompt'
    else:
        prompt_col_joined = choose_column(joined, ["prompt", "text", "input", "question"])

    prompt_col_deduped = None
    if 'prompt' in deduped.columns:
        prompt_col_deduped = next(c for c in deduped.columns if c.lower() == 'prompt')
    else:
        prompt_col_deduped = choose_column(deduped, ["prompt", "text", "input", "question"])

    # heuristics to find the model column in joined
    model_col = choose_column(joined, ["model", "model_name", "model-id", "model_id", "architecture"])
    # If the chosen model column is actually the prompt (rare), detect and fallback to None
    if prompt_col_joined == model_col:
        model_col = None

    # Build set of deduped prompts
    deduped_prompts = {
        normalize_prompts(p) for p in deduped[prompt_col_deduped].astype(str).tolist() if str(p).strip() != ""
    }

    # Build mapping model -> prompts answered (from joined file)
    answered_by_model = {}
    # If no model column found, group everything under "unknown"
    if model_col is None:
        model_col_values = ["unknown"]
        joined["unknown"] = "unknown"
        model_col = "unknown"

    for _, row in joined.iterrows():
        raw_prompt = str(row.get(prompt_col_joined, "")).strip()
        if raw_prompt == "":
            continue
        prompt = normalize_prompts(raw_prompt)
        model = str(row.get(model_col, "")).strip() or "unknown"
        answered_by_model.setdefault(model, set()).add(prompt)

    # Convert sets to sorted lists
    answered_by_model_lists = {m: sorted(list(ps)) for m, ps in answered_by_model.items()}

    # Prompts present in joined (any model)
    joined_prompts = set()
    for s in answered_by_model_lists.values():
        joined_prompts.update(s)

    # Prompts in deduped that are NOT in joined (unclassified)
    unclassified_prompts = sorted(list(deduped_prompts - joined_prompts))

    # Print lengths
    print(f"Total unique prompts in deduped file: {len(deduped_prompts)}")
    print(f"Total unique prompts present in joined file: {len(joined_prompts)}")
    print(f"Number of unclassified prompts (in deduped but not in joined): {len(unclassified_prompts)}")
    print()

    print("Answered prompts per model (model: count):")
    for model, prompts in sorted(answered_by_model_lists.items(), key=lambda x: x[0]):
        print(f"  {model}: {len(prompts)}")

    # If you want the actual lists available in memory, they are:
    #   answered_by_model_lists  -> dict model -> list of prompts answered by that model
    #   unclassified_prompts     -> list of prompts not present in joined file
    #
    # For simple verification we print the first few of each:
    print()
    SAMPLE = 5
    for model, prompts in sorted(answered_by_model_lists.items(), key=lambda x: x[0]):
        sample = prompts[:SAMPLE]
        print(f"Sample prompts for model '{model}' (up to {SAMPLE}):")
        for p in sample:
            print(f"   - {p}")
        if len(prompts) > SAMPLE:
            print(f"   ... (+{len(prompts)-SAMPLE} more)")
        print()

    print(f"Sample unclassified prompts (up to {SAMPLE}):")
    for p in unclassified_prompts[:SAMPLE]:
        print(f"   - {p}")
    if len(unclassified_prompts) > SAMPLE:
        print(f"   ... (+{len(unclassified_prompts)-SAMPLE} more)")


if __name__ == "__main__":
    main()