import pandas as pd
import os

RESULTS_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_deduped.csv"
UNIQUE_PROMPTS_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/unique_prompts.csv"

# Load deduped results
results_df = pd.read_csv(RESULTS_FILE)

# Detect correct prompt column
if "Prompt" in results_df.columns:
    prompt_col = "Prompt"
elif "prompt" in results_df.columns:
    prompt_col = "prompt"
else:
    raise ValueError("No 'Prompt' or 'prompt' column found in the results CSV.")

# Unique prompts in results
unique_prompts_in_results = set(results_df[prompt_col].dropna().astype(str))
print(f"Number of unique prompts in results: {len(unique_prompts_in_results)}")

# Unique prompts per model
if "model" in results_df.columns:
    print("\nUnique prompts per model:")
    for model, group in results_df.groupby("model"):
        print(f"{model}: {group[prompt_col].nunique()} unique prompts")
else:
    print("No 'model' column found for per-model counts.")

# Load unique prompts from the prompt library
if os.path.exists(UNIQUE_PROMPTS_FILE):
    unique_df = pd.read_csv(UNIQUE_PROMPTS_FILE)
    unique_prompts_library = set(unique_df["Prompt"].dropna().astype(str))
    print(f"\nNumber of unique prompts in prompt library: {len(unique_prompts_library)}")
    print(f"Number of prompts in both: {len(unique_prompts_in_results & unique_prompts_library)}")
    print(f"Number only in results: {len(unique_prompts_in_results - unique_prompts_library)}")
    print(f"Number only in prompt library: {len(unique_prompts_library - unique_prompts_in_results)}")
else:
    print(f"Prompt library file {UNIQUE_PROMPTS_FILE} not found.")
