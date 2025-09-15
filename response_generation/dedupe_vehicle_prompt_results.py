import pandas as pd

# File paths
input_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"
output_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_deduped.csv"

# Load the CSV
results_df = pd.read_csv(input_file)

# Detect correct prompt column
if "Prompt" in results_df.columns:
    prompt_col = "Prompt"
elif "prompt" in results_df.columns:
    prompt_col = "prompt"
else:
    raise ValueError("No 'Prompt' or 'prompt' column found in the CSV.")

# Drop duplicate prompts (keep first occurrence for each prompt/model/provider pair)
results_df.drop_duplicates(subset=[prompt_col, "provider", "model"], inplace=True)

# Save to new CSV
results_df.to_csv(output_file, index=False)

print(f"Deduplicated results saved to {output_file}")
print(f"Number of rows after deduplication: {len(results_df)}")
print(f"Number of unique prompts: {results_df[prompt_col].nunique()}")

# Count for each model
if "model" in results_df.columns:
    print("\nCounts by model:")
    for model, group in results_df.groupby("model"):
        print(f"{model}: {group[prompt_col].nunique()} unique prompts")
else:
    print("No 'model' column found for per-model counts.")
