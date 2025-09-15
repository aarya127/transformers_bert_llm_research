import pandas as pd
import csv

# File paths
csv_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"
unique_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/unique_prompts.csv"
output_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/unique_prompts_not_in_results.csv"

# Load unique prompts from the new CSV
unique_df = pd.read_csv(unique_file)
unique_prompts = set(unique_df["Prompt"].dropna().astype(str))

# Load prompts from results CSV
csv_prompts = set()
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "Prompt" in row:
            csv_prompts.add(str(row["Prompt"]))
        elif "prompt" in row:
            csv_prompts.add(str(row["prompt"]))

# Filter unique prompts to only those not in results
not_in_results = unique_df[~unique_df["Prompt"].astype(str).isin(csv_prompts)]
not_in_results.to_csv(output_file, index=False)

print(len(not_in_results))
