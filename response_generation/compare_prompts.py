import pandas as pd
import csv
from collections import Counter

# File paths
csv_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"
excel_file = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/Prompt library.xlsx"
sheet_name = "Vehicle status quo"

# Load prompts from Excel
excel_df = pd.read_excel(excel_file, sheet_name=sheet_name)
excel_prompts = excel_df["Prompt"].dropna().astype(str)

# Count duplicates in Excel
excel_counts = Counter(excel_prompts)
excel_duplicates = {p: c for p, c in excel_counts.items() if c > 1}

# Load prompts from CSV
csv_prompts = []
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "Prompt" in row:
            csv_prompts.append(str(row["Prompt"]))
        elif "prompt" in row:
            csv_prompts.append(str(row["prompt"]))

csv_counts = Counter(csv_prompts)
# Each prompt should appear 6 times (for 6 models); duplicates are (count-1)//6
csv_duplicates = {p: (c // 6 - 1) for p, c in csv_counts.items() if c > 6}

# Calculate intersections and differences
excel_set = set(excel_prompts)
csv_set = set(csv_prompts)
in_both = excel_set & csv_set
in_excel_only = excel_set - csv_set
in_csv_only = csv_set - excel_set

print(f"Prompts in both: {len(in_both)}")
print(f"Prompts only in Excel: {len(in_excel_only)}")
print(f"Prompts only in CSV: {len(in_csv_only)}")
print(f"Total in Excel: {len(excel_set)}")
print(f"Total in CSV: {len(csv_set)}")
print(f"\nExcel duplicate prompts (count > 1): {len(excel_duplicates)}")
if excel_duplicates:
    print("Sample Excel duplicates:")
    for p, c in list(excel_duplicates.items())[:5]:
        print(f'  "{p[:60]}...": {c} times')
print(f"\nCSV duplicate prompts (count > 6): {len(csv_duplicates)}")
if csv_duplicates:
    print("Sample CSV duplicates (beyond 6 per prompt):")
    for p, c in list(csv_duplicates.items())[:5]:
        print(f'  "{p[:60]}...": {c} extra sets of 6')
