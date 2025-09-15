import os
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai  # Google Gemini SDK (pip install google-genai)
import csv
import random
from model_check import check_model_availability
import concurrent.futures

# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/Prompt library.xlsx"
SHEET_NAME = "Vehicle status quo"
OUTPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"

print(f"Reading prompts from {EXCEL_FILE}, sheet: {SHEET_NAME}")
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
NUM_PROMPTS = 10
print(f"Total prompts in sheet: {len(df)}")
print
# Take a random sample of NUM_PROMPTS
sample_prompts = df.sample(n=NUM_PROMPTS, random_state=42).reset_index(drop=True)
print(f"Loaded {sample_prompts} prompts for testing.")