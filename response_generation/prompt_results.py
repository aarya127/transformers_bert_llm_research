import os
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai  # Google Gemini SDK (pip install google-genai)
import csv
import random
from model_check import check_model_availability
import concurrent.futures
from load_api_keys import load_keys
import numpy as np
import time
import functools
import requests

# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/Prompt library.xlsx"
SHEET_NAME = "Vehicle status quo"
BATCH_SIZE = 25  # number of prompt/model pairs to process per batch
OUTPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"
BATCH_SIZE = 25  # number of prompt/model pairs to process per batch
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 16))  # maximize parallelism, configurable

# Load API keys (from environment or api_keys.txt)
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------
# Retry decorator with exponential backoff
# -------------------------
def retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    print(f"Retry {attempt+1}/{max_retries} for {func.__name__} due to: {e}")
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, max_delay)
            raise RuntimeError(f"Max retries exceeded for {func.__name__}")
        return wrapper
    return decorator

# -------------------------
# Initialize clients
# -------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Models to test
# -------------------------
MODELS = [
    {"provider": "openai", "key": "openai-gpt-4o", "model": "gpt-4o"},
    {"provider": "openai", "key": "openai-gpt-3.5", "model": "gpt-3.5-turbo"},
    {"provider": "claude", "key": "claude-sonnet-3.7", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "claude", "key": "claude-haiku-3", "model": "claude-3-haiku-20240307"},
    {"provider": "claude", "key": "claude-sonnet-4.0", "model": "claude-sonnet-4-20250514"},
    {"provider": "gemini", "key": "gemini", "model": "gemini-1.5-pro-latest", "approx_tokens": True},
    {"provider": "ollama", "key": "ollama-llama3.2", "model": "llama3.2:latest", "approx_tokens": True},
]

# -------------------------
# STEP 1: Load prompts
# -------------------------
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

# Filter out rows with a "starting_vehicle" (skip these)
if "starting_vehicle" in df.columns:
    df = df[df["starting_vehicle"].isna()]

print(f"Total prompts in sheet: {len(df)}")

sample_prompts = df.reset_index(drop=True)
print(f"Loaded {len(sample_prompts)} prompts for testing.")
print(f"Models to test: {[m['key'] for m in MODELS]}")
print(f"Total prompt/model pairs to send: {len(sample_prompts) * len(MODELS)}")

# -------------------------
# STEP 1.5: Load existing results
# -------------------------
existing_responses = set()
results = []

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_responses.add((int(row['prompt_id']), row['provider'], row['model']))
            row['tokens_in'] = int(row['tokens_in'])
            row['tokens_out'] = int(row['tokens_out'])
            results.append(row)
    print(f"Loaded {len(existing_responses)} existing responses from {OUTPUT_FILE}")

# -------------------------
# STEP 2: Query functions
# -------------------------
@retry_with_backoff()
def query_openai(prompt, model):
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

@retry_with_backoff()
def query_claude(prompt, model):
    resp = claude_client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens

@retry_with_backoff()
def query_gemini(prompt, model):
    resp = gemini_client.models.generate_content(model=model, contents=prompt)
    tokens_in = len(prompt.split())
    tokens_out = len(resp.text.split()) if resp.text else 0
    return resp.text, tokens_in, tokens_out

@retry_with_backoff()
def query_ollama(prompt, model):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    text = data.get('response', '')
    tokens_in = len(prompt.split())
    tokens_out = len(text.split())
    return text, tokens_in, tokens_out

# -------------------------
# STEP 2.5: Runner
# -------------------------
def run_prompt_model_pair(args):
    idx, prompt, provider, model, key, approx_tokens = args
    print(f"Sending prompt {idx} to {provider} ({model})...")
    try:
        if provider == "openai":
            response, tin, tout = query_openai(prompt, model)
        elif provider == "claude":
            response, tin, tout = query_claude(prompt, model)
        elif provider == "gemini":
            response, tin, tout = query_gemini(prompt, model)
        elif provider == "ollama":
            response, tin, tout = query_ollama(prompt, model)
        print(f"Received response for prompt {idx} from {provider} ({model})")
        return {
            "prompt_id": idx,
            "provider": provider,
            "key": key,
            "model": model,
            "prompt": prompt,
            "response": response,
            "tokens_in": tin,
            "tokens_out": tout,
            "approx_tokens": approx_tokens
        }
    except Exception as e:
        print(f"Error with {provider} on prompt {idx}: {e}")
        return None

# -------------------------
# STEP 2.6: Prepare prompt/model pairs
# -------------------------
all_args = []
for idx, row in sample_prompts.iterrows():
    prompt = row["prompt"] if "prompt" in row else row.iloc[0]
    for m in MODELS:
        key = (idx, m['provider'], m['model'])
        if key not in existing_responses:
            all_args.append((idx, prompt, m['provider'], m['model'], m['key'], m.get('approx_tokens', False)))
        else:
            print(f"Skipping prompt {idx} for {m['provider']} ({m['model']}) -- already in CSV.")

# -------------------------
# STEP 2.7: Batch processing
# -------------------------
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch_num, batch_args in enumerate(chunk_list(all_args, BATCH_SIZE), start=1):
    print(f"Processing batch {batch_num} with {len(batch_args)} items...")

# Split all_args into API and Ollama jobs
api_args = [a for a in all_args if a[2] != "ollama"]
ollama_args = [a for a in all_args if a[2] == "ollama"]

# Phase 1: Run OpenAI/Claude/Gemini in parallel
for batch_num, batch_args in enumerate(chunk_list(api_args, BATCH_SIZE), start=1):
    print(f"Processing API batch {batch_num} with {len(batch_args)} items... (max_workers={MAX_WORKERS})")
    rate_limit_hit = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(run_prompt_model_pair, batch_args):
            if result:
                results.append(result)
            if result and isinstance(result.get("response", None), str) and "rate limit" in result["response"].lower():
                rate_limit_hit = True
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved progress after API batch {batch_num} ({len(results)} total results).")
    if rate_limit_hit:
        sleep_time = random.randint(10, 60)
        print(f"Rate limit detected, sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        time.sleep(1)

# Phase 2: Run Ollama jobs sequentially (max_workers=1)
for batch_num, batch_args in enumerate(chunk_list(ollama_args, BATCH_SIZE), start=1):
    print(f"Processing Ollama batch {batch_num} with {len(batch_args)} items... (max_workers=1)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for result in executor.map(run_prompt_model_pair, batch_args):
            if result:
                results.append(result)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved progress after Ollama batch {batch_num} ({len(results)} total results).")
    time.sleep(1)

print(f"✅ Completed. Saved {len(results)} results to {OUTPUT_FILE}")

# -------------------------
# STEP 3: Analyze results and estimate costs
# -------------------------
MODEL_COSTS = {
    "openai-gpt-4o": {"input": 0.0025, "output": 0.01},
    "openai-gpt-3.5": {"input": 0.0015, "output": 0.002},
    "claude-sonnet-3.7": {"input": 0.0030, "output": 0.0150},
    "claude-haiku-3": {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4.0": {"input": 0.003, "output": 0.015},
    "gemini": {"input": 0.00125, "output": 0.005},
}

summary_rows = []
for m in MODELS:
    model_results = [r for r in results if r['key'] == m['key']]
    if not model_results:
        continue

    avg_in = np.mean([r['tokens_in'] for r in model_results])
    avg_out = np.mean([r['tokens_out'] for r in model_results])
    total_in = np.sum([r['tokens_in'] for r in model_results])
    total_out = np.sum([r['tokens_out'] for r in model_results])

    if m['key'] in MODEL_COSTS:
        cost_in_total = (total_in / 1000) * MODEL_COSTS[m['key']]['input']
        cost_out_total = (total_out / 1000) * MODEL_COSTS[m['key']]['output']
        cost_total = cost_in_total + cost_out_total
        cost_per_1000_prompts = cost_total / (len(model_results) / 1000)
    else:
        cost_in_total = cost_out_total = cost_total = cost_per_1000_prompts = 0

    summary_rows.append({
        'model': m['key'],
        'provider': m['provider'],
        'avg_tokens_in': avg_in,
        'avg_tokens_out': avg_out,
        'total_tokens_in': total_in,
        'total_tokens_out': total_out,
        'cost_per_1000_prompts_usd': cost_per_1000_prompts,
        'cost_in_total_usd': cost_in_total,
        'cost_out_total_usd': cost_out_total,
        'approx_tokens': m.get('approx_tokens', False),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_FILE.replace('.csv', '_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"📊 Saved summary stats to {summary_csv}")