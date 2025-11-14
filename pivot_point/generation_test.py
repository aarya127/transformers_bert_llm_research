import os
import sys
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai
import csv
import random
import concurrent.futures
import numpy as np
import time
import functools
import requests
from pathlib import Path

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_api_keys import load_keys

# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = "/Users/aaryashah/Downloads/expanded_counterfactuals (1).xlsx"
OUTPUT_FILE = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results_test.csv"
BATCH_SIZE = 25  # number of prompt/model pairs to process per batch
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 16))  # maximize parallelism, configurable

# TEST MODE: Only process first N prompts
TEST_NUM_PROMPTS = 3  # Change this to test more prompts

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
# Models to test (excluding llama as requested)
# -------------------------
MODELS = [
    {"provider": "openai", "key": "openai-gpt-4o", "model": "gpt-4o"},
    {"provider": "openai", "key": "openai-gpt-3.5", "model": "gpt-3.5-turbo"},
    {"provider": "claude", "key": "claude-sonnet-3.7", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "claude", "key": "claude-haiku-3", "model": "claude-3-haiku-20240307"},
    {"provider": "claude", "key": "claude-sonnet-4.0", "model": "claude-sonnet-4-20250514"},
    {"provider": "gemini", "key": "gemini", "model": "models/gemini-2.5-pro", "approx_tokens": True},
]

# -------------------------
# STEP 1: Load prompts from Excel
# -------------------------
print(f"Reading prompts from {EXCEL_FILE}...")
df = pd.read_excel(EXCEL_FILE)

# Check if 'Prompt' column exists
if 'Prompt' not in df.columns:
    raise ValueError(f"'Prompt' column not found in Excel file. Available columns: {list(df.columns)}")

# Filter out any NaN prompts
df = df[df['Prompt'].notna()].reset_index(drop=True)

# TEST MODE: Only take first few prompts
df = df.head(TEST_NUM_PROMPTS)
print(f"\n🧪 TEST MODE: Processing only first {TEST_NUM_PROMPTS} prompts")

print(f"Total prompts loaded: {len(df)}")
print(f"Models to test: {[m['key'] for m in MODELS]}")
print(f"Total prompt/model pairs to send: {len(df) * len(MODELS)}")

# Show the prompts being tested
print("\n" + "="*80)
print("PROMPTS TO BE TESTED:")
print("="*80)
for idx, row in df.iterrows():
    print(f"\nPrompt {idx}:")
    print(f"  Position: {row.get('Position', 'N/A')}")
    print(f"  Topic: {row.get('Topic', 'N/A')}")
    print(f"  Status_quo_reversal: {row.get('Status_quo_reversal', 'N/A')}")
    print(f"  Prompt text: {row['Prompt'][:150]}...")
print("="*80 + "\n")

# -------------------------
# STEP 1.5: Load existing results to avoid re-processing
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

# -------------------------
# STEP 2.5: Runner function
# -------------------------
def run_prompt_model_pair(args):
    idx, prompt, provider, model, key, approx_tokens, metadata = args
    print(f"Sending prompt {idx} to {provider} ({model})...")
    try:
        if provider == "openai":
            response, tin, tout = query_openai(prompt, model)
        elif provider == "claude":
            response, tin, tout = query_claude(prompt, model)
        elif provider == "gemini":
            response, tin, tout = query_gemini(prompt, model)
        print(f"✓ Received response for prompt {idx} from {provider} ({model})")
        
        result = {
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
        # Add metadata columns
        result.update(metadata)
        return result
    except Exception as e:
        print(f"✗ Error with {provider} on prompt {idx}: {e}")
        return None

# -------------------------
# STEP 2.6: Prepare prompt/model pairs
# -------------------------
all_args = []
for idx, row in df.iterrows():
    prompt = row["Prompt"]
    
    # Capture metadata from the Excel file
    metadata = {
        "Position": row.get("Position", ""),
        "Scenario": row.get("Scenario", ""),
        "Topic": row.get("Topic", ""),
        "Variations": row.get("Variations", ""),
        "Status_quo_reversal": row.get("Status_quo_reversal", ""),
        "percent_value": row.get("percent_value", "")
    }
    
    for m in MODELS:
        key = (idx, m['provider'], m['model'])
        if key not in existing_responses:
            all_args.append((idx, prompt, m['provider'], m['model'], m['key'], m.get('approx_tokens', False), metadata))
        else:
            print(f"Skipping prompt {idx} for {m['provider']} ({m['model']}) -- already in CSV.")

print(f"\nTotal new prompt/model pairs to process: {len(all_args)}")

# -------------------------
# STEP 2.7: Batch processing
# -------------------------
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# All models are API-based (no Ollama), so process in parallel
for batch_num, batch_args in enumerate(chunk_list(all_args, BATCH_SIZE), start=1):
    print(f"\nProcessing batch {batch_num}/{(len(all_args) + BATCH_SIZE - 1) // BATCH_SIZE} with {len(batch_args)} items... (max_workers={MAX_WORKERS})")
    rate_limit_hit = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(run_prompt_model_pair, batch_args):
            if result:
                results.append(result)
            if result and isinstance(result.get("response", None), str) and "rate limit" in result["response"].lower():
                rate_limit_hit = True
    
    # Save progress after each batch
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved progress after batch {batch_num} ({len(results)} total results).")
    
    if rate_limit_hit:
        sleep_time = random.randint(10, 60)
        print(f"Rate limit detected, sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        time.sleep(1)

print(f"\n✅ Completed. Saved {len(results)} results to {OUTPUT_FILE}")

# -------------------------
# STEP 3: Display sample results
# -------------------------
print("\n" + "="*80)
print("SAMPLE RESULTS (First 2 responses)")
print("="*80)
for i, result in enumerate(results[:2]):
    print(f"\nResult {i+1}:")
    print(f"  Prompt ID: {result['prompt_id']}")
    print(f"  Model: {result['key']}")
    print(f"  Position: {result['Position']}")
    print(f"  Topic: {result['Topic']}")
    print(f"  Status_quo_reversal: {result['Status_quo_reversal']}")
    print(f"  Prompt: {result['prompt'][:100]}...")
    print(f"  Response: {result['response'][:200]}...")
    print(f"  Tokens: {result['tokens_in']} in, {result['tokens_out']} out")
print("="*80)

# -------------------------
# STEP 4: Analyze results and estimate costs
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
        'num_prompts': len(model_results),
        'avg_tokens_in': avg_in,
        'avg_tokens_out': avg_out,
        'total_tokens_in': total_in,
        'total_tokens_out': total_out,
        'cost_per_1000_prompts_usd': cost_per_1000_prompts,
        'cost_in_total_usd': cost_in_total,
        'cost_out_total_usd': cost_out_total,
        'cost_total_usd': cost_total,
        'approx_tokens': m.get('approx_tokens', False),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_FILE.replace('.csv', '_summary.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"\n📊 Saved summary stats to {summary_csv}")

# Display summary
print("\n" + "="*80)
print("SUMMARY STATISTICS (TEST RUN)")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)
print(f"Total cost for {TEST_NUM_PROMPTS} prompts: ${summary_df['cost_total_usd'].sum():.4f}")
print(f"Total prompts processed: {len(results)}")
print(f"\n💡 Estimated cost for all 4,800 prompts: ${(summary_df['cost_total_usd'].sum() / TEST_NUM_PROMPTS * 4800):.2f}")
print("="*80)

# Show the CSV file location
print(f"\n📁 Results saved to: {OUTPUT_FILE}")
print(f"📁 Summary saved to: {summary_csv}")
print("\n✨ Review the results, then run generation.py for the full dataset!")
