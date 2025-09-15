import pandas as pd
import csv
import os
import concurrent.futures
import time
from openai import OpenAI
import anthropic
from google import genai
import logging
import functools
import random

# CONFIG
UNIQUE_PROMPTS_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/unique_prompts_not_in_results.csv"
ALL_UNIQUE_PROMPTS_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/unique_prompts.csv"
RESULTS_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_deduped.csv"
OUTPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_deduped.csv"  # append to this
BATCH_SIZE = 25
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 16))

from load_api_keys import load_keys
# Load API keys (from environment or api_keys.txt)
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

MODELS = [
    {"provider": "openai", "key": "openai-gpt-4o", "model": "gpt-4o"},
    {"provider": "openai", "key": "openai-gpt-3.5", "model": "gpt-3.5-turbo"},
    {"provider": "claude", "key": "claude-sonnet-3.7", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "claude", "key": "claude-haiku-3", "model": "claude-3-haiku-20240307"},
    {"provider": "claude", "key": "claude-sonnet-4.0", "model": "claude-sonnet-4-20250514"},
    {"provider": "gemini", "key": "gemini", "model": "gemini-1.5-pro-latest", "approx_tokens": True},
]

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/fill_missing_prompt_results.log'),
        logging.StreamHandler()
    ]
)

# Helper: chunk a list

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Retry decorator with exponential backoff (from prompt_results.py)
def retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    logging.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__} due to: {e}")
                    time.sleep(delay + random.uniform(0, 1))
                    delay = min(delay * 2, max_delay)
            logging.error(f"Max retries exceeded for {func.__name__}")
            return None
        return wrapper
    return decorator

# API Call Wrappers (minimal, no retry for brevity)
@retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,))
def query_openai(prompt, model):
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

@retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,))
def query_claude(prompt, model):
    resp = claude_client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text

@retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,))
def query_gemini(prompt, model):
    resp = gemini_client.models.generate_content(model=model, contents=prompt)
    return resp.text

@retry_with_backoff(max_retries=5, base_delay=2, max_delay=60, allowed_exceptions=(Exception,))
def query_ollama(prompt, model):
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get('response', '')

def run_prompt_model_pair(args):
    idx, prompt, provider, model, key = args
    prompt_num = idx + 2200  # Start labeling from 2200
    try:
        if provider == "openai":
            response = query_openai(prompt, model)
        elif provider == "claude":
            response = query_claude(prompt, model)
        elif provider == "gemini":
            response = query_gemini(prompt, model)
        elif provider == "ollama":
            response = query_ollama(prompt, model)
        logging.info(f"Received response for prompt {prompt_num} from {provider} ({model})")
        return {
            "prompt_id": prompt_num,
            "provider": provider,
            "key": key,
            "model": model,
            "prompt": prompt,
            "response": response
        }
    except Exception as e:
        logging.error(f"Error with {provider} on prompt {prompt_num}: {e}")
        return None

def main():
    # Load prompts to process for all models except llama
    unique_df = pd.read_csv(UNIQUE_PROMPTS_FILE)
    prompts = unique_df["Prompt"].dropna().astype(str).tolist()

    # Load existing results
    existing = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row.get("Prompt") or row.get("prompt"), row["provider"], row["model"]))

    # Prepare all (prompt, model) pairs for non-llama
    all_args = []
    for idx, prompt in enumerate(prompts):
        for m in MODELS:
            key = (prompt, m['provider'], m['model'])
            if key not in existing:
                all_args.append((idx, prompt, m['provider'], m['model'], m['key']))

    # Batch process for non-llama
    for batch_num, batch in enumerate(chunk_list(all_args, BATCH_SIZE), start=1):
        logging.info(f"Processing batch {batch_num} with {len(batch)} items...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for result in executor.map(run_prompt_model_pair, batch):
                if result:
                    # Append to CSV
                    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=result.keys())
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(result)
        time.sleep(1)

    # Special case for llama: process all prompts in unique_prompts.csv that it hasn't seen
    llama_model = "llama3.2:latest"
    llama_provider = "ollama"
    llama_key = "ollama-llama3.2"
    all_unique_df = pd.read_csv(ALL_UNIQUE_PROMPTS_FILE)
    all_prompts = all_unique_df["Prompt"].dropna().astype(str).tolist()
    llama_existing = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("Prompt") or row.get("prompt")) and row["provider"] == llama_provider:
                    llama_existing.add((row.get("Prompt") or row.get("prompt")))
    llama_args = []
    for idx, prompt in enumerate(all_prompts):
        if prompt not in llama_existing:
            llama_args.append((idx + 25, prompt, llama_provider, llama_model, llama_key))  # Start at 25
    # Run llama jobs sequentially
    for batch_num, batch in enumerate(chunk_list(llama_args, BATCH_SIZE), start=1):
        logging.info(f"Processing Ollama batch {batch_num} with {len(batch)} items...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for result in executor.map(run_prompt_model_pair, batch):
                if result:
                    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=result.keys())
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(result)
        time.sleep(1)

if __name__ == "__main__":
    main()
