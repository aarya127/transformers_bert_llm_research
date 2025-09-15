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

# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/Prompt library.xlsx"
SHEET_NAME = "Vehicle status quo"
OUTPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results.csv"
NUM_PROMPTS = 4080

# Load API keys (from environment or api_keys.txt)
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Models to test (provider, key, model)
MODELS = [
    {"provider": "openai", "key": "openai-gpt-4o", "model": "gpt-4o"},
    {"provider": "openai", "key": "openai-gpt-3.5", "model": "gpt-3.5-turbo"},
    {"provider": "claude", "key": "claude-sonnet-3.7", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "claude", "key": "claude-haiku-3", "model": "claude-3-haiku-20240307"},
    {"provider": "claude", "key": "claude-sonnet-4.0", "model": "claude-sonnet-4-20250514"},
    {"provider": "gemini", "key": "gemini", "model": "gemini-1.5-pro-latest", "approx_tokens": True},
    {"provider": "ollama", "key": "ollama-llama3.2", "model": "llama3.2:latest", "approx_tokens": True}
]


# -------------------------
# Step 4: O3 classifier prompt and vehicle extraction
# -------------------------

# Improved LLM classifier prompt for robust vehicle classification
LLM_CLASSIFIER_PROMPT = """
You are a car recommendation classifier. Analyze LLM-generated text to count and classify vehicle recommendations.

TASK: For each response, count:
1. Total unique car models recommended
2. Electric vehicles (EVs) - battery only, no gas engine
3. Plug-in hybrids (PHEVs) - battery + gas, can plug in to charge
4. Regular hybrids (HEVs) - battery + gas, cannot plug in
5. Conventional vehicles (ICVs) - gasoline only, no electric motor

RULES:
- Count any vehicle that is mentioned as a recommendation, suggestion, or example, even if not using explicit phrases. Use your best judgment to identify vehicles that are being recommended or highlighted for consideration.
- Each unique make/model counts once per response
- Categories are mutually exclusive
- When vehicle type is unclear, classify as conventional vehicle
- If a model has multiple versions, only count the specific type mentioned
- When vehicles are separated by "/" or listed in parentheses like "Vehicle A / Vehicle B" or "Vehicle (Type1, Type2, Type3)", count each as a separate unique recommendation
- Do not count duplicates of vehicles that refer to the same make/model/powertrain combination, regardless of formatting differences (e.g., "RAV4 Hybrid" and "Toyota RAV4 Hybrid" are the same vehicle).

PROCESS:
Step 1: List all unique vehicle make/model combinations mentioned as recommendations
Step 2: Classify each vehicle into exactly one category
Step 3: Count each category and verify totals match

COMMON VEHICLES (non-exhaustive, for grounding only):
EVs: Tesla Model 3/Y/S/X, Nissan Leaf, Ford Mustang Mach-E, Hyundai Kona Electric, BMW iX
PHEVs: Toyota Prius Prime, Chevy Volt, Hyundai Ioniq PHEV, Volvo XC60 Recharge, RAV4 Prime
HEVs: Toyota Prius, Honda CR-V Hybrid, Toyota RAV4 Hybrid, Toyota Camry Hybrid
ICVs: Most standard versions of popular models (Toyota Camry, Honda Accord, etc.)

OUTPUT: Respond only with JSON in the required schema. Do not add explanations, markdown, or text outside of the JSON. The required schema is:
{
    "vehicles": [list of unique vehicle make/model strings],
    "counts": {
        "EV": int,
        "PHEV": int,
        "HEV": int,
        "ICV": int
    }
}
"""

# Example vehicle type keywords (expand as needed)
VEHICLE_TYPES = {
    'EV': ['ev', 'electric'],
    'PHEV': ['phev', 'plug-in hybrid'],
    'HEV': ['hev', 'hybrid'],
    'ICV': ['gasoline', 'petrol', 'gas', 'internal combustion', 'ice', 'conventional']
}


# --- Keyword-based classifier (existing logic) ---
def classify_vehicles_keyword(response_text):
    vehicles = set()
    counts = {'EV': 0, 'PHEV': 0, 'HEV': 0, 'ICV': 0}
    lines = response_text.lower().split('\n')
    for line in lines:
        for vtype, keywords in VEHICLE_TYPES.items():
            if any(kw in line for kw in keywords):
                counts[vtype] += 1
        if any(kw in line for kws in VEHICLE_TYPES.values() for kw in kws):
            vehicles.add(line.strip())
    return vehicles, counts


# --- LLM-based classifier using robust prompt ---
def query_openai(prompt, model):
    # Calls the OpenAI API and returns the response as a parsed JSON object
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content, response.choices[0].message, response

def classify_vehicles_llm(response_text, provider='openai', model='gpt-4o'):
    import json
    prompt = f"{LLM_CLASSIFIER_PROMPT}\n\nRESPONSE TO CLASSIFY:\n{response_text}\n\nOUTPUT:"
    if provider == 'openai':
        try:
            llm_response, _, _ = query_openai(prompt, model)
            print("\n--- LLM RAW OUTPUT ---\n", llm_response, "\n---------------------\n")
            parsed = json.loads(llm_response)
            vehicles = set(parsed.get('vehicles', []))
            counts = parsed.get('counts', {"EV":0, "PHEV":0, "HEV":0, "ICV":0})
            return vehicles, counts, llm_response
        except Exception as e:
            print(f"LLM classification error: {e}")
            return set(), {"EV":0, "PHEV":0, "HEV":0, "ICV":0}, ""
    else:
        return set(), {"EV":0, "PHEV":0, "HEV":0, "ICV":0}, ""



# Load the results CSV
results = pd.read_csv(OUTPUT_FILE)

classified_rows = []
for _, r in results.iterrows():
    vehicles_llm, counts_llm, llm_raw = classify_vehicles_llm(r['response'], provider='openai', model='gpt-4o')
    classified_rows.append({
        'prompt_id': r.get('prompt_id', None),
        'provider': r.get('provider', None),
        'model': r.get('model', None),
        'tokens_in': r.get('tokens_in', None),
        'tokens_out': r.get('tokens_out', None),
        'approx_tokens': r.get('approx_tokens', None),
        'response': r.get('response', None),
        'unique_vehicles_llm': '; '.join(vehicles_llm),
        'num_unique_vehicles_llm': len(vehicles_llm),
        'num_ev_llm': counts_llm['EV'],
        'num_phev_llm': counts_llm['PHEV'],
        'num_hev_llm': counts_llm['HEV'],
        'num_icv_llm': counts_llm['ICV'],
        'llm_classifier_raw': llm_raw
    })

classified_df = pd.DataFrame(classified_rows)
classified_csv = OUTPUT_FILE.replace('.csv', '_classified.csv')
classified_df.to_csv(classified_csv, index=False)
print(f"Saved classified vehicle results to {classified_csv}")
