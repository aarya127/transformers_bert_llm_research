import os
import sys
from pathlib import Path
import time
try:
    import pandas as pd
    import logging
    import json
    import concurrent.futures
    from collections import defaultdict
    from openai import OpenAI
except ImportError as e:
    missing = str(e).split()[-1].strip("'\n")
    print(f"Missing dependency: {missing}.\n\nPlease create and activate the project's venv, then install requirements:\n\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n\nThen re-run this script.")
    raise

# Ensure repo root is on sys.path so sibling modules (like load_api_keys) import correctly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_api_keys import load_keys

# -------------------------
# CONFIG (use repo-relative paths)
# -------------------------
SHEET_NAME = "Vehicle status quo"

# Data files relative to repo root
CSV_FILE = ROOT / "data" / "vehicle_prompt_results_deduped.csv"
CSV_OUT = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_classified.csv"

# Load API keys (from environment or api_keys.txt)
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# -------------------------
# Step 4: GPT-4o classifier prompt and vehicle extraction
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

# --- LLM-based classifier using robust prompt (GPT-4o only) ---
def classify_vehicles_llm_gpt4o(response_text):
    prompt = f"{LLM_CLASSIFIER_PROMPT}\n\nRESPONSE TO CLASSIFY:\n{response_text}\n\nOUTPUT:"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        llm_response = response.choices[0].message.content
        parsed = json.loads(llm_response)
        vehicles = set(parsed.get('vehicles', []))
        counts = parsed.get('counts', {"EV":0, "PHEV":0, "HEV":0, "ICV":0})
        return vehicles, counts, llm_response
    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        return set(), {"EV":0, "PHEV":0, "HEV":0, "ICV":0}, ""

def classify_row(row):
    prompt_id = row.get('prompt_id', None)
    response_text = row.get('response', '')
    try:
        vehicles_llm, counts_llm, llm_raw = classify_vehicles_llm_gpt4o(response_text)
        vehicles_list = list(vehicles_llm)
        counts_list = [counts_llm.get('EV', 0), counts_llm.get('PHEV', 0), counts_llm.get('HEV', 0), counts_llm.get('ICV', 0)]

        # Print high-level classification summary as list-of-lists to terminal
        summary = [prompt_id, vehicles_list, counts_list]
        print(summary)
        logger.info(f"Classified prompt_id={prompt_id}: EV={counts_list[0]} PHEV={counts_list[1]} HEV={counts_list[2]} ICV={counts_list[3]}")

        return {
            'prompt_id': prompt_id,
            'provider': row.get('provider', None),
            'model': row.get('model', None),
            'tokens_in': row.get('tokens_in', None),
            'tokens_out': row.get('tokens_out', None),
            'approx_tokens': row.get('approx_tokens', None),
            'response': response_text,
            'unique_vehicles_llm': '; '.join(vehicles_list),
            'num_unique_vehicles_llm': len(vehicles_list),
            'num_ev_llm': counts_list[0],
            'num_phev_llm': counts_list[1],
            'num_hev_llm': counts_list[2],
            'num_icv_llm': counts_list[3],
            'llm_classifier_raw': llm_raw
        }
    except Exception as e:
        logger.error(f"Error classifying row prompt_id={prompt_id}: {e}")
        return {
            'prompt_id': prompt_id,
            'provider': row.get('provider', None),
            'model': row.get('model', None),
            'tokens_in': row.get('tokens_in', None),
            'tokens_out': row.get('tokens_out', None),
            'approx_tokens': row.get('approx_tokens', None),
            'response': response_text,
            'unique_vehicles_llm': '',
            'num_unique_vehicles_llm': 0,
            'num_ev_llm': 0,
            'num_phev_llm': 0,
            'num_hev_llm': 0,
            'num_icv_llm': 0,
            'llm_classifier_raw': ''
        }




# Load the results from CSV (responses to classify)
results = pd.read_csv(CSV_FILE)
logger.info(f"Loaded {len(results)} rows from CSV. Columns: {list(results.columns)}")
logger.info(f"First row sample: {results.iloc[0].to_dict() if len(results) > 0 else 'No rows'}")

# Check for existing classified results to resume
existing_prompt_ids = set()
prompt_instance_count = defaultdict(int)
if os.path.exists(CSV_OUT):
    try:
        existing_df = pd.read_csv(CSV_OUT)
        existing_prompt_ids = set(existing_df['prompt_id'].dropna())
        # Count existing instances per prompt_id
        for pid in existing_df['prompt_id'].dropna():
            prompt_instance_count[pid] += 1
        logger.info(f"Found {len(existing_prompt_ids)} already classified prompt_ids, will skip them")
    except Exception as e:
        logger.warning(f"Could not read existing CSV for resume: {e}")

# Filter to unclassified rows
results = results[~results['prompt_id'].isin(existing_prompt_ids)]
logger.info(f"Remaining rows to classify: {len(results)}")

from collections import defaultdict

# ... existing code ...

classified_rows = []
processed_count = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(classify_row, row) for _, row in results.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        classified_rows.append(result)
        processed_count += 1
        prompt_id = result['prompt_id']
        prompt_instance_count[prompt_id] += 1
        
        # Log progress with instance
        logger.info(f"Processed {processed_count}/{len(results)} rows (prompt {prompt_id}, instance {prompt_instance_count[prompt_id]})")
        
        # Save this result immediately to CSV
        single_df = pd.DataFrame([result])
        if os.path.exists(CSV_OUT):
            single_df.to_csv(CSV_OUT, mode='a', header=False, index=False)
        else:
            single_df.to_csv(CSV_OUT, mode='w', header=True, index=False)
        
        time.sleep(0.1)  # Small delay to avoid rate limits

logger.info(f"Classification completed for {len(classified_rows)} rows")

print(f"Saved classified vehicle results continuously to {CSV_OUT}")
