import os
import sys
from pathlib import Path
try:
    import pandas as pd
    import logging
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
NUM_PROMPTS = 4080

# Data files relative to repo root
EXCEL_FILE = ROOT / "data" / "Prompt library.xlsx"
OUTPUT_FILE = ROOT / "data" / "vehicle_prompt_results.csv"

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



# --- LLM-based classifier using robust prompt (GPT-4o only) ---
import json
def classify_vehicles_llm_o3(response_text):
    prompt = f"{LLM_CLASSIFIER_PROMPT}\n\nRESPONSE TO CLASSIFY:\n{response_text}\n\nOUTPUT:"
    try:
        response = openai_client.chat.completions.create(
            model="o3",
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
        print(f"LLM classification error: {e}")
        return set(), {"EV":0, "PHEV":0, "HEV":0, "ICV":0}, ""




# Load the results CSV (responses to classify)
results = pd.read_csv(OUTPUT_FILE)

classified_rows = []
for idx, r in results.iterrows():
    prompt_id = r.get('prompt_id', None)
    try:
        logger.info(f"Classifying row {idx+1}/{len(results)} (prompt_id={prompt_id})")
        vehicles_llm, counts_llm, llm_raw = classify_vehicles_llm_o3(r.get('response', ''))
        vehicles_list = list(vehicles_llm)
        counts_list = [counts_llm.get('EV', 0), counts_llm.get('PHEV', 0), counts_llm.get('HEV', 0), counts_llm.get('ICV', 0)]

        # Print high-level classification summary as list-of-lists to terminal
        summary = [prompt_id, vehicles_list, counts_list]
        print(summary)
        logger.info(f"Classified prompt_id={prompt_id}: EV={counts_list[0]} PHEV={counts_list[1]} HEV={counts_list[2]} ICV={counts_list[3]}")

        classified_rows.append({
            'prompt_id': prompt_id,
            'provider': r.get('provider', None),
            'model': r.get('model', None),
            'tokens_in': r.get('tokens_in', None),
            'tokens_out': r.get('tokens_out', None),
            'approx_tokens': r.get('approx_tokens', None),
            'response': r.get('response', None),
            'unique_vehicles_llm': '; '.join(vehicles_list),
            'num_unique_vehicles_llm': len(vehicles_list),
            'num_ev_llm': counts_list[0],
            'num_phev_llm': counts_list[1],
            'num_hev_llm': counts_list[2],
            'num_icv_llm': counts_list[3],
            'llm_classifier_raw': llm_raw
        })
    except Exception as e:
        logger.error(f"Error classifying row idx={idx} prompt_id={prompt_id}: {e}")
        # append a row with empty/default classification so output lengths match
        classified_rows.append({
            'prompt_id': prompt_id,
            'provider': r.get('provider', None),
            'model': r.get('model', None),
            'tokens_in': r.get('tokens_in', None),
            'tokens_out': r.get('tokens_out', None),
            'approx_tokens': r.get('approx_tokens', None),
            'response': r.get('response', None),
            'unique_vehicles_llm': '',
            'num_unique_vehicles_llm': 0,
            'num_ev_llm': 0,
            'num_phev_llm': 0,
            'num_hev_llm': 0,
            'num_icv_llm': 0,
            'llm_classifier_raw': ''
        })

classified_df = pd.DataFrame(classified_rows)

CSV_OUT = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_classified.csv"

# Save to CSV, append if file exists (keep header only if creating new file)
if os.path.exists(CSV_OUT):
    try:
        classified_df.to_csv(CSV_OUT, mode='a', header=False, index=False)
        logger.info(f"Appended {len(classified_df)} rows to {CSV_OUT}")
    except Exception as e:
        logger.error(f"Failed appending to CSV {CSV_OUT}: {e}")
else:
    try:
        classified_df.to_csv(CSV_OUT, mode='w', header=True, index=False)
        logger.info(f"Wrote {len(classified_df)} rows to new CSV {CSV_OUT}")
    except Exception as e:
        logger.error(f"Failed writing CSV {CSV_OUT}: {e}")

print(f"Saved classified vehicle results to {CSV_OUT}")
