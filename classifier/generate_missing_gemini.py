import pandas as pd
import json
import time
import os
from google import genai
from openai import OpenAI
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))
from load_api_keys import load_keys

# Load API keys
load_keys()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Classifier prompt
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
        return {'vehicles': parsed.get('vehicles', []), 'counts': parsed.get('counts', {"EV":0, "PHEV":0, "HEV":0, "ICV":0}), 'raw_json': llm_response}
    except Exception as e:
        print(f"Classification error: {e}")
        return {'vehicles': [], 'counts': {"EV":0, "PHEV":0, "HEV":0, "ICV":0}, 'raw_json': ""}

# Load missing prompts
missing_df = pd.read_csv('data/prompts_not_classified.csv')
prompts = missing_df['Prompt'].tolist()

print(f"Processing {len(prompts)} prompts for Gemini...")

results = []
model = "gemini-1.5-pro-latest"

for i, prompt in enumerate(prompts, 1):
    print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:60]}...")
    
    try:
        resp = gemini_client.models.generate_content(model=model, contents=prompt)
        
        if resp.text:
            response_text = resp.text
            print(f"  ✓ Got response ({len(response_text)} chars)")
            
            # Classify
            classification = classify_vehicles_llm_gpt4o(response_text)
            
            results.append({
                'original_prompt': prompt,
                'response': response_text,
                'classification': json.dumps(classification['counts']),
                'llm_classifier_raw_json': classification['raw_json'],
                'model': model
            })
            print(f"  ✓ Classified: {classification['counts']}")
        else:
            print(f"  ✗ No response text")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Rate limiting
    if i < len(prompts):
        time.sleep(2)

# Save results
if results:
    output_df = pd.DataFrame(results)
    output_df.to_csv('data/vehicle_prompt_results_missing_gemini.csv', index=False)
    print(f"\n✓ Saved {len(results)} classified responses to vehicle_prompt_results_missing_gemini.csv")
else:
    print("\n✗ No results to save")
