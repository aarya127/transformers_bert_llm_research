import os
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_api_keys import load_keys
import json

# Load API keys
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Classification prompt (from classifier.py)
LLM_CLASSIFIER_PROMPT = """You are a car recommendation classifier. Your task is to analyze a text response that may recommend one or more vehicles and classify each recommended vehicle into one of four categories:

1. **EV** (Electric Vehicle): Battery electric vehicles with no gas engine (e.g., Tesla Model 3, Nissan Leaf, Chevy Bolt)
2. **PHEV** (Plug-in Hybrid Electric Vehicle): Vehicles with both electric battery and gas engine, can be plugged in (e.g., Toyota RAV4 Prime, Jeep Wrangler 4xe)
3. **HEV** (Hybrid Electric Vehicle): Vehicles with both electric and gas components but cannot be plugged in (e.g., Toyota Prius, Honda Accord Hybrid)
4. **ICV** (Internal Combustion Vehicle): Traditional gas or diesel vehicles with no electric components (e.g., Honda Civic, Ford F-150 gas)

**Rules:**
- Extract ALL vehicle recommendations from the text (year, make, model)
- Classify each vehicle by its powertrain type
- If a vehicle name is mentioned but no specific powertrain variant is stated, use the most common/base powertrain for that model
- If text mentions "electric" or "EV" versions without naming specific models, classify as EV
- If text mentions "hybrid" versions without naming specific models, classify as HEV (unless "plug-in" is specified → PHEV)
- If no vehicles are recommended, return empty lists

Return ONLY a JSON object with this exact structure:
{
  "vehicles": [
    {"year": "2024", "make": "Tesla", "model": "Model 3", "category": "EV"},
    {"year": "", "make": "Toyota", "model": "Prius", "category": "HEV"}
  ],
  "counts": {"EV": 1, "PHEV": 0, "HEV": 1, "ICV": 0}
}

Text to classify:
"""

# Models to use (only the 6 main models)
MODELS = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "openai", "model": "gpt-3.5-turbo"},
    {"provider": "claude", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "claude", "model": "claude-3-haiku-20240307"},
    {"provider": "claude", "model": "claude-sonnet-4-20250514"},
    {"provider": "gemini", "model": "gemini-1.5-pro-latest"},
]

INPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/prompts_not_classified.csv"
OUTPUT_FILE = "/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/data/vehicle_prompt_results_missing_classified.csv"

def get_openai_response(prompt, model_name):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return response.choices[0].message.content

def get_claude_response(prompt, model_name):
    response = claude_client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def get_gemini_response(prompt, model_name):
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": 1.0, "max_output_tokens": 2048}
    )
    return response.text

def classify_vehicle_recommendation(response_text):
    """Use GPT-4o to classify vehicle recommendations"""
    try:
        classification_prompt = LLM_CLASSIFIER_PROMPT + response_text
        result = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0,
        )
        
        raw_json = result.choices[0].message.content.strip()
        # Clean up markdown code blocks if present
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        raw_json = raw_json.strip()
        
        parsed = json.loads(raw_json)
        
        # Format classification string
        counts = parsed.get("counts", {})
        classification = f"EV:{counts.get('EV', 0)} PHEV:{counts.get('PHEV', 0)} HEV:{counts.get('HEV', 0)} ICV:{counts.get('ICV', 0)}"
        
        return {
            "classification": classification,
            "raw_json": raw_json
        }
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            "classification": "ERROR",
            "raw_json": str(e)
        }

def send_prompt(prompt_text, model_config):
    provider = model_config["provider"]
    model_name = model_config["model"]
    
    print(f"Sending to {model_name}...")
    
    try:
        if provider == "openai":
            response = get_openai_response(prompt_text, model_name)
        elif provider == "claude":
            response = get_claude_response(prompt_text, model_name)
        elif provider == "gemini":
            response = get_gemini_response(prompt_text, model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        print(f"Classifying response from {model_name}...")
        classification_result = classify_vehicle_recommendation(response)
        
        return {
            "original_prompt": prompt_text,
            "response": response,
            "classification": classification_result["classification"],
            "llm_classifier_raw_json": classification_result["raw_json"],
            "model": model_name,
        }
    except Exception as e:
        print(f"ERROR with {model_name}: {e}")
        return {
            "original_prompt": prompt_text,
            "response": f"ERROR: {e}",
            "classification": "ERROR",
            "llm_classifier_raw_json": str(e),
            "model": model_name,
        }

# Load missing prompts
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} missing prompts")
print(f"Will generate {len(df) * len(MODELS)} responses and classify them\n")

# Generate and classify responses
results = []
for idx, row in df.iterrows():
    prompt_text = row['Prompt']
    print(f"\n=== Processing prompt {idx + 1}/{len(df)} ===")
    print(f"Prompt: {prompt_text[:100]}...")
    
    for model_config in MODELS:
        result = send_prompt(prompt_text, model_config)
        results.append(result)
        time.sleep(1)  # Rate limiting

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {len(results)} classified responses to {OUTPUT_FILE}")
print(f"Expected: {len(df) * len(MODELS)} responses")
