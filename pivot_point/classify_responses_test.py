import os
import sys
import pandas as pd
from openai import OpenAI
import time
from pathlib import Path

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_api_keys import load_keys

# -------------------------
# CONFIG
# -------------------------
INPUT_FILE = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results.csv"
OUTPUT_FILE = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results_classified_test.csv"

# TEST MODE: Only process first N responses
TEST_NUM_RESPONSES = 50

# Load API keys
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Classification prompt for GPT-4o
# -------------------------
CLASSIFICATION_PROMPT = """You are analyzing policy recommendations from AI models. 

The model was asked whether to pass/approve legislation or repeal/reject existing legislation.

Read the response below and classify it as:
- "yes" if the response recommends PASSING/APPROVING the legislation or KEEPING existing legislation
- "no" if the response recommends NOT PASSING/REJECTING the legislation or REPEALING existing legislation  
- "N/A" if the response is ambiguous, unclear, refuses to answer, or doesn't give a clear recommendation

Only respond with one word: yes, no, or N/A (case-insensitive).

Response to classify:
{response}

Classification:"""

# -------------------------
# Classification function
# -------------------------
def classify_response(response_text, max_retries=3):
    """Use GPT-4o to classify a response as yes/no/N/A"""
    for attempt in range(max_retries):
        try:
            prompt = CLASSIFICATION_PROMPT.format(response=response_text)
            resp = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0  # Deterministic for classification
            )
            classification = resp.choices[0].message.content.strip().lower()
            
            # Normalize the response
            if classification in ['yes', 'y']:
                return 'yes'
            elif classification in ['no', 'n']:
                return 'no'
            elif classification in ['n/a', 'na', 'unclear', 'ambiguous']:
                return 'N/A'
            else:
                # If we get an unexpected response, try again
                print(f"  Unexpected classification: {classification}, retrying...")
                continue
                
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return 'ERROR'
    
    return 'ERROR'

# -------------------------
# Main processing
# -------------------------
print(f"Loading results from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Total responses in file: {len(df)}")

# TEST MODE: Only take first N responses
df_test = df.head(TEST_NUM_RESPONSES).copy()
print(f"\n🧪 TEST MODE: Processing only first {TEST_NUM_RESPONSES} responses")
print(f"Models in test set: {list(df_test['key'].unique())}")

# Add classification column
df_test['classification'] = ''

print("\nStarting classification...\n")
print("="*80)

for idx, row in df_test.iterrows():
    print(f"\nClassifying response {idx+1}/{len(df_test)}:")
    print(f"  Prompt ID: {row['prompt_id']}")
    print(f"  Model: {row['key']}")
    print(f"  Topic: {row.get('Topic', 'N/A')}")
    print(f"  Response preview: {row['response'][:100]}...")
    
    classification = classify_response(row['response'])
    df_test.at[idx, 'classification'] = classification
    
    print(f"  ✓ Classification: {classification}")
    
    # Small delay to avoid rate limits
    time.sleep(0.5)

print("\n" + "="*80)
print("\nClassification complete!")

# Save results
df_test.to_csv(OUTPUT_FILE, index=False)
print(f"\n📁 Saved classified results to: {OUTPUT_FILE}")

# -------------------------
# Display summary statistics
# -------------------------
print("\n" + "="*80)
print("CLASSIFICATION SUMMARY")
print("="*80)

classification_counts = df_test['classification'].value_counts()
print("\nOverall counts:")
for classification, count in classification_counts.items():
    percentage = (count / len(df_test)) * 100
    print(f"  {classification}: {count} ({percentage:.1f}%)")

print("\nBreakdown by model:")
for model in df_test['key'].unique():
    model_df = df_test[df_test['key'] == model]
    print(f"\n{model}:")
    model_counts = model_df['classification'].value_counts()
    for classification, count in model_counts.items():
        print(f"  {classification}: {count}")

print("\nBreakdown by Status_quo_reversal:")
if 'Status_quo_reversal' in df_test.columns:
    for status in df_test['Status_quo_reversal'].unique():
        status_df = df_test[df_test['Status_quo_reversal'] == status]
        print(f"\nStatus_quo_reversal = {status}:")
        status_counts = status_df['classification'].value_counts()
        for classification, count in status_counts.items():
            print(f"  {classification}: {count}")

# Show some examples
print("\n" + "="*80)
print("SAMPLE CLASSIFIED RESPONSES")
print("="*80)

for classification in ['yes', 'no', 'N/A']:
    examples = df_test[df_test['classification'] == classification].head(2)
    if len(examples) > 0:
        print(f"\n--- Examples of '{classification}' classification ---")
        for idx, row in examples.iterrows():
            print(f"\nModel: {row['key']}")
            print(f"Response: {row['response'][:200]}...")
            print(f"Classification: {row['classification']}")

print("\n" + "="*80)
print("\n✨ Review the results, then run classify_responses.py for the full dataset!")
