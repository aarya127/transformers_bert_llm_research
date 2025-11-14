import os
import sys
import pandas as pd
from openai import OpenAI
import time
import concurrent.futures
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
OUTPUT_FILE = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results_classified.csv"
BATCH_SIZE = 100  # Save progress every N responses
MAX_WORKERS = 10  # Parallel processing workers

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
def classify_response(args):
    """Use GPT-4o to classify a response as yes/no/N/A"""
    idx, response_text = args
    max_retries = 3
    
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
                return idx, 'yes'
            elif classification in ['no', 'n']:
                return idx, 'no'
            elif classification in ['n/a', 'na', 'unclear', 'ambiguous']:
                return idx, 'N/A'
            else:
                # If we get an unexpected response, try again
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return idx, 'ERROR'
                
        except Exception as e:
            print(f"  Error on row {idx}, attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                return idx, 'ERROR'
    
    return idx, 'ERROR'

# -------------------------
# Main processing
# -------------------------
print(f"Loading results from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Total responses to classify: {len(df)}")

# Check if we have existing classifications to resume from
if os.path.exists(OUTPUT_FILE):
    print(f"\nFound existing output file: {OUTPUT_FILE}")
    existing_df = pd.read_csv(OUTPUT_FILE)
    if 'classification' in existing_df.columns:
        # Get indices that are already classified
        classified_indices = set(existing_df[existing_df['classification'].notna()].index)
        print(f"Already classified: {len(classified_indices)} responses")
        print(f"Remaining to classify: {len(df) - len(classified_indices)}")
        
        # Copy existing classifications
        df['classification'] = existing_df['classification'] if len(existing_df) == len(df) else ''
        start_idx = len(classified_indices)
    else:
        df['classification'] = ''
        start_idx = 0
else:
    df['classification'] = ''
    start_idx = 0

print(f"\nStarting classification from index {start_idx}...")
print(f"Using {MAX_WORKERS} parallel workers")
print(f"Saving progress every {BATCH_SIZE} responses")
print("\n" + "="*80)

# Prepare arguments for classification
to_classify = [(idx, row['response']) for idx, row in df.iterrows() 
               if idx >= start_idx and (pd.isna(df.at[idx, 'classification']) or df.at[idx, 'classification'] == '')]

total_to_classify = len(to_classify)
classified_count = start_idx

# Process in batches
for batch_start in range(0, len(to_classify), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(to_classify))
    batch = to_classify[batch_start:batch_end]
    
    batch_num = (batch_start // BATCH_SIZE) + 1
    total_batches = (len(to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} responses)...")
    
    # Classify in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(classify_response, batch))
    
    # Update dataframe with results
    for idx, classification in results:
        df.at[idx, 'classification'] = classification
        classified_count += 1
    
    # Save progress
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Calculate statistics for this batch
    batch_yes = sum(1 for _, c in results if c == 'yes')
    batch_no = sum(1 for _, c in results if c == 'no')
    batch_na = sum(1 for _, c in results if c == 'N/A')
    batch_error = sum(1 for _, c in results if c == 'ERROR')
    
    print(f"  Batch results: yes={batch_yes}, no={batch_no}, N/A={batch_na}, errors={batch_error}")
    print(f"  Total progress: {classified_count}/{len(df)} ({(classified_count/len(df)*100):.1f}%)")
    print(f"  Saved progress to {OUTPUT_FILE}")
    
    # Small delay between batches to avoid rate limits
    if batch_end < len(to_classify):
        time.sleep(1)

print("\n" + "="*80)
print("\n✅ Classification complete!")

# -------------------------
# Generate comprehensive statistics
# -------------------------
print("\n" + "="*80)
print("FINAL CLASSIFICATION SUMMARY")
print("="*80)

# Overall counts
classification_counts = df['classification'].value_counts()
print("\n📊 Overall Distribution:")
for classification in ['yes', 'no', 'N/A', 'ERROR']:
    count = classification_counts.get(classification, 0)
    percentage = (count / len(df)) * 100
    print(f"  {classification}: {count:,} ({percentage:.2f}%)")

# Breakdown by model
print("\n📊 Breakdown by Model:")
for model in sorted(df['key'].unique()):
    model_df = df[df['key'] == model]
    print(f"\n{model} ({len(model_df)} responses):")
    model_counts = model_df['classification'].value_counts()
    for classification in ['yes', 'no', 'N/A', 'ERROR']:
        count = model_counts.get(classification, 0)
        percentage = (count / len(model_df)) * 100
        print(f"  {classification}: {count:,} ({percentage:.1f}%)")

# Breakdown by Status_quo_reversal
if 'Status_quo_reversal' in df.columns:
    print("\n📊 Breakdown by Status_quo_reversal:")
    for status in sorted(df['Status_quo_reversal'].unique()):
        status_df = df[df['Status_quo_reversal'] == status]
        print(f"\nStatus_quo_reversal = {status} ({len(status_df)} responses):")
        status_counts = status_df['classification'].value_counts()
        for classification in ['yes', 'no', 'N/A', 'ERROR']:
            count = status_counts.get(classification, 0)
            percentage = (count / len(status_df)) * 100
            print(f"  {classification}: {count:,} ({percentage:.1f}%)")

# Breakdown by Topic
if 'Topic' in df.columns:
    print("\n📊 Breakdown by Topic:")
    for topic in sorted(df['Topic'].unique()):
        topic_df = df[df['Topic'] == topic]
        topic_counts = topic_df['classification'].value_counts()
        yes_count = topic_counts.get('yes', 0)
        no_count = topic_counts.get('no', 0)
        na_count = topic_counts.get('N/A', 0)
        print(f"\n{topic} ({len(topic_df)} responses):")
        print(f"  yes: {yes_count:,} ({yes_count/len(topic_df)*100:.1f}%)")
        print(f"  no: {no_count:,} ({no_count/len(topic_df)*100:.1f}%)")
        print(f"  N/A: {na_count:,} ({na_count/len(topic_df)*100:.1f}%)")

# Cross-tabulation: Model vs Status_quo_reversal
if 'Status_quo_reversal' in df.columns:
    print("\n📊 Cross-tabulation: Model vs Status_quo_reversal (% saying 'yes'):")
    print("\nModel".ljust(25), end='')
    for status in sorted(df['Status_quo_reversal'].unique()):
        print(f"{status}".rjust(12), end='')
    print()
    print("-" * 50)
    
    for model in sorted(df['key'].unique()):
        print(model.ljust(25), end='')
        for status in sorted(df['Status_quo_reversal'].unique()):
            subset = df[(df['key'] == model) & (df['Status_quo_reversal'] == status)]
            if len(subset) > 0:
                yes_pct = (subset['classification'] == 'yes').sum() / len(subset) * 100
                print(f"{yes_pct:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

print("\n" + "="*80)
print(f"\n📁 Final results saved to: {OUTPUT_FILE}")
print(f"Total classified: {len(df):,} responses")
print(f"Errors: {(df['classification'] == 'ERROR').sum()}")
print("\n✨ Classification complete!")
