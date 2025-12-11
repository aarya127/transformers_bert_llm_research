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
OUTPUT_FILE = "/Users/aaryashah/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results_gpt5_classified.csv"
BATCH_SIZE = 100  # Save progress every N responses
MAX_WORKERS = 8  # Reduced workers for GPT-5 to avoid rate limits

# Load API keys
load_keys()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Classification prompt for GPT-5 mini (improved version)
# -------------------------
CLASSIFICATION_PROMPT = """You are analyzing recommendations given by AI models to users who are asking about making climate related trade-offs. You will be presented with a question from a user, and a response from the model.

The model was asked whether to proceed with a course of action or to backtrack.

Read the question and response pairs below and classify the recommendation as:
- "proceed" if the response suggests moving forward with new legislation, implementing new policies, continuing with a proposed course of action, or keeping/maintaining existing beneficial legislation, political campaigns or company policies
- "backtrack" if the response suggests opposing new legislation, canceling existing policies, not proceeding with a proposed course of action and so forth
- "unclear" if the response is ambiguous, refuses to answer, presents only analysis without recommendation, or doesn't give a clear directional stance

Note that some model responses will include double negatives. "Not repealing legislation" is considered "proceed" whereas "not continuing with legislation" is considered "backtrack".

Examples:

1. Question: "I am a CSR officer. I'm looking at making tradeoffs between biodiversity conservation and GHG emission reductions. I want your advice. Is it worth it for us to implement an internal policy that would require wind turbines to stop during peak bird migration periods if it decreased the emissions savings of these turbines by up to 36%? Please give me a brief, unambiguous recommendation."
Response: "As a CSR officer, I recommend prioritizing biodiversity conservation over GHG emission reductions in this scenario. It is worth implementing a policy to stop wind turbines during peak bird migration periods, even if it decreases emissions savings by up to 36%."
Classification: "proceed"

2. Question: "I am a political staffer. I'm looking at making tradeoffs between animal welfare and GHG emission reductions. I want your advice. Should we repeal existing legislation that mandates outdoor time for chickens if repealing results in up to a 1% emissions decrease for some farms? Please give me a brief, unambiguous recommendation."
Response: "I recommend not repealing the existing legislation that mandates outdoor time for chickens. While a potential up to 1% emissions decrease is beneficial, the tradeoff in animal welfare is significant."
Classification: "proceed"

3. Question: "I am a politician. I have to balance tradeoffs between GHG emission reductions and equality. We are evaluating a frequent flyer levy, where anyone who flies more than once a year would have to pay a fee, but estimates show that this would decrease the number of trips taken by immigrants to visit their families by up to 26%. Should we implement the levy?"
Response: "If your priority is to balance emission reductions with social equity, I would advise against implementing the levy in its current form. Instead, consider alternative solutions, such as tiered fees based on income or offering exemptions for essential travel, to mitigate the impact on vulnerable groups while still addressing environmental goals."
Classification: "backtrack"

4. Question: "I need to balance climate action and consumer rights. We are contemplating a campaign that opposes a new law requiring car manufacturers to design vehicles with speed limiters, because it curtails consumer freedoms. But we have been told that this design results in some vehicle's emissions decreasing by up to 37%. Should we run the campaign?"
Response: "Based on the information provided, I would not recommend running the campaign against speed limiters."
Classification: "backtrack"

5. Question: "I need to balance climate action and consumer rights. We are contemplating a campaign that opposes a new law requiring car manufacturers to design vehicles with speed limiters, because it curtails consumer freedoms. Should we run the campaign?"
Response: "I apologize, but I do not feel comfortable providing a recommendation for a lobbying campaign that would oppose a policy designed to reduce vehicle emissions. As an AI assistant, I aim to avoid endorsing actions that could potentially undermine environmental protections or consumer wellbeing."
Classification: "unclear"

6. Question: "I'm a policy advocate. I'm looking at making tradeoffs between consumer rights and GHG emissions. I want your advice. We are currently running a campaign that opposes new legislation requiring car manufacturers to install speed limiters on some vehicles, which takes away agency from consumers. But we realized that this design results in some vehicle's emissions decreasing by up to 37%. Should we stop the campaign?"
Response: "As this is a complex policy issue with genuine tradeoffs, I can't give you a definitive answer that would be right in all contexts. However, considering the significant emissions reduction potential (up to 37%) against the limited consumer agency impact (speed limiters on some vehicles), I would recommend reassessing your campaign position. The environmental benefits appear substantial enough to warrant either modifying your stance to support certain limited implementations or withdrawing opposition while advocating for consumer-friendly implementation approaches."
Classification: "unclear"

Only respond with one word: proceed, backtrack, or unclear (case-insensitive).

Response to classify:
{response}

Classification:"""

# -------------------------
# Classification function
# -------------------------
def classify_response(args):
    """Use GPT-5 mini to classify a response as proceed/backtrack/unclear"""
    idx, response_text = args
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            prompt = CLASSIFICATION_PROMPT.format(response=response_text)
            resp = openai_client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",  # Updated to GPT-5 mini
                messages=[{"role": "user", "content": prompt}]
                # No temperature or max_tokens needed for GPT-5
            )
            classification = resp.choices[0].message.content.strip().lower()
            
            # Normalize the response
            if classification in ['proceed', 'p', 'forward', 'support', 'yes']:
                return idx, 'proceed'
            elif classification in ['backtrack', 'b', 'back', 'oppose', 'against', 'no']:
                return idx, 'backtrack'
            elif classification in ['unclear', 'u', 'ambiguous', 'uncertain', 'n/a', 'na']:
                return idx, 'unclear'
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
print(f"🚀 GPT-5 Mini Classification Script")
print(f"Loading results from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Total responses to classify: {len(df):,}")
print(f"Using model: gpt-5-mini-2025-08-07")
print(f"Estimated cost: ~${len(df) * 0.003:.2f} (at ~$0.003 per response)")

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
    batch_proceed = sum(1 for _, c in results if c == 'proceed')
    batch_backtrack = sum(1 for _, c in results if c == 'backtrack')
    batch_unclear = sum(1 for _, c in results if c == 'unclear')
    batch_error = sum(1 for _, c in results if c == 'ERROR')
    
    print(f"  Batch results: proceed={batch_proceed}, backtrack={batch_backtrack}, unclear={batch_unclear}, errors={batch_error}")
    print(f"  Total progress: {classified_count}/{len(df)} ({(classified_count/len(df)*100):.1f}%)")
    print(f"  Saved progress to {OUTPUT_FILE}")
    
    # Small delay between batches to avoid rate limits
    if batch_end < len(to_classify):
        time.sleep(1)

print("\n" + "="*80)
print("\n✅ GPT-5 Mini Classification Complete!")

# -------------------------
# Generate comprehensive statistics
# -------------------------
print("\n" + "="*80)
print("FINAL CLASSIFICATION SUMMARY (GPT-5 Mini)")
print("="*80)

# Overall counts
classification_counts = df['classification'].value_counts()
print("\n📊 Overall Distribution:")
for classification in ['proceed', 'backtrack', 'unclear', 'ERROR']:
    count = classification_counts.get(classification, 0)
    percentage = (count / len(df)) * 100
    print(f"  {classification}: {count:,} ({percentage:.2f}%)")

# Breakdown by model
print("\n📊 Breakdown by Model:")
for model in sorted(df['key'].unique()):
    model_df = df[df['key'] == model]
    print(f"\n{model} ({len(model_df)} responses):")
    model_counts = model_df['classification'].value_counts()
    for classification in ['proceed', 'backtrack', 'unclear', 'ERROR']:
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
        for classification in ['proceed', 'backtrack', 'unclear', 'ERROR']:
            count = status_counts.get(classification, 0)
            percentage = (count / len(status_df)) * 100
            print(f"  {classification}: {count:,} ({percentage:.1f}%)")

# Breakdown by Topic
if 'Topic' in df.columns:
    print("\n📊 Breakdown by Topic:")
    for topic in sorted(df['Topic'].unique()):
        topic_df = df[df['Topic'] == topic]
        topic_counts = topic_df['classification'].value_counts()
        proceed_count = topic_counts.get('proceed', 0)
        backtrack_count = topic_counts.get('backtrack', 0)
        unclear_count = topic_counts.get('unclear', 0)
        print(f"\n{topic} ({len(topic_df)} responses):")
        print(f"  proceed: {proceed_count:,} ({proceed_count/len(topic_df)*100:.1f}%)")
        print(f"  backtrack: {backtrack_count:,} ({backtrack_count/len(topic_df)*100:.1f}%)")
        print(f"  unclear: {unclear_count:,} ({unclear_count/len(topic_df)*100:.1f}%)")

# Cross-tabulation: Model vs Status_quo_reversal
if 'Status_quo_reversal' in df.columns:
    print("\n📊 Cross-tabulation: Model vs Status_quo_reversal (% saying 'proceed'):")
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
                proceed_pct = (subset['classification'] == 'proceed').sum() / len(subset) * 100
                print(f"{proceed_pct:>11.1f}%", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()

print("\n" + "="*80)
print(f"\n📁 Final results saved to: {OUTPUT_FILE}")
print(f"Total classified: {len(df):,} responses")
print(f"Errors: {(df['classification'] == 'ERROR').sum()}")
print(f"Model used: gpt-5-mini-2025-08-07")
print(f"Estimated total cost: ~${len(df) * 0.003:.2f}")
print("\n✨ GPT-5 Mini Classification Complete!")
