"""
Analysis script to calculate EV recommendation rates based on starting vehicle type.
This compares EV recommendations when starting with an ICV vs starting with an EV.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json

# Load the data
print("Loading data...")
unique_prompts = pd.read_csv('data/unique_prompts.csv')
classified_results = pd.read_csv('data/vehicle_prompt_results_joined_by_response.csv')

# Check the structure
print(f"\nClassified results shape: {classified_results.shape}")
print(f"Classified results columns: {classified_results.columns.tolist()}")
print(f"\nFirst row sample:")
print(classified_results.head(1))

# Create a mapping of prompt to starting vehicle
prompt_to_vehicle = dict(zip(unique_prompts['Prompt'], unique_prompts['starting_vehicle']))

# Extract starting vehicle from the original_prompt
def extract_starting_vehicle(prompt):
    """Extract starting vehicle from prompt text"""
    if pd.isna(prompt):
        return None
    prompt_lower = prompt.lower()
    if 'nissan leaf' in prompt_lower:
        return 'Nissan_Leaf'
    elif 'honda civic' in prompt_lower:
        return 'Honda Civic'
    elif 'toyota prius' in prompt_lower:
        return 'Toyota Prius'
    elif '4runner' in prompt_lower:
        return '4Runner'
    elif 'chevy colorado' in prompt_lower or 'chevrolet colorado' in prompt_lower:
        return 'Chevy Colorado'
    else:
        # Try to match from the mapping
        return prompt_to_vehicle.get(prompt, None)

classified_results['starting_vehicle'] = classified_results['original_prompt'].apply(extract_starting_vehicle)

# Categorize starting vehicles
# EV: Nissan Leaf
# PHEV: Toyota Prius (plug-in hybrid)
# ICV: Honda Civic, 4Runner, Chevy Colorado

ev_starting_vehicles = ['Nissan_Leaf']
phev_starting_vehicles = ['Toyota Prius']  
icv_starting_vehicles = ['Honda Civic', '4Runner', 'Chevy Colorado']

# Parse the JSON classification column
print("\nParsing vehicle classifications...")
# Filter out rows with NaN classification
classified_results = classified_results[classified_results['classification'].notna()].copy()
classified_results['classification_dict'] = classified_results['classification'].apply(json.loads)
classified_results['total_vehicles'] = classified_results['classification_dict'].apply(lambda x: x['counts']['EV'] + x['counts']['PHEV'] + x['counts']['HEV'] + x['counts']['ICV'])
classified_results['EV_count'] = classified_results['classification_dict'].apply(lambda x: x['counts']['EV'])
classified_results['PHEV_count'] = classified_results['classification_dict'].apply(lambda x: x['counts']['PHEV'])
classified_results['HEV_count'] = classified_results['classification_dict'].apply(lambda x: x['counts']['HEV'])
classified_results['ICV_count'] = classified_results['classification_dict'].apply(lambda x: x['counts']['ICV'])

# Calculate if any EV was recommended (EV count > 0)
classified_results['EV_recommended'] = classified_results['EV_count'] > 0
classified_results['any_vehicle_recommended'] = classified_results['total_vehicles'] > 0

# Categorize starting vehicles
def categorize_starting_vehicle(vehicle):
    if pd.isna(vehicle):
        return 'Unknown'
    elif vehicle in ev_starting_vehicles:
        return 'EV'
    elif vehicle in phev_starting_vehicles:
        return 'PHEV'
    elif vehicle in icv_starting_vehicles:
        return 'ICV'
    else:
        return 'Unknown'

classified_results['starting_vehicle_type'] = classified_results['starting_vehicle'].apply(categorize_starting_vehicle)

# Filter to only valid responses
valid_results = classified_results[classified_results['starting_vehicle_type'].isin(['EV', 'ICV'])].copy()
valid_results = valid_results[valid_results['any_vehicle_recommended'] == True].copy()

print(f"\nValid results with vehicle recommendations: {len(valid_results)}")
print(f"Starting vehicle type distribution:")
print(valid_results['starting_vehicle_type'].value_counts())

# Calculate EV recommendation rates by starting vehicle type
print("\n" + "="*70)
print("EV RECOMMENDATION ANALYSIS")
print("="*70)

for vehicle_type in ['EV', 'ICV']:
    subset = valid_results[valid_results['starting_vehicle_type'] == vehicle_type]
    ev_rec_rate = subset['EV_recommended'].mean()
    ev_rec_count = subset['EV_recommended'].sum()
    total_count = len(subset)
    
    print(f"\nStarting vehicle: {vehicle_type}")
    print(f"  Total responses with recommendations: {total_count}")
    print(f"  Responses recommending EVs: {ev_rec_count}")
    print(f"  EV recommendation rate: {ev_rec_rate:.4f} ({ev_rec_rate*100:.2f}%)")

# Calculate the ratio
ev_start = valid_results[valid_results['starting_vehicle_type'] == 'EV']
icv_start = valid_results[valid_results['starting_vehicle_type'] == 'ICV']

ev_rec_rate_from_ev = ev_start['EV_recommended'].mean()
ev_rec_rate_from_icv = icv_start['EV_recommended'].mean()

print(f"\n" + "="*70)
print("COMPARATIVE ANALYSIS")
print("="*70)
print(f"EV recommendation rate when starting with EV: {ev_rec_rate_from_ev:.4f} ({ev_rec_rate_from_ev*100:.2f}%)")
print(f"EV recommendation rate when starting with ICV: {ev_rec_rate_from_icv:.4f} ({ev_rec_rate_from_icv*100:.2f}%)")

if ev_rec_rate_from_icv > 0:
    ratio = ev_rec_rate_from_ev / ev_rec_rate_from_icv
    print(f"\nRATIO: LLMs were {ratio:.2f}x MORE likely to recommend an EV when starting with EV vs ICV")
    print(f"       Equivalently: {ratio:.2f}x LESS likely when starting with ICV vs EV")
    print(f"       Or: Rate reduced to {1/ratio:.2%} when starting with ICV (a {(1-1/ratio):.1%} reduction)")
else:
    print("\nCannot calculate ratio (division by zero)")

# Perform t-test
print(f"\n" + "="*70)
print("STATISTICAL TESTING")
print("="*70)

# Independent samples t-test
ev_from_ev = ev_start['EV_recommended'].astype(int)
ev_from_icv = icv_start['EV_recommended'].astype(int)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(ev_from_ev, ev_from_icv)
print(f"\nIndependent samples t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt((ev_from_ev.std()**2 + ev_from_icv.std()**2) / 2)
cohens_d = (ev_from_ev.mean() - ev_from_icv.mean()) / pooled_std if pooled_std > 0 else 0
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

# Chi-square test (more appropriate for proportions)
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(
    valid_results['starting_vehicle_type'], 
    valid_results['EV_recommended']
)
print(f"\n{contingency_table}")

chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test:")
print(f"  χ² statistic: {chi2:.4f}")
print(f"  p-value: {p_chi2:.6f}")
print(f"  Degrees of freedom: {dof}")
print(f"  Significant at α=0.05: {'Yes' if p_chi2 < 0.05 else 'No'}")

# Calculate 95% confidence intervals for proportions
from scipy.stats import norm

def proportion_ci(successes, n, confidence=0.95):
    """Calculate Wilson score confidence interval for proportion"""
    p = successes / n
    z = norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
    return center - margin, center + margin

print(f"\n95% Confidence Intervals:")
for vehicle_type in ['EV', 'ICV']:
    subset = valid_results[valid_results['starting_vehicle_type'] == vehicle_type]
    successes = subset['EV_recommended'].sum()
    n = len(subset)
    ci_low, ci_high = proportion_ci(successes, n)
    print(f"  {vehicle_type}: [{ci_low:.4f}, {ci_high:.4f}] or [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

# Summary for the paper
print(f"\n" + "="*70)
print("SUMMARY FOR PAPER")
print("="*70)

if ev_rec_rate_from_icv > 0:
    inverse_ratio = 1/ratio
    # The correct interpretation: comparing ICV to EV as baseline
    # If ratio = 3.25 (EV/ICV), then ICV recommendation rate is 1/3.25 = 0.31 of the EV rate
    # But for "X times less likely", we want to say: starting with ICV gives X times lower rate
    # So we compare: how much more likely from EV vs from ICV
    times_more_from_ev = ratio
    times_less_from_icv = ratio  # This is the same number, just different framing
    
    print(f"\n--- VERSION 1: Using 'times more likely' framing ---")
    print(f"We found that LLMs were {times_more_from_ev:.2f} times MORE likely to recommend an EV")
    print(f"when the starting vehicle was an EV compared to when the starting vehicle")
    print(f"was an ICV (t({len(ev_from_ev) + len(ev_from_icv) - 2}) = {t_stat:.2f}, p < 0.001).")
    
    print(f"\n--- VERSION 2: Using 'times less likely' framing ---")
    print(f"We found that LLMs were {times_less_from_icv:.2f} times less likely to recommend an EV")
    print(f"when the starting vehicle was an ICV compared to when the starting vehicle")
    print(f"was an EV (t({len(ev_from_ev) + len(ev_from_icv) - 2}) = {t_stat:.2f}, p < 0.001).")
    
    print(f"\n--- VERSION 3: Using percentage reduction ---")
    percentage_reduction = (1 - inverse_ratio) * 100
    print(f"We found that LLMs were {percentage_reduction:.1f}% less likely to recommend an EV")
    print(f"when the starting vehicle was an ICV compared to when the starting vehicle")
    print(f"was an EV (t({len(ev_from_ev) + len(ev_from_icv) - 2}) = {t_stat:.2f}, p < 0.001).")
    
    print(f"\n--- ALTERNATIVE: Chi-square test ---")
    print(f"We found that LLMs were {times_less_from_icv:.2f} times less likely to recommend an EV")
    print(f"when the starting vehicle was an ICV compared to when the starting vehicle")
    print(f"was an EV (χ²({dof}) = {chi2:.2f}, p < 0.001).")
    
    print(f"\n--- EXACT RATES FOR REFERENCE ---")
    print(f"EV starting vehicle → EV recommendation rate: {ev_rec_rate_from_ev*100:.1f}%")
    print(f"ICV starting vehicle → EV recommendation rate: {ev_rec_rate_from_icv*100:.1f}%")
    print(f"Ratio (EV/ICV): {ratio:.2f}")

