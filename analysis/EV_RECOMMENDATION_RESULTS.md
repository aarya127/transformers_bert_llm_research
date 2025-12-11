# EV Recommendation Analysis Results
## Summary for Paper

---

### Key Finding

**We found that LLMs were 3.25 times less likely to recommend an EV when the starting vehicle was an ICV compared to when the starting vehicle was an EV (t(1900) = 28.25, p < 0.001).**

---

## Different Ways to Report This Finding:

### Option 1: "Times Less Likely" (Recommended)
"We found that LLMs were **3.25 times less likely** to recommend an EV when the starting vehicle was an ICV compared to when the starting vehicle was an EV (t(1900) = 28.25, p < 0.001)."

### Option 2: "Times More Likely" (Alternative framing)
"We found that LLMs were **3.25 times more likely** to recommend an EV when the starting vehicle was an EV compared to when the starting vehicle was an ICV (t(1900) = 28.25, p < 0.001)."

### Option 3: Percentage Reduction
"We found that LLMs were **69.3% less likely** to recommend an EV when the starting vehicle was an ICV compared to when the starting vehicle was an EV (t(1900) = 28.25, p < 0.001)."

### Option 4: Using Chi-Square Test
"We found that LLMs were **3.25 times less likely** to recommend an EV when the starting vehicle was an ICV compared to when the starting vehicle was an EV (χ²(1) = 560.46, p < 0.001)."

---

## Detailed Statistics

### Sample Sizes
- **Total responses with vehicle recommendations**: 1,902
- **Starting with EV (Nissan Leaf)**: 1,033 responses
- **Starting with ICV**: 869 responses
  - Honda Civic
  - Toyota 4Runner  
  - Chevy Colorado

### EV Recommendation Rates
| Starting Vehicle Type | EV Recommendations | Total Responses | Rate | 95% CI |
|----------------------|-------------------|-----------------|------|---------|
| EV (Nissan Leaf) | 812 | 1,033 | **78.6%** | [76.0%, 81.0%] |
| ICV | 210 | 869 | **24.2%** | [21.4%, 27.1%] |

### Statistical Tests

#### Independent Samples T-Test
- **t-statistic**: 28.25
- **p-value**: < 0.001 (highly significant)
- **Degrees of freedom**: 1,900
- **Effect size (Cohen's d)**: 1.30 (very large effect)

#### Chi-Square Test
- **χ² statistic**: 560.46
- **p-value**: < 0.001 (highly significant)
- **Degrees of freedom**: 1

### Contingency Table
|  | Did NOT Recommend EV | Recommended EV | Total |
|--|---------------------|---------------|-------|
| **Starting with EV** | 221 (21.4%) | 812 (78.6%) | 1,033 |
| **Starting with ICV** | 659 (75.8%) | 210 (24.2%) | 869 |

---

## Interpretation

This analysis reveals a **very strong status quo bias** in LLM vehicle recommendations. When users start with an electric vehicle (Nissan Leaf), LLMs recommend EVs 78.6% of the time. However, when users start with internal combustion vehicles, LLMs recommend EVs only 24.2% of the time - a **3.25-fold difference**.

This difference is:
- **Statistically highly significant** (p < 0.001)
- **Practically significant** with a very large effect size (Cohen's d = 1.30)
- **Robust** across different statistical tests (t-test and chi-square both show p < 0.001)

The 95% confidence intervals show no overlap between the two groups, further confirming this is a real and substantial effect.

---

## Files
- Analysis script: `/analysis/ev_recommendation_analysis.py`
- Data source: `/data/vehicle_prompt_results_joined_by_response.csv`
- Prompt library: `/data/unique_prompts.csv`

Generated: December 5, 2025
