# R Analysis Plots Output

This R script generates the following visualizations (exact replicas of the Python notebook plots):

## Generated Plot Files:

1. **status_quo_impact.png** (10x6)
   - Bar chart showing classification distribution by status quo reversal (N vs Y)
   - Grouped bars for proceed/backtrack/unclear classifications
   - Colors: #2ecc71 (proceed), #e74c3c (backtrack), #95a5a6 (unclear)

2. **status_quo_impact_by_model.png** (14x9)
   - 2x3 faceted plot showing status quo impact for each of the 6 models
   - Each subplot shows grouped bar chart of classifications by status quo
   - Matches Python's subplot layout exactly

3. **topic_distribution.png** (14x8)
   - Stacked horizontal bar chart of classification distribution by topic
   - Shows relative proportions of proceed/backtrack/unclear for each topic

4. **position_distribution.png** (12x6)
   - Grouped bar chart of classification distribution by position
   - X-axis labels rotated 45 degrees

5. **variations_distribution.png** (12x6)
   - Grouped bar chart of classification distribution by variations
   - X-axis labels rotated 45 degrees

6. **status_quo_percentage_heatmaps.png** (18x12)
   - 2x3 faceted heatmap grid (one per model)
   - Shows "proceed" rate by Status Quo × Percentage Value interaction
   - Color scale: red (low) → yellow (mid) → green (high)
   - Matches Python's seaborn heatmap style

7. **proceed_rate_by_percentage.png** (14x7)
   - Line plot showing "proceed" rate by percentage value
   - One line per model with markers
   - Includes legend for model comparison

8. **model_sensitivity.png** (16x6)
   - Two side-by-side horizontal bar charts:
     - Left: Standard deviation of proceed rate (orange #e67e22)
     - Right: Range of proceed rate across percentages (purple #9b59b6)

9. **proceed_rate_heatmap.png** (18x7)
   - Large heatmap: Models (rows) × Percentage Values (columns)
   - Shows "proceed" rate with red-yellow-green gradient
   - Matches Python's seaborn style with white gridlines

10. **proceed_backtrack_by_percentage.png** (14x6)
    - Two side-by-side line plots with shaded areas:
      - Left: "Proceed" rate trend (green with fill)
      - Right: "Backtrack" rate trend (red with fill)

11. **proceed_rate_by_position.png** (16x7)
    - Line plot showing "proceed" rate by percentage value
    - One line per position category

12. **model_comparison.png** (12x6)
    - Grouped bar chart of classification distribution by model
    - X-axis labels rotated 45 degrees

13. **model_classification_heatmap.png** (10x6)
    - Heatmap with text annotations
    - Models (rows) × Classifications (columns)
    - Orange gradient with percentage values displayed

14. **model_agreement_distribution.png** (10x6)
    - Bar chart showing distribution of agreement levels
    - Blue bars showing frequency of different agreement counts

15. **pairwise_agreement_heatmap.png** (12x10)
    - Symmetric matrix heatmap with text annotations
    - Shows agreement percentage between all model pairs
    - Red-yellow-green gradient with white gridlines

16. **odds_ratios_by_model.png** (10x6)
    - Horizontal bar chart of odds ratios from logistic regression
    - Blue bars with red dashed line at 1.0 (no effect)
    - Includes subtitle explaining interpretation

## Statistical Output:

The script also generates comprehensive console output including:
- Dataset dimensions and structure
- Classification frequency tables and percentages
- Status quo bias effect calculations
- Model sensitivity rankings
- Chi-square test results
- Cohen's h effect sizes
- Logistic regression summaries (overall and per-model)
- Mixed-effects model results with ICC
- Likelihood ratio tests
- Interaction effect p-values
- Final comprehensive summary with conclusions

## Color Scheme (matching Python):
- **Proceed**: #2ecc71 (green)
- **Backtrack**: #e74c3c (red)
- **Unclear**: #95a5a6 (gray)
- **Heatmaps**: RdYlGn palette (Red → Yellow → Green)
- **Sensitivity std dev**: #e67e22 (orange)
- **Sensitivity range**: #9b59b6 (purple)
- **Agreement/comparison**: #3498db (blue)

All plots saved at 300 DPI for publication quality.
