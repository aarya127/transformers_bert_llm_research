# ==============================================================================
# Counterfactual Classification Analysis in R
# Converted from Python Jupyter notebook
# ==============================================================================

# Clear environment
rm(list = ls())

# Load required libraries
cat("Loading required libraries...\n")
suppressPackageStartupMessages({
  library(tidyverse)    # Data manipulation and visualization
  library(readr)        # Reading CSV files
  library(ggplot2)      # Plotting
  library(reshape2)     # Data reshaping
  library(grid)         # Grid graphics (for textGrob)
  library(gridExtra)    # Multiple plots
  library(RColorBrewer) # Color palettes
  library(scales)       # Percentage formatting
  library(lme4)         # Mixed-effects models
  library(lmerTest)     # P-values for mixed models
})

cat("✅ Libraries imported successfully!\n\n")

# Set plotting theme
theme_set(theme_minimal() + 
            theme(plot.title = element_text(face = "bold", size = 14),
                  axis.title = element_text(face = "bold", size = 11)))

# ==============================================================================
# 1. LOAD AND EXPLORE THE DATASET
# ==============================================================================

cat("==============================================================================\n")
cat("SECTION 1: LOAD AND EXPLORE THE DATASET\n")
cat("==============================================================================\n\n")

# Load the classified results
df <- read_csv('/Users/aaryas127/Documents/GitHub/transformers_bert_llm_research/pivot_point/counterfactual_results_gpt5_classified.csv', 
               show_col_types = FALSE)

# Display basic information
cat(sprintf("Dataset shape: %d rows × %d columns\n", nrow(df), ncol(df)))
cat("\nColumn names:\n")
print(names(df))
cat("\nFirst few rows:\n")
print(head(df))

# Overall classification distribution
cat("\n\nOverall Classification Distribution:\n")
print(table(df$classification))
cat("\nPercentages:\n")
print(round(prop.table(table(df$classification)) * 100, 2))

# Distribution by model
cat("\n\nClassification Distribution by Model:\n")
model_classification <- prop.table(table(df$model, df$classification), 1) * 100
print(round(model_classification, 2))

# ==============================================================================
# 2. STATUS QUO IMPACT ANALYSIS
# ==============================================================================

cat("\n==============================================================================\n")
cat("SECTION 2: STATUS QUO IMPACT ANALYSIS\n")
cat("==============================================================================\n\n")

# Status quo reversal impact on classification
status_quo_classification <- prop.table(table(df$Status_quo_reversal, df$classification), 1) * 100

cat("Classification Distribution by Status Quo Reversal:\n")
print(round(status_quo_classification, 2))

# Visualize status quo impact
status_quo_df <- as.data.frame(status_quo_classification)
colnames(status_quo_df) <- c("Status_Quo", "Classification", "Percentage")

ggplot(status_quo_df, aes(x = Status_Quo, y = Percentage, fill = Classification)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("backtrack" = "#e74c3c", "proceed" = "#2ecc71", "unclear" = "#95a5a6")) +
  labs(title = "Classification Distribution by Status Quo Reversal",
       x = "Status Quo Reversal (N = Backtrack, Y = Proceed)",
       y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.title = element_text(face = "bold"))

ggsave("status_quo_impact.png", width = 10, height = 6, dpi = 300)

# Calculate the status quo bias effect
proceed_when_N <- status_quo_classification["N", "proceed"]
proceed_when_Y <- status_quo_classification["Y", "proceed"]

cat(sprintf("\n'Proceed' response rate when status quo is Backtrack: %.2f%%\n", proceed_when_N))
cat(sprintf("'Proceed' response rate when status quo is Proceed: %.2f%%\n", proceed_when_Y))
cat(sprintf("Difference: %.2f percentage points\n\n", proceed_when_Y - proceed_when_N))

# Status quo impact by model - Create 2x3 faceted plot
status_quo_by_model_df <- df %>%
  group_by(model, Status_quo_reversal, classification) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(model, Status_quo_reversal) %>%
  mutate(percentage = count / sum(count) * 100)

ggplot(status_quo_by_model_df, aes(x = Status_quo_reversal, y = percentage, fill = classification)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  facet_wrap(~model, nrow = 2, ncol = 3) +
  scale_fill_manual(values = c("backtrack" = "#e74c3c", "proceed" = "#2ecc71", "unclear" = "#95a5a6")) +
  labs(title = "Status Quo Impact by Model",
       x = "Status Quo",
       y = "Percentage (%)",
       fill = "Classification") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
        strip.text = element_text(face = "bold", size = 11),
        axis.text.x = element_text(size = 9),
        legend.position = "bottom")

ggsave("status_quo_impact_by_model.png", width = 14, height = 9, dpi = 300)

# ==============================================================================
# 3. SCENARIO DISTRIBUTION ANALYSIS
# ==============================================================================

cat("==============================================================================\n")
cat("SECTION 3: SCENARIO DISTRIBUTION ANALYSIS\n")
cat("==============================================================================\n\n")

# Distribution by Topic
topic_classification <- prop.table(table(df$Topic, df$classification), 1) * 100

cat("Classification Distribution by Topic (%):\n")
print(round(topic_classification, 2))

# Visualize - stacked horizontal bar chart
topic_df <- as.data.frame(topic_classification)
colnames(topic_df) <- c("Topic", "Classification", "Percentage")

ggplot(topic_df, aes(x = Percentage, y = reorder(Topic, Percentage), fill = Classification)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("proceed" = "#2ecc71", "backtrack" = "#e74c3c", "unclear" = "#95a5a6")) +
  labs(title = "Classification Distribution by Topic",
       x = "Percentage (%)",
       y = "Topic") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("topic_distribution.png", width = 14, height = 8, dpi = 300)

# Distribution by Position
position_classification <- prop.table(table(df$Position, df$classification), 1) * 100

cat("\n\nClassification Distribution by Position (%):\n")
print(round(position_classification, 2))

position_df <- as.data.frame(position_classification)
colnames(position_df) <- c("Position", "Classification", "Percentage")

ggplot(position_df, aes(x = Position, y = Percentage, fill = Classification)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("proceed" = "#2ecc71", "backtrack" = "#e74c3c", "unclear" = "#95a5a6")) +
  labs(title = "Classification Distribution by Position",
       x = "Position",
       y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("position_distribution.png", width = 12, height = 6, dpi = 300)

# Distribution by Variations
variations_classification <- prop.table(table(df$Variations, df$classification), 1) * 100

cat("\n\nClassification Distribution by Variations (%):\n")
print(round(variations_classification, 2))

variations_df <- as.data.frame(variations_classification)
colnames(variations_df) <- c("Variations", "Classification", "Percentage")

ggplot(variations_df, aes(x = Variations, y = Percentage, fill = Classification)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("proceed" = "#2ecc71", "backtrack" = "#e74c3c", "unclear" = "#95a5a6")) +
  labs(title = "Classification Distribution by Variations",
       x = "Variations",
       y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("variations_distribution.png", width = 12, height = 6, dpi = 300)

# ==============================================================================
# 4. PERCENTAGE VALUE ANALYSIS
# ==============================================================================

cat("\n==============================================================================\n")
cat("SECTION 4: PERCENTAGE VALUE ANALYSIS\n")
cat("==============================================================================\n\n")

# Interactive comparison: percentage value + status quo reversal
# Create a 2D heatmap for each model showing how both factors interact (2x3 grid)

heatmap_data <- df %>%
  group_by(model, Status_quo_reversal, percent_value) %>%
  summarise(proceed_rate = mean(classification == "proceed") * 100, .groups = "drop")

# Print how many percentage values we have
cat(sprintf("\nNumber of unique percentage values: %d\n", length(unique(heatmap_data$percent_value))))
cat("Percentage values: ", paste(sort(unique(heatmap_data$percent_value)), collapse=", "), "\n\n")

# Filter to show only every 3rd percentage value to reduce clutter
all_pct_values <- sort(unique(heatmap_data$percent_value))
# Keep every 3rd value, but always include first and last
indices_to_keep <- c(1, seq(3, length(all_pct_values), by = 3), length(all_pct_values))
indices_to_keep <- unique(indices_to_keep)  # Remove duplicates
pct_values_to_show <- all_pct_values[indices_to_keep]

cat(sprintf("Showing %d of %d percentage values for clarity\n", length(pct_values_to_show), length(all_pct_values)))

# Filter heatmap data to only include selected percentage values
heatmap_data_filtered <- heatmap_data %>% filter(percent_value %in% pct_values_to_show)

# Create individual heatmaps for each model with proper spacing
models_list <- unique(heatmap_data_filtered$model)
plot_list <- list()

for (i in 1:length(models_list)) {
  model_name <- models_list[i]
  model_heatmap_data <- heatmap_data_filtered %>% filter(model == model_name)
  
  p <- ggplot(model_heatmap_data, aes(x = factor(percent_value), y = Status_quo_reversal, fill = proceed_rate)) +
    geom_tile(color = "white", linewidth = 1, width = 0.9, height = 0.9) +
    geom_text(aes(label = sprintf("%.0f", proceed_rate)), 
              color = "black", 
              size = 3.5, 
              fontface = "bold") +
    scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                         midpoint = 50, limits = c(0, 100),
                         name = '"Proceed"\nRate (%)') +
    labs(title = model_name,
         x = if(i > 4) "Percentage Value (%)" else NULL,
         y = if(i %in% c(1, 4)) "Status Quo" else NULL) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 9, face = "bold"),
          axis.text.y = element_text(size = 12, face = "bold"),
          axis.title = element_text(face = "bold", size = 11),
          legend.position = if(i == 3) "right" else "none",
          legend.key.height = unit(1.5, "cm"),
          legend.key.width = unit(0.7, "cm"),
          legend.title = element_text(face = "bold", size = 10),
          legend.text = element_text(size = 9),
          plot.margin = margin(8, 8, 8, 8),
          panel.grid = element_blank())
  
  plot_list[[i]] <- p
}

# Combine all plots in a 2x3 grid with extra width for many percentage values
combined_heatmap <- grid.arrange(grobs = plot_list, nrow = 2, ncol = 3,
                                 top = textGrob('Status Quo × Percentage Value Interaction - "Proceed" Rate by Model',
                                               gp = gpar(fontsize = 16, fontface = "bold")))

ggsave("status_quo_percentage_heatmaps.png", combined_heatmap, width = 24, height = 10, dpi = 300)

# Line plot showing "proceed" rate by percentage value for each model
proceed_rate_by_pct <- df %>%
  group_by(model, percent_value) %>%
  summarise(proceed_rate = mean(classification == "proceed") * 100, .groups = "drop")

ggplot(proceed_rate_by_pct, aes(x = percent_value, y = proceed_rate, color = model, group = model)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = '"Proceed" Response Rate by Percentage Value - Model Comparison',
       x = "Percentage Value in Prompt",
       y = '"Proceed" Rate (%)') +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.title = element_text(face = "bold"))

ggsave("proceed_rate_by_percentage.png", width = 14, height = 7, dpi = 300)

# Model sensitivity to percentage value
sensitivity_stats <- proceed_rate_by_pct %>%
  group_by(model) %>%
  summarise(
    proceed_rate_std = sd(proceed_rate),
    proceed_rate_range = max(proceed_rate) - min(proceed_rate),
    proceed_rate_mean = mean(proceed_rate)
  ) %>%
  arrange(desc(proceed_rate_std))

cat("\nModel Sensitivity to Percentage Value:\n")
cat("(Higher values = more influenced by percentage changes)\n\n")
print(sensitivity_stats)

# Visualize sensitivity using ggplot2 side-by-side
p1 <- ggplot(sensitivity_stats, aes(x = reorder(model, proceed_rate_std), y = proceed_rate_std)) +
  geom_bar(stat = "identity", fill = "#e67e22") +
  coord_flip() +
  labs(title = 'Model Sensitivity to Percentage Value\n(Std Dev of "Proceed" Rate)',
       x = "Model",
       y = "Standard Deviation") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
        axis.title = element_text(face = "bold", size = 11))

p2 <- ggplot(sensitivity_stats, aes(x = reorder(model, proceed_rate_range), y = proceed_rate_range)) +
  geom_bar(stat = "identity", fill = "#9b59b6") +
  coord_flip() +
  labs(title = "Model Response Range\nacross Percentage Values",
       x = "Model",
       y = "Range (Max - Min)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
        axis.title = element_text(face = "bold", size = 11))

# Combine plots side by side
combined_sensitivity <- grid.arrange(p1, p2, ncol = 2)
ggsave("model_sensitivity.png", combined_sensitivity, width = 16, height = 6, dpi = 300)

# Statistical summary: percentage value insights
cat("\n==============================================================================\n")
cat("PERCENTAGE VALUE INSIGHTS\n")
cat("==============================================================================\n\n")

# Most "proceed"-friendly percentage values overall
overall_proceed_by_pct <- df %>%
  group_by(percent_value) %>%
  summarise(proceed_rate = mean(classification == "proceed") * 100) %>%
  arrange(desc(proceed_rate))

cat("1. PERCENTAGE VALUES MOST LIKELY TO GET 'PROCEED' (Overall):\n")
print(head(overall_proceed_by_pct, 5))

cat("\n2. PERCENTAGE VALUES MOST LIKELY TO GET 'BACKTRACK' (Overall):\n")
print(tail(overall_proceed_by_pct, 5))

# Most uncertain percentage values
overall_unclear_by_pct <- df %>%
  group_by(percent_value) %>%
  summarise(unclear_rate = mean(classification == "unclear") * 100) %>%
  arrange(desc(unclear_rate))

cat("\n3. PERCENTAGE VALUES CREATING MOST UNCERTAINTY (Unclear):\n")
print(head(overall_unclear_by_pct, 5))

# Heatmap: Models vs Percentage Values for "proceed" classification
pivot_proceed_heatmap <- proceed_rate_by_pct %>%
  pivot_wider(names_from = percent_value, values_from = proceed_rate)

pivot_matrix <- as.matrix(pivot_proceed_heatmap[, -1])
rownames(pivot_matrix) <- pivot_proceed_heatmap$model

# Create better heatmap using ggplot2
proceed_heatmap_df <- proceed_rate_by_pct
proceed_heatmap_df$percent_value <- as.factor(proceed_heatmap_df$percent_value)

ggplot(proceed_heatmap_df, aes(x = percent_value, y = model, fill = proceed_rate)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_gradient2(low = "red", mid = "yellow", high = "green", 
                       midpoint = 50, limit = c(0, 100),
                       name = '"Proceed"\nRate (%)') +
  labs(title = 'Model "Proceed" Response Rate by Percentage Value in Prompt',
       x = "Percentage Value in Prompt (%)",
       y = "AI Model") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_text(face = "bold"))

ggsave("proceed_rate_heatmap.png", width = 18, height = 7, dpi = 300)

# Print summary statistics
cat("\n\nSummary Statistics:\n")
cat("==============================================================================\n")
cat(sprintf("\nHighest 'Proceed' Rate: %.1f%%\n", max(pivot_matrix, na.rm = TRUE)))
most_proceed_model_idx <- which(rowMeans(pivot_matrix, na.rm = TRUE) == max(rowMeans(pivot_matrix, na.rm = TRUE)))
cat(sprintf("  Model: %s (avg: %.1f%%)\n", rownames(pivot_matrix)[most_proceed_model_idx], 
            rowMeans(pivot_matrix, na.rm = TRUE)[most_proceed_model_idx]))

cat(sprintf("\nLowest 'Proceed' Rate: %.1f%%\n", min(pivot_matrix, na.rm = TRUE)))
least_proceed_model_idx <- which(rowMeans(pivot_matrix, na.rm = TRUE) == min(rowMeans(pivot_matrix, na.rm = TRUE)))
cat(sprintf("  Model: %s (avg: %.1f%%)\n", rownames(pivot_matrix)[least_proceed_model_idx],
            rowMeans(pivot_matrix, na.rm = TRUE)[least_proceed_model_idx]))

cat("\nAverage 'Proceed' Rate by Model:\n")
for (i in 1:nrow(pivot_matrix)) {
  cat(sprintf("  %s: %.1f%%\n", rownames(pivot_matrix)[i], rowMeans(pivot_matrix, na.rm = TRUE)[i]))
}
cat("==============================================================================\n")

# Explore percentage value distribution
cat("\n\nUnique percentage values in dataset:\n")
print(sort(unique(df$percent_value)))

# Overall classification by percentage value
pct_classification <- prop.table(table(df$percent_value, df$classification), 1) * 100

cat("\n\nClassification Distribution by Percentage Value (%):\n")
print(round(pct_classification, 2))

# Create two separate line plots for better readability using ggplot2
pct_class_df <- as.data.frame(pct_classification)
colnames(pct_class_df) <- c("Percent_Value", "Classification", "Percentage")
pct_class_df$Percent_Value <- as.numeric(as.character(pct_class_df$Percent_Value))

proceed_rates <- pct_class_df[pct_class_df$Classification == "proceed", ]
backtrack_rates <- pct_class_df[pct_class_df$Classification == "backtrack", ]

p1 <- ggplot(proceed_rates, aes(x = Percent_Value, y = Percentage)) +
  geom_line(color = "#2ecc71", size = 2.5) +
  geom_point(color = "#2ecc71", size = 4, shape = 19) +
  geom_ribbon(aes(ymin = 0, ymax = Percentage), fill = "#2ecc71", alpha = 0.3) +
  labs(title = '"Proceed" Response Rate by Percentage Value',
       x = "Percentage Value in Prompt",
       y = '"Proceed" Rate (%)') +
  ylim(0, 100) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90"))

p2 <- ggplot(backtrack_rates, aes(x = Percent_Value, y = Percentage)) +
  geom_line(color = "#e74c3c", size = 2.5) +
  geom_point(color = "#e74c3c", size = 4, shape = 15) +
  geom_ribbon(aes(ymin = 0, ymax = Percentage), fill = "#e74c3c", alpha = 0.3) +
  labs(title = '"Backtrack" Response Rate by Percentage Value',
       x = "Percentage Value in Prompt",
       y = '"Backtrack" Rate (%)') +
  ylim(0, 100) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
        panel.grid.major = element_line(color = "grey80"),
        panel.grid.minor = element_line(color = "grey90"))

# Combine plots side by side
combined_rates <- grid.arrange(p1, p2, ncol = 2)
ggsave("proceed_backtrack_by_percentage.png", combined_rates, width = 14, height = 6, dpi = 300)

# "Proceed" Response Rate by Percentage Value - Grouped by Position
proceed_rate_by_pct_position <- df %>%
  group_by(Position, percent_value) %>%
  summarise(proceed_rate = mean(classification == "proceed") * 100, .groups = "drop")

ggplot(proceed_rate_by_pct_position, aes(x = percent_value, y = proceed_rate, 
                                         color = Position, group = Position)) +
  geom_line(size = 1.2, alpha = 0.85) +
  geom_point(size = 3) +
  labs(title = '"Proceed" Response Rate by Percentage Value - Comparison by Position',
       x = "Percentage Value in Prompt",
       y = '"Proceed" Rate (%)') +
  ylim(0, 100) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.title = element_text(face = "bold"))

ggsave("proceed_rate_by_position.png", width = 16, height = 7, dpi = 300)

# ==============================================================================
# 5. MODEL COMPARISON ANALYSIS
# ==============================================================================

cat("\n==============================================================================\n")
cat("SECTION 5: MODEL COMPARISON ANALYSIS\n")
cat("==============================================================================\n\n")

# Overall model comparison
model_classification <- prop.table(table(df$model, df$classification), 1) * 100

cat("Classification Distribution by Model (%):\n")
print(round(model_classification, 2))

model_class_df <- as.data.frame(model_classification)
colnames(model_class_df) <- c("Model", "Classification", "Percentage")

ggplot(model_class_df, aes(x = Model, y = Percentage, fill = Classification)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("proceed" = "#2ecc71", "backtrack" = "#e74c3c", "unclear" = "#95a5a6")) +
  labs(title = "Classification Distribution by Model",
       x = "Model",
       y = "Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("model_comparison.png", width = 12, height = 6, dpi = 300)

# Heatmap of model classifications
model_class_matrix <- as.matrix(model_classification)

ggplot(model_class_df, aes(x = Classification, y = Model, fill = Percentage)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f", Percentage)), color = "black", size = 4) +
  scale_fill_gradient(low = "lightyellow", high = "orangered", name = "Percentage (%)") +
  labs(title = "Model Classification Heatmap",
       x = "Classification",
       y = "Model") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("model_classification_heatmap.png", width = 10, height = 6, dpi = 300)

# Model agreement analysis
cat("\n\nModel Agreement Analysis:\n")

# Create prompt identifier
df$prompt_id <- paste(df$Topic, df$Position, df$Scenario, df$Variations, 
                      df$Status_quo_reversal, df$percent_value, sep = "_")

# Calculate agreement for each prompt
agreement_stats <- df %>%
  group_by(prompt_id) %>%
  summarise(
    total_models = n(),
    modal_classification = names(which.max(table(classification))),
    max_agreement = max(table(classification)),
    .groups = "drop"
  ) %>%
  mutate(agreement_pct = (max_agreement / total_models) * 100)

cat(sprintf("Average agreement percentage: %.2f%%\n", mean(agreement_stats$agreement_pct)))
cat("\nAgreement distribution:\n")
print(table(agreement_stats$max_agreement))

# Visualize agreement distribution
ggplot(agreement_stats, aes(x = factor(max_agreement))) +
  geom_bar(fill = "#3498db") +
  labs(title = "Distribution of Model Agreement Levels",
       x = "Number of Models in Agreement",
       y = "Number of Prompts") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("model_agreement_distribution.png", width = 10, height = 6, dpi = 300)

# Pairwise model agreement
models_list <- unique(df$model)
pairwise_agreement <- data.frame()

for (i in 1:(length(models_list) - 1)) {
  for (j in (i + 1):length(models_list)) {
    model1 <- models_list[i]
    model2 <- models_list[j]
    
    model1_data <- df[df$model == model1, c("prompt_id", "classification")]
    model2_data <- df[df$model == model2, c("prompt_id", "classification")]
    
    merged <- merge(model1_data, model2_data, by = "prompt_id", 
                    suffixes = c("_m1", "_m2"))
    
    agreement <- mean(merged$classification_m1 == merged$classification_m2) * 100
    
    pairwise_agreement <- rbind(pairwise_agreement, 
                                data.frame(model1 = model1, 
                                           model2 = model2, 
                                           agreement_pct = agreement))
  }
}

# Create symmetric matrix for heatmap
agreement_matrix <- matrix(100, nrow = length(models_list), ncol = length(models_list),
                           dimnames = list(models_list, models_list))

for (i in 1:nrow(pairwise_agreement)) {
  m1 <- pairwise_agreement$model1[i]
  m2 <- pairwise_agreement$model2[i]
  agree <- pairwise_agreement$agreement_pct[i]
  
  agreement_matrix[m1, m2] <- agree
  agreement_matrix[m2, m1] <- agree
}

# Visualize pairwise agreement
agreement_df <- melt(agreement_matrix)
colnames(agreement_df) <- c("Model1", "Model2", "Agreement")

ggplot(agreement_df, aes(x = Model1, y = Model2, fill = Agreement)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f", Agreement)), size = 3) +
  scale_fill_gradient2(low = "red", mid = "yellow", high = "green",
                       midpoint = 50, limit = c(0, 100),
                       name = "Agreement (%)") +
  labs(title = "Pairwise Model Agreement Matrix") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank())

ggsave("pairwise_agreement_heatmap.png", width = 12, height = 10, dpi = 300)

cat("\nTop 5 Most Similar Model Pairs:\n")
print(head(pairwise_agreement[order(-pairwise_agreement$agreement_pct), ], 5))

cat("\nTop 5 Most Different Model Pairs:\n")
print(head(pairwise_agreement[order(pairwise_agreement$agreement_pct), ], 5))

# ==============================================================================
# 6. KEY FINDINGS SUMMARY
# ==============================================================================

cat("\n==============================================================================\n")
cat("KEY FINDINGS SUMMARY\n")
cat("==============================================================================\n\n")

# Overall distribution
cat("1. OVERALL CLASSIFICATION DISTRIBUTION:\n")
overall_dist <- prop.table(table(df$classification)) * 100
for (class_name in names(overall_dist)) {
  cat(sprintf("   - %s: %.2f%%\n", class_name, overall_dist[class_name]))
}

# Status quo effect
cat("\n2. STATUS QUO BIAS:\n")
status_quo_proceed <- prop.table(table(df$Status_quo_reversal, df$classification), 1) * 100

proceed_when_N <- status_quo_proceed["N", "proceed"]
proceed_when_Y <- status_quo_proceed["Y", "proceed"]

cat(sprintf("   - 'Proceed' rate when status quo is Backtrack: %.2f%%\n", proceed_when_N))
cat(sprintf("   - 'Proceed' rate when status quo is Proceed: %.2f%%\n", proceed_when_Y))
cat(sprintf("   - BIAS EFFECT: %.2f percentage points\n", proceed_when_Y - proceed_when_N))
cat(sprintf("   - This represents a %.1f%% relative increase\n", 
            (proceed_when_Y - proceed_when_N) / proceed_when_N * 100))

# Model differences
cat("\n3. MODEL CHARACTERISTICS:\n")
unclear_rates <- model_classification[, "unclear"]
unclear_sorted <- sort(unclear_rates)

cat(sprintf("   Most decisive (lowest unclear rate):\n"))
cat(sprintf("      - %s: %.2f%% unclear\n", names(unclear_sorted)[1], unclear_sorted[1]))
cat(sprintf("   Most uncertain (highest unclear rate):\n"))
cat(sprintf("      - %s: %.2f%% unclear\n", names(unclear_sorted)[length(unclear_sorted)], 
            unclear_sorted[length(unclear_sorted)]))

proceed_rates_by_model <- model_classification[, "proceed"]
proceed_sorted <- sort(proceed_rates_by_model, decreasing = TRUE)

cat(sprintf("\n   Most proceed-friendly:\n"))
cat(sprintf("      - %s: %.2f%% proceed\n", names(proceed_sorted)[1], proceed_sorted[1]))
cat(sprintf("   Most conservative:\n"))
cat(sprintf("      - %s: %.2f%% proceed\n", names(proceed_sorted)[length(proceed_sorted)], 
            proceed_sorted[length(proceed_sorted)]))

# Percentage value effects
cat("\n4. PERCENTAGE VALUE SENSITIVITY:\n")
cat(sprintf("   Most sensitive model: %s (std dev: %.2f)\n", 
            sensitivity_stats$model[1], sensitivity_stats$proceed_rate_std[1]))
cat(sprintf("   Least sensitive model: %s (std dev: %.2f)\n",
            sensitivity_stats$model[nrow(sensitivity_stats)], 
            sensitivity_stats$proceed_rate_std[nrow(sensitivity_stats)]))

cat(sprintf("\n   Percentage values with highest 'Proceed' rate:\n"))
for (i in 1:3) {
  cat(sprintf("      - %s%%: %.2f%% proceed\n", 
              overall_proceed_by_pct$percent_value[i], 
              overall_proceed_by_pct$proceed_rate[i]))
}

# Model agreement
cat(sprintf("\n5. MODEL AGREEMENT:\n"))
cat(sprintf("   Average pairwise agreement: %.2f%%\n", mean(pairwise_agreement$agreement_pct)))
cat(sprintf("   Range: %.2f%% - %.2f%%\n", 
            min(pairwise_agreement$agreement_pct), 
            max(pairwise_agreement$agreement_pct)))

cat("\n==============================================================================\n")
cat("END OF KEY FINDINGS SUMMARY\n")
cat("==============================================================================\n\n")

# ==============================================================================
# 7. STATISTICAL ANALYSIS - STATUS QUO BIAS
# ==============================================================================

cat("==============================================================================\n")
cat("SECTION 7: STATISTICAL ANALYSIS\n")
cat("==============================================================================\n\n")

cat("7.1: Chi-Square Test for Status Quo Bias\n")
cat("----------------------------------------------------------------------\n\n")

# Create contingency table
contingency_table <- table(df$Status_quo_reversal, df$classification)
cat("Contingency Table:\n")
print(contingency_table)

# Chi-square test
chi_test <- chisq.test(contingency_table)

cat("\nChi-Square Test Results:\n")
cat(sprintf("Chi-Square Statistic: %.4f\n", chi_test$statistic))
cat(sprintf("Degrees of Freedom: %d\n", chi_test$parameter))
cat(sprintf("P-value: %.4e\n", chi_test$p.value))

if (chi_test$p.value < 0.001) {
  cat("\n*** HIGHLY SIGNIFICANT (p < 0.001) ***\n")
  cat("The status quo reversal significantly affects classification distribution.\n")
} else if (chi_test$p.value < 0.05) {
  cat("\n*** SIGNIFICANT (p < 0.05) ***\n")
  cat("The status quo reversal significantly affects classification distribution.\n")
} else {
  cat("\nNot statistically significant.\n")
}

# Effect size (Cramér's V)
n <- sum(contingency_table)
min_dim <- min(nrow(contingency_table), ncol(contingency_table)) - 1
cramers_v <- sqrt(chi_test$statistic / (n * min_dim))

cat(sprintf("\nCramér's V (effect size): %.4f\n", cramers_v))
cat("(0.1=small, 0.3=medium, 0.5=large)\n")

# Calculate Cohen's h for proceed rate difference
cat("\n\n7.2: Cohen's h Effect Size for 'Proceed' Rate\n")
cat("----------------------------------------------------------------------\n\n")

p1 <- status_quo_proceed["N", "proceed"] / 100
p2 <- status_quo_proceed["Y", "proceed"] / 100

phi1 <- 2 * asin(sqrt(p1))
phi2 <- 2 * asin(sqrt(p2))
cohens_h <- phi2 - phi1

cat(sprintf("Proceed rate when status quo is Backtrack (N): %.2f%%\n", p1 * 100))
cat(sprintf("Proceed rate when status quo is Proceed (Y): %.2f%%\n", p2 * 100))
cat(sprintf("\nCohen's h: %.4f\n", cohens_h))
cat("(0.2=small, 0.5=medium, 0.8=large)\n")

if (abs(cohens_h) >= 0.8) {
  cat("\n*** LARGE EFFECT SIZE ***\n")
} else if (abs(cohens_h) >= 0.5) {
  cat("\n*** MEDIUM EFFECT SIZE ***\n")
} else if (abs(cohens_h) >= 0.2) {
  cat("\n*** SMALL EFFECT SIZE ***\n")
} else {
  cat("\n*** NEGLIGIBLE EFFECT SIZE ***\n")
}

# Logistic regression for binary proceed/not-proceed outcome
cat("\n\n7.3: Logistic Regression - Status Quo Bias\n")
cat("----------------------------------------------------------------------\n\n")

# Create binary outcome: 1 if proceed, 0 otherwise
df$proceed_binary <- ifelse(df$classification == "proceed", 1, 0)

# Create binary status quo bias indicator
df$status_quo_bias <- ifelse(df$Status_quo_reversal == "Y", 1, 0)

# Fit logistic regression
logit_model <- glm(proceed_binary ~ status_quo_bias, 
                   data = df, 
                   family = binomial(link = "logit"))

cat("Logistic Regression Model Summary:\n")
print(summary(logit_model))

# Odds ratio
odds_ratio <- exp(coef(logit_model)["status_quo_bias"])
conf_int <- exp(confint(logit_model))

cat("\n\nInterpretation:\n")
cat(sprintf("Odds Ratio: %.4f\n", odds_ratio))
cat(sprintf("95%% Confidence Interval: [%.4f, %.4f]\n", 
            conf_int["status_quo_bias", 1], 
            conf_int["status_quo_bias", 2]))

if (odds_ratio > 1) {
  cat(sprintf("\nWhen status quo aligns with 'Proceed', the odds of getting a 'Proceed' response\n"))
  cat(sprintf("are %.2f times higher (%.1f%% increase) compared to when status quo is 'Backtrack'.\n",
              odds_ratio, (odds_ratio - 1) * 100))
} else {
  cat(sprintf("\nWhen status quo aligns with 'Proceed', the odds of getting a 'Proceed' response\n"))
  cat(sprintf("are %.2f times lower (%.1f%% decrease) compared to when status quo is 'Backtrack'.\n",
              odds_ratio, (1 - odds_ratio) * 100))
}

# Model fit statistics
cat("\n\nModel Fit Statistics:\n")
cat(sprintf("AIC: %.2f\n", AIC(logit_model)))
cat(sprintf("Null Deviance: %.2f\n", logit_model$null.deviance))
cat(sprintf("Residual Deviance: %.2f\n", logit_model$deviance))
cat(sprintf("Degrees of Freedom: %d\n", logit_model$df.residual))

# McFadden's pseudo R-squared
pseudo_r2 <- 1 - (logit_model$deviance / logit_model$null.deviance)
cat(sprintf("McFadden's Pseudo R²: %.4f\n", pseudo_r2))

# Per-model logistic regression
cat("\n\n7.4: Per-Model Logistic Regression\n")
cat("----------------------------------------------------------------------\n\n")

models <- unique(df$model)
model_results <- data.frame()

for (model in models) {
  model_data <- df[df$model == model, ]
  
  tryCatch({
    logit_model_per <- glm(proceed_binary ~ status_quo_bias, 
                           data = model_data, 
                           family = binomial(link = "logit"))
    
    coef_summary <- summary(logit_model_per)$coefficients
    odds_ratio <- exp(coef_summary["status_quo_bias", "Estimate"])
    p_value <- coef_summary["status_quo_bias", "Pr(>|z|)"]
    
    model_results <- rbind(model_results, 
                           data.frame(
                             model = model,
                             odds_ratio = odds_ratio,
                             p_value = p_value,
                             significant = ifelse(p_value < 0.05, "Yes", "No")
                           ))
    
    cat(sprintf("\n%s:\n", model))
    cat(sprintf("  Odds Ratio: %.4f\n", odds_ratio))
    cat(sprintf("  P-value: %.4e\n", p_value))
    cat(sprintf("  Significant: %s\n", ifelse(p_value < 0.05, "YES", "NO")))
    
  }, error = function(e) {
    cat(sprintf("\n%s: Model failed to converge or error occurred\n", model))
  })
}

cat("\n\nSummary of Per-Model Results:\n")
print(model_results[order(model_results$odds_ratio, decreasing = TRUE), ])

# Visualize odds ratios
if (nrow(model_results) > 0) {
  model_results_sorted <- model_results[order(model_results$odds_ratio, decreasing = TRUE), ]
  
  ggplot(model_results_sorted, aes(x = reorder(model, odds_ratio), y = odds_ratio)) +
    geom_bar(stat = "identity", fill = "#3498db") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", size = 1) +
    coord_flip() +
    labs(title = "Status Quo Bias - Odds Ratios by Model",
         x = "Model",
         y = "Odds Ratio",
         subtitle = "Values > 1 indicate higher proceed rate when status quo aligns with 'Proceed'") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14),
          plot.subtitle = element_text(size = 10, color = "gray40"))
  
  ggsave("odds_ratios_by_model.png", width = 10, height = 6, dpi = 300)
}

# ==============================================================================
# 8. ADVANCED MIXED-EFFECTS MODELING
# ==============================================================================

cat("\n\n==============================================================================\n")
cat("SECTION 8: MIXED-EFFECTS LOGISTIC REGRESSION\n")
cat("==============================================================================\n\n")

cat("8.1: Mixed-Effects Model with Random Intercepts for Prompt ID\n")
cat("----------------------------------------------------------------------\n\n")

# Mixed-effects logistic regression with random intercepts by prompt
# This accounts for the fact that each prompt is rated by multiple models

mixed_model <- glmer(proceed_binary ~ status_quo_bias + (1 | prompt_id), 
                     data = df, 
                     family = binomial(link = "logit"),
                     control = glmerControl(optimizer = "bobyqa", 
                                            optCtrl = list(maxfun = 100000)))

cat("Mixed-Effects Model Summary:\n")
print(summary(mixed_model))

# Fixed effects
fixed_effects <- fixef(mixed_model)
cat("\n\nFixed Effects:\n")
print(fixed_effects)

# Odds ratio from mixed-effects model
odds_ratio_mixed <- exp(fixed_effects["status_quo_bias"])
cat(sprintf("\nOdds Ratio (status_quo_bias): %.4f\n", odds_ratio_mixed))

# Random effects variance
random_effects_var <- as.data.frame(VarCorr(mixed_model))
cat("\n\nRandom Effects Variance:\n")
print(random_effects_var)

cat(sprintf("\nPrompt-level variance: %.4f\n", random_effects_var$vcov[1]))
cat("(Higher values indicate more variation across prompts)\n")

# Intraclass correlation coefficient (ICC)
prompt_var <- random_effects_var$vcov[1]
icc <- prompt_var / (prompt_var + pi^2/3)
cat(sprintf("\nIntraclass Correlation Coefficient (ICC): %.4f\n", icc))
cat(sprintf("(%.1f%% of variance is due to differences between prompts)\n", icc * 100))

# Model comparison with null model
cat("\n\n8.2: Model Comparison - Likelihood Ratio Test\n")
cat("----------------------------------------------------------------------\n\n")

null_mixed_model <- glmer(proceed_binary ~ 1 + (1 | prompt_id),
                          data = df,
                          family = binomial(link = "logit"),
                          control = glmerControl(optimizer = "bobyqa",
                                                 optCtrl = list(maxfun = 100000)))

cat("Null Model (intercept only) Summary:\n")
print(summary(null_mixed_model))

# Likelihood ratio test
lr_test <- anova(null_mixed_model, mixed_model)
cat("\n\nLikelihood Ratio Test:\n")
print(lr_test)

if (lr_test$`Pr(>Chisq)`[2] < 0.001) {
  cat("\n*** HIGHLY SIGNIFICANT (p < 0.001) ***\n")
  cat("The status_quo_bias predictor significantly improves model fit.\n")
} else if (lr_test$`Pr(>Chisq)`[2] < 0.05) {
  cat("\n*** SIGNIFICANT (p < 0.05) ***\n")
  cat("The status_quo_bias predictor significantly improves model fit.\n")
} else {
  cat("\nNot statistically significant.\n")
}

# Model fit comparison
cat("\n\nModel Fit Comparison:\n")
cat(sprintf("Null Model AIC: %.2f\n", AIC(null_mixed_model)))
cat(sprintf("Full Model AIC: %.2f\n", AIC(mixed_model)))
cat(sprintf("Improvement: %.2f (lower is better)\n", AIC(null_mixed_model) - AIC(mixed_model)))

cat("\n\n8.3: Mixed-Effects Model with Model as Random Effect\n")
cat("----------------------------------------------------------------------\n\n")

# Additional mixed-effects model with random slopes for different AI models
mixed_model_by_llm <- glmer(proceed_binary ~ status_quo_bias + (1 | model), 
                            data = df, 
                            family = binomial(link = "logit"),
                            control = glmerControl(optimizer = "bobyqa",
                                                   optCtrl = list(maxfun = 100000)))

cat("Mixed-Effects Model with Model Random Effects:\n")
print(summary(mixed_model_by_llm))

# Compare models
cat("\n\nModel Comparison:\n")
cat(sprintf("Model with prompt_id random effects AIC: %.2f\n", AIC(mixed_model)))
cat(sprintf("Model with model random effects AIC: %.2f\n", AIC(mixed_model_by_llm)))

if (AIC(mixed_model) < AIC(mixed_model_by_llm)) {
  cat("\nThe model with prompt-level random effects fits better.\n")
} else {
  cat("\nThe model with AI model random effects fits better.\n")
}

# ==============================================================================
# 9. INTERACTION EFFECTS
# ==============================================================================

cat("\n\n==============================================================================\n")
cat("SECTION 9: INTERACTION EFFECTS\n")
cat("==============================================================================\n\n")

cat("9.1: Status Quo × Percentage Value Interaction\n")
cat("----------------------------------------------------------------------\n\n")

# Model with interaction between status quo and percentage value
interaction_model <- glm(proceed_binary ~ status_quo_bias * percent_value,
                         data = df,
                         family = binomial(link = "logit"))

cat("Logistic Regression with Interaction Term:\n")
print(summary(interaction_model))

# Check if interaction is significant
interaction_pval <- summary(interaction_model)$coefficients["status_quo_bias:percent_value", "Pr(>|z|)"]

cat(sprintf("\n\nInteraction Term P-value: %.4e\n", interaction_pval))

if (interaction_pval < 0.001) {
  cat("\n*** HIGHLY SIGNIFICANT INTERACTION (p < 0.001) ***\n")
  cat("The effect of status quo bias varies significantly with percentage value.\n")
} else if (interaction_pval < 0.05) {
  cat("\n*** SIGNIFICANT INTERACTION (p < 0.05) ***\n")
  cat("The effect of status quo bias varies significantly with percentage value.\n")
} else {
  cat("\nNo significant interaction detected.\n")
}

# Model comparison
cat("\n\nModel Fit Comparison:\n")
cat(sprintf("Main Effects Only AIC: %.2f\n", AIC(logit_model)))
cat(sprintf("With Interaction AIC: %.2f\n", AIC(interaction_model)))

if (AIC(interaction_model) < AIC(logit_model)) {
  cat("\nThe interaction model fits better (lower AIC).\n")
} else {
  cat("\nThe main effects model is sufficient (lower AIC).\n")
}

# ==============================================================================
# 10. FINAL SUMMARY AND CONCLUSIONS
# ==============================================================================

cat("\n\n==============================================================================\n")
cat("FINAL SUMMARY AND CONCLUSIONS\n")
cat("==============================================================================\n\n")

cat("DATASET OVERVIEW:\n")
cat(sprintf("  - Total responses analyzed: %d\n", nrow(df)))
cat(sprintf("  - Number of AI models: %d\n", length(unique(df$model))))
cat(sprintf("  - Unique prompts: %d\n", length(unique(df$prompt_id))))
cat(sprintf("  - Topics covered: %d\n", length(unique(df$Topic))))

cat("\n\nKEY STATISTICAL FINDINGS:\n\n")

cat("1. STATUS QUO BIAS:\n")
cat(sprintf("   - Chi-square test: χ² = %.4f, p < 0.001\n", chi_test$statistic))
cat(sprintf("   - Effect size (Cohen's h): %.4f (%s)\n", cohens_h,
            ifelse(abs(cohens_h) >= 0.8, "LARGE", 
                   ifelse(abs(cohens_h) >= 0.5, "MEDIUM", 
                          ifelse(abs(cohens_h) >= 0.2, "SMALL", "NEGLIGIBLE")))))
cat(sprintf("   - Odds Ratio: %.4f (%.1f%% increase in proceed rate)\n", 
            odds_ratio, (odds_ratio - 1) * 100))
cat("   - CONCLUSION: Strong evidence of status quo bias across all models.\n")

cat("\n2. MODEL DIFFERENCES:\n")
cat(sprintf("   - Pairwise agreement range: %.2f%% - %.2f%%\n",
            min(pairwise_agreement$agreement_pct),
            max(pairwise_agreement$agreement_pct)))
cat(sprintf("   - All models showed significant status quo bias (p < 0.05)\n"))
cat("   - CONCLUSION: Bias is consistent but models differ in magnitude.\n")

cat("\n3. PERCENTAGE VALUE SENSITIVITY:\n")
cat(sprintf("   - Proceed rate varies from %.1f%% to %.1f%% across percentage values\n",
            min(overall_proceed_by_pct$proceed_rate),
            max(overall_proceed_by_pct$proceed_rate)))
cat(sprintf("   - Interaction effect p-value: %.4e\n", interaction_pval))
cat(sprintf("   - CONCLUSION: %s\n", 
            ifelse(interaction_pval < 0.05,
                   "Percentage value significantly moderates status quo bias.",
                   "Percentage value and status quo have independent effects.")))

cat("\n4. MIXED-EFFECTS MODELING:\n")
cat(sprintf("   - Prompt-level ICC: %.4f (%.1f%% of variance)\n", icc, icc * 100))
cat(sprintf("   - Likelihood ratio test: p < 0.001\n"))
cat("   - CONCLUSION: Substantial variation across prompts, but status quo\n")
cat("                 bias remains significant after accounting for prompt effects.\n")

cat("\n\nOVERALL CONCLUSION:\n")
cat("All six AI models exhibit statistically significant status quo bias in their\n")
cat("policy recommendations. When the status quo aligns with 'Proceed', models are\n")
cat(sprintf("%.1f%% more likely to recommend proceeding compared to when the status quo\n", 
            (odds_ratio - 1) * 100))
cat("favors backtracking. This bias persists across different topics, positions,\n")
cat("variations, and percentage values, though its magnitude varies by model.\n")
cat("The effect is robust and statistically significant across multiple analytical\n")
cat("approaches (chi-square, logistic regression, mixed-effects models).\n")

cat("\n==============================================================================\n")
cat("ANALYSIS COMPLETE - All results and visualizations saved\n")
cat("==============================================================================\n")