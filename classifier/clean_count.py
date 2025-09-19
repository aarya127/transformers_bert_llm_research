import argparse
import sys
import pandas as pd




def detect_column(columns, keywords):
        keywords = [k.lower() for k in keywords]
        for col in columns:
                low = col.lower()
                if any(k in low for k in keywords):
                        return col
        return None


def main():
        parser = argparse.ArgumentParser(description="Count unique prompts and model frequencies in a CSV")
        parser.add_argument(
                "--file",
                "-f",
                required=True,
                help="Path to CSV file (default: vehicle_prompt_results_joined_by_response.csv)",
        )
        parser.add_argument(
                "--top",
                "-t",
                type=int,
                default=20,
                help="Show top N model counts (default: 20)",
        )
        args = parser.parse_args()

        try:
                df = pd.read_csv(args.file)
        except Exception as e:
                print(f"Failed to read CSV '{args.file}': {e}", file=sys.stderr)
                sys.exit(2)

        cols = list(df.columns)
        prompt_col = detect_column(cols, ["prompt", "question", "input", "query", "text"])
        model_col = detect_column(cols, ["model", "model_name", "engine", "modelid", "model-id"])

        if prompt_col is None:
                print("Could not auto-detect a 'prompt' column. Available columns:")
                for c in cols:
                        print(f"  - {c}")
                print("Rerun with a --prompt-column argument (not implemented) or rename your column to include 'prompt'.")
                sys.exit(3)

        print(f"Detected prompt column: {prompt_col!r}")
        unique_prompts = df[prompt_col].nunique(dropna=True)
        total_prompts = len(df)
        print(f"Total rows: {total_prompts}")
        print(f"Unique prompts: {unique_prompts}")

        if model_col is None:
                print("\nCould not auto-detect a 'model' column. Available columns:")
                for c in cols:
                        print(f"  - {c}")
                print("If your dataset doesn't include model information, no model counts will be shown.")
                return

        print(f"\nDetected model column: {model_col!r}")
        counts = df[model_col].fillna("<MISSING>").value_counts(dropna=False)
        print(f"Model counts (top {args.top}):")
        for model, cnt in counts.head(args.top).items():
                print(f"  {model}: {cnt}")
        if len(counts) > args.top:
                print(f"  ... and {len(counts) - args.top} more models")

if __name__ == "__main__":
        main()