import pandas as pd
from openpyxl import load_workbook
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'vehicle_prompt_results_deduped.csv')
XLSX_PATH = os.path.join(BASE_DIR, 'data', 'Prompt library.xlsx')
SHEET_NAME = "Results Vehicle Status Quo"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
if not os.path.exists(XLSX_PATH):
    raise FileNotFoundError(f"Excel file not found: {XLSX_PATH}")

# Read the CSV
csv_df = pd.read_csv(CSV_PATH)

# Append/replace sheet in the existing workbook
with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    csv_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)

print(f"Added {os.path.basename(CSV_PATH)} as sheet '{SHEET_NAME}' in {os.path.basename(XLSX_PATH)}")
