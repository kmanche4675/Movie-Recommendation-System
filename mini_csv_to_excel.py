# mini_csv_to_excel.py
import pandas as pd
from pathlib import Path

def csv_to_excel(csv_path, excel_path=None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if excel_path is None:
        excel_path = csv_path.with_suffix(".xlsx")

    df = pd.read_csv(csv_path)

    # Auto-clean column names and formatting
    df.columns = [c.strip() for c in df.columns]

    # Save to Excel
    df.to_excel(excel_path, index=False)

    print(f"✔ Converted to Excel: {excel_path}")

if __name__ == "__main__":
    # EDIT THESE TO MATCH YOUR FILES
    csv_files = [
        "classic_svd_optuna_history.csv",
        "cult_svd_optuna_history.csv"
    ]

    for f in csv_files:
        try:
            csv_to_excel(f)
        except Exception as e:
            print(f"Failed converting {f}: {e}")
