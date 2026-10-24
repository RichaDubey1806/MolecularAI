
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "reports" / "scaffold_test_predictions.csv"

df = pd.read_csv(CSV_PATH)

print("\n===== DATASET OVERVIEW =====")
print(f"Number of test molecules: {len(df)}")
print(f"Mean absolute error: {df['absolute_error'].mean():.4f}")
print(f"Median absolute error: {df['absolute_error'].median():.4f}")
print(f"Maximum absolute error: {df['absolute_error'].max():.4f}")

print("\n===== ERROR DISTRIBUTION =====")
for threshold in [0.5, 1.0, 1.5, 2.0]:
    count = (df["absolute_error"] > threshold).sum()
    percentage = 100 * count / len(df)
    print(
        f"Errors above {threshold:.1f} log mol/L: "
        f"{count} molecules ({percentage:.1f}%)"
    )

print("\n===== MOST ACCURATE PREDICTIONS =====")
print(
    df.nsmallest(5, "absolute_error")[
        ["smiles", "actual_log_solubility",
         "predicted_log_solubility", "absolute_error"]
    ].to_string(index=False)
)

print("\n===== LARGEST PREDICTION ERRORS =====")
print(
    df.nlargest(10, "absolute_error")[
        ["smiles", "actual_log_solubility",
         "predicted_log_solubility", "absolute_error"]
    ].to_string(index=False)
)
