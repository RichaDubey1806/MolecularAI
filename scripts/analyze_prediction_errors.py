
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features

PREDICTIONS_PATH = ROOT / "reports" / "scaffold_test_predictions.csv"
OUTPUT_PATH = ROOT / "reports" / "descriptor_error_analysis.csv"

# These names match the exact order in app/predict.py.
DESCRIPTOR_NAMES = [
    "molecular_weight",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "TPSA",
    "LogP",
]


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    descriptor_rows = []

    print("Analyzing molecules:", len(df))
    print("Calculating six molecular descriptors...")

    for smiles in df["smiles"]:
        # Shape returned by calculate_features: (1, 2054).
        features = calculate_features(smiles)

        # Extract the first six values from the feature vector.
        descriptors = features[0, :6]
        descriptor_rows.append(descriptors)

    descriptor_df = pd.DataFrame(
        descriptor_rows,
        columns=DESCRIPTOR_NAMES,
    )

    # Combine prediction results with descriptor values.
    result = pd.concat(
        [
            df.reset_index(drop=True),
            descriptor_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Recalculate absolute error for consistency.
    result["absolute_error"] = (
        result["actual_log_solubility"]
        - result["predicted_log_solubility"]
    ).abs()

    result = result.sort_values(
        "absolute_error",
        ascending=False,
    )

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n===== TEN LARGEST PREDICTION ERRORS =====")

    print(
        result[
            [
                "smiles",
                "actual_log_solubility",
                "predicted_log_solubility",
                "absolute_error",
                *DESCRIPTOR_NAMES,
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTotal molecules analyzed:", len(result))
    print("Saved descriptor analysis:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
