
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "descriptor_error_analysis.csv"

FEATURES = [
    "molecular_weight",
    "TPSA",
    "LogP",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
]


def main():
    df = pd.read_csv(INPUT_PATH)

    # Residual = measured value minus predicted value.
    df["residual"] = (
        df["actual_log_solubility"]
        - df["predicted_log_solubility"]
    )

    df["absolute_error"] = df["residual"].abs()

    print("Molecules analyzed:", len(df))
    print("\n===== OVERALL ERROR STATISTICS =====")

    print(f"Mean residual: {df['residual'].mean():.4f}")
    print(f"Mean absolute error: {df['absolute_error'].mean():.4f}")
    print(f"Residual standard deviation: {df['residual'].std():.4f}")

    print("\nInterpretation:")
    print("Positive residual = model underpredicts solubility.")
    print("Negative residual = model overpredicts solubility.")

    # Examine relationships between descriptors and errors.
    print("\n===== DESCRIPTOR VS ABSOLUTE ERROR CORRELATION =====")

    correlations = df[FEATURES].corrwith(df["absolute_error"])
    correlations = correlations.sort_values(
        key=lambda values: values.abs(),
        ascending=False,
    )

    print(correlations.round(3).to_string())

    correlation_path = REPORTS_DIR / "descriptor_error_correlations.csv"
    correlations.rename("correlation_with_absolute_error").to_csv(
        correlation_path,
        header=True,
    )

    # Save a residual-analysis table.
    residual_path = REPORTS_DIR / "residual_diagnostics.csv"

    df[
        [
            "smiles",
            "actual_log_solubility",
            "predicted_log_solubility",
            "residual",
            "absolute_error",
            *FEATURES,
        ]
    ].to_csv(residual_path, index=False)

    # Plot absolute error against each key descriptor.
    for feature in ["molecular_weight", "TPSA", "LogP"]:
        plt.figure(figsize=(8, 6))

        plt.scatter(
            df[feature],
            df["absolute_error"],
            alpha=0.7,
        )

        plt.xlabel(feature)
        plt.ylabel("Absolute prediction error (log mol/L)")
        plt.title(f"MolecularAI: Error vs {feature}")
        plt.tight_layout()

        plot_path = REPORTS_DIR / f"error_vs_{feature}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print("Saved plot:", plot_path)

    print("\nSaved correlations:", correlation_path)
    print("Saved residual diagnostics:", residual_path)


if __name__ == "__main__":
    main()
