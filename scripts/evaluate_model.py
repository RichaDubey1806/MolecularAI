
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402


DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
OUTPUT_DIR = ROOT / "reports"

TARGET = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


def get_scaffold(smiles):
    """Return the Bemis-Murcko scaffold for a molecule."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df = df.drop_duplicates(subset=[SMILES_COLUMN]).reset_index(drop=True)

    print("Unique molecules:", len(df))

    # Group molecules by molecular scaffold.
    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    # Shuffle scaffold groups reproducibly.
    rng = np.random.default_rng(42)
    scaffold_groups = list(df.groupby("scaffold").groups.values())
    rng.shuffle(scaffold_groups)

    # Keep complete scaffold groups in either train or test.
    target_test_size = 0.20 * len(df)
    test_indices = []
    train_indices = []

    for group in scaffold_groups:
        if len(test_indices) < target_test_size:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split is empty. Check the dataset.")

    print("Training molecules:", len(train_df))
    print("Testing molecules:", len(test_df))
    print("Training scaffolds:", train_df["scaffold"].nunique())
    print("Testing scaffolds:", test_df["scaffold"].nunique())

    overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    print("Scaffold overlap:", len(overlap))

    if overlap:
        raise ValueError("Scaffold leakage detected between train and test.")

    # Reuse the exact feature-generation function from the app.
    X_train = np.vstack([
        calculate_features(smiles)[0]
        for smiles in train_df[SMILES_COLUMN]
    ])

    X_test = np.vstack([
        calculate_features(smiles)[0]
        for smiles in test_df[SMILES_COLUMN]
    ])

    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    # Compare two evaluation-only models on the same scaffold split.
    # Neither model overwrites the saved Streamlit app model.
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison_rows = []
    predictions_by_model = {}

    for name, candidate in models.items():
        print(f"\nTraining {name}...")
        candidate.fit(X_train, y_train)
        candidate_predictions = candidate.predict(X_test)

        candidate_mae = mean_absolute_error(y_test, candidate_predictions)
        candidate_rmse = np.sqrt(
            mean_squared_error(y_test, candidate_predictions)
        )
        candidate_r2 = r2_score(y_test, candidate_predictions)

        comparison_rows.append({
            "model": name,
            "MAE": candidate_mae,
            "RMSE": candidate_rmse,
            "R2": candidate_r2,
        })

        predictions_by_model[name] = candidate_predictions

        print(f"{name} MAE:  {candidate_mae:.4f} log mol/L")
        print(f"{name} RMSE: {candidate_rmse:.4f} log mol/L")
        print(f"{name} R²:   {candidate_r2:.4f}")

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n===== MODEL COMPARISON =====")
    print(comparison.round(4).to_string(index=False))
    print("Saved model comparison:", comparison_path)

    # Use Random Forest for the existing prediction and error reports.
    predictions = predictions_by_model["Random Forest"]
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n===== RANDOM FOREST HELD-OUT SCAFFOLD EVALUATION =====")
    print(f"MAE:  {mae:.4f} log mol/L")
    print(f"RMSE: {rmse:.4f} log mol/L")
    print(f"R²:   {r2:.4f}")

    # Calculate residuals: actual minus predicted.
    residuals = y_test - predictions

    # Save molecule-level predictions and errors.
    results = pd.DataFrame({
        "smiles": test_df[SMILES_COLUMN].to_numpy(),
        "actual_log_solubility": y_test,
        "predicted_log_solubility": predictions,
        "residual": residuals,
        "absolute_error": np.abs(residuals),
    })

    predictions_path = OUTPUT_DIR / "scaffold_test_predictions.csv"
    results.to_csv(predictions_path, index=False)

    # Plot 1: predicted versus actual solubility.
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Measured solubility (log mol/L)")
    plt.ylabel("Predicted solubility (log mol/L)")
    plt.title("MolecularAI: Predicted vs Actual Solubility")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "predicted_vs_actual_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Plot 2: residuals versus measured solubility.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Measured solubility (log mol/L)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("MolecularAI: Residual Analysis")
    plt.tight_layout()

    residual_plot_path = OUTPUT_DIR / "residual_analysis_v2.png"
    plt.savefig(residual_plot_path, dpi=150)
    plt.close()

    # Report residual statistics and output locations.
    print("\nSaved predictions:", predictions_path)
    print("Saved model comparison:", comparison_path)
    print("Saved predicted-vs-actual plot:", plot_path)
    print("Saved residual plot:", residual_plot_path)
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Residual standard deviation: {np.std(residuals):.4f}")

    # Display molecules with the largest absolute errors.
    print("\n===== FIVE LARGEST PREDICTION ERRORS =====")
    print(
        results.nlargest(5, "absolute_error")[
            [
                "smiles",
                "actual_log_solubility",
                "predicted_log_solubility",
                "residual",
                "absolute_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
