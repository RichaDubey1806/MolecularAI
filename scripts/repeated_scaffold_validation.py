
import sys
from pathlib import Path

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
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
OUTPUT_DIR = ROOT / "reports"

TARGET = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"

N_SPLITS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Identify each molecule's chemical scaffold
# --------------------------------------------------

def get_scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold)


# --------------------------------------------------
# 3. Prepare dataset and molecular features
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df = df.drop_duplicates(subset=[SMILES_COLUMN]).reset_index(drop=True)

    print("Unique molecules:", len(df))

    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    print("Unique scaffolds:", df["scaffold"].nunique())
    print("\nCalculating molecular features...")

    # Calculate molecular features once and reuse them.
    X = np.vstack([
        calculate_features(smiles)[0]
        for smiles in df[SMILES_COLUMN]
    ])

    y = df[TARGET].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()

    print("Feature matrix shape:", X.shape)
    print("Feature calculation complete.")

    # Keep each chemical scaffold entirely in train or test.
    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Evaluation-only models; the app's saved model is untouched.
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

    all_results = []

    # --------------------------------------------------
    # 4. Repeat scaffold-based evaluation
    # --------------------------------------------------

    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_scaffolds = set(groups[train_idx])
        test_scaffolds = set(groups[test_idx])
        overlap = train_scaffolds & test_scaffolds

        print(f"\n{'=' * 50}")
        print(f"SPLIT {split_number} OF {N_SPLITS}")
        print(f"{'=' * 50}")

        if overlap:
            raise ValueError(
                f"Scaffold leakage in split {split_number}!"
            )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print("Training molecules:", len(train_idx))
        print("Testing molecules:", len(test_idx))
        print("Training scaffolds:", len(train_scaffolds))
        print("Testing scaffolds:", len(test_scaffolds))
        print("Scaffold overlap: 0")

        for model_name, base_model in models.items():
            print(f"\nTraining {model_name}...")

            base_model.fit(X_train, y_train)
            predictions = base_model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            all_results.append({
                "split": split_number,
                "model": model_name,
                "train_molecules": len(train_idx),
                "test_molecules": len(test_idx),
                "train_scaffolds": len(train_scaffolds),
                "test_scaffolds": len(test_scaffolds),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            })

            print(f"MAE:  {mae:.4f} log mol/L")
            print(f"RMSE: {rmse:.4f} log mol/L")
            print(f"R²:   {r2:.4f}")

    # --------------------------------------------------
    # 5. Save detailed results
    # --------------------------------------------------

    results = pd.DataFrame(all_results)

    detailed_path = OUTPUT_DIR / "repeated_scaffold_validation.csv"
    results.to_csv(detailed_path, index=False)

    # --------------------------------------------------
    # 6. Calculate and save the summary
    # --------------------------------------------------

    summary = (
        results.groupby("model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )

    # Give every column a unique name for PowerShell compatibility.
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    summary = summary.reset_index()

    summary_path = OUTPUT_DIR / "repeated_scaffold_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------
    # 7. Display final results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL REPEATED SCAFFOLD VALIDATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    print("\nSaved detailed results:", detailed_path)
    print("Saved summary:", summary_path)

    print(
        "\nReminder: this script does not replace "
        "the app's saved model."
    )


if __name__ == "__main__":
    main()
