
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402


DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
OUTPUT_DIR = ROOT / "reports"
TARGET = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


def canonicalize_smiles(smiles):
    molecule = Chem.MolFromSmiles(str(smiles).strip())

    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule, canonical=True)


def get_scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and clean the dataset.
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()

    df["canonical_smiles"] = df[SMILES_COLUMN].apply(
        canonicalize_smiles
    )
    df = df.dropna(subset=["canonical_smiles"]).copy()

    # 2. Aggregate repeated representations of the same molecule.
    df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            **{
                TARGET: (TARGET, "mean"),
                "measurement_count": (TARGET, "size"),
            }
        )
        .rename(columns={"canonical_smiles": SMILES_COLUMN})
    )

    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    print("Unique canonical molecules:", len(df))

    # 3. Generate features once.
    print("Generating molecular features...")

    X = np.vstack([
        calculate_features(smiles)[0]
        for smiles in df[SMILES_COLUMN]
    ])

    y = df[TARGET].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()

    # 4. Reproduce the five scaffold splits.
    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=42,
    )

    results = []

    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_scaffolds = set(groups[train_idx])
        test_scaffolds = set(groups[test_idx])

        if train_scaffolds & test_scaffolds:
            raise ValueError(
                f"Scaffold leakage detected in split {split_number}"
            )

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Baseline: predict the training-set mean for every test molecule.
        baseline_predictions = np.full(
            len(test_idx), y_train.mean()
        )

        results.append({
            "split": split_number,
            "model": "Mean Baseline",
            "test_molecules": len(test_idx),
            "MAE": mean_absolute_error(
                y_test, baseline_predictions
            ),
            "RMSE": np.sqrt(
                mean_squared_error(y_test, baseline_predictions)
            ),
            "R2": r2_score(y_test, baseline_predictions),
        })

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

        print(f"\nEvaluating split {split_number}...")

        for name, model in models.items():
            model.fit(X[train_idx], y_train)
            predictions = model.predict(X[test_idx])

            results.append({
                "split": split_number,
                "model": name,
                "test_molecules": len(test_idx),
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(
                    mean_squared_error(y_test, predictions)
                ),
                "R2": r2_score(y_test, predictions),
            })

    # 5. Save detailed results and averages.
    results_df = pd.DataFrame(results)

    detailed_path = OUTPUT_DIR / "baseline_scaffold_results.csv"
    summary_path = OUTPUT_DIR / "baseline_scaffold_summary.csv"

    results_df.to_csv(detailed_path, index=False)

    summary = (
        results_df.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
        .sort_values("MAE_mean")
    )

    summary.to_csv(summary_path, index=False)

    print("\n===== BASELINE VS ML MODELS =====")
    print(summary.round(4).to_string(index=False))

    print("\nDetailed results saved:", detailed_path)
    print("Summary saved:", summary_path)
    print("Saved app model and Streamlit app were not modified.")


if __name__ == "__main__":
    main()
