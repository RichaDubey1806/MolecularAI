
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
OUTPUT_DIR = ROOT / "reports"
TARGET = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


def canonicalize_smiles(value):
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df["canonical_smiles"] = df[SMILES_COLUMN].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()

    # Aggregate repeated representations of the same molecule.
    df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg({TARGET: "mean"})
        .rename(columns={"canonical_smiles": SMILES_COLUMN})
    )

    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    print("Unique canonical molecules:", len(df))
    print("Unique scaffolds:", df["scaffold"].nunique())

    print("\nGenerating features...")
    X = np.vstack([
        calculate_features(smiles)[0]
        for smiles in df[SMILES_COLUMN]
    ])
    y = df[TARGET].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()

    configurations = {
        "Original_leaf5": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
        "Candidate_leaf2": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.20,
        random_state=42,
    )

    rows = []

    for split_no, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_scaffolds = set(groups[train_idx])
        test_scaffolds = set(groups[test_idx])

        if train_scaffolds & test_scaffolds:
            raise ValueError(f"Scaffold leakage in split {split_no}")

        print(
            f"\nSplit {split_no}: "
            f"{len(train_idx)} train, {len(test_idx)} test"
        )

        # Both configurations use the same train/test indices.
        for name, params in configurations.items():
            model = RandomForestRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])

            rows.append({
                "split": split_no,
                "model": name,
                "train_molecules": len(train_idx),
                "test_molecules": len(test_idx),
                "MAE": mean_absolute_error(y[test_idx], pred),
                "RMSE": np.sqrt(mean_squared_error(y[test_idx], pred)),
                "R2": r2_score(y[test_idx], pred),
                "scaffold_overlap": len(
                    train_scaffolds & test_scaffolds
                ),
            })

    results = pd.DataFrame(rows)

    detail_path = OUTPUT_DIR / "repeated_scaffold_config_comparison.csv"
    summary_path = OUTPUT_DIR / "repeated_scaffold_config_summary.csv"

    results.to_csv(detail_path, index=False)

    summary = (
        results.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print("\n===== FIVE-SPLIT CONFIGURATION COMPARISON =====")
    print(summary.round(4).to_string(index=False))

    print("\n===== PER-SPLIT MAE =====")
    print(
        results.pivot(index="split", columns="model", values="MAE")
        .round(4)
        .to_string()
    )

    print("\nDetailed results:", detail_path)
    print("Summary:", summary_path)
    print("Saved application models were not modified.")
    print(
        "Caution: these splits were already used during development "
        "and tuning; results are not an untouched final benchmark."
    )


if __name__ == "__main__":
    main()
