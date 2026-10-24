
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
REPORT_DIR = ROOT / "reports"
TARGET = "measured log solubility in mols per litre"
SMILES = "smiles"


def canonicalize(value):
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold_for(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load, clean, canonicalize and aggregate duplicate structures.
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES, TARGET]).copy()
    df["canonical"] = df[SMILES].apply(canonicalize)
    df = df.dropna(subset=["canonical"])

    df = (
        df.groupby("canonical", as_index=False)
        .agg({TARGET: "mean"})
        .rename(columns={"canonical": SMILES})
    )

    df["scaffold"] = df[SMILES].apply(scaffold_for)

    # Reproduce the existing seeded scaffold split.
    rng = np.random.default_rng(42)
    groups = list(df.groupby("scaffold").groups.values())
    rng.shuffle(groups)

    test_indices = []
    train_indices = []
    target_test_size = 0.20 * len(df)

    for group in groups:
        if len(test_indices) < target_test_size:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()

    scaffold_overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    structure_overlap = set(train_df[SMILES]) & set(test_df[SMILES])

    print("Unique molecules:", len(df))
    print("Training molecules:", len(train_df))
    print("Testing molecules:", len(test_df))
    print("Training scaffolds:", train_df["scaffold"].nunique())
    print("Testing scaffolds:", test_df["scaffold"].nunique())
    print("Scaffold overlap:", len(scaffold_overlap))
    print("Canonical structure overlap:", len(structure_overlap))

    if scaffold_overlap or structure_overlap:
        raise ValueError("Leakage check failed. Results are not valid.")

    # Calculate features once and use identical arrays for both models.
    X_train = np.vstack([
        calculate_features(s)[0] for s in train_df[SMILES]
    ])
    X_test = np.vstack([
        calculate_features(s)[0] for s in test_df[SMILES]
    ])
    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    configurations = {
        "Original settings (leaf=5)": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
        "Candidate settings (leaf=2)": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    results = []

    for name, params in configurations.items():
        print(f"\nTraining evaluation model: {name}")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

    comparison = pd.DataFrame(results)
    output_path = REPORT_DIR / "baseline_vs_candidate_same_split.csv"
    comparison.to_csv(output_path, index=False)

    print("\n===== SAME-SPLIT CONFIGURATION COMPARISON =====")
    print(comparison.round(4).to_string(index=False))
    print("\nReport saved to:", output_path)
    print("Saved application models were not modified.")
    print(
        "Caution: this split has been used during development; "
        "these are not untouched final benchmark results."
    )


if __name__ == "__main__":
    main()
