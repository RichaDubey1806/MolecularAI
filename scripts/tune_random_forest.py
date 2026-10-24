
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

from predict import calculate_features

DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
REPORT_DIR = ROOT / "reports"
TARGET = "measured log solubility in mols per litre"


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def scaffold_for(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna(
        subset=["smiles", TARGET]
    ).copy()

    df["smiles"] = df["smiles"].apply(canonicalize)
    df = df.dropna(subset=["smiles"])

    df = (
        df.groupby("smiles", as_index=False)
        .agg({TARGET: "mean"})
    )

    df["scaffold"] = df["smiles"].apply(scaffold_for)

    print("Unique canonical molecules:", len(df))
    print("Unique scaffolds:", df["scaffold"].nunique())
    print("Generating features...")

    X = np.vstack([
        calculate_features(smiles)[0]
        for smiles in df["smiles"]
    ])
    y = df[TARGET].to_numpy(dtype=float)
    groups = df["scaffold"].to_numpy()

    # Identical five scaffold splits for every configuration.
    splitter = GroupShuffleSplit(
        n_splits=5, test_size=0.20, random_state=42
    )

    splits = list(splitter.split(X, y, groups=groups))

    configurations = [
        {"n_estimators": 200, "max_depth": 12,
         "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 300, "max_depth": 12,
         "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 16,
         "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": None,
         "min_samples_leaf": 5, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 12,
         "min_samples_leaf": 2, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 12,
         "min_samples_leaf": 10, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 12,
         "min_samples_leaf": 5, "max_features": 0.7},
    ]

    results = []

    for config_number, params in enumerate(configurations, start=1):
        print(f"\nConfiguration {config_number}/{len(configurations)}")
        print(params)

        for split_number, (train_idx, test_idx) in enumerate(
            splits, start=1
        ):
            model = RandomForestRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])

            results.append({
                "config": config_number,
                "split": split_number,
                **params,
                "train_molecules": len(train_idx),
                "test_molecules": len(test_idx),
                "MAE": mean_absolute_error(y[test_idx], pred),
                "RMSE": np.sqrt(
                    mean_squared_error(y[test_idx], pred)
                ),
                "R2": r2_score(y[test_idx], pred),
            })

            print(
                f"  Split {split_number}: "
                f"MAE={results[-1]['MAE']:.4f}"
            )

    details = pd.DataFrame(results)
    details.to_csv(
        REPORT_DIR / "random_forest_tuning_details.csv",
        index=False,
    )

    summary = (
        details.groupby("config")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(
        pd.DataFrame([
            {"config": i, **params}
            for i, params in enumerate(configurations, start=1)
        ]),
        on="config",
    ).sort_values("MAE_mean")

    summary.to_csv(
        REPORT_DIR / "random_forest_tuning_summary.csv",
        index=False,
    )

    print("\n===== TUNING SUMMARY (LOWER MAE IS BETTER) =====")
    print(summary.round(4).to_string(index=False))
    print("\nReports saved in:", REPORT_DIR)
    print("No saved model or app files were modified.")


if __name__ == "__main__":
    main()
