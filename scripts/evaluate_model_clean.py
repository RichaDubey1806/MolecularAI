
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
    """Normalize a valid SMILES into a canonical representation."""
    if pd.isna(smiles):
        return None

    molecule = Chem.MolFromSmiles(str(smiles).strip())

    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule, canonical=True)


def get_scaffold(canonical_smiles):
    """Return the Bemis-Murcko scaffold."""
    molecule = Chem.MolFromSmiles(canonical_smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and clean formatting.
    df = pd.read_csv(DATA_PATH)
    original_rows = len(df)

    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df["canonical_smiles"] = df[SMILES_COLUMN].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()

    print("Original dataset rows:", original_rows)
    print("Rows with valid SMILES and target:", len(df))

    # 2. Report repeated chemical structures BEFORE aggregation.
    duplicate_mask = df.duplicated("canonical_smiles", keep=False)
    duplicate_rows = df[duplicate_mask].sort_values("canonical_smiles")

    print("Rows in repeated-structure groups:", len(duplicate_rows))
    print(
        "Repeated chemical structure groups:",
        duplicate_rows["canonical_smiles"].nunique(),
    )

    if not duplicate_rows.empty:
        duplicate_report = (
            duplicate_rows.groupby("canonical_smiles")
            .agg(
                measurements=(TARGET, list),
                row_count=(TARGET, "size"),
                target_min=(TARGET, "min"),
                target_max=(TARGET, "max"),
            )
            .reset_index()
        )
        duplicate_report.to_csv(
            OUTPUT_DIR / "canonical_duplicate_audit.csv", index=False
        )
        print("Duplicate audit saved: reports/canonical_duplicate_audit.csv")

    # 3. Aggregate measurements per canonical chemical structure.
    # This avoids putting duplicate representations of the same molecule
    # into different splits. Mean is used as a documented baseline choice.
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

    print("Unique canonical molecules after aggregation:", len(df))
    print("Molecules with repeated measurements:", (df["measurement_count"] > 1).sum())

    # 4. Split by Bemis-Murcko scaffold.
    df["scaffold"] = df[SMILES_COLUMN].apply(get_scaffold)

    rng = np.random.default_rng(42)
    scaffold_groups = list(df.groupby("scaffold").groups.values())
    rng.shuffle(scaffold_groups)

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

    print("\nTraining molecules:", len(train_df))
    print("Testing molecules:", len(test_df))
    print("Training scaffolds:", train_df["scaffold"].nunique())
    print("Testing scaffolds:", test_df["scaffold"].nunique())

    scaffold_overlap = set(train_df["scaffold"]) & set(test_df["scaffold"])
    structure_overlap = set(train_df[SMILES_COLUMN]) & set(test_df[SMILES_COLUMN])

    print("Scaffold overlap:", len(scaffold_overlap))
    print("Canonical structure overlap:", len(structure_overlap))

    if scaffold_overlap or structure_overlap:
        raise ValueError("Leakage check failed. Do not interpret these metrics.")

    # 5. Generate features using the same function as the app.
    X_train = np.vstack([
        calculate_features(smiles)[0] for smiles in train_df[SMILES_COLUMN]
    ])
    X_test = np.vstack([
        calculate_features(smiles)[0] for smiles in test_df[SMILES_COLUMN]
    ])

    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)

    # 6. Compare evaluation-only models. Do not overwrite the app model.
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

    rows = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

    comparison = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "cleaned_model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\n===== CANONICALIZED SCAFFOLD EVALUATION =====")
    print(comparison.round(4).to_string(index=False))
    print("\nSaved:", output_path)
    print("Original dataset, saved app model, and Streamlit app were not modified.")


if __name__ == "__main__":
    main()
