
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from predict import calculate_features

DATA_PATH = ROOT / "data" / "raw" / "delaney-processed.csv"
ORIGINAL_PATH = ROOT / "models" / "solubility_rf_baseline.joblib"
CANDIDATE_PATH = ROOT / "models" / "solubility_rf_clean_candidate.joblib"
OUTPUT_DIR = ROOT / "reports"

TARGET = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the same deterministic scaffold split.
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COLUMN, TARGET]).copy()
    df["canonical_smiles"] = df[SMILES_COLUMN].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()

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

    scaffold_overlap = (
        set(train_df["scaffold"]) & set(test_df["scaffold"])
    )
    structure_overlap = (
        set(train_df[SMILES_COLUMN]) & set(test_df[SMILES_COLUMN])
    )

    if scaffold_overlap or structure_overlap:
        raise ValueError("Scaffold or structure overlap detected.")

    print("Training molecules:", len(train_df))
    print("Held-out test molecules:", len(test_df))
    print("Scaffold overlap:", len(scaffold_overlap))
    print("Canonical structure overlap:", len(structure_overlap))

    # Load the two already-trained models.
    original = joblib.load(ORIGINAL_PATH)
    candidate = joblib.load(CANDIDATE_PATH)

    if original.n_features_in_ != candidate.n_features_in_:
        raise ValueError("The models expect different feature counts.")

    # Generate features only for held-out test molecules.
    X_test = np.vstack([
        calculate_features(smiles)[0]
        for smiles in test_df[SMILES_COLUMN]
    ])
    y_test = test_df[TARGET].to_numpy(dtype=float)

    rows = []

    for name, model in [
        ("Original app model", original),
        ("Clean candidate model", candidate),
    ]:
        predictions = model.predict(X_test)

        rows.append({
            "model": name,
            "test_molecules": len(y_test),
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

        test_df[f"{name} prediction"] = predictions

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_DIR / "original_vs_candidate_scaffold_test.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "original_vs_candidate_test_predictions.csv",
        index=False,
    )

    print("\n===== SAME HELD-OUT SCAFFOLD TEST SET =====")
    print(results.round(4).to_string(index=False))
    print("\nSaved metrics:")
    print("reports/original_vs_candidate_scaffold_test.csv")
    print("Saved predictions:")
    print("reports/original_vs_candidate_test_predictions.csv")
    print("\nNeither model file nor the Streamlit app was modified.")


if __name__ == "__main__":
    main()
